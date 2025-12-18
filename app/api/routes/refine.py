from __future__ import annotations
import os
import asyncio
import json
import uuid
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from app.core.logger import get_logger, log_exception
from app.core.mongodb_db import db as mongodb_db
from app.core.file_versions import file_version_manager
from app.core.storage import LocalSink, DriveSink
from app.core.paths import get_output_dir, get_backend_root, sanitize_path, _is_vercel
from app.utils.utils import get_google_credentials, safe_encoder, read_text_from_file
from app.core.websocket_manager import manager as ws_manager
from app.core.memory_manager import memory_manager
from app.core.state import (
    safe_jobs_snapshot_set, jobs_snapshot, active_tasks, 
    safe_active_tasks_del, safe_upsert_job, safe_uploaded_files_get,
    uploaded_files
)
from app.core.database import upsert_job
from app.core.dependencies import get_pipeline
from app.core.errors import APIError
from app.core.prompt_schema import ADVANCED_COMMANDS
from app.core.language_model import CostLimitExceeded, analytics_store

router = APIRouter()
logger = get_logger('api.refine')

MAX_REFINEMENT_PASSES = 10
MAX_HEURISTICS_SIZE = 1024 * 1024

# =====================================================
# PRESET PROFILES
# =====================================================
# Pre-configured refinement profiles for common use cases

PRESET_PROFILES = {
    "fast_cheap": {
        "name": "Fast & Cheap",
        "description": "Quick refinement using the cheapest model. Best for drafts or low-priority work.",
        "passes": 1,
        "entropy_level": "low",
        "model": "gpt-4o-mini",
        "aggressiveness": 3,
        "use_model_tiering": False,  # Always use cheap model
        "estimated_cost_per_1k_tokens": 0.00015,
        "heuristics": {
            "simplifyJargon": False,
            "condenseRedundant": True,
            "preserveFormatting": True,
            "toneFormality": 5
        }
    },
    "balanced": {
        "name": "Balanced",
        "description": "Good balance of quality and cost. Uses cheap model for initial passes, premium for final.",
        "passes": 2,
        "entropy_level": "medium",
        "model": "gpt-4o",
        "aggressiveness": 5,
        "use_model_tiering": True,  # Cheap for pass 1, premium for pass 2
        "estimated_cost_per_1k_tokens": 0.003,
        "heuristics": {
            "simplifyJargon": True,
            "condenseRedundant": True,
            "preserveFormatting": True,
            "toneFormality": 5
        }
    },
    "max_quality": {
        "name": "Max Quality",
        "description": "Best quality refinement. Uses premium model for all passes with thorough processing.",
        "passes": 3,
        "entropy_level": "high",
        "model": "gpt-4o",
        "aggressiveness": 7,
        "use_model_tiering": False,  # Premium model for all passes
        "estimated_cost_per_1k_tokens": 0.0025,
        "heuristics": {
            "simplifyJargon": True,
            "condenseRedundant": True,
            "preserveFormatting": True,
            "toneFormality": 7,
            "enhanceClarity": True,
            "improveFlow": True
        }
    },
    "academic": {
        "name": "Academic",
        "description": "Optimized for academic and research documents. Preserves citations and technical accuracy.",
        "passes": 2,
        "entropy_level": "low",
        "model": "gpt-4o",
        "aggressiveness": 4,
        "use_model_tiering": True,
        "estimated_cost_per_1k_tokens": 0.002,
        "heuristics": {
            "simplifyJargon": False,
            "condenseRedundant": True,
            "preserveFormatting": True,
            "preserveCitations": True,
            "toneFormality": 8
        }
    },
    "creative": {
        "name": "Creative",
        "description": "For creative writing. Maintains voice while improving flow and readability.",
        "passes": 2,
        "entropy_level": "high",
        "model": "gpt-4o",
        "aggressiveness": 4,
        "use_model_tiering": True,
        "estimated_cost_per_1k_tokens": 0.002,
        "heuristics": {
            "simplifyJargon": False,
            "condenseRedundant": False,
            "preserveFormatting": True,
            "preserveVoice": True,
            "toneFormality": 3
        }
    }
}


def apply_preset_to_request(request: "RefinementRequest", preset_name: str) -> "RefinementRequest":
    """Apply a preset profile to a refinement request."""
    if preset_name not in PRESET_PROFILES:
        return request
    
    preset = PRESET_PROFILES[preset_name]
    
    # Only override values that weren't explicitly set
    if request.passes == 1:  # Default value
        request.passes = preset.get("passes", 1)
    if request.entropy_level == "medium":  # Default value
        request.entropy_level = preset.get("entropy_level", "medium")
    if request.aggressiveness == 5:  # Default value
        request.aggressiveness = preset.get("aggressiveness", 5)
    
    # Merge heuristics (preset values as defaults, request values override)
    preset_heuristics = preset.get("heuristics", {})
    if request.heuristics:
        merged = {**preset_heuristics, **request.heuristics}
        request.heuristics = merged
    else:
        request.heuristics = preset_heuristics
    
    return request


# Error classification for better error handling and retry logic
class ErrorType:
    """Classification of errors for smarter handling."""
    TRANSIENT = "transient"  # Can be retried (timeouts, 5xx, rate limits)
    PERMANENT = "permanent"  # Cannot be retried (auth, config, context limit)
    COST_LIMIT = "cost_limit"  # Cost/token budget exceeded
    UNKNOWN = "unknown"

def classify_error(error: Exception) -> tuple[str, str, bool]:
    """
    Classify an error as transient or permanent.
    
    Returns:
        tuple: (error_type, user_friendly_message, should_retry)
    """
    error_str = str(error).lower()
    error_class = type(error).__name__
    
    # Transient errors (can retry)
    transient_patterns = [
        ("timeout", "The request timed out. Try again or reduce document size."),
        ("rate_limit", "Rate limit reached. Please wait a moment and try again."),
        ("429", "Too many requests. Please wait a moment."),
        ("503", "Service temporarily unavailable. Try again shortly."),
        ("502", "Server temporarily unavailable. Try again."),
        ("500", "Server error. Try again in a moment."),
        ("connection", "Connection error. Check your internet and try again."),
        ("network", "Network error. Check your connection."),
        ("econnreset", "Connection was reset. Try again."),
        ("econnrefused", "Connection refused. Try again later."),
    ]
    
    # Permanent errors (do not retry)
    permanent_patterns = [
        ("invalid_api_key", "Invalid API key. Please check your configuration."),
        ("authentication", "Authentication failed. Check your API key."),
        ("unauthorized", "Unauthorized. Please check your credentials."),
        ("401", "Authentication required. Check API key."),
        ("403", "Access forbidden. Check your permissions."),
        ("context_length", "Document too large for model. Try splitting into smaller sections."),
        ("maximum_context_length", "Document exceeds context limit. Reduce document size."),
        ("max_tokens", "Token limit exceeded. Try with fewer passes or smaller document."),
        ("invalid_request", "Invalid request format. Please check your input."),
        ("content_policy", "Content violates policy. Review and modify content."),
        ("model_not_found", "Model not available. Check model configuration."),
        ("billing", "Billing issue. Check your OpenAI account."),
        ("quota", "Quota exceeded. Check your OpenAI billing."),
        ("insufficient_quota", "Insufficient quota. Add billing to OpenAI account."),
    ]
    
    # Check transient patterns
    for pattern, message in transient_patterns:
        if pattern in error_str or pattern in error_class.lower():
            return ErrorType.TRANSIENT, message, True
    
    # Check permanent patterns
    for pattern, message in permanent_patterns:
        if pattern in error_str:
            return ErrorType.PERMANENT, message, False
    
    # Default to transient for unknown errors (safer to allow retry)
    return ErrorType.UNKNOWN, f"Unexpected error: {str(error)[:100]}", True

# CRITICAL FIX: Dedicated thread pool for pipeline execution
# This prevents blocking the main event loop and allows better resource management
# Max 4 concurrent pipeline jobs to avoid overwhelming the system
_pipeline_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline_worker")

class RefinementRequest(BaseModel):
    files: List[Dict[str, Any]]
    passes: int = 1
    startPass: int = 1  # Added for resume capability
    entropy_level: str = "medium"
    output_settings: Dict[str, Any] = None  # Will default to backend/data/output
    heuristics: Dict[str, Any] = {}
    settings: Dict[str, Any] = {}
    user_id: str = "default"
    use_memory: bool = True
    aggressiveness: int = 5  # Changed to int for preset compatibility
    earlyStop: bool = True
    scannerRisk: int = 15
    keywords: List[str] = []
    schemaLevels: Dict[str, Any] = {}
    strategy_mode: str = "model"
    entropy: Dict[str, Any] = {}
    formatting_safeguards: Dict[str, Any] = {}
    history_analysis: Dict[str, Any] = {}
    refiner_dry_run: bool = False
    annotation_mode: Dict[str, Any] = {}
    preset: str = None  # Preset profile name (fast_cheap, balanced, max_quality, academic, creative)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Normalize and be lenient to avoid 422s from common client payloads
        try:
            # user_id: fallback to default if invalid
            if not self.user_id or not isinstance(self.user_id, str):
                self.user_id = "default"
            if len(self.user_id) < 3:
                self.user_id = "user_" + self.user_id
            if len(self.user_id) > 50:
                self.user_id = self.user_id[:50]

            # passes: clamp
            if not isinstance(self.passes, int):
                try:
                    self.passes = int(self.passes)
                except Exception:
                    self.passes = 1
            self.passes = max(1, min(self.passes, MAX_REFINEMENT_PASSES))

            # entropy_level: normalize
            level_map = {"balanced": "medium", "med": "medium", "mid": "medium"}
            ent = (self.entropy_level or "medium").lower()
            ent = level_map.get(ent, ent)
            if ent not in ["low", "medium", "high"]:
                ent = "medium"
            self.entropy_level = ent

            # aggressiveness: case-insensitive
            ag = (self.aggressiveness or "Auto").strip()
            ag_norm = ag.lower()
            if ag_norm in ("auto", "low", "medium", "high"):
                self.aggressiveness = ag_norm.capitalize()
            else:
                self.aggressiveness = "Auto"

            # scannerRisk: coerce & clamp
            try:
                self.scannerRisk = int(self.scannerRisk)
            except Exception:
                self.scannerRisk = 15
            self.scannerRisk = max(0, min(self.scannerRisk, 100))

            # strategy_mode: normalize
            sm = (self.strategy_mode or "model").lower()
            if sm not in ["model", "manual", "hybrid"]:
                sm = "model"
            self.strategy_mode = sm

            # files: ensure list with minimal required keys; don't hard fail
            if not isinstance(self.files, list):
                self.files = []
            # Trim to a reasonable max
            if len(self.files) > 50:
                self.files = self.files[:50]
            # Ensure each file has id/name/type defaults
            normalized_files = []
            for idx, f in enumerate(self.files):
                if not isinstance(f, dict):
                    continue
                fid = f.get("id") or f.get("file_id") or f"file_{idx}"
                name = f.get("name") or f.get("filename") or fid
                ftype = f.get("type") or "local"
                normalized_files.append({**f, "id": fid, "name": name, "type": ftype})
            self.files = normalized_files

            # size guards for heuristics/settings
            if len(str(self.heuristics)) > MAX_HEURISTICS_SIZE:
                self.heuristics = {}
            if len(str(self.settings)) > MAX_HEURISTICS_SIZE:
                self.settings = {}

            # output_settings defaults
            if not isinstance(self.output_settings, dict):
                default_output = str(get_output_dir())
                self.output_settings = {"type": "local", "path": default_output}
            if self.output_settings.get("type") not in ("local", "drive"):
                self.output_settings["type"] = "local"
            if self.output_settings.get("type") == "local":
                # On Vercel, always use /tmp/output (writable)
                # Otherwise, use provided path or default, with sanitization
                if _is_vercel():
                    self.output_settings["path"] = str(get_output_dir())
                else:
                    path = self.output_settings.get("path") or self.output_settings.get("dir")
                    if not isinstance(path, str):
                        path = str(get_output_dir())
                    # Sanitize path to ensure it's within backend directory
                    sanitized_path = sanitize_path(path, base_dir=get_backend_root())
                    self.output_settings["path"] = str(sanitized_path)
        except Exception:
            # Never fail constructor; rely on endpoint logic to handle deeper errors
            pass

async def _validate_and_resolve_file_path(file_info: Dict[str, Any], file_id: str) -> str:
    """Validate and resolve file path with security checks"""
    from app.utils.utils import extract_drive_file_id, download_drive_file, get_google_credentials
    import tempfile
    
    # Check if this is a Google Drive file
    file_type = file_info.get("type", "local")
    drive_id = file_info.get("driveId") or file_info.get("drive_id")
    source = file_info.get("source") or file_info.get("path") or ""
    
    # If it's a drive file, download it first
    if file_type == "drive" or drive_id or ("drive.google.com" in source or "docs.google.com" in source):
        try:
            # Extract file ID from source if drive_id not provided
            if not drive_id and source:
                drive_id = extract_drive_file_id(source)
            
            if not drive_id:
                raise APIError(f"Could not extract Google Drive file ID from: {source}", 400, "INVALID_DRIVE_ID")
            
            # Check credentials
            creds = get_google_credentials()
            if not creds:
                raise APIError("Google Drive credentials not configured", 500, "DRIVE_NOT_CONFIGURED")
            
            # Get file metadata first to determine correct extension
            from app.utils.utils import get_drive_service
            service = get_drive_service()
            try:
                file_metadata = service.files().get(fileId=drive_id, fields="name, mimeType").execute()
                file_name = file_metadata.get("name", file_info.get("name", f"drive_file_{drive_id}"))
                mime_type = file_metadata.get("mimeType", "")
            except Exception as e:
                logger.warning(f"Could not get file metadata: {e}, using defaults")
                file_name = file_info.get("name", f"drive_file_{drive_id}")
                mime_type = ""
            
            # Get file extension from name, or infer from mime type
            ext = os.path.splitext(file_name)[1]
            if not ext:
                # Infer extension from mime type
                mime_to_ext = {
                    "application/pdf": ".pdf",
                    "application/vnd.google-apps.document": ".docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                    "application/msword": ".doc",
                    "text/plain": ".txt",
                    "text/markdown": ".md",
                }
                ext = mime_to_ext.get(mime_type, ".docx")
            
            if not ext.startswith('.'):
                ext = f".{ext}"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                temp_path = tmp.name
            
            # Download the file
            logger.debug(f"Downloading Google Drive file {drive_id} ({mime_type}) to {temp_path}")
            downloaded_path = download_drive_file(drive_id, temp_path)
            
            # Store in uploaded_files registry for future reference
            uploaded_files[file_id] = {
                "id": file_id,
                "name": file_name,
                "temp_path": downloaded_path,
                "type": "drive",
                "drive_id": drive_id,
                "created_at": time.time()
            }
            
            return downloaded_path
        except Exception as e:
            logger.error(f"Failed to download Google Drive file: {e}")
            raise APIError(f"Failed to download Google Drive file: {str(e)}", 500, "DRIVE_DOWNLOAD_FAILED")
    
    # Try to get file path from uploaded_files registry first
    # CRITICAL FIX: Check multiple possible file_id variations for multi-file support
    file_path = None
    stored_file_type = None  # Track the original file type
    
    # First, try the exact file_id
    if file_id in uploaded_files:
        file_info_stored = uploaded_files[file_id]
        file_path = file_info_stored.get("temp_path") or file_info_stored.get("path")
        stored_file_type = file_info_stored.get("file_type")  # Preserve file type from upload
        logger.debug(f"Found file by exact file_id: {file_id}, path: {file_path}, file_type: {stored_file_type}")
        
        # CRITICAL FIX: Verify extension matches stored file type
        if file_path and stored_file_type:
            actual_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if actual_ext != stored_file_type and stored_file_type in ['docx', 'doc', 'pdf']:
                logger.warning(f"Extension mismatch for {file_id}: temp_path ext='{actual_ext}', file_type='{stored_file_type}'")
                # The temp file should already have correct extension from upload, but log for debugging
    
    # CRITICAL FIX: If not found by file_id, try looking up by backendFileId or driveId
    # The frontend may send the backend's file_id in these fields
    if not file_path:
        backend_file_id = file_info.get("backendFileId") or file_info.get("driveId") or file_info.get("drive_id")
        if backend_file_id:
            # First, try direct lookup - the backend_file_id might BE the key in uploaded_files
            if backend_file_id in uploaded_files:
                file_info_stored = uploaded_files[backend_file_id]
                file_path = file_info_stored.get("temp_path") or file_info_stored.get("path")
                stored_file_type = file_info_stored.get("file_type")
                logger.debug(f"Found file by backendFileId direct lookup: {backend_file_id} -> {file_path}, type: {stored_file_type}")
            else:
                # Fallback: search through uploaded_files for matching drive_id or id
                for stored_file_id, stored_info in uploaded_files.items():
                    if stored_info.get("drive_id") == backend_file_id or stored_info.get("id") == backend_file_id:
                        file_path = stored_info.get("temp_path") or stored_info.get("path")
                        stored_file_type = stored_info.get("file_type")
                        logger.debug(f"Found file by backendFileId search: {backend_file_id} -> {file_path}")
                        break
        
        # Also try matching by source/path
        if not file_path:
            source = file_info.get("source") or file_info.get("path") or ""
            if source:
                for stored_file_id, stored_info in uploaded_files.items():
                    stored_path = stored_info.get("temp_path") or stored_info.get("path") or ""
                    stored_source = stored_info.get("source") or ""
                    if source in stored_path or source in stored_source or stored_path in source:
                        file_path = stored_path
                        stored_file_type = stored_info.get("file_type")
                        logger.debug(f"Found file by source/path: {source} -> {file_path}")
                        break
    
    # Fallback to file_info paths - check multiple possible path fields
    if not file_path:
        file_path = (file_info.get("path") or 
                   file_info.get("temp_path") or 
                   file_info.get("source") or "")
    
    # If still no path, try to construct from filename with strict security validation
    if not file_path and file_info.get("name"):
        filename = file_info.get("name")
        # Strict filename validation
        if filename and len(filename) <= 255 and filename.isprintable():
            # Remove any directory components and dangerous characters
            safe_filename = os.path.basename(filename)
            # Additional validation: only allow alphanumeric, dots, hyphens, underscores
            if all(c.isalnum() or c in '.-_' for c in safe_filename) and not safe_filename.startswith('.'):
                # Only check in the output directory - no other locations
                output_dir = get_output_dir()
                os.makedirs(output_dir, exist_ok=True)
                possible_path = os.path.join(output_dir, safe_filename)
                
                # Verify the resolved path is within the output directory
                try:
                    resolved_path = os.path.realpath(possible_path)
                    if resolved_path.startswith(str(output_dir)) and os.path.exists(resolved_path) and os.path.isfile(resolved_path):
                        file_path = resolved_path
                except (OSError, ValueError):
                    # Path resolution failed or is invalid
                    pass
    
    if not file_path or not os.path.exists(file_path):
        raise APIError(f'File not found: {file_path or "no path provided"}', 404, "FILE_NOT_FOUND")
    
    return file_path

async def _read_and_validate_file(file_path: str, file_id: str, job_id: str) -> str:
    """Read file content and validate it exists"""
    try:
        logger.debug(f"Reading file: {file_path}")
        original_text = read_text_from_file(file_path)
        logger.debug(f"Read {len(original_text)} characters from file")
        
        msg = {'type': 'stage_update', 'jobId': job_id, 'fileId': file_id, 'stage': 'read', 'status': 'completed', 'message': f'Read {len(original_text)} characters'}
        safe_jobs_snapshot_set(job_id, msg)
        try:
            upsert_job(job_id, {"current_stage": "read", "status": "running"})
        except Exception:
            pass
        
        # WS broadcast with proper failure handling
        try:
            await ws_manager.broadcast(job_id, msg)  # type: ignore
        except Exception as e:
            logger.warning(f"WebSocket broadcast failed: {e}")
            # Mark job as degraded due to WebSocket failure
            degraded_msg = {'type': 'warning', 'jobId': job_id, 'message': 'WebSocket connection failed - using polling fallback', 'degraded': True}
            safe_jobs_snapshot_set(job_id, degraded_msg)
            safe_upsert_job(job_id, {"status": "running", "degraded": True, "ws_failed": True})
        
        # Store original version (pass 0) for diff generation
        try:
            file_version_manager.store_version(
                file_id=file_id,
                pass_number=0,
                content=original_text,
                file_path=file_path,
                metadata={
                    "job_id": job_id,
                    "original": True,
                    "file_size": len(original_text)
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store original version: {e}")
        
        return original_text
        
    except Exception as e:
        error_msg = str(e).replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
        err = {'type': 'error', 'jobId': job_id, 'fileId': file_id, 'error': f'File read failed: {error_msg}'}
        safe_jobs_snapshot_set(job_id, err)
        try:
            upsert_job(job_id, {"current_stage": "read", "status": "failed", "error": err.get("error")})
        except Exception:
            pass
        try:
            await ws_manager.broadcast(job_id, err)  # type: ignore
        except Exception as e:
            logger.warning(f"WebSocket broadcast failed: {e}")
        raise APIError(f'File read failed: {error_msg}', 500, "FILE_READ_ERROR")

async def _check_infinite_recursion_risk(current_text: str, original_text: str, pass_num: int, file_id: str, job_id: str, last_requested_pass: int = 10) -> bool:
    """Check for infinite recursion risk and return True if should stop
    
    Args:
        last_requested_pass: The last pass NUMBER in the requested range (NOT the count).
                            e.g., if running passes 1-3, this should be 3.
    
    CRITICAL FIX: Only check for recursion risk AFTER the user's requested passes are complete.
    If user asked for passes 1-5, let all 5 run regardless of similarity.
    """
    # NEVER stop early within the requested pass range
    # User asked for passes up to last_requested_pass, so run them all
    if pass_num <= last_requested_pass:
        return False  # Always allow requested passes to run
    
    # Beyond requested passes - check for true infinite recursion
    if pass_num > last_requested_pass:
        # Check for exact duplicates - text hasn't changed at all
        if current_text == original_text:
            logger.warning(f"Pass {pass_num} (beyond requested pass {last_requested_pass}) would process original text, stopping")
            warning_msg = {'type': 'warning', 'jobId': job_id, 'fileId': file_id, 'pass': pass_num, 'message': 'Text reverted to original, stopping refinement'}
            try:
                await ws_manager.broadcast(job_id, warning_msg)  # type: ignore
            except Exception:
                pass
            safe_jobs_snapshot_set(job_id, warning_msg)
            return True
    
    return False

async def _process_refinement_pass(
    pipeline, 
    file_path: str, 
    current_text: str, 
    pass_num: int, 
    request: RefinementRequest, 
    file_id: str, 
    job_id: str,
    output_sink
) -> tuple:
    """Process a single refinement pass and return (success, final_text, metrics, pipeline_state, result)"""
    try:
        logger.debug(f"Starting pass {pass_num} of {request.passes}")
        
        # Emit pass start event
        start_evt = {'type': 'pass_start', 'jobId': job_id, 'fileId': file_id, 'pass': pass_num, 'totalPasses': request.passes}
        try:
            await ws_manager.broadcast(job_id, start_evt)  # type: ignore
        except Exception:
            pass
        safe_jobs_snapshot_set(job_id, start_evt)
        try:
            upsert_job(job_id, {"current_stage": f"pass_{pass_num}_start", "status": "running"})
        except Exception:
            pass
        
        # Emit plan event with strategy weights and entropy settings
        plan_evt = {
            'type': 'plan',
            'jobId': job_id,
            'fileId': file_id,
            'pass': pass_num,
            'weights': request.heuristics.get('strategy_weights', {
                'clarity': 0.6,
                'persuasion': 0.3,
                'brevity': 0.3,
                'formality': 0.6
            }) if request.heuristics else {
                'clarity': 0.6,
                'persuasion': 0.3,
                'brevity': 0.3,
                'formality': 0.6
            },
            'entropy': request.heuristics.get('entropy', {
                'risk_preference': 0.5,
                'repeat_penalty': 0.0,
                'phrase_penalty': 0.0
            }) if request.heuristics else {
                'risk_preference': 0.5,
                'repeat_penalty': 0.0,
                'phrase_penalty': 0.0
            },
            'formatting': request.heuristics.get('formatting_safeguards', {}).get('mode', 'smart') if request.heuristics and request.heuristics.get('formatting_safeguards') else 'smart',
            'aggressiveness': request.aggressiveness or 'Auto'
        }
        try:
            await ws_manager.broadcast(job_id, plan_evt)  # type: ignore
        except Exception:
            pass
        safe_jobs_snapshot_set(job_id, plan_evt)
        logger.debug(f"Emitted plan event for pass {pass_num}: {plan_evt}")
        
        # Run the pipeline pass in dedicated executor to avoid blocking event loop
        # CRITICAL FIX: Use dedicated thread pool for better resource management
        logger.debug(f"About to call pipeline.run_pass for pass {pass_num}")
        loop = asyncio.get_running_loop()
        # Calculate end pass for total_passes (for model tiering)
        end_pass = request.startPass + request.passes
        total_passes = end_pass - request.startPass  # Actual number of passes to run
        
        ps, rr, ft = await loop.run_in_executor(
            _pipeline_executor,  # Use dedicated pipeline thread pool
            lambda: pipeline.run_pass(
                input_path=file_path,
                pass_index=pass_num,
                prev_final_text=current_text,
                entropy_level=request.entropy_level,
                output_sink=output_sink,
                drive_title_base=Path(file_path).stem,
                heuristics_overrides=request.heuristics,
                job_id=job_id,
                total_passes=total_passes  # For model tiering (cheaper model on early passes)
            )
        )
        logger.debug(f"pipeline.run_pass completed for pass {pass_num}")
        
        # Emit strategy event if available
        try:
            if hasattr(pipeline, '_last_strategy') and pipeline._last_strategy:
                strategy_evt = {
                    'type': 'strategy',
                    'jobId': job_id,
                    'fileId': file_id,
                    'pass': pass_num,
                    'weights': pipeline._last_strategy.get('weights', {}),
                    'rationale': pipeline._last_strategy.get('rationale', ''),
                    'approach': pipeline._last_strategy.get('approach', ''),
                    'plan': pipeline._last_strategy.get('plan', {})
                }
                await ws_manager.broadcast(job_id, strategy_evt)  # type: ignore
                safe_jobs_snapshot_set(job_id, strategy_evt)
                logger.debug(f"Emitted strategy event for pass {pass_num}")
        except Exception as e:
            logger.debug(f"Failed to emit strategy event: {e}")
        
        # Emit stage updates
        for stage_name, stage_state in ps.stages.items():
            st_evt = {'type': 'stage_update', 'jobId': job_id, 'fileId': file_id, 'pass': pass_num, 'stage': stage_name, 'status': stage_state.status, 'duration': stage_state.duration_ms}
            try:
                await ws_manager.broadcast(job_id, st_evt)  # type: ignore
            except Exception:
                pass
            safe_jobs_snapshot_set(job_id, st_evt)
            try:
                upsert_job(job_id, {"current_stage": stage_name, "status": "running"})
            except Exception:
                pass
        
        # Calculate metrics
        change_percent = ps.metrics.change_pct if ps.metrics else 0.0
        tension_percent = ps.metrics.tension_pct if ps.metrics else 0.0
        scanner_risk = ps.metrics.scanner_risk if ps.metrics else 0.0
        
        metrics = {
            'changePercent': change_percent,
            'tensionPercent': tension_percent,
            'scannerRisk': scanner_risk,
            'success': rr.success,
            'localPath': rr.local_path,
            'docId': rr.doc_id,
            'originalLength': len(current_text),
            'finalLength': len(ft),
            'processingTime': sum(stage.duration_ms for stage in ps.stages.values())
        }
        # Include per-pass token counts if provided by pipeline
        try:
            tp = getattr(pipeline, '_last_pass_token_stats', None)
            if isinstance(tp, dict):
                metrics['inputTokensPreflight'] = int(tp.get('preflightInTokens', 0) or 0)
                metrics['inputTokensUsed'] = int(tp.get('usedInTokens', 0) or 0)
        except Exception:
            pass
        
        # Store file version for diff generation
        try:
            file_version_manager.store_version(
                file_id=file_id,
                pass_number=pass_num,
                content=ft,
                file_path=rr.local_path,
                metrics=metrics,
                metadata={
                    "job_id": job_id,
                    "entropy_level": request.entropy_level,
                    "heuristics": request.heuristics,
                    "processing_time": sum(stage.duration_ms for stage in ps.stages.values())
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store version: {e}")
        
        # Get cost information for this pass
        pass_cost_info = {}
        if hasattr(pipeline, '_pass_costs') and pipeline._pass_costs:
            total_pass_cost = sum(cost['total_cost'] for cost in pipeline._pass_costs)
            pass_cost_info = {
                'totalCost': total_pass_cost,
                'costBreakdown': pipeline._pass_costs,
                'requestCount': len(pipeline._pass_costs)
            }
        
        # Emit pass complete event
        pc_evt = {
            'type': 'pass_complete', 
            'jobId': job_id, 
            'fileId': file_id, 
            'pass': pass_num, 
            'metrics': metrics,
            'inputChars': len(current_text),
            'outputChars': len(ft),
            'outputPath': rr.local_path if hasattr(rr, 'local_path') and rr.local_path else None,
            'cost': pass_cost_info,
            'textContent': ft  # CRITICAL: Include textContent for diff viewer and downloads
        }
        try:
            await ws_manager.broadcast(job_id, pc_evt)  # type: ignore
        except Exception:
            pass
        safe_jobs_snapshot_set(job_id, pc_evt)
        try:
            prog = min(100.0, (pass_num / max(1, request.passes)) * 100.0)
            upsert_job(job_id, {"current_stage": "pass_complete", "progress": prog, "status": "running"})
        except Exception:
            pass
        
        return True, ft, metrics, ps, rr
        
    except CostLimitExceeded as e:
        # COST LIMIT EXCEEDED - surface this clearly to the user
        error_msg = f"Cost limit exceeded: {e.message}"
        err_evt = {
            'type': 'error', 
            'jobId': job_id, 
            'fileId': file_id, 
            'pass': pass_num, 
            'error': error_msg,
            'errorType': 'cost_limit',
            'currentCost': e.current_cost,
            'limit': e.limit,
            'limitType': e.limit_type
        }
        try:
            await ws_manager.broadcast(job_id, err_evt)  # type: ignore
        except Exception:
            pass
        safe_jobs_snapshot_set(job_id, err_evt)
        try:
            upsert_job(job_id, {"current_stage": "cost_limit", "status": "failed", "error": error_msg})
        except Exception:
            pass
        logger.warning(f"Cost limit exceeded for job {job_id}: {e.message}")
        return False, current_text, {'error': error_msg, 'errorType': 'cost_limit'}, None, None
        
    except Exception as e:
        # Classify the error for smarter handling
        error_type, user_message, can_retry = classify_error(e)
        error_msg = str(e).replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
        
        # Include the actual error message in the event so users can see what went wrong
        err_evt = {
            'type': 'error', 
            'jobId': job_id, 
            'fileId': file_id, 
            'pass': pass_num, 
            'error': f'Pass {pass_num} failed: {user_message}',
            'errorType': error_type,
            'canRetry': can_retry,
            'technicalDetails': error_msg[:200] if len(error_msg) > 200 else error_msg
        }
        try:
            await ws_manager.broadcast(job_id, err_evt)  # type: ignore
        except Exception:
            pass
        safe_jobs_snapshot_set(job_id, err_evt)
        
        # Different status based on error type
        status = "failed" if error_type == ErrorType.PERMANENT else "error_recoverable"
        try:
            upsert_job(job_id, {"current_stage": "error", "status": status, "error": err_evt.get("error")})
        except Exception:
            pass
            
        if error_type == ErrorType.PERMANENT:
            logger.error(f"Permanent error on pass {pass_num}: {e}", exc_info=True)
        else:
            logger.warning(f"Transient error on pass {pass_num} (retryable): {e}")
            
        # Return the error message so the caller can use it
        return False, current_text, {
            'error': error_msg, 
            'errorType': error_type, 
            'userMessage': user_message,
            'canRetry': can_retry
        }, None, None

async def _refine_stream(request: RefinementRequest, job_id: str) -> AsyncGenerator[str, None]:
    logger.debug(f"Starting refinement stream for job {job_id}")
    
    # Apply preset profile if specified
    if request.preset:
        request = apply_preset_to_request(request, request.preset)
        logger.info(f"Applied preset '{request.preset}' to job {job_id}")
    
    # Create job in MongoDB
    try:
        # Extract file info for the first file (assuming single file job for now or primary file)
        primary_file = request.files[0] if request.files else {}
        if mongodb_db.is_connected():
            mongodb_db.create_job(
                job_id=job_id,
                file_name=primary_file.get("name", "unknown"),
                file_id=primary_file.get("id", "unknown"),
                user_id=request.user_id,
                total_passes=request.passes,
                model=request.model if hasattr(request, 'model') else "gpt-4",
                metadata={"heuristics": request.heuristics}
            )
    except Exception as e:
        logger.warning(f"Failed to create job in MongoDB: {e}")

    pipeline = get_pipeline()
    logger.debug(f"Pipeline initialized successfully")
    
    # Set up progress callback for real-time chunk progress updates
    # This allows the pipeline to emit progress events during long-running chunk processing
    # Using thread-safe broadcast mechanism since pipeline runs in a thread pool
    
    def pipeline_progress_callback(cb_job_id: str, stage: str, progress: float, message: str):
        """Callback invoked by pipeline during chunk processing to report progress.
        Called from a thread pool, so we use asyncio.run_coroutine_threadsafe for WebSocket broadcast.
        """
        try:
            progress_evt = {
                'type': 'chunk_progress',
                'jobId': cb_job_id,
                'stage': stage,
                'progress': round(progress, 1),
                'message': message
            }
            # Broadcast via WebSocket from thread - use thread-safe method
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule the coroutine from the thread
                    asyncio.run_coroutine_threadsafe(
                        ws_manager.broadcast(cb_job_id, progress_evt),
                        loop
                    )
            except Exception:
                # If we can't get event loop, just log (progress will still show in logs)
                pass
            logger.debug(f"Progress callback: {message}")
        except Exception as e:
            logger.debug(f"Progress callback error: {e}")
    
    # Set the callback on the pipeline
    if hasattr(pipeline, 'set_progress_callback'):
        pipeline.set_progress_callback(pipeline_progress_callback)
        logger.debug("Set pipeline progress callback")
    
    memory = memory_manager.get_memory(request.user_id) if request.use_memory else None
    processed_files = 0  # Track successfully processed files
    logger.debug(f"Processing {len(request.files)} files")
    try:
        for file_idx, file_info in enumerate(request.files):
            file_id = file_info.get("id", "unknown")
            file_name = file_info.get("name", file_info.get("fileName", Path(file_id).name if '/' in file_id else file_id))
            
            logger.debug(f"Processing file {file_idx + 1}/{len(request.files)}: file_id={file_id}, file_name={file_name}, type={file_info.get('type')}, source={file_info.get('source')}")
            
            try:
                # Check if we have direct text content (Resume scenario)
                if file_info.get("textContent"):
                    original_text = file_info["textContent"]
                    # Validate it's a string
                    if not isinstance(original_text, str):
                        raise APIError("Invalid textContent provided", 400, "INVALID_INPUT")
                    
                    # Log the resume action
                    msg = {'type': 'stage_update', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'stage': 'resume', 'status': 'completed', 'message': f'Resumed with {len(original_text)} characters'}
                    yield f"{safe_encoder(msg)}\n\n"
                else:
                    # CRITICAL FIX: Enhanced file path resolution with better logging
                    logger.debug(f"Resolving file path for file_id={file_id}, checking uploaded_files registry...")
                    logger.debug(f"Available file_ids in registry: {list(uploaded_files.keys())[:10]}")  # Log first 10 for debugging
                    
                    # Validate and resolve file path
                    file_path = await _validate_and_resolve_file_path(file_info, file_id)
                    logger.debug(f"Resolved file path: {file_path}")
                    
                    # Read and validate file content
                    original_text = await _read_and_validate_file(file_path, file_id, job_id)
                    
                    # Yield the read completion message - include inputChars for frontend
                    msg = {'type': 'stage_update', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'stage': 'read', 'status': 'completed', 'message': f'Read {len(original_text)} characters', 'inputChars': len(original_text)}
                    yield f"{safe_encoder(msg)}\n\n"
                
            except APIError as e:
                # Handle file not found or other API errors
                logger.error(f"File processing error for file_id={file_id}, file_name={file_name}: {e.message}")
                err = {'type': 'error', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'error': e.message, 'error_code': e.error_code}
                safe_jobs_snapshot_set(job_id, err)
                yield f"{safe_encoder(err)}\n\n"
                # Continue processing other files instead of stopping
                continue
            except Exception as e:
                # Handle unexpected errors
                logger.error(f"Unexpected error processing file_id={file_id}, file_name={file_name}: {str(e)}", exc_info=True)
                err = {'type': 'error', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'error': f"Unexpected error: {str(e)}", 'error_code': 'UNEXPECTED_ERROR'}
                safe_jobs_snapshot_set(job_id, err)
                yield f"{safe_encoder(err)}\n\n"
                # Continue processing other files instead of stopping
                continue
            
            # Track the current text for each pass (starts with original)
            current_text = original_text
            file_processed_successfully = False
            
            # Use startPass to determine the range (Resume scenario)
            start_pass = request.startPass
            end_pass = start_pass + request.passes
            
            for pass_num in range(start_pass, end_pass):
                logger.debug(f"Starting pass {pass_num} of {end_pass - 1}")
                start_evt = {'type': 'pass_start', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'pass': pass_num, 'totalPasses': end_pass - 1}
                try:
                    await ws_manager.broadcast(job_id, start_evt)  # type: ignore
                except Exception:
                    pass
                jobs_snapshot[job_id] = start_evt
                try:
                    upsert_job(job_id, {"current_stage": f"pass_{pass_num}_start", "status": "running"})
                except Exception:
                    pass
                yield f"{safe_encoder(start_evt)}\n\n"
                
                # Emit plan event with strategy weights and entropy settings for this pass
                plan_evt = {
                    'type': 'plan',
                    'jobId': job_id,
                    'fileId': file_id,
                    'fileName': file_name,
                    'pass': pass_num,
                    'weights': request.heuristics.get('strategy_weights', {
                        'clarity': 0.6,
                        'persuasion': 0.3,
                        'brevity': 0.3,
                        'formality': 0.6
                    }) if request.heuristics else {
                        'clarity': 0.6,
                        'persuasion': 0.3,
                        'brevity': 0.3,
                        'formality': 0.6
                    },
                    'entropy': request.heuristics.get('entropy', {
                        'risk_preference': 0.5,
                        'repeat_penalty': 0.0,
                        'phrase_penalty': 0.0
                    }) if request.heuristics else {
                        'risk_preference': 0.5,
                        'repeat_penalty': 0.0,
                        'phrase_penalty': 0.0
                    },
                    'formatting': request.heuristics.get('formatting_safeguards', {}).get('mode', 'smart') if request.heuristics and request.heuristics.get('formatting_safeguards') else 'smart',
                    'aggressiveness': request.aggressiveness or 'Auto'
                }
                try:
                    await ws_manager.broadcast(job_id, plan_evt)  # type: ignore
                except Exception:
                    pass
                jobs_snapshot[job_id] = plan_evt
                yield f"{safe_encoder(plan_evt)}\n\n"
                logger.debug(f"Emitted plan event for pass {pass_num}: {plan_evt}")
                
                output_sink = None
                if request.output_settings.get("type") == "local":
                    # On Vercel, always use /tmp/output (writable)
                    # Otherwise, use provided path or default, with sanitization
                    if _is_vercel():
                        output_path = get_output_dir()
                    else:
                        output_path_env = request.output_settings.get("path")
                        if output_path_env:
                            output_path = sanitize_path(
                                output_path_env,
                                base_dir=get_backend_root()
                            )
                        else:
                            output_path = get_output_dir()
                    output_sink = LocalSink(str(output_path))
                elif request.output_settings.get("type") == "drive":
                    try:
                        output_sink = DriveSink(request.output_settings.get("folder_id"), get_google_credentials())
                    except Exception as e:
                        error_msg = str(e).replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
                        yield f"{safe_encoder({'type': 'warning', 'fileId': file_id, 'message': f'Google Drive unavailable, using local: {error_msg}'})}\n\n"
                        output_sink = LocalSink(str(get_output_dir()))
                
                # Early infinite recursion detection - check before processing
                # CRITICAL FIX: Pass the last requested pass NUMBER (not count) to allow all requested passes to run
                last_pass_in_range = end_pass - 1  # e.g., if startPass=1, passes=3, last_pass=3
                if await _check_infinite_recursion_risk(current_text, original_text, pass_num, file_id, job_id, last_pass_in_range):
                    break

                try:
                    running_evt = {'type': 'stage_update', 'jobId': job_id, 'fileId': file_id, 'pass': pass_num, 'stage': 'processing', 'status': 'running'}
                    try:
                        await ws_manager.broadcast(job_id, running_evt)  # type: ignore
                    except Exception:
                        pass
                    jobs_snapshot[job_id] = running_evt
                    try:
                        upsert_job(job_id, {"current_stage": "processing", "status": "running"})
                    except Exception:
                        pass
                    yield f"{safe_encoder(running_evt)}\n\n"
                    
                    # Log pass start to MongoDB
                    if mongodb_db.is_connected():
                        mongodb_db.log_job_event(
                            job_id=job_id,
                            event_type="pass_start",
                            message=f"Starting pass {pass_num}",
                            pass_number=pass_num
                        )
                    
                    # Process the pass (this internally calls pipeline.run_pass)
                    # CRITICAL FIX: Removed duplicate run_pass call - _process_refinement_pass already calls it
                    # CRITICAL FIX: Add timeout to prevent indefinite hanging
                    pass_timeout_seconds = 20 * 60  # 20 minutes max per pass (generous for large docs)
                    try:
                        async with asyncio.timeout(pass_timeout_seconds):
                            success, refined_text, metrics, ps, rr = await _process_refinement_pass(
                                pipeline, 
                                file_path, 
                                current_text, 
                                pass_num, 
                                request, 
                                file_id, 
                                job_id,
                                output_sink
                            )
                    except asyncio.TimeoutError:
                        logger.error(f"Pass {pass_num} timed out after {pass_timeout_seconds} seconds")
                        timeout_evt = {'type': 'error', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'pass': pass_num, 'error': f'Pass {pass_num} timed out after {pass_timeout_seconds // 60} minutes. The document may be too large or complex.'}
                        try:
                            await ws_manager.broadcast(job_id, timeout_evt)  # type: ignore
                        except Exception:
                            pass
                        jobs_snapshot[job_id] = timeout_evt
                        try:
                            upsert_job(job_id, {"current_stage": "timeout", "status": "failed", "error": timeout_evt.get("error")})
                        except Exception:
                            pass
                        yield f"{safe_encoder(timeout_evt)}\n\n"
                        break  # Stop processing this file on timeout
                    
                    # Get the final text from refined_text
                    ft = refined_text
                    
                    # If pipeline state is None (error case), skip stage updates but show actual error
                    if ps is None:
                        # Get actual error from metrics if available (we store it there on failure)
                        actual_error = metrics.get('error', 'Unknown error during pass processing') if isinstance(metrics, dict) else 'Unknown error during pass processing'
                        logger.error(f"Pipeline state is None for pass {pass_num}, error: {actual_error}")
                        # Don't emit another error - _process_refinement_pass already emitted one with the actual error
                        # Just log and continue to next pass
                        # If we want to emit pass info even on failure, emit partial pass_complete
                        partial_evt = {
                            'type': 'pass_complete',
                            'jobId': job_id,
                            'fileId': file_id,
                            'fileName': file_name,
                            'pass': pass_num,
                            'metrics': {'error': actual_error, 'success': False},
                            'inputChars': len(current_text),
                            'outputChars': len(refined_text) if refined_text else 0,
                            'error': actual_error
                        }
                        try:
                            await ws_manager.broadcast(job_id, partial_evt)  # type: ignore
                        except Exception:
                            pass
                        jobs_snapshot[job_id] = partial_evt
                        yield f"{safe_encoder(partial_evt)}\n\n"
                        # Continue to next pass instead of breaking
                        continue

                    # Log pass completion to MongoDB
                    if mongodb_db.is_connected():
                        mongodb_db.log_job_event(
                            job_id=job_id,
                            event_type="pass_complete",
                            message=f"Completed pass {pass_num}",
                            pass_number=pass_num,
                            details={
                                "metrics": metrics, 
                                "success": success,
                                "textContent": ft,  # Store textContent for diff viewer fallback
                                "fileId": file_id
                            }
                        )
                        
                        # Update job status in MongoDB
                        mongodb_db.update_job_status(
                            job_id=job_id, 
                            status="processing", 
                            current_pass=pass_num
                        )

                    # Emit actual strategy used (if available from pipeline)
                    try:
                        if hasattr(pipeline, '_last_strategy') and pipeline._last_strategy:
                            strategy_evt = {
                                'type': 'strategy',
                                'jobId': job_id,
                                'fileId': file_id,
                                'fileName': file_name,
                                'pass': pass_num,
                                'weights': pipeline._last_strategy.get('weights', {}),
                                'rationale': pipeline._last_strategy.get('rationale', ''),
                                'approach': pipeline._last_strategy.get('approach', ''),
                                'plan': pipeline._last_strategy.get('plan', {})
                            }
                            await ws_manager.broadcast(job_id, strategy_evt)  # type: ignore
                            jobs_snapshot[job_id] = strategy_evt
                            yield f"{safe_encoder(strategy_evt)}\n\n"
                            logger.debug(f"Emitted strategy event for pass {pass_num}")
                    except Exception as e:
                        logger.debug(f"Failed to emit strategy event: {e}")
                    
                    # Validate that the pass actually produced meaningful changes to prevent infinite recursion
                    # CRITICAL FIX: Only check for EXACT duplicates on passes AFTER the requested range
                    # Allow all requested passes to run - user asked for passes up to end_pass-1
                    # Only stop early if text is EXACTLY the same (no changes at all)
                    last_requested_pass = end_pass - 1  # e.g., if startPass=1, passes=3, end_pass=4, last_requested=3
                    if pass_num > last_requested_pass:
                        # Beyond requested passes - check for exact duplicates only
                        if refined_text == current_text:
                            logger.warning(f"Pass {pass_num} produced identical text (beyond requested pass {last_requested_pass}), stopping")
                            warning_msg = {'type': 'warning', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'pass': pass_num, 'message': 'Pass produced identical text, stopping refinement'}
                            try:
                                await ws_manager.broadcast(job_id, warning_msg)  # type: ignore
                            except Exception:
                                pass
                            jobs_snapshot[job_id] = warning_msg
                            yield f"{safe_encoder(warning_msg)}\n\n"
                            break  # Stop processing this file
                    elif pass_num > 1:
                        # Within requested passes - only warn but DON'T stop
                        # User explicitly requested these passes, so run them all
                        if refined_text == current_text:
                            logger.info(f"Pass {pass_num} produced identical text, but continuing as user requested passes up to {last_requested_pass}")
                            info_msg = {'type': 'info', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'pass': pass_num, 'message': f'Pass {pass_num} text unchanged, continuing to next pass'}
                            try:
                                await ws_manager.broadcast(job_id, info_msg)  # type: ignore
                            except Exception:
                                pass
                            yield f"{safe_encoder(info_msg)}\n\n"
                            # Don't break - continue to next pass as requested
                    
                    for stage_name, stage_state in ps.stages.items():
                        st_evt = {'type': 'stage_update', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'pass': pass_num, 'stage': stage_name, 'status': stage_state.status, 'duration': stage_state.duration_ms}
                        try:
                            await ws_manager.broadcast(job_id, st_evt)  # type: ignore
                        except Exception:
                            pass
                        jobs_snapshot[job_id] = st_evt
                        try:
                            upsert_job(job_id, {"current_stage": stage_name, "status": "running"})
                        except Exception:
                            pass
                        yield f"{safe_encoder(st_evt)}\n\n"
                    change_percent = ps.metrics.change_pct if ps.metrics else 0.0
                    tension_percent = ps.metrics.tension_pct if ps.metrics else 0.0
                    scanner_risk = ps.metrics.scanner_risk if ps.metrics else 0.0
                    if memory and request.use_memory:
                        # Log against the current text (previous pass output), not original
                        memory_manager.log_refinement_pass(request.user_id, current_text, ft, score=scanner_risk, notes=[f"Pass {pass_num}", f"Entropy: {request.entropy_level}"])
                    metrics = {
                        'changePercent': change_percent,
                        'tensionPercent': tension_percent,
                        'scannerRisk': scanner_risk,
                        'success': rr.success,
                        'localPath': rr.local_path,
                        'docId': rr.doc_id,
                        'originalLength': len(current_text),  # Use current text length
                        'finalLength': len(ft),
                        'processingTime': sum(stage.duration_ms for stage in ps.stages.values())
                    }
                    # Store file version for diff generation
                    try:
                        file_version_manager.store_version(
                            file_id=file_id,
                            pass_number=pass_num,
                            content=ft,
                            file_path=rr.local_path,
                            metrics=metrics,
                            metadata={
                                "job_id": job_id,
                                "entropy_level": request.entropy_level,
                                "heuristics": request.heuristics,
                                "processing_time": sum(stage.duration_ms for stage in ps.stages.values())
                            }
                        )
                    except Exception as e:
                        # Log but don't fail the refinement
                        log_exception("VERSION_STORAGE_ERROR", e)
                    
                    # Get cost information for this pass
                    pass_cost_info = {}
                    if hasattr(pipeline, '_pass_costs') and pipeline._pass_costs:
                        total_pass_cost = sum(cost['total_cost'] for cost in pipeline._pass_costs)
                        pass_cost_info = {
                            'totalCost': total_pass_cost,
                            'costBreakdown': pipeline._pass_costs,
                            'requestCount': len(pipeline._pass_costs)
                        }
                    
                    pc_evt = {
                        'type': 'pass_complete', 
                        'jobId': job_id, 
                        'fileId': file_id, 
                        'fileName': file_name, 
                        'pass': pass_num, 
                        'metrics': metrics,
                        'inputChars': len(current_text),
                        'outputChars': len(ft),
                        'outputPath': rr.local_path if hasattr(rr, 'local_path') and rr.local_path else None,
                        'cost': pass_cost_info,
                        'textContent': ft
                    }
                    logger.debug(f"About to yield pass_complete event for pass {pass_num}")
                    try:
                        await ws_manager.broadcast(job_id, pc_evt)  # type: ignore
                    except Exception:
                        pass
                    jobs_snapshot[job_id] = pc_evt
                    try:
                        # CRITICAL FIX: Calculate progress correctly accounting for start_pass
                        # pass_num is the actual pass number, start_pass is where we started
                        # current_pass_index is 1-indexed within the current batch
                        current_pass_index = pass_num - start_pass + 1
                        total_passes = request.passes
                        prog = min(100.0, (current_pass_index / max(1, total_passes)) * 100.0)
                        upsert_job(job_id, {"current_stage": "pass_complete", "progress": prog, "status": "running"})
                    except Exception:
                        pass
                    yield f"{safe_encoder(pc_evt)}\n\n"
                    # Force flush by yielding a keepalive immediately after
                    yield ":keepalive\n\n"
                    logger.debug(f"Yielded pass_complete event for pass {pass_num}")
                    # Update current text for next pass
                    current_text = ft
                    file_processed_successfully = True
                except Exception as e:
                    error_msg = str(e).replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
                    err2 = {'type': 'error', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'pass': pass_num, 'error': error_msg}
                    try:
                        await ws_manager.broadcast(job_id, err2)  # type: ignore
                    except Exception:
                        pass
                    jobs_snapshot[job_id] = err2
                    
                    # CRITICAL FIX: Determine if error is recoverable
                    # For recoverable errors, continue to next pass; for fatal errors, break
                    is_fatal_error = any(fatal in error_msg.lower() for fatal in [
                        'file not found', 'permission denied', 'timeout', 
                        'out of memory', 'quota exceeded', 'api key'
                    ])
                    
                    if is_fatal_error:
                        try:
                            upsert_job(job_id, {"current_stage": "error", "status": "failed", "error": err2.get("error")})
                        except Exception:
                            pass
                        yield f"{safe_encoder(err2)}\n\n"
                        log_exception("REFINEMENT_STREAM_ERROR_FATAL", e)
                        break  # Stop processing this file on fatal errors
                    else:
                        # Non-fatal error: log warning but try to continue
                        try:
                            upsert_job(job_id, {"current_stage": "error_recovered", "status": "running", "last_error": err2.get("error")})
                        except Exception:
                            pass
                        yield f"{safe_encoder(err2)}\n\n"
                        log_exception("REFINEMENT_STREAM_ERROR_RECOVERED", e)
                        # Continue to next pass - the text will be current_text (last good version)
                        continue
            
            # CRITICAL FIX: After pass loop completes, emit file completion event
            # This ensures the frontend knows the file processing is done
            if file_processed_successfully:
                logger.debug(f"All passes completed for file {file_id}, emitting file completion")
                file_complete_evt = {
                    'type': 'file_complete',
                    'jobId': job_id,
                    'fileId': file_id,
                    'fileName': file_name,
                    'finalPass': end_pass - 1,  # Last pass number
                    'message': f'All {request.passes} passes completed for {file_name}'
                }
                try:
                    await ws_manager.broadcast(job_id, file_complete_evt)  # type: ignore
                except Exception:
                    pass
                jobs_snapshot[job_id] = file_complete_evt
                yield f"{safe_encoder(file_complete_evt)}\n\n"
            
            # Increment counter for successfully processed file (only once per file)
            if file_processed_successfully:
                processed_files += 1
        
        # Check if any files were successfully processed (after the loop)
        if processed_files == 0:
            no_files_msg = {'type': 'error', 'jobId': job_id, 'error': 'No files were successfully processed. Please check file paths and try again.'}
            try:
                await ws_manager.broadcast(job_id, no_files_msg)  # type: ignore
            except Exception:
                pass
            jobs_snapshot[job_id] = no_files_msg
            try:
                upsert_job(job_id, {"current_stage": "failed", "status": "failed", "error": no_files_msg.get("error")})
            except Exception:
                pass
            yield f"{safe_encoder(no_files_msg)}\n\n"
            return  # Exit early if no files processed
        
        done = {'type': 'complete', 'jobId': job_id, 'message': 'Refinement processing complete', 'memory_context': memory_manager.get_memory_context(request.user_id) if request.use_memory else {}}
        try:
            await ws_manager.broadcast(job_id, done)  # type: ignore
        except Exception:
            pass
        jobs_snapshot[job_id] = done
        try:
            upsert_job(job_id, {"current_stage": "completed", "progress": 100.0, "status": "completed", "result": {"ok": True}})
        except Exception:
            pass
        yield f"{safe_encoder(done)}\n\n"
        # Signal explicit done event for SSE consumers
        yield "event: done\ndata: {}\n\n"
    except Exception as e:
        error_msg = str(e).replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
        err3 = {'type': 'error', 'jobId': job_id, 'error': f'Stream processing failed: {error_msg}'}
        try:
            await ws_manager.broadcast(job_id, err3)  # type: ignore
        except Exception:
            pass
        jobs_snapshot[job_id] = err3
        try:
            upsert_job(job_id, {"current_stage": "failed", "status": "failed", "error": err3.get("error")})
        except Exception:
            pass
        yield f"{safe_encoder(err3)}\n\n"
        # Signal explicit error completion for SSE consumers
        yield "event: error\ndata: {}\n\n"
        log_exception("REFINEMENT_STREAM_FATAL", e)
    finally:
        # CRITICAL FIX: Always emit stream_end event to ensure frontend knows processing is done
        # This helps prevent indefinite "processing" states in the UI
        try:
            stream_end_evt = {'type': 'stream_end', 'jobId': job_id, 'message': 'Stream processing finished'}
            try:
                await ws_manager.broadcast(job_id, stream_end_evt)  # type: ignore
            except Exception:
                pass
            yield f"{safe_encoder(stream_end_evt)}\n\n"
            # Also emit SSE-style end event
            yield "event: stream_end\ndata: {}\n\n"
        except Exception as cleanup_error:
            logger.error(f"Failed to emit stream_end event: {cleanup_error}")
        
        # CRITICAL FIX: Clean up job-specific pipeline data to prevent memory leaks
        try:
            pipeline = get_pipeline()
            if pipeline and hasattr(pipeline, 'cleanup_job_data'):
                pipeline.cleanup_job_data(job_id)
                logger.debug(f"Cleaned up pipeline job data for job {job_id}")
        except Exception as cleanup_error:
            logger.debug(f"Failed to cleanup pipeline job data: {cleanup_error}")

async def run_job_background(req: RefinementRequest, job_id: str) -> None:
    try:
        # Set task timeout to prevent infinite processing
        # For large files, allow up to 1 hour (3600 seconds)
        timeout_seconds = 3600  # 1 hour max per job for large files
        
        # Broadcast initial job id for WS clients listening
        head = {'type': 'job', 'jobId': job_id}
        try:
            await ws_manager.broadcast(job_id, head)  # type: ignore
        except Exception as e:
            log_exception("WS_BROADCAST_ERROR", e)
        
        # Consume stream to drive processing while updating DB via _refine_stream side-effects
        async with asyncio.timeout(timeout_seconds):
            async for _ in _refine_stream(req, job_id):
                # stream frames ignored; DB + WS already handled
                await asyncio.sleep(0)  # yield control
                
    except asyncio.TimeoutError:
        # Job timed out
        log_exception("JOB_TIMEOUT", f"Job {job_id} timed out after {timeout_seconds} seconds")
        safe_upsert_job(job_id, {"status": "timeout", "error": f"Job timed out after {timeout_seconds} seconds"})
        safe_jobs_snapshot_set(job_id, {"type": "timeout", "jobId": job_id, "error": "Job timed out"})
        
    except Exception as e:
        # Log the error and update job status
        log_exception("BACKGROUND_JOB_ERROR", e)
        safe_upsert_job(job_id, {"status": "failed", "error": str(e)})
        safe_jobs_snapshot_set(job_id, {"type": "error", "jobId": job_id, "error": str(e)})
        
    finally:
        # Always clean up the task, regardless of success/failure/timeout
        safe_active_tasks_del(job_id)

@router.post("/run")
async def refine_run(request: RefinementRequest):
    logger.debug(f"REFINE_RUN: Received request with {len(request.files)} files")
    if not request.files:
        return JSONResponse({"error": "No files provided"}, status_code=400)
    logger.debug("REFINE_RUN: Checking pipeline...")
    try:
        if not get_pipeline():
            return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)
        logger.debug("REFINE_RUN: Pipeline OK")
    except Exception as e:
        logger.error(f"REFINE_RUN: Pipeline error: {e}")
        return JSONResponse({"error": f"Pipeline initialization failed: {str(e)}"}, status_code=500)
    job_id = str(uuid.uuid4())
    # CRITICAL FIX: Store job in MongoDB for persistence across serverless invocations
    # Also immediately store in snapshot for immediate availability
    initial_job_state = {
        "type": "job",
        "jobId": job_id,
        "status": "running",
        "progress": 0.0,
        "current_stage": "initializing"
    }
    safe_jobs_snapshot_set(job_id, initial_job_state)
    
    # CRITICAL FIX: Also store in in-memory database immediately for backward compatibility
    try:
        from app.core.database import upsert_job
        upsert_job(job_id, {
            "status": "running",
            "progress": 0.0,
            "current_stage": "initializing"
        })
    except Exception as e:
        logger.warning(f"Failed to store job in in-memory DB: {e}")
    
    try:
        if mongodb_db.is_connected():
            # Get file info for job creation
            file_name = request.files[0].get("name", "Unknown") if request.files else "Unknown"
            file_id = request.files[0].get("id", "unknown") if request.files else "unknown"
            user_id = getattr(request, 'user_id', None) or "default"
            success = mongodb_db.create_job(
                job_id=job_id,
                file_name=file_name,
                file_id=file_id,
                user_id=user_id,
                total_passes=request.passes,
                model=getattr(request, 'model', 'gpt-4'),
                metadata={"entropy_level": request.entropy_level, "aggressiveness": request.aggressiveness}
            )
            if not success:
                logger.warning(f"Failed to create job {job_id} in MongoDB")
        # Also store in in-memory for backward compatibility
        upsert_job(job_id, {"status": "running", "progress": 0.0, "current_stage": "initializing"})
    except Exception as e:
        logger.warning(f"Failed to create job in storage: {e}")
        # Don't fail - snapshot is already set, so job status will work
        pass
    async def event_gen():
        # Small preamble to nudge immediate flush on some clients/proxies
        yield ":ok\n\n"
        logger.debug(f"SSE[{job_id}] preamble sent")
        # Send initial job id event
        head = {'type': 'job', 'jobId': job_id}
        yield f"data: {safe_encoder(head)}\n\n"
        logger.debug(f"SSE[{job_id}] head sent")
        event_count = 0
        newline_pair = "\n\n"
        async for chunk in _refine_stream(request, job_id):
            # Preserve properly formatted SSE frames (data:, event:, comments starting with :) and otherwise wrap as data
            if chunk.startswith("data:") or chunk.startswith(":") or chunk.startswith("event:"):
                yield chunk if chunk.endswith(newline_pair) else f"{chunk}{newline_pair}"
            else:
                yield f"data: {chunk if chunk.endswith(newline_pair) else chunk + newline_pair}"
            event_count += 1
            if event_count % 10 == 0:
                logger.debug(f"SSE[{job_id}] events forwarded: {event_count}")
        # Explicit terminal events after stream completes normally
        final_event = { 'type': 'stream_end', 'jobId': job_id, 'total_events': event_count, 'message': 'Stream completed successfully' }
        yield f"data: {safe_encoder(final_event)}\n\n"
        logger.debug(f"SSE[{job_id}] stream_end sent")
        yield ": stream-complete\n\n"
    return EventSourceResponse(
        event_gen(), 
        ping=5,
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# =====================================================
# JOB ANALYTICS ENDPOINT
# =====================================================

@router.get("/job/{job_id}/analytics")
async def get_job_analytics(job_id: str):
    """
    Get detailed analytics for a specific job.
    
    Returns:
        - Total tokens in/out per pass
        - Cost estimate per pass and total
        - Chunk counts and failure rate
        - Time per stage (read/prep/refine/post/write)
        - Warnings and errors
    """
    try:
        # Get job stats from analytics store
        job_stats = analytics_store.get_job_stats(job_id)
        
        # Get additional info from job snapshot
        job_info = jobs_snapshot.get(job_id, {})
        
        # Get pipeline stats if available
        pipeline = get_pipeline()
        pipeline_stats = {}
        if pipeline:
            # Get chunk stats if tracked
            if hasattr(pipeline, '_current_chunk_progress'):
                pipeline_stats['chunkProgress'] = pipeline._current_chunk_progress.get(job_id, 0)
            if hasattr(pipeline, '_current_chunk_total'):
                pipeline_stats['chunkTotal'] = pipeline._current_chunk_total.get(job_id, 0)
            if hasattr(pipeline, '_current_chunk_failures'):
                pipeline_stats['chunkFailures'] = pipeline._current_chunk_failures.get(job_id, 0)
        
        # Calculate failure rate
        failure_rate = 0.0
        if pipeline_stats.get('chunkTotal', 0) > 0:
            failure_rate = pipeline_stats.get('chunkFailures', 0) / pipeline_stats['chunkTotal']
        
        response = {
            "jobId": job_id,
            "cost": {
                "total": job_stats.get("total_cost", 0.0),
                "perPass": job_stats.get("pass_costs", []),
                "remaining": job_stats.get("cost_remaining", 0.0),
                "limit": job_stats.get("cost_limit", 5.0)
            },
            "tokens": {
                "total": job_stats.get("total_tokens", 0),
                "remaining": job_stats.get("tokens_remaining", 0),
                "limit": job_stats.get("token_limit", 500000)
            },
            "chunks": {
                "processed": pipeline_stats.get('chunkProgress', 0),
                "total": pipeline_stats.get('chunkTotal', 0),
                "failures": pipeline_stats.get('chunkFailures', 0),
                "failureRate": failure_rate
            },
            "passes": {
                "completed": job_stats.get("pass_count", 0),
            },
            "status": job_info.get("type", "unknown"),
            "warnings": []
        }
        
        # Add warnings based on metrics
        if failure_rate > 0.2:
            response["warnings"].append(f"{int(failure_rate * 100)}% of chunks used fallback text")
        if job_stats.get("cost_remaining", 0) < 1.0:
            response["warnings"].append("Low cost budget remaining")
        if job_stats.get("tokens_remaining", 0) < 50000:
            response["warnings"].append("Low token budget remaining")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Failed to get job analytics: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get analytics: {str(e)}"}
        )


@router.get("/presets")
async def get_presets():
    """Get available refinement preset profiles."""
    return JSONResponse(content={
        "presets": {
            name: {
                "name": profile["name"],
                "description": profile["description"],
                "passes": profile["passes"],
                "entropyLevel": profile["entropy_level"],
                "estimatedCostPer1kTokens": profile["estimated_cost_per_1k_tokens"],
                "useModelTiering": profile["use_model_tiering"]
            }
            for name, profile in PRESET_PROFILES.items()
        }
    })


@router.get("/presets/{preset_name}")
async def get_preset(preset_name: str):
    """Get a specific preset profile with full configuration."""
    if preset_name not in PRESET_PROFILES:
        return JSONResponse(
            status_code=404,
            content={"error": f"Preset '{preset_name}' not found"}
        )
    
    profile = PRESET_PROFILES[preset_name]
    return JSONResponse(content={
        "preset": preset_name,
        **profile
    })


@router.get("/job/{job_id}/report")
async def download_job_report(job_id: str):
    """
    Download a processing report for a job.
    Returns JSON sidecar with passes, metrics, and warnings.
    """
    try:
        # Get job stats
        job_stats = analytics_store.get_job_stats(job_id)
        job_info = jobs_snapshot.get(job_id, {})
        
        # Get pipeline stats
        pipeline = get_pipeline()
        chunk_failures = 0
        chunk_total = 0
        if pipeline:
            if hasattr(pipeline, '_current_chunk_failures'):
                chunk_failures = pipeline._current_chunk_failures.get(job_id, 0)
            if hasattr(pipeline, '_current_chunk_total'):
                chunk_total = pipeline._current_chunk_total.get(job_id, 0)
        
        report = {
            "jobId": job_id,
            "generatedAt": datetime.now().isoformat(),
            "summary": {
                "totalCost": job_stats.get("total_cost", 0.0),
                "totalTokens": job_stats.get("total_tokens", 0),
                "passCount": job_stats.get("pass_count", 0),
                "status": job_info.get("type", "unknown")
            },
            "passes": [
                {"passNumber": i + 1, "cost": cost}
                for i, cost in enumerate(job_stats.get("pass_costs", []))
            ],
            "chunks": {
                "total": chunk_total,
                "failures": chunk_failures,
                "fallbackUsed": chunk_failures > 0
            },
            "warnings": [],
            "errors": []
        }
        
        # Add warnings
        if chunk_failures > 0:
            report["warnings"].append(f"{chunk_failures}/{chunk_total} chunks used fallback text")
        
        if job_info.get("type") == "error":
            report["errors"].append(job_info.get("error", "Unknown error"))
        
        from fastapi.responses import Response
        return Response(
            content=json.dumps(report, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="job_{job_id}_report.json"'
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to generate job report: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate report: {str(e)}"}
        )
