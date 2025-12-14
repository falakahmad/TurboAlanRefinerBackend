from __future__ import annotations
import os
import asyncio
import json
import uuid
import time
from typing import AsyncGenerator, Dict, Any, List, Optional
from pathlib import Path

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

router = APIRouter()
logger = get_logger('api.refine')

MAX_REFINEMENT_PASSES = 10
MAX_HEURISTICS_SIZE = 1024 * 1024

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
    aggressiveness: str = "Auto"
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
    
    # First, try the exact file_id
    if file_id in uploaded_files:
        file_path = uploaded_files[file_id].get("temp_path") or uploaded_files[file_id].get("path")
    
    # If not found, try looking up by drive_id or other identifiers
    if not file_path:
        # Check if file_info has a drive_id that matches any uploaded file
        drive_id = file_info.get("driveId") or file_info.get("drive_id")
        if drive_id:
            for stored_file_id, stored_info in uploaded_files.items():
                if stored_info.get("drive_id") == drive_id or stored_info.get("id") == drive_id:
                    file_path = stored_info.get("temp_path") or stored_info.get("path")
                    logger.debug(f"Found file by drive_id: {drive_id} -> {file_path}")
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

async def _check_infinite_recursion_risk(current_text: str, original_text: str, pass_num: int, file_id: str, job_id: str) -> bool:
    """Check for infinite recursion risk and return True if should stop"""
    if pass_num > 1:
        # Check for exact duplicates before processing
        if current_text == original_text and pass_num > 2:
            logger.warning(f"Pass {pass_num} would process identical text, stopping to prevent infinite recursion")
            warning_msg = {'type': 'warning', 'jobId': job_id, 'fileId': file_id, 'pass': pass_num, 'message': 'Identical text detected, stopping refinement'}
            try:
                await ws_manager.broadcast(job_id, warning_msg)  # type: ignore
            except Exception:
                pass
            safe_jobs_snapshot_set(job_id, warning_msg)
            return True
        
        # Check for minimal changes in previous passes
        if pass_num > 3:
            # Calculate similarity with original text
            import difflib
            similarity = difflib.SequenceMatcher(None, original_text, current_text).ratio()
            if similarity > 0.99:  # 99% similar
                logger.warning(f"Pass {pass_num} shows minimal changes from original, stopping to prevent infinite recursion")
                warning_msg = {'type': 'warning', 'jobId': job_id, 'fileId': file_id, 'pass': pass_num, 'message': f'Minimal changes detected ({similarity:.1%} similarity), stopping refinement'}
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
        
        # Run the pipeline pass in executor to avoid blocking event loop
        logger.debug(f"About to call pipeline.run_pass for pass {pass_num}")
        loop = asyncio.get_event_loop()
        ps, rr, ft = await loop.run_in_executor(
            None,  # Use default ThreadPoolExecutor
            lambda: pipeline.run_pass(
                input_path=file_path,
                pass_index=pass_num,
                prev_final_text=current_text,
                entropy_level=request.entropy_level,
                output_sink=output_sink,
                drive_title_base=Path(file_path).stem,
                heuristics_overrides=request.heuristics,
                job_id=job_id
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
        
    except Exception as e:
        error_msg = str(e).replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
        err2 = {'type': 'error', 'jobId': job_id, 'fileId': file_id, 'pass': pass_num, 'error': error_msg}
        try:
            await ws_manager.broadcast(job_id, err2)  # type: ignore
        except Exception:
            pass
        safe_jobs_snapshot_set(job_id, err2)
        try:
            upsert_job(job_id, {"current_stage": "error", "status": "failed", "error": err2.get("error")})
        except Exception:
            pass
        logger.error(f"Pass {pass_num} failed: {e}")
        return False, current_text, {}, None, None

async def _refine_stream(request: RefinementRequest, job_id: str) -> AsyncGenerator[str, None]:
    logger.debug(f"Starting refinement stream for job {job_id}")
    
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
                    
                    # Yield the read completion message
                    msg = {'type': 'stage_update', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'stage': 'read', 'status': 'completed', 'message': f'Read {len(original_text)} characters'}
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
                if await _check_infinite_recursion_risk(current_text, original_text, pass_num, file_id, job_id):
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
                    
                    # Get the final text from refined_text
                    ft = refined_text
                    
                    # If pipeline state is None (error case), skip stage updates
                    if ps is None:
                        logger.error(f"Pipeline state is None for pass {pass_num}, skipping stage updates")
                        # Don't break - continue to next pass or emit error
                        error_evt = {'type': 'error', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'pass': pass_num, 'error': 'Pipeline state is None - pass may have failed'}
                        try:
                            await ws_manager.broadcast(job_id, error_evt)  # type: ignore
                        except Exception:
                            pass
                        jobs_snapshot[job_id] = error_evt
                        yield f"{safe_encoder(error_evt)}\n\n"
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
                    if pass_num > 1:
                        # Check for exact duplicates
                        if refined_text == current_text:
                            logger.warning(f"Pass {pass_num} produced identical text, stopping to prevent infinite recursion")
                            warning_msg = {'type': 'warning', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'pass': pass_num, 'message': 'Pass produced identical text, stopping refinement'}
                            try:
                                await ws_manager.broadcast(job_id, warning_msg)  # type: ignore
                            except Exception:
                                pass
                            jobs_snapshot[job_id] = warning_msg
                            yield f"{safe_encoder(warning_msg)}\n\n"
                            break  # Stop processing this file
                        
                        # Check for minimal changes (less than 0.1% difference)
                        import difflib
                        similarity = difflib.SequenceMatcher(None, current_text, ft).ratio()
                        if similarity > 0.999:  # 99.9% similar
                            logger.warning(f"Pass {pass_num} produced minimal changes ({similarity:.3f} similarity), stopping to prevent infinite recursion")
                            warning_msg = {'type': 'warning', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'pass': pass_num, 'message': f'Pass produced minimal changes ({similarity:.1%} similarity), stopping refinement'}
                            try:
                                await ws_manager.broadcast(job_id, warning_msg)  # type: ignore
                            except Exception:
                                pass
                            jobs_snapshot[job_id] = warning_msg
                            yield f"{safe_encoder(warning_msg)}\n\n"
                            break  # Stop processing this file
                        
                        # Check for diminishing returns (changes getting smaller)
                        if pass_num > 2:
                            prev_length = len(current_text)
                            current_length = len(ft)
                            change_ratio = abs(current_length - prev_length) / max(prev_length, 1)
                            if change_ratio < 0.001:  # Less than 0.1% change
                                logger.warning(f"Pass {pass_num} shows diminishing returns ({change_ratio:.4f} change ratio), stopping refinement")
                                warning_msg = {'type': 'warning', 'jobId': job_id, 'fileId': file_id, 'fileName': file_name, 'pass': pass_num, 'message': f'Diminishing returns detected ({change_ratio:.2%} change), stopping refinement'}
                                try:
                                    await ws_manager.broadcast(job_id, warning_msg)  # type: ignore
                                except Exception:
                                    pass
                                jobs_snapshot[job_id] = warning_msg
                                yield f"{safe_encoder(warning_msg)}\n\n"
                                break  # Stop processing this file
                    
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
                    try:
                        upsert_job(job_id, {"current_stage": "error", "status": "failed", "error": err2.get("error")})
                    except Exception:
                        pass
                    yield f"{safe_encoder(err2)}\n\n"
                    log_exception("REFINEMENT_STREAM_ERROR", e)
            
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


