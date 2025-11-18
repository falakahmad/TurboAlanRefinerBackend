"""
Jobs API routes.

This module handles job management endpoints including job listing, status checking,
queueing, cancellation, and retry operations.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from core.database import list_jobs, get_job, upsert_job
from core.exceptions import NotFoundError, ProcessingError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
async def jobs_list() -> JSONResponse:
    """
    List all jobs.
    
    Returns:
        JSONResponse with list of jobs (limited to 100 most recent)
    """
    try:
        jobs = list_jobs(100)
        return JSONResponse({"jobs": jobs})
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise ProcessingError(
            message="Failed to retrieve jobs list",
            details={"error": str(e)}
        )


@router.get("/{job_id}/status")
async def get_job_status(job_id: str) -> JSONResponse:
    """
    Get status of a specific job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        JSONResponse with job status information
        
    Raises:
        NotFoundError: If job is not found
    """
    try:
        job = get_job(job_id)
        if not job:
            raise NotFoundError("Job", job_id)
        
        # Convert job to dict format
        job_dict: Dict[str, Any] = {
            "id": job.id,
            "status": job.status,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "error_message": job.error_message,
            "result": job.result,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "completed_at": job.completed_at,
        }
        
        return JSONResponse(job_dict)
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}", exc_info=True)
        raise ProcessingError(
            message="Failed to retrieve job status",
            details={"job_id": job_id, "error": str(e)}
        )


@router.post("/queue")
async def queue_job(request: Dict[str, Any]) -> JSONResponse:
    """
    Queue a new job for processing.
    
    Args:
        request: Job request data
        
    Returns:
        JSONResponse with queued job information
    """
    try:
        # Implementation would go here
        # This is a placeholder for the actual queue logic
        return JSONResponse({"message": "Job queued", "job_id": "placeholder"})
    except Exception as e:
        logger.error(f"Failed to queue job: {e}", exc_info=True)
        raise ProcessingError(
            message="Failed to queue job",
            details={"error": str(e)}
        )


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> JSONResponse:
    """
    Cancel a running job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        JSONResponse with cancellation status
        
    Raises:
        NotFoundError: If job is not found
    """
    try:
        job = get_job(job_id)
        if not job:
            raise NotFoundError("Job", job_id)
        
        # Update job status to cancelled
        upsert_job(job_id, {"status": "cancelled"})
        
        return JSONResponse({"message": "Job cancelled", "job_id": job_id})
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}", exc_info=True)
        raise ProcessingError(
            message="Failed to cancel job",
            details={"job_id": job_id, "error": str(e)}
        )


@router.post("/{job_id}/retry")
async def retry_job(job_id: str) -> JSONResponse:
    """
    Retry a failed job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        JSONResponse with retry status
        
    Raises:
        NotFoundError: If job is not found
    """
    try:
        job = get_job(job_id)
        if not job:
            raise NotFoundError("Job", job_id)
        
        # Reset job status for retry
        upsert_job(job_id, {"status": "pending", "error_message": None})
        
        return JSONResponse({"message": "Job queued for retry", "job_id": job_id})
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to retry job: {e}", exc_info=True)
        raise ProcessingError(
            message="Failed to retry job",
            details={"job_id": job_id, "error": str(e)}
        )


