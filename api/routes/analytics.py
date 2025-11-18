"""
Analytics API routes.

This module handles all analytics-related endpoints including usage statistics,
cost tracking, and job metrics.
"""
from __future__ import annotations

import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from language_model import analytics_store
from core.database import list_jobs
from core.exceptions import ProcessingError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
async def get_analytics_summary() -> JSONResponse:
    """
    Get comprehensive analytics summary, including live OpenAI usage.
    
    Returns:
        JSONResponse with analytics data including:
        - Jobs statistics (total, completed, failed, running)
        - OpenAI usage (requests, tokens, costs)
        - Schema usage statistics
        - Performance metrics
    """
    try:
        logger.debug(f"Analytics endpoint called: requests={analytics_store.total_requests}, cost=${analytics_store.total_cost:.6f}")
        
        # Get all jobs from database
        jobs = list_jobs(1000)  # Get last 1000 jobs
        
        # Calculate basic metrics
        total_jobs = len(jobs)
        completed_jobs = [j for j in jobs if j.get("status") == "completed"]
        failed_jobs = [j for j in jobs if j.get("status") == "failed"]
        running_jobs = [j for j in jobs if j.get("status") == "running"]
        
        # Calculate performance metrics
        if completed_jobs:
            # Filter out jobs with missing metrics to avoid skewing averages
            jobs_with_metrics = [j for j in completed_jobs if j.get("metrics")]
            jobs_with_processing_time = [
                j for j in completed_jobs 
                if j.get("metrics", {}).get("processingTime", 0) > 0
            ]
            
            avg_change_percent = sum(
                j.get("metrics", {}).get("changePercent", 0) 
                for j in jobs_with_metrics
            ) / max(len(jobs_with_metrics), 1)
            
            avg_tension_percent = sum(
                j.get("metrics", {}).get("tensionPercent", 0) 
                for j in jobs_with_metrics
            ) / max(len(jobs_with_metrics), 1)
            
            avg_processing_time = sum(
                j.get("metrics", {}).get("processingTime", 0) 
                for j in jobs_with_processing_time
            ) / max(len(jobs_with_processing_time), 1)
            
            avg_risk_reduction = sum(
                j.get("metrics", {}).get("riskReduction", 0) 
                for j in jobs_with_metrics
            ) / max(len(jobs_with_metrics), 1)
        else:
            avg_change_percent = 0
            avg_tension_percent = 0
            avg_processing_time = 0
            avg_risk_reduction = 0
        
        # Recent activity (last 10 jobs)
        recent_activity = sorted(jobs, key=lambda x: x.get("created_at", 0), reverse=True)[:10]
        
        result: Dict[str, Any] = {
            "jobs": {
                "totalJobs": total_jobs,
                "completed": len(completed_jobs),
                "failed": len(failed_jobs),
                "running": len(running_jobs),
                "successRate": (len(completed_jobs) / total_jobs * 100) if total_jobs > 0 else 0,
                "performanceMetrics": {
                    "avgChangePercent": round(avg_change_percent, 2),
                    "avgTensionPercent": round(avg_tension_percent, 2),
                    "avgProcessingTime": round(avg_processing_time, 2),
                    "avgRiskReduction": round(avg_risk_reduction, 2),
                },
                "recentActivity": [
                    {
                        "id": job.get("id", "unknown"),
                        "fileName": job.get("fileName", "Unknown"),
                        "timestamp": datetime.fromtimestamp(job.get("created_at", 0)).isoformat(),
                        "status": job.get("status", "unknown"),
                        "action": f"Processing {'completed' if job.get('status') == 'completed' else 'failed' if job.get('status') == 'failed' else 'running'}",
                    }
                    for job in recent_activity
                ]
            },
            "openai": {
                "total_requests": analytics_store.total_requests,
                "total_tokens_in": analytics_store.total_tokens_in,
                "total_tokens_out": analytics_store.total_tokens_out,
                "total_cost": analytics_store.total_cost,
                "current_model": analytics_store.current_model,
                "last_24h": analytics_store.summary_last_24h(),
            },
            "schema_usage": analytics_store.get_schema_usage_stats()
        }
        
        logger.debug(f"Returning analytics: requests={result['openai']['total_requests']}, cost=${result['openai']['total_cost']:.6f}")
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Analytics summary error: {e}", exc_info=True)
        raise ProcessingError(
            message="Failed to generate analytics summary",
            details={"error": str(e)}
        )


@router.get("/test")
async def test_analytics() -> JSONResponse:
    """
    Test endpoint to verify analytics tracking works.
    
    Returns:
        JSONResponse with test analytics data
    """
    try:
        # Manually add some test data
        test_cost = analytics_store.add(100, 50, "gpt-4", "test-job-123")
        return JSONResponse({
            "message": "Test analytics added",
            "analytics_store": {
                "total_requests": analytics_store.total_requests,
                "total_tokens_in": analytics_store.total_tokens_in,
                "total_tokens_out": analytics_store.total_tokens_out,
                "total_cost": analytics_store.total_cost,
                "current_model": analytics_store.current_model,
            },
            "test_cost": test_cost
        })
    except Exception as e:
        logger.error(f"Test analytics error: {e}", exc_info=True)
        raise ProcessingError(
            message="Failed to add test analytics",
            details={"error": str(e)}
        )


