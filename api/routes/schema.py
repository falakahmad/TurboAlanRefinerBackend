"""
Schema API routes.

This module handles schema information endpoints.
"""
from __future__ import annotations

from typing import Dict, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from prompt_schema import ADVANCED_COMMANDS

router = APIRouter(prefix="/schema", tags=["schema"])


@router.get("")
async def get_schema_info() -> JSONResponse:
    """
    Get schema information including commands, descriptions, and categories.
    
    Returns:
        JSONResponse with schema information:
        - commands: All available schema commands
        - descriptions: Command descriptions
        - categories: Commands grouped by category
    """
    schema_info: Dict[str, Any] = {
        "commands": ADVANCED_COMMANDS,
        "descriptions": {k: v["description"] for k, v in ADVANCED_COMMANDS.items()},
        "categories": {
            "processing": [
                "microstructure_control",
                "macrostructure_analysis",
                "anti_scanner_techniques"
            ],
            "optimization": [
                "entropy_management",
                "semantic_tone_tuning"
            ],
            "safety": [
                "formatting_safeguards",
                "refiner_control"
            ],
            "analysis": [
                "history_analysis",
                "annotation_mode",
                "humanize_academic"
            ]
        }
    }
    return JSONResponse(schema_info)


