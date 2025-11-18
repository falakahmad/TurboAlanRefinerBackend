"""
Environment variable validation and configuration.

This module validates required environment variables on startup and provides
clear error messages for missing or invalid configuration.
"""
from __future__ import annotations

import os
from typing import List, Dict, Optional, Any
from core.exceptions import ConfigurationError


class EnvVar:
    """Represents an environment variable with validation rules."""
    
    def __init__(
        self,
        name: str,
        required: bool = True,
        default: Optional[str] = None,
        validator: Optional[callable] = None,
        description: Optional[str] = None
    ):
        """
        Initialize an environment variable definition.
        
        Args:
            name: Environment variable name
            required: Whether the variable is required
            default: Default value if not set
            validator: Optional validation function
            description: Human-readable description
        """
        self.name = name
        self.required = required
        self.default = default
        self.validator = validator
        self.description = description
    
    def get_value(self) -> Optional[str]:
        """Get the environment variable value."""
        value = os.getenv(self.name, self.default)
        if self.required and not value:
            raise ConfigurationError(
                f"Required environment variable '{self.name}' is not set",
                config_key=self.name,
                details={"description": self.description}
            )
        if value and self.validator:
            try:
                self.validator(value)
            except Exception as e:
                raise ConfigurationError(
                    f"Invalid value for environment variable '{self.name}': {str(e)}",
                    config_key=self.name,
                    details={"description": self.description}
                )
        return value


def validate_required_env_vars() -> Dict[str, Optional[str]]:
    """
    Validate all required environment variables on startup.
    
    Returns:
        Dictionary of validated environment variables
        
    Raises:
        ConfigurationError: If required variables are missing or invalid
    """
    env_vars = {
        "OPENAI_API_KEY": EnvVar(
            "OPENAI_API_KEY",
            required=True,
            description="OpenAI API key for language model access"
        ),
        "GOOGLE_SERVICE_ACCOUNT_JSON": EnvVar(
            "GOOGLE_SERVICE_ACCOUNT_JSON",
            required=False,
            description="Google service account JSON credentials (for Google Drive integration)"
        ),
        "GOOGLE_SERVICE_ACCOUNT_FILE": EnvVar(
            "GOOGLE_SERVICE_ACCOUNT_FILE",
            required=False,
            description="Path to Google service account JSON file"
        ),
        "REFINER_OUTPUT_DIR": EnvVar(
            "REFINER_OUTPUT_DIR",
            required=False,
            description="Output directory for refined files"
        ),
        "BACKEND_API_KEY": EnvVar(
            "BACKEND_API_KEY",
            required=False,
            description="Optional API key for endpoint protection"
        ),
        "DEBUG": EnvVar(
            "DEBUG",
            required=False,
            default="false",
            description="Enable debug logging (true/false)"
        ),
    }
    
    validated: Dict[str, Optional[str]] = {}
    errors: List[str] = []
    
    for name, env_var in env_vars.items():
        try:
            validated[name] = env_var.get_value()
        except ConfigurationError as e:
            errors.append(str(e))
    
    if errors:
        raise ConfigurationError(
            "Environment variable validation failed",
            details={"errors": errors}
        )
    
    return validated


def get_env_summary() -> Dict[str, Any]:
    """
    Get a summary of environment variable configuration (without sensitive values).
    
    Returns:
        Dictionary with configuration summary
    """
    summary = {
        "OPENAI_API_KEY": "set" if os.getenv("OPENAI_API_KEY") else "not set",
        "GOOGLE_SERVICE_ACCOUNT_JSON": "set" if os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") else "not set",
        "GOOGLE_SERVICE_ACCOUNT_FILE": os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "not set"),
        "REFINER_OUTPUT_DIR": os.getenv("REFINER_OUTPUT_DIR", "default"),
        "BACKEND_API_KEY": "set" if os.getenv("BACKEND_API_KEY") else "not set",
        "DEBUG": os.getenv("DEBUG", "false"),
        "VERCEL": "true" if os.getenv("VERCEL") else "false",
    }
    return summary


