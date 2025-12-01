"""
API configuration.
"""

from typing import Optional
from pydantic import BaseModel


class APIConfig(BaseModel):
    """API server configuration."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Session management
    session_ttl_seconds: int = 3600  # 1 hour
    max_sessions: int = 100
    cleanup_interval_seconds: int = 300  # 5 minutes
    
    # Job management
    max_concurrent_jobs: int = 4
    job_timeout_seconds: int = 600  # 10 minutes
    job_retention_seconds: int = 3600  # 1 hour
    
    # CORS
    cors_origins: list = ["*"]
    
    # Logging
    log_level: str = "INFO"


# Default configuration
default_config = APIConfig()
