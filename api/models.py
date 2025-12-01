"""
Request and response models for the API.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class WorkflowType(str, Enum):
    """Available workflow types."""
    simple = "simple"
    evaluator = "evaluator"
    multi_stage = "multi_stage"
    analysis = "analysis"
    research = "research"
    transformation = "transformation"


class JobStatus(str, Enum):
    """Job execution status."""
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


# ============================================================================
# Session Management
# ============================================================================

class SessionCreateRequest(BaseModel):
    """Request to create a new session."""
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    preprocessing_enabled: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.0,
                "max_tokens": 8192
            }
        }


class SessionResponse(BaseModel):
    """Response containing session information."""
    session_id: str
    created_at: str
    config: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "created_at": "2025-01-15T10:30:00",
                "config": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
            }
        }


class SessionConfigUpdate(BaseModel):
    """Request to update session configuration."""
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 0.5,
                "max_tokens": 16384
            }
        }


# ============================================================================
# Task Execution
# ============================================================================

class TaskRequest(BaseModel):
    """Request to execute a task."""
    task: str = Field(..., description="Task description or query")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context")
    workflow: WorkflowType = Field(WorkflowType.simple, description="Workflow type")
    workflow_params: Optional[Dict[str, Any]] = Field(None, description="Workflow-specific parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task": "Analyze the logs for error patterns",
                "workflow": "simple",
                "context": {"priority": "high"}
            }
        }


class TaskResponse(BaseModel):
    """Response after submitting a task."""
    job_id: str
    session_id: str
    status: JobStatus
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_xyz789",
                "session_id": "sess_abc123",
                "status": "pending",
                "message": "Task submitted successfully"
            }
        }


# ============================================================================
# Job Status and Results
# ============================================================================

class JobStatusResponse(BaseModel):
    """Response with job status and results."""
    job_id: str
    session_id: str
    status: JobStatus
    progress: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_xyz789",
                "session_id": "sess_abc123",
                "status": "completed",
                "result": {
                    "content": "Found 42 error patterns...",
                    "tool_calls": [],
                    "history": []
                },
                "created_at": "2025-01-15T10:30:00",
                "completed_at": "2025-01-15T10:32:15",
                "duration_seconds": 135.2
            }
        }


# ============================================================================
# Preprocessing
# ============================================================================

class PreprocessingRequest(BaseModel):
    """Request to preprocess logs."""
    enable_embeddings: bool = False
    chunk_size: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "enable_embeddings": True,
                "chunk_size": 100
            }
        }


class PreprocessingResponse(BaseModel):
    """Response after preprocessing."""
    session_id: str
    status: str
    metadata: Dict[str, Any]
    tools_available: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "status": "completed",
                "metadata": {"total_lines": 1000, "error_count": 42},
                "tools_available": ["query_logs_from_sqlite_database", "search_similar_from_embeddings"]
            }
        }


# ============================================================================
# Health and Status
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    active_sessions: int
    active_jobs: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "active_sessions": 3,
                "active_jobs": 1
            }
        }


# ============================================================================
# Error Responses
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Session not found",
                "detail": "Session sess_abc123 does not exist or has expired"
            }
        }
