"""
FastAPI server for iExplain API.

Usage:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
    
Or:
    python -m api.server
"""

import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    SessionCreateRequest,
    SessionResponse,
    SessionConfigUpdate,
    TaskRequest,
    TaskResponse,
    JobStatusResponse,
    PreprocessingRequest,
    PreprocessingResponse,
    HealthResponse,
    ErrorResponse,
    JobStatus
)
from api.session_manager import SessionManager
from api.job_manager import JobManager
from api.config import default_config


# ============================================================================
# Global managers
# ============================================================================

session_manager: Optional[SessionManager] = None
job_manager: Optional[JobManager] = None


# ============================================================================
# Lifecycle management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    # Startup
    global session_manager, job_manager
    
    session_manager = SessionManager(
        ttl_seconds=default_config.session_ttl_seconds,
        max_sessions=default_config.max_sessions
    )
    
    job_manager = JobManager(
        max_concurrent_jobs=default_config.max_concurrent_jobs,
        job_timeout_seconds=default_config.job_timeout_seconds,
        job_retention_seconds=default_config.job_retention_seconds
    )
    
    print("iExplain API started")
    print(f"- Max sessions: {default_config.max_sessions}")
    print(f"- Max concurrent jobs: {default_config.max_concurrent_jobs}")
    
    yield
    
    # Shutdown
    if job_manager:
        job_manager.shutdown()
    
    print("iExplain API shutdown")


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(
    title="iExplain API",
    description="API for intent-based system management with explainability",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=default_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Error handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


# ============================================================================
# Health check
# ============================================================================

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        active_sessions=session_manager.count_sessions(),
        active_jobs=job_manager.count_jobs(JobStatus.running)
    )


# ============================================================================
# Session management
# ============================================================================

@app.post("/api/v1/sessions", response_model=SessionResponse, status_code=201)
async def create_session(request: SessionCreateRequest):
    """
    Create a new session.
    
    A session maintains state across multiple task executions,
    including configuration and preprocessing state.
    """
    try:
        session = session_manager.create_session(
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            preprocessing_enabled=request.preprocessing_enabled
        )
        
        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at.isoformat(),
            config=session.config
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sessions/{session_id}/config", response_model=SessionResponse)
async def update_session_config(session_id: str, request: SessionConfigUpdate):
    """
    Update session configuration.
    
    Configuration updates apply to all subsequent task executions
    in this session.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session.update_config(
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at.isoformat(),
            config=session.config
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """
    Delete a session and cleanup resources.
    
    All jobs associated with the session remain accessible
    until they expire.
    """
    if not session_manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session information."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
        config=session.config
    )


# ============================================================================
# Task execution
# ============================================================================

@app.post("/api/v1/sessions/{session_id}/tasks", response_model=TaskResponse, status_code=202)
async def submit_task(session_id: str, request: TaskRequest):
    """
    Submit a task for asynchronous execution.
    
    Returns immediately with a job ID. Use the job status endpoint
    to poll for results.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        job = job_manager.submit_job(
            session_id=session_id,
            supervisor=session.supervisor,
            task=request.task,
            workflow_type=request.workflow.value,
            workflow_params=request.workflow_params,
            context=request.context
        )
        
        return TaskResponse(
            job_id=job.job_id,
            session_id=session_id,
            status=JobStatus.pending,
            message="Task submitted successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Job status and results
# ============================================================================

@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get job status and results.
    
    Poll this endpoint to check job progress and retrieve results
    once completed.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(**job.to_dict())


@app.delete("/api/v1/jobs/{job_id}", status_code=204)
async def cancel_job(job_id: str):
    """
    Cancel a pending job.
    
    Only works for jobs that haven't started execution yet.
    """
    if not job_manager.cancel_job(job_id):
        raise HTTPException(
            status_code=400,
            detail="Job cannot be cancelled (already running or completed)"
        )


# ============================================================================
# Preprocessing
# ============================================================================

@app.post("/api/v1/sessions/{session_id}/preprocessing", response_model=PreprocessingResponse)
async def setup_preprocessing(
    session_id: str,
    file: UploadFile = File(...),
    enable_embeddings: bool = False,
    chunk_size: Optional[int] = None
):
    """
    Upload log file and setup preprocessing.
    
    Enables SQL queries and optionally semantic search on logs.
    Preprocessing state is maintained for the session.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save uploaded file
    workspace_data = Path("workspace/data")
    workspace_data.mkdir(parents=True, exist_ok=True)
    
    log_path = workspace_data / file.filename
    
    try:
        with open(log_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Setup preprocessing
        metadata = session.setup_preprocessing(
            log_path=log_path,
            enable_embeddings=enable_embeddings,
            chunk_size=chunk_size
        )
        
        # Get available tools
        tools_available = []
        if session.preprocessor:
            tools_available = list(session.preprocessor.get_all_tools().keys())
        
        return PreprocessingResponse(
            session_id=session_id,
            status="completed",
            metadata=metadata,
            tools_available=tools_available
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Background cleanup
# ============================================================================

@app.post("/api/v1/admin/cleanup", status_code=200)
async def force_cleanup(background_tasks: BackgroundTasks):
    """
    Force cleanup of expired sessions and old jobs.
    
    This is automatically done periodically, but can be triggered
    manually for testing or maintenance.
    """
    def cleanup():
        sessions_cleaned = session_manager.cleanup_all()
        jobs_cleaned = job_manager.cleanup_all()
        return sessions_cleaned, jobs_cleaned
    
    background_tasks.add_task(cleanup)
    
    return {
        "status": "cleanup scheduled",
        "message": "Cleanup will run in background"
    }


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Run the API server."""
    uvicorn.run(
        "api.server:app",
        host=default_config.host,
        port=default_config.port,
        workers=default_config.workers,
        log_level=default_config.log_level.lower()
    )


if __name__ == "__main__":
    main()
