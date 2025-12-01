"""
Job manager for asynchronous task execution.
"""

import time
import uuid
from typing import Dict, Optional, Any, Callable
from datetime import datetime
from threading import Lock, Thread
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows import (
    SimpleWorkflow,
    EvaluatorWorkflow,
    MultiStageWorkflow,
    PredefinedMultiStageWorkflow
)


class JobStatus(str, Enum):
    """Job execution status."""
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Job:
    """
    Represents an asynchronous job.
    """
    
    def __init__(
        self,
        job_id: str,
        session_id: str,
        task: str,
        workflow_type: str,
        workflow_params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.job_id = job_id
        self.session_id = session_id
        self.task = task
        self.workflow_type = workflow_type
        self.workflow_params = workflow_params or {}
        self.context = context
        
        self.status = JobStatus.pending
        self.progress = None
        self.result = None
        self.error = None
        
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.duration_seconds = None
        
        self.future: Optional[Future] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dict for API response."""
        return {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds
        }
    
    def mark_running(self):
        """Mark job as running."""
        self.status = JobStatus.running
        self.started_at = datetime.now()
    
    def mark_completed(self, result: Dict[str, Any]):
        """Mark job as completed."""
        self.status = JobStatus.completed
        self.result = result
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def mark_failed(self, error: str):
        """Mark job as failed."""
        self.status = JobStatus.failed
        self.error = error
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


class JobManager:
    """
    Manages asynchronous job execution.
    """
    
    def __init__(
        self,
        max_concurrent_jobs: int = 4,
        job_timeout_seconds: int = 600,
        job_retention_seconds: int = 3600
    ):
        self.jobs: Dict[str, Job] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_timeout_seconds = job_timeout_seconds
        self.job_retention_seconds = job_retention_seconds
        
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.lock = Lock()
        self.last_cleanup = time.time()
    
    def submit_job(
        self,
        session_id: str,
        supervisor: Any,
        task: str,
        workflow_type: str = "simple",
        workflow_params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Job:
        """
        Submit a new job for execution.
        
        Args:
            session_id: Session ID
            supervisor: Supervisor instance from session
            task: Task description
            workflow_type: Workflow type (simple, evaluator, multi_stage, etc.)
            workflow_params: Workflow-specific parameters
            context: Optional context
            
        Returns:
            Job instance
        """
        with self.lock:
            # Maybe cleanup old jobs
            self._maybe_cleanup()
            
            # Generate job ID
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            
            # Create job
            job = Job(
                job_id=job_id,
                session_id=session_id,
                task=task,
                workflow_type=workflow_type,
                workflow_params=workflow_params,
                context=context
            )
            
            self.jobs[job_id] = job
            
            # Submit to executor
            future = self.executor.submit(
                self._execute_job,
                job,
                supervisor
            )
            job.future = future
            
            return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with self.lock:
            return self.jobs.get(job_id)
    
    def list_jobs(self, session_id: Optional[str] = None) -> list:
        """List all jobs, optionally filtered by session."""
        with self.lock:
            if session_id:
                return [j for j in self.jobs.values() if j.session_id == session_id]
            return list(self.jobs.values())
    
    def count_jobs(self, status: Optional[JobStatus] = None) -> int:
        """Count jobs, optionally filtered by status."""
        with self.lock:
            if status:
                return sum(1 for j in self.jobs.values() if j.status == status)
            return len(self.jobs)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job (if still pending)."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job.status == JobStatus.pending:
                if job.future:
                    job.future.cancel()
                job.mark_failed("Cancelled by user")
                return True
            return False
    
    def _execute_job(self, job: Job, supervisor: Any):
        """Execute a job (runs in thread pool)."""
        try:
            job.mark_running()
            
            # Create workflow
            workflow = self._create_workflow(
                supervisor,
                job.workflow_type,
                job.workflow_params
            )
            
            # Execute
            result = workflow.run(job.task, job.context)
            
            # Mark completed
            job.mark_completed(result)
            
        except Exception as e:
            job.mark_failed(str(e))
    
    def _create_workflow(
        self,
        supervisor: Any,
        workflow_type: str,
        workflow_params: Optional[Dict[str, Any]] = None
    ):
        """Create workflow instance."""
        params = workflow_params or {}
        
        if workflow_type == "simple":
            return SimpleWorkflow(supervisor)
        
        elif workflow_type == "evaluator":
            max_iterations = params.get("max_iterations", 3)
            return EvaluatorWorkflow(
                supervisor,
                max_iterations=max_iterations,
                verbose=False
            )
        
        elif workflow_type == "multi_stage":
            stages = params.get("stages", [])
            if not stages:
                raise ValueError("multi_stage workflow requires 'stages' parameter")
            return MultiStageWorkflow(
                supervisor,
                stages=stages,
                verbose=False
            )
        
        elif workflow_type == "analysis":
            return PredefinedMultiStageWorkflow.analysis_workflow(supervisor)
        
        elif workflow_type == "research":
            return PredefinedMultiStageWorkflow.research_workflow(supervisor)
        
        elif workflow_type == "transformation":
            return PredefinedMultiStageWorkflow.transformation_workflow(supervisor)
        
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    def _maybe_cleanup(self):
        """Cleanup old completed/failed jobs."""
        now = time.time()
        
        # Only cleanup every minute
        if now - self.last_cleanup < 60:
            return
        
        self.last_cleanup = now
        
        # Find old jobs
        cutoff = datetime.now().timestamp() - self.job_retention_seconds
        old_jobs = []
        
        for job_id, job in self.jobs.items():
            if job.status in [JobStatus.completed, JobStatus.failed]:
                if job.completed_at and job.completed_at.timestamp() < cutoff:
                    old_jobs.append(job_id)
        
        # Remove old jobs
        for job_id in old_jobs:
            del self.jobs[job_id]
    
    def cleanup_all(self):
        """Force cleanup of old jobs."""
        with self.lock:
            cutoff = datetime.now().timestamp() - self.job_retention_seconds
            old_jobs = []
            
            for job_id, job in self.jobs.items():
                if job.status in [JobStatus.completed, JobStatus.failed]:
                    if job.completed_at and job.completed_at.timestamp() < cutoff:
                        old_jobs.append(job_id)
            
            for job_id in old_jobs:
                del self.jobs[job_id]
            
            return len(old_jobs)
    
    def shutdown(self):
        """Shutdown executor."""
        self.executor.shutdown(wait=True)
