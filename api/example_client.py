#!/usr/bin/env python3
"""
Example client for iExplain API.

Demonstrates basic usage:
1. Create session
2. Upload and preprocess logs (optional)
3. Submit task
4. Poll for results
5. Cleanup

Usage:
    python api/example_client.py [--log-file logs/app.log]
"""

import argparse
import time
import requests
from pathlib import Path
from typing import Optional


class iExplainClient:
    """Simple client for iExplain API."""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session_id: Optional[str] = None
    
    def create_session(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 8192
    ) -> str:
        """Create a new session."""
        response = requests.post(
            f"{self.base_url}/sessions",
            json={
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        response.raise_for_status()
        
        data = response.json()
        self.session_id = data["session_id"]
        print(f"✓ Session created: {self.session_id}")
        return self.session_id
    
    def upload_logs(
        self,
        log_file: Path,
        enable_embeddings: bool = False,
        chunk_size: Optional[int] = None
    ) -> dict:
        """Upload and preprocess logs."""
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")
        
        print(f"Uploading logs: {log_file}")
        
        with open(log_file, "rb") as f:
            files = {"file": (log_file.name, f)}
            data = {
                "enable_embeddings": str(enable_embeddings).lower()
            }
            if chunk_size:
                data["chunk_size"] = str(chunk_size)
            
            response = requests.post(
                f"{self.base_url}/sessions/{self.session_id}/preprocessing",
                files=files,
                data=data
            )
            response.raise_for_status()
        
        result = response.json()
        print(f"✓ Preprocessing complete")
        print(f"  Tools available: {', '.join(result['tools_available'])}")
        return result
    
    def submit_task(
        self,
        task: str,
        workflow: str = "simple",
        workflow_params: Optional[dict] = None,
        context: Optional[dict] = None
    ) -> str:
        """Submit a task for execution."""
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")
        
        payload = {
            "task": task,
            "workflow": workflow
        }
        
        if workflow_params:
            payload["workflow_params"] = workflow_params
        if context:
            payload["context"] = context
        
        print(f"\nSubmitting task: {task[:60]}...")
        
        response = requests.post(
            f"{self.base_url}/sessions/{self.session_id}/tasks",
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        job_id = data["job_id"]
        print(f"✓ Job submitted: {job_id}")
        return job_id
    
    def get_job_status(self, job_id: str) -> dict:
        """Get job status and results."""
        response = requests.get(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_job(self, job_id: str, poll_interval: int = 2, timeout: int = 600) -> dict:
        """Wait for job to complete and return results."""
        print(f"\nWaiting for job to complete...")
        
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job did not complete within {timeout} seconds")
            
            status = self.get_job_status(job_id)
            
            if status["status"] == "completed":
                duration = status.get("duration_seconds", 0)
                print(f"✓ Job completed in {duration:.1f}s")
                return status
            
            elif status["status"] == "failed":
                error = status.get("error", "Unknown error")
                raise RuntimeError(f"Job failed: {error}")
            
            elif status["status"] == "running":
                elapsed = time.time() - start_time
                print(f"  Running... ({elapsed:.0f}s elapsed)")
            
            time.sleep(poll_interval)
    
    def delete_session(self):
        """Delete the session."""
        if not self.session_id:
            return
        
        response = requests.delete(f"{self.base_url}/sessions/{self.session_id}")
        response.raise_for_status()
        print(f"\n✓ Session deleted: {self.session_id}")
        self.session_id = None
    
    def health_check(self) -> dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


def main():
    """Run example client."""
    parser = argparse.ArgumentParser(description="iExplain API Example Client")
    parser.add_argument("--log-file", type=Path, help="Log file to analyze")
    parser.add_argument("--embeddings", action="store_true", help="Enable embeddings")
    parser.add_argument("--workflow", default="simple", help="Workflow type")
    parser.add_argument("--base-url", default="http://localhost:8000/api/v1", help="API base URL")
    args = parser.parse_args()
    
    client = iExplainClient(base_url=args.base_url)
    
    try:
        # Check health
        print("Checking API health...")
        health = client.health_check()
        print(f"✓ API is {health['status']}")
        print(f"  Active sessions: {health['active_sessions']}")
        print(f"  Active jobs: {health['active_jobs']}")
        print()
        
        # Create session
        client.create_session()
        
        # Upload logs if provided
        if args.log_file:
            if not args.log_file.exists():
                print(f"Error: Log file not found: {args.log_file}")
                return
            
            client.upload_logs(
                log_file=args.log_file,
                enable_embeddings=args.embeddings
            )
        
        # Submit task
        if args.log_file:
            task = f"Analyze the logs in {args.log_file.name} for error patterns and anomalies. Provide a summary of findings."
        else:
            task = "List the tools you have available and explain what you can do."
        
        job_id = client.submit_task(task, workflow=args.workflow)
        
        # Wait for results
        result = client.wait_for_job(job_id)
        
        # Display results
        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(result["result"]["content"])
        print("=" * 70)
        
        # Show statistics
        print(f"\nTool calls: {len(result['result'].get('tool_calls', []))}")
        if result["result"].get("tool_calls"):
            print("Tools used:")
            for tc in result["result"]["tool_calls"]:
                print(f"  - {tc['name']}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        client.delete_session()


if __name__ == "__main__":
    main()
