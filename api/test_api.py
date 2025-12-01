#!/usr/bin/env python3
"""
Test script for iExplain API.

Verifies that all endpoints are working correctly.

Usage:
    python api/test_api.py [--base-url http://localhost:8000/api/v1]
"""

import argparse
import sys
import time
import requests
from pathlib import Path


class APITester:
    """Test suite for iExplain API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session_id = None
        self.tests_passed = 0
        self.tests_failed = 0
    
    def test(self, name: str, func):
        """Run a test and track results."""
        print(f"\n{'='*70}")
        print(f"TEST: {name}")
        print(f"{'='*70}")
        
        try:
            func()
            print(f"✓ PASSED")
            self.tests_passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            self.tests_failed += 1
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        
        data = response.json()
        assert data["status"] == "healthy", "API not healthy"
        assert "version" in data, "Missing version"
        assert "active_sessions" in data, "Missing active_sessions"
        assert "active_jobs" in data, "Missing active_jobs"
        
        print(f"Health: {data['status']}")
        print(f"Version: {data['version']}")
        print(f"Active sessions: {data['active_sessions']}")
        print(f"Active jobs: {data['active_jobs']}")
    
    def test_create_session(self):
        """Test session creation."""
        response = requests.post(
            f"{self.base_url}/sessions",
            json={
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.0
            }
        )
        response.raise_for_status()
        
        data = response.json()
        assert "session_id" in data, "Missing session_id"
        assert "created_at" in data, "Missing created_at"
        assert "config" in data, "Missing config"
        
        self.session_id = data["session_id"]
        print(f"Session created: {self.session_id}")
        print(f"Config: {data['config']}")
    
    def test_get_session(self):
        """Test getting session info."""
        assert self.session_id, "No session created"
        
        response = requests.get(f"{self.base_url}/sessions/{self.session_id}")
        response.raise_for_status()
        
        data = response.json()
        assert data["session_id"] == self.session_id, "Session ID mismatch"
        assert "config" in data, "Missing config"
        
        print(f"Session retrieved: {self.session_id}")
    
    def test_update_session_config(self):
        """Test updating session configuration."""
        assert self.session_id, "No session created"
        
        response = requests.post(
            f"{self.base_url}/sessions/{self.session_id}/config",
            json={
                "temperature": 0.5,
                "max_tokens": 16384
            }
        )
        response.raise_for_status()
        
        data = response.json()
        assert data["config"]["temperature"] == 0.5, "Temperature not updated"
        assert data["config"]["max_tokens"] == 16384, "Max tokens not updated"
        
        print(f"Config updated: temp={data['config']['temperature']}, max_tokens={data['config']['max_tokens']}")
    
    def test_submit_simple_task(self):
        """Test submitting a simple task."""
        assert self.session_id, "No session created"
        
        response = requests.post(
            f"{self.base_url}/sessions/{self.session_id}/tasks",
            json={
                "task": "List the tools you have available",
                "workflow": "simple"
            }
        )
        response.raise_for_status()
        
        data = response.json()
        assert "job_id" in data, "Missing job_id"
        assert data["status"] == "pending", f"Unexpected status: {data['status']}"
        assert data["session_id"] == self.session_id, "Session ID mismatch"
        
        job_id = data["job_id"]
        print(f"Task submitted: {job_id}")
        
        # Wait for completion
        print("Waiting for job to complete...")
        for _ in range(30):  # 30 attempts = 60 seconds max
            time.sleep(2)
            
            response = requests.get(f"{self.base_url}/jobs/{job_id}")
            response.raise_for_status()
            status = response.json()
            
            if status["status"] == "completed":
                print(f"Job completed in {status.get('duration_seconds', 0):.1f}s")
                assert "result" in status, "Missing result"
                assert "content" in status["result"], "Missing content in result"
                print(f"Result length: {len(status['result']['content'])} chars")
                return
            
            elif status["status"] == "failed":
                raise RuntimeError(f"Job failed: {status.get('error')}")
            
            print(f"  Status: {status['status']}")
        
        raise TimeoutError("Job did not complete in time")
    
    def test_job_not_found(self):
        """Test getting non-existent job."""
        response = requests.get(f"{self.base_url}/jobs/job_nonexistent")
        assert response.status_code == 404, "Expected 404 for non-existent job"
        print("Correctly returned 404 for non-existent job")
    
    def test_session_not_found(self):
        """Test getting non-existent session."""
        response = requests.get(f"{self.base_url}/sessions/sess_nonexistent")
        assert response.status_code == 404, "Expected 404 for non-existent session"
        print("Correctly returned 404 for non-existent session")
    
    def test_delete_session(self):
        """Test deleting a session."""
        assert self.session_id, "No session created"
        
        response = requests.delete(f"{self.base_url}/sessions/{self.session_id}")
        response.raise_for_status()
        assert response.status_code == 204, "Expected 204 for successful deletion"
        
        print(f"Session deleted: {self.session_id}")
        
        # Verify it's gone
        response = requests.get(f"{self.base_url}/sessions/{self.session_id}")
        assert response.status_code == 404, "Session should be gone"
        print("Verified session is deleted")
        
        self.session_id = None
    
    def run_all_tests(self):
        """Run all tests."""
        print("\n" + "="*70)
        print("iExplain API Test Suite")
        print("="*70)
        
        # Basic tests
        self.test("Health Check", self.test_health_check)
        self.test("Create Session", self.test_create_session)
        self.test("Get Session", self.test_get_session)
        self.test("Update Session Config", self.test_update_session_config)
        self.test("Submit Simple Task", self.test_submit_simple_task)
        
        # Error handling tests
        self.test("Job Not Found", self.test_job_not_found)
        self.test("Session Not Found", self.test_session_not_found)
        
        # Cleanup
        if self.session_id:
            self.test("Delete Session", self.test_delete_session)
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Total:  {self.tests_passed + self.tests_failed}")
        print("="*70)
        
        if self.tests_failed > 0:
            print("\n✗ Some tests failed")
            sys.exit(1)
        else:
            print("\n✓ All tests passed!")


def main():
    """Run test suite."""
    parser = argparse.ArgumentParser(description="iExplain API Test Suite")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/api/v1",
        help="API base URL"
    )
    args = parser.parse_args()
    
    # Check if API is accessible
    try:
        requests.get(f"{args.base_url}/health", timeout=2)
    except requests.exceptions.RequestException:
        print(f"\n✗ Error: Cannot connect to API at {args.base_url}")
        print("\nMake sure the API server is running:")
        print("  uvicorn api.server:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Run tests
    tester = APITester(args.base_url)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
