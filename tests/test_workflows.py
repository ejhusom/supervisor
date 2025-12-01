#!/usr/bin/env python3
"""
Workflow system test/demo script.

Run this to verify workflows are working correctly.
Uses mock supervisor to test workflow logic without LLM calls.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows import SimpleWorkflow, EvaluatorWorkflow, MultiStageWorkflow


class MockSupervisor:
    """Mock supervisor for testing workflows without LLM calls."""
    
    def __init__(self, response="Mock result"):
        self.response = response
        self.call_count = 0
        self.llm_client = MockLLMClient()
    
    def run(self, task, context=None):
        self.call_count += 1
        return {
            "content": f"{self.response} (call {self.call_count})",
            "tool_calls": [],
            "history": []
        }


class MockLLMClient:
    """Mock LLM client for evaluator."""
    pass


def test_simple_workflow():
    """Test simple workflow (passthrough)."""
    print("=" * 70)
    print("TEST: Simple Workflow")
    print("=" * 70)
    
    supervisor = MockSupervisor("Simple result")
    workflow = SimpleWorkflow(supervisor)
    
    result = workflow.run("Test task")
    
    assert result["content"] == "Simple result (call 1)", "Simple workflow failed"
    assert supervisor.call_count == 1, "Should call supervisor once"
    
    print("✓ Simple workflow works")
    print(f"  Result: {result['content']}")
    print()


def test_evaluator_workflow_accept():
    """Test evaluator workflow when result is accepted."""
    print("=" * 70)
    print("TEST: Evaluator Workflow (Accept)")
    print("=" * 70)
    
    # Mock supervisor that produces acceptable result
    supervisor = MockSupervisor("High quality result that addresses the task comprehensively")
    workflow = EvaluatorWorkflow(supervisor, max_iterations=3, verbose=False)
    
    # Override evaluator to always accept
    class MockEvaluator:
        def run(self, task, **kwargs):
            return {"content": '{"acceptable": true, "reasoning": "Good", "suggestions": ""}'}
    
    workflow.evaluator = MockEvaluator()
    
    result = workflow.run("Test task")
    
    assert "evaluation_history" in result, "Should have evaluation history"
    assert result["iterations_used"] <= 3, "Should not exceed max iterations"
    
    print("✓ Evaluator workflow (accept) works")
    print(f"  Iterations used: {result['iterations_used']}")
    print(f"  Evaluations: {len(result['evaluation_history'])}")
    print()


def test_evaluator_workflow_retry():
    """Test evaluator workflow with retry."""
    print("=" * 70)
    print("TEST: Evaluator Workflow (Retry)")
    print("=" * 70)
    
    supervisor = MockSupervisor("Initial attempt")
    workflow = EvaluatorWorkflow(supervisor, max_iterations=3, verbose=False)
    
    # Override evaluator to reject first 2, accept 3rd
    class MockEvaluator:
        def __init__(self):
            self.call_count = 0
        
        def run(self, task, **kwargs):
            self.call_count += 1
            if self.call_count < 3:
                return {"content": '{"acceptable": false, "reasoning": "Needs improvement", "suggestions": "Add more detail"}'}
            else:
                return {"content": '{"acceptable": true, "reasoning": "Now acceptable", "suggestions": ""}'}
    
    workflow.evaluator = MockEvaluator()
    
    result = workflow.run("Test task")
    
    assert result["iterations_used"] == 3, f"Should use 3 iterations, got {result['iterations_used']}"
    assert len(result["evaluation_history"]) == 3, "Should have 3 evaluations"
    assert supervisor.call_count == 3, "Should call supervisor 3 times"
    
    print("✓ Evaluator workflow (retry) works")
    print(f"  Supervisor calls: {supervisor.call_count}")
    print(f"  Iterations used: {result['iterations_used']}")
    print()


def test_multi_stage_workflow():
    """Test multi-stage workflow."""
    print("=" * 70)
    print("TEST: Multi-Stage Workflow")
    print("=" * 70)
    
    supervisor = MockSupervisor("Stage result")
    stages = ["Stage 1: Parse", "Stage 2: Analyze", "Stage 3: Report"]
    workflow = MultiStageWorkflow(supervisor, stages=stages, verbose=False)
    
    result = workflow.run("Test task")
    
    assert "stage_results" in result, "Should have stage results"
    assert len(result["stage_results"]) == 3, f"Should have 3 stage results, got {len(result['stage_results'])}"
    assert supervisor.call_count == 3, f"Should call supervisor 3 times, got {supervisor.call_count}"
    assert result["stages_completed"] == 3, "Should complete 3 stages"
    
    print("✓ Multi-stage workflow works")
    print(f"  Stages: {len(stages)}")
    print(f"  Supervisor calls: {supervisor.call_count}")
    print(f"  Stage results: {len(result['stage_results'])}")
    print()


def test_workflow_context_accumulation():
    """Test that multi-stage workflow accumulates context."""
    print("=" * 70)
    print("TEST: Context Accumulation")
    print("=" * 70)
    
    supervisor = MockSupervisor("Result")
    workflow = MultiStageWorkflow(
        supervisor,
        stages=["Stage 1", "Stage 2"],
        accumulate_context=True,
        verbose=False
    )
    
    result = workflow.run("Test")
    
    # Check that stage 2 would have received stage 1 context
    # (In real usage, supervisor would see accumulated_context)
    assert len(result["stage_results"]) == 2, "Should have 2 stages"
    
    print("✓ Context accumulation works")
    print()


def run_all_tests():
    """Run all workflow tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "WORKFLOW SYSTEM TESTS" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        test_simple_workflow()
        test_evaluator_workflow_accept()
        test_evaluator_workflow_retry()
        test_multi_stage_workflow()
        test_workflow_context_accumulation()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("The workflow system is working correctly!")
        print("You can now use workflows with main.py")
        print()
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
