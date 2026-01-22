#!/usr/bin/env python3
"""Test structured query and comparison logic."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from eval.eval_structured_query import extract_json
from eval.eval_comparator import evaluate_answer


def test_json_extraction():
    """Test JSON extraction from various formats."""
    print("Testing JSON extraction...")
    
    test_cases = [
        # Plain JSON
        ('{"answer": 42}', {"answer": 42}),
        ('[1, 2, 3]', [1, 2, 3]),
        ('150', 150),
        ('"KERNEL"', "KERNEL"),
        
        # Markdown code blocks
        ('```json\n{"answer": 150}\n```', {"answer": 150}),
        ('Here is the answer:\n```\n["KERNEL", "APP"]\n```', ["KERNEL", "APP"]),
        
        # With surrounding text
        ('The answer is 150 errors found.', 150),
        ('The component is "KERNEL" which has most logs.', "KERNEL"),
    ]
    
    passed = 0
    for text, expected in test_cases:
        try:
            result = extract_json(text)
            if result == expected:
                print(f"  ✓ Extracted {result} from: {text[:50]}...")
                passed += 1
            else:
                print(f"  ✗ Got {result}, expected {expected} from: {text[:50]}...")
        except Exception as e:
            print(f"  ✗ Error extracting from {text[:50]}...: {e}")
    
    print(f"Passed {passed}/{len(test_cases)} extraction tests\n")
    return passed == len(test_cases)


def test_comparisons():
    """Test answer comparison logic."""
    print("Testing answer comparisons...")
    
    test_cases = [
        # Integer comparisons
        {
            "answer": 150,
            "spec": {"answer_type": "integer", "expected": 150, "tolerance": 0},
            "should_pass": True
        },
        {
            "answer": 148,
            "spec": {"answer_type": "integer", "expected": 150, "tolerance": 2},
            "should_pass": True
        },
        {
            "answer": 145,
            "spec": {"answer_type": "integer", "expected": 150, "tolerance": 2},
            "should_pass": False
        },
        
        # String comparisons
        {
            "answer": "KERNEL",
            "spec": {"answer_type": "string_match", "expected": ["KERNEL"], "case_sensitive": False},
            "should_pass": True
        },
        {
            "answer": "kernel",
            "spec": {"answer_type": "string_match", "expected": ["KERNEL"], "case_sensitive": False},
            "should_pass": True
        },
        {
            "answer": "The answer is KERNEL",
            "spec": {"answer_type": "string_match", "expected": ["KERNEL", "kernel"], "case_sensitive": False},
            "should_pass": True
        },
        
        # List comparisons
        {
            "answer": ["KERNEL", "APP", "RAS"],
            "spec": {"answer_type": "list", "expected": ["KERNEL", "APP", "RAS"], "order_matters": False},
            "should_pass": True
        },
        {
            "answer": ["RAS", "KERNEL", "APP"],
            "spec": {"answer_type": "list", "expected": ["KERNEL", "APP", "RAS"], "order_matters": False},
            "should_pass": True
        },
        {
            "answer": ["RAS", "KERNEL", "APP"],
            "spec": {"answer_type": "list", "expected": ["KERNEL", "APP", "RAS"], "order_matters": True},
            "should_pass": False
        },
    ]
    
    passed = 0
    for i, test in enumerate(test_cases):
        result = evaluate_answer(test["answer"], test["spec"])
        
        if result["correct"] == test["should_pass"]:
            print(f"  ✓ Test {i+1}: {result['reason']}")
            passed += 1
        else:
            print(f"  ✗ Test {i+1}: Expected {'pass' if test['should_pass'] else 'fail'}, "
                  f"got {'pass' if result['correct'] else 'fail'}")
            print(f"     Reason: {result['reason']}")
    
    print(f"Passed {passed}/{len(test_cases)} comparison tests\n")
    return passed == len(test_cases)


def main():
    print("="*70)
    print("Testing Evaluation Components")
    print("="*70)
    print()
    
    extraction_ok = test_json_extraction()
    comparison_ok = test_comparisons()
    
    print("="*70)
    if extraction_ok and comparison_ok:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)
    print("="*70)


if __name__ == "__main__":
    main()
