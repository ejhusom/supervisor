#!/usr/bin/env python3
"""
Answer comparison logic for evaluating iExplain outputs against ground truth.
"""

from typing import Any, Dict, List


class AnswerComparator:
    """Compare iExplain answers to ground truth with flexible matching."""
    
    def compare(self, answer: Any, question_spec: Dict) -> Dict[str, Any]:
        """
        Compare answer to expected value in question spec.
        
        Args:
            answer: Answer from iExplain
            question_spec: Question specification from ground truth
        
        Returns:
            Dict with:
                - correct: bool
                - reason: str (explanation)
                - answer: original answer
                - expected: expected answer
        """
        answer_type = question_spec.get("answer_type")
        expected = question_spec.get("expected")
        
        if answer_type == "integer":
            return self._compare_integer(answer, expected, question_spec)
        
        elif answer_type == "string_match":
            return self._compare_string(answer, expected, question_spec)
        
        elif answer_type == "list":
            return self._compare_list(answer, expected, question_spec)
        
        else:
            return {
                "correct": False,
                "reason": f"Unknown answer type: {answer_type}",
                "answer": answer,
                "expected": expected
            }
    
    def _compare_integer(self, answer: Any, expected: int, spec: Dict) -> Dict:
        """Compare integer answers with optional tolerance."""
        tolerance = spec.get("tolerance", 0)
        
        # Try to convert answer to int
        try:
            if isinstance(answer, str):
                # Extract number from string
                import re
                numbers = re.findall(r'\d+', answer)
                if numbers:
                    answer_int = int(numbers[0])
                else:
                    return {
                        "correct": False,
                        "reason": f"Could not extract integer from: {answer}",
                        "answer": answer,
                        "expected": expected
                    }
            else:
                answer_int = int(answer)
        except (ValueError, TypeError):
            return {
                "correct": False,
                "reason": f"Could not convert to integer: {answer}",
                "answer": answer,
                "expected": expected
            }
        
        # Check if within tolerance
        diff = abs(answer_int - expected)
        correct = diff <= tolerance
        
        if correct:
            reason = f"Correct (within tolerance ±{tolerance})"
        else:
            reason = f"Incorrect: got {answer_int}, expected {expected} (±{tolerance})"
        
        return {
            "correct": correct,
            "reason": reason,
            "answer": answer_int,
            "expected": expected,
            "difference": diff
        }
    
    def _compare_string(self, answer: Any, expected: List[str], spec: Dict) -> Dict:
        """Compare string answers with flexible matching."""
        case_sensitive = spec.get("case_sensitive", False)
        
        # Convert answer to string
        answer_str = str(answer).strip()
        
        # Prepare for comparison
        if not case_sensitive:
            answer_str = answer_str.lower()
            expected = [e.lower() for e in expected]
        
        # Check if answer matches any expected value
        for exp in expected:
            if exp in answer_str or answer_str in exp:
                return {
                    "correct": True,
                    "reason": f"Matched expected value: {exp}",
                    "answer": str(answer).strip(),
                    "expected": expected
                }
        
        return {
            "correct": False,
            "reason": f"Did not match any expected values",
            "answer": str(answer).strip(),
            "expected": expected
        }
    
    def _compare_list(self, answer: Any, expected: List, spec: Dict) -> Dict:
        """Compare list answers with optional order sensitivity."""
        order_matters = spec.get("order_matters", False)
        
        # Ensure answer is a list
        if not isinstance(answer, list):
            # Try to convert
            if isinstance(answer, str):
                # Split by common delimiters
                import re
                answer = re.split(r'[,;\n]', answer)
                answer = [a.strip() for a in answer if a.strip()]
            else:
                return {
                    "correct": False,
                    "reason": f"Answer is not a list: {type(answer)}",
                    "answer": answer,
                    "expected": expected
                }
        
        # Normalize strings in both lists
        answer_normalized = [str(a).strip() for a in answer]
        expected_normalized = [str(e).strip() for e in expected]
        
        # Compare based on order requirement
        if order_matters:
            # Exact order match required
            correct = answer_normalized == expected_normalized
            if correct:
                reason = "Correct (exact order match)"
            else:
                reason = f"Incorrect order: got {answer_normalized}, expected {expected_normalized}"
        else:
            # Set comparison (order doesn't matter)
            answer_set = set(a.lower() for a in answer_normalized)
            expected_set = set(e.lower() for e in expected_normalized)
            
            correct = answer_set == expected_set
            
            if correct:
                reason = "Correct (all items present, order ignored)"
            else:
                missing = expected_set - answer_set
                extra = answer_set - expected_set
                details = []
                if missing:
                    details.append(f"missing: {missing}")
                if extra:
                    details.append(f"extra: {extra}")
                reason = f"Incorrect: {', '.join(details)}"
        
        return {
            "correct": correct,
            "reason": reason,
            "answer": answer_normalized,
            "expected": expected_normalized
        }


def evaluate_answer(answer: Any, question_spec: Dict) -> Dict[str, Any]:
    """
    Convenience function to evaluate a single answer.
    
    Args:
        answer: Answer from iExplain
        question_spec: Question specification from ground truth
    
    Returns:
        Comparison result dict
    """
    comparator = AnswerComparator()
    return comparator.compare(answer, question_spec)
