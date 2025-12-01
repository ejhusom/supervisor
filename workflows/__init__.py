"""
Workflow orchestration layer for iExplain.

Workflows wrap the Supervisor to implement different execution strategies:
- SimpleWorkflow: Direct passthrough (default behavior)
- EvaluatorWorkflow: Evaluation + retry loop
- MultiStageWorkflow: Break tasks into sequential stages
"""

from .base import Workflow
from .simple import SimpleWorkflow
from .evaluator import EvaluatorWorkflow
from .multi_stage import MultiStageWorkflow, PredefinedMultiStageWorkflow

__all__ = [
    'Workflow',
    'SimpleWorkflow',
    'EvaluatorWorkflow',
    'MultiStageWorkflow',
    'PredefinedMultiStageWorkflow'
]