"""
Zeroth-Order Gradient Estimator

A memory-efficient, general-purpose ZO gradient estimator that supports:
- Weight perturbation (WP): Perturb parameters directly
- Node perturbation (NP): Perturb activations with pseudo-NP or true-NP
- Rule-based layer/parameter selection
- 2D and 3D activation support for vision and language models
"""

from .estimator import ZOEstimator
from .config import ZOConfig
from .objective import build_objective_function

__all__ = ['ZOEstimator', 'ZOConfig', 'build_objective_function']
