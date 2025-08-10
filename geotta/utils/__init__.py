"""
Utility functions for GeoTTA
"""

from .metrics import evaluate_model
from .memory import optimize_memory

__all__ = ['evaluate_model', 'optimize_memory']