"""
Ecstasy: A comprehensive protein structure prediction and analysis toolkit.

This package provides tools for protein structure prediction, analysis,
profiling, and permutation invariance studies.
"""

__version__ = "0.1.0"
__author__ = "Harsh Agrawal"
__email__ = "harshagrawal.1312@gmail.com"

# Import main modules to make them available at package level
from . import utils
from . import predict
from . import profiling
from . import permutation_invariance

__all__ = [
    "utils",
    "predict", 
    "profiling",
    "permutation_invariance",
] 