"""
Ecstasy: A comprehensive protein structure prediction and analysis toolkit.

This package provides tools for protein structure prediction, analysis,
profiling, and permutation invariance studies.
"""

__version__ = "0.1.0"
__author__ = "Harsh Agrawal"
__email__ = "harshagrawal.1312@gmail.com"

# Heavy submodules are imported on demand. Lightweight subpackages
# (metrics, benchmarks, models, msa, pipelines) can be imported without pulling
# in the full structure-analysis stack (biotite, seaborn, DockQ, etc.).
__all__ = ["utils", "predict", "profiling", "permutation_invariance"]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
