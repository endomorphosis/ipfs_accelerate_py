"""
Predictive Performance System

This package provides a machine learning-based framework for predicting
performance metrics of AI models on various hardware platforms.
"""

# Make key classes available at the package level
try:
    from .predict import PerformancePredictor
    from .active_learning import ActiveLearningSystem
    from .benchmark_integration import BenchmarkScheduler
except ImportError as e:
    print(f"Warning: Unable to import some modules - {e}")

__version__ = "1.0.0"