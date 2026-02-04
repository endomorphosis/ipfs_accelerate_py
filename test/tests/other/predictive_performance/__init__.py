"""
Predictive Performance System

This package provides a machine learning-based framework for predicting
performance metrics of AI models on various hardware platforms.
"""

__version__ = "1.0.0"

# Import only the modules needed for the multi-model web integration
try:
    from test.tests.other.predictive_performance.multi_model_execution import MultiModelPredictor
except ImportError:
    pass

try:
    from test.tests.other.predictive_performance.multi_model_empirical_validation import MultiModelEmpiricalValidator
except ImportError:
    pass

try:
    from test.tests.other.predictive_performance.multi_model_resource_pool_integration import MultiModelResourcePoolIntegration
except ImportError:
    pass

try:
    from test.tests.other.predictive_performance.web_resource_pool_adapter import WebResourcePoolAdapter
except ImportError:
    pass

try:
    from test.tests.other.predictive_performance.multi_model_web_integration import MultiModelWebIntegration
except ImportError:
    pass