#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Templates package for the refactored generator suite.
Provides access to all architecture and hardware templates.
"""

import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Import base hardware template
from .base_hardware import BaseHardwareTemplate, CPUHardwareTemplate

# Import hardware-specific templates
# These are new templates for our modular design
try:
    from .cuda_hardware import CudaHardwareTemplate
except ImportError:
    pass  # Optional, will be imported where needed

try:
    from .rocm_hardware import RocmHardwareTemplate
except ImportError:
    pass  # Optional, will be imported where needed

try:
    from .openvino_hardware import OpenvinoHardwareTemplate
except ImportError:
    pass  # Optional, will be imported where needed

try:
    from .apple_hardware import AppleHardwareTemplate
except ImportError:
    pass  # Optional, will be imported where needed

try:
    from .qualcomm_hardware import QualcommHardwareTemplate
except ImportError:
    pass  # Optional, will be imported where needed

# Import pipeline templates
from .base_pipeline import BasePipelineTemplate
try:
    from .text_pipeline import TextPipelineTemplate
except ImportError:
    # Fallback to the example implementation in base_pipeline
    from .base_pipeline import TextPipelineTemplate

try:
    from .image_pipeline import ImagePipelineTemplate
except ImportError:
    pass  # Optional, will be imported where needed

# Import architecture templates
from .base_architecture import BaseArchitectureTemplate

# Note: We don't import the architecture-specific template classes here
# because they may contain template syntax for code generation
# They will be imported directly where needed

# Package exports
__all__ = [
    'BaseHardwareTemplate',
    'CPUHardwareTemplate',
    'CudaHardwareTemplate',
    'RocmHardwareTemplate',
    'OpenvinoHardwareTemplate',
    'AppleHardwareTemplate',
    'QualcommHardwareTemplate',
    'BasePipelineTemplate',
    'TextPipelineTemplate',
    'ImagePipelineTemplate',
    'BaseArchitectureTemplate'
]