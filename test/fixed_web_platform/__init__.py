"""
WebNN and WebGPU platform support for the test generator.

This package provides enhanced support for WebNN and WebGPU platforms
with proper input handling, batch support detection, and modality-specific
processing for various model types.

March 2025 Updates:
- WebGPU compute shader support for audio models (20-35% performance improvement)
- Parallel model loading for multimodal models (30-45% loading time reduction)
- Shader precompilation for faster startup (reduced initial latency)
- Enhanced browser detection with Firefox support
- Performance tracking metrics integrated with benchmark database
"""

from .web_platform_handler import (
    process_for_web, 
    init_webnn, 
    init_webgpu, 
    create_mock_processors
)

__all__ = [
    'process_for_web',
    'init_webnn',
    'init_webgpu',
    'create_mock_processors'
]