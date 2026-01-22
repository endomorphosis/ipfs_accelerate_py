#!/usr/bin/env python3
"""
Pytest configuration for IPFS Accelerate tests.

This module provides global pytest configuration.
"""

import os
import sys
import logging
from pathlib import Path
import importlib.util

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "ipfs_accelerate_py"
sys.path.insert(0, str(REPO_ROOT))


def _force_real_ipfs_accelerate_package() -> None:
    """Force-load the real package directory as `ipfs_accelerate_py`.

    The repo contains repo-local `ipfs_accelerate_py.py` files (both at the
    repo root and within `test/`) that can shadow the actual
    `ipfs_accelerate_py/` package directory during imports.
    """

    init_py = PKG_DIR / "__init__.py"
    if not init_py.exists():
        return

    spec = importlib.util.spec_from_file_location(
        "ipfs_accelerate_py",
        init_py,
        submodule_search_locations=[str(PKG_DIR)],
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules["ipfs_accelerate_py"] = module
    spec.loader.exec_module(module)


_force_real_ipfs_accelerate_package()

# Import fixtures
from test.common.fixtures import *

# Register pytest markers
def pytest_configure(config):
    """
    Configure pytest.
    
    Args:
        config: pytest config object
    """
    # Register markers
    config.addinivalue_line("markers", "model: mark test as model test")
    config.addinivalue_line("markers", "text: mark test as text model test")
    config.addinivalue_line("markers", "vision: mark test as vision model test")
    config.addinivalue_line("markers", "audio: mark test as audio model test")
    config.addinivalue_line("markers", "multimodal: mark test as multimodal model test")
    config.addinivalue_line("markers", "hardware: mark test as hardware test")
    config.addinivalue_line("markers", "webgpu: mark test as WebGPU test")
    config.addinivalue_line("markers", "webnn: mark test as WebNN test")
    config.addinivalue_line("markers", "cuda: mark test as CUDA test")
    config.addinivalue_line("markers", "rocm: mark test as ROCm test")
    config.addinivalue_line("markers", "mps: mark test as MPS test")
    config.addinivalue_line("markers", "openvino: mark test as OpenVINO test")
    config.addinivalue_line("markers", "qnn: mark test as QNN test")
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "slow: mark test as slow test")
    config.addinivalue_line("markers", "distributed: mark test as distributed test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    
    # Add model-specific markers
    config.addinivalue_line("markers", "bert: mark test as BERT model test")
    config.addinivalue_line("markers", "t5: mark test as T5 model test")
    config.addinivalue_line("markers", "gpt: mark test as GPT model test")
    config.addinivalue_line("markers", "vit: mark test as ViT model test")
    config.addinivalue_line("markers", "whisper: mark test as Whisper model test")
    config.addinivalue_line("markers", "clip: mark test as CLIP model test")
    
    # Add API-specific markers
    config.addinivalue_line("markers", "openai: mark test as OpenAI API test")
    config.addinivalue_line("markers", "hf_tei: mark test as HuggingFace TEI API test")
    config.addinivalue_line("markers", "hf_tgi: mark test as HuggingFace TGI API test")
    config.addinivalue_line("markers", "ollama: mark test as Ollama API test")
    config.addinivalue_line("markers", "vllm: mark test as vLLM API test")
    config.addinivalue_line("markers", "claude: mark test as Claude API test")


# Skip functions for specific conditions
def pytest_runtest_setup(item):
    """
    Set up test before it runs.
    
    Args:
        item: Test item
    """
    # Skip WebGPU tests if marker present but WebGPU not available
    if 'webgpu' in item.keywords:
        try:
            from test.common.hardware_detection import detect_hardware
            hardware_info = detect_hardware()
            if not hardware_info['platforms']['webgpu']['available']:
                pytest.skip("WebGPU not available")
        except ImportError:
            pytest.skip("Hardware detection not available")
    
    # Skip WebNN tests if marker present but WebNN not available
    if 'webnn' in item.keywords:
        try:
            from test.common.hardware_detection import detect_hardware
            hardware_info = detect_hardware()
            if not hardware_info['platforms']['webnn']['available']:
                pytest.skip("WebNN not available")
        except ImportError:
            pytest.skip("Hardware detection not available")
    
    # Skip CUDA tests if marker present but CUDA not available
    if 'cuda' in item.keywords:
        try:
            from test.common.hardware_detection import detect_hardware
            hardware_info = detect_hardware()
            if not hardware_info['platforms']['cuda']['available']:
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("Hardware detection not available")
    
    # Skip ROCm tests if marker present but ROCm not available
    if 'rocm' in item.keywords:
        try:
            from test.common.hardware_detection import detect_hardware
            hardware_info = detect_hardware()
            if not hardware_info['platforms']['rocm']['available']:
                pytest.skip("ROCm not available")
        except ImportError:
            pytest.skip("Hardware detection not available")
    
    # Skip MPS tests if marker present but MPS not available
    if 'mps' in item.keywords:
        try:
            from test.common.hardware_detection import detect_hardware
            hardware_info = detect_hardware()
            if not hardware_info['platforms']['mps']['available']:
                pytest.skip("MPS not available")
        except ImportError:
            pytest.skip("Hardware detection not available")
    
    # Skip OpenVINO tests if marker present but OpenVINO not available
    if 'openvino' in item.keywords:
        try:
            from test.common.hardware_detection import detect_hardware
            hardware_info = detect_hardware()
            if not hardware_info['platforms']['openvino']['available']:
                pytest.skip("OpenVINO not available")
        except ImportError:
            pytest.skip("Hardware detection not available")
    
    # Skip QNN tests if marker present but QNN not available
    if 'qnn' in item.keywords:
        try:
            from test.common.hardware_detection import detect_hardware
            hardware_info = detect_hardware()
            if not hardware_info['platforms']['qnn']['available']:
                pytest.skip("QNN not available")
        except ImportError:
            pytest.skip("Hardware detection not available")