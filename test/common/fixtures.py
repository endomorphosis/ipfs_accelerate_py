#!/usr/bin/env python3
"""
Test fixtures for IPFS Accelerate tests.

This module provides pytest fixtures for tests.
"""

import os
import sys
import logging
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path

import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from our project
from test.common.hardware_detection import detect_hardware, setup_platform


# Common fixtures
@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for tests.
    
    Returns:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def hardware_info():
    """
    Get hardware information.
    
    Returns:
        Dictionary with hardware information
    """
    return detect_hardware()


# Hardware-specific fixtures
@pytest.fixture
def cpu_device():
    """
    Get CPU device.
    
    Returns:
        CPU device object or string 'cpu'
    """
    return setup_platform('cpu')


@pytest.fixture
def cuda_device():
    """
    Get CUDA device if available.
    
    Returns:
        CUDA device object or None if not available
    """
    hardware_info = detect_hardware()
    if hardware_info['platforms']['cuda']['available']:
        return setup_platform('cuda')
    pytest.skip("CUDA not available")


@pytest.fixture
def rocm_device():
    """
    Get ROCm device if available.
    
    Returns:
        ROCm device object or None if not available
    """
    hardware_info = detect_hardware()
    if hardware_info['platforms']['rocm']['available']:
        return setup_platform('rocm')
    pytest.skip("ROCm not available")


@pytest.fixture
def mps_device():
    """
    Get MPS device if available.
    
    Returns:
        MPS device object or None if not available
    """
    hardware_info = detect_hardware()
    if hardware_info['platforms']['mps']['available']:
        return setup_platform('mps')
    pytest.skip("MPS not available")


@pytest.fixture
def openvino_device():
    """
    Get OpenVINO device if available.
    
    Returns:
        OpenVINO device object or None if not available
    """
    hardware_info = detect_hardware()
    if hardware_info['platforms']['openvino']['available']:
        return setup_platform('openvino')
    pytest.skip("OpenVINO not available")


@pytest.fixture
def qnn_device():
    """
    Get QNN device if available.
    
    Returns:
        QNN device object or None if not available
    """
    hardware_info = detect_hardware()
    if hardware_info['platforms']['qnn']['available']:
        return setup_platform('qnn')
    pytest.skip("QNN not available")


# Browser-specific fixtures
@pytest.fixture
def chrome_options():
    """
    Get Chrome options for WebGPU/WebNN tests.
    
    Returns:
        Chrome options object
    """
    try:
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--enable-webgpu")
        options.add_argument("--enable-unsafe-webgpu")
        options.add_argument("--enable-features=Vulkan")
        
        return options
    except ImportError:
        pytest.skip("Selenium not available")


@pytest.fixture
def webgpu_browser(chrome_options):
    """
    Get a Chrome browser with WebGPU enabled.
    
    Args:
        chrome_options: Chrome options fixture
        
    Returns:
        Chrome WebDriver object or None if not available
    """
    try:
        from selenium import webdriver
        
        browser = webdriver.Chrome(options=chrome_options)
        yield browser
        browser.quit()
    except Exception as e:
        logger.error(f"Error creating WebGPU browser: {e}")
        pytest.skip(f"Failed to create WebGPU browser: {e}")


@pytest.fixture
def webnn_browser(chrome_options):
    """
    Get a Chrome browser with WebNN enabled.
    
    Args:
        chrome_options: Chrome options fixture
        
    Returns:
        Chrome WebDriver object or None if not available
    """
    try:
        from selenium import webdriver
        
        # Add WebNN-specific options
        chrome_options.add_argument("--enable-features=WebML")
        
        browser = webdriver.Chrome(options=chrome_options)
        yield browser
        browser.quit()
    except Exception as e:
        logger.error(f"Error creating WebNN browser: {e}")
        pytest.skip(f"Failed to create WebNN browser: {e}")


# Model-specific fixtures
@pytest.fixture
def bert_model(cpu_device):
    """
    Load BERT model.
    
    Args:
        cpu_device: CPU device fixture
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from test.common.model_helpers import load_text_model
        
        model_name = "bert-base-uncased"
        model, tokenizer = load_text_model(model_name, device=cpu_device)
        
        if model is None or tokenizer is None:
            pytest.skip(f"Failed to load {model_name}")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading BERT model: {e}")
        pytest.skip(f"Failed to load BERT model: {e}")


@pytest.fixture
def vit_model(cpu_device):
    """
    Load ViT model.
    
    Args:
        cpu_device: CPU device fixture
        
    Returns:
        Tuple of (model, feature_extractor)
    """
    try:
        from test.common.model_helpers import load_vision_model
        
        model_name = "google/vit-base-patch16-224"
        model, feature_extractor = load_vision_model(model_name, device=cpu_device)
        
        if model is None or feature_extractor is None:
            pytest.skip(f"Failed to load {model_name}")
        
        return model, feature_extractor
    except Exception as e:
        logger.error(f"Error loading ViT model: {e}")
        pytest.skip(f"Failed to load ViT model: {e}")


@pytest.fixture
def whisper_model(cpu_device):
    """
    Load Whisper model.
    
    Args:
        cpu_device: CPU device fixture
        
    Returns:
        Tuple of (model, processor)
    """
    try:
        from test.common.model_helpers import load_audio_model
        
        model_name = "openai/whisper-tiny"
        model, processor = load_audio_model(model_name, device=cpu_device)
        
        if model is None or processor is None:
            pytest.skip(f"Failed to load {model_name}")
        
        return model, processor
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")
        pytest.skip(f"Failed to load Whisper model: {e}")


@pytest.fixture
def clip_model(cpu_device):
    """
    Load CLIP model.
    
    Args:
        cpu_device: CPU device fixture
        
    Returns:
        Tuple of (model, processor)
    """
    try:
        from test.common.model_helpers import load_multimodal_model
        
        model_name = "openai/clip-vit-base-patch32"
        model, processor = load_multimodal_model(model_name, device=cpu_device)
        
        if model is None or processor is None:
            pytest.skip(f"Failed to load {model_name}")
        
        return model, processor
    except Exception as e:
        logger.error(f"Error loading CLIP model: {e}")
        pytest.skip(f"Failed to load CLIP model: {e}")


# API-specific fixtures
@pytest.fixture
def api_key():
    """
    Get API key for tests.
    
    Returns:
        API key as string
    """
    return os.environ.get("API_KEY", "test_key")


@pytest.fixture
def api_base_url():
    """
    Get API base URL for tests.
    
    Returns:
        API base URL as string
    """
    return os.environ.get("API_BASE_URL", "http://localhost:8000")