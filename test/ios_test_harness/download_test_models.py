#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Test Models Script for iOS CI

This script downloads the necessary test models for iOS CI benchmarks.
It fetches CoreML and ONNX model files and prepares them for use in the iOS test harness.

Usage:
    python download_test_models.py [--output-dir OUTPUT_DIR] [--models MODELS]

Examples:
    # Download default models to 'models' directory
    python download_test_models.py
    
    # Download specific models to a custom directory
    python download_test_models.py --output-dir /path/to/models --models bert-base-uncased,mobilenet-v2

Date: May 2025
"""

import os
import sys
import json
import argparse
import logging
import requests
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default models to download
DEFAULT_MODELS = [
    {
        "name": "bert-base-uncased",
        "url": "https://huggingface.co/apple/bert-base-uncased-coreml/resolve/main/bert-base-uncased.mlmodel",
        "type": "coreml",
        "size_mb": 438
    },
    {
        "name": "mobilenet-v2",
        "url": "https://huggingface.co/apple/mobilenet-v2-coreml/resolve/main/mobilenet-v2.mlmodel",
        "type": "coreml",
        "size_mb": 14
    },
    {
        "name": "roberta-base",
        "url": "https://huggingface.co/apple/roberta-base-coreml/resolve/main/roberta-base.mlmodel",
        "type": "coreml",
        "size_mb": 498
    },
    {
        "name": "whisper-tiny",
        "url": "https://huggingface.co/apple/whisper-tiny-coreml/resolve/main/whisper-tiny.mlmodel",
        "type": "coreml",
        "size_mb": 152
    },
    {
        "name": "clip-vit-base-patch32",
        "url": "https://huggingface.co/apple/clip-vit-base-patch32-coreml/resolve/main/clip-vit-base-patch32.mlmodel",
        "type": "coreml",
        "size_mb": 380
    },
    # ONNX models for comparison
    {
        "name": "bert-base-uncased-onnx",
        "url": "https://huggingface.co/optimum/bert-base-uncased/resolve/main/model.onnx",
        "type": "onnx",
        "size_mb": 420
    },
    {
        "name": "mobilenet-v2-onnx",
        "url": "https://huggingface.co/optimum/mobilenet-v2/resolve/main/model.onnx",
        "type": "onnx",
        "size_mb": 14
    }
]

def download_file(url: str, output_path: str) -> bool:
    """
    Download a file from a URL to the specified output path.
    
    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        
    Returns:
        Success status
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB
        
        logger.info(f"Downloading {url} to {output_path}")
        logger.info(f"File size: {total_size_in_bytes / (1024 * 1024):.1f} MB")
        
        with open(output_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
        
        logger.info(f"Download completed: {output_path}")
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        return False

def check_coreml_compatibility() -> bool:
    """
    Check if the current environment supports CoreML operations.
    
    Returns:
        True if CoreML is supported, False otherwise
    """
    try:
        # Check if running on macOS
        import platform
        if platform.system() != "Darwin":
            logger.warning("CoreML tools are only available on macOS")
            return False
        
        # Try to import coremltools
        import importlib.util
        coremltools_spec = importlib.util.find_spec("coremltools")
        if coremltools_spec is None:
            logger.warning("coremltools not found, ONNX to CoreML conversion will be skipped")
            logger.warning("Install with: pip install coremltools")
            return False
        
        return True
    
    except Exception as e:
        logger.warning(f"Error checking CoreML compatibility: {e}")
        return False

def download_models(output_dir: str, model_list: List[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Download models to the specified directory.
    
    Args:
        output_dir: Directory to save downloaded models
        model_list: List of model information dictionaries
        
    Returns:
        Dictionary mapping model names to download success status
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download each model
    results = {}
    
    for model in model_list:
        model_name = model["name"]
        model_url = model["url"]
        model_type = model.get("type", "coreml")
        
        # Determine output path
        output_path = os.path.join(output_dir, f"{model_name}.{model_type}")
        
        # Skip if file already exists
        if os.path.exists(output_path):
            logger.info(f"Model file already exists: {output_path}")
            results[model_name] = True
            continue
        
        # Download the model
        success = download_file(model_url, output_path)
        results[model_name] = success
        
        if not success:
            logger.error(f"Failed to download model: {model_name}")
    
    return results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Download test models for iOS CI benchmarks")
    
    parser.add_argument("--output-dir", default="models", help="Directory to save downloaded models")
    parser.add_argument("--models", help="Comma-separated list of model names to download (default: download all)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies if needed")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Install dependencies if requested
    if args.install_deps and check_coreml_compatibility():
        try:
            logger.info("Installing required dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "coremltools>=6.0"])
            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dependencies: {e}")
    
    # Determine which models to download
    models_to_download = DEFAULT_MODELS
    
    if args.models:
        # Filter models by provided names
        model_names = [name.strip() for name in args.models.split(',')]
        models_to_download = [model for model in DEFAULT_MODELS if model["name"] in model_names]
        
        # Check if all requested models are available
        available_names = [model["name"] for model in DEFAULT_MODELS]
        for name in model_names:
            if name not in available_names:
                logger.warning(f"Model not found in default list: {name}")
    
    # Download models
    logger.info(f"Downloading {len(models_to_download)} models to {args.output_dir}")
    results = download_models(args.output_dir, models_to_download)
    
    # Log results
    success_count = sum(1 for success in results.values() if success)
    logger.info(f"Successfully downloaded {success_count}/{len(models_to_download)} models")
    
    # Exit with error if any downloads failed
    if success_count < len(models_to_download):
        logger.error("Some model downloads failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())