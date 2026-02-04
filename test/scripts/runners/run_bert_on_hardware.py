#!/usr/bin/env python3
"""
Simple script to test BERT model on different hardware backends.
This script manually fixes the syntax issues in the template files.
"""

import os
import sys
import logging
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_bert_on_hardware(hardware="cpu", model_name="bert-base-uncased"):
    """Test BERT model on specified hardware."""
    logger.info(f"Testing {model_name} on {hardware}")
    
    # Set device
    if hardware == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA with {torch.cuda.device_count()} devices")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    elif hardware == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU")
    else:
        logger.warning(f"Hardware {hardware} not available, falling back to CPU")
        device = torch.device("cpu")
    
    try:
        # Load tokenizer and model
        logger.info(f"Loading {model_name} tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        
        # Prepare input
        text = "This is a sample text for testing BERT model."
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        logger.info("Running inference...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check outputs
        logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
        
        # Calculate embedding
        embedding = outputs.last_hidden_state.mean(dim=1)
        logger.info(f"Embedding shape: {embedding.shape}")
        
        # Test success
        logger.info(f"Successfully tested {model_name} on {hardware}")
        return True
    
    except Exception as e:
        logger.error(f"Error testing {model_name} on {hardware}: {e}")
        return False

def main():
    # Define hardware backends to test
    hardware_backends = ["cpu"]
    
    # Add CUDA if available
    if torch.cuda.is_available():
        hardware_backends.append("cuda")
    
    # Add MPS if available (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        hardware_backends.append("mps")
    
    # Test on each hardware backend
    results = {}
    for hardware in hardware_backends:
        results[hardware] = test_bert_on_hardware(hardware)
    
    # Print summary
    logger.info("=== Test Results ===")
    for hardware, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"{hardware}: {status}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())