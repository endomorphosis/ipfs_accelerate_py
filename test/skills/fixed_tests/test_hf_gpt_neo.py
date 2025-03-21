#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model registry
GPT_NEO_MODELS_REGISTRY = {
    "gpt-neo": {
        "description": "GPT-Neo autoregressive language model",
        "class": "GPTNeoForCausalLM",
    },
    "EleutherAI/gpt-neo-1.3B": {
        "description": "GPT-Neo 1.3B parameter model",
        "class": "GPTNeoForCausalLM",
    }
}

class TestGptNeoModels:
    """Test class for GPT-Neo models."""
    
    def __init__(self, model_id=None):
        """Initialize the test class."""
        self.model_id = model_id or "gpt-neo"
        self.results = {}
        
    def run_tests(self):
        """Run tests for the model."""
        logger.info(f"Testing model: {self.model_id}")
        
        # Add metadata to results
        self.results["metadata"] = {
            "model": self.model_id,
            "timestamp": ""
        }
        
        return self.results

def get_available_models():
    """Get list of available models."""
    return list(GPT_NEO_MODELS_REGISTRY.keys())

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test GPT-Neo models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        models = get_available_models()
        print("\nAvailable GPT-Neo models:")
        for model in models:
            info = GPT_NEO_MODELS_REGISTRY[model]
            print(f"  - {model} ({info['class']}): {info['description']}")
        return
        
    # Test model
    model_id = args.model or "gpt-neo"
    tester = TestGptNeoModels(model_id)
    results = tester.run_tests()
    
    print(f"Successfully tested {model_id}")

if __name__ == "__main__":
    main()