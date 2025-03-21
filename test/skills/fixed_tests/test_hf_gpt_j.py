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
GPT_J_MODELS_REGISTRY = {
    "gpt-j": {
        "description": "GPT-J autoregressive language model",
        "class": "GPTJForCausalLM",
    },
    "EleutherAI/gpt-j-6b": {
        "description": "GPT-J 6B parameter model",
        "class": "GPTJForCausalLM",
    }
}

class TestGptJModels:
    """Test class for GPT-J models."""
    
    def __init__(self, model_id=None):
        """Initialize the test class."""
        self.model_id = model_id or "gpt-j"
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
    return list(GPT_J_MODELS_REGISTRY.keys())

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test GPT-J models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        models = get_available_models()
        print("\nAvailable GPT-J models:")
        for model in models:
            info = GPT_J_MODELS_REGISTRY[model]
            print(f"  - {model} ({info['class']}): {info['description']}")
        return
        
    # Test model
    model_id = args.model or "gpt-j"
    tester = TestGptJModels(model_id)
    results = tester.run_tests()
    
    print(f"Successfully tested {model_id}")

if __name__ == "__main__":
    main()