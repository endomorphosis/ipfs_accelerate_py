#!/usr/bin/env python3
"""
Basic template for testing HuggingFace models with a focus on MODEL_TYPE models.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models registry for MODEL_TYPE models
MODEL_TYPE_MODELS_REGISTRY = {
    "model-name": {
        "description": "MODEL_TYPE base model",
        "class": "MODELTYPEBaseClass"
    }
}

class TestMODELTYPEModels:
    """Test class for MODEL_TYPE models."""
    
    def __init__(self, model_id=None):
        """Initialize with a specific model ID or default."""
        self.model_id = model_id or "model-name"
        
        # Verify model exists in registry
        if self.model_id not in MODEL_TYPE_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default")
            self.model_info = MODEL_TYPE_MODELS_REGISTRY["model-name"]
        else:
            self.model_info = MODEL_TYPE_MODELS_REGISTRY[self.model_id]
        
        # Set model properties
        self.description = self.model_info["description"]
        self.class_name = self.model_info["class"]
    
    def test_model(self):
        """Test the model and return results."""
        logger.info(f"Testing {self.model_id} ({self.class_name})")
        return {
            "model": self.model_id,
            "success": True,
            "class": self.class_name,
            "description": self.description
        }

def get_available_models():
    """Get list of available models."""
    return list(MODEL_TYPE_MODELS_REGISTRY.keys())

def test_all_models():
    """Test all available models."""
    models = get_available_models()
    results = {}
    
    for model_id in models:
        logger.info(f"Testing model: {model_id}")
        tester = TestMODELTYPEModels(model_id)
        results[model_id] = tester.test_model()
    
    return results

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MODEL_TYPE models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--all", action="store_true", help="Test all models")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list:
        models = get_available_models()
        print("Available MODEL_TYPE models:")
        for model in models:
            info = MODEL_TYPE_MODELS_REGISTRY[model]
            print(f"  - {model} ({info['class']}): {info['description']}")
        return
    
    if args.all:
        results = test_all_models()
        for model, result in results.items():
            success = "✅" if result["success"] else "❌"
            print(f"{success} {model}: {result['description']}")
        return
    
    # Test single model
    model_id = args.model or "model-name"
    tester = TestMODELTYPEModels(model_id)
    result = tester.test_model()
    
    success = "✅" if result["success"] else "❌"
    print(f"{success} {model_id}: {result['description']}")

if __name__ == "__main__":
    main()