#!/usr/bin/env python3
"""
Test script for validating model skillsets.

This script validates generated skillsets by testing their basic functionality.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set mock mode for testing
os.environ["MOCK_MODE"] = "true"

def test_skillset_by_model(model_name: str, device: str = "cpu") -> bool:
    """
    Test a skillset for a specific model.
    
    Args:
        model_name: The model name (e.g., 'bert', 't5', 'gpt2')
        device: The device to test ('cpu', 'cuda', etc.)
        
    Returns:
        True if the test succeeded, False otherwise
    """
    # Construct the skillset file path
    skillset_file = f"{model_name}_{device}_skillset.py"
    skillset_path = os.path.join("skillsets", skillset_file)
    
    logger.info(f"Testing skillset: {skillset_path}")
    
    if not os.path.exists(skillset_path):
        logger.error(f"Skillset file not found: {skillset_path}")
        return False
    
    try:
        # Import the skillset module
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"{model_name}_{device}_skillset", skillset_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Use the test_skillset function if available, otherwise create our own test
        if hasattr(module, "test_skillset"):
            module.test_skillset()
            logger.info(f"Successfully ran test_skillset for {model_name}")
            return True
        else:
            # Find the model class
            model_classes = [
                cls for name, cls in module.__dict__.items()
                if name.endswith("Skillset") and isinstance(cls, type)
            ]
            
            if not model_classes:
                logger.error(f"No skillset class found in {skillset_path}")
                return False
            
            # Instantiate the first found model class
            skillset_class = model_classes[0]
            skillset = skillset_class()
            
            # Test basic functionality
            load_result = skillset.load_model()
            logger.info(f"Load result: {{'success': {load_result['success']}, 'device': {load_result['device']}}}")
            
            if load_result["success"]:
                # Test inference
                if hasattr(skillset, "run_inference"):
                    # Try a basic inference (text varies by model type)
                    inference_text = "Hello, world!"
                    if hasattr(skillset, "task") and skillset.task == "text2text-generation":
                        inference_text = "Translate to French: Hello, how are you?"
                    
                    inference_result = skillset.run_inference(inference_text)
                    logger.info(f"Inference result: {{'success': {inference_result['success']}}}")
                    
                    if not inference_result["success"]:
                        logger.error(f"Inference failed: {inference_result.get('error', 'Unknown error')}")
                        return False
                
                return True
            else:
                logger.error(f"Model loading failed: {load_result.get('error', 'Unknown error')}")
                return False
    
    except Exception as e:
        logger.error(f"Error testing skillset {skillset_path}: {e}")
        return False
    
    return True

def test_all_skillsets() -> Dict[str, bool]:
    """
    Test all available skillsets.
    
    Returns:
        Dictionary mapping skillset names to test results
    """
    # Get all skillset files
    skillset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skillsets")
    skillset_files = [f for f in os.listdir(skillset_dir) if f.endswith("_skillset.py")]
    
    logger.info(f"Found {len(skillset_files)} skillset files")
    
    # Test each skillset
    results = {}
    
    for skillset_file in skillset_files:
        # Parse model and device from filename
        parts = skillset_file.split("_")
        if len(parts) < 3:
            logger.warning(f"Skipping invalid skillset filename: {skillset_file}")
            continue
        
        model_name = parts[0]
        device = parts[1]
        
        # Test the skillset
        success = test_skillset_by_model(model_name, device)
        results[skillset_file] = success
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model skillsets")
    parser.add_argument("--model", "-m", type=str, help="Specific model to test")
    parser.add_argument("--device", "-d", type=str, default="cpu", help="Device to test")
    parser.add_argument("--all", "-a", action="store_true", help="Test all skillsets")
    
    args = parser.parse_args()
    
    if args.all:
        # Test all skillsets
        results = test_all_skillsets()
        
        # Print summary
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Tested {len(results)} skillsets")
        logger.info(f"Success: {success_count}/{len(results)}")
        
        # Print failed skillsets
        failed = [name for name, success in results.items() if not success]
        if failed:
            logger.warning("Failed skillsets:")
            for name in failed:
                logger.warning(f"  - {name}")
        
        sys.exit(0 if success_count == len(results) else 1)
    
    elif args.model:
        # Test specific model
        success = test_skillset_by_model(args.model, args.device)
        logger.info(f"Test {'succeeded' if success else 'failed'} for {args.model} on {args.device}")
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        sys.exit(1)