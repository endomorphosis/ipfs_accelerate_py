#!/usr/bin/env python3
"""
Create Missing Critical Tests

This script creates test files for the remaining critical models that are missing.
"""

import os
import sys
from pathlib import Path

# Import the simplified_fix_hyphenated module
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(script_dir))

# Try to import the create_hyphenated_test_file function
try:
    from simplified_fix_hyphenated import create_hyphenated_test_file
    
    # Missing critical models
    missing_models = [
        # Create pix2struct test using the hyphenated approach
        {
            "name": "pix2struct",
            "model_id": "pix2struct",
            "class_name": "Pix2Struct",
            "model_upper": "PIX2STRUCT",
            "arch_type": "multimodal",
            "default_model": "google/pix2struct-base",
            "task": "image-to-text"
        },
        # Create speecht5 test using the hyphenated approach
        {
            "name": "speecht5",
            "model_id": "speecht5",
            "class_name": "SpeechT5",
            "model_upper": "SPEECHT5",
            "arch_type": "speech",
            "default_model": "microsoft/speecht5_tts",
            "task": "automatic-speech-recognition"
        }
    ]
    
    # Output directory
    output_dir = script_dir / "fixed_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test files
    for model in missing_models:
        # Create test file
        model_name = model["name"]
        print(f"Creating test file for {model_name}")
        
        # Create file content
        content = f"""#!/usr/bin/env python3

"""
        content += f'''
"""
Test file for {model["class_name"]} models.
This file tests the {model["class_name"]} model type from HuggingFace Transformers.
"""

import os
import sys
import json
import time
import logging
import argparse
from unittest.mock import MagicMock
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if dependencies are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Registry for {model["class_name"]} models
{model["model_upper"]}_MODELS_REGISTRY = {{
    "{model["default_model"]}": {{
        "full_name": "{model["class_name"]} Base",
        "architecture": "{model["arch_type"]}",
        "description": "{model["class_name"]} model for {model["task"]}",
        "model_type": "{model["name"]}",
        "parameters": "250M",
        "context_length": 512,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "recommended_tasks": ["{model["task"]}"]
    }}
}}

def select_device():
    """Select the best available device for inference."""
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

class Test{model["class_name"]}Models:
    """
    Test class for {model["class_name"]} models.
    """
    
    def __init__(self, model_id="{model["default_model"]}", device=None):
        """Initialize the test class for {model["class_name"]} models.
        
        Args:
            model_id: The model ID to test (default: "{model["default_model"]}")
            device: The device to run tests on (default: None = auto-select)
        """
        self.model_id = model_id
        self.device = device if device else select_device()
        self.performance_stats = {{}}
    
    def test_pipeline(self):
        """Test the model using the pipeline API."""
        try:
            if not HAS_TRANSFORMERS:
                logger.warning("Transformers library not available, skipping pipeline test")
                return {{"success": False, "error": "Transformers library not available"}}
                
            logger.info(f"Testing {model["class_name"]} model {{self.model_id}} with pipeline API on {{self.device}}")
            
            # Record start time
            start_time = time.time()
            
            # Initialize the pipeline with the appropriate task
            pipe = transformers.pipeline(
                "{model["task"]}", 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {{load_time:.2f}} seconds")
            
            # Test with a task-appropriate input
            test_input = "An image for {model["task"]}."
            
            # Record inference start time
            inference_start = time.time()
            
            # Run inference
            outputs = pipe(test_input)
            
            # Record inference time
            inference_time = time.time() - inference_start
            
            # Store performance stats
            self.performance_stats["pipeline"] = {{
                "load_time": load_time,
                "inference_time": inference_time
            }}
            
            return {{
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "inference_time": inference_time
            }}
        except Exception as e:
            logger.error(f"Error testing pipeline: {{e}}")
            return {{"success": False, "error": str(e)}}
    
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        results = {{}}
        
        # Run pipeline test
        pipeline_result = self.test_pipeline()
        results["pipeline"] = pipeline_result
        
        # Add metadata
        results["metadata"] = {{
            "model_id": self.model_id,
            "device": self.device,
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH
        }}
        
        return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test {model["class_name"]} HuggingFace models")
    parser.add_argument("--model", type=str, default="{model["default_model"]}", help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to run tests on (cuda, cpu)")
    
    args = parser.parse_args()
    
    # Initialize the test class
    {model["model_id"]}_tester = Test{model["class_name"]}Models(model_id=args.model, device=args.device)
    
    # Run the tests
    results = {model["model_id"]}_tester.run_tests()
    
    # Print a summary
    success = results["pipeline"].get("success", False)
    
    print("\\nTEST RESULTS SUMMARY:")
    
    if success:
        print(f"  Successfully tested {{args.model}}")
        print(f"  - Device: {{{model["model_id"]}_tester.device}}")
        print(f"  - Inference time: {{results['pipeline'].get('inference_time', 'N/A'):.4f}}s")
    else:
        print(f"  Failed to test {{args.model}}")
        print(f"  - Error: {{results['pipeline'].get('error', 'Unknown error')}}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        # Write the file
        output_file = output_dir / f"test_hf_{model['model_id']}.py"
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"Successfully created {output_file}")
        
except ImportError as e:
    print(f"Error importing simplified_fix_hyphenated: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)