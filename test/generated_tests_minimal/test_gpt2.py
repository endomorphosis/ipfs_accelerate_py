#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import libraries
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

# CUDA detection
if HAS_TORCH:
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        logger.info(f"CUDA available")
    else:
        logger.info("CUDA not available")
else:
    HAS_CUDA = False
    logger.info("CUDA detection skipped (torch not available)")

class TestGpt2Model:
    def __init__(self, model_id="gpt2"):
        self.model_id = model_id
        self.device = "cuda" if HAS_CUDA else "cpu"
        
    def test_pipeline(self):
        """Test the model using pipeline API."""
        if not HAS_TRANSFORMERS:
            logger.warning("Transformers not available, skipping test")
            return {"success": False, "error": "Transformers not available"}
            
        logger.info(f"Testing {self.model_id} with pipeline API")
        
        try:
            # Create a pipeline
            pipe = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Run inference
            test_input = "Once upon a time"
            outputs = pipe(test_input, max_length=50)
            
            # Process results
            if isinstance(outputs, list) and len(outputs) > 0:
                logger.info(f"Inference successful")
                return {"success": True}
            else:
                logger.error(f"Unexpected output format")
                return {"success": False, "error": "Unexpected output format"}
                
        except Exception as e:
            logger.error(f"Error in pipeline test: {e}")
            return {"success": False, "error": str(e)}
    
    def run_all_tests(self):
        """Run all tests for this model."""
        results = {}
        
        # Run pipeline test
        pipeline_result = self.test_pipeline()
        results["pipeline"] = pipeline_result
        
        return results

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test GPT-2 model")
    parser.add_argument("--model", type=str, default="gpt2", help="Model ID to test")
    args = parser.parse_args()
    
    # Create tester and run tests
    tester = TestGpt2Model(model_id=args.model)
    results = tester.run_all_tests()
    
    # Print results
    success = results["pipeline"].get("success", False)
    print(f"Test results: {'Success' if success else 'Failed'}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
