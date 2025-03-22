#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from unittest.mock import MagicMock
from pathlib import Path

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

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = MagicMock()
    HAS_PIL = False
    logger.warning("PIL not available, using mock")

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

class TestClipModel:
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        self.model_id = model_id
        self.device = "cuda" if HAS_CUDA else "cpu"
        
    def _get_test_image(self):
        """Get a test image or create a dummy one."""
        test_files = ["test.jpg", "test.png"]
        for file in test_files:
            if Path(file).exists():
                return file
                
        # Create a dummy image
        if HAS_PIL:
            dummy_path = "test_dummy.jpg"
            img = Image.new('RGB', (224, 224), color=(73, 109, 137))
            img.save(dummy_path)
            return dummy_path
            
        return None
        
    def test_pipeline(self):
        """Test the model using pipeline API."""
        if not HAS_TRANSFORMERS or not HAS_PIL:
            missing = []
            if not HAS_TRANSFORMERS:
                missing.append("transformers")
            if not HAS_PIL:
                missing.append("PIL")
            logger.warning(f"Missing dependencies: {', '.join(missing)}, skipping test")
            return {"success": False, "error": f"Missing dependencies: {', '.join(missing)}"}
            
        logger.info(f"Testing {self.model_id} with pipeline API")
        
        try:
            # Create a pipeline
            pipe = transformers.pipeline(
                "zero-shot-image-classification",
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Get test image
            test_image = self._get_test_image()
            if not test_image:
                return {"success": False, "error": "No test image found or created"}
            
            # Run inference
            candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a person"]
            outputs = pipe(test_image, candidate_labels=candidate_labels)
            
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
    parser = argparse.ArgumentParser(description="Test CLIP model")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="Model ID to test")
    args = parser.parse_args()
    
    # Create tester and run tests
    tester = TestClipModel(model_id=args.model)
    results = tester.run_all_tests()
    
    # Print results
    success = results["pipeline"].get("success", False)
    print(f"Test results: {'Success' if success else 'Failed'}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
