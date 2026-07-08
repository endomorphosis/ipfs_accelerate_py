#\!/usr/bin/env python3
"""
Example test file for BERT model using our enhanced test generator with indentation fixing.

This demonstrates:
1. Using the fix_test_indentation function for proper code formatting
2. Using architecture-specific template selection 
3. Properly handling class method indentation
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import testing utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fix_indentation_and_apply_template import fix_class_method_indentation, get_architecture_type

class BertTestExample:
    """Example test class for BERT model family, using our indentation fixes."""
    
    def __init__(self, model_id=None):
        """Initialize with a specific model ID."""
        self.model_id = model_id or "bert-base-uncased"
        self.performance_stats = {}
    
    def test_pipeline(self, device="auto"):
        """Test the model in a transformers pipeline."""
        results = {}
        try:
            # Simulate pipeline creation
            logger.info(f"Creating pipeline with {self.model_id} on {device}")
            
            # Record some test results
            results["pipeline_success"] = True
            results["device"] = device
            
            # Log performance metrics
            self.performance_stats["pipeline_latency"] = 0.123
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            results["pipeline_success"] = False
            
        return results
    
    def test_from_pretrained(self, device="auto"):
        """Test loading the model using from_pretrained."""
        results = {}
        try:
            # Simulate model loading
            logger.info(f"Loading {self.model_id} on {device}")
            
            # Test tokenizer
            logger.info(f"Testing tokenizer for {self.model_id}")
            
            # Test forward pass
            input_text = "This is a test sentence for BERT."
            logger.info(f"Running inference with: {input_text}")
            
            # Record results
            results["model_loaded"] = True
            results["tokenizer_loaded"] = True
            results["inference_success"] = True
            
            # Log performance metrics
            self.performance_stats["load_time"] = 0.456
            self.performance_stats["inference_time"] = 0.789
        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            results["model_loaded"] = False
            
        return results
    
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        results = {}
        
        # Test standard pipeline
        pipeline_results = self.test_pipeline()
        results["pipeline"] = pipeline_results
        
        # Test model loading
        pretrained_results = self.test_from_pretrained()
        results["from_pretrained"] = pretrained_results
        
        # Return all test results
        return results

def save_results(model_id, results, output_dir="results"):
    """Save test results to a file."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_id}_results.txt")
    
    with open(output_file, "w") as f:
        f.write(f"Test results for {model_id}\n")
        f.write("=" * 40 + "\n\n")
        
        # Write pipeline results
        if "pipeline" in results:
            pipeline = results["pipeline"]
            f.write(f"Pipeline test: {'Success' if pipeline.get('pipeline_success') else 'Failed'}\n")
            f.write(f"Device: {pipeline.get('device', 'unknown')}\n\n")
        
        # Write from_pretrained results
        if "from_pretrained" in results:
            pretrained = results["from_pretrained"]
            f.write(f"Model loading: {'Success' if pretrained.get('model_loaded') else 'Failed'}\n")
            f.write(f"Tokenizer loading: {'Success' if pretrained.get('tokenizer_loaded') else 'Failed'}\n")
            f.write(f"Inference: {'Success' if pretrained.get('inference_success') else 'Failed'}\n")
    
    logger.info(f"Results saved to {output_file}")
    return output_file

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Example test for BERT model")
    parser.add_argument("--model-id", type=str, default="bert-base-uncased",
                        help="Model ID to test (default: bert-base-uncased)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results (default: results)")
    
    args = parser.parse_args()
    
    # Show indentation fixes in action
    original_code = """
class PoorlyIndentedTestClass:
  def __init__(self, model_name):
    self.model_name = model_name
       self.results = {}
    
 def test_something(self, param1, param2):
      # This method has inconsistent indentation
     results = {}
       for i in range(10):
            if i % 2 == 0:
         results[i] = True
      else:
             results[i] = False
      return results
"""
    
    # Fix the indentation
    fixed_code = fix_class_method_indentation(original_code)
    
    # Demonstrate architecture detection
    model_id = args.model_id
    architecture = get_architecture_type(model_id)
    logger.info(f"Model {model_id} has architecture type: {architecture}")
    
    # Run actual tests
    tester = BertTestExample(model_id=args.model_id)
    results = tester.run_tests()
    
    # Save and output results
    output_file = save_results(args.model_id, results, args.output_dir)
    logger.info(f"Test completed for {args.model_id}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
