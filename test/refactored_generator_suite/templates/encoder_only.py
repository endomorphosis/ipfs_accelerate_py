#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Encoder-Only Template for the refactored generator suite.
This template is used for generating tests for encoder-only models like BERT, RoBERTa, etc.
"""

import logging
from typing import Dict, Any, List

from .base import TemplateBase


class EncoderOnlyTemplate(TemplateBase):
    """Template for encoder-only models like BERT, RoBERTa, etc."""
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this template.
        
        Returns:
            Dictionary of metadata.
        """
        metadata = super().get_metadata()
        metadata.update({
            "name": "EncoderOnlyTemplate",
            "description": "Template for encoder-only models",
            "supported_architectures": ["encoder-only"],
            "supported_models": [
                "bert", "roberta", "distilbert", "albert", "electra", "camembert", 
                "xlm-roberta", "deberta", "ernie", "rembert"
            ]
        })
        return metadata
    
    def get_imports(self) -> List[str]:
        """Get the imports required by this template.
        
        Returns:
            List of import statements.
        """
        imports = super().get_imports()
        imports.extend([
            "import torch",
            "from transformers import AutoModelForMaskedLM, AutoTokenizer",
            "from transformers import pipeline"
        ])
        return imports
    
    def get_template_str(self) -> str:
        """Get the template string.
        
        Returns:
            The template as a string.
        """
        return """#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generated test for {{ model_info.name }}
Architecture: {{ model_info.architecture }}
Generated on: {{ timestamp }}
"""

{% for import_stmt in imports %}
{{ import_stmt }}
{% endfor %}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardware detection
def select_device():
    """Select the appropriate device based on what's available."""
    {% if has_cuda %}
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using GPU.")
        return "cuda"
    {% endif %}
    {% if has_rocm %}
    if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
        logger.info("ROCm is available. Using AMD GPU.")
        return "cuda"  # ROCm uses the cuda device type
    {% endif %}
    {% if has_mps %}
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS is available. Using Apple Silicon GPU.")
        return "mps"
    {% endif %}
    logger.info("No GPU acceleration available. Using CPU.")
    return "cpu"

# Environment variable for mock mode
def is_mock_mode():
    """Check if running in mock mode."""
    return os.environ.get("MOCK_TRANSFORMERS", "").lower() == "true"

class {{ model_info.class_name }}Test:
    """Test case for {{ model_info.name }} model."""
    
    def __init__(self, model_name="{{ model_info.id }}", output_dir=None, device=None):
        """Initialize the test.
        
        Args:
            model_name: Name or path of the model to test
            output_dir: Optional directory to save outputs
            device: Device to run on (cuda, cpu, etc.)
        """
        self.model_name = model_name
        self.output_dir = output_dir or "./output"
        self.device = device or select_device()
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tracking variables
        self.results = {}
        self.start_time = None
    
    def run(self):
        """Run all tests for this model."""
        logger.info(f"Running tests for {self.model_name} on {self.device}")
        self.start_time = time.time()
        
        # Run tests
        self.results["pipeline"] = self.test_pipeline()
        self.results["masked_lm"] = self.test_masked_lm()
        {% if has_openvino %}
        self.results["openvino"] = self.test_openvino()
        {% endif %}
        
        # Summarize
        self.results["duration"] = time.time() - self.start_time
        self.results["success"] = all(r.get("success", False) for r in self.results.values() if isinstance(r, dict))
        
        self._save_results()
        return self.results
    
    def test_pipeline(self):
        """Test the pipeline API."""
        logger.info("Testing pipeline API...")
        result = {"name": "pipeline", "success": False}
        
        try:
            if is_mock_mode():
                logger.info("Mock mode enabled. Skipping actual pipeline creation.")
                result["success"] = True
                result["mock"] = True
                return result
            
            # Create pipeline
            fill_mask = pipeline(
                "fill-mask",
                model=self.model_name,
                device=self.device
            )
            
            # Test inference
            test_input = f"Paris is the {fill_mask.tokenizer.mask_token} of France."
            outputs = fill_mask(test_input)
            
            # Validate outputs
            if isinstance(outputs, list) and len(outputs) > 0:
                # Log the top prediction
                top_prediction = outputs[0]["token_str"]
                logger.info(f"Top prediction: {top_prediction}")
                
                # Check if 'capital' is in top predictions
                result["has_capital"] = any("capital" in output["token_str"].lower() for output in outputs)
                result["top_predictions"] = [output["token_str"] for output in outputs[:3]]
                result["success"] = True
            else:
                result["error"] = f"Unexpected output format: {outputs}"
        
        except Exception as e:
            logger.error(f"Error in pipeline test: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def test_masked_lm(self):
        """Test masked language modeling."""
        logger.info("Testing masked language modeling...")
        result = {"name": "masked_lm", "success": False}
        
        try:
            if is_mock_mode():
                logger.info("Mock mode enabled. Skipping actual model loading.")
                result["success"] = True
                result["mock"] = True
                return result
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
            
            # Test inference
            test_input = f"Paris is the {tokenizer.mask_token} of France."
            inputs = tokenizer(test_input, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Get predictions
            mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
            logits = outputs.logits
            mask_token_logits = logits[0, mask_token_index, :]
            top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
            top_tokens_str = [tokenizer.decode(token) for token in top_tokens]
            
            logger.info(f"Top tokens: {top_tokens_str}")
            result["top_tokens"] = top_tokens_str
            result["success"] = True
        
        except Exception as e:
            logger.error(f"Error in masked LM test: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    {% if has_openvino %}
    def test_openvino(self):
        """Test OpenVINO integration."""
        logger.info("Testing OpenVINO integration...")
        result = {"name": "openvino", "success": False}
        
        try:
            from optimum.intel import OVModelForMaskedLM
            
            if is_mock_mode():
                logger.info("Mock mode enabled. Skipping actual OpenVINO model loading.")
                result["success"] = True
                result["mock"] = True
                return result
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = OVModelForMaskedLM.from_pretrained(
                self.model_name,
                from_transformers=True,
                device="CPU"
            )
            
            # Test inference
            test_input = f"Paris is the {tokenizer.mask_token} of France."
            inputs = tokenizer(test_input, return_tensors="pt")
            
            outputs = model(**inputs)
            
            # Get predictions
            mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
            logits = outputs.logits
            mask_token_logits = logits[0, mask_token_index, :]
            top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
            top_tokens_str = [tokenizer.decode(token) for token in top_tokens]
            
            logger.info(f"OpenVINO top tokens: {top_tokens_str}")
            result["top_tokens"] = top_tokens_str
            result["success"] = True
            
        except ImportError:
            logger.warning("OpenVINO not available. Skipping test.")
            result["error"] = "OpenVINO not available"
            
        except Exception as e:
            logger.error(f"Error in OpenVINO test: {str(e)}")
            result["error"] = str(e)
            
        return result
    {% endif %}
    
    def _save_results(self):
        """Save test results to output directory."""
        if not self.output_dir:
            return
            
        try:
            results_file = os.path.join(self.output_dir, f"{self.model_name.replace('/', '_')}_results.json")
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test the {{ model_info.name }} model")
    parser.add_argument("--model", default="{{ model_info.id }}", help="Model name or path")
    parser.add_argument("--output-dir", help="Output directory for test results")
    parser.add_argument("--device", help="Device to run on (cuda, cpu, etc.)")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set mock mode if requested
    if args.mock:
        os.environ["MOCK_TRANSFORMERS"] = "true"
    
    # Run the test
    test = {{ model_info.class_name }}Test(
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device
    )
    results = test.run()
    
    # Print results summary
    success = results.get("success", False)
    print(f"\nTest results for {args.model}:")
    print(f"Success: {'Yes' if success else 'No'}")
    print(f"Duration: {results.get('duration', 0):.2f} seconds")
    
    if not success:
        for name, result in results.items():
            if isinstance(result, dict) and not result.get("success", True):
                print(f"Failed: {name} - {result.get('error', 'Unknown error')}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
"""