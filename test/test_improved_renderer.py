#!/usr/bin/env python3
"""
Test the improved template renderer.

This script tests the improved template renderer by generating tests
for representative models of each architecture.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the improved renderer
from improved_template_renderer import (
    generate_test,
    generate_all_tests,
    TemplateManager,
    TemplateRenderer,
    SyntaxValidator
)

def test_single_model(model_type, output_dir="./generated_tests_validation"):
    """Test generating a test file for a single model."""
    logger.info(f"Testing generation for model: {model_type}")
    
    # Generate the test
    result = generate_test(model_type, output_dir)
    
    # Print result
    if result["success"]:
        print(f"✅ Successfully generated test for {model_type} ({result['architecture']})")
        print(f"   Output file: {result['output_file']}")
        
        # Validate syntax
        if result.get("is_valid", False):
            print(f"   Syntax validation: ✅ Valid")
        else:
            print(f"   Syntax validation: ❌ Invalid - {result.get('validation', 'Unknown error')}")
    else:
        print(f"❌ Failed to generate test for {model_type}")
        print(f"   Error: {result['error']}")
    
    return result

def test_representative_models(output_dir="./generated_tests_validation"):
    """Test generating test files for representative models of each architecture."""
    # Define representative models for each architecture
    representative_models = {
        "encoder-only": "bert",
        "decoder-only": "gpt2",
        "encoder-decoder": "t5",
        "vision": "vit",
        "vision-text": "clip",
        "speech": "whisper"
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Test each representative model
    results = {}
    for architecture, model_type in representative_models.items():
        logger.info(f"Testing {architecture} architecture with {model_type} model")
        result = test_single_model(model_type, output_dir)
        results[model_type] = result
    
    # Save summary
    summary = {
        "total": len(results),
        "successful": sum(1 for r in results.values() if r["success"]),
        "valid_syntax": sum(1 for r in results.values() if r.get("is_valid", False)),
        "results": results
    }
    
    summary_file = os.path.join(output_dir, "validation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\nGeneration Summary:")
    print(f"Total models tested: {summary['total']}")
    print(f"Successfully generated: {summary['successful']}")
    print(f"Valid syntax: {summary['valid_syntax']}")
    print(f"Summary saved to: {summary_file}")
    
    return summary

def test_template_renderer():
    """Test the template renderer with a simple template."""
    # Define a simple template
    template = """
#!/usr/bin/env python3

# This is a test template for {{ model_info.name }}
# Architecture: {{ model_info.architecture }}

class Test{{ model_info.class_name }}Model:
    def __init__(self):
        self.model_id = "{{ model_info.id }}"
        self.task = "{{ model_info.task }}"
        
    {% if has_cuda %}
    def test_cuda(self):
        # Test with CUDA support
        print("Testing on CUDA")
        return True
    {% else %}
    def test_cpu(self):
        # Test on CPU only
        print("Testing on CPU")
        return True
    {% endif %}
    
    def run_tests(self):
        print("Running tests for {{ model_info.name }}")
        {% if has_cuda %}
        self.test_cuda()
        {% else %}
        self.test_cpu()
        {% endif %}
"""
    
    # Define context
    context = {
        "model_info": {
            "name": "bert",
            "id": "bert-base-uncased",
            "architecture": "encoder-only",
            "class_name": "Bert",
            "task": "fill-mask"
        },
        "has_cuda": True
    }
    
    # Render template
    rendered = TemplateRenderer.render(template, context)
    
    # Validate syntax
    is_valid, error, _ = SyntaxValidator.validate(rendered)
    
    # Print result
    print("\nTemplate Renderer Test:")
    print("=" * 40)
    print(rendered)
    print("=" * 40)
    print(f"Syntax validation: {'✅ Valid' if is_valid else '❌ Invalid - ' + error}")
    
    return {
        "rendered": rendered,
        "is_valid": is_valid,
        "error": error if not is_valid else None
    }

def main():
    """Main function."""
    # Test the template renderer
    renderer_result = test_template_renderer()
    
    # Test representative models
    models_result = test_representative_models()
    
    # Print overall result
    print("\nOverall Result:")
    all_valid = renderer_result["is_valid"] and models_result["valid_syntax"] == models_result["total"]
    print(f"{'✅ All tests passed' if all_valid else '❌ Some tests failed'}")
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())