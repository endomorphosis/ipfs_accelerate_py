#!/usr/bin/env python3

"""
Finalize the model lookup implementation by integrating the fix_template_syntax.py script
and testing the model registry with a minimal test file.

This script will:
1. Create a minimal test file with the model lookup functionality
2. Test the model registry with HuggingFace API integration
3. Create a basic verification test file for the most critical model types
4. Update documentation with the latest model lookup capabilities
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_minimal_test_file(output_path, model_type="bert"):
    """Create a minimal test file with model lookup functionality."""
    logger.info(f"Creating minimal test file for {model_type} at {output_path}")
    
    content = f"""#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import model lookup 
try:
    from find_models import get_recommended_default_model
    HAS_MODEL_LOOKUP = True
    logger.info("Model lookup available")
except ImportError:
    HAS_MODEL_LOOKUP = False
    logger.warning("Model lookup not available")

def get_default_model(model_type="{model_type}"):
    \"\"\"Get the default model for a model type.\"\"\""
    if HAS_MODEL_LOOKUP:
        try:
            return get_recommended_default_model(model_type)
        except Exception as e:
            logger.warning(f"Error getting recommended model: {{e}}")
    
    # Fallback defaults
    defaults = {{
        "bert": "google-bert/bert-base-uncased",
        "gpt2": "gpt2",
        "t5": "t5-small",
        "vit": "google/vit-base-patch16-224"
    }}
    
    return defaults.get(model_type, f"{{model_type}}-base")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test model lookup")
    parser.add_argument("--model-type", type=str, default="{model_type}", help="Model type to lookup")
    args = parser.parse_args()
    
    # Get default model
    default_model = get_default_model(args.model_type)
    
    print(f"Default model for {{args.model_type}}: {{default_model}}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
    
    # Write the test file
    with open(output_path, "w") as f:
        f.write(content)
    
    logger.info(f"Created minimal test file at {output_path}")
    return True

def test_model_registry():
    """Test the model registry with HuggingFace API integration."""
    logger.info("Testing model registry with HuggingFace API integration")
    
    try:
        # Try to import modules
        from find_models import get_recommended_default_model, query_huggingface_api
        
        # Test a few model types
        model_types = ["bert", "gpt2", "t5", "vit"]
        results = {}
        
        for model_type in model_types:
            try:
                default_model = get_recommended_default_model(model_type)
                results[model_type] = {"success": True, "default_model": default_model}
                logger.info(f"Default model for {model_type}: {default_model}")
            except Exception as e:
                results[model_type] = {"success": False, "error": str(e)}
                logger.error(f"Error getting default model for {model_type}: {e}")
        
        # Return results
        return results
    
    except ImportError as e:
        logger.error(f"Error importing model lookup modules: {e}")
        return {"error": str(e)}

def update_model_registry_docs(results):
    """Update documentation with the latest model lookup capabilities."""
    logger.info("Updating model registry documentation")
    
    doc_path = "MODEL_REGISTRY_README.md"
    
    content = """# HuggingFace Model Registry

This document describes the model registry system that provides dynamic model lookup
for HuggingFace models. The system queries the HuggingFace API to find popular models
and selects appropriate defaults based on download counts and model size.

## Default Models

The following default models are currently recommended by the system:

"""
    
    for model_type, info in results.items():
        if info.get("success"):
            content += f"- **{model_type}**: `{info['default_model']}`\n"
        else:
            content += f"- **{model_type}**: Error - {info.get('error', 'Unknown error')}\n"
    
    content += """
## Usage

To use the model registry in your code:

```python
from find_models import get_recommended_default_model

# Get the recommended default model for a model type
default_model = get_recommended_default_model("bert")
print(f"Using model: {default_model}")
```

The system will:
1. Try to query the HuggingFace API for popular models
2. Select an appropriate model based on popularity and size
3. Fall back to static defaults if the API is unavailable
4. Cache results to avoid repeated API calls

## Model Selection Criteria

Models are selected based on:
1. Download count (popularity)
2. Size preference (base/small models preferred over large ones)
3. Availability and stability
"""
    
    # Write the documentation
    with open(doc_path, "w") as f:
        f.write(content)
    
    logger.info(f"Updated documentation at {doc_path}")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Finalize model lookup implementation")
    parser.add_argument("--create-test", action="store_true", help="Create a minimal test file")
    parser.add_argument("--test-registry", action="store_true", help="Test the model registry")
    parser.add_argument("--update-docs", action="store_true", help="Update documentation")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    if args.all or args.test_registry:
        results = test_model_registry()
        
        if args.all or args.update_docs:
            update_model_registry_docs(results)
    
    if args.all or args.create_test:
        create_minimal_test_file("test_model_lookup.py")
    
    logger.info("Model lookup implementation finalized")
    return 0

if __name__ == "__main__":
    sys.exit(main())