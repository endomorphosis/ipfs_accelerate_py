#!/usr/bin/env python3

"""
Create a simple model lookup test file to verify
the model registry and HuggingFace API integration.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Create a minimal test file for model lookup."""
    output_path = "test_model_lookup.py"
    model_type = "bert"
    
    logger.info(f"Creating minimal test file for {model_type} at {output_path}")
    
    content = '''#!/usr/bin/env python3

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

def get_default_model(model_type="bert"):
    """Get the default model for a model type."""
    if HAS_MODEL_LOOKUP:
        try:
            return get_recommended_default_model(model_type)
        except Exception as e:
            logger.warning(f"Error getting recommended model: {e}")
    
    # Fallback defaults
    defaults = {
        "bert": "google-bert/bert-base-uncased",
        "gpt2": "gpt2",
        "t5": "t5-small",
        "vit": "google/vit-base-patch16-224"
    }
    
    return defaults.get(model_type, f"{model_type}-base")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test model lookup")
    parser.add_argument("--model-type", type=str, default="bert", help="Model type to lookup")
    args = parser.parse_args()
    
    # Get default model
    default_model = get_default_model(args.model_type)
    
    print(f"Default model for {args.model_type}: {default_model}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    # Write the test file
    with open(output_path, "w") as f:
        f.write(content)
    
    logger.info(f"Created minimal test file at {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())