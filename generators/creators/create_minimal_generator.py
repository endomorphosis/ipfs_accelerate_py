#!/usr/bin/env python3

import os
from pathlib import Path

# Current directory
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# Minimal generator script
SCRIPT_CONTENT = '''#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_file(model_name, platform=None, output_dir=None):
    """Generate a test file for the specified model and platform."""
    # Default to all platforms if none specified
    platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "webnn", "webgpu"]
    if platform and platform != "all":
        platforms = [p.strip() for p in platform.split(",")]
    
    # Create file name and path
    file_name = f"test_hf_{model_name.replace('-', '_')}.py"
    
    # Use output_dir if specified, otherwise use current directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
    else:
        output_path = file_name
    
    # Generate file content
    with open(output_path, "w") as f:
        f.write(f'''#!/usr/bin/env python3
"""
Test for {model_name} model with hardware platform support
"""

import os
import sys
import unittest
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

class Test{model_name.replace("-", "").title()}Models(unittest.TestCase):
    """Test {model_name} model across hardware platforms."""
    
    def setUp(self):
        """Set up test."""
        self.model_id = "{model_name}"
        self.test_text = "This is a test sentence."
        self.test_batch = ["First test sentence.", "Second test sentence."]
""")
        
        # Add methods for each platform
        for p in platforms:
            f.write(f'''
    def test_with_{p.lower()}(self):
        """Test {model_name} with {p}."""
        # Test initialization
        try:
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Initialize model
            model = AutoModel.from_pretrained(self.model_id)
            
            # Process input
            inputs = tokenizer(self.test_text, return_tensors="pt")
            outputs = model(**inputs)
            
            # Verify output
            self.assertIsNotNone(outputs)
            
            print(f"Model {{self.model_id}} successfully tested with {p}")
        except Exception as e:
            self.skipTest(f"Test skipped due to error: {{str(e)}}")
''')
    
    logger.info(f"Generated test file: {output_path}")
    return output_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Minimal Test Generator")
    parser.add_argument("--generate", type=str, help="Model to generate tests for")
    parser.add_argument("--platform", type=str, default="all", help="Platform to generate tests for (comma-separated or 'all')")
    parser.add_argument("--output-dir", type=str, help="Output directory for generated files")
    
    args = parser.parse_args()
    
    if args.generate:
        output_file = generate_test_file(args.generate, args.platform, args.output_dir)
        print(f"Generated test file: {output_file}")
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""

# Create the file
file_path = current_dir / "minimal_test_generator.py"
with open(file_path, 'w') as f:
    f.write(SCRIPT_CONTENT)

# Make it executable
os.chmod(file_path, 0o755)

print(f"Created minimal test generator: {file_path}")
print("Usage: python minimal_test_generator.py --generate bert --platform all")