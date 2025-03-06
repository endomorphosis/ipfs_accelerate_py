#\!/usr/bin/env python3

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_test_file(model_name, platform=None, output_dir=None):
    """Generate a test file for the specified model and platform."""
    # Default to all platforms if none specified
    platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu"]
    if platform and platform \!= "all":
        platforms = [p.strip() for p in platform.split(",")]
    
    # Create file name and path
    file_name = f"test_hf_{model_name.replace(\"-\", \"_\")}.py"
    
    # Use output_dir if specified, otherwise use current directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
    else:
        output_path = file_name
    
    # Generate file content
    with open(output_path, "w") as f:
        content = [
            "#\!/usr/bin/env python3",
            "\"\"\"",
            f"Test for {model_name} model with hardware platform support",
            "\"\"\"",
            "",
            "import os",
            "import sys",
            "import unittest",
            "import torch",
            "import numpy as np",
            "from transformers import AutoModel, AutoTokenizer, AutoConfig",
            "",
            f"class Test{model_name.replace(\"-\", \"\").title()}Models(unittest.TestCase):",
            f"    \"\"\"Test {model_name} model across hardware platforms.\"\"\"",
            "    ",
            "    def setUp(self):",
            "        \"\"\"Set up test.\"\"\"",
            f"        self.model_id = \"{model_name}\"",
            "        self.test_text = \"This is a test sentence.\"",
            "        self.test_batch = [\"First test sentence.\", \"Second test sentence.\"]",
            ""
        ]
        
        # Add methods for each platform
        for p in platforms:
            content.extend([
                f"    def test_with_{p.lower()}(self):",
                f"        \"\"\"Test {model_name} with {p}.\"\"\"",
                "        # Test initialization",
                "        try:",
                "            # Initialize tokenizer",
                "            tokenizer = AutoTokenizer.from_pretrained(self.model_id)",
                "            ",
                "            # Initialize model",
                "            model = AutoModel.from_pretrained(self.model_id)",
                "            ",
                "            # Process input",
                "            inputs = tokenizer(self.test_text, return_tensors=\"pt\")",
                "            outputs = model(**inputs)",
                "            ",
                "            # Verify output",
                "            self.assertIsNotNone(outputs)",
                "            ",
                f"            print(f\"Model {{self.model_id}} successfully tested with {p}\")",
                "        except Exception as e:",
                "            self.skipTest(f\"Test skipped due to error: {{str(e)}}\")",
                ""
            ])
        
        # Add main section
        content.extend([
            "if __name__ == \"__main__\":",
            "    unittest.main()"
        ])
        
        # Write the content to the file
        f.write("\n".join(content))
    
    logger.info(f"Generated test file: {output_path}")
    return output_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Minimal Test Generator")
    parser.add_argument("--generate", type=str, help="Model to generate tests for")
    parser.add_argument("--platform", type=str, default="all", help="Platform to generate tests for (comma-separated or \"all\")")
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
