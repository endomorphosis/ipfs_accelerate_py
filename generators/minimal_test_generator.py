#\!/usr/bin/env python3

import os
import sys
import argparse

def generate_test(model, platform="all"):
    """Generate a test for the given model and platform."""
    file_name = f"test_hf_{model.replace('-', '_')}.py"
    
    platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu"]
    if platform \!= "all":
        platforms = [p.strip() for p in platform.split(",")]
    
    with open(file_name, "w") as f:
        f.write(f"""#\!/usr/bin/env python3

import os
import sys
import unittest
import torch
from transformers import AutoModel, AutoTokenizer

# Hardware detection
HAS_CUDA = torch.cuda.is_available() if hasattr(torch, "cuda") else False
HAS_MPS = hasattr(torch, "mps") and torch.mps.is_available() if hasattr(torch, "mps") else False
HAS_ROCM = hasattr(torch, "_C") and hasattr(torch._C, "_rocm_version") if hasattr(torch, "_C") else False

class Test{model.replace('-', '').capitalize()}(unittest.TestCase):
    """Test {model} model."""
    
    def setUp(self):
        self.model_name = "{model}"
        self.tokenizer = None
        self.model = None
    
""")
        
        # Add test methods for each platform
        for p in platforms:
            f.write(f"""    def test_{p}(self):
        """Test on {p} platform."""
        # Skip if hardware not available
        if "{p}" == "cuda" and not HAS_CUDA:
            self.skipTest("CUDA not available")
        elif "{p}" == "mps" and not HAS_MPS:
            self.skipTest("MPS not available")
        elif "{p}" == "rocm" and not HAS_ROCM:
            self.skipTest("ROCm not available")
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Test basic functionality
        inputs = self.tokenizer("Hello, world\!", return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        print(f"Successfully tested {self.model_name} on {p}")
        
""")
        
        # Add main
        f.write("""
if __name__ == "__main__":
    unittest.main()
""")
    
    print(f"Generated test file: {file_name}")
    return file_name

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple test generator")
    parser.add_argument("-g", "--generate", type=str, help="Model to generate test for")
    parser.add_argument("-p", "--platform", type=str, default="all", 
                      help="Platform(s) to test on (comma-separated or 'all')")
    parser.add_argument("-o", "--output", type=str, help="Output file path (default: test_<model>.py)")
    
    args = parser.parse_args()
    
    if args.generate:
        output_file = generate_test(args.generate, args.platform)
        print(f"Test file generated: {output_file}")
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
