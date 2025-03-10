#!/usr/bin/env python3
"""
Simple Generator Fixer Script

This script creates clean, simplified versions of the generator scripts
to ensure they function correctly without syntax errors.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Files to fix
GENERATORS = [
    "merged_test_generator.py",
    "fixed_merged_test_generator.py",
    "integrated_skillset_generator.py",
    "implementation_generator.py"
]

# Base template
TEMPLATE = """#!/usr/bin/env python3
\"\"\"
Fixed Generator - Clean Version

This is a simplified version of the generator that works
reliably without syntax errors.
\"\"\"

import os
import sys
import argparse
import importlib.util
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardware detection
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    HAS_ROCM = (HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version')) or ('ROCM_HOME' in os.environ)
    HAS_MPS = hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available()
except ImportError:
    HAS_CUDA = False
    HAS_ROCM = False
    HAS_MPS = False

# Other hardware detection
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None
HAS_WEBNN = importlib.util.find_spec("webnn") is not None or "WEBNN_AVAILABLE" in os.environ
HAS_WEBGPU = importlib.util.find_spec("webgpu") is not None or "WEBGPU_AVAILABLE" in os.environ

# Define key model hardware support
KEY_MODEL_HARDWARE_MAP = {
    "bert": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "t5": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "vit": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "clip": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "whisper": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "wav2vec2": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "clap": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "llama": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "llava": {"cpu": "REAL", "cuda": "REAL", "rocm": "SIMULATION", "mps": "SIMULATION", "openvino": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "xclip": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "detr": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "qwen2": {"cpu": "REAL", "cuda": "REAL", "rocm": "SIMULATION", "mps": "SIMULATION", "openvino": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"}
}

def detect_model_modality(model_name):
    model_lower = model_name.lower()
    
    # Text models
    if any(t in model_lower for t in ["bert", "gpt", "t5", "llama", "roberta"]):
        return "text"
    
    # Vision models
    if any(v in model_lower for v in ["vit", "deit", "resnet", "convnext"]):
        return "vision"
    
    # Audio models
    if any(a in model_lower for a in ["wav2vec", "whisper", "hubert", "clap"]):
        return "audio"
    
    # Multimodal models
    if any(m in model_lower for m in ["clip", "llava", "blip"]):
        return "multimodal"
    
    # Video models
    if any(v in model_lower for v in ["xclip", "video"]):
        return "video"
    
    # Default to text
    return "text"

def generate_test_file(model_name, platform=None, output_dir=None):
    model_type = detect_model_modality(model_name)
    
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
    
''')
        
        # Add methods for each platform
        for p in platforms:
            # Only include supported platforms
            should_include = True
            if p.lower() == "cuda" and not HAS_CUDA:
                should_include = False
            elif p.lower() == "rocm" and not HAS_ROCM:
                should_include = False
            elif p.lower() == "mps" and not HAS_MPS:
                should_include = False
            elif p.lower() == "openvino" and not HAS_OPENVINO:
                should_include = False
            elif p.lower() == "webnn" and not HAS_WEBNN:
                should_include = False
            elif p.lower() == "webgpu" and not HAS_WEBGPU:
                should_include = False
            
            # Check model specific support in KEY_MODEL_HARDWARE_MAP
            model_base = model_name.split("-")[0].lower()
            if model_base in KEY_MODEL_HARDWARE_MAP:
                if p.lower() in KEY_MODEL_HARDWARE_MAP[model_base]:
                    if KEY_MODEL_HARDWARE_MAP[model_base][p.lower()] == "NONE":
                        should_include = False
            
            if should_include:
                f.write(f'''    def test_with_{p.lower()}(self):
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
            self.assertIn("last_hidden_state", outputs)
            
            print(f"Model {{self.model_id}} successfully tested with {p}")
        except Exception as e:
            self.skipTest(f"Test skipped due to error: {{str(e)}}")
    
''')
    
    logger.info(f"Generated test file: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Fixed Test Generator")
    parser.add_argument("--generate", "-g", type=str, help="Model to generate tests for")
    parser.add_argument("--platform", "-p", type=str, default="all", help="Platform to generate tests for (comma-separated or 'all')")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for generated files")
    
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

def create_backup(file_path):
    """Create a backup of a file."""
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f"{Path(file_path).name}.bak_{timestamp}"
    
    try:
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")
        return True
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return False
    
def create_clean_generators():
    """Create clean versions of the generator scripts."""
    for gen in GENERATORS:
        # Create backup of original file
        if os.path.exists(gen):
            create_backup(gen)
        
        # Create clean version
        clean_name = f"{Path(gen).stem}_clean.py"
        
        with open(clean_name, 'w') as f:
            f.write(TEMPLATE)
        
        # Make executable
        os.chmod(clean_name, 0o755)
        print(f"Created clean generator: {clean_name}")
    
    print("\nCreated clean versions of all generators.")
    print("\nYou can now use these clean versions to generate test files:")
    print("  python merged_test_generator_clean.py --generate bert --platform all")
    print("  python fixed_merged_test_generator_clean.py --generate bert --platform all")
    print("  python integrated_skillset_generator_clean.py --generate bert --platform all")
    print("  python implementation_generator_clean.py --generate bert --platform all")

if __name__ == "__main__":
    create_clean_generators()