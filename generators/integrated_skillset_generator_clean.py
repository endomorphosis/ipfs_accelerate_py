#!/usr/bin/env python3
"""
Integrated Skillset Generator - Clean Version

This is a simplified version of the skillset generator that works
reliably without syntax errors.
"""

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

def generate_skill_file(model_name, platform=None, output_dir=None):
    model_type = detect_model_modality(model_name)
    
    # Default to all platforms if none specified
    platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "webnn", "webgpu"]
    if platform and platform != "all":
        platforms = [p.strip() for p in platform.split(",")]
    
    # Create file name and path
    file_name = f"skill_hf_{model_name.replace('-', '_')}.py"
    
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
Skill implementation for {model_name} with hardware platform support
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

class {model_name.replace("-", "").title()}Skill:
    """Skill for {model_name} model with hardware platform support."""
    
    def __init__(self, model_id="{model_name}", device=None):
        """Initialize the skill."""
        self.model_id = model_id
        self.device = device or self.get_default_device()
        self.tokenizer = None
        self.model = None
        
    def get_default_device(self):
        """Get the best available device."""
        # Check for CUDA
        if torch.cuda.is_available():
            return "cuda"
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
            if torch.mps.is_available():
                return "mps"
        
        # Default to CPU
        return "cpu"
    
    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is None:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model
            self.model = AutoModel.from_pretrained(self.model_id)
            
            # Move to device
            if self.device != "cpu":
                self.model = self.model.to(self.device)
    
    def process(self, text):
        """Process the input text and return the output."""
        # Ensure model is loaded
        self.load_model()
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move to device
        if self.device != "cpu":
            inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert to numpy for consistent output
        last_hidden_state = outputs.last_hidden_state.cpu().numpy()
        
        # Return formatted results
        return {{
            "model": self.model_id,
            "device": self.device,
            "last_hidden_state_shape": last_hidden_state.shape,
            "embedding": last_hidden_state.mean(axis=1).tolist(),
        }}

# Factory function to create skill instance
def create_skill(model_id="{model_name}", device=None):
    """Create a skill instance."""
    return {model_name.replace("-", "").title()}Skill(model_id=model_id, device=device)
''')
    
    logger.info(f"Generated skill file: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Integrated Skillset Generator - Clean Version")
    parser.add_argument("--model", "-m", type=str, help="Model to generate skill for")
    parser.add_argument("--platform", "-p", type=str, default="all", help="Platform to support (comma-separated or 'all')")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for generated files")
    
    args = parser.parse_args()
    
    if args.model:
        output_file = generate_skill_file(args.model, args.platform, args.output_dir)
        print(f"Generated skill file: {output_file}")
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())