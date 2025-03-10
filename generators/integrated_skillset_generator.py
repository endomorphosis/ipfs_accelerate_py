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

# Template database integration
try:
    from hardware_test_templates.template_database import TemplateDatabase
    HAS_TEMPLATE_DB = True
except ImportError:
    HAS_TEMPLATE_DB = False
    logger.warning("Template database not available. Using hardcoded templates.")

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
HAS_QNN = importlib.util.find_spec("qnn_wrapper") is not None or importlib.util.find_spec("qti") is not None
HAS_WEBNN = importlib.util.find_spec("webnn") is not None or "WEBNN_AVAILABLE" in os.environ
HAS_WEBGPU = importlib.util.find_spec("webgpu") is not None or "WEBGPU_AVAILABLE" in os.environ

# Model registry for common test models
MODEL_REGISTRY = {
    "bert": "bert-base-uncased",
    "t5": "t5-small",
    "vit": "google/vit-base-patch16-224",
    "clip": "openai/clip-vit-base-patch32",
    "whisper": "openai/whisper-tiny",
    "wav2vec2": "facebook/wav2vec2-base",
    "clap": "laion/clap-htsat-unfused",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava_next": "llava-hf/llava-1.6-34b-hf",
    "xclip": "microsoft/xclip-base-patch32",
    "detr": "facebook/detr-resnet-50",
    "qwen2": "Qwen/Qwen2-7B-Instruct"
}

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

def resolve_model_name(model_name):
    """Resolve model name to get the full model ID if it's a short name."""
    # If it's a key in MODEL_REGISTRY, return the full model ID
    model_base = model_name.split("-")[0].lower() if "-" in model_name else model_name.lower()
    if model_base in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_base]
    # Otherwise, return the model name as is
    return model_name

def get_template_from_db(model_base):
    """Get a template from the template database."""
    if not HAS_TEMPLATE_DB:
        return None
    
    try:
        db = TemplateDatabase()
        # Try skill specific template first
        template = db.get_template(f"skill_{model_base}")
        if template:
            logger.info(f"Found skill template for {model_base} in database")
            return template
        
        # Try with model name
        template = db.get_template(model_base)
        if template:
            logger.info(f"Found template for {model_base} in database")
            return template
        
        # Try with common variations
        variations = [
            model_base.replace("-", "_"),
            model_base.replace("_", ""),
            model_base.lower()
        ]
        
        for variation in variations:
            template = db.get_template(f"skill_{variation}")
            if template:
                logger.info(f"Found skill template for {variation} in database")
                return template
        
        logger.warning(f"No template found for {model_base} in database")
        return None
    except Exception as e:
        logger.error(f"Error accessing template database: {e}")
        return None

def generate_skill_file(model_name, platform=None, output_dir=None, use_db_templates=False):
    model_type = detect_model_modality(model_name)
    resolved_model_id = resolve_model_name(model_name)
    
    # Extract base model name without version
    model_base = model_name.split("-")[0].lower() if "-" in model_name else model_name.lower()
    
    # Default to all platforms if none specified
    platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
    if platform and platform != "all":
        platforms = [p.strip() for p in platform.split(",")]
        
    # Try to get template from database if use_db_templates is True
    template_content = None
    if use_db_templates and HAS_TEMPLATE_DB:
        template_content = get_template_from_db(model_base)
    
    # Create file name and path
    file_name = f"skill_hf_{model_name.replace('-', '_')}.py"
    
    # Use output_dir if specified, otherwise use current directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
    else:
        output_path = file_name
    
    # Generate file content based on either database template or default generation
    if template_content and use_db_templates:
        try:
            # Use template from database with context variables replaced
            context = {
                "model_name": model_name,
                "model_id": resolved_model_id,
                "model_type": model_type,
                "model_base": model_base,
                "platforms": ",".join(platforms),
                "modality": model_type,
                "class_name": f"{model_name.replace('-', '').title()}Skill"
            }
            
            # Use Python's string formatting to replace variables
            file_content = template_content.format(**context)
            
            # Write template-based content
            with open(output_path, "w") as f:
                f.write(file_content)
            
            logger.info(f"Generated skill file from template: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error applying template: {e}")
            logger.info("Falling back to default generation")
            # Fall back to default generation if template fails
            template_content = None
    
    # If no template or template failed, use default skill file generation
    with open(output_path, "w") as f:
        f.write(f'''#!/usr/bin/env python3
"""
Skill implementation for {model_name} with hardware platform support
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoFeatureExtractor, AutoProcessor, AutoImageProcessor, AutoModelForImageClassification, AutoModelForAudioClassification, AutoModelForVideoClassification

class {model_name.replace("-", "").title()}Skill:
    """Skill for {model_name} model with hardware platform support."""
    
    def __init__(self, model_id="{resolved_model_id}", device=None):
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
        """Load the model and tokenizer based on modality."""
        if self.model is None:
            # Determine model modality
            modality = "{model_type}"
            
            # Load appropriate tokenizer/processor and model based on modality
            if modality == "audio":
                self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            elif modality == "vision":
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
            elif modality == "multimodal":
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
            elif modality == "video":
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForVideoClassification.from_pretrained(self.model_id)
            else:
                # Default to text
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
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

def detect_available_hardware():
    """Detect available hardware platforms."""
    hardware = []
    if True:  # CPU is always available
        hardware.append("cpu")
    if HAS_CUDA:
        hardware.append("cuda")
    if HAS_ROCM:
        hardware.append("rocm")
    if HAS_MPS:
        hardware.append("mps")
    if HAS_OPENVINO:
        hardware.append("openvino")
    if HAS_QNN:
        hardware.append("qnn")
    if HAS_WEBNN:
        hardware.append("webnn")
    if HAS_WEBGPU:
        hardware.append("webgpu")
    
    return hardware

def generate_template_database_info():
    """Generate information about templates in the database."""
    if not HAS_TEMPLATE_DB:
        return "Template database not available. Install hardware_test_templates first."
    
    try:
        db = TemplateDatabase()
        templates = db.list_templates()
        
        result = []
        result.append(f"Found {len(templates)} templates in database:")
        
        # Group by model family
        families = {}
        for template in templates:
            family = template['model_family']
            if family not in families:
                families[family] = []
            families[family].append(template)
        
        # Output templates by family
        for family, templates in families.items():
            result.append(f"\n{family.upper()} ({len(templates)} templates):")
            for template in templates:
                result.append(f"  - {template['model_id']} ({template['modality']})")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error accessing template database: {e}"

def main():
    parser = argparse.ArgumentParser(description="Integrated Skillset Generator - Enhanced Version")
    parser.add_argument("--model", "-m", type=str, help="Model to generate skill for")
    parser.add_argument("--platform", "-p", type=str, default="all", help="Platform to support (comma-separated or 'all')")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for generated files")
    parser.add_argument("--use-db-templates", "-t", action="store_true", help="Use templates from the database")
    parser.add_argument("--list-templates", "-l", action="store_true", help="List available templates in the database")
    parser.add_argument("--all-models", "-a", action="store_true", help="Generate skills for all models in MODEL_REGISTRY")
    parser.add_argument("--cross-platform", "-c", action="store_true", help="Generate skill with cross-platform support")
    parser.add_argument("--detect-hardware", "-d", action="store_true", help="Detect available hardware platforms")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # List templates
    if args.list_templates:
        print(generate_template_database_info())
        return 0
    
    # Detect hardware
    if args.detect_hardware:
        hardware = detect_available_hardware()
        print(f"Available hardware platforms: {', '.join(hardware)}")
        return 0
    
    # Generate for all models
    if args.all_models:
        generated_files = []
        for model_name in MODEL_REGISTRY.keys():
            output_file = generate_skill_file(
                model_name, 
                args.platform, 
                args.output_dir, 
                use_db_templates=args.use_db_templates
            )
            generated_files.append(output_file)
            
        print(f"Generated {len(generated_files)} skill files")
        return 0
    
    # Generate for a specific model
    if args.model:
        output_file = generate_skill_file(
            args.model, 
            args.platform, 
            args.output_dir, 
            use_db_templates=args.use_db_templates
        )
        print(f"Generated skill file: {output_file}")
        return 0
    
    # If no action specified, print help
    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())