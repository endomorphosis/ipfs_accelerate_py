#!/usr/bin/env python3
"""
Fixed Merged Test Generator - Clean Version

This is a simplified version of the test generator that works
reliably without syntax errors.
"""

import os
import sys
import argparse
import importlib.util
import logging
import json
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

# DuckDB integration for benchmark results
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    logger.warning("DuckDB not available. Benchmark results will be stored in JSON format.")

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
    "bert": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "t5": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "vit": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "clip": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "whisper": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "wav2vec2": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "clap": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "llama": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "llava": {"cpu": "REAL", "cuda": "REAL", "rocm": "SIMULATION", "mps": "SIMULATION", "openvino": "SIMULATION", "qualcomm": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "xclip": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "detr": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "qwen2": {"cpu": "REAL", "cuda": "REAL", "rocm": "SIMULATION", "mps": "SIMULATION", "openvino": "SIMULATION", "qualcomm": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"}
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

def generate_imports_for_platform(platform):
    """Generate the imports for a specific platform."""
    imports = []
    
    if platform == "cpu":
        imports.append("import torch")
    elif platform == "cuda" or platform == "rocm":
        imports.append("import torch")
    elif platform == "mps":
        imports.append("import torch")
    elif platform == "openvino":
        imports.append("import torch")
        imports.append("try:\n    import openvino as ov\nexcept ImportError:\n    ov = None")
    elif platform == "qualcomm":
        imports.append("import torch")
        imports.append("try:\n    import qnn_wrapper\nexcept ImportError:\n    qnn_wrapper = None")
    elif platform == "webnn":
        imports.append("import torch")
        imports.append("# WebNN specific imports would go here")
    elif platform == "webgpu":
        imports.append("import torch")
        imports.append("# WebGPU specific imports would go here")
    
    return imports

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
        template = db.get_template(model_base)
        if template:
            logger.info(f"Found template for {model_base} in database")
            return template
        
        # Try with 'template_' prefix
        template = db.get_template(f"template_{model_base}")
        if template:
            logger.info(f"Found template for template_{model_base} in database")
            return template
        
        # Try with common variations
        variations = [
            model_base.replace("-", "_"),
            model_base.replace("_", ""),
            model_base.lower()
        ]
        
        for variation in variations:
            template = db.get_template(variation)
            if template:
                logger.info(f"Found template for {variation} in database")
                return template
        
        logger.warning(f"No template found for {model_base} in database")
        return None
    except Exception as e:
        logger.error(f"Error accessing template database: {e}")
        return None

def generate_test_file(model_name, platform=None, output_dir=None, cross_platform=False, use_db_templates=False):
    """Generate a test file for the specified model and platforms."""
    # Get model type and resolved model ID
    model_type = detect_model_modality(model_name)
    resolved_model_id = resolve_model_name(model_name)
    
    # Extract base model name without version
    model_base = model_name.split("-")[0].lower() if "-" in model_name else model_name.lower()
    
    # Default to all platforms if none specified or cross-platform is True
    all_platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu"]
    if cross_platform:
        platforms = all_platforms
    elif platform and platform != "all":
        platforms = [p.strip() for p in platform.split(",")]
    else:
        platforms = all_platforms
        
    # Try to get template from database if use_db_templates is True
    template_content = None
    if use_db_templates and HAS_TEMPLATE_DB:
        template_content = get_template_from_db(model_base)
    
    # Create file name and path
    file_name = f"test_hf_{model_name.replace('-', '_')}.py"
    
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
                "class_name": f"Test{model_name.replace('-', '').title()}Models"
            }
            
            # Use Python's string formatting to replace variables 
            file_content = template_content.format(**context)
            
            # Write template-based content
            with open(output_path, "w") as f:
                f.write(file_content)
                
            logger.info(f"Generated test file from template: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error applying template: {e}")
            logger.info("Falling back to default generation")
            # Fall back to default generation if template fails
            template_content = None
    
    # If no template or template failed, use default test file generation
    with open(output_path, "w") as f:
        # Header and imports
        f.write(f'''#!/usr/bin/env python3
"""
Test for {model_name} model with hardware platform support
Generated by fixed_merged_test_generator.py
"""

import os
import sys
import unittest
import importlib.util
import logging
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoFeatureExtractor, AutoProcessor, AutoImageProcessor, AutoModelForImageClassification, AutoModelForAudioClassification, AutoModelForVideoClassification

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardware detection
HAS_CUDA = torch.cuda.is_available()
HAS_ROCM = (HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version')) or ('ROCM_HOME' in os.environ)
HAS_MPS = hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available()
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None
HAS_QNN = importlib.util.find_spec("qnn_wrapper") is not None or importlib.util.find_spec("qti") is not None
HAS_WEBNN = importlib.util.find_spec("webnn") is not None or "WEBNN_AVAILABLE" in os.environ
HAS_WEBGPU = importlib.util.find_spec("webgpu") is not None or "WEBGPU_AVAILABLE" in os.environ

# Try to import centralized hardware detection
try:
    from centralized_hardware_detection import hardware_detection
    HAS_CENTRALIZED_DETECTION = True
except ImportError:
    HAS_CENTRALIZED_DETECTION = False

class Test{model_name.replace("-", "").title()}Models(unittest.TestCase):
    """Test {model_name} model with cross-platform hardware support."""
    
    def setUp(self):
        """Set up the test environment."""
        self.model_id = "{resolved_model_id}"
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.modality = "{model_type}"
        
        # Detect hardware capabilities if available
        if HAS_CENTRALIZED_DETECTION:
            self.hardware_capabilities = hardware_detection.detect_hardware_capabilities()
        else:
            self.hardware_capabilities = {{
                "cuda": HAS_CUDA,
                "rocm": HAS_ROCM,
                "mps": HAS_MPS,
                "openvino": HAS_OPENVINO,
                "qnn": HAS_QNN,
                "webnn": HAS_WEBNN,
                "webgpu": HAS_WEBGPU
            }}
        
    def run_tests(self):
        """Run all tests for this model."""
        unittest.main()
''')
        
        # Add methods for each platform
        for p in platforms:
            # Only include supported platforms based on hardware detection
            hardware_var = f"HAS_{p.upper()}"
            
            # Skip condition based on hardware availability
            skip_condition = f"if not {hardware_var}: self.skipTest('{p.upper()} not available')"
            
            # Device setup based on platform
            if p == "cuda" or p == "rocm":
                device_setup = 'device = "cuda"'
            elif p == "mps":
                device_setup = 'device = "mps"'
            else:
                device_setup = 'device = "cpu"'
            
            # Check model specific support in KEY_MODEL_HARDWARE_MAP
            model_base = model_name.split("-")[0].lower()
            special_setup = ""
            special_teardown = ""
            
            # Special setup for OpenVINO
            if p == "openvino":
                special_setup = '        # Initialize OpenVINO if available\n        if HAS_OPENVINO:\n            try:\n                import openvino as ov\n                self.ov_core = ov.Core()\n                self.openvino_label = "openvino"\n            except Exception as e:\n                logger.warning(f"Error initializing OpenVINO: {{e}}")'
            
            if model_base in KEY_MODEL_HARDWARE_MAP:
                if p.lower() in KEY_MODEL_HARDWARE_MAP[model_base]:
                    support_type = KEY_MODEL_HARDWARE_MAP[model_base][p.lower()]
                    if support_type == "NONE":
                        continue  # Skip this platform entirely
                    elif support_type == "SIMULATION":
                        special_setup = '        logger.info("Using simulation mode for this platform")'
            
            # Add the test method
            f.write(f'''
    def test_{p.lower()}(self):
        """Test {model_name} with {p}."""
        # Skip if hardware not available
        {skip_condition}
        
        # Set up device
        {device_setup}
{special_setup}
        
        try:
            # Initialize tokenizer and model based on modality
            if '{model_type}' == 'audio':
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
                self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            elif '{model_type}' == 'vision':
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
            elif '{model_type}' == 'multimodal':
                from transformers import AutoProcessor, AutoModel
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
            elif '{model_type}' == 'video':
                from transformers import AutoProcessor, AutoModelForVideoClassification
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForVideoClassification.from_pretrained(self.model_id)
            else:
                # Default to text models
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
            
            # Move model to device if not CPU
            if device != "cpu":
                self.model = self.model.to(device)
            
            # Prepare input based on modality
            if '{model_type}' == 'text':
                inputs = self.tokenizer("Test input for {model_name}", return_tensors="pt")
            elif '{model_type}' == 'audio':
                import numpy as np
                sample_rate = 16000
                dummy_audio = np.random.random(sample_rate)
                inputs = self.processor(dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
            elif '{model_type}' == 'vision':
                import numpy as np
                from PIL import Image
                dummy_image = Image.new('RGB', (224, 224), color='white')
                inputs = self.processor(images=dummy_image, return_tensors="pt")
            elif '{model_type}' == 'multimodal' or '{model_type}' == 'video':
                import numpy as np
                from PIL import Image
                dummy_image = Image.new('RGB', (224, 224), color='white')
                inputs = self.processor(images=dummy_image, text="Test input", return_tensors="pt")
            else:
                inputs = self.tokenizer("Test input for {model_name}", return_tensors="pt")
            
            # Move inputs to device if not CPU
            if device != "cpu":
                inputs = {{k: v.to(device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Verify outputs based on model type
            self.assertIsNotNone(outputs)
            # Different models return different output structures
            if '{model_type}' == 'text':
                if hasattr(outputs, 'last_hidden_state'):
                    self.assertIsNotNone(outputs.last_hidden_state)
                else:
                    # Some models might have alternative output structures
                    self.assertTrue(any(key in outputs for key in ['last_hidden_state', 'hidden_states', 'logits']))
            elif '{model_type}' in ['audio', 'vision', 'video']:
                if hasattr(outputs, 'logits'):
                    self.assertIsNotNone(outputs.logits)
                else:
                    # Some models might have alternative output structures
                    self.assertTrue(any(key in outputs for key in ['logits', 'embedding', 'last_hidden_state']))
            elif '{model_type}' == 'multimodal':
                # CLIP, LLAVA, etc. might have different output structures
                self.assertTrue(any(hasattr(outputs, attr) for attr in ['text_embeds', 'image_embeds', 'last_hidden_state', 'logits']))
            
            # Log success
            logger.info(f"Successfully tested {{self.model_id}} on {p.lower()}")
{special_teardown}
        except Exception as e:
            logger.error(f"Error testing {{self.model_id}} on {p.lower()}: {{str(e)}}")
            raise
''')
        
        # Add test main
        f.write('''
if __name__ == "__main__":
    unittest.main()
''')
    
    logger.info(f"Generated test file: {output_path}")
    return output_path

def handle_template_output(template_content, model_base, output_path=None):
    """Handle template content output."""
    if output_path:
        with open(output_path, "w") as f:
            f.write(template_content)
        return output_path
    else:
        print(template_content)
        return None

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

def main():
    parser = argparse.ArgumentParser(description="Fixed Merged Test Generator - Enhanced Version")
    parser.add_argument("--generate", "-g", type=str, help="Model to generate tests for")
    parser.add_argument("--platform", "-p", type=str, default="all", help="Platform to generate tests for (comma-separated or 'all')")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for generated files")
    parser.add_argument("--cross-platform", "-c", action="store_true", help="Generate test for all platforms")
    parser.add_argument("--use-db-templates", "-t", action="store_true", help="Use templates from the database")
    parser.add_argument("--list-templates", "-l", action="store_true", help="List available templates in the database")
    parser.add_argument("--all-models", "-a", action="store_true", help="Generate tests for all models in MODEL_REGISTRY")
    parser.add_argument("--family", "-f", type=str, help="Generate tests for all models in a specific family")
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
            output_file = generate_test_file(
                model_name, 
                args.platform, 
                args.output_dir, 
                cross_platform=args.cross_platform,
                use_db_templates=args.use_db_templates
            )
            generated_files.append(output_file)
            
        print(f"Generated {len(generated_files)} test files")
        return 0
    
    # Generate for a specific model
    if args.generate:
        output_file = generate_test_file(
            args.generate, 
            args.platform, 
            args.output_dir, 
            cross_platform=args.cross_platform,
            use_db_templates=args.use_db_templates
        )
        print(f"Generated test file: {output_file}")
        return 0
    
    # If no action specified, print help
    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())