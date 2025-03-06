#!/usr/bin/env python3
"""
Enhanced Test Generator with Hardware Support and Template Integration

This generator creates test files for Hugging Face models with comprehensive
hardware platform support, including CPU, CUDA, ROCm, MPS, OpenVINO, Qualcomm,
WebNN, and WebGPU.

Usage:
  python simple_test_generator.py -g bert -p all
  python simple_test_generator.py -g vit -p cpu,cuda,webgpu -o test_vit_web.py
  python simple_test_generator.py -g llama -p cpu,cuda --use-template
"""

import os
import sys
import argparse
import importlib.util
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_generator")

# Check for DuckDB availability for template database
HAS_DUCKDB = importlib.util.find_spec("duckdb") is not None
if HAS_DUCKDB:
    try:
        import duckdb
        logger.info("DuckDB available for template database integration")
    except ImportError:
        HAS_DUCKDB = False
        logger.warning("Failed to import DuckDB")

# Define constants
DEFAULT_TEMPLATE_DB = "template_db.duckdb"

def detect_hardware():
    """Detect available hardware platforms."""
    hardware = {
        "cpu": True,  # CPU is always available
        "cuda": False,
        "rocm": False,
        "mps": False,
        "openvino": False,
        "qualcomm": False,
        "webnn": False,
        "webgpu": False
    }
    
    # Check for PyTorch
    try:
        import torch
        
        # CUDA detection
        hardware["cuda"] = torch.cuda.is_available()
        
        # ROCm detection
        if hasattr(torch, "_C") and hasattr(torch._C, "_rocm_version"):
            hardware["rocm"] = True
        elif "ROCM_HOME" in os.environ:
            hardware["rocm"] = True
            
        # MPS (Apple Silicon) detection
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
            hardware["mps"] = torch.mps.is_available()
        
        logger.info(f"PyTorch detected with CUDA: {hardware['cuda']}, ROCm: {hardware['rocm']}, MPS: {hardware['mps']}")
    except ImportError:
        logger.warning("PyTorch not available, hardware detection limited")
    
    # OpenVINO detection
    hardware["openvino"] = importlib.util.find_spec("openvino") is not None
    
    # Qualcomm detection
    hardware["qualcomm"] = (
        importlib.util.find_spec("qnn_wrapper") is not None or
        importlib.util.find_spec("qti") is not None or
        "QUALCOMM_SDK" in os.environ
    )
    
    # WebNN detection
    hardware["webnn"] = (
        importlib.util.find_spec("webnn") is not None or
        "WEBNN_AVAILABLE" in os.environ or
        "WEBNN_SIMULATION" in os.environ
    )
    
    # WebGPU detection
    hardware["webgpu"] = (
        importlib.util.find_spec("webgpu") is not None or
        importlib.util.find_spec("wgpu") is not None or
        "WEBGPU_AVAILABLE" in os.environ or
        "WEBGPU_SIMULATION" in os.environ
    )
    
    # Log detected hardware
    available_hw = [hw for hw, available in hardware.items() if available]
    logger.info(f"Detected hardware: {', '.join(available_hw)}")
    
    return hardware

def get_template_from_db(model_type, template_type="test", platform=None, db_path=None):
    """
    Get a template from the template database.
    
    Args:
        model_type: Type of model (bert, t5, etc.)
        template_type: Type of template (test, base, etc.)
        platform: Optional platform specific template
        db_path: Path to template database
        
    Returns:
        Template string or None if not found
    """
    if not HAS_DUCKDB:
        logger.warning("DuckDB not available, cannot use template database")
        return None
    
    db_path = db_path or DEFAULT_TEMPLATE_DB
    if not os.path.exists(db_path):
        logger.warning(f"Template database not found at {db_path}")
        return None
    
    try:
        conn = duckdb.connect(db_path)
        
        # Query for the template
        if platform:
            query = """
            SELECT template FROM templates 
            WHERE model_type = ? AND template_type = ? AND platform = ?
            ORDER BY id DESC LIMIT 1
            """
            result = conn.execute(query, [model_type, template_type, platform]).fetchone()
        else:
            query = """
            SELECT template FROM templates 
            WHERE model_type = ? AND template_type = ? AND platform IS NULL
            ORDER BY id DESC LIMIT 1
            """
            result = conn.execute(query, [model_type, template_type]).fetchone()
        
        conn.close()
        
        if result and result[0]:
            logger.info(f"Found template for {model_type} ({template_type})")
            return result[0]
        else:
            logger.warning(f"No template found for {model_type} ({template_type})")
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving template from database: {e}")
        return None

def detect_model_category(model_name):
    """
    Detect the category of a model based on its name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Category string: text, vision, audio, multimodal, or video
    """
    model_lower = model_name.lower()
    
    # Text models
    if any(name in model_lower for name in ["bert", "gpt", "t5", "llama", "roberta", "mistral"]):
        return "text"
    
    # Vision models
    if any(name in model_lower for name in ["vit", "resnet", "deit", "detr"]):
        return "vision"
    
    # Audio models
    if any(name in model_lower for name in ["wav2vec", "whisper", "clap", "hubert"]):
        return "audio"
    
    # Multimodal models
    if any(name in model_lower for name in ["clip", "blip", "llava", "fuyu"]):
        return "multimodal"
    
    # Video models
    if any(name in model_lower for name in ["videomae", "xclip", "vivit"]):
        return "video"
    
    # Default to text
    return "text"

def generate_test(model, platform="all", output_file=None, use_template=False, template_db=None):
    """
    Generate a test file for the given model and platform.
    
    Args:
        model: Model name to generate test for
        platform: Hardware platform(s) to include (comma-separated or 'all')
        output_file: Optional output file path
        use_template: Whether to use templates from the database
        template_db: Optional path to template database
        
    Returns:
        Path to the generated file
    """
    # Set output file path
    if output_file:
        file_name = output_file
    else:
        file_name = f"test_hf_{model.replace('-', '_')}.py"
    
    # Determine platforms to include
    all_platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu"]
    platforms = all_platforms if platform == "all" else [p.strip() for p in platform.split(",")]
    
    # Detect model category
    model_category = detect_model_category(model)
    logger.info(f"Detected model category: {model_category}")
    
    # Check if we should use a template
    model_base = model.split("-")[0].lower()
    template = None
    if use_template:
        template = get_template_from_db(model_base, "test", None, template_db)
        if not template:
            # Try using category template as fallback
            template = get_template_from_db(model_category, "test", None, template_db)
            if template:
                logger.info(f"Using category template for {model_category}")
    
    # Generate the test file
    with open(file_name, "w") as f:
        if template:
            # Replace placeholders in the template
            filled_template = template.replace("{{model_name}}", model)
            filled_template = filled_template.replace("{{model_category}}", model_category)
            
            # Fix the class name placeholder if it exists
            class_name_placeholder = "{{model_name.replace(\"-\", \"\").capitalize()}}"
            if class_name_placeholder in filled_template:
                # Get properly capitalized class name
                class_name = model.replace("-", "").capitalize()
                filled_template = filled_template.replace(class_name_placeholder, class_name)
            
            # Write the filled template
            f.write(filled_template)
            logger.info(f"Generated test file using template from database")
        else:
            # Use the standard built-in template
            logger.info(f"Using built-in template (no database template found)")
            
            # Write file header and imports
            f.write(f'''#!/usr/bin/env python3
"""
Test file for {model} with cross-platform hardware support
"""

import os
import sys
import unittest
import importlib.util
import logging
import torch
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardware detection
HAS_CUDA = torch.cuda.is_available() if hasattr(torch, "cuda") else False
HAS_MPS = hasattr(torch, "mps") and torch.mps.is_available() if hasattr(torch, "mps") else False
HAS_ROCM = (hasattr(torch, "_C") and hasattr(torch._C, "_rocm_version")) if hasattr(torch, "_C") else False
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None
HAS_QUALCOMM = (
    importlib.util.find_spec("qnn_wrapper") is not None or 
    importlib.util.find_spec("qti") is not None or
    "QUALCOMM_SDK" in os.environ
)
HAS_WEBNN = (
    importlib.util.find_spec("webnn") is not None or
    "WEBNN_AVAILABLE" in os.environ or
    "WEBNN_SIMULATION" in os.environ
)
HAS_WEBGPU = (
    importlib.util.find_spec("webgpu") is not None or
    importlib.util.find_spec("wgpu") is not None or
    "WEBGPU_AVAILABLE" in os.environ or
    "WEBGPU_SIMULATION" in os.environ
)

class Test{model.replace("-", "").capitalize()}(unittest.TestCase):
    """Test {model} model with hardware platform support."""
    
    def setUp(self):
        """Set up the test environment."""
        self.model_name = "{model}"
        self.tokenizer = None
        self.model = None
''')
            
            # Add test methods for each platform
            for p in platforms:
                skip_checks = []
                device_setup = []
                
                # Add platform-specific checks
                if p == "cuda":
                    skip_checks.append('if not HAS_CUDA: self.skipTest("CUDA not available")')
                    device_setup.append('device = "cuda"')
                elif p == "rocm":
                    skip_checks.append('if not HAS_ROCM: self.skipTest("ROCm not available")')
                    device_setup.append('device = "cuda"  # ROCm uses CUDA API')
                elif p == "mps":
                    skip_checks.append('if not HAS_MPS: self.skipTest("MPS not available")')
                    device_setup.append('device = "mps"')
                elif p == "openvino":
                    skip_checks.append('if not HAS_OPENVINO: self.skipTest("OpenVINO not available")')
                    device_setup.append('device = "cpu"  # OpenVINO uses CPU for PyTorch API')
                elif p == "qualcomm":
                    skip_checks.append('if not HAS_QUALCOMM: self.skipTest("Qualcomm AI Engine not available")')
                    device_setup.append('device = "cpu"  # Qualcomm uses CPU for PyTorch API')
                elif p == "webnn":
                    skip_checks.append('if not HAS_WEBNN: self.skipTest("WebNN not available")')
                    device_setup.append('device = "cpu"  # WebNN uses CPU for PyTorch API')
                elif p == "webgpu":
                    skip_checks.append('if not HAS_WEBGPU: self.skipTest("WebGPU not available")')
                    device_setup.append('device = "cpu"  # WebGPU uses CPU for PyTorch API')
                else:
                    # Default to CPU
                    device_setup.append('device = "cpu"')
                
                # Create skip checks string
                skip_checks_str = "\n        ".join(skip_checks)
                
                # Create device setup string
                device_setup_str = "\n        ".join(device_setup)
                
                # Add the test method
                f.write(f'''
    def test_{p}(self):
        """Test {model} on {p} platform."""
        # Skip if hardware not available
        {skip_checks_str}
        
        # Set up device
        {device_setup_str}
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move model to device if needed
            if device != "cpu":
                self.model = self.model.to(device)
            
            # Test basic functionality
            inputs = self.tokenizer("Hello, world!", return_tensors="pt")
            
            # Move inputs to device if needed
            if device != "cpu":
                inputs = {{k: v.to(device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            
            # Log success
            logger.info(f"Successfully tested {{self.model_name}} on {p}")
            
        except Exception as e:
            logger.error(f"Error testing {{self.model_name}} on {p}: {{str(e)}}")
            raise
''')
            
            # Add main section
            f.write('''
if __name__ == "__main__":
    unittest.main()
''')
    
    logger.info(f"Generated test file: {file_name}")
    return file_name

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced test generator with hardware support")
    parser.add_argument("-g", "--generate", type=str, help="Model to generate test for")
    parser.add_argument("-p", "--platform", type=str, default="all", 
                      help="Platform(s) to test on (comma-separated or 'all')")
    parser.add_argument("-o", "--output", type=str, help="Output file path (default: test_<model>.py)")
    parser.add_argument("-t", "--use-template", action="store_true", 
                      help="Use template from database if available")
    parser.add_argument("-d", "--template-db", type=str, default=DEFAULT_TEMPLATE_DB,
                      help=f"Path to template database (default: {DEFAULT_TEMPLATE_DB})")
    parser.add_argument("--detect-hardware", action="store_true",
                      help="Detect available hardware platforms and exit")
    parser.add_argument("--list-templates", action="store_true",
                      help="List available templates in the database")
    
    args = parser.parse_args()
    
    # Check if we should detect hardware
    if args.detect_hardware:
        detect_hardware()
        return 0
    
    # Check if we should list templates
    if args.list_templates:
        if not HAS_DUCKDB:
            logger.error("DuckDB not available, cannot list templates")
            return 1
        
        if not os.path.exists(args.template_db):
            logger.error(f"Template database not found at {args.template_db}")
            return 1
        
        try:
            conn = duckdb.connect(args.template_db)
            templates = conn.execute("""
                SELECT model_type, template_type, platform, id, created_at
                FROM templates
                ORDER BY model_type, template_type, platform
            """).fetchall()
            
            if templates:
                print(f"Found {len(templates)} templates in {args.template_db}:")
                for t in templates:
                    platform_str = f", platform={t[2]}" if t[2] else ""
                    print(f"  - {t[0]} (type={t[1]}{platform_str}, id={t[3]})")
            else:
                print(f"No templates found in {args.template_db}")
                
            conn.close()
            return 0
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return 1
    
    # Generate test file if model is specified
    if args.generate:
        output_file = generate_test(
            model=args.generate, 
            platform=args.platform, 
            output_file=args.output,
            use_template=args.use_template,
            template_db=args.template_db
        )
        print(f"Generated test file: {output_file}")
        return 0
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())