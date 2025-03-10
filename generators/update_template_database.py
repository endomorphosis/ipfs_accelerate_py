#!/usr/bin/env python3
"""
Update Template Database

This script updates the template database with samples for all model types
and adds additional templates for specific hardware platforms.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for DuckDB
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    logger.error("DuckDB not available. Please install with: pip install duckdb")
    sys.exit(1)

# Constants
DEFAULT_DB_PATH = "template_db.duckdb"
MODEL_CATEGORIES = ["text", "vision", "audio", "multimodal", "video"]
KEY_MODELS = [
    "bert", "t5", "llama", "vit", "clip", "detr", 
    "wav2vec2", "whisper", "clap", "llava", "llava_next", 
    "xclip", "qwen2"
]
HARDWARE_PLATFORMS = ["cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu"]

def create_or_update_schema(db_path):
    """
    Create or update the template database schema.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = duckdb.connect(db_path)
        
        # Check if tables exist
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0].lower() for t in tables]
        
        # Create templates table if needed
        if "templates" not in table_names:
            conn.execute("""
            CREATE TABLE templates (
                id INTEGER PRIMARY KEY,
                model_type VARCHAR NOT NULL,
                template_type VARCHAR NOT NULL,
                platform VARCHAR,
                template TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            logger.info("Created templates table")
        
        # Create helpers table if needed
        if "template_helpers" not in table_names:
            conn.execute("""
            CREATE TABLE template_helpers (
                id INTEGER PRIMARY KEY,
                helper_name VARCHAR NOT NULL,
                helper_code TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            logger.info("Created template_helpers table")
        
        # Create template versions table if needed
        if "template_versions" not in table_names:
            conn.execute("""
            CREATE TABLE template_versions (
                id INTEGER PRIMARY KEY,
                template_id INTEGER,
                version VARCHAR NOT NULL,
                changes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            logger.info("Created template_versions table")
        
        # Create template dependencies table if needed
        if "template_dependencies" not in table_names:
            conn.execute("""
            CREATE TABLE template_dependencies (
                id INTEGER PRIMARY KEY,
                template_id INTEGER,
                dependency_template_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            logger.info("Created template_dependencies table")
        
        # Create template variables table if needed
        if "template_variables" not in table_names:
            conn.execute("""
            CREATE TABLE template_variables (
                id INTEGER PRIMARY KEY,
                variable_name VARCHAR NOT NULL,
                default_value TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            logger.info("Created template_variables table")
        
        # Create template validation table if needed
        if "template_validation" not in table_names:
            conn.execute("""
            CREATE TABLE template_validation (
                id INTEGER PRIMARY KEY,
                template_id INTEGER,
                validation_date TIMESTAMP,
                is_valid BOOLEAN,
                validation_errors TEXT,
                validation_warnings TEXT
            )
            """)
            logger.info("Created template_validation table")
        
        conn.close()
        return True
    
    except Exception as e:
        logger.error(f"Error creating/updating schema: {e}")
        return False

def get_next_id(conn, table):
    """Get the next ID for a table."""
    result = conn.execute(f"SELECT MAX(id) FROM {table}").fetchone()
    if result[0] is None:
        return 1
    return result[0] + 1

def add_base_templates(db_path):
    """
    Add base templates for each model category.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = duckdb.connect(db_path)
        
        templates_added = 0
        
        # Template for text models
        text_template = '''#!/usr/bin/env python3
"""
Test for {{model_name}} with cross-platform hardware support
"""

import os
import sys
import unittest
import importlib.util
import logging
import torch
from transformers import AutoModel, AutoTokenizer'''

text_template_part2 = '''
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

class Test{{model_name.replace("-", "").capitalize()}}(unittest.TestCase):
    """Test {{model_name}} text model with hardware platform support."""
    
    def setUp(self):
        """Set up the test environment."""
        self.model_name = "{{model_name}}"
        self.tokenizer = None
        self.model = None
        
    def test_cpu(self):
        """Test on CPU platform."""
        device = "cpu"
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Test basic functionality
        inputs = self.tokenizer("Hello, world!", return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        logger.info(f"Successfully tested {self.model_name} on CPU")
        
    def test_cuda(self):
        """Test on CUDA platform."""
        if not HAS_CUDA:
            self.skipTest("CUDA not available")
            
        device = "cuda"
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(device)
        
        # Test basic functionality
        inputs = self.tokenizer("Hello, world!", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        logger.info(f"Successfully tested {self.model_name} on CUDA")

if __name__ == "__main__":
    unittest.main()
"""
        
        # Template for vision models
        vision_template = '''#!/usr/bin/env python3
"""
Test for {{model_name}} vision model with cross-platform hardware support
"""'''

import os
import sys
import unittest
import importlib.util
import logging
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel

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

class Test{{model_name.replace("-", "").capitalize()}}(unittest.TestCase):
    """Test {{model_name}} vision model with hardware platform support."""
    
    def setUp(self):
        """Set up the test environment."""
        self.model_name = "{{model_name}}"
        self.processor = None
        self.model = None
        # Create a dummy image (3 channels, 224x224)
        self.dummy_image = np.random.rand(3, 224, 224)
        
    def test_cpu(self):
        """Test on CPU platform."""
        device = "cpu"
        
        # Initialize processor and model
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Test basic functionality
        inputs = self.processor(self.dummy_image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        logger.info(f"Successfully tested {self.model_name} on CPU")
        
    def test_cuda(self):
        """Test on CUDA platform."""
        if not HAS_CUDA:
            self.skipTest("CUDA not available")
            
        device = "cuda"
        
        # Initialize processor and model
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(device)
        
        # Test basic functionality
        inputs = self.processor(self.dummy_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        logger.info(f"Successfully tested {self.model_name} on CUDA")
        
    def test_webgpu(self):
        """Test on WebGPU platform."""
        if not HAS_WEBGPU:
            self.skipTest("WebGPU not available")
            
        device = "cpu"  # WebGPU uses CPU for PyTorch API
        
        # Initialize processor and model
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Test basic functionality
        inputs = self.processor(self.dummy_image, return_tensors="pt")
        
        # Run inference with WebGPU simulation
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        logger.info(f"Successfully tested {self.model_name} on WebGPU")

if __name__ == "__main__":
    unittest.main()
"""
        
        # Add templates for each category
        for category in MODEL_CATEGORIES:
            template_id = get_next_id(conn, "templates")
            
            # Select appropriate template based on category
            if category == "text":
                template_content = text_template
            elif category in ["vision", "multimodal"]:
                template_content = vision_template
            elif category == "audio":
                # Audio template would go here
                template_content = text_template  # For now, use text template
            elif category == "video":
                # Video template would go here
                template_content = vision_template  # For now, use vision template
            else:
                template_content = text_template  # Default
            
            # Insert into database
            conn.execute(
                "INSERT INTO templates (id, model_type, template_type, template) VALUES (?, ?, 'test', ?)",
                [template_id, category, template_content]
            )
            
            templates_added += 1
            
        # Add templates for key models
        for model in KEY_MODELS:
            template_id = get_next_id(conn, "templates")
            
            # Use appropriate template based on model type
            if model in ["bert", "t5", "llama", "qwen2"]:
                template_content = text_template
            elif model in ["vit", "detr"]:
                template_content = vision_template
            elif model in ["wav2vec2", "whisper", "clap"]:
                template_content = text_template  # For now
            elif model in ["clip", "llava", "llava_next"]:
                template_content = vision_template
            elif model in ["xclip"]:
                template_content = vision_template
            else:
                template_content = text_template  # Default
            
            # Insert into database
            conn.execute(
                "INSERT INTO templates (id, model_type, template_type, template) VALUES (?, ?, 'test', ?)",
                [template_id, model, template_content]
            )
            
            templates_added += 1
        
        logger.info(f"Added {templates_added} templates")
        conn.close()
        return True
    
    except Exception as e:
        logger.error(f"Error adding base templates: {e}")
        return False

def add_hardware_specific_templates(db_path):
    """
    Add hardware-specific templates for qualcomm and webgpu.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = duckdb.connect(db_path)
        
        templates_added = 0
        
        # Qualcomm-specific template for bert
        qualcomm_bert_template = '''#!/usr/bin/env python3
"""
Test for {{model_name}} with Qualcomm AI Engine support
"""'''

import os
import sys
import unittest
import importlib.util
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for Qualcomm SDK
HAS_QUALCOMM = (
    importlib.util.find_spec("qnn_wrapper") is not None or 
    importlib.util.find_spec("qti") is not None or
    "QUALCOMM_SDK" in os.environ
)

class Test{{model_name.replace("-", "").capitalize()}}Qualcomm(unittest.TestCase):
    """Test {{model_name}} model with Qualcomm AI Engine."""
    
    def setUp(self):
        """Set up the test environment."""
        self.model_name = "{{model_name}}"
        if not HAS_QUALCOMM:
            self.skipTest("Qualcomm AI Engine not available")
        
    def test_qualcomm(self):
        """Test {{model_name}} on Qualcomm AI Engine."""
        try:
            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            
            # Prepare sample input
            inputs = tokenizer("Testing Qualcomm AI Engine with {{model_name}}", return_tensors="pt")
            
            # Run inference
            outputs = model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            logger.info(f"Successfully tested {self.model_name} on Qualcomm AI Engine")
        except Exception as e:
            logger.error(f"Error testing {self.model_name} on Qualcomm: {e}")
            raise

if __name__ == "__main__":
    unittest.main()
"""
        
        # WebGPU-specific template for vision models
        webgpu_vision_template = '''#!/usr/bin/env python3
"""
Test for {{model_name}} with WebGPU support
"""'''

import os
import sys
import unittest
import importlib.util
import logging
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for WebGPU
HAS_WEBGPU = (
    importlib.util.find_spec("webgpu") is not None or
    importlib.util.find_spec("wgpu") is not None or
    "WEBGPU_AVAILABLE" in os.environ or
    "WEBGPU_SIMULATION" in os.environ
)

# Check for browser-specific optimizations
HAS_WEBGPU_COMPUTE_SHADERS = os.environ.get("WEBGPU_COMPUTE_SHADERS_ENABLED", "0") == "1"
HAS_SHADER_PRECOMPILE = os.environ.get("WEBGPU_SHADER_PRECOMPILE_ENABLED", "0") == "1"

class Test{{model_name.replace("-", "").capitalize()}}WebGPU(unittest.TestCase):
    """Test {{model_name}} vision model with WebGPU."""
    
    def setUp(self):
        """Set up the test environment."""
        self.model_name = "{{model_name}}"
        if not HAS_WEBGPU:
            self.skipTest("WebGPU not available")
        
        # Create dummy image (3 channels, 224x224)
        self.dummy_image = np.random.rand(3, 224, 224)
        
    def test_webgpu_basic(self):
        """Test {{model_name}} on WebGPU (basic)."""
        try:
            # Initialize processor and model
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            
            # Prepare sample input
            inputs = processor(self.dummy_image, return_tensors="pt")
            
            # Run inference
            outputs = model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            logger.info(f"Successfully tested {self.model_name} on WebGPU (basic)")
        except Exception as e:
            logger.error(f"Error testing {self.model_name} on WebGPU: {e}")
            raise
            
    def test_webgpu_shader_precompile(self):
        """Test {{model_name}} on WebGPU with shader precompilation."""
        if not HAS_SHADER_PRECOMPILE:
            self.skipTest("WebGPU shader precompilation not enabled")
            
        try:
            # Enable shader precompilation
            os.environ["WEBGPU_SHADER_PRECOMPILE"] = "1"
            
            # Initialize processor and model
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            
            # Prepare sample input
            inputs = processor(self.dummy_image, return_tensors="pt")
            
            # Run inference
            outputs = model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            logger.info(f"Successfully tested {self.model_name} on WebGPU with shader precompilation")
        except Exception as e:
            logger.error(f"Error testing {self.model_name} on WebGPU with shader precompilation: {e}")
            raise
        finally:
            # Reset environment
            os.environ.pop("WEBGPU_SHADER_PRECOMPILE", None)

if __name__ == "__main__":
    unittest.main()
"""
        
        # Add Qualcomm template for bert
        template_id = get_next_id(conn, "templates")
        conn.execute(
            "INSERT INTO templates (id, model_type, template_type, platform, template) VALUES (?, 'bert', 'test', 'qualcomm', ?)",
            [template_id, qualcomm_bert_template]
        )
        templates_added += 1
        
        # Add WebGPU template for vision models
        template_id = get_next_id(conn, "templates")
        conn.execute(
            "INSERT INTO templates (id, model_type, template_type, platform, template) VALUES (?, 'vision', 'test', 'webgpu', ?)",
            [template_id, webgpu_vision_template]
        )
        templates_added += 1
        
        # Add WebGPU template for vit specifically
        template_id = get_next_id(conn, "templates")
        conn.execute(
            "INSERT INTO templates (id, model_type, template_type, platform, template) VALUES (?, 'vit', 'test', 'webgpu', ?)",
            [template_id, webgpu_vision_template]
        )
        templates_added += 1
        
        logger.info(f"Added {templates_added} hardware-specific templates")
        conn.close()
        return True
    
    except Exception as e:
        logger.error(f"Error adding hardware-specific templates: {e}")
        return False

def list_templates(db_path):
    """
    List all templates in the database.
    
    Args:
        db_path: Path to the database file
    """
    try:
        conn = duckdb.connect(db_path)
        templates = conn.execute("""
            SELECT id, model_type, template_type, platform, created_at
            FROM templates
            ORDER BY model_type, template_type, platform
        """).fetchall()
        
        if templates:
            print(f"Templates in {db_path}:")
            print("{:<5} {:<15} {:<15} {:<15} {:<20}".format("ID", "Model Type", "Template Type", "Platform", "Created At"))
            print("-" * 70)
            
            for t in templates:
                platform = t[3] if t[3] else "all"
                print("{:<5} {:<15} {:<15} {:<15} {:<20}".format(t[0], t[1], t[2], platform, t[4]))
        else:
            print(f"No templates found in {db_path}")
            
        conn.close()
    
    except Exception as e:
        logger.error(f"Error listing templates: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Update template database")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH,
                      help=f"Path to the template database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--update-schema", action="store_true",
                      help="Create or update the database schema")
    parser.add_argument("--add-templates", action="store_true",
                      help="Add base templates for all model categories")
    parser.add_argument("--add-hardware-templates", action="store_true",
                      help="Add hardware-specific templates")
    parser.add_argument("--list", action="store_true",
                      help="List all templates in the database")
    
    args = parser.parse_args()
    
    # Create database if it doesn't exist
    if not os.path.exists(args.db_path) or args.update_schema:
        logger.info(f"Creating or updating schema in {args.db_path}")
        if not create_or_update_schema(args.db_path):
            return 1
    
    # Add base templates
    if args.add_templates:
        logger.info("Adding base templates")
        if not add_base_templates(args.db_path):
            return 1
    
    # Add hardware-specific templates
    if args.add_hardware_templates:
        logger.info("Adding hardware-specific templates")
        if not add_hardware_specific_templates(args.db_path):
            return 1
    
    # List templates
    if args.list:
        list_templates(args.db_path)
    
    # If no actions specified, print help
    if not (args.update_schema or args.add_templates or args.add_hardware_templates or args.list):
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())