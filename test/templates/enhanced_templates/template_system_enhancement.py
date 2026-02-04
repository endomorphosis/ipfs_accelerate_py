#!/usr/bin/env python3
"""
Template System Enhancement Script
This script enhances the DuckDB-based template system with improved validation,
better placeholder handling, and template inheritance.

Key features:
1. Template validation system to verify hardware platform support
2. Improved placeholder handling for consistent variable replacement
3. Template inheritance system for better code reuse and structure
"""

import os
import sys
import json
import logging
import argparse
import importlib
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import duckdb
try:
    import duckdb
    DUCKDB_AVAILABLE = True
    logger.info("DuckDB is available, will use database storage")
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.error("DuckDB not available. This script requires DuckDB.")
    sys.exit(1)

# Define common constants
DEFAULT_DB_PATH = "./template_db.duckdb"

# Model type definitions
MODEL_TYPES = [
    "bert", "t5", "llama", "vit", "clip", "whisper", "wav2vec2", 
    "clap", "llava", "xclip", "qwen", "detr", "default"
]

# Hardware platform definitions
HARDWARE_PLATFORMS = [
    "cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "samsung", "webnn", "webgpu"
]

# Template types
TEMPLATE_TYPES = [
    "test", "benchmark", "skill", "helper", "hardware_specific"
]

# Modality types for template categorization
MODALITY_TYPES = {
    "text": ["bert", "t5", "llama", "roberta", "gpt2"],
    "vision": ["vit", "resnet", "detr"],
    "audio": ["whisper", "wav2vec2", "clap"],
    "multimodal": ["clip", "llava", "xclip"]
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhance the template database system with validation, improved placeholder handling, and inheritance"
    )
    parser.add_argument(
        "--db-path", type=str, default=DEFAULT_DB_PATH,
        help=f"Path to template database file (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--check-db", action="store_true",
        help="Check if database exists and has proper schema"
    )
    parser.add_argument(
        "--validate-templates", action="store_true",
        help="Validate all templates in the database for syntax and hardware support"
    )
    parser.add_argument(
        "--validate-model-type", type=str,
        help="Validate templates for a specific model type"
    )
    parser.add_argument(
        "--list-templates", action="store_true",
        help="List all templates in the database with validation status"
    )
    parser.add_argument(
        "--add-inheritance", action="store_true",
        help="Add inheritance system to templates"
    )
    parser.add_argument(
        "--enhance-placeholders", action="store_true",
        help="Enhance placeholder handling in templates"
    )
    parser.add_argument(
        "--apply-all-enhancements", action="store_true",
        help="Apply all enhancements (validation, inheritance, placeholders)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment and configure logging"""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

def check_database(db_path: str) -> bool:
    """Check if database exists and has the correct schema"""
    if not os.path.exists(db_path):
        logger.error(f"Database file {db_path} does not exist")
        return False

    try:
        conn = duckdb.connect(db_path)
        
        # Check if templates table exists
        result = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name = 'templates'
        """).fetchone()
        
        if result[0] == 0:
            logger.error("Templates table not found in database")
            return False
        
        # Check if templates table has the expected columns
        result = conn.execute("""
        PRAGMA table_info(templates)
        """).fetchall()
        
        columns = [row[1] for row in result]
        required_columns = ['model_type', 'template_type', 'template', 'hardware_platform']
        
        for column in required_columns:
            if column not in columns:
                logger.error(f"Required column '{column}' not found in templates table")
                return False
        
        # Check if database has templates
        result = conn.execute("""
        SELECT COUNT(*) FROM templates
        """).fetchone()
        
        template_count = result[0]
        if template_count == 0:
            logger.warning("Database exists but contains no templates")
        else:
            logger.info(f"Database contains {template_count} templates")
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return False

def enhance_schema(db_path: str) -> bool:
    """Enhance the database schema to support template inheritance and validation"""
    try:
        conn = duckdb.connect(db_path)
        
        # Check if validation columns already exist
        result = conn.execute("""
        PRAGMA table_info(templates)
        """).fetchall()
        
        columns = [row[1] for row in result]
        
        # Add validation column if it doesn't exist
        if 'validation_status' not in columns:
            logger.info("Adding validation_status column to templates table")
            conn.execute("""
            ALTER TABLE templates ADD COLUMN validation_status VARCHAR
            """)
        
        # Add parent_template column for inheritance if it doesn't exist
        if 'parent_template' not in columns:
            logger.info("Adding parent_template column to templates table")
            conn.execute("""
            ALTER TABLE templates ADD COLUMN parent_template VARCHAR
            """)
        
        # Add modality column for better categorization if it doesn't exist
        if 'modality' not in columns:
            logger.info("Adding modality column to templates table")
            conn.execute("""
            ALTER TABLE templates ADD COLUMN modality VARCHAR
            """)
        
        # Add last_updated column for tracking changes if it doesn't exist
        if 'last_updated' not in columns:
            logger.info("Adding last_updated column to templates table")
            conn.execute("""
            ALTER TABLE templates ADD COLUMN last_updated TIMESTAMP
            """)
        
        # Create a new template_validation table if it doesn't exist
        conn.execute("""
        CREATE TABLE IF NOT EXISTS template_validation (
            id INTEGER PRIMARY KEY,
            template_id INTEGER,
            validation_date TIMESTAMP,
            validation_type VARCHAR,
            success BOOLEAN,
            errors TEXT,
            hardware_support TEXT
        )
        """)
        
        # Create a template_placeholders table if it doesn't exist
        conn.execute("""
        CREATE TABLE IF NOT EXISTS template_placeholders (
            id INTEGER PRIMARY KEY,
            placeholder VARCHAR,
            description TEXT,
            default_value VARCHAR,
            required BOOLEAN
        )
        """)
        
        conn.close()
        logger.info("Database schema enhanced successfully")
        return True
    except Exception as e:
        logger.error(f"Error enhancing database schema: {e}")
        return False

def extract_placeholders(template: str) -> Set[str]:
    """Extract all placeholders from a template"""
    # Find all patterns like {placeholder_name}
    pattern = r'\{([a-zA-Z0-9_]+)\}'
    placeholders = set(re.findall(pattern, template))
    return placeholders

def validate_template_syntax(template: str) -> Tuple[bool, List[str]]:
    """Validate template syntax (check for balanced braces, valid Python syntax, etc.)"""
    errors = []
    
    # Check for balanced braces in placeholders
    if template.count('{') != template.count('}'):
        errors.append("Unbalanced braces in template")
    
    # Check for Python syntax errors
    try:
        # We need to replace all placeholder patterns with actual values for compilation
        placeholders = extract_placeholders(template)
        test_template = template
        
        for placeholder in placeholders:
            test_template = test_template.replace(f"{{{placeholder}}}", f'"{placeholder}"')
        
        # Try to compile the template as Python code
        compile(test_template, '<template>', 'exec')
    except SyntaxError as e:
        errors.append(f"Python syntax error: {e}")
    
    # Check for common template issues
    if "{{" in template or "}}" in template:
        errors.append("Double braces detected: {{ or }} should be single { or }")
    
    if "\\n" in template and '"""' in template:
        # This could be legitimate in some cases, so just add a warning
        errors.append("Warning: \\n escape sequence found in triple-quoted string")
    
    return len(errors) == 0, errors

def validate_hardware_support(template: str, hardware_platform: str = None) -> Tuple[bool, Dict[str, bool]]:
    """Validate hardware support in a template"""
    # Initialize hardware support status for all platforms
    hardware_support = {platform: False for platform in HARDWARE_PLATFORMS}
    hardware_support['cpu'] = True  # CPU support is assumed for all templates
    
    # Check for hardware-specific imports and configurations
    if "torch.cuda" in template or "device = 'cuda'" in template:
        hardware_support['cuda'] = True
    
    if "rocm" in template or "AMD" in template:
        hardware_support['rocm'] = True
    
    if "mps" in template or "torch.backends.mps" in template:
        hardware_support['mps'] = True
    
    if "openvino" in template or "OpenVINO" in template:
        hardware_support['openvino'] = True
    
    if "qualcomm" in template or "QNN" in template:
        hardware_support['qualcomm'] = True
    
    if "samsung" in template or "Exynos" in template:
        hardware_support['samsung'] = True
    
    if "webnn" in template or "WebNN" in template:
        hardware_support['webnn'] = True
    
    if "webgpu" in template or "WebGPU" in template:
        hardware_support['webgpu'] = True
    
    # If a specific hardware platform is specified, check if it's supported
    if hardware_platform:
        return hardware_support.get(hardware_platform, False), hardware_support
    
    # Otherwise, return overall validation status and hardware support dict
    return True, hardware_support

def validate_template(template: str, template_type: str, model_type: str, hardware_platform: str = None) -> Tuple[bool, Dict[str, Any]]:
    """Validate a template for syntax, hardware support, and mandatory placeholders"""
    validation_results = {
        'syntax': {'success': False, 'errors': []},
        'hardware': {'success': False, 'support': {}},
        'placeholders': {'success': False, 'missing': [], 'all': []}
    }
    
    # Validate syntax
    syntax_valid, syntax_errors = validate_template_syntax(template)
    validation_results['syntax']['success'] = syntax_valid
    validation_results['syntax']['errors'] = syntax_errors
    
    # Validate hardware support
    hardware_valid, hardware_support = validate_hardware_support(template, hardware_platform)
    validation_results['hardware']['success'] = hardware_valid
    validation_results['hardware']['support'] = hardware_support
    
    # Extract and validate placeholders
    placeholders = extract_placeholders(template)
    validation_results['placeholders']['all'] = list(placeholders)
    
    # Check for mandatory placeholders based on template type
    mandatory_placeholders = {'model_name', 'normalized_name', 'generated_at'}
    missing_placeholders = mandatory_placeholders - placeholders
    
    validation_results['placeholders']['success'] = len(missing_placeholders) == 0
    validation_results['placeholders']['missing'] = list(missing_placeholders)
    
    # Determine overall validation status
    validation_success = syntax_valid and hardware_valid and validation_results['placeholders']['success']
    
    return validation_success, validation_results

def validate_all_templates(db_path: str, model_type: str = None) -> bool:
    """Validate all templates in the database or templates for a specific model type"""
    try:
        conn = duckdb.connect(db_path)
        
        # Query templates to validate
        if model_type:
            logger.info(f"Validating templates for model type: {model_type}")
            query = """
            SELECT rowid, model_type, template_type, template, hardware_platform
            FROM templates
            WHERE model_type = ?
            """
            results = conn.execute(query, [model_type]).fetchall()
        else:
            logger.info("Validating all templates")
            query = """
            SELECT rowid, model_type, template_type, template, hardware_platform
            FROM templates
            """
            results = conn.execute(query).fetchall()
        
        if not results:
            logger.warning(f"No templates found to validate")
            return False
        
        # Validate each template
        success_count = 0
        fail_count = 0
        
        for rowid, model_type, template_type, template, hardware_platform in results:
            logger.info(f"Validating template: {model_type}/{template_type}/{hardware_platform or 'generic'}")
            
            # Validate template
            success, validation_results = validate_template(
                template, template_type, model_type, hardware_platform
            )
            
            # Update template with validation status
            if success:
                status = "VALID"
                success_count += 1
            else:
                status = "INVALID"
                fail_count += 1
                
                # Log validation errors
                if not validation_results['syntax']['success']:
                    logger.error(f"Syntax errors: {validation_results['syntax']['errors']}")
                
                if not validation_results['placeholders']['success']:
                    logger.error(f"Missing placeholders: {validation_results['placeholders']['missing']}")
            
            # Update template validation status in database
            conn.execute("""
            UPDATE templates 
            SET validation_status = ?, 
                last_updated = CURRENT_TIMESTAMP
            WHERE rowid = ?
            """, [status, rowid])
            
            # Store detailed validation results in template_validation table
            hardware_support_json = json.dumps(validation_results['hardware']['support'])
            conn.execute("""
            INSERT INTO template_validation
            (template_id, validation_date, validation_type, success, errors, hardware_support)
            VALUES (?, CURRENT_TIMESTAMP, 'full', ?, ?, ?)
            """, [
                rowid, 
                success, 
                json.dumps(validation_results['syntax']['errors']), 
                hardware_support_json
            ])
        
        logger.info(f"Validation complete: {success_count} valid, {fail_count} invalid")
        conn.close()
        return success_count > 0
    except Exception as e:
        logger.error(f"Error validating templates: {e}")
        return False

def list_templates_with_validation(db_path: str) -> bool:
    """List all templates in the database with their validation status"""
    try:
        conn = duckdb.connect(db_path)
        
        # Query templates with validation status
        query = """
        SELECT t.model_type, t.template_type, t.hardware_platform, 
               t.validation_status, t.modality,
               v.validation_date, v.success as latest_validation,
               v.hardware_support
        FROM templates t
        LEFT JOIN (
            SELECT template_id, MAX(validation_date) as validation_date
            FROM template_validation
            GROUP BY template_id
        ) latest ON t.rowid = latest.template_id
        LEFT JOIN template_validation v ON latest.template_id = v.template_id 
            AND latest.validation_date = v.validation_date
        ORDER BY t.model_type, t.template_type, t.hardware_platform
        """
        
        results = conn.execute(query).fetchall()
        
        if not results:
            logger.warning("No templates found in database")
            return False
        
        # Display template information
        print("\nTemplates with Validation Status:")
        print("-" * 100)
        print(f"{'Model Type':<15} {'Template Type':<15} {'Hardware':<10} {'Status':<10} {'Modality':<12} {'Latest Validation':<20} {'Hardware Support'}")
        print("-" * 100)
        
        for row in results:
            model_type, template_type, hardware, status, modality, latest_validation, latest_success, hardware_support = row
            
            # Format hardware platform display
            hardware = hardware or "generic"
            
            # Format status display
            status = status or "UNKNOWN"
            
            # Format modality display
            modality = modality or "unknown"
            
            # Format latest validation display
            validation_date = latest_validation or "Never"
            if latest_success is not None:
                validation_status = "✅ PASS" if latest_success else "❌ FAIL"
            else:
                validation_status = "⚠️ NONE"
            
            # Format hardware support display
            if hardware_support:
                hardware_info = json.loads(hardware_support)
                supported_hw = [hw for hw, supported in hardware_info.items() if supported]
                hw_display = ", ".join(supported_hw)
            else:
                hw_display = "Unknown"
            
            print(f"{model_type:<15} {template_type:<15} {hardware:<10} {status:<10} {modality:<12} {validation_date} {validation_status:<10} {hw_display}")
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        return False

def add_template_inheritance(db_path: str) -> bool:
    """Add inheritance system to templates"""
    try:
        conn = duckdb.connect(db_path)
        
        # Step 1: Define parent-child relationships for model types
        model_inheritance = {
            # Text models inherit from default text template
            "bert": {"parent": "default_text"},
            "t5": {"parent": "default_text"},
            "llama": {"parent": "default_text"},
            "gpt2": {"parent": "default_text"},
            
            # Vision models inherit from default vision template
            "vit": {"parent": "default_vision"},
            "resnet": {"parent": "default_vision"},
            "detr": {"parent": "default_vision"},
            
            # Audio models inherit from default audio template
            "whisper": {"parent": "default_audio"},
            "wav2vec2": {"parent": "default_audio"},
            "clap": {"parent": "default_audio"},
            
            # Multimodal models inherit from default multimodal template
            "clip": {"parent": "default_multimodal"},
            "llava": {"parent": "default_multimodal"},
            "xclip": {"parent": "default_multimodal"}
        }
        
        # Step 2: Define default templates for each modality if they don't exist
        default_templates = {
            "default_text": {
                "test": """#!/usr/bin/env python3
\"\"\"
Text model test for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

import os
import unittest
import logging
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    \"\"\"Test {model_name} with resource pool integration.\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up test environment.\"\"\"
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Set up device for hardware acceleration if available
        cls.device = "cpu"
        if {has_cuda} and cls.torch.cuda.is_available():
            cls.device = "cuda"
        elif {has_mps} and hasattr(cls.torch, "mps") and cls.torch.backends.mps.is_available():
            cls.device = "mps"
        logger.info(f"Using device: {cls.device}")
        
        # Load model and tokenizer
        try:
            cls.tokenizer = cls.transformers.AutoTokenizer.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            if cls.device != "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        \"\"\"Test that model loaded successfully.\"\"\"
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)
    
    def test_inference(self):
        \"\"\"Test basic inference.\"\"\"
        # Prepare input
        text = "This is a test sentence for a text model."
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move inputs to device if needed:
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertIn("last_hidden_state", outputs)
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
"""
            },
            "default_vision": {
                "test": """#!/usr/bin/env python3
\"\"\"
Vision model test for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

import os
import unittest
import logging
import numpy as np
from PIL import Image
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    \"\"\"Test {model_name} with resource pool integration.\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up test environment.\"\"\"
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Set up device for hardware acceleration if available
        cls.device = "cpu"
        if {has_cuda} and cls.torch.cuda.is_available():
            cls.device = "cuda"
        elif {has_mps} and hasattr(cls.torch, "mps") and cls.torch.backends.mps.is_available():
            cls.device = "mps"
        logger.info(f"Using device: {cls.device}")
        
        # Create a test image if it doesn't exist
        cls.test_image_path = "test.jpg"
        if not os.path.exists(cls.test_image_path):
            # Create a simple test image (100x100 black square)
            img = Image.new('RGB', (100, 100), color='black')
            img.save(cls.test_image_path)
            logger.info(f"Created test image at {cls.test_image_path}")
        
        # Load model and feature extractor/processor
        try:
            cls.processor = cls.transformers.AutoFeatureExtractor.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            if cls.device != "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        \"\"\"Test that model loaded successfully.\"\"\"
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.processor)
    
    def test_inference(self):
        \"\"\"Test basic inference.\"\"\"
        # Load and process image
        image = Image.open(self.test_image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move inputs to device if needed:
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertIn("last_hidden_state", outputs)
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
"""
            },
            "default_audio": {
                "test": """#!/usr/bin/env python3
\"\"\"
Audio model test for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

import os
import unittest
import logging
import numpy as np
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    \"\"\"Test {model_name} with resource pool integration.\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up test environment.\"\"\"
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Set up device for hardware acceleration if available
        cls.device = "cpu"
        if {has_cuda} and cls.torch.cuda.is_available():
            cls.device = "cuda"
        elif {has_mps} and hasattr(cls.torch, "mps") and cls.torch.backends.mps.is_available():
            cls.device = "mps"
        logger.info(f"Using device: {cls.device}")
        
        # Create a test audio array or use existing file
        cls.test_audio_path = "test.mp3"
        cls.sampling_rate = 16000
        
        if not os.path.exists(cls.test_audio_path):
            # Create a simple silence audio array (1 second)
            logger.info(f"No test audio found, using synthetic array")
            cls.audio_array = np.zeros(cls.sampling_rate)  # 1 second of silence
        else:
            try:
                # Try to load audio file if available
                import librosa
                cls.audio_array, cls.sampling_rate = librosa.load(cls.test_audio_path, sr=cls.sampling_rate)
                logger.info(f"Loaded test audio from {cls.test_audio_path}")
            except (ImportError, Exception) as e:
                logger.warning(f"Could not load audio file: {e}")
                cls.audio_array = np.zeros(cls.sampling_rate)  # 1 second of silence
        
        # Load model and processor
        try:
            cls.processor = cls.transformers.AutoProcessor.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            if cls.device != "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        \"\"\"Test that model loaded successfully.\"\"\"
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.processor)
    
    def test_inference(self):
        \"\"\"Test basic inference.\"\"\"
        # Process audio input
        inputs = self.processor(
            self.audio_array, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt"
        )
        
        # Move inputs to device if needed:
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
"""
            },
            "default_multimodal": {
                "test": """#!/usr/bin/env python3
\"\"\"
Multimodal model test for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

import os
import unittest
import logging
import numpy as np
from PIL import Image
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    \"\"\"Test {model_name} with resource pool integration.\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up test environment.\"\"\"
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Set up device for hardware acceleration if available
        cls.device = "cpu"
        if {has_cuda} and cls.torch.cuda.is_available():
            cls.device = "cuda"
        elif {has_mps} and hasattr(cls.torch, "mps") and cls.torch.backends.mps.is_available():
            cls.device = "mps"
        logger.info(f"Using device: {cls.device}")
        
        # Create a test image if it doesn't exist
        cls.test_image_path = "test.jpg"
        if not os.path.exists(cls.test_image_path):
            # Create a simple test image (100x100 black square)
            img = Image.new('RGB', (100, 100), color='black')
            img.save(cls.test_image_path)
            logger.info(f"Created test image at {cls.test_image_path}")
        
        # Test text prompt
        cls.test_text = "What's in this image?"
        
        # Load model and processor
        try:
            cls.processor = cls.transformers.AutoProcessor.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            if cls.device != "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        \"\"\"Test that model loaded successfully.\"\"\"
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.processor)
    
    def test_inference(self):
        \"\"\"Test basic inference.\"\"\"
        # Load image
        image = Image.open(self.test_image_path)
        
        # Process inputs
        inputs = self.processor(
            text=self.test_text,
            images=image, 
            return_tensors="pt"
        )
        
        # Move inputs to device if needed:
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
"""
            }
        }
        
        # Step 3: Add default templates to database if they don't exist
        for parent_name, templates in default_templates.items():
            for template_type, template_content in templates.items():
                # Check if parent template exists
                result = conn.execute("""
                SELECT COUNT(*) FROM templates
                WHERE model_type = ? AND template_type = ?
                """, [parent_name, template_type]).fetchone()
                
                if result[0] == 0:
                    logger.info(f"Adding parent template {parent_name}/{template_type}")
                    
                    # Determine modality
                    if parent_name == "default_text":
                        modality = "text"
                    elif parent_name == "default_vision":
                        modality = "vision"
                    elif parent_name == "default_audio":
                        modality = "audio"
                    elif parent_name == "default_multimodal":
                        modality = "multimodal"
                    else:
                        modality = None
                    
                    # Insert parent template
                    conn.execute("""
                    INSERT INTO templates
                    (model_type, template_type, template, hardware_platform, validation_status, modality, last_updated)
                    VALUES (?, ?, ?, NULL, 'VALID', ?, CURRENT_TIMESTAMP)
                    """, [parent_name, template_type, template_content, modality])
        
        # Step 4: Update existing templates with parent information
        for model_type, inheritance_info in model_inheritance.items():
            parent_type = inheritance_info["parent"]
            
            # Determine modality
            if parent_type == "default_text":
                modality = "text"
            elif parent_type == "default_vision":
                modality = "vision"
            elif parent_type == "default_audio":
                modality = "audio"
            elif parent_type == "default_multimodal":
                modality = "multimodal"
            else:
                modality = None
            
            # Get templates for this model type
            results = conn.execute("""
            SELECT rowid, model_type, template_type, hardware_platform
            FROM templates
            WHERE model_type = ?
            """, [model_type]).fetchall()
            
            for rowid, model_type, template_type, hardware_platform in results:
                # Set parent_template and modality
                logger.info(f"Updating template {model_type}/{template_type}/{hardware_platform or 'generic'} with parent {parent_type}")
                conn.execute("""
                UPDATE templates
                SET parent_template = ?, modality = ?, last_updated = CURRENT_TIMESTAMP
                WHERE rowid = ?
                """, [parent_type, modality, rowid])
        
        conn.close()
        logger.info("Template inheritance system added successfully")
        return True
    except Exception as e:
        logger.error(f"Error adding template inheritance: {e}")
        return False

def enhance_placeholders(db_path: str) -> bool:
    """Enhance placeholder handling in templates"""
    try:
        conn = duckdb.connect(db_path)
        
        # Step 1: Define standard placeholders and their properties
        standard_placeholders = {
            # Core placeholders
            "model_name": {"description": "Full model name", "default_value": None, "required": True},
            "normalized_name": {"description": "Normalized model name for class names", "default_value": None, "required": True},
            "generated_at": {"description": "Generation timestamp", "default_value": None, "required": True},
            
            # Hardware-related placeholders
            "best_hardware": {"description": "Best available hardware for the model", "default_value": "cpu", "required": False},
            "torch_device": {"description": "PyTorch device to use", "default_value": "cpu", "required": False},
            "has_cuda": {"description": "Boolean indicating CUDA availability", "default_value": "False", "required": False},
            "has_rocm": {"description": "Boolean indicating ROCm availability", "default_value": "False", "required": False},
            "has_mps": {"description": "Boolean indicating MPS availability", "default_value": "False", "required": False},
            "has_openvino": {"description": "Boolean indicating OpenVINO availability", "default_value": "False", "required": False},
            "has_webnn": {"description": "Boolean indicating WebNN availability", "default_value": "False", "required": False},
            "has_webgpu": {"description": "Boolean indicating WebGPU availability", "default_value": "False", "required": False},
            
            # Model-related placeholders
            "model_family": {"description": "Model family classification", "default_value": "default", "required": False},
            "model_subfamily": {"description": "Model subfamily classification", "default_value": None, "required": False},
        }
        
        # Step 2: Clear existing placeholders and add standard ones
        conn.execute("DELETE FROM template_placeholders")
        
        for placeholder_name, properties in standard_placeholders.items():
            conn.execute("""
            INSERT INTO template_placeholders
            (placeholder, description, default_value, required)
            VALUES (?, ?, ?, ?)
            """, [
                placeholder_name,
                properties["description"],
                properties["default_value"],
                properties["required"]
            ])
        
        # Step 3: Extract additional placeholders from existing templates
        query = """
        SELECT template FROM templates
        """
        templates = conn.execute(query).fetchall()
        
        additional_placeholders = set()
        for template, in templates:
            placeholders = extract_placeholders(template)
            additional_placeholders.update(placeholders)
        
        # Step 4: Add any additional placeholders found
        for placeholder in additional_placeholders:
            if placeholder not in standard_placeholders:
                conn.execute("""
                INSERT INTO template_placeholders
                (placeholder, description, default_value, required)
                VALUES (?, ?, NULL, FALSE)
                """, [placeholder, f"Auto-detected placeholder: {placeholder}"])
        
        # Step 5: Create helper functions for placeholder documentation (utilities for test/benchmark generators)
        
        # First, check if the utilities directory exists, create if not
        utilities_dir = os.path.join(os.path.dirname(db_path), "template_utilities")
        os.makedirs(utilities_dir, exist_ok=True)
        
        # Create a placeholder helper module
        helper_path = os.path.join(utilities_dir, "placeholder_helpers.py")
        with open(helper_path, "w") as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Placeholder helper functions for template rendering.
This module provides utilities for working with template placeholders.
\"\"\"

import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def get_standard_placeholders() -> Dict[str, Dict[str, Any]]:
    \"\"\"Get standard placeholders and their properties\"\"\"
    # Standard placeholders used across all templates
    return {
        # Core placeholders
        "model_name": {"description": "Full model name", "default_value": None, "required": True},
        "normalized_name": {"description": "Normalized model name for class names", "default_value": None, "required": True},
        "generated_at": {"description": "Generation timestamp", "default_value": None, "required": True},
        
        # Hardware-related placeholders
        "best_hardware": {"description": "Best available hardware for the model", "default_value": "cpu", "required": False},
        "torch_device": {"description": "PyTorch device to use", "default_value": "cpu", "required": False},
        "has_cuda": {"description": "Boolean indicating CUDA availability", "default_value": "False", "required": False},
        "has_rocm": {"description": "Boolean indicating ROCm availability", "default_value": "False", "required": False},
        "has_mps": {"description": "Boolean indicating MPS availability", "default_value": "False", "required": False},
        "has_openvino": {"description": "Boolean indicating OpenVINO availability", "default_value": "False", "required": False},
        "has_webnn": {"description": "Boolean indicating WebNN availability", "default_value": "False", "required": False},
        "has_webgpu": {"description": "Boolean indicating WebGPU availability", "default_value": "False", "required": False},
        
        # Model-related placeholders
        "model_family": {"description": "Model family classification", "default_value": "default", "required": False},
        "model_subfamily": {"description": "Model subfamily classification", "default_value": None, "required": False},
    }

def detect_missing_placeholders(template: str, context: Dict[str, Any]) -> List[str]:
    \"\"\"Detect missing placeholders in a template\"\"\"
    # Find all patterns like {placeholder_name}
    import re
    pattern = r'\{([a-zA-Z0-9_]+)\}'
    placeholders = set(re.findall(pattern, template))
    
    # Find placeholders that are not in context
    missing = [p for p in placeholders if p not in context]
    return missing

def get_default_context(model_name: str) -> Dict[str, Any]:
    \"\"\"Get default context for template rendering\"\"\"
    import datetime
    import re
    
    # Normalize model name for class names
    normalized_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name).title()
    
    # Hardware detection
    import torch
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch, 'mps') and torch.backends.mps.is_available()
    
    # Default context
    context = {
        "model_name": model_name,
        "normalized_name": normalized_name,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "torch_device": "cuda" if has_cuda else "mps" if has_mps else "cpu",
        "has_cuda": str(has_cuda),
        "has_mps": str(has_mps),
        "has_rocm": "False",
        "has_openvino": "False",
        "has_webnn": "False",
        "has_webgpu": "False",
    }
    
    return context

def render_template(template: str, context: Dict[str, Any]) -> str:
    \"\"\"Render a template with placeholder substitution\"\"\"
    # Ensure all required placeholders are present
    missing = detect_missing_placeholders(template, context)
    
    if missing:
        # Try to fill in defaults
        standard_placeholders = get_standard_placeholders()
        for placeholder in missing:
            if placeholder in standard_placeholders and standard_placeholders[placeholder]["default_value"] is not None:
                context[placeholder] = standard_placeholders[placeholder]["default_value"]
        
        # Check again after filling defaults
        missing = detect_missing_placeholders(template, context)
        
        if missing:
            logger.warning(f"Missing placeholders: {missing}")
            # For missing placeholders, use a placeholder name
            for placeholder in missing:
                context[placeholder] = f"<<MISSING:{placeholder}>>"
    
    # Render template
    result = template.format(**context)
    return result
""")
        
        logger.info(f"Created placeholder helper module at {helper_path}")
        
        # Create an initialization file for the utilities directory
        init_path = os.path.join(utilities_dir, "__init__.py")
        with open(init_path, "w") as f:
            f.write("""\"\"\"Template utilities package\"\"\"

from test.templates.enhanced_templates.placeholder_helpers import (
    get_standard_placeholders,
    detect_missing_placeholders,
    get_default_context,
    render_template
)

__all__ = [
    'get_standard_placeholders',
    'detect_missing_placeholders',
    'get_default_context',
    'render_template'
]
""")
        
        logger.info(f"Created utilities package initialization file at {init_path}")
        
        conn.close()
        logger.info("Placeholder system enhanced successfully")
        return True
    except Exception as e:
        logger.error(f"Error enhancing placeholders: {e}")
        return False

def apply_all_enhancements(db_path: str) -> bool:
    """Apply all template system enhancements"""
    logger.info("Applying all template system enhancements")
    
    # Step 1: Check if database exists and has proper schema
    if not check_database(db_path):
        logger.error("Database check failed")
        return False
    
    # Step 2: Enhance database schema
    if not enhance_schema(db_path):
        logger.error("Schema enhancement failed")
        return False
    
    # Step 3: Validate all templates
    if not validate_all_templates(db_path):
        logger.warning("Template validation found issues (continuing with other enhancements)")
    
    # Step 4: Add template inheritance
    if not add_template_inheritance(db_path):
        logger.error("Template inheritance enhancement failed")
        return False
    
    # Step 5: Enhance placeholders
    if not enhance_placeholders(db_path):
        logger.error("Placeholder enhancement failed")
        return False
    
    # Step 6: List templates with validation status
    list_templates_with_validation(db_path)
    
    logger.info("All template system enhancements applied successfully")
    return True

def main():
    """Main function"""
    args = parse_args()
    setup_environment(args)
    
    # Apply operations based on command-line arguments
    if args.check_db:
        check_database(args.db_path)
    
    if args.validate_templates:
        validate_all_templates(args.db_path)
    
    if args.validate_model_type:
        validate_all_templates(args.db_path, args.validate_model_type)
    
    if args.list_templates:
        list_templates_with_validation(args.db_path)
    
    if args.add_inheritance:
        add_template_inheritance(args.db_path)
    
    if args.enhance_placeholders:
        enhance_placeholders(args.db_path)
    
    if args.apply_all_enhancements:
        apply_all_enhancements(args.db_path)
    
    # If no specific operation was specified, show usage
    if not any([
        args.check_db, args.validate_templates, args.validate_model_type,
        args.list_templates, args.add_inheritance, args.enhance_placeholders,
        args.apply_all_enhancements
    ]):
        logger.error("No operation specified")
        logger.info("Use --help to see available operations")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())