#!/usr/bin/env python3
"""
Template Database Validator for IPFS Accelerate

This script helps validate and manage templates stored in DuckDB database.
It provides functionality to:
1. Validate syntax and structure of templates
2. Verify hardware compatibility across templates
3. Check for template inheritance and dependencies
4. Migrate templates from JSON files to DuckDB database

Usage:
    python create_template_db_validator.py --validate-db [db_path]
    python create_template_db_validator.py --migrate-templates [source_dir] [db_path]
    python create_template_db_validator.py --check-hardware --report [output_file]
"""

import os
import sys
import argparse
import re
import ast
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Set

# Check for DuckDB availability
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

# Set up logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define hardware platforms to check for
HARDWARE_PLATFORMS = [
    ('cuda', r'(cuda|gpu)'),
    ('cpu', r'cpu'),
    ('mps', r'(mps|apple|m1|m2)'),
    ('rocm', r'(rocm|amd)'),
    ('openvino', r'(openvino|intel)'),
    ('qualcomm', r'(qualcomm|qnn|hexagon)'),
    ('webnn', r'webnn'),
    ('webgpu', r'webgpu')
]

# Hardware detection patterns
HARDWARE_CHECKS = {
    'cuda': [
        r'HAS_CUDA', 
        r'torch\.cuda\.is_available', 
        r'device\s*=\s*[\'"]cuda[\'"]'
    ],
    'cpu': [
        r'device\s*=\s*[\'"]cpu[\'"]'
    ],
    'mps': [
        r'HAS_MPS', 
        r'torch\.mps\.is_available', 
        r'device\s*=\s*[\'"]mps[\'"]'
    ],
    'rocm': [
        r'HAS_ROCM', 
        r'_rocm_version', 
        r'ROCM_HOME'
    ],
    'openvino': [
        r'HAS_OPENVINO',
        r'openvino',
        r'import\s+openvino'
    ],
    'qualcomm': [
        r'HAS_QUALCOMM',
        r'QUALCOMM_SDK',
        r'qnn_wrapper',
        r'import\s+qti'
    ],
    'webnn': [
        r'HAS_WEBNN',
        r'WEBNN_AVAILABLE',
        r'WEBNN_SIMULATION'
    ],
    'webgpu': [
        r'HAS_WEBGPU',
        r'WEBGPU_AVAILABLE',
        r'WEBGPU_SIMULATION'
    ]
}

# Model types for categorization
MODEL_TYPES = [
    "text_embedding",
    "text_generation",
    "vision",
    "audio",
    "multimodal",
    "video",
    "vision_language",
    "text_to_image",
    "text_to_audio",
    "text_to_video"
]

def validate_python_syntax(content: str) -> Tuple[bool, List[str]]:
    """
    Validate Python syntax in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors)
    """
    errors = []
    try:
        ast.parse(content)
        return True, []
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return False, errors

def validate_imports(content: str) -> Tuple[bool, List[str]]:
    """
    Validate imports in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors)
    """
    errors = []
    required_imports = {'os', 'sys', 'torch', 'logging'}
    found_imports = set()
    
    # Find all import statements
    import_pattern = re.compile(r'import\s+([\w\.]+)|from\s+([\w\.]+)\s+import')
    for match in import_pattern.finditer(content):
        if match.group(1):
            # 'import x' form
            module = match.group(1).split('.')[0]
            found_imports.add(module)
        elif match.group(2):
            # 'from x import y' form
            module = match.group(2).split('.')[0]
            found_imports.add(module)
    
    # Check for missing required imports
    missing_imports = required_imports - found_imports
    if missing_imports:
        errors.append(f"Missing required imports: {', '.join(missing_imports)}")
    
    return len(errors) == 0, errors

def validate_class_structure(content: str) -> Tuple[bool, List[str]]:
    """
    Validate class structure in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors)
    """
    errors = []
    
    # Parse the AST to analyze structure
    try:
        tree = ast.parse(content)
        
        # Find class definitions
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not classes:
            errors.append("No class definitions found in template")
            return False, errors
        
        # Check for test class naming convention
        test_classes = [cls for cls in classes if cls.name.startswith('Test')]
        if not test_classes:
            errors.append("No test classes found (class names should start with 'Test')")
        
        # Check for methods in test classes
        for cls in test_classes:
            methods = [node.name for node in cls.body if isinstance(node, ast.FunctionDef)]
            
            # Check for at least one test method
            test_methods = [m for m in methods if m.startswith('test_') or m == 'run']
            if not test_methods:
                errors.append(f"Class {cls.name} has no test methods (should start with 'test_' or include 'run' method)")
        
        return len(errors) == 0, errors
    except SyntaxError:
        # This should be caught by validate_python_syntax
        errors.append("Cannot analyze class structure due to syntax errors")
        return False, errors

def validate_hardware_awareness(content: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate hardware awareness and cross-platform support in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors, list of supported platforms)
    """
    errors = []
    warnings = []
    
    supported_platforms = []
    
    # Check for centralized hardware detection
    uses_central_detection = False
    if 'detect_available_hardware' in content or 'hardware_detection' in content:
        uses_central_detection = True
        logger.info("Found centralized hardware detection")
    
    # Check for explicit hardware checks
    for platform, patterns in HARDWARE_CHECKS.items():
        for pattern in patterns:
            if re.search(pattern, content):
                if platform not in supported_platforms:
                    supported_platforms.append(platform)
                break
    
    # If still not found hardware, check for mentions
    if not supported_platforms:
        for platform_name, pattern in HARDWARE_PLATFORMS:
            if re.search(pattern, content, re.IGNORECASE):
                supported_platforms.append(platform_name)
    
    # Core platforms that should be supported
    core_platforms = {'cuda', 'cpu'}
    missing_core = core_platforms - set(supported_platforms)
    
    if missing_core:
        errors.append(f"Missing support for core hardware platforms: {', '.join(missing_core)}")
    
    # Recommended platforms
    recommended_platforms = {'mps', 'rocm', 'openvino'}
    missing_recommended = recommended_platforms - set(supported_platforms)
    
    if missing_recommended:
        warnings.append(f"Missing recommended hardware platforms: {', '.join(missing_recommended)}")
    
    # Web platforms
    web_platforms = {'webnn', 'webgpu'}
    has_web = any(p in supported_platforms for p in web_platforms)
    
    if not has_web:
        warnings.append("No web platform support detected (WebNN/WebGPU)")
    
    # Check for Qualcomm support (new in March 2025)
    has_qualcomm = 'qualcomm' in supported_platforms
    if not has_qualcomm:
        warnings.append("No Qualcomm AI Engine support detected")
    
    # Check for all platforms
    if len(supported_platforms) < 3:
        warnings.append(f"Limited hardware support detected, only found: {', '.join(supported_platforms)}")
    
    # Add warnings to errors
    if warnings:
        errors.extend([f"WARNING: {w}" for w in warnings])
    
    # Only count serious errors (not warnings) for validity
    success = not any(not e.startswith("WARNING:") for e in errors)
    
    return success, errors, supported_platforms

def validate_template_variables(content: str) -> Tuple[bool, List[str]]:
    """
    Validate template variables in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors)
    """
    errors = []
    
    # Check for template variables
    template_vars = re.findall(r'{{(.*?)}}', content)
    
    # Common required variables
    required_vars = ['model_name']
    
    # Check if required variables are present
    found_vars = [var.strip() for var in template_vars]
    
    # Extract variable names from more complex expressions
    cleaned_vars = []
    for var in found_vars:
        # Handle expressions like model_name.replace("-", "")
        if '.' in var:
            base_var = var.split('.')[0].strip()
            cleaned_vars.append(base_var)
        else:
            cleaned_vars.append(var)
    
    # Find missing required variables
    missing_vars = []
    for req_var in required_vars:
        if req_var not in cleaned_vars:
            missing_vars.append(req_var)
    
    if missing_vars:
        errors.append(f"Missing required template variables: {', '.join(missing_vars)}")
    
    # Verify variable patterns are valid
    invalid_vars = []
    for var in template_vars:
        # Check for common errors in variable expressions
        if var.count('{') != var.count('}'):
            invalid_vars.append(var)
        elif var.count('(') != var.count(')'):
            invalid_vars.append(var)
    
    if invalid_vars:
        errors.append(f"Invalid template variable syntax: {', '.join(invalid_vars)}")
    
    return len(errors) == 0, errors

def validate_template_file(file_path: str) -> Dict[str, Any]:
    """
    Validate a template file with multiple validation rules
    
    Args:
        file_path: Path to the template file
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating template: {file_path}")
    
    # Read template content
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        return {
            'valid': False,
            'file': file_path,
            'errors': [f"Failed to read file: {str(e)}"]
        }
    
    # Run all validators
    validators = [
        ('syntax', validate_python_syntax),
        ('imports', validate_imports),
        ('class_structure', validate_class_structure),
        ('template_vars', validate_template_variables),
    ]
    
    all_valid = True
    all_errors = []
    results_by_validator = {}
    
    for validator_name, validator_func in validators:
        if validator_name == 'template_vars':
            valid, errors = validator_func(content)
        else:
            valid, errors = validator_func(content)
        
        results_by_validator[validator_name] = {
            'valid': valid,
            'errors': errors
        }
        
        all_valid = all_valid and valid
        all_errors.extend([f"{validator_name}: {error}" for error in errors])
    
    # Run hardware awareness check separately to capture supported platforms
    hw_valid, hw_errors, supported_platforms = validate_hardware_awareness(content)
    results_by_validator['hardware_awareness'] = {
        'valid': hw_valid,
        'errors': hw_errors,
        'supported_platforms': supported_platforms
    }
    all_valid = all_valid and hw_valid
    all_errors.extend([f"hardware_awareness: {error}" for error in hw_errors])
    
    # Combine all results
    result = {
        'valid': all_valid,
        'file': file_path,
        'errors': all_errors,
        'validators': results_by_validator,
        'supported_platforms': supported_platforms
    }
    
    return result

def validate_template_directory(directory_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Validate all templates in a directory
    
    Args:
        directory_path: Path to directory containing templates
        
    Returns:
        Dictionary mapping file names to validation results
    """
    results = {}
    
    # Find all Python files in the directory
    for file_path in Path(directory_path).glob('*.py'):
        # Skip files starting with underscore
        if file_path.name.startswith('_'):
            continue
        
        # Skip non-template files (basic check)
        if not (file_path.name.startswith('template_') or file_path.name.endswith('template.py')):
            continue
        
        # Validate the template
        result = validate_template_file(str(file_path))
        results[file_path.name] = result
    
    return results

def validate_duckdb_templates(db_path: str = "template_db.duckdb") -> Dict[str, Any]:
    """
    Validate templates stored in a DuckDB database
    
    Args:
        db_path: Path to the DuckDB database
        
    Returns:
        Dictionary with validation results
    """
    if not HAS_DUCKDB:
        logger.warning("DuckDB not available. Using JSON-based template storage instead.")
        
        # Determine JSON database path
        db_dir = os.path.dirname(db_path)
        json_db_path = os.path.join(db_dir, "template_db.json")
        
        if not os.path.exists(json_db_path):
            return {
                'valid': False,
                'error': f"JSON database file not found: {json_db_path}"
            }
            
        try:
            # Load the JSON database
            with open(json_db_path, 'r') as f:
                template_db = json.load(f)
                
            if 'templates' not in template_db:
                return {
                    'valid': False,
                    'error': "No templates found in JSON database"
                }
                
            templates = template_db['templates']
            if not templates:
                return {
                    'valid': False,
                    'error': "No templates found in JSON database"
                }
                
            logger.info(f"Found {len(templates)} templates in JSON database")
            
            results = {}
            valid_count = 0
            
            # Validate each template
            for template_id, template_data in templates.items():
                model_type = template_data.get('model_type', 'unknown')
                template_type = template_data.get('template_type', 'unknown')
                platform = template_data.get('platform')
                content = template_data.get('template', '')
                
                platform_str = f"{model_type}/{template_type}"
                if platform:
                    platform_str += f"/{platform}"
                
                logger.info(f"Validating template {template_id}: {platform_str}")
                
                # Run validations
                validators = [
                    ('syntax', validate_python_syntax),
                    ('imports', validate_imports),
                    ('class_structure', validate_class_structure),
                    ('template_vars', validate_template_variables)
                ]
                
                all_valid = True
                all_errors = []
                results_by_validator = {}
                
                for validator_name, validator_func in validators:
                    valid, errors = validator_func(content)
                    results_by_validator[validator_name] = {
                        'valid': valid,
                        'errors': errors
                    }
                    
                    all_valid = all_valid and valid
                    all_errors.extend([f"{validator_name}: {error}" for error in errors])
                
                # Run hardware awareness check separately
                hw_valid, hw_errors, supported_platforms = validate_hardware_awareness(content)
                results_by_validator['hardware_awareness'] = {
                    'valid': hw_valid,
                    'errors': hw_errors,
                    'supported_platforms': supported_platforms
                }
                all_valid = all_valid and hw_valid
                all_errors.extend([f"hardware_awareness: {error}" for error in hw_errors])
                
                # Store results
                results[template_id] = {
                    'id': template_id,
                    'model_type': model_type,
                    'template_type': template_type,
                    'platform': platform,
                    'valid': all_valid,
                    'errors': all_errors,
                    'validators': results_by_validator,
                    'supported_platforms': supported_platforms
                }
                
                if all_valid:
                    valid_count += 1
            
            return {
                'valid': valid_count == len(templates),
                'total': len(templates),
                'valid_count': valid_count,
                'invalid_count': len(templates) - valid_count,
                'templates': results
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Error validating JSON templates: {str(e)}"
            }
    
    try:
        import duckdb
    except ImportError:
        return {
            'valid': False,
            'error': "Failed to import DuckDB"
        }
    
    if not os.path.exists(db_path):
        return {
            'valid': False,
            'error': f"Database file not found: {db_path}"
        }
    
    try:
        conn = duckdb.connect(db_path)
        
        # Check if templates table exists
        table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='templates'").fetchall()
        if not table_check:
            return {
                'valid': False,
                'error': "No 'templates' table found in database"
            }
        
        # Get all templates
        templates = conn.execute("SELECT id, model_type, template_type, platform, template FROM templates").fetchall()
        if not templates:
            return {
                'valid': False,
                'error': "No templates found in database"
            }
        
        logger.info(f"Found {len(templates)} templates in database")
        
        results = {}
        valid_count = 0
        
        # Validate each template
        for template in templates:
            template_id, model_type, template_type, platform, content = template
            
            platform_str = f"{model_type}/{template_type}"
            if platform:
                platform_str += f"/{platform}"
            
            logger.info(f"Validating template {template_id}: {platform_str}")
            
            # Run validations
            validators = [
                ('syntax', validate_python_syntax),
                ('imports', validate_imports),
                ('class_structure', validate_class_structure),
                ('template_vars', validate_template_variables)
            ]
            
            all_valid = True
            all_errors = []
            results_by_validator = {}
            
            for validator_name, validator_func in validators:
                valid, errors = validator_func(content)
                results_by_validator[validator_name] = {
                    'valid': valid,
                    'errors': errors
                }
                
                all_valid = all_valid and valid
                all_errors.extend([f"{validator_name}: {error}" for error in errors])
            
            # Run hardware awareness check separately
            hw_valid, hw_errors, supported_platforms = validate_hardware_awareness(content)
            results_by_validator['hardware_awareness'] = {
                'valid': hw_valid,
                'errors': hw_errors,
                'supported_platforms': supported_platforms
            }
            all_valid = all_valid and hw_valid
            all_errors.extend([f"hardware_awareness: {error}" for error in hw_errors])
            
            # Store results
            template_key = f"{template_id}_{model_type}_{template_type}"
            if platform:
                template_key += f"_{platform}"
                
            results[template_key] = {
                'id': template_id,
                'model_type': model_type,
                'template_type': template_type,
                'platform': platform,
                'valid': all_valid,
                'errors': all_errors,
                'validators': results_by_validator,
                'supported_platforms': supported_platforms
            }
            
            if all_valid:
                valid_count += 1
        
        conn.close()
        
        return {
            'valid': valid_count == len(templates),
            'total': len(templates),
            'valid_count': valid_count,
            'invalid_count': len(templates) - valid_count,
            'templates': results
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': f"Error validating database templates: {str(e)}"
        }

def create_template_database(db_path: str) -> bool:
    """
    Create a new template database with the required schema
    
    Args:
        db_path: Path to create the DuckDB database
        
    Returns:
        Boolean indicating success or failure
    """
    if not HAS_DUCKDB:
        logger.warning("DuckDB not available. Creating JSON-based template storage instead.")
        db_dir = os.path.dirname(db_path)
        json_db_path = os.path.join(db_dir, "template_db.json")
        
        # Create basic structure
        template_db = {
            "templates": {},
            "template_helpers": {},
            "hardware_platforms": {
                "cuda": {"id": 1, "type": "GPU", "description": "NVIDIA CUDA GPU"},
                "cpu": {"id": 2, "type": "CPU", "description": "Central Processing Unit"},
                "mps": {"id": 3, "type": "GPU", "description": "Apple Metal Performance Shaders"},
                "rocm": {"id": 4, "type": "GPU", "description": "AMD ROCm GPU"},
                "openvino": {"id": 5, "type": "ACCEL", "description": "Intel OpenVINO"},
                "qualcomm": {"id": 6, "type": "MOBILE", "description": "Qualcomm AI Engine"},
                "webnn": {"id": 7, "type": "WEB", "description": "Web Neural Network API"},
                "webgpu": {"id": 8, "type": "WEB", "description": "Web GPU API"}
            },
            "model_types": {
                "text_embedding": {"id": 1, "description": "Text embedding models like BERT, Sentence Transformers"},
                "text_generation": {"id": 2, "description": "Text generation models like GPT, LLAMA, T5"},
                "vision": {"id": 3, "description": "Vision models like ViT, ResNet, DETR"},
                "audio": {"id": 4, "description": "Audio models like Whisper, Wav2Vec2"},
                "multimodal": {"id": 5, "description": "Multimodal models like CLIP, BLIP"},
                "video": {"id": 6, "description": "Video models like XCLIP, VideoMAE"},
                "vision_language": {"id": 7, "description": "Vision-language models like LLaVA"},
                "text_to_image": {"id": 8, "description": "Text-to-image models like Stable Diffusion"},
                "text_to_audio": {"id": 9, "description": "Text-to-audio models like MusicGen"},
                "text_to_video": {"id": 10, "description": "Text-to-video models like Video Diffusion"}
            },
            "created_at": datetime.now().isoformat()
        }
        
        try:
            with open(json_db_path, 'w') as f:
                # Convert datetime to string for JSON serialization
                json.dump(template_db, f, indent=2)
            
            logger.info(f"Created JSON-based template database at {json_db_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating JSON template database: {str(e)}")
            return False
    
    try:
        # Create database connection
        conn = duckdb.connect(db_path)
        
        # Create templates table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY,
            model_type VARCHAR,
            template_type VARCHAR,
            platform VARCHAR,
            template TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create template_helpers table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS template_helpers (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            helper_type VARCHAR,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create template_dependencies table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS template_dependencies (
            template_id INTEGER,
            dependency_id INTEGER,
            dependency_type VARCHAR,
            FOREIGN KEY (template_id) REFERENCES templates(id),
            FOREIGN KEY (dependency_id) REFERENCES templates(id)
        )
        """)
        
        # Create hardware_platforms table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_platforms (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            type VARCHAR,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Insert default hardware platforms
        hw_platforms = [
            (1, 'cuda', 'GPU', 'NVIDIA CUDA GPU'),
            (2, 'cpu', 'CPU', 'Central Processing Unit'),
            (3, 'mps', 'GPU', 'Apple Metal Performance Shaders'),
            (4, 'rocm', 'GPU', 'AMD ROCm GPU'),
            (5, 'openvino', 'ACCEL', 'Intel OpenVINO'),
            (6, 'qualcomm', 'MOBILE', 'Qualcomm AI Engine'),
            (7, 'webnn', 'WEB', 'Web Neural Network API'),
            (8, 'webgpu', 'WEB', 'Web GPU API')
        ]
        
        # Check if hardware platforms already exist
        has_platforms = conn.execute("SELECT COUNT(*) FROM hardware_platforms").fetchone()[0]
        
        if has_platforms == 0:
            conn.executemany("""
            INSERT INTO hardware_platforms (id, name, type, description)
            VALUES (?, ?, ?, ?)
            """, hw_platforms)
            
            logger.info(f"Inserted {len(hw_platforms)} hardware platforms")
        
        # Create model_types table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS model_types (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Insert default model types
        model_type_data = [
            (1, 'text_embedding', 'Text embedding models like BERT, Sentence Transformers'),
            (2, 'text_generation', 'Text generation models like GPT, LLAMA, T5'),
            (3, 'vision', 'Vision models like ViT, ResNet, DETR'),
            (4, 'audio', 'Audio models like Whisper, Wav2Vec2'),
            (5, 'multimodal', 'Multimodal models like CLIP, BLIP'),
            (6, 'video', 'Video models like XCLIP, VideoMAE'),
            (7, 'vision_language', 'Vision-language models like LLaVA'),
            (8, 'text_to_image', 'Text-to-image models like Stable Diffusion'),
            (9, 'text_to_audio', 'Text-to-audio models like MusicGen'),
            (10, 'text_to_video', 'Text-to-video models like Video Diffusion')
        ]
        
        # Check if model types already exist
        has_model_types = conn.execute("SELECT COUNT(*) FROM model_types").fetchone()[0]
        
        if has_model_types == 0:
            conn.executemany("""
            INSERT INTO model_types (id, name, description)
            VALUES (?, ?, ?)
            """, model_type_data)
            
            logger.info(f"Inserted {len(model_type_data)} model types")
        
        # Commit and close
        conn.close()
        
        logger.info(f"Successfully created template database at {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating template database: {str(e)}")
        return False

def migrate_template_files_to_db(source_dir: str, db_path: str) -> Dict[str, Any]:
    """
    Migrate template files from a directory to the DuckDB database
    
    Args:
        source_dir: Directory containing template files
        db_path: Path to the DuckDB database
        
    Returns:
        Dictionary with migration results
    """
    if not HAS_DUCKDB:
        logger.warning("DuckDB not available. Using JSON-based template storage instead.")
        
        # Determine JSON database path
        db_dir = os.path.dirname(db_path)
        json_db_path = os.path.join(db_dir, "template_db.json")
        
        # Check if database exists, create if not
        if not os.path.exists(json_db_path):
            logger.info(f"JSON database {json_db_path} not found, creating...")
            if not create_template_database(db_path):  # This will create a JSON file instead
                return {
                    'success': False,
                    'error': f"Failed to create JSON database at {json_db_path}"
                }
        
        # Load existing database
        try:
            with open(json_db_path, 'r') as f:
                template_db = json.load(f)
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to load JSON database: {str(e)}"
            }
        
        # Find template files
        template_files = []
        for file_path in Path(source_dir).glob('**/*.py'):
            # Skip files starting with underscore
            if file_path.name.startswith('_'):
                continue
            
            # Only include template files
            if file_path.name.startswith('template_') or 'template' in file_path.name.lower():
                template_files.append(str(file_path))
        
        logger.info(f"Found {len(template_files)} template files in {source_dir}")
        
        # Process each template file
        processed = 0
        skipped = 0
        errors = []
        
        for file_path in template_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Parse file name to determine model type and template type
                file_name = os.path.basename(file_path)
                
                # Default values
                model_type = 'unknown'
                template_type = 'test'
                platform = None
                
                # Parse file name to extract information
                if file_name.startswith('template_'):
                    # Format: template_<model_type>[_<platform>].py
                    parts = file_name[9:-3].split('_')  # Remove 'template_' and '.py'
                    if len(parts) > 0:
                        # Try to identify model type
                        for mt in MODEL_TYPES:
                            if mt in parts[0] or parts[0] in mt:
                                model_type = mt
                                break
                        
                        # If not found, use first part
                        if model_type == 'unknown':
                            model_type = parts[0]
                        
                        # Check for platform
                        if len(parts) > 1:
                            for hw_platform, _ in HARDWARE_PLATFORMS:
                                if hw_platform in parts[1]:
                                    platform = hw_platform
                                    break
                
                # Create a unique template ID
                template_id = f"{model_type}_{template_type}"
                if platform:
                    template_id += f"_{platform}"
                template_id += f"_{file_name}"
                
                # Add to database
                template_db['templates'][template_id] = {
                    'id': template_id,
                    'model_type': model_type,
                    'template_type': template_type,
                    'platform': platform,
                    'template': content,
                    'file_path': file_path,
                    'updated_at': datetime.now().isoformat()
                }
                
                processed += 1
                
            except Exception as e:
                errors.append({
                    'file': file_path,
                    'error': str(e)
                })
                skipped += 1
        
        # Save updated database
        try:
            with open(json_db_path, 'w') as f:
                json.dump(template_db, f, indent=2)
            
            logger.info(f"Updated JSON database at {json_db_path}")
            
            return {
                'success': True,
                'total': len(template_files),
                'processed': processed,
                'skipped': skipped,
                'errors': errors,
                'db_path': json_db_path
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error saving JSON database: {str(e)}"
            }
    # Continue with DuckDB implementation if available
    
    if not os.path.exists(source_dir):
        return {
            'success': False,
            'error': f"Source directory not found: {source_dir}"
        }
    
    # Check if database exists, create if not
    if not os.path.exists(db_path):
        logger.info(f"Database {db_path} not found, creating...")
        if not create_template_database(db_path):
            return {
                'success': False,
                'error': f"Failed to create database at {db_path}"
            }
    
    try:
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Find template files
        template_files = []
        for file_path in Path(source_dir).glob('**/*.py'):
            # Skip files starting with underscore
            if file_path.name.startswith('_'):
                continue
            
            # Only include template files
            if file_path.name.startswith('template_') or 'template' in file_path.name.lower():
                template_files.append(str(file_path))
        
        logger.info(f"Found {len(template_files)} template files in {source_dir}")
        
        # Process each template file
        processed = 0
        skipped = 0
        errors = []
        
        for file_path in template_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Parse file name to determine model type and template type
                file_name = os.path.basename(file_path)
                
                # Default values
                model_type = 'unknown'
                template_type = 'test'
                platform = None
                
                # Parse file name to extract information
                if file_name.startswith('template_'):
                    # Format: template_<model_type>[_<platform>].py
                    parts = file_name[9:-3].split('_')  # Remove 'template_' and '.py'
                    if len(parts) > 0:
                        # Try to identify model type
                        for mt in MODEL_TYPES:
                            if mt in parts[0] or parts[0] in mt:
                                model_type = mt
                                break
                        
                        # If not found, use first part
                        if model_type == 'unknown':
                            model_type = parts[0]
                        
                        # Check for platform
                        if len(parts) > 1:
                            for hw_platform, _ in HARDWARE_PLATFORMS:
                                if hw_platform in parts[1]:
                                    platform = hw_platform
                                    break
                
                # Check if template already exists
                exists = conn.execute("""
                SELECT COUNT(*) FROM templates 
                WHERE model_type = ? AND template_type = ? AND (platform = ? OR (platform IS NULL AND ? IS NULL))
                """, [model_type, template_type, platform, platform]).fetchone()[0]
                
                if exists:
                    logger.info(f"Template already exists for {model_type}/{template_type}/{platform}, updating...")
                    conn.execute("""
                    UPDATE templates 
                    SET template = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE model_type = ? AND template_type = ? AND (platform = ? OR (platform IS NULL AND ? IS NULL))
                    """, [content, model_type, template_type, platform, platform])
                else:
                    # Insert new template
                    conn.execute("""
                    INSERT INTO templates (model_type, template_type, platform, template)
                    VALUES (?, ?, ?, ?)
                    """, [model_type, template_type, platform, content])
                
                processed += 1
                
            except Exception as e:
                errors.append({
                    'file': file_path,
                    'error': str(e)
                })
                skipped += 1
        
        # Commit and close
        conn.close()
        
        return {
            'success': True,
            'total': len(template_files),
            'processed': processed,
            'skipped': skipped,
            'errors': errors
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error migrating templates: {str(e)}"
        }

def generate_hardware_compatibility_report(db_path: str, output_file: str = None) -> Dict[str, Any]:
    """
    Generate a hardware compatibility report for templates in the database
    
    Args:
        db_path: Path to the DuckDB database
        output_file: Path to write the report (if None, return as dictionary)
        
    Returns:
        Dictionary with hardware compatibility results
    """
    if not HAS_DUCKDB:
        logger.warning("DuckDB not available. Using JSON-based template storage instead.")
        
        # Determine JSON database path
        db_dir = os.path.dirname(db_path)
        json_db_path = os.path.join(db_dir, "template_db.json")
        
        if not os.path.exists(json_db_path):
            return {
                'success': False,
                'error': f"JSON database file not found: {json_db_path}"
            }
            
        try:
            # Load the JSON database
            with open(json_db_path, 'r') as f:
                template_db = json.load(f)
                
            if 'templates' not in template_db:
                return {
                    'success': False,
                    'error': "No templates found in JSON database"
                }
                
            templates = template_db['templates']
            if not templates:
                return {
                    'success': False,
                    'error': "No templates found in JSON database"
                }
                
            logger.info(f"Found {len(templates)} templates in JSON database")
            
            # Analyze hardware compatibility for each template
            compatibility_matrix = {}
            platform_support = {hw: 0 for hw, _ in HARDWARE_PLATFORMS}
            model_type_counts = {}
            
            for template_id, template_data in templates.items():
                model_type = template_data.get('model_type', 'unknown')
                template_type = template_data.get('template_type', 'unknown')
                platform = template_data.get('platform')
                content = template_data.get('template', '')
                
                # Initialize counters for model types
                if model_type not in model_type_counts:
                    model_type_counts[model_type] = 0
                model_type_counts[model_type] += 1
                
                # Check hardware support
                _, _, supported_platforms = validate_hardware_awareness(content)
                
                # Update compatibility matrix
                if model_type not in compatibility_matrix:
                    compatibility_matrix[model_type] = {hw: 0 for hw, _ in HARDWARE_PLATFORMS}
                
                # Update support counters
                for hw in supported_platforms:
                    if hw in platform_support:
                        platform_support[hw] += 1
                        compatibility_matrix[model_type][hw] += 1
            
            # Calculate percentages
            total_templates = len(templates)
            platform_percentages = {hw: (count / total_templates) * 100 for hw, count in platform_support.items()}
            
            # Calculate percentages by model type
            model_compatibility = {}
            for model_type, hw_counts in compatibility_matrix.items():
                model_compatibility[model_type] = {}
                type_count = model_type_counts[model_type]
                for hw, count in hw_counts.items():
                    model_compatibility[model_type][hw] = (count / type_count) * 100 if type_count > 0 else 0
            
            # Generate markdown report
            if output_file:
                report = "# Hardware Compatibility Report (JSON Database)\n\n"
                report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                report += f"Total templates analyzed: {total_templates}\n\n"
                
                # Overall platform support
                report += "## Overall Platform Support\n\n"
                report += "| Hardware Platform | Templates | Percentage |\n"
                report += "|-------------------|-----------|------------|\n"
                
                for hw, count in platform_support.items():
                    percentage = platform_percentages[hw]
                    report += f"| {hw} | {count} | {percentage:.1f}% |\n"
                
                # Compatibility matrix by model type
                report += "\n## Compatibility Matrix by Model Type\n\n"
                report += "| Model Type | Count | " + " | ".join([hw for hw, _ in HARDWARE_PLATFORMS]) + " |\n"
                report += "|------------|-------|" + "|".join(["---" for _ in HARDWARE_PLATFORMS]) + "|\n"
                
                for model_type, type_count in model_type_counts.items():
                    row = f"| {model_type} | {type_count} |"
                    for hw, _ in HARDWARE_PLATFORMS:
                        percentage = model_compatibility[model_type][hw]
                        status = "✅" if percentage > 75 else "⚠️" if percentage > 25 else "❌"
                        row += f" {status} {percentage:.1f}% |"
                    report += row + "\n"
                
                # High compatibility pairs
                report += "\n## Highly Compatible Model-Hardware Pairs\n\n"
                report += "These combinations have >75% compatibility:\n\n"
                
                for model_type, hw_percentages in model_compatibility.items():
                    high_compat = [(hw, pct) for hw, pct in hw_percentages.items() if pct > 75]
                    if high_compat:
                        report += f"### {model_type}\n"
                        for hw, pct in high_compat:
                            report += f"- {hw}: {pct:.1f}%\n"
                        report += "\n"
                
                # Improvement opportunities
                report += "\n## Improvement Opportunities\n\n"
                report += "These combinations have <25% compatibility and need improvement:\n\n"
                
                for model_type, hw_percentages in model_compatibility.items():
                    low_compat = [(hw, pct) for hw, pct in hw_percentages.items() if pct < 25]
                    if low_compat:
                        report += f"### {model_type}\n"
                        for hw, pct in low_compat:
                            report += f"- {hw}: {pct:.1f}%\n"
                        report += "\n"
                
                # Write report to file
                with open(output_file, 'w') as f:
                    f.write(report)
                
                logger.info(f"Hardware compatibility report written to {output_file}")
            
            return {
                'success': True,
                'total_templates': total_templates,
                'platform_support': platform_support,
                'platform_percentages': platform_percentages,
                'model_compatibility': model_compatibility,
                'model_type_counts': model_type_counts
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error generating hardware compatibility report: {str(e)}"
            }
    
    if not os.path.exists(db_path):
        return {
            'success': False,
            'error': f"Database file not found: {db_path}"
        }
    
    try:
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Get all templates
        templates = conn.execute("""
        SELECT id, model_type, template_type, platform, template 
        FROM templates
        """).fetchall()
        
        if not templates:
            return {
                'success': False,
                'error': "No templates found in database"
            }
        
        logger.info(f"Found {len(templates)} templates in database")
        
        # Analyze hardware compatibility for each template
        compatibility_matrix = {}
        platform_support = {hw: 0 for hw, _ in HARDWARE_PLATFORMS}
        model_type_counts = {}
        
        for template in templates:
            template_id, model_type, template_type, platform, content = template
            
            # Initialize counters for model types
            if model_type not in model_type_counts:
                model_type_counts[model_type] = 0
            model_type_counts[model_type] += 1
            
            # Check hardware support
            _, _, supported_platforms = validate_hardware_awareness(content)
            
            # Update compatibility matrix
            if model_type not in compatibility_matrix:
                compatibility_matrix[model_type] = {hw: 0 for hw, _ in HARDWARE_PLATFORMS}
            
            # Update support counters
            for hw in supported_platforms:
                if hw in platform_support:
                    platform_support[hw] += 1
                    compatibility_matrix[model_type][hw] += 1
        
        # Calculate percentages
        total_templates = len(templates)
        platform_percentages = {hw: (count / total_templates) * 100 for hw, count in platform_support.items()}
        
        # Calculate percentages by model type
        model_compatibility = {}
        for model_type, hw_counts in compatibility_matrix.items():
            model_compatibility[model_type] = {}
            type_count = model_type_counts[model_type]
            for hw, count in hw_counts.items():
                model_compatibility[model_type][hw] = (count / type_count) * 100 if type_count > 0 else 0
        
        # Generate markdown report
        if output_file:
            report = "# Hardware Compatibility Report\n\n"
            report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += f"Total templates analyzed: {total_templates}\n\n"
            
            # Overall platform support
            report += "## Overall Platform Support\n\n"
            report += "| Hardware Platform | Templates | Percentage |\n"
            report += "|-------------------|-----------|------------|\n"
            
            for hw, count in platform_support.items():
                percentage = platform_percentages[hw]
                report += f"| {hw} | {count} | {percentage:.1f}% |\n"
            
            # Compatibility matrix by model type
            report += "\n## Compatibility Matrix by Model Type\n\n"
            report += "| Model Type | Count | " + " | ".join([hw for hw, _ in HARDWARE_PLATFORMS]) + " |\n"
            report += "|------------|-------|" + "|".join(["---" for _ in HARDWARE_PLATFORMS]) + "|\n"
            
            for model_type, type_count in model_type_counts.items():
                row = f"| {model_type} | {type_count} |"
                for hw, _ in HARDWARE_PLATFORMS:
                    percentage = model_compatibility[model_type][hw]
                    status = "✅" if percentage > 75 else "⚠️" if percentage > 25 else "❌"
                    row += f" {status} {percentage:.1f}% |"
                report += row + "\n"
            
            # High compatibility pairs
            report += "\n## Highly Compatible Model-Hardware Pairs\n\n"
            report += "These combinations have >75% compatibility:\n\n"
            
            for model_type, hw_percentages in model_compatibility.items():
                high_compat = [(hw, pct) for hw, pct in hw_percentages.items() if pct > 75]
                if high_compat:
                    report += f"### {model_type}\n"
                    for hw, pct in high_compat:
                        report += f"- {hw}: {pct:.1f}%\n"
                    report += "\n"
            
            # Improvement opportunities
            report += "\n## Improvement Opportunities\n\n"
            report += "These combinations have <25% compatibility and need improvement:\n\n"
            
            for model_type, hw_percentages in model_compatibility.items():
                low_compat = [(hw, pct) for hw, pct in hw_percentages.items() if pct < 25]
                if low_compat:
                    report += f"### {model_type}\n"
                    for hw, pct in low_compat:
                        report += f"- {hw}: {pct:.1f}%\n"
                    report += "\n"
            
            # Write report to file
            with open(output_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Hardware compatibility report written to {output_file}")
        
        # Close connection
        conn.close()
        
        return {
            'success': True,
            'total_templates': total_templates,
            'platform_support': platform_support,
            'platform_percentages': platform_percentages,
            'model_compatibility': model_compatibility,
            'model_type_counts': model_type_counts
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error generating hardware compatibility report: {str(e)}"
        }

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Template Database Validator")
    parser.add_argument("--validate-db", action="store_true", help="Validate templates in the database")
    parser.add_argument("--migrate-templates", action="store_true", help="Migrate template files to database")
    parser.add_argument("--check-hardware", action="store_true", help="Check hardware compatibility of templates")
    parser.add_argument("--create-db", action="store_true", help="Create a new template database")
    parser.add_argument("--source-dir", type=str, help="Source directory for template files")
    parser.add_argument("--db-path", type=str, default="../generators/templates/template_db.duckdb", 
                      help="Path to the DuckDB database")
    parser.add_argument("--report", type=str, help="Path to write hardware compatibility report")
    args = parser.parse_args()
    
    if args.validate_db:
        print(f"Validating templates in database: {args.db_path}")
        db_results = validate_duckdb_templates(args.db_path)
        
        if 'error' in db_results:
            print(f"❌ Error: {db_results['error']}")
            return 1
            
        valid_count = db_results['valid_count']
        invalid_count = db_results['invalid_count']
        total = db_results['total']
        
        print(f"\nValidation Results: {valid_count} valid, {invalid_count} invalid templates (out of {total})")
        
        # Platform support stats
        platform_counts = {
            'cuda': 0,
            'cpu': 0,
            'mps': 0,
            'rocm': 0,
            'openvino': 0,
            'qualcomm': 0,
            'webnn': 0,
            'webgpu': 0
        }
        
        for result in db_results['templates'].values():
            for platform in result.get('supported_platforms', []):
                if platform in platform_counts:
                    platform_counts[platform] += 1
        
        print("\nHardware Platform Support:")
        for platform, count in platform_counts.items():
            percentage = count/total*100 if total else 0
            print(f"  - {platform}: {count} templates ({percentage:.1f}%)")
            
        # Show details for invalid templates
        if invalid_count > 0:
            print("\nInvalid Templates:")
            invalid_templates = {k: v for k, v in db_results['templates'].items() if not v['valid']}
            
            for name, result in invalid_templates.items():
                model_type = result['model_type']
                template_type = result['template_type']
                platform = result['platform'] or 'all'
                template_id = result['id']
                
                print(f"  - Template {template_id} ({model_type}/{template_type}/{platform}):")
                for error in result['errors'][:5]:  # Show first 5 errors
                    print(f"    - {error}")
                if len(result['errors']) > 5:
                    print(f"    - ... and {len(result['errors']) - 5} more errors")
    
    elif args.migrate_templates:
        if not args.source_dir:
            print("❌ Error: --source-dir required for template migration")
            return 1
            
        print(f"Migrating templates from {args.source_dir} to database {args.db_path}")
        migration_results = migrate_template_files_to_db(args.source_dir, args.db_path)
        
        if not migration_results['success']:
            print(f"❌ Error: {migration_results['error']}")
            return 1
            
        print(f"\nMigration Results:")
        print(f"  - Total files: {migration_results['total']}")
        print(f"  - Processed: {migration_results['processed']}")
        print(f"  - Skipped: {migration_results['skipped']}")
        
        if migration_results['errors']:
            print("\nErrors during migration:")
            for error in migration_results['errors']:
                print(f"  - {error['file']}: {error['error']}")
    
    elif args.check_hardware:
        print(f"Checking hardware compatibility in database: {args.db_path}")
        report_results = generate_hardware_compatibility_report(args.db_path, args.report)
        
        if not report_results['success']:
            print(f"❌ Error: {report_results['error']}")
            return 1
            
        print(f"\nHardware Compatibility Summary:")
        print(f"  - Total templates: {report_results['total_templates']}")
        
        print("\nPlatform Support:")
        for platform, count in report_results['platform_support'].items():
            percentage = report_results['platform_percentages'][platform]
            print(f"  - {platform}: {count} templates ({percentage:.1f}%)")
        
        print("\nModel Type Coverage:")
        for model_type, count in report_results['model_type_counts'].items():
            print(f"  - {model_type}: {count} templates")
            for platform, percentage in report_results['model_compatibility'][model_type].items():
                status = "✅" if percentage > 75 else "⚠️" if percentage > 25 else "❌"
                print(f"    - {platform}: {status} {percentage:.1f}%")
        
        if args.report:
            print(f"\nDetailed report written to: {args.report}")
    
    elif args.create_db:
        print(f"Creating new template database at: {args.db_path}")
        if create_template_database(args.db_path):
            print(f"✅ Successfully created template database")
        else:
            print(f"❌ Failed to create template database")
            return 1
    
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())