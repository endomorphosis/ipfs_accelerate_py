#!/usr/bin/env python3
"""
Phase 16 Generator Fixes

This script addresses issues in the test generators and benchmark generators
to ensure they properly generate correct Python code and support all hardware 
platforms, including cross-platform testing and data storage in the DB.

It fixes:
1. Syntax issues in merged_test_generator.py and integrated_skillset_generator.py
2. Indentation problems in template generation
3. Proper handling of template string formatting
4. WebNN and WebGPU platform support
5. Ensuring database compatibility in benchmark generators

Usage: python fix_generators_phase16.py [--all] [--test-only] [--benchmark-only]
"""

import os
import sys
import re
import shutil
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("generator_fixes")

# Paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = SCRIPT_DIR
BACKUP_DIR = TEST_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

FIXED_MERGED_GENERATOR = TEST_DIR / "fixed_merged_test_generator.py"
MERGED_GENERATOR = TEST_DIR / "merged_test_generator.py"
INTEGRATED_GENERATOR = TEST_DIR / "integrated_skillset_generator.py"
BENCHMARK_GENERATOR = TEST_DIR / "benchmark_all_key_models.py"
BENCHMARK_RUNNER = TEST_DIR / "run_model_benchmarks.py"
HARDWARE_TEMPLATES_DIR = TEST_DIR / "hardware_test_templates"

# List of files to check and fix
TEST_GENERATORS = [
    MERGED_GENERATOR,
    FIXED_MERGED_GENERATOR,
    INTEGRATED_GENERATOR
]

BENCHMARK_GENERATORS = [
    BENCHMARK_GENERATOR,
    BENCHMARK_RUNNER
]

def create_backup(file_path):
    """Create a backup of the specified file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = BACKUP_DIR / f"{file_path.name}.bak_{timestamp}"
    
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup of {file_path.name} at {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup of {file_path.name}: {e}")
        return False

def fix_template_string_syntax(file_path):
    """Fix template string syntax issues."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix triple-quote issues in template declarations
        content = re.sub(r'template_database\["[^"]+"\] = """"""',
                         lambda m: m.group(0).replace('""""""', '"""'),
                         content)
        
        # Fix docstring termination within templates
        content = re.sub(r'(template_database\["[^"]+"\] = """\n.*?)\n"""',
                         lambda m: m.group(1) + '\n"""',
                         content, flags=re.DOTALL)
        
        # Fix nested triple quotes in template strings
        content = re.sub(r'(""".+?)"""(.+?)"""', 
                        lambda m: m.group(1) + '\'\'\'\\"\'\'' + m.group(2) + '\'\'\'\\"\'\'' if '"""' in m.group(2) else m.group(0), 
                        content, flags=re.DOTALL)
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Fixed template string syntax in {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error fixing template string syntax in {file_path.name}: {e}")
        return False

def fix_indentation_issues(file_path):
    """Fix indentation issues in template generation."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix indentation in if statements and class definitions
        # Pattern 1: Missing indentation after if statements
        pattern1 = re.compile(r'(\s+if [^\n:]+:\n)(\s+)([^\s])', re.MULTILINE)
        content = pattern1.sub(r'\1\2    \3', content)
        
        # Pattern 2: Missing indentation after class definitions
        pattern2 = re.compile(r'(\s+class [^\n:]+:\n)(\s+)([^\s])', re.MULTILINE)
        content = pattern2.sub(r'\1\2    \3', content)
        
        # Fix indentation in generated functions and methods
        # Pattern 3: Ensure consistent indent in methods
        pattern3 = re.compile(r'(\s+def [^\n:]+:\n)(\s+)([^\s])', re.MULTILINE)
        content = pattern3.sub(r'\1\2    \3', content)
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Fixed indentation issues in {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error fixing indentation issues in {file_path.name}: {e}")
        return False

def fix_web_platform_support(file_path):
    """Add proper WebNN and WebGPU platform support."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for existing web platform support
        if "from fixed_web_platform import" not in content:
            # Add web platform import
            web_platform_import = """
# Import fixed WebNN and WebGPU platform support
try:
    from fixed_web_platform import process_for_web, init_webnn, init_webgpu, create_mock_processors
    WEB_PLATFORM_SUPPORT = True
except ImportError:
    WEB_PLATFORM_SUPPORT = False
    print("WebNN and WebGPU platform support not available - install the fixed_web_platform module")
from unittest.mock import MagicMock
"""
            
            # Find import section
            import_end = content.find("# Configure logging")
            if import_end == -1:
                import_end = content.find("# Constants")
                if import_end == -1:
                    import_end = content.find("def ")
                    
            content = content[:import_end] + web_platform_import + content[import_end:]
            
        # Add enhanced WebNN and WebGPU handlers
        webnn_handler_template = """
def create_webnn_handler(self, model_name=None, model_path=None, model_type=None, device="webnn", web_api_mode="simulation", tokenizer=None, **kwargs):
        """Initialize model for WebNN inference."""
        try:
            model_name = model_name or self.model_name
            model_path = model_path or self.model_path
            model_type = model_type or getattr(self, 'model_type', 'text')
            
            # Process specific WebNN options if any
            web_api_mode = web_api_mode.lower()
            implementation_type = "REAL_WEBNN" if web_api_mode == "real" else "SIMULATION_WEBNN"
            
            # Create handler function for WebNN
            def handler(input_data, **kwargs):
                # Process with web platform support if available
                if WEB_PLATFORM_SUPPORT:
                    try:
                        result = process_for_web(model_type, input_data, platform="webnn")
                        return {
                            "output": result,
                            "implementation_type": implementation_type,
                            "model": model_name,
                            "platform": "webnn"
                        }
                    except Exception as e:
                        print(f"Error in WebNN processing: {e}")
                
                # Fallback to mock output
                mock_output = {"mock_output": f"WebNN mock output for {model_name}"}
                return {
                    "output": mock_output,
                    "implementation_type": "MOCK_WEBNN",
                    "model": model_name,
                    "platform": "webnn"
                }
            
            return handler
        except Exception as e:
            print(f"Error creating WebNN handler: {e}")
            # Return simple mock handler
            return lambda x: {"output": "Error in WebNN handler", "implementation_type": "ERROR", "error": str(e)}
"""

        webgpu_handler_template = """
def create_webgpu_handler(self, model_name=None, model_path=None, model_type=None, device="webgpu", web_api_mode="simulation", tokenizer=None, **kwargs):
        """Initialize model for WebGPU inference."""
        try:
            model_name = model_name or self.model_name
            model_path = model_path or self.model_path
            model_type = model_type or getattr(self, 'model_type', 'text')
            
            # Process specific WebGPU options if any
            web_api_mode = web_api_mode.lower()
            implementation_type = "REAL_WEBGPU" if web_api_mode == "real" else "SIMULATION_WEBGPU"
            
            # Create handler function for WebGPU
            def handler(input_data, **kwargs):
                # Process with web platform support if available
                if WEB_PLATFORM_SUPPORT:
                    try:
                        result = process_for_web(model_type, input_data, platform="webgpu")
                        return {
                            "output": result,
                            "implementation_type": implementation_type,
                            "model": model_name,
                            "platform": "webgpu"
                        }
                    except Exception as e:
                        print(f"Error in WebGPU processing: {e}")
                
                # Fallback to mock output
                mock_output = {"mock_output": f"WebGPU mock output for {model_name}"}
                return {
                    "output": mock_output,
                    "implementation_type": "MOCK_WEBGPU",
                    "model": model_name,
                    "platform": "webgpu"
                }
            
            return handler
        except Exception as e:
            print(f"Error creating WebGPU handler: {e}")
            # Return simple mock handler
            return lambda x: {"output": "Error in WebGPU handler", "implementation_type": "ERROR", "error": str(e)}
"""
        
        # Add WebNN and WebGPU handlers if they don't exist
        if "def create_webnn_handler" not in content:
            # Find a good location to insert
            class_def = re.search(r'class\s+\w+\s*\(.*\):', content)
            if class_def:
                class_match = class_def.group(0)
                class_index = content.index(class_match)
                class_end = content.find("\n\n", class_index)
                if class_end == -1:
                    class_end = content.find("def ", class_index)
                
                # Insert handlers after class definition
                if class_end != -1:
                    content = content[:class_end] + "\n" + webnn_handler_template + "\n" + webgpu_handler_template + content[class_end:]
            
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Enhanced web platform support in {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error enhancing web platform support in {file_path.name}: {e}")
        return False

def fix_hardware_platform_handling(file_path):
    """Fix hardware platform handling code."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix hardware detection code
        hardware_detection_code = """
def detect_available_hardware():
    """Detect available hardware platforms on the current system."""
    available_hardware = {
        "cpu": True  # CPU is always available
    }
    
    # CUDA (NVIDIA GPUs)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        available_hardware["cuda"] = cuda_available
        if cuda_available:
            logger.info(f"CUDA available with {torch.cuda.device_count()} devices")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        available_hardware["cuda"] = False
        logger.info("CUDA not available: torch not installed")
    
    # MPS (Apple Silicon)
    try:
        import torch
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        available_hardware["mps"] = mps_available
        if mps_available:
            logger.info("MPS (Apple Silicon) available")
    except ImportError:
        available_hardware["mps"] = False
        logger.info("MPS not available: torch not installed")
    except AttributeError:
        available_hardware["mps"] = False
        logger.info("MPS not available: torch version does not support mps")
    
    # ROCm (AMD GPUs)
    try:
        import torch
        rocm_available = torch.cuda.is_available() and hasattr(torch.version, "hip")
        available_hardware["rocm"] = rocm_available
        if rocm_available:
            logger.info(f"ROCm available with {torch.cuda.device_count()} devices")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        available_hardware["rocm"] = False
        logger.info("ROCm not available: torch not installed")
    except AttributeError:
        available_hardware["rocm"] = False
        logger.info("ROCm not available: torch version does not support hip")
    
    # OpenVINO (Intel)
    try:
        import openvino
        available_hardware["openvino"] = True
        logger.info(f"OpenVINO available: {openvino.__version__}")
        # Get available devices
        try:
            core = openvino.Core()
            devices = core.available_devices
            logger.info(f"OpenVINO devices: {devices}")
        except:
            logger.info("Could not get OpenVINO devices")
    except ImportError:
        available_hardware["openvino"] = False
        logger.info("OpenVINO not available: openvino not installed")
    
    # WebNN and WebGPU - check if fixed_web_platform module is available
    # For WebNN
    if WEB_PLATFORM_SUPPORT:
        available_hardware["webnn"] = True
        available_hardware["webgpu"] = True
        logger.info("WebNN and WebGPU simulation available via fixed_web_platform module")
    else:
        available_hardware["webnn"] = False
        available_hardware["webgpu"] = False
        logger.info("WebNN and WebGPU not available: fixed_web_platform module not found")
    
    # Browser environment detection - for simulation vs real implementation
    try:
        # Check if we're in a browser environment
        import js
        if hasattr(js, 'navigator'):
            if hasattr(js.navigator, 'ml'):
                logger.info("WebNN API detected in browser environment")
                available_hardware["webnn"] = True
            if hasattr(js.navigator, 'gpu'):
                logger.info("WebGPU API detected in browser environment")
                available_hardware["webgpu"] = True
    except ImportError:
        # Not in a browser environment, use simulation if WEB_PLATFORM_SUPPORT is True
        pass
    
    # Check for NPU (Neural Processing Unit) support
    try:
        # Different approaches to detect NPUs
        npu_detected = False
        
        # Check for Intel NPU
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices
            if any(d for d in devices if any(npu_name in d for npu_name in ["NPU", "MYRIAD", "HDDL", "GNA"])):
                npu_detected = True
                logger.info("Intel NPU detected via OpenVINO")
        except:
            pass
            
        # Check for Apple Neural Engine
        try:
            import platform
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                try:
                    import coremltools
                    npu_detected = True
                    logger.info("Apple Neural Engine detected")
                except ImportError:
                    pass
        except:
            pass
        
        # Check for Qualcomm NPU
        try:
            import qnn
            npu_detected = True
            logger.info("Qualcomm NPU detected")
        except ImportError:
            pass
        
        available_hardware["npu"] = npu_detected
    except:
        available_hardware["npu"] = False
    
    return available_hardware
"""
        # If hardware detection function is missing or incomplete, replace it
        if "def detect_available_hardware" not in content or "WebNN and WebGPU - check if fixed_web_platform module is available" not in content:
            # Find where to add hardware detection (after imports)
            constants_start = content.find("# Constants")
            if constants_start != -1:
                # Add before constants
                content = content[:constants_start] + hardware_detection_code + content[constants_start:]
            else:
                # Add at the end of imports
                import_end = content.find("import ")
                if import_end != -1:
                    last_import = content.rfind("import ", 0, 1000)  # Look in first 1000 chars
                    if last_import != -1:
                        import_end = content.find("\n", last_import)
                        if import_end != -1:
                            content = content[:import_end+1] + "\n" + hardware_detection_code + content[import_end+1:]
        
        # Ensure hardware selection is implemented
        hardware_selection_method = """
def get_hardware_platform_code(platform, model_type, modality):
    """Generate code for specific hardware platform."""
    platform = platform.lower()
    
    # Base import code common to all platforms
    platform_imports = [f"# {platform.upper()} platform specific imports"]
    
    # Platform-specific imports and initialization
    if platform == "cpu":
        platform_imports.append("# CPU does not require specific imports")
        platform_init = "self.init_cpu()"
        platform_handler = "handler = self.create_cpu_handler()"
    elif platform == "cuda":
        platform_imports.append("import torch")
        platform_init = "self.init_cuda()"
        platform_handler = "handler = self.create_cuda_handler()"
    elif platform == "openvino":
        platform_imports.append("import openvino")
        platform_init = "self.init_openvino()"
        platform_handler = "handler = self.create_openvino_handler()"
    elif platform == "mps":
        platform_imports.append("import torch")
        platform_init = "self.init_mps()"
        platform_handler = "handler = self.create_mps_handler()"
    elif platform == "rocm":
        platform_imports.append("import torch")
        platform_init = "self.init_rocm()"
        platform_handler = "handler = self.create_rocm_handler()"
    elif platform == "webnn":
        platform_imports.append("# WebNN specific imports")
        if WEB_PLATFORM_SUPPORT:
            platform_imports.append("from fixed_web_platform import process_for_web, init_webnn, create_mock_processors")
        platform_init = "self.init_webnn()"
        platform_handler = "handler = self.create_webnn_handler()"
    elif platform == "webgpu":
        platform_imports.append("# WebGPU specific imports")
        if WEB_PLATFORM_SUPPORT:
            platform_imports.append("from fixed_web_platform import process_for_web, init_webgpu, create_mock_processors")
            # Add model-specific optimizations for WebGPU
            if modality == "audio":
                platform_imports.append("from fixed_web_platform.webgpu_audio_compute_shaders import optimize_for_firefox")
            elif modality in ["multimodal", "vision_language"]:
                platform_imports.append("from fixed_web_platform.progressive_model_loader import load_models_in_parallel")
            elif modality == "text_generation":
                platform_imports.append("from fixed_web_platform.webgpu_shader_precompilation import precompile_shaders")
        platform_init = "self.init_webgpu()"
        platform_handler = "handler = self.create_webgpu_handler()"
    else:
        # Default to CPU
        platform_imports.append(f"# Unsupported platform '{platform}', using CPU")
        platform_init = "self.init_cpu()"
        platform_handler = "handler = self.create_cpu_handler()"
    
    return {
        "imports": platform_imports,
        "init": platform_init,
        "handler": platform_handler,
        "platform_name": platform.upper()
    }
"""
        # If hardware selection method is missing, add it
        if "def get_hardware_platform_code" not in content:
            # Find a good location to insert
            detect_hardware_end = content.find("def detect_available_hardware")
            if detect_hardware_end != -1:
                detect_hardware_end = content.find("\ndef ", detect_hardware_end + 20)
                if detect_hardware_end != -1:
                    content = content[:detect_hardware_end] + hardware_selection_method + content[detect_hardware_end:]
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Fixed hardware platform handling in {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error fixing hardware platform handling in {file_path.name}: {e}")
        return False

def fix_benchmark_generators(file_path):
    """Fix benchmark generators to ensure database integration."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Ensure database saving code is present and correctly implemented
        db_save_check = re.search(r'if\s+not\s+os\.environ\.get\s*\(\s*[\'"](DEPRECATE_JSON_OUTPUT)[\'"]', content)
        if db_save_check:
            # Fix found but may need correction - check implementation
            db_var = db_save_check.group(1)
            
            # Ensure we have correct DB condition
            db_condition_pattern = re.compile(r'(if\s+not\s+os\.environ\.get\s*\(\s*[\'"]DEPRECATE_JSON_OUTPUT[\'"].*?\))(.*?)(else\s*:)', re.DOTALL)
            matches = db_condition_pattern.findall(content)
            
            if matches:
                for match in matches:
                    # Check if the if/else is properly handling DB vs JSON
                    if "save_to_json" in match[1] and "save_to_database" in match[2]:
                        # Already correctly implemented
                        pass
                    else:
                        # Fix implementation
                        old_condition = match[0] + match[1] + match[2]
                        new_condition = """if not os.environ.get("DEPRECATE_JSON_OUTPUT", "0") == "1":
            # Legacy JSON output mode
            save_to_json(result, output_dir)
        else:
            # Database storage mode - save to DuckDB
            save_to_database(result, db_path)"""
                        content = content.replace(old_condition, new_condition)
            
            # If no matches found, add the code
            else:
                # Find a place to add the DB saving code
                save_result_pattern = re.compile(r'def\s+save_results\s*\(.*?\)', re.DOTALL)
                save_result_match = save_result_pattern.search(content)
                
                if save_result_match:
                    save_func_start = save_result_match.start()
                    save_func_end = content.find("def ", save_func_start + 10)
                    
                    # Add new save_results function with database support
                    new_save_function = """def save_results(result, output_dir=None, db_path=None):
    """Save benchmark results to file or database."""
    if not os.environ.get("DEPRECATE_JSON_OUTPUT", "0") == "1":
        # Legacy JSON output mode
        save_to_json(result, output_dir)
    else:
        # Database storage mode - save to DuckDB
        save_to_database(result, db_path)
"""
                    
                    # Add database functions if not present
                    db_function = """
def save_to_database(result, db_path=None):
    """Save benchmark results to DuckDB database."""
    if db_path is None:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    try:
        # Import database utilities
        import duckdb
        import pandas as pd
        from datetime import datetime
        
        # Convert result to DataFrame
        df = pd.DataFrame([result])
        
        # Add timestamp
        df["timestamp"] = datetime.now().isoformat()
        
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Create table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_results (
                model_name VARCHAR,
                hardware VARCHAR,
                batch_size INTEGER,
                throughput DOUBLE,
                latency DOUBLE,
                memory_usage DOUBLE,
                precision VARCHAR,
                platform VARCHAR,
                timestamp VARCHAR,
                additional_info VARCHAR
            )
        ''')
        
        # Insert data
        conn.execute('''
            INSERT INTO benchmark_results 
            SELECT * FROM df
        ''')
        
        # Close connection
        conn.close()
        
        print(f"Results saved to database: {db_path}")
    except Exception as e:
        print(f"Error saving to database: {e}")
        # Fallback to JSON if database fails
        save_to_json(result, "./benchmark_results_fallback")
"""
                    
                    # Check if database functions exist
                    if "def save_to_database" not in content:
                        content = content.replace(save_result_match.group(0), new_save_function + db_function)
        
        # Ensure database path is configured properly
        db_path_pattern = re.compile(r'(parser\.add_argument\s*\(\s*[\'"]-+db-path[\'"].*?)\)') 
        db_path_match = db_path_pattern.search(content)
        
        if not db_path_match:
            # Add db_path argument if missing
            parser_pattern = re.compile(r'parser\s*=\s*argparse\.ArgumentParser\(.*?\)', re.DOTALL)
            parser_match = parser_pattern.search(content)
            
            if parser_match:
                # Find the end of argument declarations
                parser_end = content.find("args = parser.parse_args()", parser_match.end())
                if parser_end != -1:
                    # Add db_path argument
                    db_path_arg = """
    parser.add_argument("--db-path", type=str, default=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"),
                        help="Path to benchmark database")
    parser.add_argument("--no-db-store", action="store_true", 
                        help="Disable storing results in database, even if DEPRECATE_JSON_OUTPUT is set")
"""
                    content = content[:parser_end] + db_path_arg + content[parser_end:]
        
        # Ensure the database imports are present
        if "import duckdb" not in content:
            # Find import section
            import_end = content.find("\n\n", content.find("import "))
            if import_end != -1:
                # Add database imports
                db_imports = """
# Database imports
try:
    import duckdb
    import pandas as pd
    HAS_DB_SUPPORT = True
except ImportError:
    HAS_DB_SUPPORT = False
    print("Warning: Database support not available. Install duckdb and pandas.")
"""
                content = content[:import_end] + db_imports + content[import_end:]
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Fixed benchmark generator database integration in {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error fixing benchmark generator database integration in {file_path.name}: {e}")
        return False

def fix_template_loading(file_path):
    """Ensure proper template loading from hardware_test_templates."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if template loading from hardware_test_templates is implemented
        if "hardware_test_templates" not in content or "load_template_from_file" not in content:
            # Add template loading code
            template_loading_code = """
def load_template_from_hardware_templates(model_name, model_category=None):
    """Load a template from hardware_test_templates directory."""
    # Try model-specific template first
    template_path = Path(os.path.dirname(os.path.abspath(__file__))) / "hardware_test_templates"
    model_template = template_path / f"template_{model_name}.py"
    
    if model_template.exists():
        with open(model_template, 'r') as f:
            return f.read()
    
    # Try category template next
    if model_category:
        category_template = template_path / f"template_{model_category}.py"
        if category_template.exists():
            with open(category_template, 'r') as f:
                return f.read()
    
    # Use default templates from template_database as fallback
    return None
"""
            # Find a good location to insert
            template_section = content.find("template_database = {}")
            if template_section != -1:
                template_section_end = content.find("\n\n", template_section)
                if template_section_end != -1:
                    content = content[:template_section_end+2] + template_loading_code + content[template_section_end+2:]
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Fixed template loading in {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error fixing template loading in {file_path.name}: {e}")
        return False

def fix_platform_cli_args(file_path):
    """Ensure platform-related CLI arguments are properly handled."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if platform CLI arguments are implemented
        platform_args_pattern = re.compile(r'parser\.add_argument\s*\(\s*[\'"]-+platform[\'"]')
        if not platform_args_pattern.search(content):
            # Find the argument parser
            parser_pattern = re.compile(r'parser\s*=\s*argparse\.ArgumentParser\(.*?\)', re.DOTALL)
            parser_match = parser_pattern.search(content)
            
            if parser_match:
                # Find the end of argument declarations
                parser_end = content.find("args = parser.parse_args()", parser_match.end())
                if parser_end != -1:
                    # Add platform arguments
                    platform_args = """
    # Hardware platform options
    hardware_group = parser.add_argument_group('Hardware Platform Options')
    hardware_group.add_argument("--platform", type=str, 
                               choices=["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu", "all"],
                               default="all", help="Hardware platform to generate tests for")
    hardware_group.add_argument("--cross-platform", action="store_true", 
                               help="Ensure cross-platform compatibility in tests")
    hardware_group.add_argument("--enhance-webnn", action="store_true", 
                               help="Generate tests with enhanced WebNN support")
    hardware_group.add_argument("--enhance-webgpu", action="store_true", 
                               help="Generate tests with enhanced WebGPU support")
"""
                    content = content[:parser_end] + platform_args + content[parser_end:]
        
        # Check if platform argument handling is implemented in main function
        main_function_pattern = re.compile(r'def\s+main\s*\(\s*\)\s*:')
        main_match = main_function_pattern.search(content)
        
        if main_match:
            # Find platform argument handling
            platform_handling_pattern = re.compile(r'args\.platform')
            if not platform_handling_pattern.search(content[main_match.end():]):
                # Find a place to add platform handling
                main_body_start = content.find("    ", main_match.end())
                if main_body_start != -1:
                    # Add platform handling code
                    platform_handling = """
    # Process hardware platform options
    platform = args.platform.lower() if hasattr(args, 'platform') else "all"
    cross_platform = args.cross_platform if hasattr(args, 'cross_platform') else False
    
    if platform != "all":
        print(f"Targeting specific hardware platform: {platform.upper()}")
    
    if cross_platform:
        print("Ensuring cross-platform compatibility in generated tests")
"""
                    content = content[:main_body_start] + platform_handling + content[main_body_start:]
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Fixed platform CLI arguments in {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error fixing platform CLI arguments in {file_path.name}: {e}")
        return False

def fix_test_generator(file_path):
    """Apply all fixes to a test generator file."""
    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return False
    
    # Create backup
    if not create_backup(file_path):
        logger.error(f"Failed to create backup for {file_path.name}, skipping")
        return False
    
    success = True
    # Apply all fixes
    success = fix_template_string_syntax(file_path) and success
    success = fix_indentation_issues(file_path) and success
    success = fix_web_platform_support(file_path) and success
    success = fix_hardware_platform_handling(file_path) and success
    success = fix_template_loading(file_path) and success
    success = fix_platform_cli_args(file_path) and success
    
    if success:
        logger.info(f"Successfully fixed {file_path.name}")
    else:
        logger.error(f"Failed to fix {file_path.name}")
    
    return success

def fix_benchmark_generator(file_path):
    """Apply all fixes to a benchmark generator file."""
    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return False
    
    # Create backup
    if not create_backup(file_path):
        logger.error(f"Failed to create backup for {file_path.name}, skipping")
        return False
    
    success = True
    # Apply benchmark-specific fixes
    success = fix_benchmark_generators(file_path) and success
    
    if success:
        logger.info(f"Successfully fixed benchmark generator {file_path.name}")
    else:
        logger.error(f"Failed to fix benchmark generator {file_path.name}")
    
    return success

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix test and benchmark generators for Phase 16")
    parser.add_argument("--all", action="store_true", help="Fix all generators")
    parser.add_argument("--test-only", action="store_true", help="Fix only test generators")
    parser.add_argument("--benchmark-only", action="store_true", help="Fix only benchmark generators")
    args = parser.parse_args()
    
    # Determine which generators to fix
    fix_test = args.all or args.test_only or (not args.benchmark_only)
    fix_benchmark = args.all or args.benchmark_only
    
    success = True
    
    if fix_test:
        logger.info("Fixing test generators...")
        for generator in TEST_GENERATORS:
            success = fix_test_generator(generator) and success
    
    if fix_benchmark:
        logger.info("Fixing benchmark generators...")
        for generator in BENCHMARK_GENERATORS:
            success = fix_benchmark_generator(generator) and success
    
    if success:
        logger.info("Successfully fixed all generators")
        return 0
    else:
        logger.error("Failed to fix one or more generators")
        return 1

if __name__ == "__main__":
    sys.exit(main())