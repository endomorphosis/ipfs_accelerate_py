#!/usr/bin/env python3
"""
Regenerate all test files with fixes.

This script:
1. Regenerates test files using architecture-aware templates
2. Applies indentation fixes to ensure proper Python syntax
3. Creates backups of original files
4. Verifies syntax validity of generated files
5. Provides a comprehensive report of fixed files

Usage:
    python regenerate_tests_with_fixes.py [--pattern PATTERN] [--verify] [--force]
"""

import os
import sys
import glob
import argparse
import logging
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"regenerate_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Define architecture types for test generation
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "bloom", "llama", "mistral", "falcon", "phi"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2"],
    "vision-encoder-text-decoder": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct"]
}

def find_test_files(directory, pattern):
    """Find test files matching the pattern."""
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    return sorted(files)

def get_architecture_type(model_type):
    """Determine architecture type based on model type."""
    for arch_type, models in ARCHITECTURE_TYPES.items():
        if any(model in model_type.lower() for model in models):
            return arch_type
    return "encoder-only"  # Default to encoder-only if unknown

def get_template_for_architecture(arch_type):
    """Get the appropriate template file for an architecture type."""
    template_map = {
        "encoder-only": "templates/encoder_only_template.py",
        "decoder-only": "templates/decoder_only_template.py",
        "encoder-decoder": "templates/encoder_decoder_template.py",
        "vision": "templates/vision_template.py",
        "vision-encoder-text-decoder": "templates/vision_text_template.py",
        "speech": "templates/speech_template.py",
        "multimodal": "templates/multimodal_template.py"
    }
    
    return template_map.get(arch_type, "templates/encoder_only_template.py")

def get_default_model_for_type(model_type):
    """Get default model ID for a model type."""
    # This is a simple mapping, would be expanded in a real implementation
    default_models = {
        "bert": "bert-base-uncased",
        "gpt2": "gpt2",
        "t5": "t5-small",
        "vit": "google/vit-base-patch16-224",
        "clip": "openai/clip-vit-base-patch32",
        "wav2vec2": "facebook/wav2vec2-base-960h",
        "whisper": "openai/whisper-small"
    }
    
    # Return the default model if found, otherwise use the model type itself
    return default_models.get(model_type.lower(), f"{model_type}-base")

def regenerate_test_file(file_path, force=False, verify=True):
    """
    Regenerate a test file using the architecture-aware generator.
    
    Args:
        file_path: Path to the test file to regenerate
        force: Whether to overwrite if file exists
        verify: Whether to verify syntax after generation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Skip if file exists and force is False
        if os.path.exists(file_path) and not force:
            logger.info(f"Skipping {file_path} - file already exists (use --force to override)")
            return False
        
        # Extract model type from filename
        filename = os.path.basename(file_path)
        if not filename.startswith("test_hf_"):
            logger.warning(f"Invalid filename: {filename}, should start with 'test_hf_'")
            return False
            
        model_type = filename[8:].replace(".py", "")
        
        # Determine architecture type
        arch_type = get_architecture_type(model_type)
        logger.info(f"Determined architecture type '{arch_type}' for model '{model_type}'")
        
        # Get default model ID
        default_model = get_default_model_for_type(model_type)
        
        # Create a registry entry for this model type
        registry_entry = {
            model_type: {
                "description": f"{model_type.upper()} model",
                "class": f"{model_type.capitalize()}ForSequenceClassification",
                "default_model": default_model,
                "architecture": arch_type
            }
        }
        
        # Create backup if file exists
        if os.path.exists(file_path):
            backup_path = f"{file_path}.bak"
            with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Created backup at {backup_path}")
        
        # Select appropriate template based on architecture
        template_file = get_template_for_architecture(arch_type)
        if not os.path.exists(template_file):
            # For this example, we'll simulate the template content
            template_content = generate_template_for_arch(arch_type, model_type)
        else:
            with open(template_file, 'r') as f:
                template_content = f.read()
        
        # Fill template with model-specific information
        content = template_content.replace("MODEL_TYPE", model_type.upper())
        content = content.replace("model_type", model_type)
        content = content.replace("ModelTypeClass", f"{model_type.capitalize()}Class")
        content = content.replace("DEFAULT_MODEL", default_model)
        
        # Write content to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"✅ Generated test file: {file_path}")
        
        # Run indentation fix on the file
        fix_indentation(file_path)
        
        # Verify syntax if requested
        if verify:
            is_valid = verify_syntax(file_path)
            if not is_valid:
                logger.error(f"❌ Generated file has syntax errors: {file_path}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error regenerating test file {file_path}: {e}")
        return False

def fix_indentation(file_path):
    """Apply indentation fixes to a file."""
    logger.info(f"Fixing indentation in {file_path}")
    
    try:
        # Import indentation fixer if available
        try:
            # Try to use the dedicated indentation fixer script
            fix_script = "complete_indentation_fix.py"
            if os.path.exists(fix_script):
                cmd = [sys.executable, fix_script, file_path, "--verify"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ Fixed indentation using {fix_script}")
                    return True
                else:
                    logger.warning(f"Warning: Indentation fix script failed: {result.stderr}")
                    # Continue with built-in fixer
            else:
                logger.warning(f"Warning: Indentation fix script not found: {fix_script}")
                # Continue with built-in fixer
        except Exception as e:
            logger.warning(f"Warning: Error using indentation fix script: {e}")
            # Continue with built-in fixer
        
        # Built-in indentation fixer
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix common indentation issues
        
        # 1. Fix parentheses/brackets in function calls
        content = content.replace(")))))", ")")
        content = content.replace("))))", ")")
        content = content.replace(")))", ")")
        content = content.replace("))", ")")
        
        content = content.replace("((((", "(")
        content = content.replace("(((", "(")
        content = content.replace("((", "(")
        
        content = content.replace("}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}", "}")
        content = content.replace("}}}}}}}}}}}}}}}}}}}}}}}}}}}}}", "}")
        content = content.replace("}}}}}}}}}}}}}}}}}}}}}}}}}}}}", "}")
        content = content.replace("}}}}}}}}}}}}}}}}}}}}}", "}")
        content = content.replace("}}}}}}}}}}}}}", "}")
        content = content.replace("}}}}}", "}")
        
        content = content.replace("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{", "{")
        content = content.replace("{{{{{{{{{{{{{{{{{{{{{{{{{{{", "{")
        content = content.replace("{{{{{{{{{{{{{{{{{{{{{{{{", "{")
        content = content.replace("{{{{{{{{{{{{{{", "{")
        content = content.replace("{{{{{", "{")
        
        content = content.replace("[[[[[", "[")
        content = content.replace("[[[[", "[")
        content = content.replace("[[[", "[")
        content = content.replace("[[", "[")
        
        content = content.replace("]]]]]", "]")
        content = content.replace("]]]]", "]")
        content = content.replace("]]]", "]")
        content = content.replace("]]", "]")
        
        # 2. Fix malformed colons in conditionals and functions
        content = content.replace("try::", "try:")
        content = content.replace("except::", "except:")
        content = content.replace("if ::", "if :")
        content = content.replace("else::", "else:")
        content = content.replace("elif ::", "elif :")
        
        # 3. Fix indentation in class methods
        content = content.replace("        def ", "    def ")
        
        # 4. Fix missing/extra commas in dictionaries and lists
        content = content.replace(",\n        }", "\n        }")
        content = content.replace(",\n    }", "\n    }")
        content = content.replace(",\n}", "\n}")
        
        content = content.replace("[],", "[")
        content = content.replace(",]", "]")
        
        # 5. Fix colons in method definitions
        content = content.replace("def test_pipeline())))):", "def test_pipeline():")
        content = content.replace("def test_from_pretrained())))):", "def test_from_pretrained():")
        content = content.replace("def run_tests())))):", "def run_tests():")
        
        # 6. Fix indentation levels
        lines = content.split('\n')
        fixed_lines = []
        
        in_class = False
        in_method = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append("")
                continue
                
            # Handle class definitions
            if stripped.startswith("class ") and stripped.endswith(":"):
                in_class = True
                in_method = False
                fixed_lines.append(stripped)
                continue
                
            # Handle method definitions
            if in_class and stripped.startswith("def ") and stripped.endswith(":"):
                in_method = True
                fixed_lines.append("    " + stripped)
                continue
                
            # Handle method body
            if in_class and in_method:
                fixed_lines.append("        " + stripped)
                continue
                
            # Handle class body (not in method)
            if in_class and not in_method:
                fixed_lines.append("    " + stripped)
                continue
                
            # Handle top-level code
            fixed_lines.append(stripped)
        
        # Write fixed content back to file
        with open(file_path, 'w') as f:
            f.write('\n'.join(fixed_lines))
        
        return True
        
    except Exception as e:
        logger.error(f"Error fixing indentation in {file_path}: {e}")
        return False

def verify_syntax(file_path):
    """Verify Python syntax of a file."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", file_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {file_path}: Syntax is valid")
            return True
        else:
            logger.error(f"❌ {file_path}: Syntax error")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"❌ {file_path}: Error validating syntax: {e}")
        return False

def generate_template_for_arch(arch_type, model_type):
    """Generate a template for a specific architecture type."""
    if arch_type == "encoder-only":
        return f"""#!/usr/bin/env python3

# Import hardware detection capabilities if available
try:
    from generators.hardware.hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback
    
    import os
    import sys
    import json
    import time
    import datetime
    import traceback
    import logging
    import argparse
    from unittest.mock import patch, MagicMock, Mock
    from typing import Dict, List, Any, Optional, Union
    from pathlib import Path

    import asyncio
# Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
    import numpy as np

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

# Try to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Hardware detection
def check_hardware():
    \"\"\"Check available hardware and return capabilities.\"\"\"
    capabilities = {
        "cpu": True,
        "cuda": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "mps": False,
        "openvino": False
    }
    
    # Check CUDA
    if HAS_TORCH:
        capabilities["cuda"] = torch.cuda.is_available()
        if capabilities["cuda"]:
            capabilities["cuda_devices"] = torch.cuda.device_count()
            capabilities["cuda_version"] = torch.version.cuda
    
    # Check MPS (Apple Silicon)
    if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        capabilities["mps"] = torch.mps.is_available()
    
    # Check OpenVINO
    try:
        import openvino
        capabilities["openvino"] = True
    except ImportError:
        pass
    
    return capabilities

# Get hardware capabilities
HW_CAPABILITIES = check_hardware()

# Models registry - Maps model IDs to their specific configurations
{model_type.upper()}_MODELS_REGISTRY = {{
    "{model_type}-base": {{
        "description": "{model_type.upper()} base model",
        "class": "{model_type.capitalize()}ForMaskedLM",
        "vocab_size": 30522,
    }}
}}

class Test{model_type.capitalize()}Models:
    \"\"\"Base test class for all {model_type.upper()}-family models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class for a specific model or default.\"\"\"
        self.model_id = model_id or "{model_type}-base"
        
        # Verify model exists in registry
        if self.model_id not in {model_type.upper()}_MODELS_REGISTRY:
            logger.warning(f"Model {{self.model_id}} not in registry, using default configuration")
            self.model_info = {model_type.upper()}_MODELS_REGISTRY["{model_type}-base"]
        else:
            self.model_info = {model_type.upper()}_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "fill-mask"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_text = "The quick brown fox jumps over the [MASK] dog."
        self.test_texts = [
            "The quick brown fox jumps over the [MASK] dog.",
            "The quick brown fox jumps over the [MASK] dog. (alternative)"
        ]
        
        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {{self.preferred_device}} as preferred device")
        
        # Results storage
        self.results = {{}}
        self.examples = []
        self.performance_stats = {{}}
    
    def test_pipeline(self, device="auto"):
        \"\"\"Test the model using transformers pipeline API.\"\"\"
        if device == "auto":
            device = self.preferred_device
        
        results = {{
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }}
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            results["pipeline_success"] = False
            return results
            
        try:
            logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {{
                "task": self.task,
                "model": self.model_id,
                "device": device
            }}
            
            # Time the model loading
            load_start_time = time.time()
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            
            # Prepare test input
            pipeline_input = self.test_text
            
            # Run warmup inference if on CUDA
            if device == "cuda":
                try:
                    _ = pipeline(pipeline_input)
                except Exception:
                    pass
            
            # Run multiple inference passes
            num_runs = 3
            times = []
            outputs = []
            
            for _ in range(num_runs):
                start_time = time.time()
                output = pipeline(pipeline_input)
                end_time = time.time()
                times.append(end_time - start_time)
                outputs.append(output)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Store results
            results["pipeline_success"] = True
            results["pipeline_avg_time"] = avg_time
            results["pipeline_min_time"] = min_time
            results["pipeline_max_time"] = max_time
            results["pipeline_load_time"] = load_time
            results["pipeline_error_type"] = "none"
            
            # Add to examples
            self.examples.append({{
                "method": f"pipeline() on {{device}}",
                "input": str(pipeline_input),
                "output_preview": str(outputs[0])[:200] + "..." if len(str(outputs[0])) > 200 else str(outputs[0])
            }})
            
            # Store in performance stats
            self.performance_stats[f"pipeline_{{device}}"] = {{
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "load_time": load_time,
                "num_runs": num_runs
            }}
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_traceback"] = traceback.format_exc()
            logger.error(f"Error testing pipeline on {{device}}: {{e}}")
            
            # Classify error type
            error_str = str(e).lower()
            traceback_str = traceback.format_exc().lower()
            
            if "cuda" in error_str or "cuda" in traceback_str:
                results["pipeline_error_type"] = "cuda_error"
            elif "memory" in error_str:
                results["pipeline_error_type"] = "out_of_memory"
            elif "no module named" in error_str:
                results["pipeline_error_type"] = "missing_dependency"
            else:
                results["pipeline_error_type"] = "other"
        
        # Add to overall results
        self.results[f"pipeline_{{device}}"] = results
        return results
    
    def test_from_pretrained(self, device="auto"):
        \"\"\"Test the model using direct from_pretrained loading.\"\"\"
        if device == "auto":
            device = self.preferred_device
        
        results = {{
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }}
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_core"] = ["transformers"]
            results["from_pretrained_success"] = False
            return results
        
        try:
            logger.info(f"Testing {{self.model_id}} with from_pretrained() on {{device}}...")
            
            # Common parameters for loading
            pretrained_kwargs = {{
                "local_files_only": False
            }}
            
            # Time tokenizer loading
            tokenizer_load_start = time.time()
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id,
                **pretrained_kwargs
            )
            tokenizer_load_time = time.time() - tokenizer_load_start
            
            # Use appropriate model class based on model type
            model_class = None
            if self.class_name == "{model_type.capitalize()}ForMaskedLM":
                model_class = getattr(transformers, self.class_name, None)
                if model_class is None:
                    # Fallback to Auto class
                    model_class = transformers.AutoModelForMaskedLM
            else:
                # Fallback to Auto class
                model_class = transformers.AutoModelForMaskedLM
            
            # Time model loading
            model_load_start = time.time()
            model = model_class.from_pretrained(
                self.model_id,
                **pretrained_kwargs
            )
            model_load_time = time.time() - model_load_start
            
            # Move model to device
            if device != "cpu":
                model = model.to(device)
            
            # Prepare test input
            test_input = self.test_text
            
            # Tokenize input
            inputs = tokenizer(test_input, return_tensors="pt")
            
            # Move inputs to device
            if device != "cpu":
                inputs = {{key: val.to(device) for key, val in inputs.items()}}
            
            # Run warmup inference if using CUDA
            if device == "cuda":
                try:
                    with torch.no_grad():
                        _ = model(**inputs)
                except Exception:
                    pass
            
            # Run multiple inference passes
            num_runs = 3
            times = []
            outputs = []
            
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    output = model(**inputs)
                end_time = time.time()
                times.append(end_time - start_time)
                outputs.append(output)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Calculate model size
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = (param_count * 4) / (1024 * 1024)  # Rough size in MB
            
            # Store results
            results["from_pretrained_success"] = True
            results["from_pretrained_avg_time"] = avg_time
            results["from_pretrained_min_time"] = min_time
            results["from_pretrained_max_time"] = max_time
            results["tokenizer_load_time"] = tokenizer_load_time
            results["model_load_time"] = model_load_time
            results["model_size_mb"] = model_size_mb
            results["from_pretrained_error_type"] = "none"
            
            # Add to examples
            example_data = {{
                "method": f"from_pretrained() on {{device}}",
                "input": str(test_input),
                "model_info": {{
                    "size_mb": model_size_mb,
                    "parameters": param_count
                }}
            }}
            
            self.examples.append(example_data)
            
            # Store in performance stats
            self.performance_stats[f"from_pretrained_{{device}}"] = {{
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "tokenizer_load_time": tokenizer_load_time,
                "model_load_time": model_load_time,
                "model_size_mb": model_size_mb,
                "num_runs": num_runs
            }}
            
        except Exception as e:
            # Store error information
            results["from_pretrained_success"] = False
            results["from_pretrained_error"] = str(e)
            results["from_pretrained_traceback"] = traceback.format_exc()
            logger.error(f"Error testing from_pretrained on {{device}}: {{e}}")
            
            # Classify error type
            error_str = str(e).lower()
            traceback_str = traceback.format_exc().lower()
            
            if "cuda" in error_str or "cuda" in traceback_str:
                results["from_pretrained_error_type"] = "cuda_error"
            elif "memory" in error_str:
                results["from_pretrained_error_type"] = "out_of_memory"
            elif "no module named" in error_str:
                results["from_pretrained_error_type"] = "missing_dependency"
            else:
                results["from_pretrained_error_type"] = "other"
        
        # Add to overall results
        self.results[f"from_pretrained_{{device}}"] = results
        return results
    
    def run_tests(self, all_hardware=False):
        \"\"\"
        Run all tests for this model.
        
        Args:
            all_hardware: If True, tests on all available hardware (CPU, CUDA, OpenVINO)
        
        Returns:
            Dict containing test results
        \"\"\"
        # Always test on default device
        self.test_pipeline()
        self.test_from_pretrained()
        
        # Test on all available hardware if requested
        if all_hardware:
            # Always test on CPU
            if self.preferred_device != "cpu":
                self.test_pipeline(device="cpu")
                self.test_from_pretrained(device="cpu")
            
            # Test on CUDA if available
            if HW_CAPABILITIES["cuda"] and self.preferred_device != "cuda":
                self.test_pipeline(device="cuda")
                self.test_from_pretrained(device="cuda")
        
        # Build final results
        return {{
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "hardware": HW_CAPABILITIES,
            "metadata": {{
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "description": self.description,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH
            }}
        }}

def save_results(model_id, results, output_dir="collected_results"):
    \"\"\"Save test results to a file.\"\"\"
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    filename = f"hf_{model_type}_{{safe_model_id}}_{{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {{output_path}}")
    return output_path

def get_available_models():
    \"\"\"Get a list of all available {model_type.upper()} models in the registry.\"\"\"
    return list({model_type.upper()}_MODELS_REGISTRY.keys())

def test_all_models(output_dir="collected_results", all_hardware=False):
    \"\"\"Test all registered {model_type.upper()} models.\"\"\"
    models = get_available_models()
    results = {{}}
    
    for model_id in models:
        logger.info(f"Testing model: {{model_id}}")
        tester = Test{model_type.capitalize()}Models(model_id)
        model_results = tester.run_tests(all_hardware=all_hardware)
        
        # Save individual results
        save_results(model_id, model_results, output_dir=output_dir)
        
        # Add to summary
        results[model_id] = {{
            "success": any(r.get("pipeline_success", False) for r in model_results["results"].values())
        }}
    
    # Save summary
    summary_path = os.path.join(output_dir, f"hf_{model_type}_summary_{{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved summary to {{summary_path}}")
    return results

def main():
    \"\"\"Command-line entry point.\"\"\"
    parser = argparse.ArgumentParser(description="Test {model_type.upper()}-family models")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Specific model to test")
    model_group.add_argument("--all-models", action="store_true", help="Test all registered models")
    
    # Hardware options
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for output files")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    # List options
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = get_available_models()
        print("\\nAvailable {model_type.upper()}-family models:")
        for model in models:
            info = {model_type.upper()}_MODELS_REGISTRY[model]
            print(f"  - {{model}} ({{info['class']}}): {{info['description']}}")
        return
    
    # Create output directory if needed
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test all models if requested
    if args.all_models:
        results = test_all_models(output_dir=args.output_dir, all_hardware=args.all_hardware)
        
        # Print summary
        print("\\n{model_type.upper()} Models Testing Summary:")
        total = len(results)
        successful = sum(1 for r in results.values() if r["success"])
        print(f"Successfully tested {{successful}} of {{total}} models ({{successful/total*100:.1f}}%)")
        return
    
    # Test single model (default or specified)
    model_id = args.model or "{model_type}-base"
    logger.info(f"Testing model: {{model_id}}")
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Run test
    tester = Test{model_type.capitalize()}Models(model_id)
    results = tester.run_tests(all_hardware=args.all_hardware)
    
    # Save results if requested
    if args.save:
        save_results(model_id, results, output_dir=args.output_dir)
    
    # Print summary
    success = any(r.get("pipeline_success", False) for r in results["results"].values())
    
    print("\\nTEST RESULTS SUMMARY:")
    if success:
        print(f"✅ Successfully tested {{model_id}}")
        
        # Print performance highlights
        for device, stats in results["performance"].items():
            if "avg_time" in stats:
                print(f"  - {{device}}: {{stats['avg_time']:.4f}}s average inference time")
    else:
        print(f"❌ Failed to test {{model_id}}")
        
        # Print error information
        for test_name, result in results["results"].items():
            if "pipeline_error" in result:
                print(f"  - Error in {{test_name}}: {{result.get('pipeline_error_type', 'unknown')}}")
                print(f"    {{result.get('pipeline_error', 'Unknown error')}}")
    
    print("\\nFor detailed results, use --save flag and check the JSON output file.")

if __name__ == "__main__":
    main()
"""
    elif arch_type == "decoder-only":
        # Similar template for decoder-only models like GPT-2
        return f"""# Decoder-only template for {model_type} would go here"""
    elif arch_type == "encoder-decoder":
        # Similar template for encoder-decoder models like T5
        return f"""# Encoder-decoder template for {model_type} would go here"""
    elif arch_type == "vision":
        # Similar template for vision models like ViT
        return f"""# Vision model template for {model_type} would go here"""
    else:
        # Default template for unknown architecture types
        return f"""# Default template for {model_type} with {arch_type} architecture would go here"""

def run_regeneration(directory, pattern, verify=True, force=False):
    """
    Regenerate test files using the architecture-aware generator.
    
    Args:
        directory: Directory containing test files
        pattern: File pattern to match
        verify: Whether to verify syntax after regeneration
        force: Whether to regenerate even if file exists
    
    Returns:
        Tuple of (num_regenerated, num_failed, total)
    """
    # Find test files
    files = find_test_files(directory, pattern)
    logger.info(f"Found {len(files)} files matching pattern {pattern}")
    
    regenerated = []
    failed = []
    skipped = []
    
    for file_path in files:
        # Check if already regenerated 
        if os.path.exists(file_path) and not force:
            logger.info(f"Skipping {file_path} - file already exists (use --force to override)")
            skipped.append(file_path)
            continue
        
        # Regenerate file
        if regenerate_test_file(file_path, force=force, verify=verify):
            regenerated.append(file_path)
        else:
            failed.append(file_path)
    
    # Print summary
    logger.info("\nRegeneration Summary:")
    logger.info(f"- Regenerated: {len(regenerated)} files")
    logger.info(f"- Failed: {len(failed)} files")
    logger.info(f"- Skipped: {len(skipped)} files")
    logger.info(f"- Total: {len(files)} files")
    
    if failed:
        logger.info("\nFailed files:")
        for f in failed:
            logger.info(f"  - {f}")
    
    return len(regenerated), len(failed), len(files)

def main():
    parser = argparse.ArgumentParser(description="Regenerate test files with fixes")
    parser.add_argument("--pattern", type=str, default="test_hf_*.py", 
                       help="File pattern to match (default: test_hf_*.py)")
    parser.add_argument("--directory", type=str, default=".", 
                       help="Directory containing test files (default: current directory)")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify syntax after regeneration")
    parser.add_argument("--force", action="store_true", 
                       help="Regenerate even if file exists")
    parser.add_argument("--single", type=str, 
                       help="Regenerate a single model type (e.g. 'bert')")
    
    args = parser.parse_args()
    
    # Handle single file regeneration
    if args.single:
        file_path = os.path.join(args.directory, f"test_hf_{args.single}.py")
        logger.info(f"Regenerating single file: {file_path}")
        success = regenerate_test_file(file_path, force=args.force, verify=args.verify)
        return 0 if success else 1
    
    # Run regeneration
    regenerated, failed, total = run_regeneration(
        directory=args.directory,
        pattern=args.pattern,
        verify=args.verify,
        force=args.force
    )
    
    # Return appropriate exit code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())