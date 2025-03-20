#!/usr/bin/env python3
"""
Comprehensive management tool for HuggingFace test files.

This script provides a unified interface for:
1. Creating minimal test files with correct indentation
2. Trying to fix indentation in existing files
3. Validating syntax in test files
4. Batch processing multiple files

Usage:
    python manage_test_files.py create <family> <output_path>
    python manage_test_files.py fix <file_path>
    python manage_test_files.py validate <file_path>
    python manage_test_files.py batch-create <families> [--output-dir DIRECTORY]
"""

import sys
import os
import re
import glob
import shutil
import subprocess
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model family templates
MODEL_TEMPLATES = {
    "bert": {
        "class_name": "BertModel",
        "model_class": "BertForMaskedLM",
        "tokenizer_class": "BertTokenizer",
        "task": "fill-mask",
        "model_id": "bert-base-uncased",
        "test_text": "The man worked as a [MASK].",
        "architecture_type": "encoder_only",
    },
    "gpt2": {
        "class_name": "GPT2LMHeadModel",
        "model_class": "GPT2LMHeadModel",
        "tokenizer_class": "GPT2Tokenizer",
        "task": "text-generation",
        "model_id": "gpt2",
        "test_text": "Once upon a time",
        "architecture_type": "decoder_only",
    },
    "t5": {
        "class_name": "T5ForConditionalGeneration",
        "model_class": "T5ForConditionalGeneration",
        "tokenizer_class": "T5Tokenizer",
        "task": "translation_en_to_fr",
        "model_id": "t5-small",
        "test_text": "translate English to French: Hello, how are you?",
        "architecture_type": "encoder_decoder",
    },
    "vit": {
        "class_name": "ViTForImageClassification",
        "model_class": "ViTForImageClassification",
        "processor_class": "ViTImageProcessor",
        "task": "image-classification",
        "model_id": "google/vit-base-patch16-224",
        "test_image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "architecture_type": "encoder_only",
        "is_vision": True,
    },
}

# Add more model families based on architecture types
EXTENDED_MODEL_FAMILIES = {
    "encoder_only": [
        "roberta", "distilbert", "albert", "electra", 
        "convnext", "clip", "bert"
    ],
    "decoder_only": [
        "gpt_neo", "gpt_neox", "gptj", "opt", "llama", "bloom",
        "gpt2"
    ],
    "encoder_decoder": [
        "bart", "pegasus", "mbart", "mt5", "longt5", "t5"
    ],
    "vision": [
        "detr", "swin", "convnext", "vit"
    ]
}

def generate_minimal_imports():
    """Generate minimal import statements with correct indentation."""
    return """#!/usr/bin/env python3
\"\"\"
Minimal test file for HuggingFace model.
\"\"\"

import os
import sys
import json
import time
import logging
import argparse
from unittest.mock import MagicMock
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        "mps": False
    }
    
    # Check CUDA
    if HAS_TORCH:
        capabilities["cuda"] = torch.cuda.is_available()
        
    # Check MPS (Apple Silicon)
    if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        capabilities["mps"] = torch.mps.is_available()
        
    return capabilities

HW_CAPABILITIES = check_hardware()
"""

def generate_minimal_class(family):
    """Generate a minimal test class with correct indentation."""
    
    family_info = MODEL_TEMPLATES.get(family, MODEL_TEMPLATES["bert"])
    
    # For families not explicitly defined, find by architecture type
    if family not in MODEL_TEMPLATES:
        for arch_type, families in EXTENDED_MODEL_FAMILIES.items():
            if family in families:
                # Find a template for this architecture
                for template_family, template_info in MODEL_TEMPLATES.items():
                    if template_info.get("architecture_type") == arch_type:
                        family_info = template_info.copy()
                        # Update with family-specific info
                        family_info["model_id"] = family
                        break
                break
    
    upper_family = family.upper()
    cap_family = family.capitalize()
    class_name = family_info["class_name"]
    model_class = family_info["model_class"]
    tokenizer_class = family_info.get("tokenizer_class", "AutoTokenizer")
    model_id = family_info["model_id"]
    task = family_info["task"]
    test_text = family_info.get("test_text", "Test input")
    
    # For vision models, use different inputs
    is_vision = family_info.get("is_vision", False)
    
    if is_vision:
        test_input = f"""        # Vision input
        self.test_image_url = "{family_info.get('test_image_url', 'http://example.com/image.jpg')}"
"""
    else:
        test_input = f"""        # Text input
        self.test_text = "{test_text}"
"""
    
    return f"""
# Model registry
{upper_family}_MODELS_REGISTRY = {{
    "{model_id}": {{
        "description": "{family} base model",
        "class": "{class_name}",
    }},
}}

class Test{cap_family}Models:
    \"\"\"Test class for {family}-family models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class for a specific model or default.\"\"\"
        self.model_id = model_id or "{model_id}"
        
        # Verify model exists in registry
        if self.model_id not in {upper_family}_MODELS_REGISTRY:
            logger.warning(f"Model {{self.model_id}} not in registry, using default")
            self.model_info = {upper_family}_MODELS_REGISTRY["{model_id}"]
        else:
            self.model_info = {upper_family}_MODELS_REGISTRY[self.model_id]
            
        # Define model parameters
        self.task = "{task}"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
{test_input}
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
            {'pipeline_input = self.test_image_url' if is_vision else 'pipeline_input = self.test_text'}
            
            # Run inference passes
            num_runs = 1
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
            
            # Store results
            results["pipeline_success"] = True
            results["pipeline_avg_time"] = avg_time
            results["pipeline_load_time"] = load_time
            
            # Add to examples
            self.examples.append({{
                "method": f"pipeline() on {{device}}",
                "input": str(pipeline_input),
                "output_preview": str(outputs[0])[:200] if len(str(outputs[0])) > 200 else str(outputs[0])
            }})
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            logger.error(f"Error testing pipeline on {{device}}: {{e}}")
        
        # Add to overall results
        self.results[f"pipeline_{{device}}"] = results
        return results
    
    def run_tests(self):
        \"\"\"Run all tests for this model.\"\"\"
        # Run test
        self.test_pipeline()
        
        # Return results
        return {{
            "results": self.results,
            "examples": self.examples,
            "metadata": {{
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name
            }}
        }}


def save_results(model_id, results, output_dir="results"):
    \"\"\"Save test results to a file.\"\"\"
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    filename = f"hf_{family}_{{safe_model_id}}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {{output_path}}")
    return output_path


def main():
    \"\"\"Command-line entry point.\"\"\"
    parser = argparse.ArgumentParser(description="Test {family}-family models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output files")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Test single model (default or specified)
    model_id = args.model or "{model_id}"
    logger.info(f"Testing model: {{model_id}}")
    
    # Run test
    tester = Test{cap_family}Models(model_id)
    results = tester.run_tests()
    
    # Save results if requested
    if args.save:
        save_results(model_id, results, output_dir=args.output_dir)
    
    # Print summary
    success = any(r.get("pipeline_success", False) for r in results["results"].values())
    
    print("\\nTEST RESULTS SUMMARY:")
    if success:
        print(f"✅ Successfully tested {{model_id}}")
    else:
        print(f"❌ Failed to test {{model_id}}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())"""

def get_supported_families():
    """Get list of all supported model families."""
    families = set(MODEL_TEMPLATES.keys())
    for arch_families in EXTENDED_MODEL_FAMILIES.values():
        families.update(arch_families)
    return sorted(families)

def create_minimal_test_file(family, output_path):
    """
    Create a minimal test file with correct indentation.
    
    Args:
        family: Model family name
        output_path: Output file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supported_families = get_supported_families()
        
        # Check if family is supported
        if family not in supported_families:
            logger.error(f"Unsupported model family: {family}")
            logger.info(f"Supported families include: {', '.join(list(supported_families)[:10])}...")
            return False
        
        # Generate content
        imports = generate_minimal_imports()
        class_content = generate_minimal_class(family)
        
        content = imports + class_content
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Created minimal test file at {output_path}")
        
        # Verify syntax
        try:
            compile(content, output_path, 'exec')
            logger.info(f"✅ {output_path}: Syntax is valid")
            return True
        except SyntaxError as e:
            logger.error(f"❌ {output_path}: Syntax error: {e}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error creating minimal test file: {e}")
        return False

def attempt_fix_indentation(file_path, backup=True):
    """
    Attempt to fix indentation issues in an existing file.
    Uses multiple strategies and falls back if needed.
    
    Args:
        file_path: Path to the file to fix
        backup: Whether to create a backup
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create backup if requested
        if backup:
            backup_path = f"{file_path}.bak"
            with open(backup_path, 'w') as f:
                f.write(content)
            logger.info(f"Created backup at {backup_path}")
        
        # Extract the model family name from filename
        base_name = os.path.basename(file_path)
        match = re.search(r'test_hf_(\w+)\.py', base_name)
        
        if match:
            family = match.group(1)
            logger.info(f"Detected model family: {family}")
            
            # Check if we have a template for this family
            if family in get_supported_families():
                # Option 1: Create a new file from template
                logger.info(f"Creating new file from template for {family}")
                minimal_path = f"{file_path}.minimal"
                if create_minimal_test_file(family, minimal_path):
                    # Compare the files
                    with open(minimal_path, 'r') as f:
                        minimal_content = f.read()
                    
                    logger.info(f"Created minimal version at {minimal_path}")
                    logger.info("You can compare the files and use the minimal version if needed")
                    
                    # Don't overwrite by default
                    return True
            
        # Option 2: Try to fix indentation with basic patterns
        # This is simplified since previously attempted fixers had issues
        
        # Basic fixes
        fixed_content = content
        
        # 1. Fix class definition indent (must be at column 0)
        fixed_content = re.sub(r'^\s+class\s+(\w+):', r'class \1:', fixed_content, flags=re.MULTILINE)
        
        # 2. Fix method definition indent (must be 4 spaces inside class)
        fixed_content = re.sub(r'^\s*def\s+(\w+)\(self', r'    def \1(self', fixed_content, flags=re.MULTILINE)
        
        # 3. Fix top-level function definition indent (must be at column 0)
        fixed_content = re.sub(r'^\s+def\s+(\w+)\((?!self)', r'def \1(', fixed_content, flags=re.MULTILINE)
        
        # 4. Fix import statements (should be at column 0)
        fixed_content = re.sub(r'^\s+import\s+', r'import ', fixed_content, flags=re.MULTILINE)
        fixed_content = re.sub(r'^\s+from\s+', r'from ', fixed_content, flags=re.MULTILINE)
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        # Verify syntax
        try:
            compile(fixed_content, file_path, 'exec')
            logger.info(f"✅ {file_path}: Fixed file now has valid syntax")
            return True
        except SyntaxError as e:
            logger.error(f"❌ {file_path}: Syntax error remains: {e}")
            logger.error(f"  Line {e.lineno}, column {e.offset}: {e.text.strip() if e.text else ''}")
            
            # Revert to original if backup exists
            if backup:
                logger.info(f"Reverting to backup")
                shutil.copy2(backup_path, file_path)
            
            return False
        
    except Exception as e:
        logger.error(f"❌ Error fixing file: {e}")
        return False

def verify_syntax(file_path):
    """
    Verify Python syntax of a file.
    
    Args:
        file_path: Path to the file to verify
        
    Returns:
        bool: True if syntax is valid, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        try:
            compile(content, file_path, 'exec')
            logger.info(f"✅ {file_path}: Syntax is valid")
            return True
        except SyntaxError as e:
            logger.error(f"❌ {file_path}: Syntax error: {e}")
            logger.error(f"  Line {e.lineno}, column {e.offset}: {e.text.strip() if e.text else ''}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error verifying syntax: {e}")
        return False

def batch_create_test_files(families, output_dir="."):
    """
    Create minimal test files for multiple families.
    
    Args:
        families: List of model families
        output_dir: Output directory
        
    Returns:
        Tuple of (successful, failed, total)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    successful = []
    failed = []
    
    for family in families:
        output_path = os.path.join(output_dir, f"test_hf_{family}.py")
        logger.info(f"Creating minimal test file for {family}...")
        
        if create_minimal_test_file(family, output_path):
            successful.append(family)
        else:
            failed.append(family)
    
    # Print summary
    logger.info("\nBatch Creation Summary:")
    logger.info(f"- Successful: {len(successful)} files")
    logger.info(f"- Failed: {len(failed)} files")
    logger.info(f"- Total: {len(families)} files")
    
    if failed:
        logger.info("\nFailed families:")
        for f in failed:
            logger.info(f"  - {f}")
    
    return len(successful), len(failed), len(families)

def main():
    parser = argparse.ArgumentParser(description="Manage HuggingFace test files")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a minimal test file")
    create_parser.add_argument("family", help="Model family")
    create_parser.add_argument("output_path", help="Output file path")
    
    # Fix command
    fix_parser = subparsers.add_parser("fix", help="Fix indentation in an existing file")
    fix_parser.add_argument("file_path", help="Path to the file to fix")
    fix_parser.add_argument("--no-backup", action="store_true", help="Do not create a backup")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate syntax of a file")
    validate_parser.add_argument("file_path", help="Path to the file to validate")
    
    # Batch-create command
    batch_parser = subparsers.add_parser("batch-create", help="Create minimal test files for multiple families")
    batch_parser.add_argument("families", nargs="+", help="Model families")
    batch_parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List supported model families")
    
    args = parser.parse_args()
    
    if args.command == "create":
        return 0 if create_minimal_test_file(args.family, args.output_path) else 1
    
    elif args.command == "fix":
        return 0 if attempt_fix_indentation(args.file_path, not args.no_backup) else 1
    
    elif args.command == "validate":
        return 0 if verify_syntax(args.file_path) else 1
    
    elif args.command == "batch-create":
        successful, failed, total = batch_create_test_files(args.families, args.output_dir)
        return 0 if failed == 0 else 1
    
    elif args.command == "list":
        families = get_supported_families()
        print("\nSupported model families:")
        for family in families:
            if family in MODEL_TEMPLATES:
                print(f"  - {family} (template available)")
            else:
                # Find architecture type
                arch_type = "unknown"
                for at, fams in EXTENDED_MODEL_FAMILIES.items():
                    if family in fams:
                        arch_type = at
                        break
                print(f"  - {family} ({arch_type})")
        return 0

if __name__ == "__main__":
    sys.exit(main())