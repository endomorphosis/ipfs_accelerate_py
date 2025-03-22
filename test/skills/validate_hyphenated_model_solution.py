#!/usr/bin/env python3
"""
Expanded validation for hyphenated model solution with model-specific validation rules.

This script:
1. Validates generated test files against model architecture requirements
2. Performs model inference validation when possible
3. Generates comprehensive validation reports with actionable insights
4. Provides recommendations for test improvements

Usage:
    python validate_hyphenated_model_solution.py [--files FILE_PATTERNS] [--all] [--model MODEL_ID] [--inference] [--report]
"""

import os
import sys
import re
import json
import time
import glob
import argparse
import logging
import traceback
import importlib.util
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add timestamped file handler
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"validation_{timestamp}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FIXED_TESTS_DIR = CURRENT_DIR / "fixed_tests"
TEMPLATES_DIR = CURRENT_DIR / "templates"
VALIDATION_REPORTS_DIR = CURRENT_DIR / "validation_reports"

# Create validation directories if they don't exist
os.makedirs(FIXED_TESTS_DIR, exist_ok=True)
os.makedirs(VALIDATION_REPORTS_DIR, exist_ok=True)

# Architecture mapping for validation rules
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta", "albert"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5", "speech-to-text", "speech-to-text-2", "wav2vec2-bert"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava", "chinese-clip"]
}

# Known hyphenated model families to validate
KNOWN_HYPHENATED_MODELS = [
    "gpt-j",
    "gpt-neo",
    "gpt-neox",
    "xlm-roberta", 
    "vision-text-dual-encoder",
    "speech-to-text",
    "speech-to-text-2",
    "chinese-clip",
    "data2vec-text",
    "data2vec-audio",
    "data2vec-vision",
    "wav2vec2-bert"
]

# Common model class names by architecture type for validation
MODEL_CLASS_MAPPING = {
    "encoder-only": {
        "bert": "BertForMaskedLM",
        "distilbert": "DistilBertForMaskedLM",
        "roberta": "RobertaForMaskedLM",
        "xlm-roberta": "XLMRobertaForMaskedLM",
        "albert": "AlbertForMaskedLM",
        "electra": "ElectraForMaskedLM"
    },
    "decoder-only": {
        "gpt2": "GPT2LMHeadModel",
        "gpt-j": "GPTJForCausalLM",
        "gpt-neo": "GPTNeoForCausalLM",
        "gpt-neox": "GPTNeoXForCausalLM",
        "bloom": "BloomForCausalLM",
        "llama": "LlamaForCausalLM"
    },
    "encoder-decoder": {
        "t5": "T5ForConditionalGeneration",
        "mt5": "MT5ForConditionalGeneration",
        "bart": "BartForConditionalGeneration",
        "mbart": "MBartForConditionalGeneration",
        "pegasus": "PegasusForConditionalGeneration"
    },
    "vision": {
        "vit": "ViTForImageClassification",
        "swin": "SwinForImageClassification",
        "deit": "DeiTForImageClassification",
        "beit": "BeitForImageClassification"
    },
    "vision-text": {
        "clip": "CLIPModel",
        "vision-text-dual-encoder": "VisionTextDualEncoderModel",
        "chinese-clip": "ChineseCLIPModel"
    },
    "speech": {
        "wav2vec2": "Wav2Vec2ForCTC",
        "hubert": "HubertForCTC",
        "whisper": "WhisperForConditionalGeneration",
        "speech-to-text": "Speech2TextForConditionalGeneration",
        "speech-to-text-2": "Speech2Text2ForCTC",
        "wav2vec2-bert": "Wav2Vec2BertForCTC"
    }
}

# Validation rules by architecture type
ARCHITECTURE_VALIDATION_RULES = {
    "encoder-only": {
        "required_methods": ["test_pipeline", "test_from_pretrained", "run_tests"],
        "required_tasks": ["fill-mask"],
        "required_variables": ["HAS_TRANSFORMERS", "HAS_TORCH", "HAS_TOKENIZERS"],
        "model_inputs": ["text"],
        "result_keys": ["predictions", "pipeline_success", "from_pretrained_success"]
    },
    "decoder-only": {
        "required_methods": ["test_pipeline", "test_from_pretrained", "run_tests"],
        "required_tasks": ["text-generation"],
        "required_variables": ["HAS_TRANSFORMERS", "HAS_TORCH", "HAS_TOKENIZERS"],
        "model_inputs": ["text"],
        "result_keys": ["predictions", "pipeline_success", "from_pretrained_success"]
    },
    "encoder-decoder": {
        "required_methods": ["test_pipeline", "test_from_pretrained", "run_tests"],
        "required_tasks": ["text2text-generation", "translation"],
        "required_variables": ["HAS_TRANSFORMERS", "HAS_TORCH", "HAS_TOKENIZERS", "HAS_SENTENCEPIECE"],
        "model_inputs": ["text"],
        "result_keys": ["predictions", "pipeline_success", "from_pretrained_success"]
    },
    "vision": {
        "required_methods": ["test_pipeline", "test_from_pretrained", "run_tests"],
        "required_tasks": ["image-classification"],
        "required_variables": ["HAS_TRANSFORMERS", "HAS_TORCH", "HAS_TOKENIZERS"],
        "model_inputs": ["image"],
        "result_keys": ["predictions", "pipeline_success", "from_pretrained_success"]
    },
    "vision-text": {
        "required_methods": ["test_pipeline", "test_from_pretrained", "run_tests"],
        "required_tasks": ["image-to-text", "zero-shot-image-classification"],
        "required_variables": ["HAS_TRANSFORMERS", "HAS_TORCH", "HAS_TOKENIZERS"],
        "model_inputs": ["image", "text"],
        "result_keys": ["predictions", "pipeline_success", "from_pretrained_success"]
    },
    "speech": {
        "required_methods": ["test_pipeline", "test_from_pretrained", "run_tests"],
        "required_tasks": ["automatic-speech-recognition", "audio-classification"],
        "required_variables": ["HAS_TRANSFORMERS", "HAS_TORCH", "HAS_TOKENIZERS"],
        "model_inputs": ["audio"],
        "result_keys": ["predictions", "pipeline_success", "from_pretrained_success"]
    },
    "multimodal": {
        "required_methods": ["test_pipeline", "test_from_pretrained", "run_tests"],
        "required_tasks": ["image-to-text", "document-question-answering"],
        "required_variables": ["HAS_TRANSFORMERS", "HAS_TORCH", "HAS_TOKENIZERS"],
        "model_inputs": ["image", "text"],
        "result_keys": ["predictions", "pipeline_success", "from_pretrained_success"]
    }
}

def to_valid_identifier(text):
    """Convert hyphenated model names to valid Python identifiers."""
    # Replace hyphens with underscores
    text = text.replace("-", "_")
    # Remove any other invalid characters
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    # Ensure it doesn't start with a number
    if text and text[0].isdigit():
        text = '_' + text
    return text

def get_class_name(model_name):
    """Get proper class name capitalization for a model name."""
    # Special cases for known model names with specific capitalization
    special_cases = {
        "gpt-j": "GPTJ",
        "gpt-neo": "GPTNeo",
        "gpt-neox": "GPTNeoX",
        "gpt2": "GPT2",
        "xlm-roberta": "XLMRoBERTa",
        "wav2vec2-bert": "Wav2Vec2BERT",
        "t5": "T5",
        "mt5": "MT5",
        "vit": "ViT",
        "bert": "BERT",
        "clip": "CLIP",
        "chinese-clip": "ChineseCLIP",
        "vision-text-dual-encoder": "VisionTextDualEncoder"
    }
    
    # Check for special cases first
    if model_name.lower() in special_cases:
        return special_cases[model_name.lower()]
    
    # For other hyphenated names, capitalize each part
    if "-" in model_name:
        return ''.join(part.capitalize() for part in model_name.split('-'))
    
    # Default: just capitalize
    return model_name.capitalize()

def get_upper_case_name(model_name):
    """Generate uppercase constants for registry variables."""
    return to_valid_identifier(model_name).upper()

def get_architecture_type(model_name):
    """Determine architecture type based on model type."""
    model_type_lower = model_name.lower()
    for arch_type, models in ARCHITECTURE_TYPES.items():
        if any(model in model_type_lower for model in models):
            return arch_type
    return "encoder-only"  # Default to encoder-only if unknown

def check_syntax(file_path):
    """Check if a file has valid Python syntax."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Compile the source code 
        compile(content, file_path, 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error on line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def validate_imports_and_variables(content, required_variables):
    """Validate that required imports and variables are present."""
    issues = []
    
    # Check for imports
    if "import transformers" not in content and "from transformers import" not in content:
        issues.append("Missing transformers import")
    
    if "import torch" not in content and "from torch import" not in content:
        issues.append("Missing torch import")
    
    # Check for required variables
    for var in required_variables:
        if var not in content:
            issues.append(f"Missing required variable: {var}")
    
    # Check for mock detection variables
    mock_detection_vars = [
        "using_real_inference = HAS_TRANSFORMERS and HAS_TORCH",
        "using_mocks = not using_real_inference",
        '"test_type": "REAL INFERENCE"'
    ]
    
    for var in mock_detection_vars:
        if var not in content:
            issues.append(f"Missing mock detection: {var}")
    
    return issues

def validate_class_name(content, model_name):
    """Validate that the test class name follows proper capitalization rules."""
    valid_model_id = to_valid_identifier(model_name)
    class_name_prefix = get_class_name(model_name)
    
    # Expected class name pattern
    expected_class_name = f"Test{class_name_prefix}Models"
    
    # Use regex to find actual class name
    class_match = re.search(r'class\s+(Test\w+Models)', content)
    if not class_match:
        return False, f"Could not find test class definition"
    
    actual_class_name = class_match.group(1)
    
    if actual_class_name != expected_class_name:
        return False, f"Incorrect test class name: found '{actual_class_name}', expected '{expected_class_name}'"
    
    return True, None

def validate_model_class_imports(content, model_name, arch_type):
    """Validate that the model class imports follow proper capitalization."""
    # Get expected model class name
    # E.g., "GPTJForCausalLM" for "gpt-j" 
    model_base = model_name.split('-')[0] if '-' in model_name else model_name
    expected_class = None
    
    # Get expected class from mapping
    if model_name in MODEL_CLASS_MAPPING.get(arch_type, {}):
        expected_class = MODEL_CLASS_MAPPING[arch_type][model_name]
    elif model_base in MODEL_CLASS_MAPPING.get(arch_type, {}):
        expected_class = MODEL_CLASS_MAPPING[arch_type][model_base]
    
    if not expected_class:
        return False, f"Unknown model class for {model_name} in {arch_type} architecture"
    
    # Check if the expected class name appears in the content
    if expected_class not in content:
        return False, f"Missing expected model class: {expected_class}"
    
    return True, None

def validate_registry_name(content, model_name):
    """Validate that the registry variable name is properly formatted."""
    upper_case_name = get_upper_case_name(model_name)
    expected_registry = f"{upper_case_name}_MODELS_REGISTRY"
    
    # Check for registry variable
    if expected_registry not in content:
        return False, f"Missing or incorrect registry variable, expected: {expected_registry}"
    
    return True, None

def validate_required_methods(content, required_methods):
    """Validate that required methods are present."""
    missing_methods = []
    
    for method in required_methods:
        pattern = re.compile(r'def\s+' + method + r'\s*\(')
        if not pattern.search(content):
            missing_methods.append(method)
    
    if missing_methods:
        return False, f"Missing required methods: {', '.join(missing_methods)}"
    
    return True, None

def validate_mock_detection(content):
    """Validate proper mock detection implementation."""
    # Check for mock detection in run_tests method
    run_tests_pattern = r'def run_tests\(.*?\):(.*?)return'
    run_tests_match = re.search(run_tests_pattern, content, re.DOTALL)
    
    if not run_tests_match:
        return False, "Could not find run_tests method"
    
    run_tests_body = run_tests_match.group(1)
    
    # Check for mock detection variables
    if "using_real_inference = HAS_TRANSFORMERS and HAS_TORCH" not in run_tests_body:
        return False, "Missing real inference detection in run_tests"
    
    if "using_mocks = not using_real_inference" not in run_tests_body:
        return False, "Missing mock detection in run_tests"
    
    # Check for mock detection in metadata
    metadata_pattern = r'"metadata":\s*{(.*?)}'
    metadata_match = re.search(metadata_pattern, content, re.DOTALL)
    
    if not metadata_match:
        return False, "Could not find metadata dictionary"
    
    metadata_body = metadata_match.group(1)
    
    required_metadata = [
        '"has_transformers": HAS_TRANSFORMERS',
        '"has_torch": HAS_TORCH',
        '"using_real_inference": using_real_inference',
        '"using_mocks": using_mocks',
        '"test_type":'
    ]
    
    for item in required_metadata:
        if item not in metadata_body:
            return False, f"Missing {item} in metadata"
    
    return True, None

def try_model_inference(model_name, file_path):
    """Try running the test file with actual model inference if possible."""
    try:
        # Run the test file with --list-models to check if it works without errors
        result = subprocess.run(
            [sys.executable, file_path, "--list-models"],
            capture_output=True, 
            text=True,
            timeout=30  # Timeout after 30 seconds
        )
        
        if result.returncode != 0:
            return False, f"Test execution failed with code {result.returncode}: {result.stderr}"
        
        # Check if the output contains the model name
        output = result.stdout
        if model_name not in output and to_valid_identifier(model_name) not in output:
            return False, f"Model {model_name} not found in test output"
        
        return True, output
        
    except subprocess.TimeoutExpired:
        return False, "Test execution timed out after 30 seconds"
    except Exception as e:
        return False, f"Error running test: {str(e)}"

def validate_file(file_path, run_inference=False):
    """Validate a test file against architecture-specific rules."""
    # Get the model name from the file name
    model_name = None
    try:
        file_name = os.path.basename(file_path)
        if file_name.startswith("test_hf_"):
            model_with_ext = file_name[8:]  # Remove "test_hf_" prefix
            model_name = os.path.splitext(model_with_ext)[0]  # Remove file extension
            # Convert back to hyphenated format if needed
            model_name = model_name.replace("_", "-")
            
            # Try to identify the model from known hyphenated models
            matching_models = [m for m in KNOWN_HYPHENATED_MODELS if to_valid_identifier(m) == model_name.replace("-", "_")]
            if matching_models:
                model_name = matching_models[0]  # Use the original hyphenated name
    except Exception as e:
        return {
            "file_path": file_path,
            "status": "error",
            "message": f"Error extracting model name: {str(e)}",
            "issues": [str(e)],
            "raw_model_name": os.path.basename(file_path)[8:-3] if os.path.basename(file_path).startswith("test_hf_") else None
        }
    
    if not model_name:
        return {
            "file_path": file_path,
            "status": "error",
            "message": "Could not determine model name from file path",
            "issues": ["Unknown model name"],
            "raw_model_name": os.path.basename(file_path)[8:-3] if os.path.basename(file_path).startswith("test_hf_") else None
        }
    
    # Read the file content
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        return {
            "file_path": file_path,
            "status": "error",
            "message": f"Error reading file: {str(e)}",
            "issues": [str(e)],
            "model_name": model_name
        }
    
    # Determine the architecture type
    arch_type = get_architecture_type(model_name)
    
    # Check syntax first
    syntax_valid, syntax_error = check_syntax(file_path)
    if not syntax_valid:
        return {
            "file_path": file_path,
            "status": "error",
            "message": syntax_error,
            "issues": [syntax_error],
            "model_name": model_name,
            "architecture_type": arch_type
        }
    
    # Prepare validation results
    validation_results = {
        "file_path": file_path,
        "model_name": model_name,
        "architecture_type": arch_type,
        "status": "validating",
        "issues": [],
        "inference_results": None
    }
    
    # Get validation rules for this architecture
    rules = ARCHITECTURE_VALIDATION_RULES.get(arch_type, ARCHITECTURE_VALIDATION_RULES["encoder-only"])
    
    # Run all validations
    # 1. Check imports and variables
    import_issues = validate_imports_and_variables(content, rules["required_variables"])
    if import_issues:
        validation_results["issues"].extend(import_issues)
    
    # 2. Check class name capitalization
    class_valid, class_error = validate_class_name(content, model_name)
    if not class_valid:
        validation_results["issues"].append(class_error)
    
    # 3. Check model class imports
    model_class_valid, model_class_error = validate_model_class_imports(content, model_name, arch_type)
    if not model_class_valid:
        validation_results["issues"].append(model_class_error)
    
    # 4. Check registry name
    registry_valid, registry_error = validate_registry_name(content, model_name)
    if not registry_valid:
        validation_results["issues"].append(registry_error)
    
    # 5. Check required methods
    methods_valid, methods_error = validate_required_methods(content, rules["required_methods"])
    if not methods_valid:
        validation_results["issues"].append(methods_error)
    
    # 6. Check mock detection
    mock_valid, mock_error = validate_mock_detection(content)
    if not mock_valid:
        validation_results["issues"].append(mock_error)
    
    # 7. Run inference test if requested
    if run_inference:
        inference_valid, inference_result = try_model_inference(model_name, file_path)
        validation_results["inference_results"] = {
            "success": inference_valid,
            "message": inference_result if not inference_valid else "Model listed successfully",
            "output": inference_result if inference_valid else None
        }
        
        if not inference_valid:
            validation_results["issues"].append(f"Inference test failed: {inference_result}")
    
    # Set final validation status
    if validation_results["issues"]:
        validation_results["status"] = "warning" if run_inference and validation_results["inference_results"]["success"] else "failed"
        validation_results["message"] = f"Validation completed with {len(validation_results['issues'])} issues"
    else:
        validation_results["status"] = "passed"
        validation_results["message"] = "All validation checks passed"
    
    return validation_results

def generate_actionable_recommendations(validation_results):
    """Generate actionable recommendations based on validation results."""
    recommendations = []
    
    if validation_results["status"] == "passed":
        recommendations.append("✅ No issues found. The test file follows all best practices.")
        return recommendations
    
    # Add specific recommendations based on issues
    for issue in validation_results["issues"]:
        if "Missing required variable" in issue:
            var_name = issue.split(": ")[1]
            recommendations.append(f"📝 Add {var_name} variable to enable proper mock detection")
        
        elif "Missing mock detection" in issue:
            recommendations.append("📝 Add proper mock detection to ensure CI/CD compatibility")
        
        elif "Incorrect test class name" in issue:
            match = re.search(r"found '(.*?)', expected '(.*?)'", issue)
            if match:
                found, expected = match.groups()
                recommendations.append(f"📝 Rename class from {found} to {expected} to follow naming convention")
        
        elif "Missing expected model class" in issue:
            model_class = issue.split(": ")[1]
            recommendations.append(f"📝 Use {model_class} for consistency with HuggingFace's class naming")
        
        elif "Missing or incorrect registry variable" in issue:
            registry = issue.split(", expected: ")[1]
            recommendations.append(f"📝 Use {registry} as the model registry variable name")
        
        elif "Missing required methods" in issue:
            methods = issue.split(": ")[1]
            recommendations.append(f"📝 Implement missing methods: {methods}")
        
        elif "Inference test failed" in issue:
            recommendations.append("🔍 Test the model with the actual HuggingFace dependency to verify functionality")
    
    # Add general recommendations
    if validation_results["status"] == "failed":
        recommendations.append("🔄 Regenerate the test file using the most recent template")
    
    if validation_results.get("inference_results", {}).get("success") is False:
        recommendations.append("📦 Verify that the transformers, torch, and tokenizers packages are installed")
    
    return recommendations

def generate_markdown_report(validation_results_list, file_path=None):
    """Generate a markdown report from validation results."""
    # Get timestamp for the report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"# Hyphenated Model Validation Report\n\n"
    report += f"Generated: {timestamp}\n\n"
    
    # Summary section
    total = len(validation_results_list)
    passed = sum(1 for r in validation_results_list if r["status"] == "passed")
    warnings = sum(1 for r in validation_results_list if r["status"] == "warning")
    failed = sum(1 for r in validation_results_list if r["status"] == "error" or r["status"] == "failed")
    
    report += f"## Summary\n\n"
    report += f"- Total files validated: {total}\n"
    report += f"- Passed: {passed} ({passed/total*100:.1f}%)\n"
    report += f"- Warnings: {warnings} ({warnings/total*100:.1f}%)\n"
    report += f"- Failed: {failed} ({failed/total*100:.1f}%)\n\n"
    
    # Architecture breakdown
    arch_counts = defaultdict(int)
    arch_passed = defaultdict(int)
    
    for result in validation_results_list:
        arch_type = result.get("architecture_type", "unknown")
        arch_counts[arch_type] += 1
        if result["status"] == "passed":
            arch_passed[arch_type] += 1
    
    report += f"## Architecture Breakdown\n\n"
    report += f"| Architecture | Count | Passed | Pass Rate |\n"
    report += f"|-------------|-------|--------|----------|\n"
    
    for arch, count in sorted(arch_counts.items()):
        passed = arch_passed[arch]
        pass_rate = passed / count * 100 if count > 0 else 0
        report += f"| {arch} | {count} | {passed} | {pass_rate:.1f}% |\n"
    
    # Detailed results section
    report += f"\n## Detailed Results\n\n"
    
    for i, result in enumerate(validation_results_list):
        model_name = result.get("model_name", "unknown")
        arch_type = result.get("architecture_type", "unknown")
        status = result.get("status", "unknown").upper()
        issues = result.get("issues", [])
        
        status_icon = "✅" if status == "PASSED" else "⚠️" if status == "WARNING" else "❌"
        
        report += f"### {i+1}. {model_name} ({arch_type}) - {status_icon} {status}\n\n"
        report += f"File: `{os.path.basename(result['file_path'])}`\n\n"
        
        if issues:
            report += f"#### Issues ({len(issues)})\n\n"
            for j, issue in enumerate(issues):
                report += f"{j+1}. {issue}\n"
            report += "\n"
        
        # Add recommendations
        recommendations = generate_actionable_recommendations(result)
        if recommendations:
            report += f"#### Recommendations\n\n"
            for rec in recommendations:
                report += f"- {rec}\n"
            report += "\n"
        
        # Add inference results if available
        if result.get("inference_results"):
            inference = result["inference_results"]
            inf_status = "✅ Passed" if inference["success"] else "❌ Failed"
            report += f"#### Inference Test: {inf_status}\n\n"
            if not inference["success"]:
                report += f"Error: {inference['message']}\n\n"
        
        report += f"---\n\n"
    
    # Appendix with useful commands
    report += f"## Appendix: Useful Commands\n\n"
    report += f"```bash\n"
    report += f"# Regenerate tests for all hyphenated models\n"
    report += f"python integrate_generator_fixes.py --generate-all --output-dir fixed_tests\n\n"
    report += f"# Validate tests with inference\n"
    report += f"python validate_hyphenated_model_solution.py --all --inference --report\n\n"
    report += f"# Run a specific test with real model\n"
    report += f"python fixed_tests/test_hf_gpt_j.py --model gpt-j\n"
    report += f"```\n"
    
    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(report)
        logger.info(f"Report written to {file_path}")
    
    return report

def validate_files(file_patterns, run_inference=False, max_workers=4):
    """Validate multiple files matching patterns using parallel processing."""
    file_paths = []
    
    # Process each file pattern
    for pattern in file_patterns:
        # Handle directory paths
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "test_hf_*.py")
        
        # Find matching files
        matching_files = glob.glob(pattern)
        file_paths.extend(matching_files)
    
    # Remove duplicates
    file_paths = list(set(file_paths))
    
    if not file_paths:
        logger.warning(f"No files found matching patterns: {file_patterns}")
        return []
    
    logger.info(f"Validating {len(file_paths)} files...")
    
    # Use ThreadPoolExecutor for parallel processing
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(validate_file, file_path, run_inference): file_path for file_path in file_paths}
        for future in future_to_file:
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Validated {result['file_path']}: {result['status']}")
            except Exception as e:
                file_path = future_to_file[future]
                logger.error(f"Error validating {file_path}: {str(e)}")
                results.append({
                    "file_path": file_path,
                    "status": "error",
                    "message": str(e),
                    "issues": [str(e)]
                })
    
    return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Validate hyphenated model solution")
    
    # File selection options
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("--files", nargs="+", help="File patterns to validate")
    file_group.add_argument("--all", action="store_true", help="Validate all test files in fixed_tests directory")
    file_group.add_argument("--model", type=str, help="Validate test file for specific model")
    
    # Validation options
    parser.add_argument("--inference", action="store_true", help="Run model inference validation")
    parser.add_argument("--report", action="store_true", help="Generate validation report")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--output-dir", type=str, default="validation_reports", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Determine which files to validate
    if args.all:
        file_patterns = [str(FIXED_TESTS_DIR / "test_hf_*.py")]
    elif args.model:
        model_id = to_valid_identifier(args.model)
        file_patterns = [str(FIXED_TESTS_DIR / f"test_hf_{model_id}.py")]
    else:
        file_patterns = args.files
    
    # Run validation
    results = validate_files(file_patterns, args.inference, args.max_workers)
    
    # Print summary to console
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "passed")
    warnings = sum(1 for r in results if r["status"] == "warning")
    failed = sum(1 for r in results if r["status"] in ["error", "failed"])
    
    print(f"\nValidation Results:")
    print(f"- Total files validated: {total}")
    print(f"- Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"- Warnings: {warnings} ({warnings/total*100:.1f}%)")
    print(f"- Failed: {failed} ({failed/total*100:.1f}%)")
    
    # Generate report if requested
    if args.report or args.all:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(args.output_dir, f"validation_report_{timestamp}.md")
        
        # Save JSON results
        json_file = os.path.join(args.output_dir, f"validation_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = []
            for r in results:
                # Copy the result and handle potential non-serializable objects
                json_result = r.copy()
                json_results.append(json_result)
            
            json.dump(json_results, f, indent=2)
        
        # Generate markdown report
        generate_markdown_report(results, report_file)
        print(f"\nReport saved to: {report_file}")
        print(f"Detailed results saved to: {json_file}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())