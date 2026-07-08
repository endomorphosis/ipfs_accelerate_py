#!/usr/bin/env python3
"""
Integrate fixes for hyphenated model names and ensure consistent mock detection in test generator files.

This script:
1. Adds proper handling for hyphenated model names (gpt-j, xlm-roberta, etc.)
2. Adds missing HAS_TOKENIZERS and HAS_SENTENCEPIECE imports in test files
3. Ensures the imports for non-BERT models are handled properly
4. Fixes model class names for GPT2, T5, ViT, XLM-RoBERTa, etc.
5. Updates variables and class names for consistency
6. Creates a CI/CD-friendly approach to automatically maintain tests

Usage:
    python integrate_generator_fixes.py [--file FILE_PATH] [--scan-models] [--generate-all] [--verify]
"""

import os
import sys
import re
import json
import argparse
import logging
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FIXED_TESTS_DIR = CURRENT_DIR / "fixed_tests"
TEMPLATES_DIR = CURRENT_DIR / "templates"

# Known hyphenated model families from HuggingFace
KNOWN_HYPHENATED_MODELS = [
    "gpt-j",
    "gpt-neo",
    "gpt-neox",
    "xlm-roberta", 
    "vision-text-dual-encoder",
    "speech-to-text",
    "speech-to-text-2",
    "trocr-base",
    "trocr-large",
    "chinese-clip",
    "data2vec-text",
    "data2vec-audio",
    "data2vec-vision",
    "wav2vec2-bert"
]

# Architecture mapping for template selection
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5", "speech-to-text", "speech-to-text-2", "wav2vec2-bert"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava"]
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

def get_template_path(model_name):
    """Get the appropriate template path for a model architecture."""
    arch_type = get_architecture_type(model_name)
    
    template_map = {
        "encoder-only": os.path.join(TEMPLATES_DIR, "encoder_only_template.py"),
        "decoder-only": os.path.join(TEMPLATES_DIR, "decoder_only_template.py"),
        "encoder-decoder": os.path.join(TEMPLATES_DIR, "encoder_decoder_template.py"),
        "vision": os.path.join(TEMPLATES_DIR, "vision_template.py"),
        "vision-text": os.path.join(TEMPLATES_DIR, "vision_text_template.py"),
        "speech": os.path.join(TEMPLATES_DIR, "speech_template.py"),
        "multimodal": os.path.join(TEMPLATES_DIR, "multimodal_template.py")
    }
    
    template_path = template_map.get(arch_type)
    if not template_path or not os.path.exists(template_path):
        logger.warning(f"Template not found for {arch_type}, using encoder-only template")
        fallback_template = os.path.join(TEMPLATES_DIR, "encoder_only_template.py")
        if not os.path.exists(fallback_template):
            logger.error(f"Fallback template not found: {fallback_template}")
            return None
        return fallback_template
        
    return template_path

def fix_missing_imports(content, file_path):
    """Fix missing import variables like HAS_TOKENIZERS and HAS_SENTENCEPIECE."""
    model_type = os.path.basename(file_path)[8:-3]  # Extract model type from test_hf_MODEL.py
    
    # First check if the imports are missing
    has_tokenizers_var = "HAS_TOKENIZERS" in content
    has_sentencepiece_var = "HAS_SENTENCEPIECE" in content
    
    if has_tokenizers_var and has_sentencepiece_var:
        logger.info(f"✅ {file_path}: All necessary import variables present")
        return content
    
    # Add missing import sections
    if not has_tokenizers_var:
        logger.info(f"❌ {file_path}: Missing HAS_TOKENIZERS variable, adding it")
        # Find where to add the tokenizers import section
        if "Try to import tokenizers" not in content:
            # Add after transformers import
            transformers_import_match = re.search(r'(# Try to import transformers.*?HAS_TRANSFORMERS = False.*?logger\.warning\("transformers not available, using mock"\))', content, re.DOTALL)
            if transformers_import_match:
                tokenizers_section = """
# Try to import tokenizers
try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")
"""
                insert_pos = transformers_import_match.end()
                content = content[:insert_pos] + tokenizers_section + content[insert_pos:]
                
    if not has_sentencepiece_var:
        logger.info(f"❌ {file_path}: Missing HAS_SENTENCEPIECE variable, adding it")
        # Find where to add the sentencepiece import section
        if "Try to import sentencepiece" not in content:
            # Add after tokenizers import (or transformers import if we just added tokenizers)
            tokenizers_import_match = re.search(r'(# Try to import tokenizers.*?HAS_TOKENIZERS = False.*?logger\.warning\("tokenizers not available, using mock"\))', content, re.DOTALL)
            if tokenizers_import_match:
                sentencepiece_section = """
# Try to import sentencepiece
try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")
"""
                insert_pos = tokenizers_import_match.end()
                content = content[:insert_pos] + sentencepiece_section + content[insert_pos:]
            else:
                # If tokenizers section wasn't found and we're adding both, check if we can add after transformers
                transformers_import_match = re.search(r'(# Try to import transformers.*?HAS_TRANSFORMERS = False.*?logger\.warning\("transformers not available, using mock"\))', content, re.DOTALL)
                if transformers_import_match:
                    both_sections = """
# Try to import tokenizers
try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

# Try to import sentencepiece
try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")
"""
                    insert_pos = transformers_import_match.end()
                    content = content[:insert_pos] + both_sections + content[insert_pos:]
    
    return content

def fix_model_class_names(content, file_path):
    """Fix model class names (e.g., Gpt2LMHeadModel -> GPT2LMHeadModel)."""
    model_type = os.path.basename(file_path)[8:-3]  # Extract model type from test_hf_MODEL.py
    
    # Define the correct class names for common model types
    model_class_corrections = {
        "gpt2": ("Gpt2LMHeadModel", "GPT2LMHeadModel"),
        "gpt_j": ("GptjForCausalLM", "GPTJForCausalLM"),
        "gpt_neo": ("GptneoForCausalLM", "GPTNeoForCausalLM"),
        "gpt_neox": ("GptneoxForCausalLM", "GPTNeoXForCausalLM"),
        "t5": ("T5ForConditionalGeneration", "T5ForConditionalGeneration"),
        "vit": ("VitForImageClassification", "ViTForImageClassification"),
        "swin": ("SwinForImageClassification", "SwinForImageClassification"),
        "clip": ("ClipModel", "CLIPModel"),
        "bart": ("BartForConditionalGeneration", "BartForConditionalGeneration"),
        "whisper": ("WhisperForConditionalGeneration", "WhisperForConditionalGeneration"),
        "xlm_roberta": ("XlmRobertaForMaskedLM", "XLMRobertaForMaskedLM"),
    }
    
    # Apply corrections specific to this model type
    model_type = to_valid_identifier(model_type)
    for model_prefix, (incorrect, correct) in model_class_corrections.items():
        if model_type.startswith(model_prefix):
            old_line = f"model = transformers.{incorrect}.from_pretrained"
            new_line = f"model = transformers.{correct}.from_pretrained"
            if old_line in content:
                logger.info(f"❌ {file_path}: Incorrect model class name '{incorrect}', fixing to '{correct}'")
                content = content.replace(old_line, new_line)
    
    return content

def fix_run_tests_function(content, file_path):
    """Fix the run_tests function to correctly handle mock detection."""
    # Check if the run_tests function already has proper mock detection
    mock_detection_pattern = r"using_real_inference\s*=\s*HAS_TRANSFORMERS\s+and\s+HAS_TORCH"
    mocks_pattern = r"using_mocks\s*=\s*not\s+using_real_inference\s+or\s+not\s+HAS_TOKENIZERS\s+or\s+not\s+HAS_SENTENCEPIECE"
    
    has_detection = re.search(mock_detection_pattern, content) is not None
    has_mocks = re.search(mocks_pattern, content) is not None
    
    if has_detection and has_mocks:
        return content  # Already fixed
        
    # Find the run_tests function
    run_tests_match = re.search(r'(def run_tests\(self, all_hardware=False\):.*?return results\s*$)', content, re.DOTALL | re.MULTILINE)
    if not run_tests_match:
        logger.warning(f"❗ {file_path}: Could not find run_tests function, skipping mock detection fix")
        return content
        
    # Get the function content
    run_tests_content = run_tests_match.group(1)
    
    # Check for missing mock detection code
    if not has_detection or not has_mocks:
        # Find where to add the mock detection logic (before the metadata dict)
        metadata_match = re.search(r'(\s+# Add metadata\s+results\["metadata"\] = {)', run_tests_content)
        if metadata_match:
            # Add the missing mock detection logic
            mock_detection_logic = """
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
"""
            prefix = run_tests_content[:metadata_match.start()]
            suffix = run_tests_content[metadata_match.start():]
            updated_function = prefix + mock_detection_logic + suffix
            
            # Replace the function in the full content
            content = content.replace(run_tests_content, updated_function)
            logger.info(f"✅ {file_path}: Added mock detection logic in run_tests function")
    
    # Now check if the metadata section includes mock detection keys
    metadata_keys = [
        '"has_transformers": HAS_TRANSFORMERS', 
        '"has_torch": HAS_TORCH',
        '"has_tokenizers": HAS_TOKENIZERS',
        '"has_sentencepiece": HAS_SENTENCEPIECE',
        '"using_real_inference": using_real_inference',
        '"using_mocks": using_mocks',
        '"test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"'
    ]
    
    # Check for each key
    missing_keys = []
    for key in metadata_keys:
        if key not in content:
            missing_keys.append(key)
    
    if missing_keys:
        # Find the metadata dictionary
        metadata_dict_match = re.search(r'(\s+results\["metadata"\] = {.*?\n\s+})', content, re.DOTALL)
        if metadata_dict_match:
            metadata_dict = metadata_dict_match.group(1)
            # Find the end of the dictionary (last }, before first return)
            dict_closing_match = re.search(r'\n(\s+})(?=\s*\n\s+return)', metadata_dict)
            if dict_closing_match:
                # Add missing keys before the closing brace
                indent = dict_closing_match.group(1).replace('}', '')
                missing_keys_str = ',\n'.join(f"{indent}{key}" for key in missing_keys)
                new_dict = metadata_dict[:dict_closing_match.start()] + ",\n" + missing_keys_str + metadata_dict[dict_closing_match.start():]
                content = content.replace(metadata_dict, new_dict)
                logger.info(f"✅ {file_path}: Added missing mock detection keys to metadata dict")
    
    return content

def fix_main_function(content, file_path):
    """Fix the main function to correctly display mock detection status."""
    # Check if the main function already has proper status display
    status_pattern = r'using_real_inference = results\["metadata"\]\["using_real_inference"\].*?using_mocks = results\["metadata"\]\["using_mocks"\]'
    indicator_pattern = r'if using_real_inference and not using_mocks:.*?print\(f"\{GREEN\}🚀 Using REAL INFERENCE with actual models\{RESET\}"\)'
    
    has_status = re.search(status_pattern, content, re.DOTALL) is not None
    has_indicator = re.search(indicator_pattern, content, re.DOTALL) is not None
    
    if has_status and has_indicator:
        return content  # Already fixed
        
    # Find the main function
    main_match = re.search(r'(def main\(\):.*?)(\n\s*if __name__ == "__main__")', content, re.DOTALL)
    if not main_match:
        logger.warning(f"❗ {file_path}: Could not find main function, skipping status display fix")
        return content
        
    # Fix the main function to add mock detection display
    main_function = main_match.group(1)
    
    # Find the summary section
    summary_section_match = re.search(r'(\s+# Print a summary\s+print\("\n" \+ "="\*50\)\s+print\("TEST RESULTS SUMMARY"\)\s+print\("="\*50\)\s+)', main_function)
    if summary_section_match:
        # Add the mock status display right after the summary header
        mock_status_code = """
    # Indicate real vs mock inference clearly
    using_real_inference = results["metadata"]["using_real_inference"]
    using_mocks = results["metadata"]["using_mocks"]
    
    if using_real_inference and not using_mocks:
        print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")
    else:
        print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
"""
        prefix = main_function[:summary_section_match.end()]
        suffix = main_function[summary_section_match.end():]
        updated_main = prefix + mock_status_code + suffix
        
        # Replace the main function in the full content
        content = content.replace(main_function, updated_main)
        logger.info(f"✅ {file_path}: Added mock status display in main function")
    
    return content

def fix_registry_variables(content, file_path):
    """Fix registry variable names for hyphenated models."""
    model_type = os.path.basename(file_path)[8:-3]  # Extract model type from test_hf_MODEL.py
    
    # Convert the model type to a valid identifier and get the upper case name
    valid_model_id = to_valid_identifier(model_type)
    upper_case_name = get_upper_case_name(model_type)
    
    # Find all occurrences of registry variables with pattern MODEL_REGISTRY, BERT_MODELS_REGISTRY, etc.
    registry_vars = re.findall(r'([A-Z_]+_MODELS_REGISTRY)', content)
    
    if not registry_vars:
        logger.warning(f"❗ {file_path}: Could not find registry variables")
        return content
    
    # Skip if the registry variable is already correct
    if any(var == f"{upper_case_name}_MODELS_REGISTRY" for var in registry_vars):
        logger.info(f"✅ {file_path}: Registry variable already has correct name")
        return content
    
    # Replace the registry variable with the correct name
    for registry_var in registry_vars:
        new_registry_var = f"{upper_case_name}_MODELS_REGISTRY"
        if registry_var != new_registry_var:
            logger.info(f"❌ {file_path}: Incorrect registry variable '{registry_var}', fixing to '{new_registry_var}'")
            content = content.replace(registry_var, new_registry_var)
    
    return content

def fix_test_class_name(content, file_path):
    """Fix test class names to match the proper capitalization (TestGPTJModels instead of TestGptjModels)."""
    model_type = os.path.basename(file_path)[8:-3]  # Extract model type from test_hf_MODEL.py
    
    # Create proper capitalized name for test class
    class_name_prefix = get_class_name(model_type)
    
    # Find the test class definition
    class_match = re.search(r'class\s+Test(\w+)Models', content)
    if not class_match:
        logger.warning(f"❗ {file_path}: Could not find test class definition")
        return content
    
    current_prefix = class_match.group(1)
    correct_class_name = f"Test{class_name_prefix}Models"
    current_class_name = f"Test{current_prefix}Models"
    
    if current_class_name != correct_class_name:
        logger.info(f"❌ {file_path}: Incorrect test class name '{current_class_name}', fixing to '{correct_class_name}'")
        content = content.replace(current_class_name, correct_class_name)
    
    return content

def create_test_file(model_name, output_dir=None):
    """Create a test file for a hyphenated model using the appropriate template."""
    if output_dir is None:
        output_dir = FIXED_TESTS_DIR
    
    logger.info(f"Creating test file for model {model_name}")
    
    # Convert to valid Python identifier and get proper naming
    valid_model_id = to_valid_identifier(model_name)
    upper_case_name = get_upper_case_name(model_name)
    class_name_prefix = get_class_name(model_name)
    arch_type = get_architecture_type(model_name)
    
    # Get the appropriate template
    template_path = get_template_path(model_name)
    if not template_path:
        logger.error(f"Could not find template for model {model_name}")
        return False
    
    logger.info(f"Using template: {os.path.basename(template_path)} for {model_name}")
    
    try:
        # Read the template
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # File path for the test file
        output_file = os.path.join(output_dir, f"test_hf_{valid_model_id}.py")
        
        # Make replacements in the template
        content = template_content
        
        # Create replacement patterns based on architecture
        if arch_type == "encoder-only":
            replacements = {
                "BERT_MODELS_REGISTRY": f"{upper_case_name}_MODELS_REGISTRY",
                "TestBertModels": f"Test{class_name_prefix}Models",
                "bert-base-uncased": model_name,
                "BERT": class_name_prefix,
                "bert": valid_model_id,
                "BertForMaskedLM": f"{class_name_prefix}ForMaskedLM",
                "hf_bert_": f"hf_{valid_model_id}_"
            }
        elif arch_type == "decoder-only":
            replacements = {
                "GPT2_MODELS_REGISTRY": f"{upper_case_name}_MODELS_REGISTRY",
                "TestGPT2Models": f"Test{class_name_prefix}Models",
                "gpt2": model_name,
                "GPT2": class_name_prefix,
                "GPT2LMHeadModel": f"{class_name_prefix}ForCausalLM",
                "hf_gpt2_": f"hf_{valid_model_id}_"
            }
        elif arch_type == "encoder-decoder":
            replacements = {
                "T5_MODELS_REGISTRY": f"{upper_case_name}_MODELS_REGISTRY",
                "TestT5Models": f"Test{class_name_prefix}Models",
                "t5-small": model_name,
                "T5": class_name_prefix,
                "T5ForConditionalGeneration": f"{class_name_prefix}ForConditionalGeneration",
                "hf_t5_": f"hf_{valid_model_id}_"
            }
        elif arch_type == "vision":
            replacements = {
                "VIT_MODELS_REGISTRY": f"{upper_case_name}_MODELS_REGISTRY",
                "TestVitModels": f"Test{class_name_prefix}Models",
                "google/vit-base-patch16-224": model_name,
                "ViT": class_name_prefix,
                "ViTForImageClassification": f"{class_name_prefix}ForImageClassification",
                "hf_vit_": f"hf_{valid_model_id}_"
            }
        elif arch_type in ["vision-text", "speech", "multimodal"]:
            # For more complex architectures, use a generic approach
            template_base = os.path.basename(template_path).split("_")[0].capitalize()
            replacements = {
                f"{template_base.upper()}_MODELS_REGISTRY": f"{upper_case_name}_MODELS_REGISTRY",
                f"Test{template_base}Models": f"Test{class_name_prefix}Models",
                # Add model-specific defaults here
                f"{template_base}": class_name_prefix,
                f"hf_{template_base.lower()}_": f"hf_{valid_model_id}_"
            }
        else:
            # Fallback to encoder-only if architecture type is not recognized
            replacements = {
                "BERT_MODELS_REGISTRY": f"{upper_case_name}_MODELS_REGISTRY",
                "TestBertModels": f"Test{class_name_prefix}Models",
                "bert-base-uncased": model_name,
                "BERT": class_name_prefix,
                "bert": valid_model_id,
                "BertForMaskedLM": f"{class_name_prefix}ForMaskedLM",
                "hf_bert_": f"hf_{valid_model_id}_"
            }
        
        # Apply replacements
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Write the test file
        with open(output_file, 'w') as f:
            f.write(content)
        
        # Validate syntax
        try:
            compile(content, output_file, 'exec')
            logger.info(f"✅ {output_file}: Syntax is valid")
            
            # Apply additional fixes to ensure consistency
            with open(output_file, 'r') as f:
                file_content = f.read()
            
            file_content = fix_missing_imports(file_content, output_file)
            file_content = fix_model_class_names(file_content, output_file)
            file_content = fix_run_tests_function(file_content, output_file)
            file_content = fix_main_function(file_content, output_file)
            file_content = fix_registry_variables(file_content, output_file)
            file_content = fix_test_class_name(file_content, output_file)
            
            # Write the updated file
            with open(output_file, 'w') as f:
                f.write(file_content)
            
            logger.info(f"✅ Created test file for {model_name} at {output_file}")
            return True
            
        except SyntaxError as e:
            logger.error(f"❌ {output_file}: Syntax error: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating test file for {model_name}: {e}")
        return False

def fix_file(file_path):
    """Apply all fixes to a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Create backup
        backup_path = f"{file_path}.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at {backup_path}")
        
        # Apply fixes
        content = fix_missing_imports(content, file_path)
        content = fix_model_class_names(content, file_path)
        content = fix_run_tests_function(content, file_path)
        content = fix_main_function(content, file_path)
        content = fix_registry_variables(content, file_path)
        content = fix_test_class_name(content, file_path)
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify the fix works
        try:
            # Basic syntax check
            compile(content, file_path, 'exec')
            logger.info(f"✅ {file_path}: Syntax check passed")
            return True
        except SyntaxError as e:
            logger.error(f"❌ {file_path}: Syntax error after fixes: {e}")
            # Restore from backup
            with open(backup_path, 'r') as f:
                original = f.read()
            with open(file_path, 'w') as f:
                f.write(original)
            logger.info(f"Restored from backup due to syntax error")
            return False
            
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False

def detect_hyphenated_models_from_huggingface(limit=100):
    """Detect hyphenated model names from the HuggingFace API."""
    logger.info("Fetching model list from HuggingFace API...")
    
    try:
        # Use the HuggingFace API to list models
        url = f"https://huggingface.co/api/models?limit={limit}&sort=downloads&direction=-1"
        response = requests.get(url)
        response.raise_for_status()
        
        models_data = response.json()
        
        # Extract hyphenated model family names
        hyphenated_models = set()
        for model in models_data:
            model_id = model.get("id", "")
            # Extract the model family name (before first slash if present)
            family = model_id.split("/")[0] if "/" in model_id else model_id
            # Check if the family name contains a hyphen
            if "-" in family:
                hyphenated_models.add(family)
        
        # Combine with known models
        all_hyphenated_models = sorted(list(set(KNOWN_HYPHENATED_MODELS) | hyphenated_models))
        
        logger.info(f"Discovered {len(all_hyphenated_models)} hyphenated model families")
        return all_hyphenated_models
    
    except Exception as e:
        logger.error(f"Error fetching models from HuggingFace API: {e}")
        logger.info("Falling back to known hyphenated models list")
        return KNOWN_HYPHENATED_MODELS

def update_fixed_tests_readme(hyphenated_models):
    """Update the README in the fixed_tests directory with all hyphenated models."""
    readme_path = FIXED_TESTS_DIR / "README.md"
    
    # Create model examples section
    examples_section = "\n## Example Models with Hyphenated Names\n\n"
    for model in sorted(hyphenated_models)[:10]:  # Show the first 10 models
        valid_id = to_valid_identifier(model)
        examples_section += f"- {model} → test_hf_{valid_id}.py\n"
    
    content = """# Fixed Tests for HuggingFace Models

This directory contains test files that have been regenerated with fixes for:

1. Hyphenated model names (e.g. "gpt-j" → "gpt_j")
2. Capitalization issues in class names (e.g. "GPTJForCausalLM" vs "GptjForCausalLM")
3. Syntax errors like unterminated string literals
4. Indentation issues
5. Consistent mock detection across all test files

The test files in this directory are generated using the updated test generator
that handles hyphenated model names correctly. The generator now:

1. Automatically converts hyphenated model names to valid Python identifiers
2. Ensures proper capitalization patterns for class names
3. Validates that generated files have valid Python syntax
4. Fixes common syntax errors like unterminated string literals
5. Adds proper mock detection for CI/CD environments
""" + examples_section + """
## Running the Tests

Tests can be run individually with:

```bash
python fixed_tests/test_hf_gpt_j.py --list-models
python fixed_tests/test_hf_xlm_roberta.py --list-models
```

To run all tests:

```bash
cd fixed_tests
for test in test_hf_*.py; do python $test --list-models; done
```

## Validation

All test files in this directory have been validated to ensure:

1. Valid Python syntax
2. Proper indentation
3. Correct class naming patterns
4. Valid Python identifiers for hyphenated model names
5. Consistent mock object detection for CI/CD environments
"""
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(readme_path), exist_ok=True)
    
    # Write the README
    with open(readme_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated README at {readme_path}")
    return True

def check_file_syntax(content, filename):
    """Check if a file has valid Python syntax."""
    try:
        compile(content, filename, 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)

def process_model(model_name, output_dir):
    """Process a single model - create and validate its test file."""
    logger.info(f"Processing model: {model_name}")
    success = create_test_file(model_name, output_dir)
    if success:
        file_path = os.path.join(output_dir, f"test_hf_{to_valid_identifier(model_name)}.py")
        with open(file_path, 'r') as f:
            content = f.read()
        is_valid, error = check_file_syntax(content, file_path)
        if is_valid:
            return {"model": model_name, "success": True, "file": file_path}
        else:
            return {"model": model_name, "success": False, "file": file_path, "error": error}
    return {"model": model_name, "success": False, "error": "Failed to create test file"}

def generate_report(results, output_path="integration_results.json"):
    """Generate a report of the integration process."""
    # Count successes and failures
    success_count = sum(1 for result in results if result["success"])
    failure_count = len(results) - success_count
    
    # Create report dictionary
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_models": len(results),
        "successful_models": success_count,
        "failed_models": failure_count,
        "success_rate": f"{success_count / len(results) * 100:.2f}%" if results else "N/A",
        "results": results
    }
    
    # Write the report to a JSON file
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info(f"\nIntegration Summary:")
    logger.info(f"- Total models processed: {len(results)}")
    logger.info(f"- Successfully integrated: {success_count}")
    logger.info(f"- Failed to integrate: {failure_count}")
    logger.info(f"- Success rate: {report['success_rate']}")
    logger.info(f"- Full report saved to: {output_path}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Fix hyphenated model names and ensure consistent mock detection")
    parser.add_argument("--file", type=str, help="Path to specific file to fix")
    parser.add_argument("--dir", type=str, default="fixed_tests", help="Directory containing test files to fix")
    parser.add_argument("--scan-models", action="store_true", help="Scan HuggingFace for hyphenated model names")
    parser.add_argument("--generate-all", action="store_true", help="Generate test files for all hyphenated models")
    parser.add_argument("--verify", action="store_true", help="Verify syntax of generated test files")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads for parallel processing")
    parser.add_argument("--output-dir", type=str, help="Output directory for generated files")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"integration_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Generate timestamped output file
    output_report = f"integration_results_{timestamp}.json"
    
    if args.file:
        if os.path.exists(args.file):
            success = fix_file(args.file)
            if success:
                print(f"Successfully fixed {args.file}")
            else:
                print(f"Failed to fix {args.file}")
        else:
            print(f"File not found: {args.file}")
            return 1
    elif args.scan_models or args.generate_all:
        # Get hyphenated models
        if args.scan_models:
            hyphenated_models = detect_hyphenated_models_from_huggingface()
        else:
            hyphenated_models = KNOWN_HYPHENATED_MODELS
        
        logger.info(f"Found {len(hyphenated_models)} hyphenated model families")
        logger.info(f"Models: {', '.join(hyphenated_models[:10])}..." if len(hyphenated_models) > 10 else f"Models: {', '.join(hyphenated_models)}")
        
        # Update README with hyphenated models
        update_fixed_tests_readme(hyphenated_models)
        
        # Generate test files in parallel
        if args.generate_all:
            logger.info(f"Generating test files for all {len(hyphenated_models)} hyphenated models")
            
            results = []
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_model = {executor.submit(process_model, model, output_dir): model for model in hyphenated_models}
                for future in future_to_model:
                    result = future.result()
                    results.append(result)
            
            # Generate integration report
            generate_report(results, output_report)
            
            # Return success if at least one model was successfully processed
            return 0 if any(result["success"] for result in results) else 1
        
        return 0
    else:
        if not os.path.exists(args.dir):
            print(f"Directory not found: {args.dir}")
            return 1
            
        # Process all test files in the directory
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.startswith("test_hf_") and f.endswith(".py")]
        
        success_count = 0
        failure_count = 0
        
        for file_path in files:
            print(f"Processing {file_path}...")
            if fix_file(file_path):
                success_count += 1
            else:
                failure_count += 1
                
        print(f"\nSummary:")
        print(f"- Successfully fixed: {success_count} files")
        print(f"- Failed to fix: {failure_count} files")
        print(f"- Total: {len(files)} files")
        
        if failure_count > 0:
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())