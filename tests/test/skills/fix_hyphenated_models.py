#!/usr/bin/env python3
"""
Fix HuggingFace test files with hyphenated model names.

This script addresses two main issues with hyphenated model names:
1. Python identifiers cannot contain hyphens (gpt-j -> gpt_j)
2. Class names need proper capitalization (gpt-j -> GPTJ)

Usage:
    python fix_hyphenated_models.py [--model MODEL_NAME] [--all-hyphenated] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import re
import logging
import argparse
import traceback
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple, Set

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = CURRENT_DIR / "fixed_templates"
OUTPUT_DIR = CURRENT_DIR / "fixed_tests"

# Maps for hyphenated model names
# Format: 'original-name': ('valid_identifier', 'ClassName', 'CLASS_NAME')
HYPHENATED_MODEL_MAPS = {
    'gpt-j': ('gpt_j', 'GPTJ', 'GPT_J'),
    'gpt-neo': ('gpt_neo', 'GPTNeo', 'GPT_NEO'),
    'gpt-neox': ('gpt_neox', 'GPTNeoX', 'GPT_NEOX'),
    'xlm-roberta': ('xlm_roberta', 'XLMRoBERTa', 'XLM_ROBERTA'),
    'vision-text-dual-encoder': ('vision_text_dual_encoder', 'VisionTextDualEncoder', 'VISION_TEXT_DUAL_ENCODER'),
    't5-base': ('t5_base', 'T5Base', 'T5_BASE'),
    'wav2vec2-base': ('wav2vec2_base', 'Wav2Vec2Base', 'WAV2VEC2_BASE'),
}

# Architecture types for model mapping
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava"]
}

# Class name capitalization fixes
CLASS_NAME_FIXES = {
    "VitForImageClassification": "ViTForImageClassification",
    "SwinForImageClassification": "SwinForImageClassification",
    "DeitForImageClassification": "DeiTForImageClassification",
    "BeitForImageClassification": "BEiTForImageClassification",
    "ConvnextForImageClassification": "ConvNextForImageClassification",
    "Gpt2LMHeadModel": "GPT2LMHeadModel",
    "GptjForCausalLM": "GPTJForCausalLM",
    "GptneoForCausalLM": "GPTNeoForCausalLM",
    "GptneoxForCausalLM": "GPTNeoXForCausalLM",
    "XlmRobertaForMaskedLM": "XLMRobertaForMaskedLM",
    "XlmRobertaModel": "XLMRobertaModel"
}

def to_valid_identifier(text: str) -> str:
    """Convert a hyphenated model name to a valid Python identifier."""
    # Replace hyphens with underscores
    text = text.replace("-", "_")
    # Remove any other invalid characters
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    # Ensure it doesn't start with a number
    if text and text[0].isdigit():
        text = '_' + text
    return text

def get_class_name_capitalization(model_name: str) -> str:
    """Get proper class name capitalization for a model name."""
    # Check if we have a predefined mapping
    if model_name in HYPHENATED_MODEL_MAPS:
        return HYPHENATED_MODEL_MAPS[model_name][1]
    
    # Otherwise, generate a capitalization based on rules:
    # 1. Split by hyphens
    # 2. Capitalize each part
    # 3. Join without hyphens
    parts = model_name.split("-")
    
    # Special case handling for known patterns
    special_cases = {
        "gpt": "GPT",
        "xlm": "XLM",
        "t5": "T5",
        "bert": "BERT",
        "roberta": "RoBERTa",
        "wav2vec2": "Wav2Vec2",
        "neox": "NeoX",
        "neo": "Neo"
    }
    
    capitalized_parts = []
    for part in parts:
        if part.lower() in special_cases:
            capitalized_parts.append(special_cases[part.lower()])
        else:
            capitalized_parts.append(part.capitalize())
    
    return "".join(capitalized_parts)

def get_upper_case_name(model_name: str) -> str:
    """Get upper case name for a model name (for constants)."""
    # Check if we have a predefined mapping
    if model_name in HYPHENATED_MODEL_MAPS:
        return HYPHENATED_MODEL_MAPS[model_name][2]
    
    # Otherwise, replace hyphens with underscores and uppercase
    return model_name.replace("-", "_").upper()

def get_architecture_type(model_name: str) -> str:
    """Determine the architecture type for a model name."""
    model_name_lower = model_name.lower()
    
    for arch_type, models in ARCHITECTURE_TYPES.items():
        for model in models:
            if model_name_lower.startswith(model):
                return arch_type
    
    # Default to encoder-only if no match found
    logger.warning(f"Could not determine architecture type for {model_name}, defaulting to encoder-only")
    return "encoder-only"

def get_template_path(model_name: str) -> str:
    """Get the template path for a specific model architecture."""
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
        return os.path.join(TEMPLATES_DIR, "encoder_only_template.py")
        
    return template_path

def check_file_syntax(content: str, filename: str = "<string>") -> Tuple[bool, Optional[str]]:
    """Check if a Python file has valid syntax."""
    try:
        compile(content, filename, 'exec')
        return True, None
    except SyntaxError as e:
        error_message = f"Syntax error on line {e.lineno}: {e.msg}"
        if hasattr(e, 'text') and e.text:
            error_message += f"\n{e.text}"
            if hasattr(e, 'offset') and e.offset:
                error_message += "\n" + " " * (e.offset - 1) + "^"
        return False, error_message

def fix_registry_references(content: str, original_name: str) -> str:
    """Fix registry variable references for hyphenated model names."""
    valid_id = to_valid_identifier(original_name)
    upper_case = get_upper_case_name(original_name)
    
    # Fix model registry references
    content = re.sub(
        rf"{original_name.upper()}_MODELS_REGISTRY", 
        f"{upper_case}_MODELS_REGISTRY", 
        content
    )
    
    # Fix lowercase references
    content = re.sub(
        rf"{original_name.lower()}\.", 
        f"{valid_id}.", 
        content
    )
    
    # Fix lowercase references in strings
    content = re.sub(
        rf'hf_{original_name.lower()}_', 
        f'hf_{valid_id}_', 
        content
    )
    
    return content

def fix_class_declarations(content: str, original_name: str) -> str:
    """Fix class declarations for hyphenated model names."""
    class_name = get_class_name_capitalization(original_name)
    
    # Create the test class name pattern (e.g., TestGPTJModels)
    test_class_pattern = rf'class Test{original_name.title().replace("-", "")}Models'
    test_class_replacement = f'class Test{class_name}Models'
    
    # Fix test class declarations
    content = re.sub(test_class_pattern, test_class_replacement, content)
    
    # Fix model class references in from_pretrained
    for model_cls, fixed_cls in CLASS_NAME_FIXES.items():
        if model_cls.lower().startswith(original_name.lower().replace("-", "")):
            pattern = rf'transformers\.{model_cls}'
            replacement = f'transformers.{fixed_cls}'
            content = re.sub(pattern, replacement, content)
    
    return content

def fix_method_references(content: str, original_name: str) -> str:
    """Fix method references for hyphenated model names."""
    valid_id = to_valid_identifier(original_name)
    
    # Fix method and function names
    content = re.sub(
        rf'test_{original_name.lower()}', 
        f'test_{valid_id}', 
        content
    )
    
    # Fix variable references
    content = re.sub(
        rf'{original_name.lower()}_tester', 
        f'{valid_id}_tester', 
        content
    )
    
    return content

def fix_template_syntax_errors(content: str) -> str:
    """Fix common syntax errors in templates."""
    # Fix unterminated string literals
    content = re.sub(r'print\(\"\n', 'print(f"\n', content)
    content = re.sub(r'print\(\"\r\n', 'print(f"\r\n', content)
    
    # Fix triple quote issues
    content = re.sub(r'\"\"\"\"', '"""', content)
    
    # Fix escaped characters in string literals
    content = re.sub(r'\\([^tnrb\'\"\\])', r'\\\\\1', content)
    
    return content

def create_test_file(model_name: str, output_dir: Path) -> Tuple[bool, Optional[str]]:
    """Create a test file for a model with a hyphenated name."""
    try:
        # Ensure the model name is hyphenated
        if "-" not in model_name:
            logger.warning(f"{model_name} is not a hyphenated model name, skipping")
            return False, f"{model_name} is not a hyphenated model name"
        
        # Get valid identifier and class name
        valid_id = to_valid_identifier(model_name)
        class_name = get_class_name_capitalization(model_name)
        upper_name = get_upper_case_name(model_name)
        
        logger.info(f"Processing {model_name} -> id: {valid_id}, class: {class_name}, upper: {upper_name}")
        
        # Get appropriate template
        template_path = get_template_path(model_name)
        logger.info(f"Using template: {os.path.basename(template_path)}")
        
        # Read template
        with open(template_path, 'r') as f:
            template_content = f.read()
            
        # Fix any syntax errors in the template
        template_content = fix_template_syntax_errors(template_content)
        
        # Extract template type from filename
        template_type = os.path.basename(template_path).replace("_template.py", "")
        
        # Determine replacement patterns based on template type
        if template_type == "encoder_only":
            base_model = "bert"
            base_class = "BERT"
            base_model_id = "bert-base-uncased"
        elif template_type == "decoder_only":
            base_model = "gpt2"
            base_class = "GPT2"
            base_model_id = "gpt2"
        elif template_type == "encoder_decoder":
            base_model = "t5"
            base_class = "T5"
            base_model_id = "t5-small"
        elif template_type == "vision":
            base_model = "vit"
            base_class = "ViT"
            base_model_id = "google/vit-base-patch16-224"
        elif template_type == "vision_text":
            base_model = "clip"
            base_class = "CLIP"
            base_model_id = "openai/clip-vit-base-patch32"
        elif template_type == "speech":
            base_model = "wav2vec2"
            base_class = "Wav2Vec2"
            base_model_id = "facebook/wav2vec2-base"
        elif template_type == "multimodal":
            base_model = "llava"
            base_class = "LLaVA"
            base_model_id = "llava-hf/llava-1.5-7b-hf"
        else:
            logger.warning(f"Unknown template type: {template_type}, using encoder_only defaults")
            base_model = "bert"
            base_class = "BERT"
            base_model_id = "bert-base-uncased"
        
        # Make replacements
        replacements = {
            # Replace registry name
            f"{base_class}_MODELS_REGISTRY": f"{upper_name}_MODELS_REGISTRY",
            
            # Replace class names
            f"Test{base_class}Models": f"Test{class_name}Models",
            
            # Replace model types
            base_model_id: f"{model_name}",
            
            # Replace class identifiers
            base_class: class_name,
            
            # Replace lowercase identifiers
            base_model: valid_id,
            
            # Replace file path references
            f"hf_{base_model}_": f"hf_{valid_id}_"
        }
        
        content = template_content
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Apply specific fixes for hyphenated names
        content = fix_registry_references(content, model_name)
        content = fix_class_declarations(content, model_name)
        content = fix_method_references(content, model_name)
        
        # Fix specific syntax issues
        content = re.sub(r'print\("([^"]*?)$', r'print(f"\1")', content, flags=re.MULTILINE)
        content = re.sub(r'print\("(.*?)$', r'print(f"\1")', content, flags=re.MULTILINE)
        content = re.sub(r'print\(f"\n([^"]*)"', r'print(f"\n\1")', content)
        
        # Try to fix syntax errors incrementally
        attempts = 0
        max_attempts = 3
        syntax_valid = False
        
        while not syntax_valid and attempts < max_attempts:
            # Validate syntax
            syntax_valid, error = check_file_syntax(content)
            
            if not syntax_valid:
                logger.warning(f"Syntax error in attempt {attempts+1} for {model_name}: {error}")
                
                # Try to fix based on error message
                if "unterminated string literal" in error.lower():
                    # Fix common string errors
                    content = re.sub(r'print\(["\'](.*?)$', r'print(f"\1")', content, flags=re.MULTILINE)
                    # Add closing quote to print statements
                    content = re.sub(r'print\((f?")[^\n"]*$', r'print(\1")', content, flags=re.MULTILINE)
                
                # Fix other common errors as needed
                attempts += 1
        
        if not syntax_valid:
            logger.error(f"Failed to fix syntax errors after {max_attempts} attempts for {model_name}")
            
            # Write the content to a temporary file for debugging
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{model_name}.py")
            with open(temp_file.name, 'w') as f:
                f.write(content)
            logger.error(f"Wrote problematic content to {temp_file.name}")
            return False, error
        
        # Write the file
        output_file = output_dir / f"test_hf_{valid_id}.py"
        with open(output_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully created {output_file}")
        return True, None
        
    except Exception as e:
        logger.error(f"Error creating test file for {model_name}: {str(e)}")
        traceback.print_exc()
        return False, str(e)

def find_hyphenated_models() -> List[str]:
    """Find all hyphenated model names in the architecture types."""
    hyphenated = []
    for models in ARCHITECTURE_TYPES.values():
        for model in models:
            if "-" in model:
                hyphenated.append(model)
    return sorted(hyphenated)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix test files for hyphenated model names")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Specific hyphenated model name to fix")
    group.add_argument("--all-hyphenated", action="store_true", help="Process all hyphenated model names")
    group.add_argument("--list", action="store_true", help="List all known hyphenated model names")
    
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory for fixed test files")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.list:
        # List all hyphenated model names
        hyphenated_models = find_hyphenated_models()
        print("\nKnown hyphenated model names:")
        for model in hyphenated_models:
            print(f"  - {model}")
        return 0
    
    if args.all_hyphenated:
        # Process all hyphenated model names
        hyphenated_models = find_hyphenated_models()
        logger.info(f"Found {len(hyphenated_models)} hyphenated model names")
        
        success_count = 0
        failure_count = 0
        
        for model_name in hyphenated_models:
            success, error = create_test_file(model_name, args.output_dir)
            if success:
                success_count += 1
            else:
                failure_count += 1
                logger.error(f"Failed to create test file for {model_name}: {error}")
        
        logger.info(f"Processed {len(hyphenated_models)} hyphenated model names")
        logger.info(f"Success: {success_count}, Failed: {failure_count}")
        
        return 0 if failure_count == 0 else 1
    
    if args.model:
        # Process a specific model
        if "-" not in args.model:
            logger.error(f"{args.model} is not a hyphenated model name")
            return 1
        
        success, error = create_test_file(args.model, args.output_dir)
        if not success:
            logger.error(f"Failed to create test file for {args.model}: {error}")
            return 1
        
        logger.info(f"Successfully created test file for {args.model}")
        return 0

if __name__ == "__main__":
    sys.exit(main())