#!/usr/bin/env python3
"""
Regenerate fixed test files using architecture-specific templates.

This script:
1. Uses the architecture-specific templates (encoder-only, decoder-only, etc.)
2. Fixes indentation using the complete_indentation_fix.py script
3. Ensures that the mock detection system is implemented in all test files
4. Verifies syntax of generated files
5. Outputs the results to the fixed_tests directory

Usage:
    python regenerate_fixed_tests.py [--model MODEL_TYPE] [--all] [--verify]
"""

import os
import sys
import argparse
import logging
import subprocess
import shutil
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

# Define architecture types for model mapping
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "albert", "camembert", "xlm-roberta", "deberta"],
    "decoder-only": ["gpt2", "gpt-j", "gptj", "gpt-neo", "gpt_neo", "gpt_neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt", "opt"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "mt5", "longt5", "led", "marian"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2"],
    "vision-text": ["clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava"]
}

def get_architecture_type(model_type):
    """Determine architecture type based on model type."""
    model_type_lower = model_type.lower()
    for arch_type, models in ARCHITECTURE_TYPES.items():
        if any(model in model_type_lower for model in models):
            return arch_type
    return "encoder-only"  # Default to encoder-only if unknown

def get_template_for_architecture(arch_type):
    """Get the template path for a specific architecture type."""
    # Define base directory for templates
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    
    template_map = {
        "encoder-only": os.path.join(template_dir, "encoder_only_template.py"),
        "decoder-only": os.path.join(template_dir, "decoder_only_template.py"),
        "encoder-decoder": os.path.join(template_dir, "encoder_decoder_template.py"),
        "vision": os.path.join(template_dir, "vision_template.py"),
        "vision-text": os.path.join(template_dir, "vision_text_template.py"),
        "speech": os.path.join(template_dir, "speech_template.py"),
        "multimodal": os.path.join(template_dir, "multimodal_template.py")
    }
    
    template_path = template_map.get(arch_type)
    if not template_path or not os.path.exists(template_path):
        logger.warning(f"Template not found for {arch_type}, using encoder-only template")
        fallback_template = os.path.join(template_dir, "encoder_only_template.py")
        if not os.path.exists(fallback_template):
            logger.error(f"Fallback template not found: {fallback_template}")
            # List available templates
            available_templates = [f for f in os.listdir(template_dir) if f.endswith("_template.py")]
            logger.info(f"Available templates: {available_templates}")
            # Use first available template if any exist
            if available_templates:
                return os.path.join(template_dir, available_templates[0])
            return None
        return fallback_template
        
    return template_path

def get_default_model_for_type(model_type):
    """Get default model ID for a model type."""
    default_models = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased",
        "electra": "google/electra-small-discriminator",
        "albert": "albert-base-v2",
        "gpt2": "gpt2",
        "gptj": "EleutherAI/gpt-j-6b",
        "gpt_neo": "EleutherAI/gpt-neo-125m",
        "gpt_neox": "EleutherAI/gpt-neox-20b",
        "bloom": "bigscience/bloom-560m",
        "llama": "meta-llama/Llama-2-7b",
        "opt": "facebook/opt-125m",
        "t5": "t5-small",
        "bart": "facebook/bart-base",
        "pegasus": "google/pegasus-xsum",
        "mbart": "facebook/mbart-large-cc25",
        "mt5": "google/mt5-small",
        "vit": "google/vit-base-patch16-224",
        "swin": "microsoft/swin-tiny-patch4-window7-224",
        "deit": "facebook/deit-base-patch16-224",
        "beit": "microsoft/beit-base-patch16-224",
        "convnext": "facebook/convnext-tiny-224",
        "clip": "openai/clip-vit-base-patch32",
        "blip": "Salesforce/blip-vqa-base",
        "llava": "llava-hf/llava-1.5-7b-hf",
        "wav2vec2": "facebook/wav2vec2-base-960h",
        "hubert": "facebook/hubert-base-ls960",
        "whisper": "openai/whisper-small"
    }
    
    # Return the default model if found, otherwise construct a reasonable default
    model_type_lower = model_type.lower()
    if model_type_lower in default_models:
        return default_models[model_type_lower]
        
    # Try to find a close match
    for key in default_models:
        if key in model_type_lower:
            return default_models[key]
            
    # Fallback to a generic model name
    return f"{model_type}-base"

def check_mock_detection(content):
    """
    Check if mock detection system is implemented in content.
    
    Args:
        content: File content as string
        
    Returns:
        bool: True if mock detection is implemented, False otherwise
    """
    # Check for key mock detection patterns
    has_using_real_inference = "using_real_inference = HAS_TRANSFORMERS and HAS_TORCH" in content
    has_using_mocks = "using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE" in content
    has_visual_indicators = "🚀 Using REAL INFERENCE with actual models" in content and "🔷 Using MOCK OBJECTS for CI/CD testing only" in content
    has_metadata = '"using_real_inference": using_real_inference,' in content and '"using_mocks": using_mocks,' in content
    
    return has_using_real_inference and has_using_mocks and has_visual_indicators and has_metadata

def ensure_mock_detection(content):
    """
    Ensure mock detection system is implemented in content.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with mock detection implemented
    """
    if check_mock_detection(content):
        return content
        
    # Add mock detection logic in the run_tests method
    if "def run_tests(" in content and "return {" in content:
        # Find the return statement in run_tests method
        return_index = content.find("return {", content.find("def run_tests("))
        
        if return_index != -1:
            # Add mock detection logic before the return statement
            mock_detection_code = """
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
"""
            # Insert the mock detection code
            content = content[:return_index] + mock_detection_code + content[return_index:]
            
            # Now add the mock detection metadata to the return dictionary
            if '"metadata":' in content:
                metadata_index = content.find('"metadata":', return_index)
                if metadata_index != -1:
                    # Find the closing brace of the metadata dictionary
                    closing_brace_index = content.find("}", metadata_index)
                    if closing_brace_index != -1:
                        # Add the mock detection metadata
                        mock_metadata = """
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_tokenizers": HAS_TOKENIZERS,
                "has_sentencepiece": HAS_SENTENCEPIECE,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
"""
                        content = content[:closing_brace_index] + mock_metadata + content[closing_brace_index:]
            
    # Add the visual indicators to the main function
    if "def main():" in content and "print(f\"Successfully tested" in content:
        success_print_index = content.find("print(f\"Successfully tested", content.find("def main():"))
        
        if success_print_index != -1:
            # Find the appropriate location to insert the visual indicators
            # Look for Test Results Summary section
            summary_index = content.find("TEST RESULTS SUMMARY", success_print_index - 200, success_print_index)
            if summary_index != -1:
                next_line_index = content.find("\n", summary_index)
                if next_line_index != -1:
                    # Add the visual indicators
                    visual_indicators_code = """
    
    # Indicate real vs mock inference clearly
    if using_real_inference and not using_mocks:
        print(f"🚀 Using REAL INFERENCE with actual models")
    else:
        print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
"""
                    content = content[:next_line_index] + visual_indicators_code + content[next_line_index:]
            
    return content

def fix_indentation(file_path):
    """
    Fix indentation issues in a file.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Verify the file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        # Skip indentation fix if the file is already valid
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            compile(content, file_path, 'exec')
            logger.info(f"✅ {file_path}: Syntax is already valid, no indentation fix needed")
            return True
        except SyntaxError:
            # File needs fixing
            pass
            
        # First try direct_fix.py which is more reliable for templates
        direct_fix_script = os.path.join(os.path.dirname(__file__), "direct_fix.py")
        if os.path.exists(direct_fix_script):
            logger.info(f"Trying direct fix on {file_path}")
            try:
                direct_fix_cmd = [sys.executable, direct_fix_script, file_path, "--apply"]
                direct_fix_result = subprocess.run(direct_fix_cmd, capture_output=True, text=True)
                
                # Check if the fix was successful by verifying the syntax
                with open(file_path, 'r') as f:
                    content = f.read()
                compile(content, file_path, 'exec')
                logger.info(f"✅ {file_path}: Successfully fixed with direct_fix.py")
                return True
            except Exception as direct_fix_error:
                logger.warning(f"Direct fix unsuccessful: {direct_fix_error}")
                # Continue to next approach if this fails
        
        # Next try complete_indentation_fix.py
        complete_fix_script = os.path.join(os.path.dirname(__file__), "complete_indentation_fix.py")
        if os.path.exists(complete_fix_script):
            logger.info(f"Trying complete indentation fix on {file_path}")
            complete_fix_cmd = [sys.executable, complete_fix_script, file_path, "--verify"]
            complete_fix_result = subprocess.run(complete_fix_cmd, capture_output=True, text=True)
            
            if complete_fix_result.returncode == 0:
                logger.info(f"✅ {file_path}: Successfully fixed with complete_indentation_fix.py")
                return True
            else:
                logger.warning(f"Complete indentation fix unsuccessful: {complete_fix_result.stderr}")
                # Continue to next approach if this fails
                
        # As a last resort, manually copy the template directly
        logger.info(f"Attempting direct template copy for {file_path}")
        basename = os.path.basename(file_path)
        model_type = basename[8:-3]  # Extract from test_hf_MODEL.py
        arch_type = get_architecture_type(model_type)
        template_path = get_template_for_architecture(arch_type)
        
        if os.path.exists(template_path):
            # Read template content
            with open(template_path, 'r') as f:
                template_content = f.read()
                
            # Replace BERT with appropriate model type
            model_upper = model_type.upper()
            model_capitalized = model_type.capitalize()
            
            replacements = {
                "BERT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestBertModels": f"Test{model_capitalized}Models",
                "bert": model_type,
                "bert-base-uncased": get_default_model_for_type(model_type),
                "hf_bert": f"hf_{model_type}"
            }
            
            for old, new in replacements.items():
                template_content = template_content.replace(old, new)
                
            # Write directly to the output file
            with open(file_path, 'w') as f:
                f.write(template_content)
                
            # Verify the syntax
            try:
                compile(template_content, file_path, 'exec')
                logger.info(f"✅ {file_path}: Successfully fixed with direct template copy")
                return True
            except SyntaxError as e:
                logger.error(f"❌ {file_path}: Syntax error after direct template copy: {e}")
                return False
                
        return False
            
    except Exception as e:
        logger.error(f"Error fixing indentation: {e}")
        return False

def regenerate_test_file(model_type, output_dir="fixed_tests", verify=True):
    """
    Regenerate a test file using the architecture-specific template.
    
    Args:
        model_type: Type of model (e.g., bert, gpt2, t5)
        output_dir: Directory to save the fixed test file
        verify: Whether to verify syntax after generation
        
    Returns:
        Tuple of (success, output_path)
    """
    try:
        # Get input and output paths
        input_file = f"test_hf_{model_type}.py"
        output_path = os.path.join(output_dir, input_file)
        
        # Determine architecture type
        arch_type = get_architecture_type(model_type)
        logger.info(f"Model {model_type} has architecture type: {arch_type}")
        
        # Get template file
        template_path = get_template_for_architecture(arch_type)
        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            return False, None
        
        # Get default model for this type
        default_model = get_default_model_for_type(model_type)
        
        # Read template
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Check if template has mock detection
        if not check_mock_detection(template_content):
            logger.warning(f"Template {template_path} does not have mock detection, adding it")
            template_content = ensure_mock_detection(template_content)
        
        # Replace template placeholders
        content = template_content
        model_upper = model_type.upper()
        model_capitalized = model_type.capitalize()
        
        # Replace standard placeholders based on architecture type
        if arch_type == "encoder-only":
            replacements = {
                "BERT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestBertModels": f"Test{model_capitalized}Models",
                "bert": model_type,
                "BertForMaskedLM": f"{model_capitalized}ForMaskedLM",
                "bert-base-uncased": default_model,
                "fill-mask": "fill-mask"
            }
        elif arch_type == "decoder-only":
            replacements = {
                "GPT2_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestGpt2Models": f"Test{model_capitalized}Models",
                "gpt2": model_type,
                "GPT2LMHeadModel": f"{model_capitalized}LMHeadModel",
                "gpt2-medium": default_model,
                "text-generation": "text-generation" 
            }
        elif arch_type == "encoder-decoder":
            replacements = {
                "T5_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestT5Models": f"Test{model_capitalized}Models",
                "t5": model_type,
                "T5ForConditionalGeneration": f"{model_capitalized}ForConditionalGeneration",
                "t5-small": default_model,
                "text2text-generation": "text2text-generation"
            }
        elif arch_type == "vision":
            replacements = {
                "VIT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestVitModels": f"Test{model_capitalized}Models",
                "vit": model_type,
                "ViTForImageClassification": f"{model_capitalized}ForImageClassification",
                "google/vit-base-patch16-224": default_model,
                "image-classification": "image-classification"
            }
        elif arch_type in ["vision-text", "multimodal"]:
            replacements = {
                "CLIP_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestClipModels": f"Test{model_capitalized}Models",
                "clip": model_type,
                "CLIPModel": f"{model_capitalized}Model",
                "openai/clip-vit-base-patch32": default_model,
                "image-to-text": "image-to-text"
            }
        elif arch_type == "speech":
            replacements = {
                "WHISPER_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestWhisperModels": f"Test{model_capitalized}Models",
                "whisper": model_type,
                "WhisperForConditionalGeneration": f"{model_capitalized}ForConditionalGeneration",
                "openai/whisper-small": default_model,
                "automatic-speech-recognition": "automatic-speech-recognition"
            }
        else:
            replacements = {
                "BERT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestBertModels": f"Test{model_capitalized}Models",
                "bert": model_type,
                "bert-base-uncased": default_model
            }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Make directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create backup if file exists
        if os.path.exists(output_path):
            backup_path = f"{output_path}.bak"
            shutil.copy2(output_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Write new file
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Created test file: {output_path}")
        
        # Apply indentation fixes
        logger.info(f"Applying indentation fixes to {output_path}")
        if not fix_indentation(output_path):
            logger.warning(f"Failed to fix indentation for {output_path}")
        
        # Verify syntax if requested
        if verify:
            logger.info(f"Verifying syntax of {output_path}")
            try:
                with open(output_path, 'r') as f:
                    code = f.read()
                compile(code, output_path, 'exec')
                logger.info(f"✅ {output_path}: Syntax is valid")
            except SyntaxError as e:
                logger.error(f"❌ {output_path}: Syntax error: {e}")
                return False, output_path
        
        # Verify mock detection
        with open(output_path, 'r') as f:
            content = f.read()
        
        if not check_mock_detection(content):
            logger.error(f"❌ {output_path}: Mock detection not implemented after regeneration")
            return False, output_path
        
        logger.info(f"✅ {output_path}: Mock detection is implemented")
        return True, output_path
        
    except Exception as e:
        logger.error(f"Error regenerating test file for {model_type}: {e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(description="Regenerate test files using architecture-specific templates")
    parser.add_argument("--model", type=str, help="Model type to regenerate (e.g., bert, gpt2, t5)")
    parser.add_argument("--all", action="store_true", help="Regenerate all test files")
    parser.add_argument("--verify", action="store_true", help="Verify syntax after generation")
    parser.add_argument("--output-dir", type=str, default="fixed_tests", help="Directory to save fixed files")
    parser.add_argument("--list", action="store_true", help="List all model types that can be regenerated")
    parser.add_argument("--list-architectures", action="store_true", help="List all supported architecture types")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List architectures if requested
    if args.list_architectures:
        print("\nSupported Architecture Types:")
        for arch_type, models in ARCHITECTURE_TYPES.items():
            print(f"  Architecture: {arch_type}")
            print(f"  Model families: {', '.join(models)}")
            print(f"  Template: {os.path.basename(get_template_for_architecture(arch_type) or 'Not found')}")
            print()
        return 0
    
    # List model types if requested
    if args.list:
        print("\nAvailable model types for regeneration:")
        model_families = []
        # Check for test files in current directory
        test_dir = args.output_dir
        if os.path.exists(test_dir) and os.path.isdir(test_dir):
            for file in os.listdir(test_dir):
                if file.startswith("test_hf_") and file.endswith(".py"):
                    model_type = file[8:-3]  # Extract model type from test_hf_MODEL.py
                    model_families.append(model_type)
        
        # Add predefined model types for each architecture
        for arch_type, models in ARCHITECTURE_TYPES.items():
            for model in models:
                if model not in model_families:
                    model_families.append(model)
        
        # Print sorted list of model types
        model_families.sort()
        for model in model_families:
            arch_type = get_architecture_type(model)
            default_model = get_default_model_for_type(model)
            print(f"  - {model} (Architecture: {arch_type}, Default model: {default_model})")
        print()
        return 0
    
    if not args.model and not args.all:
        logger.error("Must specify either --model or --all")
        return 1
    
    # Collect models to regenerate
    models_to_regenerate = []
    
    if args.all:
        # Try to find existing test files first
        test_dir = args.output_dir
        if os.path.exists(test_dir) and os.path.isdir(test_dir):
            for file in os.listdir(test_dir):
                if file.startswith("test_hf_") and file.endswith(".py"):
                    model_type = file[8:-3]  # Extract model type from test_hf_MODEL.py
                    models_to_regenerate.append(model_type)
        
        # If no existing files found, use predefined model types
        if not models_to_regenerate:
            # Add at least one model from each architecture type
            for arch_type, models in ARCHITECTURE_TYPES.items():
                if models:
                    models_to_regenerate.append(models[0])
    else:
        models_to_regenerate.append(args.model)
    
    logger.info(f"Found {len(models_to_regenerate)} models to regenerate")
    
    # Regenerate each model
    successes = 0
    failures = 0
    
    for model_type in models_to_regenerate:
        logger.info(f"Regenerating model: {model_type}")
        success, output_path = regenerate_test_file(
            model_type, 
            output_dir=args.output_dir, 
            verify=args.verify
        )
        
        if success:
            successes += 1
        else:
            failures += 1
    
    # Print summary
    logger.info("\nRegeneration Summary:")
    logger.info(f"- Successfully regenerated: {successes} models")
    logger.info(f"- Failed: {failures} models")
    logger.info(f"- Total: {len(models_to_regenerate)} models")
    
    if failures > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())