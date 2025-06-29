#!/usr/bin/env python3
"""
Generate a test file for a HuggingFace model using a fixed template with advanced model selection.
"""
import os
import sys
import logging
import re
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Try to import advanced model selection capabilities
try:
    from advanced_model_selection import (
        select_model_advanced, 
        get_hardware_profile, 
        TASK_TO_MODEL_TYPES,
        HARDWARE_PROFILES,
        estimate_model_size
    )
    HAS_ADVANCED_SELECTION = True
    logger.info("Advanced model selection is available")
except ImportError:
    HAS_ADVANCED_SELECTION = False
    logger.warning("Advanced model selection not available, using basic selection")
    # Define basic task mapping for standalone operation
    TASK_TO_MODEL_TYPES = {
        "text-classification": ["bert", "roberta", "distilbert", "albert", "electra", "xlm-roberta"],
        "token-classification": ["bert", "roberta", "distilbert", "electra"],
        "question-answering": ["bert", "roberta", "distilbert", "albert", "electra"],
        "text-generation": ["gpt2", "gpt-j", "gpt-neo", "bloom", "llama", "opt", "mistral", "falcon", "phi"],
        "summarization": ["t5", "bart", "pegasus", "led"],
        "translation": ["t5", "mbart", "m2m_100", "mt5"],
        "image-classification": ["vit", "resnet", "deit", "convnext", "swin"],
        "image-segmentation": ["mask2former", "segformer", "detr"],
        "object-detection": ["yolos", "detr", "mask2former"],
        "image-to-text": ["blip", "git", "pix2struct"],
        "text-to-image": ["stable-diffusion", "dall-e"],
        "automatic-speech-recognition": ["whisper", "wav2vec2", "hubert"],
        "audio-classification": ["wav2vec2", "hubert", "audio-spectrogram-transformer"],
        "visual-question-answering": ["llava", "blip", "git"],
        "document-question-answering": ["layoutlm", "donut", "pix2struct"],
        "fill-mask": ["bert", "roberta", "distilbert", "albert", "electra", "xlm-roberta"]
    }

# Define architecture types for model mapping
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava"]
}

def to_valid_identifier(text):
    """Convert text to a valid Python identifier."""
    # Replace hyphens with underscores
    text = text.replace("-", "_")
    # Remove any other invalid characters
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    # Ensure it doesn't start with a number
    if text and text[0].isdigit():
        text = '_' + text
    return text

def get_task_for_model(model_type):
    """Get the appropriate task for a model type."""
    model_type_lower = model_type.lower()
    
    # Check each task's model list
    for task, model_types in TASK_TO_MODEL_TYPES.items():
        if any(model_type_lower == model.lower() or 
               model_type_lower == model.lower().replace("-", "_") 
               for model in model_types):
            return task
    
    # Default tasks by architecture type
    for arch, models in ARCHITECTURE_TYPES.items():
        if any(model_type_lower == model.lower() or 
               model_type_lower == model.lower().replace("-", "_") 
               for model in models):
            if arch == "encoder-only":
                return "fill-mask"
            elif arch == "decoder-only":
                return "text-generation"
            elif arch == "encoder-decoder":
                return "text2text-generation"
            elif arch == "vision":
                return "image-classification"
            elif arch == "speech":
                return "automatic-speech-recognition"
    
    # Final fallback
    return "fill-mask"

def get_architecture_type(model_type):
    """Determine architecture type based on model type."""
    model_type_lower = model_type.lower()
    for arch_type, models in ARCHITECTURE_TYPES.items():
        if any(model in model_type_lower for model in models):
            return arch_type
    return "encoder-only"  # Default to encoder-only if unknown

def get_template_for_architecture(model_type, templates_dir="templates"):
    """Get the template path for a specific model type's architecture."""
    # For now, use the minimal_bert_template.py for all model types
    # This is a temporary solution until all templates are fixed
    minimal_template = os.path.join(templates_dir, "minimal_bert_template.py")
    if os.path.exists(minimal_template):
        logger.info(f"Using minimal template for {model_type}")
        return minimal_template
    
    # If minimal template doesn't exist, try to use the architecture-specific template
    arch_type = get_architecture_type(model_type)
    
    template_map = {
        "encoder-only": os.path.join(templates_dir, "encoder_only_template.py"),
        "decoder-only": os.path.join(templates_dir, "decoder_only_template.py"),
        "encoder-decoder": os.path.join(templates_dir, "encoder_decoder_template.py"),
        "vision": os.path.join(templates_dir, "vision_template.py"),
        "vision-text": os.path.join(templates_dir, "vision_text_template.py"),
        "speech": os.path.join(templates_dir, "speech_template.py"),
        "multimodal": os.path.join(templates_dir, "multimodal_template.py")
    }
    
    template_path = template_map.get(arch_type)
    if not template_path or not os.path.exists(template_path):
        logger.warning(f"Template not found for {arch_type}, using encoder-only template")
        fallback_template = os.path.join(templates_dir, "encoder_only_template.py")
        if not os.path.exists(fallback_template):
            logger.error(f"Fallback template not found: {fallback_template}")
            return None
        return fallback_template
        
    return template_path

def select_best_model(model_type, task=None, hardware_profile=None, max_size_mb=None, framework=None):
    """Select the best model based on available constraints."""
    # If advanced selection is available, use it
    if HAS_ADVANCED_SELECTION:
        try:
            model = select_model_advanced(
                model_type, 
                task=task, 
                hardware_profile=hardware_profile,
                max_size_mb=max_size_mb,
                framework=framework
            )
            logger.info(f"Selected model using advanced selection: {model}")
            return model
        except Exception as e:
            logger.warning(f"Advanced model selection failed: {e}, falling back to basic selection")
    
    # Basic fallback models
    default_models = {
        "bert": "google-bert/bert-base-uncased",
        "gpt2": "gpt2",
        "t5": "t5-small",
        "vit": "google/vit-base-patch16-224",
        "roberta": "roberta-base",
        "electra": "google/electra-small-discriminator",
        "llama": "meta-llama/Llama-2-7b",
        "blip": "Salesforce/blip-image-captioning-base",
        "whisper": "openai/whisper-base.en",
        "wav2vec2": "facebook/wav2vec2-base",
        "clip": "openai/clip-vit-base-patch32",
        "bart": "facebook/bart-base",
        "dinov2": "facebook/dinov2-base",
        "swin": "microsoft/swin-base-patch4-window7-224",
        "gpt-j": "EleutherAI/gpt-j-6b",
        "gpt-neo": "EleutherAI/gpt-neo-1.3B",
        "distilbert": "distilbert-base-uncased",
        "xlm-roberta": "xlm-roberta-base"
    }
    
    # Basic size categories for when hardware_profile is specified but advanced selection is unavailable
    size_variants = {
        "cpu-small": {
            "bert": "prajjwal1/bert-tiny",
            "gpt2": "distilgpt2",
            "t5": "t5-small",
            "vit": "google/vit-base-patch16-224-in21k",
            "roberta": "distilroberta-base",
            "electra": "google/electra-small-discriminator",
            "whisper": "openai/whisper-tiny.en",
        },
        "cpu-medium": {
            "bert": "google-bert/bert-base-uncased",
            "gpt2": "gpt2",
            "t5": "t5-base",
            "vit": "google/vit-base-patch16-224",
            "roberta": "roberta-base",
            "whisper": "openai/whisper-base.en",
        },
        "gpu-small": {
            "bert": "google-bert/bert-large-uncased",
            "gpt2": "gpt2-medium",
            "t5": "t5-large",
            "vit": "google/vit-large-patch16-224",
            "roberta": "roberta-large",
            "whisper": "openai/whisper-medium.en",
        }
    }
    
    # If hardware profile is specified, try to select a model based on it
    if hardware_profile and hardware_profile in size_variants:
        hw_models = size_variants[hardware_profile]
        if model_type.lower() in hw_models:
            logger.info(f"Selected model based on hardware profile {hardware_profile}: {hw_models[model_type.lower()]}")
            return hw_models[model_type.lower()]
    
    # Try default models dictionary
    if model_type.lower() in default_models:
        return default_models[model_type.lower()]
    
    # Final fallback: use model-type with -base suffix
    return f"{model_type}-base"

def generate_test_file(model_type, output_dir="test_output", template_path=None, 
                     task=None, hardware_profile=None, max_size_mb=None, framework=None):
    """
    Generate a test file for a specific model type using a template with advanced model selection.
    
    Args:
        model_type: Type of the model (e.g., bert, gpt2, t5, vit)
        output_dir: Directory to store the generated file
        template_path: Optional path to a specific template
        task: Optional specific task for model selection
        hardware_profile: Optional hardware profile for size constraints
        max_size_mb: Optional maximum model size in MB
        framework: Optional framework compatibility
        
    Returns:
        Path to the generated file if successful, None otherwise
    """
    try:
        # Get the template path based on model architecture if not provided
        if not template_path:
            template_path = get_template_for_architecture(
                model_type, 
                os.path.join(os.path.dirname(__file__), "templates")
            )
            
        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            return None
            
        # Create valid Python identifier
        model_valid = to_valid_identifier(model_type)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine output path
        output_path = os.path.join(output_dir, f"test_hf_{model_valid}.py")
        
        # Read template
        logger.info(f"Using template: {template_path}")
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Determine the best task if not provided
        if not task:
            task = get_task_for_model(model_type)
            logger.info(f"Determined task for {model_type}: {task}")
        
        # Select the best model using advanced selection if available
        default_model = select_best_model(
            model_type, 
            task=task, 
            hardware_profile=hardware_profile,
            max_size_mb=max_size_mb,
            framework=framework
        )
        logger.info(f"Selected model: {default_model}")
        
        # Prepare proper capitalization
        if '-' in model_type:
            # Handle special capitalization for hyphenated models
            parts = model_type.split('-')
            if model_type.lower() in ["gpt-j", "gpt-neo", "gpt-neox"]:
                # GPT-J should be GPTJ, GPT-Neo should be GPTNeo
                model_capitalized = ''.join(part.upper() if i == 0 else part.capitalize() 
                                       for i, part in enumerate(parts))
            elif model_type.lower() in ["xlm-roberta"]:
                # XLM-RoBERTa should be XLMRoBERTa
                model_capitalized = 'XLMRoBERTa'
            else:
                model_capitalized = ''.join(part.capitalize() for part in parts)
        else:
            model_capitalized = model_type.capitalize()
            
        # Properly handle special model names
        model_caps = {
            "bert": "BERT",
            "gpt2": "GPT2",
            "t5": "T5",
            "vit": "ViT",
            "roberta": "RoBERTa",
            "electra": "ELECTRA",
            "llama": "LLaMA",
            "clip": "CLIP",
            "whisper": "Whisper",
            "wav2vec2": "Wav2Vec2",
            "hubert": "HuBERT",
            "bart": "BART",
            "blip": "BLIP"
        }
        model_upper = model_caps.get(model_type.lower(), model_type.upper())
        
        # Ensure model_upper is valid for variable names (replace hyphens with underscores)
        model_upper_valid = model_upper.replace('-', '_')
        
        # Generate task-specific input text
        if task == "fill-mask":
            input_text = f"The quick brown fox jumps over the [MASK] dog."
            mask_token = "[MASK]"
        elif task == "text-generation":
            input_text = f"{model_upper} is a model that"
            mask_token = "<mask>"  # Not actually used for text generation
        elif task == "text2text-generation":
            input_text = f"translate English to German: The house is wonderful."
            mask_token = "<mask>"  # Not actually used for T5-like models
        elif task == "image-classification":
            input_text = f"{model_upper} processes images to classify them."
            mask_token = "<mask>"  # Not used for vision models
        elif task == "automatic-speech-recognition":
            input_text = f"{model_upper} processes audio to transcribe speech."
            mask_token = "<mask>"  # Not used for audio models
        else:
            input_text = f"{model_upper} is performing a {task} task."
            mask_token = "<mask>"
        
        # Prepare replacements
        replacements = {
            "BERT_MODELS_REGISTRY": f"{model_upper_valid}_MODELS_REGISTRY",
            "TestBertModels": f"Test{model_capitalized.replace('-', '')}Models",
            "bert": model_valid,
            "\"google-bert/bert-base-uncased\"": f"\"{default_model}\"",
            "\"fill-mask\"": f"\"{task}\"",
            "hf_bert_": f"hf_{model_valid}_",
            "BERT Base": f"{model_upper.replace('-', ' ')} Base",
            "bert-base-uncased": default_model.split('/')[-1] if '/' in default_model else default_model,
            "[MASK]": mask_token,
            "The quick brown fox jumps over the [MASK] dog": input_text,
            "Testing BERT model": f"Testing {model_upper.replace('-', ' ')} model",
            "Test BERT HuggingFace": f"Test {model_upper.replace('-', ' ')} HuggingFace"
        }
        
        # Apply replacements
        content = template_content
        for old, new in replacements.items():
            content = content.replace(old, new)
            
        # Write output file
        with open(output_path, 'w') as f:
            f.write(content)
            
        # Validate syntax
        try:
            compile(content, output_path, 'exec')
            logger.info(f"✅ Syntax is valid for {output_path}")
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in generated file: {e}")
            if hasattr(e, 'lineno') and e.lineno is not None:
                lines = content.split('\n')
                line_no = e.lineno - 1  # 0-based index
                if 0 <= line_no < len(lines):
                    logger.error(f"Problematic line {e.lineno}: {lines[line_no].rstrip()}")
            return None
            
        logger.info(f"Successfully generated test file: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating test file: {e}")
        import traceback
        traceback.print_exc()
        return None
        
def main():
    parser = argparse.ArgumentParser(description="Generate a test file for a HuggingFace model")
    parser.add_argument("--model", type=str, required=True, help="Model type (e.g., bert, gpt2, t5)")
    parser.add_argument("--output-dir", type=str, default="test_output", help="Output directory")
    parser.add_argument("--template", type=str, help="Path to a specific template")
    
    # Advanced selection arguments
    parser.add_argument("--task", type=str, help="Specific task for model selection")
    parser.add_argument("--hardware", type=str, help="Hardware profile for model selection")
    parser.add_argument("--max-size", type=int, help="Maximum model size in MB")
    parser.add_argument("--framework", type=str, help="Framework compatibility")
    
    # List available options
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--list-architectures", action="store_true", help="List available architecture types")
    
    args = parser.parse_args()
    
    # List tasks if requested
    if args.list_tasks:
        print("\nAvailable Tasks:")
        for task, model_types in sorted(TASK_TO_MODEL_TYPES.items()):
            print(f"  - {task}: {', '.join(model_types[:3])}{'...' if len(model_types) > 3 else ''}")
        return 0
    
    # List architectures if requested
    if args.list_architectures:
        print("\nAvailable Architecture Types:")
        for arch, model_types in sorted(ARCHITECTURE_TYPES.items()):
            print(f"  - {arch}: {', '.join(model_types[:3])}{'...' if len(model_types) > 3 else ''}")
        return 0
    
    # Generate the test file
    output_path = generate_test_file(
        args.model, 
        args.output_dir, 
        args.template,
        task=args.task,
        hardware_profile=args.hardware,
        max_size_mb=args.max_size,
        framework=args.framework
    )
    
    if output_path:
        print(f"\nSuccessfully generated test file:")
        print(f"  {output_path}")
        return 0
    else:
        print("\nFailed to generate test file.")
        return 1
        
if __name__ == "__main__":
    sys.exit(main())