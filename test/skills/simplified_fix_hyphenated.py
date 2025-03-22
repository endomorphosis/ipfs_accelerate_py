#!/usr/bin/env python3
"""
Simplified script to fix hyphenated model names in test files.

This script uses a simple direct approach to create test files for hyphenated model names,
bypassing the complex template replacement that causes syntax errors.
"""

import os
import sys
import re
import logging
import argparse
from pathlib import Path
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = CURRENT_DIR / "templates" / "hyphenated_name_template.py"
OUTPUT_DIR = CURRENT_DIR / "fixed_tests"

# Maps for hyphenated model names
# Format: 'original-name': ('valid_identifier', 'ClassName', 'CLASS_NAME')
HYPHENATED_MODEL_MAPS = {
    'gpt-j': ('gpt_j', 'GPTJ', 'GPT_J'),
    'gpt-neo': ('gpt_neo', 'GPTNeo', 'GPT_NEO'),
    'gpt-neox': ('gpt_neox', 'GPTNeoX', 'GPT_NEOX'),
    'xlm-roberta': ('xlm_roberta', 'XLMRoBERTa', 'XLM_ROBERTA'),
    'vision-text-dual-encoder': ('vision_text_dual_encoder', 'VisionTextDualEncoder', 'VISION_TEXT_DUAL_ENCODER'),
    'vision-encoder-decoder': ('vision_encoder_decoder', 'VisionEncoderDecoder', 'VISION_ENCODER_DECODER'),
    'wav2vec2-base': ('wav2vec2_base', 'Wav2Vec2Base', 'WAV2VEC2_BASE'),
    'video-llava': ('video_llava', 'VideoLlava', 'VIDEO_LLAVA'),
}

# Default models for each type
DEFAULT_MODELS = {
    'gpt-j': 'EleutherAI/gpt-j-6b',
    'gpt-neo': 'EleutherAI/gpt-neo-1.3B',
    'xlm-roberta': 'xlm-roberta-base',
    'vision-text-dual-encoder': 'openai/clip-vit-base-patch32',
    'vision-encoder-decoder': 'nlpconnect/vit-gpt2-image-captioning',
    'wav2vec2-base': 'facebook/wav2vec2-base',
    'video-llava': 'llava-hf/llava-1.5-7b-hf',
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

def to_valid_identifier(text):
    """Convert a hyphenated model name to a valid Python identifier."""
    # Replace hyphens with underscores
    text = text.replace("-", "_")
    # Remove any other invalid characters
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    # Ensure it doesn't start with a number
    if text and text[0].isdigit():
        text = '_' + text
    return text

def get_class_name_capitalization(model_name):
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

def get_upper_case_name(model_name):
    """Get upper case name for a model name (for constants)."""
    # Check if we have a predefined mapping
    if model_name in HYPHENATED_MODEL_MAPS:
        return HYPHENATED_MODEL_MAPS[model_name][2]
    
    # Otherwise, replace hyphens with underscores and uppercase
    return model_name.replace("-", "_").upper()

def get_architecture_type(model_name):
    """Determine the architecture type for a model name."""
    model_name_lower = model_name.lower()
    
    for arch_type, models in ARCHITECTURE_TYPES.items():
        for model in models:
            if model_name_lower.startswith(model):
                return arch_type
    
    # Default to encoder-only if no match found
    logger.warning(f"Could not determine architecture type for {model_name}, defaulting to encoder-only")
    return "encoder-only"

def check_file_syntax(content, filename="<string>"):
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

def create_hyphenated_test_file(model_name, output_dir=None):
    """Create a fixed test file for a model with a hyphenated name using direct string replacement."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    try:
        # Check if model name has a hyphen
        if "-" not in model_name:
            logger.warning(f"{model_name} is not a hyphenated model name, skipping")
            return False, f"{model_name} is not a hyphenated model name"
        
        # Get model properties
        model_id = to_valid_identifier(model_name)
        class_name = get_class_name_capitalization(model_name)
        model_upper = get_upper_case_name(model_name)
        arch_type = get_architecture_type(model_name)
        default_model = DEFAULT_MODELS.get(model_name, f"{model_name}-base")
        
        logger.info(f"Creating test file for {model_name} -> id: {model_id}, class: {class_name}, type: {arch_type}")
        
        # Generate file content directly to avoid template syntax issues
        content = f"""#!/usr/bin/env python3

"""
        content += f'''
"""
Test file for {class_name} models.
This file tests the {class_name} model type from HuggingFace Transformers.
"""

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

# Check if dependencies are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Registry for {class_name} models
{model_upper}_MODELS_REGISTRY = {{
    "{default_model}": {{
        "full_name": "{class_name} Base",
        "architecture": "{arch_type}",
        "description": "{class_name} model for text generation",
        "model_type": "{model_name}",
        "parameters": "1.3B",
        "context_length": 2048,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "recommended_tasks": ["text-generation"]
    }}
}}

def select_device():
    """Select the best available device for inference."""
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

class Test{class_name}Models:
    """
    Test class for {class_name} models.
    """
    
    def __init__(self, model_id="{default_model}", device=None):
        """Initialize the test class for {class_name} models.
        
        Args:
            model_id: The model ID to test (default: "{default_model}")
            device: The device to run tests on (default: None = auto-select)
        """
        self.model_id = model_id
        self.device = device if device else select_device()
        self.performance_stats = {{}}
    
    def test_pipeline(self):
        """Test the model using the pipeline API."""
        try:
            if not HAS_TRANSFORMERS:
                logger.warning("Transformers library not available, skipping pipeline test")
                return {{"success": False, "error": "Transformers library not available"}}
                
            logger.info(f"Testing {class_name} model {{self.model_id}} with pipeline API on {{self.device}}")
            
            # Record start time
            start_time = time.time()
            
            # Initialize the pipeline
            pipe = transformers.pipeline(
                "text-generation", 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {{load_time:.2f}} seconds")
            
            # Test with a simple input
            test_input = "{class_name} is a model that"
            
            # Record inference start time
            inference_start = time.time()
            
            # Run inference
            outputs = pipe(test_input)
            
            # Record inference time
            inference_time = time.time() - inference_start
            
            # Store performance stats
            self.performance_stats["pipeline"] = {{
                "load_time": load_time,
                "inference_time": inference_time
            }}
            
            return {{
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "inference_time": inference_time
            }}
        except Exception as e:
            logger.error(f"Error testing pipeline: {{e}}")
            return {{"success": False, "error": str(e)}}
    
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        results = {{}}
        
        # Run pipeline test
        pipeline_result = self.test_pipeline()
        results["pipeline"] = pipeline_result
        
        # Add metadata
        results["metadata"] = {{
            "model_id": self.model_id,
            "device": self.device,
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH
        }}
        
        return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test {class_name} HuggingFace models")
    parser.add_argument("--model", type=str, default="{default_model}", help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to run tests on (cuda, cpu)")
    
    args = parser.parse_args()
    
    # Initialize the test class
    {model_id}_tester = Test{class_name}Models(model_id=args.model, device=args.device)
    
    # Run the tests
    results = {model_id}_tester.run_tests()
    
    # Print a summary
    success = results["pipeline"].get("success", False)
    
    print("\\nTEST RESULTS SUMMARY:")
    
    if success:
        print(f"  Successfully tested {{args.model}}")
        print(f"  - Device: {{{model_id}_tester.device}}")
        print(f"  - Inference time: {{results['pipeline'].get('inference_time', 'N/A'):.4f}}s")
    else:
        print(f"  Failed to test {{args.model}}")
        print(f"  - Error: {{results['pipeline'].get('error', 'Unknown error')}}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        # Validate syntax
        syntax_valid, error = check_file_syntax(content)
        if not syntax_valid:
            logger.error(f"Syntax error in generated file for {model_name}: {error}")
            return False, error
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Write the file
        output_file = os.path.join(output_dir, f"test_hf_{model_id}.py")
        with open(output_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully created {output_file}")
        return True, None
        
    except Exception as e:
        logger.error(f"Error creating test file for {model_name}: {str(e)}")
        traceback.print_exc()
        return False, str(e)

def find_hyphenated_models():
    """Find all hyphenated model names in the architecture types."""
    hyphenated = []
    for models in ARCHITECTURE_TYPES.values():
        for model in models:
            if "-" in model:
                hyphenated.append(model)
    return sorted(list(set(hyphenated)))  # Remove duplicates

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create test files for models with hyphenated names")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Specific hyphenated model name to process")
    group.add_argument("--all-hyphenated", action="store_true", help="Process all hyphenated model names")
    group.add_argument("--list", action="store_true", help="List all known hyphenated model names")
    
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for test files")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or OUTPUT_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
            success, error = create_hyphenated_test_file(model_name, output_dir)
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
        
        success, error = create_hyphenated_test_file(args.model, output_dir)
        if not success:
            logger.error(f"Failed to create test file for {args.model}: {error}")
            return 1
        
        logger.info(f"Successfully created test file for {args.model}")
        return 0

if __name__ == "__main__":
    sys.exit(main())