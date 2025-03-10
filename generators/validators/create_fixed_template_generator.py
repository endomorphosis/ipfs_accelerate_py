#!/usr/bin/env python3
"""
Fixed Template-Based Test Generator

This script is a fixed version of create_template_based_test_generator.py
that uses the proper indentation for all templates.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the fixed templates
from generators.validators.fixed_template_example import (
    STANDARD_TEMPLATE,
    MODEL_INPUT_TEMPLATES,
    OUTPUT_CHECK_TEMPLATES,
    MODEL_SPECIFIC_CODE_TEMPLATES,
    CUSTOM_MODEL_LOADING_TEMPLATES
)

# Import validator
from generators.validators.template_validator_integration import validate_template_for_generator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model family mapping from original script
MODEL_FAMILIES = {
    "text_embedding": ["bert", "sentence-transformers", "distilbert", "roberta", "mpnet"],
    "text_generation": ["gpt2", "llama", "opt", "t5", "bloom", "mistral", "qwen", "falcon"],
    "vision": ["vit", "resnet", "detr", "deit", "convnext", "beit"],
    "audio": ["whisper", "wav2vec2", "hubert", "speecht5", "clap"],
    "multimodal": ["clip", "llava", "xclip", "blip", "flava"]
}

# Reverse mapping from model name to family
MODEL_TO_FAMILY = {}
for family, models in MODEL_FAMILIES.items():
    for model in models:
        MODEL_TO_FAMILY[model] = family

class FixedTemplateGenerator:
    """
    Generator for test files using fixed templates.
    """
    
    def __init__(self, args=None):
        """
        Initialize the generator.
        
        Args:
            args: Command line arguments
        """
        self.args = args or argparse.Namespace()  # Default empty args
        
        # Set default validation behavior if not specified
        if not hasattr(self.args, "validate"):
            self.args.validate = True
        if not hasattr(self.args, "skip_validation"):
            self.args.skip_validation = False
        if not hasattr(self.args, "strict_validation"):
            self.args.strict_validation = False
    
    def get_model_family(self, model_name: str) -> str:
        """
        Determine the model family for a given model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model family name
        """
        # Check direct mapping
        model_prefix = model_name.split('/')[0] if '/' in model_name else model_name
        model_prefix = model_prefix.split('-')[0] if '-' in model_prefix else model_prefix
        
        if model_prefix in MODEL_TO_FAMILY:
            return MODEL_TO_FAMILY[model_prefix]
        
        # Try pattern matching
        for family, models in MODEL_FAMILIES.items():
            for model in models:
                if model in model_name.lower():
                    return family
        
        # Default to text_embedding if unknown
        return "text_embedding"
    
    def generate_test_file(self, model_name: str, output_file: str = None, model_type: str = None) -> str:
        """
        Generate a test file for a specific model.
        
        Args:
            model_name: Name of the model
            output_file: Path to output file (optional)
            model_type: Model type/family (optional)
            
        Returns:
            Generated test file content
        """
        if not model_type:
            model_type = self.get_model_family(model_name)
        
        logger.info(f"Generating test file for model {model_name} of type {model_type}")
        
        # Get model class name from model name
        model_class_name = model_name.split('/')[-1] if '/' in model_name else model_name
        model_class_name = ''.join(part.capitalize() for part in model_class_name.replace('-', ' ').split())
        
        # Get appropriate templates for this model type
        model_input_code = MODEL_INPUT_TEMPLATES.get(model_type, MODEL_INPUT_TEMPLATES["text_embedding"])
        output_check_code = OUTPUT_CHECK_TEMPLATES.get(model_type, OUTPUT_CHECK_TEMPLATES["text_embedding"])
        custom_model_loading = CUSTOM_MODEL_LOADING_TEMPLATES.get(model_type, CUSTOM_MODEL_LOADING_TEMPLATES["text_embedding"])
        model_specific_code = MODEL_SPECIFIC_CODE_TEMPLATES.get(model_type, MODEL_SPECIFIC_CODE_TEMPLATES["text_embedding"])
        
        # Create test file content
        content = STANDARD_TEMPLATE
        content = content.replace("{{model_name}}", model_name)
        content = content.replace("{{model_class_name}}", model_class_name)
        content = content.replace("{{model_type}}", model_type)
        content = content.replace("{{generation_date}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        content = content.replace("{{model_input_code}}", model_input_code)
        content = content.replace("{{output_check_code}}", output_check_code)
        content = content.replace("{{custom_model_loading}}", custom_model_loading)
        content = content.replace("{{model_specific_code}}", model_specific_code)
        
        # Validate the generated template content
        if (getattr(self.args, "validate", True) and not getattr(self.args, "skip_validation", False)):
            logger.info(f"Validating template for {model_name}...")
            is_valid, validation_errors = validate_template_for_generator(
                content, 
                "fixed_template_generator",
                validate_hardware=True,
                check_resource_pool=True,
                strict_indentation=False  # Always be lenient with indentation for templates
            )
            
            if not is_valid:
                logger.warning(f"Generated template has validation errors:")
                for error in validation_errors:
                    logger.warning(f"  - {error}")
                
                # Do not fail for template indentation warnings
                # These are expected in template generators
                if getattr(self.args, "strict_validation", False) and not all(
                    "indentation" in error.lower() for error in validation_errors
                ):
                    raise ValueError(f"Template validation failed for {model_name}")
                else:
                    logger.warning("Continuing despite validation errors (use --strict-validation to fail on errors)")
            else:
                logger.info(f"Template validation passed for {model_name}")
        
        # Write to file if requested
        if output_file:
            output_path = Path(output_file)
            os.makedirs(output_path.parent, exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Generated test file saved to {output_file}")
            
            # Make file executable
            os.chmod(output_file, 0o755)
        
        return content
    
    def generate_family_tests(self, family: str, output_dir: str):
        """
        Generate test files for all models in a family.
        
        Args:
            family: Model family name
            output_dir: Directory to save test files
        """
        if family not in MODEL_FAMILIES:
            logger.error(f"Unknown model family: {family}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_prefix in MODEL_FAMILIES[family]:
            # Use a standard model for each prefix
            if model_prefix == "bert":
                model_name = "bert-base-uncased"
            elif model_prefix == "sentence-transformers":
                model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
            elif model_prefix == "distilbert":
                model_name = "distilbert-base-uncased"
            elif model_prefix == "roberta":
                model_name = "roberta-base"
            elif model_prefix == "gpt2":
                model_name = "gpt2"
            elif model_prefix == "llama":
                model_name = "meta-llama/Llama-2-7b-hf"
            elif model_prefix == "t5":
                model_name = "t5-small"
            elif model_prefix == "vit":
                model_name = "google/vit-base-patch16-224"
            elif model_prefix == "whisper":
                model_name = "openai/whisper-tiny"
            elif model_prefix == "wav2vec2":
                model_name = "facebook/wav2vec2-base-960h"
            elif model_prefix == "clip":
                model_name = "openai/clip-vit-base-patch32"
            else:
                model_name = f"{model_prefix}-base"
            
            output_file = os.path.join(output_dir, f"test_{model_prefix}.py")
            self.generate_test_file(model_name, output_file, family)
    
    def list_models(self):
        """
        List all model types/families.
        """
        print("Available model families:")
        for family, models in MODEL_FAMILIES.items():
            print(f"- {family} ({len(models)} models)")
            for model in models[:3]:  # Show first 3 models
                print(f"  - {model}")
            if len(models) > 3:
                print(f"  - ... ({len(models) - 3} more)")
    
    def list_families(self):
        """
        List all model families.
        """
        print("Available model families:")
        for family in MODEL_FAMILIES:
            print(f"- {family}")

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Fixed Template-Based Test Generator")
    parser.add_argument("--model", type=str, help="Generate test file for specific model")
    parser.add_argument("--family", type=str, help="Generate test files for specific model family")
    parser.add_argument("--output", type=str, help="Output file or directory (depends on mode)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-families", action="store_true", help="List available model families")
    # Validation options
    parser.add_argument("--validate", action="store_true", 
                     help="Validate templates before generation (default if validator available)")
    parser.add_argument("--skip-validation", action="store_true",
                     help="Skip template validation even if validator is available")
    parser.add_argument("--strict-validation", action="store_true",
                     help="Fail on validation errors")
    
    args = parser.parse_args()
    
    # Create generator
    generator = FixedTemplateGenerator(args)
    
    if args.list_models:
        generator.list_models()
    elif args.list_families:
        generator.list_families()
    elif args.model:
        # Generate test file for specific model
        output_file = args.output if args.output else f"test_{args.model.split('/')[-1]}.py"
        content = generator.generate_test_file(args.model, output_file)
        if not args.output:
            print(content)
    elif args.family:
        # Generate test files for family
        output_dir = args.output if args.output else f"tests_{args.family}"
        generator.generate_family_tests(args.family, output_dir)
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())