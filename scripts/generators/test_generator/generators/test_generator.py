#!/usr/bin/env python3
"""
Test generator for HuggingFace model tests.

This module provides classes and functions to generate standardized test files
for HuggingFace models based on their architecture type.
"""

import os
import sys
import glob
import json
import logging
import importlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"generate_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Import architecture detector
try:
    from .architecture_detector import (
        get_architecture_type, normalize_model_name, get_model_metadata,
        MODEL_NAME_MAPPING, ARCHITECTURE_TYPES
    )
except ImportError:
    # Try with absolute import
    try:
        from scripts.generators.architecture_detector import (
            get_architecture_type, normalize_model_name, get_model_metadata,
            MODEL_NAME_MAPPING, ARCHITECTURE_TYPES
        )
    except ImportError:
        # Try with full path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from refactored_test_suite.generators.architecture_detector import (
                get_architecture_type, normalize_model_name, get_model_metadata,
                MODEL_NAME_MAPPING, ARCHITECTURE_TYPES
            )
        except ImportError:
            logger.error("Failed to import architecture_detector module")
            sys.exit(1)

# Define priority models based on current priorities
PRIORITY_MODELS = {
    "high": [
        # Text Models
        "roberta", "albert", "distilbert", "deberta", "bart", "llama", 
        "mistral", "phi", "falcon", "mpt",
        # Vision Models
        "swin", "deit", "resnet", "convnext",
        # Multimodal Models
        "clip", "blip", "llava",
        # Audio Models
        "whisper", "wav2vec2", "hubert",
        # New Architecture Types
        "stable-diffusion", "mixtral", "mamba", "rag-token"
    ],
    "medium": [
        # Text Models
        "xlm-roberta", "electra", "ernie", "rembert", "gpt-neo", "gpt-j", 
        "opt", "gemma", "mbart", "pegasus", "prophetnet", "led",
        # Vision Models
        "beit", "segformer", "detr", "mask2former", "yolos", "sam", "dinov2",
        # Multimodal Models
        "flava", "git", "idefics", "paligemma", "imagebind",
        # Audio Models
        "sew", "unispeech", "clap", "musicgen", "encodec",
        # New Architecture Types
        "stable-diffusion-xl", "sdxl", "dalle", "qwen-moe", "rwkv", "rag-sequence"
    ],
    "low": [
        # Additional text models
        "reformer", "xlnet", "blenderbot", "gptj", "gpt_neo", "bigbird", "longformer", 
        "camembert", "marian", "mt5", "opus-mt", "m2m_100", "transfo-xl", "ctrl",
        "squeezebert", "funnel", "t5", "tapas", "canine", "roformer", "layoutlm", "layoutlmv2",
        # Additional vision models
        "bit", "dpt", "levit", "mlp-mixer", "mobilevit", "poolformer", "regnet", "segformer",
        "mobilenet_v1", "mobilenet_v2", "efficientnet", "donut", "beit", 
        # Additional multimodal
        "vilt", "vinvl", "align", "flava", "blip-2", "flamingo", "florence",
        # Additional speech
        "wavlm", "data2vec", "unispeech-sat", "hubert", "sew-d", "usm", "seamless_m4t",
        # Additional new architectures
        "kandinsky", "imagen", "pixart", "latent-diffusion", "switchht", "olmoe", 
        "hyena", "vim", "gamba", "rag-document", "rag-end2end"
    ]
}

class ModelTestGenerator:
    """Generator for model test files."""
    
    def __init__(self, output_dir="generated_tests", template_dir=None):
        """
        Initialize the generator.
        
        Args:
            output_dir: Directory to save generated files
            template_dir: Directory containing templates (or None to use default)
        """
        self.output_dir = output_dir
        
        # Set template directory
        if template_dir is None:
            # Use default templates directory relative to this file
            self.template_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "templates"
            )
        else:
            self.template_dir = template_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def get_template_for_architecture(self, arch_type: str) -> str:
        """
        Get the template content for an architecture type.
        
        Args:
            arch_type: Architecture type
            
        Returns:
            Template content as string
        """
        # Map architecture type to template file
        template_map = {
            "encoder-only": "encoder_only_template.py",
            "decoder-only": "decoder_only_template.py",
            "encoder-decoder": "encoder_decoder_template.py",
            "vision": "vision_template.py",
            "vision-encoder-text-decoder": "vision_text_template.py",
            "speech": "speech_template.py",
            "multimodal": "multimodal_template.py",
            "diffusion": "diffusion_model_template.py",
            "mixture-of-experts": "moe_model_template.py",
            "state-space": "ssm_model_template.py",
            "rag": "rag_model_template.py"
        }
        
        # Get template file path
        template_file = os.path.join(self.template_dir, template_map.get(arch_type, "encoder_only_template.py"))
        
        # Check if template file exists
        if not os.path.exists(template_file):
            logger.warning(f"Template file not found: {template_file}, using fallback")
            return self.get_fallback_template(arch_type)
        
        # Read template content
        try:
            with open(template_file, "r") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading template file: {e}")
            return self.get_fallback_template(arch_type)
    
    def get_fallback_template(self, arch_type: str) -> str:
        """
        Get a fallback template for an architecture type.
        
        Args:
            arch_type: Architecture type
            
        Returns:
            Template content as string
        """
        # Basic template for encoder-only models
        if arch_type == "encoder-only":
            return """#!/usr/bin/env python3
\"\"\"
Test file for {model_type_upper} models (encoder-only architecture).

This test verifies the functionality of the {model_type} model.
\"\"\"

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import model test base
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refactored_test_suite.model_test_base import EncoderOnlyModelTest


class {test_class_name}(EncoderOnlyModelTest):
    \"\"\"Test class for {model_type} model.\"\"\"
    
    def __init__(self, model_id=None, device=None):
        \"\"\"Initialize the test.\"\"\"
        # Set model type explicitly
        self.model_type = "{model_type}"
        self.task = "fill-mask"
        self.architecture_type = "encoder-only"
        
        # Call parent initializer
        super().__init__(model_id, device)
    
    def get_default_model_id(self) -> str:
        \"\"\"Get the default model ID for this model type.\"\"\"
        return "{default_model_id}"
    
    def run_all_tests(self):
        \"\"\"Run all tests for this model.\"\"\"
        # Run basic tests through parent class
        results = self.run_tests()
        
        # Optionally run additional model-specific tests
        # ...
        
        return results


def main():
    \"\"\"Command-line entry point.\"\"\"
    parser = argparse.ArgumentParser(description="Test {model_type} model")
    parser.add_argument("--model-id", type=str, help="Specific model ID to test")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="Device to test on")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for results")
    
    args = parser.parse_args()
    
    # Create and run the test
    test = {test_class_name}(args.model_id, args.device)
    results = test.run_all_tests()
    
    # Print a summary
    success = results.get("model_loading", {}).get("success", False)
    model_id = results.get("metadata", {}).get("model", test.model_id)
    device = results.get("metadata", {}).get("device", test.device)
    
    if success:
        print(f"✅ Successfully tested {model_id} on {device}")
    else:
        print(f"❌ Failed to test {model_id} on {device}")
        error = results.get("model_loading", {}).get("error", "Unknown error")
        print(f"Error: {error}")
    
    # Save results if requested
    if args.save:
        output_path = test.save_results(args.output_dir)
        if output_path:
            print(f"Results saved to {output_path}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
"""
        
        # Similar fallback templates for other architecture types
        # (would implement others as needed, but using encoder-only as a default)
        return self.get_fallback_template("encoder-only")

    def fill_template(self, template_content: str, model_name: str, arch_type: str) -> str:
        """
        Fill in the template with model-specific information.
        
        Args:
            template_content: Template content
            model_name: Model name
            arch_type: Architecture type
            
        Returns:
            Filled template content
        """
        # Get model metadata
        metadata = get_model_metadata(model_name)
        
        # Determine model information based on architecture type rather than specific model
        # Extract model class from the architecture type
        arch_class_mapping = {
            "encoder-only": "Bert",
            "decoder-only": "GPT2",
            "encoder-decoder": "T5",
            "vision": "ViT",
            "vision-encoder-text-decoder": "CLIP",
            "speech": "Whisper",
            "multimodal": "FLAVA",
            "diffusion": "StableDiffusion",
            "mixture-of-experts": "Mixtral",
            "state-space": "Mamba",
            "rag": "RAG"
        }
        
        # Use the architecture type to determine the model class name
        model_class = arch_class_mapping.get(arch_type, "Model")
        
        # Use the model class name to create the test class name
        test_class_name = f"Test{model_class}Model"
        
        # For compatibility with existing code, still extract model_type from the specific model
        normalized_name = metadata["normalized_name"]
        # But use it primarily for documentation, not for the class name
        model_type = normalized_name
        model_type_upper = model_type.upper()
        
        # Use the model name as the default model ID for testing
        default_model_id = model_name
        
        # Architecture-specific configurations
        sampling_rate = "16000"  # Default for speech models
        
        # Set specific values based on architecture
        if arch_type == "speech":
            # Specific sampling rates for speech models
            if "whisper" in model_name:
                sampling_rate = "16000"
            elif "wav2vec2" in model_name:
                sampling_rate = "16000"
            elif "hubert" in model_name:
                sampling_rate = "16000"
            elif "encodec" in model_name:
                sampling_rate = "24000"
            elif "seamless" in model_name or "seamless_m4t" in model_name:
                sampling_rate = "16000"
            elif "musicgen" in model_name:
                sampling_rate = "32000"
        
        # Replace placeholders in template
        content = template_content
        replacements = {
            "{model_type}": model_type,
            "{model_type_upper}": model_type_upper,
            "{test_class_name}": test_class_name,
            "{default_model_id}": default_model_id,
            "{architecture_type}": arch_type,
            "{task}": metadata["task"],
            "{example_input}": metadata["example_input"],
            "{sampling_rate}": sampling_rate
        }
        
        for key, value in replacements.items():
            if key in content:
                content = content.replace(key, value)
        
        return content
    
    def generate_test_file(self, model_name: str, force: bool = False, verify: bool = True) -> Tuple[bool, str]:
        """
        Generate a test file for a model class based on its architecture type.
        
        Args:
            model_name: Model name used as an example for the architecture type
            force: Whether to overwrite existing files
            verify: Whether to verify the generated file
            
        Returns:
            Tuple of (success, file_path)
        """
        try:
            # Detect architecture type
            arch_type = get_architecture_type(model_name)
            logger.info(f"Detected architecture type '{arch_type}' for model '{model_name}'")
            
            # Use model name for file generation to make unique files for each model
            # Normalize the model name to create a valid filename
            # This ensures we create individual test files for each model
            normalized_model = model_name.replace("-", "_").lower()
            
            # For architecture-specific templates, keep the naming as before
            arch_to_class_file = {
                "encoder-only": "bert",
                "decoder-only": "gpt2",
                "encoder-decoder": "t5",
                "vision": "vit",
                "vision-encoder-text-decoder": "clip",
                "speech": "whisper",
                "multimodal": "flava",
                "diffusion": "stable_diffusion",
                "mixture-of-experts": "mixtral",
                "state-space": "mamba",
                "rag": "rag"
            }
            
            # Get the class name for the file
            class_name = arch_to_class_file.get(arch_type, "base_model")
            
            # Generate file path - use model name for newer architectures or model variants
            if arch_type in ["diffusion", "mixture-of-experts", "state-space", "rag"]:
                file_name = f"test_hf_{normalized_model}.py"
            elif model_name.upper() == "NLLB":
                # Special case for NLLB
                file_name = f"test_hf_nllb.py"
            elif "-" in model_name or "_" in model_name or any(x in model_name.lower() for x in ["1", "2", "3", "7b", "8b", "13b", "70b"]):
                # For models with size variants or numbered versions, use model-specific name
                normalized_name = model_name.replace("/", "_").replace("-", "_").lower()
                file_name = f"test_hf_{normalized_name}.py"
            elif model_name.lower().startswith("m") and len(model_name) > 1 and model_name[1].isupper():
                # For multilingual models like mT0, mBART, mGPT
                normalized_name = model_name.lower()
                file_name = f"test_hf_{normalized_name}.py"
            else:
                # Use architecture class for common architectures
                file_name = f"test_hf_{class_name}.py"
            file_path = os.path.join(self.output_dir, file_name)
            
            # Check if file already exists and we're not forcing overwrite
            if os.path.exists(file_path) and not force:
                logger.info(f"File already exists: {file_path} (use force=True to overwrite)")
                return False, file_path
            
            # Get template for architecture
            template_content = self.get_template_for_architecture(arch_type)
            
            # Fill template with model information
            content = self.fill_template(template_content, model_name, arch_type)
            
            # Write to file
            with open(file_path, "w") as f:
                f.write(content)
            
            logger.info(f"✅ Generated test file: {file_path}")
            
            # Verify if requested
            if verify:
                if self.verify_test_file(file_path):
                    logger.info(f"✅ Verification passed: {file_path}")
                else:
                    logger.error(f"❌ Verification failed: {file_path}")
                    return False, file_path
            
            return True, file_path
        except Exception as e:
            logger.error(f"❌ Error generating test file for {model_name}: {e}")
            return False, ""
    
    def verify_test_file(self, file_path: str) -> bool:
        """
        Verify a generated test file.
        
        Args:
            file_path: Path to file to verify
            
        Returns:
            True if verification passed, False otherwise
        """
        # Check syntax
        syntax_valid = self.verify_syntax(file_path)
        
        # Check ModelTest pattern
        pattern_valid = self.verify_model_test_pattern(file_path)
        
        return syntax_valid and pattern_valid
    
    def verify_syntax(self, file_path: str) -> bool:
        """
        Verify Python syntax of a file.
        
        Args:
            file_path: Path to file to verify
            
        Returns:
            True if syntax is valid, False otherwise
        """
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
    
    def verify_model_test_pattern(self, file_path: str) -> bool:
        """
        Verify that a file follows the ModelTest pattern.
        
        Args:
            file_path: Path to file to verify
            
        Returns:
            True if file follows pattern, False otherwise
        """
        try:
            # Read the file
            with open(file_path, "r") as f:
                content = f.read()
            
            # Check for required imports/inheritance
            required_terms = [
                "ModelTest",
                "def get_default_model_id",
                "super().__init__",
                "model_type",
                "run_tests"
            ]
            
            for term in required_terms:
                if term not in content:
                    logger.error(f"❌ {file_path}: Missing required term '{term}'")
                    return False
            
            logger.info(f"✅ {file_path}: Follows ModelTest pattern")
            return True
        except Exception as e:
            logger.error(f"❌ {file_path}: Error validating ModelTest pattern: {e}")
            return False

    def generate_models_by_priority(self, priority: str = "high", verify: bool = True, force: bool = False) -> Tuple[int, int, int]:
        """
        Generate test files for models with the given priority.
        
        Args:
            priority: Priority level (high, medium, low, all)
            verify: Whether to verify generated files
            force: Whether to overwrite existing files
            
        Returns:
            Tuple of (num_generated, num_failed, total)
        """
        # Determine which models to generate
        models_to_generate = []
        if priority == "all":
            for p in PRIORITY_MODELS:
                models_to_generate.extend(PRIORITY_MODELS[p])
        else:
            models_to_generate = PRIORITY_MODELS.get(priority, [])
        
        # Generate each model
        generated = []
        failed = []
        
        for model_type in models_to_generate:
            logger.info(f"Generating test for {model_type}")
            success, file_path = self.generate_test_file(model_type, force=force, verify=verify)
            
            if success:
                generated.append(file_path)
            else:
                failed.append(model_type)
        
        # Print summary
        logger.info("\nGeneration Summary:")
        logger.info(f"- Generated: {len(generated)} files")
        logger.info(f"- Failed: {len(failed)} models")
        logger.info(f"- Total: {len(models_to_generate)} models")
        
        if failed:
            logger.info("\nFailed models:")
            for model in sorted(failed):
                logger.info(f"  - {model}")
        
        return len(generated), len(failed), len(models_to_generate)

    def generate_coverage_report(self, output_file: str = "model_test_coverage.md") -> bool:
        """
        Generate a coverage report for model tests.
        
        Args:
            output_file: Path to save report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Gather model information
            all_models = {}
            for priority, models in PRIORITY_MODELS.items():
                for model in models:
                    if model not in all_models:
                        all_models[model] = {
                            "priority": priority,
                            "architecture": get_architecture_type(model),
                            "has_test": False,
                            "test_file": None
                        }
            
            # Check for existing test files
            for model in all_models:
                model_snake = normalize_model_name(model)
                potential_files = [
                    f"test_hf_{model_snake}.py",
                    f"test_hf_{model}.py"
                ]
                
                for file in potential_files:
                    matches = glob.glob(f"**/test_hf_{model_snake}.py", recursive=True)
                    if matches:
                        all_models[model]["has_test"] = True
                        all_models[model]["test_file"] = matches[0]
                        break
            
            # Generate statistics
            total_models = len(all_models)
            models_with_tests = sum(1 for model in all_models.values() if model["has_test"])
            coverage_pct = (models_with_tests / total_models) * 100 if total_models > 0 else 0
            
            arch_stats = {}
            for arch_type in ARCHITECTURE_TYPES:
                models_in_arch = [m for m, info in all_models.items() if info["architecture"] == arch_type]
                models_with_tests_in_arch = [m for m in models_in_arch if all_models[m]["has_test"]]
                
                arch_stats[arch_type] = {
                    "total": len(models_in_arch),
                    "with_tests": len(models_with_tests_in_arch),
                    "coverage_pct": (len(models_with_tests_in_arch) / len(models_in_arch)) * 100 if len(models_in_arch) > 0 else 0
                }
            
            priority_stats = {}
            for priority in PRIORITY_MODELS:
                models_in_priority = [m for m, info in all_models.items() if info["priority"] == priority]
                models_with_tests_in_priority = [m for m in models_in_priority if all_models[m]["has_test"]]
                
                priority_stats[priority] = {
                    "total": len(models_in_priority),
                    "with_tests": len(models_with_tests_in_priority),
                    "coverage_pct": (len(models_with_tests_in_priority) / len(models_in_priority)) * 100 if len(models_in_priority) > 0 else 0
                }
            
            # Write report
            with open(output_file, "w") as f:
                f.write("# HuggingFace Model Test Coverage Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Overview\n\n")
                f.write(f"- **Total models**: {total_models}\n")
                f.write(f"- **Models with tests**: {models_with_tests} ({coverage_pct:.1f}%)\n")
                f.write(f"- **Models missing tests**: {total_models - models_with_tests}\n\n")
                
                f.write("## Coverage by Architecture\n\n")
                f.write("| Architecture | Total Models | Models with Tests | Coverage |\n")
                f.write("|--------------|--------------|-------------------|----------|\n")
                
                for arch_type, stats in arch_stats.items():
                    f.write(f"| {arch_type} | {stats['total']} | {stats['with_tests']} | {stats['coverage_pct']:.1f}% |\n")
                
                f.write("\n## Coverage by Priority\n\n")
                f.write("| Priority | Total Models | Models with Tests | Coverage |\n")
                f.write("|----------|--------------|-------------------|----------|\n")
                
                for priority, stats in priority_stats.items():
                    f.write(f"| {priority} | {stats['total']} | {stats['with_tests']} | {stats['coverage_pct']:.1f}% |\n")
                
                f.write("\n## Missing Tests\n\n")
                f.write("### High Priority\n\n")
                high_missing = [m for m, info in all_models.items() if info["priority"] == "high" and not info["has_test"]]
                if high_missing:
                    for model in sorted(high_missing):
                        f.write(f"- {model} ({all_models[model]['architecture']})\n")
                else:
                    f.write("✅ All high priority models have tests!\n")
                
                f.write("\n### Medium Priority\n\n")
                medium_missing = [m for m, info in all_models.items() if info["priority"] == "medium" and not info["has_test"]]
                if medium_missing:
                    for model in sorted(medium_missing):
                        f.write(f"- {model} ({all_models[model]['architecture']})\n")
                else:
                    f.write("✅ All medium priority models have tests!\n")
            
            logger.info(f"✅ Generated coverage report: {output_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Error generating coverage report: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    generator = ModelTestGenerator(output_dir="./generated_tests")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate HuggingFace model tests")
    parser.add_argument("--model", help="Model name to generate test for")
    parser.add_argument("--priority", choices=["high", "medium", "low", "all"], help="Generate tests for models with this priority")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing files")
    parser.add_argument("--verify", action="store_true", help="Verify generated files")
    
    args = parser.parse_args()
    
    if args.model:
        success, file_path = generator.generate_test_file(args.model, force=args.force, verify=args.verify)
        if success:
            print(f"✅ Generated test file: {file_path}")
        else:
            print(f"❌ Failed to generate test file for {args.model}")
    elif args.priority:
        generator.generate_models_by_priority(args.priority, verify=args.verify, force=args.force)
    else:
        parser.print_help()