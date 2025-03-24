#!/usr/bin/env python3
"""
Generate HuggingFace model tests for all model architectures.

This script:
1. Detects model architecture using introspection and pattern matching
2. Selects the appropriate test class from model_test_base
3. Generates standardized test files compliant with ModelTest pattern
4. Validates generated files for syntax and pattern compliance
5. Creates model coverage reports

Usage:
    python generate_model_tests.py --model MODEL_NAME
    python generate_model_tests.py --priority {high,medium,low,all}
    python generate_model_tests.py --report
"""

import os
import sys
import glob
import json
import logging
import argparse
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

# Import model_test_base
try:
    from model_test_base import (
        ModelTest, EncoderOnlyModelTest, DecoderOnlyModelTest, 
        EncoderDecoderModelTest, VisionModelTest, get_model_test_class
    )
    HAS_MODEL_TEST_BASE = True
except ImportError:
    logger.error("model_test_base.py not found. Make sure it's in the same directory.")
    HAS_MODEL_TEST_BASE = False

# Define architecture types for model detection
ARCHITECTURE_TYPES = {
    "encoder-only": [
        # Core encoder models
        "bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "xlm_roberta",
        # Additional encoder models
        "albert", "canine", "ernie", "layoutlm", "rembert", "squeezebert", "funnel", "reformer", 
        "mpt", "xlnet", "bigbird", "longformer", "roformer", "tapas", "flava"
    ],
    "decoder-only": [
        # Core decoder models
        "gpt2", "gpt-j", "gptj", "gpt-neo", "gpt_neo", "bloom", "llama", "mistral", "falcon", "phi",
        # Additional decoder models
        "opt", "gptj", "ctrl", "transfo-xl", "gemma", "codellama"
    ],
    "encoder-decoder": [
        # Core encoder-decoder models
        "t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5",
        # Additional encoder-decoder models
        "blenderbot", "m2m100", "prophetnet", "opus_mt"
    ],
    "vision": [
        # Core vision models
        "vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2",
        # Additional vision models
        "bit", "dpt", "levit", "regnet", "segformer", "efficientnet", "donut", "mobilevit",
        "mlp-mixer", "yolos", "mask2former", "detr", "sam", "resnet"
    ],
    "vision-encoder-text-decoder": [
        # Core vision-text models
        "vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip", "blip2", "blip-2",
        # Additional vision-text models
        "vilt", "vinvl", "align", "florence", "paligemma", "donut", "git"
    ],
    "speech": [
        # Core speech models
        "wav2vec2", "hubert", "whisper", "bark", "speecht5",
        # Additional speech models
        "wavlm", "data2vec", "unispeech", "unispeech_sat", "unispeech-sat", "sew", "sew_d", 
        "sew-d", "usm", "seamless_m4t", "clap", "encodec", "musicgen"
    ],
    "multimodal": [
        # Core multimodal models
        "llava", "clip", "blip", "git", "pix2struct",
        # Additional multimodal models
        "idefics", "flava", "flamingo", "imagebind"
    ]
}

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
        "whisper", "wav2vec2", "hubert"
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
        "sew", "unispeech", "clap", "musicgen", "encodec"
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
        "wavlm", "data2vec", "unispeech-sat", "hubert", "sew-d", "usm", "seamless_m4t"
    ]
}

# Map for replacing hyphenated names with underscore versions
MODEL_NAME_MAPPING = {
    "xlm-roberta": "xlm_roberta",
    "gpt-neo": "gpt_neo",
    "gpt-j": "gptj",
    "blip-2": "blip2",
    "sew-d": "sew_d",
    "opus-mt": "opus_mt",
    "m2m_100": "m2m100",
    "unispeech-sat": "unispeech_sat"
}


def detect_architecture_type(model_name: str) -> str:
    """
    Detect architecture type of a model using multiple detection methods.
    
    Args:
        model_name: Name of the model to detect
        
    Returns:
        Architecture type as a string
    """
    # Normalize model name
    model_name_lower = model_name.lower()
    
    # Try introspection with transformers
    try:
        import transformers
        if hasattr(transformers, "AutoConfig"):
            config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            # Check model architecture based on config properties
            # Check if it's a text-to-text model (encoder-decoder)
            if hasattr(config, "is_encoder_decoder") and config.is_encoder_decoder:
                return "encoder-decoder"
            
            # Check if it's a vision model
            if hasattr(config, "num_channels") and config.num_channels > 0:
                # Check if it also has text components
                if hasattr(config, "vocab_size") and config.vocab_size > 0:
                    return "vision-encoder-text-decoder"
                return "vision"
            
            # Check if it's a speech model
            if "speech" in type(config).__name__.lower() or "audio" in type(config).__name__.lower():
                return "speech"
            
            # Check if it's a decoder-only model
            if hasattr(config, "is_decoder") and config.is_decoder:
                return "decoder-only"
            
            # Default to encoder-only for remaining text models
            return "encoder-only"
    except Exception:
        # Introspection failed, use pattern matching
        pass
    
    # Match by model name patterns
    for arch_type, models in ARCHITECTURE_TYPES.items():
        if any(model.lower() in model_name_lower for model in models):
            return arch_type
    
    # Default to encoder-only if unknown
    return "encoder-only"


def get_template_for_architecture(arch_type: str) -> Dict[str, str]:
    """
    Get the appropriate template details for an architecture type.
    
    Args:
        arch_type: Architecture type
        
    Returns:
        Dictionary with template details (base_class, imports, etc.)
    """
    templates = {
        "encoder-only": {
            "base_class": "EncoderOnlyModelTest",
            "imports": [
                "from model_test_base import EncoderOnlyModelTest"
            ],
            "test_class_name": "Test{model_type_pascal}Model",
            "task": "fill-mask",
            "test_input": "The [MASK] runs quickly."
        },
        "decoder-only": {
            "base_class": "DecoderOnlyModelTest",
            "imports": [
                "from model_test_base import DecoderOnlyModelTest"
            ],
            "test_class_name": "Test{model_type_pascal}Model",
            "task": "text-generation",
            "test_input": "Once upon a time"
        },
        "encoder-decoder": {
            "base_class": "EncoderDecoderModelTest",
            "imports": [
                "from model_test_base import EncoderDecoderModelTest"
            ],
            "test_class_name": "Test{model_type_pascal}Model",
            "task": "text2text-generation",
            "test_input": "translate English to French: Hello, how are you?"
        },
        "vision": {
            "base_class": "VisionModelTest",
            "imports": [
                "from model_test_base import VisionModelTest",
                "from PIL import Image"
            ],
            "test_class_name": "Test{model_type_pascal}Model",
            "task": "image-classification",
            "test_input": "dummy_image = Image.new('RGB', (224, 224), color='red')"
        },
        "vision-encoder-text-decoder": {
            "base_class": "VisionTextModelTest",
            "imports": [
                "from model_test_base import ModelTest",
                "from PIL import Image"
            ],
            "test_class_name": "Test{model_type_pascal}Model",
            "task": "image-to-text",
            "test_input": "dummy_image = Image.new('RGB', (224, 224), color='red')"
        },
        "speech": {
            "base_class": "SpeechModelTest",
            "imports": [
                "from model_test_base import ModelTest",
                "import numpy as np"
            ],
            "test_class_name": "Test{model_type_pascal}Model",
            "task": "automatic-speech-recognition",
            "test_input": "dummy_audio = np.zeros(16000)"
        },
        "multimodal": {
            "base_class": "MultimodalModelTest",
            "imports": [
                "from model_test_base import ModelTest",
                "from PIL import Image",
                "import numpy as np"
            ],
            "test_class_name": "Test{model_type_pascal}Model",
            "task": "multimodal",
            "test_input": "text_input = 'Describe this image', image_input = Image.new('RGB', (224, 224), color='red')"
        }
    }
    
    # Default to encoder-only if not found
    return templates.get(arch_type, templates["encoder-only"])


def convert_to_snake_case(name: str) -> str:
    """Convert a string to snake_case."""
    # Replace hyphens with underscores
    name = name.replace('-', '_')
    
    # Apply mapping for special cases
    if name in MODEL_NAME_MAPPING:
        return MODEL_NAME_MAPPING[name]
    
    return name


def convert_to_pascal_case(name: str) -> str:
    """Convert a string to PascalCase."""
    # Replace hyphens with underscores and split
    parts = name.replace('-', '_').split('_')
    
    # Capitalize each part
    return ''.join(part.capitalize() for part in parts)


def generate_test_file(model_type: str, output_dir: str = "generated_tests", overwrite: bool = False) -> Tuple[bool, str]:
    """
    Generate a test file for a specific model type.
    
    Args:
        model_type: Type of model to generate a test for
        output_dir: Directory to save the generated file
        overwrite: Whether to overwrite existing files
        
    Returns:
        Tuple of (success, output_path)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply name mapping for special cases
    model_type_safe = convert_to_snake_case(model_type)
    
    # Generate file path
    file_path = os.path.join(output_dir, f"test_hf_{model_type_safe}.py")
    
    # Check if file already exists and should not be overwritten
    if os.path.exists(file_path) and not overwrite:
        logger.info(f"Skipping {file_path} - file already exists (use --force to overwrite)")
        return False, file_path
    
    # Detect architecture type
    arch_type = detect_architecture_type(model_type)
    logger.info(f"Detected architecture type {arch_type} for model {model_type}")
    
    # Get template for architecture
    template = get_template_for_architecture(arch_type)
    
    # Generate model-specific variables
    model_type_pascal = convert_to_pascal_case(model_type_safe)
    test_class_name = template["test_class_name"].format(model_type_pascal=model_type_pascal)
    
    # Generate test file content
    content = f"""#!/usr/bin/env python3
\"\"\"
Test file for {model_type.upper()} model.

This test verifies the functionality of the {model_type} model,
which is a {arch_type} architecture.
\"\"\"

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import model test base
{"".join(f"{imp}\n" for imp in template["imports"])}


class {test_class_name}({template["base_class"]}):
    \"\"\"Test class for {model_type} model.\"\"\"
    
    def __init__(self, model_id=None, device=None):
        \"\"\"Initialize the test.\"\"\"
        # Set model type explicitly
        self.model_type = "{model_type_safe}"
        self.task = "{template["task"]}"
        self.architecture_type = "{arch_type}"
        
        # Call parent initializer
        super().__init__(model_id, device)
    
    def get_default_model_id(self) -> str:
        \"\"\"Get the default model ID for this model type.\"\"\"
        return "{model_type}"
    
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
    
    try:
        # Write the file
        with open(file_path, "w") as f:
            f.write(content)
            
        logger.info(f"✅ Generated test file: {file_path}")
        return True, file_path
    except Exception as e:
        logger.error(f"❌ Error generating test file: {e}")
        return False, file_path


def verify_syntax(file_path: str) -> bool:
    """
    Verify Python syntax of a file.
    
    Args:
        file_path: Path to the file to verify
        
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


def verify_model_test_pattern(file_path: str) -> bool:
    """
    Verify that a file follows the ModelTest pattern.
    
    Args:
        file_path: Path to the file to verify
        
    Returns:
        True if the file follows the pattern, False otherwise
    """
    try:
        # Read the file
        with open(file_path, "r") as f:
            content = f.read()
            
        # Check for required imports
        if "from model_test_base import" not in content:
            logger.error(f"❌ {file_path}: Missing import from model_test_base")
            return False
            
        # Check for required methods
        required_methods = [
            "get_default_model_id",
            "run_all_tests"
        ]
        
        for method in required_methods:
            if f"def {method}" not in content:
                logger.error(f"❌ {file_path}: Missing required method {method}")
                return False
                
        logger.info(f"✅ {file_path}: Follows ModelTest pattern")
        return True
    except Exception as e:
        logger.error(f"❌ {file_path}: Error validating pattern: {e}")
        return False


def generate_models_by_priority(priority: str, output_dir: str = "generated_tests", verify: bool = True, overwrite: bool = False) -> Tuple[int, int, int]:
    """
    Generate test files for models with the given priority.
    
    Args:
        priority: Priority level (high, medium, low, all)
        output_dir: Directory to save generated files
        verify: Whether to verify generated files
        overwrite: Whether to overwrite existing files
        
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
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate each model
    generated = []
    failed = []
    
    for model_type in models_to_generate:
        logger.info(f"Generating test for {model_type}")
        success, file_path = generate_test_file(model_type, output_dir, overwrite)
        
        if success:
            generated.append(file_path)
            
            # Verify if requested
            if verify:
                syntax_valid = verify_syntax(file_path)
                pattern_valid = verify_model_test_pattern(file_path)
                
                if not syntax_valid or not pattern_valid:
                    failed.append(file_path)
                    logger.error(f"❌ {file_path}: Failed verification")
        else:
            failed.append(model_type)
    
    # Print summary
    logger.info("\nGeneration Summary:")
    logger.info(f"- Generated: {len(generated)} files")
    logger.info(f"- Failed: {len(failed)}")
    logger.info(f"- Total: {len(models_to_generate)}")
    
    if failed:
        logger.info("\nFailed models/files:")
        for f in failed:
            logger.info(f"  - {f}")
    
    return len(generated), len(failed), len(models_to_generate)


def generate_coverage_report(output_file: str = "model_test_coverage.md") -> bool:
    """
    Generate a coverage report for model tests.
    
    Args:
        output_file: Path to save the report to
        
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
                        "architecture": detect_architecture_type(model),
                        "has_test": False,
                        "test_file": None
                    }
        
        # Check for existing test files
        for model in all_models:
            model_snake = convert_to_snake_case(model)
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate HuggingFace model tests")
    
    # Model selection options
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Specific model to generate a test for")
    model_group.add_argument("--priority", type=str, choices=["high", "medium", "low", "all"], help="Generate tests for models with this priority")
    model_group.add_argument("--report", action="store_true", help="Generate a coverage report")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="generated_tests", help="Directory to save generated files")
    parser.add_argument("--verify", action="store_true", help="Verify generated files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    # Check requirements
    if not HAS_MODEL_TEST_BASE:
        logger.error("model_test_base.py not found. Please make sure it's in the same directory.")
        return 1
    
    # Generate a single model test
    if args.model:
        logger.info(f"Generating test for model: {args.model}")
        success, file_path = generate_test_file(args.model, args.output_dir, args.force)
        
        if success and args.verify:
            syntax_valid = verify_syntax(file_path)
            pattern_valid = verify_model_test_pattern(file_path)
            
            if not syntax_valid or not pattern_valid:
                logger.error(f"❌ {file_path}: Failed verification")
                return 1
        
        return 0 if success else 1
    
    # Generate tests by priority
    elif args.priority:
        logger.info(f"Generating tests for {args.priority} priority models")
        generated, failed, total = generate_models_by_priority(
            args.priority, 
            args.output_dir, 
            args.verify, 
            args.force
        )
        
        return 0 if failed == 0 else 1
    
    # Generate coverage report
    elif args.report:
        logger.info("Generating coverage report")
        success = generate_coverage_report()
        
        return 0 if success else 1
    
    # No operation specified
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())