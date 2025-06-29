#!/usr/bin/env python3
"""
Track implementation progress for model tests.

This script generates a report on progress toward implementing tests for all 
required models, categorized by architecture and priority.
"""

import os
import sys
import glob
import json
import logging
import datetime
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"implementation_progress_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Define architecture patterns (since we can't rely on import)
ARCHITECTURE_PATTERNS = {
    "encoder-only": ["bert", "roberta", "albert", "distilbert", "electra", "deberta", "funnel", "xlm-roberta"],
    "decoder-only": ["gpt2", "gpt-neo", "gpt-j", "llama", "falcon", "mistral", "phi", "gemma"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "mt5", "led", "prophetnet"],
    "vision": ["vit", "swin", "beit", "deit", "resnet", "convnext", "detr", "sam"],
    "vision-encoder-text-decoder": ["clip", "blip", "git", "donut"],
    "speech": ["whisper", "wav2vec2", "hubert", "encodec", "clap", "musicgen"],
    "multimodal": ["llava", "flava", "idefics", "paligemma"]
}

# Define priority models (since we can't rely on import)
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
        "whisper", "wav2vec2", "wavtovec2", "wave2vec2", "hubert" # Different name variations
    ],
    "medium": [
        # Text Models
        "xlm-roberta", "electra", "ernie", "rembert", "gpt-neo", "gpt-j", 
        "gptj", "gpt_j", # Different name variations
        "opt", "gemma", "mbart", "pegasus", "prophetnet", "led",
        # Vision Models
        "beit", "segformer", "detr", "mask2former", "yolos", "sam", "dinov2",
        "dino-v2", "dino_v2", # Different name variations
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

def normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name to a standard format.
    
    Args:
        model_name: Model name
        
    Returns:
        Normalized model name
    """
    # Debug logging
    original_name = model_name
    
    # Extract the base model name (remove organization)
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    
    # Remove version numbers and sizes
    model_name = re.sub(r"-\d+b.*$", "", model_name.lower())
    model_name = re.sub(r"\.?v\d+.*$", "", model_name)
    
    # Handle common prefixes
    prefixes = ["hf-", "hf_", "huggingface-", "huggingface_"]
    for prefix in prefixes:
        if model_name.startswith(prefix):
            model_name = model_name[len(prefix):]
    
    # Remove common version suffixes
    suffixes = ["-base", "-small", "-large", "-tiny", "-mini", "-medium"]
    for suffix in suffixes:
        if model_name.endswith(suffix):
            model_name = model_name[:-len(suffix)]
    
    # Handle special cases directly in normalization
    special_case_mapping = {
        "wav2vec2": ["wav2vec2", "wavtovec2", "wave2vec2", "wav-2-vec-2", "wav_2_vec_2"],
        "gpt_j": ["gpt-j", "gptj", "gpt_j"],
        "dinov2": ["dinov2", "dino-v2", "dino_v2", "dino_v_2", "dino-v-2"]
    }
    
    # Check if the model name is one of our special case variants
    for base_name, variants in special_case_mapping.items():
        if model_name in variants:
            logger.info(f"Normalized special case: {original_name} -> {base_name} (matched variant {model_name})")
            return base_name
    
    # Normalize hyphens (do this last to avoid interfering with special case matching)
    model_name = model_name.replace("-", "_")
    
    # Debug logging for normalization
    if original_name != model_name:
        logger.info(f"Normalized: {original_name} -> {model_name}")
    
    return model_name

def find_implemented_tests(search_dirs: List[str], pattern: str = "test_hf_*.py") -> Dict[str, List[str]]:
    """
    Find implemented tests across multiple directories.
    
    Args:
        search_dirs: List of directories to search
        pattern: Glob pattern for test files
        
    Returns:
        Dict of model name to test file list
    """
    implemented = {}
    special_case_mapping = {
        # Map normalized names from filenames to expected models in PRIORITY_MODELS
        "wav2vec2": ["wav2vec2", "wavtovec2", "wave2vec2"],
        "gpt_j": ["gpt-j", "gptj", "gpt_j"],
        "dinov2": ["dinov2", "dino-v2", "dino_v2"]
    }
    
    for search_dir in search_dirs:
        # Skip if directory doesn't exist
        if not os.path.exists(search_dir):
            logger.warning(f"Directory not found: {search_dir}")
            continue
        
        # Find all test files
        for root, _, files in os.walk(search_dir):
            matches = []
            for file in files:
                if file.startswith("test_hf_") and file.endswith(".py"):
                    matches.append(os.path.join(root, file))
            
            # Process matches
            for match in matches:
                # Extract model name from file name
                file_name = os.path.basename(match)
                model_name = file_name[len("test_hf_"):-len(".py")]
                
                # Add to implemented dict
                if model_name not in implemented:
                    implemented[model_name] = []
                implemented[model_name].append(match)
                
                # Add special case mappings
                for original, variants in special_case_mapping.items():
                    if model_name.lower() == original.lower():  # Case-insensitive comparison
                        logger.info(f"Found special case match for {model_name}: {original} with variants {variants}")
                        for variant in variants:
                            if variant not in implemented:
                                implemented[variant] = []
                            implemented[variant].append(match)
    
    # Debug logging to check the implemented dictionary
    logger.info(f"Special case models we're looking for: wav2vec2, dinov2, dino-v2, dino_v2")
    logger.info(f"Models found in implemented dictionary: {sorted(implemented.keys())}")
    
    # Double-check specific special cases
    for key, variants in special_case_mapping.items():
        if key in implemented:
            logger.info(f"Special case {key} is recognized with {len(implemented[key])} test files")
        else:
            logger.warning(f"Special case {key} was not recognized")
        
        for variant in variants:
            if variant in implemented:
                logger.info(f"Special case variant {variant} is recognized with {len(implemented[variant])} test files")
            else:
                logger.warning(f"Special case variant {variant} was not recognized")
    
    return implemented

def get_required_models() -> Dict[str, Dict[str, Any]]:
    """
    Get all required model tests.
    
    Returns:
        Dict of model name to model info
    """
    required = {}
    
    # Add all models from priority lists
    for priority, models in PRIORITY_MODELS.items():
        for model_name in models:
            if model_name not in required:
                required[model_name] = {
                    "priority": priority,
                    "architecture": None
                }
    
    # Determine architecture for each model
    for model_name in required:
        architecture = None
        
        # Check against architecture patterns
        for arch, patterns in ARCHITECTURE_PATTERNS.items():
            for pattern in patterns:
                if pattern in model_name:
                    architecture = arch
                    break
            if architecture:
                break
        
        # Set architecture (default to encoder-only if not found)
        required[model_name]["architecture"] = architecture or "encoder-only"
    
    return required

def get_models_by_architecture() -> Dict[str, List[str]]:
    """
    Get required models grouped by architecture.
    
    Returns:
        Dict of architecture to list of model names
    """
    by_architecture = {}
    
    # Initialize with known architectures
    for arch in ARCHITECTURE_PATTERNS.keys():
        by_architecture[arch] = []
    
    # Group required models by architecture
    required = get_required_models()
    
    for model_name, info in required.items():
        architecture = info["architecture"]
        if architecture not in by_architecture:
            by_architecture[architecture] = []
        by_architecture[architecture].append(model_name)
    
    return by_architecture

def get_models_by_priority() -> Dict[str, List[str]]:
    """
    Get required models grouped by priority.
    
    Returns:
        Dict of priority to list of model names
    """
    by_priority = {
        "high": [],
        "medium": [],
        "low": []
    }
    
    # Group required models by priority
    required = get_required_models()
    
    for model_name, info in required.items():
        priority = info["priority"]
        by_priority[priority].append(model_name)
    
    return by_priority

def get_implementation_status(required: Dict[str, Dict[str, Any]], implemented: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Get implementation status.
    
    Args:
        required: Dict of required models
        implemented: Dict of implemented models
        
    Returns:
        Dict with implementation status
    """
    # Initialize status
    status = {
        "total_required": len(required),
        "total_implemented": 0,
        "percentage": 0.0,
        "missing": {},
        "by_architecture": {},
        "by_priority": {}
    }
    
    # Log all implemented models for debugging
    logger.info(f"All implemented models: {sorted(implemented.keys())}")
    
    # Special case direct mapping
    special_case_mapping = {
        "wav2vec2": ["wav2vec2", "wavtovec2", "wave2vec2", "wav-2-vec-2", "wav_2_vec_2"],
        "gpt_j": ["gpt-j", "gptj", "gpt_j"],
        "dinov2": ["dinov2", "dino-v2", "dino_v2", "dino_v_2", "dino-v-2"]
    }
    
    # Check which models are implemented
    for model_name, info in required.items():
        normalized_name = normalize_model_name(model_name)
        
        # Direct lookup first
        implemented_flag = normalized_name in implemented
        
        # If not found, try matching against all normalized keys in implemented
        if not implemented_flag:
            for impl_name in implemented.keys():
                normalized_impl = normalize_model_name(impl_name)
                if normalized_impl == normalized_name:
                    implemented_flag = True
                    logger.info(f"Found match after normalization: {model_name} -> {impl_name}")
                    break
        
        # If still not found, try special case variants
        if not implemented_flag:
            for original, variants in special_case_mapping.items():
                if model_name in variants or normalized_name in variants:
                    # Check if the original or any other variant is implemented
                    if original in implemented:
                        implemented_flag = True
                        logger.info(f"Found special case match: {model_name} -> {original}")
                        break
                    for variant in variants:
                        if variant in implemented:
                            implemented_flag = True
                            logger.info(f"Found variant match: {model_name} -> {variant}")
                            break
        
        logger.info(f"Model {model_name} (normalized: {normalized_name}): {'✅ IMPLEMENTED' if implemented_flag else '❌ MISSING'}")
        
        # Update total implemented
        if implemented_flag:
            status["total_implemented"] += 1
        else:
            # Add to missing
            status["missing"][model_name] = info
        
        # Update by architecture
        architecture = info["architecture"]
        if architecture not in status["by_architecture"]:
            status["by_architecture"][architecture] = {
                "total": 0,
                "implemented": 0,
                "percentage": 0.0
            }
        
        status["by_architecture"][architecture]["total"] += 1
        if implemented_flag:
            status["by_architecture"][architecture]["implemented"] += 1
        
        # Update by priority
        priority = info["priority"]
        if priority not in status["by_priority"]:
            status["by_priority"][priority] = {
                "total": 0,
                "implemented": 0,
                "percentage": 0.0
            }
        
        status["by_priority"][priority]["total"] += 1
        if implemented_flag:
            status["by_priority"][priority]["implemented"] += 1
    
    # Calculate percentages
    if status["total_required"] > 0:
        status["percentage"] = (status["total_implemented"] / status["total_required"]) * 100
    
    for architecture, info in status["by_architecture"].items():
        if info["total"] > 0:
            info["percentage"] = (info["implemented"] / info["total"]) * 100
    
    for priority, info in status["by_priority"].items():
        if info["total"] > 0:
            info["percentage"] = (info["implemented"] / info["total"]) * 100
    
    return status

def generate_report(status: Dict[str, Any], output_file: str) -> None:
    """
    Generate implementation status report.
    
    Args:
        status: Implementation status
        output_file: Output file path
    """
    with open(output_file, "w") as f:
        f.write("# Model Test Implementation Progress\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write overall progress
        f.write("## Overall Progress\n\n")
        f.write(f"- **Total required models**: {status['total_required']}\n")
        f.write(f"- **Models with tests**: {status['total_implemented']}\n")
        f.write(f"- **Implementation percentage**: {status['percentage']:.1f}%\n")
        f.write(f"- **Missing models**: {len(status['missing'])}\n\n")
        
        # Write progress bar
        progress = int(status["percentage"] / 10)
        f.write("```\n")
        f.write(f"[{'#' * progress}{' ' * (10 - progress)}] {status['percentage']:.1f}%\n")
        f.write("```\n\n")
        
        # Write by architecture
        f.write("## Progress by Architecture\n\n")
        f.write("| Architecture | Required | Implemented | Percentage |\n")
        f.write("|--------------|----------|-------------|------------|\n")
        
        for architecture, info in sorted(status["by_architecture"].items()):
            f.write(f"| {architecture} | {info['total']} | {info['implemented']} | {info['percentage']:.1f}% |\n")
        
        # Write by priority
        f.write("\n## Progress by Priority\n\n")
        f.write("| Priority | Required | Implemented | Percentage |\n")
        f.write("|----------|----------|-------------|------------|\n")
        
        for priority in ["high", "medium", "low"]:
            info = status["by_priority"].get(priority, {"total": 0, "implemented": 0, "percentage": 0.0})
            f.write(f"| {priority} | {info['total']} | {info['implemented']} | {info['percentage']:.1f}% |\n")
        
        # Write missing high-priority models
        f.write("\n## Missing High-Priority Models\n\n")
        
        missing_high = [model for model, info in status["missing"].items() if info["priority"] == "high"]
        if missing_high:
            for model in sorted(missing_high):
                info = status["missing"][model]
                f.write(f"- {model} ({info['architecture']})\n")
        else:
            f.write("✅ All high-priority models have tests!\n")
        
        # Write missing medium-priority models
        f.write("\n## Missing Medium-Priority Models\n\n")
        
        missing_medium = [model for model, info in status["missing"].items() if info["priority"] == "medium"]
        if missing_medium:
            for model in sorted(missing_medium):
                info = status["missing"][model]
                f.write(f"- {model} ({info['architecture']})\n")
        else:
            f.write("✅ All medium-priority models have tests!\n")
        
        # Write next steps
        f.write("\n## Next Steps\n\n")
        
        if missing_high:
            f.write("1. **Implement High-Priority Models**: Focus on implementing tests for remaining high-priority models:\n")
            for model in sorted(missing_high)[:5]:  # Show up to 5
                info = status["missing"][model]
                f.write(f"   - {model} ({info['architecture']})\n")
            if len(missing_high) > 5:
                f.write(f"   - ...and {len(missing_high) - 5} more\n")
        elif missing_medium:
            f.write("1. **Implement Medium-Priority Models**: Focus on implementing tests for medium-priority models:\n")
            for model in sorted(missing_medium)[:5]:  # Show up to 5
                info = status["missing"][model]
                f.write(f"   - {model} ({info['architecture']})\n")
            if len(missing_medium) > 5:
                f.write(f"   - ...and {len(missing_medium) - 5} more\n")
        else:
            missing_low = [model for model, info in status["missing"].items() if info["priority"] == "low"]
            if missing_low:
                f.write("1. **Implement Low-Priority Models**: Focus on implementing tests for low-priority models:\n")
                for model in sorted(missing_low)[:5]:  # Show up to 5
                    info = status["missing"][model]
                    f.write(f"   - {model} ({info['architecture']})\n")
                if len(missing_low) > 5:
                    f.write(f"   - ...and {len(missing_low) - 5} more\n")
            else:
                f.write("1. **Enhance Existing Tests**: All required models have tests! Focus on enhancing test coverage.\n")
        
        f.write("\n2. **Validate Test Coverage**: Ensure all tests cover key functionality:\n")
        f.write("   - Model loading\n")
        f.write("   - Input processing\n")
        f.write("   - Forward pass / inference\n")
        f.write("   - Output validation\n")
        
        f.write("\n3. **Integration Tests**: Implement integration tests for model interactions.\n")
        
        f.write("\n4. **CI/CD Integration**: Ensure all tests run successfully in CI/CD pipelines.\n")
        
        f.write("\n5. **Performance Testing**: Add performance benchmarks for key models.\n")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Track implementation progress for model tests")
    
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["./generated_tests", "./models"],
        help="Directories to search for test files"
    )
    
    parser.add_argument(
        "--output",
        default="implementation_progress.md",
        help="Output file path"
    )
    
    return parser.parse_args()

def main():
    """Command-line entry point."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Find implemented tests
        implemented = find_implemented_tests(args.dirs)
        logger.info(f"Found {len(implemented)} implemented models")
        
        # Get required models
        required = get_required_models()
        logger.info(f"Found {len(required)} required models")
        
        # Get implementation status
        status = get_implementation_status(required, implemented)
        
        # Generate report
        generate_report(status, args.output)
        logger.info(f"Generated implementation progress report: {args.output}")
        
        # Return 0 if all required models are implemented, 1 otherwise
        return 0 if status["total_implemented"] == status["total_required"] else 1
    
    except Exception as e:
        logger.error(f"Error tracking implementation progress: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())