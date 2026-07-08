#!/usr/bin/env python3
"""
Script to generate test files for missing models identified in
the HF_MODEL_COVERAGE_ROADMAP.md that are not yet implemented.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure enhanced_generator is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path is set
try:
    from enhanced_generator import (
        MODEL_REGISTRY, 
        generate_test, 
        get_model_architecture,
        validate_generated_file
    )
except ImportError as e:
    logger.error(f"Failed to import from enhanced_generator: {e}")
    logger.error("Make sure to update enhanced_generator.py first with update_model_registry.py")
    sys.exit(1)

# Define priority model lists from the roadmap
CRITICAL_PRIORITY_MODELS = [
    "gpt-j", 
    "flan-t5", 
    "xlm-roberta", 
    "vision-text-dual-encoder"
]

HIGH_PRIORITY_MODELS = [
    "codellama", 
    "gpt-neo", 
    "gpt-neox", 
    "qwen2", 
    "qwen3",
    "longt5", 
    "pegasus-x", 
    "luke", 
    "mpnet",
    "fuyu", 
    "kosmos-2", 
    "llava-next", 
    "video-llava",
    "bark", 
    "mobilenet-v2",
    "blip-2", 
    "chinese-clip", 
    "clipseg"
]

# Medium priority models are the remaining models in MODEL_REGISTRY
# that are not in CRITICAL_PRIORITY_MODELS or HIGH_PRIORITY_MODELS

def get_normalized_model_name(model_name: str) -> List[str]:
    """
    Generate normalized variants of model names to handle different formats.
    
    Args:
        model_name: The original model name
        
    Returns:
        List of possible name variations
    """
    variants = [model_name.lower()]
    
    # Handle hyphenated names
    if '-' in model_name:
        variants.append(model_name.replace('-', '_').lower())
        variants.append(model_name.replace('-', '').lower())
    
    # Handle special cases for common prefixes/suffixes
    if model_name.lower().startswith('t5-'):
        base = model_name.lower()[3:]  # Remove 't5-'
        variants.append(f"t5_{base}")
        variants.append(f"t5{base}")
    
    # Handle version numbers (e.g., v2 -> _v2, -v2)
    if 'v2' in model_name.lower() or 'v3' in model_name.lower():
        for v in ['v2', 'v3', 'v4']:
            if v in model_name.lower():
                without_v = model_name.lower().replace(v, '')
                variants.append(f"{without_v}_{v}")
                variants.append(f"{without_v}-{v}")
    
    return variants

def get_missing_models() -> Dict[str, List[str]]:
    """
    Identify missing models by comparing the roadmap priorities
    against what's currently implemented.
    """
    # Get lists of model names from the registry with all variants
    implemented_models = set()
    for model in MODEL_REGISTRY.keys():
        implemented_models.update(get_normalized_model_name(model))
    
    # Check critical priority models
    missing_critical = []
    for model in CRITICAL_PRIORITY_MODELS:
        variants = get_normalized_model_name(model)
        if not any(variant in implemented_models for variant in variants):
            missing_critical.append(model)
    
    # Check high priority models
    missing_high = []
    for model in HIGH_PRIORITY_MODELS:
        variants = get_normalized_model_name(model)
        if not any(variant in implemented_models for variant in variants):
            missing_high.append(model)
    
    # Get remaining models from the roadmap (medium priority)
    with open('skills/HF_MODEL_COVERAGE_ROADMAP.md', 'r') as f:
        roadmap_content = f.read()
    
    # Extract all models marked as implemented in the roadmap
    all_marked_implemented = set()
    for line in roadmap_content.splitlines():
        if line.startswith('- [x]'):
            try:
                model_name = line.split('- [x]')[1].strip().split(' ')[0].lower()
                model_name = model_name.rstrip(",:;.")  # Remove trailing punctuation
                all_marked_implemented.add(model_name)
            except IndexError:
                continue
    
    # Identify medium priority models not in critical or high
    medium_priority = all_marked_implemented - set(CRITICAL_PRIORITY_MODELS) - set(HIGH_PRIORITY_MODELS)
    missing_medium = []
    
    for model in medium_priority:
        variants = get_normalized_model_name(model)
        if not any(variant in implemented_models for variant in variants):
            missing_medium.append(model)
    
    return {
        "critical": missing_critical,
        "high": missing_high,
        "medium": missing_medium
    }

def determine_model_architecture(model_type: str) -> str:
    """
    Determine the model architecture based on the model name or by analyzing
    its similarities to other models.
    
    Args:
        model_type: The model type to analyze
        
    Returns:
        Architecture type string
    """
    # Use get_model_architecture from enhanced_generator
    arch = get_model_architecture(model_type)
    if arch != "unknown":
        return arch
    
    # Try with normalized variants
    for variant in get_normalized_model_name(model_type):
        arch = get_model_architecture(variant)
        if arch != "unknown":
            return arch
    
    # Analyze model name for common patterns
    model_lower = model_type.lower()
    
    # Enhanced encoder-only patterns
    if any(term in model_lower for term in ["bert", "electra", "albert", "roberta", "xlm", "deberta", 
                                          "funnel", "canine", "luke", "mpnet", "ernie", "xlnet", 
                                          "rembert", "convbert", "roformer", "bigbird", "esm", 
                                          "flaubert", "ibert", "nezha", "mra"]):
        return "encoder-only"
    
    # Enhanced decoder-only patterns
    elif any(term in model_lower for term in ["gpt", "llama", "mistral", "falcon", "phi", "starcoder", 
                                           "bloom", "opt", "mosaicmpt", "mosaic-mpt", "mosaic_mpt", 
                                           "codegen", "olmo", "olmoe", "nemotron", "mamba", 
                                           "stablelm", "codellama", "pythia", "xglm", "rwkv", 
                                           "recurrent-gemma", "command-r", "open-llama", "open_llama"]):
        return "decoder-only"
    
    # Enhanced encoder-decoder patterns
    elif any(term in model_lower for term in ["t5", "bart", "pegasus", "mbart", "led", "longt5", 
                                          "prophetnet", "bigbird-pegasus", "bigbird_pegasus", 
                                          "m2m-100", "m2m_100", "nllb", "switch-transformers", 
                                          "switch_transformers", "seamless-m4t", "seamless_m4t", 
                                          "umt5", "plbart"]):
        return "encoder-decoder"
    
    # Enhanced vision patterns
    elif any(term in model_lower for term in ["vit", "swin", "deit", "beit", "convnext", "resnet", 
                                          "dinov2", "segformer", "mask2former", "detr", "yolos", 
                                          "sam", "mobilenet", "efficientnet", "convnextv2", 
                                          "poolformer", "perceiver", "mobilevit", "cvt", "levit", 
                                          "conditional-detr", "swinv2", "imagegpt", "dinat", 
                                          "depth-anything", "vitdet", "van"]):
        return "vision"
    
    # Enhanced vision-text patterns
    elif any(term in model_lower for term in ["clip", "blip", "dual-encoder", "vilt", "xclip", 
                                           "vision-text", "vision_text", "vision-encoder-decoder", 
                                           "clipseg", "chinese-clip", "chinese_clip", "owlvit", 
                                           "siglip", "groupvit", "instructblip"]):
        return "vision-text"
    
    # Enhanced speech patterns
    elif any(term in model_lower for term in ["wav2vec", "whisper", "hubert", "speech", "bark", 
                                          "encodec", "clap", "musicgen", "speecht5", "wavlm", 
                                          "unispeech", "sew", "audioldm", "data2vec-audio", 
                                          "data2vec_audio"]):
        return "speech"
    
    # Enhanced multimodal patterns
    elif any(term in model_lower for term in ["llava", "paligemma", "idefics", "fuyu", "git", 
                                         "flava", "video-llava", "kosmos", "mllama", "imagebind", 
                                         "llava-next", "qwen-vl", "qwen_vl"]):
        return "multimodal"
    
    return "unknown"

def generate_missing_model_tests(output_dir: str, priority: str = "all") -> Dict[str, Any]:
    """
    Generate test files for missing models.
    
    Args:
        output_dir: Directory to output the generated files
        priority: Priority level to generate ("critical", "high", "medium", or "all")
        
    Returns:
        Dictionary with status information about generated files
    """
    missing_models = get_missing_models()
    results = {
        "generated": 0,
        "failed": 0,
        "skipped": 0,
        "details": {}
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which priority levels to process
    priorities_to_process = ["critical", "high", "medium"] if priority == "all" else [priority]
    
    for current_priority in priorities_to_process:
        if current_priority not in missing_models:
            logger.warning(f"No models found for priority: {current_priority}")
            continue
            
        models = missing_models[current_priority]
        logger.info(f"Processing {len(models)} {current_priority} priority models")
        
        for model_type in models:
            # Generate all normalized name variants 
            model_variants = get_normalized_model_name(model_type)
            
            success = False
            error_msg = ""
            best_result = None
            
            # Try different approaches for generation
            # 1. Try direct generation with each variant
            for variant in model_variants:
                try:
                    logger.info(f"Trying direct generation for {variant} (original: {model_type})")
                    result = generate_test(variant, output_dir)
                    
                    # Validate the generated file
                    is_valid, validation_msg = validate_generated_file(result["file_path"])
                    if is_valid:
                        logger.info(f"Successfully generated and validated test: {result['file_path']}")
                        best_result = result
                        success = True
                        break
                    else:
                        error_msg = f"Validation failed: {validation_msg}"
                except Exception as e:
                    error_msg = f"Direct generation failed: {str(e)}"
                    continue
            
            # 2. If direct generation failed, try to determine architecture first
            if not success:
                try:
                    # Determine the model architecture
                    arch = determine_model_architecture(model_type)
                    if arch != "unknown":
                        logger.info(f"Determined architecture {arch} for {model_type}, trying generation again")
                        # Check if generate_test accepts 'architecture' parameter
                        try:
                            # Try with the architecture parameter
                            result = generate_test(model_type, output_dir, architecture=arch)
                        except TypeError:
                            # If it fails, try without the architecture parameter
                            logger.info(f"generate_test() doesn't accept architecture parameter, trying without it")
                            result = generate_test(model_type, output_dir)
                        
                        # Validate the generated file
                        is_valid, validation_msg = validate_generated_file(result["file_path"])
                        if is_valid:
                            logger.info(f"Successfully generated and validated test: {result['file_path']}")
                            best_result = result
                            success = True
                        else:
                            error_msg = f"Architecture-based generation validation failed: {validation_msg}"
                    else:
                        error_msg = f"Could not determine architecture for {model_type}"
                except Exception as e:
                    error_msg = f"Architecture-based generation failed: {str(e)}"
            
            # Update results
            if success:
                results["generated"] += 1
                results["details"][model_type] = {
                    "status": "success", 
                    "file_path": best_result["file_path"],
                    "architecture": best_result["architecture"],
                    "variants_tried": model_variants
                }
            else:
                logger.error(f"Failed to generate test for {model_type}: {error_msg}")
                results["failed"] += 1
                results["details"][model_type] = {
                    "status": "failed", 
                    "error": error_msg,
                    "variants_tried": model_variants
                }
    
    # Generate a report
    report_path = os.path.join(output_dir, "generation_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Model Test Generation Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Generated:** {results['generated']}\n")
        f.write(f"- **Failed:** {results['failed']}\n")
        f.write(f"- **Skipped:** {results['skipped']}\n\n")
        
        f.write(f"## Details\n\n")
        
        # First show successful generations
        if any(details["status"] == "success" for details in results["details"].values()):
            f.write(f"### Successfully Generated Models\n\n")
            for model_type, details in sorted(results["details"].items()):
                if details["status"] == "success":
                    f.write(f"#### ✅ {model_type}\n")
                    f.write(f"- Architecture: {details['architecture']}\n")
                    f.write(f"- File path: {details['file_path']}\n")
                    f.write(f"- Variants tried: {', '.join(details['variants_tried'])}\n\n")
        
        # Then show failed generations
        if any(details["status"] == "failed" for details in results["details"].values()):
            f.write(f"### Failed Generations\n\n")
            for model_type, details in sorted(results["details"].items()):
                if details["status"] == "failed":
                    f.write(f"#### ❌ {model_type}\n")
                    f.write(f"- Error: {details['error']}\n")
                    f.write(f"- Variants tried: {', '.join(details['variants_tried'])}\n\n")
    
    logger.info(f"Generation report written to: {report_path}")
    return results

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate test files for missing models")
    parser.add_argument("--output-dir", default="missing_model_tests", 
                        help="Directory to output the generated files")
    parser.add_argument("--priority", choices=["critical", "high", "medium", "all"], 
                        default="all", help="Priority level to generate")
    args = parser.parse_args()
    
    results = generate_missing_model_tests(args.output_dir, args.priority)
    
    logger.info(f"Summary:")
    logger.info(f"  Generated: {results['generated']}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Skipped: {results['skipped']}")
    
    return 0 if results["failed"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())