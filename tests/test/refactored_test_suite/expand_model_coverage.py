#!/usr/bin/env python3
"""
Expand model test coverage to include more HuggingFace models.

This script analyzes the transformers library and generates test files for
models that are not yet covered by the existing test suite, with the goal
of achieving 300+ model coverage.
"""

import os
import sys
import time
import json
import logging
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"expand_coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Import required modules
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add current directory to Python path for imports
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add generators directory to path
generators_dir = os.path.join(current_dir, "generators")
if generators_dir not in sys.path:
    sys.path.insert(0, generators_dir)

try:
    # Import directly from paths using importlib
    architecture_detector_path = os.path.join(generators_dir, "architecture_detector.py")
    test_generator_path = os.path.join(generators_dir, "test_generator.py")
    track_implementation_progress_path = os.path.join(current_dir, "track_implementation_progress.py")
    
    # Import architecture_detector
    import importlib.util
    spec = importlib.util.spec_from_file_location("architecture_detector", architecture_detector_path)
    architecture_detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(architecture_detector)
    
    get_architecture_type = architecture_detector.get_architecture_type
    normalize_model_name = architecture_detector.normalize_model_name
    get_model_metadata = architecture_detector.get_model_metadata
    MODEL_NAME_MAPPING = architecture_detector.MODEL_NAME_MAPPING
    ARCHITECTURE_TYPES = architecture_detector.ARCHITECTURE_TYPES
    
    logger.info("Successfully imported architecture_detector module from path")
    
    # Import test_generator
    spec = importlib.util.spec_from_file_location("test_generator", test_generator_path)
    test_generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_generator)
    
    ModelTestGenerator = test_generator.ModelTestGenerator
    
    logger.info("Successfully imported test_generator module from path")
    
    # Import track_implementation_progress
    spec = importlib.util.spec_from_file_location("track_implementation_progress", track_implementation_progress_path)
    track_implementation_progress = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(track_implementation_progress)
    
    find_implemented_tests = track_implementation_progress.find_implemented_tests
    get_required_models = track_implementation_progress.get_required_models
    get_implementation_status = track_implementation_progress.get_implementation_status
    
    logger.info("Successfully imported track_implementation_progress module from path")
except Exception as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error(f"Current directory: {current_dir}")
    logger.error(f"Generators directory: {generators_dir}")
    logger.error(f"Python path: {sys.path}")
    sys.exit(1)

# Define additional models to expand coverage
ADDITIONAL_MODELS = {
    # New encoder-only models
    "encoder-only": [
        "albert", "bert", "bert-japanese", "big_bird", "bigbird_pegasus", "biogpt", "bloom", 
        "camembert", "canine", "chinese_clip", "clip", "clipseg", "code_llama", "codegen", 
        "convbert", "convnext", "cpmant", "ctrl", "deberta", "deberta_v2", "decision_transformer", 
        "distilbert", "dpr", "electra", "ernie", "esm", "flaubert", "fnet", "funnel", 
        "gptj", "herbert", "ibert", "layoutlm", "layoutlmv2", "layoutlmv3", "led", "lilt", 
        "longformer", "luke", "lxmert", "markuplm", "mega", "megatron_bert", "mobilebert", 
        "mpnet", "nystromformer", "openai", "perceiver", "qdqbert", "reformer", "rembert", 
        "roberta", "roberta_prelayernorm", "roformer", "squeezebert", "tapas", "tapex", 
        "transfo_xl", "visual_bert", "xglm", "xlm", "xlm_prophetnet", "xlm_roberta", "xlnet", 
        "xmod", "yoso", "bart", "bert_generation", "bigbird", "plbart"
    ],
    
    # New decoder-only models
    "decoder-only": [
        "gpt2", "gpt_neo", "gpt_neox", "falcon", "opt", "mpt", "llama", "llama2", "mistral", 
        "qwen", "qwen2", "gemma", "phi", "stablelm", "pythia", "dolly", "bloom", "baichuan", 
        "cerebras", "cohere", "cpm", "ctranslate2", "ctrl", "galactica", "gpt_bigcode", 
        "codegen", "codellama", "starcoder", "starcoder2", "santacoder", "incoder", "bloom", 
        "rwkv", "jais", "open_llama", "bloomz", "phi-1", "phi-1.5", "phi-2", "phi-3", "mixtral"
    ],
    
    # New encoder-decoder models
    "encoder-decoder": [
        "t5", "mt5", "bart", "mbart", "pegasus", "pegasus_x", "marian", "m2m_100", "nllb", 
        "opus_mt", "led", "longt5", "bigbird_pegasus", "prophetnet", "xlm_prophetnet",
        "fsmt", "flant5", "ulmt5", "switchtransformers", "flan-t5", "flan-ul2", "palm",
        "mms", "seamless_m4t", "seamless_m4t_v2"
    ],
    
    # New vision models
    "vision": [
        "vit", "deit", "beit", "swin", "convnext", "resnet", "regnet", "mobilenet_v1", 
        "mobilenet_v2", "efficientnet", "bit", "dpt", "segformer", "detr", "yolos", 
        "mask2former", "sam", "dinov2", "poolformer", "convnextv2", "levit", "mlp_mixer", 
        "mobilevit", "perceiver", "pvt", "segformer", "swiftformer", "timm_backbone", 
        "van", "videomae", "vit_hybrid", "vitdet", "vitmatte", "vitmsn", "bridgetower", 
        "deta", "dinat", "conditional_detr", "convnext", "data2vec_vision", "dino", "dit", 
        "donut_swin", "efficientformer", "focalnet", "nat", "owlvit", "perceiver_image"
    ],
    
    # New vision-encoder-text-decoder models
    "vision-encoder-text-decoder": [
        "clip", "blip", "blip2", "git", "donut", "pix2struct", "layoutlmv3", "lilt", 
        "vilt", "vinvl", "align", "bridgetower", "chinese_clip", "clipseg", "owlvit", 
        "siglip", "groupvit", "albef", "alt_clip", "clap", "florence"
    ],
    
    # New speech models
    "speech": [
        "whisper", "wav2vec2", "hubert", "wavlm", "unispeech", "unispeech_sat", "sew", 
        "sew_d", "encodec", "clap", "musicgen", "usm", "seamless_m4t", "data2vec_audio", 
        "mctct", "mms", "seamlessm4t", "speech_to_text", "speecht5", "umt5", "wav2vec2_phoneme", 
        "wavlm", "whisper", "xlsr_wav2vec2"
    ],
    
    # New multimodal models
    "multimodal": [
        "llava", "flava", "idefics", "paligemma", "imagebind", "flamingo", "blip2", 
        "fuyu", "instructblip", "kosmos2", "mgp_str", "mplug_owl", "mplug_owl2", "siglip", 
        "trocr", "vitdet", "xclip", "xvlm", "clvp", "clip_vision_model", "donut", "fuyu"
    ]
}

def get_all_models() -> List[str]:
    """
    Get all models from both MODEL_NAME_MAPPING and ADDITIONAL_MODELS.
    
    Returns:
        List of all model names
    """
    all_models = set()
    
    # Add models from MODEL_NAME_MAPPING
    for model_name in MODEL_NAME_MAPPING.keys():
        all_models.add(model_name)
    
    # Add models from ADDITIONAL_MODELS
    for architecture, models in ADDITIONAL_MODELS.items():
        for model_name in models:
            all_models.add(model_name)
    
    return sorted(list(all_models))

def get_models_by_architecture() -> Dict[str, List[str]]:
    """
    Get models grouped by architecture.
    
    Returns:
        Dict of architecture to list of model names
    """
    models_by_arch = {arch: [] for arch in ARCHITECTURE_TYPES}
    
    # Add models from MODEL_NAME_MAPPING
    for model_name, architecture in MODEL_NAME_MAPPING.items():
        if architecture in models_by_arch:
            models_by_arch[architecture].append(model_name)
    
    # Add models from ADDITIONAL_MODELS
    for architecture, models in ADDITIONAL_MODELS.items():
        if architecture in models_by_arch:
            # Add only models that are not already in the list
            for model_name in models:
                if model_name not in models_by_arch[architecture]:
                    models_by_arch[architecture].append(model_name)
    
    return models_by_arch

def analyze_coverage():
    """
    Analyze current coverage and identify missing models.
    
    Returns:
        Dict with coverage analysis
    """
    # Get implemented models
    implemented = find_implemented_tests(["./generated_tests"])
    
    # Get all models
    all_models = get_all_models()
    
    # Get models by architecture
    models_by_architecture = get_models_by_architecture()
    
    # Calculate totals
    total_models = len(all_models)
    total_implemented = len(implemented)
    
    # Calculate by architecture
    by_architecture = {}
    for architecture, models in models_by_architecture.items():
        total_in_arch = len(models)
        implemented_in_arch = 0
        
        missing_in_arch = []
        for model_name in models:
            normalized_name = normalize_model_name(model_name)
            if normalized_name in implemented:
                implemented_in_arch += 1
            else:
                missing_in_arch.append(model_name)
        
        percentage = (implemented_in_arch / total_in_arch) * 100 if total_in_arch > 0 else 0
        
        by_architecture[architecture] = {
            "total": total_in_arch,
            "implemented": implemented_in_arch,
            "percentage": percentage,
            "missing": missing_in_arch
        }
    
    # Calculate overall percentage
    percentage = (total_implemented / total_models) * 100 if total_models > 0 else 0
    
    # Identify missing models
    missing_models = []
    for model_name in all_models:
        normalized_name = normalize_model_name(model_name)
        if normalized_name not in implemented:
            missing_models.append(model_name)
    
    return {
        "total_models": total_models,
        "total_implemented": total_implemented,
        "percentage": percentage,
        "missing_models": missing_models,
        "by_architecture": by_architecture
    }

def print_coverage_report(analysis: Dict[str, Any]):
    """
    Print coverage report.
    
    Args:
        analysis: Coverage analysis
    """
    print("\n====== HuggingFace Model Test Coverage Report ======\n")
    
    # Print overall stats
    print(f"Total models: {analysis['total_models']}")
    print(f"Implemented models: {analysis['total_implemented']}")
    print(f"Coverage percentage: {analysis['percentage']:.1f}%")
    print(f"Missing models: {len(analysis['missing_models'])}")
    
    # Print by architecture
    print("\nCoverage by architecture:")
    for architecture, stats in analysis['by_architecture'].items():
        print(f"  {architecture}: {stats['implemented']}/{stats['total']} ({stats['percentage']:.1f}%)")
    
    # Print missing models by architecture
    print("\nMissing models by architecture:")
    for architecture, stats in analysis['by_architecture'].items():
        if len(stats['missing']) > 0:
            print(f"\n  {architecture} ({len(stats['missing'])} missing):")
            for model_name in sorted(stats['missing'])[:10]:  # Show only top 10
                print(f"    - {model_name}")
            
            if len(stats['missing']) > 10:
                print(f"    - ... and {len(stats['missing']) - 10} more")

def generate_missing_tests(architecture: str, num_models: int = 10):
    """
    Generate test files for missing models of a specific architecture.
    
    Args:
        architecture: Architecture type
        num_models: Number of models to generate tests for
    """
    # Get coverage analysis
    analysis = analyze_coverage()
    
    # Get missing models for the specified architecture
    missing_models = analysis['by_architecture'].get(architecture, {}).get('missing', [])
    
    if not missing_models:
        logger.info(f"No missing models for architecture: {architecture}")
        return
    
    # Limit number of models to generate
    models_to_generate = missing_models[:num_models]
    
    # Create generator
    generator = ModelTestGenerator(output_dir="./generated_tests")
    
    # Generate test files
    for model_name in models_to_generate:
        logger.info(f"Generating test for model: {model_name}")
        
        try:
            success, file_path = generator.generate_test_file(model_name, force=True)
            
            if success:
                logger.info(f"✅ Successfully generated test for {model_name}: {file_path}")
            else:
                logger.error(f"❌ Failed to generate test for {model_name}")
        except Exception as e:
            logger.error(f"Error generating test for {model_name}: {e}")

def generate_all_missing_tests(limit_per_arch: int = 50):
    """
    Generate test files for all missing models.
    
    Args:
        limit_per_arch: Limit number of models per architecture
    """
    for architecture in ARCHITECTURE_TYPES:
        logger.info(f"Generating tests for architecture: {architecture}")
        generate_missing_tests(architecture, limit_per_arch)

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Expand model test coverage")
    
    # Command groups
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--analyze", action="store_true", help="Analyze current coverage")
    group.add_argument("--generate", type=str, choices=ARCHITECTURE_TYPES + ["all"], help="Generate tests for a specific architecture or 'all'")
    
    # Options
    parser.add_argument("--num", type=int, default=10, help="Number of models to generate tests for")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        if args.analyze:
            # Analyze coverage
            analysis = analyze_coverage()
            print_coverage_report(analysis)
            
            # Calculate models needed to reach 300
            target = 300
            current = analysis['total_implemented']
            needed = max(0, target - current)
            
            print(f"\nTo reach {target}+ model coverage, need to implement {needed} more models.")
            
            return 0
        
        elif args.generate:
            if args.generate == "all":
                # Generate tests for all architectures
                generate_all_missing_tests(args.num)
            else:
                # Generate tests for specific architecture
                generate_missing_tests(args.generate, args.num)
            
            # Re-analyze coverage after generation
            analysis = analyze_coverage()
            print_coverage_report(analysis)
            
            return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())