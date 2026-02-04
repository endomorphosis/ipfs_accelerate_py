#!/usr/bin/env python3
"""
Script to generate tests for missing HuggingFace models in batches.
This script identifies missing model tests and generates them in batches.
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"batch_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_TYPES_JSON = "huggingface_model_types.json"
OUTPUT_DIR = "../"
TEMP_DIR = "temp_generated"

# Architecture mapping
ARCHITECTURE_MAPPING = {
    # Encoder-only models
    "bert": "encoder_only",
    "roberta": "encoder_only",
    "albert": "encoder_only",
    "electra": "encoder_only",
    "camembert": "encoder_only",
    "distilbert": "encoder_only",
    "xlm": "encoder_only",
    
    # Decoder-only models
    "gpt2": "decoder_only",
    "gptj": "decoder_only",
    "gpt_neo": "decoder_only",
    "llama": "decoder_only",
    "opt": "decoder_only",
    "falcon": "decoder_only",
    "mistral": "decoder_only",
    "phi": "decoder_only",
    
    # Encoder-decoder models
    "t5": "encoder_decoder",
    "bart": "encoder_decoder",
    "pegasus": "encoder_decoder",
    "mbart": "encoder_decoder",
    "m2m_100": "encoder_decoder",
    "led": "encoder_decoder",
    
    # Vision models
    "vit": "vision",
    "swin": "vision",
    "beit": "vision",
    "deit": "vision",
    "convnext": "vision",
    "sam": "vision",
    
    # Multimodal models
    "clip": "multimodal",
    "blip": "multimodal",
    "llava": "multimodal",
    "flava": "multimodal",
    "idefics": "multimodal",
    
    # Audio models
    "wav2vec2": "audio",
    "hubert": "audio",
    "whisper": "audio",
    "clap": "audio",
    "encodec": "audio"
}

def load_model_types():
    """Load model types from JSON file."""
    if os.path.exists(MODEL_TYPES_JSON):
        with open(MODEL_TYPES_JSON, 'r') as f:
            return json.load(f)
    
    # If model types file doesn't exist, return a subset
    logger.warning(f"Model types file {MODEL_TYPES_JSON} not found. Using default subset.")
    return [
        "bert", "gpt2", "t5", "vit", "roberta", "llama", "mistral", "falcon", 
        "phi", "bart", "wav2vec2", "whisper", "clip", "blip"
    ]

def get_implemented_models():
    """Get list of implemented models from test files."""
    test_files = glob.glob(os.path.join(OUTPUT_DIR, "test_hf_*.py"))
    implemented = []
    
    for test_file in test_files:
        model_name = os.path.basename(test_file).replace('test_hf_', '').replace('.py', '')
        implemented.append(model_name)
    
    return implemented

def get_missing_models(model_types, implemented_models):
    """Get list of missing model tests."""
    return [model for model in model_types if model not in implemented_models]

def guess_architecture(model_name):
    """Guess the architecture of a model based on its name."""
    for prefix, arch in ARCHITECTURE_MAPPING.items():
        if model_name == prefix or model_name.startswith(prefix + "_"):
            return arch
    
    # Default to encoder_only if unknown
    return "encoder_only"

def generate_test_for_model(model_name, template_model=None):
    """Generate a test for a specific model."""
    
    # If template model is not specified, guess the architecture and use an appropriate template
    if template_model is None:
        arch = guess_architecture(model_name)
        
        # Use an appropriate template for each architecture
        if arch == "encoder_only":
            template_model = "bert"
        elif arch == "decoder_only":
            template_model = "gpt2"
        elif arch == "encoder_decoder":
            template_model = "t5"
        elif arch == "vision":
            template_model = "vit"
        elif arch == "multimodal":
            template_model = "clip"
        elif arch == "audio":
            template_model = "wav2vec2"
        else:
            template_model = "bert"  # Default
    
    # Create temp directory if needed
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Run generator script
    command = [
        sys.executable,
        os.path.join(OUTPUT_DIR, "test_generator.py"),
        "--family", model_name,
        "--template", template_model,
        "--output", TEMP_DIR
    ]
    
    logger.info(f"Generating test for {model_name} using template {template_model}...")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error generating test for {model_name}: {result.stderr}")
            return False
        
        # Verify syntax
        test_file = os.path.join(TEMP_DIR, f"test_hf_{model_name}.py")
        if not os.path.exists(test_file):
            logger.error(f"Generated test file does not exist: {test_file}")
            return False
        
        syntax_check = subprocess.run(
            [sys.executable, "-m", "py_compile", test_file],
            capture_output=True,
            text=True
        )
        
        if syntax_check.returncode != 0:
            logger.error(f"Syntax check failed for {model_name}: {syntax_check.stderr}")
            return False
        
        # Copy to output directory
        output_file = os.path.join(OUTPUT_DIR, f"test_hf_{model_name}.py")
        with open(test_file, 'r') as src, open(output_file, 'w') as dst:
            dst.write(src.read())
        
        logger.info(f"✅ Successfully generated test for {model_name}")
        return True
    
    except Exception as e:
        logger.error(f"Exception generating test for {model_name}: {e}")
        return False

def generate_batch(missing_models, batch_size, num_workers=4):
    """Generate tests for a batch of missing models."""
    batch = missing_models[:batch_size]
    
    results = {}
    successful = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_model = {executor.submit(generate_test_for_model, model): model for model in batch}
        
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                success = future.result()
                results[model] = success
                
                if success:
                    successful.append(model)
                else:
                    failed.append(model)
            
            except Exception as e:
                logger.error(f"Exception processing {model}: {e}")
                results[model] = False
                failed.append(model)
    
    return results, successful, failed

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate tests for missing HuggingFace models")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of tests to generate in a batch")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for generated tests")
    parser.add_argument("--temp-dir", type=str, default=TEMP_DIR,
                        help="Temporary directory for generated files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--all", action="store_true",
                        help="Generate tests for all missing models")
    parser.add_argument("--list-missing", action="store_true",
                        help="List missing models without generating tests")
    
    args = parser.parse_args()
    
    global OUTPUT_DIR, TEMP_DIR
    OUTPUT_DIR = args.output_dir
    TEMP_DIR = args.temp_dir
    
    # Load model types
    logger.info("Loading model types...")
    model_types = load_model_types()
    
    # Get implemented models
    logger.info("Getting implemented models...")
    implemented_models = get_implemented_models()
    
    # Get missing models
    missing_models = get_missing_models(model_types, implemented_models)
    
    if args.list_missing:
        logger.info(f"Missing models ({len(missing_models)}):")
        for i, model in enumerate(missing_models, 1):
            arch = guess_architecture(model)
            logger.info(f"{i}. {model} (architecture: {arch})")
        return
    
    logger.info(f"Total models: {len(model_types)}")
    logger.info(f"Implemented models: {len(implemented_models)}")
    logger.info(f"Missing models: {len(missing_models)}")
    
    if not missing_models:
        logger.info("No missing models to generate!")
        return
    
    # Create directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Generate tests
    if args.all:
        logger.info(f"Generating tests for all {len(missing_models)} missing models...")
        
        successful_all = []
        failed_all = []
        
        for i in range(0, len(missing_models), args.batch_size):
            batch = missing_models[i:i+args.batch_size]
            logger.info(f"Processing batch {i//args.batch_size + 1} ({len(batch)} models)...")
            
            results, successful, failed = generate_batch(batch, len(batch), args.workers)
            
            successful_all.extend(successful)
            failed_all.extend(failed)
            
            logger.info(f"Batch {i//args.batch_size + 1} results: {len(successful)} successful, {len(failed)} failed")
        
        logger.info("\nGeneration complete!")
        logger.info(f"Successfully generated: {len(successful_all)}/{len(missing_models)}")
        logger.info(f"Failed: {len(failed_all)}/{len(missing_models)}")
        
        if failed_all:
            logger.info("\nFailed models:")
            for model in failed_all:
                logger.info(f"- {model}")
    
    else:
        # Generate a single batch
        batch_size = min(args.batch_size, len(missing_models))
        logger.info(f"Generating tests for {batch_size} missing models...")
        
        results, successful, failed = generate_batch(missing_models, batch_size, args.workers)
        
        logger.info("\nBatch generation complete!")
        logger.info(f"Successfully generated: {len(successful)}/{batch_size}")
        logger.info(f"Failed: {len(failed)}/{batch_size}")
        
        if failed:
            logger.info("\nFailed models:")
            for model in failed:
                logger.info(f"- {model}")
    
    # Generate coverage report
    try:
        logger.info("\nGenerating updated coverage report...")
        subprocess.run([
            sys.executable,
            "visualize_test_coverage.py",
            "--output-dir", "coverage_visualizations"
        ])
    except Exception as e:
        logger.error(f"Error generating coverage report: {e}")

if __name__ == "__main__":
    main()