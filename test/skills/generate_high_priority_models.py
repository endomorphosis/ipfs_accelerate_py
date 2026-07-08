#!/usr/bin/env python3
"""
Script to generate tests for high-priority HuggingFace models.
This script focuses on the Phase 2 high-priority models from the roadmap.
"""

import os
import sys
import json
import glob
import logging
import subprocess
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"high_priority_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SKILLS_DIR)
TEMP_DIR = os.path.join(SKILLS_DIR, "temp_generated")
COVERAGE_DIR = os.path.join(SKILLS_DIR, "coverage_visualizations")

# These can be overridden by command-line arguments
OUTPUT_DIR = PARENT_DIR

# High priority models from Phase 2 of the roadmap
HIGH_PRIORITY_MODELS = {
    # Text Models
    "roberta": {
        "template": "bert",
        "architecture": "encoder-only",
        "default_model": "roberta-base",
        "tasks": ["fill-mask"],
    },
    "albert": {
        "template": "bert",
        "architecture": "encoder-only",
        "default_model": "albert-base-v2",
        "tasks": ["fill-mask"],
    },
    "distilbert": {
        "template": "bert",
        "architecture": "encoder-only",
        "default_model": "distilbert-base-uncased",
        "tasks": ["fill-mask"],
    },
    "deberta": {
        "template": "bert",
        "architecture": "encoder-only",
        "default_model": "microsoft/deberta-base",
        "tasks": ["fill-mask"],
    },
    "bart": {
        "template": "t5",
        "architecture": "encoder-decoder",
        "default_model": "facebook/bart-base",
        "tasks": ["summarization", "translation"],
    },
    "llama": {
        "template": "gpt2",
        "architecture": "decoder-only",
        "default_model": "meta-llama/Llama-2-7b-hf",
        "tasks": ["text-generation"],
    },
    "mistral": {
        "template": "gpt2",
        "architecture": "decoder-only",
        "default_model": "mistralai/Mistral-7B-v0.1",
        "tasks": ["text-generation"],
    },
    "phi": {
        "template": "gpt2",
        "architecture": "decoder-only",
        "default_model": "microsoft/phi-2",
        "tasks": ["text-generation"],
    },
    "falcon": {
        "template": "gpt2",
        "architecture": "decoder-only",
        "default_model": "tiiuae/falcon-7b",
        "tasks": ["text-generation"],
    },
    "mpt": {
        "template": "gpt2", 
        "architecture": "decoder-only",
        "default_model": "mosaicml/mpt-7b",
        "tasks": ["text-generation"],
    },
    
    # Vision Models
    "swin": {
        "template": "vit",
        "architecture": "vision",
        "default_model": "microsoft/swin-base-patch4-window7-224-in22k",
        "tasks": ["image-classification"],
    },
    "deit": {
        "template": "vit",
        "architecture": "vision",
        "default_model": "facebook/deit-base-patch16-224",
        "tasks": ["image-classification"],
    },
    "resnet": {
        "template": "vit",
        "architecture": "vision",
        "default_model": "microsoft/resnet-50",
        "tasks": ["image-classification"],
    },
    "convnext": {
        "template": "vit",
        "architecture": "vision",
        "default_model": "facebook/convnext-base-224-22k-1k",
        "tasks": ["image-classification"],
    },
    
    # Multimodal Models
    "clip": {
        "template": "clip",
        "architecture": "multimodal",
        "default_model": "openai/clip-vit-base-patch32",
        "tasks": ["zero-shot-image-classification"],
    },
    "blip": {
        "template": "clip",
        "architecture": "multimodal",
        "default_model": "Salesforce/blip-image-captioning-base",
        "tasks": ["image-to-text"],
    },
    "llava": {
        "template": "clip",
        "architecture": "multimodal",
        "default_model": "llava-hf/llava-1.5-7b-hf",
        "tasks": ["visual-question-answering"],
    },
    
    # Audio Models
    "whisper": {
        "template": "wav2vec2",
        "architecture": "audio",
        "default_model": "openai/whisper-base",
        "tasks": ["automatic-speech-recognition"],
    },
    "wav2vec2": {
        "template": "wav2vec2",
        "architecture": "audio",
        "default_model": "facebook/wav2vec2-base",
        "tasks": ["automatic-speech-recognition"],
    },
    "hubert": {
        "template": "wav2vec2",
        "architecture": "audio",
        "default_model": "facebook/hubert-base-ls960",
        "tasks": ["automatic-speech-recognition"],
    },
}

def get_implemented_models(output_dir=OUTPUT_DIR):
    """Get list of implemented models from test files."""
    test_files = glob.glob(os.path.join(output_dir, "test_hf_*.py"))
    implemented = []
    
    for test_file in test_files:
        model_name = os.path.basename(test_file).replace('test_hf_', '').replace('.py', '')
        implemented.append(model_name)
    
    logger.info(f"Found {len(implemented)} implemented models")
    return implemented

def get_missing_high_priority_models(implemented_models):
    """Get high priority models that have not been implemented yet."""
    missing = []
    
    for model, config in HIGH_PRIORITY_MODELS.items():
        if model not in implemented_models:
            missing.append(model)
    
    logger.info(f"Found {len(missing)} missing high-priority models")
    return missing

def generate_test_for_model(model_name, output_dir=OUTPUT_DIR):
    """Generate a test for a specific model."""
    model_config = HIGH_PRIORITY_MODELS.get(model_name)
    
    if not model_config:
        logger.error(f"Model {model_name} not found in high priority models list")
        return False
    
    template_model = model_config["template"]
    
    # Create temp directory if needed
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Run generator script
    test_generator_path = os.path.join(PARENT_DIR, "test_generator.py")
    if not os.path.exists(test_generator_path):
        test_generator_path = os.path.join(SKILLS_DIR, "test_generator.py")
    
    command = [
        sys.executable,
        test_generator_path,
        "--family", model_name,
        "--template", template_model,
        "--output", TEMP_DIR
    ]
    
    logger.info(f"Generating test for {model_name} using template {template_model}...")
    
    try:
        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error generating test for {model_name}: {result.stderr}")
            logger.error(f"Command output: {result.stdout}")
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
        output_file = os.path.join(output_dir, f"test_hf_{model_name}.py")
        with open(test_file, 'r') as src, open(output_file, 'w') as dst:
            dst.write(src.read())
        
        logger.info(f"✅ Successfully generated test for {model_name}")
        return True
    
    except Exception as e:
        logger.error(f"Exception generating test for {model_name}: {e}")
        return False

def generate_batch(missing_models, max_models=None, num_workers=4, output_dir=OUTPUT_DIR):
    """Generate tests for a batch of missing models."""
    if max_models:
        batch = missing_models[:max_models]
    else:
        batch = missing_models
    
    results = {}
    successful = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_model = {executor.submit(generate_test_for_model, model, output_dir): model for model in batch}
        
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

def update_model_coverage_roadmap(successful_models):
    """Update the HF_MODEL_COVERAGE_ROADMAP.md file to mark successful models as implemented."""
    roadmap_path = os.path.join(SKILLS_DIR, "HF_MODEL_COVERAGE_ROADMAP.md")
    
    if not os.path.exists(roadmap_path):
        logger.warning(f"Roadmap file not found: {roadmap_path}")
        return
    
    try:
        with open(roadmap_path, 'r') as f:
            content = f.read()
        
        # Create a backup of the original file
        backup_path = f"{roadmap_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_path, 'w') as f:
            f.write(content)
        
        # Process each model one by one to ensure more precise replacements
        updated_content = content
        for model in successful_models:
            # Try different patterns to find the right line
            patterns = [
                f"- [ ] {model} ",  # Standard format with space
                f"- [ ] {model}(",  # Format with opening parenthesis
                f"- [ ] {model}\n"  # Format at end of line
            ]
            
            for pattern in patterns:
                if pattern in updated_content:
                    replacement = pattern.replace("[ ]", "[x]")
                    updated_content = updated_content.replace(pattern, replacement)
                    logger.info(f"Updated status for model: {model}")
                    break
            
            # For case-insensitive matching (e.g., "RoBERTa" vs "roberta")
            if model.lower() != model:
                for pattern in patterns:
                    capitalized_pattern = pattern.replace(model, model[0].upper() + model[1:])
                    if capitalized_pattern in updated_content:
                        replacement = capitalized_pattern.replace("[ ]", "[x]")
                        updated_content = updated_content.replace(capitalized_pattern, replacement)
                        logger.info(f"Updated status for model (capitalized): {model}")
                        break
        
        # Write the updated content back to the file
        with open(roadmap_path, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"Updated roadmap file: {roadmap_path}")
    
    except Exception as e:
        logger.error(f"Error updating roadmap file: {e}")

def update_coverage_statistic(successful_count):
    """Update the model coverage percentage in CLAUDE.md."""
    claude_path = os.path.join(PARENT_DIR, "CLAUDE.md")
    
    if not os.path.exists(claude_path):
        logger.warning(f"CLAUDE.md file not found: {claude_path}")
        return
    
    try:
        # Get current implementation count from HF_MODEL_COVERAGE_ROADMAP.md
        roadmap_path = os.path.join(SKILLS_DIR, "HF_MODEL_COVERAGE_ROADMAP.md")
        current_implemented = 0
        total_models = 315  # From roadmap
        
        if os.path.exists(roadmap_path):
            with open(roadmap_path, 'r') as f:
                content = f.read()
                current_implemented = content.count("[x]")
        
        # Add new successful implementations
        new_implemented = current_implemented + successful_count
        percentage = int((new_implemented / total_models) * 100)
        
        # Update CLAUDE.md
        with open(claude_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            if "**Priority 2: Comprehensive HuggingFace Model Testing (300+ classes)**" in line and "IN PROGRESS" in line:
                # Update the progress percentage
                updated_line = line.replace(f"(🔄 IN PROGRESS - 40%)", f"(🔄 IN PROGRESS - {percentage}%)")
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)
        
        with open(claude_path, 'w') as f:
            f.writelines(updated_lines)
        
        logger.info(f"Updated coverage statistic in CLAUDE.md to {percentage}%")
    
    except Exception as e:
        logger.error(f"Error updating coverage statistic: {e}")

def generate_coverage_report():
    """Generate an updated coverage report."""
    try:
        coverage_tool = os.path.join(SKILLS_DIR, "visualize_test_coverage.py")
        if not os.path.exists(coverage_tool):
            logger.warning(f"Coverage tool not found: {coverage_tool}")
            return
        
        os.makedirs(COVERAGE_DIR, exist_ok=True)
        
        subprocess.run([
            sys.executable,
            coverage_tool,
            "--output-dir", COVERAGE_DIR
        ])
        
        logger.info(f"Generated updated coverage report in {COVERAGE_DIR}")
    
    except Exception as e:
        logger.error(f"Error generating coverage report: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate tests for high-priority HuggingFace models")
    parser.add_argument("--max-models", type=int, help="Maximum number of models to generate")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for generated tests")
    parser.add_argument("--list-missing", action="store_true", help="List missing high-priority models without generating tests")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--specific-model", type=str, help="Generate test for a specific high-priority model")
    parser.add_argument("--skip-update", action="store_true", help="Skip updating the roadmap and coverage files")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    # Get implemented models
    implemented_models = get_implemented_models(output_dir)
    
    # Get missing high-priority models
    missing_models = get_missing_high_priority_models(implemented_models)
    
    if args.list_missing:
        logger.info(f"Missing high-priority models ({len(missing_models)}):")
        for i, model in enumerate(missing_models, 1):
            config = HIGH_PRIORITY_MODELS[model]
            logger.info(f"{i}. {model} (architecture: {config['architecture']}, template: {config['template']})")
        return
    
    # Handle specific model request
    if args.specific_model:
        if args.specific_model in HIGH_PRIORITY_MODELS:
            success = generate_test_for_model(args.specific_model, output_dir)
            if success and not args.skip_update:
                update_model_coverage_roadmap([args.specific_model])
                update_coverage_statistic(1)
                generate_coverage_report()
            return
        else:
            logger.error(f"Model {args.specific_model} is not in the high-priority list")
            return
    
    # Generate tests for missing high-priority models
    if not missing_models:
        logger.info("All high-priority models have already been implemented!")
        return
    
    logger.info(f"Generating tests for {len(missing_models)} missing high-priority models...")
    
    # Create temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Generate tests
    max_models = args.max_models if args.max_models else len(missing_models)
    results, successful, failed = generate_batch(missing_models, max_models, args.workers, output_dir)
    
    logger.info("\nGeneration complete!")
    logger.info(f"Successfully generated: {len(successful)}/{len(missing_models[:max_models])}")
    logger.info(f"Failed: {len(failed)}/{len(missing_models[:max_models])}")
    
    if failed:
        logger.info("\nFailed models:")
        for model in failed:
            logger.info(f"- {model}")
    
    # Update roadmap and coverage files
    if successful and not args.skip_update:
        update_model_coverage_roadmap(successful)
        update_coverage_statistic(len(successful))
        generate_coverage_report()

if __name__ == "__main__":
    main()