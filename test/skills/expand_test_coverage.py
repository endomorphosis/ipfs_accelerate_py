#!/usr/bin/env python3

"""
Expand test coverage to additional models, focusing on hyphenated model names.

This script:
1. Uses find_models.py to discover all available HuggingFace models
2. Adds hyphenated models and other missing models to the MODEL_REGISTRY
3. Updates CLASS_NAME_FIXES with proper capitalization patterns
4. Generates test files for all hyphenated models and other prioritized models
5. Validates and fixes the generated test files

Usage:
    python expand_test_coverage.py [--hyphenated-only] [--output-dir OUTPUT_DIR] [--verify]
"""

import os
import sys
import json
import subprocess
import logging
import argparse
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FIXED_TESTS_DIR = CURRENT_DIR / "fixed_tests"
GENERATOR_PATH = CURRENT_DIR / "test_generator_fixed.py"

def run_find_models(hyphenated_only=True, output_file=None):
    """Run the find_models.py script to discover models."""
    find_models_path = CURRENT_DIR / "find_models.py"
    
    if not os.path.exists(find_models_path):
        logger.error(f"Could not find find_models.py at {find_models_path}")
        return None
    
    # Create temporary output file if none provided
    if not output_file:
        output_file = CURRENT_DIR / f"model_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Build command
    cmd = [sys.executable, str(find_models_path)]
    if hyphenated_only:
        cmd.append("--hyphenated-only")
    cmd.extend(["--output", str(output_file)])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the command
        process = subprocess.run(cmd, check=True, text=True, capture_output=True)
        logger.info(f"find_models.py output: {process.stdout}")
        
        # Check if the output file was created
        if os.path.exists(output_file):
            logger.info(f"Model data saved to {output_file}")
            return output_file
        else:
            logger.error(f"Expected output file {output_file} not created")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running find_models.py: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def update_generator(model_data_file):
    """Update test_generator_fixed.py with new models and class name fixes."""
    logger.info(f"Updating generator with data from {model_data_file}")
    
    # Read the model data
    try:
        with open(model_data_file, 'r') as f:
            model_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading model data: {e}")
        return False
    
    # Read the current generator content
    try:
        with open(GENERATOR_PATH, 'r') as f:
            generator_content = f.read()
    except Exception as e:
        logger.error(f"Error reading generator file: {e}")
        return False
    
    # Modify the generator content to update MODEL_REGISTRY
    registry_entries = model_data.get("model_registry_entries", {})
    class_name_fixes = model_data.get("class_name_fixes", {})
    
    # Format the MODEL_REGISTRY entries
    model_registry_entries = []
    for model_type, entry in registry_entries.items():
        # Check if the model already exists in the generator
        if f'"{model_type}"' in generator_content and f'model_type": "{model_type}"' in generator_content:
            logger.info(f"Model {model_type} already exists in MODEL_REGISTRY, skipping")
            continue
        
        # Format the entry
        entry_lines = []
        entry_lines.append(f'    "{model_type}": {{')
        for key, value in entry.items():
            if isinstance(value, str):
                entry_lines.append(f'        "{key}": "{value}",')
            elif isinstance(value, list):
                entry_lines.append(f'        "{key}": {value},')
            elif isinstance(value, dict):
                entry_lines.append(f'        "{key}": {{')
                for k, v in value.items():
                    if isinstance(v, dict):
                        entry_lines.append(f'            "{k}": {{')
                        for inner_k, inner_v in v.items():
                            if isinstance(inner_v, str):
                                entry_lines.append(f'                "{inner_k}": "{inner_v}",')
                            else:
                                entry_lines.append(f'                "{inner_k}": {inner_v},')
                        entry_lines.append('            },')
                    elif isinstance(v, str):
                        entry_lines.append(f'            "{k}": "{v}",')
                    else:
                        entry_lines.append(f'            "{k}": {v},')
                entry_lines.append('        },')
        entry_lines.append('    },')
        model_registry_entries.append('\n'.join(entry_lines))
    
    # Format the CLASS_NAME_FIXES entries
    class_name_fix_entries = []
    for model_type, correct_capitalization in class_name_fixes.items():
        parts = model_type.split('-')
        incorrect_name = ''.join(part.capitalize() for part in parts)
        
        # Add common model class suffixes
        for suffix in ["ForCausalLM", "Model", "ForMaskedLM", "ForSequenceClassification", 
                       "ForTokenClassification", "ForQuestionAnswering", "ForMultipleChoice", 
                       "ForImageClassification", "ForAudioClassification", "ForCTC"]:
            # Check if this specific fix already exists
            fix_entry = f'    "{incorrect_name}{suffix}": "{correct_capitalization}{suffix}",'
            if fix_entry not in generator_content:
                class_name_fix_entries.append(fix_entry)
    
    # Update the MODEL_REGISTRY
    if model_registry_entries:
        # Find the MODEL_REGISTRY definition
        registry_start = generator_content.find("MODEL_REGISTRY = {")
        registry_end = generator_content.find("}", registry_start)
        
        if registry_start != -1 and registry_end != -1:
            # Insert the new models
            updated_content = (
                generator_content[:registry_end] + 
                ",\n" + 
                "\n".join(model_registry_entries) + 
                generator_content[registry_end:]
            )
            generator_content = updated_content
            logger.info(f"Added {len(model_registry_entries)} new models to MODEL_REGISTRY")
        else:
            logger.error("Could not find MODEL_REGISTRY in generator file")
            return False
    
    # Update the CLASS_NAME_FIXES
    if class_name_fix_entries:
        # Find the CLASS_NAME_FIXES definition
        fixes_start = generator_content.find("CLASS_NAME_FIXES = {")
        fixes_end = generator_content.find("}", fixes_start)
        
        if fixes_start != -1 and fixes_end != -1:
            # Insert the new fixes
            updated_content = (
                generator_content[:fixes_end] + 
                "\n" + 
                "\n".join(class_name_fix_entries) + 
                generator_content[fixes_end:]
            )
            generator_content = updated_content
            logger.info(f"Added {len(class_name_fix_entries)} new class name fixes")
        else:
            logger.error("Could not find CLASS_NAME_FIXES in generator file")
            return False
    
    # Write the updated generator content
    try:
        with open(GENERATOR_PATH, 'w') as f:
            f.write(generator_content)
        logger.info(f"Updated generator file at {GENERATOR_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error writing generator file: {e}")
        return False

def run_generator(output_dir=None, hyphenated_only=True, verify=True):
    """Run the test generator to create test files."""
    if not output_dir:
        output_dir = FIXED_TESTS_DIR
    
    # Build command
    cmd = [sys.executable, str(GENERATOR_PATH)]
    if hyphenated_only:
        cmd.append("--hyphenated-only")
    else:
        cmd.append("--all")
    cmd.extend(["--output-dir", str(output_dir)])
    if verify:
        cmd.append("--verify")
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the command
        process = subprocess.run(cmd, check=True, text=True, capture_output=True)
        logger.info(f"Generator output: {process.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running generator: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def verify_test_files(test_dir=None):
    """Verify that all test files have valid syntax."""
    if not test_dir:
        test_dir = FIXED_TESTS_DIR
    
    # Find all test files
    test_files = list(Path(test_dir).glob("test_hf_*.py"))
    logger.info(f"Found {len(test_files)} test files to verify")
    
    success_count = 0
    failure_count = 0
    
    for test_file in test_files:
        try:
            # Try to compile the file
            with open(test_file, 'r') as f:
                code = f.read()
                compile(code, str(test_file), 'exec')
                logger.info(f"✅ {test_file.name}: Syntax is valid")
                success_count += 1
        except SyntaxError as e:
            logger.error(f"❌ {test_file.name}: Syntax error: {e}")
            failure_count += 1
        except Exception as e:
            logger.error(f"❌ {test_file.name}: Error: {e}")
            failure_count += 1
    
    logger.info(f"Verification summary: {success_count} files passed, {failure_count} files failed")
    return failure_count == 0

def update_coverage_report(test_dir=None, output_file=None):
    """Generate and update the test coverage report."""
    if not test_dir:
        test_dir = FIXED_TESTS_DIR
    
    if not output_file:
        output_file = CURRENT_DIR / "MODEL_TEST_COVERAGE.md"
    
    # Find all test files
    test_files = list(Path(test_dir).glob("test_hf_*.py"))
    
    # Extract model names from test files
    model_names = [f.stem[8:] for f in test_files]  # Remove "test_hf_" prefix
    
    # Group models by architecture type
    models_by_architecture = {}
    for model_name in model_names:
        # Convert underscores back to hyphens for matching
        search_name = model_name.replace('_', '-')
        
        # Determine architecture type
        architecture_type = "unknown"
        for arch_type, models in ARCHITECTURE_TYPES.items():
            if any(model in search_name for model in models):
                architecture_type = arch_type
                break
        
        if architecture_type not in models_by_architecture:
            models_by_architecture[architecture_type] = []
        models_by_architecture[architecture_type].append(model_name)
    
    # Create the report content
    report_content = f"""# Model Test Coverage Report

Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

Total test files: {len(test_files)}

## Coverage by Architecture Type

"""
    
    # Add sections for each architecture type
    for arch_type, models in sorted(models_by_architecture.items()):
        report_content += f"### {arch_type.capitalize()} ({len(models)} models)\n\n"
        for model in sorted(models):
            report_content += f"- `{model}`\n"
        report_content += "\n"
    
    # Add special section for hyphenated models
    hyphenated_models = [m for m in model_names if "_" in m and m.replace("_", "-") != m]
    if hyphenated_models:
        report_content += f"## Hyphenated Models ({len(hyphenated_models)} models)\n\n"
        for model in sorted(hyphenated_models):
            original_name = model.replace("_", "-")
            report_content += f"- `{original_name}` → `{model}`\n"
        report_content += "\n"
    
    # Write the report
    try:
        with open(output_file, 'w') as f:
            f.write(report_content)
        logger.info(f"Updated coverage report at {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error writing coverage report: {e}")
        return False

def main():
    """Main function to expand test coverage."""
    parser = argparse.ArgumentParser(description="Expand test coverage to additional models")
    parser.add_argument("--hyphenated-only", action="store_true", help="Only process hyphenated model names")
    parser.add_argument("--output-dir", type=str, default=str(FIXED_TESTS_DIR), help="Output directory for test files")
    parser.add_argument("--verify", action="store_true", help="Verify syntax of generated tests")
    parser.add_argument("--skip-find-models", action="store_true", help="Skip finding models step (use existing data)")
    parser.add_argument("--model-data", type=str, help="Path to existing model data file (if skipping find-models)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Find models
    model_data_file = None
    if not args.skip_find_models:
        logger.info("Step 1: Finding models...")
        model_data_file = run_find_models(hyphenated_only=args.hyphenated_only)
        if not model_data_file:
            logger.error("Failed to find models, stopping")
            return 1
    else:
        if args.model_data:
            model_data_file = args.model_data
            logger.info(f"Using existing model data file: {model_data_file}")
        else:
            logger.error("Must provide --model-data when using --skip-find-models")
            return 1
    
    # Step 2: Update generator
    logger.info("Step 2: Updating generator...")
    if not update_generator(model_data_file):
        logger.error("Failed to update generator, stopping")
        return 1
    
    # Step 3: Run generator
    logger.info("Step 3: Running generator to create test files...")
    if not run_generator(output_dir=args.output_dir, hyphenated_only=args.hyphenated_only, verify=args.verify):
        logger.error("Failed to run generator, stopping")
        return 1
    
    # Step 4: Verify test files
    if args.verify:
        logger.info("Step 4: Verifying test files...")
        if not verify_test_files(test_dir=args.output_dir):
            logger.warning("Some test files failed verification")
    
    # Step 5: Update coverage report
    logger.info("Step 5: Updating coverage report...")
    update_coverage_report(test_dir=args.output_dir)
    
    logger.info("Successfully expanded test coverage")
    return 0

# Define architecture types (needed for coverage report)
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta", "albert", "ernie", "rembert"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt", "gemma", "opt", "codegen"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan", "prophetnet"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2", "segformer", "detr", "yolos", "mask2former"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip", "git", "flava", "paligemma"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5", "sew", "encodec", "musicgen", "audio", "clap"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava", "idefics", "imagebind"]
}

if __name__ == "__main__":
    sys.exit(main())