#!/usr/bin/env python3
"""
Fix Hyphenated Model Report

This script updates the model tracking system to properly recognize both hyphenated and
underscore model names as the same model.
"""

import os
import sys
import json
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyphenated to underscore mappings
HYPHENATED_MODELS = {
    'gpt-j': 'gpt_j',
    'gpt-neo': 'gpt_neo',
    'gpt-neox': 'gpt_neox',
    'flan-t5': 'flan_t5',
    'xlm-roberta': 'xlm_roberta',
    'vision-text-dual-encoder': 'vision_text_dual_encoder',
    'speech-to-text': 'speech_to_text',
    'speech-to-text-2': 'speech_to_text_2',
    'data2vec-text': 'data2vec_text',
    'data2vec-audio': 'data2vec_audio',
    'data2vec-vision': 'data2vec_vision',
    'wav2vec2-conformer': 'wav2vec2_conformer',
    'transfo-xl': 'transfo_xl',
    'mlp-mixer': 'mlp_mixer'
}

def update_model_tracking_file():
    """Update model tracking metrics to correctly count both hyphenated and underscore versions."""
    model_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_tracking.json')
    
    # Create the file if it doesn't exist
    if not os.path.exists(model_file_path):
        tracking_data = {
            "models": {},
            "architectures": {},
            "priorities": {
                "critical": [],
                "high": [],
                "medium": []
            }
        }
    else:
        # Load existing data
        with open(model_file_path, 'r') as f:
            tracking_data = json.load(f)
    
    # Update each hyphenated model to include both versions
    for hyphenated, underscore in HYPHENATED_MODELS.items():
        # Check if either version is in the model list
        if hyphenated in tracking_data["models"] or underscore in tracking_data["models"]:
            # Standardize to use the underscore version
            if hyphenated in tracking_data["models"]:
                model_data = tracking_data["models"].pop(hyphenated)
                tracking_data["models"][underscore] = model_data
                logger.info(f"Standardized {hyphenated} -> {underscore}")
            
            # Update priority lists
            for priority in ["critical", "high", "medium"]:
                if hyphenated in tracking_data["priorities"][priority]:
                    tracking_data["priorities"][priority].remove(hyphenated)
                    if underscore not in tracking_data["priorities"][priority]:
                        tracking_data["priorities"][priority].append(underscore)
                        logger.info(f"Updated {priority} priority list: {hyphenated} -> {underscore}")
    
    # Save updated data
    with open(model_file_path, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    logger.info(f"Updated model tracking file: {model_file_path}")

def copy_test_files():
    """Copy test files with underscore names to their hyphenated equivalents."""
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    fixed_tests_dir = script_dir / "fixed_tests"
    
    for hyphenated, underscore in HYPHENATED_MODELS.items():
        # Format file names
        underscore_file = fixed_tests_dir / f"test_hf_{underscore}.py"
        hyphenated_file = fixed_tests_dir / f"test_hf_{hyphenated}.py"
        
        # Copy from underscore to hyphenated if needed
        if underscore_file.exists() and not hyphenated_file.exists():
            with open(underscore_file, 'r') as src:
                content = src.read()
                
            with open(hyphenated_file, 'w') as dst:
                dst.write(content)
            
            logger.info(f"Created symlink for {hyphenated_file}")
        
        # Copy from hyphenated to underscore if needed
        elif hyphenated_file.exists() and not underscore_file.exists():
            with open(hyphenated_file, 'r') as src:
                content = src.read()
                
            with open(underscore_file, 'w') as dst:
                dst.write(content)
            
            logger.info(f"Created {underscore_file}")

def create_model_symlinks():
    """Creates symbolic links for test files to ensure both naming conventions work."""
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    for hyphenated, underscore in HYPHENATED_MODELS.items():
        # Format file names
        underscore_file = script_dir / f"test_hf_{underscore}.py"
        hyphenated_file = script_dir / f"test_hf_{hyphenated}.py"
        
        # Create symlink from underscore to hyphenated if needed
        if underscore_file.exists() and not hyphenated_file.exists():
            try:
                os.symlink(underscore_file, hyphenated_file)
                logger.info(f"Created symlink: {hyphenated_file} -> {underscore_file}")
            except Exception as e:
                logger.error(f"Error creating symlink {hyphenated_file}: {e}")
        
        # Create symlink from hyphenated to underscore if needed
        elif hyphenated_file.exists() and not underscore_file.exists():
            try:
                os.symlink(hyphenated_file, underscore_file)
                logger.info(f"Created symlink: {underscore_file} -> {hyphenated_file}")
            except Exception as e:
                logger.error(f"Error creating symlink {underscore_file}: {e}")

def update_report_generator():
    """Update the missing model report generator to handle both naming conventions."""
    report_generator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate_missing_model_report.py')
    
    if not os.path.exists(report_generator_path):
        logger.error(f"Report generator not found: {report_generator_path}")
        return
    
    with open(report_generator_path, 'r') as f:
        content = f.read()
    
    # Create a function to standardize model names when checking
    standardize_func = '''
def standardize_model_name(name):
    """Standardize model names to handle both hyphenated and underscore versions."""
    # Hyphenated to underscore mappings
    HYPHENATED_MODELS = {
        'gpt-j': 'gpt_j',
        'gpt-neo': 'gpt_neo',
        'gpt-neox': 'gpt_neox',
        'flan-t5': 'flan_t5',
        'xlm-roberta': 'xlm_roberta',
        'vision-text-dual-encoder': 'vision_text_dual_encoder',
        'speech-to-text': 'speech_to_text',
        'speech-to-text-2': 'speech_to_text_2',
        'data2vec-text': 'data2vec_text',
        'data2vec-audio': 'data2vec_audio',
        'data2vec-vision': 'data2vec_vision',
        'wav2vec2-conformer': 'wav2vec2_conformer',
        'transfo-xl': 'transfo_xl',
        'mlp-mixer': 'mlp_mixer'
    }
    
    # Check if the name exists in the mapping
    if name in HYPHENATED_MODELS:
        return HYPHENATED_MODELS[name]
    
    # Check if this is an underscore version of a hyphenated name
    for hyphenated, underscore in HYPHENATED_MODELS.items():
        if name == underscore:
            return name
    
    # Default to original name
    return name
'''
    
    # Check if the function already exists
    if "def standardize_model_name(" not in content:
        # Find imports section to add after
        import_section_end = max(content.rfind("import "), content.rfind("from "))
        if import_section_end == -1:
            import_section_end = 0
        
        # Find the end of the imports block
        while import_section_end < len(content) and content[import_section_end] != '\n':
            import_section_end += 1
        import_section_end += 1
        
        # Insert the function
        content = content[:import_section_end] + standardize_func + content[import_section_end:]
    
    # Update model name extraction logic
    if "model_name = file_path.stem.replace(\"test_hf_\", \"\")" in content:
        content = content.replace(
            "model_name = file_path.stem.replace(\"test_hf_\", \"\")",
            "model_name = standardize_model_name(file_path.stem.replace(\"test_hf_\", \"\"))"
        )
    
    # Write updated file
    with open(report_generator_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated report generator: {report_generator_path}")

def main():
    """Main entry point."""
    # Update model tracking
    update_model_tracking_file()
    
    # Copy test files to ensure both naming conventions work
    copy_test_files()
    
    # Create symbolic links between the files
    create_model_symlinks()
    
    # Update report generator
    update_report_generator()
    
    logger.info("Completed fixing hyphenated model naming issues")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())