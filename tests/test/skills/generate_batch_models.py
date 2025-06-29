#!/usr/bin/env python3
"""
Script to generate tests for batch model implementation.
"""

import os
import sys
import json
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
        logging.FileHandler(f"batch_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SKILLS_DIR)
GENERATOR_SCRIPT = os.path.join(SKILLS_DIR, "test_generator_fixed.py")
ROADMAP_PATH = os.path.join(SKILLS_DIR, "HF_MODEL_COVERAGE_ROADMAP.md")

def syntax_check(file_path):
    """Check Python syntax of a file."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", file_path],
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stderr

def generate_model_test(model_name, architecture, template=None, task=None, original_name=None, output_dir=None):
    """Generate a test file for a specific model."""
    if output_dir is None:
        output_dir = os.path.join(SKILLS_DIR, "fixed_tests")
    
    logger.info(f"Generating test for {model_name}...")
    
    # Build command
    command = [
        sys.executable,
        GENERATOR_SCRIPT,
        "--generate", model_name,
        "--output-dir", output_dir
    ]
    
    if task:
        command.extend(["--task", task])
    
    # Run generator
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check if the file exists even if there was an error
        # The generator might show errors but still create the file
        output_path = os.path.join(output_dir, f"test_hf_{model_name}.py")
        
        if result.returncode != 0:
            logger.warning(f"Generator reported error for {model_name}: {result.stderr}")
            
            if not os.path.exists(output_path):
                logger.error(f"Generated file not found: {output_path}")
                return False
            else:
                logger.warning(f"Generator reported error but file was created: {output_path}")
        
        # Check syntax
        syntax_valid, syntax_error = syntax_check(output_path)
        if not syntax_valid:
            logger.error(f"Syntax check failed for {model_name}: {syntax_error}")
            return False
        
        logger.info(f"✅ Successfully generated test for {model_name}: {output_path}")
        
        # If the model has a hyphenated name, update the roadmap
        if original_name and original_name != model_name:
            logger.info(f"Model {model_name} has hyphenated original name: {original_name}")
        
        return True
    
    except Exception as e:
        logger.error(f"Exception generating {model_name}: {e}")
        return False

def update_roadmap_mark_implemented(model_name, section_name):
    """Update the roadmap to mark a model as implemented."""
    try:
        # Read the current roadmap
        with open(ROADMAP_PATH, 'r') as f:
            roadmap_content = f.read()
        
        # Look for the model in the medium priority section
        section_start = roadmap_content.find(f"### {section_name}")
        if section_start == -1:
            logger.warning(f"Section {section_name} not found in roadmap")
            return False
        
        # Find the model entry (both hyphenated and underscore versions)
        model_entry_format1 = f"- [ ] {model_name}"
        model_entry_format2 = f"- [ ] {model_name.replace('_', '-')}"
        
        # Try to replace with the first format
        if model_entry_format1 in roadmap_content:
            updated_content = roadmap_content.replace(
                model_entry_format1,
                f"- [x] {model_name}"
            )
            # Write back the updated content
            with open(ROADMAP_PATH, 'w') as f:
                f.write(updated_content)
            logger.info(f"Updated roadmap: Marked {model_name} as implemented")
            return True
        
        # Try with the second format (hyphenated)
        elif model_entry_format2 in roadmap_content:
            updated_content = roadmap_content.replace(
                model_entry_format2,
                f"- [x] {model_name.replace('_', '-')}"
            )
            # Write back the updated content
            with open(ROADMAP_PATH, 'w') as f:
                f.write(updated_content)
            logger.info(f"Updated roadmap: Marked {model_name.replace('_', '-')} as implemented")
            return True
        
        else:
            logger.warning(f"Model entry for {model_name} not found in roadmap")
            return False
    
    except Exception as e:
        logger.error(f"Error updating roadmap for {model_name}: {e}")
        return False

def generate_batch_report(batch_name, successful_models, failed_models):
    """Generate a report for the batch implementation."""
    report_dir = os.path.join(SKILLS_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(
        report_dir, 
        f"{batch_name}_implementation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    
    try:
        with open(report_path, 'w') as f:
            f.write(f"# {batch_name} Implementation Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            total = len(successful_models) + len(failed_models)
            success_rate = len(successful_models) / total * 100 if total > 0 else 0
            f.write(f"- **Total Models:** {total}\n")
            f.write(f"- **Successfully Implemented:** {len(successful_models)} ({success_rate:.1f}%)\n")
            f.write(f"- **Failed:** {len(failed_models)}\n\n")
            
            f.write("## Successfully Implemented Models\n\n")
            if successful_models:
                for model in successful_models:
                    f.write(f"- **{model}**\n")
            else:
                f.write("No models were successfully implemented.\n")
            
            f.write("\n## Failed Models\n\n")
            if failed_models:
                for model in failed_models:
                    f.write(f"- **{model}**\n")
            else:
                f.write("No models failed implementation.\n")
        
        logger.info(f"Batch report generated: {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Error generating batch report: {e}")
        return None

def main():
    """Main function to generate batch model tests."""
    parser = argparse.ArgumentParser(description="Generate tests for batch model implementation")
    parser.add_argument("--batch-file", required=True, help="JSON file containing batch model definitions")
    parser.add_argument("--output-dir", default=None, help="Output directory for generated files")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads")
    args = parser.parse_args()
    
    # Ensure batch file exists
    if not os.path.exists(args.batch_file):
        logger.error(f"Batch file not found: {args.batch_file}")
        return 1
    
    # Set up output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(SKILLS_DIR, "fixed_tests")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load batch model definitions
    try:
        with open(args.batch_file, 'r') as f:
            batch_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading batch file: {e}")
        return 1
    
    # Generate tests
    successful_models = []
    failed_models = []
    
    # Process models by architecture type
    for architecture, models in batch_data.items():
        logger.info(f"Processing {len(models)} models for architecture: {architecture}")
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            
            for model_data in models:
                model_name = model_data["name"]
                futures[executor.submit(
                    generate_model_test,
                    model_name=model_name,
                    architecture=model_data.get("architecture", architecture),
                    template=model_data.get("template"),
                    task=model_data.get("task"),
                    original_name=model_data.get("original_name"),
                    output_dir=output_dir
                )] = model_name
            
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    success = future.result()
                    if success:
                        successful_models.append(model_name)
                        # Update roadmap to mark as implemented
                        if architecture == "decoder_only":
                            section_name = "Text Decoder Models"
                        elif architecture == "encoder_decoder":
                            section_name = "Text Encoder-Decoder Models"
                        elif architecture == "encoder_only":
                            section_name = "Text Encoder Models"
                        elif architecture == "vision":
                            section_name = "Vision Models"
                        elif architecture == "vision_text":
                            section_name = "Vision-text Models"
                        elif architecture == "multimodal":
                            section_name = "Multimodal Models"
                        elif architecture == "speech":
                            section_name = "Audio Models"
                        else:
                            section_name = f"{architecture.capitalize()} Models"
                        
                        update_roadmap_mark_implemented(model_name, section_name)
                    else:
                        failed_models.append(model_name)
                except Exception as e:
                    logger.error(f"Error processing {model_name}: {e}")
                    failed_models.append(model_name)
    
    # Generate report
    batch_name = os.path.splitext(os.path.basename(args.batch_file))[0]
    report_path = generate_batch_report(batch_name, successful_models, failed_models)
    
    # Print summary
    logger.info("\nBatch Generation Summary:")
    logger.info(f"- Total Models: {len(successful_models) + len(failed_models)}")
    logger.info(f"- Successfully Implemented: {len(successful_models)}")
    logger.info(f"- Failed: {len(failed_models)}")
    
    if report_path:
        logger.info(f"- Implementation Report: {report_path}")
    
    if failed_models:
        logger.warning("Some models failed to generate. See the report for details.")
        return 1
    else:
        logger.info("All models were successfully generated!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
