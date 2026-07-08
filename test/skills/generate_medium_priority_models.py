#!/usr/bin/env python3
"""
Script to generate tests for medium-priority HuggingFace models.
This script focuses on Phase 4 medium-priority models from the roadmap.
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
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"medium_priority_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SKILLS_DIR)
TEMP_DIR = os.path.join(SKILLS_DIR, "temp_generated")
COVERAGE_DIR = os.path.join(SKILLS_DIR, "coverage_visualizations")
MODELS_JSON = os.path.join(SKILLS_DIR, "medium_priority_models.json")

# These can be overridden by command-line arguments
OUTPUT_DIR = PARENT_DIR

def load_medium_priority_models():
    """Load the medium priority models from JSON file."""
    try:
        with open(MODELS_JSON, 'r') as f:
            data = json.load(f)
        
        # Flatten the model categories into a single list
        models = {}
        for category, model_list in data["medium_priority_models"].items():
            for model in model_list:
                models[model["name"]] = model
        
        logger.info(f"Loaded {len(models)} medium-priority models")
        return models
    
    except Exception as e:
        logger.error(f"Error loading medium priority models: {e}")
        return {}

def get_implemented_models(output_dir=OUTPUT_DIR):
    """Get list of implemented models from test files."""
    test_files = glob.glob(os.path.join(output_dir, "test_hf_*.py"))
    implemented = []
    
    for test_file in test_files:
        model_name = os.path.basename(test_file).replace('test_hf_', '').replace('.py', '')
        implemented.append(model_name)
    
    logger.info(f"Found {len(implemented)} implemented models")
    return implemented

def get_missing_medium_priority_models(medium_priority_models, implemented_models):
    """Get medium priority models that have not been implemented yet."""
    missing = {}
    
    for model_name, model_config in medium_priority_models.items():
        if model_name not in implemented_models:
            missing[model_name] = model_config
    
    logger.info(f"Found {len(missing)} missing medium-priority models")
    return missing

def generate_test_for_model(model_name, model_config, output_dir=OUTPUT_DIR):
    """Generate a test for a specific model."""
    template_model = model_config.get("template")
    
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
        # Get a subset of the missing models
        model_names = list(missing_models.keys())[:max_models]
        models_to_process = {name: missing_models[name] for name in model_names}
    else:
        models_to_process = missing_models
    
    results = {}
    successful = {}
    failed = {}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_model = {
            executor.submit(generate_test_for_model, model_name, model_config, output_dir): model_name 
            for model_name, model_config in models_to_process.items()
        }
        
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                success = future.result()
                results[model_name] = success
                
                if success:
                    successful[model_name] = models_to_process[model_name]
                else:
                    failed[model_name] = models_to_process[model_name]
            
            except Exception as e:
                logger.error(f"Exception processing {model_name}: {e}")
                results[model_name] = False
                failed[model_name] = models_to_process[model_name]
    
    return results, successful, failed

def update_medium_priority_roadmap():
    """Create or update Phase 4 section in HF_MODEL_COVERAGE_ROADMAP.md with the medium priority models."""
    roadmap_path = os.path.join(SKILLS_DIR, "HF_MODEL_COVERAGE_ROADMAP.md")
    
    if not os.path.exists(roadmap_path):
        logger.warning(f"Roadmap file not found: {roadmap_path}")
        return
    
    try:
        # Load the medium priority models
        models_data = load_medium_priority_models()
        if not models_data:
            logger.error("No medium priority models found")
            return
        
        # Read the roadmap file
        with open(roadmap_path, 'r') as f:
            content = f.read()
        
        # Check if Phase 4 already exists
        if "## Phase 4: Medium-Priority Models" in content:
            logger.info("Phase 4 section already exists in roadmap")
            return
        
        # Create the Phase 4 section
        phase4_content = "\n## Phase 4: Medium-Priority Models (April 6-15, 2025)\n\n"
        phase4_content += "These models represent medium-priority architectures with wide usage:\n\n"
        
        # Get implemented models
        implemented_models = get_implemented_models()
        
        # Create subsections for each category
        categories = {
            "text_encoder_models": "### Text Encoder Models",
            "text_decoder_models": "### Text Decoder Models",
            "text_encoder_decoder_models": "### Text Encoder-Decoder Models",
            "vision_models": "### Vision Models",
            "multimodal_models": "### Multimodal Models",
            "audio_models": "### Audio Models"
        }
        
        # Load the original JSON to preserve categories
        with open(MODELS_JSON, 'r') as f:
            categories_data = json.load(f)["medium_priority_models"]
        
        # Add models by category
        for category, heading in categories.items():
            phase4_content += f"{heading}\n"
            
            for model in categories_data[category]:
                model_name = model["name"]
                architecture = model["architecture"]
                
                # Check if already implemented
                checkbox = "[x]" if model_name in implemented_models else "[ ]"
                
                phase4_content += f"- {checkbox} {model_name} ({architecture})\n"
            
            phase4_content += "\n"
        
        # Find the position to insert Phase 4 section
        if "## Phase 3: Architecture Expansion" in content:
            parts = content.split("## Phase 3: Architecture Expansion")
            
            # Find the end of Phase 3 section
            phase3_part = parts[1]
            if "##" in phase3_part:
                phase3_end = phase3_part.find("##")
                insert_position = len(parts[0]) + len("## Phase 3: Architecture Expansion") + phase3_end
            else:
                insert_position = len(parts[0]) + len("## Phase 3: Architecture Expansion") + len(phase3_part)
            
            # Insert Phase 4 section
            updated_content = content[:insert_position] + phase4_content + content[insert_position:]
            
            # Write updated content
            with open(roadmap_path, 'w') as f:
                f.write(updated_content)
            
            logger.info(f"Added Phase 4 section to roadmap file: {roadmap_path}")
        else:
            logger.warning("Could not find Phase 3 section in roadmap file")
    
    except Exception as e:
        logger.error(f"Error updating roadmap file: {e}")

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
        
        # Check if Phase 4 exists
        if "## Phase 4: Medium-Priority Models" not in content:
            logger.warning("Phase 4 section not found in roadmap, creating it...")
            update_medium_priority_roadmap()
            
            # Reload the content
            with open(roadmap_path, 'r') as f:
                content = f.read()
        
        # Process each model one by one to ensure more precise replacements
        updated_content = content
        for model_name in successful_models.keys():
            # Try different patterns to find the right line
            patterns = [
                f"- [ ] {model_name} ",  # Standard format with space
                f"- [ ] {model_name}(",  # Format with opening parenthesis
                f"- [ ] {model_name}\n"  # Format at end of line
            ]
            
            for pattern in patterns:
                if pattern in updated_content:
                    replacement = pattern.replace("[ ]", "[x]")
                    updated_content = updated_content.replace(pattern, replacement)
                    logger.info(f"Updated status for model: {model_name}")
                    break
            
            # For case-insensitive matching (e.g., "RoBERTa" vs "roberta")
            if model_name.lower() != model_name:
                for pattern in patterns:
                    capitalized_pattern = pattern.replace(model_name, model_name[0].upper() + model_name[1:])
                    if capitalized_pattern in updated_content:
                        replacement = capitalized_pattern.replace("[ ]", "[x]")
                        updated_content = updated_content.replace(capitalized_pattern, replacement)
                        logger.info(f"Updated status for model (capitalized): {model_name}")
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
        
        # Calculate new percentage
        percentage = int((current_implemented / total_models) * 100)
        
        # Update CLAUDE.md
        with open(claude_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            if "**Priority 2: Comprehensive HuggingFace Model Testing (300+ classes)**" in line and "IN PROGRESS" in line:
                # Update the progress percentage
                updated_line = line.replace(f"(🔄 IN PROGRESS - 45%)", f"(🔄 IN PROGRESS - {percentage}%)")
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
    parser = argparse.ArgumentParser(description="Generate tests for medium-priority HuggingFace models")
    parser.add_argument("--max-models", type=int, help="Maximum number of models to generate")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for generated tests")
    parser.add_argument("--list-missing", action="store_true", help="List missing medium-priority models without generating tests")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--specific-model", type=str, help="Generate test for a specific medium-priority model")
    parser.add_argument("--skip-update", action="store_true", help="Skip updating the roadmap and coverage files")
    parser.add_argument("--init-roadmap", action="store_true", help="Initialize Phase 4 section in roadmap without generating tests")
    parser.add_argument("--category", type=str, choices=["text_encoder_models", "text_decoder_models", 
                                                        "text_encoder_decoder_models", "vision_models", 
                                                        "multimodal_models", "audio_models"],
                      help="Process only models from a specific category")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    # Initialize Phase 4 in roadmap if requested
    if args.init_roadmap:
        update_medium_priority_roadmap()
        return
    
    # Load medium priority models
    medium_priority_models = load_medium_priority_models()
    if not medium_priority_models:
        logger.error("Failed to load medium priority models")
        return
    
    # Filter by category if specified
    if args.category:
        with open(MODELS_JSON, 'r') as f:
            categories_data = json.load(f)["medium_priority_models"]
            
        category_models = {}
        for model in categories_data[args.category]:
            category_models[model["name"]] = model
        
        medium_priority_models = category_models
        logger.info(f"Filtered to {len(medium_priority_models)} models in category: {args.category}")
    
    # Get implemented models
    implemented_models = get_implemented_models(output_dir)
    
    # Get missing medium-priority models
    missing_models = get_missing_medium_priority_models(medium_priority_models, implemented_models)
    
    if args.list_missing:
        logger.info(f"Missing medium-priority models ({len(missing_models)}):")
        for i, (model_name, config) in enumerate(missing_models.items(), 1):
            logger.info(f"{i}. {model_name} (architecture: {config['architecture']}, template: {config['template']})")
        return
    
    # Handle specific model request
    if args.specific_model:
        if args.specific_model in medium_priority_models:
            success = generate_test_for_model(args.specific_model, medium_priority_models[args.specific_model], output_dir)
            if success and not args.skip_update:
                update_model_coverage_roadmap({args.specific_model: medium_priority_models[args.specific_model]})
                update_coverage_statistic(1)
                generate_coverage_report()
            return
        else:
            logger.error(f"Model {args.specific_model} is not in the medium-priority list")
            return
    
    # Generate tests for missing medium-priority models
    if not missing_models:
        logger.info("All medium-priority models have already been implemented!")
        return
    
    logger.info(f"Generating tests for {len(missing_models)} missing medium-priority models...")
    
    # Create temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Generate tests
    max_models = args.max_models if args.max_models else len(missing_models)
    results, successful, failed = generate_batch(missing_models, max_models, args.workers, output_dir)
    
    logger.info("\nGeneration complete!")
    logger.info(f"Successfully generated: {len(successful)}/{min(max_models, len(missing_models))}")
    logger.info(f"Failed: {len(failed)}/{min(max_models, len(missing_models))}")
    
    if failed:
        logger.info("\nFailed models:")
        for model_name in failed:
            logger.info(f"- {model_name}")
    
    # Update roadmap and coverage files
    if successful and not args.skip_update:
        # Ensure Phase 4 exists in roadmap
        update_medium_priority_roadmap()
        
        # Mark implemented models as complete
        update_model_coverage_roadmap(successful)
        
        # Update coverage percentage
        update_coverage_statistic(len(successful))
        
        # Generate updated coverage report
        generate_coverage_report()

if __name__ == "__main__":
    main()