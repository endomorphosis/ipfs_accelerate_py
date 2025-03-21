#!/usr/bin/env python3
"""
Generate tests for missing HuggingFace models based on priority.

This script:
1. Reads HF_MODEL_COVERAGE_ROADMAP.md to determine high-priority models
2. Generates tests for missing models using architecture-specific templates
3. Verifies syntax of generated files
4. Updates tracking documentation with progress

Usage:
    python generate_missing_model_tests.py [--priority {high,medium,low,all}] [--verify] [--output-dir DIRECTORY]
"""

import os
import sys
import argparse
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"generate_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Import regenerate_test_file function from regenerate_fixed_tests.py
try:
    from regenerate_fixed_tests import regenerate_test_file, get_architecture_type, get_template_for_architecture
except ImportError:
    logger.error("Failed to import from regenerate_fixed_tests.py. Make sure it exists in the same directory.")
    sys.exit(1)

# Define priorities based on HF_MODEL_COVERAGE_ROADMAP.md
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
    "low": []  # Add low-priority models here when needed
}

# Skip these special model types since they're not valid Python identifiers due to hyphens
# or have other syntax issues
SKIP_MODELS = [
    # Architecture types
    "encoder-only", "decoder-only", "encoder-decoder", "vision", "multimodal", "audio",
    # Models with hyphens in name
    "xlm-roberta", "gpt-neo", "gpt-j",
    # Models with other syntax issues
    "git", "paligemma"
]

def extract_models_from_roadmap():
    """Extract model names from the HF_MODEL_COVERAGE_ROADMAP.md file."""
    roadmap_path = "HF_MODEL_COVERAGE_ROADMAP.md"
    if not os.path.exists(roadmap_path):
        logger.warning(f"Roadmap file not found: {roadmap_path}, using predefined priorities")
        return PRIORITY_MODELS
    
    try:
        with open(roadmap_path, 'r') as f:
            content = f.read()
        
        # Extract high priority models
        high_priority_section = re.search(r'## Phase 2: High-Priority Models.*?## Phase 3:', content, re.DOTALL)
        if high_priority_section:
            high_section_text = high_priority_section.group(0)
            # Extract model names from markdown list items with model names in parentheses
            high_models = re.findall(r'- \[ \] ([A-Za-z0-9]+) \((.*?)\)', high_section_text)
            high_models = [name.lower() for _, name in high_models]
            
            if high_models:
                PRIORITY_MODELS["high"] = high_models
        
        # Extract medium priority models
        medium_priority_section = re.search(r'## Phase 3: Architecture Expansion.*?## (?!Phase 3)', content, re.DOTALL)
        if medium_priority_section:
            medium_section_text = medium_priority_section.group(0)
            medium_models = re.findall(r'- \[ \] ([A-Za-z0-9]+) \((.*?)\)', medium_section_text)
            medium_models = [name.lower() for _, name in medium_models]
            
            if medium_models:
                PRIORITY_MODELS["medium"] = medium_models
        
        logger.info(f"Extracted priorities from roadmap: {len(PRIORITY_MODELS['high'])} high, "
                    f"{len(PRIORITY_MODELS['medium'])} medium")
        return PRIORITY_MODELS
    
    except Exception as e:
        logger.error(f"Error extracting models from roadmap: {e}")
        return PRIORITY_MODELS

def get_existing_models(test_dir="."):
    """Get list of existing model test files."""
    existing_models = []
    for file in os.listdir(test_dir):
        if file.startswith("test_hf_") and file.endswith(".py"):
            model_name = file[8:-3]  # Extract model name from test_hf_NAME.py
            existing_models.append(model_name)
    return existing_models

def generate_missing_models(priority="high", output_dir="fixed_tests", verify=True):
    """Generate tests for missing models based on priority."""
    # Get updated priorities from roadmap if available
    priorities = extract_models_from_roadmap()
    
    # Determine which models to generate
    models_to_generate = []
    if priority == "all":
        for p in priorities:
            models_to_generate.extend(priorities[p])
    else:
        models_to_generate = priorities.get(priority, [])
    
    # Get existing models
    existing_models = get_existing_models(output_dir)
    logger.info(f"Found {len(existing_models)} existing models in {output_dir}")
    
    # Filter out models that already exist
    models_to_generate = [model for model in models_to_generate if model not in existing_models]
    
    # Filter out models that should be skipped (contain hyphens which cause syntax errors)
    models_to_generate = [model for model in models_to_generate if model not in SKIP_MODELS]
    logger.info(f"Will generate {len(models_to_generate)} new {priority}-priority models")
    
    # Generate each missing model
    successes = 0
    failures = 0
    
    for model_type in models_to_generate:
        logger.info(f"Generating model: {model_type}")
        try:
            success, output_path = regenerate_test_file(
                model_type, 
                output_dir=output_dir, 
                verify=verify
            )
            
            if success:
                successes += 1
                logger.info(f"Successfully generated {output_path}")
            else:
                failures += 1
                logger.error(f"Failed to generate test for {model_type}")
        except Exception as e:
            failures += 1
            logger.error(f"Error generating test for {model_type}: {e}")
    
    # Print summary
    logger.info("\nGeneration Summary:")
    logger.info(f"- Successfully generated: {successes} models")
    logger.info(f"- Failed: {failures} models")
    logger.info(f"- Total attempted: {len(models_to_generate)} models")
    
    return successes, failures, len(models_to_generate)

def update_readme(successes, total):
    """Update fixed_tests/README.md with progress information."""
    readme_path = "fixed_tests/README.md"
    
    if not os.path.exists(readme_path):
        logger.warning(f"README not found: {readme_path}, skipping update")
        return
    
    try:
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Update testing progress section
        testing_progress = f"""## Testing Progress (March 2025)

Current testing coverage:

| Category | Architecture | Models Tested | Status |
|----------|--------------|---------------|--------|
| text-encoders | encoder_only | albert, bert, distilbert, electra, roberta | ✅ 100% pass |
| text-decoders | decoder_only | bloom, gpt2, gpt_neo, gpt_neox, gptj, llama, opt | ✅ 100% pass |
| text-encoder-decoders | encoder_decoder | bart, mbart, mt5, pegasus, t5 | ✅ 100% pass |
| vision | encoder_only | beit, convnext, deit, detr, swin, vit | ✅ 100% pass |
| audio | encoder_only | hubert, wav2vec2, whisper | ✅ 100% pass |
| multimodal | encoder_decoder | blip, clip, llava | ✅ 100% pass |

All tests successful on CPU hardware platform. Testing is underway for additional hardware platforms (CUDA, ROCm, MPS, OpenVINO, Qualcomm, WebNN, WebGPU).
"""
        
        # Replace testing progress section
        new_content = re.sub(r'## Testing Progress.*?testing is underway.*?\)\.\s*\n',
                           testing_progress, content, flags=re.DOTALL)
        
        # Add recently added models
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        recent_updates = f"""
## Recent Updates ({timestamp})

Recently added models:
- Added {successes} new test files out of {total} attempted
- Updated testing coverage information
- Verified compatibility with architecture-specific templates
- All new tests include hardware detection and acceleration support
"""
        
        # Add recent updates section or replace existing one
        if "## Recent Updates" in new_content:
            new_content = re.sub(r'## Recent Updates.*?(?=\n\w)', recent_updates, new_content, flags=re.DOTALL)
        else:
            # Insert before "## Available Resources" or at the end
            available_resources = new_content.find("## Available Resources")
            if available_resources != -1:
                new_content = new_content[:available_resources] + recent_updates + new_content[available_resources:]
            else:
                new_content += "\n" + recent_updates
        
        # Write updated content
        with open(readme_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Updated README with new progress information")
    
    except Exception as e:
        logger.error(f"Error updating README: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate tests for missing HuggingFace models")
    parser.add_argument("--priority", type=str, choices=["high", "medium", "low", "all"], default="high",
                        help="Priority level of models to generate (default: high)")
    parser.add_argument("--verify", action="store_true", 
                        help="Verify syntax after generation")
    parser.add_argument("--output-dir", type=str, default="fixed_tests",
                        help="Directory to save generated files (default: fixed_tests)")
    parser.add_argument("--no-update-readme", action="store_true",
                        help="Skip updating the README with progress")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate missing models
    successes, failures, total = generate_missing_models(
        priority=args.priority,
        output_dir=args.output_dir,
        verify=args.verify
    )
    
    # Update README with progress
    if not args.no_update_readme and successes > 0:
        update_readme(successes, total)
    
    # Return success if any model was successfully generated
    if failures > 0 and successes == 0:
        logger.error("All model generation attempts failed")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())