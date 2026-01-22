#!/usr/bin/env python3
"""
Standardize Task Configurations

This script standardizes the task configurations in HuggingFace model test files to ensure
they use appropriate tasks for each model architecture. It fixes common issues identified
in the validation process, focusing on pipeline task configuration.

Usage:
    python standardize_task_configurations.py --directory TESTS_DIR [--dry-run] [--verbose]
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define architecture types for model mapping
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta", "albert"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt", "gpt_j"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan", "speecht5"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2", "resnet"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava", "vision_text_dual_encoder"]
}

# Define appropriate tasks for each architecture type
RECOMMENDED_TASKS = {
    "encoder-only": "fill-mask",
    "decoder-only": "text-generation",
    "encoder-decoder": "text2text-generation",
    "vision": "image-classification",
    "vision-text": "image-to-text",
    "speech": "automatic-speech-recognition",
    "multimodal": "image-to-text"
}

# Special case overrides for specific models
SPECIAL_TASK_OVERRIDES = {
    "clip": "zero-shot-image-classification",
    "chinese-clip": "zero-shot-image-classification",
    "vision-text-dual-encoder": "zero-shot-image-classification",
    "vision_text_dual_encoder": "zero-shot-image-classification",
    "wav2vec2-bert": "automatic-speech-recognition",
    "speech-to-text": "automatic-speech-recognition",
    "speech-to-text-2": "translation",
    "speecht5": "automatic-speech-recognition",
    "blip-2": "image-to-text",
    "video-llava": "image-to-text",  # Changed from video-to-text which isn't supported
    "conditional-detr": "object-detection",
    "detr": "object-detection",
    "mask2former": "image-segmentation",
    "segformer": "image-segmentation",
    "sam": "image-segmentation",
    "gpt_j": "text-generation",
    "pix2struct": "image-to-text"
}

# Define test_input patterns for each task
TEST_INPUTS = {
    "fill-mask": '"The <mask> is a language model."',
    "text-generation": '"This model can"',
    "text2text-generation": '"translate English to French: Hello, how are you?"',
    "image-classification": '"An image of a cat."',
    "image-to-text": '"An image of a landscape."',
    "automatic-speech-recognition": '"A short audio clip."',
    "zero-shot-image-classification": '"An image with labels: dog, cat, bird."',
    "translation": '"Hello, how are you?"',
    "object-detection": '"An image of a street scene."',
    "image-segmentation": '"An image for segmentation."'
}

class TaskStandardizer:
    """Standardizes task configurations in HuggingFace model test files."""
    
    def __init__(self, directory: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the standardizer.
        
        Args:
            directory: Directory containing test files
            dry_run: If True, don't actually write changes to files
            verbose: If True, print verbose output
        """
        self.directory = Path(directory)
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = {
            "total_files": 0,
            "files_updated": 0,
            "files_already_correct": 0,
            "files_with_errors": 0,
            "by_architecture": {}
        }
    
    def run(self):
        """Run the standardization process."""
        # Find all test files
        test_files = list(self.directory.glob("test_hf_*.py"))
        self.stats["total_files"] = len(test_files)
        
        logger.info(f"Found {len(test_files)} test files to process")
        
        # Process each file
        for file_path in test_files:
            try:
                self._process_file(file_path)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                self.stats["files_with_errors"] += 1
        
        # Print summary
        self._print_summary()
    
    def _process_file(self, file_path: Path):
        """Process a single test file."""
        model_name = self._extract_model_name(file_path)
        architecture = self._get_model_architecture(model_name)
        
        # Update architecture stats
        if architecture not in self.stats["by_architecture"]:
            self.stats["by_architecture"][architecture] = {
                "total": 0, "updated": 0, "already_correct": 0, "errors": 0
            }
        self.stats["by_architecture"][architecture]["total"] += 1
        
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if the file already has a pipeline configuration
        pipeline_match = re.search(r'transformers\.pipeline\(\s*["\']([^"\']+)["\']', content)
        
        if pipeline_match:
            current_task = pipeline_match.group(1)
            recommended_task = self._get_recommended_task(model_name, architecture)
            
            if current_task != recommended_task:
                # Need to update the task
                if self.verbose:
                    logger.info(f"Updating task for {model_name} from '{current_task}' to '{recommended_task}'")
                
                # Replace the task
                updated_content = re.sub(
                    r'transformers\.pipeline\(\s*["\']([^"\']+)["\']',
                    f'transformers.pipeline("{recommended_task}"',
                    content
                )
                
                # Also update the test input if needed
                if recommended_task in TEST_INPUTS:
                    test_input_pattern = r'test_input\s*=\s*["\'][^"\']*["\']'
                    test_input_replacement = f'test_input = {TEST_INPUTS[recommended_task]}'
                    
                    # Check if test_input exists
                    if re.search(test_input_pattern, updated_content):
                        updated_content = re.sub(
                            test_input_pattern,
                            test_input_replacement,
                            updated_content
                        )
                
                # Write the updated content
                if not self.dry_run:
                    with open(file_path, 'w') as f:
                        f.write(updated_content)
                
                self.stats["files_updated"] += 1
                self.stats["by_architecture"][architecture]["updated"] += 1
                
                if self.verbose:
                    logger.info(f"✅ Updated: {file_path}")
            else:
                # Task is already correct
                self.stats["files_already_correct"] += 1
                self.stats["by_architecture"][architecture]["already_correct"] += 1
                
                if self.verbose:
                    logger.info(f"✓ Already correct: {file_path}")
        else:
            # No pipeline configuration found, need to add one
            if self.verbose:
                logger.warning(f"No pipeline configuration found in {file_path}")
            
            # This is a more complex fix that would require understanding the file structure
            # For now, we'll just log it as an error
            self.stats["files_with_errors"] += 1
            self.stats["by_architecture"][architecture]["errors"] += 1
            
            if self.verbose:
                logger.error(f"❌ No pipeline configuration found: {file_path}")
    
    def _extract_model_name(self, file_path: Path) -> str:
        """Extract the model name from a test file path."""
        return file_path.stem.replace("test_hf_", "")
    
    def _get_model_architecture(self, model_name: str) -> str:
        """Determine the model architecture from the model name."""
        model_name_lower = model_name.lower()
        
        for arch_type, models in ARCHITECTURE_TYPES.items():
            for model in models:
                if model.lower() in model_name_lower:
                    return arch_type
        
        return "unknown"
    
    def _get_recommended_task(self, model_name: str, architecture: str) -> str:
        """Get the recommended task for a model."""
        # Check for special case overrides
        for special_model, task in SPECIAL_TASK_OVERRIDES.items():
            if special_model.lower() in model_name.lower():
                return task
        
        # Otherwise use the architecture default
        return RECOMMENDED_TASKS.get(architecture, "fill-mask")
    
    def _print_summary(self):
        """Print a summary of the standardization process."""
        total = self.stats["total_files"]
        updated = self.stats["files_updated"]
        already_correct = self.stats["files_already_correct"]
        errors = self.stats["files_with_errors"]
        
        logger.info("\n" + "="*50)
        logger.info("Task Standardization Summary")
        logger.info("="*50)
        logger.info(f"Total files processed: {total}")
        logger.info(f"Files updated: {updated} ({updated/total*100:.1f}%)")
        logger.info(f"Files already correct: {already_correct} ({already_correct/total*100:.1f}%)")
        logger.info(f"Files with errors: {errors} ({errors/total*100:.1f}%)")
        logger.info("\nBy architecture:")
        
        for arch, stats in sorted(self.stats["by_architecture"].items()):
            arch_total = stats["total"]
            arch_updated = stats["updated"]
            arch_correct = stats["already_correct"]
            arch_errors = stats["errors"]
            
            logger.info(f"  {arch}: {arch_total} files")
            logger.info(f"    - Updated: {arch_updated} ({arch_updated/arch_total*100:.1f}%)")
            logger.info(f"    - Already correct: {arch_correct} ({arch_correct/arch_total*100:.1f}%)")
            logger.info(f"    - Errors: {arch_errors} ({arch_errors/arch_total*100:.1f}%)")
        
        if self.dry_run:
            logger.info("\nThis was a dry run. No files were actually modified.")

def main():
    """Main entry point for the task standardizer."""
    parser = argparse.ArgumentParser(description="Standardize task configurations in HuggingFace model test files")
    parser.add_argument("--directory", "-d", type=str, default="fixed_tests",
                        help="Directory containing test files")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Don't actually write changes to files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print verbose output")
    
    args = parser.parse_args()
    
    # Resolve directory path
    directory = args.directory
    if not os.path.isabs(directory):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    
    # Create and run the standardizer
    standardizer = TaskStandardizer(directory, args.dry_run, args.verbose)
    standardizer.run()

if __name__ == "__main__":
    main()