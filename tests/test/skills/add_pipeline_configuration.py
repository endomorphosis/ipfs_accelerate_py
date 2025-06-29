#!/usr/bin/env python3
"""
Add Pipeline Configuration

This script adds missing pipeline configurations to HuggingFace model test files.
It identifies files without a transformers.pipeline() call and adds the appropriate
configuration based on the model architecture.

Usage:
    python add_pipeline_configuration.py --directory TESTS_DIR [--dry-run] [--verbose]
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
import traceback
from typing import Dict, List, Set, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import architecture and task mappings from the standardization script
from standardize_task_configurations import (
    ARCHITECTURE_TYPES,
    RECOMMENDED_TASKS,
    SPECIAL_TASK_OVERRIDES,
    TEST_INPUTS
)

class PipelineConfigurator:
    """Adds missing pipeline configurations to HuggingFace model test files."""
    
    def __init__(self, directory: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the configurator.
        
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
            "files_skipped": 0,
            "files_with_errors": 0,
            "by_architecture": {}
        }
    
    def run(self):
        """Run the pipeline configuration process."""
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
                logger.error(traceback.format_exc())
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
                "total": 0, "updated": 0, "skipped": 0, "errors": 0
            }
        self.stats["by_architecture"][architecture]["total"] += 1
        
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if the file already has a pipeline configuration
        pipeline_match = re.search(r'transformers\.pipeline\(\s*["\']([^"\']+)["\']', content)
        
        if pipeline_match:
            # Already has pipeline configuration, skip it
            self.stats["files_skipped"] += 1
            self.stats["by_architecture"][architecture]["skipped"] += 1
            
            if self.verbose:
                logger.info(f"✓ Already has pipeline: {file_path}")
            return
        
        # No pipeline configuration found, need to add one
        if self.verbose:
            logger.info(f"Adding pipeline configuration to {file_path}")
        
        # Get the recommended task for this model
        task = self._get_recommended_task(model_name, architecture)
        test_input = TEST_INPUTS.get(task, f'"The model is processing data for {task}."')
        
        # Look for the test_pipeline method
        test_pipeline_match = re.search(r'def\s+test_pipeline\s*\([^)]*\):\s*(?:""".*?""")?\s*(.*?)(?=\s*def|\s*$)', content, re.DOTALL)
        
        if test_pipeline_match:
            # Found test_pipeline method, add pipeline configuration
            method_body = test_pipeline_match.group(1)
            
            # Check if method has try-except block
            try_match = re.search(r'try:', method_body)
            
            if try_match:
                # Insert after the try block
                updated_content = content.replace(
                    method_body,
                    self._insert_after_pattern(
                        method_body,
                        r'try:',
                        f'''
            # Initialize the pipeline with the appropriate task
            pipe = transformers.pipeline(
                "{task}", 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {{load_time:.2f}} seconds")
            
            # Test with a task-appropriate input
            test_input = {test_input}
'''
                    )
                )
            else:
                # No try block, insert after checking for transformers availability
                updated_content = content.replace(
                    method_body,
                    self._insert_after_pattern(
                        method_body,
                        r'if not HAS_TRANSFORMERS:.*?return.*?}',
                        f'''
            
            # Record start time
            start_time = time.time()
            
            # Initialize the pipeline with the appropriate task
            pipe = transformers.pipeline(
                "{task}", 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {{load_time:.2f}} seconds")
            
            # Test with a task-appropriate input
            test_input = {test_input}
'''
                    )
                )
            
            # Write the updated content
            if not self.dry_run:
                with open(file_path, 'w') as f:
                    f.write(updated_content)
            
            self.stats["files_updated"] += 1
            self.stats["by_architecture"][architecture]["updated"] += 1
            
            if self.verbose:
                logger.info(f"✅ Added pipeline configuration to {file_path}")
        else:
            # Could not find test_pipeline method
            logger.warning(f"Could not find test_pipeline method in {file_path}")
            self.stats["files_with_errors"] += 1
            self.stats["by_architecture"][architecture]["errors"] += 1
    
    def _insert_after_pattern(self, text: str, pattern: str, insertion: str) -> str:
        """Insert text after a pattern match."""
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return text
        
        match_end = match.end()
        return text[:match_end] + insertion + text[match_end:]
    
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
        """Print a summary of the configuration process."""
        total = self.stats["total_files"]
        updated = self.stats["files_updated"]
        skipped = self.stats["files_skipped"]
        errors = self.stats["files_with_errors"]
        
        logger.info("\n" + "="*50)
        logger.info("Pipeline Configuration Summary")
        logger.info("="*50)
        logger.info(f"Total files processed: {total}")
        logger.info(f"Files updated with pipeline config: {updated} ({updated/total*100:.1f}%)")
        logger.info(f"Files already had pipeline: {skipped} ({skipped/total*100:.1f}%)")
        logger.info(f"Files with errors: {errors} ({errors/total*100:.1f}%)")
        logger.info("\nBy architecture:")
        
        for arch, stats in sorted(self.stats["by_architecture"].items()):
            arch_total = stats["total"]
            arch_updated = stats["updated"]
            arch_skipped = stats["skipped"]
            arch_errors = stats["errors"]
            
            logger.info(f"  {arch}: {arch_total} files")
            logger.info(f"    - Updated: {arch_updated} ({arch_updated/arch_total*100:.1f}%)")
            logger.info(f"    - Already had pipeline: {arch_skipped} ({arch_skipped/arch_total*100:.1f}%)")
            logger.info(f"    - Errors: {arch_errors} ({arch_errors/arch_total*100:.1f}%)")
        
        if self.dry_run:
            logger.info("\nThis was a dry run. No files were actually modified.")

def main():
    """Main entry point for the pipeline configurator."""
    parser = argparse.ArgumentParser(description="Add pipeline configurations to HuggingFace model test files")
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
    
    # Create and run the configurator
    configurator = PipelineConfigurator(directory, args.dry_run, args.verbose)
    configurator.run()

if __name__ == "__main__":
    main()