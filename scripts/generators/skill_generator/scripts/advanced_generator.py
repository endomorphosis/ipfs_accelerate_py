#!/usr/bin/env python3
"""
Advanced Generator

This script provides an advanced interface for generating and fixing test files
with enhanced features like automatic validation and fixing.
"""

import os
import sys
import json
import logging
import argparse
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

# Add parent directory to path to allow imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from generator_core.generator import GeneratorCore
from generator_core.config import ConfigManager
from model_selection.registry import ModelRegistry
from templates import get_template_for_architecture
from syntax.validator import SyntaxValidator
from syntax.fixer import SyntaxFixer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedGenerator:
    """
    Advanced generator with validation and automatic fixing.
    """
    
    def __init__(self, config_file=None, output_dir=None):
        """
        Initialize the advanced generator.
        
        Args:
            config_file: Path to configuration file
            output_dir: Directory to output generated files
        """
        self.config = ConfigManager(config_file)
        self.registry = ModelRegistry(self.config)
        self.generator = GeneratorCore(config=self.config)
        self.validator = SyntaxValidator()
        self.fixer = SyntaxFixer()
        self.output_dir = output_dir or "generated_tests"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            "generated": 0,
            "fixed": 0,
            "errors": 0
        }
    
    def validate_and_fix(self, file_path: str) -> bool:
        """
        Validate and fix a generated file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is valid (or was fixed), False otherwise
        """
        # Validate the file
        validation_result = self.validator.validate_file(file_path)
        
        if validation_result["valid"]:
            return True
        
        # Try to fix the file
        logger.info(f"Fixing issues in {file_path}...")
        
        with open(file_path, "r") as f:
            content = f.read()
        
        # Apply fixes
        fixed_content = self.fixer.fix_content(content)
        
        # Write fixed content
        with open(file_path, "w") as f:
            f.write(fixed_content)
        
        # Validate again
        validation_result = self.validator.validate_file(file_path)
        
        if validation_result["valid"]:
            self.stats["fixed"] += 1
            return True
        else:
            self.stats["errors"] += 1
            return False
    
    def generate_file(self, model_info: Dict[str, Any]) -> Optional[str]:
        """
        Generate a test file for a model.
        
        Args:
            model_info: Model information dictionary
            
        Returns:
            Path to the generated file or None if generation failed
        """
        architecture = model_info.get("architecture")
        if not architecture:
            logger.error(f"No architecture specified for model: {model_info}")
            return None
        
        # Get the appropriate template
        template_class = get_template_for_architecture(architecture)
        if not template_class:
            logger.error(f"No template found for architecture: {architecture}")
            return None
        
        template = template_class()
        
        # Prepare context
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context = {
            "model_info": model_info,
            "timestamp": timestamp,
            "has_openvino": model_info.get("has_openvino", False)
        }
        
        # Generate content
        file_content = template.render(context)
        
        # Create filename
        model_type = model_info["type"]
        filename = f"test_hf_{model_type}.py"
        file_path = os.path.join(self.output_dir, filename)
        
        # Write file
        with open(file_path, "w") as f:
            f.write(file_content)
        
        # Validate and fix
        if self.validate_and_fix(file_path):
            self.stats["generated"] += 1
            logger.info(f"Generated valid test file: {filename}")
            return file_path
        else:
            logger.error(f"Generated file has unfixable issues: {filename}")
            return None
    
    def generate_batch(self, models: List[Dict[str, Any]]) -> List[str]:
        """
        Generate test files for a batch of models.
        
        Args:
            models: List of model information dictionaries
            
        Returns:
            List of paths to generated files
        """
        generated_files = []
        
        for model_info in models:
            file_path = self.generate_file(model_info)
            if file_path:
                generated_files.append(file_path)
        
        return generated_files
    
    def search_huggingface_api(self, query: str, model_type: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the HuggingFace API for models.
        
        Args:
            query: Search query
            model_type: Model type to filter by
            limit: Maximum number of results
            
        Returns:
            List of model information dictionaries
        """
        try:
            # Try to import requests
            import requests
            
            # Set up API URL
            api_url = f"https://huggingface.co/api/models?search={query}&sort=downloads&direction=-1&limit={limit}"
            
            # Make request
            response = requests.get(api_url)
            data = response.json()
            
            # Map to model_info format
            models = []
            for model in data:
                # Determine architecture based on pipeline tag or model type
                architecture = None
                tags = model.get("pipeline_tag", [])
                
                if "fill-mask" in tags:
                    architecture = "encoder-only"
                elif "text-generation" in tags:
                    architecture = "decoder-only"
                elif "translation" in tags or "summarization" in tags:
                    architecture = "encoder-decoder"
                elif "image-classification" in tags:
                    architecture = "vision"
                elif "zero-shot-image-classification" in tags:
                    architecture = "vision-text"
                elif "automatic-speech-recognition" in tags:
                    architecture = "speech"
                else:
                    # Determine from model_type
                    if model_type in ["bert", "roberta", "distilbert", "electra"]:
                        architecture = "encoder-only"
                    elif model_type in ["gpt2", "llama", "mistral", "falcon"]:
                        architecture = "decoder-only"
                    elif model_type in ["t5", "bart", "pegasus"]:
                        architecture = "encoder-decoder"
                    elif model_type in ["vit", "resnet", "swin"]:
                        architecture = "vision"
                    elif model_type in ["clip", "blip"]:
                        architecture = "vision-text"
                    elif model_type in ["whisper", "wav2vec2"]:
                        architecture = "speech"
                
                if architecture:
                    models.append({
                        "name": model["id"],
                        "type": model_type,
                        "architecture": architecture
                    })
            
            return models
        except Exception as e:
            logger.error(f"Error searching HuggingFace API: {str(e)}")
            return []
    
    def generate_from_huggingface(self, model_type: str, query: str = None, limit: int = 5) -> List[str]:
        """
        Generate test files by searching HuggingFace.
        
        Args:
            model_type: Model type to search for
            query: Optional search query
            limit: Maximum number of models to generate
            
        Returns:
            List of paths to generated files
        """
        # Use model_type as query if not provided
        query = query or model_type
        
        # Search for models
        models = self.search_huggingface_api(query, model_type, limit)
        
        if not models:
            logger.error(f"No models found for query: {query}")
            return []
        
        # Generate files
        return self.generate_batch(models)
    
    def print_stats(self):
        """Print generation statistics."""
        print("\n\033[1mGeneration Statistics\033[0m")
        print(f"Generated files: {self.stats['generated']}")
        print(f"Fixed files: {self.stats['fixed']}")
        print(f"Files with errors: {self.stats['errors']}")

def generate_from_file(input_file: str, output_dir: str = "generated_tests") -> List[str]:
    """
    Generate test files from a JSON file containing model information.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to output generated files
        
    Returns:
        List of paths to generated files
    """
    try:
        # Load model information
        with open(input_file, "r") as f:
            data = json.load(f)
        
        # Initialize generator
        generator = AdvancedGenerator(output_dir=output_dir)
        
        # Generate files
        if isinstance(data, list):
            return generator.generate_batch(data)
        elif isinstance(data, dict) and "models" in data:
            return generator.generate_batch(data["models"])
        else:
            logger.error(f"Invalid input file format: {input_file}")
            return []
    except Exception as e:
        logger.error(f"Error generating from file: {str(e)}")
        return []

def generate_from_report(report_file: str, priority: str = "high", architecture: str = None, 
                      batch_size: int = 10, start_index: int = 0, 
                      output_dir: str = "generated_tests") -> List[str]:
    """
    Generate test files from a coverage report.
    
    Args:
        report_file: Path to coverage report JSON file
        priority: Priority level to generate (high, medium)
        architecture: Architecture to filter by
        batch_size: Maximum number of models to generate
        start_index: Index to start from
        output_dir: Directory to output generated files
        
    Returns:
        List of paths to generated files
    """
    try:
        # Load report data
        with open(report_file, "r") as f:
            data = json.load(f)
        
        # Extract models based on criteria
        if architecture:
            models = data.get("missing_by_architecture", {}).get(architecture, [])
        else:
            models = data.get(f"{priority}_priority_models", [])
            models = [model for model in models if not model.get("implemented", False)]
        
        # Apply batch limits
        end_index = min(start_index + batch_size, len(models))
        batch_models = models[start_index:end_index]
        
        # Initialize generator
        generator = AdvancedGenerator(output_dir=output_dir)
        
        # Generate files
        generated = generator.generate_batch(batch_models)
        
        # Print statistics
        generator.print_stats()
        
        # Print next command to run
        if end_index < len(models):
            next_index = end_index
            logger.info("\nTo generate the next batch, run:")
            
            if architecture:
                logger.info(f"python {sys.argv[0]} --report {report_file} --output-dir {output_dir} " +
                          f"--batch-size {batch_size} --start-index {next_index} --architecture {architecture}")
            else:
                logger.info(f"python {sys.argv[0]} --report {report_file} --output-dir {output_dir} " +
                          f"--batch-size {batch_size} --start-index {next_index} --priority {priority}")
        else:
            logger.info(f"\nAll {'architecture' if architecture else priority + ' priority'} models have been generated!")
        
        return generated
    except Exception as e:
        logger.error(f"Error generating from report: {str(e)}")
        return []

def generate_from_huggingface(model_type: str, query: str = None, limit: int = 5, 
                           output_dir: str = "generated_tests") -> List[str]:
    """
    Generate test files by searching HuggingFace.
    
    Args:
        model_type: Model type to search for
        query: Optional search query
        limit: Maximum number of models to generate
        output_dir: Directory to output generated files
        
    Returns:
        List of paths to generated files
    """
    # Initialize generator
    generator = AdvancedGenerator(output_dir=output_dir)
    
    # Generate files
    generated = generator.generate_from_huggingface(model_type, query, limit)
    
    # Print statistics
    generator.print_stats()
    
    return generated

def main():
    parser = argparse.ArgumentParser(description="Advanced model test generator")
    parser.add_argument("--output-dir", default="generated_tests", help="Directory to output generated files")
    
    # Input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", help="JSON file containing model information")
    input_group.add_argument("--report", help="Coverage report JSON file")
    input_group.add_argument("--huggingface", dest="model_type", help="Generate from HuggingFace API (provide model type)")
    
    # Report options
    parser.add_argument("--priority", choices=["high", "medium"], default="high", help="Priority level to generate (for report)")
    parser.add_argument("--architecture", help="Generate models for a specific architecture (for report)")
    parser.add_argument("--batch-size", type=int, default=10, help="Maximum number of models to generate (for report)")
    parser.add_argument("--start-index", type=int, default=0, help="Index to start from (for report)")
    
    # HuggingFace options
    parser.add_argument("--query", help="Search query for HuggingFace API")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of models to generate (for HuggingFace)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.file:
        # Generate from file
        generate_from_file(args.file, args.output_dir)
    elif args.report:
        # Generate from report
        generate_from_report(
            args.report,
            args.priority,
            args.architecture,
            args.batch_size,
            args.start_index,
            args.output_dir
        )
    elif args.model_type:
        # Generate from HuggingFace
        generate_from_huggingface(
            args.model_type,
            args.query,
            args.limit,
            args.output_dir
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())