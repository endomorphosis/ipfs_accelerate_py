#!/usr/bin/env python3
"""
Model generator module for the refactored generator suite.

This module provides functionality to generate model-specific code
based on templates and architecture detection.
"""

import os
import re
import sys
import time
import json
import logging
import importlib
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from scripts.generators.architecture_detector import (
    get_architecture_type,
    get_model_metadata,
    get_model_class_name,
    get_model_class_name_short,
    get_default_model_id,
    normalize_model_name
)
from hardware.hardware_detection import (
    get_optimal_device,
    get_model_hardware_recommendations,
    is_device_compatible_with_model,
    get_device_code_snippet
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define priority levels for models
PRIORITY_MODELS = {
    "critical": [
        "bert", "roberta", "gpt2", "t5", "llama", "mistral", "vit", "clip", "whisper"
    ],
    "high": [
        "albert", "electra", "deberta", "camembert", "xlm-roberta", "distilbert",
        "opt", "phi", "bloomz", "falcon", "flan-t5", "bart", "mbart", "pegasus",
        "beit", "deit", "swin", "convnext", "dino", "blip", "wav2vec2", "hubert",
        "flava", "mixtral"
    ],
    "medium": [
        "layoutlm", "canine", "roformer", "transfo-xl", "ctrl", "gpt-neo", "gpt-j", 
        "marian", "m2m100", "prophetnet", "reformer", "big-bird", "detr", "segformer",
        "yolos", "donut", "imagebind", "stable-diffusion"
    ],
    "low": [
        "mt5", "longt5", "led", "fsmt", "rag", "mlp-mixer", "mobilenet", "nllb",
        "trocr", "clap", "musicgen", "videomae", "mamba"
    ]
}


class ModelSkillsetGenerator:
    """Generator for model skillset files from templates."""
    
    def __init__(self, template_dir: str = None, output_dir: str = None):
        """
        Initialize the model skillset generator.
        
        Args:
            template_dir: Directory containing templates. If None, uses default.
            output_dir: Directory for generated files. If None, uses default.
        """
        # Set directories
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.template_dir = template_dir or os.path.join(base_dir, "templates")
        self.output_dir = output_dir or os.path.join(base_dir, "skillsets")
        
        # Ensure directories exist
        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Map of architecture types to template files
        self.arch_templates = {
            "encoder-only": "encoder_only_template.py",
            "decoder-only": "decoder_only_template.py",
            "encoder-decoder": "encoder_decoder_template.py",
            "vision": "vision_template.py",
            "vision-encoder-text-decoder": "vision_text_template.py",
            "speech": "speech_template.py",
            "multimodal": "multimodal_template.py",
            "diffusion": "diffusion_model_template.py",
            "mixture-of-experts": "moe_model_template.py",
            "state-space": "ssm_model_template.py",
            "rag": "rag_model_template.py"
        }
        
        logger.info(f"Initialized ModelSkillsetGenerator with templates from {self.template_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def get_template_for_architecture(self, arch_type: str) -> str:
        """
        Get the template path for a given architecture type.
        
        Args:
            arch_type: The architecture type.
            
        Returns:
            Path to the template file.
        """
        if arch_type not in self.arch_templates:
            logger.warning(f"Unknown architecture type: {arch_type}, using encoder-only as fallback")
            arch_type = "encoder-only"
        
        template_file = self.arch_templates[arch_type]
        template_path = os.path.join(self.template_dir, template_file)
        
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        return template_path
    
    def fill_template(self, template_content: str, model_name: str, arch_type: str, device: str) -> str:
        """
        Fill in the template with model-specific information.
        
        Args:
            template_content: The template content.
            model_name: The model name.
            arch_type: The architecture type.
            device: The target device.
            
        Returns:
            The filled template content.
        """
        # Get model metadata
        metadata = get_model_metadata(model_name)
        
        # Extract information from metadata
        model_type = metadata["model_type"]
        model_type_upper = model_type.upper()
        model_class_name = metadata["model_class_name"]
        processor_class_name = metadata["processor_class_name"]
        model_class_name_short = metadata["model_class_name_short"]
        default_model_id = metadata["default_model_id"]
        
        # Create class name for the skillset
        model_parts = re.split(r'[-_]', model_type)
        skillset_class_name = ''.join(part.capitalize() for part in model_parts) + "Skillset"
        test_class_name = "Test" + ''.join(part.capitalize() for part in model_parts) + "Model"
        
        # Get device initialization code
        raw_device_code = get_device_code_snippet(device, model_class_name, arch_type)
        
        # Fix indentation for device code
        # Find where the device_init_code placeholder is in the template and determine its indentation
        placeholder = "{device_init_code}"
        if placeholder in template_content:
            lines = template_content.splitlines()
            for line in lines:
                if placeholder in line:
                    indent = line[:line.find(placeholder)]
                    # Add indentation to each line of device code
                    device_init_code = ""
                    for device_line in raw_device_code.splitlines():
                        if device_line.strip():  # If line is not empty
                            device_init_code += indent + device_line.lstrip() + "\n"
                    break
            else:
                # If placeholder is not found in lines, use raw device code
                device_init_code = raw_device_code
        else:
            device_init_code = raw_device_code
        
        # Define replacements
        replacements = {
            "{model_type}": model_type,
            "{model_type_upper}": model_type_upper,
            "{model_class_name}": model_class_name,
            "{processor_class_name}": processor_class_name,
            "{model_class_name_short}": model_class_name_short,
            "{default_model_id}": default_model_id,
            "{architecture_type}": arch_type,
            "{test_class_name}": test_class_name,
            "{skillset_class_name}": skillset_class_name,
            "{device_init_code}": device_init_code.rstrip(),  # Remove trailing newlines
            "{sampling_rate}": "16000"  # Default sampling rate for speech models
        }
        
        # Replace placeholders in template
        filled_content = template_content
        for placeholder, value in replacements.items():
            filled_content = filled_content.replace(placeholder, str(value))
        
        return filled_content
    
    def generate_skillset_file(self, model_name: str, device: str = "all", force: bool = False, verify: bool = True) -> Tuple[bool, List[str]]:
        """
        Generate a skillset file for a specific model.
        
        Args:
            model_name: The model name.
            device: The target device, or "all" for all compatible devices.
            force: Whether to overwrite existing files.
            verify: Whether to verify generated files with Python syntax check.
            
        Returns:
            Tuple of (success, list of generated file paths).
        """
        logger.info(f"Generating skillset for model: {model_name}, device: {device}")
        
        # Determine architecture type
        arch_type = get_architecture_type(model_name)
        logger.info(f"Detected architecture type: {arch_type}")
        
        # Get template
        try:
            template_path = self.get_template_for_architecture(arch_type)
            with open(template_path, 'r') as f:
                template_content = f.read()
        except FileNotFoundError as e:
            logger.error(f"Error reading template: {e}")
            return False, []
        
        # Normalize model name for file naming
        model_type = normalize_model_name(model_name)
        
        # Determine target devices
        devices_to_generate = []
        if device == "all":
            # Get all compatible devices
            devices_to_generate = get_model_hardware_recommendations(arch_type)
        else:
            # Check if specified device is compatible
            if is_device_compatible_with_model(device, arch_type):
                devices_to_generate = [device]
            else:
                logger.warning(f"Device {device} is not compatible with {arch_type}. Using CPU instead.")
                devices_to_generate = ["cpu"]
        
        # Generate files for each device
        generated_files = []
        success = True
        
        for dev in devices_to_generate:
            # Generate output file path
            output_filename = f"{model_type}_{dev}_skillset.py"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Check if file exists
            if os.path.exists(output_path) and not force:
                logger.warning(f"File already exists: {output_path}. Use force=True to overwrite.")
                continue
            
            # Fill template
            filled_content = self.fill_template(template_content, model_name, arch_type, dev)
            
            # Write to file
            try:
                with open(output_path, 'w') as f:
                    f.write(filled_content)
                logger.info(f"Generated file: {output_path}")
                generated_files.append(output_path)
            except Exception as e:
                logger.error(f"Error writing file {output_path}: {e}")
                success = False
                continue
            
            # Verify file
            if verify and not self.verify_skillset_file(output_path):
                logger.error(f"Verification failed for {output_path}")
                success = False
        
        return success, generated_files
    
    def verify_skillset_file(self, file_path: str) -> bool:
        """
        Verify a generated skillset file for syntax correctness.
        
        Args:
            file_path: Path to the file to verify.
            
        Returns:
            True if verification succeeded, False otherwise.
        """
        try:
            # Use Python's builtin compile function to check syntax
            with open(file_path, 'r') as f:
                content = f.read()
            
            compile(content, file_path, 'exec')
            logger.info(f"Syntax verification passed for {file_path}")
            return True
        
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return False
    
    def generate_models_by_priority(self, priority: str = "critical", device: str = "all", verify: bool = True, force: bool = False) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Generate skillset files for models of a given priority level.
        
        Args:
            priority: Priority level ("critical", "high", "medium", "low").
            device: Target device, or "all" for all compatible devices.
            verify: Whether to verify generated files.
            force: Whether to overwrite existing files.
            
        Returns:
            Tuple of (overall success, dict mapping model names to generated file paths).
        """
        if priority not in PRIORITY_MODELS:
            logger.error(f"Unknown priority level: {priority}")
            return False, {}
        
        # Get models for this priority level
        models = PRIORITY_MODELS[priority]
        logger.info(f"Generating {len(models)} models with priority '{priority}'")
        
        # Generate each model
        results = {}
        overall_success = True
        
        for model_name in models:
            success, files = self.generate_skillset_file(model_name, device, force, verify)
            results[model_name] = files
            
            if not success:
                overall_success = False
                logger.warning(f"Generation failed or partially succeeded for {model_name}")
        
        # Summarize results
        total_files = sum(len(files) for files in results.values())
        logger.info(f"Generated {total_files} files for {len(models)} models with priority '{priority}'")
        
        return overall_success, results
    
    def generate_all_models(self, device: str = "all", verify: bool = True, force: bool = False) -> Tuple[bool, Dict[str, Dict[str, List[str]]]]:
        """
        Generate skillset files for all models in all priority levels.
        
        Args:
            device: Target device, or "all" for all compatible devices.
            verify: Whether to verify generated files.
            force: Whether to overwrite existing files.
            
        Returns:
            Tuple of (overall success, nested dict mapping priority to model results).
        """
        # Process each priority level
        overall_success = True
        results = {}
        
        for priority in PRIORITY_MODELS.keys():
            success, priority_results = self.generate_models_by_priority(priority, device, verify, force)
            results[priority] = priority_results
            
            if not success:
                overall_success = False
        
        # Summarize results
        total_files = sum(len(files) for priority_dict in results.values() for files in priority_dict.values())
        total_models = sum(len(priority_dict) for priority_dict in results.values())
        logger.info(f"Generated {total_files} files for {total_models} models across all priority levels")
        
        return overall_success, results


# For direct execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate model skillset files")
    parser.add_argument("--model", "-m", type=str, help="Specific model to generate")
    parser.add_argument("--device", "-d", type=str, default="all", 
                       choices=["all", "cpu", "cuda", "rocm", "mps", "openvino", "qnn"],
                       help="Target device")
    parser.add_argument("--priority", "-p", type=str, default="critical",
                       choices=["critical", "high", "medium", "low", "all"],
                       help="Model priority level to generate")
    parser.add_argument("--template-dir", "-t", type=str, help="Template directory")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing files")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification of generated files")
    
    args = parser.parse_args()
    
    # Create generator
    generator = ModelSkillsetGenerator(args.template_dir, args.output_dir)
    
    # Determine what to generate
    if args.model:
        # Generate single model
        success, files = generator.generate_skillset_file(
            args.model, args.device, args.force, not args.no_verify
        )
        
        if success:
            print(f"Successfully generated {len(files)} file(s) for {args.model}:")
            for file_path in files:
                print(f"  {file_path}")
            sys.exit(0)
        else:
            print(f"Failed to generate all files for {args.model}")
            sys.exit(1)
    
    else:
        # Generate models by priority
        if args.priority == "all":
            success, results = generator.generate_all_models(
                args.device, not args.no_verify, args.force
            )
        else:
            success, results = generator.generate_models_by_priority(
                args.priority, args.device, not args.no_verify, args.force
            )
        
        if success:
            print(f"Successfully generated all files")
            sys.exit(0)
        else:
            print(f"Failed to generate some files")
            sys.exit(1)