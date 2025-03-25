#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main generator class for the refactored generator suite.
Orchestrates the generation process by coordinating all components.
"""

import os
import sys
import json
import time
import logging
import datetime
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, Callable


class GeneratorCore:
    """Core generator class for HuggingFace model tests."""
    
    def __init__(self, config, registry, hardware_manager=None, 
                 dependency_manager=None, model_selector=None,
                 syntax_validator=None, syntax_fixer=None):
        """Initialize the generator core.
        
        Args:
            config: The configuration manager instance.
            registry: The component registry instance.
            hardware_manager: Optional hardware detection manager.
            dependency_manager: Optional dependency manager.
            model_selector: Optional model selector.
            syntax_validator: Optional syntax validator.
            syntax_fixer: Optional syntax fixer.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.registry = registry
        self.hardware_manager = hardware_manager
        self.dependency_manager = dependency_manager
        self.model_selector = model_selector
        self.syntax_validator = syntax_validator
        self.syntax_fixer = syntax_fixer
    
    def generate(self, model_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a test file for the given model type.
        
        Args:
            model_type: The model type (bert, gpt2, t5, etc.)
            options: Dictionary of generation options.
            
        Returns:
            Dictionary with generation results.
        """
        start_time = time.time()
        self.logger.info(f"Generating test for model type: {model_type}")
        
        # Get the template for this model
        template = self.registry.get_template(model_type)
        if not template:
            error_msg = f"No template found for model type: {model_type}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model_type": model_type,
                "duration": time.time() - start_time
            }
        
        # Get model information
        model_info = self._get_model_info(model_type, options)
        
        # Get hardware information
        hardware_info = self._get_hardware_info()
        
        # Get dependency information
        dependency_info = self._get_dependency_info()
        
        # Build context for template rendering
        context = self._build_context(model_type, model_info, hardware_info, 
                                     dependency_info, options)
        
        try:
            # Render the template
            content = template.render(context)
            
            # Validate and fix syntax if needed
            content = self._validate_and_fix(content, options)
            
            # Write the output
            output_file = self._get_output_file(model_type, options)
            success = self._write_output(output_file, content)
            
            if not success:
                return {
                    "success": False,
                    "error": f"Failed to write output file: {output_file}",
                    "model_type": model_type,
                    "duration": time.time() - start_time
                }
            
            return {
                "success": True,
                "output_file": str(output_file),
                "model_type": model_type,
                "model_info": model_info,
                "architecture": model_info["architecture"],
                "duration": time.time() - start_time
            }
        
        except Exception as e:
            self.logger.exception(f"Error generating test for model type {model_type}")
            return {
                "success": False,
                "error": str(e),
                "model_type": model_type,
                "duration": time.time() - start_time
            }
    
    def generate_batch(self, model_types: List[str], common_options: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test files for multiple model types.
        
        Args:
            model_types: List of model types to generate.
            common_options: Dictionary of common generation options.
            
        Returns:
            Dictionary with batch generation results.
        """
        start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        
        for model_type in model_types:
            # Generate test for this model type
            result = self.generate(model_type, common_options)
            results.append(result)
            
            if result["success"]:
                success_count += 1
            else:
                error_count += 1
        
        return {
            "success": error_count == 0,
            "results": results,
            "success_count": success_count,
            "error_count": error_count,
            "total_count": len(model_types),
            "duration": time.time() - start_time
        }
    
    def _get_model_info(self, model_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Get model information, using provided options if available.
        
        Args:
            model_type: The model type.
            options: Dictionary of generation options.
            
        Returns:
            Dictionary with model information.
        """
        # If model_info is provided in options, use it
        if "model_info" in options:
            return options["model_info"]
        
        # If model_selector is provided, use it
        if self.model_selector and hasattr(self.model_selector, "select_model"):
            # Extract model selection criteria from options
            task = options.get("task")
            hardware = options.get("hardware")
            max_size = options.get("max_size")
            framework = options.get("framework")
            
            # Select model based on criteria
            model_info = self.model_selector.select_model(
                model_type, 
                task=task, 
                hardware=hardware, 
                max_size=max_size, 
                framework=framework
            )
            
            if model_info:
                return model_info
        
        # Otherwise, get basic model info from registry
        return self.registry.get_model_info(model_type)
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information.
        
        Returns:
            Dictionary with hardware information.
        """
        if self.hardware_manager and hasattr(self.hardware_manager, "detect_all"):
            return self.hardware_manager.detect_all()
        
        # If no hardware manager is provided, use registry
        return self.registry.get_hardware_info()
    
    def _get_dependency_info(self) -> Dict[str, Any]:
        """Get dependency information.
        
        Returns:
            Dictionary with dependency information.
        """
        if self.dependency_manager and hasattr(self.dependency_manager, "check_all"):
            return self.dependency_manager.check_all()
        
        # If no dependency manager is provided, return empty dict
        return {}
    
    def _build_context(self, model_type: str, model_info: Dict[str, Any],
                      hardware_info: Dict[str, Any], dependency_info: Dict[str, Any],
                      options: Dict[str, Any]) -> Dict[str, Any]:
        """Build the template rendering context.
        
        Args:
            model_type: The model type.
            model_info: Dictionary with model information.
            hardware_info: Dictionary with hardware information.
            dependency_info: Dictionary with dependency information.
            options: Dictionary of generation options.
            
        Returns:
            Dictionary with context for template rendering.
        """
        # Start with basic context
        context = {
            "model_type": model_type,
            "model_info": model_info,
            "hardware_info": hardware_info,
            "dependencies": dependency_info,
            "options": options,
            "timestamp": datetime.datetime.now().isoformat(),
            "uuid": str(uuid.uuid4())
        }
        
        # Add hardware availability flags
        context.update({
            "has_cuda": hardware_info.get("cuda", {}).get("available", False),
            "has_rocm": hardware_info.get("rocm", {}).get("available", False),
            "has_mps": hardware_info.get("mps", {}).get("available", False),
            "has_openvino": hardware_info.get("openvino", {}).get("available", False),
            "has_webnn": hardware_info.get("webnn", {}).get("available", False),
            "has_webgpu": hardware_info.get("webgpu", {}).get("available", False)
        })
        
        # Add template-specific context
        if "template_context" in options:
            context.update(options["template_context"])
        
        return context
    
    def _validate_and_fix(self, content: str, options: Dict[str, Any]) -> str:
        """Validate and fix syntax in the generated content.
        
        Args:
            content: The generated content.
            options: Dictionary of generation options.
            
        Returns:
            The validated and fixed content.
        """
        # Check if syntax fixing is enabled
        fix_syntax = options.get("fix_syntax", self.config.get("syntax.auto_fix", True))
        if not fix_syntax:
            return content
        
        # If syntax validator is provided, use it
        if self.syntax_validator and hasattr(self.syntax_validator, "validate"):
            is_valid, error_info = self.syntax_validator.validate(content)
            
            # If valid, return the content as is
            if is_valid:
                return content
            
            # If not valid and syntax fixer is provided, try to fix it
            if not is_valid and self.syntax_fixer and hasattr(self.syntax_fixer, "fix"):
                self.logger.info(f"Fixing syntax errors: {error_info}")
                return self.syntax_fixer.fix(content)
        
        # Otherwise, apply simple fixes
        return self._simple_fixes(content)
    
    def _simple_fixes(self, content: str) -> str:
        """Apply simple syntax fixes to the content.
        
        Args:
            content: The content to fix.
            
        Returns:
            The fixed content.
        """
        # Replace multiple consecutive quotes
        content = content.replace('""""', '"""')
        content = content.replace("''''", "'''")
        
        # Fix incomplete multiline strings
        lines = content.split('\n')
        triple_double_quote_count = 0
        triple_single_quote_count = 0
        
        for i, line in enumerate(lines):
            triple_double_quote_count += line.count('"""')
            triple_single_quote_count += line.count("'''")
        
        # If odd number of triple quotes, add closing quotes
        if triple_double_quote_count % 2 != 0:
            lines.append('"""')
        if triple_single_quote_count % 2 != 0:
            lines.append("'''")
        
        content = '\n'.join(lines)
        
        # Fix missing parentheses
        open_paren_count = content.count('(')
        close_paren_count = content.count(')')
        if open_paren_count > close_paren_count:
            content += ')' * (open_paren_count - close_paren_count)
        
        # Fix missing brackets
        open_bracket_count = content.count('[')
        close_bracket_count = content.count(']')
        if open_bracket_count > close_bracket_count:
            content += ']' * (open_bracket_count - close_bracket_count)
        
        # Fix missing braces
        open_brace_count = content.count('{')
        close_brace_count = content.count('}')
        if open_brace_count > close_brace_count:
            content += '}' * (open_brace_count - close_brace_count)
        
        return content
    
    def _get_output_file(self, model_type: str, options: Dict[str, Any]) -> Path:
        """Get the output file path.
        
        Args:
            model_type: The model type.
            options: Dictionary of generation options.
            
        Returns:
            Path object for the output file.
        """
        # If output_file is provided in options, use it
        if "output_file" in options:
            output_file = options["output_file"]
            if isinstance(output_file, str):
                return Path(output_file)
            return output_file
        
        # Otherwise, construct a path based on the model type
        output_dir = options.get("output_dir", self.config.get("output_dir", "./generated_tests"))
        output_dir = Path(output_dir)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct filename
        filename = f"test_{model_type}.py"
        
        return output_dir / filename
    
    def _write_output(self, output_file: Path, content: str) -> bool:
        """Write the generated content to the output file.
        
        Args:
            output_file: Path to the output file.
            content: Content to write.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Create the parent directory if it doesn't exist
            os.makedirs(output_file.parent, exist_ok=True)
            
            # Write the content
            with open(output_file, 'w') as f:
                f.write(content)
            
            self.logger.info(f"Wrote output to: {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error writing output to {output_file}: {e}")
            return False