#!/usr/bin/env python3
"""
Template Renderer for End-to-End Testing Framework

This module provides a template rendering system that works with the TemplateDatabase
to render templates for model skills, tests, benchmarks, and documentation. The renderer
handles variable substitution, template inheritance, and model-specific customizations.

Usage:
    renderer = TemplateRenderer(db_path="./template_database.duckdb")
    rendered_content = renderer.render_template(
        model_name="bert-base-uncased",
        template_type="skill",
        hardware_platform="cuda",
        variables={"batch_size": 4}
    )
"""

import os
import re
import json
import uuid
import logging
import datetime
import inspect
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# Import template database
from template_database import TemplateDatabase, DEFAULT_DB_PATH

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class TemplateRenderer:
    """
    Renderer for templates stored in the template database.
    
    This class provides methods for rendering templates with variable substitution,
    template inheritance, and model-specific customizations.
    """
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH, verbose: bool = False):
        """
        Initialize the template renderer.
        
        Args:
            db_path: Path to the template database
            verbose: Enable verbose logging
        """
        self.db = TemplateDatabase(db_path, verbose)
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
            
    def _process_variable_transforms(self, content: str, variables: Dict[str, Any]) -> str:
        """
        Process variable transformations in template content.
        
        This handles expressions like ${variable.replace('-', '_')} by evaluating
        the Python expression with the variable value.
        
        Args:
            content: Template content with variable transforms
            variables: Dictionary of variable values
            
        Returns:
            Processed content with transformations applied
        """
        import re
        
        # Pattern to match variable transformations like ${variable.replace('-', '_')}
        pattern = r'\${([a-zA-Z0-9_]+)\.([^}]+)}'
        
        def replace_with_transform(match):
            var_name = match.group(1)
            transform = match.group(2)
            
            if var_name not in variables:
                logger.warning(f"Variable '{var_name}' not found in variables dictionary")
                return f"${{{var_name}.{transform}}}"
                
            var_value = variables[var_name]
            
            try:
                # Create a safe local environment with just the variable value
                local_env = {"value": var_value}
                # Convert the transform to apply to the value variable
                transform_code = f"value.{transform}"
                # Evaluate the transformation
                result = eval(transform_code, {"__builtins__": {}}, local_env)
                return str(result)
            except Exception as e:
                logger.warning(f"Error processing transformation '{transform}' for variable '{var_name}': {e}")
                return f"${{{var_name}.{transform}}}"
        
        # Replace all transformations
        processed_content = re.sub(pattern, replace_with_transform, content)
        return processed_content
            
    def render_template(self,
                        model_name: str,
                        template_type: str,
                        hardware_platform: Optional[str] = None,
                        variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Render a template for a specific model and hardware platform.
        
        Args:
            model_name: Name of the model
            template_type: Type of template (skill, test, benchmark, documentation)
            hardware_platform: Hardware platform (optional, defaults to "cpu")
            variables: Additional variables to use in template rendering (optional)
            
        Returns:
            Rendered template content
        """
        # Get model family
        model_family = self.db.get_model_family(model_name)
        if not model_family:
            raise ValueError(f"Could not determine model family for {model_name}")
            
        # Get template
        template = self.db.get_template(
            model_family=model_family,
            template_type=template_type,
            hardware_platform=hardware_platform
        )
        
        if not template:
            raise ValueError(f"No template found for {model_family} {template_type} on {hardware_platform}")
            
        # Set up basic variables
        base_variables = {
            "model_name": model_name,
            "model_family": model_family,
            "hardware_type": hardware_platform or "cpu",
            "test_id": str(uuid.uuid4()),
            "batch_size": 1,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add additional variables
        if variables:
            base_variables.update(variables)
            
        # Add derived variables with common transformations
        derived_variables = {
            # Model name transformations
            "model_name_safe": model_name.replace('-', '_').replace('/', '_'),
            "model_name_class": model_name.replace('-', '_').replace('/', '_').title(),
            "model_name_file": model_name.replace('/', '_'),
            
            # Model family transformations
            "model_family_display": model_family.replace('_', ' '),
            
            # Hardware transformations
            "hardware_name": hardware_platform or "cpu",
            
            # Documentation placeholders
            "test_results": "No test results available yet.",
            "benchmark_results": "No benchmark results available yet.",
            "limitations": f"This implementation may have limitations specific to {hardware_platform or 'cpu'} hardware. "
                          f"Please refer to hardware documentation for details."
        }
        base_variables.update(derived_variables)
            
        # Render the template
        rendered_content = self.db.render_template(
            template_id=template["template_id"],
            variables=base_variables,
            render_dependencies=True
        )
        
        # Process variable transformations
        rendered_content = self._process_variable_transforms(rendered_content, base_variables)
        
        # Add header comment with metadata
        header = f"""#!/usr/bin/env python3
# Generated by TemplateRenderer on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Model: {model_name}
# Template: {template["template_name"]} ({template["template_id"]})
# Hardware: {hardware_platform or "cpu"}
# Type: {template_type}

"""
        
        return header + rendered_content
        
    def render_component_set(self,
                            model_name: str,
                            hardware_platform: Optional[str] = None,
                            variables: Optional[Dict[str, Any]] = None,
                            output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Render a complete set of components (skill, test, benchmark, documentation) for a model.
        
        Args:
            model_name: Name of the model
            hardware_platform: Hardware platform (optional, defaults to "cpu")
            variables: Additional variables to use in template rendering (optional)
            output_dir: Directory to output the files (optional)
            
        Returns:
            Dictionary of rendered content by template type
        """
        # Set default hardware platform
        hardware_platform = hardware_platform or "cpu"
        
        # Create a dictionary to store rendered content
        rendered_content = {}
        
        # Set template types to render
        template_types = ["skill", "test", "benchmark", "documentation"]
        
        # Get model family
        model_family = self.db.get_model_family(model_name)
        if not model_family:
            raise ValueError(f"Could not determine model family for {model_name}")
            
        # Create base variables
        base_variables = {
            "model_name": model_name,
            "model_family": model_family,
            "hardware_type": hardware_platform,
            "test_id": str(uuid.uuid4()),
            "batch_size": 1,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add additional variables
        if variables:
            base_variables.update(variables)
            
        # Add model family-specific variables
        self._add_model_family_variables(model_family, base_variables)
        
        # Add hardware-specific variables
        self._add_hardware_specific_variables(hardware_platform, base_variables)
        
        # Add derived variables with common transformations
        derived_variables = {
            # Model name transformations
            "model_name_safe": model_name.replace('-', '_').replace('/', '_'),
            "model_name_class": model_name.replace('-', '_').replace('/', '_').title(),
            "model_name_file": model_name.replace('/', '_'),
            
            # Model family transformations
            "model_family_display": model_family.replace('_', ' '),
            
            # Hardware transformations
            "hardware_name": hardware_platform,
            
            # Documentation placeholders
            "test_results": "No test results available yet.",
            "benchmark_results": "No benchmark results available yet.",
            "limitations": f"This implementation may have limitations specific to {hardware_platform} hardware. "
                          f"Please refer to hardware documentation for details."
        }
        base_variables.update(derived_variables)
        
        # Render each template type
        for template_type in template_types:
            try:
                # Get template
                template = self.db.get_template(
                    model_family=model_family,
                    template_type=template_type,
                    hardware_platform=hardware_platform
                )
                
                if not template:
                    logger.warning(f"No {template_type} template found for {model_family} on {hardware_platform}")
                    continue
                    
                # Render template
                rendered = self.db.render_template(
                    template_id=template["template_id"],
                    variables=base_variables,
                    render_dependencies=True
                )
                
                # Process variable transformations
                rendered = self._process_variable_transforms(rendered, base_variables)
                
                # Add header
                header = f"""#!/usr/bin/env python3
# Generated by TemplateRenderer on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Model: {model_name}
# Template: {template["template_name"]} ({template["template_id"]})
# Hardware: {hardware_platform}
# Type: {template_type}

"""
                rendered = header + rendered
                
                # Store rendered content
                rendered_content[template_type] = rendered
                
                # Write to file if output directory specified
                if output_dir:
                    # Ensure output directory exists
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Determine output file name
                    if template_type == "skill":
                        filename = f"{model_name.replace('/', '_')}_{hardware_platform}_skill.py"
                    elif template_type == "test":
                        filename = f"test_{model_name.replace('/', '_')}_{hardware_platform}.py"
                    elif template_type == "benchmark":
                        filename = f"benchmark_{model_name.replace('/', '_')}_{hardware_platform}.py"
                    elif template_type == "documentation":
                        filename = f"{model_name.replace('/', '_')}_{hardware_platform}_docs.md"
                    else:
                        filename = f"{template_type}_{model_name.replace('/', '_')}_{hardware_platform}.py"
                        
                    # Write to file
                    file_path = os.path.join(output_dir, filename)
                    with open(file_path, 'w') as f:
                        f.write(rendered)
                        
                    logger.info(f"Wrote {template_type} template to {file_path}")
                    
            except Exception as e:
                logger.error(f"Error rendering {template_type} template for {model_name} on {hardware_platform}: {e}")
                
        return rendered_content
        
    def _add_model_family_variables(self, model_family: str, variables: Dict[str, Any]) -> None:
        """
        Add model family-specific variables to the variables dictionary.
        
        Args:
            model_family: Model family
            variables: Variables dictionary to update
        """
        # Text embedding models
        if model_family == "text_embedding":
            variables.update({
                "input_type": "text",
                "output_type": "embedding",
                "typical_sequence_length": 128,
                "typical_output_dims": 768,
                "common_use_case": "semantic search, clustering, classification"
            })
            
        # Text generation models
        elif model_family == "text_generation":
            variables.update({
                "input_type": "text",
                "output_type": "text",
                "typical_sequence_length": 1024,
                "typical_output_dims": None,
                "common_use_case": "question answering, completion, summarization"
            })
            
        # Vision models
        elif model_family == "vision":
            variables.update({
                "input_type": "image",
                "output_type": "embedding",
                "typical_sequence_length": None,
                "typical_output_dims": 768,
                "common_use_case": "image classification, feature extraction"
            })
            
        # Audio models
        elif model_family == "audio":
            variables.update({
                "input_type": "audio",
                "output_type": "text",
                "typical_sequence_length": None,
                "typical_output_dims": None,
                "common_use_case": "speech recognition, audio classification"
            })
            
        # Multimodal models
        elif model_family == "multimodal":
            variables.update({
                "input_type": "multiple",
                "output_type": "multiple",
                "typical_sequence_length": None,
                "typical_output_dims": None,
                "common_use_case": "image-text understanding, visual question answering"
            })
            
    def _add_hardware_specific_variables(self, hardware_platform: str, variables: Dict[str, Any]) -> None:
        """
        Add hardware-specific variables to the variables dictionary.
        
        Args:
            hardware_platform: Hardware platform
            variables: Variables dictionary to update
        """
        # CPU-specific variables
        if hardware_platform == "cpu":
            variables.update({
                "hardware_specific_optimizations": "- CPU threading optimizations\n- Cache-friendly operations\n- SSE/AVX instructions where applicable",
                "memory_management": "host_memory",
                "precision": "float32",
                "threading_model": "parallel",
                "initialization_code": "import torch\ndevice = 'cpu'"
            })
            
        # CUDA-specific variables
        elif hardware_platform == "cuda":
            variables.update({
                "hardware_specific_optimizations": "- CUDA kernel optimizations\n- Mixed precision inference\n- Memory optimization for GPU",
                "memory_management": "device_memory",
                "precision": "float16",
                "threading_model": "cuda_streams",
                "initialization_code": "import torch\ndevice = 'cuda' if torch.cuda.is_available() else 'cpu'"
            })
            
        # WebGPU-specific variables
        elif hardware_platform == "webgpu":
            variables.update({
                "hardware_specific_optimizations": "- WebGPU shader optimizations\n- Browser-specific optimizations\n- Memory management for browser environment",
                "memory_management": "device_memory",
                "precision": "float16",
                "threading_model": "browser_worker",
                "initialization_code": "from fixed_web_platform.webgpu_utils import get_device\ndevice = get_device()"
            })
            
        # Default variables for other platforms
        else:
            variables.update({
                "hardware_specific_optimizations": f"- Platform-specific optimizations for {hardware_platform}",
                "memory_management": "host_memory",
                "precision": "float32",
                "threading_model": "default",
                "initialization_code": f"# Initialize {hardware_platform} device\ndevice = '{hardware_platform}'"
            })
            
    def get_compatible_hardware_platforms(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get compatible hardware platforms for a given model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of compatible hardware platforms with compatibility level
        """
        # Get model family
        model_family = self.db.get_model_family(model_name)
        if not model_family:
            raise ValueError(f"Could not determine model family for {model_name}")
            
        # Get compatible hardware platforms
        return self.db.get_compatible_hardware_platforms(model_family)
        
    def initialize_database_with_defaults(self) -> None:
        """Initialize the template database with default templates."""
        from template_database import add_default_templates
        add_default_templates(self.db.db_path)
        logger.info(f"Initialized template database with default templates at {self.db.db_path}")
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Template Renderer")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH,
                       help="Path to the template database")
    parser.add_argument("--model", type=str, required=False,
                       help="Model name to render templates for")
    parser.add_argument("--hardware", type=str, default="cpu",
                       help="Hardware platform to render templates for")
    parser.add_argument("--output-dir", type=str, default="./generated",
                       help="Directory to output rendered templates")
    parser.add_argument("--template-type", type=str, choices=["skill", "test", "benchmark", "documentation"],
                       help="Specific template type to render")
    parser.add_argument("--list-compatible-hardware", action="store_true",
                       help="List compatible hardware platforms for the model")
    parser.add_argument("--initialize-db", action="store_true",
                       help="Initialize the template database with default templates")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    # Create renderer
    renderer = TemplateRenderer(db_path=args.db_path, verbose=args.verbose)
    
    # Initialize database if requested
    if args.initialize_db:
        renderer.initialize_database_with_defaults()
        print(f"Initialized template database at {args.db_path}")
        
    # List compatible hardware platforms if requested
    if args.list_compatible_hardware and args.model:
        try:
            platforms = renderer.get_compatible_hardware_platforms(args.model)
            print(f"Compatible hardware platforms for {args.model}:")
            for platform in platforms:
                print(f"- {platform['hardware_platform']}: {platform['compatibility_level']}")
                if platform['description']:
                    print(f"  {platform['description']}")
        except Exception as e:
            print(f"Error listing compatible hardware platforms: {e}")
            
    # Render template if model is specified
    if args.model:
        try:
            if args.template_type:
                # Render specific template type
                rendered = renderer.render_template(
                    model_name=args.model,
                    template_type=args.template_type,
                    hardware_platform=args.hardware
                )
                
                # Create output directory if it doesn't exist
                os.makedirs(args.output_dir, exist_ok=True)
                
                # Determine output file name
                if args.template_type == "skill":
                    filename = f"{args.model.replace('/', '_')}_{args.hardware}_skill.py"
                elif args.template_type == "test":
                    filename = f"test_{args.model.replace('/', '_')}_{args.hardware}.py"
                elif args.template_type == "benchmark":
                    filename = f"benchmark_{args.model.replace('/', '_')}_{args.hardware}.py"
                elif args.template_type == "documentation":
                    filename = f"{args.model.replace('/', '_')}_{args.hardware}_docs.md"
                else:
                    filename = f"{args.template_type}_{args.model.replace('/', '_')}_{args.hardware}.py"
                    
                # Write to file
                file_path = os.path.join(args.output_dir, filename)
                with open(file_path, 'w') as f:
                    f.write(rendered)
                    
                print(f"Rendered {args.template_type} template for {args.model} on {args.hardware} to {file_path}")
                
            else:
                # Render all template types
                rendered_content = renderer.render_component_set(
                    model_name=args.model,
                    hardware_platform=args.hardware,
                    output_dir=args.output_dir
                )
                
                print(f"Rendered templates for {args.model} on {args.hardware} to {args.output_dir}")
                for template_type in rendered_content:
                    print(f"- {template_type}")
                    
        except Exception as e:
            print(f"Error rendering templates: {e}")
    elif not args.initialize_db and not args.list_compatible_hardware:
        parser.print_help()