#!/usr/bin/env python3
"""
Documentation Template Fixer

This script fixes issues with the documentation templates by modifying how the
ModelDocGenerator and TemplateRenderer handle variable substitution. It ensures
all required variables are included in the variables dictionary and handles
variable transformation properly.
"""

import os
import sys
import re
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DocTemplateFixer")

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import the modules we need to fix
from model_documentation_generator import ModelDocGenerator
from template_renderer import TemplateRenderer

def monkey_patch_model_doc_generator():
    """Monkey patch the ModelDocGenerator to fix variables."""
    
    # Store original method for reference
    original_generate_documentation = ModelDocGenerator.generate_documentation
    
    def patched_generate_documentation(self) -> str:
        """Patched version of generate_documentation that ensures all variables are defined."""
        logger.info(f"Using patched generate_documentation for {self.model_name} on {self.hardware}")
        
        # Create output directory if it doesn't exist
        model_doc_dir = os.path.join(self.output_dir, self.model_name.replace('/', '_'))
        os.makedirs(model_doc_dir, exist_ok=True)
        
        # Generate documentation file path
        doc_path = os.path.join(model_doc_dir, f"{self.model_name.replace('/', '_')}_{self.hardware}_docs.md")
        
        # Use template system if available
        if self.use_templates:
            try:
                # Determine model family from model name
                model_family = self.template_db.get_model_family(self.model_name)
                if not model_family:
                    logger.warning(f"Could not determine model family for {self.model_name}, falling back to manual generation")
                    return self._generate_documentation_manual(doc_path)
                
                # Extract information from files
                skill_docstrings = self.extract_docstrings(self.skill_path)
                test_docstrings = self.extract_docstrings(self.test_path)
                benchmark_docstrings = self.extract_docstrings(self.benchmark_path)
                
                skill_snippets = self.extract_key_code_snippets(self.skill_path)
                test_snippets = self.extract_key_code_snippets(self.test_path)
                benchmark_snippets = self.extract_key_code_snippets(self.benchmark_path)
                
                expected_results = self.load_expected_results()
                
                # Extract the model class name from the skill file
                model_class_name = self._extract_class_name(self.skill_path)
                
                # Extract API details for better documentation (safely)
                try:
                    api_details = self._extract_api_details(self.skill_path)
                except Exception as e:
                    logger.warning(f"Error extracting API details: {e}")
                    api_details = {"class_name": model_class_name, "methods": {}}
                
                # Extract model architecture details
                model_architecture = self._get_model_architecture_description(model_family)
                
                # Extract hardware capability details
                hardware_capabilities = self._get_hardware_capability_details(self.hardware)
                
                # Get model-specific features and use cases
                model_specific_features = self._get_model_specific_features(model_family)
                model_common_use_cases = self._get_model_common_use_cases(model_family)
                
                # Format features and use cases for better markdown rendering
                formatted_features = "\n".join([f"- {feature}" for feature in model_specific_features])
                formatted_use_cases = "\n".join([f"- {use_case}" for use_case in model_common_use_cases])
                
                # Get hardware-specific notes
                hardware_specific_notes = self._get_hardware_specific_notes(self.hardware)
                
                # Create variables for template rendering
                variables = {
                    "model_name": self.model_name,
                    "model_family": model_family,
                    "hardware_type": self.hardware,
                    "timestamp": os.environ.get('DATE', '2025-03-16'),
                    "user": os.environ.get('USER', 'auto-generated'),
                    "test_id": "test-id-12345",
                    "model_class_name": model_class_name,
                    
                    # Code snippets
                    "class_definition": skill_snippets.get("class_definition", "# No class definition found"),
                    "setup_method": skill_snippets.get("setup_method", "# No setup method found"),
                    "run_method": skill_snippets.get("run_method", "# No run method found"),
                    "test_class": test_snippets.get("test_class", "# No test class found"),
                    "benchmark_function": benchmark_snippets.get("benchmark_function", "# No benchmark function found"),
                    "main_execution": benchmark_snippets.get("main_execution", "# No main execution block found"),
                    
                    # Expected results
                    "expected_results_json": "{}",
                    "expected_results_available": False,
                    
                    # Hardware-specific information
                    "hardware_specific_notes": hardware_specific_notes,
                    "hardware_capabilities": hardware_capabilities,
                    
                    # Model architecture details
                    "model_architecture": model_architecture,
                    
                    # Module docstring
                    "module_docstring": skill_docstrings.get("module", ""),
                    "class_docstring": skill_docstrings.get(model_class_name, ""),
                    
                    # Model type-specific details
                    "model_specific_features": model_specific_features,
                    "formatted_model_specific_features": formatted_features,
                    "model_common_use_cases": model_common_use_cases,
                    "formatted_model_common_use_cases": formatted_use_cases,
                    
                    # Usage example
                    "usage_example": self._generate_usage_example(model_class_name, model_family),
                    
                    # Prevent common undefined variable errors
                    "features": "Features not available",
                    "formatted_api_docs": "API documentation not available",
                    "test_results": "Test results not available",
                    "benchmark_results": "Benchmark results not available",
                    "limitations": f"This implementation may have limitations specific to {self.hardware} hardware."
                }
                
                # Add test method snippets
                test_methods = [k for k in test_snippets.keys() if k.startswith("test_method_")]
                test_methods_content = ""
                for method_key in test_methods:
                    test_methods_content += f"```python\n{test_snippets[method_key]}\n```\n\n"
                variables["test_methods_content"] = test_methods_content
                
                # Add expected results if available
                if expected_results:
                    import json
                    variables["expected_results_json"] = json.dumps(expected_results, indent=2)
                    variables["expected_results_available"] = True
                
                # Generate formatted API documentation
                try:
                    api_docs = self._format_api_documentation(api_details)
                    variables["formatted_api_docs"] = api_docs
                except Exception as e:
                    logger.warning(f"Error formatting API documentation: {e}")
                    variables["formatted_api_docs"] = "API documentation could not be generated."
                
                # Render documentation template
                try:
                    logger.info("Rendering documentation template with template system")
                    rendered_doc = self.template_renderer.render_template(
                        model_name=self.model_name,
                        template_type="documentation",
                        hardware_platform=self.hardware,
                        variables=variables
                    )
                    
                    # Write to file
                    with open(doc_path, 'w') as f:
                        f.write(rendered_doc)
                        
                    logger.info(f"Documentation generated with template: {doc_path}")
                    return doc_path
                    
                except Exception as e:
                    logger.warning(f"Error rendering documentation template: {e}")
                    logger.info("Falling back to manual documentation generation")
                    return self._generate_documentation_manual(doc_path)
                    
            except Exception as e:
                logger.warning(f"Error using template system for documentation: {e}")
                logger.info("Falling back to manual documentation generation")
                return self._generate_documentation_manual(doc_path)
        else:
            # No template system available, use manual generation
            return self._generate_documentation_manual(doc_path)
    
    # Replace the original method with our patched version
    ModelDocGenerator.generate_documentation = patched_generate_documentation
    logger.info("ModelDocGenerator.generate_documentation has been patched")

def monkey_patch_template_renderer():
    """Monkey patch the TemplateRenderer to handle variable substitution better."""
    
    # Store original method for reference
    original_process_variable_transforms = TemplateRenderer._process_variable_transforms
    original_render_template = TemplateRenderer.render_template
    
    def patched_process_variable_transforms(self, content: str, variables: Dict[str, Any]) -> str:
        """Patched version of _process_variable_transforms to handle missing variables better."""
        import re
        
        # First apply the normal variable substitution for ${var} pattern
        pattern = r'\${([a-zA-Z0-9_]+)}'
        
        def replace_var(match):
            var_name = match.group(1)
            
            if var_name not in variables:
                logger.warning(f"Variable '{var_name}' not found in variables dictionary, using placeholder")
                return f"PLACEHOLDER_{var_name}"
                
            var_value = variables[var_name]
            return str(var_value)
        
        # Replace simple variables first
        processed_content = re.sub(pattern, replace_var, content)
        
        # Pattern to match variable transformations like ${variable.replace('-', '_')}
        transform_pattern = r'\${([a-zA-Z0-9_]+)\.([^}]+)}'
        
        def replace_with_transform(match):
            var_name = match.group(1)
            transform = match.group(2)
            
            if var_name not in variables:
                logger.warning(f"Variable '{var_name}' not found in variables dictionary for transform")
                return f"PLACEHOLDER_{var_name}_{transform}"
                
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
                return f"PLACEHOLDER_{var_name}_{transform}"
        
        # Replace all transformations
        processed_content = re.sub(transform_pattern, replace_with_transform, processed_content)
        return processed_content
        
    def patched_render_template(self, model_name: str, template_type: str, 
                              hardware_platform: str = None, variables: Dict[str, Any] = None) -> str:
        """Patched version of render_template that ensures all required variables are present."""
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
            "test_id": "test-id-12345",
            "batch_size": 1,
            "timestamp": "2025-03-16"
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
            
            # Documentation variables
            "test_results": "No test results available yet.",
            "benchmark_results": "No benchmark results available yet.",
            "limitations": f"This implementation may have limitations specific to {hardware_platform or 'cpu'} hardware.",
            "features": "Default features",
            "formatted_api_docs": "API documentation not available",
            "model_architecture": "Model architecture details not available",
            "hardware_specific_notes": "Hardware-specific notes not available",
            "formatted_model_specific_features": "Model-specific features not available",
            "formatted_model_common_use_cases": "Common use cases not available",
            "usage_example": "Usage example not available",
            "class_definition": "Class definition not available"
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
# Generated by TemplateRenderer on 2025-03-16
# Model: {model_name}
# Template: {template["template_name"]} ({template["template_id"]})
# Hardware: {hardware_platform or "cpu"}
# Type: {template_type}

"""
        
        return header + rendered_content
        
    # Replace the original methods with our patched versions
    TemplateRenderer._process_variable_transforms = patched_process_variable_transforms
    TemplateRenderer.render_template = patched_render_template
    logger.info("TemplateRenderer methods have been patched")

def main():
    """Main function to patch the classes and run a test."""
    
    # Patch the classes
    monkey_patch_model_doc_generator()
    monkey_patch_template_renderer()
    
    # Print success message
    logger.info("Documentation template system has been patched successfully")
    logger.info("Run test_enhanced_documentation.py again to verify the fix")
    
if __name__ == "__main__":
    main()