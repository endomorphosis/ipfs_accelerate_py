#!/usr/bin/env python
"""
Template Verifier - Validate and test model templates.
Ensures templates are properly structured and compatible with model registry.
"""

import os
import sys
import json
import logging
import argparse
import importlib.util
import inspect
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

# Add parent directory for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates.model_template_registry import get_template_registry

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemplateVerifier:
    """
    Validates and tests model templates to ensure they are compliant with registry requirements.
    
    Attributes:
        template_dir: Directory containing templates
        registry: Template registry instance
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize template verifier.
        
        Args:
            template_dir: Directory containing templates
        """
        self.template_dir = template_dir or os.path.join(os.path.dirname(__file__))
        self.registry = get_template_registry()
    
    def verify_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Verify all templates in the template directory.
        
        Returns:
            Dictionary with verification results for each template
        """
        results = {}
        
        # Get all template names
        template_names = self.registry.get_all_template_names()
        
        for template_name in template_names:
            try:
                results[template_name] = self.verify_template(template_name)
            except Exception as e:
                logger.error(f"Error verifying template {template_name}: {str(e)}")
                results[template_name] = {
                    "status": "error",
                    "errors": [f"Exception during verification: {str(e)}"]
                }
        
        return results
    
    def verify_template(self, template_name: str) -> Dict[str, Any]:
        """
        Verify a single template.
        
        Args:
            template_name: The template name
            
        Returns:
            Dictionary with verification results
        """
        logger.info(f"Verifying template: {template_name}")
        
        # Get template info
        template_info = self.registry.get_template_info(template_name)
        if not template_info:
            return {
                "status": "error",
                "errors": [f"Template {template_name} not found in registry"]
            }
        
        # Check for required template metadata
        required_metadata = ["name", "description", "version"]
        errors = []
        warnings = []
        
        for field in required_metadata:
            if field not in template_info or not template_info[field]:
                errors.append(f"Missing required metadata: {field}")
        
        # Check for inheritance consistency
        parent = template_info.get("inherits_from")
        if parent:
            # Verify parent exists
            if parent not in self.registry.templates:
                errors.append(f"Parent template {parent} not found")
            else:
                # Get inheritance chain
                chain = self.registry.resolve_template_chain(template_name)
                if not chain:
                    errors.append(f"Failed to resolve inheritance chain for {template_name}")
                elif chain[0] != "hf_template.py":
                    warnings.append(f"Inheritance chain does not start with base template: {chain}")
        
        # Check for required sections
        required_sections = ["imports", "class_definition", "init", "methods"]
        merged_template = self.registry.get_merged_template(template_name)
        
        for section in required_sections:
            if section not in merged_template.get("sections", {}):
                warnings.append(f"Missing section after inheritance: {section}")
        
        # Render test with a mock context
        mock_context = {
            "model_name": "test_model",
            "model_type": "test",
            "model_description": "Test model description",
            "modality": "text",
            "supports_quantization": "False",
            "requires_gpu": "False"
        }
        
        try:
            rendered = self.registry.render_template(template_name, mock_context)
            
            # Check if rendering was successful
            if not rendered:
                errors.append("Template rendering failed with empty result")
            elif "{{" in rendered or "}}" in rendered:
                warnings.append("Template contains unresolved variables after rendering")
                
        except Exception as e:
            errors.append(f"Template rendering failed: {str(e)}")
        
        # Determine verification status
        status = "success"
        if errors:
            status = "error"
        elif warnings:
            status = "warning"
        
        return {
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "template_info": template_info,
            "inheritance_chain": merged_template.get("inheritance_chain", []),
            "sections": list(merged_template.get("sections", {}).keys())
        }
    
    def test_template_inheritance(self, template_name: str) -> Dict[str, Any]:
        """
        Test template inheritance resolution.
        
        Args:
            template_name: The template name
            
        Returns:
            Dictionary with inheritance test results
        """
        logger.info(f"Testing inheritance for template: {template_name}")
        
        # Get template chain
        chain = self.registry.resolve_template_chain(template_name)
        
        if not chain:
            return {
                "status": "error",
                "errors": [f"Failed to resolve inheritance chain for {template_name}"]
            }
        
        # Get merged template
        merged = self.registry.get_merged_template(template_name)
        
        # Check if each section in the merged template is from the expected template
        section_sources = {}
        
        for template in chain:
            template_info = self.registry.templates.get(template, {})
            for section_name, section_content in template_info.get("sections", {}).items():
                # If this section hasn't been seen yet or is overridden by this template
                if section_name not in section_sources or section_content == merged["sections"].get(section_name):
                    section_sources[section_name] = template
        
        return {
            "status": "success",
            "chain": chain,
            "section_sources": section_sources,
            "section_count": len(merged.get("sections", {}))
        }
    
    def generate_test_implementation(self, template_name: str, model_family: str) -> Dict[str, Any]:
        """
        Generate a test implementation from a template.
        
        Args:
            template_name: The template name
            model_family: The model family for template context
            
        Returns:
            Dictionary with generation results
        """
        logger.info(f"Generating test implementation from template {template_name} for {model_family}")
        
        # Create context for the template
        context = {
            "model_name": f"test_{model_family}",
            "model_type": model_family,
            "model_description": f"Test model for {model_family}",
            "modality": "text" if model_family == "text_generation" else "embeddings" if model_family == "embedding" else "multimodal",
            "supports_quantization": "True",
            "requires_gpu": "False"
        }
        
        try:
            # Render the template
            implementation = self.registry.render_template(template_name, context)
            
            # Check if rendering was successful
            if not implementation:
                return {
                    "status": "error",
                    "errors": ["Template rendering failed with empty result"]
                }
            
            # Check for syntax errors
            try:
                compile(implementation, "<string>", "exec")
                syntax_valid = True
            except SyntaxError as e:
                syntax_valid = False
                error_message = f"Syntax error in generated code: {str(e)}"
            
            return {
                "status": "success" if syntax_valid else "error",
                "errors": [] if syntax_valid else [error_message],
                "implementation": implementation,
                "syntax_valid": syntax_valid,
                "template_name": template_name,
                "model_family": model_family
            }
            
        except Exception as e:
            return {
                "status": "error",
                "errors": [f"Template rendering failed: {str(e)}"],
                "template_name": template_name,
                "model_family": model_family
            }
    
    def validate_template_compatibility(self, template_name: str, hardware_types: List[str]) -> Dict[str, Any]:
        """
        Validate template hardware compatibility.
        
        Args:
            template_name: The template name
            hardware_types: List of hardware types to check
            
        Returns:
            Dictionary with compatibility results
        """
        logger.info(f"Validating hardware compatibility for template: {template_name}")
        
        # Get template info
        template_info = self.registry.get_template_info(template_name)
        if not template_info:
            return {
                "status": "error",
                "errors": [f"Template {template_name} not found in registry"]
            }
        
        # Get merged template
        merged = self.registry.get_merged_template(template_name)
        
        # Check hardware compatibility
        supported_hardware = merged.get("supports_hardware", [])
        compatible = {}
        
        for hw_type in hardware_types:
            compatible[hw_type] = hw_type in supported_hardware
        
        # Check for hardware-specific sections
        hardware_sections = {}
        
        for hw_type in hardware_types:
            hardware_sections[hw_type] = []
            
            # Look for hardware-specific sections or code
            for section_name, section_content in merged.get("sections", {}).items():
                if re.search(rf'\b{hw_type}\b', section_content, re.IGNORECASE):
                    hardware_sections[hw_type].append(section_name)
        
        return {
            "status": "success",
            "hardware_compatibility": compatible,
            "hardware_sections": hardware_sections,
            "template_name": template_name
        }
    
    def generate_compatibility_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive compatibility report for all templates.
        
        Returns:
            Dictionary with compatibility report
        """
        logger.info("Generating template compatibility report")
        
        report = {
            "templates": {},
            "model_families": {},
            "hardware_compatibility": {},
            "inheritance": {},
            "summary": {}
        }
        
        # Get all template names
        template_names = self.registry.get_all_template_names()
        
        # Hardware types to check
        hardware_types = ["cpu", "cuda", "rocm", "mps", "openvino"]
        
        # Model families to check
        model_families = ["embedding", "text_generation", "vision", "audio", "multimodal"]
        
        # Verify all templates
        for template_name in template_names:
            # Verify template
            verify_result = self.verify_template(template_name)
            report["templates"][template_name] = verify_result
            
            if verify_result["status"] != "error":
                # Test inheritance
                inheritance_result = self.test_template_inheritance(template_name)
                report["inheritance"][template_name] = inheritance_result
                
                # Validate hardware compatibility
                compatibility_result = self.validate_template_compatibility(template_name, hardware_types)
                report["hardware_compatibility"][template_name] = compatibility_result
        
        # Check template selection for each model family
        for model_family in model_families:
            # Select template for each family
            selected_template = self.registry.select_template_for_model(model_family)
            
            # Generate test implementation
            generation_result = self.generate_test_implementation(selected_template, model_family)
            
            report["model_families"][model_family] = {
                "selected_template": selected_template,
                "generation_result": generation_result
            }
        
        # Generate summary
        summary = {
            "template_count": len(template_names),
            "error_count": sum(1 for t in report["templates"].values() if t["status"] == "error"),
            "warning_count": sum(1 for t in report["templates"].values() if t["status"] == "warning"),
            "success_count": sum(1 for t in report["templates"].values() if t["status"] == "success"),
            "model_family_coverage": sum(1 for f in report["model_families"].values() if f["generation_result"]["status"] == "success") / len(model_families) if model_families else 0
        }
        
        report["summary"] = summary
        
        return report

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Template Verifier")
    parser.add_argument("--template", type=str, help="Specific template to verify")
    parser.add_argument("--all", action="store_true", help="Verify all templates")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive report")
    parser.add_argument("--test-family", type=str, choices=["embedding", "text_generation", "vision", "audio", "multimodal"], 
                        help="Test template with specific model family")
    parser.add_argument("--output", type=str, help="Output file for report or generated code")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create verifier
    verifier = TemplateVerifier()
    
    # Process command
    if args.template:
        # Verify single template
        result = verifier.verify_template(args.template)
        print(f"Template verification result for {args.template}:")
        print(f"Status: {result['status']}")
        
        if result["errors"]:
            print("Errors:")
            for error in result["errors"]:
                print(f"- {error}")
        
        if result["warnings"]:
            print("Warnings:")
            for warning in result["warnings"]:
                print(f"- {warning}")
        
        # Test with model family
        if args.test_family:
            generation_result = verifier.generate_test_implementation(args.template, args.test_family)
            
            if generation_result["status"] == "success":
                print(f"Successfully generated implementation for {args.test_family}")
                
                # Output generated code
                if args.output:
                    with open(args.output, "w") as f:
                        f.write(generation_result["implementation"])
                    print(f"Generated code written to {args.output}")
                else:
                    print("\nGenerated code:")
                    print("---------------")
                    print(generation_result["implementation"])
            else:
                print(f"Failed to generate implementation: {generation_result['errors']}")
    
    elif args.all:
        # Verify all templates
        results = verifier.verify_all_templates()
        
        print("Template verification results:")
        for template_name, result in results.items():
            print(f"{template_name}: {result['status']}")
            
            if result["errors"]:
                print("  Errors:")
                for error in result["errors"]:
                    print(f"  - {error}")
            
            if result["warnings"]:
                print("  Warnings:")
                for warning in result["warnings"]:
                    print(f"  - {warning}")
        
        # Output report
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Report written to {args.output}")
    
    elif args.report:
        # Generate comprehensive report
        report = verifier.generate_compatibility_report()
        
        print("Template Compatibility Report Summary:")
        print(f"Total templates: {report['summary']['template_count']}")
        print(f"Success: {report['summary']['success_count']}")
        print(f"Warnings: {report['summary']['warning_count']}")
        print(f"Errors: {report['summary']['error_count']}")
        print(f"Model family coverage: {report['summary']['model_family_coverage'] * 100:.1f}%")
        
        # Output report
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Comprehensive report written to {args.output}")
    else:
        # No command specified, show help
        parser.print_help()

if __name__ == "__main__":
    main()