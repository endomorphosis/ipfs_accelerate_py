#!/usr/bin/env python
"""
Template Inheritance System Demonstration

This script demonstrates the template inheritance system with model-specific template generation.
It showcases the key features of template inheritance, hardware-aware template selection, and
customized template rendering for different model families.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional

# Add parent directory for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates.model_template_registry import get_template_registry, select_template, render_template

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_template_inheritance(model_family: str, model_name: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Demonstrate template inheritance with a specific model family and model.
    
    Args:
        model_family: Model family (e.g., 'embedding', 'text_generation')
        model_name: Specific model name (e.g., 'bert-base-uncased')
        output_file: Optional file to save the generated code
        
    Returns:
        Dictionary with demonstration results
    """
    logger.info(f"Demonstrating template inheritance for model family: {model_family}, model: {model_name}")
    
    # Get template registry
    registry = get_template_registry()
    
    # 1. Select appropriate template based on model family
    template_name = select_template(model_family)
    logger.info(f"Selected template: {template_name}")
    
    # 2. Get template inheritance chain
    chain = registry.resolve_template_chain(template_name)
    logger.info(f"Template inheritance chain: {chain}")
    
    # 3. Get merged template
    merged_template = registry.get_merged_template(template_name)
    
    # 4. Create context for template rendering
    context = {
        "model_name": model_name,
        "model_type": model_family,
        "model_description": f"{model_name} model for {model_family}",
        "modality": "text" if model_family in ["text_generation", "embedding"] else 
                   "image" if model_family == "vision" else 
                   "audio" if model_family == "audio" else "multimodal",
        "supports_quantization": "True",
        "requires_gpu": "False" if model_family == "embedding" else "True"
    }
    
    # 5. Render the template
    implementation = registry.render_template(template_name, context)
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(implementation)
        logger.info(f"Generated implementation saved to {output_file}")
    
    # Return results
    return {
        "model_family": model_family,
        "model_name": model_name,
        "template_name": template_name,
        "inheritance_chain": chain,
        "section_count": len(merged_template.get("sections", {})),
        "sections": list(merged_template.get("sections", {}).keys()),
        "implementation_length": len(implementation),
        "output_file": output_file
    }

def demonstrate_hardware_aware_selection(model_family: str, hardware_info: Dict[str, bool]) -> Dict[str, Any]:
    """
    Demonstrate hardware-aware template selection.
    
    Args:
        model_family: Model family (e.g., 'embedding', 'text_generation')
        hardware_info: Hardware information (e.g., {"cuda": True, "mps": False})
        
    Returns:
        Dictionary with demonstration results
    """
    logger.info(f"Demonstrating hardware-aware template selection for {model_family}")
    logger.info(f"Hardware information: {hardware_info}")
    
    # Get template registry
    registry = get_template_registry()
    
    # Select template with hardware information
    template_name = select_template(model_family, hardware_info)
    logger.info(f"Selected template with hardware awareness: {template_name}")
    
    # Get merged template
    merged_template = registry.get_merged_template(template_name)
    
    # Get supported hardware
    supported_hardware = merged_template.get("supports_hardware", [])
    logger.info(f"Template supports hardware: {supported_hardware}")
    
    # Return results
    return {
        "model_family": model_family,
        "hardware_info": hardware_info,
        "template_name": template_name,
        "supported_hardware": supported_hardware
    }

def demonstrate_multiple_model_families(output_dir: str) -> Dict[str, Any]:
    """
    Demonstrate template inheritance across multiple model families.
    
    Args:
        output_dir: Directory to save generated code files
        
    Returns:
        Dictionary with demonstration results
    """
    logger.info("Demonstrating template inheritance across multiple model families")
    
    # Model families to demonstrate
    model_families = {
        "embedding": "bert-base-uncased",
        "text_generation": "t5-small",
        "vision": "vit-base-patch16-224",
        "audio": "whisper-small",
        "multimodal": "clip-vit-base-patch32"
    }
    
    results = {}
    
    for family, model in model_families.items():
        # Create output file path
        output_file = os.path.join(output_dir, f"generated_{family}_model.py")
        
        # Demonstrate template inheritance
        result = demonstrate_template_inheritance(family, model, output_file)
        results[family] = result
    
    # Log summary
    logger.info("Demonstration summary:")
    for family, result in results.items():
        logger.info(f"  {family}: {result['template_name']} -> {result['output_file']}")
    
    return results

def generate_markdown_report(results: Dict[str, Any], output_file: str) -> None:
    """
    Generate a Markdown report from demonstration results.
    
    Args:
        results: Results from demonstration functions
        output_file: File to save the report
    """
    logger.info(f"Generating Markdown report: {output_file}")
    
    # Create report content
    report = f"""# Template Inheritance System Demonstration Report

## Overview

This report summarizes the demonstration of the Template Inheritance System.
It shows how templates are selected, inheritance chains are resolved, and
implementations are generated for different model families.

## Model Family Templates

| Model Family | Selected Template | Inheritance Chain | Section Count |
|--------------|------------------|-------------------|---------------|
"""
    
    # Add model family results
    for family, result in results.items():
        chain = " → ".join(result["inheritance_chain"])
        report += f"| {family} | {result['template_name']} | {chain} | {result['section_count']} |\n"
    
    # Add hardware-aware selection section
    report += """
## Hardware-Aware Template Selection

The system can select templates based on available hardware, ensuring optimal
implementation for different hardware configurations.

| Model Family | Hardware Configuration | Selected Template | Supported Hardware |
|--------------|------------------------|------------------|-------------------|
"""
    
    # Example hardware configurations
    hardware_configs = {
        "embedding": {"cuda": True, "mps": False},
        "text_generation": {"cuda": False, "mps": True},
        "vision": {"cuda": True, "rocm": True},
        "audio": {"cuda": False, "cpu": True}
    }
    
    # Add hardware selection results
    for family, hw_info in hardware_configs.items():
        hw_result = demonstrate_hardware_aware_selection(family, hw_info)
        hw_config = ", ".join([k for k, v in hw_info.items() if v])
        supported = ", ".join(hw_result["supported_hardware"])
        report += f"| {family} | {hw_config} | {hw_result['template_name']} | {supported} |\n"
    
    # Add section descriptions
    report += """
## Template Sections

Each template is divided into sections that can be inherited or overridden.
Below is a sample of sections from the embedding model template:

| Section | Purpose | Inheritance |
|---------|---------|------------|
| imports | Import statements for required modules | Usually inherited from base |
| class_definition | Class declaration and metadata | Usually specialized by model family |
| init | Initialization method | Often specialized by model family |
| methods | Model-specific methods | Highly specialized by model type |
| utility_methods | Helper functions | Usually inherited from base |
| hardware_support | Hardware-specific code | Specialized based on hardware support |
"""
    
    # Add conclusion
    report += """
## Conclusion

The Template Inheritance System provides a flexible and extensible way to create
model implementations with shared core functionality while allowing for
hardware-aware specialization. It enables:

1. Code reuse through inheritance
2. Hardware-specific optimizations
3. Model family specialization
4. Consistent implementation patterns
5. Simple extension to new model types

For more information, see the [Template Inheritance Guide](../TEMPLATE_INHERITANCE_GUIDE.md).
"""
    
    # Save report
    with open(output_file, "w") as f:
        f.write(report)
    
    logger.info(f"Report generated: {output_file}")

def main():
    """Main function for the demonstration script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demonstrate Template Inheritance System")
    parser.add_argument("--family", type=str, choices=["embedding", "text_generation", "vision", "audio", "multimodal"],
                       help="Specific model family to demonstrate")
    parser.add_argument("--model", type=str, default=None, help="Specific model to use")
    parser.add_argument("--output-dir", type=str, default="./generated", help="Output directory for generated code")
    parser.add_argument("--report", type=str, default="template_demo_report.md", help="Output file for Markdown report")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.family and args.model:
        # Demonstrate single model
        output_file = os.path.join(args.output_dir, f"generated_{args.family}_model.py")
        result = demonstrate_template_inheritance(args.family, args.model, output_file)
        
        # Generate report for single model
        results = {args.family: result}
        generate_markdown_report(results, args.report)
    else:
        # Demonstrate all model families
        results = demonstrate_multiple_model_families(args.output_dir)
        generate_markdown_report(results, args.report)
    
    logger.info("Demonstration completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())