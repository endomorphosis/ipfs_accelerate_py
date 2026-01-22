#!/usr/bin/env python3
"""
Test script for the multimodal pipeline template.

This script tests the multimodal pipeline template by generating a reference
implementation for a multimodal model (FLAVA) and verifying that it contains
the expected pipeline-specific code.
"""

import os
import sys
import argparse
import importlib.util
from typing import Dict, Any, List, Tuple

# Add parent directory to path to import from templates
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from templates.multimodal_pipeline import MultimodalPipelineTemplate
from templates.template_composer import TemplateComposer
from templates.base_architecture import BaseArchitectureTemplate
from templates.base_hardware import BaseHardwareTemplate
from templates.cpu_hardware import CPUHardwareTemplate
from generators.architecture_detector import get_architecture_type


def import_module_from_path(path: str, module_name: str = None):
    """Import a module from a file path."""
    if module_name is None:
        module_name = os.path.basename(path).replace(".py", "")
    
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyMultimodalTemplate(BaseArchitectureTemplate):
    """Dummy multimodal architecture template for testing."""
    
    def __init__(self):
        """Initialize the dummy multimodal template."""
        super().__init__()
        self.architecture_type = "multimodal"
        self.model_type = "flava"
        self.supported_task_types = [
            "multimodal_classification",
            "multimodal_generation",
            "multimodal_question_answering",
            "multimodal_retrieval"
        ]
        self.default_task_type = "multimodal_classification"
        self.hidden_size = 768
        self.model_description = "A multimodal model that processes images, text, and audio."
    
    def get_model_class(self, task_type: str) -> str:
        """Get the model class for a given task type."""
        return "self.transformers.AutoModel"
    
    def get_processor_class(self, task_type: str) -> str:
        """Get the processor class for a given task type."""
        return "self.transformers.AutoProcessor"
    
    def get_input_processing_code(self, task_type: str) -> str:
        """Get input processing code."""
        return """
        # Process inputs for multimodal model
        inputs = tokenizer(
            text=text_input if text_input is not None else None,
            images=image if image is not None else None,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        """
    
    def get_output_processing_code(self, task_type: str) -> str:
        """Get output processing code."""
        if task_type == "multimodal_classification":
            return """
            # Process outputs for multimodal classification
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                probabilities = self.torch.nn.functional.softmax(logits, dim=-1)
                predictions = probabilities[0].cpu().tolist()
            elif hasattr(outputs, "image_embeds") and hasattr(outputs, "text_embeds"):
                # Calculate similarity for CLIP-like models
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize embeddings
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)
                predictions = similarity[0].cpu().tolist()
            else:
                predictions = None
            """
        elif task_type == "multimodal_generation":
            return """
            # Process outputs for multimodal generation
            if hasattr(model, "generate"):
                # Add extra args for generation
                generate_args = {**inputs, **generation_params}
                output_ids = model.generate(**generate_args)
                
                # Decode the generated output
                generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            else:
                generated_text = ["Mock multimodal generation output"]
            """
        elif task_type == "multimodal_question_answering":
            return """
            # Process outputs for multimodal question answering
            if hasattr(model, "generate"):
                # Generate answer
                output_ids = model.generate(**inputs)
                
                # Decode the generated answer
                answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            else:
                answers = ["Mock multimodal QA answer"]
            """
        else:
            return """
            # Default output processing
            if hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
            else:
                embeddings = None
            """
    
    def get_model_config(self, model_name: str) -> str:
        """Get model configuration code."""
        return """
    def get_model_config(self):
        \"\"\"Get model configuration for multimodal model.\"\"\"
        return {
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "model_type": "flava"
        }
        """
    
    def get_mock_output_code(self) -> str:
        """Get mock output code."""
        return """
                # Create mock output tensors (flava-like structure)
                device = self.torch.device(device if "cuda" in device else "cpu")
                batch_size = kwargs.get("input_ids", kwargs.get("pixel_values", self.torch.ones((1, 3, 224, 224), device=device))).shape[0]
                hidden_size = 768
                
                # Create multimodal output structure
                mock_outputs = type('MultimodalOutput', (), {})()
                mock_outputs.text_embeds = self.torch.randn(batch_size, hidden_size, device=device)
                mock_outputs.image_embeds = self.torch.randn(batch_size, hidden_size, device=device)
                mock_outputs.multimodal_embeds = self.torch.randn(batch_size, hidden_size, device=device)
                mock_outputs.logits = self.torch.randn(batch_size, 100, device=device)  # 100 classes
                
                return mock_outputs
        """
    
    def get_mock_processor_code(self) -> str:
        """Get mock processor code."""
        return """
                # Dummy processor function to mimic multimodal processor
                def mock_tokenize(text=None, images=None, audio=None, return_tensors=None, padding=None, truncation=None):
                    import torch
                    
                    # Create dummy inputs for different modalities
                    inputs = {}
                    
                    if text is not None:
                        inputs["input_ids"] = torch.randint(0, 30000, (1, 50))
                        inputs["attention_mask"] = torch.ones((1, 50))
                    
                    if images is not None:
                        inputs["pixel_values"] = torch.rand((1, 3, 224, 224))
                    
                    if audio is not None:
                        inputs["audio_values"] = torch.rand((1, 16000))
                    
                    return inputs
        """


def generate_test_implementation(output_dir: str = "generated_test_models") -> Tuple[bool, str]:
    """
    Generate a test implementation for a multimodal model using the multimodal
    pipeline template.
    
    Args:
        output_dir: Output directory for generated files
        
    Returns:
        Tuple of (success, output_file_path)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model metadata
    model_name = "flava"
    arch_type = "multimodal"
    hardware_types = ["cpu"]
    
    # Create architecture template
    arch_template = DummyMultimodalTemplate()
    
    # Create hardware templates
    hardware_templates = {
        "cpu": CPUHardwareTemplate()
    }
    
    # Create pipeline templates
    pipeline_templates = {
        "multimodal": MultimodalPipelineTemplate()
    }
    
    # Create template composer
    composer = TemplateComposer(
        hardware_templates=hardware_templates,
        architecture_templates={"multimodal": arch_template},
        pipeline_templates=pipeline_templates,
        output_dir=output_dir
    )
    
    # Generate implementation
    print(f"Generating implementation for {model_name} ({arch_type})...")
    success, output_file = composer.generate_model_implementation(
        model_name=model_name,
        arch_type=arch_type,
        hardware_types=hardware_types,
        force=True
    )
    
    # Report success/failure
    if success:
        print(f"Successfully generated {output_file}")
    else:
        print(f"Failed to generate implementation for {model_name}")
    
    return success, output_file


def verify_implementation(output_file: str) -> bool:
    """
    Verify that the generated implementation contains multimodal pipeline-specific code.
    
    Args:
        output_file: Path to the generated implementation file
        
    Returns:
        True if verification successful, False otherwise
    """
    # Check if file exists
    if not os.path.exists(output_file):
        print(f"Error: Generated file {output_file} does not exist")
        return False
    
    # Read file content
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Prepare verification report
    report_file = os.path.join(os.path.dirname(output_file), "verification_report.md")
    report = ["# Multimodal Pipeline Verification Report\n"]
    verification_success = True
    
    # Check file size (should be substantial)
    file_size = os.path.getsize(output_file)
    report.append(f"## File Information\n- File: {os.path.basename(output_file)}\n- Size: {file_size} bytes\n")
    
    if file_size < 1000:
        report.append(f"❌ File size is suspiciously small: {file_size} bytes\n")
        verification_success = False
    else:
        report.append(f"✅ File size looks reasonable: {file_size} bytes\n")
    
    # Check for multimodal pipeline imports
    report.append("## Checking for Multimodal Pipeline Imports\n")
    if "# Multimodal pipeline imports" in content:
        report.append("✅ Found multimodal pipeline imports\n")
    else:
        report.append("❌ Missing multimodal pipeline imports\n")
        verification_success = False
    
    # Check for multimodal task types
    report.append("## Checking for Multimodal Task Types\n")
    
    # Only checking for the default task type (multimodal_classification)
    # as the template_composer only generates handlers for the default task type
    if "# Preprocess for multimodal classification" in content:
        report.append(f"✅ Found preprocessing for multimodal_classification (default task type)\n")
    else:
        report.append(f"❌ Missing preprocessing for multimodal_classification\n")
        verification_success = False
    
    # Note that other task types are supported by the pipeline but not generated by default
    report.append("\n> Note: The template_composer only generates handlers for the default task type.\n")
    report.append("> To test other task types, you would need to modify the task_type parameter in the generation process.\n")
    
    # Check for multimodal utility functions
    report.append("## Checking for Multimodal Utility Functions\n")
    utility_functions = [
        "resize_image",
        "encode_image_base64",
        "encode_audio_base64",
        "normalize_embedding",
        "compute_similarity"
    ]
    
    for func in utility_functions:
        if f"def {func}" in content:
            report.append(f"✅ Found utility function: {func}\n")
        else:
            report.append(f"❌ Missing utility function: {func}\n")
            verification_success = False
    
    # Check for multimodal-specific handling
    report.append("## Checking for Multimodal-Specific Handling\n")
    multimodal_features = [
        "image_input",
        "text_input",
        "audio_input",
        "multimodal_embeds",
        "image_embeds",
        "text_embeds"
    ]
    
    for feature in multimodal_features:
        if feature in content:
            report.append(f"✅ Found multimodal feature: {feature}\n")
        else:
            report.append(f"❌ Missing multimodal feature: {feature}\n")
            verification_success = False
    
    # Write verification report
    with open(report_file, 'w') as f:
        f.write("\n".join(report))
    
    print(f"Verification report written to {report_file}")
    
    # Return verification result
    if verification_success:
        print("✅ Verification successful! The multimodal pipeline template is working correctly.")
    else:
        print("❌ Verification failed! The multimodal pipeline template is not working correctly.")
    
    return verification_success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test multimodal pipeline template")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="generated_test_models",
        help="Output directory for generated models"
    )
    args = parser.parse_args()
    
    # Generate test implementation
    success, output_file = generate_test_implementation(args.output_dir)
    
    if success:
        # Verify implementation
        verify_implementation(output_file)
    

if __name__ == "__main__":
    main()