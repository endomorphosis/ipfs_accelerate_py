#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to verify that all architecture templates compile correctly and handle
hardware detection properly, including ROCm support.
"""

import os
import sys
import json
import ast
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verify_templates")

# List of architecture template files to check
TEMPLATES_TO_CHECK = [
    "encoder_only_template.py",
    "decoder_only_template.py",
    "encoder_decoder_template.py",
    "vision_template.py",
    "vision_text_template.py",
    "speech_template.py",
    "multimodal_template.py",
    "rag_model_template.py",
    "graph_model_template.py",
    "time_series_model_template.py",
    "object_detection_model_template.py",
    "diffusion_model_template.py",
    "moe_model_template.py",
    "ssm_model_template.py",
]

# Hardware backends to verify in templates
HARDWARE_BACKENDS = ["cpu", "cuda", "rocm", "openvino", "apple", "qualcomm"]

def check_template_syntax(file_path):
    """Check if a template file has valid Python syntax.
    
    Args:
        file_path: Path to the template file.
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fill in placeholder variables to make it valid Python
        # This is a simplified approach - in reality, templates may have complex placeholders
        test_content = content
        test_content = test_content.replace("{model_type}", "model_name")
        test_content = test_content.replace("{model_type_upper}", "MODEL_NAME")
        test_content = test_content.replace("{task_type}", "task_name")
        test_content = test_content.replace("{task_class}", "TaskName")
        test_content = test_content.replace("{test_input}", "This is a test.")
        test_content = test_content.replace("{automodel_class}", "AutoModel")
        test_content = test_content.replace("{hidden_size}", "768")
        test_content = test_content.replace("{model_description}", "This is a test model.")
        
        # Try to parse with AST to check syntax
        ast.parse(test_content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_hardware_support(file_path, hardware_type):
    """Check if a template file includes support for a specific hardware backend.
    
    Args:
        file_path: Path to the template file.
        hardware_type: Hardware backend to check for.
        
    Returns:
        Tuple of (has_support, details)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for initialization method for this hardware
        init_method = f"init_{hardware_type}"
        has_init = init_method in content
        
        # Check for hardware-specific handler
        handler_method = f"create_{hardware_type}_"
        has_handler = handler_method in content
        
        # Check for hardware detection
        detection_patterns = {
            "cpu": ["cpu"],
            "cuda": ["cuda.is_available()", "cuda.device_count()", "cuda.get_device"],
            "rocm": ["hip.is_available()", "AMD", "Radeon", "HIP_VISIBLE_DEVICES", "rocm"],
            "openvino": ["openvino", "OVModelFor", "OpenVINO"],
            "apple": ["mps.is_available()", "Apple Silicon", "apple"],
            "qualcomm": ["qnn_wrapper", "qti.aisw", "QUALCOMM_SDK", "qualcomm"]
        }
        
        patterns = detection_patterns.get(hardware_type, [hardware_type])
        has_detection = any(pattern in content for pattern in patterns)
        
        # Additional check for ROCm support
        rocm_specifics = {}
        if hardware_type == "rocm":
            rocm_specifics["checks_hip_api"] = "hip.is_available()" in content
            rocm_specifics["checks_amd_in_name"] = ("AMD" in content and "device_name" in content)
            rocm_specifics["handles_hip_visible_devices"] = "HIP_VISIBLE_DEVICES" in content
            rocm_specifics["handles_half_precision"] = "half precision" in content.lower() and "AMD GPU" in content
        
        return {
            "has_support": has_init and has_handler and has_detection,
            "has_init_method": has_init,
            "has_handler_method": has_handler,
            "has_detection": has_detection,
            "file_name": os.path.basename(file_path),
            **rocm_specifics
        }
        
    except Exception as e:
        return {
            "has_support": False,
            "error": str(e),
            "file_name": os.path.basename(file_path)
        }

def main():
    """Main function."""
    print("\n=== Template Verification ===\n")
    
    # Find the templates directory
    templates_dir = Path(__file__).parent / "templates"
    if not templates_dir.exists():
        print(f"Templates directory not found: {templates_dir}")
        return 1
    
    results = {
        "syntax_valid": 0,
        "syntax_invalid": 0,
        "hardware_support": {hw: 0 for hw in HARDWARE_BACKENDS},
        "templates": []
    }
    
    # Process each template
    for template_file in TEMPLATES_TO_CHECK:
        template_path = templates_dir / template_file
        if not template_path.exists():
            print(f"❌ Template not found: {template_file}")
            continue
        
        # Check syntax
        is_valid, error = check_template_syntax(template_path)
        
        template_result = {
            "file_name": template_file,
            "syntax_valid": is_valid,
            "hardware_support": {}
        }
        
        if is_valid:
            results["syntax_valid"] += 1
            print(f"✅ {template_file}: Syntax valid")
            
            # Check hardware support
            for hw in HARDWARE_BACKENDS:
                hw_support = check_hardware_support(template_path, hw)
                template_result["hardware_support"][hw] = hw_support
                
                if hw_support["has_support"]:
                    results["hardware_support"][hw] += 1
        else:
            results["syntax_invalid"] += 1
            template_result["syntax_error"] = error
            print(f"❌ {template_file}: Syntax invalid - {error}")
        
        results["templates"].append(template_result)
    
    # Print summary
    print("\n=== Summary ===\n")
    print(f"Templates checked: {len(results['templates'])}")
    print(f"Syntax valid: {results['syntax_valid']}")
    print(f"Syntax invalid: {results['syntax_invalid']}")
    
    print("\nHardware support:")
    for hw, count in results["hardware_support"].items():
        print(f"  - {hw}: {count}/{results['syntax_valid']} templates")
    
    # Check ROCm support specifically
    rocm_supported_templates = []
    for template in results["templates"]:
        if template.get("syntax_valid") and \
           template.get("hardware_support", {}).get("rocm", {}).get("has_support", False):
            rocm_supported_templates.append(template["file_name"])
    
    print("\nROCm support details:")
    if rocm_supported_templates:
        print(f"  - {len(rocm_supported_templates)} templates support ROCm:")
        for template in rocm_supported_templates:
            print(f"    - {template}")
    else:
        print("  - No templates with full ROCm support found")
    
    # Write detailed results to file
    output_file = Path(__file__).parent / "template_verification_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results written to: {output_file}")
    
    return 0 if results["syntax_invalid"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())