#!/usr/bin/env python3
"""
Script to diagnose and fix issues with failed model test generations.
"""

import os
import sys
import re
import json
import time
import logging
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the enhanced_generator is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import from enhanced_generator
try:
    from enhanced_generator import (
        MODEL_REGISTRY, 
        ARCHITECTURE_TYPES,
        get_model_architecture,
        generate_test
    )
except ImportError as e:
    logger.error(f"Failed to import from enhanced_generator: {e}")
    sys.exit(1)

# Common pattern errors in failed test generations
COMMON_ERRORS = {
    "syntax error": "The generated file has Python syntax errors.",
    "KeyError": "The model type or a key is not found in the registry.",
    "ImportError": "There was an issue importing a required module.",
    "AttributeError": "A required attribute was not found, likely due to a missing class or method.",
    "TypeError": "A function received an argument of the wrong type.",
    "ValueError": "A function received an invalid value."
}

# Solutions for common errors
ERROR_SOLUTIONS = {
    "syntax error": "Check indentation and string formatting in the template.",
    "KeyError": "Ensure the model is properly registered in MODEL_REGISTRY.",
    "ImportError": "Check that all required modules are available in the current environment.",
    "AttributeError": "Verify that the model class exists and is correctly specified.",
    "TypeError": "Check the argument types passed to functions.",
    "ValueError": "Verify the values provided are within expected ranges."
}

def get_generation_reports(directory: str) -> Dict[str, Any]:
    """
    Parse generation report files in a directory.
    
    Args:
        directory: Directory to search for generation reports
        
    Returns:
        Dictionary of {model_name: details} for failed generations
    """
    report_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "generation_report.md":
                report_files.append(os.path.join(root, file))
    
    if not report_files:
        logger.warning(f"No generation report files found in {directory}")
        return {}
    
    # Parse each report
    failed_models = {}
    
    for report_file in report_files:
        logger.info(f"Parsing report: {report_file}")
        with open(report_file, 'r') as f:
            content = f.read()
        
        # Extract failed model sections
        failed_sections = re.findall(r'#### ❌ ([^\n]+).*?- Error: (.*?)(?=\n\n|\Z)', content, re.DOTALL)
        
        for model, error in failed_sections:
            model = model.strip('`').strip()
            error = error.strip()
            
            # Determine error type
            error_type = "unknown"
            for key in COMMON_ERRORS:
                if key in error.lower():
                    error_type = key
                    break
            
            failed_models[model] = {
                "model": model,
                "error": error,
                "error_type": error_type,
                "report_file": report_file,
                "directory": os.path.dirname(report_file)
            }
    
    logger.info(f"Found {len(failed_models)} failed models across {len(report_files)} reports")
    return failed_models

def analyze_error(model: str, error_msg: str) -> Dict[str, Any]:
    """
    Analyze an error message to determine the root cause.
    
    Args:
        model: Model name
        error_msg: Error message from the generation attempt
        
    Returns:
        Dictionary with analysis results
    """
    # Determine error type
    error_type = "unknown"
    for key in COMMON_ERRORS:
        if key in error_msg.lower():
            error_type = key
            break
    
    # Get the architecture
    try:
        architecture = get_model_architecture(model)
    except:
        architecture = "unknown"
    
    # Check if model is in registry
    in_registry = model in MODEL_REGISTRY
    
    # Find similar models
    similar_models = []
    if architecture != "unknown":
        for m in MODEL_REGISTRY:
            if get_model_architecture(m) == architecture and m != model:
                similar_models.append(m)
    
    # Limit to 5 similar models
    similar_models = similar_models[:5]
    
    # Check for specific error patterns
    specific_issues = []
    
    if "not in MODEL_REGISTRY" in error_msg:
        specific_issues.append("Model not found in MODEL_REGISTRY")
    elif "unexpected indent" in error_msg:
        specific_issues.append("Indentation error in generated code")
    elif "expected an indented block" in error_msg:
        specific_issues.append("Missing indentation in generated code")
    elif "SyntaxError: invalid syntax" in error_msg:
        specific_issues.append("Invalid Python syntax in generated code")
    elif "ImportError" in error_msg:
        specific_issues.append("Module import failure")
    
    # Provide suggested solution
    solution = ERROR_SOLUTIONS.get(error_type, "Inspect the specific error details and template code.")
    
    # For models not in registry, suggest adding to MODEL_REGISTRY
    if not in_registry:
        solution = "Add this model to MODEL_REGISTRY in enhanced_generator.py or additional_models.py"
    
    return {
        "model": model,
        "error_type": error_type,
        "error_description": COMMON_ERRORS.get(error_type, "Unknown error type"),
        "architecture": architecture,
        "in_registry": in_registry,
        "similar_models": similar_models,
        "specific_issues": specific_issues,
        "suggested_solution": solution
    }

def generate_model_fix(model: str, analysis: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Attempt to fix a failed model generation based on analysis.
    
    Args:
        model: Model name
        analysis: Error analysis dictionary
        output_dir: Directory to output fixed model
        
    Returns:
        Dictionary with fix results
    """
    logger.info(f"Attempting to fix generation for model: {model}")
    
    result = {
        "model": model,
        "success": False,
        "approach": "None",
        "message": "No fix attempted",
        "file_path": None
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Try different approaches based on analysis
    
    # Approach 1: Try with similar model as template
    if analysis["similar_models"]:
        result["approach"] = "similar_model_template"
        similar_model = analysis["similar_models"][0]
        logger.info(f"Trying to generate {model} using {similar_model} as template")
        
        try:
            # Use the architecture and template model
            gen_result = generate_test(
                model, 
                output_dir, 
                architecture=analysis["architecture"],
                template_model=similar_model
            )
            
            result["success"] = True
            result["message"] = f"Generated using {similar_model} as template"
            result["file_path"] = gen_result["file_path"]
            return result
        except Exception as e:
            result["message"] = f"Failed to generate using similar model: {str(e)}"
    
    # Approach 2: Try with direct architecture specification
    if analysis["architecture"] != "unknown":
        result["approach"] = "direct_architecture"
        logger.info(f"Trying to generate {model} with direct architecture: {analysis['architecture']}")
        
        try:
            # Use direct architecture specification
            gen_result = generate_test(
                model, 
                output_dir, 
                architecture=analysis["architecture"]
            )
            
            result["success"] = True
            result["message"] = f"Generated using direct architecture specification: {analysis['architecture']}"
            result["file_path"] = gen_result["file_path"]
            return result
        except Exception as e:
            result["message"] = f"Failed to generate with direct architecture: {str(e)}"
    
    # Approach 3: Try with name variations
    name_variations = [
        model.replace("-", "_"),
        model.replace("_", "-"),
        model.replace("-", ""),
        model.replace("_", "")
    ]
    
    result["approach"] = "name_variations"
    for variant in name_variations:
        if variant != model:
            logger.info(f"Trying to generate {model} using name variation: {variant}")
            
            try:
                # Try with name variation
                gen_result = generate_test(variant, output_dir)
                
                result["success"] = True
                result["message"] = f"Generated using name variation: {variant}"
                result["file_path"] = gen_result["file_path"]
                return result
            except Exception as e:
                pass  # Try next variation
    
    result["message"] = "Failed to generate with all name variations"
    
    # Approach 4: Create custom entry if not in registry
    if not analysis["in_registry"]:
        result["approach"] = "custom_registry_entry"
        logger.info(f"Trying to create custom registry entry for {model}")
        
        # Determine best architecture and class
        if analysis["architecture"] == "unknown":
            if "bert" in model.lower():
                arch = "encoder-only"
                model_class = "BertForMaskedLM"
                task = "fill-mask"
                test_input = "The quick brown fox jumps over the [MASK] dog."
            elif "gpt" in model.lower():
                arch = "decoder-only"
                model_class = "GPT2LMHeadModel"
                task = "text-generation"
                test_input = "Once upon a time"
            elif "t5" in model.lower():
                arch = "encoder-decoder"
                model_class = "T5ForConditionalGeneration"
                task = "text2text-generation"
                test_input = "translate English to German: Hello, how are you?"
            elif "vit" in model.lower() or "vision" in model.lower():
                arch = "vision"
                model_class = "ViTForImageClassification"
                task = "image-classification"
                test_input = "test.jpg"
            elif "clip" in model.lower():
                arch = "vision-text"
                model_class = "CLIPModel"
                task = "zero-shot-image-classification"
                test_input = ["test.jpg", ["a photo of a cat", "a photo of a dog"]]
            elif "whisper" in model.lower() or "speech" in model.lower() or "audio" in model.lower():
                arch = "speech"
                model_class = "WhisperForConditionalGeneration"
                task = "automatic-speech-recognition"
                test_input = "test.mp3"
            elif "llava" in model.lower() or "multimodal" in model.lower():
                arch = "multimodal"
                model_class = "LlavaForConditionalGeneration"
                task = "image-to-text"
                test_input = ["test.jpg", "What is in this image?"]
            else:
                # Default to encoder-only as a fallback
                arch = "encoder-only"
                model_class = "AutoModel"
                task = "fill-mask"
                test_input = "The quick brown fox jumps over the [MASK] dog."
        else:
            arch = analysis["architecture"]
            
            # Get default class for architecture
            if arch == "encoder-only":
                model_class = "BertForMaskedLM"
                task = "fill-mask"
                test_input = "The quick brown fox jumps over the [MASK] dog."
            elif arch == "decoder-only":
                model_class = "GPT2LMHeadModel"
                task = "text-generation"
                test_input = "Once upon a time"
            elif arch == "encoder-decoder":
                model_class = "T5ForConditionalGeneration"
                task = "text2text-generation"
                test_input = "translate English to German: Hello, how are you?"
            elif arch == "vision":
                model_class = "ViTForImageClassification"
                task = "image-classification"
                test_input = "test.jpg"
            elif arch == "vision-text":
                model_class = "CLIPModel"
                task = "zero-shot-image-classification"
                test_input = ["test.jpg", ["a photo of a cat", "a photo of a dog"]]
            elif arch == "speech":
                model_class = "WhisperForConditionalGeneration"
                task = "automatic-speech-recognition"
                test_input = "test.mp3"
            elif arch == "multimodal":
                model_class = "LlavaForConditionalGeneration"
                task = "image-to-text"
                test_input = ["test.jpg", "What is in this image?"]
            else:
                model_class = "AutoModel"
                task = "feature-extraction"
                test_input = "The quick brown fox jumps over the lazy dog."
        
        # Use model name components to guess a default model
        model_parts = re.split(r'[-_]', model.lower())
        org_prefix = ""
        
        # Guess organization prefix
        common_orgs = ["google", "facebook", "microsoft", "openai", "meta", "huggingface"]
        for org in common_orgs:
            if any(part == org for part in model_parts):
                org_prefix = f"{org}/"
                break
        
        # Create a temporary registry entry
        temp_registry = MODEL_REGISTRY.copy()
        model_id = model.replace("-", "_")
        
        temp_registry[model_id] = {
            "default_model": f"{org_prefix}{model.lower()}",
            "task": task,
            "class": model_class,
            "test_input": test_input
        }
        
        try:
            # Try to generate with the temporary registry
            with open("temp_registry_fix.py", "w") as f:
                f.write(f"""
import sys
from enhanced_generator import generate_test as original_generate_test

# Add custom entry
MODEL_REGISTRY = {temp_registry}

def modified_generate_test(model_type, output_dir, **kwargs):
    return original_generate_test(model_type, output_dir, **kwargs)

if __name__ == "__main__":
    # Generate test with custom registry
    result = modified_generate_test("{model_id}", "{output_dir}")
    print(result["file_path"])
""")
            
            # Run the temporary script
            from subprocess import run, PIPE
            proc = run([sys.executable, "temp_registry_fix.py"], stdout=PIPE, stderr=PIPE, text=True)
            
            if proc.returncode == 0 and proc.stdout.strip():
                file_path = proc.stdout.strip()
                if os.path.exists(file_path):
                    result["success"] = True
                    result["message"] = f"Generated using custom registry entry with {model_class}"
                    result["file_path"] = file_path
                    os.remove("temp_registry_fix.py")
                    return result
            
            result["message"] = f"Failed to generate with custom registry: {proc.stderr}"
            
        except Exception as e:
            result["message"] = f"Failed to generate with custom registry: {str(e)}"
        
        finally:
            # Clean up temp file
            if os.path.exists("temp_registry_fix.py"):
                os.remove("temp_registry_fix.py")
    
    return result

def generate_diagnostics_report(failed_models: Dict[str, Any], fixes: Dict[str, Any], output_file: str) -> None:
    """
    Generate a diagnostics report for failed model generations.
    
    Args:
        failed_models: Dictionary of failed model details
        fixes: Dictionary of fix attempts and results
        output_file: Output file path for the report
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'w') as f:
        f.write(f"# Model Generation Diagnostics Report\n\n")
        f.write(f"**Generated on:** {timestamp}\n\n")
        
        # Overall statistics
        total_failed = len(failed_models)
        total_fixed = sum(1 for fix in fixes.values() if fix["success"])
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Total failed models:** {total_failed}\n")
        f.write(f"- **Successfully fixed:** {total_fixed} ({round(total_fixed/total_failed*100, 1)}%)\n")
        f.write(f"- **Remaining issues:** {total_failed - total_fixed}\n\n")
        
        # Error type statistics
        error_types = {}
        for model_info in failed_models.values():
            error_type = model_info.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        f.write(f"## Error Types\n\n")
        for error_type, count in error_types.items():
            f.write(f"- **{error_type}:** {count} ({round(count/total_failed*100, 1)}%)\n")
        f.write("\n")
        
        # Fix approach statistics
        fix_approaches = {}
        for fix in fixes.values():
            approach = fix.get("approach", "None")
            if fix["success"]:
                fix_approaches[approach] = fix_approaches.get(approach, 0) + 1
        
        f.write(f"## Successful Fix Approaches\n\n")
        for approach, count in fix_approaches.items():
            f.write(f"- **{approach}:** {count} ({round(count/total_fixed*100, 1)}%)\n")
        f.write("\n")
        
        # Successfully fixed models
        fixed_models = [m for m, fix in fixes.items() if fix["success"]]
        if fixed_models:
            f.write(f"## Successfully Fixed Models ({len(fixed_models)})\n\n")
            for model in sorted(fixed_models):
                fix = fixes[model]
                f.write(f"### ✅ {model}\n\n")
                f.write(f"- **Approach:** {fix['approach']}\n")
                f.write(f"- **Message:** {fix['message']}\n")
                f.write(f"- **File:** {os.path.basename(fix['file_path'])}\n\n")
        
        # Remaining issues
        remaining = [m for m, fix in fixes.items() if not fix["success"]]
        if remaining:
            f.write(f"## Remaining Issues ({len(remaining)})\n\n")
            for model in sorted(remaining):
                f.write(f"### ❌ {model}\n\n")
                f.write(f"- **Original error:** {failed_models[model]['error']}\n")
                f.write(f"- **Fix approach tried:** {fixes[model]['approach']}\n")
                f.write(f"- **Fix result:** {fixes[model]['message']}\n\n")
                
                # Add suggestions for manual intervention
                analysis = analyze_error(model, failed_models[model]['error'])
                f.write(f"**Suggested manual fix:**\n\n")
                f.write(f"1. {analysis['suggested_solution']}\n")
                if analysis["architecture"] != "unknown":
                    f.write(f"2. Use architecture type: `{analysis['architecture']}`\n")
                if analysis["similar_models"]:
                    f.write(f"3. Check similar models for reference: {', '.join(analysis['similar_models'])}\n")
                f.write("\n")
        
        # Recommendations for next steps
        f.write(f"## Next Steps\n\n")
        f.write(f"1. Commit the successfully fixed models\n")
        f.write(f"2. Manually address the remaining {total_failed - total_fixed} issues using the suggestions above\n")
        f.write(f"3. Re-run the verification script to ensure all tests pass syntactically\n")
        f.write(f"4. Execute the test suite with the new implementations\n")
    
    logger.info(f"Diagnostics report written to {output_file}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Diagnose and fix failed model test generations")
    parser.add_argument("--input-dir", default=".", help="Directory containing generation reports")
    parser.add_argument("--output-dir", default="fixed_tests", help="Directory to output fixed tests")
    parser.add_argument("--report", default="model_diagnosis_report.md", help="Output file for diagnostics report")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix failed generations")
    args = parser.parse_args()
    
    # Get failed model details
    failed_models = get_generation_reports(args.input_dir)
    
    if not failed_models:
        logger.error("No failed models found to diagnose")
        return 1
    
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze and attempt fixes for each failed model
    fixes = {}
    
    for model, model_info in failed_models.items():
        logger.info(f"Analyzing model: {model}")
        analysis = analyze_error(model, model_info["error"])
        
        # Display analysis
        logger.info(f"  Error type: {analysis['error_type']}")
        logger.info(f"  Architecture: {analysis['architecture']}")
        logger.info(f"  In registry: {analysis['in_registry']}")
        logger.info(f"  Suggested solution: {analysis['suggested_solution']}")
        
        # Attempt fix if requested
        if args.fix:
            fix_result = generate_model_fix(model, analysis, args.output_dir)
            fixes[model] = fix_result
            
            if fix_result["success"]:
                logger.info(f"✅ Successfully fixed model: {model}")
                logger.info(f"   Approach: {fix_result['approach']}")
                logger.info(f"   Output: {fix_result['file_path']}")
            else:
                logger.warning(f"❌ Failed to fix model: {model}")
                logger.warning(f"   Approach tried: {fix_result['approach']}")
                logger.warning(f"   Result: {fix_result['message']}")
        
    # Generate diagnostics report
    generate_diagnostics_report(failed_models, fixes, args.report)
    
    # Return success if all models were fixed
    if args.fix:
        success_count = sum(1 for fix in fixes.values() if fix["success"])
        logger.info(f"Fixed {success_count} out of {len(failed_models)} failed models")
        return 0 if success_count == len(failed_models) else 1
    else:
        logger.info(f"Analyzed {len(failed_models)} failed models. Run with --fix to attempt repairs.")
        return 0

if __name__ == "__main__":
    sys.exit(main())