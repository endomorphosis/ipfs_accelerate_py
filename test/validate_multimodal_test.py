#!/usr/bin/env python3

import os
import sys
import ast
import importlib.util
import logging
import argparse
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Required methods for ModelTest compliance
REQUIRED_METHODS = [
    "test_model_loading",
    "load_model",
    "verify_model_output",
    "detect_preferred_device"
]

def load_module_from_path(file_path: str) -> Optional[Any]:
    """Load a Python module from file path."""
    try:
        module_name = os.path.basename(file_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            logger.error(f"Failed to create spec for {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error loading module from {file_path}: {e}")
        return None

def analyze_file_ast(file_path: str) -> Dict[str, Any]:
    """Analyze Python file using AST to verify ModelTest compliance."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # Find all class definitions
        classes = {}
        model_test_subclasses = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                classes[node.name] = {
                    "name": node.name,
                    "bases": bases,
                    "methods": methods
                }
                
                # Check if this is a ModelTest subclass
                if "ModelTest" in bases:
                    model_test_subclasses.append(node.name)
        
        # Analyze model test subclasses
        results = []
        for class_name in model_test_subclasses:
            class_info = classes[class_name]
            
            # Check for required methods
            missing_methods = [method for method in REQUIRED_METHODS if method not in class_info["methods"]]
            
            results.append({
                "class_name": class_name,
                "has_all_required_methods": len(missing_methods) == 0,
                "missing_methods": missing_methods,
                "methods": class_info["methods"],
                "bases": class_info["bases"]
            })
        
        return {
            "file_path": file_path,
            "classes": classes,
            "model_test_subclasses": model_test_subclasses,
            "results": results,
            "overall_compliant": all(r["has_all_required_methods"] for r in results) if results else False
        }
    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {e}")
        return {
            "file_path": file_path,
            "error": str(e),
            "overall_compliant": False
        }

def check_model_type(file_path: str) -> str:
    """Determine model type category (text, vision, audio, multimodal)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for multimodal indicators
        multimodal_indicators = [
            "CLIP", "clip", "BLIP", "blip", "vision-text", "vision_text", "VisionText",
            "image_and_text", "image and text", "vision and text", "multimodal"
        ]
        
        for indicator in multimodal_indicators:
            if indicator in content:
                return "multimodal"
        
        # Check for vision indicators
        vision_indicators = [
            "vision", "image", "Vision", "Image", "ViT", "Swin", "ConvNext", 
            "classification", "segmentation", "detection", "visual"
        ]
        
        for indicator in vision_indicators:
            if indicator in content:
                return "vision"
        
        # Check for audio indicators
        audio_indicators = [
            "audio", "Audio", "speech", "Speech", "Whisper", "wav2vec", 
            "ASR", "asr", "transcription", "hubert", "librosa", "soundfile"
        ]
        
        for indicator in audio_indicators:
            if indicator in content:
                return "audio"
        
        # Default to text
        return "text"
    except Exception as e:
        logger.error(f"Error determining model type for {file_path}: {e}")
        return "unknown"

def validate_test_file(file_path: str) -> Dict[str, Any]:
    """Validate if a test file is compliant with the ModelTest pattern."""
    if not os.path.exists(file_path):
        return {
            "file_path": file_path,
            "exists": False,
            "compliant": False,
            "error": "File does not exist"
        }
    
    try:
        # Determine model type
        model_type = check_model_type(file_path)
        
        # Analyze the file structure
        analysis = analyze_file_ast(file_path)
        
        # Determine if the file is compliant with ModelTest pattern
        is_compliant = analysis.get("overall_compliant", False)
        
        return {
            "file_path": file_path,
            "exists": True,
            "compliant": is_compliant,
            "model_type": model_type,
            "analysis": analysis,
            "error": analysis.get("error", None)
        }
    except Exception as e:
        logger.error(f"Error validating file {file_path}: {e}")
        return {
            "file_path": file_path,
            "exists": True,
            "compliant": False,
            "error": str(e)
        }

def generate_validation_report(validation_results: List[Dict[str, Any]]) -> str:
    """Generate a formatted report of validation results."""
    report = "# Test Files Compliance Report\n\n"
    
    # Compliance statistics
    total_files = len(validation_results)
    compliant_files = sum(1 for r in validation_results if r.get("compliant", False))
    compliance_rate = (compliant_files / total_files * 100) if total_files > 0 else 0
    
    report += f"## Summary\n\n"
    report += f"- Total files: {total_files}\n"
    report += f"- Compliant files: {compliant_files}\n"
    report += f"- Compliance rate: {compliance_rate:.1f}%\n\n"
    
    # Files by model type
    model_types = {}
    for result in validation_results:
        model_type = result.get("model_type", "unknown")
        model_types[model_type] = model_types.get(model_type, 0) + 1
    
    report += "## Files by Model Type\n\n"
    for model_type, count in model_types.items():
        report += f"- {model_type.capitalize()}: {count}\n"
    report += "\n"
    
    # Compliant files list
    report += "## Compliant Files\n\n"
    for result in sorted(validation_results, key=lambda x: x["file_path"]):
        if result.get("compliant", False):
            model_type = result.get("model_type", "unknown")
            report += f"- ✅ [{os.path.basename(result['file_path'])}]({result['file_path']}) ({model_type})\n"
    report += "\n"
    
    # Non-compliant files list with details
    report += "## Non-compliant Files\n\n"
    for result in sorted(validation_results, key=lambda x: x["file_path"]):
        if not result.get("compliant", False):
            model_type = result.get("model_type", "unknown")
            report += f"- ❌ [{os.path.basename(result['file_path'])}]({result['file_path']}) ({model_type})\n"
            
            # Show missing methods if available
            analysis = result.get("analysis", {})
            results = analysis.get("results", [])
            
            if results:
                for class_result in results:
                    missing_methods = class_result.get("missing_methods", [])
                    if missing_methods:
                        report += f"  - Class `{class_result['class_name']}` missing methods: {', '.join(missing_methods)}\n"
            
            # Show error if available
            error = result.get("error")
            if error:
                report += f"  - Error: {error}\n"
    
    return report

def find_test_files(directory: str, pattern: str = "test_*.py") -> List[str]:
    """Find test files matching the pattern in the given directory."""
    import glob
    
    # Find all test files
    search_pattern = os.path.join(directory, pattern)
    test_files = glob.glob(search_pattern)
    
    return test_files

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Validate test files for ModelTest compliance")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a specific test file")
    group.add_argument("--directory", type=str, help="Directory to scan for test files")
    parser.add_argument("--pattern", type=str, default="test_*.py", help="File pattern to match (default: test_*.py)")
    parser.add_argument("--report", type=str, help="Output path for validation report")
    
    args = parser.parse_args()
    
    # Validate files
    validation_results = []
    
    if args.file:
        # Validate a single file
        result = validate_test_file(args.file)
        validation_results.append(result)
        
        # Print results
        if result["compliant"]:
            logger.info(f"✅ {args.file} is compliant with ModelTest pattern")
        else:
            logger.info(f"❌ {args.file} is not compliant with ModelTest pattern")
            if "error" in result and result["error"]:
                logger.info(f"   Error: {result['error']}")
            if "analysis" in result and "results" in result["analysis"]:
                for class_result in result["analysis"]["results"]:
                    if not class_result["has_all_required_methods"]:
                        logger.info(f"   Class {class_result['class_name']} is missing methods: {class_result['missing_methods']}")
    
    elif args.directory:
        # Find test files in directory
        test_files = find_test_files(args.directory, args.pattern)
        logger.info(f"Found {len(test_files)} test files in {args.directory}")
        
        # Validate each file
        for file_path in test_files:
            result = validate_test_file(file_path)
            validation_results.append(result)
            
            # Print result
            status = "✅" if result["compliant"] else "❌"
            model_type = result.get("model_type", "unknown")
            logger.info(f"{status} {os.path.basename(file_path)} ({model_type})")
    
    # Calculate compliance rate
    total_files = len(validation_results)
    compliant_files = sum(1 for r in validation_results if r.get("compliant", False))
    compliance_rate = (compliant_files / total_files * 100) if total_files > 0 else 0
    
    logger.info(f"Compliance rate: {compliance_rate:.1f}% ({compliant_files}/{total_files})")
    
    # Generate report if requested
    if args.report:
        report = generate_validation_report(validation_results)
        try:
            with open(args.report, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Validation report saved to {args.report}")
        except Exception as e:
            logger.error(f"Error saving report to {args.report}: {e}")
    
if __name__ == "__main__":
    main()