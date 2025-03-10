#!/usr/bin/env python3
"""
Generator Verification Script (March 6, 2025)

This script verifies that the fixed generators are working correctly by:
1. Generating test files for key models
2. Checking that the files have the critical fixes
3. Verifying Python syntax in the generated files
"""

import os
import sys
import subprocess
import importlib
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Key models to test
KEY_MODELS = [
    "bert", "t5", "vit", "clip", "whisper", "wav2vec2", 
    "clap", "llama", "llava", "detr", "xclip", "qwen2"
]

# Critical fixes to check for
CRITICAL_FIXES = [
    "run_tests",                 # Check for run_tests method
    "resolve_model_name",        # Check for model name resolution
    "MODEL_REGISTRY",            # Check for model registry
    "TestBertModels",            # Check for Models suffix
    "openvino_label",            # Check for OpenVINO label parameter
    "AutoFeatureExtractor",      # Check for modality-specific processors
    "AutoImageProcessor",
    "AutoProcessor",
    "modality",                  # Check for modality-based initialization
]

def check_file_for_fixes(file_path, fixes):
    """Check if a file contains the critical fixes."""
    results = {}
    with open(file_path, 'r') as f:
        content = f.read()
        for fix in fixes:
            results[fix] = fix in content
    return results

def verify_python_syntax(file_path):
    """Verify that a file has valid Python syntax."""
    try:
        # Use Python to compile the file
        result = subprocess.run(
            ["python", "-m", "py_compile", file_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return True, "Syntax OK"
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def run_generator_test(generator_script, model, output_dir):
    """Run a generator test with a specific model."""
    try:
        # Create command
        cmd = [
            "python", generator_script,
            "--generate", model,
            "--cross-platform",
            "--output-dir", output_dir
        ]
        
        # Run generator
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Generated test for {model}")
            return True, f"test_hf_{model.replace('-', '_')}.py"
        else:
            logger.error(f"❌ Failed to generate test for {model}: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        logger.error(f"Error running generator: {e}")
        return False, str(e)

def main():
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Test results
        generator_results = []
        
        # Test each key model
        for model in KEY_MODELS:
            # Run the generator
            success, file_or_error = run_generator_test(
                "fixed_merged_test_generator.py", 
                model, 
                temp_dir
            )
            
            if success:
                output_file = os.path.join(temp_dir, file_or_error)
                
                # Verify Python syntax
                syntax_ok, syntax_msg = verify_python_syntax(output_file)
                
                # Check for critical fixes
                fix_results = check_file_for_fixes(output_file, CRITICAL_FIXES)
                
                # Calculate fix percentage
                fix_count = sum(1 for fix in fix_results.values() if fix)
                fix_pct = (fix_count / len(CRITICAL_FIXES)) * 100
                
                # Store results
                generator_results.append({
                    "model": model,
                    "success": True,
                    "syntax_ok": syntax_ok,
                    "fixes": fix_results,
                    "fix_pct": fix_pct
                })
                
                # Log results
                logger.info(f"Model: {model}, Syntax: {'✅' if syntax_ok else '❌'}, Fixes: {fix_pct:.1f}%")
                for fix, present in fix_results.items():
                    logger.info(f"  - {fix}: {'✅' if present else '❌'}")
            else:
                generator_results.append({
                    "model": model,
                    "success": False,
                    "error": file_or_error
                })
        
        # Print summary
        logger.info("\n===== GENERATOR VERIFICATION SUMMARY =====")
        succeeded = sum(1 for r in generator_results if r["success"])
        syntax_ok = sum(1 for r in generator_results if r.get("syntax_ok", False))
        
        logger.info(f"Generator success: {succeeded}/{len(KEY_MODELS)} models ({succeeded/len(KEY_MODELS)*100:.1f}%)")
        logger.info(f"Syntax correctness: {syntax_ok}/{len(KEY_MODELS)} models ({syntax_ok/len(KEY_MODELS)*100:.1f}%)")
        
        # Calculate average fix percentage
        fix_pcts = [r["fix_pct"] for r in generator_results if r.get("fix_pct") is not None]
        if fix_pcts:
            avg_fix_pct = sum(fix_pcts) / len(fix_pcts)
            logger.info(f"Critical fixes: {avg_fix_pct:.1f}% present on average")
        
        # Overall assessment
        if succeeded == len(KEY_MODELS) and avg_fix_pct > 90:
            logger.info("✅ VERIFICATION PASSED: Generators are working correctly!")
            return 0
        else:
            logger.warning("⚠️ VERIFICATION ISSUES: Generators need further improvements.")
            return 1

if __name__ == "__main__":
    sys.exit(main())