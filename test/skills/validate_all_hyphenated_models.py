#!/usr/bin/env python3
"""
Complete validation suite for hyphenated model solution.

This script:
1. Validates all hyphenated model test files using model-specific validation rules
2. Performs inference testing with tiny model versions
3. Generates comprehensive reports with actionable recommendations
4. Provides integration with CI/CD environments

Usage:
    python validate_all_hyphenated_models.py [--inference] [--ci] [--report-dir REPORT_DIR]
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FIXED_TESTS_DIR = CURRENT_DIR / "fixed_tests"
VALIDATION_REPORTS_DIR = CURRENT_DIR / "validation_reports"
COMPREHENSIVE_REPORT_FILE = VALIDATION_REPORTS_DIR / f"comprehensive_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

# Known hyphenated models to validate
KNOWN_HYPHENATED_MODELS = [
    "gpt-j",
    "gpt-neo",
    "gpt-neox",
    "xlm-roberta", 
    "vision-text-dual-encoder",
    "speech-to-text",
    "speech-to-text-2",
    "chinese-clip",
    "data2vec-text",
    "data2vec-audio",
    "data2vec-vision",
    "wav2vec2-bert"
]

def to_valid_identifier(text):
    """Convert hyphenated model names to valid Python identifiers."""
    return text.replace("-", "_")

def regenerate_test_files():
    """Regenerate all test files for hyphenated models."""
    logger.info("Regenerating test files for all hyphenated models...")
    try:
        result = subprocess.run(
            [sys.executable, "integrate_generator_fixes.py", "--generate-all", "--output-dir", "fixed_tests"],
            capture_output=True,
            text=True,
            cwd=CURRENT_DIR
        )
        
        if result.returncode != 0:
            logger.error(f"Error regenerating test files: {result.stderr}")
            return False
        
        logger.info("Test files regenerated successfully")
        return True
    except Exception as e:
        logger.error(f"Error regenerating test files: {str(e)}")
        return False

def run_syntax_validation():
    """Run syntax validation on all test files."""
    logger.info("Running syntax validation on all test files...")
    try:
        result = subprocess.run(
            [sys.executable, "validate_hyphenated_model_solution.py", "--all", "--report", 
             "--output-dir", str(VALIDATION_REPORTS_DIR)],
            capture_output=True,
            text=True,
            cwd=CURRENT_DIR
        )
        
        if result.returncode != 0:
            logger.error(f"Error running syntax validation: {result.stderr}")
            return None
        
        # Look for the latest validation report
        reports = sorted(VALIDATION_REPORTS_DIR.glob("validation_report_*.md"), key=os.path.getmtime, reverse=True)
        if reports:
            logger.info(f"Syntax validation completed: {reports[0]}")
            return reports[0]
        
        logger.warning("No validation report found")
        return None
    except Exception as e:
        logger.error(f"Error running syntax validation: {str(e)}")
        return None

def run_inference_validation(model_name, use_small=True):
    """Run inference validation for a specific model."""
    logger.info(f"Running inference validation for {model_name}...")
    try:
        cmd = [sys.executable, "validate_model_inference.py", "--model", model_name, 
               "--output-dir", str(VALIDATION_REPORTS_DIR)]
        
        if use_small:
            cmd.append("--use-small")
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=CURRENT_DIR
        )
        
        status = "succeeded" if result.returncode == 0 else "failed"
        logger.info(f"Inference validation for {model_name} {status}")
        
        # Parse stdout for results summary
        return {
            "model": model_name,
            "status": status,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        logger.error(f"Error running inference validation for {model_name}: {str(e)}")
        return {
            "model": model_name,
            "status": "error",
            "error": str(e)
        }

def run_parallel_inference_validation(models, use_small=True, max_workers=4):
    """Run inference validation for multiple models in parallel."""
    logger.info(f"Running parallel inference validation for {len(models)} models...")
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {executor.submit(run_inference_validation, model, use_small): model for model in models}
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results[model] = result
                logger.info(f"Completed inference validation for {model}: {result['status']}")
            except Exception as e:
                logger.error(f"Error in inference validation for {model}: {str(e)}")
                results[model] = {"model": model, "status": "error", "error": str(e)}
    
    return results

def collect_validation_reports():
    """Collect validation reports from the validation reports directory."""
    logger.info("Collecting validation reports...")
    
    reports = {
        "syntax": [],
        "inference": []
    }
    
    # Collect syntax validation reports
    syntax_reports = sorted(VALIDATION_REPORTS_DIR.glob("validation_report_*.md"), key=os.path.getmtime, reverse=True)
    for report in syntax_reports[:5]:  # Get the 5 most recent
        reports["syntax"].append(str(report))
    
    # Collect inference validation reports
    inference_reports = sorted(VALIDATION_REPORTS_DIR.glob("inference_validation_*.json"), key=os.path.getmtime, reverse=True)
    for report in inference_reports:
        try:
            with open(report, 'r') as f:
                data = json.load(f)
                reports["inference"].append({"file": str(report), "data": data})
        except Exception as e:
            logger.error(f"Error reading inference report {report}: {str(e)}")
    
    return reports

def extract_model_results(reports):
    """Extract model-specific results from validation reports."""
    model_results = {}
    
    # Process syntax reports
    for report_file in reports["syntax"]:
        try:
            with open(report_file, 'r') as f:
                content = f.read()
                
                # Extract model information from the markdown
                for line in content.split('\n'):
                    if line.startswith("### ") and ". " in line and " - " in line:
                        parts = line.split(" - ")
                        if len(parts) >= 2:
                            name_part = parts[0]
                            status_part = parts[1]
                            
                            if "." in name_part:
                                model_num, model_info = name_part.split(".", 1)
                                if "(" in model_info and ")" in model_info:
                                    model_name = model_info.split("(")[0].strip()
                                    arch_type = model_info.split("(")[1].split(")")[0].strip()
                                    
                                    status = "passed" if "✅" in status_part else "failed"
                                    
                                    if model_name not in model_results:
                                        model_results[model_name] = {"syntax": status, "inference": "unknown", "arch_type": arch_type}
                                    else:
                                        model_results[model_name]["syntax"] = status
                                        model_results[model_name]["arch_type"] = arch_type
        except Exception as e:
            logger.error(f"Error processing syntax report {report_file}: {str(e)}")
    
    # Process inference reports
    for report in reports["inference"]:
        try:
            data = report["data"]
            model_name = data.get("model_name")
            success = data.get("success", False)
            
            if model_name:
                status = "passed" if success else "failed"
                if model_name not in model_results:
                    model_results[model_name] = {"syntax": "unknown", "inference": status}
                else:
                    model_results[model_name]["inference"] = status
        except Exception as e:
            logger.error(f"Error processing inference report: {str(e)}")
    
    return model_results

def generate_comprehensive_report(model_results, inference_results, output_file=None):
    """Generate a comprehensive validation report."""
    if output_file is None:
        output_file = COMPREHENSIVE_REPORT_FILE
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"# Comprehensive Hyphenated Model Validation Report\n\n"
    report += f"Generated: {timestamp}\n\n"
    
    # Summary statistics
    total_models = len(model_results)
    syntax_passed = sum(1 for model, data in model_results.items() if data.get("syntax") == "passed")
    inference_passed = sum(1 for model, data in model_results.items() if data.get("inference") == "passed")
    
    report += f"## Summary\n\n"
    report += f"- Total hyphenated models: {total_models}\n"
    report += f"- Syntax validation passed: {syntax_passed}/{total_models} ({syntax_passed/max(1,total_models)*100:.1f}%)\n"
    report += f"- Inference validation passed: {inference_passed}/{total_models} ({inference_passed/max(1,total_models)*100:.1f}%)\n\n"
    
    # Overall status
    overall_passed = all(data.get("syntax") == "passed" for model, data in model_results.items())
    overall_icon = "✅" if overall_passed else "❌"
    report += f"## Overall Status: {overall_icon} {'PASSED' if overall_passed else 'FAILED'}\n\n"
    
    # Model breakdown table
    report += f"## Model Validation Status\n\n"
    report += f"| Model | Architecture | Syntax | Inference |\n"
    report += f"|-------|--------------|--------|----------|\n"
    
    for model, data in sorted(model_results.items()):
        syntax_status = data.get("syntax", "unknown")
        inference_status = data.get("inference", "unknown")
        arch_type = data.get("arch_type", "unknown")
        
        syntax_icon = "✅" if syntax_status == "passed" else "❌" if syntax_status == "failed" else "❓"
        inference_icon = "✅" if inference_status == "passed" else "❌" if inference_status == "failed" else "❓"
        
        report += f"| {model} | {arch_type} | {syntax_icon} {syntax_status} | {inference_icon} {inference_status} |\n"
    
    # Recommendations section
    report += f"\n## Recommendations\n\n"
    
    # Add general recommendations
    report += "### General Recommendations\n\n"
    report += "- Run the regenerate script to update all test files: `python integrate_generator_fixes.py --generate-all`\n"
    report += "- Use architecture-specific templates for different model types\n"
    report += "- Verify capitalization in model class names (e.g., GPTJForCausalLM, XLMRoBERTaModel)\n"
    report += "- Ensure proper mock detection is implemented in all test files\n\n"
    
    # Add specific recommendations for each model
    report += "### Model-Specific Recommendations\n\n"
    
    for model, data in sorted(model_results.items()):
        syntax_status = data.get("syntax", "unknown")
        inference_status = data.get("inference", "unknown")
        
        if syntax_status != "passed" or inference_status != "passed":
            report += f"**{model}**:\n"
            
            if syntax_status != "passed":
                report += f"- Fix syntax validation issues: Regenerate the test file using `python integrate_generator_fixes.py --generate {model}`\n"
            
            if inference_status != "passed":
                # Look up specific inference issues from results
                specific_issues = []
                if model in inference_results:
                    result = inference_results[model]
                    if "stdout" in result:
                        for line in result["stdout"].split("\n"):
                            if "Issue" in line or "Error" in line:
                                specific_issues.append(line.strip())
                
                report += f"- Fix inference validation issues:\n"
                if specific_issues:
                    for issue in specific_issues[:3]:  # Show the first 3 issues
                        report += f"  - {issue}\n"
                else:
                    report += f"  - Verify model class name and initialization parameters\n"
                    report += f"  - Check for dependency issues (transformers, torch, tokenizers)\n"
            
            report += "\n"
    
    # Write the report to file
    with open(output_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Comprehensive report written to {output_file}")
    return output_file

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Complete validation suite for hyphenated model solution")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate test files before validation")
    parser.add_argument("--inference", action="store_true", help="Run inference validation tests")
    parser.add_argument("--ci", action="store_true", help="Run in CI mode with minimal output")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--report-dir", type=str, default=str(VALIDATION_REPORTS_DIR), help="Directory for validation reports")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to validate (defaults to all)")
    
    args = parser.parse_args()
    
    # Configure output directory
    global VALIDATION_REPORTS_DIR
    VALIDATION_REPORTS_DIR = Path(args.report_dir)
    os.makedirs(VALIDATION_REPORTS_DIR, exist_ok=True)
    
    # Determine which models to validate
    models_to_validate = []
    if args.models:
        models_to_validate = [model.strip() for model in args.models.split(",")]
    else:
        models_to_validate = KNOWN_HYPHENATED_MODELS
    
    logger.info(f"Validating {len(models_to_validate)} hyphenated models")
    
    # Step 1: Regenerate test files if requested
    if args.regenerate:
        regenerate_test_files()
    
    # Step 2: Run syntax validation
    syntax_report = run_syntax_validation()
    
    # Step 3: Run inference validation if requested
    inference_results = {}
    if args.inference:
        inference_results = run_parallel_inference_validation(
            models_to_validate, 
            use_small=True,
            max_workers=args.max_workers
        )
    
    # Step 4: Collect validation reports
    validation_reports = collect_validation_reports()
    
    # Step 5: Extract model results
    model_results = extract_model_results(validation_reports)
    
    # Step 6: Generate comprehensive report
    report_file = generate_comprehensive_report(model_results, inference_results)
    
    # Print summary to console if not in CI mode
    if not args.ci:
        print("\nValidation Summary:")
        print(f"- Total models validated: {len(model_results)}")
        syntax_passed = sum(1 for model, data in model_results.items() if data.get("syntax") == "passed")
        print(f"- Syntax validation passed: {syntax_passed}/{len(model_results)}")
        
        if args.inference:
            inference_passed = sum(1 for model, data in model_results.items() if data.get("inference") == "passed")
            print(f"- Inference validation passed: {inference_passed}/{len(model_results)}")
        
        print(f"\nComprehensive report saved to: {report_file}")
    
    # Return success code if all validations passed
    all_syntax_passed = all(data.get("syntax") == "passed" for model, data in model_results.items())
    all_inference_passed = all(data.get("inference") == "passed" for model, data in model_results.items() 
                               if args.inference and data.get("inference") != "unknown")
    
    if args.inference:
        return 0 if all_syntax_passed and all_inference_passed else 1
    else:
        return 0 if all_syntax_passed else 1

if __name__ == "__main__":
    sys.exit(main())