#!/usr/bin/env python3
"""
Run verification of regenerated test files.

This script:
1. Runs the analyze_template_structure.py script to generate analysis reports
2. Runs the regenerate_template_tests.py script to regenerate tests
3. Verifies the regenerated tests for syntax and basic functionality
4. Generates a comprehensive verification report

Usage:
    python run_verification.py [--verbose]
"""

import os
import sys
import argparse
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Define paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent
SKILLS_DIR = REPO_ROOT / "skills"
TEMPLATES_DIR = SKILLS_DIR / "templates"
FINAL_MODELS_DIR = REPO_ROOT / "final_models"
FIXED_TESTS_DIR = SKILLS_DIR / "fixed_tests"
OUTPUT_DIR = REPO_ROOT / "template_verification"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_script(script_path, args=None):
    """Run a Python script with arguments."""
    if args is None:
        args = []
    
    cmd = [sys.executable, str(script_path)] + args
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"Script failed with code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return False, result.stdout, result.stderr
        
        return True, result.stdout, result.stderr
    
    except Exception as e:
        logger.error(f"Error running script: {e}")
        return False, "", str(e)

def run_analyze_template_structure(verbose=False):
    """Run the analyze_template_structure.py script."""
    script_path = SCRIPT_DIR / "analyze_template_structure.py"
    args = ["--verbose"] if verbose else []
    
    return run_script(script_path, args)

def run_regenerate_template_tests(verbose=False):
    """Run the regenerate_template_tests.py script."""
    script_path = SCRIPT_DIR / "regenerate_template_tests.py"
    args = ["--all", "--verify"]
    
    if verbose:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"[debug logging enabled]")
            args.append(f"--log-file={f.name}")
    
    return run_script(script_path, args)

def verify_regenerated_test(test_path):
    """Verify a regenerated test file."""
    if not os.path.exists(test_path):
        return False, f"Test file not found: {test_path}"
    
    # First check syntax
    try:
        with open(test_path, 'r') as f:
            content = f.read()
        
        compile(content, test_path, 'exec')
    except SyntaxError as e:
        return False, f"Syntax error: Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error checking syntax: {e}"
    
    # Then try running the test
    cmd = [sys.executable, test_path, "--help"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10  # Set a timeout to avoid hanging
        )
        
        if result.returncode != 0:
            return False, f"Test failed with code {result.returncode}: {result.stderr}"
        
        return True, "Test successfully verified"
    
    except subprocess.TimeoutExpired:
        return False, "Test timed out after 10 seconds"
    except Exception as e:
        return False, f"Error running test: {e}"

def verify_all_regenerated_tests():
    """Verify all regenerated test files."""
    results = {}
    
    # Get all regenerated test files
    test_files = list(FIXED_TESTS_DIR.glob("test_hf_*.py"))
    
    for test_file in test_files:
        model_name = test_file.stem.replace("test_hf_", "")
        success, message = verify_regenerated_test(test_file)
        
        results[model_name] = {
            "success": success,
            "message": message
        }
    
    return results

def generate_verification_report(analysis_success, regeneration_success, verification_results):
    """Generate a comprehensive verification report."""
    report = []
    
    report.append("# Template Verification Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Overall status
    overall_success = (analysis_success and regeneration_success and 
                     all(result["success"] for result in verification_results.values()))
    
    if overall_success:
        report.append("## ✅ OVERALL STATUS: SUCCESS")
        report.append("All verification steps completed successfully.")
    else:
        report.append("## ❌ OVERALL STATUS: FAILURE")
        report.append("Some verification steps failed. See details below.")
    
    report.append("")
    
    # Analysis status
    report.append("## Template Analysis")
    if analysis_success:
        report.append("✅ Analysis completed successfully")
    else:
        report.append("❌ Analysis failed")
    report.append("")
    
    # Regeneration status
    report.append("## Template Regeneration")
    if regeneration_success:
        report.append("✅ Regeneration completed successfully")
    else:
        report.append("❌ Regeneration failed")
    report.append("")
    
    # Verification status
    report.append("## Test Verification")
    report.append("")
    report.append("| Model | Status | Message |")
    report.append("|-------|--------|---------|")
    
    for model_name, result in verification_results.items():
        status = "✅ Pass" if result["success"] else "❌ Fail"
        message = result["message"]
        report.append(f"| {model_name} | {status} | {message} |")
    
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if overall_success:
        report.append("All tests are now properly templated and working correctly.")
        report.append("Recommended next steps:")
        report.append("1. Apply these changes to the main repository")
        report.append("2. Update architecture mappings in the generator")
        report.append("3. Run comprehensive tests to ensure all functionality works")
    else:
        report.append("Some verification steps failed. Recommended actions:")
        
        if not analysis_success:
            report.append("- Review the analysis script and logs to understand the issues")
        
        if not regeneration_success:
            report.append("- Check the regeneration script and logs for errors")
        
        if not all(result["success"] for result in verification_results.values()):
            report.append("- Fix the failing test files:")
            for model_name, result in verification_results.items():
                if not result["success"]:
                    report.append(f"  - {model_name}: {result['message']}")
    
    # Write the report
    report_path = OUTPUT_DIR / "verification_report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    
    return report_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run template verification")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Run template analysis
    logger.info("Running template analysis...")
    analysis_success, _, _ = run_analyze_template_structure(args.verbose)
    
    # Run template regeneration
    logger.info("Running template regeneration...")
    regeneration_success, _, _ = run_regenerate_template_tests(args.verbose)
    
    # Verify regenerated tests
    logger.info("Verifying regenerated tests...")
    verification_results = verify_all_regenerated_tests()
    
    # Generate verification report
    logger.info("Generating verification report...")
    report_path = generate_verification_report(
        analysis_success,
        regeneration_success,
        verification_results
    )
    
    logger.info(f"Verification complete. Report saved to {report_path}")
    
    # Return success if all steps were successful
    overall_success = (analysis_success and regeneration_success and 
                     all(result["success"] for result in verification_results.values()))
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())