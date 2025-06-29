#!/usr/bin/env python3
"""
Verify mock detection in all HuggingFace test files.

This script:
1. Checks all test files for proper mock detection implementation
2. Applies fixes to files missing proper mock detection
3. Verifies mock detection by testing with different environment variable combinations
4. Generates a comprehensive report of the results

Usage:
    python verify_all_mock_detection.py [--check-only] [--fix] [--verify]
"""

import os
import sys
import re
import argparse
import subprocess
import logging
import concurrent.futures
from datetime import datetime
from pathlib import Path

# Set up logging
log_file = f"mock_detection_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
GREEN = "\033[32m"
BLUE = "\033[34m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def check_mock_detection(file_path):
    """
    Check if a file has proper mock detection implemented.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        dict: Results of the check with status and missing components
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Define patterns to check for
        patterns = {
            "env_vars": {
                "MOCK_TORCH": "MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'",
                "MOCK_TRANSFORMERS": "MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'",
                "MOCK_TOKENIZERS": "MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'",
                "MOCK_SENTENCEPIECE": "MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'"
            },
            "dependencies": {
                "torch": {"import": "import torch", "has_var": "HAS_TORCH = True", "mock_check": "if MOCK_TORCH:"},
                "transformers": {"import": "import transformers", "has_var": "HAS_TRANSFORMERS = True", "mock_check": "if MOCK_TRANSFORMERS:"},
                "tokenizers": {"import": "import tokenizers", "has_var": "HAS_TOKENIZERS = True", "mock_check": "if MOCK_TOKENIZERS:"},
                "sentencepiece": {"import": "import sentencepiece", "has_var": "HAS_SENTENCEPIECE = True", "mock_check": "if MOCK_SENTENCEPIECE:"}
            },
            "real_inference": "using_real_inference = HAS_TRANSFORMERS and HAS_TORCH",
            "using_mocks": "using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE",
            "visual_indicators": {
                "real": "Using REAL INFERENCE with actual models",
                "mock": "Using MOCK OBJECTS for CI/CD testing only"
            },
            "color_codes": {
                "GREEN": "GREEN = \"\\033[32m\"",
                "BLUE": "BLUE = \"\\033[34m\"",
                "RESET": "RESET = \"\\033[0m\""
            }
        }
        
        # Check for each pattern
        results = {
            "file": file_path,
            "status": "OK",
            "missing": [],
            "has_env_vars": True,
            "has_mock_checks": True,
            "has_detection_logic": True,
            "has_visual_indicators": True,
            "has_color_codes": True
        }
        
        # Check environment variables
        for var_name, var_pattern in patterns["env_vars"].items():
            if var_pattern not in content:
                results["has_env_vars"] = False
                results["missing"].append(f"Environment variable: {var_name}")
        
        # Check dependency imports and mock checks
        for dep_name, dep_patterns in patterns["dependencies"].items():
            # Skip sentencepiece and tokenizers for some model types
            if dep_name in ["sentencepiece", "tokenizers"]:
                # For certain models, these might not be needed
                if "test_hf_vit" in file_path or "test_hf_vision" in file_path:
                    continue
            
            if dep_patterns["import"] not in content or dep_patterns["has_var"] not in content:
                results["missing"].append(f"Dependency import: {dep_name}")
            
            if dep_patterns["mock_check"] not in content:
                results["has_mock_checks"] = False
                results["missing"].append(f"Mock check for: {dep_name}")
        
        # Check detection logic
        if patterns["real_inference"] not in content:
            results["has_detection_logic"] = False
            results["missing"].append("Detection logic: using_real_inference")
        
        if patterns["using_mocks"] not in content:
            results["has_detection_logic"] = False
            results["missing"].append("Detection logic: using_mocks")
        
        # Check visual indicators
        for indicator_name, indicator_pattern in patterns["visual_indicators"].items():
            if indicator_pattern not in content:
                results["has_visual_indicators"] = False
                results["missing"].append(f"Visual indicator: {indicator_name}")
        
        # Check color codes
        for code_name, code_pattern in patterns["color_codes"].items():
            if code_pattern not in content:
                results["has_color_codes"] = False
                results["missing"].append(f"Color code: {code_name}")
        
        # Update overall status
        if results["missing"]:
            results["status"] = "NEEDS_FIX"
        
        return results
    
    except Exception as e:
        logger.error(f"Error checking {file_path}: {e}")
        return {
            "file": file_path,
            "status": "ERROR",
            "error": str(e),
            "missing": ["Error occurred during check"],
            "has_env_vars": False,
            "has_mock_checks": False,
            "has_detection_logic": False,
            "has_visual_indicators": False,
            "has_color_codes": False
        }

def fix_mock_detection(file_path, check_results):
    """
    Fix mock detection in a file based on check results.
    
    Args:
        file_path: Path to the file to fix
        check_results: Results from check_mock_detection
        
    Returns:
        bool: True if fixes were applied successfully, False otherwise
    """
    try:
        # Create backup of the file
        backup_path = f"{file_path}.mock_fix.bak"
        with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        logger.info(f"Created backup at {backup_path}")
        
        # Determine which fixes need to be applied
        fixes_needed = []
        if not check_results["has_env_vars"]:
            fixes_needed.append("env_vars")
        if not check_results["has_mock_checks"]:
            fixes_needed.append("mock_checks")
        if not check_results["has_detection_logic"]:
            fixes_needed.append("detection_logic")
        if not check_results["has_visual_indicators"]:
            fixes_needed.append("visual_indicators")
        if not check_results["has_color_codes"]:
            fixes_needed.append("color_codes")
        
        # Apply fixes using existing scripts if possible
        if "env_vars" in fixes_needed:
            logger.info(f"Adding environment variables to {file_path}")
            subprocess.run([sys.executable, "add_env_mock_support.py", "--template", file_path, "--no-backup"], 
                           check=True)
        
        if "visual_indicators" in fixes_needed or "color_codes" in fixes_needed:
            logger.info(f"Adding colorized output to {file_path}")
            subprocess.run([sys.executable, "add_colorized_output.py", "--template", file_path, "--no-backup"], 
                           check=True)
        
        if "detection_logic" in fixes_needed:
            logger.info(f"Adding mock detection logic to {file_path}")
            subprocess.run([sys.executable, "add_mock_detection_to_templates.py", "--template", file_path, "--no-backup"], 
                           check=True)
        
        # Apply mock checks fix if needed
        if "mock_checks" in fixes_needed:
            logger.info(f"Fixing mock checks in {file_path}")
            subprocess.run([sys.executable, "fix_mock_detection_errors.py", "--file", file_path], 
                           check=True)
        
        # Verify that fixes were applied correctly
        new_check_results = check_mock_detection(file_path)
        if new_check_results["status"] == "OK":
            logger.info(f"✅ Successfully fixed {file_path}")
            return True
        else:
            logger.error(f"❌ Failed to fix all issues in {file_path}")
            logger.error(f"Remaining issues: {new_check_results['missing']}")
            return False
    
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False

def verify_mock_detection_function(file_path):
    """
    Verify that mock detection works correctly in a file by testing with different environment variables.
    
    Args:
        file_path: Path to the file to verify
        
    Returns:
        dict: Results of the verification
    """
    logger.info(f"Verifying mock detection for {file_path}")
    
    # Define the different environment variable configurations to test
    configs = [
        {"MOCK_TORCH": "False", "MOCK_TRANSFORMERS": "False", "expected": "REAL INFERENCE"},
        {"MOCK_TORCH": "True", "MOCK_TRANSFORMERS": "False", "expected": "MOCK OBJECTS"},
        {"MOCK_TORCH": "False", "MOCK_TRANSFORMERS": "True", "expected": "MOCK OBJECTS"},
        {"MOCK_TORCH": "True", "MOCK_TRANSFORMERS": "True", "expected": "MOCK OBJECTS"}
    ]
    
    verification_results = {
        "file": file_path,
        "status": "OK",
        "configs_tested": 0,
        "configs_passed": 0,
        "configs_failed": 0,
        "details": []
    }
    
    for config in configs:
        # Set up environment for this config
        env = os.environ.copy()
        for key, value in config.items():
            if key != "expected":
                env[key] = value
        
        expected = config["expected"]
        desc = ", ".join([f"{k}={v}" for k, v in config.items() if k != "expected"])
        
        logger.info(f"Testing {file_path} with {desc}")
        
        try:
            # Run the test file
            result = subprocess.run(
                [sys.executable, file_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            verification_results["configs_tested"] += 1
            
            # Check if execution was successful
            if result.returncode != 0:
                logger.error(f"Test failed with exit code {result.returncode}")
                logger.error(f"Error: {result.stderr}")
                verification_results["configs_failed"] += 1
                verification_results["details"].append({
                    "config": desc,
                    "status": "ERROR",
                    "error": f"Exit code: {result.returncode}"
                })
                continue
            
            # Check for expected output
            if expected in result.stdout:
                logger.info(f"✅ Test passed: Detected '{expected}' as expected")
                verification_results["configs_passed"] += 1
                verification_results["details"].append({
                    "config": desc,
                    "status": "PASS",
                    "expected": expected,
                    "detected": expected
                })
            else:
                logger.error(f"❌ Test failed: Did not detect '{expected}'")
                verification_results["configs_failed"] += 1
                
                # Try to determine what was detected instead
                detected = "UNKNOWN"
                if "REAL INFERENCE" in result.stdout:
                    detected = "REAL INFERENCE"
                elif "MOCK OBJECTS" in result.stdout:
                    detected = "MOCK OBJECTS"
                
                verification_results["details"].append({
                    "config": desc,
                    "status": "FAIL",
                    "expected": expected,
                    "detected": detected
                })
        
        except subprocess.TimeoutExpired:
            logger.error(f"Test timed out after 30 seconds")
            verification_results["configs_failed"] += 1
            verification_results["details"].append({
                "config": desc,
                "status": "TIMEOUT",
                "error": "Test timed out after 30 seconds"
            })
        except Exception as e:
            logger.error(f"Error running test: {e}")
            verification_results["configs_failed"] += 1
            verification_results["details"].append({
                "config": desc,
                "status": "ERROR",
                "error": str(e)
            })
    
    # Update overall status
    if verification_results["configs_failed"] > 0:
        verification_results["status"] = "FAIL"
    
    return verification_results

def process_file(file_path, args):
    """
    Process a single file - check, fix, and verify as needed.
    
    Args:
        file_path: Path to the file to process
        args: Command-line arguments
        
    Returns:
        dict: Results of processing the file
    """
    results = {
        "file": file_path,
        "check": None,
        "fix": None,
        "verify": None
    }
    
    # Step 1: Check mock detection
    logger.info(f"Checking mock detection in {file_path}")
    check_results = check_mock_detection(file_path)
    results["check"] = check_results
    
    # Step 2: Fix if needed and requested
    if check_results["status"] == "NEEDS_FIX" and args.fix:
        logger.info(f"Applying fixes to {file_path}")
        fix_success = fix_mock_detection(file_path, check_results)
        results["fix"] = {"status": "SUCCESS" if fix_success else "FAIL"}
    
    # Step 3: Verify if requested
    if args.verify:
        logger.info(f"Verifying mock detection in {file_path}")
        verification_results = verify_mock_detection_function(file_path)
        results["verify"] = verification_results
    
    return results

def generate_report(results, output_file):
    """
    Generate a comprehensive report of the results.
    
    Args:
        results: List of results from processing files
        output_file: Path to save the report
        
    Returns:
        None
    """
    # Collect statistics
    stats = {
        "total_files": len(results),
        "check": {
            "ok": 0,
            "needs_fix": 0,
            "error": 0
        },
        "fix": {
            "success": 0,
            "fail": 0,
            "not_attempted": 0
        },
        "verify": {
            "pass": 0,
            "fail": 0,
            "not_verified": 0
        }
    }
    
    for result in results:
        # Check stats
        if result["check"]["status"] == "OK":
            stats["check"]["ok"] += 1
        elif result["check"]["status"] == "NEEDS_FIX":
            stats["check"]["needs_fix"] += 1
        else:
            stats["check"]["error"] += 1
        
        # Fix stats
        if result["fix"] is None:
            stats["fix"]["not_attempted"] += 1
        elif result["fix"]["status"] == "SUCCESS":
            stats["fix"]["success"] += 1
        else:
            stats["fix"]["fail"] += 1
        
        # Verify stats
        if result["verify"] is None:
            stats["verify"]["not_verified"] += 1
        elif result["verify"]["status"] == "OK":
            stats["verify"]["pass"] += 1
        else:
            stats["verify"]["fail"] += 1
    
    # Format the report
    report = f"""
=============================================================
       MOCK DETECTION VERIFICATION REPORT
=============================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total files processed: {stats['total_files']}

=== CHECK RESULTS ===
✅ OK: {stats['check']['ok']} files
⚠️ Needs Fix: {stats['check']['needs_fix']} files
❌ Error: {stats['check']['error']} files

=== FIX RESULTS ===
✅ Successfully Fixed: {stats['fix']['success']} files
❌ Failed to Fix: {stats['fix']['fail']} files
⏭️ Not Attempted: {stats['fix']['not_attempted']} files

=== VERIFICATION RESULTS ===
✅ Passed: {stats['verify']['pass']} files
❌ Failed: {stats['verify']['fail']} files
⏭️ Not Verified: {stats['verify']['not_verified']} files

=============================================================
       DETAILED RESULTS
=============================================================
"""
    
    # Add detailed results for each file
    for result in results:
        file_name = os.path.basename(result["file"])
        report += f"\n--- {file_name} ---\n"
        
        # Check details
        report += "CHECK: "
        if result["check"]["status"] == "OK":
            report += "✅ OK\n"
        elif result["check"]["status"] == "NEEDS_FIX":
            report += f"⚠️ Needs Fix - Missing {len(result['check']['missing'])} components\n"
            for missing in result["check"]["missing"]:
                report += f"  - {missing}\n"
        else:
            report += f"❌ Error: {result['check'].get('error', 'Unknown error')}\n"
        
        # Fix details
        report += "FIX: "
        if result["fix"] is None:
            report += "⏭️ Not Attempted\n"
        elif result["fix"]["status"] == "SUCCESS":
            report += "✅ Success\n"
        else:
            report += "❌ Failed\n"
        
        # Verify details
        report += "VERIFY: "
        if result["verify"] is None:
            report += "⏭️ Not Verified\n"
        else:
            if result["verify"]["status"] == "OK":
                report += f"✅ Passed {result['verify']['configs_passed']}/{result['verify']['configs_tested']} configurations\n"
            else:
                report += f"❌ Failed {result['verify']['configs_failed']}/{result['verify']['configs_tested']} configurations\n"
                
                # Add details for failed configurations
                for detail in result["verify"]["details"]:
                    if detail["status"] != "PASS":
                        report += f"  - Config: {detail['config']}\n"
                        report += f"    Status: {detail['status']}\n"
                        if "expected" in detail:
                            report += f"    Expected: {detail['expected']}, Detected: {detail['detected']}\n"
                        if "error" in detail:
                            report += f"    Error: {detail['error']}\n"
    
    # Write the report to file
    with open(output_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Report written to {output_file}")
    
    # Print summary to console
    print(f"\n{GREEN}=== MOCK DETECTION VERIFICATION SUMMARY ==={RESET}")
    print(f"Total files processed: {stats['total_files']}")
    print(f"✅ Files with correct mock detection: {stats['check']['ok']}")
    print(f"⚠️ Files needing fixes: {stats['check']['needs_fix']}")
    print(f"❌ Files with errors: {stats['check']['error']}")
    
    if stats["fix"]["not_attempted"] < stats["total_files"]:
        print(f"\n{GREEN}=== FIX RESULTS ==={RESET}")
        print(f"✅ Successfully fixed: {stats['fix']['success']}")
        print(f"❌ Failed to fix: {stats['fix']['fail']}")
    
    if stats["verify"]["not_verified"] < stats["total_files"]:
        print(f"\n{GREEN}=== VERIFICATION RESULTS ==={RESET}")
        print(f"✅ Passed verification: {stats['verify']['pass']}")
        print(f"❌ Failed verification: {stats['verify']['fail']}")
    
    print(f"\nDetailed report available at: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Verify and fix mock detection in all HuggingFace test files")
    parser.add_argument("--check-only", action="store_true", help="Only check files without fixing or verifying")
    parser.add_argument("--fix", action="store_true", help="Fix files with missing mock detection")
    parser.add_argument("--verify", action="store_true", help="Verify mock detection with different environment variables")
    parser.add_argument("--dir", type=str, default="fixed_tests", help="Directory containing test files")
    parser.add_argument("--file", type=str, help="Process a specific file instead of all files")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--output", type=str, help="Output file for the report")
    
    args = parser.parse_args()
    
    # Set default flags if none specified
    if not (args.check_only or args.fix or args.verify):
        args.check_only = True
    
    # Determine output file
    if not args.output:
        args.output = f"mock_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Find test files
    if args.file:
        if os.path.exists(args.file):
            files = [args.file]
        else:
            logger.error(f"File not found: {args.file}")
            return 1
    else:
        if not os.path.exists(args.dir):
            logger.error(f"Directory not found: {args.dir}")
            return 1
        
        files = []
        for root, _, filenames in os.walk(args.dir):
            for filename in filenames:
                if filename.startswith('test_hf_') and filename.endswith('.py'):
                    files.append(os.path.join(root, filename))
    
    logger.info(f"Found {len(files)} test files to process")
    
    # Process files in parallel
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create a list to hold the future => file mapping
        future_to_file = {executor.submit(process_file, file_path, args): file_path for file_path in files}
        
        # Process as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                all_results.append(result)
                
                # Print progress indicator
                status = "🟢"  # Default - OK
                if result["check"]["status"] == "NEEDS_FIX":
                    status = "🟡"  # Needs fix
                    if result["fix"] and result["fix"]["status"] == "SUCCESS":
                        status = "🔵"  # Fixed
                elif result["check"]["status"] == "ERROR":
                    status = "🔴"  # Error
                
                print(f"{status} {os.path.basename(file_path)}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                all_results.append({
                    "file": file_path,
                    "check": {"status": "ERROR", "error": str(e)},
                    "fix": None,
                    "verify": None
                })
    
    # Generate report
    generate_report(all_results, args.output)
    
    # Determine exit code
    success = True
    for result in all_results:
        if result["check"]["status"] != "OK":
            if not (args.fix and result["fix"] and result["fix"]["status"] == "SUCCESS"):
                success = False
                break
        
        if args.verify and result["verify"] and result["verify"]["status"] != "OK":
            success = False
            break
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())