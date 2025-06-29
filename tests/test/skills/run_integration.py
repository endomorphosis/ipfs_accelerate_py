#!/usr/bin/env python3
"""
Automated integration script for deploying fixed HuggingFace test files.
This script runs the full integration process:
1. Test all fixed files to ensure they work properly
2. Deploy the files to the main project
3. Verify deployed files are working correctly
4. Generate a report of the integration status
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Key models to test functionality
TEST_MODELS = {
    "bert": "bert-base-uncased",
    "gpt2": "gpt2",
    "t5": "t5-small",
    "vit": "google/vit-base-patch16-224"
}

def verify_syntax(file_paths):
    """
    Verify Python syntax for files.
    
    Args:
        file_paths: List of paths to Python files to verify
        
    Returns:
        Dict mapping each file to its verification result (True/False)
    """
    results = {}
    
    for file_path in file_paths:
        if not file_path.endswith('.py'):
            # Skip non-Python files
            results[file_path] = True
            continue
            
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'py_compile', file_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {file_path} - Valid syntax")
                results[file_path] = True
            else:
                logger.error(f"❌ {file_path} - Syntax error: {result.stderr.strip()}")
                results[file_path] = False
        except Exception as e:
            logger.error(f"❌ {file_path} - Error during verification: {e}")
            results[file_path] = False
    
    return results

def test_functionality(test_files, cpu_only=True):
    """
    Test the functionality of the test files.
    
    Args:
        test_files: Dict mapping model families to their test file paths
        cpu_only: Whether to use CPU only for testing
        
    Returns:
        Dict mapping each model family to its test result (True/False)
    """
    results = {}
    
    for family, test_file in test_files.items():
        model_id = TEST_MODELS.get(family, None)
        if not model_id:
            logger.warning(f"⚠️ No test model specified for {family}, using default")
        
        # Prepare command
        cmd = [sys.executable, test_file]
        if cpu_only:
            cmd.append("--cpu-only")
        if model_id:
            cmd.extend(["--model", model_id])
        
        try:
            logger.info(f"Testing functionality: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and "Successfully tested" in result.stdout:
                logger.info(f"✅ {family} - Functionality test passed")
                results[family] = True
            else:
                error_message = result.stderr if result.stderr else result.stdout
                logger.error(f"❌ {family} - Functionality test failed: {error_message}")
                results[family] = False
        except Exception as e:
            logger.error(f"❌ {family} - Error during testing: {e}")
            results[family] = False
    
    return results

def run_integration(args):
    """
    Run the complete integration process.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dict with integration results
    """
    results = {
        "syntax_verification": {},
        "functionality_tests": {},
        "deployment": {},
        "post_deployment_tests": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # 0. Preparation
    fixed_test_dir = os.path.abspath(args.fixed_tests_dir)
    dest_dir = os.path.abspath(args.dest_dir)
    backup_dir = os.path.abspath(args.backup_dir)
    os.makedirs(backup_dir, exist_ok=True)
    
    # Collect files
    fixed_test_files = {}
    for family in TEST_MODELS.keys():
        test_file_path = os.path.join(fixed_test_dir, f"test_hf_{family}.py")
        if os.path.exists(test_file_path):
            fixed_test_files[family] = test_file_path
    
    # 1. Verify syntax of fixed files
    logger.info("\n=== 1. Verifying syntax of fixed files ===")
    syntax_results = verify_syntax(list(fixed_test_files.values()))
    results["syntax_verification"] = syntax_results
    
    all_syntax_valid = all(syntax_results.values())
    if not all_syntax_valid:
        logger.error("❌ Syntax verification failed for some files. Aborting.")
        if not args.force:
            return results
    
    # 2. Test functionality of fixed files
    logger.info("\n=== 2. Testing functionality of fixed files ===")
    functionality_results = test_functionality(fixed_test_files, cpu_only=args.cpu_only)
    results["functionality_tests"] = functionality_results
    
    all_functionality_valid = all(functionality_results.values())
    if not all_functionality_valid:
        logger.error("❌ Functionality tests failed for some files. Aborting.")
        if not args.force:
            return results
    
    # 3. Deploy files
    if args.deploy:
        logger.info("\n=== 3. Deploying files ===")
        
        # Use our existing integration script
        integration_script = os.path.join(os.path.dirname(__file__), "integrate_generator_fixes.py")
        
        cmd = [
            sys.executable,
            integration_script,
            "--all",
            "--dest-dir", dest_dir,
            "--backup-dir", backup_dir
        ]
        
        if args.test_only:
            cmd.append("--test-only")
        
        try:
            logger.info(f"Running integration: {' '.join(cmd)}")
            deployment_result = subprocess.run(cmd, capture_output=True, text=True)
            
            deployment_success = deployment_result.returncode == 0
            results["deployment"] = {
                "success": deployment_success,
                "output": deployment_result.stdout,
                "error": deployment_result.stderr if deployment_result.stderr else None
            }
            
            if deployment_success:
                logger.info("✅ Deployment successful")
            else:
                logger.error(f"❌ Deployment failed: {deployment_result.stderr}")
                if not args.force:
                    return results
        except Exception as e:
            logger.error(f"❌ Error during deployment: {e}")
            results["deployment"] = {"success": False, "error": str(e)}
            if not args.force:
                return results
    
        # 4. Test deployed files
        if not args.test_only and args.deploy:
            logger.info("\n=== 4. Testing deployed files ===")
            
            deployed_test_files = {}
            for family in TEST_MODELS.keys():
                test_file_path = os.path.join(dest_dir, f"test_hf_{family}.py")
                if os.path.exists(test_file_path):
                    deployed_test_files[family] = test_file_path
            
            post_deployment_results = test_functionality(deployed_test_files, cpu_only=args.cpu_only)
            results["post_deployment_tests"] = post_deployment_results
            
            all_post_deployment_valid = all(post_deployment_results.values())
            if not all_post_deployment_valid:
                logger.error("❌ Post-deployment tests failed for some files.")
    
    # Save integration results
    if args.save_results:
        results_file = f"integration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Integration results saved to {results_file}")
    
    # Print summary
    print_summary(results)
    
    return results

def print_summary(results):
    """Print a summary of the integration results."""
    logger.info("\n===== INTEGRATION SUMMARY =====")
    
    # Syntax verification
    syntax_success = sum(1 for r in results["syntax_verification"].values() if r)
    syntax_total = len(results["syntax_verification"])
    logger.info(f"Syntax verification: {syntax_success}/{syntax_total} files passed")
    
    # Functionality tests
    func_success = sum(1 for r in results["functionality_tests"].values() if r)
    func_total = len(results["functionality_tests"])
    logger.info(f"Functionality tests: {func_success}/{func_total} models passed")
    
    # Deployment
    if "deployment" in results and results["deployment"]:
        deployment_success = results["deployment"].get("success", False)
        logger.info(f"Deployment: {'Successful' if deployment_success else 'Failed'}")
    
    # Post-deployment tests
    if "post_deployment_tests" in results and results["post_deployment_tests"]:
        post_success = sum(1 for r in results["post_deployment_tests"].values() if r)
        post_total = len(results["post_deployment_tests"])
        logger.info(f"Post-deployment tests: {post_success}/{post_total} models passed")
    
    # Overall status
    syntax_all_passed = all(results["syntax_verification"].values())
    func_all_passed = all(results["functionality_tests"].values())
    deployment_passed = results.get("deployment", {}).get("success", True)
    post_all_passed = all(results.get("post_deployment_tests", {}).values())
    
    overall_success = syntax_all_passed and func_all_passed and deployment_passed and post_all_passed
    
    logger.info("\nOverall integration status: " + 
                ("✅ SUCCESS" if overall_success else "❌ FAILED"))
    
    # Next steps
    logger.info("\nNext steps:")
    if overall_success:
        logger.info("1. Run cleanup_integration.py to remove temporary files")
        logger.info("2. Update documentation to reflect the new approach")
        logger.info("3. Implement CI/CD integration for syntax validation")
    else:
        logger.info("1. Fix any issues reported in the log")
        logger.info("2. Re-run the integration with the --force flag if needed")
        if results.get("deployment", {}).get("success", False):
            logger.info("3. Use the restore script to revert changes if necessary")

def main():
    parser = argparse.ArgumentParser(description="Automate HuggingFace test file integration process")
    
    # Basic options
    parser.add_argument("--fixed-tests-dir", type=str, default="fixed_tests",
                      help="Directory containing fixed test files (default: fixed_tests)")
    parser.add_argument("--dest-dir", type=str, default="..",
                      help="Destination directory for integration (default: parent directory)")
    parser.add_argument("--backup-dir", type=str, default="backups",
                      help="Directory for backups (default: backups)")
    
    # Test options
    parser.add_argument("--cpu-only", action="store_true",
                      help="Use CPU only for testing (default: true)")
    parser.add_argument("--test-only", action="store_true",
                      help="Test integration without copying files")
    
    # Integration options
    parser.add_argument("--deploy", action="store_true",
                      help="Actually deploy the files (default: false)")
    parser.add_argument("--force", action="store_true",
                      help="Force integration even if tests fail")
    parser.add_argument("--save-results", action="store_true",
                      help="Save integration results to JSON file")
    
    args = parser.parse_args()
    
    # Run integration
    run_integration(args)

if __name__ == "__main__":
    main()