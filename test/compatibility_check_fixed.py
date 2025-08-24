#!/usr/bin/env python
"""
Compatibility Check for IPFS Accelerate Python Package

This script checks compatibility between the installed ipfs_accelerate_py
package and the test framework by analyzing the expected vs. actual structure.
Fixed version with proper Python 3.12 syntax.
"""

import os
import sys
import json
import logging
import importlib
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("compatibility_check")

def check_package_structure():
    """Check the structure of the installed package"""
    results = {
        "installed_package": {}
    }
    
    # Check main package
    try:
        import ipfs_accelerate_py
        results["installed_package"]["main_package"] = {
            "status": "Present",
            "version": getattr(ipfs_accelerate_py, '__version__', 'Unknown'),
            "location": ipfs_accelerate_py.__file__ if hasattr(ipfs_accelerate_py, '__file__') else 'Unknown'
        }
        logger.info("✅ Main package ipfs_accelerate_py found")
        
        # Check for main attributes/modules
        expected_attrs = ['backends', 'config', 'ipfs_accelerate', 'load_checkpoint_and_dispatch']
        for attr in expected_attrs:
            if hasattr(ipfs_accelerate_py, attr):
                results["installed_package"][attr] = "Present"
                logger.info(f"✅ Found {attr}")
            else:
                results["installed_package"][attr] = "Missing"
                logger.warning(f"⚠️ Missing {attr}")
        
    except ImportError as e:
        results["installed_package"]["main_package"] = {
            "status": "Missing",
            "error": str(e)
        }
        logger.error(f"Failed to import main package: {e}")
    
    return results

def check_test_framework_expectations():
    """Check what the test framework expects"""
    results = {
        "test_framework": {}
    }
    
    # Expected package structure based on tests
    expected_structure = {
        "main_package": "ipfs_accelerate_py",
        "core_modules": ["backends", "config", "ipfs_accelerate"],
        "main_functions": ["load_checkpoint_and_dispatch"],
        "backend_methods": ["docker_tunnel", "marketplace", "start_container", "stop_container"]
    }
    
    results["test_framework"]["expected_structure"] = expected_structure
    logger.info("📋 Test framework expectations recorded")
    
    return results

def check_repo_structure():
    """Check the repository structure"""
    results = {
        "repository": {}
    }
    
    # Try to find the repository root
    current_dir = Path(__file__).resolve().parent
    repo_root = None
    
    # Look for common repository indicators
    for parent in [current_dir] + list(current_dir.parents):
        if any((parent / indicator).exists() for indicator in ['.git', 'setup.py', 'pyproject.toml']):
            repo_root = parent
            break
    
    if repo_root:
        results["repository"]["root"] = str(repo_root)
        results["repository"]["has_setup_py"] = (repo_root / "setup.py").exists()
        results["repository"]["has_pyproject_toml"] = (repo_root / "pyproject.toml").exists()
        results["repository"]["has_requirements"] = (repo_root / "requirements.txt").exists()
        
        # Check for key directories
        key_dirs = ["test", "ipfs_accelerate_py", "examples", "docs"]
        for dir_name in key_dirs:
            results["repository"][f"has_{dir_name}"] = (repo_root / dir_name).exists()
        
        logger.info(f"✅ Repository root found: {repo_root}")
    else:
        results["repository"]["root"] = "Not found"
        logger.warning("⚠️ Could not locate repository root")
    
    return results

def analyze_compatibility(package_results, framework_results, repo_results):
    """Analyze compatibility between components"""
    results = {
        "compatibility_analysis": {}
    }
    
    # Check if main package meets expectations
    if package_results.get("installed_package", {}).get("main_package", {}).get("status") == "Present":
        results["compatibility_analysis"]["main_package"] = "Compatible"
        logger.info("✅ Main package compatibility: OK")
    else:
        results["compatibility_analysis"]["main_package"] = "Incompatible"
        logger.error("❌ Main package compatibility: FAILED")
    
    # Check attribute compatibility
    expected_attrs = framework_results.get("test_framework", {}).get("expected_structure", {}).get("core_modules", [])
    present_attrs = []
    missing_attrs = []
    
    for attr in expected_attrs:
        if package_results.get("installed_package", {}).get(attr) == "Present":
            present_attrs.append(attr)
        else:
            missing_attrs.append(attr)
    
    results["compatibility_analysis"]["attributes"] = {
        "present": present_attrs,
        "missing": missing_attrs,
        "compatibility": "Compatible" if not missing_attrs else "Partial"
    }
    
    if missing_attrs:
        logger.warning(f"⚠️ Missing attributes: {missing_attrs}")
    else:
        logger.info("✅ All expected attributes present")
    
    # Overall compatibility score
    total_checks = len(expected_attrs) + 1  # +1 for main package
    passed_checks = len(present_attrs) + (1 if results["compatibility_analysis"]["main_package"] == "Compatible" else 0)
    compatibility_score = (passed_checks / total_checks) * 100
    
    results["compatibility_analysis"]["overall_score"] = compatibility_score
    results["compatibility_analysis"]["overall_status"] = (
        "Fully Compatible" if compatibility_score == 100 else
        "Mostly Compatible" if compatibility_score >= 75 else
        "Partially Compatible" if compatibility_score >= 50 else
        "Incompatible"
    )
    
    logger.info(f"📊 Overall compatibility score: {compatibility_score:.1f}%")
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Check compatibility between IPFS Accelerate Python package and test framework")
    parser.add_argument("--output", "-o", help="Output file for compatibility results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run checks
    logger.info("Checking installed package structure")
    package_results = check_package_structure()
    
    logger.info("Checking test framework expectations")
    framework_results = check_test_framework_expectations()
    
    logger.info("Checking repository structure")
    repo_results = check_repo_structure()
    
    # Analyze compatibility
    logger.info("Analyzing compatibility")
    compatibility_results = analyze_compatibility(package_results, framework_results, repo_results)
    
    # Combine results
    results = {
        **package_results,
        **framework_results,
        **repo_results,
        **compatibility_results
    }
    
    # Print summary
    print("\n" + "="*60)
    print("COMPATIBILITY CHECK SUMMARY")
    print("="*60)
    
    overall_status = compatibility_results["compatibility_analysis"]["overall_status"]
    overall_score = compatibility_results["compatibility_analysis"]["overall_score"]
    print(f"Overall Status: {overall_status} ({overall_score:.1f}%)")
    
    if package_results.get("installed_package", {}).get("main_package", {}).get("status") == "Present":
        version = package_results["installed_package"]["main_package"].get("version", "Unknown")
        print(f"Package Version: {version}")
    
    missing_attrs = compatibility_results["compatibility_analysis"]["attributes"]["missing"]
    if missing_attrs:
        print(f"Missing Attributes: {', '.join(missing_attrs)}")
    
    print("="*60)
    
    # Save results if output file specified
    if args.output:
        output_path = Path(args.output)
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    # Return appropriate exit code
    return 0 if overall_score >= 75 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)