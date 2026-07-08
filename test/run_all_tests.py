#!/usr/bin/env python3
"""
Comprehensive Test Runner for IPFS Accelerate Python

This script runs all test suites and provides detailed reporting.
Designed for both development and CI/CD environments.
"""

import sys
import os
import subprocess
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

def run_command(cmd: List[str], description: str) -> Tuple[bool, str, float]:
    """
    Run a command and return success status, output, and execution time.
    """
    print(f"🔄 {description}...")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        end_time = time.time()
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        duration = end_time - start_time
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status} ({duration:.2f}s)")
        
        return success, output, duration
        
    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        print(f"   ⏰ TIMEOUT ({duration:.2f}s)")
        return False, "Command timed out", duration
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"   💥 ERROR ({duration:.2f}s): {e}")
        return False, str(e), duration

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is supported."""
    version = sys.version_info
    supported_versions = [(3, 8), (3, 9), (3, 10), (3, 11), (3, 12)]
    
    current = (version.major, version.minor)
    if current in supported_versions:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (unsupported)"

def check_dependencies() -> Dict[str, bool]:
    """Check if required test dependencies are available."""
    dependencies = {}
    
    # Check pytest
    try:
        import pytest
        dependencies['pytest'] = True
    except ImportError:
        dependencies['pytest'] = False
    
    # Check hardware_detection
    try:
        import hardware_detection
        dependencies['hardware_detection'] = True
    except ImportError:
        dependencies['hardware_detection'] = False
    
    # Check main module
    try:
        import ipfs_accelerate_py
        dependencies['ipfs_accelerate_py'] = True
    except ImportError:
        dependencies['ipfs_accelerate_py'] = False
    
    return dependencies

def run_test_suite() -> Dict[str, Any]:
    """Run all test suites and collect results."""
    
    print("🧪 IPFS Accelerate Python - Comprehensive Test Runner")
    print("=" * 60)
    
    # Environment info
    python_ok, python_version = check_python_version()
    print(f"🐍 Python Version: {python_version}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not python_ok:
        print("❌ Unsupported Python version")
        return {"success": False, "error": "Unsupported Python version"}
    
    # Check dependencies
    print("\n🔍 Checking Dependencies...")
    deps = check_dependencies()
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"   {status} {dep}")
    
    if not all(deps.values()):
        print("⚠️  Some dependencies are missing, tests may fail")
    
    # Test configuration
    test_suites = [
        {
            'name': 'Smoke Tests',
            'script': 'test_smoke_basic.py',
            'description': 'Basic functionality verification',
            'expected_tests': 6,
        },
        {
            'name': 'Comprehensive Tests', 
            'script': 'test_comprehensive.py',
            'description': 'Detailed core functionality with mocking',
            'expected_tests': 16,
        },
        {
            'name': 'Integration Tests',
            'script': 'test_integration.py', 
            'description': 'End-to-end integration workflows',
            'expected_tests': 10,
        }
    ]
    
    results = {
        'start_time': datetime.now().isoformat(),
        'environment': {
            'python_version': python_version,
            'working_directory': os.getcwd(),
            'dependencies': deps,
        },
        'test_suites': {},
        'summary': {
            'total_suites': len(test_suites),
            'passed_suites': 0,
            'failed_suites': 0,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'total_duration': 0.0,
        }
    }
    
    print(f"\n🎯 Running {len(test_suites)} Test Suites...")
    print("=" * 60)
    
    suite_start_time = time.time()
    
    # Run each test suite
    for suite in test_suites:
        print(f"\n📊 {suite['name']}")
        print(f"   {suite['description']}")
        print("-" * 40)
        
        suite_results = {
            'expected_tests': suite['expected_tests'],
            'direct_execution': {},
            'pytest_execution': {},
        }
        
        # Test 1: Direct execution
        success, output, duration = run_command(
            [sys.executable, suite['script']],
            f"Direct execution of {suite['script']}"
        )
        
        suite_results['direct_execution'] = {
            'success': success,
            'duration': duration,
            'output_lines': len(output.split('\n')) if output else 0,
        }
        
        # Test 2: pytest execution
        success_pytest, output_pytest, duration_pytest = run_command(
            [sys.executable, '-m', 'pytest', suite['script'], '-v'],
            f"pytest execution of {suite['script']}"
        )
        
        suite_results['pytest_execution'] = {
            'success': success_pytest,
            'duration': duration_pytest,
            'output_lines': len(output_pytest.split('\n')) if output_pytest else 0,
        }
        
        # Determine suite success (both direct and pytest should pass)
        suite_success = success and success_pytest
        suite_total_duration = duration + duration_pytest
        
        if suite_success:
            results['summary']['passed_suites'] += 1
            results['summary']['passed_tests'] += suite['expected_tests']
            print(f"   🎉 {suite['name']} PASSED (expected {suite['expected_tests']} tests)")
        else:
            results['summary']['failed_suites'] += 1
            results['summary']['failed_tests'] += suite['expected_tests']
            print(f"   💥 {suite['name']} FAILED")
            if not success:
                print(f"      Direct execution failed")
            if not success_pytest:
                print(f"      pytest execution failed")
        
        results['summary']['total_tests'] += suite['expected_tests']
        results['summary']['total_duration'] += suite_total_duration
        results['test_suites'][suite['name']] = suite_results
    
    suite_end_time = time.time()
    results['end_time'] = datetime.now().isoformat()
    results['summary']['total_duration'] = suite_end_time - suite_start_time
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)
    
    summary = results['summary']
    print(f"🐍 Python Version: {python_version}")
    print(f"⏱️  Total Duration: {summary['total_duration']:.2f}s")
    print(f"📊 Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed")
    print(f"🧪 Test Cases: {summary['passed_tests']}/{summary['total_tests']} passed")
    
    if summary['failed_suites'] == 0:
        print("🎉 ALL TESTS PASSED!")
        results['success'] = True
    else:
        print(f"❌ {summary['failed_suites']} test suite(s) failed")
        results['success'] = False
    
    # Performance summary
    avg_duration = summary['total_duration'] / summary['total_suites'] if summary['total_suites'] > 0 else 0
    print(f"⚡ Average suite duration: {avg_duration:.2f}s")
    
    if summary['total_duration'] < 30:
        print("✅ Performance: Excellent (< 30s)")
    elif summary['total_duration'] < 60:
        print("🟡 Performance: Good (< 60s)")
    else:
        print("🟠 Performance: Slow (> 60s)")
    
    return results

def save_results(results: Dict[str, Any], filename: str = "test_results.json"):
    """Save test results to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")
    except Exception as e:
        print(f"\n⚠️  Failed to save results: {e}")

def main():
    """Main test runner function."""
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run the test suite
    results = run_test_suite()
    
    # Save results
    save_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results.get('success', False) else 1)

if __name__ == "__main__":
    main()