#!/usr/bin/env python3
"""
Direct Check for July 2025 Enhancements in Resource Pool Bridge Integration

This script examines the ResourcePoolBridgeIntegrationEnhanced implementation file
to verify that all July 2025 enhancements are properly implemented.
"""

import os
import sys
import re
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_implementation_file():
    """Check the implementation file for July 2025 enhancements"""
    # Path to implementation file
    implementation_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fixed_web_platform",
        "resource_pool_bridge_integration_enhanced.py"
    )
    
    if not os.path.exists(implementation_path):
        logger.error(f"Implementation file not found at {implementation_path}")
        return False, {}
    
    # Read the file
    with open(implementation_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check for key methods
    key_methods = [
        {"name": "def get_metrics", "enhancement": "Performance metrics retrieval"},
        {"name": "def get_health_status", "enhancement": "Health monitoring with circuit breaker"},
        {"name": "def get_performance_report", "enhancement": "Comprehensive performance analysis"},
        {"name": "def detect_performance_regressions", "enhancement": "Performance regression detection"},
        {"name": "def get_browser_recommendations", "enhancement": "Browser recommendations based on performance"}
    ]
    
    # Check for key components
    key_components = [
        {"name": "CircuitBreaker", "enhancement": "Circuit breaker pattern"},
        {"name": "BrowserCircuitBreakerManager", "enhancement": "Circuit breaker management"},
        {"name": "PerformanceTrendAnalyzer", "enhancement": "Performance trend analysis"},
        {"name": "ConnectionPoolManager", "enhancement": "Advanced connection pooling"},
        {"name": "TensorSharingManager", "enhancement": "Cross-model tensor sharing"},
        {"name": "UltraLowPrecisionManager", "enhancement": "Ultra-low precision support"},
        {"name": "BrowserPerformanceHistory", "enhancement": "Browser performance history"}
    ]
    
    # Check for July 2025 enhancements
    july_2025_enhancements = [
        {"text": "# July 2025 enhancements", "category": "Documentation"},
        {"text": "Enhanced error recovery", "category": "Error Recovery"},
        {"text": "Performance history tracking", "category": "Performance Analysis"},
        {"text": "Performance trend analysis", "category": "Performance Analysis"},
        {"text": "Circuit breaker pattern", "category": "Fault Tolerance"},
        {"text": "Regression detection", "category": "Performance Analysis"},
        {"text": "Browser-specific optimizations", "category": "Optimization"}
    ]
    
    # Results
    results = {
        "file_exists": True,
        "file_size_bytes": len(content),
        "line_count": len(lines),
        "methods": [],
        "components": [],
        "enhancements": [],
        "metrics": {}
    }
    
    # Check methods
    for method in key_methods:
        pattern = method["name"]
        matches = [line for line in lines if pattern in line]
        method_found = len(matches) > 0
        
        results["methods"].append({
            "name": method["name"],
            "enhancement": method["enhancement"],
            "found": method_found,
            "line_number": next((i+1 for i, line in enumerate(lines) if pattern in line), None)
        })
    
    # Check components
    for component in key_components:
        pattern = component["name"]
        matches = [line for line in lines if pattern in line and not line.strip().startswith('#')]
        component_found = len(matches) > 0
        
        results["components"].append({
            "name": component["name"],
            "enhancement": component["enhancement"],
            "found": component_found,
            "count": len(matches),
            "line_number": next((i+1 for i, line in enumerate(lines) if pattern in line and not line.strip().startswith('#')), None)
        })
    
    # Check enhancements
    for enhancement in july_2025_enhancements:
        pattern = enhancement["text"]
        matches = [line for line in lines if pattern.lower() in line.lower()]
        enhancement_found = len(matches) > 0
        
        results["enhancements"].append({
            "text": enhancement["text"],
            "category": enhancement["category"],
            "found": enhancement_found,
            "count": len(matches),
            "line_number": next((i+1 for i, line in enumerate(lines) if pattern.lower() in line.lower()), None)
        })
    
    # Check for performance analyzer initialization
    performance_analyzer_init = re.search(r'self\.performance_analyzer\s*=\s*PerformanceTrendAnalyzer', content) is not None
    
    # Check for circuit breaker manager initialization
    circuit_breaker_init = re.search(r'self\.circuit_breaker_manager\s*=\s*BrowserCircuitBreakerManager', content) is not None
    
    # Metrics
    results["metrics"] = {
        "methods_found": sum(1 for m in results["methods"] if m["found"]),
        "methods_total": len(results["methods"]),
        "components_found": sum(1 for c in results["components"] if c["found"]),
        "components_total": len(results["components"]),
        "enhancements_found": sum(1 for e in results["enhancements"] if e["found"]),
        "enhancements_total": len(results["enhancements"]),
        "performance_analyzer_init": performance_analyzer_init,
        "circuit_breaker_init": circuit_breaker_init
    }
    
    # Calculate completion percentage
    total_checks = (
        results["metrics"]["methods_total"] + 
        results["metrics"]["components_total"] + 
        results["metrics"]["enhancements_total"] + 
        (1 if performance_analyzer_init else 0) + 
        (1 if circuit_breaker_init else 0)
    )
    
    found_checks = (
        results["metrics"]["methods_found"] + 
        results["metrics"]["components_found"] + 
        results["metrics"]["enhancements_found"] + 
        (1 if performance_analyzer_init else 0) + 
        (1 if circuit_breaker_init else 0)
    )
    
    results["metrics"]["completion_percentage"] = (found_checks / total_checks) * 100 if total_checks > 0 else 0
    
    # Determine implementation status
    if results["metrics"]["completion_percentage"] >= 95:
        implementation_status = "COMPLETE"
    elif results["metrics"]["completion_percentage"] >= 80:
        implementation_status = "MOSTLY COMPLETE"
    elif results["metrics"]["completion_percentage"] >= 50:
        implementation_status = "PARTIALLY COMPLETE"
    else:
        implementation_status = "INCOMPLETE"
    
    results["implementation_status"] = implementation_status
    
    return results["metrics"]["completion_percentage"] >= 95, results

def check_claude_md_file():
    """Check if CLAUDE.md has been updated to reflect the completion of the July 2025 enhancements"""
    # Path to CLAUDE.md
    claude_md_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "CLAUDE.md"
    )
    
    if not os.path.exists(claude_md_path):
        logger.error(f"CLAUDE.md file not found at {claude_md_path}")
        return False, {}
    
    # Read the file
    with open(claude_md_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Results
    results = {
        "file_exists": True,
        "file_size_bytes": len(content),
        "line_count": len(lines),
        "checks": []
    }
    
    # Check if the WebGPU/WebNN Resource Pool Integration is marked as 100% complete
    check_100_percent = {
        "description": "WebGPU/WebNN Resource Pool Integration marked as 100% complete",
        "passed": False,
        "evidence": None
    }
    
    for i, line in enumerate(lines):
        # Look for WebGPU/WebNN Resource Pool Integration 100% complete
        if "WebGPU/WebNN Resource Pool Integration" in line and "100%" in line:
            check_100_percent["passed"] = True
            check_100_percent["evidence"] = line
            break
    
    results["checks"].append(check_100_percent)
    
    # Check if July 2025 features are mentioned
    july_2025_features = [
        "Enhanced error recovery",
        "Performance history tracking",
        "Circuit breaker pattern",
        "Regression detection",
        "Performance trend analysis"
    ]
    
    for feature in july_2025_features:
        feature_check = {
            "description": f"Feature '{feature}' mentioned in CLAUDE.md",
            "passed": False,
            "evidence": None
        }
        
        for i, line in enumerate(lines):
            if feature.lower() in line.lower():
                feature_check["passed"] = True
                feature_check["evidence"] = line
                break
        
        results["checks"].append(feature_check)
    
    # Check if completion date is mentioned
    check_completion_date = {
        "description": "Completion date for WebGPU/WebNN Resource Pool Integration mentioned",
        "passed": False,
        "evidence": None
    }
    
    for i, line in enumerate(lines):
        if "COMPLETED: July 15, 2025" in line or "COMPLETED: July 2025" in line:
            check_completion_date["passed"] = True
            check_completion_date["evidence"] = line
            break
    
    results["checks"].append(check_completion_date)
    
    # Check if Current Implementation Priorities have been updated
    check_priorities_updated = {
        "description": "Current Implementation Priorities updated to remove WebGPU/WebNN Resource Pool Integration",
        "passed": False,
        "evidence": None
    }
    
    # Look for a section that lists current priorities but does not include WebGPU/WebNN Resource Pool
    in_priorities_section = False
    has_resource_pool = False
    
    for i, line in enumerate(lines):
        if "Current Implementation Priorities" in line or "## Current Implementation Priorities" in line:
            in_priorities_section = True
            continue
        
        if in_priorities_section and "WebGPU/WebNN Resource Pool" in line:
            has_resource_pool = True
            break
        
        if in_priorities_section and line.startswith("##"):  # End of priorities section
            break
    
    # If we found the priorities section but it doesn't have resource pool, it's updated
    if in_priorities_section and not has_resource_pool:
        check_priorities_updated["passed"] = True
        check_priorities_updated["evidence"] = "Current priorities section does not include WebGPU/WebNN Resource Pool"
    
    results["checks"].append(check_priorities_updated)
    
    # Calculate metrics
    results["metrics"] = {
        "checks_passed": sum(1 for c in results["checks"] if c["passed"]),
        "checks_total": len(results["checks"]),
        "completion_percentage": (sum(1 for c in results["checks"] if c["passed"]) / len(results["checks"])) * 100 if results["checks"] else 0
    }
    
    # Determine documentation status
    if results["metrics"]["completion_percentage"] >= 90:
        documentation_status = "COMPLETE"
    elif results["metrics"]["completion_percentage"] >= 70:
        documentation_status = "MOSTLY COMPLETE"
    elif results["metrics"]["completion_percentage"] >= 40:
        documentation_status = "PARTIALLY COMPLETE"
    else:
        documentation_status = "INCOMPLETE"
    
    results["documentation_status"] = documentation_status
    
    return results["metrics"]["completion_percentage"] >= 90, results

def check_test_coverage():
    """Check if there are tests for the July 2025 enhancements"""
    # Path to test file
    test_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_resource_pool_enhanced.py"
    )
    
    # Path to integration test file
    integration_test_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_web_resource_pool_integration.py"
    )
    
    # Results
    results = {
        "test_file_exists": os.path.exists(test_path),
        "integration_test_file_exists": os.path.exists(integration_test_path),
        "tests": []
    }
    
    if results["test_file_exists"]:
        # Read the test file
        with open(test_path, 'r') as f:
            test_content = f.read()
            test_lines = test_content.split('\n')
        
        results["test_file_size_bytes"] = len(test_content)
        results["test_file_line_count"] = len(test_lines)
    else:
        logger.warning(f"Test file not found at {test_path}")
        results["test_file_size_bytes"] = 0
        results["test_file_line_count"] = 0
    
    if results["integration_test_file_exists"]:
        # Read the integration test file
        with open(integration_test_path, 'r') as f:
            integration_test_content = f.read()
            integration_test_lines = integration_test_content.split('\n')
        
        results["integration_test_file_size_bytes"] = len(integration_test_content)
        results["integration_test_file_line_count"] = len(integration_test_lines)
    else:
        logger.warning(f"Integration test file not found at {integration_test_path}")
        results["integration_test_file_size_bytes"] = 0
        results["integration_test_file_line_count"] = 0
    
    # Key test cases to look for
    test_cases = [
        {"name": "test_performance_trend_analysis", "enhancement": "Performance trend analysis"},
        {"name": "test_circuit_breaker", "enhancement": "Circuit breaker pattern"},
        {"name": "test_browser_selection", "enhancement": "Browser-specific optimizations"},
        {"name": "test_error_recovery", "enhancement": "Enhanced error recovery"},
        {"name": "test_performance_regression_detection", "enhancement": "Regression detection"}
    ]
    
    # Check for each test case
    for test_case in test_cases:
        test_result = {
            "name": test_case["name"],
            "enhancement": test_case["enhancement"],
            "found_in_test_file": False,
            "found_in_integration_test": False
        }
        
        # Check in test file
        if results["test_file_exists"]:
            for i, line in enumerate(test_lines):
                if f"def {test_case['name']}" in line:
                    test_result["found_in_test_file"] = True
                    test_result["test_file_line_number"] = i + 1
                    break
        
        # Check in integration test file
        if results["integration_test_file_exists"]:
            for i, line in enumerate(integration_test_lines):
                if f"def {test_case['name']}" in line or f"test_{test_case['enhancement'].lower().replace(' ', '_')}" in line:
                    test_result["found_in_integration_test"] = True
                    test_result["integration_test_line_number"] = i + 1
                    break
        
        results["tests"].append(test_result)
    
    # Calculate metrics
    results["metrics"] = {
        "tests_in_test_file": sum(1 for t in results["tests"] if t["found_in_test_file"]),
        "tests_in_integration_test": sum(1 for t in results["tests"] if t["found_in_integration_test"]),
        "tests_total": len(results["tests"]),
        "test_coverage_percentage": (sum(1 for t in results["tests"] if t["found_in_test_file"] or t["found_in_integration_test"]) / len(results["tests"])) * 100 if results["tests"] else 0
    }
    
    # Determine test coverage status
    if results["metrics"]["test_coverage_percentage"] >= 80:
        test_coverage_status = "COMPLETE"
    elif results["metrics"]["test_coverage_percentage"] >= 60:
        test_coverage_status = "MOSTLY COMPLETE"
    elif results["metrics"]["test_coverage_percentage"] >= 40:
        test_coverage_status = "PARTIALLY COMPLETE"
    else:
        test_coverage_status = "INCOMPLETE"
    
    results["test_coverage_status"] = test_coverage_status
    
    return results["metrics"]["test_coverage_percentage"] >= 80, results

def check_completion_documentation():
    """Check if there's a completion report for the July 2025 enhancements"""
    # Path to completion report
    completion_report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "WEB_RESOURCE_POOL_JULY2025_COMPLETION.md"
    )
    
    # Path to README
    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "WEB_RESOURCE_POOL_README.md"
    )
    
    # Results
    results = {
        "completion_report_exists": os.path.exists(completion_report_path),
        "readme_exists": os.path.exists(readme_path),
        "checks": []
    }
    
    # If completion report exists, check its content
    if results["completion_report_exists"]:
        with open(completion_report_path, 'r') as f:
            completion_content = f.read()
            completion_lines = completion_content.split('\n')
        
        results["completion_report_size_bytes"] = len(completion_content)
        results["completion_report_line_count"] = len(completion_lines)
        
        # Check if report mentions all enhancements
        enhancements = [
            "Enhanced error recovery",
            "Performance history tracking",
            "Performance trend analysis",
            "Circuit breaker pattern",
            "Regression detection"
        ]
        
        for enhancement in enhancements:
            check = {
                "description": f"Enhancement '{enhancement}' mentioned in completion report",
                "passed": False,
                "evidence": None
            }
            
            for i, line in enumerate(completion_lines):
                if enhancement.lower() in line.lower():
                    check["passed"] = True
                    check["evidence"] = line
                    break
            
            results["checks"].append(check)
        
        # Check if the report mentions performance improvements
        performance_check = {
            "description": "Performance improvements mentioned in completion report",
            "passed": False,
            "evidence": None
        }
        
        for i, line in enumerate(completion_lines):
            if "performance" in line.lower() and any(x in line.lower() for x in ["improvement", "increase", "better", "faster", "reduction"]):
                performance_check["passed"] = True
                performance_check["evidence"] = line
                break
        
        results["checks"].append(performance_check)
    else:
        logger.warning(f"Completion report not found at {completion_report_path}")
    
    # If README exists, check its content
    if results["readme_exists"]:
        with open(readme_path, 'r') as f:
            readme_content = f.read()
            readme_lines = readme_content.split('\n')
        
        results["readme_size_bytes"] = len(readme_content)
        results["readme_line_count"] = len(readme_lines)
        
        # Check if README mentions the WebGPU/WebNN Resource Pool Integration is complete
        readme_check = {
            "description": "README mentions WebGPU/WebNN Resource Pool Integration is complete",
            "passed": False,
            "evidence": None
        }
        
        for i, line in enumerate(readme_lines):
            if "WebGPU/WebNN Resource Pool" in line.lower() and "complete" in line.lower():
                readme_check["passed"] = True
                readme_check["evidence"] = line
                break
        
        results["checks"].append(readme_check)
    else:
        logger.warning(f"README not found at {readme_path}")
    
    # Calculate metrics
    results["metrics"] = {
        "checks_passed": sum(1 for c in results["checks"] if c["passed"]),
        "checks_total": len(results["checks"]),
        "completion_percentage": (sum(1 for c in results["checks"] if c["passed"]) / len(results["checks"])) * 100 if results["checks"] else 0
    }
    
    # Determine documentation status
    if results["metrics"]["completion_percentage"] >= 80:
        documentation_status = "COMPLETE"
    elif results["metrics"]["completion_percentage"] >= 60:
        documentation_status = "MOSTLY COMPLETE"
    elif results["metrics"]["completion_percentage"] >= 40:
        documentation_status = "PARTIALLY COMPLETE"
    else:
        documentation_status = "INCOMPLETE"
    
    results["documentation_status"] = documentation_status
    
    return results["metrics"]["completion_percentage"] >= 80, results

def main():
    """Main entry point"""
    logger.info("Checking July 2025 enhancements in ResourcePoolBridgeIntegrationEnhanced")
    
    # Check implementation file
    implementation_success, implementation_results = check_implementation_file()
    
    # Check CLAUDE.md
    claude_md_success, claude_md_results = check_claude_md_file()
    
    # Check test coverage
    test_coverage_success, test_coverage_results = check_test_coverage()
    
    # Check completion documentation
    completion_doc_success, completion_doc_results = check_completion_documentation()
    
    # Combine results
    results = {
        "timestamp": datetime.now().isoformat(),
        "implementation_results": implementation_results,
        "claude_md_results": claude_md_results,
        "test_coverage_results": test_coverage_results,
        "completion_doc_results": completion_doc_results,
        "summary": {
            "implementation_success": implementation_success,
            "claude_md_success": claude_md_success,
            "test_coverage_success": test_coverage_success,
            "completion_doc_success": completion_doc_success,
            "overall_success": implementation_success and claude_md_success and test_coverage_success and completion_doc_success
        }
    }
    
    # Calculate overall score
    checks_passed = (
        implementation_success + 
        claude_md_success + 
        test_coverage_success + 
        completion_doc_success
    )
    
    overall_percentage = (checks_passed / 4) * 100
    
    results["summary"]["overall_percentage"] = overall_percentage
    
    # Determine overall status
    if overall_percentage >= 95:
        overall_status = "COMPLETE"
    elif overall_percentage >= 75:
        overall_status = "MOSTLY COMPLETE"
    elif overall_percentage >= 50:
        overall_status = "PARTIALLY COMPLETE"
    else:
        overall_status = "INCOMPLETE"
    
    results["summary"]["overall_status"] = overall_status
    
    # Print summary
    logger.info(f"Implementation: {implementation_results['implementation_status']} ({implementation_results['metrics']['completion_percentage']:.1f}%)")
    logger.info(f"CLAUDE.md: {claude_md_results['documentation_status']} ({claude_md_results['metrics']['completion_percentage']:.1f}%)")
    logger.info(f"Test Coverage: {test_coverage_results['test_coverage_status']} ({test_coverage_results['metrics']['test_coverage_percentage']:.1f}%)")
    logger.info(f"Completion Documentation: {completion_doc_results['documentation_status']} ({completion_doc_results['metrics']['completion_percentage']:.1f}%)")
    logger.info(f"Overall Status: {overall_status} ({overall_percentage:.1f}%)")
    
    # Save detailed results to file
    with open("july_2025_enhancements_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Detailed results saved to july_2025_enhancements_validation.json")
    
    # Return appropriate exit code
    return 0 if overall_percentage >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())