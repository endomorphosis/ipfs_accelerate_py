#!/usr/bin/env python3
"""
Check the implementation status of the unified web framework.

This script checks which components of the unified web framework have been
implemented and produces a comprehensive status report.

Usage:
    python check_unified_framework_status.py
"""

import os
import sys
import json
import time
import importlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("check_unified_framework")

# Expected components
EXPECTED_COMPONENTS = {
    # Core components
    "UnifiedWebPlatform": {"file": "__init__.py", "status": "Partial"},
    "ConfigurationManager": {"file": "configuration_manager.py", "status": "Complete"},
    "ErrorHandler": {"file": "error_handling.py", "status": "Complete"},
    "PlatformDetector": {"file": "platform_detector.py", "status": "Complete"},
    "ResultFormatter": {"file": "result_formatter.py", "status": "Complete"},
    "ModelShardingManager": {"file": "model_sharding.py", "status": "Partial"},
    
    # Configuration sub-components
    "ConfigValidationRule": {"file": "configuration_manager.py", "status": "Complete"},
    "BrowserProfile": {"file": "configuration_manager.py", "status": "Complete"},
    
    # Error handling sub-components
    "WebPlatformError": {"file": "error_handling.py", "status": "Complete"},
    "ConfigurationError": {"file": "error_handling.py", "status": "Complete"},
    "BrowserCompatibilityError": {"file": "error_handling.py", "status": "Complete"},
    "RuntimeError": {"file": "error_handling.py", "status": "Complete"},
    
    # Utility functions
    "format_inference_result": {"file": "result_formatter.py", "status": "Complete"},
    "format_error_response": {"file": "result_formatter.py", "status": "Complete"},
    "get_browser_capabilities": {"file": "platform_detector.py", "status": "Complete"},
    "get_hardware_capabilities": {"file": "platform_detector.py", "status": "Complete"},
    "create_platform_profile": {"file": "platform_detector.py", "status": "Complete"},
}

def check_component_existence() -> Dict[str, Any]:
    """
    Check which unified framework components exist.
    
    Returns:
        Dictionary with component existence status
    """
    # Initialize result
    result = {
        "exists": {},
        "missing": {},
        "importable": {},
        "not_importable": {}
    }
    
    # Check if unified_framework directory exists
    framework_dir = Path("fixed_web_platform/unified_framework")
    if not framework_dir.exists():
        logger.error(f"Unified framework directory {framework_dir} does not exist")
        result["directory_exists"] = False
        return result
    
    result["directory_exists"] = True
    
    # Check for each expected file
    expected_files = set(comp["file"] for comp in EXPECTED_COMPONENTS.values())
    existing_files = set()
    missing_files = set()
    
    for file in expected_files:
        file_path = framework_dir / file
        if file_path.exists():
            existing_files.add(file)
        else:
            missing_files.add(file)
    
    result["existing_files"] = list(existing_files)
    result["missing_files"] = list(missing_files)
    
    # Try to import each component
    for component, info in EXPECTED_COMPONENTS.items():
        # Skip import checks for utility functions
        if component.islower() and '_' in component:
            continue
            
        module_name = f"fixed_web_platform.unified_framework"
        if info["file"] != "__init__.py":
            module_name += f".{info['file'][:-3]}"
            
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, component):
                result["importable"][component] = {
                    "module": module_name,
                    "status": info["status"]
                }
            else:
                result["not_importable"][component] = {
                    "module": module_name,
                    "reason": "Component not found in module",
                    "status": info["status"]
                }
        except ImportError as e:
            result["not_importable"][component] = {
                "module": module_name,
                "reason": str(e),
                "status": info["status"]
            }
    
    return result

def generate_status_report(status: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive status report.
    
    Args:
        status: Component existence status
        
    Returns:
        Status report dictionary
    """
    # Initialize report
    report = {
        "timestamp": time.time(),
        "implementation_status": {},
        "summary": {},
        "components_by_file": {}
    }
    
    # Collect implementation status
    if not status.get("directory_exists", False):
        report["implementation_status"]["unified_framework"] = "Missing"
    else:
        implemented = len(status.get("importable", {}))
        total = len(EXPECTED_COMPONENTS)
        missing_files = len(status.get("missing_files", []))
        
        report["implementation_status"]["unified_framework"] = {
            "status": "Partial" if missing_files > 0 else "Complete",
            "implemented_components": implemented,
            "total_components": total,
            "completion_percentage": round(implemented / total * 100, 1),
            "missing_files": status.get("missing_files", [])
        }
        
        # Check specific components
        for component, info in EXPECTED_COMPONENTS.items():
            if component in status.get("importable", {}):
                report["implementation_status"][component] = info["status"]
            else:
                report["implementation_status"][component] = "Missing"
    
    # Generate summary
    complete_count = sum(1 for status in report["implementation_status"].values() 
                        if status == "Complete")
    partial_count = sum(1 for status in report["implementation_status"].values() 
                       if status == "Partial")
    missing_count = sum(1 for status in report["implementation_status"].values() 
                       if status == "Missing")
    total_count = len(report["implementation_status"])
    
    report["summary"] = {
        "total_components": total_count,
        "complete_components": complete_count,
        "partial_components": partial_count,
        "missing_components": missing_count,
        "completion_percentage": round((complete_count + partial_count * 0.5) / total_count * 100, 1)
    }
    
    # Organize by file
    components_by_file = {}
    for component, info in EXPECTED_COMPONENTS.items():
        file = info["file"]
        if file not in components_by_file:
            components_by_file[file] = []
            
        components_by_file[file].append({
            "name": component,
            "status": report["implementation_status"].get(component, "Unknown")
        })
    
    report["components_by_file"] = components_by_file
    
    return report

def print_status_report(report: Dict[str, Any]) -> None:
    """
    Print status report in a user-friendly format.
    
    Args:
        report: Status report to print
    """
    # Print header
    print("\n" + "=" * 80)
    print("Unified Web Framework Implementation Status Report")
    print("=" * 80)
    
    # Print summary
    summary = report["summary"]
    print(f"\nSummary:")
    print(f"- Total components: {summary['total_components']}")
    print(f"- Complete components: {summary['complete_components']}")
    print(f"- Partial components: {summary['partial_components']}")
    print(f"- Missing components: {summary['missing_components']}")
    print(f"- Overall completion: {summary['completion_percentage']}%")
    
    # Print components by file
    print("\nComponents by file:")
    for file, components in report["components_by_file"].items():
        if not os.path.exists(f"fixed_web_platform/unified_framework/{file}"):
            print(f"\n{file} (MISSING)")
        else:
            print(f"\n{file}")
            
        for component in components:
            status_emoji = "✅" if component["status"] == "Complete" else "🟡" if component["status"] == "Partial" else "❌"
            print(f"  {status_emoji} {component['name']}: {component['status']}")
    
    # Print implementation status
    framework_status = report["implementation_status"].get("unified_framework", {})
    if isinstance(framework_status, dict):
        print(f"\nFramework Status: {framework_status['status']}")
        print(f"- Implemented: {framework_status['implemented_components']}/{framework_status['total_components']} components ({framework_status['completion_percentage']}%)")
        
        if framework_status.get("missing_files"):
            print("- Missing files:")
            for file in framework_status["missing_files"]:
                print(f"  - {file}")
    
    # Print completion information
    print("\nCompletion Information:")
    if summary["completion_percentage"] < 30:
        print("- Status: Initial implementation in progress")
        print("- Next steps: Implement core components")
    elif summary["completion_percentage"] < 60:
        print("- Status: Basic implementation in progress")
        print("- Next steps: Complete core components and implement utility functions")
    elif summary["completion_percentage"] < 90:
        print("- Status: Advanced implementation in progress")
        print("- Next steps: Complete remaining components and finalize integration")
    else:
        print("- Status: Implementation nearly complete")
        print("- Next steps: Finalize and test all components")
    
    print("\n" + "=" * 80)

def save_status_report(report: Dict[str, Any], output_file: str) -> None:
    """
    Save status report to JSON file.
    
    Args:
        report: Status report to save
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Status report saved to {output_file}")

def main():
    """Check unified framework status and generate report."""
    # Check component existence
    status = check_component_existence()
    
    # Generate status report
    report = generate_status_report(status)
    
    # Print report
    print_status_report(report)
    
    # Save report
    save_status_report(report, "unified_framework_status.json")

if __name__ == "__main__":
    main()