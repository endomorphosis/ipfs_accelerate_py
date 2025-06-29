#!/usr/bin/env python3
"""
Verify Real WebNN and WebGPU Implementation Status

This script checks if the current implementations of WebNN and WebGPU
are using real browser-based implementations or simulations.

Usage:
    python verify_real_web_implementation.py
"""

import os
import sys
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_implementation_file(file_path):
    """Check if an implementation file is using real browser integration.
    
    Args:
        file_path: Path to implementation file
        
    Returns:
        Tuple of (implementation_type, status, details)
        implementation_type: "webgpu" or "webnn"
        status: "real", "simulation", or "unknown"
        details: Dict with details about the implementation
    """
    if not os.path.exists(file_path):
        return "unknown", "missing", {"error": f"File not found: {file_path}"}
    
    # Determine implementation type from filename
    implementation_type = "unknown"
    if "webgpu" in file_path.lower():
        implementation_type = "webgpu"
    elif "webnn" in file_path.lower():
        implementation_type = "webnn"
    
    # Check if file has real implementation markers
    real_markers = [
        "RealWebGPUConnection",
        "RealWebNNConnection",
        "RealWebGPUImplementation",
        "RealWebNNImplementation",
        "browser_connection",
        "navigator.gpu",
        "navigator.ml",
        "WebGPU is available",
        "WebNN is available",
        "transformers.js"
    ]
    
    simulation_markers = [
        "SimulatedWebGPU",
        "SimulatedWebNN",
        "is_simulation = True",
        "simulation = True",
        "SimulatedBuffer",
        "SimulatedShaderModule",
        "SimulatedComputePipeline",
        "SIMULATED PERFORMANCE",
        "Fake latency",
        "Fake results"
    ]
    
    found_real_markers = []
    found_simulation_markers = []
    
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Check for real implementation markers
        for marker in real_markers:
            if marker in content:
                found_real_markers.append(marker)
        
        # Check for simulation markers
        for marker in simulation_markers:
            if marker in content:
                found_simulation_markers.append(marker)
    
    # Determine status
    status = "unknown"
    if found_real_markers and not found_simulation_markers:
        status = "real"
    elif found_simulation_markers:
        if found_real_markers:
            status = "hybrid"
        else:
            status = "simulation"
    
    # Return details
    details = {
        "file_path": file_path,
        "real_markers": found_real_markers,
        "simulation_markers": found_simulation_markers,
        "lines_of_code": len(content.splitlines())
    }
    
    return implementation_type, status, details

def find_implementation_files():
    """Find all WebNN and WebGPU implementation files.
    
    Returns:
        List of file paths
    """
    implementation_files = []
    
    # Look in fixed_web_platform directory
    web_platform_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "fixed_web_platform"
    if web_platform_dir.exists():
        for file in web_platform_dir.glob("**/*"):
            if file.is_file() and file.suffix == ".py":
                if "webgpu" in file.name.lower() or "webnn" in file.name.lower():
                    implementation_files.append(str(file))
    
    # Look for other implementation files
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    for file in current_dir.glob("**/*web*implementation*.py"):
        if file.is_file():
            implementation_files.append(str(file))
    
    # Look for direct_web_integration.py
    direct_web_path = current_dir / "direct_web_integration.py"
    if direct_web_path.exists():
        implementation_files.append(str(direct_web_path))
    
    # Look for implement_real_webnn_webgpu.py
    real_impl_path = current_dir / "implement_real_webnn_webgpu.py"
    if real_impl_path.exists():
        implementation_files.append(str(real_impl_path))
    
    return sorted(list(set(implementation_files)))

def verify_implementations():
    """Verify all WebNN and WebGPU implementations.
    
    Returns:
        Dict with verification results
    """
    # Find implementation files
    implementation_files = find_implementation_files()
    logger.info(f"Found {len(implementation_files)} implementation files")
    
    # Check each file
    results = {
        "webgpu": {
            "files": [],
            "real_count": 0,
            "simulation_count": 0,
            "hybrid_count": 0,
            "unknown_count": 0,
            "status": "unknown"
        },
        "webnn": {
            "files": [],
            "real_count": 0,
            "simulation_count": 0,
            "hybrid_count": 0,
            "unknown_count": 0,
            "status": "unknown"
        },
        "unknown": {
            "files": [],
            "count": 0
        }
    }
    
    for file_path in implementation_files:
        implementation_type, status, details = check_implementation_file(file_path)
        
        # Add to results
        if implementation_type in ["webgpu", "webnn"]:
            results[implementation_type]["files"].append({
                "file_path": file_path,
                "status": status,
                "details": details
            })
            
            if status == "real":
                results[implementation_type]["real_count"] += 1
            elif status == "simulation":
                results[implementation_type]["simulation_count"] += 1
            elif status == "hybrid":
                results[implementation_type]["hybrid_count"] += 1
            else:
                results[implementation_type]["unknown_count"] += 1
        else:
            results["unknown"]["files"].append({
                "file_path": file_path,
                "status": status,
                "details": details
            })
            results["unknown"]["count"] += 1
    
    # Determine overall status for each implementation type
    for impl_type in ["webgpu", "webnn"]:
        if results[impl_type]["real_count"] > 0 and results[impl_type]["simulation_count"] == 0:
            results[impl_type]["status"] = "real"
        elif results[impl_type]["simulation_count"] > 0 and results[impl_type]["real_count"] == 0:
            results[impl_type]["status"] = "simulation"
        elif results[impl_type]["real_count"] > 0 and results[impl_type]["simulation_count"] > 0:
            results[impl_type]["status"] = "hybrid"
        else:
            results[impl_type]["status"] = "unknown"
    
    return results

def display_verification_results(results):
    """Display verification results.
    
    Args:
        results: Verification results
    """
    # Display header
    print("\n========== WebNN and WebGPU Implementation Verification ==========\n")
    
    # Display WebGPU status
    webgpu_status = results["webgpu"]["status"]
    print(f"WebGPU Implementation Status: {webgpu_status.upper()}")
    print(f"  - Real implementation files: {results['webgpu']['real_count']}")
    print(f"  - Simulation files: {results['webgpu']['simulation_count']}")
    print(f"  - Hybrid files: {results['webgpu']['hybrid_count']}")
    print(f"  - Unknown files: {results['webgpu']['unknown_count']}")
    print()
    
    # Display WebNN status
    webnn_status = results["webnn"]["status"]
    print(f"WebNN Implementation Status: {webnn_status.upper()}")
    print(f"  - Real implementation files: {results['webnn']['real_count']}")
    print(f"  - Simulation files: {results['webnn']['simulation_count']}")
    print(f"  - Hybrid files: {results['webnn']['hybrid_count']}")
    print(f"  - Unknown files: {results['webnn']['unknown_count']}")
    print()
    
    # Display overall verdict
    if webgpu_status == "real" and webnn_status == "real":
        print("\033[92mVERDICT: REAL IMPLEMENTATIONS - Both WebGPU and WebNN are using real browser-based implementations.\033[0m")
    elif webgpu_status == "simulation" and webnn_status == "simulation":
        print("\033[91mVERDICT: SIMULATIONS - Both WebGPU and WebNN are using simulations.\033[0m")
    elif webgpu_status == "hybrid" or webnn_status == "hybrid":
        print("\033[93mVERDICT: HYBRID - One or both implementations are using a mix of real and simulated code.\033[0m")
    else:
        print("\033[93mVERDICT: INCOMPLETE - Implementation status could not be fully determined.\033[0m")
    
    print("\n=================================================================")
    
    # Display real implementation files
    print("\nReal Implementation Files:")
    for impl_type in ["webgpu", "webnn"]:
        for file in results[impl_type]["files"]:
            if file["status"] == "real":
                print(f"  - [{impl_type.upper()}] {file['file_path']}")
    
    # Display simulation files
    print("\nSimulation Files:")
    for impl_type in ["webgpu", "webnn"]:
        for file in results[impl_type]["files"]:
            if file["status"] == "simulation":
                print(f"  - [{impl_type.upper()}] {file['file_path']}")
    
    # Display hybrid files
    print("\nHybrid Files:")
    for impl_type in ["webgpu", "webnn"]:
        for file in results[impl_type]["files"]:
            if file["status"] == "hybrid":
                print(f"  - [{impl_type.upper()}] {file['file_path']}")
                print(f"    Real markers: {', '.join(file['details']['real_markers'])}")
                print(f"    Simulation markers: {', '.join(file['details']['simulation_markers'])}")
    
    print("\nNext Steps:")
    if webgpu_status != "real" or webnn_status != "real":
        print("  1. Run the direct_web_integration.py script to implement real WebNN and WebGPU support")
        print("     $ python direct_web_integration.py --browser chrome --platform both")
        print("  2. Install necessary dependencies")
        print("     $ pip install selenium webdriver-manager")
        print("  3. Verify the implementation status again")
        print("     $ python verify_real_web_implementation.py")
    else:
        print("  - The implementation is already using real browser-based code.")
        print("  - You can update the NEXT_STEPS.md document to reflect this achievement.")

def main():
    """Main function."""
    # Verify implementations
    results = verify_implementations()
    
    # Display results
    display_verification_results(results)
    
    # Return exit code based on implementation status
    if results["webgpu"]["status"] == "real" and results["webnn"]["status"] == "real":
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())