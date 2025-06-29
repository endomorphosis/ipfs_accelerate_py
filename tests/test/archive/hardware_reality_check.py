#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware Reality Check

This script detects which hardware platforms are actually available and logs the results
to the hardware_availability_log table in the benchmark database.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
import duckdb
import torch
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_cpu():
    """Detect CPU information."""
    details = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }
    
    # Try to get more detailed CPU info from /proc/cpuinfo on Linux
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.strip().startswith("model name"):
                        details["model_name"] = line.split(":")[1].strip()
                        break
        except Exception as e:
            logger.warning(f"Could not read /proc/cpuinfo: {e}")
    
    return True, "Direct CPU detection", details

def detect_cuda():
    """Detect CUDA availability and information."""
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() returned False", {}
    
    # Get basic CUDA information
    try:
        device_count = torch.cuda.device_count()
        
        if device_count == 0:
            return False, "No CUDA devices found via torch.cuda.device_count()", {}
        
        details = {
            "device_count": device_count,
            "devices": [],
            "cuda_version": torch.version.cuda,
        }
        
        # Get information for each device
        for i in range(device_count):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "name": device_props.name,
                "total_memory_gb": round(device_props.total_memory / (1024**3), 2),
                "compute_capability": f"{device_props.major}.{device_props.minor}",
                "multi_processor_count": device_props.multi_processor_count,
            }
            details["devices"].append(device_info)
        
        return True, "Direct CUDA device detection via torch", details
    except Exception as e:
        return False, f"Error detecting CUDA: {str(e)}", {}

def detect_rocm():
    """Detect ROCm (AMD GPU) availability."""
    # Check for ROCm through torch's AMD backend
    try:
        if hasattr(torch, 'hip') and torch.hip.is_available():
            return True, "torch.hip.is_available() returned True", {"device_count": torch.hip.device_count()}
        else:
            return False, "torch.hip is not available", {}
    except Exception as e:
        # Check if the rocm command is available
        try:
            import subprocess
            result = subprocess.run(["rocm-smi", "--showallinfo"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True, "rocm-smi command succeeded", {"rocm_smi_output": result.stdout}
            else:
                return False, "rocm-smi command failed", {"error": result.stderr}
        except Exception as e2:
            return False, f"ROCm detection failed via torch.hip and rocm-smi: {str(e)}, {str(e2)}", {}

def detect_mps():
    """Detect MPS (Apple Silicon) availability."""
    # Check if we're on macOS
    if platform.system() != "Darwin":
        return False, "Not running on macOS", {}
    
    # Check if we're on Apple Silicon
    arch = platform.machine()
    if arch != "arm64":
        return False, f"Not running on arm64 architecture (found {arch})", {}
    
    # Check for MPS support in PyTorch
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Get device information
            details = {
                "os_version": platform.mac_ver()[0],
                "model": platform.machine(),
                # Unfortunately torch doesn't expose detailed MPS device info easily
                "mps_enabled": torch.backends.mps.is_available(),
                "mps_built": torch.backends.mps.is_built(),
            }
            
            # Try to get more detailed info about the device
            try:
                import subprocess
                result = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    details["system_info"] = result.stdout
            except Exception as e:
                logger.warning(f"Could not get detailed system info: {e}")
            
            return True, "torch.backends.mps.is_available() returned True", details
        else:
            if hasattr(torch.backends, "mps"):
                return False, "torch.backends.mps.is_available() returned False", {"mps_built": torch.backends.mps.is_built()}
            else:
                return False, "torch.backends.mps is not available", {}
    except Exception as e:
        return False, f"Error detecting MPS: {str(e)}", {}

def detect_openvino():
    """Detect OpenVINO availability."""
    try:
        import openvino
        details = {"version": getattr(openvino, "__version__", "unknown")}
        
        # Try to get available devices
        try:
            from openvino.runtime import Core
            core = Core()
            devices = core.available_devices
            details["available_devices"] = devices
            
            # Get device properties for each device
            device_props = {}
            for device in devices:
                try:
                    props = core.get_property(device, "FULL_DEVICE_NAME")
                    device_props[device] = props
                except Exception as e:
                    device_props[device] = str(e)
            
            details["device_properties"] = device_props
            
            return True, "OpenVINO runtime detected with available devices", details
        except Exception as e:
            return True, "OpenVINO package detected but could not query devices", {"error": str(e), "version": details["version"]}
    except ImportError:
        return False, "OpenVINO package not installed", {}

def detect_qnn():
    """Detect QNN (Qualcomm Neural Networks) availability."""
    try:
        import qnn
        return True, "QNN package detected", {"version": getattr(qnn, "__version__", "unknown")}
    except ImportError:
        # Try alternative method by checking for the Qualcomm SDK
        sdk_paths = [
            "/opt/qcom/aistack",
            "/opt/qualcomm/aistack",
            "/usr/local/Qualcomm/AIStack",
            os.path.expanduser("~/Qualcomm/AIStack"),
        ]
        
        for path in sdk_paths:
            if os.path.exists(path):
                return True, f"Qualcomm AI Stack found at {path}", {"path": path}
        
        return False, "QNN package not installed and Qualcomm AI Stack not found", {}

def detect_webnn():
    """Check for WebNN capability."""
    # WebNN requires a browser, so we can't detect it directly
    # but we can check for Node.js WebNN implementations
    try:
        import subprocess
        result = subprocess.run(["npm", "list", "-g", "webnn"], capture_output=True, text=True, timeout=5)
        if "webnn" in result.stdout:
            return True, "Node.js WebNN package detected", {"npm_output": result.stdout}
        else:
            return False, "WebNN requires a browser environment and is not natively available", {}
    except Exception:
        return False, "WebNN requires a browser environment and is not natively available", {}

def detect_webgpu():
    """Check for WebGPU capability."""
    # WebGPU requires a browser, so we can't detect it directly
    # but we can check for Node.js WebGPU implementations
    try:
        import subprocess
        result = subprocess.run(["npm", "list", "-g", "webgpu"], capture_output=True, text=True, timeout=5)
        if "webgpu" in result.stdout:
            return True, "Node.js WebGPU package detected", {"npm_output": result.stdout}
        else:
            return False, "WebGPU requires a browser environment and is not natively available", {}
    except Exception:
        return False, "WebGPU requires a browser environment and is not natively available", {}

def detect_all_hardware():
    """Detect all hardware platforms and return the results."""
    detectors = {
        "cpu": detect_cpu,
        "cuda": detect_cuda,
        "rocm": detect_rocm,
        "mps": detect_mps,
        "openvino": detect_openvino,
        "qnn": detect_qnn,
        "webnn": detect_webnn,
        "webgpu": detect_webgpu,
    }
    
    results = {}
    
    for hw_type, detector in detectors.items():
        logger.info(f"Detecting {hw_type}...")
        try:
            is_available, detection_method, details = detector()
            results[hw_type] = {
                "is_available": is_available,
                "detection_method": detection_method,
                "detection_details": details,
            }
            availability = "AVAILABLE" if is_available else "NOT AVAILABLE"
            logger.info(f"  {hw_type}: {availability} - {detection_method}")
        except Exception as e:
            logger.error(f"Error detecting {hw_type}: {e}")
            results[hw_type] = {
                "is_available": False,
                "detection_method": f"Error during detection: {str(e)}",
                "detection_details": {},
            }
    
    return results

def log_hardware_availability(db_path, results):
    """Log hardware availability to the database."""
    try:
        # Connect to the database
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        
        # Insert hardware availability results
        for hw_type, result in results.items():
            # Convert details to JSON string
            detection_details = json.dumps(result["detection_details"])
            
            # Create sequence if it doesn't exist
            conn.execute("CREATE SEQUENCE IF NOT EXISTS hardware_availability_log_id_seq")
            
            # Get the next ID from the sequence
            next_id = conn.execute("SELECT nextval('hardware_availability_log_id_seq')").fetchone()[0]
            
            # Insert into the hardware_availability_log table
            conn.execute("""
            INSERT INTO hardware_availability_log 
            (id, hardware_type, is_available, detection_method, detection_details, detected_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (next_id, hw_type, result["is_available"], result["detection_method"], detection_details))
        
        # Close the connection
        conn.close()
        logger.info("Hardware availability logged to the database")
        
        return True
    except Exception as e:
        logger.error(f"Failed to log hardware availability: {e}")
        return False

def generate_report(results):
    """Generate a report of hardware availability."""
    report = []
    report.append("# Hardware Availability Report")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Available Hardware")
    report.append("")
    
    # List available hardware
    available = [hw for hw, result in results.items() if result["is_available"]]
    if available:
        for hw in available:
            details = results[hw]["detection_details"]
            detail_str = ""
            
            if hw == "cuda" and "devices" in details:
                devices = details["devices"]
                device_names = [d["name"] for d in devices]
                detail_str = f" ({', '.join(device_names)})"
            elif hw == "openvino" and "available_devices" in details:
                detail_str = f" ({', '.join(details['available_devices'])})"
            
            report.append(f"- ✅ **{hw.upper()}**{detail_str}: {results[hw]['detection_method']}")
    else:
        report.append("- ❌ No hardware platforms detected as available")
    
    report.append("")
    report.append("## Unavailable Hardware")
    report.append("")
    
    # List unavailable hardware
    unavailable = [hw for hw, result in results.items() if not result["is_available"]]
    if unavailable:
        for hw in unavailable:
            report.append(f"- ❌ **{hw.upper()}**: {results[hw]['detection_method']}")
    else:
        report.append("- ✅ All hardware platforms are available")
    
    report.append("")
    report.append("## Detailed Information")
    report.append("")
    
    # Add detailed information for each hardware platform
    for hw in sorted(results.keys()):
        result = results[hw]
        available = "✅ Available" if result["is_available"] else "❌ Not Available"
        report.append(f"### {hw.upper()} ({available})")
        report.append("")
        report.append(f"Detection method: {result['detection_method']}")
        report.append("")
        
        # Add details if available
        details = result["detection_details"]
        if details:
            report.append("Details:")
            report.append("```json")
            report.append(json.dumps(details, indent=2))
            report.append("```")
        else:
            report.append("No additional details available.")
        
        report.append("")
    
    return "\n".join(report)

def main():
    """Main function to detect hardware and update the database."""
    parser = argparse.ArgumentParser(description="Hardware Reality Check")
    parser.add_argument("--db-path", type=str, help="Path to the benchmark database (default: uses BENCHMARK_DB_PATH environment variable)")
    parser.add_argument("--output", type=str, help="Path to save the hardware report (default: hardware_availability_report.md)")
    parser.add_argument("--no-db", action="store_true", help="Skip logging to the database")
    
    args = parser.parse_args()
    
    # Get database path
    db_path = args.db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    db_path = Path(db_path)
    
    if not args.no_db and not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1
    
    # Detect hardware
    results = detect_all_hardware()
    
    # Log to database if requested
    if not args.no_db:
        if not log_hardware_availability(db_path, results):
            logger.error("Failed to log hardware availability to the database")
    
    # Generate report
    report = generate_report(results)
    
    # Save report
    output_path = args.output or "hardware_availability_report.md"
    with open(output_path, "w") as f:
        f.write(report)
    
    logger.info(f"Hardware availability report saved to {output_path}")
    
    # Print summary
    available = [hw for hw, result in results.items() if result["is_available"]]
    unavailable = [hw for hw, result in results.items() if not result["is_available"]]
    
    logger.info(f"Available hardware: {', '.join(available) if available else 'None'}")
    logger.info(f"Unavailable hardware: {', '.join(unavailable) if unavailable else 'None'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())