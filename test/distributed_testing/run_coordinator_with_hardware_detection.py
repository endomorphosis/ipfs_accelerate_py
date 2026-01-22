#!/usr/bin/env python3
"""
Run Coordinator with Hardware Capability Detection

This script runs the Distributed Testing Coordinator with hardware capability detection
and visualization, demonstrating the integration and functionality of the hardware-aware
task distribution system.

Usage:
    python run_coordinator_with_hardware_detection.py --db-path ./test_db.duckdb
"""

import os
import sys
import json
import logging
import argparse
import asyncio
import signal
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_hardware_run")

# Import coordinator and hardware capability components
from coordinator import DistributedTestingCoordinator
from coordinator_hardware_integration import CoordinatorHardwareIntegration
from hardware_capability_detector import (
    HardwareCapabilityDetector,
    HardwareType,
    HardwareVendor,
    PrecisionType,
    CapabilityScore,
    HardwareCapability,
    WorkerHardwareCapabilities
)
from hardware_aware_visualization import HardwareSchedulingVisualizer


async def run_coordinator_demo(args):
    """
    Run the coordinator with hardware capability detection.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting coordinator with hardware capability detection")
    
    # Create the coordinator
    coordinator = DistributedTestingCoordinator(
        db_path=args.db_path,
        host=args.host,
        port=args.port,
        enable_advanced_scheduler=True,
        enable_health_monitor=True,
        enable_load_balancer=True
    )
    
    # Create hardware integration
    hardware_integration = CoordinatorHardwareIntegration(
        coordinator=coordinator,
        db_path=args.db_path,
        enable_browser_detection=args.enable_browser_detection,
        browser_executable_path=args.browser_executable_path,
        cache_capabilities=True
    )
    
    # Initialize hardware integration
    await hardware_integration.initialize()
    
    # Create visualizer
    visualizer = HardwareSchedulingVisualizer(
        output_dir=args.output_dir,
        file_format="html"
    )
    
    # Create hardware capability detector for demo purposes
    detector = HardwareCapabilityDetector(
        db_path=args.db_path,
        enable_browser_detection=args.enable_browser_detection,
        browser_executable_path=args.browser_executable_path
    )
    
    # Detect hardware capabilities on the current machine
    capabilities = detector.detect_all_capabilities_with_browsers() if args.enable_browser_detection else detector.detect_all_capabilities()
    
    # Print capabilities summary
    logger.info(f"Detected {len(capabilities.hardware_capabilities)} hardware capabilities")
    logger.info(f"OS: {capabilities.os_type} {capabilities.os_version}")
    logger.info(f"CPU Count: {capabilities.cpu_count}")
    logger.info(f"Total Memory: {capabilities.total_memory_gb:.2f} GB")
    
    # Print detected hardware
    for hw in capabilities.hardware_capabilities:
        hw_type = hw.hardware_type.name if isinstance(hw.hardware_type, Enum) else hw.hardware_type
        vendor = hw.vendor.name if isinstance(hw.vendor, Enum) else hw.vendor
        logger.info(f"  {hw_type} - {vendor} - {hw.model}")
    
    # Store capabilities in database
    if args.store_capabilities:
        detector.store_capabilities(capabilities)
        logger.info("Stored hardware capabilities in database")
    
    # Start the coordinator
    try:
        # Hook SIGINT for graceful shutdown
        loop = asyncio.get_event_loop()
        
        # Register shutdown handler
        for signal_name in ('SIGINT', 'SIGTERM'):
            loop.add_signal_handler(
                getattr(signal, signal_name),
                lambda: asyncio.create_task(shutdown(coordinator))
            )
        
        # Start coordinator server
        site, runner = await coordinator.start()
        
        logger.info(f"Coordinator server started at http://{args.host}:{args.port}")
        logger.info("Press Ctrl+C to stop the server")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info("Server shutting down")
    finally:
        # Stop the coordinator
        await coordinator.stop()
        logger.info("Server stopped")


async def shutdown(coordinator):
    """Gracefully shut down the coordinator."""
    logger.info("Shutting down coordinator...")
    await coordinator.stop()
    
    # Stop the event loop
    asyncio.get_event_loop().stop()


async def run_simulated_tasks(args):
    """
    Run a simulation of hardware-aware task assignment.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting hardware-aware task assignment simulation")
    
    # Create hardware capability detector
    detector = HardwareCapabilityDetector(
        db_path=args.db_path,
        enable_browser_detection=args.enable_browser_detection
    )
    
    # Detect current hardware for task requirements
    capabilities = detector.detect_all_capabilities()
    
    # Create simulated workers with various hardware
    workers = [
        {
            "worker_id": "worker-1",
            "hostname": "worker-node-1",
            "hardware_capabilities": {
                "hostname": "worker-node-1",
                "os_type": "Linux",
                "os_version": "5.10.0",
                "cpu_count": 8,
                "total_memory_gb": 16.0,
                "hardware_capabilities": [
                    {
                        "hardware_type": "cpu",
                        "vendor": "intel",
                        "model": "Intel Core i7",
                        "cores": 8,
                        "memory_gb": 16.0,
                        "supported_precisions": ["fp32", "fp16", "int8"]
                    }
                ]
            }
        },
        {
            "worker_id": "worker-2",
            "hostname": "worker-node-2",
            "hardware_capabilities": {
                "hostname": "worker-node-2",
                "os_type": "Linux",
                "os_version": "5.10.0",
                "cpu_count": 16,
                "total_memory_gb": 32.0,
                "hardware_capabilities": [
                    {
                        "hardware_type": "cpu",
                        "vendor": "amd",
                        "model": "AMD Ryzen 9",
                        "cores": 16,
                        "memory_gb": 32.0,
                        "supported_precisions": ["fp32", "fp16", "int8"]
                    },
                    {
                        "hardware_type": "gpu",
                        "vendor": "nvidia",
                        "model": "NVIDIA RTX 3080",
                        "compute_units": 68,
                        "memory_gb": 10.0,
                        "supported_precisions": ["fp32", "fp16", "int8"]
                    }
                ]
            }
        },
        {
            "worker_id": "worker-3",
            "hostname": "worker-node-3",
            "hardware_capabilities": {
                "hostname": "worker-node-3",
                "os_type": "macOS",
                "os_version": "12.4",
                "cpu_count": 10,
                "total_memory_gb": 32.0,
                "hardware_capabilities": [
                    {
                        "hardware_type": "cpu",
                        "vendor": "apple",
                        "model": "Apple M1 Pro",
                        "cores": 10,
                        "memory_gb": 32.0,
                        "supported_precisions": ["fp32", "fp16", "int8"]
                    }
                ]
            }
        },
        {
            "worker_id": "worker-4",
            "hostname": "worker-node-4",
            "hardware_capabilities": {
                "hostname": "worker-node-4",
                "os_type": "Linux",
                "os_version": "5.10.0",
                "cpu_count": 4,
                "total_memory_gb": 8.0,
                "hardware_capabilities": [
                    {
                        "hardware_type": "cpu",
                        "vendor": "intel",
                        "model": "Intel Core i5",
                        "cores": 4,
                        "memory_gb": 8.0,
                        "supported_precisions": ["fp32", "fp16", "int8"]
                    },
                    {
                        "hardware_type": "webgpu",
                        "vendor": "firefox",
                        "model": "Firefox WebGPU",
                        "memory_gb": 4.0,
                        "supported_precisions": ["fp32", "fp16"]
                    }
                ]
            }
        }
    ]
    
    # Create tasks with hardware requirements
    tasks = [
        {
            "task_id": "task-1",
            "type": "test",
            "requirements": {
                "hardware": {
                    "required_types": ["cpu"]
                }
            }
        },
        {
            "task_id": "task-2",
            "type": "benchmark",
            "requirements": {
                "hardware": {
                    "required_types": ["gpu"],
                    "min_memory_gb": 8.0,
                    "precision_types": ["fp16"]
                }
            }
        },
        {
            "task_id": "task-3",
            "type": "test",
            "requirements": {
                "hardware": {
                    "required_types": ["cpu"],
                    "min_memory_gb": 16.0
                }
            }
        },
        {
            "task_id": "task-4",
            "type": "benchmark",
            "requirements": {
                "hardware": {
                    "required_types": ["webgpu"]
                }
            }
        },
        {
            "task_id": "task-5",
            "type": "test",
            "requirements": {
                "hardware": {
                    "required_types": ["tpu", "cpu"],
                    "min_memory_gb": 32.0
                }
            }
        }
    ]
    
    # Create task compatibility tracker
    task_compatibility = {}
    
    # Parse each worker's hardware capabilities
    parsed_capabilities = {}
    for worker in workers:
        worker_id = worker["worker_id"]
        hw_caps = worker["hardware_capabilities"]
        parsed_caps = detector._parse_hardware_capabilities(worker_id, hw_caps)
        parsed_capabilities[worker_id] = parsed_caps
        
        # Store capabilities in database if requested
        if args.store_capabilities and parsed_caps:
            detector.store_capabilities(parsed_caps)
            logger.info(f"Stored hardware capabilities for worker {worker_id} in database")
    
    # Check task compatibility for each worker
    for worker_id, worker_caps in parsed_capabilities.items():
        if not worker_caps:
            logger.warning(f"No parsed capabilities for worker {worker_id}")
            continue
            
        task_compatibility[worker_id] = []
        
        # Check compatibility for each task
        for task in tasks:
            task_id = task["task_id"]
            requirements = task.get("requirements", {})
            hardware_requirements = requirements.get("hardware", {})
            
            # If no hardware requirements, worker can handle the task
            if not hardware_requirements:
                task_compatibility[worker_id].append(task_id)
                continue
            
            # Check hardware requirements against worker capabilities
            is_compatible = True
            
            # Check required hardware types
            if "required_types" in hardware_requirements:
                required_types = hardware_requirements["required_types"]
                
                worker_hardware_types = set()
                for hw in worker_caps.hardware_capabilities:
                    hw_type = hw.hardware_type.value if isinstance(hw.hardware_type, Enum) else str(hw.hardware_type)
                    worker_hardware_types.add(hw_type)
                
                if not all(req_type in worker_hardware_types for req_type in required_types):
                    is_compatible = False
            
            # Check minimum memory requirements
            if is_compatible and "min_memory_gb" in hardware_requirements:
                min_memory_gb = hardware_requirements["min_memory_gb"]
                
                has_sufficient_memory = False
                for hw in worker_caps.hardware_capabilities:
                    if hw.memory_gb and hw.memory_gb >= min_memory_gb:
                        has_sufficient_memory = True
                        break
                
                if not has_sufficient_memory and worker_caps.total_memory_gb >= min_memory_gb:
                    has_sufficient_memory = True
                
                if not has_sufficient_memory:
                    is_compatible = False
            
            # Check precision type requirements
            if is_compatible and "precision_types" in hardware_requirements:
                required_precision_types = hardware_requirements["precision_types"]
                
                has_required_precision_types = False
                for hw in worker_caps.hardware_capabilities:
                    supported_precisions = set()
                    for p in hw.supported_precisions:
                        p_str = p.value if isinstance(p, Enum) else str(p)
                        supported_precisions.add(p_str)
                    
                    if all(req_type in supported_precisions for req_type in required_precision_types):
                        has_required_precision_types = True
                        break
                
                if not has_required_precision_types:
                    is_compatible = False
            
            # Add to compatible tasks if all checks passed
            if is_compatible:
                task_compatibility[worker_id].append(task_id)
    
    # Create visualizer
    visualizer = HardwareSchedulingVisualizer(
        output_dir=args.output_dir,
        file_format="html"
    )
    
    # Create worker assignments for visualization
    worker_assignments = {}
    for worker_id, compatible_tasks in task_compatibility.items():
        worker_assignments[worker_id] = compatible_tasks
    
    # Define worker types for visualization
    worker_types = {
        "worker-1": "cpu",
        "worker-2": "gpu",
        "worker-3": "cpu",
        "worker-4": "browser"
    }
    
    # Visualize workload distribution
    distribution_path = visualizer.visualize_workload_distribution(
        worker_assignments=worker_assignments,
        worker_types=worker_types,
        filename="simulated_workload_distribution"
    )
    
    # Calculate efficiency scores for visualization
    efficiency_scores = {}
    for worker_id, worker_caps in parsed_capabilities.items():
        # Calculate a simple efficiency score based on number of compatible tasks
        num_tasks = len(tasks)
        compatible_count = len(task_compatibility.get(worker_id, []))
        
        # Generate a weighted score
        if worker_id == "worker-1":  # CPU-only worker
            score = 0.7  # Good for CPU tasks
        elif worker_id == "worker-2":  # GPU worker
            score = 0.9  # Great for GPU tasks
        elif worker_id == "worker-3":  # Apple M1
            score = 0.8  # Very good for most tasks
        elif worker_id == "worker-4":  # WebGPU worker
            score = 0.6  # Good for WebGPU tasks
        else:
            score = 0.5  # Default
        
        efficiency_scores[worker_id] = score
    
    # Get efficiency scores for each worker
    for worker_id, score in efficiency_scores.items():
        logger.info(f"Worker {worker_id} efficiency score: {score}")
    
    # Print compatibility matrix
    logger.info("\nTask Compatibility Matrix:")
    logger.info("-" * 60)
    header = "Worker ID       | " + " | ".join([f"{task['task_id']:8}" for task in tasks])
    logger.info(header)
    logger.info("-" * len(header))
    
    for worker_id in sorted(task_compatibility.keys()):
        compatible_task_ids = task_compatibility[worker_id]
        row = f"{worker_id:15} | "
        
        for task in tasks:
            task_id = task["task_id"]
            if task_id in compatible_task_ids:
                row += "  ✓     | "
            else:
                row += "  ✗     | "
        
        logger.info(row)
    
    logger.info("-" * len(header))
    
    # Generate detailed HTML report
    report_path = os.path.join(args.output_dir, "hardware_compatibility_report.html")
    
    # Create HTML report
    with open(report_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Hardware Capability and Task Compatibility Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .section {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .compatible {{
            color: green;
            font-weight: bold;
        }}
        .incompatible {{
            color: red;
        }}
        .visual-section {{
            margin-top: 30px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>Hardware Capability and Task Compatibility Report</h1>
    
    <div class="section">
        <h2>Worker Hardware Summary</h2>
        <table>
            <tr>
                <th>Worker ID</th>
                <th>Hostname</th>
                <th>OS</th>
                <th>CPU Count</th>
                <th>Memory (GB)</th>
                <th>Hardware Components</th>
            </tr>
""")
        
        # Add worker hardware rows
        for worker in workers:
            worker_id = worker["worker_id"]
            hw_caps = worker["hardware_capabilities"]
            hw_components = ", ".join([h["hardware_type"] for h in hw_caps.get("hardware_capabilities", [])])
            
            f.write(f"""
            <tr>
                <td>{worker_id}</td>
                <td>{hw_caps.get("hostname", "unknown")}</td>
                <td>{hw_caps.get("os_type", "unknown")} {hw_caps.get("os_version", "")}</td>
                <td>{hw_caps.get("cpu_count", 0)}</td>
                <td>{hw_caps.get("total_memory_gb", 0):.2f}</td>
                <td>{hw_components}</td>
            </tr>
""")
        
        f.write("""
        </table>
    </div>
    
    <div class="section">
        <h2>Task Requirements</h2>
        <table>
            <tr>
                <th>Task ID</th>
                <th>Type</th>
                <th>Hardware Requirements</th>
            </tr>
""")
        
        # Add task requirement rows
        for task in tasks:
            task_id = task["task_id"]
            task_type = task["type"]
            hw_reqs = task.get("requirements", {}).get("hardware", {})
            
            # Format hardware requirements
            req_str = ""
            if "required_types" in hw_reqs:
                req_str += f"Required Hardware: {', '.join(hw_reqs['required_types'])}<br>"
            if "min_memory_gb" in hw_reqs:
                req_str += f"Minimum Memory: {hw_reqs['min_memory_gb']} GB<br>"
            if "precision_types" in hw_reqs:
                req_str += f"Precision Types: {', '.join(hw_reqs['precision_types'])}<br>"
            
            f.write(f"""
            <tr>
                <td>{task_id}</td>
                <td>{task_type}</td>
                <td>{req_str if req_str else "None"}</td>
            </tr>
""")
        
        f.write("""
        </table>
    </div>
    
    <div class="section">
        <h2>Task Compatibility Matrix</h2>
        <table>
            <tr>
                <th>Worker ID</th>
""")
        
        # Add task headers
        for task in tasks:
            f.write(f"""
                <th>{task['task_id']}</th>
""")
        
        f.write("""
            </tr>
""")
        
        # Add compatibility rows
        for worker_id in sorted(task_compatibility.keys()):
            compatible_task_ids = task_compatibility[worker_id]
            
            f.write(f"""
            <tr>
                <td>{worker_id}</td>
""")
            
            for task in tasks:
                task_id = task["task_id"]
                if task_id in compatible_task_ids:
                    f.write(f"""
                <td class="compatible">✓ Compatible</td>
""")
                else:
                    f.write(f"""
                <td class="incompatible">✗ Incompatible</td>
""")
            
            f.write("""
            </tr>
""")
        
        f.write("""
        </table>
    </div>
    
    <div class="visual-section">
        <h2>Workload Distribution Visualization</h2>
        <img src="simulated_workload_distribution.html" alt="Workload Distribution">
    </div>
""")
        
        f.write("""
</body>
</html>
""")
    
    logger.info(f"Generated compatibility report at {report_path}")
    
    # Print summary
    logger.info("\nSimulation completed:")
    logger.info(f"- Workers analyzed: {len(workers)}")
    logger.info(f"- Tasks evaluated: {len(tasks)}")
    logger.info(f"- Compatibility report: {report_path}")
    logger.info(f"- Distribution visualization: {distribution_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Coordinator with Hardware Capability Detection")
    parser.add_argument("--db-path", default="./hardware_db.duckdb", help="Path to DuckDB database")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--enable-browser-detection", action="store_true", help="Enable browser-based WebGPU/WebNN detection")
    parser.add_argument("--browser-executable-path", help="Path to browser executable for browser-based detection")
    parser.add_argument("--output-dir", default="./visualizations", help="Directory to save visualizations")
    parser.add_argument("--store-capabilities", action="store_true", help="Store capabilities in database")
    parser.add_argument("--run-coordinator", action="store_true", help="Run the coordinator server")
    parser.add_argument("--run-simulation", action="store_true", help="Run a simulation of task assignment")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.run_coordinator:
        # Run the coordinator server
        asyncio.run(run_coordinator_demo(args))
    elif args.run_simulation:
        # Run the simulation
        asyncio.run(run_simulated_tasks(args))
    else:
        # By default, run the simulation
        asyncio.run(run_simulated_tasks(args))


if __name__ == "__main__":
    main()