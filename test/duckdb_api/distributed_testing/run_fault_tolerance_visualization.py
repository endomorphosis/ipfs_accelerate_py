#!/usr/bin/env python3
"""
Fault Tolerance Visualization Script

This script visualizes fault tolerance data from a running or simulated system,
generating comprehensive visualizations and reports.

Usage:
    python run_fault_tolerance_visualization.py [--simulation] [--output-dir DIR]
"""

import os
import sys
import argparse
import logging
import tempfile
import random
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("fault_tolerance_viz_runner")

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import fault tolerance components
from data.duckdb.distributed_testing.hardware_aware_fault_tolerance import (
    HardwareAwareFaultToleranceManager,
    FailureType,
    RecoveryStrategy,
    FailureContext,
    RecoveryAction,
    visualize_fault_tolerance,
    create_recovery_manager
)
from data.duckdb.distributed_testing.hardware_taxonomy import (
    HardwareClass,
    HardwareArchitecture,
    HardwareVendor,
    SoftwareBackend,
    PrecisionType,
    HardwareCapabilityProfile,
    MemoryProfile
)
from data.duckdb.distributed_testing.ml_pattern_detection import (
    MLPatternDetector
)


def create_simulation_data(coordinator=None):
    """
    Create a simulated fault tolerance manager with sample data.
    
    Args:
        coordinator: Optional coordinator mock
        
    Returns:
        HardwareAwareFaultToleranceManager with simulated data
    """
    logger.info("Creating simulation data...")
    
    # Create a fault tolerance manager with ML detection
    manager = HardwareAwareFaultToleranceManager(
        db_manager=None,
        coordinator=coordinator,
        enable_ml=True
    )
    
    # Start the manager
    manager.start()
    
    # Create hardware profiles for testing
    cpu_profile = HardwareCapabilityProfile(
        hardware_class=HardwareClass.CPU,
        architecture=HardwareArchitecture.X86_64,
        vendor=HardwareVendor.INTEL,
        memory=MemoryProfile(
            total_bytes=8 * 1024 * 1024 * 1024,
            available_bytes=6 * 1024 * 1024 * 1024
        ),
        compute_units=8
    )
    
    gpu_profile = HardwareCapabilityProfile(
        hardware_class=HardwareClass.GPU,
        architecture=HardwareArchitecture.GPU_CUDA,
        vendor=HardwareVendor.NVIDIA,
        memory=MemoryProfile(
            total_bytes=16 * 1024 * 1024 * 1024,
            available_bytes=14 * 1024 * 1024 * 1024
        ),
        compute_units=128
    )
    
    webgpu_profile = HardwareCapabilityProfile(
        hardware_class=HardwareClass.GPU,
        architecture=HardwareArchitecture.GPU_WEBGPU,
        vendor=HardwareVendor.GOOGLE,
        memory=MemoryProfile(
            total_bytes=4 * 1024 * 1024 * 1024,
            available_bytes=3 * 1024 * 1024 * 1024
        ),
        compute_units=32
    )
    
    # Add various failure types
    _add_cpu_failures(manager, cpu_profile)
    _add_gpu_failures(manager, gpu_profile)
    _add_webgpu_failures(manager, webgpu_profile)
    _add_timeout_failures(manager, cpu_profile)
    
    # Add recovery actions
    _add_recovery_actions(manager)
    
    # Detect patterns from simulated failures
    _detect_patterns(manager)
    
    logger.info(f"Created simulation with {len(manager.failure_history)} failures")
    return manager


def _add_cpu_failures(manager, cpu_profile):
    """Add CPU failures to the manager."""
    # Software errors on CPU
    for i in range(20):
        failure_context = FailureContext(
            task_id=f"task_cpu_{i}",
            worker_id=f"worker_cpu_{i % 5}",
            hardware_profile=cpu_profile,
            error_message=f"Software exception: {random.choice(['IndexError', 'ValueError', 'TypeError', 'AssertionError'])}",
            error_type=FailureType.SOFTWARE_ERROR,
            timestamp=datetime.now() - timedelta(days=i//4, hours=i%24),
            attempt=1
        )
        manager.failure_history.append(failure_context)


def _add_gpu_failures(manager, gpu_profile):
    """Add GPU failures to the manager."""
    # Hardware errors on GPU
    for i in range(15):
        error_type = random.choice([
            FailureType.HARDWARE_ERROR, 
            FailureType.RESOURCE_EXHAUSTION
        ])
        
        error_message = "Generic GPU error"
        if error_type == FailureType.RESOURCE_EXHAUSTION:
            error_message = "CUDA out of memory: Tried to allocate 2.00 GiB"
        elif error_type == FailureType.HARDWARE_ERROR:
            error_message = random.choice([
                "CUDA error: device-side assert triggered",
                "CUDA error: an illegal memory access was encountered",
                "CUDA driver version is insufficient"
            ])
        
        failure_context = FailureContext(
            task_id=f"task_gpu_{i}",
            worker_id=f"worker_gpu_{i % 3}",
            hardware_profile=gpu_profile,
            error_message=error_message,
            error_type=error_type,
            timestamp=datetime.now() - timedelta(days=i//3, hours=i%24),
            attempt=1
        )
        manager.failure_history.append(failure_context)


def _add_webgpu_failures(manager, webgpu_profile):
    """Add WebGPU failures to the manager."""
    # Browser failures on WebGPU
    for i in range(10):
        error_type = random.choice([
            FailureType.BROWSER_FAILURE,
            FailureType.RESOURCE_EXHAUSTION
        ])
        
        error_message = "Generic WebGPU error"
        if error_type == FailureType.BROWSER_FAILURE:
            error_message = random.choice([
                "WebGPU context lost",
                "Browser crashed unexpectedly",
                "WebGPU validation error"
            ])
        elif error_type == FailureType.RESOURCE_EXHAUSTION:
            error_message = "WebGPU out of memory: insufficient storage buffer size"
        
        failure_context = FailureContext(
            task_id=f"task_webgpu_{i}",
            worker_id=f"worker_webgpu_{i % 2}",
            hardware_profile=webgpu_profile,
            error_message=error_message,
            error_type=error_type,
            timestamp=datetime.now() - timedelta(days=i//2, hours=i%24),
            attempt=1
        )
        manager.failure_history.append(failure_context)


def _add_timeout_failures(manager, cpu_profile):
    """Add timeout failures to the manager."""
    # Timeout errors on CPU
    for i in range(5):
        failure_context = FailureContext(
            task_id=f"task_timeout_{i}",
            worker_id=f"worker_cpu_{i % 5}",
            hardware_profile=cpu_profile,
            error_message="Task execution timed out after 300 seconds",
            error_type=FailureType.TIMEOUT,
            timestamp=datetime.now() - timedelta(days=i, hours=12),
            attempt=1
        )
        manager.failure_history.append(failure_context)


def _add_recovery_actions(manager):
    """Add recovery actions to the manager."""
    # Add recovery actions for each failure
    for failure in manager.failure_history:
        task_id = failure.task_id
        worker_id = failure.worker_id
        error_type = failure.error_type
        
        # Determine recovery strategy based on error type
        if error_type == FailureType.SOFTWARE_ERROR:
            # Software errors: first try delayed retry, then different worker
            actions = [
                RecoveryAction(
                    strategy=RecoveryStrategy.DELAYED_RETRY,
                    worker_id=worker_id,
                    message="Software error: retry with delay",
                    delay=random.uniform(1.0, 5.0)
                )
            ]
            
            # Add a second action for some tasks
            if random.random() < 0.4:
                actions.append(
                    RecoveryAction(
                        strategy=RecoveryStrategy.DIFFERENT_WORKER,
                        worker_id=None,
                        message="Software error: try different worker"
                    )
                )
            
        elif error_type == FailureType.HARDWARE_ERROR:
            # Hardware errors: try different worker
            actions = [
                RecoveryAction(
                    strategy=RecoveryStrategy.DIFFERENT_WORKER,
                    worker_id=None,
                    message="Hardware error: try different worker"
                )
            ]
            
        elif error_type == FailureType.RESOURCE_EXHAUSTION:
            # OOM errors: reduce batch size, reduce precision, fallback to CPU
            actions = [
                RecoveryAction(
                    strategy=RecoveryStrategy.REDUCED_BATCH_SIZE,
                    worker_id=worker_id,
                    message="OOM error: reduce batch size"
                )
            ]
            
            # Add more actions for some tasks
            if random.random() < 0.7:
                actions.append(
                    RecoveryAction(
                        strategy=RecoveryStrategy.REDUCED_PRECISION,
                        worker_id=worker_id,
                        message="OOM error: reduce precision"
                    )
                )
                
                if random.random() < 0.3:
                    actions.append(
                        RecoveryAction(
                            strategy=RecoveryStrategy.FALLBACK_CPU,
                            worker_id=None,
                            message="OOM error: fallback to CPU"
                        )
                    )
            
        elif error_type == FailureType.BROWSER_FAILURE:
            # Browser failures: restart browser, try different browser
            actions = [
                RecoveryAction(
                    strategy=RecoveryStrategy.BROWSER_RESTART,
                    worker_id=worker_id,
                    message="Browser error: restart browser"
                )
            ]
            
            # Add a second action for some tasks
            if random.random() < 0.5:
                actions.append(
                    RecoveryAction(
                        strategy=RecoveryStrategy.DIFFERENT_WORKER,
                        worker_id=None,
                        message="Browser error: try different browser"
                    )
                )
            
        elif error_type == FailureType.TIMEOUT:
            # Timeout errors: increase timeout, then different worker
            actions = [
                RecoveryAction(
                    strategy=RecoveryStrategy.DELAYED_RETRY,
                    worker_id=worker_id,
                    message="Timeout error: retry with increased timeout",
                    delay=random.uniform(5.0, 10.0)
                )
            ]
            
            # Add a second action for some tasks
            if random.random() < 0.6:
                actions.append(
                    RecoveryAction(
                        strategy=RecoveryStrategy.DIFFERENT_WORKER,
                        worker_id=None,
                        message="Timeout error: try different worker"
                    )
                )
            
        else:
            # Default: delayed retry
            actions = [
                RecoveryAction(
                    strategy=RecoveryStrategy.DELAYED_RETRY,
                    worker_id=worker_id,
                    message="Unknown error: retry with delay",
                    delay=random.uniform(1.0, 3.0)
                )
            ]
        
        # Add to recovery history
        manager.recovery_history[task_id] = actions
        
        # Update ML detector with success/failure status
        if manager.ml_detector:
            for action in actions:
                # Randomly decide if the recovery was successful
                success = random.random() < 0.7
                manager.ml_detector.update_recovery_result(
                    task_id=task_id,
                    strategy=action.strategy,
                    success=success
                )


def _detect_patterns(manager):
    """Detect patterns from simulated failures."""
    # CPU software errors pattern
    manager.failure_patterns["pattern_cpu_software"] = {
        "type": "error_type",
        "key": "SOFTWARE_ERROR",
        "count": 15,
        "first_seen": (datetime.now() - timedelta(days=5)).isoformat(),
        "last_seen": datetime.now().isoformat(),
        "task_ids": [f"task_cpu_{i}" for i in range(5)],
        "worker_ids": [f"worker_cpu_{i}" for i in range(3)],
        "error_types": ["SOFTWARE_ERROR"],
        "recommended_action": "Review application code for common errors"
    }
    
    # GPU OOM errors pattern
    manager.failure_patterns["pattern_gpu_oom"] = {
        "type": "hardware_class",
        "key": "GPU",
        "count": 8,
        "first_seen": (datetime.now() - timedelta(days=3)).isoformat(),
        "last_seen": datetime.now().isoformat(),
        "task_ids": [f"task_gpu_{i}" for i in range(3)],
        "worker_ids": [f"worker_gpu_{i}" for i in range(2)],
        "error_types": ["RESOURCE_EXHAUSTION"],
        "recommended_action": "Consider using smaller batch sizes or reduced precision for GPU tasks"
    }
    
    # Specific worker pattern
    manager.failure_patterns["pattern_worker_gpu_0"] = {
        "type": "worker_id",
        "key": "worker_gpu_0",
        "count": 6,
        "first_seen": (datetime.now() - timedelta(days=2)).isoformat(),
        "last_seen": datetime.now().isoformat(),
        "task_ids": [f"task_gpu_{i}" for i in range(3)],
        "worker_ids": ["worker_gpu_0"],
        "error_types": ["HARDWARE_ERROR", "RESOURCE_EXHAUSTION"],
        "recommended_action": "Take worker worker_gpu_0 offline for investigation"
    }
    
    # Browser failures pattern
    manager.failure_patterns["pattern_browser_failures"] = {
        "type": "error_type",
        "key": "BROWSER_FAILURE",
        "count": 7,
        "first_seen": (datetime.now() - timedelta(days=4)).isoformat(),
        "last_seen": datetime.now().isoformat(),
        "task_ids": [f"task_webgpu_{i}" for i in range(4)],
        "worker_ids": [f"worker_webgpu_{i}" for i in range(2)],
        "error_types": ["BROWSER_FAILURE"],
        "recommended_action": "Check for browser updates or consider using a different browser"
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fault Tolerance Visualization Script")
    parser.add_argument("--simulation", action="store_true", help="Use simulated data")
    parser.add_argument("--output-dir", default="./visualizations", help="Output directory for visualizations")
    parser.add_argument("--open-browser", action="store_true", help="Open the report in a web browser")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.simulation:
        # Use simulated data
        manager = create_simulation_data()
    else:
        # Try to connect to a running system (not implemented)
        logger.error("Direct connection to running system not implemented.")
        logger.info("Use --simulation flag to generate visualizations from simulated data.")
        return 1
    
    # Generate visualizations
    report_path = visualize_fault_tolerance(manager, args.output_dir)
    
    if report_path:
        logger.info(f"Visualization report generated successfully: {report_path}")
        
        # Open in browser if requested
        if args.open_browser:
            report_url = f"file://{os.path.abspath(report_path)}"
            logger.info(f"Opening report in browser: {report_url}")
            webbrowser.open(report_url)
        
        return 0
    else:
        logger.error("Failed to generate visualization report.")
        return 1


if __name__ == "__main__":
    sys.exit(main())