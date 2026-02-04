#!/usr/bin/env python3
"""
Worker Auto-Discovery with CI/CD Integration Example

This example demonstrates how to use the worker auto-discovery feature with CI/CD integration.
It shows how to:

1. Start a coordinator with worker auto-discovery enabled
2. Allow workers to auto-register with capability detection
3. Submit tasks to the coordinator
4. Report results to CI systems using the TestResultReporter
5. Generate comprehensive reports with performance metrics

This creates a complete end-to-end workflow for a distributed testing environment
with automatic worker discovery and reporting.
"""

import anyio
import json
import logging
import os
import sys
import tempfile
import time
import random
from datetime import datetime
from pathlib import Path
import socket
import uuid

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import from distributed_testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import necessary modules
from .coordinator import DistributedTestingCoordinator
from .worker import Worker
from .create_task import create_benchmark_task
from .ci.api_interface import CIProviderFactory, TestRunResult
from .ci.result_reporter import TestResultReporter
from .ci.register_providers import register_all_providers


def detect_hardware_capabilities():
    """
    Detect hardware capabilities of the current machine.
    
    Returns:
        Dict containing hardware capabilities
    """
    capabilities = {
        "hardware": ["cpu"],
        "cpu_info": {},
        "memory_gb": 0,
        "os_info": {},
        "models": ["bert", "t5", "vit"]  # Default supported models
    }
    
    # Basic OS info
    capabilities["os_info"] = {
        "system": sys.platform,
        "python_version": sys.version.split()[0],
        "hostname": socket.gethostname()
    }
    
    # Try to detect CPU info
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logical_count = psutil.cpu_count(logical=True)
        memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        
        capabilities["cpu_info"] = {
            "physical_cores": cpu_count,
            "logical_cores": cpu_logical_count,
            "percent_available": 100 - psutil.cpu_percent(interval=0.1)
        }
        capabilities["memory_gb"] = memory_gb
    except ImportError:
        logger.warning("psutil not available, using fallback CPU detection")
        import multiprocessing
        capabilities["cpu_info"] = {
            "logical_cores": multiprocessing.cpu_count(),
            "percent_available": 100  # Assume 100% available
        }
        capabilities["memory_gb"] = 8  # Assume 8GB of RAM
    
    # Try to detect CUDA
    try:
        # Check if CUDA is available by trying to import torch and check CUDA
        import torch
        if torch.cuda.is_available():
            capabilities["hardware"].append("cuda")
            cuda_devices = []
            for i in range(torch.cuda.device_count()):
                cuda_devices.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                })
            capabilities["cuda_info"] = {
                "device_count": torch.cuda.device_count(),
                "devices": cuda_devices
            }
    except (ImportError, AttributeError):
        logger.warning("PyTorch CUDA detection not available")
    
    # Try to detect ROCm
    try:
        # Check for ROCm by looking for hip module in torch
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            capabilities["hardware"].append("rocm")
            capabilities["rocm_info"] = {
                "device_count": torch.hip.device_count()
            }
    except (ImportError, AttributeError):
        logger.warning("PyTorch ROCm detection not available")
    
    # Try to detect MPS (Apple Silicon GPU)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            capabilities["hardware"].append("mps")
            capabilities["mps_info"] = {
                "is_built": torch.backends.mps.is_built(),
                "is_available": torch.backends.mps.is_available()
            }
    except (ImportError, AttributeError):
        logger.warning("PyTorch MPS detection not available")
    
    # Try to detect other AI accelerators
    # OpenVINO
    try:
        import openvino
        capabilities["hardware"].append("openvino")
        capabilities["openvino_info"] = {
            "version": openvino.__version__
        }
    except ImportError:
        logger.warning("OpenVINO not available")
    
    # Add additional capabilities based on detected hardware
    if "cuda" in capabilities["hardware"] or "rocm" in capabilities["hardware"]:
        # Add GPU-intensive models to capabilities
        capabilities["models"].extend(["llava", "llama", "xclip"])
    
    if "mps" in capabilities["hardware"]:
        # Add MPS-compatible models
        capabilities["models"].extend(["whisper", "clip"])
    
    if "openvino" in capabilities["hardware"]:
        # Add OpenVINO-optimized models
        capabilities["models"].extend(["detr", "whisper"])
    
    return capabilities


async def run_example(ci_provider_type=None, ci_config=None, num_workers=2):
    """
    Run the example with CI integration and worker auto-discovery.
    
    Args:
        ci_provider_type: CI provider type (github, gitlab, etc.)
        ci_config: CI provider configuration
        num_workers: Number of simulated workers to create
    """
    # Use local CI provider if none specified
    ci_provider_type = ci_provider_type or "local"
    ci_config = ci_config or {}
    
    # Register all CI providers
    register_all_providers()
    
    # Create temporary directories for database, reports, and artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        db_path = temp_dir_path / "coordinator.db"
        reports_dir = temp_dir_path / "reports"
        artifacts_dir = temp_dir_path / "artifacts"
        reports_dir.mkdir()
        artifacts_dir.mkdir()
        
        logger.info(f"Using temporary directory: {temp_dir}")
        logger.info(f"Reports directory: {reports_dir}")
        logger.info(f"Artifacts directory: {artifacts_dir}")
        
        try:
            # Create and start coordinator with worker auto-discovery enabled
            coordinator = DistributedTestingCoordinator(
                host="127.0.0.1", 
                port=8080,
                db_path=str(db_path),
                enable_batch_processing=True,        # Enable batch processing
                worker_auto_discovery=True,          # Enable worker auto-discovery
                discovery_interval=5,                # Check for new workers every 5 seconds
                auto_register_workers=True,          # Allow workers to auto-register
                enable_plugins=False                 # Don't use plugins for this example
            )
            
            # Start coordinator
            logger.info("Starting coordinator with worker auto-discovery...")
            await coordinator.start()
            logger.info("Coordinator started")
            
            # Create and start multiple workers
            workers = []
            for i in range(num_workers):
                # Simulate different hardware capabilities for each worker
                worker_capabilities = detect_hardware_capabilities()
                
                # Simulate different worker capabilities
                if i % 2 == 0:
                    # Make some workers more powerful than others
                    worker_capabilities["memory_gb"] = 32 + (i * 16)
                    worker_capabilities["cpu_info"]["physical_cores"] = 16 + (i * 4)
                    worker_capabilities["models"].extend(["whisper", "wav2vec2"])
                    if "cuda" not in worker_capabilities["hardware"]:
                        # Add simulated CUDA capability
                        worker_capabilities["hardware"].append("cuda")
                        worker_capabilities["cuda_info"] = {
                            "device_count": 1,
                            "devices": [
                                {
                                    "index": 0,
                                    "name": "NVIDIA RTX 4090",
                                    "memory_total": 24
                                }
                            ]
                        }
                
                # Create a unique worker ID
                worker_id = f"auto-worker-{uuid.uuid4().hex[:8]}"
                
                # Create worker
                worker = Worker(
                    coordinator_url="http://127.0.0.1:8080",
                    worker_id=worker_id,
                    capabilities=worker_capabilities,
                    max_concurrent_tasks=2,         # Worker can run 2 tasks in parallel
                    auto_register=True              # Enable auto-registration
                )
                
                # Connect worker to coordinator with auto-discovery
                logger.info(f"Connecting worker {worker_id} to coordinator...")
                await worker.connect()
                logger.info(f"Worker {worker_id} connected with capabilities: {worker_capabilities['hardware']}")
                
                workers.append(worker)
            
            # Wait a bit for all workers to be discovered and registered
            logger.info("Waiting for all workers to be registered...")
            await anyio.sleep(2)
            
            # Create CI provider
            logger.info(f"Creating {ci_provider_type} CI provider...")
            ci_provider = await CIProviderFactory.create_provider(ci_provider_type, ci_config)
            
            # Create test result reporter
            reporter = TestResultReporter(
                ci_provider=ci_provider,
                report_dir=str(reports_dir),
                artifact_dir=str(artifacts_dir)
            )
            
            # Create a test run
            logger.info("Creating test run...")
            test_run_data = {
                "name": "Worker Auto-Discovery Example",
                "build_id": f"example-{int(time.time())}",
                "commit_sha": ci_config.get("commit_sha", "HEAD"),
                "branch": "main"
            }
            
            test_run = await ci_provider.create_test_run(test_run_data)
            test_run_id = test_run["id"]
            logger.info(f"Created test run with ID: {test_run_id}")
            
            # Create a test result object
            test_result = TestRunResult(
                test_run_id=test_run_id,
                status="running",
                total_tests=0,     # Will be updated later
                passed_tests=0,    # Will be updated later
                failed_tests=0,    # Will be updated later
                skipped_tests=0,   # Will be updated later
                duration_seconds=0 # Will be updated later
            )
            
            # Initialize metrics for the test result
            test_result.metadata = {
                "test_details": True,
                "performance_metrics": {},
                "environment": {
                    "Python Version": sys.version.split()[0],
                    "Platform": sys.platform,
                    "Coordinator": "DistributedTestingCoordinator with worker auto-discovery",
                    "Worker Count": len(workers),
                    "Total CPU Cores": sum(w.capabilities.get("cpu_info", {}).get("physical_cores", 0) for w in workers),
                    "Total GPU Count": sum(w.capabilities.get("cuda_info", {}).get("device_count", 0) for w in workers)
                }
            }
            
            # Create task groups for different model types
            model_types = [
                {"name": "bert-base-uncased", "type": "text", "count": 3},
                {"name": "vit-base", "type": "vision", "count": 2},
                {"name": "t5-small", "type": "text_generation", "count": 2},
                {"name": "whisper-tiny", "type": "audio", "count": 2}
            ]
            
            # Create and submit tasks
            submitted_tasks = []
            total_task_count = sum(m["count"] for m in model_types)
            
            for model in model_types:
                logger.info(f"Creating {model['count']} tasks for {model['name']}...")
                for i in range(model["count"]):
                    hardware_type = random.choice(["cpu", "cuda"]) if i % 2 == 0 else "cpu"
                    task_data = create_benchmark_task(
                        model_name=model["name"],
                        model_type=model["type"],
                        batch_sizes=[1, 2, 4] if hardware_type == "cuda" else [1],
                        hardware_type=hardware_type,
                        task_id=f"{model['name']}-task-{i+1}"
                    )
                    task_data["priority"] = random.randint(1, 5)
                    submitted_tasks.append(task_data)
                    
                    # Submit task to coordinator
                    await coordinator.submit_task(task_data)
                    logger.info(f"Submitted {model['name']} task {i+1} with ID {task_data['task_id']}")
            
            # Update the total tests count
            test_result.total_tests = total_task_count
            logger.info(f"Total tasks submitted: {total_task_count}")
            
            # Create passed/failed/skipped lists
            test_result.metadata["passed_tests"] = []
            test_result.metadata["failed_tests"] = []
            test_result.metadata["skipped_tests"] = []
            
            # Update the test run status to reflect we're running
            await ci_provider.update_test_run(
                test_run_id,
                {
                    "status": "running",
                    "summary": {
                        "total_tasks": total_task_count,
                        "task_statuses": {"running": total_task_count}
                    }
                }
            )
            
            # Start time for performance tracking
            start_time = time.time()
            
            # Create performance tracking artifact
            worker_discovery_metrics = {
                "discovery_metrics": {
                    "total_workers": len(workers),
                    "worker_capabilities": [w.capabilities for w in workers],
                    "auto_discovery_enabled": True,
                    "auto_registration_enabled": True
                },
                "task_assignment_metrics": {
                    "total_tasks": total_task_count,
                    "task_types": {model["name"]: model["count"] for model in model_types},
                    "expected_distribution": {}
                }
            }
            
            # Create expected distribution based on worker capabilities
            for model in model_types:
                model_name = model["name"]
                worker_discovery_metrics["task_assignment_metrics"]["expected_distribution"][model_name] = []
                for i, worker in enumerate(workers):
                    # Check if worker supports this model
                    if any(m.lower() in model_name.lower() for m in worker.capabilities.get("models", [])):
                        # Worker supports this model, it's a candidate for tasks
                        worker_discovery_metrics["task_assignment_metrics"]["expected_distribution"][model_name].append(
                            {"worker_id": worker.worker_id, "probability": 1.0 / len(workers)}
                        )
            
            # Write metrics to file
            metrics_path = artifacts_dir / "worker_discovery_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(worker_discovery_metrics, f, indent=2)
            
            # Wait for tasks to be processed
            logger.info("Waiting for tasks to complete...")
            
            # In a real example, we would wait for tasks to complete by checking the coordinator
            # For this example, we'll simulate task completion
            await anyio.sleep(5)  # Simulate task processing time
            
            # Create task history artifact
            task_history = {
                "tasks": []
            }
            
            # Simulate task completion
            passed_count = total_task_count - 2  # Simulate two failed tasks
            failed_count = 1
            skipped_count = 1
            
            # Update test result with completion data
            test_result.passed_tests = passed_count
            test_result.failed_tests = failed_count
            test_result.skipped_tests = skipped_count
            test_result.status = "failure" if failed_count > 0 else "success"
            test_result.duration_seconds = time.time() - start_time
            
            # Add sample passed tests
            for i in range(passed_count):
                task_idx = i % len(submitted_tasks)
                task = submitted_tasks[task_idx]
                
                # Create simulated worker assignment
                worker_idx = i % len(workers)
                assigned_worker = workers[worker_idx]
                
                # Add to task history
                task_history["tasks"].append({
                    "task_id": task["task_id"],
                    "status": "completed",
                    "assigned_worker": assigned_worker.worker_id,
                    "worker_capabilities": assigned_worker.capabilities.get("hardware", []),
                    "duration_seconds": random.uniform(0.5, 3.0),
                    "batch_size": random.choice([1, 2, 4]),
                    "model_name": task["model_name"],
                    "model_type": task["model_type"],
                    "hardware_type": task["hardware_type"],
                    "assigned_at": (datetime.now().isoformat(),)
                })
                
                # Add to passed tests
                test_result.metadata["passed_tests"].append({
                    "name": task["task_id"],
                    "duration_seconds": task_history["tasks"][-1]["duration_seconds"],
                    "batch_size": task_history["tasks"][-1]["batch_size"],
                    "hardware": task["hardware_type"],
                    "worker": assigned_worker.worker_id
                })
            
            # Add a sample failed test
            failed_task = submitted_tasks[-1]
            failed_worker = workers[-1]
            
            task_history["tasks"].append({
                "task_id": failed_task["task_id"],
                "status": "failed",
                "assigned_worker": failed_worker.worker_id,
                "worker_capabilities": failed_worker.capabilities.get("hardware", []),
                "error": "CUDA out of memory. Tried to allocate 2.5 GiB but only 1.8 GiB available",
                "duration_seconds": 2.4,
                "batch_size": 1,
                "model_name": failed_task["model_name"],
                "model_type": failed_task["model_type"],
                "hardware_type": failed_task["hardware_type"],
                "assigned_at": (datetime.now().isoformat(),)
            })
            
            test_result.metadata["failed_tests"].append({
                "name": failed_task["task_id"],
                "error": "CUDA out of memory. Tried to allocate 2.5 GiB but only 1.8 GiB available",
                "duration_seconds": 2.4,
                "batch_size": 1,
                "hardware": failed_task["hardware_type"],
                "worker": failed_worker.worker_id
            })
            
            # Add a sample skipped test
            skipped_task = submitted_tasks[-2]
            
            task_history["tasks"].append({
                "task_id": skipped_task["task_id"],
                "status": "skipped",
                "reason": "No compatible worker found with required hardware",
                "required_hardware": "rocm",  # This wasn't available in any worker
                "batch_size": 1,
                "model_name": skipped_task["model_name"],
                "model_type": skipped_task["model_type"]
            })
            
            test_result.metadata["skipped_tests"].append({
                "name": skipped_task["task_id"],
                "reason": "No compatible worker found with required hardware"
            })
            
            # Write task history to file
            task_history_path = artifacts_dir / "task_history.json"
            with open(task_history_path, "w") as f:
                json.dump(task_history, f, indent=2)
            
            # Create worker performance analysis
            worker_performance = {
                "worker_performance": []
            }
            
            for worker in workers:
                worker_tasks = [task for task in task_history["tasks"] 
                              if task.get("assigned_worker") == worker.worker_id]
                
                completed_tasks = [task for task in worker_tasks if task["status"] == "completed"]
                failed_tasks = [task for task in worker_tasks if task["status"] == "failed"]
                
                if not worker_tasks:
                    continue
                
                worker_performance["worker_performance"].append({
                    "worker_id": worker.worker_id,
                    "capabilities": worker.capabilities.get("hardware", []),
                    "assigned_tasks": len(worker_tasks),
                    "completed_tasks": len(completed_tasks),
                    "failed_tasks": len(failed_tasks),
                    "average_duration": sum(task["duration_seconds"] for task in completed_tasks) / max(1, len(completed_tasks)),
                    "task_details": {
                        "by_model": {},
                        "by_hardware": {}
                    }
                })
                
                # Calculate model-specific metrics
                model_counts = {}
                for task in completed_tasks:
                    model_name = task["model_name"]
                    if model_name not in model_counts:
                        model_counts[model_name] = {"count": 0, "duration": 0}
                    model_counts[model_name]["count"] += 1
                    model_counts[model_name]["duration"] += task["duration_seconds"]
                
                for model_name, data in model_counts.items():
                    worker_performance["worker_performance"][-1]["task_details"]["by_model"][model_name] = {
                        "count": data["count"],
                        "average_duration": data["duration"] / data["count"]
                    }
                
                # Calculate hardware-specific metrics
                hardware_counts = {}
                for task in completed_tasks:
                    hardware_type = task["hardware_type"]
                    if hardware_type not in hardware_counts:
                        hardware_counts[hardware_type] = {"count": 0, "duration": 0}
                    hardware_counts[hardware_type]["count"] += 1
                    hardware_counts[hardware_type]["duration"] += task["duration_seconds"]
                
                for hardware_type, data in hardware_counts.items():
                    worker_performance["worker_performance"][-1]["task_details"]["by_hardware"][hardware_type] = {
                        "count": data["count"],
                        "average_duration": data["duration"] / data["count"]
                    }
            
            # Write worker performance to file
            worker_performance_path = artifacts_dir / "worker_performance.json"
            with open(worker_performance_path, "w") as f:
                json.dump(worker_performance, f, indent=2)
            
            # Add performance metrics to test result
            test_result.metadata["performance_metrics"] = {
                "average_throughput": round(passed_count / max(1, test_result.duration_seconds), 2),
                "average_latency_ms": round(sum(task.get("duration_seconds", 0) * 1000 for task in task_history["tasks"]) / len(task_history["tasks"]), 2),
                "memory_usage_mb": 4256,
                "worker_utilization_percent": round(100 * sum(len([t for t in task_history["tasks"] if t.get("assigned_worker") == w.worker_id]) for w in workers) / (max(1, len(workers) * total_task_count)), 2),
                "auto_discovery_time_seconds": 2.0,
                "worker_registration_time_seconds": 0.5
            }
            
            # Collect and upload artifacts
            logger.info("Collecting and uploading artifacts...")
            artifact_patterns = [str(artifacts_dir / "*.json")]
            artifacts = await reporter.collect_and_upload_artifacts(
                test_run_id,
                artifact_patterns
            )
            
            if artifacts:
                logger.info(f"Collected and uploaded {len(artifacts)} artifacts")
                test_result.metadata["artifacts"] = artifacts
            
            # Generate and upload reports
            logger.info("Generating and uploading test reports...")
            report_files = await reporter.report_test_result(
                test_result,
                formats=["markdown", "html", "json"]
            )
            
            for fmt, file_path in report_files.items():
                logger.info(f"Generated {fmt.upper()} report: {file_path}")
            
            # Update final status
            logger.info("Updating test run status...")
            await ci_provider.update_test_run(
                test_run_id,
                {
                    "status": test_result.status,
                    "end_time": datetime.now().isoformat(),
                    "summary": {
                        "total_tests": test_result.total_tests,
                        "passed_tests": test_result.passed_tests,
                        "failed_tests": test_result.failed_tests,
                        "skipped_tests": test_result.skipped_tests,
                        "duration_seconds": test_result.duration_seconds
                    }
                }
            )
            
            logger.info("Example completed successfully")
            logger.info(f"Test run status: {test_result.status}")
            logger.info(f"Tests: {test_result.passed_tests} passed, {test_result.failed_tests} failed, {test_result.skipped_tests} skipped")
            logger.info(f"Reports directory: {reports_dir}")
            logger.info(f"Artifacts directory: {artifacts_dir}")
            
        except Exception as e:
            logger.error(f"Error in example: {str(e)}")
        finally:
            # Clean up workers
            if 'workers' in locals():
                for worker in workers:
                    logger.info(f"Disconnecting worker {worker.worker_id}...")
                    await worker.disconnect()
            
            # Clean up coordinator
            if 'coordinator' in locals():
                logger.info("Shutting down coordinator...")
                await coordinator.shutdown()
            
            # Clean up CI provider
            if 'ci_provider' in locals():
                logger.info("Closing CI provider...")
                await ci_provider.close()


async def main():
    """Main entry point for the example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Worker Auto-Discovery with CI Integration Example")
    parser.add_argument("--ci-provider", default="local", help="CI provider type (github, gitlab, jenkins, etc.)")
    parser.add_argument("--config", help="Path to JSON configuration file for the CI provider")
    parser.add_argument("--workers", type=int, default=2, help="Number of simulated workers to create")
    
    args = parser.parse_args()
    
    provider_type = args.ci_provider
    provider_config = {}
    
    # Load configuration from file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                if provider_type in file_config:
                    provider_config = file_config[provider_type]
                else:
                    provider_config = file_config
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
    
    # Run the example
    await run_example(provider_type, provider_config, args.workers)


if __name__ == "__main__":
    anyio.run(main())