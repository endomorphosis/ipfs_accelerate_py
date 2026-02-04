#!/usr/bin/env python3
"""
CI Integration with Coordinator and Batch Processing Example

This example demonstrates how to integrate the CI/CD reporting system with the
Coordinator's batch task processing capabilities. It shows how to:

1. Configure the coordinator with batch processing
2. Submit tasks of different types and group them intelligently
3. Report results to CI systems using the TestResultReporter
4. Generate comprehensive reports with performance metrics
5. Collect and upload artifacts

This creates an end-to-end workflow from task creation to result reporting.
"""

import anyio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import from distributed_testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import necessary modules
from test.tests.distributed.distributed_testing.coordinator import DistributedTestingCoordinator
from test.tests.distributed.distributed_testing.worker import Worker
from test.tests.distributed.distributed_testing.create_task import create_benchmark_task
from test.tests.distributed.distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from test.tests.distributed.distributed_testing.ci.result_reporter import TestResultReporter
from test.tests.distributed.distributed_testing.ci.register_providers import register_all_providers


async def run_example(ci_provider_type=None, ci_config=None):
    """
    Run the example with CI integration and batch task processing.
    
    Args:
        ci_provider_type: CI provider type (github, gitlab, etc.)
        ci_config: CI provider configuration
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
            # Create and start coordinator with batch processing enabled
            coordinator = DistributedTestingCoordinator(
                host="127.0.0.1", 
                port=8080,
                db_path=str(db_path),
                enable_batch_processing=True,  # Enable batch processing
                batch_size_limit=5,            # Maximum tasks per batch
                model_grouping=True,           # Group tasks by model
                hardware_grouping=True,        # Group tasks by hardware requirements
                enable_plugins=False           # Don't use plugins for this example
            )
            
            # Start coordinator
            logger.info("Starting coordinator with batch processing...")
            await coordinator.start()
            logger.info("Coordinator started")
            
            # Create and start worker
            worker = Worker(
                coordinator_url="http://127.0.0.1:8080",
                worker_id="example-worker-1",
                capabilities={
                    "hardware": ["cpu", "cuda"],
                    "memory_gb": 32,
                    "models": ["bert", "t5", "vit"]
                },
                max_concurrent_tasks=2        # Worker can run 2 tasks in parallel
            )
            
            # Connect worker to coordinator
            logger.info("Connecting worker to coordinator...")
            await worker.connect()
            logger.info("Worker connected")
            
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
                "name": "Batch Processing Example",
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
                    "Coordinator": "DistributedTestingCoordinator with batch processing",
                    "Worker Count": 1,
                    "Max Concurrent Tasks": 2
                }
            }
            
            # Create sample benchmark tasks for different models
            # Group 1: BERT tasks (will be batched together)
            bert_tasks = []
            logger.info("Creating BERT benchmark tasks...")
            for i in range(3):
                task_data = create_benchmark_task(
                    model_name="bert-base-uncased",
                    model_type="text",
                    batch_sizes=[1, 2, 4],
                    hardware_type="cpu" if i % 2 == 0 else "cuda",
                    task_id=f"bert-task-{i+1}"
                )
                task_data["priority"] = random.randint(1, 5)
                bert_tasks.append(task_data)
                
                # Submit task to coordinator
                await coordinator.submit_task(task_data)
                logger.info(f"Submitted BERT task {i+1} with ID {task_data['task_id']}")
            
            # Group 2: ViT tasks (will be batched together)
            vit_tasks = []
            logger.info("Creating ViT benchmark tasks...")
            for i in range(2):
                task_data = create_benchmark_task(
                    model_name="vit-base-patch16-224",
                    model_type="vision",
                    batch_sizes=[1, 4],
                    hardware_type="cuda",
                    task_id=f"vit-task-{i+1}"
                )
                task_data["priority"] = random.randint(1, 5)
                vit_tasks.append(task_data)
                
                # Submit task to coordinator
                await coordinator.submit_task(task_data)
                logger.info(f"Submitted ViT task {i+1} with ID {task_data['task_id']}")
            
            # Group 3: T5 tasks (will be batched together)
            t5_tasks = []
            logger.info("Creating T5 benchmark tasks...")
            for i in range(2):
                task_data = create_benchmark_task(
                    model_name="t5-small",
                    model_type="text_generation",
                    batch_sizes=[1],
                    hardware_type="cpu" if i % 2 == 0 else "cuda",
                    task_id=f"t5-task-{i+1}"
                )
                task_data["priority"] = random.randint(1, 5)
                t5_tasks.append(task_data)
                
                # Submit task to coordinator
                await coordinator.submit_task(task_data)
                logger.info(f"Submitted T5 task {i+1} with ID {task_data['task_id']}")
            
            # Update the total tests count
            total_tasks = len(bert_tasks) + len(vit_tasks) + len(t5_tasks)
            test_result.total_tests = total_tasks
            logger.info(f"Total tasks submitted: {total_tasks}")
            
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
                        "total_tasks": total_tasks,
                        "task_statuses": {"running": total_tasks}
                    }
                }
            )
            
            # Wait for tasks to be processed
            start_time = time.time()
            logger.info("Waiting for tasks to complete...")
            
            # Create some performance metrics files as artifacts
            await create_sample_artifacts(artifacts_dir, bert_tasks, vit_tasks, t5_tasks)
            
            # Simulate task processing
            # In a real example, we would monitor the task status in the coordinator
            await anyio.sleep(5)  # Simulate task processing time
            
            # Simulate task completion (in a real example, we would get this from the coordinator)
            passed_count = total_tasks - 1  # Simulate one failed task
            failed_count = 1
            skipped_count = 0
            
            # Update test result with completion data
            test_result.passed_tests = passed_count
            test_result.failed_tests = failed_count
            test_result.skipped_tests = skipped_count
            test_result.status = "failure" if failed_count > 0 else "success"
            test_result.duration_seconds = time.time() - start_time
            
            # Add sample passed tests
            for i in range(passed_count):
                task_id = f"task-{i+1}"
                task_type = "bert" if i < 3 else ("vit" if i < 5 else "t5")
                test_result.metadata["passed_tests"].append({
                    "name": f"{task_type}-{task_id}",
                    "duration_seconds": random.uniform(0.5, 3.0),
                    "batch_size": random.choice([1, 2, 4]),
                    "hardware": random.choice(["cpu", "cuda"])
                })
            
            # Add a sample failed test
            test_result.metadata["failed_tests"].append({
                "name": "t5-task-2",
                "error": "CUDA out of memory. Tried to allocate 2.5 GiB but only 1.8 GiB available",
                "duration_seconds": 2.4,
                "batch_size": 1,
                "hardware": "cuda"
            })
            
            # Add performance metrics
            test_result.metadata["performance_metrics"] = {
                "average_throughput": 56.3,
                "average_latency_ms": 17.8,
                "memory_usage_mb": 4256,
                "batch_processing_efficiency": 82.5,
                "task_batching_time_saved_seconds": 7.3
            }
            
            # Collect and upload artifacts
            logger.info("Collecting and uploading artifacts...")
            artifact_patterns = [str(artifacts_dir / "*.json"), str(artifacts_dir / "*.csv")]
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
            # Clean up
            if 'worker' in locals():
                logger.info("Disconnecting worker...")
                await worker.disconnect()
            
            if 'coordinator' in locals():
                logger.info("Shutting down coordinator...")
                await coordinator.shutdown()
            
            if 'ci_provider' in locals():
                logger.info("Closing CI provider...")
                await ci_provider.close()


async def create_sample_artifacts(artifact_dir, bert_tasks, vit_tasks, t5_tasks):
    """Create sample artifacts for the example."""
    # Create performance metrics JSON file
    perf_metrics = {
        "benchmark_summary": {
            "total_tasks": len(bert_tasks) + len(vit_tasks) + len(t5_tasks),
            "batch_processing": {
                "batches_created": 3,
                "average_batch_size": 2.33,
                "time_saved_percent": 42.5
            },
            "model_metrics": {
                "bert-base-uncased": {
                    "avg_throughput": 45.2,
                    "avg_latency_ms": 22.1,
                    "memory_mb": 768
                },
                "vit-base-patch16-224": {
                    "avg_throughput": 28.7,
                    "avg_latency_ms": 34.8,
                    "memory_mb": 1024
                },
                "t5-small": {
                    "avg_throughput": 18.5,
                    "avg_latency_ms": 54.1,
                    "memory_mb": 1536
                }
            }
        }
    }
    
    # Write metrics to file
    metrics_path = artifact_dir / "performance_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(perf_metrics, f, indent=2)
    
    # Create batch efficiency CSV file
    batch_efficiency_path = artifact_dir / "batch_efficiency.csv"
    with open(batch_efficiency_path, "w") as f:
        f.write("model,hardware,batch_size,single_task_ms,batched_task_ms,efficiency_percent\n")
        f.write("bert-base-uncased,cpu,1,45.2,35.6,21.2\n")
        f.write("bert-base-uncased,cpu,2,78.3,58.9,24.8\n")
        f.write("bert-base-uncased,cuda,1,22.1,17.3,21.7\n")
        f.write("vit-base-patch16-224,cuda,1,34.8,27.2,21.8\n")
        f.write("vit-base-patch16-224,cuda,4,112.5,84.7,24.7\n")
        f.write("t5-small,cpu,1,88.3,69.5,21.3\n")
        f.write("t5-small,cuda,1,54.1,41.8,22.7\n")
    
    logger.info(f"Created sample artifacts in {artifact_dir}")
    return [metrics_path, batch_efficiency_path]


async def main():
    """Main entry point for the example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CI Integration with Coordinator and Batch Processing Example")
    parser.add_argument("--ci-provider", default="local", help="CI provider type (github, gitlab, jenkins, etc.)")
    parser.add_argument("--config", help="Path to JSON configuration file for the CI provider")
    
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
    await run_example(provider_type, provider_config)


if __name__ == "__main__":
    anyio.run(main())