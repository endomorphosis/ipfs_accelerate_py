#!/usr/bin/env python3
"""
Enhanced CI/CD Integration Example

This example demonstrates how to use the enhanced CI/CD Integration Plugin with the Distributed Testing Framework.
It shows how to create and manage test runs, work with artifacts, add PR comments, and analyze performance trends.
It showcases the standardized API architecture and the history tracking features.
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import from distributed_testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from distributed_testing.plugin_architecture import Plugin, PluginType, HookType
from distributed_testing.coordinator import DistributedTestingCoordinator
from distributed_testing.task_scheduler import Task


async def main():
    """Run the example."""
    logger.info("Starting Enhanced CI/CD integration example")
    
    # Create temporary directory for artifacts
    artifact_dir = Path("./test_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    
    # Create example HTML report
    html_report_path = artifact_dir / "test_report.html"
    with open(html_report_path, "w") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .success { color: green; }
                .failure { color: red; }
            </style>
        </head>
        <body>
            <h1>Test Report</h1>
            <p>Generated on: <span id="timestamp"></span></p>
            <script>
                document.getElementById("timestamp").textContent = new Date().toLocaleString();
            </script>
            <h2>Test Results</h2>
            <table border="1" cellpadding="5">
                <tr>
                    <th>Test</th>
                    <th>Status</th>
                    <th>Duration</th>
                </tr>
                <tr>
                    <td>Model Test 1</td>
                    <td class="success">PASS</td>
                    <td>1.23s</td>
                </tr>
                <tr>
                    <td>Model Test 2</td>
                    <td class="success">PASS</td>
                    <td>0.87s</td>
                </tr>
                <tr>
                    <td>Model Test 3</td>
                    <td class="failure">FAIL</td>
                    <td>2.15s</td>
                </tr>
            </table>
        </body>
        </html>
        """)
    
    try:
        # Create coordinator with plugin support
        coordinator = DistributedTestingCoordinator(
            db_path=":memory:",  # Use in-memory database for example
            enable_plugins=True,
            plugin_dirs=["distributed_testing/plugins"]
        )
        
        # Configure CI/CD integration plugin
        ci_plugin_config = {
            "ci_system": "local",  # Use local mode for example
            "artifact_dir": str(artifact_dir),
            "enable_history_tracking": True,
            "history_retention_days": 30,
            "track_performance_trends": True,
            "result_format": "all",
            "detailed_logging": True
        }
        
        # Start coordinator
        await coordinator.start()
        
        # Find CI/CD integration plugin
        ci_plugin = None
        for plugin_id, plugin in coordinator.plugin_manager.get_plugins_by_type(PluginType.INTEGRATION).items():
            if "CICDIntegration" in plugin_id:
                ci_plugin = plugin
                break
        
        if not ci_plugin:
            logger.error("CI/CD integration plugin not found")
            return
        
        # Configure plugin
        await coordinator.plugin_manager.configure_plugin(ci_plugin.id, ci_plugin_config)
        
        # Get CI client
        if hasattr(ci_plugin, 'ci_client') and hasattr(ci_plugin.ci_client, 'create_test_run'):
            ci_client = ci_plugin.ci_client
        else:
            logger.error("CI client not available")
            return
        
        # Create test run
        logger.info("Creating test run")
        test_run = await ci_client.create_test_run({
            "name": "Example Test Run",
            "build_id": f"example-{int(time.time())}",
            "commit_sha": "abcdef123456789",
            "branch": "main",
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown"
            }
        })
        
        logger.info(f"Created test run: {test_run['id']}")
        
        # Simulate tasks
        task_ids = []
        
        # Create tasks for different model types to demonstrate trend analysis
        model_types = ["text", "vision", "audio"]
        
        for i in range(15):
            task_id = f"task-{i+1}"
            model_type = model_types[i % len(model_types)]
            task_data = {
                "name": f"Example {model_type.capitalize()} Task {i+1}",
                "type": f"{model_type}_model_test",
                "model": f"{model_type}-model-{i%3+1}"
            }
            
            # Create task
            await coordinator.plugin_manager.invoke_hook(
                HookType.TASK_CREATED,
                task_id,
                task_data
            )
            
            task_ids.append(task_id)
            
            # Simulate task execution
            await asyncio.sleep(0.1)
            
            # Complete or fail task (fail every 5th task)
            if i % 5 != 4:
                await coordinator.plugin_manager.invoke_hook(
                    HookType.TASK_COMPLETED,
                    task_id,
                    {"status": "completed", "result": f"Result for task {i+1}"}
                )
                
                # Record performance metrics
                # - execution_time: how long the task took
                # - memory_usage: memory usage in MB
                # - accuracy: model accuracy (for completed tasks only)
                await ci_client.record_performance_metric(
                    test_run_id=test_run['id'],
                    task_id=task_id,
                    metric_name="execution_time",
                    metric_value=0.5 + (i % 3) * 0.2,
                    unit="seconds"
                )
                
                await ci_client.record_performance_metric(
                    test_run_id=test_run['id'],
                    task_id=task_id,
                    metric_name="memory_usage",
                    metric_value=100 + (i % 3) * 50,
                    unit="MB"
                )
                
                await ci_client.record_performance_metric(
                    test_run_id=test_run['id'],
                    task_id=task_id,
                    metric_name="accuracy",
                    metric_value=0.9 - (i % 3) * 0.05,
                    unit="pct"
                )
            else:
                await coordinator.plugin_manager.invoke_hook(
                    HookType.TASK_FAILED,
                    task_id,
                    f"Simulated failure in task {i+1}: Out of memory error"
                )
        
        # Upload artifact
        logger.info("Uploading artifact")
        success = await ci_client.upload_artifact(
            test_run['id'],
            str(html_report_path),
            "Test Report"
        )
        logger.info(f"Artifact upload {'succeeded' if success else 'failed'}")
        
        # Update test run status
        logger.info("Updating test run")
        success = await ci_client.update_test_run(
            test_run['id'],
            {
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "summary": {
                    "total_tasks": len(task_ids),
                    "task_statuses": {"completed": len(task_ids) - 3, "failed": 3},
                    "duration": 5.5
                }
            }
        )
        logger.info(f"Test run update {'succeeded' if success else 'failed'}")
        
        # Create two more test runs to demonstrate history tracking
        for run_num in range(2):
            logger.info(f"Creating additional test run {run_num+2}")
            additional_run = await ci_client.create_test_run({
                "name": f"Example Test Run {run_num+2}",
                "build_id": f"example-{int(time.time())}-{run_num+2}",
                "commit_sha": f"abcdef{run_num+2}",
                "branch": "feature/new-tests" if run_num == 0 else "main",
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown"
                }
            })
            
            # Add a few tasks to this run
            for i in range(5):
                task_id = f"run{run_num+2}-task-{i+1}"
                model_type = model_types[i % len(model_types)]
                task_data = {
                    "name": f"Example {model_type.capitalize()} Task {i+1}",
                    "type": f"{model_type}_model_test",
                    "model": f"{model_type}-model-{i%3+1}"
                }
                
                # Create and complete task
                await coordinator.plugin_manager.invoke_hook(
                    HookType.TASK_CREATED,
                    task_id,
                    task_data
                )
                
                await asyncio.sleep(0.1)
                
                await coordinator.plugin_manager.invoke_hook(
                    HookType.TASK_COMPLETED,
                    task_id,
                    {"status": "completed", "result": f"Result for run {run_num+2}, task {i+1}"}
                )
                
                # Record performance metrics with slight variations to show trends
                # - execution_time: progressively improving (lower is better)
                # - memory_usage: progressively improving (lower is better)
                # - accuracy: progressively improving (higher is better)
                await ci_client.record_performance_metric(
                    test_run_id=additional_run['id'],
                    task_id=task_id,
                    metric_name="execution_time",
                    metric_value=0.5 + (i % 3) * 0.2 - run_num * 0.1,  # Gets faster in newer runs
                    unit="seconds"
                )
                
                await ci_client.record_performance_metric(
                    test_run_id=additional_run['id'],
                    task_id=task_id,
                    metric_name="memory_usage",
                    metric_value=100 + (i % 3) * 50 - run_num * 10,  # Uses less memory in newer runs
                    unit="MB"
                )
                
                await ci_client.record_performance_metric(
                    test_run_id=additional_run['id'],
                    task_id=task_id,
                    metric_name="accuracy",
                    metric_value=0.9 - (i % 3) * 0.05 + run_num * 0.03,  # Gets more accurate in newer runs
                    unit="pct"
                )
            
            # Complete this run
            await ci_client.update_test_run(
                additional_run['id'],
                {
                    "status": "completed",
                    "end_time": datetime.now().isoformat(),
                    "summary": {
                        "total_tasks": 5,
                        "task_statuses": {"completed": 5, "failed": 0},
                        "duration": 3.0
                    }
                }
            )
        
        # Get test history
        logger.info("Getting test history")
        history = await ci_client.get_test_history(limit=5)
        logger.info(f"Found {len(history)} test runs in history")
        
        for idx, run in enumerate(history):
            logger.info(f"  Run {idx+1}: {run.get('id')} - {run.get('status')} - Branch: {run.get('branch')}")
        
        # Get performance metrics for execution time
        logger.info("Getting performance metrics for execution time")
        metrics = await ci_client.get_performance_metrics(
            metric_name="execution_time"
        )
        logger.info(f"Found {len(metrics)} execution time metrics")
        
        # Analyze performance trends by branch
        logger.info("Analyzing performance trends by branch")
        branch_trends = await ci_client.analyze_performance_trends(
            metric_name="execution_time",
            grouping="branch",
            timeframe="all"
        )
        
        if "error" not in branch_trends:
            logger.info(f"Performance trend analysis by branch completed:")
            logger.info(f"  Metric: {branch_trends.get('metric_name')}")
            logger.info(f"  Unit: {branch_trends.get('unit')}")
            logger.info(f"  Groups: {len(branch_trends.get('groups', []))}")
            
            for group in branch_trends.get('groups', []):
                logger.info(f"  Branch {group['name']}: avg={group['avg']:.2f} {branch_trends.get('unit', '')}, min={group['min']:.2f}, max={group['max']:.2f}")
                
            if branch_trends.get('overall'):
                logger.info(f"  Overall average: {branch_trends['overall']['avg']:.2f} {branch_trends.get('unit', '')}")
        else:
            logger.warning(f"Performance trend analysis by branch failed: {branch_trends.get('error')}")
        
        # Analyze performance trends by task type
        logger.info("Analyzing performance trends by task type")
        type_trends = await ci_client.analyze_performance_trends(
            metric_name="execution_time",
            grouping="task_type",
            timeframe="all"
        )
        
        if "error" not in type_trends:
            logger.info(f"Performance trend analysis by task type completed:")
            logger.info(f"  Metric: {type_trends.get('metric_name')}")
            logger.info(f"  Unit: {type_trends.get('unit')}")
            logger.info(f"  Groups: {len(type_trends.get('groups', []))}")
            
            for group in type_trends.get('groups', []):
                logger.info(f"  Type {group['name']}: avg={group['avg']:.2f} {type_trends.get('unit', '')}, min={group['min']:.2f}, max={group['max']:.2f}")
                
            if type_trends.get('overall'):
                logger.info(f"  Overall average: {type_trends['overall']['avg']:.2f} {type_trends.get('unit', '')}")
        else:
            logger.warning(f"Performance trend analysis by task type failed: {type_trends.get('error')}")
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
    finally:
        # Clean up
        if 'coordinator' in locals():
            await coordinator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())