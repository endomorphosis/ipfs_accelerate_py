#!/usr/bin/env python3
"""
GitLab CI/CD Integration Example

This example demonstrates how to use the TestResultReporter and GitLabClient implementation
to report test results to GitLab with multiple output formats, artifact management,
and merge request comments. It shows a practical implementation of the CI/CD integration pipeline.
"""

import anyio
import json
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

# Import CI specific modules
from test.tests.distributed.distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from test.tests.distributed.distributed_testing.ci.gitlab_client import GitLabClient
from test.tests.distributed.distributed_testing.ci.result_reporter import TestResultReporter
from test.tests.distributed.distributed_testing.ci.register_providers import register_all_providers


async def run_example(gitlab_token=None, project_id=None, commit_sha=None, mr_iid=None):
    """
    Run the GitLab CI integration example with the TestResultReporter.
    
    Args:
        gitlab_token: GitLab API token (optional, will use env var if not provided)
        project_id: GitLab project ID (optional, will use env var)
        commit_sha: Commit SHA for status reports (optional)
        mr_iid: Merge request IID for comments (optional)
    """
    
    # Use environment variables if parameters not provided
    gitlab_token = gitlab_token or os.environ.get('GITLAB_TOKEN')
    project_id = project_id or os.environ.get('CI_PROJECT_ID') 
    commit_sha = commit_sha or os.environ.get('CI_COMMIT_SHA')
    mr_iid = mr_iid or os.environ.get('CI_MERGE_REQUEST_IID')
    
    if not gitlab_token:
        logger.warning("GitLab token not provided. Running in demo mode with simulated responses.")
    
    if not project_id and gitlab_token:
        logger.warning("GitLab project ID not provided. Running in demo mode with simulated responses.")
    
    # Register all providers with the factory
    register_all_providers()
    
    # Create directories for reports and artifacts
    report_dir = Path("./ci_reports")
    artifact_dir = Path("./ci_artifacts")
    report_dir.mkdir(exist_ok=True)
    artifact_dir.mkdir(exist_ok=True)
    
    # Create some sample test artifacts and result data
    sample_artifacts = create_sample_artifacts(artifact_dir)
    test_result = create_sample_test_result(mr_iid)
    
    try:
        # Create GitLab client
        gitlab_config = {
            "token": gitlab_token,
            "project_id": project_id,
            "commit_sha": commit_sha,
            "api_url": os.environ.get('CI_API_V4_URL', 'https://gitlab.com/api/v4')
        }
        
        gitlab_client = await CIProviderFactory.create_provider("gitlab", gitlab_config)
        
        # Create the test result reporter
        reporter = TestResultReporter(
            ci_provider=gitlab_client,
            report_dir=str(report_dir),
            artifact_dir=str(artifact_dir)
        )
        
        logger.info("Creating test run in GitLab...")
        test_run_data = {
            "name": "Distributed Testing Example",
            "commit_sha": commit_sha,
            "build_id": f"example-{int(time.time())}"
        }
        
        test_run = await gitlab_client.create_test_run(test_run_data)
        test_result.test_run_id = test_run["id"]
        
        logger.info(f"Created test run with ID: {test_run['id']}")
        
        # Collect artifacts
        if sample_artifacts:
            logger.info("Collecting and uploading artifacts...")
            artifact_patterns = [str(path) for path in sample_artifacts]
            artifacts = await reporter.collect_and_upload_artifacts(
                test_run["id"],
                artifact_patterns
            )
            
            if artifacts:
                logger.info(f"Collected and uploaded {len(artifacts)} artifacts")
                test_result.metadata["artifacts"] = artifacts
        
        # Generate and upload reports in multiple formats
        logger.info("Generating and uploading test reports...")
        report_files = await reporter.report_test_result(
            test_result,
            formats=["markdown", "html", "json"]
        )
        
        for fmt, file_path in report_files.items():
            logger.info(f"Generated {fmt.upper()} report: {file_path}")
        
        # Add comment to MR if MR IID provided
        if mr_iid:
            logger.info(f"Adding comment to MR !{mr_iid}...")
            comment_success = await gitlab_client.add_pr_comment(
                mr_iid,
                f"## Distributed Testing Results\n\n"
                f"Test run completed with status: **{test_result.status.upper()}**\n\n"
                f"- Total Tests: {test_result.total_tests}\n"
                f"- Passed: {test_result.passed_tests}\n"
                f"- Failed: {test_result.failed_tests}\n"
                f"- Skipped: {test_result.skipped_tests}\n\n"
                f"[View Full Report]({test_run.get('url', '#')})"
            )
            logger.info(f"MR comment {'added successfully' if comment_success else 'failed'}")
        
        # Update final status
        logger.info("Updating test run status...")
        update_success = await gitlab_client.update_test_run(
            test_run["id"],
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
        logger.info(f"Status update {'succeeded' if update_success else 'failed'}")
        
        # Set build status if commit SHA provided
        if commit_sha:
            logger.info("Setting build status...")
            status_success = await gitlab_client.set_build_status(
                "success" if test_result.failed_tests == 0 else "failed",
                f"Tests: {test_result.passed_tests} passed, {test_result.failed_tests} failed, {test_result.skipped_tests} skipped"
            )
            logger.info(f"Build status {'set successfully' if status_success else 'failed'}")
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
    finally:
        # Clean up GitLab client
        if 'gitlab_client' in locals():
            await gitlab_client.close()


def create_sample_artifacts(artifact_dir):
    """Create sample artifacts for the example."""
    artifacts = []
    
    # Create sample report files
    report_examples = {
        "performance_metrics.json": json.dumps({
            "throughput": 124.5,
            "latency_ms": 8.7,
            "memory_mb": 512,
            "startup_time_ms": 350,
            "inference_time_ms": 7.2
        }, indent=2),
        
        "benchmark_summary.txt": "\n".join([
            "# Benchmark Summary",
            "",
            "Model: bert-base-uncased",
            "Hardware: NVIDIA A100",
            "Batch Size: 16",
            "",
            "- Average Throughput: 124.5 items/sec",
            "- p50 Latency: 7.8 ms",
            "- p95 Latency: 9.2 ms",
            "- p99 Latency: 12.5 ms",
            "- Memory Usage: 512 MB",
            "",
            "Compared to baseline:",
            "- Throughput: +15.3%",
            "- p50 Latency: -8.2%",
            "- Memory Usage: -5.1%"
        ]),
        
        "error_log.txt": "\n".join([
            "2025-03-15 08:12:34 ERROR: Failed to initialize model 'llama-7b' - CUDA out of memory",
            "2025-03-15 08:12:35 ERROR: Stack trace:",
            "  File 'model_loader.py', line 123, in initialize_model",
            "    model = Model.from_pretrained(model_name)",
            "  File 'base_model.py', line 45, in from_pretrained",
            "    return cls._load_weights(config, weights_path)",
            "2025-03-15 08:12:36 INFO: Attempting fallback to CPU",
            "2025-03-15 08:12:40 WARNING: CPU fallback will be significantly slower"
        ]),
        
        "coverage.xml": """<?xml version="1.0" ?>
<coverage version="5.5" timestamp="1615754444" lines-valid="145" lines-covered="125" line-rate="0.862" branches-valid="32" branches-covered="24" branch-rate="0.75">
    <packages>
        <package name="distributed_testing" line-rate="0.862" branch-rate="0.75">
            <classes>
                <class name="coordinator.py" filename="coordinator.py" line-rate="0.93" branch-rate="0.85">
                    <methods/>
                    <lines>
                        <line number="1" hits="1"/>
                        <line number="2" hits="1"/>
                        <line number="3" hits="1"/>
                    </lines>
                </class>
                <class name="worker.py" filename="worker.py" line-rate="0.82" branch-rate="0.70">
                    <methods/>
                    <lines>
                        <line number="1" hits="1"/>
                        <line number="2" hits="1"/>
                        <line number="3" hits="0"/>
                    </lines>
                </class>
            </classes>
        </package>
    </packages>
</coverage>
"""
    }
    
    # Write sample files
    for filename, content in report_examples.items():
        file_path = artifact_dir / filename
        with open(file_path, "w") as f:
            f.write(content)
        artifacts.append(file_path)
    
    logger.info(f"Created {len(artifacts)} sample artifacts")
    return artifacts


def create_sample_test_result(mr_iid=None):
    """Create a sample test result for the example."""
    
    # Basic test run information
    test_result = TestRunResult(
        test_run_id="pending",  # Will be filled in later
        status="success",
        total_tests=42,
        passed_tests=38,
        failed_tests=2,
        skipped_tests=2,
        duration_seconds=125.7
    )
    
    # Add detailed metadata
    test_result.metadata = {
        "test_details": True,
        "performance_metrics": {
            "average_throughput": 124.5,
            "average_latency_ms": 8.7,
            "memory_usage_mb": 512,
            "cpu_utilization_percent": 78.3,
            "gpu_utilization_percent": 92.5,
            "power_consumption_watts": 285
        },
        "environment": {
            "Python Version": sys.version.split()[0],
            "Platform": sys.platform,
            "CUDA Version": "12.1",
            "GPU": "NVIDIA A100",
            "CPU": "AMD EPYC 7543 32-Core Processor",
            "Memory": "128GB"
        },
        "coverage": {
            "line_rate": 0.862,
            "branch_rate": 0.75,
            "lines_covered": 125,
            "lines_valid": 145
        }
    }
    
    # Add failed tests
    test_result.metadata["failed_tests"] = [
        {
            "name": "test_model_inference_large_batch",
            "error": "CUDA out of memory. Tried to allocate 8.5 GiB but only 4.2 GiB available",
            "duration_seconds": 3.2
        },
        {
            "name": "test_model_quantization_int4",
            "error": "Assertion failed: Expected output tensor shape to be [1, 768] but got [1, 384]",
            "duration_seconds": 2.8
        }
    ]
    
    # Add passed tests (just a sample)
    test_result.metadata["passed_tests"] = [
        {"name": "test_model_load", "duration_seconds": 1.2},
        {"name": "test_model_inference_small_batch", "duration_seconds": 2.5},
        {"name": "test_model_save", "duration_seconds": 0.8},
        {"name": "test_model_tokenization", "duration_seconds": 0.3},
        {"name": "test_model_embedding", "duration_seconds": 1.7}
    ]
    
    # Add MR information if provided
    if mr_iid:
        test_result.metadata["pr_number"] = mr_iid  # Using pr_number as the key for consistency
    
    return test_result


async def main():
    """Main entry point for the example."""
    # Get credentials from environment if available (GitLab CI environment variables)
    gitlab_token = os.environ.get('GITLAB_TOKEN') or os.environ.get('CI_JOB_TOKEN')
    project_id = os.environ.get('CI_PROJECT_ID')
    commit_sha = os.environ.get('CI_COMMIT_SHA')
    mr_iid = os.environ.get('CI_MERGE_REQUEST_IID')
    
    if gitlab_token and project_id:
        logger.info(f"Running example with actual GitLab integration for project {project_id}")
    else:
        logger.info("Running example in simulation mode (no GitLab credentials provided)")
        logger.info("Set GITLAB_TOKEN and CI_PROJECT_ID environment variables for actual GitLab integration")
    
    await run_example(gitlab_token, project_id, commit_sha, mr_iid)


if __name__ == "__main__":
    anyio.run(main())