#!/usr/bin/env python3
"""
Generic CI/CD Integration Example

This example demonstrates how to use the TestResultReporter with any CI provider implementation.
It shows how to create a provider-agnostic test result reporting workflow that can be used
with any CI/CD system through the standardized interface.
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
from distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.register_providers import register_all_providers


async def run_example(provider_type, provider_config, test_artifacts_dir=None):
    """
    Run the generic CI integration example with the TestResultReporter.
    
    Args:
        provider_type: CI provider type (github, gitlab, jenkins, etc.)
        provider_config: Configuration dictionary for the provider
        test_artifacts_dir: Directory containing test artifacts (optional)
    """
    
    if not provider_type:
        logger.error("Provider type not specified")
        return
    
    # Register all providers with the factory
    register_all_providers()
    
    # Create directories for reports and artifacts
    report_dir = Path("./ci_reports")
    artifact_dir = Path(test_artifacts_dir) if test_artifacts_dir else Path("./ci_artifacts")
    report_dir.mkdir(exist_ok=True)
    artifact_dir.mkdir(exist_ok=True)
    
    # Create some sample test artifacts and result data
    sample_artifacts = create_sample_artifacts(artifact_dir)
    test_result = create_sample_test_result()
    
    try:
        logger.info(f"Creating {provider_type} CI provider...")
        
        # List available providers
        available_providers = CIProviderFactory.get_available_providers()
        logger.info(f"Available providers: {', '.join(available_providers)}")
        
        if provider_type not in available_providers:
            logger.warning(f"Provider type '{provider_type}' not found. Available types: {', '.join(available_providers)}")
            logger.warning("Using simulation mode")
        
        # Create the CI provider
        ci_provider = await CIProviderFactory.create_provider(provider_type, provider_config)
        
        # Create the test result reporter
        reporter = TestResultReporter(
            ci_provider=ci_provider,
            report_dir=str(report_dir),
            artifact_dir=str(artifact_dir)
        )
        
        logger.info("Creating test run...")
        test_run_data = {
            "name": f"Generic {provider_type.title()} Test Run",
            "build_id": f"example-{int(time.time())}",
            "commit_sha": provider_config.get("commit_sha", "HEAD"),
            "branch": provider_config.get("branch", "main")
        }
        
        test_run = await ci_provider.create_test_run(test_run_data)
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
        
        # Add PR comment if PR info provided
        pr_number = provider_config.get("pr_number") or provider_config.get("mr_iid")
        if pr_number:
            pr_label = "PR" if provider_type in ["github", "azure"] else "MR"
            logger.info(f"Adding comment to {pr_label} #{pr_number}...")
            comment_success = await ci_provider.add_pr_comment(
                pr_number,
                f"## Distributed Testing Results\n\n"
                f"Test run completed with status: **{test_result.status.upper()}**\n\n"
                f"- Total Tests: {test_result.total_tests}\n"
                f"- Passed: {test_result.passed_tests}\n"
                f"- Failed: {test_result.failed_tests}\n"
                f"- Skipped: {test_result.skipped_tests}\n\n"
                f"[View Full Report]({test_run.get('url', '#')})"
            )
            logger.info(f"{pr_label} comment {'added successfully' if comment_success else 'failed'}")
        
        # Update final status
        logger.info("Updating test run status...")
        update_success = await ci_provider.update_test_run(
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
        commit_sha = provider_config.get("commit_sha")
        if commit_sha:
            logger.info("Setting build status...")
            status_success = await ci_provider.set_build_status(
                "success" if test_result.failed_tests == 0 else "failure",
                f"Tests: {test_result.passed_tests} passed, {test_result.failed_tests} failed, {test_result.skipped_tests} skipped"
            )
            logger.info(f"Build status {'set successfully' if status_success else 'failed'}")
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
    finally:
        # Clean up CI provider
        if 'ci_provider' in locals():
            await ci_provider.close()


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
        ])
    }
    
    # Write sample files
    for filename, content in report_examples.items():
        file_path = artifact_dir / filename
        with open(file_path, "w") as f:
            f.write(content)
        artifacts.append(file_path)
    
    logger.info(f"Created {len(artifacts)} sample artifacts")
    return artifacts


def create_sample_test_result():
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
    
    return test_result


def detect_ci_environment():
    """
    Detect the CI environment based on environment variables.
    
    Returns:
        Tuple of (provider_type, provider_config) or (None, None) if not detected
    """
    # GitHub Actions
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        logger.info("Detected GitHub Actions environment")
        return "github", {
            "token": os.environ.get('GITHUB_TOKEN'),
            "repository": os.environ.get('GITHUB_REPOSITORY'),
            "commit_sha": os.environ.get('GITHUB_SHA'),
            "pr_number": os.environ.get('GITHUB_PR_NUMBER')
        }
    
    # GitLab CI
    elif os.environ.get('GITLAB_CI') == 'true':
        logger.info("Detected GitLab CI environment")
        return "gitlab", {
            "token": os.environ.get('GITLAB_TOKEN') or os.environ.get('CI_JOB_TOKEN'),
            "project_id": os.environ.get('CI_PROJECT_ID'),
            "commit_sha": os.environ.get('CI_COMMIT_SHA'),
            "mr_iid": os.environ.get('CI_MERGE_REQUEST_IID'),
            "api_url": os.environ.get('CI_API_V4_URL', 'https://gitlab.com/api/v4')
        }
    
    # Azure Pipelines
    elif os.environ.get('TF_BUILD') == 'True':
        logger.info("Detected Azure Pipelines environment")
        return "azure", {
            "organization": os.environ.get('SYSTEM_TEAMFOUNDATIONCOLLECTIONURI'),
            "project": os.environ.get('SYSTEM_TEAMPROJECT'),
            "token": os.environ.get('SYSTEM_ACCESSTOKEN'),
            "build_id": os.environ.get('BUILD_BUILDID'),
            "commit_sha": os.environ.get('BUILD_SOURCEVERSION')
        }
    
    # Jenkins
    elif os.environ.get('JENKINS_URL'):
        logger.info("Detected Jenkins environment")
        return "jenkins", {
            "url": os.environ.get('JENKINS_URL'),
            "job_name": os.environ.get('JOB_NAME'),
            "build_number": os.environ.get('BUILD_NUMBER'),
            "token": os.environ.get('JENKINS_TOKEN')
        }
    
    # CircleCI
    elif os.environ.get('CIRCLECI') == 'true':
        logger.info("Detected CircleCI environment")
        return "circleci", {
            "token": os.environ.get('CIRCLE_TOKEN'),
            "project_slug": f"{os.environ.get('CIRCLE_PROJECT_USERNAME')}/{os.environ.get('CIRCLE_PROJECT_REPONAME')}",
            "build_num": os.environ.get('CIRCLE_BUILD_NUM')
        }
    
    # TravisCI
    elif os.environ.get('TRAVIS') == 'true':
        logger.info("Detected Travis CI environment")
        return "travis", {
            "token": os.environ.get('TRAVIS_API_TOKEN'),
            "repo_slug": os.environ.get('TRAVIS_REPO_SLUG'),
            "build_id": os.environ.get('TRAVIS_BUILD_ID')
        }
    
    # Not in a CI environment or unsupported CI
    logger.info("No CI environment detected, using local mode")
    return "local", {}


async def main():
    """Main entry point for the example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generic CI Integration Example")
    parser.add_argument("--ci-provider", help="CI provider type (github, gitlab, jenkins, etc.)")
    parser.add_argument("--config", help="Path to JSON configuration file for the CI provider")
    parser.add_argument("--artifacts-dir", help="Directory containing test artifacts")
    parser.add_argument("--auto-detect", action="store_true", help="Auto-detect CI environment")
    
    args = parser.parse_args()
    
    provider_type = None
    provider_config = {}
    
    # Auto-detect CI environment if requested
    if args.auto_detect:
        provider_type, provider_config = detect_ci_environment()
    
    # Use command line arguments if provided
    if args.ci_provider:
        provider_type = args.ci_provider
    
    # Load configuration from file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                provider_config.update(file_config)
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
    
    if not provider_type:
        logger.error("No CI provider specified. Use --ci-provider or --auto-detect")
        return
    
    # Run the example
    await run_example(provider_type, provider_config, args.artifacts_dir)


if __name__ == "__main__":
    anyio.run(main())