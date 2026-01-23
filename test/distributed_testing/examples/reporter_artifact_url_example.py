#!/usr/bin/env python3
"""
Example showing how to use the TestResultReporter with the artifact URL retrieval system.

This example demonstrates:
1. Creating a test result
2. Generating reports in multiple formats
3. Uploading and collecting artifacts
4. Retrieving artifact URLs
5. Including artifact URLs in test reports and PR comments
"""

import anyio
import json
import logging
import os
import tempfile
import sys
import argparse
from typing import Dict, Any, Optional, List

# Ensure parent directory is in the path
sys.path.append('/home/barberb/ipfs_accelerate_py/test')

# Import CI system components
from distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.register_providers import register_all_providers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_example(args):
    """Run the example."""
    # Register all CI providers
    register_all_providers()
    
    # Create a CI provider based on arguments
    ci_provider = None
    if args.provider:
        try:
            # Load provider config from file if specified
            if args.config:
                with open(args.config, 'r') as f:
                    config = json.load(f)
            else:
                # Use minimal config for testing
                config = {
                    "token": args.token or "test_token",
                    "repository": args.repository or "test_user/test_repo"
                }
            
            ci_provider = await CIProviderFactory.create_provider(args.provider, config)
            logger.info(f"Created {args.provider} CI provider")
        except Exception as e:
            logger.error(f"Failed to create CI provider: {str(e)}")
            # Continue without CI provider for demo purposes
    
    # Create the test result reporter
    reporter = TestResultReporter(
        ci_provider=ci_provider,
        report_dir=args.report_dir,
        artifact_dir=args.artifact_dir
    )
    
    # Create a unique test run ID
    import uuid
    test_run_id = f"reporter-example-{uuid.uuid4().hex[:8]}"
    
    # Create a test run in the CI system if a provider is available
    if ci_provider:
        try:
            test_run = await ci_provider.create_test_run({
                "name": "Reporter Artifact URL Example",
                "project": "distributed-testing",
                "commit_sha": args.commit_sha or "abcdef1234567890"
            })
            
            if test_run and "id" in test_run:
                test_run_id = test_run["id"]
                logger.info(f"Created test run with ID: {test_run_id}")
        except Exception as e:
            logger.error(f"Failed to create test run: {str(e)}")
    
    # Create a test result
    test_result = TestRunResult(
        test_run_id=test_run_id,
        status="success",
        total_tests=10,
        passed_tests=8,
        failed_tests=1,
        skipped_tests=1,
        duration_seconds=45.6,
        metadata={
            "pr_number": args.pr_number or "123",
            "performance_metrics": {
                "average_throughput": 125.4,
                "average_latency_ms": 7.9,
                "memory_usage_mb": 256
            },
            "environment": {
                "platform": "linux",
                "python_version": "3.9.10",
                "cpu_cores": 4,
                "memory_gb": 8
            }
        }
    )
    
    # Create some test artifacts
    artifacts_to_create = [
        {"name": "test_results.json", "content": json.dumps({"tests": [{"name": "test1", "result": "pass"}]})},
        {"name": "test_metrics.csv", "content": "test,result,duration\ntest1,pass,0.5\ntest2,fail,1.2"},
        {"name": "test_log.txt", "content": "INFO: Starting tests\nERROR: Test2 failed\nINFO: Tests completed"}
    ]
    
    artifact_paths = []
    for artifact in artifacts_to_create:
        # Create a temporary file
        file_path = os.path.join(args.artifact_dir, artifact["name"])
        with open(file_path, "w") as f:
            f.write(artifact["content"])
        artifact_paths.append(file_path)
    
    # Collect and upload artifacts
    if ci_provider:
        logger.info("Collecting and uploading artifacts...")
        artifacts = await reporter.collect_and_upload_artifacts(
            test_run_id=test_run_id,
            artifact_patterns=artifact_paths
        )
        
        # Add artifacts to test result metadata
        if artifacts:
            test_result.metadata["artifacts"] = artifacts
            logger.info(f"Added {len(artifacts)} artifacts to test result metadata")
    
    # Generate reports with the artifact URLs included
    logger.info("Generating and uploading reports...")
    report_files = await reporter.report_test_result(
        test_result,
        formats=["markdown", "html", "json"]
    )
    
    # Display the report file paths
    for fmt, file_path in report_files.items():
        print(f"{fmt.upper()} report: {file_path}")
    
    # Clean up
    if ci_provider:
        await ci_provider.close()
        
    # Print success message
    print(f"\nTest run '{test_run_id}' completed successfully.")
    if "artifacts" in test_result.metadata:
        print(f"\nArtifacts with URLs:")
        for artifact in test_result.metadata["artifacts"]:
            url = artifact.get("url", "No URL available")
            print(f"  - {artifact['name']}: {url}")

def main():
    """Parse arguments and run the example."""
    parser = argparse.ArgumentParser(description="Test Reporter with Artifact URL Retrieval Example")
    
    # CI provider options
    parser.add_argument("--provider", choices=["github", "gitlab", "jenkins", "circleci", "azure", "bitbucket", "teamcity", "travis"], 
                        help="CI provider to use")
    parser.add_argument("--token", help="Token for CI provider authentication")
    parser.add_argument("--repository", help="Repository name (e.g. user/repo)")
    parser.add_argument("--config", help="Path to CI provider configuration file")
    parser.add_argument("--commit-sha", help="Commit SHA for test run")
    parser.add_argument("--pr-number", help="Pull request number for comments")
    
    # Reporter options
    parser.add_argument("--report-dir", default="./reports", help="Directory for reports")
    parser.add_argument("--artifact-dir", default="./artifacts", help="Directory for artifacts")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)
    
    # Run the example
    anyio.run(run_example(args))

if __name__ == "__main__":
    main()