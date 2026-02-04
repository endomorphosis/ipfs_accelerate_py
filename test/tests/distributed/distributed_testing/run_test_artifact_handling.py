#!/usr/bin/env python3
"""
Demo script for the standardized artifact handling system.

This script demonstrates how to use the artifact handling system to upload artifacts
to different CI providers and retrieve them later.
"""

import anyio
import logging
import os
import json
import argparse
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional

from test.tests.distributed.distributed_testing.ci.api_interface import CIProviderInterface, CIProviderFactory
from test.tests.distributed.distributed_testing.ci.register_providers import register_all_providers
from test.tests.distributed.distributed_testing.ci.artifact_handler import get_artifact_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def initialize_provider(provider_name: str, config: Dict[str, Any]):
    """
    Initialize a CI provider.
    
    Args:
        provider_name: Name of the provider
        config: Configuration for the provider
    
    Returns:
        Provider instance
    """
    # Register all providers first
    register_all_providers()
    
    # Create provider
    provider = await CIProviderFactory.create_provider(provider_name, config)
    
    return provider

async def create_test_artifacts(temp_dir: str):
    """
    Create test artifact files.
    
    Args:
        temp_dir: Directory to create files in
    
    Returns:
        Dictionary of created files
    """
    # Create various types of artifacts
    artifacts = {}
    
    # Log file
    log_path = os.path.join(temp_dir, "test.log")
    with open(log_path, "w") as f:
        f.write(f"Log file created at {datetime.now().isoformat()}\n")
        f.write("INFO: Test started\n")
        f.write("INFO: Running test case 1\n")
        f.write("INFO: Test case 1 passed\n")
        f.write("INFO: Running test case 2\n")
        f.write("WARN: Test case 2 had a warning\n")
        f.write("INFO: Test case 2 passed\n")
        f.write("INFO: Test completed\n")
    
    artifacts["log"] = {
        "path": log_path,
        "name": "test.log",
        "type": "log"
    }
    
    # Report file (JSON)
    report_path = os.path.join(temp_dir, "test_report.json")
    report_data = {
        "testSuite": "Sample Test Suite",
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": 5,
            "passed": 4,
            "failed": 1,
            "skipped": 0
        },
        "testCases": [
            {"name": "Test Case 1", "status": "passed", "duration": 0.5},
            {"name": "Test Case 2", "status": "passed", "duration": 0.8},
            {"name": "Test Case 3", "status": "passed", "duration": 0.2},
            {"name": "Test Case 4", "status": "failed", "duration": 1.2, "error": "Expected value did not match"},
            {"name": "Test Case 5", "status": "passed", "duration": 0.4}
        ]
    }
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    
    artifacts["report"] = {
        "path": report_path,
        "name": "test_report.json",
        "type": "report"
    }
    
    # Coverage report (HTML)
    coverage_path = os.path.join(temp_dir, "coverage.html")
    with open(coverage_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Coverage Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .passed { color: green; }
        .failed { color: red; }
        table { border-collapse: collapse; width: 100%; }
        td, th { border: 1px solid #ddd; padding: 8px; }
    </style>
</head>
<body>
    <h1>Test Coverage Report</h1>
    <p>Generated on: <span id="date">""" + datetime.now().isoformat() + """</span></p>
    
    <h2>Summary</h2>
    <p>Overall coverage: <strong>85.2%</strong></p>
    
    <table>
        <tr>
            <th>Module</th>
            <th>Coverage</th>
        </tr>
        <tr>
            <td>core.py</td>
            <td>92.5%</td>
        </tr>
        <tr>
            <td>utils.py</td>
            <td>88.3%</td>
        </tr>
        <tr>
            <td>handlers.py</td>
            <td>75.0%</td>
        </tr>
    </table>
</body>
</html>
""")
    
    artifacts["coverage"] = {
        "path": coverage_path,
        "name": "coverage.html",
        "type": "coverage"
    }
    
    # Return all artifacts
    return artifacts

async def run_demo(provider_name: str, config: Dict[str, Any]):
    """
    Run the artifact handling demo.
    
    Args:
        provider_name: Name of the provider to use
        config: Configuration for the provider
    """
    logger.info(f"Running artifact handling demo with {provider_name}...")
    
    # Create a temporary directory for artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create test artifacts
            artifacts = await create_test_artifacts(temp_dir)
            logger.info(f"Created {len(artifacts)} test artifacts")
            
            # Initialize the provider
            provider = await initialize_provider(provider_name, config)
            logger.info(f"Initialized {provider_name} provider")
            
            # Create a test run
            test_run_data = {
                "name": f"Artifact Handling Demo - {datetime.now().isoformat()}"
            }
            
            if provider_name == "github":
                # Add repository-specific data for GitHub
                test_run_data["commit_sha"] = "HEAD"
            
            test_run = await provider.create_test_run(test_run_data)
            test_run_id = test_run.get("id")
            logger.info(f"Created test run: {test_run_id}")
            
            # Get the artifact handler
            artifact_handler = get_artifact_handler()
            
            # Register provider with handler
            artifact_handler.register_provider(provider_name, provider)
            
            # Upload artifacts
            results = {}
            for artifact_type, artifact in artifacts.items():
                logger.info(f"Uploading {artifact_type} artifact: {artifact['name']}")
                
                result = await artifact_handler.upload_artifact(
                    source_path=artifact["path"],
                    artifact_name=artifact["name"],
                    artifact_type=artifact["type"],
                    test_run_id=test_run_id,
                    provider_name=provider_name,
                    store_locally=True
                )
                
                results[artifact_type] = result
            
            # Show results
            logger.info("==== Upload Results ====")
            for artifact_type, (success, metadata) in results.items():
                logger.info(f"{artifact_type}: {'✅ Success' if success else '❌ Failed'}")
                if metadata:
                    logger.info(f"  - Size: {metadata.file_size} bytes")
                    logger.info(f"  - Stored at: {metadata.artifact_path}")
            
            # Get all artifacts for test run
            stored_artifacts = artifact_handler.get_artifacts_for_test_run(test_run_id)
            
            logger.info("==== Stored Artifacts ====")
            for artifact in stored_artifacts:
                logger.info(f"{artifact.artifact_name} ({artifact.artifact_type}):")
                logger.info(f"  - Size: {artifact.file_size} bytes")
                logger.info(f"  - Created: {artifact.creation_time}")
                logger.info(f"  - Hash: {artifact.content_hash}")
            
            # Update test run with results
            await provider.update_test_run(
                test_run_id,
                {
                    "status": "completed",
                    "summary": {
                        "total_tests": 5,
                        "passed_tests": 4,
                        "failed_tests": 1,
                        "skipped_tests": 0,
                        "duration_seconds": 3.1
                    },
                    "end_time": datetime.now().isoformat()
                }
            )
            logger.info(f"Updated test run: {test_run_id}")
            
            # Clean up provider
            await provider.close()
            logger.info(f"Closed {provider_name} provider")
            
            logger.info("Artifact handling demo completed successfully")
        
        except Exception as e:
            logger.error(f"Error in artifact handling demo: {str(e)}")
            raise

async def main():
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(description="Artifact Handling Demo")
    parser.add_argument("--provider", choices=["github", "gitlab", "jenkins", "azure"], default="github",
                        help="CI provider to use")
    parser.add_argument("--token", help="API token for the CI provider")
    parser.add_argument("--repository", help="Repository name (for GitHub, GitLab)")
    parser.add_argument("--project", help="Project name (for Azure)")
    parser.add_argument("--organization", help="Organization name (for Azure)")
    args = parser.parse_args()
    
    # Set up provider-specific config
    config = {
        "token": args.token
    }
    
    if args.provider == "github" and args.repository:
        config["repository"] = args.repository
    elif args.provider == "gitlab" and args.repository:
        config["repository"] = args.repository
    elif args.provider == "azure" and args.organization and args.project:
        config["organization"] = args.organization
        config["project"] = args.project
    
    # Use environment variables if not provided
    if not config.get("token"):
        env_token = os.environ.get(f"{args.provider.upper()}_TOKEN")
        if env_token:
            config["token"] = env_token
    
    if args.provider == "github" and not config.get("repository"):
        env_repo = os.environ.get("GITHUB_REPOSITORY")
        if env_repo:
            config["repository"] = env_repo
    
    # Run the demo
    await run_demo(args.provider, config)

if __name__ == "__main__":
    anyio.run(main())