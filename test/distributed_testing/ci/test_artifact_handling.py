#!/usr/bin/env python3
"""
Test script for standardized artifact handling.

This script tests the artifact handling system with different CI providers.
"""

import asyncio
import logging
import os
import json
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional

from distributed_testing.ci.api_interface import CIProviderInterface, CIProviderFactory
from distributed_testing.ci.github_client import GitHubClient
from distributed_testing.ci.gitlab_client import GitLabClient
from distributed_testing.ci.jenkins_client import JenkinsClient
from distributed_testing.ci.azure_client import AzureDevOpsClient
from distributed_testing.ci.artifact_handler import (
    ArtifactMetadata, 
    ArtifactStorage, 
    ArtifactHandler, 
    get_artifact_handler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create mock CI providers for testing
class MockCIProvider(CIProviderInterface):
    """Mock CI provider for testing."""
    
    def __init__(self, provider_name):
        """Initialize the mock provider."""
        self.provider_name = provider_name
        self.artifacts = {}
        self.test_runs = {}
        self.pr_comments = {}
        self.build_status = {}
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the CI provider with configuration."""
        logger.info(f"Initializing {self.provider_name} provider with config: {config}")
        return True
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new test run."""
        test_run_id = f"{self.provider_name}-test-run-{len(self.test_runs) + 1}"
        self.test_runs[test_run_id] = {
            "id": test_run_id,
            "name": test_run_data.get("name", f"Test Run {test_run_id}"),
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "data": test_run_data
        }
        logger.info(f"Created test run {test_run_id} for {self.provider_name}")
        return self.test_runs[test_run_id]
    
    async def update_test_run(self, test_run_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a test run."""
        if test_run_id not in self.test_runs:
            logger.error(f"Test run {test_run_id} not found for {self.provider_name}")
            return False
        
        self.test_runs[test_run_id].update(update_data)
        logger.info(f"Updated test run {test_run_id} for {self.provider_name}")
        return True
    
    async def add_pr_comment(self, pr_number: str, comment: str) -> bool:
        """Add a comment to a pull request."""
        if pr_number not in self.pr_comments:
            self.pr_comments[pr_number] = []
        
        self.pr_comments[pr_number].append(comment)
        logger.info(f"Added comment to PR {pr_number} for {self.provider_name}")
        return True
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """Upload an artifact for a test run."""
        if test_run_id not in self.artifacts:
            self.artifacts[test_run_id] = []
        
        self.artifacts[test_run_id].append({
            "path": artifact_path,
            "name": artifact_name,
            "upload_time": datetime.now().isoformat()
        })
        
        logger.info(f"Uploaded artifact {artifact_name} for test run {test_run_id} with {self.provider_name}")
        return True
    
    async def get_test_run_status(self, test_run_id: str) -> Dict[str, Any]:
        """Get the status of a test run."""
        if test_run_id not in self.test_runs:
            logger.error(f"Test run {test_run_id} not found for {self.provider_name}")
            return {"status": "unknown", "error": "Test run not found"}
        
        return self.test_runs[test_run_id]
    
    async def set_build_status(self, status: str, description: str) -> bool:
        """Set the build status in the CI system."""
        self.build_status = {
            "status": status,
            "description": description,
            "time": datetime.now().isoformat()
        }
        
        logger.info(f"Set build status to {status} for {self.provider_name}")
        return True
    
    async def close(self) -> None:
        """Close the CI provider and clean up resources."""
        logger.info(f"Closed {self.provider_name} provider")

async def test_artifact_lifecycle():
    """Test artifact lifecycle with different providers."""
    logger.info("Testing artifact lifecycle...")
    
    # Set up test environment
    temp_dir = tempfile.mkdtemp()
    storage_dir = os.path.join(temp_dir, "artifacts")
    os.makedirs(storage_dir, exist_ok=True)
    
    try:
        # Create test artifact files
        log_file_path = os.path.join(temp_dir, "test.log")
        report_file_path = os.path.join(temp_dir, "test_report.json")
        
        with open(log_file_path, "w") as f:
            f.write("Sample log content\nLine 2\nLine 3\n")
        
        with open(report_file_path, "w") as f:
            json.dump({
                "tests": 10,
                "passed": 8,
                "failed": 2
            }, f, indent=2)
        
        # Create artifact handler
        handler = ArtifactHandler(storage_dir)
        
        # Create mock providers
        providers = {
            "github": MockCIProvider("github"),
            "gitlab": MockCIProvider("gitlab"),
            "jenkins": MockCIProvider("jenkins"),
            "azure": MockCIProvider("azure")
        }
        
        # Register providers with handler
        for name, provider in providers.items():
            handler.register_provider(name, provider)
            await provider.initialize({})
        
        # Create test runs for each provider
        test_runs = {}
        for name, provider in providers.items():
            test_run = await provider.create_test_run({
                "name": f"{name} Test Run"
            })
            test_runs[name] = test_run
        
        # Upload artifacts to each provider
        results = {}
        for name, test_run in test_runs.items():
            # Upload log artifact
            log_result = await handler.upload_artifact(
                source_path=log_file_path,
                artifact_name="test.log",
                artifact_type="log",
                test_run_id=test_run["id"],
                provider_name=name,
                store_locally=True
            )
            
            # Upload report artifact
            report_result = await handler.upload_artifact(
                source_path=report_file_path,
                artifact_name="test_report.json",
                artifact_type="report",
                test_run_id=test_run["id"],
                provider_name=name,
                store_locally=True
            )
            
            results[name] = {
                "log": log_result,
                "report": report_result
            }
        
        # Verify results
        for name, result in results.items():
            log_success, log_metadata = result["log"]
            report_success, report_metadata = result["report"]
            
            logger.info(f"Provider {name}:")
            logger.info(f"  Log upload success: {log_success}")
            logger.info(f"  Report upload success: {report_success}")
            
            assert log_success, f"Log upload failed for {name}"
            assert report_success, f"Report upload failed for {name}"
            assert log_metadata is not None, f"Log metadata missing for {name}"
            assert report_metadata is not None, f"Report metadata missing for {name}"
        
        # Verify we can get artifacts
        for name, test_run in test_runs.items():
            artifacts = handler.get_artifacts_for_test_run(test_run["id"])
            logger.info(f"Provider {name} has {len(artifacts)} artifacts")
            assert len(artifacts) == 2, f"Expected 2 artifacts for {name}, found {len(artifacts)}"
            
            log_artifact = handler.get_artifact_by_name(test_run["id"], "test.log")
            assert log_artifact is not None, f"Log artifact not found for {name}"
            assert log_artifact.artifact_type == "log", f"Wrong artifact type for log: {log_artifact.artifact_type}"
            
            report_artifact = handler.get_artifact_by_name(test_run["id"], "test_report.json")
            assert report_artifact is not None, f"Report artifact not found for {name}"
            assert report_artifact.artifact_type == "report", f"Wrong artifact type for report: {report_artifact.artifact_type}"
        
        # Test batch upload
        batch_artifacts = [
            {
                "source_path": log_file_path,
                "artifact_name": "batch_test.log",
                "artifact_type": "log",
                "test_run_id": test_runs["github"]["id"],
                "store_locally": True
            },
            {
                "source_path": report_file_path,
                "artifact_name": "batch_report.json",
                "artifact_type": "report",
                "test_run_id": test_runs["github"]["id"],
                "store_locally": True
            }
        ]
        
        batch_results = await handler.upload_artifacts_batch(batch_artifacts, "github")
        
        logger.info(f"Batch upload results: {len(batch_results)} artifacts")
        assert len(batch_results) == 2, f"Expected 2 batch results, found {len(batch_results)}"
        assert "batch_test.log" in batch_results, "batch_test.log not in batch results"
        assert "batch_report.json" in batch_results, "batch_report.json not in batch results"
        
        # Test artifact purging
        for name, test_run in test_runs.items():
            purge_result = await handler.purge_artifacts_for_test_run(test_run["id"])
            logger.info(f"Purged artifacts for {name}: {purge_result}")
            assert purge_result, f"Failed to purge artifacts for {name}"
            
            # Verify artifacts are gone
            artifacts = handler.get_artifacts_for_test_run(test_run["id"])
            assert len(artifacts) == 0, f"Expected 0 artifacts after purge for {name}, found {len(artifacts)}"
        
        logger.info("Artifact lifecycle tests passed!")
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

async def test_singleton_handler():
    """Test the singleton artifact handler."""
    logger.info("Testing singleton artifact handler...")
    
    # Set up test environment
    temp_dir = tempfile.mkdtemp()
    storage_dir = os.path.join(temp_dir, "artifacts")
    os.makedirs(storage_dir, exist_ok=True)
    
    try:
        # Get the singleton handler
        handler1 = get_artifact_handler(storage_dir)
        handler2 = get_artifact_handler(storage_dir)
        
        # Verify we have the same handler instance
        assert handler1 is handler2, "Singleton handler instances are different"
        logger.info("Singleton handler test passed!")
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

async def main():
    """Run all tests."""
    logger.info("Testing artifact handling...")
    
    # Test artifact lifecycle
    await test_artifact_lifecycle()
    
    # Test singleton handler
    await test_singleton_handler()
    
    logger.info("All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())