#!/usr/bin/env python3
"""
Test CI Client Implementations

This module tests the implementation of all CI clients to ensure they correctly
implement the standardized CIProviderInterface.
"""

import asyncio
import logging
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import CI client interfaces and implementations
from distributed_testing.ci.api_interface import CIProviderInterface, CIProviderFactory, TestRunResult
from distributed_testing.ci.register_providers import register_all_providers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestCIClientImplementations(unittest.TestCase):
    """Test the implementation of all CI clients."""
    
    def setUp(self):
        """Set up the test environment."""
        # Register all providers
        register_all_providers()
        
        # Create temp directory for artifacts
        self.artifact_dir = Path("./temp_artifacts")
        self.artifact_dir.mkdir(exist_ok=True)
        
        # Create test artifact file
        self.artifact_path = self.artifact_dir / "test_artifact.txt"
        with open(self.artifact_path, "w") as f:
            f.write("This is a test artifact file.")
        
        # Set up test configs for each provider
        self.configs = {
            "github": {
                "token": "fake_github_token",
                "repository": "fake_owner/fake_repo",
                "commit_sha": "abcdef1234567890"
            },
            "gitlab": {
                "token": "fake_gitlab_token",
                "project_id": "12345",
                "ref": "main"
            },
            "jenkins": {
                "url": "https://jenkins.example.com/",
                "user": "fake_jenkins_user",
                "token": "fake_jenkins_token",
                "job_name": "test-job",
                "build_id": "123"
            },
            "azure": {
                "org_url": "https://dev.azure.com/fake-org",
                "project": "fake-project",
                "token": "fake_azure_token",
                "pipeline_id": "123",
                "build_id": "456"
            },
            "circleci": {
                "token": "fake_circleci_token",
                "project_slug": "github/fake-org/fake-repo",
                "build_num": "123"
            },
            "bitbucket": {
                "username": "fake_bitbucket_user",
                "app_password": "fake_bitbucket_password",
                "workspace": "fake-workspace",
                "repository": "fake-repo",
                "commit_hash": "abcdef1234567890"
            },
            "teamcity": {
                "url": "https://teamcity.example.com",
                "username": "fake_teamcity_user",
                "password": "fake_teamcity_password",
                "build_id": "123",
                "build_type_id": "FakeBuildType"
            },
            "travis": {
                "token": "fake_travis_token",
                "repository": "fake-owner/fake-repo",
                "build_id": "123456789"
            }
        }
    
    def tearDown(self):
        """Clean up the test environment."""
        # Remove test artifact file and directory
        if self.artifact_path.exists():
            self.artifact_path.unlink()
        
        if self.artifact_dir.exists():
            self.artifact_dir.rmdir()
    
    async def _test_client_implementation(self, provider_type: str, config: Dict[str, Any]):
        """Test a specific CI client implementation."""
        logger.info(f"Testing {provider_type} client implementation")
        
        try:
            # Create client using factory
            client = await CIProviderFactory.create_provider(provider_type, config)
            
            # Ensure client is an instance of CIProviderInterface
            self.assertIsInstance(client, CIProviderInterface)
            
            # Test create_test_run
            logger.info(f"Testing create_test_run for {provider_type}")
            test_run = await client.create_test_run({
                "name": f"Test Run for {provider_type}",
                "commit_sha": config.get("commit_sha", "abcdef1234567890")
            })
            
            # Verify test run has required fields
            self.assertIsInstance(test_run, dict)
            self.assertIn("id", test_run)
            self.assertIn("name", test_run)
            self.assertIn("status", test_run)
            
            test_run_id = test_run["id"]
            
            # Test update_test_run
            logger.info(f"Testing update_test_run for {provider_type}")
            result = await client.update_test_run(test_run_id, {
                "status": "running",
                "summary": {
                    "total_tests": 10,
                    "passed_tests": 8,
                    "failed_tests": 1,
                    "skipped_tests": 1,
                    "duration": 5.5
                }
            })
            self.assertTrue(result)
            
            # Test upload_artifact
            logger.info(f"Testing upload_artifact for {provider_type}")
            result = await client.upload_artifact(
                test_run_id,
                str(self.artifact_path),
                "test_artifact.txt"
            )
            # Note: Some clients may not fully implement artifact upload,
            # but they should return True for the simulation
            self.assertTrue(result)
            
            # Test get_test_run_status
            logger.info(f"Testing get_test_run_status for {provider_type}")
            status = await client.get_test_run_status(test_run_id)
            self.assertIsInstance(status, dict)
            self.assertIn("id", status)
            self.assertIn("status", status)
            
            # Test set_build_status
            logger.info(f"Testing set_build_status for {provider_type}")
            # Note: This might not be fully implemented in all clients
            await client.set_build_status("success", f"Tests passed for {provider_type}")
            
            # Complete the test run
            logger.info(f"Completing test run for {provider_type}")
            result = await client.update_test_run(test_run_id, {
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "summary": {
                    "total_tests": 10,
                    "passed_tests": 9,
                    "failed_tests": 0,
                    "skipped_tests": 1,
                    "duration": 8.5
                }
            })
            self.assertTrue(result)
            
            # Close the client
            await client.close()
            
            logger.info(f"Successfully tested {provider_type} client implementation")
            return True
            
        except Exception as e:
            logger.error(f"Error testing {provider_type} client: {str(e)}")
            self.fail(f"Error testing {provider_type} client: {str(e)}")
    
    def test_all_client_implementations(self):
        """Test all CI client implementations."""
        async def run_all_tests():
            # Test each provider type
            results = {}
            for provider_type, config in self.configs.items():
                results[provider_type] = await self._test_client_implementation(provider_type, config)
            return results
        
        # Run all tests
        results = asyncio.run(run_all_tests())
        
        # Verify all tests passed
        for provider_type, result in results.items():
            self.assertTrue(result, f"Test failed for {provider_type}")
        
        logger.info("All CI client implementations tested successfully")


if __name__ == "__main__":
    unittest.main()