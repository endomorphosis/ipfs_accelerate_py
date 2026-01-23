#!/usr/bin/env python3
"""
Test Script for TestResultReporter Integration with Artifact URL Retrieval

This script tests the complete integration between TestResultReporter and the
artifact URL retrieval system across different CI providers.
"""

import anyio
import json
import logging
import os
import tempfile
import sys
from typing import Dict, Any, Optional, List
import unittest

# Add the parent directory to the path
sys.path.append('/home/barberb/ipfs_accelerate_py/test')

from distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.register_providers import register_all_providers

# Import mock CI providers for testing
from distributed_testing.test_artifact_url_retrieval import (
    MockGitHubClient, 
    MockJenkinsClient,
    MockCircleCIClient,
    MockAzureDevOpsClient,
    MockBitbucketClient,
    MockTeamCityClient,
    MockTravisClient
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestReporterArtifactIntegration(unittest.TestCase):
    """Test the TestResultReporter integration with artifact URL retrieval."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for reports and artifacts
        self.report_dir = tempfile.mkdtemp()
        self.artifact_dir = tempfile.mkdtemp()
        
        # Register all providers
        register_all_providers()
        
        # Map provider classes to their names for testing
        self.provider_map = {
            "github": MockGitHubClient,
            "jenkins": MockJenkinsClient,
            "circleci": MockCircleCIClient,
            "azure": MockAzureDevOpsClient,
            "bitbucket": MockBitbucketClient,
            "teamcity": MockTeamCityClient,
            "travis": MockTravisClient
        }
        
        # Create temporary test artifacts
        self.artifact_files = []
        self.create_test_artifacts()

    def tearDown(self):
        """Clean up temporary files and directories."""
        # Remove artifact files
        for file_path in self.artifact_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        
        # Remove temporary directories
        os.rmdir(self.report_dir)
        os.rmdir(self.artifact_dir)

    def create_test_artifacts(self):
        """Create temporary test artifacts for testing."""
        artifacts_to_create = [
            {"name": "test_results.json", "content": json.dumps({"tests": [{"name": "test1", "result": "pass"}]})},
            {"name": "performance.csv", "content": "metric,value\nthroughput,125.4\nlatency,7.9"},
            {"name": "test.log", "content": "INFO: Starting tests\nERROR: Test2 failed\nINFO: Tests completed"}
        ]
        
        for artifact in artifacts_to_create:
            file_path = os.path.join(self.artifact_dir, artifact["name"])
            with open(file_path, "w") as f:
                f.write(artifact["content"])
            self.artifact_files.append(file_path)

    async def test_with_provider(self, provider_name):
        """Test the TestResultReporter integration with a specific provider."""
        logger.info(f"Testing TestResultReporter integration with {provider_name}...")
        
        # Create mock CI provider
        provider_class = self.provider_map[provider_name]
        provider = provider_class()
        await provider.initialize({})
        
        # Create test result reporter
        reporter = TestResultReporter(
            ci_provider=provider,
            report_dir=self.report_dir,
            artifact_dir=self.artifact_dir
        )
        
        # Create a test result
        test_result = TestRunResult(
            test_run_id=f"{provider_name}-test-123",
            status="success",
            total_tests=10,
            passed_tests=8,
            failed_tests=1,
            skipped_tests=1,
            duration_seconds=45.6,
            metadata={
                "pr_number": "123",
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
        
        # Test bulk URL retrieval method
        logger.info("Testing get_artifact_urls method...")
        artifact_names = [os.path.basename(file_path) for file_path in self.artifact_files]
        
        # Upload artifacts manually first
        for file_path in self.artifact_files:
            artifact_name = os.path.basename(file_path)
            success = await provider.upload_artifact(
                test_run_id=test_result.test_run_id,
                artifact_path=file_path,
                artifact_name=artifact_name
            )
            self.assertTrue(success, f"Failed to upload artifact {artifact_name}")
        
        # Get artifact URLs in bulk
        urls = await reporter.get_artifact_urls(
            test_run_id=test_result.test_run_id,
            artifact_names=artifact_names
        )
        
        # Verify that we got URLs for all artifacts
        self.assertEqual(len(urls), len(artifact_names), 
                         f"Expected {len(artifact_names)} URLs, got {len(urls)}")
        
        for name in artifact_names:
            self.assertIn(name, urls, f"Missing URL for artifact {name}")
            self.assertIsNotNone(urls[name], f"URL for {name} is None")
            self.assertIn(name, urls[name], f"URL doesn't contain artifact name: {urls[name]}")
        
        logger.info("Bulk URL retrieval test passed")
        
        # Test collect_and_upload_artifacts with automatic URL retrieval
        logger.info("Testing collect_and_upload_artifacts with automatic URL retrieval...")
        artifacts = await reporter.collect_and_upload_artifacts(
            test_run_id=f"{provider_name}-test-456",
            artifact_patterns=self.artifact_files
        )
        
        # Verify that artifacts were collected and URLs were retrieved
        self.assertEqual(len(artifacts), len(self.artifact_files), 
                         f"Expected {len(self.artifact_files)} artifacts, got {len(artifacts)}")
        
        for artifact in artifacts:
            self.assertIn("name", artifact, "Artifact missing 'name' field")
            self.assertIn("path", artifact, "Artifact missing 'path' field")
            self.assertIn("size_bytes", artifact, "Artifact missing 'size_bytes' field")
            self.assertIn("url", artifact, "Artifact missing 'url' field")
            self.assertIsNotNone(artifact["url"], f"URL for {artifact['name']} is None")
            
        logger.info("collect_and_upload_artifacts test with automatic URL retrieval passed")
        
        # Test report_test_result with artifact URLs
        logger.info("Testing report_test_result with artifact URLs...")
        test_result.metadata["artifacts"] = artifacts
        
        # Generate reports
        report_files = await reporter.report_test_result(
            test_result,
            formats=["markdown", "html", "json"]
        )
        
        # Verify reports were generated
        self.assertIn("markdown", report_files, "Markdown report not generated")
        self.assertIn("html", report_files, "HTML report not generated")
        self.assertIn("json", report_files, "JSON report not generated")
        
        # Check markdown report for artifact URLs
        with open(report_files["markdown"], "r") as f:
            markdown_content = f.read()
            
        self.assertIn("## Artifacts", markdown_content, "Markdown report missing Artifacts section")
        for artifact in artifacts:
            self.assertIn(artifact["name"], markdown_content, 
                         f"Markdown report missing artifact {artifact['name']}")
            self.assertIn(artifact["url"], markdown_content, 
                         f"Markdown report missing URL for {artifact['name']}")
        
        # Check HTML report for artifact URLs
        with open(report_files["html"], "r") as f:
            html_content = f.read()
            
        self.assertIn("<h2>Artifacts</h2>", html_content, "HTML report missing Artifacts section")
        for artifact in artifacts:
            self.assertIn(artifact["name"], html_content, 
                         f"HTML report missing artifact {artifact['name']}")
            self.assertIn(artifact["url"], html_content, 
                         f"HTML report missing URL for {artifact['name']}")
        
        # Check JSON report for artifact URLs
        with open(report_files["json"], "r") as f:
            json_content = json.load(f)
            
        self.assertIn("metadata", json_content, "JSON report missing metadata")
        self.assertIn("artifacts", json_content["metadata"], "JSON report missing artifacts in metadata")
        
        json_artifacts = json_content["metadata"]["artifacts"]
        self.assertEqual(len(json_artifacts), len(artifacts), 
                         f"JSON report has {len(json_artifacts)} artifacts, expected {len(artifacts)}")
        
        for i, artifact in enumerate(artifacts):
            self.assertEqual(artifact["name"], json_artifacts[i]["name"], 
                            f"JSON report artifact name mismatch: {artifact['name']} vs {json_artifacts[i]['name']}")
            self.assertEqual(artifact["url"], json_artifacts[i]["url"], 
                            f"JSON report artifact URL mismatch: {artifact['url']} vs {json_artifacts[i]['url']}")
        
        logger.info("report_test_result test with artifact URLs passed")
        
        # Test integration with PR comments
        logger.info("Testing artifact URL integration with PR comments...")
        
        # Mock the add_pr_comment method to capture the comment
        pr_comment = None
        
        original_add_pr_comment = provider.add_pr_comment
        
        async def mock_add_pr_comment(pr_number, comment):
            nonlocal pr_comment
            pr_comment = comment
            return True
        
        provider.add_pr_comment = mock_add_pr_comment
        
        # Generate report with PR comment
        await reporter.report_test_result(
            test_result,
            formats=["markdown"]
        )
        
        # Restore original method
        provider.add_pr_comment = original_add_pr_comment
        
        # Verify the PR comment contains artifact URLs
        self.assertIsNotNone(pr_comment, "PR comment not generated")
        self.assertIn("## Artifacts", pr_comment, "PR comment missing Artifacts section")
        
        for artifact in artifacts:
            self.assertIn(artifact["name"], pr_comment, 
                         f"PR comment missing artifact {artifact['name']}")
            self.assertIn(artifact["url"], pr_comment, 
                         f"PR comment missing URL for {artifact['name']}")
        
        logger.info("PR comment integration with artifact URLs passed")
        
        # Test report artifacts also get their URLs
        logger.info("Testing report artifacts also get their URLs...")
        
        # Get test result with updated artifacts
        updated_result = TestRunResult(
            test_run_id=test_result.test_run_id,
            status=test_result.status,
            total_tests=test_result.total_tests,
            passed_tests=test_result.passed_tests,
            failed_tests=test_result.failed_tests,
            skipped_tests=test_result.skipped_tests,
            duration_seconds=test_result.duration_seconds,
            metadata=test_result.metadata
        )
        
        # Generate reports again to capture report artifacts
        report_files = await reporter.report_test_result(
            updated_result,
            formats=["markdown", "html", "json"]
        )
        
        # Check final artifacts
        self.assertIn("artifacts", updated_result.metadata, "Result metadata missing artifacts")
        
        # Count report artifacts
        report_artifacts = [a for a in updated_result.metadata["artifacts"] 
                            if a.get("type") == "report"]
        
        # Should have 3 report artifacts (markdown, html, json)
        self.assertEqual(len(report_artifacts), 3, 
                         f"Expected 3 report artifacts, got {len(report_artifacts)}")
        
        # Check that each report has a URL
        for report_artifact in report_artifacts:
            self.assertIn("url", report_artifact, f"Report artifact missing URL: {report_artifact}")
            self.assertIsNotNone(report_artifact["url"], 
                               f"URL for report {report_artifact['name']} is None")
        
        logger.info("Report artifacts with URLs test passed")
        
        # Test with missing artifact
        logger.info("Testing get_artifact_urls with missing artifact...")
        
        # Try to get a URL for a non-existent artifact
        urls = await reporter.get_artifact_urls(
            test_run_id=test_result.test_run_id,
            artifact_names=["non_existent.json"]
        )
        
        self.assertIn("non_existent.json", urls, "Missing entry for non-existent artifact")
        self.assertIsNone(urls["non_existent.json"], "URL for non-existent artifact should be None")
        
        logger.info("Missing artifact test passed")
        
        # Clean up
        await provider.close()
        
        logger.info(f"All tests passed for {provider_name} provider!")

    async def run_all_provider_tests(self):
        """Run tests for all providers."""
        for provider_name in self.provider_map:
            await self.test_with_provider(provider_name)


async def test_parallel_url_retrieval_performance():
    """Test the performance improvement of parallel URL retrieval."""
    logger.info("Testing parallel URL retrieval performance...")
    
    # Register all providers
    register_all_providers()
    
    # Create mock GitHub client
    provider = MockGitHubClient()
    await provider.initialize({})
    
    # Create test result reporter
    report_dir = tempfile.mkdtemp()
    artifact_dir = tempfile.mkdtemp()
    
    reporter = TestResultReporter(
        ci_provider=provider,
        report_dir=report_dir,
        artifact_dir=artifact_dir
    )
    
    # Create test run
    test_run_id = "performance-test-123"
    
    # Create 20 test artifacts for performance testing
    artifact_names = [f"test_artifact_{i}.json" for i in range(20)]
    
    # Upload mock artifacts
    for name in artifact_names:
        await provider.upload_artifact(
            test_run_id=test_run_id,
            artifact_path=os.path.join(artifact_dir, "dummy.txt"),
            artifact_name=name
        )
    
    # Implement sequential URL retrieval for comparison
    async def get_urls_sequentially():
        urls = {}
        for name in artifact_names:
            try:
                urls[name] = await provider.get_artifact_url(test_run_id, name)
            except Exception as e:
                logger.error(f"Error retrieving URL for {name}: {str(e)}")
                urls[name] = None
        return urls
    
    # Test sequential retrieval time
    import time
    
    start_time = time.time()
    sequential_urls = await get_urls_sequentially()
    sequential_time = time.time() - start_time
    
    # Test parallel retrieval time
    start_time = time.time()
    parallel_urls = await reporter.get_artifact_urls(test_run_id, artifact_names)
    parallel_time = time.time() - start_time
    
    # Verify results are the same
    for name in artifact_names:
        assert sequential_urls[name] == parallel_urls[name], \
            f"URL mismatch for {name}: {sequential_urls[name]} vs {parallel_urls[name]}"
    
    # Log performance improvement
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    logger.info(f"Sequential retrieval time: {sequential_time:.4f} seconds")
    logger.info(f"Parallel retrieval time: {parallel_time:.4f} seconds")
    logger.info(f"Speedup factor: {speedup:.2f}x")
    
    # Clean up
    await provider.close()
    os.rmdir(report_dir)
    os.rmdir(artifact_dir)
    
    assert speedup > 1.5, f"Expected significant speedup (>1.5x), got {speedup:.2f}x"
    logger.info("Parallel URL retrieval performance test passed")


async def test_edge_cases():
    """Test edge cases for TestResultReporter artifact URL integration."""
    logger.info("Testing edge cases...")
    
    # Register all providers
    register_all_providers()
    
    # Create temporary directories
    report_dir = tempfile.mkdtemp()
    artifact_dir = tempfile.mkdtemp()
    
    try:
        # Test case 1: No CI provider
        logger.info("Testing with no CI provider...")
        
        reporter = TestResultReporter(
            ci_provider=None,
            report_dir=report_dir,
            artifact_dir=artifact_dir
        )
        
        # Try to get artifact URLs without a provider
        urls = await reporter.get_artifact_urls(
            test_run_id="test-123",
            artifact_names=["test.json", "test.log"]
        )
        
        # Should return None for all artifacts
        assert len(urls) == 2, f"Expected 2 URL entries, got {len(urls)}"
        assert urls["test.json"] is None, "URL should be None when no provider is available"
        assert urls["test.log"] is None, "URL should be None when no provider is available"
        
        logger.info("No CI provider test passed")
        
        # Test case 2: Provider without get_artifact_url method
        logger.info("Testing with provider missing get_artifact_url method...")
        
        # Create a minimal provider without get_artifact_url
        class MinimalProvider:
            async def upload_artifact(self, *args, **kwargs):
                return True
            
            async def close(self):
                pass
        
        reporter = TestResultReporter(
            ci_provider=MinimalProvider(),
            report_dir=report_dir,
            artifact_dir=artifact_dir
        )
        
        # Try to get artifact URLs
        urls = await reporter.get_artifact_urls(
            test_run_id="test-123",
            artifact_names=["test.json", "test.log"]
        )
        
        # Should return None for all artifacts
        assert len(urls) == 2, f"Expected 2 URL entries, got {len(urls)}"
        assert urls["test.json"] is None, "URL should be None when provider doesn't support get_artifact_url"
        assert urls["test.log"] is None, "URL should be None when provider doesn't support get_artifact_url"
        
        logger.info("Provider without get_artifact_url test passed")
        
        # Test case 3: Empty artifact names list
        logger.info("Testing with empty artifact names list...")
        
        # Create a GitHub provider
        provider = MockGitHubClient()
        await provider.initialize({})
        
        reporter = TestResultReporter(
            ci_provider=provider,
            report_dir=report_dir,
            artifact_dir=artifact_dir
        )
        
        # Try to get URLs with empty list
        urls = await reporter.get_artifact_urls(
            test_run_id="test-123",
            artifact_names=[]
        )
        
        # Should return an empty dictionary
        assert isinstance(urls, dict), "Should return a dictionary even for empty input"
        assert len(urls) == 0, f"Expected empty dictionary, got {len(urls)} entries"
        
        logger.info("Empty artifact names test passed")
        
        # Test case 4: URL retrieval failure
        logger.info("Testing URL retrieval failure handling...")
        
        # Create a GitHub provider with modified get_artifact_url to sometimes fail
        class FailingProvider(MockGitHubClient):
            async def get_artifact_url(self, test_run_id, artifact_name):
                if artifact_name == "failing.json":
                    raise Exception("Simulated retrieval failure")
                return await super().get_artifact_url(test_run_id, artifact_name)
        
        provider = FailingProvider()
        await provider.initialize({})
        
        reporter = TestResultReporter(
            ci_provider=provider,
            report_dir=report_dir,
            artifact_dir=artifact_dir
        )
        
        # Upload some artifacts first
        test_run_id = "test-failures-123"
        await provider.upload_artifact(test_run_id, os.path.join(artifact_dir, "dummy.txt"), "test.json")
        await provider.upload_artifact(test_run_id, os.path.join(artifact_dir, "dummy.txt"), "failing.json")
        
        # Try to get URLs with one failing
        urls = await reporter.get_artifact_urls(
            test_run_id=test_run_id,
            artifact_names=["test.json", "failing.json"]
        )
        
        # Should have both entries, but one is None
        assert len(urls) == 2, f"Expected 2 URL entries, got {len(urls)}"
        assert urls["test.json"] is not None, "URL for test.json should not be None"
        assert urls["failing.json"] is None, "URL for failing.json should be None due to error"
        
        logger.info("URL retrieval failure test passed")
        
        # Clean up
        await provider.close()
        
    finally:
        # Clean up temporary directories
        os.rmdir(report_dir)
        os.rmdir(artifact_dir)
    
    logger.info("All edge case tests passed")


async def test_ci_coordinator_integration():
    """Test integration with the CI coordinator."""
    logger.info("Testing integration with CI coordinator...")
    
    # Skip if CI coordinator is not available
    try:
        # Try to import the coordinator
        from distributed_testing.coordinator import DistributedTestingCoordinator
    except ImportError:
        logger.warning("DistributedTestingCoordinator not available, skipping CI coordinator integration test")
        return
    
    # Create temporary directories
    report_dir = tempfile.mkdtemp()
    artifact_dir = tempfile.mkdtemp()
    db_path = os.path.join(tempfile.mkdtemp(), "coordinator.db")
    
    try:
        # Create a mock CI provider
        provider = MockGitHubClient()
        await provider.initialize({})
        
        # Create a test result reporter
        reporter = TestResultReporter(
            ci_provider=provider,
            report_dir=report_dir,
            artifact_dir=artifact_dir
        )
        
        # Create a coordinator with batch processing
        coordinator = DistributedTestingCoordinator(
            db_path=db_path,
            enable_batch_processing=True,
            batch_size_limit=5
        )
        
        # Create test result with artifacts
        test_run_id = "ci-coordinator-test-123"
        
        # Create some test artifacts
        artifact_files = []
        artifact_data = [
            {"name": "test_results.json", "content": json.dumps({"tests": [{"name": "test1", "result": "pass"}]})},
            {"name": "metrics.csv", "content": "test,result\ntest1,pass\ntest2,fail"},
            {"name": "log.txt", "content": "INFO: Test log"}
        ]
        
        for artifact in artifact_data:
            file_path = os.path.join(artifact_dir, artifact["name"])
            with open(file_path, "w") as f:
                f.write(artifact["content"])
            artifact_files.append(file_path)
        
        # Register a task with the coordinator
        task_id = await coordinator.register_task({
            "name": "Integration Test",
            "type": "test",
            "priority": 1,
            "parameters": {
                "test_file": "test_integration.py",
                "timeout": 30
            },
            "metadata": {
                "test_run_id": test_run_id
            }
        })
        
        # Update task status to success
        await coordinator.update_task_status(task_id, "completed", {
            "status": "success",
            "total_tests": 10,
            "passed_tests": 9,
            "failed_tests": 1,
            "skipped_tests": 0,
            "duration_seconds": 15.5
        })
        
        # Create a test result
        test_result = TestRunResult(
            test_run_id=test_run_id,
            status="success",
            total_tests=10,
            passed_tests=9,
            failed_tests=1,
            skipped_tests=0,
            duration_seconds=15.5,
            metadata={
                "task_id": task_id,
                "pr_number": "123",
                "performance_metrics": {
                    "average_throughput": 125.4,
                    "average_latency_ms": 7.9,
                    "memory_usage_mb": 256
                }
            }
        )
        
        # Collect and upload artifacts
        artifacts = await reporter.collect_and_upload_artifacts(
            test_run_id=test_run_id,
            artifact_patterns=artifact_files
        )
        
        # Add artifacts to test result metadata
        test_result.metadata["artifacts"] = artifacts
        
        # Send test result to coordinator
        await coordinator.process_test_result(test_result)
        
        # Get task details from coordinator
        task = await coordinator.get_task(task_id)
        
        # Verify that artifacts were attached to the task
        assert "artifacts" in task["result_metadata"], "Artifacts not found in task result metadata"
        
        task_artifacts = task["result_metadata"]["artifacts"]
        assert len(task_artifacts) == len(artifacts), f"Expected {len(artifacts)} artifacts, got {len(task_artifacts)}"
        
        # Verify artifact URLs
        for artifact in task_artifacts:
            assert "url" in artifact, f"URL not found in artifact {artifact['name']}"
            assert artifact["url"] is not None, f"URL is None for artifact {artifact['name']}"
            
        logger.info("CI coordinator integration test passed")
        
    except Exception as e:
        logger.error(f"CI coordinator integration test failed: {str(e)}")
        raise
    
    finally:
        # Clean up temporary directories
        for dir_path in [report_dir, artifact_dir, os.path.dirname(db_path)]:
            if os.path.exists(dir_path):
                try:
                    for file in os.listdir(dir_path):
                        os.unlink(os.path.join(dir_path, file))
                    os.rmdir(dir_path)
                except Exception as e:
                    logger.error(f"Error cleaning up directory {dir_path}: {str(e)}")

async def main():
    """Run the test suite."""
    logger.info("Starting TestResultReporter Artifact URL Integration Tests...")
    
    # Create test suite
    test_suite = TestReporterArtifactIntegration()
    
    try:
        # Run all provider tests
        await test_suite.run_all_provider_tests()
        
        # Test parallel URL retrieval performance
        await test_parallel_url_retrieval_performance()
        
        # Test edge cases
        await test_edge_cases()
        
        # Test CI coordinator integration
        await test_ci_coordinator_integration()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    anyio.run(main())