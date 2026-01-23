#!/usr/bin/env python3
"""
Test CI/CD Integration with Distributed Testing Framework

This module provides comprehensive tests for the CI/CD integration functionality
including the TestResultReporter, CIProviderInterface implementations, and the
integration with the coordinator.
"""

import anyio
import json
import logging
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import socket
from unittest.mock import patch, MagicMock, AsyncMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import from distributed_testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import necessary modules
from distributed_testing.ci.api_interface import CIProviderFactory, CIProviderInterface, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.register_providers import register_all_providers


class MockCIProvider(CIProviderInterface):
    """Mock CI provider for testing."""
    
    def __init__(self):
        self.test_runs = {}
        self.artifacts = {}
        self.pr_comments = {}
        self.build_statuses = {}
    
    async def initialize(self, config):
        """Initialize the mock CI provider."""
        self.config = config
        return True
    
    async def create_test_run(self, test_run_data):
        """Create a test run."""
        test_run_id = f"test-{len(self.test_runs) + 1}"
        test_run = {
            "id": test_run_id,
            "name": test_run_data.get("name", f"Test Run {test_run_id}"),
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "url": f"http://example.com/test-runs/{test_run_id}"
        }
        self.test_runs[test_run_id] = test_run
        return test_run
    
    async def update_test_run(self, test_run_id, update_data):
        """Update a test run."""
        if test_run_id not in self.test_runs:
            return False
        
        self.test_runs[test_run_id].update(update_data)
        return True
    
    async def add_pr_comment(self, pr_number, comment):
        """Add a comment to a PR."""
        if pr_number not in self.pr_comments:
            self.pr_comments[pr_number] = []
        
        self.pr_comments[pr_number].append(comment)
        return True
    
    async def upload_artifact(self, test_run_id, artifact_path, artifact_name):
        """Upload an artifact."""
        if test_run_id not in self.artifacts:
            self.artifacts[test_run_id] = []
        
        self.artifacts[test_run_id].append({
            "path": artifact_path,
            "name": artifact_name
        })
        return True
    
    async def get_test_run_status(self, test_run_id):
        """Get test run status."""
        if test_run_id not in self.test_runs:
            return {"status": "unknown"}
        
        return self.test_runs[test_run_id]
    
    async def set_build_status(self, status, description):
        """Set build status."""
        self.build_statuses[status] = description
        return True
    
    async def close(self):
        """Close the provider."""
        pass


class TestCIIntegration(unittest.IsolatedAsyncioTestCase):
    """Test CI integration functionality."""
    
    async def asyncSetUp(self):
        """Set up for tests."""
        # Create temporary directories for tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.reports_dir = self.temp_path / "reports"
        self.artifacts_dir = self.temp_path / "artifacts"
        self.reports_dir.mkdir()
        self.artifacts_dir.mkdir()
        
        # Create a sample artifact
        self.artifact_path = self.artifacts_dir / "sample.json"
        with open(self.artifact_path, "w") as f:
            json.dump({"test": "data"}, f)
        
        # Register the mock provider
        CIProviderFactory.register_provider("mock", MockCIProvider)
        
        # Create a mock CI provider
        self.mock_provider = await CIProviderFactory.create_provider("mock", {})
        
        # Create a test result reporter
        self.reporter = TestResultReporter(
            ci_provider=self.mock_provider,
            report_dir=str(self.reports_dir),
            artifact_dir=str(self.artifacts_dir)
        )
    
    async def asyncTearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    async def test_create_test_run(self):
        """Test creating a test run."""
        test_run_data = {
            "name": "Test Run",
            "build_id": "build-123",
            "commit_sha": "abcdef123456"
        }
        
        test_run = await self.mock_provider.create_test_run(test_run_data)
        
        self.assertIn("id", test_run)
        self.assertEqual(test_run["name"], "Test Run")
        self.assertEqual(test_run["status"], "running")
        self.assertIn("start_time", test_run)
    
    async def test_update_test_run(self):
        """Test updating a test run."""
        test_run = await self.mock_provider.create_test_run({"name": "Test Run"})
        test_run_id = test_run["id"]
        
        update_data = {
            "status": "completed",
            "end_time": datetime.now().isoformat()
        }
        
        result = await self.mock_provider.update_test_run(test_run_id, update_data)
        self.assertTrue(result)
        
        updated_run = self.mock_provider.test_runs[test_run_id]
        self.assertEqual(updated_run["status"], "completed")
        self.assertIn("end_time", updated_run)
    
    async def test_add_pr_comment(self):
        """Test adding a PR comment."""
        pr_number = "123"
        comment = "Test comment"
        
        result = await self.mock_provider.add_pr_comment(pr_number, comment)
        self.assertTrue(result)
        
        self.assertIn(pr_number, self.mock_provider.pr_comments)
        self.assertEqual(self.mock_provider.pr_comments[pr_number][0], comment)
    
    async def test_upload_artifact(self):
        """Test uploading an artifact."""
        test_run = await self.mock_provider.create_test_run({"name": "Test Run"})
        test_run_id = test_run["id"]
        
        result = await self.mock_provider.upload_artifact(
            test_run_id,
            str(self.artifact_path),
            "sample.json"
        )
        self.assertTrue(result)
        
        self.assertIn(test_run_id, self.mock_provider.artifacts)
        artifact = self.mock_provider.artifacts[test_run_id][0]
        self.assertEqual(artifact["name"], "sample.json")
        self.assertEqual(artifact["path"], str(self.artifact_path))
    
    async def test_set_build_status(self):
        """Test setting build status."""
        status = "success"
        description = "Build succeeded"
        
        result = await self.mock_provider.set_build_status(status, description)
        self.assertTrue(result)
        
        self.assertIn(status, self.mock_provider.build_statuses)
        self.assertEqual(self.mock_provider.build_statuses[status], description)
    
    async def test_result_reporter_report_test_result(self):
        """Test reporting test results."""
        test_run = await self.mock_provider.create_test_run({"name": "Test Run"})
        test_run_id = test_run["id"]
        
        test_result = TestRunResult(
            test_run_id=test_run_id,
            status="success",
            total_tests=10,
            passed_tests=9,
            failed_tests=1,
            skipped_tests=0,
            duration_seconds=5.0
        )
        
        test_result.metadata = {
            "performance_metrics": {
                "average_throughput": 123.4,
                "average_latency_ms": 8.5
            }
        }
        
        # Report test results
        report_files = await self.reporter.report_test_result(
            test_result,
            formats=["markdown", "html", "json"]
        )
        
        # Check that report files were created
        self.assertEqual(len(report_files), 3)
        self.assertTrue(os.path.exists(report_files["markdown"]))
        self.assertTrue(os.path.exists(report_files["html"]))
        self.assertTrue(os.path.exists(report_files["json"]))
        
        # Check markdown report content
        with open(report_files["markdown"], "r") as f:
            markdown_content = f.read()
            self.assertIn(f"# Test Run Report: {test_run_id}", markdown_content)
            self.assertIn("**Status:** SUCCESS", markdown_content)
            self.assertIn("Total Tests: 10", markdown_content)
            self.assertIn("Passed: 9", markdown_content)
            self.assertIn("Failed: 1", markdown_content)
        
        # Check HTML report content
        with open(report_files["html"], "r") as f:
            html_content = f.read()
            self.assertIn("<!DOCTYPE html>", html_content)
            self.assertIn(f"Test Run Report: {test_run_id}", html_content)
            self.assertIn("Total Tests", html_content)
            self.assertIn("Passed", html_content)
            self.assertIn("Failed", html_content)
        
        # Check JSON report content
        with open(report_files["json"], "r") as f:
            json_content = json.load(f)
            self.assertEqual(json_content["test_run_id"], test_run_id)
            self.assertEqual(json_content["status"], "success")
            self.assertEqual(json_content["total_tests"], 10)
            self.assertEqual(json_content["passed_tests"], 9)
            self.assertEqual(json_content["failed_tests"], 1)
            self.assertIn("performance_metrics", json_content["metadata"])
    
    async def test_result_reporter_collect_artifacts(self):
        """Test collecting artifacts."""
        test_run = await self.mock_provider.create_test_run({"name": "Test Run"})
        test_run_id = test_run["id"]
        
        # Create additional artifacts
        for i in range(3):
            artifact_path = self.artifacts_dir / f"artifact_{i}.json"
            with open(artifact_path, "w") as f:
                json.dump({"id": i}, f)
        
        # Collect artifacts
        artifacts = await self.reporter.collect_and_upload_artifacts(
            test_run_id,
            [str(self.artifacts_dir / "*.json")]
        )
        
        # Check that artifacts were collected and uploaded
        self.assertEqual(len(artifacts), 4)  # sample.json + 3 new artifacts
        self.assertEqual(len(self.mock_provider.artifacts[test_run_id]), 4)
        
        # Check artifact details
        for artifact in artifacts:
            self.assertIn("name", artifact)
            self.assertIn("path", artifact)
            self.assertIn("size_bytes", artifact)
            self.assertTrue(os.path.exists(artifact["path"]))
    
    async def test_factory_registration(self):
        """Test provider factory registration."""
        # Register providers
        register_all_providers()
        
        # Get available providers
        providers = CIProviderFactory.get_available_providers()
        
        # Should include at least our mock provider
        self.assertIn("mock", providers)
        
        # Check creating a provider that doesn't exist
        with self.assertRaises(ValueError):
            await CIProviderFactory.create_provider("nonexistent", {})


if __name__ == "__main__":
    unittest.main()