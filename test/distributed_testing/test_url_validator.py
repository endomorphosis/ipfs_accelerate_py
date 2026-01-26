#!/usr/bin/env python3
"""
Test Script for URL Validation System

This script tests the URL validation functionality for artifact URLs.
"""

import pytest

# URL validation uses aiohttp for its default HTTP client. In minimal dependency
# environments, skip these tests instead of failing.
pytest.importorskip("aiohttp")

import anyio
import inspect
import logging
import os
import sys
import tempfile
import unittest
from typing import Dict, List, Optional, Any, Tuple

# Add the parent directory to the path
sys.path.append('/home/barberb/ipfs_accelerate_py/test')

# Import the URL validator
from distributed_testing.ci.url_validator import (
    ArtifactURLValidator,
    get_validator,
    validate_url,
    validate_urls,
    generate_health_report,
    close_validator
)

# Import TestResultReporter for integration testing
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.api_interface import TestRunResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockHTTPResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, url, status=200):
        self.url = url
        self.status = status
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockSession:
    """Mock aiohttp session for testing."""
    
    def __init__(self, urls_status=None):
        self.urls_status = urls_status or {}
    
    def head(self, url, timeout=None, allow_redirects=None):
        # aiohttp's session.head() returns an awaitable async-context-manager.
        # For the validator, it's sufficient to return an object that supports
        # async enter/exit and exposes .status.
        status = self.urls_status.get(url, 200)
        return MockHTTPResponse(url, status)
    
    async def close(self):
        pass


class MockCIProvider:
    """Mock CI provider for testing."""
    
    def __init__(self, urls=None):
        self.urls = urls or {}
        self.uploaded_artifacts = {}
    
    async def get_artifact_url(self, test_run_id, artifact_name):
        return self.urls.get(artifact_name, f"https://example.com/artifacts/{test_run_id}/{artifact_name}")
    
    async def upload_artifact(self, test_run_id, artifact_path, artifact_name):
        self.uploaded_artifacts[(test_run_id, artifact_name)] = artifact_path
        return True
    
    async def close(self):
        pass


class _TestURLValidatorBase(unittest.TestCase):
    """Test the URL validation functionality."""

    # Prevent pytest/unittest from collecting this base class directly.
    __test__ = False
    
    def setUp(self):
        """Set up test environment."""
        self.test_urls = [
            "https://example.com/artifact1.json",
            "https://example.com/artifact2.png",
            "https://example.com/not-found.txt",
            "https://invalid-domain.example/artifact.log"
        ]
        self.urls_status = {
            "https://example.com/artifact1.json": 200,
            "https://example.com/artifact2.png": 200,
            "https://example.com/not-found.txt": 404,
            "https://invalid-domain.example/artifact.log": 500
        }
        
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a temporary artifact
        self.artifact_path = os.path.join(self.temp_dir, "test-artifact.json")
        with open(self.artifact_path, "w") as f:
            f.write('{"test": "data"}')
    
    def tearDown(self):
        """Clean up temporary files and directories."""
        os.unlink(self.artifact_path)
        os.rmdir(self.temp_dir)
    
    async def test_validate_url(self):
        """Test validating a single URL."""
        # Create validator with mock session
        mock_session = MockSession(self.urls_status)
        validator = ArtifactURLValidator(session=mock_session)
        await validator.initialize()
        
        # Test a valid URL
        is_valid, status_code, error_message = await validator.validate_url(
            "https://example.com/artifact1.json"
        )
        self.assertTrue(is_valid)
        self.assertEqual(status_code, 200)
        self.assertIsNone(error_message)
        
        # Test an invalid URL
        is_valid, status_code, error_message = await validator.validate_url(
            "https://example.com/not-found.txt"
        )
        self.assertFalse(is_valid)
        self.assertEqual(status_code, 404)
        self.assertIsNotNone(error_message)
        
        # Close validator
        await validator.close()
    
    async def test_validate_urls(self):
        """Test validating multiple URLs in parallel."""
        # Create validator with mock session
        mock_session = MockSession(self.urls_status)
        validator = ArtifactURLValidator(session=mock_session)
        await validator.initialize()
        
        # Test multiple URLs
        results = await validator.validate_urls(self.test_urls)
        
        # Check that all URLs were validated
        self.assertEqual(len(results), len(self.test_urls))
        
        # Check results for valid URLs
        self.assertTrue(results["https://example.com/artifact1.json"][0])
        self.assertEqual(results["https://example.com/artifact1.json"][1], 200)
        
        self.assertTrue(results["https://example.com/artifact2.png"][0])
        self.assertEqual(results["https://example.com/artifact2.png"][1], 200)
        
        # Check results for invalid URLs
        self.assertFalse(results["https://example.com/not-found.txt"][0])
        self.assertEqual(results["https://example.com/not-found.txt"][1], 404)
        
        self.assertFalse(results["https://invalid-domain.example/artifact.log"][0])
        self.assertEqual(results["https://invalid-domain.example/artifact.log"][1], 500)
        
        # Close validator
        await validator.close()
    
    async def test_url_caching(self):
        """Test URL validation caching."""
        # Create validator with mock session
        mock_session = MockSession(self.urls_status)
        validator = ArtifactURLValidator(session=mock_session)
        await validator.initialize()
        
        # Validate a URL
        await validator.validate_url("https://example.com/artifact1.json")
        
        # Modify the session to simulate a URL status change
        modified_urls_status = dict(self.urls_status)
        modified_urls_status["https://example.com/artifact1.json"] = 404
        mock_session.urls_status = modified_urls_status
        
        # Validate the same URL again (should use cache)
        is_valid, status_code, error_message = await validator.validate_url(
            "https://example.com/artifact1.json"
        )
        
        # It should still be valid because of caching
        self.assertTrue(is_valid)
        self.assertEqual(status_code, 200)
        
        # Now validate with cache disabled
        is_valid, status_code, error_message = await validator.validate_url(
            "https://example.com/artifact1.json",
            use_cache=False
        )
        
        # It should now reflect the new status
        self.assertFalse(is_valid)
        self.assertEqual(status_code, 404)
        
        # Close validator
        await validator.close()
    
    async def test_health_report(self):
        """Test generating a health report."""
        # Create validator with mock session
        mock_session = MockSession(self.urls_status)
        validator = ArtifactURLValidator(session=mock_session)
        await validator.initialize()
        
        # Validate a few URLs to populate history
        for url in self.test_urls:
            await validator.validate_url(url)
        
        # Generate a health report
        report = validator.generate_health_report(format="dict")
        
        # Check the report structure
        self.assertIn("timestamp", report)
        self.assertIn("total_urls", report)
        self.assertIn("valid_urls", report)
        self.assertIn("invalid_urls", report)
        self.assertIn("overall_availability", report)
        self.assertIn("urls", report)
        
        # Check that all URLs are in the report
        for url in self.test_urls:
            self.assertIn(url, report["urls"])
            
            # Check URL health information
            health_info = report["urls"][url]
            self.assertIn("is_valid", health_info)
            self.assertIn("status_code", health_info)
            self.assertIn("availability", health_info)
        
        # Generate a markdown report
        markdown_report = validator.generate_health_report(format="markdown")
        self.assertIsInstance(markdown_report, str)
        self.assertIn("# URL Health Report", markdown_report)
        
        # Generate an HTML report
        html_report = validator.generate_health_report(format="html")
        self.assertIsInstance(html_report, str)
        self.assertIn("<!DOCTYPE html>", html_report)
        
        # Close validator
        await validator.close()
    
    async def test_global_validator_functions(self):
        """Test global validator functions."""
        # Override the session in the global validator (must set module global).
        import distributed_testing.ci.url_validator as url_validator_module

        if url_validator_module._global_validator:
            await url_validator_module._global_validator.close()

        url_validator_module._global_validator = ArtifactURLValidator(
            session=MockSession(self.urls_status)
        )
        await url_validator_module._global_validator.initialize()
        
        # Validate a URL
        is_valid, status_code, error_message = await validate_url(
            "https://example.com/artifact1.json"
        )
        self.assertTrue(is_valid)
        
        # Validate multiple URLs
        results = await validate_urls(self.test_urls)
        self.assertEqual(len(results), len(self.test_urls))
        
        # Generate a health report
        report = await generate_health_report(format="dict")
        self.assertIn("total_urls", report)
        
        # Close the global validator
        await close_validator()
    
    async def test_integration_with_result_reporter(self):
        """Test integration with TestResultReporter."""
        # Create a mock CI provider
        ci_provider = MockCIProvider()
        
        # Create a test result reporter
        reporter = TestResultReporter(
            ci_provider=ci_provider,
            report_dir=self.temp_dir,
            artifact_dir=self.temp_dir
        )
        
        # Create a validator with mock session
        mock_session = MockSession(self.urls_status)
        validator = ArtifactURLValidator(session=mock_session)
        await validator.initialize()
        
        # Override the global validator
        import distributed_testing.ci.url_validator as url_validator_module
        if url_validator_module._global_validator:
            await url_validator_module._global_validator.close()

        url_validator_module._global_validator = validator
        
        # Test get_artifact_urls with validation
        ci_provider.urls = {
            "artifact1.json": "https://example.com/artifact1.json",
            "not-found.txt": "https://example.com/not-found.txt"
        }
        
        urls = await reporter.get_artifact_urls(
            test_run_id="test-123",
            artifact_names=["artifact1.json", "not-found.txt"],
            validate=True
        )
        
        self.assertEqual(len(urls), 2)
        self.assertEqual(urls["artifact1.json"], "https://example.com/artifact1.json")
        self.assertEqual(urls["not-found.txt"], "https://example.com/not-found.txt")
        
        # Test collect_and_upload_artifacts with validation
        artifacts = await reporter.collect_and_upload_artifacts(
            test_run_id="test-123",
            artifact_patterns=[self.artifact_path],
            validate_urls=True,
            include_health_info=True
        )
        
        self.assertEqual(len(artifacts), 1)
        self.assertIn("url_validated", artifacts[0])
        self.assertIn("url_valid", artifacts[0])
        
        # URL health info should be included
        if "url_health" in artifacts[0]:
            self.assertIn("url", artifacts[0]["url_health"])
            self.assertIn("is_valid", artifacts[0]["url_health"])
        
        # Clean up
        await close_validator()


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests using unittest under pytest."""

    def run_async(self, test_method, *args, **kwargs):
        async def _runner():
            result = test_method(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return anyio.run(_runner)


class TestURLValidator(AsyncTestCase, _TestURLValidatorBase):
    """Pytest-compatible wrappers for async tests."""

    def test_validate_url(self):
        self.run_async(super().test_validate_url)

    def test_validate_urls(self):
        self.run_async(super().test_validate_urls)

    def test_url_caching(self):
        self.run_async(super().test_url_caching)

    def test_health_report(self):
        self.run_async(super().test_health_report)

    def test_global_validator_functions(self):
        self.run_async(super().test_global_validator_functions)

    def test_integration_with_result_reporter(self):
        self.run_async(super().test_integration_with_result_reporter)


async def main():
    """Run the test suite."""
    logger.info("Starting URL Validator Tests")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTest(unittest.makeSuite(TestURLValidator))
    
    # Run tests with AnyioTestRunner
    class AnyioTestRunner(unittest.TextTestRunner):
        async def run_async(self, test):
            self.run(test)
    
    runner = AnyioTestRunner(verbosity=2)
    await runner.run_async(suite)
    
    logger.info("URL Validator Tests completed")


if __name__ == "__main__":
    anyio.run(main())