#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Error Reporting System

This module contains tests for the automated error reporting functionality.

Author: IPFS Accelerate Python Framework Team
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.error_reporter import ErrorReporter, get_error_reporter, install_global_exception_handler


class TestErrorReporter(unittest.TestCase):
    """Test cases for ErrorReporter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = ErrorReporter(
            github_token='test_token',
            github_repo='test_owner/test_repo',
            enabled=True
        )
        # Override cache file location for testing
        self.reporter.error_cache_file = Path(self.temp_dir) / 'test_reported_errors.json'
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ErrorReporter initialization"""
        self.assertEqual(self.reporter.github_token, 'test_token')
        self.assertEqual(self.reporter.github_repo, 'test_owner/test_repo')
        self.assertTrue(self.reporter.enabled)
        self.assertTrue(self.reporter.include_system_info)
        self.assertTrue(self.reporter.auto_label)
    
    def test_initialization_from_env(self):
        """Test ErrorReporter initialization from environment variables"""
        with patch.dict(os.environ, {
            'GITHUB_TOKEN': 'env_token',
            'GITHUB_REPO': 'env_owner/env_repo'
        }):
            reporter = ErrorReporter()
            self.assertEqual(reporter.github_token, 'env_token')
            self.assertEqual(reporter.github_repo, 'env_owner/env_repo')
    
    def test_disabled_when_missing_config(self):
        """Test that reporter is disabled when configuration is missing"""
        reporter = ErrorReporter(github_token=None, github_repo=None)
        self.assertFalse(reporter.enabled)
    
    def test_compute_error_hash(self):
        """Test error hash computation"""
        error_info1 = {
            'error_type': 'ValueError',
            'error_message': 'Invalid value',
            'source_component': 'test',
            'traceback': 'line 1\nline 2\nline 3'
        }
        
        error_info2 = {
            'error_type': 'ValueError',
            'error_message': 'Invalid value',
            'source_component': 'test',
            'traceback': 'line 1\nline 2\nline 3'
        }
        
        error_info3 = {
            'error_type': 'TypeError',
            'error_message': 'Invalid type',
            'source_component': 'test',
            'traceback': 'line 1\nline 2\nline 3'
        }
        
        hash1 = self.reporter._compute_error_hash(error_info1)
        hash2 = self.reporter._compute_error_hash(error_info2)
        hash3 = self.reporter._compute_error_hash(error_info3)
        
        # Same errors should have same hash
        self.assertEqual(hash1, hash2)
        # Different errors should have different hash
        self.assertNotEqual(hash1, hash3)
    
    def test_gather_system_info(self):
        """Test system information gathering"""
        info = self.reporter._gather_system_info()
        
        self.assertIn('python_version', info)
        self.assertIn('platform', info)
        self.assertIn('architecture', info)
        self.assertIn('timestamp', info)
        self.assertIn('environment', info)
    
    def test_create_issue_body(self):
        """Test issue body creation"""
        error_info = {
            'error_type': 'ValueError',
            'error_message': 'Test error message',
            'source_component': 'test-component',
            'timestamp': '2025-11-06T08:00:00Z',
            'traceback': 'Traceback line 1\nTraceback line 2',
            'context': {'key': 'value'},
            'system_info': {'platform': 'test'}
        }
        
        body = self.reporter._create_issue_body(error_info)
        
        self.assertIn('ValueError', body)
        self.assertIn('Test error message', body)
        self.assertIn('test-component', body)
        self.assertIn('Traceback line 1', body)
        self.assertIn('"key": "value"', body)
        self.assertIn('"platform": "test"', body)
    
    def test_determine_labels(self):
        """Test label determination"""
        # MCP server error
        error_info1 = {
            'error_type': 'RuntimeError',
            'source_component': 'mcp-server'
        }
        labels1 = self.reporter._determine_labels(error_info1)
        self.assertIn('bug', labels1)
        self.assertIn('automated-report', labels1)
        self.assertIn('mcp-server', labels1)
        
        # Dashboard error
        error_info2 = {
            'error_type': 'TypeError',
            'source_component': 'dashboard'
        }
        labels2 = self.reporter._determine_labels(error_info2)
        self.assertIn('dashboard', labels2)
        
        # Critical error
        error_info3 = {
            'error_type': 'CriticalError',
            'source_component': 'docker'
        }
        labels3 = self.reporter._determine_labels(error_info3)
        self.assertIn('priority:high', labels3)
        self.assertIn('docker', labels3)
    
    def test_save_and_load_reported_errors(self):
        """Test saving and loading reported errors cache"""
        # Add some error hashes
        self.reporter.reported_errors.add('hash1')
        self.reporter.reported_errors.add('hash2')
        
        # Save
        self.reporter._save_reported_errors()
        
        # Create new reporter with same cache file
        new_reporter = ErrorReporter(
            github_token='test_token',
            github_repo='test_owner/test_repo'
        )
        new_reporter.error_cache_file = self.reporter.error_cache_file
        new_reporter._load_reported_errors()
        
        # Check that errors were loaded
        self.assertIn('hash1', new_reporter.reported_errors)
        self.assertIn('hash2', new_reporter.reported_errors)
    
    @patch('utils.error_reporter.requests.post')
    def test_create_github_issue_success(self, mock_post):
        """Test successful GitHub issue creation"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            'html_url': 'https://github.com/test/repo/issues/123'
        }
        mock_post.return_value = mock_response
        
        error_info = {
            'error_type': 'TestError',
            'error_message': 'Test message',
            'source_component': 'test',
            'timestamp': '2025-11-06T08:00:00Z',
            'traceback': 'Test traceback'
        }
        
        issue_url = self.reporter._create_github_issue(error_info)
        
        self.assertEqual(issue_url, 'https://github.com/test/repo/issues/123')
        mock_post.assert_called_once()
    
    @patch('utils.error_reporter.requests.post')
    def test_create_github_issue_failure(self, mock_post):
        """Test GitHub issue creation failure"""
        # Mock failed response
        mock_post.side_effect = Exception('API error')
        
        error_info = {
            'error_type': 'TestError',
            'error_message': 'Test message',
            'source_component': 'test',
            'timestamp': '2025-11-06T08:00:00Z',
            'traceback': 'Test traceback'
        }
        
        issue_url = self.reporter._create_github_issue(error_info)
        
        self.assertIsNone(issue_url)
    
    @patch('utils.error_reporter.requests.post')
    def test_report_error_with_exception(self, mock_post):
        """Test reporting an error with exception object"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            'html_url': 'https://github.com/test/repo/issues/124'
        }
        mock_post.return_value = mock_response
        
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            issue_url = self.reporter.report_error(
                exception=e,
                source_component='test',
                context={'test': True}
            )
        
        self.assertIsNotNone(issue_url)
        self.assertIn('github.com', issue_url)
    
    @patch('utils.error_reporter.requests.post')
    def test_report_error_duplicate_prevention(self, mock_post):
        """Test that duplicate errors are not reported"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            'html_url': 'https://github.com/test/repo/issues/125'
        }
        mock_post.return_value = mock_response
        
        # Report first error
        issue_url1 = self.reporter.report_error(
            error_type='TestError',
            error_message='Same error',
            source_component='test'
        )
        
        # Try to report same error again
        issue_url2 = self.reporter.report_error(
            error_type='TestError',
            error_message='Same error',
            source_component='test'
        )
        
        # First should succeed, second should be None (duplicate)
        self.assertIsNotNone(issue_url1)
        self.assertIsNone(issue_url2)
        # API should only be called once
        self.assertEqual(mock_post.call_count, 1)
    
    def test_report_error_disabled(self):
        """Test that reporting does nothing when disabled"""
        reporter = ErrorReporter(enabled=False)
        
        issue_url = reporter.report_error(
            error_type='TestError',
            error_message='Test',
            source_component='test'
        )
        
        self.assertIsNone(issue_url)
    
    def test_get_error_reporter_singleton(self):
        """Test that get_error_reporter returns singleton"""
        reporter1 = get_error_reporter()
        reporter2 = get_error_reporter()
        
        self.assertIs(reporter1, reporter2)


class TestGlobalExceptionHandler(unittest.TestCase):
    """Test cases for global exception handler"""
    
    def test_install_global_exception_handler(self):
        """Test installing global exception handler"""
        original_excepthook = sys.excepthook
        
        install_global_exception_handler('test-component')
        
        # Check that excepthook was modified
        self.assertNotEqual(sys.excepthook, original_excepthook)
        
        # Restore original
        sys.excepthook = original_excepthook


if __name__ == '__main__':
    unittest.main()
