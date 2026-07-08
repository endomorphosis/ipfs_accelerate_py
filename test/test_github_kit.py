"""
Tests for GitHub Kit Module

These tests verify the core GitHub operations functionality.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ipfs_accelerate_py.kit.github_kit import (
        GitHubKit,
        GitHubResult
    )
    HAVE_GITHUB_KIT = True
except ImportError:
    HAVE_GITHUB_KIT = False


@unittest.skipUnless(HAVE_GITHUB_KIT, "GitHub kit module not available")
class TestGitHubKit(unittest.TestCase):
    """Test GitHub kit core functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.kit = GitHubKit()
        
    def test_github_kit_initialization(self):
        """Test GitHubKit can be initialized"""
        self.assertIsNotNone(self.kit)
        self.assertTrue(hasattr(self.kit, '_run_command'))
    
    @patch('subprocess.run')
    def test_list_repos_success(self, mock_run):
        """Test listing repositories successfully"""
        mock_output = '{"repositories": [{"name": "test-repo", "full_name": "owner/test-repo"}]}'
        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_output,
            stderr=""
        )
        
        result = self.kit.list_repos(owner="testowner", limit=10)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_list_repos_failure(self, mock_run):
        """Test handling list repos failure"""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Authentication failed"
        )
        
        result = self.kit.list_repos(owner="testowner")
        
        self.assertFalse(result.success)
        self.assertIn("Authentication failed", result.error)
    
    @patch('subprocess.run')
    def test_get_repo_success(self, mock_run):
        """Test getting repository details"""
        mock_output = '{"name": "test-repo", "full_name": "owner/test-repo", "description": "Test"}'
        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_output,
            stderr=""
        )
        
        result = self.kit.get_repo(repo="owner/test-repo")
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
    
    @patch('subprocess.run')
    def test_list_prs_success(self, mock_run):
        """Test listing pull requests"""
        mock_output = '[{"number": 1, "title": "Test PR"}]'
        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_output,
            stderr=""
        )
        
        result = self.kit.list_prs(repo="owner/test-repo", state="open")
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
    
    @patch('subprocess.run')
    def test_list_issues_success(self, mock_run):
        """Test listing issues"""
        mock_output = '[{"number": 1, "title": "Test Issue"}]'
        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_output,
            stderr=""
        )
        
        result = self.kit.list_issues(repo="owner/test-repo", state="open")
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
    
    @patch('subprocess.run')
    def test_list_workflow_runs(self, mock_run):
        """Test listing workflow runs"""
        mock_output = '{"workflow_runs": [{"id": 123, "status": "completed"}]}'
        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_output,
            stderr=""
        )
        
        result = self.kit.list_workflow_runs(repo="owner/test-repo")
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
    
    def test_github_result_dataclass(self):
        """Test GitHubResult dataclass"""
        result = GitHubResult(
            success=True,
            data={"test": "data"},
            error=None
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.data, {"test": "data"})
        self.assertIsNone(result.error)


if __name__ == '__main__':
    unittest.main()
