#!/usr/bin/env python3
"""
Tests for CI Client implementations

This module contains tests for the CI client implementations used by the
CI/CD Integration plugin for the Distributed Testing Framework.
"""

import logging
import os
import unittest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# These CI client implementations rely on aiohttp. In minimal-dependency
# environments we skip this entire module rather than failing during import/setup.
pytest.importorskip("aiohttp")

# Import CI clients
from distributed_testing.ci import GitHubClient, GitLabClient, JenkinsClient, AzureClient

if GitHubClient is None or GitLabClient is None or JenkinsClient is None or AzureClient is None:
    pytest.skip("One or more CI clients are unavailable in this environment", allow_module_level=True)

class TestGitHubClient(unittest.IsolatedAsyncioTestCase):
    """Tests for GitHubClient implementation."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        self.token = "mock_github_token"
        self.repository = "owner/repo"
        self.client = GitHubClient()
        assert await self.client.initialize({"token": self.token, "repository": self.repository})
        # Avoid creating a real aiohttp.ClientSession (which would need closing).
        self.client.session = MagicMock()
    
    async def test_create_test_run(self):
        """Test creating a test run in GitHub."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={
            "id": 12345,
            "name": "Test Run",
            "started_at": datetime.now().isoformat(),
            "html_url": "https://github.com/owner/repo/runs/12345"
        })
        self.client.session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        self.client.session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Create test run
        test_run_data = {
            "name": "GitHub Test Run",
            "commit_sha": "abc123"
        }
        
        test_run = await self.client.create_test_run(test_run_data)
        
        # Assert test run was created
        self.assertEqual(test_run["id"], "12345")
        self.assertEqual(test_run["name"], "Test Run")
        self.assertEqual(test_run["status"], "running")
    
    async def test_update_test_run(self):
        """Test updating a test run in GitHub."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 200
        self.client.session.patch.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        self.client.session.patch.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Update test run
        update_data = {
            "status": "completed",
            "summary": {
                "total_tasks": 10,
                "task_statuses": {
                    "completed": 8,
                    "failed": 2
                },
                "duration": 120.5
            }
        }
        
        result = await self.client.update_test_run("12345", update_data)
        
        # Assert test run was updated
        self.assertTrue(result)
    
    async def test_add_pr_comment(self):
        """Test adding a PR comment in GitHub."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 201
        self.client.session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        self.client.session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Add PR comment
        pr_number = "42"
        comment = "Test comment"
        
        result = await self.client.add_pr_comment(pr_number, comment)
        
        # Assert comment was added
        self.assertTrue(result)

class TestGitLabClient(unittest.IsolatedAsyncioTestCase):
    """Tests for GitLabClient implementation."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        self.token = "mock_gitlab_token"
        self.project = "group/project"
        self.client = GitLabClient()
        assert await self.client.initialize({"token": self.token, "project": self.project})
        # Avoid creating a real aiohttp.ClientSession (which would need closing).
        self.client.session = MagicMock()
    
    async def test_create_test_run_with_commit(self):
        """Test creating a test run in GitLab using commit status."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={
            "id": 12345,
            "sha": "abc123",
            "status": "running"
        })
        self.client.session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        self.client.session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Create test run
        test_run_data = {
            "name": "GitLab Test Run",
            "commit_sha": "abc123"
        }
        
        test_run = await self.client.create_test_run(test_run_data)
        
        # Assert test run was created
        self.assertTrue(test_run["id"].startswith("gl-status-"))
        self.assertEqual(test_run["status"], "running")
    
    async def test_update_test_run(self):
        """Test updating a test run in GitLab."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 201
        self.client.session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        self.client.session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Update test run
        test_run_id = "gl-status-abc123-12345"
        update_data = {
            "status": "completed",
            "summary": {
                "total_tasks": 10,
                "task_statuses": {
                    "completed": 8,
                    "failed": 2
                },
                "duration": 120.5
            }
        }
        
        result = await self.client.update_test_run(test_run_id, update_data)
        
        # Assert test run was updated
        self.assertTrue(result)
    
    async def test_add_pr_comment(self):
        """Test adding a MR comment in GitLab."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 201
        self.client.session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        self.client.session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Add MR comment
        mr_number = "42"
        comment = "Test comment"
        
        result = await self.client.add_pr_comment(mr_number, comment)
        
        # Assert comment was added
        self.assertTrue(result)

class TestJenkinsClient(unittest.IsolatedAsyncioTestCase):
    """Tests for JenkinsClient implementation."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        self.url = "https://jenkins.example.com/"
        self.user = "jenkins_user"
        self.token = "jenkins_token"
        self.client = JenkinsClient()
        assert await self.client.initialize({"url": self.url, "user": self.user, "token": self.token})
        # Avoid creating a real aiohttp.ClientSession (which would need closing).
        self.client.session = MagicMock()
    
    async def test_create_test_run(self):
        """Test creating a test run in Jenkins."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 200
        self.client.session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        self.client.session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Create test run
        test_run_data = {
            "name": "Jenkins Test Run",
            "job_name": "test-job",
            "build_id": "42"
        }
        
        test_run = await self.client.create_test_run(test_run_data)
        
        # Assert test run was created
        self.assertEqual(test_run["id"], "jenkins-test-job-42")
        self.assertEqual(test_run["status"], "running")
    
    async def test_update_test_run(self):
        """Test updating a test run in Jenkins."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 200
        self.client.session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        self.client.session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Update test run
        test_run_id = "jenkins-test-job-42"
        update_data = {
            "status": "completed",
            "summary": {
                "total_tasks": 10,
                "task_statuses": {
                    "completed": 8,
                    "failed": 2
                },
                "duration": 120.5
            }
        }
        
        result = await self.client.update_test_run(test_run_id, update_data)
        
        # Assert test run was updated
        self.assertTrue(result)

class TestAzureClient(unittest.IsolatedAsyncioTestCase):
    """Tests for AzureClient implementation."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        self.token = "azure_token"
        self.organization = "org"
        self.project = "project"
        self.client = AzureClient()
        assert await self.client.initialize({"token": self.token, "organization": self.organization, "project": self.project})
        # Avoid creating a real aiohttp.ClientSession (which would need closing).
        self.client.session = MagicMock()
    
    async def test_create_test_run(self):
        """Test creating a test run in Azure DevOps."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "id": 12345,
            "name": "Test Run"
        })
        self.client.session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        self.client.session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Create test run
        test_run_data = {
            "name": "Azure Test Run",
            "build_id": "42"
        }
        
        test_run = await self.client.create_test_run(test_run_data)
        
        # Assert test run was created
        self.assertEqual(test_run["id"], "12345")
        self.assertEqual(test_run["status"], "running")
    
    async def test_update_test_run(self):
        """Test updating a test run in Azure DevOps."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 200
        self.client.session.patch.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        self.client.session.patch.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Update test run
        test_run_id = "12345"
        update_data = {
            "status": "completed",
            "summary": {
                "total_tasks": 10,
                "task_statuses": {
                    "completed": 8,
                    "failed": 2
                },
                "duration": 120.5
            }
        }
        
        result = await self.client.update_test_run(test_run_id, update_data)
        
        # Assert test run was updated
        self.assertTrue(result)
    
    async def test_add_pr_comment(self):
        """Test adding a PR comment in Azure DevOps."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 200
        self.client.session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        self.client.session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Add PR comment
        pr_number = "42"
        comment = "Test comment"
        
        result = await self.client.add_pr_comment(pr_number, comment)
        
        # Assert comment was added
        self.assertTrue(result)

