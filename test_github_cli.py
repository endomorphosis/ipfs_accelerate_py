"""
Tests for GitHub CLI integration
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from ipfs_accelerate_py.github_cli import GitHubCLI, WorkflowQueue, RunnerManager


class TestGitHubCLI:
    """Test GitHub CLI wrapper"""
    
    def test_init(self):
        """Test GitHubCLI initialization"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="gh version 2.0.0", stderr="")
            cli = GitHubCLI()
            assert cli.gh_path == "gh"
    
    def test_run_command_success(self):
        """Test successful command execution"""
        with patch('subprocess.run') as mock_run:
            # Setup mock for version check
            mock_run.return_value = Mock(returncode=0, stdout="gh version 2.0.0", stderr="")
            cli = GitHubCLI()
            
            # Setup mock for actual command
            mock_run.return_value = Mock(returncode=0, stdout='{"status": "ok"}', stderr="")
            result = cli._run_command(["api", "repos/owner/repo"])
            
            assert result["success"] is True
            assert result["returncode"] == 0
            assert '{"status": "ok"}' in result["stdout"]
    
    def test_run_command_failure(self):
        """Test failed command execution"""
        with patch('subprocess.run') as mock_run:
            # Setup mock for version check
            mock_run.return_value = Mock(returncode=0, stdout="gh version 2.0.0", stderr="")
            cli = GitHubCLI()
            
            # Setup mock for actual command
            mock_run.return_value = Mock(returncode=1, stdout="", stderr="error")
            result = cli._run_command(["api", "invalid"])
            
            assert result["success"] is False
            assert result["returncode"] == 1
            assert "error" in result["stderr"]
    
    def test_get_auth_status(self):
        """Test getting authentication status"""
        with patch('subprocess.run') as mock_run:
            # Setup mock for version check
            mock_run.return_value = Mock(returncode=0, stdout="gh version 2.0.0", stderr="")
            cli = GitHubCLI()
            
            # Setup mock for auth status
            mock_run.return_value = Mock(returncode=0, stdout="✓ Logged in", stderr="")
            result = cli.get_auth_status()
            
            assert result["authenticated"] is True
            assert "Logged in" in result["output"]
    
    def test_list_repos(self):
        """Test listing repositories"""
        with patch('subprocess.run') as mock_run:
            # Setup mock for version check
            mock_run.return_value = Mock(returncode=0, stdout="gh version 2.0.0", stderr="")
            cli = GitHubCLI()
            
            # Setup mock for list repos
            repos_data = json.dumps([
                {"name": "repo1", "owner": {"login": "user"}, "url": "https://github.com/user/repo1"},
                {"name": "repo2", "owner": {"login": "user"}, "url": "https://github.com/user/repo2"}
            ])
            mock_run.return_value = Mock(returncode=0, stdout=repos_data, stderr="")
            repos = cli.list_repos()
            
            assert len(repos) == 2
            assert repos[0]["name"] == "repo1"
            assert repos[1]["name"] == "repo2"


class TestWorkflowQueue:
    """Test Workflow Queue manager"""
    
    def test_init(self):
        """Test WorkflowQueue initialization"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="gh version 2.0.0", stderr="")
            queue = WorkflowQueue()
            assert queue.gh is not None
    
    def test_list_workflow_runs(self):
        """Test listing workflow runs"""
        with patch('subprocess.run') as mock_run:
            # Setup mock for version check
            mock_run.return_value = Mock(returncode=0, stdout="gh version 2.0.0", stderr="")
            queue = WorkflowQueue()
            
            # Setup mock for workflow runs
            runs_data = json.dumps([
                {
                    "databaseId": 123,
                    "name": "CI",
                    "status": "completed",
                    "conclusion": "success",
                    "createdAt": "2024-01-01T00:00:00Z"
                }
            ])
            mock_run.return_value = Mock(returncode=0, stdout=runs_data, stderr="")
            runs = queue.list_workflow_runs("owner/repo")
            
            assert len(runs) == 1
            assert runs[0]["databaseId"] == 123
            assert runs[0]["status"] == "completed"
    
    def test_list_failed_runs(self):
        """Test filtering failed workflow runs"""
        with patch('subprocess.run') as mock_run:
            # Setup mock for version check
            mock_run.return_value = Mock(returncode=0, stdout="gh version 2.0.0", stderr="")
            queue = WorkflowQueue()
            
            # Setup mock for workflow runs
            from datetime import datetime
            now = datetime.now().isoformat()
            runs_data = json.dumps([
                {
                    "databaseId": 123,
                    "conclusion": "failure",
                    "createdAt": now,
                    "status": "completed"
                },
                {
                    "databaseId": 124,
                    "conclusion": "success",
                    "createdAt": now,
                    "status": "completed"
                }
            ])
            mock_run.return_value = Mock(returncode=0, stdout=runs_data, stderr="")
            failed = queue.list_failed_runs("owner/repo", since_days=1)
            
            assert len(failed) == 1
            assert failed[0]["conclusion"] == "failure"


class TestRunnerManager:
    """Test Runner Manager"""
    
    def test_init(self):
        """Test RunnerManager initialization"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="gh version 2.0.0", stderr="")
            manager = RunnerManager()
            assert manager.gh is not None
    
    def test_get_system_cores(self):
        """Test getting system CPU cores"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="gh version 2.0.0", stderr="")
            manager = RunnerManager()
            cores = manager.get_system_cores()
            
            assert cores > 0
            assert isinstance(cores, int)
    
    def test_get_runner_registration_token(self):
        """Test getting runner registration token"""
        with patch('subprocess.run') as mock_run:
            # Setup mock for version check
            mock_run.return_value = Mock(returncode=0, stdout="gh version 2.0.0", stderr="")
            manager = RunnerManager()
            
            # Setup mock for token generation
            mock_run.return_value = Mock(returncode=0, stdout="token_abc123", stderr="")
            token = manager.get_runner_registration_token(repo="owner/repo")
            
            assert token == "token_abc123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
