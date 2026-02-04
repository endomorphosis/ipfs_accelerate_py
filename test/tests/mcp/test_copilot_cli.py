"""
Tests for Copilot CLI integration
"""

import pytest
from unittest.mock import Mock, patch
from ipfs_accelerate_py.copilot_cli import CopilotCLI


class TestCopilotCLI:
    """Test Copilot CLI wrapper"""
    
    def test_init(self):
        """Test CopilotCLI initialization"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            cli = CopilotCLI()
            assert cli.copilot_path == "github-copilot-cli"
    
    def test_suggest_command(self):
        """Test command suggestion"""
        with patch('subprocess.run') as mock_run:
            # Setup mock for verification
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            cli = CopilotCLI()
            
            # Setup mock for suggestion
            mock_run.return_value = Mock(
                returncode=0,
                stdout="ls -la | grep txt",
                stderr=""
            )
            result = cli.suggest_command("list all text files")
            
            assert result["success"] is True
            assert "ls" in result["suggestion"]
    
    def test_explain_command(self):
        """Test command explanation"""
        with patch('subprocess.run') as mock_run:
            # Setup mock for verification
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            cli = CopilotCLI()
            
            # Setup mock for explanation
            mock_run.return_value = Mock(
                returncode=0,
                stdout="This command lists all files in the current directory",
                stderr=""
            )
            result = cli.explain_command("ls -la")
            
            assert result["success"] is True
            assert "lists all files" in result["explanation"]
    
    def test_suggest_git_command(self):
        """Test Git command suggestion"""
        with patch('subprocess.run') as mock_run:
            # Setup mock for verification
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            cli = CopilotCLI()
            
            # Setup mock for Git suggestion
            mock_run.return_value = Mock(
                returncode=0,
                stdout="git commit -am 'message'",
                stderr=""
            )
            result = cli.suggest_git_command("commit all changes with message")
            
            assert result["success"] is True
            assert "git commit" in result["suggestion"]
    
    def test_command_failure(self):
        """Test failed command execution"""
        with patch('subprocess.run') as mock_run:
            # Setup mock for verification
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            cli = CopilotCLI()
            
            # Setup mock for failed command
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="Error: Invalid input"
            )
            result = cli.suggest_command("invalid prompt")
            
            assert result["success"] is False
            assert "Invalid input" in result["stderr"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
