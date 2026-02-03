"""
Tests for MCP Docker Tools

These tests verify the MCP tool wrappers for Docker execution.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from ipfs_accelerate_py.mcp.tools.docker_tools import (
        execute_docker_container,
        build_and_execute_github_repo,
        execute_with_payload,
        list_running_containers,
        stop_container,
        pull_docker_image,
        register_docker_tools,
        HAVE_DOCKER_EXECUTOR
    )
    HAVE_DOCKER_TOOLS = True
except ImportError as e:
    print(f"Failed to import docker tools: {e}")
    HAVE_DOCKER_TOOLS = False


@unittest.skipUnless(HAVE_DOCKER_TOOLS, "Docker tools not available")
class TestDockerMCPTools(unittest.TestCase):
    """Test MCP Docker tool wrappers"""
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.execute_docker_hub_container')
    def test_execute_docker_container_success(self, mock_execute):
        """Test execute_docker_container MCP tool"""
        # Mock successful execution
        from ipfs_accelerate_py.docker_executor import DockerExecutionResult
        
        mock_execute.return_value = DockerExecutionResult(
            success=True,
            exit_code=0,
            stdout="Hello from Docker",
            stderr="",
            execution_time=1.5
        )
        
        result = execute_docker_container(
            image="python:3.9",
            command="python -c 'print(\"Hello from Docker\")'",
            memory_limit="512m",
            timeout=60
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"], "Hello from Docker")
        self.assertIn("execution_time", result)
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.execute_docker_hub_container')
    def test_execute_docker_container_failure(self, mock_execute):
        """Test execute_docker_container with failure"""
        from ipfs_accelerate_py.docker_executor import DockerExecutionResult
        
        mock_execute.return_value = DockerExecutionResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Command failed",
            execution_time=0.5,
            error_message="Execution failed"
        )
        
        result = execute_docker_container(
            image="ubuntu:20.04",
            command="false"
        )
        
        self.assertFalse(result["success"])
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("error_message", result)
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.execute_docker_hub_container')
    def test_execute_docker_container_with_environment(self, mock_execute):
        """Test execute_docker_container with environment variables"""
        from ipfs_accelerate_py.docker_executor import DockerExecutionResult
        
        mock_execute.return_value = DockerExecutionResult(
            success=True,
            exit_code=0,
            stdout="VAR1=value1",
            stderr="",
            execution_time=1.0
        )
        
        result = execute_docker_container(
            image="ubuntu:20.04",
            command="env",
            environment={"VAR1": "value1", "VAR2": "value2"}
        )
        
        self.assertTrue(result["success"])
        # Verify environment was passed to execute function
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args[1]
        self.assertIn("environment", call_kwargs)
        self.assertEqual(call_kwargs["environment"]["VAR1"], "value1")
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.build_and_execute_from_github')
    def test_build_and_execute_github_repo_success(self, mock_build):
        """Test build_and_execute_github_repo MCP tool"""
        from ipfs_accelerate_py.docker_executor import DockerExecutionResult
        
        mock_build.return_value = DockerExecutionResult(
            success=True,
            exit_code=0,
            stdout="Application executed successfully",
            stderr="",
            execution_time=30.0
        )
        
        result = build_and_execute_github_repo(
            repo_url="https://github.com/user/repo",
            branch="main",
            command="python app.py"
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("execution_time", result)
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.build_and_execute_from_github')
    def test_build_and_execute_github_repo_with_build_args(self, mock_build):
        """Test build_and_execute_github_repo with build arguments"""
        from ipfs_accelerate_py.docker_executor import DockerExecutionResult
        
        mock_build.return_value = DockerExecutionResult(
            success=True,
            exit_code=0,
            stdout="Built and executed",
            stderr="",
            execution_time=25.0
        )
        
        result = build_and_execute_github_repo(
            repo_url="https://github.com/user/python-app",
            branch="develop",
            dockerfile_path="docker/Dockerfile",
            build_args={"PYTHON_VERSION": "3.9", "ENV": "production"}
        )
        
        self.assertTrue(result["success"])
        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args[1]
        self.assertEqual(call_kwargs["build_args"]["PYTHON_VERSION"], "3.9")
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.execute_docker_hub_container')
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.chmod')
    @patch('os.unlink')
    def test_execute_with_payload(self, mock_unlink, mock_chmod, mock_temp, mock_execute):
        """Test execute_with_payload MCP tool"""
        from ipfs_accelerate_py.docker_executor import DockerExecutionResult
        
        # Mock temp file
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/test_payload"
        mock_temp_file.__enter__.return_value = mock_temp_file
        mock_temp.return_value = mock_temp_file
        
        mock_execute.return_value = DockerExecutionResult(
            success=True,
            exit_code=0,
            stdout="4",  # Result of 2+2
            stderr="",
            execution_time=1.0
        )
        
        result = execute_with_payload(
            image="python:3.9",
            payload="print(2+2)",
            payload_path="/app/script.py",
            entrypoint="python /app/script.py"
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["exit_code"], 0)
        # Verify file was written
        mock_temp_file.write.assert_called_with("print(2+2)")
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.execute_docker_hub_container')
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.chmod')
    @patch('os.unlink')
    @patch('os.path.exists')
    def test_execute_with_payload_script(self, mock_exists, mock_unlink, mock_chmod, mock_temp, mock_execute):
        """Test execute_with_payload with shell script"""
        from ipfs_accelerate_py.docker_executor import DockerExecutionResult
        
        mock_exists.return_value = True
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/test_script.sh"
        mock_temp_file.__enter__.return_value = mock_temp_file
        mock_temp.return_value = mock_temp_file
        
        mock_execute.return_value = DockerExecutionResult(
            success=True,
            exit_code=0,
            stdout="Script executed",
            stderr="",
            execution_time=0.8
        )
        
        payload = "#!/bin/bash\necho 'Script executed'"
        
        result = execute_with_payload(
            image="ubuntu:20.04",
            payload=payload,
            payload_path="/tmp/script.sh",
            entrypoint="bash /tmp/script.sh"
        )
        
        self.assertTrue(result["success"])
        # Verify chmod was called for executable script
        mock_chmod.assert_called_once()
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.DockerExecutor')
    def test_list_running_containers(self, mock_executor_class):
        """Test list_running_containers MCP tool"""
        mock_executor = MagicMock()
        mock_executor.list_running_containers.return_value = [
            {"ID": "abc123", "Names": "container1", "Status": "Up 5 minutes"},
            {"ID": "def456", "Names": "container2", "Status": "Up 10 minutes"}
        ]
        mock_executor_class.return_value = mock_executor
        
        result = list_running_containers()
        
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["containers"]), 2)
        self.assertEqual(result["containers"][0]["ID"], "abc123")
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.DockerExecutor')
    def test_stop_container_success(self, mock_executor_class):
        """Test stop_container MCP tool"""
        mock_executor = MagicMock()
        mock_executor.stop_container.return_value = True
        mock_executor_class.return_value = mock_executor
        
        result = stop_container(container_id="test_container")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["container_id"], "test_container")
        self.assertIn("stopped successfully", result["message"].lower())
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.DockerExecutor')
    def test_stop_container_failure(self, mock_executor_class):
        """Test stop_container with failure"""
        mock_executor = MagicMock()
        mock_executor.stop_container.return_value = False
        mock_executor_class.return_value = mock_executor
        
        result = stop_container(container_id="nonexistent")
        
        self.assertFalse(result["success"])
        self.assertEqual(result["container_id"], "nonexistent")
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.DockerExecutor')
    def test_pull_docker_image_success(self, mock_executor_class):
        """Test pull_docker_image MCP tool"""
        mock_executor = MagicMock()
        mock_executor.pull_image.return_value = True
        mock_executor_class.return_value = mock_executor
        
        result = pull_docker_image(image="python:3.9")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["image"], "python:3.9")
        self.assertIn("pulled successfully", result["message"].lower())
    
    def test_register_docker_tools(self):
        """Test registering Docker tools with MCP server"""
        mock_server = MagicMock()
        mock_tool_decorator = MagicMock(return_value=lambda f: f)
        mock_server.tool.return_value = mock_tool_decorator
        
        register_docker_tools(mock_server)
        
        # Should have attempted to register tools
        # Number of calls depends on HAVE_DOCKER_EXECUTOR
        if HAVE_DOCKER_EXECUTOR:
            self.assertGreater(mock_server.tool.call_count, 0)


@unittest.skipUnless(HAVE_DOCKER_TOOLS, "Docker tools not available")
class TestDockerToolsEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.HAVE_DOCKER_EXECUTOR', False)
    def test_docker_executor_not_available(self):
        """Test tools when Docker executor is not available"""
        result = execute_docker_container(
            image="python:3.9",
            command="echo test"
        )
        
        self.assertFalse(result["success"])
        self.assertIn("not available", result["error_message"])
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.execute_docker_hub_container')
    def test_execute_docker_container_exception(self, mock_execute):
        """Test exception handling in execute_docker_container"""
        mock_execute.side_effect = Exception("Unexpected error")
        
        result = execute_docker_container(
            image="python:3.9",
            command="python -c 'print(1)'"
        )
        
        self.assertFalse(result["success"])
        self.assertEqual(result["exit_code"], -1)
        self.assertIn("Unexpected error", result["error_message"])
    
    @patch('ipfs_accelerate_py.mcp.tools.docker_tools.execute_docker_hub_container')
    def test_execute_with_payload_cleanup(self, mock_execute):
        """Test that temporary files are cleaned up"""
        from ipfs_accelerate_py.docker_executor import DockerExecutionResult
        
        mock_execute.return_value = DockerExecutionResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            execution_time=1.0
        )
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.unlink') as mock_unlink:
                with patch('os.path.exists', return_value=True):
                    mock_temp_file = MagicMock()
                    mock_temp_file.name = "/tmp/test"
                    mock_temp_file.__enter__.return_value = mock_temp_file
                    mock_temp.return_value = mock_temp_file
                    
                    execute_with_payload(
                        image="python:3.9",
                        payload="test",
                        payload_path="/app/test"
                    )
                    
                    # Verify cleanup was attempted
                    mock_unlink.assert_called_once()


if __name__ == '__main__':
    unittest.main()
