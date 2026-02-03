"""
Tests for Docker Executor Module

These tests verify the core Docker execution functionality.
"""

import os
import sys
import unittest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ipfs_accelerate_py.docker_executor import (
        DockerExecutor,
        DockerExecutionConfig,
        GitHubDockerConfig,
        DockerExecutionResult,
        execute_docker_hub_container,
        build_and_execute_from_github
    )
    HAVE_DOCKER_EXECUTOR = True
except ImportError:
    HAVE_DOCKER_EXECUTOR = False


@unittest.skipUnless(HAVE_DOCKER_EXECUTOR, "Docker executor module not available")
class TestDockerExecutor(unittest.TestCase):
    """Test Docker executor core functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.executor = None
        
    def test_docker_executor_initialization(self):
        """Test DockerExecutor can be initialized"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            
            executor = DockerExecutor()
            self.assertIsNotNone(executor)
            self.assertEqual(executor.docker_command, "docker")
    
    def test_docker_not_available(self):
        """Test handling when Docker is not available"""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("docker not found")
            
            with self.assertRaises(RuntimeError):
                DockerExecutor()
    
    def test_build_docker_command_basic(self):
        """Test building basic Docker run command"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            
            executor = DockerExecutor()
            config = DockerExecutionConfig(
                image="python:3.9",
                command=["python", "-c", "print('hello')"]
            )
            
            cmd = executor._build_docker_command(config)
            
            self.assertIn("docker", cmd)
            self.assertIn("run", cmd)
            self.assertIn("--rm", cmd)
            self.assertIn("python:3.9", cmd)
            self.assertIn("python", cmd)
    
    def test_build_docker_command_with_resources(self):
        """Test building Docker command with resource limits"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            
            executor = DockerExecutor()
            config = DockerExecutionConfig(
                image="ubuntu:20.04",
                memory_limit="1g",
                cpu_limit=2.0,
                command=["echo", "test"]
            )
            
            cmd = executor._build_docker_command(config)
            
            self.assertIn("--memory", cmd)
            self.assertIn("1g", cmd)
            self.assertIn("--cpus", cmd)
            self.assertIn("2.0", cmd)
    
    def test_build_docker_command_with_environment(self):
        """Test building Docker command with environment variables"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            
            executor = DockerExecutor()
            config = DockerExecutionConfig(
                image="python:3.9",
                environment={"VAR1": "value1", "VAR2": "value2"},
                command=["env"]
            )
            
            cmd = executor._build_docker_command(config)
            
            self.assertIn("-e", cmd)
            # Check that environment variables are in the command
            env_args = [arg for arg in cmd if arg.startswith("VAR")]
            self.assertTrue(any("VAR1=value1" in arg for arg in cmd))
    
    def test_execute_container_success(self):
        """Test successful container execution"""
        with patch('subprocess.run') as mock_run:
            # Mock Docker version check
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            executor = DockerExecutor()
            
            # Mock container execution
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Hello from container",
                stderr=""
            )
            
            config = DockerExecutionConfig(
                image="ubuntu:20.04",
                command=["echo", "Hello from container"]
            )
            
            result = executor.execute_container(config)
            
            self.assertTrue(result.success)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.stdout, "Hello from container")
            self.assertEqual(result.stderr, "")
    
    def test_execute_container_failure(self):
        """Test container execution failure"""
        with patch('subprocess.run') as mock_run:
            # Mock Docker version check
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            executor = DockerExecutor()
            
            # Mock container execution failure
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="Command failed"
            )
            
            config = DockerExecutionConfig(
                image="ubuntu:20.04",
                command=["false"]
            )
            
            result = executor.execute_container(config)
            
            self.assertFalse(result.success)
            self.assertEqual(result.exit_code, 1)
            self.assertEqual(result.stderr, "Command failed")
    
    def test_execute_container_timeout(self):
        """Test container execution timeout"""
        with patch('subprocess.run') as mock_run:
            # Mock Docker version check
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            executor = DockerExecutor()
            
            # Mock timeout
            mock_run.side_effect = subprocess.TimeoutExpired("docker", 10)
            
            config = DockerExecutionConfig(
                image="ubuntu:20.04",
                command=["sleep", "1000"],
                timeout=1
            )
            
            result = executor.execute_container(config)
            
            self.assertFalse(result.success)
            self.assertEqual(result.exit_code, -1)
            # Check for timeout in stderr or error_message
            self.assertTrue(
                "timeout" in result.stderr.lower() or 
                (result.error_message and "timeout" in result.error_message.lower())
            )
    
    def test_list_running_containers(self):
        """Test listing running containers"""
        with patch('subprocess.run') as mock_run:
            # Mock Docker version check
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            executor = DockerExecutor()
            
            # Mock container list
            mock_containers = [
                '{"ID":"abc123","Names":"container1","Status":"Up 5 minutes"}',
                '{"ID":"def456","Names":"container2","Status":"Up 10 minutes"}'
            ]
            mock_run.return_value = Mock(
                returncode=0,
                stdout='\n'.join(mock_containers)
            )
            
            containers = executor.list_running_containers()
            
            self.assertEqual(len(containers), 2)
            self.assertEqual(containers[0]["ID"], "abc123")
            self.assertEqual(containers[1]["ID"], "def456")
    
    def test_stop_container_success(self):
        """Test stopping a container successfully"""
        with patch('subprocess.run') as mock_run:
            # Mock Docker version check
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            executor = DockerExecutor()
            
            # Mock stop success
            mock_run.return_value = Mock(returncode=0, stdout="")
            
            success = executor.stop_container("test_container")
            
            self.assertTrue(success)
    
    def test_stop_container_failure(self):
        """Test stopping a container failure"""
        with patch('subprocess.run') as mock_run:
            # Mock Docker version check
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            executor = DockerExecutor()
            
            # Mock stop failure
            mock_run.return_value = Mock(
                returncode=1,
                stderr="No such container"
            )
            
            success = executor.stop_container("nonexistent")
            
            self.assertFalse(success)
    
    def test_pull_image_success(self):
        """Test pulling Docker image successfully"""
        with patch('subprocess.run') as mock_run:
            # Mock Docker version check
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            executor = DockerExecutor()
            
            # Mock pull success
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Successfully pulled python:3.9"
            )
            
            success = executor.pull_image("python:3.9")
            
            self.assertTrue(success)
    
    def test_convenience_function_execute_docker_hub(self):
        """Test convenience function for Docker Hub execution"""
        with patch('subprocess.run') as mock_run:
            # Mock Docker version check
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            
            # Mock container execution
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Test output",
                stderr=""
            )
            
            result = execute_docker_hub_container(
                image="python:3.9",
                command=["python", "-c", "print('test')"],
                timeout=60
            )
            
            self.assertTrue(result.success)
            self.assertEqual(result.exit_code, 0)


@unittest.skipUnless(HAVE_DOCKER_EXECUTOR, "Docker executor module not available")
class TestGitHubDockerBuild(unittest.TestCase):
    """Test GitHub repository dockerization"""
    
    def test_github_config_creation(self):
        """Test creating GitHub configuration"""
        config = GitHubDockerConfig(
            repo_url="https://github.com/user/repo",
            branch="main",
            dockerfile_path="Dockerfile"
        )
        
        self.assertEqual(config.repo_url, "https://github.com/user/repo")
        self.assertEqual(config.branch, "main")
        self.assertEqual(config.dockerfile_path, "Dockerfile")
    
    @patch('subprocess.run')
    @patch('tempfile.mkdtemp')
    @patch('shutil.rmtree')
    def test_build_and_execute_github_repo(self, mock_rmtree, mock_mkdtemp, mock_run):
        """Test building and executing from GitHub repo"""
        # Mock temp directory
        mock_mkdtemp.return_value = "/tmp/test_repo"
        
        # Mock Docker version check
        mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
        executor = DockerExecutor()
        
        # Mock git clone success
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Mock docker build success
        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0]
            if "clone" in cmd:
                return Mock(returncode=0, stdout="", stderr="")
            elif "build" in cmd:
                return Mock(returncode=0, stdout="Built successfully", stderr="")
            elif "run" in cmd:
                return Mock(returncode=0, stdout="Executed successfully", stderr="")
            elif "rmi" in cmd:
                return Mock(returncode=0, stdout="", stderr="")
            return Mock(returncode=0, stdout="", stderr="")
        
        mock_run.side_effect = mock_run_side_effect
        
        github_config = GitHubDockerConfig(
            repo_url="https://github.com/user/repo",
            branch="main"
        )
        
        execution_config = DockerExecutionConfig(
            image="placeholder",
            command=["python", "app.py"]
        )
        
        result = executor.build_and_execute_github_repo(
            github_config,
            execution_config
        )
        
        # Result should be successful based on our mocks
        self.assertTrue(result.success)


@unittest.skipUnless(HAVE_DOCKER_EXECUTOR, "Docker executor module not available")
class TestDockerExecutionConfig(unittest.TestCase):
    """Test Docker execution configuration"""
    
    def test_config_defaults(self):
        """Test configuration default values"""
        config = DockerExecutionConfig(image="test:latest")
        
        self.assertEqual(config.image, "test:latest")
        self.assertEqual(config.memory_limit, "2g")
        self.assertEqual(config.timeout, 300)
        self.assertEqual(config.network_mode, "none")
        self.assertTrue(config.no_new_privileges)
        self.assertFalse(config.read_only)
    
    def test_config_custom_values(self):
        """Test configuration with custom values"""
        config = DockerExecutionConfig(
            image="custom:tag",
            memory_limit="1g",
            cpu_limit=2.0,
            timeout=600,
            network_mode="bridge",
            read_only=True,
            user="1000:1000"
        )
        
        self.assertEqual(config.memory_limit, "1g")
        self.assertEqual(config.cpu_limit, 2.0)
        self.assertEqual(config.timeout, 600)
        self.assertEqual(config.network_mode, "bridge")
        self.assertTrue(config.read_only)
        self.assertEqual(config.user, "1000:1000")


if __name__ == '__main__':
    unittest.main()
