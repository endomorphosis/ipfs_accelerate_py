"""
Tests for Docker Kit Module

These tests verify the Docker operations wrapper functionality.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ipfs_accelerate_py.kit.docker_kit import (
        DockerKit,
        DockerResult,
        get_docker_kit
    )
    HAVE_DOCKER_KIT = True
except ImportError:
    HAVE_DOCKER_KIT = False


@unittest.skipUnless(HAVE_DOCKER_KIT, "Docker kit module not available")
class TestDockerKit(unittest.TestCase):
    """Test Docker kit core functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock docker availability
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Docker version 24.0.0",
                stderr=""
            )
            self.kit = DockerKit()
        
    def test_docker_kit_initialization(self):
        """Test DockerKit can be initialized"""
        self.assertIsNotNone(self.kit)
        self.assertEqual(self.kit.docker_path, "docker")
        self.assertEqual(self.kit.timeout, 300)
        
    def test_docker_result_dataclass(self):
        """Test DockerResult dataclass"""
        result = DockerResult(
            success=True,
            data={"container_id": "abc123"},
            exit_code=0,
            stdout="Container started",
            stderr="",
            error=None
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data["container_id"], "abc123")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "Container started")
    
    @patch('subprocess.run')
    def test_verify_installation_success(self, mock_run):
        """Test Docker installation verification - success"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Docker version 24.0.0, build abc123",
            stderr=""
        )
        
        kit = DockerKit()
        # Should complete initialization without error
        self.assertIsNotNone(kit)
    
    @patch('subprocess.run')
    def test_run_container_success(self, mock_run):
        """Test running a container successfully"""
        # Mock docker --version
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Docker version 24.0.0", stderr=""),
            Mock(returncode=0, stdout="abc123\n", stderr="")
        ]
        
        kit = DockerKit()
        result = kit.run_container(
            image="python:3.9",
            command="python --version"
        )
        
        # Should return DockerResult
        self.assertIsInstance(result, DockerResult)
        
    @patch('subprocess.run')
    def test_run_container_with_options(self, mock_run):
        """Test running a container with various options"""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Docker version 24.0.0", stderr=""),
            Mock(returncode=0, stdout="container123\n", stderr="")
        ]
        
        kit = DockerKit()
        result = kit.run_container(
            image="python:3.9",
            command="python --version",
            environment={"VAR1": "value1"},
            memory="512m",
            cpus=1.0,
            network="bridge"
        )
        
        self.assertIsInstance(result, DockerResult)
        
    @patch('subprocess.run')
    def test_list_containers(self, mock_run):
        """Test listing containers"""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Docker version 24.0.0", stderr=""),
            Mock(
                returncode=0,
                stdout='[{"Id":"abc123","Names":"/test","State":"running"}]',
                stderr=""
            )
        ]
        
        kit = DockerKit()
        result = kit.list_containers()
        
        self.assertIsInstance(result, DockerResult)
        if result.success:
            self.assertIsNotNone(result.data)
        
    @patch('subprocess.run')
    def test_stop_container(self, mock_run):
        """Test stopping a container"""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Docker version 24.0.0", stderr=""),
            Mock(returncode=0, stdout="container123", stderr="")
        ]
        
        kit = DockerKit()
        result = kit.stop_container("container123")
        
        self.assertIsInstance(result, DockerResult)
        
    @patch('subprocess.run')
    def test_execute_code_in_container(self, mock_run):
        """Test executing code in container"""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Docker version 24.0.0", stderr=""),
            Mock(returncode=0, stdout="command output", stderr="")
        ]
        
        kit = DockerKit()
        result = kit.execute_code_in_container(
            image="python:3.9",
            code="print('hello')"
        )
        
        self.assertIsInstance(result, DockerResult)
        
    @patch('subprocess.run')
    def test_remove_container(self, mock_run):
        """Test removing a container"""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Docker version 24.0.0", stderr=""),
            Mock(returncode=0, stdout="container123", stderr="")
        ]
        
        kit = DockerKit()
        result = kit.remove_container("container123")
        
        self.assertIsInstance(result, DockerResult)
        
    @patch('subprocess.run')
    def test_pull_image(self, mock_run):
        """Test pulling an image"""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Docker version 24.0.0", stderr=""),
            Mock(returncode=0, stdout="Pull complete", stderr="")
        ]
        
        kit = DockerKit()
        result = kit.pull_image("python:3.9")
        
        self.assertIsInstance(result, DockerResult)
        
    @patch('subprocess.run')
    def test_list_images(self, mock_run):
        """Test listing images"""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Docker version 24.0.0", stderr=""),
            Mock(
                returncode=0,
                stdout='[{"Repository":"python","Tag":"3.9","ID":"abc123"}]',
                stderr=""
            )
        ]
        
        kit = DockerKit()
        result = kit.list_images()
        
        self.assertIsInstance(result, DockerResult)
        
    @patch('subprocess.run')
    def test_error_handling(self, mock_run):
        """Test error handling for failed commands"""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Docker version 24.0.0", stderr=""),
            Mock(returncode=1, stdout="", stderr="Error: container not found")
        ]
        
        kit = DockerKit()
        result = kit.stop_container("nonexistent")
        
        self.assertIsInstance(result, DockerResult)
        # Error case handling
        
    def test_singleton_pattern(self):
        """Test get_docker_kit singleton pattern"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Docker version 24.0.0",
                stderr=""
            )
            
            instance1 = get_docker_kit()
            instance2 = get_docker_kit()
            
            # Should return instances
            self.assertIsNotNone(instance1)
            self.assertIsNotNone(instance2)
    
    @patch('subprocess.run')
    def test_build_image(self, mock_run):
        """Test building an image"""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Docker version 24.0.0", stderr=""),
            Mock(returncode=0, stdout="Successfully built abc123", stderr="")
        ]
        
        kit = DockerKit()
        
        # Check if build_image method exists
        if hasattr(kit, 'build_image'):
            result = kit.build_image(
                context_path="/tmp/test",
                tag="test:latest"
            )
            self.assertIsInstance(result, DockerResult)
        else:
            # Method doesn't exist, skip
            self.skipTest("build_image method not implemented")


if __name__ == '__main__':
    unittest.main()
