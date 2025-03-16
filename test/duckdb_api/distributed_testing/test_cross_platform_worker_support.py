#!/usr/bin/env python3
"""
Unit tests for the CrossPlatformWorkerSupport module.

These tests verify the functionality of the cross-platform worker support
including platform detection, hardware detection, and script generation.
"""

import os
import sys
import json
import uuid
import unittest
import tempfile
import platform
import subprocess
from unittest import mock
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the module to test
from duckdb_api.distributed_testing.cross_platform_worker_support import (
    CrossPlatformWorkerSupport,
    LinuxPlatformHandler,
    WindowsPlatformHandler,
    MacOSPlatformHandler,
    ContainerPlatformHandler
)


class TestCrossPlatformWorkerSupport(unittest.TestCase):
    """Test cases for CrossPlatformWorkerSupport class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temp directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Save the actual platform for cleanup
        self.actual_platform = platform.system().lower()
        
        # Common test parameters
        self.test_coordinator_url = "http://test-coordinator:8080"
        self.test_api_key = "test_api_key_123"
        self.test_worker_id = f"worker_{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_platform_detection(self):
        """Test platform detection works correctly."""
        # Test detection with different mock platforms
        with mock.patch('platform.system', return_value='Linux'):
            with mock.patch('os.path.exists', return_value=False):
                support = CrossPlatformWorkerSupport()
                self.assertEqual(support.current_platform, 'linux')
                self.assertIsInstance(support.handler, LinuxPlatformHandler)

        with mock.patch('platform.system', return_value='Windows'):
            support = CrossPlatformWorkerSupport()
            self.assertEqual(support.current_platform, 'windows')
            self.assertIsInstance(support.handler, WindowsPlatformHandler)

        with mock.patch('platform.system', return_value='Darwin'):
            support = CrossPlatformWorkerSupport()
            self.assertEqual(support.current_platform, 'darwin')
            self.assertIsInstance(support.handler, MacOSPlatformHandler)

    def test_container_detection(self):
        """Test container detection works correctly."""
        # Test detection with container environment
        with mock.patch('os.path.exists', lambda path: path == '/.dockerenv'):
            support = CrossPlatformWorkerSupport()
            self.assertEqual(support.current_platform, 'container')
            self.assertIsInstance(support.handler, ContainerPlatformHandler)

        # Test detection with environment variable
        with mock.patch.dict('os.environ', {'CONTAINER_ENV': '1'}):
            with mock.patch('os.path.exists', return_value=False):
                support = CrossPlatformWorkerSupport()
                self.assertEqual(support.current_platform, 'container')
                self.assertIsInstance(support.handler, ContainerPlatformHandler)

    def test_config_loading(self):
        """Test configuration loading from a file."""
        # Create a temporary config file
        config_path = os.path.join(self.temp_dir.name, 'test_config.json')
        test_config = {
            'coordinator_url': self.test_coordinator_url,
            'api_key': self.test_api_key,
            'worker_id': self.test_worker_id
        }
        
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Test loading valid config
        support = CrossPlatformWorkerSupport(config_path=config_path)
        self.assertEqual(support.config, test_config)
        
        # Test loading invalid config (should not raise exception)
        with open(config_path, 'w') as f:
            f.write('invalid json')
        
        with self.assertLogs(level='ERROR') as cm:
            support = CrossPlatformWorkerSupport(config_path=config_path)
            self.assertIn('Error loading configuration', cm.output[0])

    def test_worker_command_generation(self):
        """Test worker command generation for different platforms."""
        config = {
            'coordinator_url': self.test_coordinator_url,
            'api_key': self.test_api_key,
            'worker_id': self.test_worker_id,
            'log_to_file': True
        }
        
        # Test Linux command
        with mock.patch('platform.system', return_value='Linux'):
            support = CrossPlatformWorkerSupport()
            cmd = support.get_worker_command(config)
            self.assertIn('python3', cmd)
            self.assertIn(self.test_coordinator_url, cmd)
            self.assertIn(self.test_api_key, cmd)
            self.assertIn(self.test_worker_id, cmd)
            self.assertIn('--log-file', cmd)
        
        # Test Windows command
        with mock.patch('platform.system', return_value='Windows'):
            support = CrossPlatformWorkerSupport()
            cmd = support.get_worker_command(config)
            self.assertIn('python', cmd)  # Windows uses 'python' not 'python3'
            self.assertIn(self.test_coordinator_url, cmd)
            self.assertIn(self.test_api_key, cmd)
            self.assertIn(self.test_worker_id, cmd)
            self.assertIn('--log-file', cmd)
        
        # Test Container command
        with mock.patch('os.path.exists', lambda path: path == '/.dockerenv'):
            support = CrossPlatformWorkerSupport()
            cmd = support.get_worker_command(config)
            self.assertIn('python', cmd)
            self.assertIn(self.test_coordinator_url, cmd)
            self.assertIn(self.test_api_key, cmd)
            self.assertIn(self.test_worker_id, cmd)
            self.assertIn('--container-mode', cmd)

    def test_deployment_script_creation(self):
        """Test creation of deployment scripts for different platforms."""
        config = {
            'coordinator_url': self.test_coordinator_url,
            'api_key': self.test_api_key,
            'worker_id': self.test_worker_id
        }
        
        # Test Linux script
        with mock.patch('platform.system', return_value='Linux'):
            # Make sure os.path.exists returns False for /.dockerenv to avoid container detection
            with mock.patch('os.path.exists', lambda path: path != '/.dockerenv'):
                support = CrossPlatformWorkerSupport()
                output_path = os.path.join(self.temp_dir.name, 'linux_deploy.sh')
                
                # Mock the actual file write operation
                with mock.patch('builtins.open', mock.mock_open()) as mock_file:
                    with mock.patch('os.chmod') as mock_chmod:
                        script_path = support.create_deployment_script(config, output_path)
                        
                        # Verify the correct calls were made
                        mock_file.assert_called_once_with(output_path, 'w')
                        mock_chmod.assert_called_once_with(output_path, 0o755)
                        
                        # Since we're mocking the file operations, manually check script contents
                        # by examining what was written to the mock file
                        file_handle = mock_file()
                        file_content = ''.join(call.args[0] for call in file_handle.write.call_args_list)
                        
                        self.assertIn('#!/bin/bash', file_content)
                        self.assertIn(self.test_coordinator_url, file_content)
                        self.assertIn(self.test_api_key, file_content)
                        self.assertIn(self.test_worker_id, file_content)
        
        # Test Windows script
        with mock.patch('platform.system', return_value='Windows'):
            support = CrossPlatformWorkerSupport()
            output_path = os.path.join(self.temp_dir.name, 'windows_deploy')
            
            # Mock the file operations
            with mock.patch('builtins.open', mock.mock_open()) as mock_file:
                script_path = support.create_deployment_script(config, output_path)
                
                # Verify the correct calls were made
                mock_file.assert_called_once_with(output_path + '.bat', 'w')
                
                # Since we're mocking the file operations, manually check script contents
                file_handle = mock_file()
                file_content = ''.join(call.args[0] for call in file_handle.write.call_args_list)
                
                self.assertIn('@echo off', file_content)
                self.assertIn(self.test_coordinator_url, file_content)
                self.assertIn(self.test_api_key, file_content)
                self.assertIn(self.test_worker_id, file_content)
        
        # Test Container script (docker-compose.yml)
        with mock.patch('os.path.exists', lambda path: path == '/.dockerenv'):
            support = CrossPlatformWorkerSupport()
            output_path = os.path.join(self.temp_dir.name, 'docker-compose.yml')
            
            # Mock the file operations with a more complex mock to handle multiple files
            open_mock = mock.mock_open()
            file_handles = {}
            
            def get_mock_file(filename, mode):
                if filename not in file_handles:
                    file_handles[filename] = mock.MagicMock()
                return file_handles[filename]
            
            open_mock.side_effect = get_mock_file
            
            with mock.patch('builtins.open', open_mock):
                script_path = support.create_deployment_script(config, output_path)
                
                # Verify docker-compose.yml was created
                self.assertEqual(script_path, output_path)
                
                # Verify Dockerfile.worker was created
                dockerfile_path = os.path.join(os.path.dirname(output_path), 'Dockerfile.worker')
                
                # Verify both files were opened
                open_calls = [call[0][0] for call in open_mock.call_args_list]
                self.assertIn(output_path, open_calls)
                self.assertIn(dockerfile_path, open_calls)

    @mock.patch('subprocess.run')
    def test_install_dependencies(self, mock_run):
        """Test dependency installation for different platforms."""
        # Create a mock subprocess response
        mock_process = mock.MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Test with default dependencies
        with mock.patch('platform.system', return_value='Linux'):
            # Make sure os.path.exists returns False for /.dockerenv to avoid container detection
            with mock.patch('os.path.exists', lambda path: path != '/.dockerenv'):
                support = CrossPlatformWorkerSupport()
                result = support.install_dependencies()
                
                self.assertTrue(result)
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                self.assertEqual(args[0], sys.executable)
                self.assertEqual(args[1:3], ['-m', 'pip'])
            
        # Reset mock
        mock_run.reset_mock()
        mock_run.return_value = mock_process
        
        # Test with custom dependencies
        custom_deps = ['requests', 'pandas', 'pytest']
        with mock.patch('platform.system', return_value='Linux'):
            # Make sure os.path.exists returns False for /.dockerenv to avoid container detection
            with mock.patch('os.path.exists', lambda path: path != '/.dockerenv'):
                support = CrossPlatformWorkerSupport()
                result = support.install_dependencies(dependencies=custom_deps)
                
                self.assertTrue(result)
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                self.assertEqual(args[0], sys.executable)
                self.assertEqual(args[1:3], ['-m', 'pip'])
                for dep in custom_deps:
                    self.assertIn(dep, args)
        
        # Test installation failure
        mock_run.reset_mock()
        # Create a side effect that raises an exception
        def raise_error(*args, **kwargs):
            raise subprocess.CalledProcessError(1, 'pip install')
        mock_run.side_effect = raise_error
        
        with mock.patch('platform.system', return_value='Linux'):
            # Make sure os.path.exists returns False for /.dockerenv to avoid container detection
            with mock.patch('os.path.exists', lambda path: path != '/.dockerenv'):
                support = CrossPlatformWorkerSupport()
                with self.assertLogs(level='ERROR') as cm:
                    result = support.install_dependencies()
                    self.assertFalse(result)
                    self.assertIn('Failed to install dependencies', cm.output[0])

    @mock.patch('platform.system')
    @mock.patch('subprocess.run')
    def test_hardware_detection(self, mock_run, mock_platform):
        """Test hardware detection for different platforms."""
        # Mock subprocess run to simulate hardware detection commands
        mock_process = mock.MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Test Linux hardware detection
        mock_platform.return_value = 'Linux'
        
        # Mock /proc/cpuinfo
        with mock.patch('builtins.open', mock.mock_open(read_data="model name : Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz")) as mock_file:
            with mock.patch('os.path.exists', lambda path: path != "/.dockerenv"):
                support = CrossPlatformWorkerSupport()
                hardware_info = support.detect_hardware()
                
                self.assertEqual(hardware_info['platform'], 'linux')
                self.assertIn('cpu', hardware_info)
                self.assertIn('memory', hardware_info)
                self.assertIn('gpu', hardware_info)
                self.assertIn('disk', hardware_info)
        
        # Test Windows hardware detection
        mock_platform.return_value = 'Windows'
        
        with mock.patch('builtins.open', mock.mock_open()) as mock_file:
            # Ensure we're not detecting as a container
            with mock.patch('os.path.exists', lambda path: path != '/.dockerenv'):
                # Mock psutil for Windows memory detection
                with mock.patch.dict('sys.modules', {'psutil': mock.MagicMock()}):
                    sys.modules['psutil'].virtual_memory.return_value.total = 16 * 1024**3  # 16 GB
                    
                    support = CrossPlatformWorkerSupport()
                    hardware_info = support.detect_hardware()
                    
                    self.assertEqual(hardware_info['platform'], 'windows')
                    self.assertIn('cpu', hardware_info)
                    self.assertIn('memory', hardware_info)
                    self.assertEqual(hardware_info['memory']['total_gb'], 16.0)
                    self.assertIn('gpu', hardware_info)
                    self.assertIn('disk', hardware_info)
        
        # Test macOS hardware detection
        mock_platform.return_value = 'Darwin'
        
        # Mock sysctl output for CPU model
        mock_process.stdout = "Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz"
        
        with mock.patch('builtins.open', mock.mock_open()) as mock_file:
            # Ensure we're not detecting as a container
            with mock.patch('os.path.exists', lambda path: path != '/.dockerenv'):
                support = CrossPlatformWorkerSupport()
                hardware_info = support.detect_hardware()
                
                self.assertEqual(hardware_info['platform'], 'darwin')
                self.assertIn('cpu', hardware_info)
                self.assertIn('memory', hardware_info)
                self.assertIn('gpu', hardware_info)
                self.assertIn('disk', hardware_info)
        
        # Test container hardware detection
        with mock.patch('os.path.exists', lambda path: path == '/.dockerenv'):
            with mock.patch('builtins.open', mock.mock_open()) as mock_file:
                support = CrossPlatformWorkerSupport()
                hardware_info = support.detect_hardware()
                
                self.assertEqual(hardware_info['platform'], 'container')
                self.assertIn('container', hardware_info)
                self.assertIn('cpu', hardware_info)
                self.assertIn('memory', hardware_info)
                self.assertIn('gpu', hardware_info)
                self.assertIn('disk', hardware_info)

    def test_startup_script_generation(self):
        """Test startup script generation for different platforms."""
        # Test Linux startup script
        with mock.patch('platform.system', return_value='Linux'):
            support = CrossPlatformWorkerSupport()
            script = support.get_startup_script(
                self.test_coordinator_url, 
                self.test_api_key,
                self.test_worker_id
            )
            
            self.assertIn('#!/bin/bash', script)
            self.assertIn(self.test_coordinator_url, script)
            self.assertIn(self.test_api_key, script)
            self.assertIn(self.test_worker_id, script)
            self.assertIn('export WORKER_ID', script)
            self.assertIn('mkdir -p logs', script)
            self.assertIn('nohup python3', script)
        
        # Test Windows startup script
        with mock.patch('platform.system', return_value='Windows'):
            support = CrossPlatformWorkerSupport()
            script = support.get_startup_script(
                self.test_coordinator_url, 
                self.test_api_key,
                self.test_worker_id
            )
            
            self.assertIn('@echo off', script)
            self.assertIn(self.test_coordinator_url, script)
            self.assertIn(self.test_api_key, script)
            self.assertIn(self.test_worker_id, script)
            self.assertIn('set WORKER_ID', script)
            self.assertIn('if not exist logs mkdir logs', script)
            self.assertIn('start /b python', script)
        
        # Test macOS startup script
        with mock.patch('platform.system', return_value='Darwin'):
            support = CrossPlatformWorkerSupport()
            script = support.get_startup_script(
                self.test_coordinator_url, 
                self.test_api_key,
                self.test_worker_id
            )
            
            self.assertIn('#!/bin/bash', script)
            self.assertIn(self.test_coordinator_url, script)
            self.assertIn(self.test_api_key, script)
            self.assertIn(self.test_worker_id, script)
            self.assertIn('export WORKER_ID', script)
            self.assertIn('mkdir -p logs', script)
            self.assertIn('nohup python3', script)
        
        # Test container startup script
        with mock.patch('os.path.exists', lambda path: path == '/.dockerenv'):
            support = CrossPlatformWorkerSupport()
            script = support.get_startup_script(
                self.test_coordinator_url, 
                self.test_api_key,
                self.test_worker_id
            )
            
            self.assertIn('#!/bin/bash', script)
            self.assertIn(self.test_coordinator_url, script)
            self.assertIn(self.test_api_key, script)
            self.assertIn(self.test_worker_id, script)
            self.assertIn('export WORKER_ID', script)
            self.assertIn('export CONTAINER_ENV=1', script)
            self.assertIn('mkdir -p /app/logs', script)
            self.assertIn('--container-mode', script)

    def test_path_conversion(self):
        """Test path conversion for different platforms."""
        test_path = "/home/user/test/path.txt"
        
        # Test Linux path conversion (should remain the same)
        with mock.patch('platform.system', return_value='Linux'):
            support = CrossPlatformWorkerSupport()
            converted_path = support.convert_path_for_platform(test_path)
            self.assertEqual(converted_path, str(Path(test_path)))
            self.assertIn('/', converted_path)
        
        # Test Windows path conversion (should use backslashes)
        with mock.patch('platform.system', return_value='Windows'):
            support = CrossPlatformWorkerSupport()
            converted_path = support.convert_path_for_platform(test_path)
            self.assertEqual(converted_path, str(Path(test_path)).replace('/', '\\'))
            self.assertIn('\\', converted_path)
        
        # Test macOS path conversion (should be like Linux)
        with mock.patch('platform.system', return_value='Darwin'):
            support = CrossPlatformWorkerSupport()
            converted_path = support.convert_path_for_platform(test_path)
            self.assertEqual(converted_path, str(Path(test_path)))
            self.assertIn('/', converted_path)
        
        # Test container path conversion (should be like Linux)
        with mock.patch('os.path.exists', lambda path: path == '/.dockerenv'):
            support = CrossPlatformWorkerSupport()
            converted_path = support.convert_path_for_platform(test_path)
            self.assertEqual(converted_path, str(Path(test_path)))
            self.assertIn('/', converted_path)

    def test_get_platform_info(self):
        """Test platform information collection."""
        # Test Linux platform info
        with mock.patch('platform.system', return_value='Linux'):
            support = CrossPlatformWorkerSupport()
            platform_info = support.get_platform_info()
            
            self.assertEqual(platform_info['platform'], 'linux')
            self.assertEqual(platform_info['system'], 'Linux')
            self.assertIn('release', platform_info)
            self.assertIn('architecture', platform_info)
        
        # Test Windows platform info
        with mock.patch('platform.system', return_value='Windows'):
            support = CrossPlatformWorkerSupport()
            platform_info = support.get_platform_info()
            
            self.assertEqual(platform_info['platform'], 'windows')
            self.assertEqual(platform_info['system'], 'Windows')
            self.assertIn('release', platform_info)
            self.assertIn('architecture', platform_info)
            self.assertIn('version', platform_info)
        
        # Test macOS platform info
        with mock.patch('platform.system', return_value='Darwin'):
            support = CrossPlatformWorkerSupport()
            platform_info = support.get_platform_info()
            
            self.assertEqual(platform_info['platform'], 'darwin')
            self.assertEqual(platform_info['system'], 'Darwin')
            self.assertIn('release', platform_info)
            self.assertIn('architecture', platform_info)
        
        # Test container platform info
        with mock.patch('os.path.exists', lambda path: path == '/.dockerenv'):
            support = CrossPlatformWorkerSupport()
            platform_info = support.get_platform_info()
            
            self.assertEqual(platform_info['platform'], 'container')
            self.assertIn('container_type', platform_info)


if __name__ == '__main__':
    unittest.main()