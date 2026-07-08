"""
Extended tests for unified CLI - tests actual command execution and error handling.

These tests validate:
- Command execution for all modules
- Error handling and exit codes
- Help system completeness
- Module import verification
"""

import unittest
import sys
import io
from unittest.mock import patch, Mock
from ipfs_accelerate_py import unified_cli


class TestUnifiedCLIExtended(unittest.TestCase):
    """Extended tests for unified CLI command execution and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv.copy()
        
    def tearDown(self):
        """Clean up after tests."""
        sys.argv = self.original_argv

    def test_help_command_all_modules(self):
        """Test that help works for all modules."""
        modules = ['github', 'docker', 'hardware', 'runner', 'ipfs-files', 'network']
        
        for module in modules:
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                sys.argv = ['unified_cli.py', module, '--help']
                
                try:
                    unified_cli.main()
                except SystemExit as e:
                    # Help should exit with 0
                    self.assertEqual(e.code, 0)
                    
                output = mock_stdout.getvalue()
                self.assertIn(module, output.lower())

    def test_invalid_module_error(self):
        """Test error handling for invalid module."""
        with patch('sys.stderr', new_callable=io.StringIO):
            sys.argv = ['unified_cli.py', 'invalid-module', 'some-command']
            
            with self.assertRaises(SystemExit) as cm:
                unified_cli.main()
                
            # Should exit with error code
            self.assertNotEqual(cm.exception.code, 0)

    def test_github_module_import(self):
        """Test github module import function."""
        module = unified_cli.import_kit_module('github')
        self.assertIsNotNone(module)
        self.assertTrue(hasattr(module, 'GitHubKit') or hasattr(module, 'get_github_kit'))

    def test_docker_module_import(self):
        """Test docker module import function."""
        module = unified_cli.import_kit_module('docker')
        self.assertIsNotNone(module)
        self.assertTrue(hasattr(module, 'DockerKit') or hasattr(module, 'get_docker_kit'))

    def test_hardware_module_import(self):
        """Test hardware module import function."""
        module = unified_cli.import_kit_module('hardware')
        self.assertIsNotNone(module)
        self.assertTrue(hasattr(module, 'HardwareKit') or hasattr(module, 'get_hardware_kit'))

    def test_runner_module_import(self):
        """Test runner module import function."""
        module = unified_cli.import_kit_module('runner')
        self.assertIsNotNone(module)
        self.assertTrue(hasattr(module, 'RunnerKit') or hasattr(module, 'get_runner_kit'))

    def test_ipfs_files_module_import(self):
        """Test ipfs_files module import function."""
        module = unified_cli.import_kit_module('ipfs_files')
        self.assertIsNotNone(module)
        self.assertTrue(hasattr(module, 'IPFSFilesKit') or hasattr(module, 'get_ipfs_files_kit'))

    def test_network_module_import(self):
        """Test network module import function."""
        module = unified_cli.import_kit_module('network')
        self.assertIsNotNone(module)
        self.assertTrue(hasattr(module, 'NetworkKit') or hasattr(module, 'get_network_kit'))

    def test_invalid_module_import_returns_none(self):
        """Test that invalid module import returns None."""
        module = unified_cli.import_kit_module('invalid_module_name')
        self.assertIsNone(module)

    def test_main_help_displays_all_modules(self):
        """Test that main --help displays all 6 modules."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            sys.argv = ['unified_cli.py', '--help']
            
            try:
                unified_cli.main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)
                
            output = mock_stdout.getvalue()
            # Check all modules are listed
            self.assertIn('github', output.lower())
            self.assertIn('docker', output.lower())
            self.assertIn('hardware', output.lower())
            self.assertIn('runner', output.lower())
            self.assertIn('ipfs-files', output.lower())
            self.assertIn('network', output.lower())

    def test_github_commands_listed_in_help(self):
        """Test that github --help shows all github commands."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            sys.argv = ['unified_cli.py', 'github', '--help']
            
            try:
                unified_cli.main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)
                
            output = mock_stdout.getvalue()
            # Check some GitHub commands are listed
            self.assertIn('list-repos', output.lower())
            self.assertIn('get-repo', output.lower())

    def test_docker_commands_listed_in_help(self):
        """Test that docker --help shows all docker commands."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            sys.argv = ['unified_cli.py', 'docker', '--help']
            
            try:
                unified_cli.main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)
                
            output = mock_stdout.getvalue()
            # Check some Docker commands are listed
            self.assertIn('run', output.lower())
            self.assertIn('list', output.lower())


if __name__ == '__main__':
    unittest.main()
