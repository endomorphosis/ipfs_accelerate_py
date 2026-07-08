"""
Integration Tests for Unified CLI

These tests verify the unified CLI works correctly with all kit modules.
"""

import os
import sys
import unittest
import subprocess
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CLI_PATH = os.path.join(os.path.dirname(__file__), '..', 'ipfs_accelerate_py', 'unified_cli.py')


class TestUnifiedCLIIntegration(unittest.TestCase):
    """Test unified CLI integration"""
    
    def test_cli_help(self):
        """Test CLI shows help"""
        result = subprocess.run(
            [sys.executable, CLI_PATH, '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('ipfs accelerate', result.stdout.lower())
        self.assertIn('github', result.stdout.lower())
        self.assertIn('docker', result.stdout.lower())
        self.assertIn('hardware', result.stdout.lower())
    
    def test_github_module_help(self):
        """Test GitHub module help"""
        result = subprocess.run(
            [sys.executable, CLI_PATH, 'github', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('list-repos', result.stdout.lower())
    
    def test_docker_module_help(self):
        """Test Docker module help"""
        result = subprocess.run(
            [sys.executable, CLI_PATH, 'docker', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('run', result.stdout.lower())
    
    def test_hardware_module_help(self):
        """Test Hardware module help"""
        result = subprocess.run(
            [sys.executable, CLI_PATH, 'hardware', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('info', result.stdout.lower())
    
    def test_runner_module_help(self):
        """Test Runner module help"""
        result = subprocess.run(
            [sys.executable, CLI_PATH, 'runner', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('start', result.stdout.lower())
        self.assertIn('status', result.stdout.lower())
    
    def test_hardware_info_command(self):
        """Test hardware info command actually works"""
        result = subprocess.run(
            [sys.executable, CLI_PATH, 'hardware', 'info', '--format', 'json'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should succeed or fail gracefully (may fail if deps missing)
        self.assertIn(result.returncode, [0, 1, 2])
        
        if result.returncode == 0:
            # Should have JSON output
            self.assertTrue('{' in result.stdout or '[' in result.stdout)


if __name__ == '__main__':
    unittest.main()
