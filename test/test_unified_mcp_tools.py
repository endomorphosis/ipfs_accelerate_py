"""
Unit tests for MCP unified tools - tests tool registration and schemas.

These tests validate:
- Tool registration for all modules
- Tool counts are correct
- All registration functions exist and are callable
"""

import unittest
from unittest.mock import Mock, patch
from ipfs_accelerate_py.mcp import unified_tools


class TestUnifiedMCPTools(unittest.TestCase):
    """Unit tests for MCP unified tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock()
        self.mock_mcp.register_tool = Mock()

    def test_register_github_tools_exists(self):
        """Test GitHub tools registration function exists."""
        self.assertTrue(callable(unified_tools.register_github_tools))
        
        # Should not raise exception
        unified_tools.register_github_tools(self.mock_mcp)
        
        # Should have been called
        self.assertTrue(self.mock_mcp.register_tool.called)

    def test_register_docker_tools_exists(self):
        """Test Docker tools registration function exists."""
        self.assertTrue(callable(unified_tools.register_docker_tools))
        
        unified_tools.register_docker_tools(self.mock_mcp)
        self.assertTrue(self.mock_mcp.register_tool.called)

    def test_register_hardware_tools_exists(self):
        """Test Hardware tools registration function exists."""
        self.assertTrue(callable(unified_tools.register_hardware_tools))
        
        unified_tools.register_hardware_tools(self.mock_mcp)
        self.assertTrue(self.mock_mcp.register_tool.called)

    def test_register_runner_tools_exists(self):
        """Test Runner tools registration function exists."""
        self.assertTrue(callable(unified_tools.register_runner_tools))
        
        unified_tools.register_runner_tools(self.mock_mcp)
        self.assertTrue(self.mock_mcp.register_tool.called)

    def test_register_ipfs_files_tools_exists(self):
        """Test IPFS Files tools registration function exists."""
        self.assertTrue(callable(unified_tools.register_ipfs_files_tools))
        
        unified_tools.register_ipfs_files_tools(self.mock_mcp)
        self.assertTrue(self.mock_mcp.register_tool.called)

    def test_register_network_tools_exists(self):
        """Test Network tools registration function exists."""
        self.assertTrue(callable(unified_tools.register_network_tools))
        
        unified_tools.register_network_tools(self.mock_mcp)
        self.assertTrue(self.mock_mcp.register_tool.called)

    def test_register_unified_tools_exists(self):
        """Test unified tools registration function exists."""
        self.assertTrue(callable(unified_tools.register_unified_tools))
        
        unified_tools.register_unified_tools(self.mock_mcp)
        self.assertTrue(self.mock_mcp.register_tool.called)

    def test_register_unified_tools_calls_all_modules(self):
        """Test that register_unified_tools calls all module registrations."""
        with patch.object(unified_tools, 'register_github_tools') as mock_github, \
             patch.object(unified_tools, 'register_docker_tools') as mock_docker, \
             patch.object(unified_tools, 'register_hardware_tools') as mock_hardware, \
             patch.object(unified_tools, 'register_runner_tools') as mock_runner, \
             patch.object(unified_tools, 'register_ipfs_files_tools') as mock_ipfs, \
             patch.object(unified_tools, 'register_network_tools') as mock_network:
            
            unified_tools.register_unified_tools(self.mock_mcp)
            
            # All module registrations should be called
            mock_github.assert_called_once_with(self.mock_mcp)
            mock_docker.assert_called_once_with(self.mock_mcp)
            mock_hardware.assert_called_once_with(self.mock_mcp)
            mock_runner.assert_called_once_with(self.mock_mcp)
            mock_ipfs.assert_called_once_with(self.mock_mcp)
            mock_network.assert_called_once_with(self.mock_mcp)

    def test_all_registration_functions_in_module(self):
        """Test that all expected registration functions exist in module."""
        expected_functions = [
            'register_github_tools',
            'register_docker_tools',
            'register_hardware_tools',
            'register_runner_tools',
            'register_ipfs_files_tools',
            'register_network_tools',
            'register_unified_tools'
        ]
        
        for func_name in expected_functions:
            self.assertTrue(
                hasattr(unified_tools, func_name),
                f"Missing function: {func_name}"
            )
            self.assertTrue(
                callable(getattr(unified_tools, func_name)),
                f"Not callable: {func_name}"
            )

    def test_github_tools_registration_count(self):
        """Test GitHub tools registration count."""
        unified_tools.register_github_tools(self.mock_mcp)
        
        # GitHub should register 6 tools
        self.assertEqual(self.mock_mcp.register_tool.call_count, 6)

    def test_docker_tools_registration_count(self):
        """Test Docker tools registration count."""
        unified_tools.register_docker_tools(self.mock_mcp)
        
        # Docker should register 4 tools
        self.assertEqual(self.mock_mcp.register_tool.call_count, 4)

    def test_hardware_tools_registration_count(self):
        """Test Hardware tools registration count."""
        unified_tools.register_hardware_tools(self.mock_mcp)
        
        # Hardware should register 3 tools
        self.assertEqual(self.mock_mcp.register_tool.call_count, 3)

    def test_runner_tools_registration_count(self):
        """Test Runner tools registration count."""
        unified_tools.register_runner_tools(self.mock_mcp)
        
        # Runner should register 7 tools
        self.assertEqual(self.mock_mcp.register_tool.call_count, 7)

    def test_ipfs_files_tools_registration_count(self):
        """Test IPFS Files tools registration count."""
        unified_tools.register_ipfs_files_tools(self.mock_mcp)
        
        # IPFS Files should register 7 tools
        self.assertEqual(self.mock_mcp.register_tool.call_count, 7)

    def test_network_tools_registration_count(self):
        """Test Network tools registration count."""
        unified_tools.register_network_tools(self.mock_mcp)
        
        # Network should register 8 tools
        self.assertEqual(self.mock_mcp.register_tool.call_count, 8)

    def test_total_tools_count(self):
        """Test total number of tools registered."""
        unified_tools.register_unified_tools(self.mock_mcp)
        
        # Should register 35 tools total (6+4+3+7+7+8)
        self.assertEqual(self.mock_mcp.register_tool.call_count, 35)


if __name__ == '__main__':
    unittest.main()
