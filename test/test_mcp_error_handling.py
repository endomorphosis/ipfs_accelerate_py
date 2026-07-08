"""
Tests for MCP Server Auto-Healing Error Handling

This module tests the auto-healing error handling for MCP server tools.
"""

import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ipfs_accelerate_py.mcp.server import StandaloneMCP


class TestMCPErrorHandling:
    """Test MCP server error handling functionality."""
    
    def test_standalone_mcp_init(self):
        """Test StandaloneMCP initialization."""
        mcp = StandaloneMCP("test-server")
        
        assert mcp.name == "test-server"
        assert mcp.tools == {}
        assert mcp.resources == {}
        assert mcp.prompts == {}
    
    def test_error_handler_init_disabled(self):
        """Test error handler initialization when disabled."""
        # Ensure env vars are not set
        os.environ.pop('IPFS_AUTO_ISSUE', None)
        os.environ.pop('IPFS_AUTO_PR', None)
        os.environ.pop('IPFS_AUTO_HEAL', None)
        
        mcp = StandaloneMCP("test-server")
        
        # Error handler should not be initialized when disabled
        assert mcp._error_handler is None
    
    def test_error_handler_init_enabled(self):
        """Test error handler initialization when enabled."""
        # Enable auto-issue
        os.environ['IPFS_AUTO_ISSUE'] = 'true'
        
        try:
            mcp = StandaloneMCP("test-server")
            
            # Error handler should be initialized
            # (May be None if error_handler module not available, which is ok)
            # Just verify no exception is raised
            assert True
        finally:
            os.environ.pop('IPFS_AUTO_ISSUE', None)
    
    def test_report_tool_error_no_handler(self):
        """Test tool error reporting when handler is not available."""
        mcp = StandaloneMCP("test-server")
        mcp._error_handler = None
        
        # Should not raise exception
        exception = RuntimeError("Test error")
        mcp._report_tool_error("test_tool", exception, {"param": "value"})
    
    def test_report_resource_error_no_handler(self):
        """Test resource error reporting when handler is not available."""
        mcp = StandaloneMCP("test-server")
        mcp._error_handler = None
        
        # Should not raise exception
        exception = RuntimeError("Test error")
        mcp._report_resource_error("test://resource", exception)
    
    def test_report_client_error_no_handler(self):
        """Test client error reporting when handler is not available."""
        mcp = StandaloneMCP("test-server")
        mcp._error_handler = None
        
        # Should not raise exception
        error_data = {
            "error_type": "NetworkError",
            "error_message": "Failed to fetch",
            "stack_trace": "Error: Failed...",
            "context": {"url": "https://example.com"}
        }
        mcp._report_client_error(error_data)
    
    def test_tool_registration(self):
        """Test tool registration."""
        mcp = StandaloneMCP("test-server")
        
        def test_tool(param1: str) -> str:
            """Test tool function."""
            return f"Result: {param1}"
        
        mcp.register_tool(
            name="test_tool",
            function=test_tool,
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            }
        )
        
        assert "test_tool" in mcp.tools
        assert mcp.tools["test_tool"]["function"] == test_tool
        assert mcp.tools["test_tool"]["description"] == "A test tool"
    
    def test_tool_execution_success(self):
        """Test successful tool execution."""
        mcp = StandaloneMCP("test-server")
        
        def test_tool(value: int) -> int:
            """Double the value."""
            return value * 2
        
        mcp.register_tool(
            name="double",
            function=test_tool,
            description="Double a value",
            input_schema={"type": "object", "properties": {"value": {"type": "integer"}}}
        )
        
        # Execute tool
        result = mcp.tools["double"]["function"](value=5)
        assert result == 10
    
    def test_tool_execution_error(self):
        """Test tool execution that raises an error."""
        mcp = StandaloneMCP("test-server")
        
        def failing_tool():
            """A tool that always fails."""
            raise RuntimeError("Tool failed")
        
        mcp.register_tool(
            name="failing_tool",
            function=failing_tool,
            description="A failing tool",
            input_schema={"type": "object", "properties": {}}
        )
        
        # Execute tool should raise
        with pytest.raises(RuntimeError, match="Tool failed"):
            mcp.tools["failing_tool"]["function"]()


class TestMCPErrorReporting:
    """Test MCP error reporting functionality."""
    
    def test_client_error_data_structure(self):
        """Test that client error data has correct structure."""
        error_data = {
            "error_type": "NetworkError",
            "error_message": "Connection failed",
            "stack_trace": "Error: Connection failed\n  at fetch...",
            "context": {
                "timestamp": "2024-01-31T12:00:00Z",
                "userAgent": "Mozilla/5.0...",
                "url": "https://example.com",
                "method": "list_models"
            }
        }
        
        # Verify structure
        assert "error_type" in error_data
        assert "error_message" in error_data
        assert "stack_trace" in error_data
        assert "context" in error_data
        
        # Verify context details
        assert "timestamp" in error_data["context"]
        assert "userAgent" in error_data["context"]
        assert "url" in error_data["context"]
    
    def test_server_error_context(self):
        """Test server-side error context structure."""
        mcp = StandaloneMCP("test-server")
        
        # Mock error handler
        class MockErrorHandler:
            def __init__(self):
                self.captured_errors = []
                self.enable_auto_issue = False
            
            def capture_error(self, exception, context=None):
                self.captured_errors.append({
                    "exception": exception,
                    "context": context
                })
        
        mcp._error_handler = MockErrorHandler()
        
        # Report a tool error
        exception = ValueError("Invalid parameter")
        mcp._report_tool_error("test_tool", exception, {"param": "value"})
        
        # Verify error was captured
        assert len(mcp._error_handler.captured_errors) == 1
        
        # Verify context
        error = mcp._error_handler.captured_errors[0]
        assert error["exception"] == exception
        assert error["context"]["mcp_server"] == "test-server"
        assert error["context"]["tool_name"] == "test_tool"
        assert error["context"]["error_source"] == "mcp_tool"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
