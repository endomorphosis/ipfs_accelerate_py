"""
Tests for Copilot SDK integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from ipfs_accelerate_py.copilot_sdk import CopilotSDK


class TestCopilotSDK:
    """Test Copilot SDK wrapper"""
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_init(self, mock_client_class):
        """Test CopilotSDK initialization"""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o")
        assert sdk.default_model == "gpt-4o"
        assert sdk.enable_cache is True
        assert sdk._client is None  # Not initialized yet
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', False)
    def test_init_without_sdk(self):
        """Test initialization fails without SDK"""
        with pytest.raises(ImportError) as exc_info:
            CopilotSDK()
        
        assert "GitHub Copilot SDK is not installed" in str(exc_info.value)
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_create_session(self, mock_client_class):
        """Test session creation"""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session._config = {"model": "gpt-4o"}
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o")
        
        # Create session (this will call async internally)
        # We need to mock the _run_async to avoid event loop issues
        with patch.object(sdk, '_run_async', return_value=mock_session):
            session = sdk.create_session(model="gpt-5")
            
            assert session == mock_session
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_send_message(self, mock_client_class):
        """Test sending a message"""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session._config = {"model": "gpt-4o"}
        mock_session.send = AsyncMock()
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o")
        
        # Mock the response
        expected_response = {
            "prompt": "Hello",
            "messages": [{"type": "message", "content": "Hi there!"}],
            "success": True
        }
        
        with patch.object(sdk, '_run_async', return_value=expected_response):
            result = sdk.send_message(mock_session, "Hello")
            
            assert result["success"] is True
            assert result["prompt"] == "Hello"
            assert len(result["messages"]) > 0
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_stream_message(self, mock_client_class):
        """Test streaming a message"""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session._config = {"model": "gpt-4o"}
        mock_session.send = AsyncMock()
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o")
        
        # Mock the streaming response
        expected_response = {
            "prompt": "Tell me a story",
            "chunks": ["Once", " upon", " a", " time"],
            "full_text": "Once upon a time",
            "success": True
        }
        
        chunks_received = []
        def on_chunk(chunk):
            chunks_received.append(chunk)
        
        with patch.object(sdk, '_run_async', return_value=expected_response):
            result = sdk.stream_message(mock_session, "Tell me a story", on_chunk=on_chunk)
            
            assert result["success"] is True
            assert result["prompt"] == "Tell me a story"
            assert "full_text" in result
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_destroy_session(self, mock_client_class):
        """Test session destruction"""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.destroy = AsyncMock()
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o")
        
        # Add session to active sessions
        session_id = id(mock_session)
        sdk._active_sessions[session_id] = mock_session
        
        # Mock the response
        expected_response = {
            "success": True,
            "message": f"Session {session_id} destroyed"
        }
        
        with patch.object(sdk, '_run_async', return_value=expected_response):
            result = sdk.destroy_session(mock_session)
            
            assert result["success"] is True
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.Tool')
    def test_register_tool(self, mock_tool_class, mock_client_class):
        """Test tool registration"""
        mock_tool = Mock()
        mock_tool_class.return_value = mock_tool
        
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o")
        
        def my_handler(invocation):
            return {"result": "success"}
        
        tool = sdk.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            handler=my_handler
        )
        
        assert tool == mock_tool
        assert "test_tool" in sdk._tools
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_stop(self, mock_client_class):
        """Test stopping the SDK"""
        mock_session1 = AsyncMock()
        mock_session1.destroy = AsyncMock()
        mock_session2 = AsyncMock()
        mock_session2.destroy = AsyncMock()
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o")
        sdk._client = mock_client
        sdk._active_sessions = {
            "session1": mock_session1,
            "session2": mock_session2
        }
        
        # Mock the response
        expected_response = {
            "success": True,
            "message": "Copilot SDK stopped"
        }
        
        with patch.object(sdk, '_run_async', return_value=expected_response):
            result = sdk.stop()
            
            assert result["success"] is True
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_context_manager(self, mock_client_class):
        """Test context manager support"""
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client_class.return_value = mock_client
        
        with patch('ipfs_accelerate_py.copilot_sdk.wrapper.CopilotSDK.stop') as mock_stop:
            mock_stop.return_value = {"success": True}
            
            with CopilotSDK(model="gpt-4o") as sdk:
                assert sdk.default_model == "gpt-4o"
            
            # Verify stop was called
            mock_stop.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
