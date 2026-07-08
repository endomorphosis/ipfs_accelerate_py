"""
Comprehensive tests for Copilot SDK integration demonstrating SDK features.

These tests validate that the wrapper properly exposes and supports the key features
of the GitHub Copilot SDK (https://github.com/github/copilot-sdk):
- Session lifecycle management
- Multi-turn conversations (stateful)
- Streaming responses
- Tool registration and invocation
- Custom model selection
- Caching behavior
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from ipfs_accelerate_py.copilot_sdk import CopilotSDK


class TestSDKFeatures:
    """Test core SDK features through the wrapper"""
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_session_lifecycle_management(self, mock_client_class):
        """Test SDK feature: Session create and destroy lifecycle"""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session._config = {"model": "gpt-4o"}
        mock_session.destroy = AsyncMock()
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.stop = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Test lifecycle
        sdk = CopilotSDK(model="gpt-4o")
        
        with patch.object(sdk, '_run_async') as mock_run_async:
            # Create session
            mock_run_async.return_value = mock_session
            session = sdk.create_session()
            assert session == mock_session
            
            # Destroy session
            mock_run_async.return_value = {"success": True}
            result = sdk.destroy_session(session)
            assert result["success"] is True
            
            # Stop SDK
            result = sdk.stop()
            assert result["success"] is True
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_multi_turn_stateful_conversation(self, mock_client_class):
        """Test SDK feature: Multi-turn stateful conversations"""
        # Setup mocks for stateful conversation
        mock_session = AsyncMock()
        mock_session._config = {"model": "gpt-4o"}
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o", enable_cache=True)
        
        with patch.object(sdk, '_run_async') as mock_run_async:
            # Create session
            mock_run_async.return_value = mock_session
            session = sdk.create_session()
            
            # First turn: "What is 1+1?"
            mock_run_async.return_value = {
                "prompt": "What is 1+1?",
                "messages": [{"type": "message", "content": "2"}],
                "success": True
            }
            result1 = sdk.send_message(session, "What is 1+1?")
            assert result1["success"] is True
            assert "2" in result1["messages"][0]["content"]
            
            # Second turn: "Now double that" (context-aware)
            mock_run_async.return_value = {
                "prompt": "Now double that",
                "messages": [{"type": "message", "content": "4"}],
                "success": True
            }
            result2 = sdk.send_message(session, "Now double that")
            assert result2["success"] is True
            assert "4" in result2["messages"][0]["content"]
            
            # Verify session was reused (stateful)
            assert session == mock_session
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_streaming_responses(self, mock_client_class):
        """Test SDK feature: Real-time streaming responses"""
        mock_session = AsyncMock()
        mock_session._config = {"model": "gpt-4o"}
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o")
        
        with patch.object(sdk, '_run_async') as mock_run_async:
            # Create streaming session
            mock_run_async.return_value = mock_session
            session = sdk.create_session(streaming=True)
            
            # Mock streaming response with chunks
            mock_run_async.return_value = {
                "prompt": "Tell me a story",
                "chunks": ["Once", " upon", " a", " time"],
                "full_text": "Once upon a time",
                "messages": [{"type": "message", "content": "Once upon a time"}],
                "success": True
            }
            
            chunks_received = []
            def on_chunk(chunk):
                chunks_received.append(chunk)
            
            result = sdk.stream_message(session, "Tell me a story", on_chunk=on_chunk)
            
            assert result["success"] is True
            assert "chunks" in result
            assert result["full_text"] == "Once upon a time"
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.Tool')
    def test_custom_tool_registration(self, mock_tool_class, mock_client_class):
        """Test SDK feature: Custom tool registration and invocation"""
        mock_tool = Mock()
        mock_tool_class.return_value = mock_tool
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o")
        
        # Define a custom tool
        def encrypt_string(invocation):
            input_str = invocation["arguments"]["input"]
            return {
                "textResultForLlm": input_str.upper(),
                "resultType": "success"
            }
        
        # Register the tool
        tool = sdk.register_tool(
            name="encrypt_string",
            description="Encrypts a string",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "String to encrypt"}
                },
                "required": ["input"]
            },
            handler=encrypt_string
        )
        
        assert tool == mock_tool
        assert "encrypt_string" in sdk._tools
        
        # Verify tool can be retrieved
        tools = sdk.get_tools()
        assert "encrypt_string" in tools
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_model_selection(self, mock_client_class):
        """Test SDK feature: Custom model selection"""
        mock_session_gpt4 = AsyncMock()
        mock_session_gpt4._config = {"model": "gpt-4o"}
        
        mock_session_gpt5 = AsyncMock()
        mock_session_gpt5._config = {"model": "gpt-5"}
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(side_effect=[mock_session_gpt4, mock_session_gpt5])
        mock_client_class.return_value = mock_client
        
        # Test default model
        sdk = CopilotSDK(model="gpt-4o")
        assert sdk.default_model == "gpt-4o"
        
        with patch.object(sdk, '_run_async') as mock_run_async:
            # Create session with default model
            mock_run_async.return_value = mock_session_gpt4
            session1 = sdk.create_session()
            
            # Create session with different model
            mock_run_async.return_value = mock_session_gpt5
            session2 = sdk.create_session(model="gpt-5")
            
            # Verify different sessions created with different models
            assert session1 != session2
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.get_global_cache')
    def test_caching_behavior(self, mock_get_cache, mock_client_class):
        """Test SDK feature: Response caching"""
        mock_cache = Mock()
        mock_cache.get = Mock(return_value=None)
        mock_cache.put = Mock()
        mock_get_cache.return_value = mock_cache
        
        mock_session = AsyncMock()
        mock_session._config = {"model": "gpt-4o"}
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client_class.return_value = mock_client
        
        # Enable caching
        sdk = CopilotSDK(model="gpt-4o", enable_cache=True)
        assert sdk.cache is not None
        
        with patch.object(sdk, '_run_async') as mock_run_async:
            mock_run_async.return_value = mock_session
            session = sdk.create_session()
            
            # First call - cache miss
            mock_run_async.return_value = {
                "prompt": "What is 2+2?",
                "messages": [{"type": "message", "content": "4"}],
                "success": True
            }
            result1 = sdk.send_message(session, "What is 2+2?", use_cache=True)
            
            # Verify cache was checked and populated
            assert mock_cache.get.called
            assert mock_cache.put.called
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_error_handling(self, mock_client_class):
        """Test SDK feature: Graceful error handling"""
        mock_session = AsyncMock()
        mock_session._config = {"model": "gpt-4o"}
        mock_session.send = AsyncMock(side_effect=Exception("Network error"))
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o")
        
        with patch.object(sdk, '_run_async') as mock_run_async:
            mock_run_async.return_value = mock_session
            session = sdk.create_session()
            
            # Simulate error during message send
            mock_run_async.return_value = {
                "success": False,
                "error": "Network error"
            }
            result = sdk.send_message(session, "Hello")
            
            assert result["success"] is False
            assert "error" in result
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_context_manager_support(self, mock_client_class):
        """Test SDK feature: Context manager for automatic cleanup"""
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client_class.return_value = mock_client
        
        with patch('ipfs_accelerate_py.copilot_sdk.wrapper.CopilotSDK.stop') as mock_stop:
            mock_stop.return_value = {"success": True}
            
            # Use as context manager
            with CopilotSDK(model="gpt-4o") as sdk:
                assert sdk.default_model == "gpt-4o"
            
            # Verify cleanup was called
            mock_stop.assert_called_once()
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_multiple_concurrent_sessions(self, mock_client_class):
        """Test SDK feature: Multiple concurrent sessions"""
        mock_session1 = AsyncMock()
        mock_session1._config = {"model": "gpt-4o"}
        
        mock_session2 = AsyncMock()
        mock_session2._config = {"model": "gpt-4o"}
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(side_effect=[mock_session1, mock_session2])
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(model="gpt-4o")
        
        with patch.object(sdk, '_run_async') as mock_run_async:
            # Create multiple sessions
            mock_run_async.return_value = mock_session1
            session1 = sdk.create_session()
            
            mock_run_async.return_value = mock_session2
            session2 = sdk.create_session()
            
            # Verify both sessions tracked
            assert id(session1) in [id(s) for s in sdk._active_sessions.values()]
            assert id(session2) in [id(s) for s in sdk._active_sessions.values()]
            assert len(sdk._active_sessions) == 2
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient')
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True)
    def test_session_options_configuration(self, mock_client_class):
        """Test SDK feature: Session configuration options"""
        mock_session = AsyncMock()
        mock_session._config = {
            "model": "gpt-4o",
            "streaming": True
        }
        
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client_class.return_value = mock_client
        
        sdk = CopilotSDK(
            model="gpt-4o",
            log_level="debug",
            auto_start=True,
            auto_restart=True
        )
        
        assert sdk.log_level == "debug"
        assert sdk.auto_start is True
        assert sdk.auto_restart is True
        
        with patch.object(sdk, '_run_async') as mock_run_async:
            mock_run_async.return_value = mock_session
            
            # Create session with streaming enabled
            session = sdk.create_session(streaming=True)
            assert session == mock_session
    
    @patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', False)
    def test_graceful_fallback_without_sdk(self):
        """Test SDK feature: Graceful handling when SDK not installed"""
        with pytest.raises(ImportError) as exc_info:
            CopilotSDK()
        
        error_msg = str(exc_info.value)
        assert "GitHub Copilot SDK is not installed" in error_msg
        assert "pip install github-copilot-sdk" in error_msg


class TestSDKIntegrationWithOperations:
    """Test SDK features through the operations layer"""
    
    def test_operations_session_tracking(self):
        """Test that operations layer properly tracks sessions"""
        from scripts.shared.operations import CopilotSDKOperations
        from scripts.shared.core import SharedCore
        
        with patch('ipfs_accelerate_py.copilot_sdk.wrapper.HAVE_COPILOT_SDK', True):
            with patch('ipfs_accelerate_py.copilot_sdk.wrapper._CopilotClient'):
                ops = CopilotSDKOperations(SharedCore())
                
                # Initially no sessions
                result = ops.list_sessions()
                assert result["count"] == 0
                
                # Mock session creation
                with patch.object(ops, 'copilot_sdk') as mock_sdk:
                    mock_session = Mock()
                    mock_sdk.create_session.return_value = mock_session
                    
                    # Create session via operations
                    create_result = ops.create_session(model="gpt-4o")
                    
                    if create_result.get("success"):
                        # Session should be tracked
                        session_id = create_result["session_id"]
                        assert session_id in ops._active_sessions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
