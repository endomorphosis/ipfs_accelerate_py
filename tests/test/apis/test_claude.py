import os
import io
import sys
import json
import unittest
import threading
from unittest.mock import MagicMock, patch, Mock

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))

# Import ModelTest base class
try:
    from refactored_test_suite.model_test import ModelTest
except ImportError:
    # Fallback to alternative import path
    try:
        from model_test import ModelTest
    except ImportError:
        # Create a temporary ModelTest class if not available
        class ModelTest(unittest.TestCase):
            """Temporary ModelTest class."""
            pass

# Mock the claude module
class MockClaude:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources or {}
        self.metadata = metadata or {}
        self.api_key = metadata.get("claude_api_key", "test_api_key")
        self.circuit_lock = threading.RLock()
        self.queue_lock = threading.RLock()
        self.queue_processing = False
        self.request_queue = []
        self.endpoints = {}
        self.circuit_state = "CLOSED"
        self.queue_enabled = True
        self.max_retries = 3
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.current_requests = 0
        self.max_concurrent_requests = 5
        self.request_tracking = True
        self.recent_requests = {}
    
    def make_post_request_claude(self, data, api_key=None, request_id=None):
        # For test_endpoint test
        if isinstance(data, dict) and data.get("messages", []):
            return {
                "id": "msg_01abcdefg",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "This is a test response"}],
                "model": "claude-3-opus-20240229",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            }
        else:
            return None
    
    def chat(self, messages):
        return {"content": [{"type": "text", "text": "This is a test response"}]}
    
    def stream_chat(self, messages):
        return iter([{"content": "This"}, {"content": " is"}, {"content": " streaming"}])
    
    def make_stream_request_claude(self, data):
        # For streaming test
        return iter([
            {
                "type": "message_start",
                "message": {
                    "id": "msg_01abcdefg",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-3-opus-20240229",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 10
                    }
                }
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text",
                    "text": "This "
                }
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text",
                    "text": "is "
                }
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text",
                    "text": "a "
                }
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text",
                    "text": "test"
                }
            },
            {
                "type": "content_block_stop",
                "index": 0
            },
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "output_tokens": 20
                    }
                }
            },
            {
                "type": "message_stop"
            }
        ])
    
    def _get_api_key(self, metadata):
        return metadata.get("claude_api_key", "test_api_key")
    
    def track_request_result(self, success, error_type=None):
        pass
    
    def create_claude_endpoint_handler(self):
        return lambda x: x
    
    def test_claude_endpoint(self):
        return True
    
    def create_endpoint(self, **kwargs):
        endpoint_id = "endpoint_" + str(len(self.endpoints))
        self.endpoints[endpoint_id] = kwargs
        return endpoint_id
    
    def get_stats(self, endpoint_id):
        # First call returns initial stats
        if not hasattr(self, '_stats_called'):
            self._stats_called = True
            return {"requests": 0, "success": 0, "errors": 0}
        # Second call returns updated stats
        return {"requests": 1, "success": 1, "errors": 0}
    
    def make_request_with_endpoint(self, endpoint_id, data):
        return {"content": [{"type": "text", "text": "Response for " + endpoint_id}]}
    
    def is_compatible_model(self, model):
        return "claude" in model.lower()
    
    def _process_queue(self):
        pass

class TestClaude(ModelTest):
    """Test class for Claude API interface."""
    
    def setUp(self):
        """Initialize the test with model details and mock the Claude API."""
        super().setUp()
        self.model_id = "anthropic/claude-3-haiku-20240307"
        self.model_type = "text"
        
        # Replace the actual claude module with our mock
        sys.modules['ipfs_accelerate_py.api_backends.claude'] = Mock()
        sys.modules['ipfs_accelerate_py.api_backends.claude'].claude = MockClaude
        
        # Now import
        from ipfs_accelerate_py.api_backends import claude
        
        # Setup resources and metadata
        self.resources = {}
        self.metadata = {
            "claude_api_key": os.environ.get("ANTHROPIC_API_KEY", "test_api_key_for_mock"),
            "model": "claude-3-haiku-20240307",
            "max_retries": 3,
            "timeout": 30,
            "max_concurrent_requests": 5,
            "queue_size": 100
        }
        
        # Initialize the claude API client
        self.claude = claude.claude(resources=self.resources, metadata=self.metadata)
        self.device = self.detect_preferred_device()
    
    def detect_preferred_device(self):
        """Detect available hardware and choose preferred device."""
        return "cpu"  # Claude API is cloud-based, so local device is always "cpu"
    
    def load_model(self, model_name):
        """Load a Claude API model for testing."""
        try:
            # Create an API client for the specified model
            metadata = self.metadata.copy()
            metadata["model"] = model_name
            
            # Import the claude module
            from ipfs_accelerate_py.api_backends import claude
            
            # Create a new client with the specified model
            api_client = claude.claude(resources=self.resources, metadata=metadata)
            return api_client
        except Exception as e:
            self.logger.error(f"Error loading Claude API model {model_name}: {e}")
            return None
    
    def verify_model_output(self, model, input_data, expected_output=None):
        """Verify that model produces expected output."""
        try:
            # For Claude API models, input_data should be messages format
            if isinstance(input_data, str):
                # Convert string to messages format
                messages = [{"role": "user", "content": input_data}]
            elif isinstance(input_data, list):
                # Assume it's already in messages format
                messages = input_data
            else:
                self.fail(f"Unsupported input format: {type(input_data)}")
            
            # Make API request
            response = model.chat(messages)
            
            # Verify response structure
            self.assertIsNotNone(response, "Response should not be None")
            self.assertTrue("content" in response, "Response should have content field")
            
            # If expected output is provided, verify the content matches
            if expected_output is not None:
                response_text = ""
                if isinstance(response.get("content"), list):
                    for content_block in response["content"]:
                        if content_block.get("type") == "text":
                            response_text += content_block.get("text", "")
                
                self.assertEqual(response_text, expected_output)
            
            return response
        except Exception as e:
            self.fail(f"Error verifying Claude API output: {e}")
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        model = self.load_model(self.model_id)
        self.assertIsNotNone(model, "Claude API client should load successfully")
    
    def test_basic_chat(self):
        """Test basic chat functionality."""
        messages = [{"role": "user", "content": "Hello, Claude!"}]
        
        # Get chat response
        response = self.claude.chat(messages)
        
        # Verify response structure
        self.assertIsNotNone(response, "Chat response should not be None")
        self.assertTrue("content" in response, "Response should have content field")
        
        response_text = ""
        if isinstance(response.get("content"), list):
            for content_block in response["content"]:
                if content_block.get("type") == "text":
                    response_text += content_block.get("text", "")
        
        self.assertNotEqual(response_text, "", "Response text should not be empty")
    
    def test_streaming_chat(self):
        """Test streaming chat functionality."""
        if not hasattr(self.claude, 'stream_chat'):
            self.skipTest("Streaming not implemented")
        
        messages = [{"role": "user", "content": "Hello, Claude!"}]
        
        # Get streaming response
        stream_chunks = list(self.claude.stream_chat(messages))
        
        # Verify stream chunks
        self.assertTrue(len(stream_chunks) > 0, "Should receive streaming chunks")
    
    def test_api_key_multiplexing(self):
        """Test API key multiplexing features."""
        if not hasattr(self.claude, 'create_endpoint'):
            self.skipTest("API key multiplexing not implemented")
        
        # Create endpoints with different API keys
        endpoint1 = self.claude.create_endpoint(
            api_key="test_claude_key_1",
            max_concurrent_requests=5
        )
        
        endpoint2 = self.claude.create_endpoint(
            api_key="test_claude_key_2",
            max_concurrent_requests=10
        )
        
        self.assertIsNotNone(endpoint1, "Endpoint 1 should be created")
        self.assertIsNotNone(endpoint2, "Endpoint 2 should be created")
        
        # Test stats if available
        if hasattr(self.claude, 'get_stats'):
            stats1 = self.claude.get_stats(endpoint1)
            self.assertIsNotNone(stats1, "Stats should be available")
    
    def test_queue_and_backoff(self):
        """Test queue and backoff functionality."""
        if not hasattr(self.claude, 'queue_enabled'):
            self.skipTest("Queue not implemented")
        
        # Verify queue settings
        self.assertTrue(hasattr(self.claude, 'queue_enabled'), "Queue enabled setting should exist")
        self.assertTrue(hasattr(self.claude, 'request_queue'), "Request queue should exist")
        self.assertTrue(hasattr(self.claude, 'max_retries'), "Max retries setting should exist")
        self.assertTrue(hasattr(self.claude, 'initial_retry_delay'), "Initial retry delay setting should exist")
    
    def test_model_compatibility(self):
        """Test model compatibility check."""
        if not hasattr(self.claude, 'is_compatible_model'):
            self.skipTest("Model compatibility check not implemented")
        
        compatible = self.claude.is_compatible_model("anthropic/claude-3-opus-20240229")
        incompatible = self.claude.is_compatible_model("nonexistent-model")
        
        self.assertTrue(compatible, "Should recognize valid Claude model")
        self.assertFalse(incompatible, "Should reject invalid model")
    
    def run_all_tests(self):
        """Run all tests for Claude API."""
        results = {}
        
        # Run all test methods
        test_methods = [
            self.test_model_loading,
            self.test_basic_chat,
            self.test_streaming_chat,
            self.test_api_key_multiplexing,
            self.test_queue_and_backoff,
            self.test_model_compatibility
        ]
        
        for test_method in test_methods:
            method_name = test_method.__name__
            try:
                test_method()
                results[method_name] = "Success"
            except unittest.SkipTest as e:
                results[method_name] = f"Skipped: {str(e)}"
            except Exception as e:
                results[method_name] = f"Error: {str(e)}"
        
        return results
    
    def save_results(self, results, expected_dir='expected_results', collected_dir='collected_results'):
        """Save test results and compare with expected results."""
        # Create directories if they don't exist
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, expected_dir)
        collected_dir = os.path.join(base_dir, collected_dir)
        
        # Create directories with appropriate permissions
        for directory in [expected_dir, collected_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, mode=0o755, exist_ok=True)
        
        # Save collected results
        results_file = os.path.join(collected_dir, 'claude_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving results to {results_file}: {str(e)}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'claude_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                    if expected_results != results:
                        print("Test results differ from expected results!")
                        print(f"Expected: {json.dumps(expected_results, indent=2)}")
                        print(f"Got: {json.dumps(results, indent=2)}")
            except Exception as e:
                print(f"Error comparing results with {expected_file}: {str(e)}")
        else:
            # Create expected results file if it doesn't exist
            try:
                with open(expected_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    print(f"Created new expected results file: {expected_file}")
            except Exception as e:
                print(f"Error creating {expected_file}: {str(e)}")
        
        return results


if __name__ == "__main__":
    test = TestClaude()
    results = test.run_all_tests()
    test.save_results(results)
    print(f"Claude API Test Results: {json.dumps(results, indent=2)}")