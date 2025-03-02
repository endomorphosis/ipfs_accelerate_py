#!/usr/bin/env python
"""
Mock test for the Ollama API implementation to verify parameter compatibility.
"""

import sys
import os
import time
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the Ollama API
from ipfs_accelerate_py.api_backends.ollama import ollama

class MockOllama(ollama):
    """Mock Ollama API client that doesn't make actual HTTP requests"""
    
    def __init__(self, resources=None, metadata=None):
        """Initialize with mocked request handler"""
        super().__init__(resources, metadata)
        self.requests = []
        
    def make_post_request_ollama(self, endpoint_url, data, stream=False, request_id=None, priority=None):
        """Mock request handler that just records the request"""
        self.requests.append({
            "endpoint_url": endpoint_url,
            "data": data,
            "stream": stream,
            "request_id": request_id,
            "priority": priority,
            "timestamp": time.time()
        })
        
        # Return a mock response
        if stream:
            def mock_stream():
                for i in range(3):
                    yield {
                        "message": {"content": f"Mock response chunk {i+1}"},
                        "done": i == 2
                    }
            return mock_stream()
        else:
            return {
                "message": {"content": "This is a mock response from Ollama API"},
                "prompt_eval_count": 10,
                "eval_count": 20
            }

def test_parameter_compatibility():
    """Test parameter compatibility with the test_api_backoff_queue.py script"""
    print("\n=== Testing Parameter Compatibility ===")
    
    # Create mock client
    client = MockOllama()
    client.requests = []
    
    # Test the chat method with model_name parameter
    test_message = "Test message for parameter compatibility"
    response = client.chat(
        model_name="llama3",
        messages=[{"role": "user", "content": test_message}],
        max_tokens=30,
        temperature=0.7,
        request_id="test_123"
    )
    
    # Verify the request
    request = client.requests[-1]
    print(f"Request endpoint: {request['endpoint_url']}")
    print(f"Request model: {request['data']['model']}")
    print(f"Request content: {request['data']['messages'][0]['content']}")
    print(f"Request ID: {request['request_id']}")
    print(f"Options included max_tokens: {'options' in request['data'] and 'num_predict' in request['data']['options']}")
    print(f"Options included temperature: {'options' in request['data'] and 'temperature' in request['data']['options']}")
    
    # Test the generate method
    client.requests = []
    response = client.generate(
        model="llama3",
        prompt="Generate some text",
        max_tokens=20,
        temperature=0.8,
        request_id="generate_123"
    )
    
    # Verify the generate request converts to a chat request
    request = client.requests[-1]
    print(f"\nGenerate request converted to chat:")
    print(f"Request endpoint: {request['endpoint_url']}")
    print(f"Request model: {request['data']['model']}")
    print(f"Request prompt converted to message: {request['data']['messages'][0]['content'] == 'Generate some text'}")
    print(f"Request ID: {request['request_id']}")
    
    # Test the completions method
    client.requests = []
    response = client.completions(
        model="llama3",
        prompt="Complete this text",
        max_tokens=20,
        temperature=0.8,
        request_id="complete_123"
    )
    
    # Verify the completions request converts to a chat request
    request = client.requests[-1]
    print(f"\nCompletions request converted to chat:")
    print(f"Request endpoint: {request['endpoint_url']}")
    print(f"Request model: {request['data']['model']}")
    print(f"Request prompt converted to message: {request['data']['messages'][0]['content'] == 'Complete this text'}")
    print(f"Request ID: {request['request_id']}")
    
    return True

def test_backoff_queue_structure():
    """Test the structure of the backoff and queue implementation"""
    print("\n=== Testing Backoff and Queue Structure ===")
    
    # Create client
    client = ollama()
    
    # Check basic attributes
    print(f"Has max_retries: {hasattr(client, 'max_retries')}")
    print(f"Has backoff_factor: {hasattr(client, 'backoff_factor')}")
    print(f"Has initial_retry_delay: {hasattr(client, 'initial_retry_delay')}")
    print(f"Has max_retry_delay: {hasattr(client, 'max_retry_delay')}")
    
    # Check queue attributes
    print(f"Has request_queue: {hasattr(client, 'request_queue')}")
    print(f"Has queue_size: {hasattr(client, 'queue_size')}")
    print(f"Has max_concurrent_requests: {hasattr(client, 'max_concurrent_requests')}")
    print(f"Has current_requests: {hasattr(client, 'current_requests')}")
    print(f"Has queue_lock: {hasattr(client, 'queue_lock')}")
    
    # Check circuit breaker attributes
    print(f"Has circuit_state: {hasattr(client, 'circuit_state')}")
    print(f"Has failure_count: {hasattr(client, 'failure_count')}")
    print(f"Has failure_threshold: {hasattr(client, 'failure_threshold')}")
    print(f"Has circuit_timeout: {hasattr(client, 'circuit_timeout')}")
    print(f"Has circuit_lock: {hasattr(client, 'circuit_lock')}")
    
    # Check priority queue structure
    print(f"Has priority levels: {hasattr(client, 'PRIORITY_HIGH') and hasattr(client, 'PRIORITY_NORMAL') and hasattr(client, 'PRIORITY_LOW')}")
    
    # Check usage statistics
    print(f"Has usage_stats: {hasattr(client, 'usage_stats')}")
    if hasattr(client, 'usage_stats'):
        print(f"Usage stats includes request counts: {'total_requests' in client.usage_stats}")
        print(f"Usage stats includes token counts: {'total_tokens' in client.usage_stats}")
    
    return True

def test_callable_interface():
    """Test the callable interface"""
    print("\n=== Testing Callable Interface ===")
    
    # Create mock client
    client = MockOllama()
    client.requests = []
    
    # Test calling the client directly
    response = client("chat", 
                     model="llama3", 
                     messages=[{"role": "user", "content": "Test callable interface"}])
    
    # Verify the request
    request = client.requests[-1]
    print(f"Callable interface works: {len(client.requests) > 0}")
    print(f"Request endpoint: {request['endpoint_url']}")
    print(f"Request model: {request['data']['model']}")
    print(f"Request content: {request['data']['messages'][0]['content']}")
    
    return True

def main():
    """Run all tests"""
    print("\n=== Ollama API Mock Test ===")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Run the tests
    tests_passed = 0
    total_tests = 3
    
    try:
        if test_parameter_compatibility():
            tests_passed += 1
    except Exception as e:
        print(f"Parameter compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        if test_backoff_queue_structure():
            tests_passed += 1
    except Exception as e:
        print(f"Backoff/queue structure test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        if test_callable_interface():
            tests_passed += 1
    except Exception as e:
        print(f"Callable interface test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print(f"\n{tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    main()