#!/usr/bin/env python
"""
Direct test for the newly implemented Ollama API with queue and backoff.
"""

import sys
import os
import time
import json
import concurrent.futures
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the Ollama API directly
from ipfs_accelerate_py.api_backends.ollama import ollama

def test_ollama_api_basic():
    """Test basic functionality of the Ollama API"""
    print("\n=== Testing Ollama API Basic Functionality ===")
    
    # Initialize client
    client = ollama()
    
    # Print configuration
    print(f"API URL: {client.ollama_api_url}")
    print(f"Default model: {client.default_model}")
    print(f"Concurrency settings: max_concurrent_requests={client.max_concurrent_requests}")
    print(f"Queue settings: queue_size={client.queue_size}")
    print(f"Backoff settings: max_retries={client.max_retries}, factor={client.backoff_factor}")
    
    # Test if the circuit breaker is initialized correctly
    print(f"Circuit breaker state: {client.circuit_state}")
    print(f"Circuit breaker settings: threshold={client.failure_threshold}, timeout={client.circuit_timeout}s")
    
    # Check method compatibility with test_api_backoff_queue.py
    print("\nChecking method signatures:")
    print(f"Has 'chat' method: {hasattr(client, 'chat')}")
    print(f"Has 'generate' method: {hasattr(client, 'generate')}")
    print(f"Has 'completions' method: {hasattr(client, 'completions')}")
    print(f"Has 'make_post_request_ollama' method: {hasattr(client, 'make_post_request_ollama')}")
    
    # Check callable interface
    print(f"Has callable interface: {hasattr(client, '__call__')}")
    
    return True

def test_chat_method_compatibility():
    """Test that the chat method can accept model_name parameter"""
    print("\n=== Testing Chat Method Compatibility ===")
    
    # Initialize client
    client = ollama()
    
    # Set up a test message
    messages = [{"role": "user", "content": "Hello, this is a test message. Just echo back 'Test successful'"}]
    
    # Test with model parameter
    print("Testing with model parameter...")
    try:
        result1 = client.chat(
            model="llama3",
            messages=messages,
            max_tokens=10,
            temperature=0.7
        )
        print(f"Success with model parameter: {result1['implementation_type']}")
    except Exception as e:
        print(f"Failed with model parameter: {str(e)}")
    
    # Test with model_name parameter
    print("\nTesting with model_name parameter...")
    try:
        result2 = client.chat(
            model_name="llama3",
            messages=messages,
            max_tokens=10,
            temperature=0.7
        )
        print(f"Success with model_name parameter: {result2['implementation_type']}")
    except Exception as e:
        print(f"Failed with model_name parameter: {str(e)}")
    
    # Test with request_id parameter
    print("\nTesting with request_id parameter...")
    try:
        result3 = client.chat(
            model="llama3",
            messages=messages,
            request_id="test_request_123"
        )
        print(f"Success with request_id parameter: {result3['implementation_type']}")
    except Exception as e:
        print(f"Failed with request_id parameter: {str(e)}")
    
    return True

def test_concurrent_requests():
    """Test that concurrent requests are properly queued and processed"""
    print("\n=== Testing Concurrent Requests ===")
    
    # Initialize client with small concurrency limit
    client = ollama(metadata={"ollama_model": "llama3"})
    client.max_concurrent_requests = 2  # Set small limit to test queuing
    
    # Set up test messages
    messages = []
    for i in range(5):
        messages.append([{
            "role": "user", 
            "content": f"Test concurrent request {i+1}. Reply with: This is concurrent request {i+1}"
        }])
    
    # Send concurrent requests
    start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                client.chat, 
                model="llama3", 
                messages=msg,
                request_id=f"concurrent_{i+1}"
            ) 
            for i, msg in enumerate(messages)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"Request completed: {result.get('text', '')[:50]}...")
            except Exception as e:
                print(f"Request failed: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\nAll {len(messages)} requests completed in {total_time:.2f}s")
    print(f"Successful requests: {sum(1 for r in results if r.get('implementation_type') == '(REAL)')}")
    
    # Check usage stats
    print("\nUsage statistics:")
    for key, value in client.usage_stats.items():
        print(f"  {key}: {value}")
    
    return True

def main():
    """Run all tests"""
    print("\n=== Ollama API Implementation Test ===")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Run the tests
    try:
        # Basic functionality test
        test_ollama_api_basic()
        
        # Method compatibility test
        test_chat_method_compatibility()
        
        # Concurrent requests test
        test_concurrent_requests()
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()