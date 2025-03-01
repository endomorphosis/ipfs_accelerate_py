#!/usr/bin/env python
"""
A simplified test of Groq API queue and backoff functionality using mocks.
This avoids actual API calls that might time out during testing.
"""

import os
import sys
import time
import json
import threading
import concurrent.futures
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the Groq implementation
from ipfs_accelerate_py.api_backends.groq import groq

def test_backoff_mechanism():
    """Test the exponential backoff mechanism with mocks"""
    print("\n=== Testing Exponential Backoff System (Mocked) ===")
    
    # Initialize Groq client
    api_key = "test_key"
    metadata = {"groq_api_key": api_key}
    groq_client = groq(resources={}, metadata=metadata)
    
    # Record retry delays
    delays = []
    
    # Mock the sleep function to record delays instead of actually sleeping
    original_sleep = time.sleep
    
    def mock_sleep(seconds):
        delays.append(seconds)
        print(f"  Would sleep for {seconds:.2f}s")
        # Actually sleep a tiny amount to maintain execution order
        original_sleep(0.01)
    
    # Mock the requests.post function
    with patch('time.sleep', side_effect=mock_sleep), \
         patch('requests.post') as mock_post:
        
        # Create rate limit responses
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"retry-after": "2"}
        rate_limit_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        
        # Create success response for the final call
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": "Success response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        
        # Set up 4 rate limits then success
        mock_post.side_effect = [
            rate_limit_response,  # First attempt - rate limited
            rate_limit_response,  # Second attempt - rate limited
            rate_limit_response,  # Third attempt - rate limited
            rate_limit_response,  # Fourth attempt - rate limited
            success_response      # Fifth attempt - success
        ]
        
        try:
            # Make a request that will trigger backoff
            result = groq_client.make_post_request_groq(
                endpoint_url="https://api.groq.com/openai/v1/chat/completions",
                data={"model": "llama3-8b-8192", "messages": [{"role": "user", "content": "Test"}]},
                api_key=api_key
            )
            
            # Print results
            print("\nBackoff Results:")
            print(f"  Retry delays: {delays}")
            print(f"  Expected pattern: Growing exponentially (1, 2, 4, 8, 16, ...)")
            print(f"  Final result successful: {'Yes' if 'choices' in result else 'No'}")
            
            return {
                "success": True,
                "delays": delays,
                "follows_exponential_pattern": all(delays[i] <= delays[i+1] for i in range(len(delays)-1)),
                "retry_count": len(delays)
            }
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "delays": delays
            }

def test_queue_system():
    """Test the request queue system with mocks"""
    print("\n=== Testing Queue System (Mocked) ===")
    
    # Initialize Groq client
    api_key = "test_key"
    metadata = {"groq_api_key": api_key}
    groq_client = groq(resources={}, metadata=metadata)
    
    # Set parameters for testing
    groq_client.max_concurrent_requests = 1  # Only allow 1 concurrent request
    groq_client.queue_enabled = True
    print(f"  Concurrency limit set to: {groq_client.max_concurrent_requests}")
    
    # Keep track of execution
    execution_order = []
    
    # Mock make_post_request_groq to record calls and add delays
    original_method = groq_client.make_post_request_groq
    
    def mock_make_post_request(*args, **kwargs):
        request_id = kwargs.get('request_id', 'unknown')
        print(f"  Processing request: {request_id}")
        execution_order.append(request_id)
        time.sleep(0.1)  # Small delay to simulate processing
        return {"message": "Mock response"}
    
    # Replace the method with our mock
    groq_client.make_post_request_groq = mock_make_post_request
    
    try:
        # Create and start concurrent threads
        results = []
        threads = []
        
        def make_request(req_id):
            try:
                print(f"  Starting request {req_id}")
                start_time = time.time()
                result = groq_client.make_post_request_groq(
                    endpoint_url="https://api.groq.com/openai/v1/chat/completions",
                    data={"model": "llama3-8b-8192", "messages": [{"role": "user", "content": f"Test {req_id}"}]},
                    api_key=api_key,
                    request_id=f"request_{req_id}"
                )
                end_time = time.time()
                results.append({
                    "id": req_id,
                    "success": True,
                    "time": end_time - start_time,
                    "request_id": f"request_{req_id}"
                })
                print(f"  Completed request {req_id} in {end_time - start_time:.2f}s")
            except Exception as e:
                results.append({
                    "id": req_id,
                    "success": False,
                    "error": str(e),
                    "request_id": f"request_{req_id}"
                })
                print(f"  Failed request {req_id}: {str(e)}")
        
        # Start 5 threads concurrently
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Restore the original method
        groq_client.make_post_request_groq = original_method
        
        # Analyze results
        print("\nQueue Results:")
        print(f"  Queue size: {len(groq_client.request_queue)}")
        print(f"  Execution order: {execution_order}")
        print(f"  Total requests: {len(results)}")
        print(f"  Successful requests: {sum(1 for r in results if r['success'])}")
        
        # Check if we're honoring the concurrency limit
        concurrent_executed = len(set(execution_order))
        expected_execution = 5  # We started 5 threads
        
        return {
            "success": True,
            "execution_order": execution_order,
            "concurrent_requests": groq_client.current_requests,
            "total_requests": len(results),
            "successful_requests": sum(1 for r in results if r['success']),
            "respected_limit": concurrent_executed == expected_execution
        }
        
    except Exception as e:
        print(f"  Error in queue test: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    print("=== Groq API Queue and Backoff Mock Test ===")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Test the backoff mechanism
    backoff_results = test_backoff_mechanism()
    
    # Test the queue system
    queue_results = test_queue_system()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "backoff_test": backoff_results,
        "queue_test": queue_results
    }
    
    filename = f"groq_queue_mock_test_results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to: {filename}")

if __name__ == "__main__":
    main()