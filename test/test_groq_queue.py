#!/usr/bin/env python
import os
import sys
import time
import threading
import concurrent.futures
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the Groq implementation with our queue/backoff enhancements
from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.groq import groq

def send_request(groq_client, message, model="llama3-8b-8192", tag=""):
    """Send a request and capture the timing information"""
    start_time = time.time()
    request_id = f"req_{tag}_{time.time()}"
    
    try:
        response = groq_client.chat(
            model_name=model,
            messages=[{"role": "user", "content": message}],
            max_tokens=30,
            temperature=0.7,
            request_id=request_id
        )
        
        end_time = time.time()
        
        return {
            "success": True,
            "request_id": request_id,
            "tag": tag,
            "time_taken": end_time - start_time,
            "text": response.get("text", ""),
            "model": model,
            "thread": threading.current_thread().name
        }
    except Exception as e:
        end_time = time.time()
        
        return {
            "success": False,
            "request_id": request_id,
            "tag": tag,
            "time_taken": end_time - start_time,
            "error": str(e),
            "model": model,
            "thread": threading.current_thread().name
        }

def test_backoff_system():
    """Test the backoff system by deliberately hitting rate limits"""
    print("\n=== Testing Exponential Backoff System ===")
    
    # Initialize Groq client
    api_key = "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"
    metadata = {"groq_api_key": api_key}
    groq_client = groq(resources={}, metadata=metadata)
    
    # Make fewer requests to complete faster
    results = []
    
    for i in range(3):
        print(f"Sending request {i+1}/3...")
        result = send_request(
            groq_client, 
            f"Count from 1 to 3. Request #{i+1}", 
            tag=f"backoff_{i+1}"
        )
        
        results.append(result)
        print(f"  {'✓' if result['success'] else '✗'} Request {i+1} completed in {result['time_taken']:.2f}s")
        
        # Pause briefly to ensure we don't overload the API
        if i < 4:  # Don't sleep after the last request
            time.sleep(0.1)
    
    # Print summary
    print("\nBackoff Test Results:")
    for i, result in enumerate(results):
        if result["success"]:
            print(f"Request {i+1}: Success in {result['time_taken']:.2f}s - {result['text'][:30]}...")
        else:
            print(f"Request {i+1}: Failed in {result['time_taken']:.2f}s - {result['error']}")
    
    return results

def test_queue_system():
    """Test the queue system by sending concurrent requests"""
    print("\n=== Testing Queue System ===")
    
    # Initialize Groq client
    api_key = "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"
    metadata = {"groq_api_key": api_key}
    groq_client = groq(resources={}, metadata=metadata)
    
    # Set concurrency limit artificially low for testing
    groq_client.max_concurrent_requests = 2
    print(f"Concurrency limit set to: {groq_client.max_concurrent_requests}")
    
    # Prepare to send fewer concurrent requests to complete faster
    num_requests = 4
    messages = [
        f"What is {i+1} + {i+2}? Answer with just the number." for i in range(num_requests)
    ]
    
    # Use a thread pool to simulate concurrent requests
    start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(send_request, groq_client, msg, "llama3-8b-8192", f"queue_{i+1}"): i 
            for i, msg in enumerate(messages)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Request {idx+1} completed: {'✓ Success' if result['success'] else '✗ Failed'} in {result['time_taken']:.2f}s")
            except Exception as e:
                print(f"Request {idx+1} raised exception: {str(e)}")
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\nAll {num_requests} requests completed in {total_time:.2f}s")
    print(f"Average time per request: {total_time/num_requests:.2f}s")
    
    # Sort results by request ID to show order of execution
    results.sort(key=lambda x: x["tag"])
    
    print("\nQueue Test Results:")
    for result in results:
        if result["success"]:
            print(f"{result['tag']}: {result['thread']} - {result['time_taken']:.2f}s - {result['text'][:30]}...")
        else:
            print(f"{result['tag']}: {result['thread']} - {result['time_taken']:.2f}s - {result['error']}")
    
    return results

def main():
    print("=== Groq API Queue and Backoff Test ===")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Test the backoff system
    backoff_results = test_backoff_system()
    
    # Shorter wait to speed up testing
    print("\nWaiting 3 seconds before testing queue system...")
    time.sleep(3)
    
    # Test the queue system
    queue_results = test_queue_system()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "backoff_test": backoff_results,
        "queue_test": queue_results
    }
    
    filename = f"groq_queue_test_results_{timestamp}.json"
    import json
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to: {filename}")

if __name__ == "__main__":
    main()