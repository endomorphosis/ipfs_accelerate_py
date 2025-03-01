#!/usr/bin/env python
"""
Comprehensive test for Ollama API backoff queue functionality.
This script:
1. Tests the basic queue and backoff mechanism
2. Tests different queue sizes and concurrent request limits
3. Tests handling of rate limits and error recovery
4. Compares performance with and without the queue system
"""

import sys
import os
import time
import json
import threading
import concurrent.futures
import random
import argparse
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ipfs_accelerate_py"))
from api_backends import ollama

# Default settings
DEFAULT_MODEL = "llama3"
DEFAULT_HOST = "http://localhost:11434"
NUM_REQUESTS = 8
MAX_CONCURRENT = 2
QUEUE_SIZE = 20

def send_request(client, prompt, model, tag="test"):
    """Send a request to the Ollama API and track metrics"""
    request_id = f"req_{tag}_{int(time.time())}"
    start_time = time.time()
    
    try:
        # Send the actual request
        response = client.generate(
            model=model,
            prompt=prompt,
            max_tokens=30,
            temperature=0.7,
            request_id=request_id
        )
        
        success = True
        error = None
        # Extract content from response
        content = response.get("text", response.get("generated_text", str(response)))
    except Exception as e:
        success = False
        error = str(e)
        content = None
    
    end_time = time.time()
    
    # Return metrics
    return {
        "success": success,
        "request_id": request_id,
        "time_taken": end_time - start_time,
        "content": content[:100] if content else None,  # Truncate long content
        "error": error,
        "tag": tag,
        "thread": threading.current_thread().name,
        "timestamp": datetime.now().isoformat()
    }

def test_basic_queue(client, model, num_requests=5, concurrent_requests=2):
    """Basic test of the queue system by sending multiple concurrent requests"""
    print(f"\n=== Testing Basic Queue Functionality ===")
    
    # Set concurrency limit
    if hasattr(client, "max_concurrent_requests"):
        original_limit = client.max_concurrent_requests
        client.max_concurrent_requests = concurrent_requests
        print(f"Set concurrent requests limit to {concurrent_requests} (was {original_limit})")
    
    # Generate test prompts
    prompts = [
        f"What is {i+1} + {i+2}? Respond with just the number." 
        for i in range(num_requests)
    ]
    
    # Send requests concurrently
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(send_request, client, prompt, model, f"basic_{i+1}"): i
            for i, prompt in enumerate(prompts)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
                status = "✓" if result["success"] else "✗"
                print(f"Request {idx+1}/{num_requests}: {status} in {result['time_taken']:.2f}s")
            except Exception as e:
                print(f"Request {idx+1}/{num_requests} raised exception: {str(e)}")
    
    total_time = time.time() - start_time
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    print(f"\nBasic Queue Test - {successful}/{num_requests} successful in {total_time:.2f}s")
    print(f"Average time per request: {total_time/num_requests:.2f}s")
    
    # Reset concurrency limit if we changed it
    if hasattr(client, "max_concurrent_requests"):
        client.max_concurrent_requests = original_limit
    
    return results

def test_backoff_recovery(client, model, num_requests=3):
    """Test the backoff and recovery mechanism by simulating errors"""
    print(f"\n=== Testing Backoff and Recovery ===")
    
    # Check if client has required attributes
    has_backoff = hasattr(client, "max_retries") and hasattr(client, "backoff_factor")
    if has_backoff:
        print(f"Client configuration: max_retries={client.max_retries}, " +
              f"backoff_factor={client.backoff_factor}")
    else:
        print("Warning: Client does not have backoff attributes configured")
    
    # Generate test prompts that are likely to cause errors (very large context)
    prompts = [
        "Explain the theory of relativity in extreme detail." * 10  # Large prompt to potentially trigger errors
        for _ in range(num_requests)
    ]
    
    # Send requests with small delay to avoid overwhelming the API
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Sending potentially problematic request {i+1}/{num_requests}...")
        result = send_request(client, prompt, model, f"backoff_{i+1}")
        results.append(result)
        
        status = "✓" if result["success"] else "✗"
        print(f"  Request {i+1}: {status} in {result['time_taken']:.2f}s")
        
        if i < num_requests - 1:
            time.sleep(0.5)  # Small delay between requests
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    print(f"\nBackoff Test - {successful}/{num_requests} successful")
    for i, result in enumerate(results):
        if result["success"]:
            print(f"Request {i+1}: Success in {result['time_taken']:.2f}s")
        else:
            print(f"Request {i+1}: Failed in {result['time_taken']:.2f}s - {result['error']}")
    
    return results

def test_queue_sizes(client, model, sizes=[1, 5, 10], requests_per_size=4):
    """Test different queue sizes to observe behavior"""
    print(f"\n=== Testing Different Queue Sizes ===")
    
    # Check if client has queue size attribute
    if not hasattr(client, "queue_size"):
        print("Warning: Client does not support queue_size attribute")
        return []
    
    all_results = []
    original_size = client.queue_size
    
    # Test each queue size
    for size in sizes:
        print(f"\nTesting queue size: {size}")
        client.queue_size = size
        
        # Generate test prompts
        prompts = [
            f"Count from {i+1} to {i+5} in order. Be brief." 
            for i in range(requests_per_size)
        ]
        
        # Send requests concurrently
        size_results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=requests_per_size) as executor:
            future_to_idx = {
                executor.submit(send_request, client, prompt, model, f"size{size}_{i+1}"): i
                for i, prompt in enumerate(prompts)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    size_results.append(result)
                    status = "✓" if result["success"] else "✗"
                    print(f"Request {idx+1}/{requests_per_size}: {status} in {result['time_taken']:.2f}s")
                except Exception as e:
                    print(f"Request {idx+1}/{requests_per_size} exception: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Summarize this size test
        successful = sum(1 for r in size_results if r["success"])
        print(f"Queue size {size}: {successful}/{requests_per_size} successful in {total_time:.2f}s")
        print(f"Average time per request: {total_time/requests_per_size:.2f}s")
        
        # Add to overall results
        all_results.extend(size_results)
    
    # Reset queue size
    client.queue_size = original_size
    return all_results

def test_with_without_queue(client, model, num_requests=5):
    """Compare performance with and without queue enabled"""
    print(f"\n=== Comparing Performance With and Without Queue ===")
    
    # Check if client has queue_enabled attribute
    if not hasattr(client, "queue_enabled"):
        print("Warning: Client does not support queue_enabled attribute")
        return {}
    
    results = {"with_queue": [], "without_queue": []}
    
    # Generate test prompts
    prompts = [
        f"What day comes after {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][i % 7]}?"
        for i in range(num_requests)
    ]
    
    # Test with queue
    print("\nTesting WITH queue enabled:")
    client.queue_enabled = True
    client.max_concurrent_requests = 2  # Limit concurrency to test queue
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        future_to_idx = {
            executor.submit(send_request, client, prompt, model, f"with_{i+1}"): i
            for i, prompt in enumerate(prompts)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results["with_queue"].append(result)
                status = "✓" if result["success"] else "✗"
                print(f"Request {idx+1}: {status} in {result['time_taken']:.2f}s")
            except Exception as e:
                print(f"Request {idx+1} exception: {str(e)}")
    
    with_queue_time = time.time() - start_time
    with_queue_successes = sum(1 for r in results["with_queue"] if r["success"])
    print(f"WITH queue: {with_queue_successes}/{num_requests} successful in {with_queue_time:.2f}s")
    
    # Wait a bit before the next test
    time.sleep(2)
    
    # Test without queue
    print("\nTesting WITHOUT queue enabled:")
    client.queue_enabled = False
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        future_to_idx = {
            executor.submit(send_request, client, prompt, model, f"without_{i+1}"): i
            for i, prompt in enumerate(prompts)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results["without_queue"].append(result)
                status = "✓" if result["success"] else "✗"
                print(f"Request {idx+1}: {status} in {result['time_taken']:.2f}s")
            except Exception as e:
                print(f"Request {idx+1} exception: {str(e)}")
    
    without_queue_time = time.time() - start_time
    without_queue_successes = sum(1 for r in results["without_queue"] if r["success"])
    print(f"WITHOUT queue: {without_queue_successes}/{num_requests} successful in {without_queue_time:.2f}s")
    
    # Compare results
    print("\nComparison:")
    print(f"WITH queue: {with_queue_time:.2f}s total, {with_queue_time/num_requests:.2f}s per request")
    print(f"WITHOUT queue: {without_queue_time:.2f}s total, {without_queue_time/num_requests:.2f}s per request")
    
    if with_queue_time < without_queue_time:
        print(f"Queue is {(without_queue_time/with_queue_time - 1)*100:.1f}% faster")
    else:
        print(f"Direct requests are {(with_queue_time/without_queue_time - 1)*100:.1f}% faster")
    
    # Reset queue setting
    client.queue_enabled = True
    return results

def run_all_tests(client, model, args):
    """Run all test suites and collect results"""
    print(f"\n=== Starting Comprehensive Ollama API Queue and Backoff Tests ===")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Model: {model}")
    print(f"Configuration: Queue size: {client.queue_size if hasattr(client, 'queue_size') else 'unknown'}, " +
          f"Max concurrent: {client.max_concurrent_requests if hasattr(client, 'max_concurrent_requests') else 'unknown'}")
    
    results = {}
    
    # Run test suites
    print("\nRunning test suite 1/4: Basic Queue...")
    results["basic_queue"] = test_basic_queue(client, model, args.num_requests, args.max_concurrent)
    
    print("\nRunning test suite 2/4: Backoff Recovery...")
    results["backoff_recovery"] = test_backoff_recovery(client, model, 3)
    
    print("\nRunning test suite 3/4: Queue Sizes...")
    results["queue_sizes"] = test_queue_sizes(client, model, [1, 3, 5], 3)
    
    print("\nRunning test suite 4/4: With/Without Queue...")
    results["with_without_queue"] = test_with_without_queue(client, model, 4)
    
    # Generate summary
    all_requests = (
        results["basic_queue"] + 
        results["backoff_recovery"] + 
        results["queue_sizes"] + 
        results["with_without_queue"]["with_queue"] + 
        results["with_without_queue"]["without_queue"]
    )
    
    successful_requests = sum(1 for r in all_requests if r["success"])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "model": model,
        "total_requests": len(all_requests),
        "successful_requests": successful_requests,
        "success_rate": successful_requests / len(all_requests) if all_requests else 0,
        "results": results
    }
    
    filename = f"ollama_backoff_queue_comprehensive_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    print(f"\n=== Test Complete: {successful_requests}/{len(all_requests)} successful requests ({successful_requests/len(all_requests)*100:.1f}%) ===")
    
    return output

def main():
    parser = argparse.ArgumentParser(description="Comprehensive test for Ollama API queue and backoff")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama API host (default: {DEFAULT_HOST})")
    parser.add_argument("--num_requests", "-n", type=int, default=NUM_REQUESTS, 
                       help=f"Number of requests to send (default: {NUM_REQUESTS})")
    parser.add_argument("--max_concurrent", "-c", type=int, default=MAX_CONCURRENT,
                       help=f"Maximum concurrent requests (default: {MAX_CONCURRENT})")
    parser.add_argument("--queue_size", "-q", type=int, default=QUEUE_SIZE,
                       help=f"Queue size to use (default: {QUEUE_SIZE})")
    
    args = parser.parse_args()
    
    # Initialize client
    metadata = {"ollama_host": args.host}
    client = ollama(resources={}, metadata=metadata)
    
    # Configure client
    if hasattr(client, "queue_size"):
        client.queue_size = args.queue_size
    if hasattr(client, "max_concurrent_requests"):
        client.max_concurrent_requests = args.max_concurrent
    
    # Run all tests
    run_all_tests(client, args.model, args)

if __name__ == "__main__":
    main()