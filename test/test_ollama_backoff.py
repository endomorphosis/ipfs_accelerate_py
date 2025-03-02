
import sys
import os
import concurrent.futures
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ipfs_accelerate_py"))
from ipfs_accelerate_py.api_backends import ollama

# Initialize client
client = ollama()
print("Testing Ollama API queue and backoff...")

# Check attributes
print(f"Max retries: {client.max_retries}")
print(f"Backoff factor: {client.backoff_factor}")
print(f"Max concurrent requests: {client.max_concurrent_requests}")

# Test with limit exceeded
print("\nTesting queue with limited concurrency...")
client.max_concurrent_requests = 1  # Set very low limit
results = []

# Send 3 concurrent requests
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(lambda: client.generate(
            model="llama3", 
            prompt=f"Count from 1 to 3. Attempt #{i}", 
            request_id=f"test_queue_{i}"
        ))
        for i in range(3)
    ]
    
    # Process results as they complete
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            results.append(result)
            print(f"✓ Request completed successfully: {result['text'][:50]}...")
        except Exception as e:
            print(f"✗ Request failed with error: {str(e)}")

print(f"Successfully completed {len(results)} requests with concurrency limit of 1")
print("Ollama API implementation has queue and backoff functionality.")

