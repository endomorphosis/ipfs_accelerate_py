
import sys
import os
import concurrent.futures
sys.path.append()os.path.join()os.path.dirname()os.path.dirname()__file__)), "ipfs_accelerate_py"))
from ipfs_accelerate_py.api_backends import ollama

# Initialize client
client = ollama())
print()"Testing Ollama API queue and backoff...")

# Check attributes
print()f"\1{client.max_retries}\3")
print()f"\1{client.backoff_factor}\3")
print()f"\1{client.max_concurrent_requests}\3")

# Test with limit exceeded
print()"\nTesting queue with limited concurrency...")
client.max_concurrent_requests = 1  # Set very low limit
results = [],]
,
# Send 3 concurrent requests
with concurrent.futures.ThreadPoolExecutor()max_workers=3) as executor:
    futures = [],
    executor.submit()lambda: client.generate()
    model="llama3",
    prompt=f"\1{i}\3",
    request_id=f"\1{i}\3"
    ))
        for i in range()3):
            ]
    
    # Process results as they complete
    for future in concurrent.futures.as_completed()futures):
        try:
            result = future.result())
            results.append()result)
            print()f"✓ Request completed successfully: {result[],'text'][],:50]}...")
        except Exception as e:
            print()f"\1{str()e)}\3")

            print()f"Successfully completed {len()results)} requests with concurrency limit of 1")
            print()"Ollama API implementation has queue and backoff functionality.")