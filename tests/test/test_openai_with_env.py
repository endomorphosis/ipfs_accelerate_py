#!/usr/bin/env python
"""
Test the OpenAI API implementation with environment variables.
This will perform a real API call if an API key is provided in the .env file.
"""

import os
import sys
import time
from dotenv import load_dotenv
import threading

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ipfs_accelerate_py'))
:
try:
    # Import the OpenAI API implementation
    from ipfs_accelerate_py.api_backends import openai_api
    
    print("Successfully imported the OpenAI API implementation")
    
    # Create an instance of the API
    api = openai_api(resources={}, metadata={})
    
    # Check if API key is available:
    if not api.api_key:
        print("No API key found in environment variables. Please add OPENAI_API_KEY to your .env file.")
        sys.exit(1)
    
        print(f"\1{api.api_key[-4:] if len(api.api_key) > 8 else ''}\3")
        ,
    # Test queue functionality by simulating concurrent requests
        print("\nTesting queue functionality...")
    
    # Set a low concurrent limit for testing
        api.max_concurrent_requests = 2
        results = [],,
    :
    def make_request(idx):
        """Make a test request to the API"""
        try:
            # Simply use the embedding function as it's usually cheaper than completions
            result = api.embedding("text-embedding-3-small", f"\1{idx}\3", "float")
            results.append((idx, "Success"))
            print(f"✓ Request {idx} completed successfully")
        except Exception as e:
            results.append((idx, f"\1{str(e)}\3"))
            print(f"\1{str(e)}\3")
    
    # Start multiple threads to test queueing
            threads = [],,
            for i in range(5):  # Launch 5 concurrent requests when max is 2
            t = threading.Thread(target=make_request, args=(i,))
            threads.append(t)
            t.start()
            time.sleep(0.1)  # Small delay to ensure predictable order
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
        print(f"\nQueue test results: {len([r for r in results if 'Success' in r[1]])}/5 successful")
        ,
    # Test backoff functionality (this is harder to test without mocking)
        print("\nBackoff functionality would be triggered automatically if rate limits are hit")
    
        print("\nAll tests completed!")
:
except ImportError as e:
    print(f"\1{e}\3")
except Exception as e:
    print(f"\1{e}\3")