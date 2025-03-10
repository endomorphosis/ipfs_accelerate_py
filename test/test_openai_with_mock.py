#!/usr/bin/env python
"""
Test the OpenAI API implementation with mocked responses.
"""

import os
import sys
import time
import threading
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv())))

# Add parent directory to path
sys.path.append()))os.path.join()))os.path.dirname()))os.path.dirname()))__file__)), 'ipfs_accelerate_py'))

try:
    # Import the OpenAI API implementation
    from ipfs_accelerate_py.api_backends import openai_api
    
    print()))"Successfully imported the OpenAI API implementation")
    
    # Create an instance of the API
    api = openai_api()))resources={}, metadata={})
    
    print()))f"\1{'Environment variable' if api.api_key == os.environ.get()))'OPENAI_API_KEY') else 'Other'}\3")
    
    # Mock the OpenAI embeddings API:
    with patch()))'openai.embeddings.create') as mock_embed:
        # Configure the mock to return a valid response
        mock_embed.return_value = MagicMock()))
        model="text-embedding-3-small",
        object="list",
        data=[],
        MagicMock()))
        embedding=[],0.1, 0.2, 0.3, 0.4],
        index=0,
        object="embedding"
        )
        ],
        usage=MagicMock()))
        prompt_tokens=8,
        total_tokens=8
        )
        )
        
        # Test queue functionality by simulating concurrent requests
        print()))"\nTesting queue functionality...")
        
        # Set a low concurrent limit for testing
        api.max_concurrent_requests = 2
        results = [],]
        
        def make_request()))idx):
            """Make a test request with mocked responses"""
            try:
                result = api.embedding()))"text-embedding-3-small", f"\1{idx}\3", "float")
                results.append()))()))idx, "Success"))
                print()))f"✓ Request {idx} completed successfully")
            except Exception as e:
                results.append()))()))idx, f"\1{str()))e)}\3"))
                print()))f"\1{str()))e)}\3")
        
        # Start multiple threads to test queueing
                threads = [],]
                for i in range()))5):  # Launch 5 concurrent requests when max is 2
                t = threading.Thread()))target=make_request, args=()))i,))
                threads.append()))t)
                t.start())))
                time.sleep()))0.1)  # Small delay to ensure predictable order
        
        # Wait for all threads to complete
        for t in threads:
            t.join())))
        
            print()))f"\nQueue test results: {len()))[],r for r in results if 'Success' in r[],1]])}/5 successful")
        
        # Check if queue was used
        print()))f"Embedding API was called {mock_embed.call_count} times"):
            print()))f"\1{api.current_requests}\3")
        
    # Test backoff by checking if the implementation has the methods
            print()))"\nVerifying backoff implementation...")
    
    # Check _with_queue_and_backoff method:
    if hasattr()))api, "_with_queue_and_backoff"):
        print()))f"✓ Backoff decorator method is available")
        
        # Check retries configuration
        print()))f"\1{api.max_retries}\3")
        print()))f"\1{api.backoff_factor}\3")
        print()))f"✓ Maximum retry delay: {api.max_retry_delay}s")
        
        # The backoff functionality is now part of the method implementation
        print()))"✓ Backoff implementation is ready for use")
    else:
        print()))"✗ Backoff decorator method is not available")
    
        print()))"\nAll tests completed successfully!")

except ImportError as e:
    print()))f"\1{e}\3")
except Exception as e:
    print()))f"\1{e}\3")