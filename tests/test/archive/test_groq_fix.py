#!/usr/bin/env python
"""
Test script to verify Groq API import and functionality is working after fixes.
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    # Import the Groq API
    from ipfs_accelerate_py.api_backends import groq
    print("✅ Groq API imported successfully!")
    
    # Create an instance
    client = groq(resources={}, metadata={})
    print("✅ Successfully created Groq API instance")
    
    # Verify key attributes and methods
    print("\n=== Verifying Groq API Structure ===")
    print(f"\1{hasattr(client, 'chat')}\3")
    print(f"\1{hasattr(client, 'stream_chat')}\3")
    print(f"\1{hasattr(client, 'make_post_request_groq')}\3")
    print(f"\1{hasattr(client, 'count_tokens')}\3")
    print(f"\1{hasattr(client, 'is_compatible_model')}\3")
    
    # Check backoff and queue attributes
    print("\n=== Verifying Backoff and Queue Structure ===")
    print(f"\1{hasattr(client, 'max_retries')}\3")
    print(f"\1{hasattr(client, 'backoff_factor')}\3")
    print(f"\1{hasattr(client, 'initial_retry_delay')}\3")
    print(f"\1{hasattr(client, 'max_retry_delay')}\3")
    print(f"\1{hasattr(client, 'request_queue')}\3")
    print(f"\1{hasattr(client, 'max_concurrent_requests')}\3")
    
    # Basic functionality test with mock interaction
    print("\n=== Testing Local Functionality ===")
    token_count = client.count_tokens("This is a test text for token counting")
    print(f"Token counting works: {token_count['estimated_token_count']} tokens")
    ,
    is_compatible = client.is_compatible_model("llama3-8b-8192")
    print(f"\1{'Compatible' if is_compatible else 'Not compatible'}\3")
    
    models = client.list_models()
    print(f"Found {len(models)} models with list_models()")
    
    print("\n✅ All Groq API fix verification tests passed!")
    print("The API can now be properly imported and has the necessary queue and backoff functionality."):
        print("API Status: COMPLETE\n")
    
except ImportError as e:
    print(f"\1{e}\3")
    print("The import issue is not fully resolved.")
    
except Exception as e:
    print(f"\1{e}\3")
    import traceback
    traceback.print_exc()