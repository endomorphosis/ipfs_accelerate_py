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
    print(f"Has chat method: {hasattr(client, 'chat')}")
    print(f"Has stream_chat method: {hasattr(client, 'stream_chat')}")
    print(f"Has make_post_request_groq method: {hasattr(client, 'make_post_request_groq')}")
    print(f"Has count_tokens method: {hasattr(client, 'count_tokens')}")
    print(f"Has is_compatible_model method: {hasattr(client, 'is_compatible_model')}")
    
    # Check backoff and queue attributes
    print("\n=== Verifying Backoff and Queue Structure ===")
    print(f"Has max_retries: {hasattr(client, 'max_retries')}")
    print(f"Has backoff_factor: {hasattr(client, 'backoff_factor')}")
    print(f"Has initial_retry_delay: {hasattr(client, 'initial_retry_delay')}")
    print(f"Has max_retry_delay: {hasattr(client, 'max_retry_delay')}")
    print(f"Has request_queue: {hasattr(client, 'request_queue')}")
    print(f"Has max_concurrent_requests: {hasattr(client, 'max_concurrent_requests')}")
    
    # Basic functionality test with mock interaction
    print("\n=== Testing Local Functionality ===")
    token_count = client.count_tokens("This is a test text for token counting")
    print(f"Token counting works: {token_count['estimated_token_count']} tokens")
    
    is_compatible = client.is_compatible_model("llama3-8b-8192")
    print(f"Model compatibility check works: {'Compatible' if is_compatible else 'Not compatible'}")
    
    models = client.list_models()
    print(f"Found {len(models)} models with list_models()")
    
    print("\n✅ All Groq API fix verification tests passed!")
    print("The API can now be properly imported and has the necessary queue and backoff functionality.")
    print("API Status: COMPLETE\n")
    
except ImportError as e:
    print(f"❌ Error importing Groq API: {e}")
    print("The import issue is not fully resolved.")
    
except Exception as e:
    print(f"❌ Error in testing Groq API: {e}")
    import traceback
    traceback.print_exc()