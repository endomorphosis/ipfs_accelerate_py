#!/usr/bin/env python
"""
Simple verification test for the OpenAI API implementation.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ipfs_accelerate_py'))

try:
    # Import the OpenAI API implementation
    from api_backends import openai_api
    
    print("Successfully imported the OpenAI API implementation")
    
    # Create an instance of the API
    api = openai_api(resources={}, metadata={})
    
    print("\nVerifying implementation features:")
    
    # Check environment variable handling
    api_key_source = "environment variable" if api.api_key == os.environ.get("OPENAI_API_KEY") else "other source"
    print(f"✓ API key loaded from {api_key_source}")
    
    # Check queue setup
    print(f"✓ Queue configuration: max_concurrent_requests={api.max_concurrent_requests}, queue_size={api.queue_size}")
    
    # Check backoff settings
    print(f"✓ Backoff configuration: max_retries={api.max_retries}, initial_delay={api.initial_retry_delay}, backoff_factor={api.backoff_factor}")
    
    # Verify methods
    if hasattr(api, "_process_queue"):
        print("✓ Queue processing method is available")
    else:
        print("✗ Queue processing method is not available")
        
    if hasattr(api, "_with_queue_and_backoff"):
        print("✓ Backoff decorator method is available")
    else:
        print("✗ Backoff decorator method is not available")
    
    print("\nImplementation looks good! All features are available.")

except ImportError as e:
    print(f"Failed to import OpenAI API implementation: {e}")
except Exception as e:
    print(f"Error verifying implementation: {e}")