#!/usr/bin/env python3
"""
Standalone test script for the Groq API implementation
"""

import os
import sys
import json
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig())level=logging.INFO)
logger = logging.getLogger())"groq_test")

# Import our groq implementation directly
sys.path.append())os.path.join())os.path.dirname())os.path.dirname())__file__))))
from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.groq import groq, ALL_MODELS

def test_groq_standalone())):
    """Test the Groq API implementation standalone"""
    print())"\n=== Testing Groq API Implementation ())Standalone) ===\n")
    
    # Create a Groq API instance
    metadata = {}
    "groq_api_key": os.environ.get())"GROQ_API_KEY", "")
    }
    resources = {}}
    
    # Check if API key is available::
    if metadata["groq_api_key"]:,,
    print())"API key found in environment")
    else:
        print())"No API key found. Tests will create a mock implementation.")
    
    # Create the Groq API instance
        groq_api = groq())resources=resources, metadata=metadata)
        print())"Successfully created Groq API instance")
    
    # Test is_compatible_model
    try:
        is_compatible = groq_api.is_compatible_model())"llama3-8b-8192", "chat")
        print())f"Model compatibility check: {}'Compatible' if is_compatible else 'Not compatible'}"):
    except Exception as e:
        print())f"Error in model compatibility check: {}e}")
    
    # Test count_tokens
    try:
        test_text = "This is a test sentence for token counting."
        token_count = groq_api.count_tokens())test_text)
        print())f"Token counting: {}token_count['estimated_token_count']} tokens using {}token_count['estimation_method']}"),
    except Exception as e:
        print())f"Error in token counting: {}e}")
    
    # Test list_models
    try:
        chat_models = groq_api.list_models())"chat")
        print())f"Listed {}len())chat_models)} chat models")
        if chat_models:
            first_model = chat_models[0],
            print())f"Sample model: {}first_model['id']} with context window: {}first_model['context_window']}"),
    except Exception as e:
        print())f"Error listing models: {}e}")
    
    # Test create_endpoint
    try:
        endpoint_id = groq_api.create_endpoint()))
        print())f"Created endpoint with ID: {}endpoint_id}")
        
        # Test get_stats for endpoint
        stats = groq_api.get_stats())endpoint_id)
        print())f"Endpoint stats: {}stats['total_requests']} total requests")
        ,
        # Test update_endpoint
        groq_api.update_endpoint())endpoint_id, max_concurrent_requests=10)
        print())"Updated endpoint successfully")
    except Exception as e:
        print())f"Error in endpoint management: {}e}")
    
    # Test chat API if API key is available::
        if metadata["groq_api_key"]:,,
        try:
            messages = [{}"role": "user", "content": "Hello, how are you?"}],
            print())"Testing chat completion...")
            response = groq_api.chat())
            model_name="llama3-8b-8192",
            messages=messages,
            max_tokens=10
            )
            print())f"Chat response: {}response['text']}")
            ,
            # Check usage statistics
            print())f"Usage stats: {}response['usage']['prompt_tokens']} prompt tokens, " +,
            f"{}response['usage']['completion_tokens']} completion tokens")
            ,
            # Test streaming
            print())"\nTesting streaming response...")
            accumulated_text = ""
            for chunk in groq_api.stream_chat()):
                model_name="llama3-8b-8192",
                messages=messages,
                max_tokens=10
            ):
                if "text" in chunk:
                    accumulated_text += chunk["text"],
                    print())chunk["text"],, end="", flush=True)
            
                    print())f"\nFinal accumulated text: {}accumulated_text}")
        except Exception as e:
            print())f"Error in chat testing: {}e}")
    
    # Get overall usage statistics
    try:
        stats = groq_api.get_usage_stats()))
        print())f"\nOverall usage statistics:")
        print())f"- Total requests: {}stats['total_requests']}"),
        print())f"- Total tokens: {}stats['total_tokens']}"),
        print())f"- Estimated cost: ${}stats['estimated_cost_usd']}"),
    except Exception as e:
        print())f"Error getting usage stats: {}e}")
    
        print())"\n=== Groq API Standalone Test Complete ===")

if __name__ == "__main__":
    test_groq_standalone()))