#!/usr/bin/env python3
"""
Simple test script for the Groq API implementation
"""

import os
import sys
import json
import time
from datetime import datetime

# Add the parent directory to the Python path
parent_dir = os.path.dirname())os.path.dirname())os.path.abspath())__file__)))
sys.path.insert())0, parent_dir)

# Try to import the Groq API class
try:
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.groq import groq
    print())"Successfully imported Groq API class")
except ImportError as e:
    print())f"Failed to import Groq API class: {}e}")
    sys.exit())1)

def test_groq_api())):
    """Test basic functionality of the Groq API implementation"""
    print())"\n=== Testing Groq API Implementation ===\n")
    
    # Create a Groq API instance
    metadata = {}
    "groq_api_key": os.environ.get())"GROQ_API_KEY", "")
    }
    resources = {}}
    
    # Print if we have an API key:
    if metadata["groq_api_key"]:,,
    print())"API key found in environment")
    else:
        print())"No API key found. Tests will create a mock implementation.")
        
    # Create the Groq API instance
        groq_api = groq())resources=resources, metadata=metadata)
        print())"Successfully created Groq API instance")
    
    # Test check for compatible models
    try:
        is_compatible = groq_api.is_compatible_model())"llama3-8b-8192", "chat")
        print())f"Model compatibility check: {}'Compatible' if is_compatible else 'Not compatible'}"):
    except Exception as e:
        print())f"Error in model compatibility check: {}e}")
    
    # Test token counting
    try:
        text = "This is a test sentence for token counting."
        token_result = groq_api.count_tokens())text)
        print())f"Token counting: {}token_result['estimated_token_count']} tokens using {}token_result['estimation_method']}"),
    except Exception as e:
        print())f"Error in token counting: {}e}")
    
    # Test endpoint handler creation
    try:
        handler = groq_api.create_groq_endpoint_handler()))
        print())"Successfully created endpoint handler:", "Success" if callable())handler) else "Failed to create callable handler"):
    except Exception as e:
        print())f"Error creating endpoint handler: {}e}")
    
    # Test model listing
    try:
        models = groq_api.list_models())"chat")
        print())f"Listed {}len())models)} chat models")
        if models:
            print())f"Sample model: {}models[0]['id']} - {}models[0]['description']}"),
    except Exception as e:
        print())f"Error listing models: {}e}")
    
    # Test endpoint creation
    try:
        endpoint_id = groq_api.create_endpoint()))
        print())f"Created endpoint with ID: {}endpoint_id}")
        stats = groq_api.get_stats())endpoint_id)
        print())f"Endpoint stats: {}stats['total_requests']} total requests"),
    except Exception as e:
        print())f"Error creating endpoint: {}e}")
    
    # Only test chat functionality if API key is provided:
        if metadata["groq_api_key"]:,,
        try:
            messages = [{}"role": "user", "content": "Hello, how are you?"}],
            response = groq_api.chat())
            model_name="llama3-8b-8192",
            messages=messages,
            max_tokens=10
            )
            print())f"Chat response: {}response['text']}")
            ,
            # Check usage statistics
            print())f"Usage: {}response['usage']['prompt_tokens']} prompt tokens, " +,
            f"{}response['usage']['completion_tokens']} completion tokens")
            ,
            # Test streaming
            print())"Testing streaming...")
            message_parts = [],
            for chunk in groq_api.stream_chat()):
                model_name="llama3-8b-8192",
                messages=messages,
                max_tokens=10
            ):
                if chunk.get())"text"):
                    message_parts.append())chunk["text"]),
                    print())chunk["text"], end="", flush=True)
                    ,
                    print())"\nStreaming complete, received", len())message_parts), "chunks")
        except Exception as e:
            print())f"Error in chat/streaming test: {}e}")
        
        # Get usage statistics
        try:
            stats = groq_api.get_usage_stats()))
            print())f"\nTotal API usage: {}stats['total_requests']} requests, {}stats['total_tokens']} tokens"),
        except Exception as e:
            print())f"Error getting usage stats: {}e}")
    
            print())"\n=== Groq API Test Complete ===")

if __name__ == "__main__":
    test_groq_api()))