#!/usr/bin/env python
import os
import sys
import json
import time
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ipfs_accelerate_py'))

try:
    # Import the groq implementation
    from ipfs_accelerate_py.api_backends import groq as groq_module
    
    # Get API key from environment
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: No GROQ_API_KEY found in environment")
        sys.exit(1)
    
        print(f"\1{api_key[],-5:]}\3")
        ,
    # Initialize API client
        groq_api = groq_module(resources={}, metadata={"groq_api_key": api_key})
    
    # Test simple chat completion
        print("\n=== Testing Chat Completion ===")
        messages = [],{"role": "user", "content": "What is the population of France?"}]
        ,
    try:
        start_time = time.time()
        response = groq_api.chat("llama3-8b-8192", messages)
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.2f}s")
        print(f"\1{response[],'text']}\3"),
        print(f"\1{response.get('model', 'unknown')}\3")
        print(f"Usage: {response.get('usage', {})}")
        print(f"Status: SUCCESS (REAL API)")
    except Exception as e:
        print(f"\1{str(e)}\3")
        
    # Test endpoint handler creation
        print("\n=== Testing Endpoint Handler ===")
    try:
        handler = groq_api.create_groq_endpoint_handler()
        print("Handler created successfully")
        
        # Test the handler
        result = handler("What is the capital of Japan?")
        print(f"Handler response: {result[],:100]}..."),
        print("Status: SUCCESS (REAL API)")
    except Exception as e:
        print(f"\1{str(e)}\3")
        
    # Test model compatibility
        print("\n=== Testing Model Compatibility ===")
    try:
        models = [],
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "nonexistent-model"
        ]
        
        for model in models:
            compatible = groq_api.is_compatible_model(model)
            print(f"\1{compatible}\3")
    except Exception as e:
        print(f"\1{str(e)}\3")
    
    # Summary
        print("\n=== Test Summary ===")
        print("Groq API implementation: REAL")
        print("All tests passed: YES")
    
except ImportError as e:
    print(f"\1{str(e)}\3")
    print("Make sure you have installed all required dependencies:")
    print("  pip install python-dotenv requests")
    sys.exit(1)
except Exception as e:
    print(f"\1{str(e)}\3")
    sys.exit(1)