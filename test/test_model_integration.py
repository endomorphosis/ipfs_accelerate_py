#!/usr/bin/env python3
"""
Test model integration with WebNN and WebGPU platforms.

This script demonstrates basic usage of the fixed_web_platform module.

Usage:
    python test_model_integration.py
    """

    import os
    import sys
    import time
    import logging
    from pathlib import Path

# Add the parent directory to the path for importing
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, str(current_dir))

# Import web platform handlers
try:
    from test.web_platform import process_for_web, init_webnn, init_webgpu, create_mock_processors
    WEB_PLATFORM_SUPPORT = True
except ImportError:
    print("WebNN and WebGPU platform support not available")
    WEB_PLATFORM_SUPPORT = False

def test_webnn_integration():
    """Test WebNN integration with a simple class instance."""
    if not WEB_PLATFORM_SUPPORT:
        print("WebNN support not available")
    return False
    
    # Create a simple class to test WebNN integration
    class SimpleModelTest:
        def __init__(self):
            self.model_name = "bert-base-uncased"
            self.mode = "text"
            
        def _create_mock_processor(self):
            """Create a mock processor for testing."""
            return lambda x: {"input_ids": [[101, 102, 103]], "attention_mask": [[1, 1, 1]]}
            ,
    # Create an instance
            model_test = SimpleModelTest()
    
    # Initialize WebNN
            init_result = init_webnn(model_test,
            model_name="bert-base-uncased",
            model_type="text",
            web_api_mode="simulation")
    
    if init_result and "endpoint" in init_result:
        print("WebNN initialization successful!")
        
        # Test the endpoint
        endpoint = init_result["endpoint"],,
        processor = init_result["processor"]
        ,,
        # Process some text
        test_input = "Hello world"
        processed = process_for_web("text", test_input)
        print(f"\1{processed}\3")
        
        # Test the endpoint
        result = endpoint(processed)
        print(f"\1{type(result)}\3")
        if isinstance(result, dict) and "implementation_type" in result:
            print(f"\1{result['implementation_type']}\3")
            ,,
        return True
    else:
        print("WebNN initialization failed")
        return False

def test_webgpu_integration():
    """Test WebGPU integration with a simple class instance."""
    if not WEB_PLATFORM_SUPPORT:
        print("WebGPU support not available")
    return False
    
    # Create a simple class to test WebGPU integration
    class SimpleModelTest:
        def __init__(self):
            self.model_name = "vit-base-patch16-224"
            self.mode = "vision"
            
        def _create_mock_processor(self):
            """Create a mock processor for testing."""
            return lambda x: {"pixel_values": [[[[0.5]]]]}
            ,
    # Create an instance
            model_test = SimpleModelTest()
    
    # Initialize WebGPU
            init_result = init_webgpu(model_test,
            model_name="vit-base-patch16-224",
            model_type="vision",
            web_api_mode="simulation")
    
    if init_result and "endpoint" in init_result:
        print("WebGPU initialization successful!")
        
        # Test the endpoint
        endpoint = init_result["endpoint"],,
        processor = init_result["processor"]
        ,,
        # Process an image
        test_input = "test.jpg"
        processed = process_for_web("vision", test_input)
        print(f"\1{processed}\3")
        
        # Test the endpoint
        result = endpoint(processed)
        print(f"\1{type(result)}\3")
        if isinstance(result, dict) and "implementation_type" in result:
            print(f"\1{result['implementation_type']}\3")
            ,,
        return True
    else:
        print("WebGPU initialization failed")
        return False

def main():
    """Run the integration tests."""
    print("Testing WebNN and WebGPU platform integration")
    
    # Test WebNN integration
    print("\n=== Testing WebNN Integration ===")
    webnn_success = test_webnn_integration()
    
    # Test WebGPU integration
    print("\n=== Testing WebGPU Integration ===")
    webgpu_success = test_webgpu_integration()
    
    # Print summary
    print("\n=== Integration Test Summary ===")
    print(f"\1{'Success' if webnn_success else 'Failed'}\3"):
        print(f"\1{'Success' if webgpu_success else 'Failed'}\3")
    
    # Return success if both tests pass
    return 0 if webnn_success and webgpu_success else 1
:
if __name__ == "__main__":
    sys.exit(main())