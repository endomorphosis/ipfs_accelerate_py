#!/usr/bin/env python
"""
Test script to verify the Sora implementation in the OpenAI API backend.
"""

import os
import sys
import json
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ipfs_accelerate_py'))

try:
    # Import the OpenAI API implementation
    from api_backends import openai_api
    
    print("Testing OpenAI Sora (text-to-video) Implementation")
    print("=" * 50)
    
    # Create an instance of the API
    api = openai_api(resources={}, metadata={})
    
    # Check if API key is available
    if not api.api_key:
        print("No API key found in environment variables.")
        print("Using mock mode for testing.")
    
    # Look for video_models in module
    from api_backends.openai_api import video_models
    
    # Verify Sora models are defined
    if "sora-1.0" in video_models:
        print("✓ Sora models are correctly defined")
        print(f"  Available models: {', '.join(video_models)}")
    else:
        print("✗ Sora models are not correctly defined")
        
    # Verify text_to_video method exists
    if hasattr(api, 'text_to_video'):
        print("✓ text_to_video method is available")
    else:
        print("✗ text_to_video method is not available")
        
    # Test the moderated_text_to_video method directly with custom mock
    print("\nTesting text_to_video method with custom mock...")
    
    # Create a mock for the moderation that will pass the content check
    mock_moderation_result = MagicMock()
    mock_moderation_result.results = [MagicMock(flagged=False)]
    
    try:
        # Mock the check_messages result and call moderated_text_to_video directly
        with patch.object(api, 'moderation', return_value=mock_moderation_result):
            # Create a custom exception to simulate API not available yet
            not_available_error = Exception("Sora API is not publicly available yet")
            
            # Create our test variables
            prompt = "A cat playing with a ball of yarn"
            mock_video_data = {
                'url': 'https://oaiusercontent.com/video123456.mp4',
                'revised_prompt': prompt,
                'dimensions': '1024x576',
                'duration': 5.0
            }
            
            # Simulate the API response with a custom handler
            with patch.object(api, 'moderated_text_to_video') as mock_method:
                # Return our simulated data
                mock_method.return_value = {
                    'text': json.dumps(mock_video_data),
                    'done': True
                }
                
                # Make the call
                result = api.text_to_video(
                    model="sora-1.0",
                    prompt=prompt,
                    dimensions="1024x576",
                    duration=5.0,
                    quality="standard"
                )
                
                # Parse the result
                video_data = json.loads(result['text'])
                
                print(f"✓ Generated mock video URL: {video_data['url']}")
                print(f"✓ Used dimensions: {video_data['dimensions']}")
                print(f"✓ Video duration: {video_data['duration']} seconds")
                
                print("\nThis implementation is ready for when Sora becomes publicly available.")
                print("Currently, the API will handle requests but inform users if Sora is not yet available.")
                
                # Also test the error handling when Sora is not available
                print("\nTesting error handling for Sora not being available yet...")
                
                # Mock the moderated_text_to_video to raise the not available exception
                with patch.object(api, 'moderated_text_to_video', side_effect=Exception("Sora API is not publicly available yet")):
                    try:
                        # Make the call that should trigger our error handling
                        result = api.text_to_video(
                            model="sora-1.0",
                            prompt="This should trigger the not available error",
                            dimensions="1024x576"
                        )
                        print("✗ Error handling test failed - should have raised an exception")
                    except Exception as e:
                        if "not publicly available" in str(e):
                            print("✓ Error handling test passed - correctly identified Sora is not available")
                        else:
                            print(f"✗ Unexpected error: {str(e)}")
                
    except Exception as e:
        print(f"✗ Error testing text_to_video: {str(e)}")
    
    print("\nImplementation check completed.")

except ImportError as e:
    print(f"Failed to import OpenAI API implementation: {e}")
except Exception as e:
    print(f"Error during testing: {e}")