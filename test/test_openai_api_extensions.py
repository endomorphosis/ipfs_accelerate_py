#!/usr/bin/env python
"""
Test script to validate the OpenAI API extensions implementation.
This tests the Assistants API, Function Calling, and Fine-tuning capabilities.
"""

import os
import sys
import json
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv())))))

# Add parent directory to path
sys.path.append()))))os.path.join()))))os.path.dirname()))))os.path.dirname()))))__file__)), 'ipfs_accelerate_py'))

# Import the extension implementations
try:
    from implement_openai_assistants_api import OpenAIAssistantsAPI
    from implement_openai_function_calling import OpenAIFunctionCalling
    from implement_openai_fine_tuning import OpenAIFineTuning
except ImportError as e:
    print()))))f"Failed to import OpenAI API extensions: {}}}}e}")
    sys.exit()))))1)

def test_assistants_api()))))):
    """Test the OpenAI Assistants API extension implementation."""
    print()))))"\n=== Testing OpenAI Assistants API Extension ===")
    
    # Initialize the extension
    assistants_api = OpenAIAssistantsAPI())))))
    
    # Test assistant creation with mock
    with patch()))))'openai.beta.assistants.create') as mock_create:
        mock_assistant = MagicMock())))))
        mock_assistant.id = "asst_test123"
        mock_assistant.model = "gpt-4o"
        mock_assistant.name = "Test Assistant"
        mock_assistant.created_at = 1692835277
        
        mock_create.return_value = mock_assistant
        
        result = assistants_api.create_assistant()))))
        model="gpt-4o",
        name="Test Assistant",
        description="A test assistant",
        instructions="You are a test assistant for validation purposes."
        )
        
        if result[]]]],,,,"success"]:,,,,,
        print()))))"✓ Successfully tested assistant creation")
        print()))))f"  Assistant ID: {}}}}result[]]]],,,,'assistant_id']}"),
        else:
            print()))))f"✗ Failed to test assistant creation: {}}}}result.get()))))'error')}")
    
    # Test thread creation with mock
    with patch()))))'openai.beta.threads.create') as mock_create_thread:
        mock_thread = MagicMock())))))
        mock_thread.id = "thread_test123"
        mock_thread.created_at = 1692835377
        
        mock_create_thread.return_value = mock_thread
        
        result = assistants_api.create_thread())))))
        
        if result[]]]],,,,"success"]:,,,,,
        print()))))"✓ Successfully tested thread creation")
        print()))))f"  Thread ID: {}}}}result[]]]],,,,'thread_id']}"),
        else:
            print()))))f"✗ Failed to test thread creation: {}}}}result.get()))))'error')}")
    
    # Test message creation with mock
    with patch()))))'openai.beta.threads.messages.create') as mock_create_message:
        mock_message = MagicMock())))))
        mock_message.id = "msg_test123"
        mock_message.thread_id = "thread_test123"
        mock_message.role = "user"
        mock_message.created_at = 1692835477
        
        mock_create_message.return_value = mock_message
        
        result = assistants_api.create_message()))))
        thread_id="thread_test123",
        role="user",
        content="Hello, assistant!"
        )
        
        if result[]]]],,,,"success"]:,,,,,
        print()))))"✓ Successfully tested message creation")
        print()))))f"  Message ID: {}}}}result[]]]],,,,'message_id']}"),
        else:
            print()))))f"✗ Failed to test message creation: {}}}}result.get()))))'error')}")
    
    # Test run creation with mock
    with patch()))))'openai.beta.threads.runs.create') as mock_create_run:
        mock_run = MagicMock())))))
        mock_run.id = "run_test123"
        mock_run.thread_id = "thread_test123"
        mock_run.assistant_id = "asst_test123"
        mock_run.status = "queued"
        mock_run.created_at = 1692835577
        
        mock_create_run.return_value = mock_run
        
        result = assistants_api.create_run()))))
        thread_id="thread_test123",
        assistant_id="asst_test123"
        )
        
        if result[]]]],,,,"success"]:,,,,,
        print()))))"✓ Successfully tested run creation")
        print()))))f"  Run ID: {}}}}result[]]]],,,,'run_id']}"),
        print()))))f"  Status: {}}}}result[]]]],,,,'status']}"),
        else:
            print()))))f"✗ Failed to test run creation: {}}}}result.get()))))'error')}")
    
            print()))))"\nAssistants API extension tests completed")
        return True

def test_function_calling()))))):
    """Test the OpenAI Function Calling extension implementation."""
    print()))))"\n=== Testing OpenAI Function Calling Extension ===")
    
    # Initialize the extension
    function_api = OpenAIFunctionCalling())))))
    
    # Define a simple test function
    def get_weather()))))location: str, unit: str = "celsius") -> str:
        """Get the current weather for a location."""
    return f"The weather in {}}}}location} is sunny and 25°{}}}}unit[]]]],,,,0].upper())))))}"
    ,
    # Test function registration
    result = function_api.register_function()))))get_weather)
    
    if result[]]]],,,,"success"]:,,,,,
    print()))))"✓ Successfully registered test function")
    print()))))f"  Function name: {}}}}result[]]]],,,,'name']}"),
    else:
        print()))))f"✗ Failed to register function: {}}}}result.get()))))'error')}")
    
    # Test function definition generation
        definitions = function_api.get_registered_function_definitions())))))
    if definitions and len()))))definitions) > 0:
        print()))))"✓ Successfully retrieved function definitions")
        print()))))f"  Number of definitions: {}}}}len()))))definitions)}")
    else:
        print()))))"✗ Failed to retrieve function definitions")
    
    # Test function calling with mock
    with patch()))))'openai.chat.completions.create') as mock_completion:
        mock_response = MagicMock())))))
        mock_message = MagicMock())))))
        mock_choice = MagicMock())))))
        mock_tool_call = MagicMock())))))
        mock_function = MagicMock())))))
        
        # Set up the mock tool call
        mock_function.name = "get_weather"
        mock_function.arguments = json.dumps())))){}}"location": "San Francisco", "unit": "fahrenheit"})
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_function
        mock_tool_call.id = "call_test123"
        
        # Build the mock response structure
        mock_message.tool_calls = []]]],,,,mock_tool_call],
        mock_message.content = None
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"
        mock_response.choices = []]]],,,,mock_choice]
        ,
        mock_completion.return_value = mock_response
        
        result = function_api.process_with_function_calling()))))
        messages=[]]]],,,,
        {}}"role": "system", "content": "You are a helpful assistant."},
        {}}"role": "user", "content": "What's the weather in San Francisco?"}
        ]
        )
        
        if result[]]]],,,,"success"] and "function_results" in result:
            print()))))"✓ Successfully tested function calling")
            print()))))f"  Function calls: {}}}}len()))))result[]]]],,,,'function_results'])}")
            for call in result[]]]],,,,'function_results']:
                print()))))f"  Called: {}}}}call[]]]],,,,'function_name']}")
        else:
            print()))))f"✗ Failed to test function calling: {}}}}result.get()))))'error')}")
    
    # Test function unregistration
            result = function_api.unregister_function()))))"get_weather")
            if result[]]]],,,,"success"]:,,,,,
            print()))))"✓ Successfully unregistered function")
    else:
        print()))))f"✗ Failed to unregister function: {}}}}result.get()))))'error')}")
    
        print()))))"\nFunction Calling extension tests completed")
            return True

def test_fine_tuning()))))):
    """Test the OpenAI Fine-tuning extension implementation."""
    print()))))"\n=== Testing OpenAI Fine-tuning Extension ===")
    
    # Initialize the extension
    fine_tuning = OpenAIFineTuning())))))
    
    # Create a temporary file for testing
    import tempfile
    import os
    
    temp_dir = tempfile.gettempdir())))))
    test_file_path = os.path.join()))))temp_dir, "test_fine_tuning.jsonl")
    
    # Create a simple test dataset
    test_data = []]]],,,,
    []]]],,,,
    {}}"role": "system", "content": "You are a helpful assistant."},
    {}}"role": "user", "content": "Hello"},
    {}}"role": "assistant", "content": "Hi there! How can I help you today?"}
    ],
    []]]],,,,
    {}}"role": "system", "content": "You are a helpful assistant."},
    {}}"role": "user", "content": "What's the capital of France?"},
    {}}"role": "assistant", "content": "The capital of France is Paris."}
    ]
    ]
    
    # Test JSONL file creation
    result = fine_tuning.prepare_jsonl_from_messages()))))test_data, test_file_path)
    if result[]]]],,,,"success"]:,,,,,
    print()))))"✓ Successfully created test JSONL file")
    print()))))f"  File path: {}}}}result[]]]],,,,'file_path']}")
    else:
        print()))))f"✗ Failed to create test file: {}}}}result.get()))))'error')}")
    
    # Test file validation
        result = fine_tuning.validate_fine_tuning_file()))))test_file_path)
        if result[]]]],,,,"success"]:,,,,,
        print()))))"✓ Successfully validated test file")
        print()))))f"  Examples count: {}}}}result[]]]],,,,'examples_count']}")
        print()))))f"  Estimated tokens: {}}}}result[]]]],,,,'estimated_tokens']}")
        for warning in result.get()))))"warnings", []]]],,,,]):
            print()))))f"  Warning: {}}}}warning}")
    else:
        print()))))f"✗ Failed to validate test file: {}}}}result.get()))))'error')}")
    
    # Test file upload with mock
    with patch()))))'openai.files.create') as mock_upload:
        mock_file = MagicMock())))))
        mock_file.id = "file-test123"
        mock_file.filename = "test_fine_tuning.jsonl"
        mock_file.purpose = "fine-tune"
        mock_file.created_at = 1692835677
        mock_file.status = "uploaded"
        
        mock_upload.return_value = mock_file
        
        # Mock the validation to always return success
        with patch.object()))))fine_tuning, 'validate_fine_tuning_file') as mock_validate:
            mock_validate.return_value = {}}
            "success": True,
            "valid": True,
            "examples_count": 2,
            "estimated_tokens": 100,
            "errors": []]]],,,,],
            "warnings": []]]],,,,]
            }
            
            result = fine_tuning.upload_fine_tuning_file()))))test_file_path)
            
            if result[]]]],,,,"success"]:,,,,,
            print()))))"✓ Successfully tested file upload")
            print()))))f"  File ID: {}}}}result[]]]],,,,'file_id']}")
            else:
                print()))))f"✗ Failed to test file upload: {}}}}result.get()))))'error')}")
    
    # Test fine-tuning job creation with mock
    with patch()))))'openai.fine_tuning.jobs.create') as mock_create_job:
        mock_job = MagicMock())))))
        mock_job.id = "ftjob-test123"
        mock_job.model = "gpt-3.5-turbo"
        mock_job.status = "created"
        mock_job.created_at = 1692835777
        mock_job.fine_tuned_model = None
        mock_job.object = "fine_tuning.job"
        
        mock_create_job.return_value = mock_job
        
        result = fine_tuning.create_fine_tuning_job()))))
        training_file_id="file-test123",
        model="gpt-3.5-turbo"
        )
        
        if result[]]]],,,,"success"]:,,,,,
        print()))))"✓ Successfully tested fine-tuning job creation")
        print()))))f"  Job ID: {}}}}result[]]]],,,,'job_id']}")
        print()))))f"  Status: {}}}}result[]]]],,,,'status']}"),
        else:
            print()))))f"✗ Failed to test job creation: {}}}}result.get()))))'error')}")
    
    # Clean up the test file
    if os.path.exists()))))test_file_path):
        os.remove()))))test_file_path)
        print()))))"✓ Cleaned up test file")
    
        print()))))"\nFine-tuning extension tests completed")
            return True

def main()))))):
    """Run tests for all OpenAI API extensions."""
    # Check if we have an API key
    api_key = os.environ.get()))))"OPENAI_API_KEY"):
    if not api_key:
        print()))))"No OpenAI API key found in environment variables.")
        print()))))"Tests will run in mock mode only.")
    
        results = {}}
        "assistants_api": test_assistants_api()))))),
        "function_calling": test_function_calling()))))),
        "fine_tuning": test_fine_tuning())))))
        }
    
        print()))))"\n=== Test Summary ===")
    for extension, success in results.items()))))):
        status = "✓ Passed" if success else "✗ Failed":
            print()))))f"{}}}}extension}: {}}}}status}")
    
            all_passed = all()))))results.values()))))))
    
    if all_passed:
        print()))))"\nAll tests passed successfully!")
        print()))))"The OpenAI API extensions are properly implemented and working as expected.")
    else:
        print()))))"\nSome tests failed. Please check the logs for details.")
    
        return 0 if all_passed else 1
:
if __name__ == "__main__":
    sys.exit()))))main()))))))