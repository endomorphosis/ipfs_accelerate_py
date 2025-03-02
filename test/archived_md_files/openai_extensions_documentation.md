# OpenAI API Extensions Documentation

This document provides comprehensive information about the OpenAI API extensions implemented in the IPFS Accelerate Python framework. These extensions enhance the base OpenAI API implementation with support for Assistants API, Function Calling, and Fine-tuning capabilities.

## Overview

The OpenAI API extensions consist of three main components:

1. **Assistants API** - For creating and managing assistants, threads, and conversations
2. **Function Calling** - For integrating custom functions, parallel execution, and tool use
3. **Fine-tuning API** - For creating and managing fine-tuned models

Each extension is implemented as a separate class that can be used independently or integrated with the base OpenAI API implementation.

## 1. Assistants API

The Assistants API extension provides a complete implementation of OpenAI's Assistants API, allowing you to create and manage AI assistants that can use tools and maintain conversational context.

### Key Features

- **Assistant Management**: Create, list, retrieve, update, and delete assistants
- **Thread Management**: Create, retrieve, and delete conversation threads
- **Message Management**: Create, list, and retrieve messages in threads
- **Run Management**: Create, monitor, and control assistant runs
- **Function Calling**: Support for tool outputs and function call handling
- **File Management**: Upload, list, retrieve, and delete files for assistants
- **Helper Methods**: Simplified conversation interfaces and run monitoring

### Usage Example

```python
from implement_openai_assistants_api import OpenAIAssistantsAPI

# Initialize the API
assistants_api = OpenAIAssistantsAPI()

# Create an assistant
assistant = assistants_api.create_assistant(
    model="gpt-4o",
    name="Research Assistant",
    description="A helpful assistant for research tasks",
    instructions="You are a research assistant. Help users find and analyze information."
)

# Create a thread and have a conversation
conversation = assistants_api.simple_conversation(
    assistant_id=assistant["assistant_id"],
    messages=[
        "Hello! I'm researching climate change. Can you help me understand its impact on agriculture?",
        "What are the main crops affected by rising temperatures?"
    ]
)

# Access the messages from the conversation
for message in conversation["messages"]:
    print(f"{message.role.upper()}: {message.content[0].text.value}")
```

## 2. Function Calling

The Function Calling extension allows you to register Python functions and have them called by OpenAI models, with support for parallel function calling and advanced tool use.

### Key Features

- **Function Registration**: Register Python functions with automatic schema generation
- **Function Execution**: Execute function calls made by the model
- **Parallel Function Calling**: Call multiple functions concurrently for efficiency
- **Tool Integration**: Support for code interpreter, retrieval, and file search tools
- **Conversation Management**: Complete back-and-forth conversations with function calls
- **Automatic Parameter Extraction**: Parse function parameters from docstrings

### Usage Example

```python
from implement_openai_function_calling import OpenAIFunctionCalling

# Initialize the API
function_api = OpenAIFunctionCalling()

# Define and register a function
def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and state, e.g. "San Francisco, CA"
        unit: The unit of temperature, either "celsius" or "fahrenheit"
        
    Returns:
        A string with the weather information
    """
    import random
    temp = random.randint(0, 30) if unit == "celsius" else random.randint(32, 86)
    conditions = random.choice(["sunny", "cloudy", "rainy", "snowy"])
    return f"The current weather in {location} is {temp}°{unit[0].upper()} and {conditions}."

function_api.register_function(get_weather)

# Use the function in a conversation
response = function_api.complete_function_conversation(
    messages=[
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather like in San Francisco and Tokyo?"}
    ]
)

# Print the conversation with function calls
for message in response["conversation"]:
    if message["role"] == "tool":
        print(f"FUNCTION ({message['name']}): {message['content']}")
    else:
        print(f"{message['role'].upper()}: {message.get('content', '')}")
```

## 3. Fine-tuning API

The Fine-tuning API extension provides a complete implementation of OpenAI's Fine-tuning API, allowing you to create and manage fine-tuned models.

### Key Features

- **File Management**: Prepare, validate, upload, and manage training files
- **Job Management**: Create, list, retrieve, and cancel fine-tuning jobs
- **Model Management**: List and delete fine-tuned models
- **Validation**: Validate training files for common errors
- **Workflow Automation**: Complete fine-tuning workflow from data to model
- **Job Monitoring**: Monitor fine-tuning job progress and metrics

### Usage Example

```python
from implement_openai_fine_tuning import OpenAIFineTuning

# Initialize the API
fine_tuning = OpenAIFineTuning()

# Prepare training data
training_data = [
    [
        {"role": "system", "content": "You are a helpful assistant that provides information about astronomy."},
        {"role": "user", "content": "What is a black hole?"},
        {"role": "assistant", "content": "A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves, can escape from it."}
    ],
    # ... more training examples ...
]

# Create a training file
result = fine_tuning.prepare_jsonl_from_messages(training_data, "training_data.jsonl")

# Execute the complete fine-tuning workflow
workflow = fine_tuning.complete_fine_tuning_workflow(
    training_data_path="training_data.jsonl",
    model="gpt-3.5-turbo",
    suffix="astronomy-expert",
    wait_for_completion=True
)

# Use the fine-tuned model
if workflow["success"]:
    model_id = workflow["fine_tuned_model"]
    response = fine_tuning.use_fine_tuned_model(
        model=model_id,
        messages=[
            {"role": "user", "content": "Tell me about neutron stars."}
        ]
    )
    print(response["content"])
```

## Integration with Base OpenAI API

All three extensions are designed to work seamlessly with the base OpenAI API implementation in the IPFS Accelerate Python framework. Each extension takes the same `resources` and `metadata` parameters as the base implementation and can delegate to it for shared functionality.

Example of integration:

```python
from api_backends import openai_api
from implement_openai_assistants_api import OpenAIAssistantsAPI
from implement_openai_function_calling import OpenAIFunctionCalling
from implement_openai_fine_tuning import OpenAIFineTuning

# Initialize the base API
resources = {}
metadata = {"openai_api_key": "your-api-key"}
api = openai_api(resources=resources, metadata=metadata)

# Initialize the extensions with the same resources and metadata
assistants_api = OpenAIAssistantsAPI(resources=resources, metadata=metadata)
function_api = OpenAIFunctionCalling(resources=resources, metadata=metadata)
fine_tuning = OpenAIFineTuning(resources=resources, metadata=metadata)

# Now you can use both the base API and extensions with shared configuration
```

## Error Handling

All extensions implement consistent error handling that follows the same pattern:

1. Methods return a dictionary with a `success` boolean key
2. Successful operations include the relevant data in the response
3. Failed operations include an `error` key with details
4. All API calls are wrapped in try/except blocks to prevent crashes

Example error handling:

```python
result = assistants_api.create_assistant(model="gpt-4o", name="Test Assistant")

if result["success"]:
    assistant_id = result["assistant_id"]
    print(f"Created assistant: {assistant_id}")
else:
    print(f"Error creating assistant: {result['error']}")
```

## Mock Implementation

All three extensions include support for mock implementations that allow testing without actual API calls. The mock mode is automatically used when no API key is provided or when explicitly requested.

To run tests using mocks:

```python
from unittest.mock import patch, MagicMock
from implement_openai_assistants_api import OpenAIAssistantsAPI

# Test with mocks
with patch('openai.beta.assistants.create') as mock_create:
    mock_assistant = MagicMock()
    mock_assistant.id = "asst_test123"
    mock_create.return_value = mock_assistant
    
    assistants_api = OpenAIAssistantsAPI()
    result = assistants_api.create_assistant(model="gpt-4o", name="Test Assistant")
    
    assert result["success"] == True
    assert result["assistant_id"] == "asst_test123"
```

## Advanced Usage

### Assistants with Function Calling

Combine the Assistants API and Function Calling extensions for powerful applications:

```python
# Register functions
function_api.register_function(get_weather)
function_api.register_function(search_database)

# Get function definitions
functions = function_api.get_registered_function_definitions()

# Create an assistant with functions as tools
tools = [{"type": "function", "function": func} for func in functions]
assistant = assistants_api.create_assistant(model="gpt-4o", tools=tools)

# Start a conversation
thread = assistants_api.create_thread()
assistants_api.create_message(thread["thread_id"], "user", "What's the weather in NYC?")
run = assistants_api.create_run(thread["thread_id"], assistant["assistant_id"])

# Wait for completion and handle function calls
run_status = assistants_api.wait_for_run_completion(thread["thread_id"], run["run_id"])
if run_status["status"] == "requires_action":
    # Execute the function calls
    tool_outputs = []
    for tool_call in run_status["tool_calls"]:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Execute the appropriate function
        if function_name == "get_weather":
            result = get_weather(**arguments)
        elif function_name == "search_database":
            result = search_database(**arguments)
            
        tool_outputs.append({
            "tool_call_id": tool_call.id,
            "output": result
        })
    
    # Submit the results back to the assistant
    assistants_api.submit_tool_outputs(thread["thread_id"], run["run_id"], tool_outputs)
```

## Further Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Assistants API Guide](https://platform.openai.com/docs/assistants/overview)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)

## Troubleshooting

Common issues and their solutions:

1. **Authentication Errors**:
   - Ensure your API key is set in the environment variables or passed directly
   - Check if your API key has the necessary permissions for the features you're using

2. **Rate Limit Errors**:
   - The extensions include automatic retry with exponential backoff
   - Consider implementing your own rate limiting for high-volume applications

3. **Model Compatibility**:
   - Not all models support all features (e.g., function calling, vision)
   - Check the OpenAI documentation for feature compatibility by model

4. **Large Requests**:
   - Be mindful of token limits for different models
   - Function calling and assistants may require additional tokens

For additional support, please open an issue in the GitHub repository.