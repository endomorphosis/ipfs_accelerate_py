import asyncio
import json
import requests
import os
from typing import Dict, List, Optional, Union, Any, Callable
try:
    import anthropic
except ImportError:
    print("Anthropic package not installed. Install with: pip install anthropic")

class claude:
    """Anthropic Claude API Backend Integration
    
    This class provides a comprehensive interface for interacting with Anthropic's Claude models.
    It supports multiple types of interactions including:
    - Text completion generation
    - Chat completions
    - System prompts
    - Tool/function calling
    - Model management
    
    Features:
    - Dynamic endpoint management
    - Multiple response types (completion, chat)
    - Async and sync request options
    - Request queueing and batching
    - Built-in retry logic and error handling
    
    The handler methods follow these patterns:
    - completion: Takes a prompt, returns generated text
    - chat: Takes message list, returns assistant response
    - tool_use: Takes messages and tool definitions, returns responses with tool calls
    """

    def __init__(self, resources=None, metadata=None):
        """Initialize Claude backend interface
        
        Args:
            resources: Resources configuration dictionary
            metadata: Additional metadata dictionary
        """
        self.resources = resources
        self.metadata = metadata
        # Register method references
        self.create_remote_claude_endpoint_handler = self.create_remote_claude_endpoint_handler
        self.create_remote_claude_chat_endpoint_handler = self.create_remote_claude_chat_endpoint_handler
        self.request_claude_endpoint = self.request_claude_endpoint
        self.test_claude_endpoint = self.test_claude_endpoint
        self.init = self.init
        self.__test__ = self.__test__
        # Add endpoints tracking
        self.endpoints = {}
        self.endpoint_status = {}
        self.registered_models = {}
        # Add queue for managing requests
        self.request_queue = asyncio.Queue(64)
        return None

    def init(self, api_key=None, model_name=None, endpoint_type="completion"):
        """Initialize connection to Anthropic's Claude API
        
        Supported endpoint_types:
        - "completion": Standard text completion
        - "chat": Structured chat completion with system prompts
        
        Supported models:
        - "claude-3-opus"
        - "claude-3-sonnet" 
        - "claude-3-haiku"
        - "claude-2.1"
        - "claude-2.0"
        - "claude-instant-1.2"
        
        Args:
            api_key: Anthropic API key for authentication
            model_name: Name of the Claude model to use
            endpoint_type: Type of endpoint to initialize
            
        Returns:
            tuple: (None, api_key, handler, queue, batch_size)
        """
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError("Anthropic API key is required")

        # Initialize Anthropic client
        self.client = anthropic.Client(api_key=api_key)

        # Create appropriate handler based on endpoint type
        if endpoint_type == "chat":
            endpoint_handler = self.create_remote_claude_chat_endpoint_handler(api_key, model_name)
        else:
            endpoint_handler = self.create_remote_claude_endpoint_handler(api_key, model_name)

        # Register the model
        if model_name not in self.registered_models:
            self.registered_models[model_name] = {
                "types": [endpoint_type]
            }
        elif endpoint_type not in self.registered_models[model_name]["types"]:
            self.registered_models[model_name]["types"].append(endpoint_type)
            
        return None, api_key, endpoint_handler, self.request_queue, 32  # Default batch size

    def create_remote_claude_endpoint_handler(self, api_key=None, model_name=None):
        """Create a handler for text completion with Claude
        
        Example:
            ```python
            handler = claude.create_remote_claude_endpoint_handler(
                api_key="your-api-key",
                model_name="claude-3-opus"
            )
            
            # Basic usage
            response = handler("Explain quantum computing")
            
            # With parameters
            response = handler(
                "Explain quantum computing",
                parameters={
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "top_p": 0.9,
                    "system": "You are a helpful science teacher"
                }
            )
            ```
        """
        def handler(prompt, parameters=None, api_key=api_key, model_name=model_name):
            try:
                msg_params = {
                    "model": model_name,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                }
                
                if parameters:
                    msg_params.update({
                        "temperature": parameters.get("temperature", 0.7),
                        "max_tokens": parameters.get("max_tokens", 1024),
                        "top_p": parameters.get("top_p", 1.0),
                        "top_k": parameters.get("top_k", None),
                    })
                    
                    # Add system prompt if provided
                    if "system" in parameters:
                        msg_params["system"] = parameters["system"]

                message = self.client.messages.create(
                    messages=[{"role": "user", "content": prompt}],
                    **msg_params
                )
                
                return message.content[0].text

            except Exception as e:
                print(f"Error in Claude completion handler: {e}")
                return None
        
        return handler

    def create_remote_claude_chat_endpoint_handler(self, api_key=None, model_name=None):
        """Create a handler for chat completion with Claude
        
        Example:
            ```python
            handler = claude.create_remote_claude_chat_endpoint_handler(
                api_key="your-api-key",
                model_name="claude-3-opus"
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is fusion energy?"},
                {"role": "assistant", "content": "Fusion energy is..."},
                {"role": "user", "content": "What are its advantages?"}
            ]
            
            response = handler(messages)
            
            # With parameters
            response = handler(
                messages,
                parameters={
                    "temperature": 0.8,
                    "max_tokens": 2000
                }
            )
            ```
        """
        def handler(messages, parameters=None, api_key=api_key, model_name=model_name):
            try:
                msg_params = {
                    "model": model_name,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                }
                
                if parameters:
                    msg_params.update({
                        "temperature": parameters.get("temperature", 0.7),
                        "max_tokens": parameters.get("max_tokens", 1024),
                        "top_p": parameters.get("top_p", 1.0),
                        "top_k": parameters.get("top_k", None),
                    })

                # Extract system message if present
                system_msg = None
                chat_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_msg = msg["content"]
                    else:
                        chat_messages.append(msg)

                if system_msg:
                    msg_params["system"] = system_msg

                message = self.client.messages.create(
                    messages=chat_messages,
                    **msg_params
                )
                
                return message.content[0].text

            except Exception as e:
                print(f"Error in Claude chat handler: {e}")
                return None
        
        return handler

    def __test__(self, api_key, endpoint_handler, endpoint_label, endpoint_type="completion"):
        """Test the Claude endpoint
        
        Args:
            api_key: API key for authentication
            endpoint_handler: The handler function
            endpoint_label: Label for the endpoint
            endpoint_type: Type of endpoint to test
            
        Returns:
            bool: True if test passes, False otherwise
        """
        if endpoint_type == "chat":
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"}
            ]
            try:
                result = endpoint_handler(test_messages)
                if result:
                    print(f"Claude chat test passed for {endpoint_label}")
                    return True
                print(f"Claude chat test failed for {endpoint_label}: No result")
                return False
            except Exception as e:
                print(f"Claude chat test failed for {endpoint_label}: {e}")
                return False
                
        else:  # completion
            test_prompt = "Say hello"
            try:
                result = endpoint_handler(test_prompt)
                if result:
                    print(f"Claude completion test passed for {endpoint_label}")
                    return True
                print(f"Claude completion test failed for {endpoint_label}: No result")
                return False
            except Exception as e:
                print(f"Claude completion test failed for {endpoint_label}: {e}")
                return False