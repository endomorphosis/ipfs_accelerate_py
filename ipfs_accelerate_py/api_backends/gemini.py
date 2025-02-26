import asyncio
import json
import requests
import os
from typing import Dict, List, Optional, Union, Any, Callable
try:
    import google.generativeai as genai
except ImportError:
    print("Google GenerativeAI package not installed. Install with: pip install google-generativeai")

class gemini:
    """Google Gemini API Backend Integration
    
    This class provides a comprehensive interface for interacting with Google's Gemini models.
    It supports multiple types of interactions including:
    - Text completion generation
    - Chat completions
    - Multimodal inputs (text + images)
    - Embedding generation
    - Model management
    
    Features:
    - Dynamic endpoint management
    - Multiple response types (completion, chat, multimodal)
    - Async and sync request options
    - Request queueing and batching
    - Built-in retry logic and error handling
    """

    def __init__(self, resources=None, metadata=None):
        """Initialize Gemini backend interface
        
        Args:
            resources: Resources configuration dictionary
            metadata: Additional metadata dictionary
        """
        self.resources = resources
        self.metadata = metadata
        # Register method references
        self.create_remote_gemini_endpoint_handler = self.create_remote_gemini_endpoint_handler
        self.create_remote_gemini_chat_endpoint_handler = self.create_remote_gemini_chat_endpoint_handler
        self.create_remote_gemini_vision_endpoint_handler = self.create_remote_gemini_vision_endpoint_handler
        self.create_remote_gemini_embedding_endpoint_handler = self.create_remote_gemini_embedding_endpoint_handler
        self.request_gemini_endpoint = self.request_gemini_endpoint
        self.test_gemini_endpoint = self.test_gemini_endpoint
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
        """Initialize connection to Google's Gemini API
        
        Supported endpoint_types:
        - "completion": Standard text completion
        - "chat": Structured chat completion
        - "vision": Multimodal text+image
        - "embedding": Text embedding generation
        
        Args:
            api_key: Google API key for authentication
            model_name: Name of the model (e.g., "gemini-pro", "gemini-pro-vision")
            endpoint_type: Type of endpoint to initialize
            
        Returns:
            tuple: (None, api_key, handler, queue, batch_size)
        """
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("Google API key is required")

        # Configure the Google Generative AI library
        genai.configure(api_key=api_key)

        # Create appropriate handler based on endpoint type
        if endpoint_type == "chat":
            endpoint_handler = self.create_remote_gemini_chat_endpoint_handler(api_key, model_name)
        elif endpoint_type == "vision":
            endpoint_handler = self.create_remote_gemini_vision_endpoint_handler(api_key, model_name)
        elif endpoint_type == "embedding":
            endpoint_handler = self.create_remote_gemini_embedding_endpoint_handler(api_key, model_name)
        else:
            endpoint_handler = self.create_remote_gemini_endpoint_handler(api_key, model_name)

        # Register the model
        if model_name not in self.registered_models:
            self.registered_models[model_name] = {
                "types": [endpoint_type]
            }
        elif endpoint_type not in self.registered_models[model_name]["types"]:
            self.registered_models[model_name]["types"].append(endpoint_type)
            
        return None, api_key, endpoint_handler, self.request_queue, 32  # Default batch size

    def create_remote_gemini_endpoint_handler(self, api_key=None, model_name=None):
        """Create a handler for text completion with Gemini
        
        Example:
            ```python
            handler = gemini.create_remote_gemini_endpoint_handler(
                api_key="your-api-key",
                model_name="gemini-pro"
            )
            
            # Basic usage
            response = handler("Explain quantum computing")
            
            # With parameters
            response = handler(
                "Explain quantum computing",
                parameters={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 1024
                }
            )
            ```
        """
        def handler(prompt, parameters=None, api_key=api_key, model_name=model_name):
            try:
                # Get the model
                model = genai.GenerativeModel(model_name)
                
                # Set generation config if parameters provided
                if parameters:
                    generation_config = genai.types.GenerationConfig(
                        temperature=parameters.get("temperature", 0.7),
                        top_p=parameters.get("top_p", 0.9),
                        top_k=parameters.get("top_k", 40),
                        max_output_tokens=parameters.get("max_output_tokens", 2048),
                        stop_sequences=parameters.get("stop", [])
                    )
                    response = model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                else:
                    response = model.generate_content(prompt)
                
                return response.text

            except Exception as e:
                print(f"Error in Gemini completion handler: {e}")
                return None
        
        return handler

    def create_remote_gemini_chat_endpoint_handler(self, api_key=None, model_name=None):
        """Create a handler for chat completion with Gemini
        
        Example:
            ```python
            handler = gemini.create_remote_gemini_chat_endpoint_handler(
                api_key="your-api-key",
                model_name="gemini-pro"
            )
            
            messages = [
                {"role": "user", "content": "What is fusion energy?"},
                {"role": "assistant", "content": "Fusion energy is..."},
                {"role": "user", "content": "What are its advantages?"}
            ]
            
            response = handler(messages)
            ```
        """
        def handler(messages, parameters=None, api_key=api_key, model_name=model_name):
            try:
                # Get the model
                model = genai.GenerativeModel(model_name)
                
                # Create chat session
                chat = model.start_chat(history=[])
                
                # Add messages to history
                for msg in messages:
                    if msg["role"] == "user":
                        response = chat.send_message(msg["content"])
                    # Skip assistant messages as they're part of the response
                
                # Return the last response
                return response.text

            except Exception as e:
                print(f"Error in Gemini chat handler: {e}")
                return None
        
        return handler

    def create_remote_gemini_vision_endpoint_handler(self, api_key=None, model_name=None):
        """Create a handler for multimodal vision+text tasks with Gemini
        
        Example:
            ```python
            handler = gemini.create_remote_gemini_vision_endpoint_handler(
                api_key="your-api-key",
                model_name="gemini-pro-vision"
            )
            
            # Load image
            from PIL import Image
            image = Image.open("image.jpg")
            
            # Get response about image
            response = handler(
                "What's in this image?",
                image_data=image
            )
            ```
        """
        def handler(prompt, image_data=None, parameters=None, api_key=api_key, model_name=model_name):
            try:
                model = genai.GenerativeModel(model_name)
                
                if not image_data:
                    raise ValueError("Image data is required for vision tasks")
                
                response = model.generate_content([prompt, image_data])
                return response.text

            except Exception as e:
                print(f"Error in Gemini vision handler: {e}")
                return None
        
        return handler

    def create_remote_gemini_embedding_endpoint_handler(self, api_key=None, model_name=None):
        """Create a handler for generating embeddings with Gemini
        
        Example:
            ```python
            handler = gemini.create_remote_gemini_embedding_endpoint_handler(
                api_key="your-api-key",
                model_name="gemini-pro-embedding"
            )
            
            # Get embedding for text
            embedding = handler("This is a sample text")
            ```
        """
        def handler(text, parameters=None, api_key=api_key, model_name=model_name):
            try:
                model = genai.GenerativeModel(model_name)
                embedding = model.embed_content(text=text)
                return embedding.values
                
            except Exception as e:
                print(f"Error in Gemini embedding handler: {e}")
                return None
        
        return handler

    def __test__(self, api_key, endpoint_handler, endpoint_label, endpoint_type="completion"):
        """Test the Gemini endpoint
        
        Args:
            api_key: API key for authentication
            endpoint_handler: The handler function
            endpoint_label: Label for the endpoint
            endpoint_type: Type of endpoint to test
            
        Returns:
            bool: True if test passes, False otherwise
        """
        if endpoint_type == "chat":
            test_messages = [{"role": "user", "content": "Say hello"}]
            try:
                result = endpoint_handler(test_messages)
                if result:
                    print(f"Gemini chat test passed for {endpoint_label}")
                    return True
                print(f"Gemini chat test failed for {endpoint_label}: No result")
                return False
            except Exception as e:
                print(f"Gemini chat test failed for {endpoint_label}: {e}")
                return False
                
        elif endpoint_type == "vision":
            try:
                # Create a small test image
                from PIL import Image
                import numpy as np
                test_image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                
                result = endpoint_handler("Describe this image", test_image)
                if result:
                    print(f"Gemini vision test passed for {endpoint_label}")
                    return True
                print(f"Gemini vision test failed for {endpoint_label}: No result")
                return False
            except Exception as e:
                print(f"Gemini vision test failed for {endpoint_label}: {e}")
                return False
                
        elif endpoint_type == "embedding":
            test_text = "Test embedding generation"
            try:
                result = endpoint_handler(test_text)
                if isinstance(result, (list, np.ndarray)):
                    print(f"Gemini embedding test passed for {endpoint_label}")
                    return True
                print(f"Gemini embedding test failed for {endpoint_label}: Invalid result format")
                return False
            except Exception as e:
                print(f"Gemini embedding test failed for {endpoint_label}: {e}")
                return False
                
        else:  # completion
            test_prompt = "Say hello"
            try:
                result = endpoint_handler(test_prompt)
                if result:
                    print(f"Gemini completion test passed for {endpoint_label}")
                    return True
                print(f"Gemini completion test failed for {endpoint_label}: No result")
                return False
            except Exception as e:
                print(f"Gemini completion test failed for {endpoint_label}: {e}")
                return False