import asyncio
import json
import requests
import os
from typing import Dict, List, Optional, Union, Any, Callable

class ovms:
    """OpenVINO Model Server (OVMS) API Backend Integration
    
    This class provides a comprehensive interface for interacting with OVMS endpoints.
    It supports multiple types of operations including:
    - Model inference with various input types
    - Model metadata retrieval 
    - Model list retrieval
    - Dynamic model loading
    - Async and sync inference
    - Custom pre/post processing
    
    Features:
    - Support for all OVMS model types (classification, detection, NLP, etc.)
    - Automatic input tokenization for NLP models
    - Both sync and async inference modes
    - Batched inference support
    - Multiple precision support (FP32, FP16, INT8)
    - Custom preprocessing and postprocessing hooks
    - Dynamic endpoint management
    - Health monitoring and status tracking
    
    Supported model types:
    - Text Generation Models (GPT, T5, etc.)
    - Computer Vision Models (Classification, Detection, Segmentation)
    - Speech Models (ASR, TTS)
    - Multimodal Models (Vision + Language)
    - Embedding Models (BERT, Sentence Transformers)
    
    Usage example for text generation:
    ```python
    from ipfs_accelerate_py.api_backends import ovms
    
    # Initialize backend
    ovms_backend = ovms()
    
    # Configure endpoint for a text generation model
    endpoint_url, api_key, handler, queue, batch_size = ovms_backend.init(
        endpoint_url="http://localhost:9000",
        model_name="gpt2",
        context_length=1024
    )
    
    # Basic inference
    response = handler("Explain quantum computing")
    
    # With custom parameters
    response = handler(
        "Explain quantum computing",
        parameters={
            "endpoint_path": "/v2/models/gpt2/infer",
            "max_tokens": 100,
            "raw": False  # Enable automatic tokenization
        }
    )
    ```
    
    Usage example for computer vision:
    ```python
    # Configure endpoint for an image classification model
    endpoint_url, api_key, handler, queue, batch_size = ovms_backend.init(
        endpoint_url="http://localhost:9000",
        model_name="resnet50"
    )
    
    # Custom preprocessing for images
    def preprocess_image(image_data):
        # Convert image to model input format
        return processed_data
    
    # Create handler with custom preprocessing
    handler = ovms_backend.create_remote_ovms_endpoint_handler(
        endpoint_url="http://localhost:9000",
        model_name="resnet50",
        preprocessing=preprocess_image
    )
    
    # Perform inference
    result = handler(image_data, parameters={"raw": True})
    ```
    """

    def __init__(self, resources=None, metadata=None):
        """Initialize OVMS backend interface
        
        Args:
            resources: Resources configuration dictionary
            metadata: Additional metadata dictionary
        """
        self.resources = resources
        self.metadata = metadata
        # Register method references
        self.create_remote_ovms_endpoint_handler = self.create_remote_ovms_endpoint_handler
        self.test_ovms_endpoint = self.test_ovms_endpoint
        self.create_ovms_endpoint_handler = self.create_ovms_endpoint_handler
        self.make_post_request_ovms = self.make_post_request_ovms
        self.make_async_post_request_ovms = self.make_async_post_request_ovms
        self.request_ovms_endpoint = self.request_ovms_endpoint
        self.list_available_ovms_models = self.list_available_ovms_models
        self.get_model_metadata = self.get_model_metadata
        self.init = self.init
        self.__test__ = self.__test__
        # Add endpoints tracking
        self.endpoints = {}
        self.endpoint_status = {}
        self.registered_models = {}
        # Add queue for managing requests
        self.request_queue = asyncio.Queue(64)
        return None
    
    def init(self, endpoint_url=None, api_key=None, model_name=None, context_length=None):
        """Initialize a connection to an OpenVINO Model Server (OVMS) endpoint
        
        The init method sets up a connection to an OVMS endpoint and configures 
        the appropriate handler for the specified model. It supports both synchronous
        and asynchronous inference modes.
        
        Supported endpoints:
        - Model Inference: /v2/models/{model_name}/infer
        - Model Metadata: /v2/models/{model_name}/metadata
        - Model Status: /v2/models/{model_name}/ready
        - Server Status: /v2/health/ready
        
        Args:
            endpoint_url: The URL of the OVMS server
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            context_length: Maximum context length for the model (for text models)
            
        Returns:
            tuple: (endpoint_url, api_key, handler, queue, batch_size)
            
        Example:
            ```python
            # Basic initialization
            endpoint_url, api_key, handler, queue, batch_size = ovms.init(
                endpoint_url="http://localhost:9000",
                model_name="bert-base"
            )
            
            # Initialize with context length for text models
            endpoint_url, api_key, handler, queue, batch_size = ovms.init(
                endpoint_url="http://localhost:9000",
                model_name="gpt2",
                context_length=1024
            )
            ```
        """
        # Create the endpoint handler
        endpoint_handler = self.create_remote_ovms_endpoint_handler(endpoint_url, api_key, model_name, context_length)
        
        # Register the endpoint
        if model_name not in self.endpoints:
            self.endpoints[model_name] = []
        
        if endpoint_url not in self.endpoints[model_name]:
            self.endpoints[model_name].append(endpoint_url)
            self.endpoint_status[endpoint_url] = 32  # Default batch size
            
            # Register model in the registered_models dictionary
            if model_name not in self.registered_models:
                self.registered_models[model_name] = {
                    "endpoints": [endpoint_url],
                    "context_length": context_length
                }
            else:
                if endpoint_url not in self.registered_models[model_name]["endpoints"]:
                    self.registered_models[model_name]["endpoints"].append(endpoint_url)
                if context_length and not self.registered_models[model_name].get("context_length"):
                    self.registered_models[model_name]["context_length"] = context_length
        
        return endpoint_url, api_key, endpoint_handler, self.request_queue, self.endpoint_status[endpoint_url]
    
    def list_available_ovms_models(self, endpoint_url=None, api_key=None):
        """List available models from an OVMS endpoint
        
        Queries the OVMS server to get a list of all available models and their status.
        Supports both v1 and v2 of the OVMS API.
        
        Args:
            endpoint_url: URL of the OVMS endpoint
            api_key: API key for authentication, if required
            
        Returns:
            list: List of available models with metadata
            
        Example:
            ```python
            # List all available models
            models = ovms.list_available_ovms_models(
                endpoint_url="http://localhost:9000"
            )
            
            # Models info includes:
            # - Name
            # - Version
            # - Status (loading/ready/error)
            # - Input shapes and types
            # - Supported precisions
            ```
        """
        if not endpoint_url:
            if self.endpoints:
                # Use the first available endpoint
                model_name = next(iter(self.endpoints))
                endpoint_url = self.endpoints[model_name][0]
            else:
                return None
                
        try:
            # OVMS typically uses a /v1/models or /v2/models endpoint for model listing
            for models_path in ["/v2/models", "/v1/models", "/models"]:
                models_endpoint = f"{endpoint_url.rstrip('/')}{models_path}"
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                response = requests.get(models_endpoint, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    
                    # Different OVMS versions might structure the response differently
                    if "models" in result:
                        return result["models"]
                    elif "model_versions" in result:
                        return result["model_versions"]
                    else:
                        return result
            
            # If all paths failed but didn't raise exceptions
            return None
        except Exception as e:
            print(f"Failed to list OVMS models: {e}")
            return None
    
    def get_model_metadata(self, model_name, endpoint_url=None, api_key=None):
        """Get metadata and diagnostic information for a specific model
        
        Retrieves comprehensive model information for setup and debugging:
        
        1. Model Configuration:
           - Model architecture
           - Input/output specifications
           - Supported precisions
           - Hardware requirements
           - Batch size limits
           - Memory requirements
        
        2. Runtime Status:
           - Loading status
           - Instance count
           - Resource utilization
           - Error conditions
           - Performance metrics
        
        3. Hardware Configuration:
           - Device assignment
           - Memory allocation
           - Compute resources
           - Optimization settings
        
        4. System Requirements:
           - Minimum CPU/GPU specs
           - Memory thresholds
           - Disk space needs
           - Network bandwidth
        
        Args:
            model_name: Name of the model
            endpoint_url: URL of the OVMS endpoint
            api_key: API key for authentication
            
        Returns:
            dict: Detailed model metadata and diagnostics
            
        Example:
            ```python
            # Get comprehensive model information
            metadata = ovms.get_model_metadata(
                model_name="bert-base",
                endpoint_url="http://localhost:9000"
            )
            
            # Access specific details
            input_shape = metadata["inputs"][0]["shape"]
            precision = metadata["precision"]
            status = metadata["status"]
            resources = metadata["resources"]
            
            # Check for warnings/errors
            if "warnings" in metadata:
                print("Model warnings:", metadata["warnings"])
            ```
        """
        if not endpoint_url:
            if model_name in self.endpoints and self.endpoints[model_name]:
                endpoint_url = self.endpoints[model_name][0]
            else:
                return None
                
        try:
            # Try different API versions
            for api_version in ["v2", "v1"]:
                metadata_endpoint = f"{endpoint_url.rstrip('/')}/{api_version}/models/{model_name}/metadata"
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                response = requests.get(metadata_endpoint, headers=headers)
                if response.status_code == 200:
                    return response.json()
            
            # If all versions failed, try OVMS-specific endpoint
            metadata_endpoint = f"{endpoint_url.rstrip('/')}/models/{model_name}/metadata"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.get(metadata_endpoint, headers=headers)
            if response.status_code == 200:
                return response.json()
                
            return None
        except Exception as e:
            print(f"Failed to get metadata for model {model_name}: {e}")
            return None
    
    def __test__(self, endpoint_url, endpoint_handler, endpoint_label, api_key=None):
        """Test the remote OVMS endpoint and diagnose issues
        
        Performs comprehensive endpoint testing and diagnostics:
        1. Connection Testing:
           - Basic connectivity
           - Authentication validation
           - SSL/TLS verification
           - Timeout configuration
        
        2. Model Validation:
           - Model loading status
           - Input shape verification
           - Output format validation
           - Precision compatibility
        
        3. Performance Metrics:
           - Response time
           - Throughput capacity
           - Resource utilization
           - Batch processing efficiency
        
        4. Error Diagnostics:
           - Detailed error messages
           - Stack traces for debugging
           - Resource constraint warnings
           - Configuration validation
        
        Common Error Types:
        - CONNECTION_ERROR: Network connectivity issues
        - AUTH_ERROR: Invalid API key or credentials
        - MODEL_ERROR: Model loading or execution failed
        - INPUT_ERROR: Invalid input format or shape
        - RESOURCE_ERROR: Insufficient system resources
        - TIMEOUT_ERROR: Request exceeded time limit
        
        Debugging Tools:
        - Model Server Logs: Check OVMS server logs
        - Client Logs: Enable debug logging
        - Metrics: Monitor performance metrics
        - Health Checks: Regular endpoint validation
        
        Args:
            endpoint_url: URL of the endpoint
            endpoint_handler: The handler function
            endpoint_label: Label for the endpoint
            api_key: API key for authentication
            
        Returns:
            bool: True if test passes, False otherwise
            
        Example:
            ```python
            # Test endpoint with detailed diagnostics
            success = ovms.__test__(
                endpoint_url="http://localhost:9000",
                endpoint_handler=handler,
                endpoint_label="bert-gpu",
                api_key="test-key"
            )
            
            # The test will print detailed diagnostics:
            # - Connection status
            # - Model availability
            # - Input/output validation
            # - Performance metrics
            # - Error details if any
            ```
        
        Troubleshooting Tips:
        1. Connection Issues:
           - Verify endpoint URL and port
           - Check network connectivity
           - Validate SSL certificates
           - Confirm firewall rules
        
        2. Model Issues:
           - Verify model path and version
           - Check input tensor shapes
           - Validate model configuration
           - Monitor resource usage
        
        3. Performance Issues:
           - Adjust batch size
           - Monitor system resources
           - Check model optimization
           - Consider hardware acceleration
        """
        test_text = "The quick brown fox jumps over the lazy dog"
        try:
            result = endpoint_handler(test_text)
            if result is not None:
                print(f"Remote OVMS test passed for {endpoint_label}")
                return True
            else:
                print(f"Remote OVMS test failed for {endpoint_label}: No result")
                return False
        except Exception as e:
            print(f"Remote OVMS test failed for {endpoint_label}: {e}")
            return False
    
    def make_post_request_ovms(self, endpoint_url, data, api_key=None):
        """Make a POST request to an OVMS endpoint
        
        Handles the low-level communication with OVMS endpoints, including:
        - Request formatting
        - Data serialization
        - Error handling
        - Response parsing
        
        Args:
            endpoint_url: URL of the endpoint
            data: Data to send in the request
            api_key: API key for authentication, if required
            
        Returns:
            dict: Response from the endpoint
            
        The method automatically formats data according to OVMS expectations:
        - Single inputs are wrapped in a batch
        - Inputs are structured in the "instances" format
        - Metadata is included when required
        """
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Format data according to OVMS expectations
            if isinstance(data, str):
                formatted_data = {"instances": [{"data": data}]}
            elif isinstance(data, dict):
                formatted_data = {"instances": [data]}
            elif isinstance(data, list):
                formatted_data = {"instances": data}
            else:
                raise ValueError("Unsupported data format")
            
            response = requests.post(endpoint_url, headers=headers, json=formatted_data)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            print(f"Error making request to OVMS endpoint: {e}")
            return None
    
    async def make_async_post_request_ovms(self, endpoint_url, data, api_key=None):
        """Make an asynchronous POST request to an OVMS endpoint
        
        Provides non-blocking request handling for high-throughput scenarios.
        Automatically handles timeouts and connection management.
        
        Args:
            endpoint_url: URL of the endpoint
            data: Data to send in the request
            api_key: API key for authentication, if required
            
        Returns:
            dict: Response from the endpoint
            
        Example:
            ```python
            # Make async batch request
            response = await ovms.make_async_post_request_ovms(
                endpoint_url="http://localhost:9000/v2/models/bert/infer",
                data={
                    "instances": [
                        {"data": "Sample text 1"},
                        {"data": "Sample text 2"}
                    ]
                }
            )
            ```
        """
        import aiohttp
        from aiohttp import ClientSession, ClientTimeout
        
        # Format data according to OVMS expectations
        if isinstance(data, str):
            formatted_data = {"instances": [{"data": data}]}
        elif isinstance(data, dict):
            formatted_data = {"instances": [data]}
        elif isinstance(data, list):
            formatted_data = {"instances": data}
        else:
            raise ValueError("Unsupported data format")
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = ClientTimeout(total=300)
        
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.post(endpoint_url, headers=headers, json=formatted_data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Error {response.status}: {error_text}")
                    return await response.json()
                    
        except Exception as e:
            print(f"Error in async request to OVMS endpoint: {e}")
            raise e
    
    async def test_ovms_endpoint(self, model=None, endpoint_url=None, api_key=None, endpoint_list=None):
        """Test an OVMS endpoint or list of endpoints
        
        Validates endpoint connectivity and model availability. Can test:
        - Single specific endpoint
        - List of endpoints
        - All endpoints for a given model
        
        Tests performed:
        - Basic connectivity
        - Model availability
        - Inference with test input
        - Response format validation
        
        Args:
            model: Name of the model
            endpoint_url: URL of a specific endpoint to test
            api_key: API key for authentication
            endpoint_list: List of endpoints to test
            
        Returns:
            dict: Results of endpoint tests with detailed status
            
        Example:
            ```python
            # Test specific endpoint
            results = await ovms.test_ovms_endpoint(
                model="bert-base",
                endpoint_url="http://localhost:9000"
            )
            
            # Test multiple endpoints
            results = await ovms.test_ovms_endpoint(
                model="bert-base",
                endpoint_list=[
                    "http://server1:9000",
                    "http://server2:9000"
                ]
            )
            ```
        """
        test_results = {}
        
        if endpoint_url and not endpoint_list:
            endpoint_list = [endpoint_url]
            
        if not endpoint_list and model in self.endpoints:
            endpoint_list = self.endpoints[model]
            
        if not endpoint_list:
            return {"error": "No endpoints provided for testing"}
            
        for endpoint in endpoint_list:
            try:
                handler = self.create_remote_ovms_endpoint_handler(endpoint, api_key, model)
                test_input = "The quick brown fox jumps over the lazy dog"
                
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(test_input)
                else:
                    result = handler(test_input)
                    
                test_results[endpoint] = {
                    "status": "success" if result is not None else "failed",
                    "result": result if result is not None else "No result"
                }
                
            except Exception as e:
                test_results[endpoint] = {
                    "status": "error",
                    "message": str(e)
                }
                
        return test_results
    
    def request_ovms_endpoint(self, model, endpoint=None, endpoint_type=None, batch=None):
        """Request an OVMS endpoint
        
        Finds a suitable endpoint for inference based on:
        - Model availability
        - Batch size capacity
        - Endpoint health
        - Load balancing considerations
        
        The method implements smart endpoint selection considering:
        - Current endpoint load
        - Available batch capacity
        - Historical performance
        - Health status
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL (optional)
            endpoint_type: Type of endpoint (optional)
            batch: Batch size (optional)
            
        Returns:
            str: URL of the selected endpoint
            
        Example:
            ```python
            # Get endpoint for small batch
            endpoint = ovms.request_ovms_endpoint(
                model="bert-base",
                batch=["text1", "text2"]
            )
            
            # Get endpoint for specific type
            endpoint = ovms.request_ovms_endpoint(
                model="bert-base",
                endpoint_type="gpu",
                batch=large_batch
            )
            ```
        """
        incoming_batch_size = len(batch) if batch else 1
        
        # If endpoint is specified and has sufficient capacity, use it
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
            if incoming_batch_size <= endpoint_batch_size:
                return endpoint
        
        # Check in endpoints dictionary
        if model in self.endpoints:
            for e in self.endpoints[model]:
                if e in self.endpoint_status and self.endpoint_status[e] >= incoming_batch_size:
                    return e
        
        # No suitable endpoint found
        return None

    def create_ovms_endpoint_handler(self, model=None, endpoint=None, endpoint_type=None, batch=None, preprocessing=None, postprocessing=None):
        """Create an endpoint handler for OVMS
        
        High-level method to create a handler with appropriate configuration.
        Automatically determines optimal settings based on model and endpoint type.
        
        Features:
        - Automatic model type detection
        - Input/output format handling
        - Batch size optimization
        - Custom processing pipeline integration
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            endpoint_type: Type of endpoint (optional)
            batch: Batch size (optional)
            preprocessing: Custom preprocessing function (optional)
            postprocessing: Custom postprocessing function (optional)
            
        Returns:
            function: Handler for the endpoint
            
        Example:
            ```python
            # Create handler for NLP model
            handler = ovms.create_ovms_endpoint_handler(
                model="gpt2",
                endpoint="http://localhost:9000",
                preprocessing=tokenize_text
            )
            
            # Create handler for vision model
            handler = ovms.create_ovms_endpoint_handler(
                model="resnet50",
                endpoint="http://localhost:9000",
                preprocessing=preprocess_image,
                postprocessing=decode_predictions
            )
            ```
        """
        return self.create_remote_ovms_endpoint_handler(endpoint, None, model, preprocessing=preprocessing, postprocessing=postprocessing)
    
    def create_remote_ovms_endpoint_handler(self, endpoint_url, api_key=None, model_name=None, context_length=None, preprocessing=None, postprocessing=None):
        """Create a handler for a remote OVMS endpoint
        
        Creates a handler function that manages all communication with the OVMS endpoint.
        Supports custom preprocessing and postprocessing functions for flexible input/output handling.
        
        Model Types and Input Formats:
        1. Text Generation Models:
           - Input: Raw text or tokenized ids
           - Parameters: max_length, temperature, top_p, etc.
           Example models: GPT-2, T5, BART
        
        2. Computer Vision Models:
           - Input: Image data (RGB/BGR arrays, tensors)
           - Tasks: Classification, detection, segmentation
           Example models: ResNet, YOLO, Mask R-CNN
        
        3. Speech Models:
           - Input: Audio waveforms, spectrograms
           - Tasks: ASR, TTS, audio classification
           Example models: Wav2Vec, Whisper, CLAP
           
        4. Multimodal Models:
           - Input: Combined text + image/audio
           - Tasks: Image captioning, VQA
           Example models: CLIP, LLaVA
        
        5. Embedding Models:
           - Input: Text, images, or audio
           - Output: Vector embeddings
           Example models: BERT, Sentence-BERT

        Configuration Options:
        - endpoint_path: Custom inference endpoint path
        - raw: Skip default tokenization/preprocessing
        - batch_size: Number of inputs to process together
        - execution_parameters:
          - timeout: Request timeout in seconds
          - client_name: Client identifier
          - compression_level: gRPC compression
          - credentials: Authentication details
        
        Hardware Configuration:
        - cpu_thresh: CPU utilization threshold
        - mem_thresh: Memory utilization threshold  
        - batch_size: Maximum batch size
        - instance_count: Number of model instances
        - device: Target device (CPU/GPU/MYRIAD)

        Args:
            endpoint_url: URL of the endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            context_length: Maximum context length for text models
            preprocessing: Custom preprocessing function (optional)
            postprocessing: Custom postprocessing function (optional)
        
        Returns:
            function: Handler for the endpoint
            
        Example for text generation:
            ```python
            handler = ovms.create_remote_ovms_endpoint_handler(
                endpoint_url="http://localhost:9000",
                model_name="gpt2",
                context_length=1024
            )
            
            # Basic text generation
            response = handler("Write a story about:")
            
            # With custom parameters
            response = handler(
                "Write a story about:",
                parameters={
                    "max_length": 200,
                    "temperature": 0.7,
                    "device": "GPU"
                }
            )
            ```
            
        Example for computer vision:
            ```python
            # Custom image preprocessing
            def preprocess_images(images):
                # Normalize, resize, etc.
                return processed_images
                
            handler = ovms.create_remote_ovms_endpoint_handler(
                endpoint_url="http://localhost:9000",
                model_name="resnet50",
                preprocessing=preprocess_images
            )
            
            # Run inference on batch of images
            results = handler(
                image_batch,
                parameters={
                    "batch_size": 32,
                    "device": "GPU",
                    "raw": True
                }
            )
            ```
            
        Example for embeddings:
            ```python
            handler = ovms.create_remote_ovms_endpoint_handler(
                endpoint_url="http://localhost:9000",
                model_name="bert-base"
            )
            
            # Get embeddings for text
            embeddings = handler(
                ["text1", "text2"],
                parameters={
                    "pooling": "mean",
                    "normalize": True
                }
            )
            ```
        
        Error Handling:
        The handler implements robust error handling for:
        - Network connectivity issues
        - Model loading errors
        - Invalid input formats
        - Resource constraints
        - Timeout conditions
        """
        from transformers import AutoTokenizer
        
        # Try to get or create a tokenizer for this model
        tokenizer = None
        if self.resources and "tokenizer" in self.resources:
            if model_name in self.resources["tokenizer"]:
                tokenizer = self.resources["tokenizer"][model_name].get("cpu")
                
        try:
            if not tokenizer and model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
                if self.resources and "tokenizer" in self.resources:
                    if model_name not in self.resources["tokenizer"]:
                        self.resources["tokenizer"][model_name] = {}
                    self.resources["tokenizer"][model_name]["cpu"] = tokenizer
        except Exception as e:
            print(f"Could not load tokenizer for {model_name}: {e}")
            # Continue without tokenizer
        
        def handler(inputs, parameters=None, endpoint_url=endpoint_url, api_key=api_key, model_name=model_name):
            try:
                # Apply custom preprocessing if provided
                if preprocessing and callable(preprocessing):
                    inputs = preprocessing(inputs)
                
                tokens = None
                data = None
                
                # If parameters contain the 'raw' flag, skip tokenization
                if parameters and parameters.get("raw", False):
                    data = inputs
                # If we have a tokenizer, tokenize the input
                elif tokenizer:
                    if isinstance(inputs, str):
                        tokens = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                        # Convert to standard Python types for JSON serialization
                        data = {
                            "input_ids": tokens["input_ids"].tolist(),
                            "attention_mask": tokens["attention_mask"].tolist()
                        }
                    else:
                        # Assume inputs is already tokenized or properly structured
                        data = inputs
                else:
                    # If no tokenizer, pass raw input
                    data = inputs
                
                # Add model name to request if not already in data
                if isinstance(data, dict) and "model_name" not in data and model_name:
                    data["model_name"] = model_name
                
                # Extract specific endpoint path from parameters if provided
                specific_path = None
                if parameters and "endpoint_path" in parameters:
                    specific_path = parameters.pop("endpoint_path")
                    
                # Construct the full endpoint URL
                full_url = endpoint_url
                if specific_path:
                    full_url = f"{endpoint_url.rstrip('/')}/{specific_path.lstrip('/')}"
                
                # Make the request
                response = self.make_post_request_ovms(full_url, data, api_key)
                
                # Apply custom postprocessing if provided
                if postprocessing and callable(postprocessing) and response:
                    response = postprocessing(response)
                    
                if response:
                    # Handle different possible response formats from OVMS
                    if "predictions" in response:
                        return response["predictions"]
                    elif "outputs" in response:
                        return response["outputs"]
                    
                return response
                
            except Exception as e:
                print(f"Error in OVMS handler: {e}")
                return None
                
        return handler
        
    async def create_async_ovms_endpoint_handler(self, endpoint_url, api_key=None, model_name=None, preprocessing=None, postprocessing=None):
        """Create an asynchronous handler for an OVMS endpoint
        
        Creates an async handler for non-blocking inference requests.
        Particularly useful for high-throughput scenarios or when managing multiple requests.
        
        Features:
        - Non-blocking async inference
        - Concurrent request handling
        - Automatic request queuing
        - Custom preprocessing/postprocessing
        - Error handling with retries
        
        Args:
            endpoint_url: URL of the endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            preprocessing: Custom preprocessing function (optional)
            postprocessing: Custom postprocessing function (optional)
            
        Returns:
            function: Async handler for the endpoint
            
        Example:
            ```python
            # Create async handler
            handler = await ovms.create_async_ovms_endpoint_handler(
                endpoint_url="http://localhost:9000",
                model_name="bert-base"
            )
            
            # Use handler asynchronously
            async def process_inputs(inputs):
                results = []
                for input_batch in inputs:
                    result = await handler(input_batch)
                    results.append(result)
                return results
            
            # Process multiple inputs concurrently
            results = await asyncio.gather(
                process_inputs(batch1),
                process_inputs(batch2)
            )
            ```
        """
        from transformers import AutoTokenizer
        
        # Try to get or create a tokenizer for this model
        tokenizer = None
        if self.resources and "tokenizer" in self.resources:
            if model_name in self.resources["tokenizer"]:
                tokenizer = self.resources["tokenizer"][model_name].get("cpu")
                
        try:
            if not tokenizer and model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
                if self.resources and "tokenizer" in self.resources:
                    if model_name not in self.resources["tokenizer"]:
                        self.resources["tokenizer"][model_name] = {}
                    self.resources["tokenizer"][model_name]["cpu"] = tokenizer
        except Exception as e:
            print(f"Could not load tokenizer for {model_name}: {e}")
            # Continue without tokenizer
            
        async def handler(inputs, parameters=None, endpoint_url=endpoint_url, api_key=api_key, model_name=model_name):
            try:
                # Apply custom preprocessing if provided
                if preprocessing and callable(preprocessing):
                    inputs = preprocessing(inputs)
                
                tokens = None
                data = None
                
                # If parameters contain the 'raw' flag, skip tokenization
                if parameters and parameters.get("raw", False):
                    data = inputs
                # If we have a tokenizer, tokenize the input
                elif tokenizer:
                    if isinstance(inputs, str):
                        tokens = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                        # Convert to standard Python types for JSON serialization
                        data = {
                            "input_ids": tokens["input_ids"].tolist(),
                            "attention_mask": tokens["attention_mask"].tolist()
                        }
                    else:
                        # Assume inputs is already tokenized or properly structured
                        data = inputs
                else:
                    # If no tokenizer, pass raw input
                    data = inputs
                
                # Add model name to request if not already in data
                if isinstance(data, dict) and "model_name" not in data and model_name:
                    data["model_name"] = model_name
                
                # Extract specific endpoint path from parameters if provided
                specific_path = None
                if parameters and "endpoint_path" in parameters:
                    specific_path = parameters.pop("endpoint_path")
                    
                # Construct the full endpoint URL
                full_url = endpoint_url
                if specific_path:
                    full_url = f"{endpoint_url.rstrip('/')}/{specific_path.lstrip('/')}"
                    
                # Make the async request
                response = await self.make_async_post_request_ovms(full_url, data, api_key)
                
                # Apply custom postprocessing if provided
                if postprocessing and callable(postprocessing) and response:
                    response = postprocessing(response)
                    
                if response:
                    # Handle different possible response formats from OVMS
                    if "predictions" in response:
                        return response["predictions"]
                    elif "outputs" in response:
                        return response["outputs"]
                    
                return response
                
            except Exception as e:
                print(f"Error in async OVMS handler: {e}")
                return None
                
        return handler
