import asyncio
import os
import requests
import json

class hf_tei:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        # Register method references
        self.create_remote_text_embedding_endpoint_handler = self.create_remote_text_embedding_endpoint_handler
        self.request_hf_tei_endpoint = self.request_hf_tei_endpoint
        self.make_post_request_hf_tei = self.make_post_request_hf_tei
        self.create_hf_tei_endpoint_handler = self.create_hf_tei_endpoint_handler
        self.test_hf_tei_endpoint = self.test_hf_tei_endpoint
        self.init = self.init
        self.__test__ = self.__test__
        # Add queue for managing requests
        self.request_queue = asyncio.Queue(64)
        
    def init(self, endpoint_url=None, api_key=None, model_name=None):
        """Initialize a connection to a remote text embedding interface endpoint
        
        Args:
            endpoint_url: The URL of the remote endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            
        Returns:
            tuple: (endpoint_url, api_key, handler, queue, batch_size)
        """
        # Create the endpoint handler
        endpoint_handler = self.create_remote_text_embedding_endpoint_handler(endpoint_url, api_key, model_name)
        
        return endpoint_url, api_key, endpoint_handler, self.request_queue, 0
    
    def __test__(self, endpoint_url, endpoint_handler, endpoint_label, api_key=None):
        """Test the remote text embedding endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            endpoint_handler: The handler function
            endpoint_label: Label for the endpoint
            api_key: API key for authentication
            
        Returns:
            bool: True if test passes, False otherwise
        """
        test_text = "The quick brown fox jumps over the lazy dog"
        try:
            result = endpoint_handler(test_text)
            if result is not None:
                print(f"Remote text embedding test passed for {endpoint_label}")
                return True
            else:
                print(f"Remote text embedding test failed for {endpoint_label}: No result")
                return False
        except Exception as e:
            print(f"Remote text embedding test failed for {endpoint_label}: {e}")
            return False
    
    def request_hf_tei_endpoint(self, model, endpoint=None, endpoint_type=None, batch=None):
        """Request a text embedding endpoint
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL (optional)
            endpoint_type: Type of endpoint (optional)
            batch: Batch size (optional)
            
        Returns:
            dict: Information about the endpoint
        """
        # Implementation details
        pass
    
    async def test_hf_tei_endpoint(self, model, endpoint_list=None):
        """Test a list of text embedding endpoints
        
        Args:
            model: Name of the model
            endpoint_list: List of endpoints to test
            
        Returns:
            list: Results of endpoint tests
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        local_endpoints = self.resources["tei_endpoints"]
        local_endpoints_types = [x[1] for x in local_endpoints]
        local_endpoints_by_model = self.endpoints["tei_endpoints"][model]
        endpoint_handlers_by_model = self.resources["tei_endpoints"][model]
        local_endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
        local_endpoints_by_model_by_endpoint = [ x for x in local_endpoints_by_model_by_endpoint if x in local_endpoints_by_model if x in local_endpoints_types]
        if len(local_endpoints_by_model_by_endpoint) > 0:
            for endpoint in local_endpoints_by_model_by_endpoint:
                endpoint_handler = endpoint_handlers_by_model[endpoint]
                try:
                    test = await endpoint_handler("hello world")
                    test_results[endpoint] = test
                except Exception as e:
                    try:
                        test = endpoint_handler("hello world")
                        test_results[endpoint] = test
                    except Exception as e:
                        test_results[endpoint] = e
                    pass
        else:
            return ValueError("No endpoint_handlers found")
        return test_results
    
    async def create_hf_tei_endpoint_handler(self, model, endpoint=None, endpoint_type=None, batch=None):
        """Create an endpoint handler for text embedding
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL (optional)
            endpoint_type: Type of endpoint (optional)
            batch: Batch size (optional)
            
        Returns:
            function: Handler for the endpoint
        """
        if batch == None:
            incoming_batch_size = 0
        else:
            incoming_batch_size = len(batch)
        endpoint_batch_size = 0
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
        elif endpoint_type == None:
            for endpoint_type in self.endpoint_types:
                if endpoint_type in self.__dict__.keys():
                    if model in self.__dict__[endpoint_type]:
                        for endpoint in self.__dict__[endpoint_type][model]:
                            endpoint_batch_size = self.endpoint_status[endpoint]
                            if self.endpoint_status[endpoint] >= incoming_batch_size:
                                return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
                else:
                    pass
        else:
            if model in self.__dict__[endpoint_type]:
                for endpoint in self.__dict__[endpoint_type][model]:
                    endpoint_batch_size = self.endpoint_status[endpoint]
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
            else:
                return None
                
        if incoming_batch_size > endpoint_batch_size:
            return ValueError("Batch size too large")
        else:
            if model in self.endpoints:
                for endpoint in self.tei_endpoints[model]:
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
            return None
    
    def make_post_request_hf_tei(self, endpoint, data=None, api_key=None):
        """Make a POST request to a remote text embedding interface
        
        Args:
            endpoint: URL of the endpoint
            data: Data to send in the request
            api_key: API key for authentication, if required
            
        Returns:
            dict: Response from the endpoint
            
        Raises:
            ValueError: If the request fails or the response status is not 200
        """
        try:
            # Set up headers
            headers = {'Content-Type': 'application/json'}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
                
            # Format the data properly
            if data is None:
                return None
                
            # Convert different input formats to the expected format
            if isinstance(data, dict):
                if "inputs" not in data:
                    data = {"inputs": data}
            elif isinstance(data, list):
                data = {"inputs": data}
            elif isinstance(data, str):
                data = {"inputs": data}
                
            # Make the synchronous request
            response = requests.post(endpoint, headers=headers, json=data)
            
            # Handle error status codes
            if response.status_code == 401:
                raise ValueError(f"Authentication failed (401): Please check your API key")
            elif response.status_code == 404:
                raise ValueError(f"Resource not found (404): Model or endpoint may not exist")
            elif response.status_code >= 400:
                error_msg = f"Request failed with status code {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error']}"
                except:
                    pass
                raise ValueError(error_msg)
                
            # Parse and return the response
            return response.json()
            
        except ValueError:
            # Re-raise ValueError exceptions
            raise
        except Exception as e:
            # Convert other exceptions to ValueError
            error_msg = str(e)
            if "Connection" in error_msg:
                raise ValueError(f"Connection error: {error_msg}")
            elif "Timeout" in error_msg:
                raise ValueError(f"Timeout error: {error_msg}")
            else:
                raise ValueError(f"Error in request: {error_msg}")
                
    async def make_async_post_request_hf_tei(self, endpoint, data=None, api_key=None):
        """Make an asynchronous POST request to a remote text embedding interface
        
        Args:
            endpoint: URL of the endpoint
            data: Data to send in the request
            api_key: API key for authentication, if required
            
        Returns:
            dict: Response from the endpoint
            
        Raises:
            ValueError: If the request fails or the response status is not 200
        """
        import aiohttp
        from aiohttp import ClientSession, ClientTimeout
        
        # Format the data properly
        if data is None:
            return None
            
        # Convert different input formats to the expected format
        if isinstance(data, dict):
            if "inputs" not in data:
                data = {"inputs": data}
        elif isinstance(data, list):
            data = {"inputs": data}
        elif isinstance(data, str):
            data = {"inputs": data}
            
        # Set up headers
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            
        timeout = ClientTimeout(total=300) 
        
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    # Handle error status codes
                    if response.status == 401:
                        raise ValueError(f"Authentication failed (401): Please check your API key")
                    elif response.status == 404:
                        raise ValueError(f"Resource not found (404): Model or endpoint may not exist")
                    elif response.status >= 400:
                        error_text = await response.text()
                        try:
                            error_data = json.loads(error_text)
                            if "error" in error_data:
                                error_msg = f"Error {response.status}: {error_data['error']}"
                            else:
                                error_msg = f"Error {response.status}: {error_text}"
                        except:
                            error_msg = f"Error {response.status}: {error_text}"
                        raise ValueError(error_msg)
                        
                    # Parse and return the response
                    return await response.json()
                    
        except aiohttp.ClientPayloadError as e:
            raise ValueError(f"ClientPayloadError: {str(e)}")
        except asyncio.TimeoutError as e:
            raise ValueError(f"Timeout error: {str(e)}")
        except ValueError:
            # Re-raise ValueError exceptions
            raise
        except Exception as e:
            # Convert other exceptions to ValueError
            raise ValueError(f"Unexpected error: {str(e)}")

    def create_remote_text_embedding_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for a remote text embedding endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            
        Returns:
            function: Handler for the endpoint
        """
        def handler(input_text, endpoint_url=endpoint_url, api_key=api_key, model_name=model_name):
            try:
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                data = {
                    "input": input_text,
                    "model": model_name
                }
                
                response = requests.post(endpoint_url, headers=headers, json=data)
                response.raise_for_status()
                
                result = response.json()
                return result.get("embeddings", result)
            
            except Exception as e:
                print(f"Error making request to text embedding endpoint: {e}")
                return None
        
        return handler