import asyncio
import os
import sys
import json
import time
import traceback
from datetime import datetime

# Set environment variables to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set environment variable to avoid fork warnings in multiprocessing
# This helps prevent the "This process is multi-threaded, use of fork() may lead to deadlocks" warnings
# Reference: https://github.com/huggingface/transformers/issues/5486
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"

# Configure to use spawn instead of fork to prevent deadlocks
import multiprocessing
if hasattr(multiprocessing, "set_start_method"):
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("Could not set multiprocessing start method to 'spawn' - already set")

# Add parent directory to sys.path for proper imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class test_ipfs_accelerate:
    """
    Test class for IPFS Accelerate Python Framework.
    
    This class provides methods to test the IPFS Accelerate Python framework and its components:
    1. Hardware backend testing
    2. IPFS accelerate model endpoint testing
    3. Local endpoints (CUDA, OpenVINO, CPU)
    4. API endpoints (TEI, OVMS)
    5. Network endpoints (libp2p, WebNN)
    
    The test process follows these phases:
    - Phase 1: Test with models defined in global metadata
    - Phase 2: Test with models from mapped_models.json
    - Phase 3: Collect and analyze test results
    - Phase 4: Generate test reports
    """
    
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the test_ipfs_accelerate class.
        
        Args:
            resources (dict, optional): Dictionary containing resources like endpoints. Defaults to None.
            metadata (dict, optional): Dictionary containing metadata like models list. Defaults to None.
        """
        # Initialize resources
        if resources is None:
            self.resources = {}
        else:
            self.resources = resources
        
        # Initialize metadata
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata
        
        # Initialize ipfs_accelerate_py
        if "ipfs_accelerate_py" not in dir(self):
            if "ipfs_accelerate_py" not in list(self.resources.keys()):
                try:
                    from ipfs_accelerate_py import ipfs_accelerate_py
                    self.resources["ipfs_accelerate_py"] = ipfs_accelerate_py(resources, metadata)
                    self.ipfs_accelerate_py = self.resources["ipfs_accelerate_py"]
                except Exception as e:
                    print(f"Error initializing ipfs_accelerate_py: {str(e)}")
                    print(traceback.format_exc())
                    self.resources["ipfs_accelerate_py"] = None
                    self.ipfs_accelerate_py = None
            else:
                self.ipfs_accelerate_py = self.resources["ipfs_accelerate_py"]
        
        # Initialize test_hardware_backend
        if "test_hardware_backend" not in dir(self):
            if "test_hardware_backend" not in list(self.resources.keys()):
                try:
                    from test_hardware_backend import test_hardware_backend
                    self.resources["test_backend"] = test_hardware_backend(resources, metadata)
                    self.test_backend = self.resources["test_backend"]
                except Exception as e:
                    print(f"Error initializing test_hardware_backend: {str(e)}")
                    print(traceback.format_exc())
                    self.resources["test_backend"] = None
                    self.test_backend = None
            else:
                self.test_backend = self.resources["test_backend"]
        
        # Initialize test_api_backend
        if "test_api_backend" not in dir(self):
            if "test_api_backend" not in list(self.resources.keys()):
                try:
                    from test_api_backend import test_api_backend
                    self.resources["test_api_backend"] = test_api_backend(resources, metadata)
                    self.test_api_backend = self.resources["test_api_backend"]
                except Exception as e:
                    print(f"Error initializing test_api_backend: {str(e)}")
                    print(traceback.format_exc())
                    self.resources["test_api_backend"] = None
                    self.test_api_backend = None
            else:
                self.test_api_backend = self.resources["test_api_backend"]
        
        # Initialize torch
        if "torch" not in dir(self):
            if "torch" not in list(self.resources.keys()):
                try:
                    import torch
                    self.resources["torch"] = torch
                    self.torch = self.resources["torch"]
                except Exception as e:
                    print(f"Error importing torch: {str(e)}")
                    self.resources["torch"] = None
                    self.torch = None
            else:
                self.torch = self.resources["torch"]
                
        # Initialize transformers module - needed for most skill tests
        if "transformers" not in list(self.resources.keys()):
            try:
                import transformers
                self.resources["transformers"] = transformers
                print("  Added transformers module to resources")
            except Exception as e:
                print(f"Error importing transformers: {str(e)}")
                # Create MagicMock for transformers if import fails
                try:
                    from unittest.mock import MagicMock
                    self.resources["transformers"] = MagicMock()
                    print("  Added MagicMock for transformers to resources")
                except Exception as mock_error:
                    print(f"Error creating MagicMock for transformers: {str(mock_error)}")
                    self.resources["transformers"] = None
        
        # Ensure required resource dictionaries exist and are properly structured
        required_resource_keys = [
            "local_endpoints", 
            "openvino_endpoints", 
            "tokenizer"
        ]
        
        for key in required_resource_keys:
            if key not in self.resources:
                self.resources[key] = {}
        
        # Convert list structures to dictionary structures if needed
        self._convert_resource_structures()
        
        return None
        
    def _convert_resource_structures(self):
        """
        Convert list-based resources to dictionary-based structures.
        
        This method ensures all resources use the proper dictionary structure expected by
        ipfs_accelerate_py.init_endpoints. It handles conversion of:
        - local_endpoints: from list to nested dictionary
        - tokenizer: from list to nested dictionary
        """
        # Convert local_endpoints from list to dictionary if needed
        if isinstance(self.resources.get("local_endpoints"), list) and self.resources["local_endpoints"]:
            local_endpoints_dict = {}
            
            # Convert list entries to dictionary structure
            for endpoint_entry in self.resources["local_endpoints"]:
                if len(endpoint_entry) >= 2:
                    model = endpoint_entry[0]
                    endpoint_type = endpoint_entry[1]
                    
                    # Create nested structure
                    if model not in local_endpoints_dict:
                        local_endpoints_dict[model] = []
                    
                    # Add endpoint entry to the model's list
                    local_endpoints_dict[model].append(endpoint_entry)
            
            # Replace list with dictionary
            self.resources["local_endpoints"] = local_endpoints_dict
            print(f"  Converted local_endpoints from list to dictionary with {len(local_endpoints_dict)} models")
        
        # Convert tokenizer from list to dictionary if needed
        if isinstance(self.resources.get("tokenizer"), list) and self.resources["tokenizer"]:
            tokenizer_dict = {}
            
            # Convert list entries to dictionary structure
            for tokenizer_entry in self.resources["tokenizer"]:
                if len(tokenizer_entry) >= 2:
                    model = tokenizer_entry[0]
                    endpoint_type = tokenizer_entry[1]
                    
                    # Create nested structure
                    if model not in tokenizer_dict:
                        tokenizer_dict[model] = {}
                    
                    # Initialize with None, will be filled during endpoint creation
                    tokenizer_dict[model][endpoint_type] = None
            
            # Replace list with dictionary
            self.resources["tokenizer"] = tokenizer_dict
            print(f"  Converted tokenizer from list to dictionary with {len(tokenizer_dict)} models")
        
        # Add endpoint_handler dictionary if it doesn't exist
        if "endpoint_handler" not in self.resources:
            self.resources["endpoint_handler"] = {}
            
        # Ensure proper structure for endpoint_handler
        for model in self.resources.get("local_endpoints", {}):
            if model not in self.resources["endpoint_handler"]:
                self.resources["endpoint_handler"][model] = {}
                
        # Create structured resources dictionary for ipfs_accelerate_py.init_endpoints
        if "structured_resources" not in self.resources:
            self.resources["structured_resources"] = {
                "tokenizer": self.resources.get("tokenizer", {}),
                "endpoint_handler": self.resources.get("endpoint_handler", {}),
                "endpoints": {
                    "local_endpoints": self.resources.get("local_endpoints", {}),
                    "api_endpoints": self.resources.get("tei_endpoints", {})
                }
            }
    
    async def get_huggingface_model_types(self):
        """
        Get a list of all Hugging Face model types.
        
        Returns:
            list: Sorted list of model types
        """
        # Initialize transformers if not already done
        if "transformers" not in dir(self):
            if "transformers" not in list(self.resources.keys()):
                try:
                    import transformers
                    self.resources["transformers"] = transformers
                    self.transformers = self.resources["transformers"]
                except Exception as e:
                    print(f"Error importing transformers: {str(e)}")
                    return []
            else:
                self.transformers = self.resources["transformers"]

        try:
            # Get all model types from the MODEL_MAPPING
            model_types = []
            for config in self.transformers.MODEL_MAPPING.keys():
                if hasattr(config, 'model_type'):
                    model_types.append(config.model_type)

            # Add model types from the AutoModel registry
            model_types.extend(list(self.transformers.MODEL_MAPPING._model_mapping.keys()))
            
            # Remove duplicates and sort
            model_types = sorted(list(set(model_types)))
            return model_types
        except Exception as e:
            print(f"Error getting Hugging Face model types: {str(e)}")
            print(traceback.format_exc())
            return []    
    
    def get_model_type(self, model_name=None, model_type=None):
        """
        Get the model type for a given model name.
        
        Args:
            model_name (str, optional): The model name. Defaults to None.
            model_type (str, optional): The model type. Defaults to None.
            
        Returns:
            str: The model type
        """
        # Initialize transformers if not already done
        if "transformers" not in dir(self):
            if "transformers" not in list(self.resources.keys()):
                try:
                    import transformers
                    self.resources["transformers"] = transformers
                    self.transformers = self.resources["transformers"]
                except Exception as e:
                    print(f"Error importing transformers: {str(e)}")
                    return None
            else:
                self.transformers = self.resources["transformers"]

        # Get model type based on model name
        if model_name is not None:
            try:
                config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                model_type = config.__class__.model_type
            except Exception as e:
                print(f"Error getting model type for {model_name}: {str(e)}")
        
        return model_type
    
    async def test(self):
        """
        Main test method that tests both hardware backend and IPFS accelerate endpoints.
        
        This method performs the following tests:
        1. Test hardware backend with the models defined in metadata
        2. Test IPFS accelerate endpoints for both CUDA and OpenVINO platforms
        
        Returns:
            dict: Dictionary containing test results for hardware backend and IPFS accelerate
        """
        test_results = {
            "timestamp": str(datetime.now()),
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Running"
        }
        
        # Test hardware backend
        try:
            print("Testing hardware backend...")
            if self.test_backend is None:
                raise ValueError("test_backend is not initialized")
                
            if not hasattr(self.test_backend, "__test__") or not callable(self.test_backend.__test__):
                raise AttributeError("test_backend.__test__ method is not defined or not callable")
                
            # Check if test_backend exists and has __test__ method
            if not hasattr(self.test_backend, "__test__"):
                raise AttributeError("test_backend does not have __test__ method")
                
            # Check the signature of __test__ method to determine how to call it
            import inspect
            sig = inspect.signature(self.test_backend.__test__)
            param_count = len(sig.parameters)
            
            print(f"TestHardwareBackend.__test__() has {param_count} parameters: {list(sig.parameters.keys())}")
            
            # Call with appropriate number of parameters, handling various signature formats
            if asyncio.iscoroutinefunction(self.test_backend.__test__):
                # Handle async method
                if param_count == 1:  # Just self
                    test_results["test_backend"] = await self.test_backend.__test__()
                elif param_count == 2:  # Could be (self, resources) or (self, metadata)
                    # Check parameter names to determine what to pass
                    param_names = list(sig.parameters.keys())
                    if 'resources' in param_names:
                        test_results["test_backend"] = await self.test_backend.__test__(self.resources)
                    elif 'metadata' in param_names:
                        test_results["test_backend"] = await self.test_backend.__test__(self.metadata)
                    else:
                        # If parameter names aren't resources or metadata, try resources as default
                        test_results["test_backend"] = await self.test_backend.__test__(self.resources)
                elif param_count == 3:  # self, resources, metadata
                    test_results["test_backend"] = await self.test_backend.__test__(self.resources, self.metadata)
                else:
                    # For any other parameter count, attempt to call without params as fallback
                    print(f"Warning: Unexpected parameter count {param_count} for test_backend.__test__")
                    print(f"Attempting to call with no parameters as fallback")
                    test_results["test_backend"] = await self.test_backend.__test__()
            else:
                # Handle sync method
                if param_count == 1:  # Just self
                    test_results["test_backend"] = self.test_backend.__test__()
                elif param_count == 2:  # Could be (self, resources) or (self, metadata)
                    # Check parameter names to determine what to pass
                    param_names = list(sig.parameters.keys())
                    if 'resources' in param_names:
                        test_results["test_backend"] = self.test_backend.__test__(self.resources)
                    elif 'metadata' in param_names:
                        test_results["test_backend"] = self.test_backend.__test__(self.metadata)
                    else:
                        # If parameter names aren't resources or metadata, try resources as default
                        test_results["test_backend"] = self.test_backend.__test__(self.resources)
                elif param_count == 3:  # self, resources, metadata
                    test_results["test_backend"] = self.test_backend.__test__(self.resources, self.metadata)
                else:
                    # For any other parameter count, attempt to call without params as fallback
                    print(f"Warning: Unexpected parameter count {param_count} for test_backend.__test__")
                    print(f"Attempting to call with no parameters as fallback")
                    test_results["test_backend"] = self.test_backend.__test__()
                
            test_results["hardware_backend_status"] = "Success"
        except Exception as e:
            error = {
                "status": "Error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            test_results["test_backend"] = error
            test_results["hardware_backend_status"] = "Failed"
            print(f"Error testing hardware backend: {str(e)}")
            print(traceback.format_exc())
        
        # Test IPFS accelerate endpoints
        try:
            print("Testing IPFS accelerate endpoints...")
            if self.ipfs_accelerate_py is None:
                raise ValueError("ipfs_accelerate_py is not initialized")
                
            results = {}
            
            # Initialize endpoints
            if not hasattr(self.ipfs_accelerate_py, "init_endpoints") or not callable(self.ipfs_accelerate_py.init_endpoints):
                raise AttributeError("ipfs_accelerate_py.init_endpoints method is not defined or not callable")
                
            print("Initializing endpoints...")
            # Get models list and validate it
            models_list = self.metadata.get('models', [])
            if not models_list:
                print("Warning: No models provided for init_endpoints")
                # Create an empty fallback structure
                ipfs_accelerate_init = {
                    "queues": {}, "queue": {}, "batch_sizes": {}, 
                    "endpoint_handler": {}, "consumer_tasks": {}, 
                    "caches": {}, "tokenizer": {},
                    "endpoints": {"local_endpoints": {}, "api_endpoints": {}, "libp2p_endpoints": {}}
                }
            else:
                # Try initialization with multi-tier fallback strategy
                try:
                    # First approach: Use properly structured resources with correct dictionary format
                    # Make sure resources are converted to proper dictionary structure
                    self._convert_resource_structures()
                    
                    # Use the structured_resources with correct nested format
                    print(f"Initializing endpoints for {len(models_list)} models using structured resources...")
                    ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints(
                        models_list, 
                        self.resources.get("structured_resources", {})
                    )
                except Exception as e:
                    print(f"Error in first init_endpoints attempt with structured resources: {str(e)}")
                    try:
                        # Second approach: Use simplified endpoint structure
                        # Create a simple endpoint dictionary with correct structure
                        simple_endpoint = {
                            "endpoints": {
                                "local_endpoints": self.resources.get("local_endpoints", {}),
                                "libp2p_endpoints": self.resources.get("libp2p_endpoints", {}),
                                "tei_endpoints": self.resources.get("tei_endpoints", {})
                            },
                            "tokenizer": self.resources.get("tokenizer", {}),
                            "endpoint_handler": self.resources.get("endpoint_handler", {})
                        }
                        print(f"Trying second approach with simple_endpoint dictionary structure...")
                        ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints(models_list, simple_endpoint)
                    except Exception as e2:
                        print(f"Error in second init_endpoints attempt: {str(e2)}")
                        try:
                            # Third approach: Create endpoint structure on-the-fly
                            # Do a fresh conversion with simplified structure
                            endpoint_resources = {}
                            
                            # Convert list-based resources to dict format where needed
                            for key, value in self.resources.items():
                                if isinstance(value, list) and key in ["local_endpoints", "tokenizer"]:
                                    # Convert list to dict for these specific resources
                                    if key == "local_endpoints":
                                        endpoints_dict = {}
                                        for entry in value:
                                            if len(entry) >= 2:
                                                model, endpoint_type = entry[0], entry[1]
                                                if model not in endpoints_dict:
                                                    endpoints_dict[model] = []
                                                endpoints_dict[model].append(entry)
                                        endpoint_resources[key] = endpoints_dict
                                    elif key == "tokenizer":
                                        tokenizers_dict = {}
                                        for entry in value:
                                            if len(entry) >= 2:
                                                model, endpoint_type = entry[0], entry[1]
                                                if model not in tokenizers_dict:
                                                    tokenizers_dict[model] = {}
                                                tokenizers_dict[model][endpoint_type] = None
                                        endpoint_resources[key] = tokenizers_dict
                                else:
                                    endpoint_resources[key] = value
                            
                            print(f"Trying third approach with on-the-fly conversion...")
                            ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints(models_list, endpoint_resources)
                        except Exception as e3:
                            print(f"Error in third init_endpoints attempt: {str(e3)}")
                            # Final fallback - create a minimal viable endpoint structure for testing
                            print("Using fallback empty endpoint structure")
                            ipfs_accelerate_init = {
                                "queues": {}, "queue": {}, "batch_sizes": {}, 
                                "endpoint_handler": {}, "consumer_tasks": {}, 
                                "caches": {}, "tokenizer": {},
                                "endpoints": {"local_endpoints": {}, "api_endpoints": {}, "libp2p_endpoints": {}}
                            }
            
            # Test each model
            model_list = self.metadata.get('models', [])
            print(f"Testing {len(model_list)} models...")
            
            for model_idx, model in enumerate(model_list):
                print(f"Testing model {model_idx+1}/{len(model_list)}: {model}")
                
                if model not in results:
                    results[model] = {
                        "status": "Running",
                        "local_endpoint": {},
                        "api_endpoint": {}
                    }
                
                # Test local endpoint (tests both CUDA and OpenVINO internally)
                try:
                    print(f"  Testing local endpoint for {model}...")
                    local_result = await self.test_local_endpoint(model)
                    results[model]["local_endpoint"] = local_result
                    
                    # Determine if test was successful
                    if isinstance(local_result, dict) and not any("error" in str(k).lower() for k in local_result.keys()):
                        results[model]["local_endpoint_status"] = "Success"
                        
                        # Try to determine implementation type
                        impl_type = "MOCK"
                        for key, value in local_result.items():
                            if isinstance(value, dict) and "implementation_type" in value:
                                if "REAL" in value["implementation_type"]:
                                    impl_type = "REAL"
                                    break
                            elif isinstance(value, str) and "REAL" in value:
                                impl_type = "REAL"
                                break
                        
                        results[model]["local_endpoint_implementation"] = impl_type
                    else:
                        results[model]["local_endpoint_status"] = "Failed"
                except Exception as e:
                    error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc()
                    }
                    results[model]["local_endpoint_error"] = error_info
                    results[model]["local_endpoint_status"] = "Failed"
                    print(f"  Error testing local endpoint for {model}: {str(e)}")
                
                # Test API endpoint
                try:
                    print(f"  Testing API endpoint for {model}...")
                    api_result = await self.test_api_endpoint(model)
                    results[model]["api_endpoint"] = api_result
                    
                    # Determine if test was successful
                    if isinstance(api_result, dict) and not any("error" in str(k).lower() for k in api_result.keys()):
                        results[model]["api_endpoint_status"] = "Success"
                        
                        # Try to determine implementation type
                        impl_type = "MOCK"
                        for key, value in api_result.items():
                            if isinstance(value, dict) and "implementation_type" in value:
                                if "REAL" in value["implementation_type"]:
                                    impl_type = "REAL"
                                    break
                            elif isinstance(value, str) and "REAL" in value:
                                impl_type = "REAL"
                                break
                        
                        results[model]["api_endpoint_implementation"] = impl_type
                    else:
                        results[model]["api_endpoint_status"] = "Failed"
                except Exception as e:
                    error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc()
                    }
                    results[model]["api_endpoint_error"] = error_info
                    results[model]["api_endpoint_status"] = "Failed"
                    print(f"  Error testing API endpoint for {model}: {str(e)}")
                    
                # Determine overall model status
                if results[model].get("local_endpoint_status") == "Success" or results[model].get("api_endpoint_status") == "Success":
                    results[model]["status"] = "Success"
                else:
                    results[model]["status"] = "Failed"
            
            # Collect success/failure counts
            success_count = sum(1 for model_results in results.values() if model_results.get("status") == "Success")
            failure_count = sum(1 for model_results in results.values() if model_results.get("status") == "Failed")
            
            # Add summary data
            results["summary"] = {
                "total_models": len(model_list),
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": f"{success_count / len(model_list) * 100:.1f}%" if model_list else "N/A"
            }
            
            test_results["ipfs_accelerate_tests"] = results
            test_results["ipfs_accelerate_status"] = "Success"
        except Exception as e:
            error = {
                "status": "Error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            test_results["ipfs_accelerate_tests"] = error
            test_results["ipfs_accelerate_status"] = "Failed"
            print(f"Error testing IPFS accelerate: {str(e)}")
            print(traceback.format_exc())

        # Set overall test status
        if (test_results.get("hardware_backend_status") == "Success" and 
            test_results.get("ipfs_accelerate_status") == "Success"):
            test_results["status"] = "Success"
        else:
            test_results["status"] = "Partial Success" if (test_results.get("hardware_backend_status") == "Success" or 
                                                           test_results.get("ipfs_accelerate_status") == "Success") else "Failed"

        return test_results

    async def test_local_endpoint(self, model, endpoint_list=None):
        """
        Test local endpoint for a model with proper error handling and resource cleanup.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if "local_endpoints" not in self.resources:
                return {"error": "Missing local_endpoints in resources"}
            if "tokenizer" not in self.resources:
                return {"error": "Missing tokenizer in resources"}
            
            # Convert resource structure if needed
            self._convert_resource_structures()
            
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "local_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "local_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("local_endpoints", {}):
                return {"error": f"Model {model} not found in local_endpoints"}
                
            # Get model endpoints from ipfs_accelerate_py
            local_endpoints_by_model = self.ipfs_accelerate_py.endpoints["local_endpoints"][model]
            
            # Check if model exists in endpoint handler and tokenizer
            if not hasattr(self.ipfs_accelerate_py, "resources") or "endpoint_handler" not in self.ipfs_accelerate_py.resources:
                return {"error": "endpoint_handler not found in ipfs_accelerate_py.resources"}
                
            if model not in self.ipfs_accelerate_py.resources.get("endpoint_handler", {}):
                return {"error": f"Model {model} not found in endpoint_handler"}
                
            if "tokenizer" not in self.ipfs_accelerate_py.resources:
                return {"error": "tokenizer not found in ipfs_accelerate_py.resources"}
                
            if model not in self.ipfs_accelerate_py.resources.get("tokenizer", {}):
                return {"error": f"Model {model} not found in tokenizer"}
                
            # Get handlers and tokenizers for this model
            endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["endpoint_handler"][model]
            tokenizers_by_model = self.ipfs_accelerate_py.resources["tokenizer"][model]
            
            # Get available endpoint types
            endpoint_types = list(endpoint_handlers_by_model.keys())
            
            # Filter endpoints based on input or default behavior
            if endpoint_list is not None:
                # Filter by specified endpoint list
                local_endpoints_by_model_by_endpoint_list = [
                    x for x in local_endpoints_by_model 
                    if ("openvino:" in json.dumps(x) or "cuda:" in json.dumps(x)) 
                    and x[1] in endpoint_list 
                    and x[1] in endpoint_types
                ]
            else:
                # Use all CUDA and OpenVINO endpoints
                local_endpoints_by_model_by_endpoint_list = [
                    x for x in local_endpoints_by_model 
                    if ("openvino:" in json.dumps(x) or "cuda:" in json.dumps(x))
                    and x[1] in endpoint_types
                ]      
            
            # If no endpoints found, return error
            if len(local_endpoints_by_model_by_endpoint_list) == 0:
                return {"status": f"No valid endpoints found for model {model}"}
            
            # Test each endpoint
            for endpoint in local_endpoints_by_model_by_endpoint_list:
                # Clean up CUDA cache before testing to prevent memory issues
                if hasattr(self, "torch") and self.torch is not None and hasattr(self.torch, "cuda") and hasattr(self.torch.cuda, "empty_cache"):
                    self.torch.cuda.empty_cache()
                    print(f"  Cleared CUDA cache before testing endpoint {endpoint}")
                
                # Add timeout handling for model operations
                start_time = time.time()
                max_test_time = 300  # 5 minutes max per endpoint test
                
                # Get model type and validate it's supported
                try:
                    model_type = self.get_model_type(model)
                    if not model_type:
                        test_results[endpoint[1]] = {"error": f"Could not determine model type for {model}"}
                        continue
                        
                    # Load supported model types
                    hf_model_types_path = os.path.join(os.path.dirname(__file__), "hf_model_types.json")
                    if not os.path.exists(hf_model_types_path):
                        test_results[endpoint[1]] = {"error": "hf_model_types.json not found"}
                        continue
                        
                    with open(hf_model_types_path, "r") as f:
                        hf_model_types = json.load(f)
                        
                    method_name = "hf_" + model_type
                    
                    # Check if model type is supported
                    if model_type not in hf_model_types:
                        test_results[endpoint[1]] = {"error": f"Model type {model_type} not supported"}
                        continue
                        
                    # Check if endpoint exists in handlers
                    if endpoint[1] not in endpoint_handlers_by_model:
                        test_results[endpoint[1]] = {"error": f"Endpoint {endpoint[1]} not found for model {model}"}
                        continue
                        
                    endpoint_handler = endpoint_handlers_by_model[endpoint[1]]
                    
                    # Import the module and test the endpoint
                    try:
                        module = __import__('worker.skillset', fromlist=[method_name])
                        this_method = getattr(module, method_name)
                        this_hf = this_method(self.resources, self.metadata)
                        
                        # Check if test method is async and add timeout protection
                        if asyncio.iscoroutinefunction(this_hf.__test__):
                            try:
                                # Use asyncio.wait_for to add timeout protection
                                test = await asyncio.wait_for(
                                    this_hf.__test__(
                                        model, 
                                        endpoint_handlers_by_model[endpoint[1]], 
                                        endpoint[1], 
                                        tokenizers_by_model[endpoint[1]]
                                    ),
                                    timeout=max_test_time
                                )
                            except asyncio.TimeoutError:
                                test = {
                                    "error": f"Test timed out after {max_test_time} seconds",
                                    "timeout": True,
                                    "time_elapsed": time.time() - start_time
                                }
                        else:
                            # For sync methods, we still track time but don't have a clean timeout mechanism
                            # The global timeout in the Bash tool will still catch it if it runs too long
                            test = this_hf.__test__(
                                model, 
                                endpoint_handlers_by_model[endpoint[1]], 
                                endpoint[1], 
                                tokenizers_by_model[endpoint[1]]
                            )
                            # Check if we're past the timeout limit
                            if time.time() - start_time > max_test_time:
                                test = {
                                    "error": f"Test execution exceeded time limit of {max_test_time} seconds",
                                    "timeout": True,
                                    "time_elapsed": time.time() - start_time
                                }
                            
                        test_results[endpoint[1]] = test
                        
                        # Clean up resources
                        del this_hf
                        del this_method
                        del module
                        del test
                        
                        # Explicitly clean up CUDA memory
                        if hasattr(self, "torch") and self.torch is not None and hasattr(self.torch, "cuda") and hasattr(self.torch.cuda, "empty_cache"):
                            self.torch.cuda.empty_cache()
                            print(f"  Cleared CUDA cache after testing endpoint {endpoint[1]}")
                    except Exception as e:
                        test_results[endpoint[1]] = {
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }
                except Exception as e:
                    test_results[endpoint[1]] = {
                        "error": f"Error processing endpoint {endpoint[1]}: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_local_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
    
    async def test_api_endpoint(self, model, endpoint_list=None):
        """
        Test API endpoints (TEI, OVMS) for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if not hasattr(self.ipfs_accelerate_py, "resources") or "tei_endpoints" not in self.ipfs_accelerate_py.resources:
                return {"error": "Missing tei_endpoints in resources"}
                
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "tei_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "tei_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("tei_endpoints", {}):
                return {"error": f"Model {model} not found in tei_endpoints"}
                
            # Check if model exists in endpoint handlers
            if model not in self.ipfs_accelerate_py.resources.get("tei_endpoints", {}):
                return {"error": f"Model {model} not found in tei_endpoint handlers"}
                
            local_endpoints = self.ipfs_accelerate_py.resources["tei_endpoints"]
            local_endpoints_types = [x[1] for x in local_endpoints]
            local_endpoints_by_model = self.ipfs_accelerate_py.endpoints["tei_endpoints"][model]
            endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["tei_endpoints"][model]
            
            # Get list of valid endpoints for the model
            local_endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
            local_endpoints_by_model_by_endpoint = [
                x for x in local_endpoints_by_model_by_endpoint 
                if x in local_endpoints_by_model 
                if x in local_endpoints_types
            ]
            
            # Filter by provided endpoint list if specified
            if endpoint_list is not None:
                local_endpoints_by_model_by_endpoint = [
                    x for x in local_endpoints_by_model_by_endpoint 
                    if x in endpoint_list
                ]
            
            # If no endpoints found, return error
            if len(local_endpoints_by_model_by_endpoint) == 0:
                return {"status": f"No valid API endpoints found for model {model}"}
            
            # Test each endpoint
            for endpoint in local_endpoints_by_model_by_endpoint:
                try:
                    endpoint_handler = endpoint_handlers_by_model[endpoint]
                    implementation_type = "Unknown"
                    
                    # Try async call first, then fallback to sync
                    try:
                        # Determine if handler is async
                        if asyncio.iscoroutinefunction(endpoint_handler):
                            test = await endpoint_handler("hello world")
                            implementation_type = "REAL (async)"
                        else:
                            test = endpoint_handler("hello world")
                            implementation_type = "REAL (sync)"
                            
                        # Record successful test results
                        test_results[endpoint] = {
                            "status": "Success",
                            "implementation_type": implementation_type,
                            "result": test
                        }
                    except Exception as e:
                        # If async call fails, try sync call as fallback
                        try:
                            if asyncio.iscoroutinefunction(endpoint_handler):
                                # Already tried async and it failed
                                raise e
                            else:
                                test = endpoint_handler("hello world")
                                implementation_type = "REAL (sync fallback)"
                                test_results[endpoint] = {
                                    "status": "Success (with fallback)",
                                    "implementation_type": implementation_type,
                                    "result": test
                                }
                        except Exception as fallback_error:
                            # Both async and sync approaches failed
                            test_results[endpoint] = {
                                "status": "Error",
                                "error": str(fallback_error),
                                "traceback": traceback.format_exc()
                            }
                except Exception as e:
                    test_results[endpoint] = {
                        "status": "Error",
                        "error": f"Error processing endpoint {endpoint}: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_api_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
    
    async def test_libp2p_endpoint(self, model, endpoint_list=None):
        """
        Test libp2p endpoint for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if not hasattr(self.ipfs_accelerate_py, "resources") or "libp2p_endpoints" not in self.ipfs_accelerate_py.resources:
                return {"error": "Missing libp2p_endpoints in resources"}
                
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "libp2p_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "libp2p_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("libp2p_endpoints", {}):
                return {"error": f"Model {model} not found in libp2p_endpoints"}
                
            # Check if model exists in endpoint handlers
            if model not in self.ipfs_accelerate_py.resources.get("libp2p_endpoints", {}):
                return {"error": f"Model {model} not found in libp2p_endpoint handlers"}
                
            libp2p_endpoints_by_model = self.ipfs_accelerate_py.endpoints["libp2p_endpoints"][model]
            endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["libp2p_endpoints"][model]
            
            # Get list of valid endpoints for the model
            local_endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
            
            # Filter by provided endpoint list if specified
            if endpoint_list is not None:
                local_endpoints_by_model_by_endpoint = [
                    x for x in local_endpoints_by_model_by_endpoint 
                    if x in endpoint_list
                ]
            
            # If no endpoints found, return error
            if len(local_endpoints_by_model_by_endpoint) == 0:
                return {"status": f"No valid libp2p endpoints found for model {model}"}
            
            # Test each endpoint
            for endpoint in local_endpoints_by_model_by_endpoint:
                try:
                    endpoint_handler = endpoint_handlers_by_model[endpoint]
                    implementation_type = "Unknown"
                    
                    # Try async call first, then fallback to sync
                    try:
                        # Determine if handler is async
                        if asyncio.iscoroutinefunction(endpoint_handler):
                            test = await endpoint_handler("hello world")
                            implementation_type = "REAL (async)"
                        else:
                            test = endpoint_handler("hello world")
                            implementation_type = "REAL (sync)"
                            
                        # Record successful test results
                        test_results[endpoint] = {
                            "status": "Success",
                            "implementation_type": implementation_type,
                            "result": test
                        }
                    except Exception as e:
                        # If async call fails, try sync call as fallback
                        try:
                            if asyncio.iscoroutinefunction(endpoint_handler):
                                # Already tried async and it failed
                                raise e
                            else:
                                test = endpoint_handler("hello world")
                                implementation_type = "REAL (sync fallback)"
                                test_results[endpoint] = {
                                    "status": "Success (with fallback)",
                                    "implementation_type": implementation_type,
                                    "result": test
                                }
                        except Exception as fallback_error:
                            # Both async and sync approaches failed
                            test_results[endpoint] = {
                                "status": "Error",
                                "error": str(fallback_error),
                                "traceback": traceback.format_exc()
                            }
                except Exception as e:
                    test_results[endpoint] = {
                        "status": "Error",
                        "error": f"Error processing endpoint {endpoint}: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_libp2p_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
    
    async def test_ovms_endpoint(self, model, endpoint_list=None):
        """
        Test OpenVINO Model Server (OVMS) endpoints for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if not hasattr(self.ipfs_accelerate_py, "resources") or "ovms_endpoints" not in self.ipfs_accelerate_py.resources:
                return {"error": "Missing ovms_endpoints in resources"}
                
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "ovms_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "ovms_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("ovms_endpoints", {}):
                return {"error": f"Model {model} not found in ovms_endpoints"}
                
            # Check if model exists in endpoint handlers
            if model not in self.ipfs_accelerate_py.resources.get("ovms_endpoints", {}):
                return {"error": f"Model {model} not found in ovms_endpoint handlers"}
                
            ovms_endpoints_by_model = self.ipfs_accelerate_py.endpoints["ovms_endpoints"][model]
            endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["ovms_endpoints"][model]
            
            # Get list of valid endpoints for the model
            local_endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
            
            # Filter by provided endpoint list if specified
            if endpoint_list is not None:
                local_endpoints_by_model_by_endpoint = [
                    x for x in local_endpoints_by_model_by_endpoint 
                    if x in endpoint_list
                ]
            
            # If no endpoints found, return error
            if len(local_endpoints_by_model_by_endpoint) == 0:
                return {"status": f"No valid OVMS endpoints found for model {model}"}
            
            # Test each endpoint
            for endpoint in local_endpoints_by_model_by_endpoint:
                try:
                    endpoint_handler = endpoint_handlers_by_model[endpoint]
                    implementation_type = "Unknown"
                    
                    # Try async call first, then fallback to sync
                    # Since OVMS typically requires structured input, we'll create a simple tensor
                    try:
                        # Create a sample input (assuming a simple input tensor)
                        import numpy as np
                        sample_input = np.ones((1, 3, 224, 224), dtype=np.float32)  # Simple image-like tensor
                        
                        # Determine if handler is async
                        if asyncio.iscoroutinefunction(endpoint_handler):
                            test = await endpoint_handler(sample_input)
                            implementation_type = "REAL (async)"
                        else:
                            test = endpoint_handler(sample_input)
                            implementation_type = "REAL (sync)"
                            
                        # Record successful test results
                        test_results[endpoint] = {
                            "status": "Success",
                            "implementation_type": implementation_type,
                            "result": str(test)  # Convert numpy arrays to strings for JSON serialization
                        }
                    except Exception as e:
                        # If async call fails, try sync call as fallback
                        try:
                            if asyncio.iscoroutinefunction(endpoint_handler):
                                # Already tried async and it failed
                                raise e
                            else:
                                # Try with string input instead as a last resort
                                test = endpoint_handler("hello world")
                                implementation_type = "REAL (sync fallback)"
                                test_results[endpoint] = {
                                    "status": "Success (with fallback)",
                                    "implementation_type": implementation_type,
                                    "result": str(test)
                                }
                        except Exception as fallback_error:
                            # Both async and sync approaches failed
                            test_results[endpoint] = {
                                "status": "Error",
                                "error": str(fallback_error),
                                "traceback": traceback.format_exc()
                            }
                except Exception as e:
                    test_results[endpoint] = {
                        "status": "Error",
                        "error": f"Error processing endpoint {endpoint}: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_ovms_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
    
    async def test_tei_endpoint(self, model, endpoint_list=None):
        """
        Test Text Embedding Inference (TEI) endpoints for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if not hasattr(self.ipfs_accelerate_py, "resources") or "tei_endpoints" not in self.ipfs_accelerate_py.resources:
                return {"error": "Missing tei_endpoints in resources"}
                
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "tei_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "tei_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("tei_endpoints", {}):
                return {"error": f"Model {model} not found in tei_endpoints"}
                
            # Check if model exists in endpoint handlers
            if model not in self.ipfs_accelerate_py.resources.get("tei_endpoints", {}):
                return {"error": f"Model {model} not found in tei_endpoint handlers"}
                
            local_endpoints = self.ipfs_accelerate_py.resources["tei_endpoints"]
            local_endpoints_types = [x[1] if isinstance(x, list) and len(x) > 1 else None for x in local_endpoints]
            local_endpoints_by_model = self.ipfs_accelerate_py.endpoints["tei_endpoints"][model]
            endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["tei_endpoints"][model]
            
            # Get list of valid endpoints for the model
            local_endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
            
            # Filter by provided endpoint list if specified
            if endpoint_list is not None:
                local_endpoints_by_model_by_endpoint = [
                    x for x in local_endpoints_by_model_by_endpoint 
                    if x in endpoint_list
                ]
            
            # If no endpoints found, return error
            if len(local_endpoints_by_model_by_endpoint) == 0:
                return {"status": f"No valid TEI endpoints found for model {model}"}
            
            # Test each endpoint
            for endpoint in local_endpoints_by_model_by_endpoint:
                try:
                    endpoint_handler = endpoint_handlers_by_model[endpoint]
                    implementation_type = "Unknown"
                    
                    # For TEI endpoints, we'll use a text sample
                    # Text Embedding Inference API typically expects a list of strings
                    try:
                        # Create a sample input
                        sample_input = ["This is a sample text for embedding generation."]
                        
                        # Determine if handler is async
                        if asyncio.iscoroutinefunction(endpoint_handler):
                            test = await endpoint_handler(sample_input)
                            implementation_type = "REAL (async)"
                        else:
                            test = endpoint_handler(sample_input)
                            implementation_type = "REAL (sync)"
                            
                        # Record successful test results, ensuring serializable format
                        if hasattr(test, "tolist"):  # Handle numpy arrays
                            result_data = test.tolist()
                        elif isinstance(test, list) and hasattr(test[0], "tolist"):  # List of numpy arrays
                            result_data = [item.tolist() if hasattr(item, "tolist") else item for item in test]
                        else:
                            result_data = test
                            
                        test_results[endpoint] = {
                            "status": "Success",
                            "implementation_type": implementation_type,
                            "result": {
                                "shape": str(test.shape) if hasattr(test, "shape") else "unknown",
                                "type": str(type(test)),
                                "sample": str(result_data)[:100] + "..." if len(str(result_data)) > 100 else str(result_data)
                            }
                        }
                    except Exception as e:
                        # If async call fails, try sync call as fallback
                        try:
                            if asyncio.iscoroutinefunction(endpoint_handler):
                                # Already tried async and it failed
                                raise e
                            else:
                                # Try with single string input instead
                                test = endpoint_handler("hello world")
                                implementation_type = "REAL (sync fallback)"
                                # Format result for JSON serialization
                                if hasattr(test, "tolist"):  # Handle numpy arrays
                                    result_data = test.tolist()
                                elif isinstance(test, list) and hasattr(test[0], "tolist"):  # List of numpy arrays
                                    result_data = [item.tolist() if hasattr(item, "tolist") else item for item in test]
                                else:
                                    result_data = test
                                    
                                test_results[endpoint] = {
                                    "status": "Success (with fallback)",
                                    "implementation_type": implementation_type,
                                    "result": {
                                        "shape": str(test.shape) if hasattr(test, "shape") else "unknown",
                                        "type": str(type(test)),
                                        "sample": str(result_data)[:100] + "..." if len(str(result_data)) > 100 else str(result_data)
                                    }
                                }
                        except Exception as fallback_error:
                            # Both async and sync approaches failed
                            test_results[endpoint] = {
                                "status": "Error",
                                "error": str(fallback_error),
                                "traceback": traceback.format_exc()
                            }
                except Exception as e:
                    test_results[endpoint] = {
                        "status": "Error",
                        "error": f"Error processing endpoint {endpoint}: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_tei_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
    
    async def test_endpoint(self, model, endpoint=None):
        """
        Test a specific endpoint for a model.
        
        Args:
            model (str): The model to test
            endpoint (str, optional): The endpoint to test. Defaults to None.
            
        Returns:
            dict: Test results for the endpoint
        """
        test_results = {}
        
        try:
            # Test different endpoint types
            try:    
                test_results["local_endpoint"] = await self.test_local_endpoint(model, endpoint)
            except Exception as e:
                test_results["local_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
            try:
                test_results["libp2p_endpoint"] = await self.test_libp2p_endpoint(model, endpoint)
            except Exception as e:
                test_results["libp2p_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
            try:
                test_results["api_endpoint"] = await self.test_api_endpoint(model, endpoint)
            except Exception as e:
                test_results["api_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
            try:
                test_results["ovms_endpoint"] = await self.test_ovms_endpoint(model, endpoint)
            except Exception as e:
                test_results["ovms_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
            try:
                test_results["tei_endpoint"] = await self.test_tei_endpoint(model, endpoint)
            except Exception as e:
                test_results["tei_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
            # WebNN endpoint not implemented yet
            test_results["webnn_endpoint"] = {"status": "Not implemented"}
        except Exception as e:
            test_results["global_error"] = {
                "status": "Error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
        return test_results
        
    async def test_endpoints(self, models, endpoint_handler_object=None):
        """
        Test all available endpoints for each model.
        
        Args:
            models (list): List of models to test
            endpoint_handler_object (object, optional): Endpoint handler object. Defaults to None.
            
        Returns:
            dict: Test results for all endpoints
        """
        test_results = {}
        
        # Track overall stats
        test_stats = {
            "total_models": len(models),
            "successful_tests": 0,
            "failed_tests": 0,
            "models_tested": []
        }
        
        # Test each model
        for model_idx, model in enumerate(models):
            print(f"Testing endpoints for model {model_idx+1}/{len(models)}: {model}")
            
            if model not in test_results:
                test_results[model] = {}
                
            model_success = True
            test_stats["models_tested"].append(model)
            
            # Test local endpoint (CUDA/OpenVINO)
            try: 
                print(f"  Testing local endpoint...")
                local_result = await self.test_local_endpoint(model)
                test_results[model]["local_endpoint"] = local_result
                if isinstance(local_result, Exception) or (isinstance(local_result, dict) and any("Error" in str(v) for v in local_result.values())):
                    model_success = False
            except Exception as e:
                test_results[model]["local_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                model_success = False
                print(f"  Error testing local endpoint for {model}: {str(e)}")

            # Test WebNN endpoint (currently not implemented)
            try:
                test_results[model]["webnn_endpoint"] = {"status": "Not implemented"}
            except Exception as e:
                test_results[model]["webnn_endpoint"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                print(f"  Error testing WebNN endpoint for {model}: {str(e)}")

            # Update test stats
            if model_success:
                test_stats["successful_tests"] += 1
            else:
                test_stats["failed_tests"] += 1

        # Add endpoint handler resources if provided
        if endpoint_handler_object:
            try:
                test_results["endpoint_handler_resources"] = endpoint_handler_object
            except Exception as e:
                test_results["endpoint_handler_resources"] = {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
        # Add test stats to results
        test_results["test_stats"] = test_stats
                
        return test_results
    
    async def test_ipfs_accelerate(self):
        """
        Test IPFS accelerate endpoints for all models.
        
        Returns:
            dict: Test results for IPFS accelerate endpoints
        """
        test_results = {}
        
        try:
            print("Testing IPFS accelerate...")
            
            # Use the existing ipfs_accelerate_py instance
            if self.ipfs_accelerate_py is None:
                raise ValueError("ipfs_accelerate_py is not initialized")
                
            print("Initializing endpoints...")
            # Pass models explicitly when calling init_endpoints to avoid unbound 'model' error
            endpoint_resources = {}
            for key in self.resources:
                endpoint_resources[key] = self.resources[key]
                
            # Make resources a dict-like structure to avoid type issues
            if isinstance(endpoint_resources, list):
                endpoint_resources = {i: v for i, v in enumerate(endpoint_resources)}
                
            # Get models list and validate it
            models_list = self.metadata.get('models', [])
            if not models_list:
                print("Warning: No models provided for init_endpoints")
                # Create an empty fallback structure
                ipfs_accelerate_init = {
                    "queues": {}, "queue": {}, "batch_sizes": {}, 
                    "endpoint_handler": {}, "consumer_tasks": {}, 
                    "caches": {}, "tokenizer": {},
                    "endpoints": {"local_endpoints": {}, "api_endpoints": {}, "libp2p_endpoints": {}}
                }
            else:
                # Try the initialization with different approaches
                try:
                    print(f"Initializing endpoints for {len(models_list)} models...")
                    ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints(models_list, endpoint_resources)
                except Exception as e:
                    print(f"Error in first init_endpoints attempt: {str(e)}")
                    try:
                        # Alternative approach - creating a simple endpoint structure with actual resource data
                        simple_endpoint = {
                            "local_endpoints": self.resources.get("local_endpoints", []),
                            "libp2p_endpoints": self.resources.get("libp2p_endpoints", []),
                            "tei_endpoints": self.resources.get("tei_endpoints", [])
                        }
                        print(f"Trying second approach with simple_endpoint structure")
                        ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints(models_list, simple_endpoint)
                    except Exception as e2:
                        print(f"Error in second init_endpoints attempt: {str(e2)}")
                        # Final fallback - create a minimal viable endpoint structure
                        print("Using fallback empty endpoint structure")
                        ipfs_accelerate_init = {
                            "queues": {}, "queue": {}, "batch_sizes": {}, 
                            "endpoint_handler": {}, "consumer_tasks": {}, 
                            "caches": {}, "tokenizer": {},
                            "endpoints": {"local_endpoints": {}, "api_endpoints": {}, "libp2p_endpoints": {}}
                        }
            
            # Test endpoints for all models
            model_list = self.metadata.get('models', [])
            print(f"Testing endpoints for {len(model_list)} models...")
            
            test_endpoints = await self.test_endpoints(model_list, ipfs_accelerate_init)
            test_results["test_endpoints"] = test_endpoints
            test_results["status"] = "Success"
            
            return test_results
        except Exception as e:
            error = {
                "status": "Error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            test_results["error"] = error
            test_results["status"] = "Failed"
            print(f"Error testing IPFS accelerate endpoints: {str(e)}")
            print(traceback.format_exc())
            
            return test_results
    
    async def __test__(self, resources=None, metadata=None):
        """
        Main test entry point that runs all tests and collects results.
        
        This method follows the 4-phase testing approach defined in the class documentation:
        - Phase 1: Test with models defined in global metadata
        - Phase 2: Test with models from mapped_models.json
        - Phase 3: Collect and analyze test results
        - Phase 4: Generate test reports
        
        Args:
            resources (dict, optional): Dictionary of resources. Defaults to None.
            metadata (dict, optional): Dictionary of metadata. Defaults to None.
            
        Returns:
            dict: Comprehensive test results
        """
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Starting test suite at {start_time}")
        
        # Initialize resources if not provided
        if resources is not None:
            self.resources = resources
        if metadata is not None:
            self.metadata = metadata
            
        # Ensure required resource dictionaries exist
        required_resource_keys = [
            "local_endpoints", "tei_endpoints", "libp2p_endpoints", 
            "openvino_endpoints", "tokenizer", "endpoint_handler"
        ]
        
        print("Initializing resources...")
        for key in required_resource_keys:
            if key not in self.resources:
                self.resources[key] = {}
                print(f"  Created empty {key} dictionary")
            
        # Load mapped models from JSON
        mapped_models = {}
        mapped_models_values = []
        mapped_models_path = os.path.join(os.path.dirname(__file__), "mapped_models.json")
        
        print("Loading mapped models...")
        if os.path.exists(mapped_models_path):
            try:
                with open(mapped_models_path, "r") as f:
                    mapped_models = json.load(f)
                mapped_models_values = list(mapped_models.values())
                print(f"  Loaded {len(mapped_models)} model mappings")
                
                # Update metadata with models from mapped_models.json
                if "models" not in self.metadata or not self.metadata["models"]:
                    self.metadata["models"] = mapped_models_values
                    print("  Updated self.metadata with mapped models")
            except Exception as e:
                print(f"Error loading mapped_models.json: {str(e)}")
                print(traceback.format_exc())
        else:
            print("  Warning: mapped_models.json not found")
        
        # Initialize the transformers module if available
        if "transformers" not in self.resources:
            try:
                import transformers
                self.resources["transformers"] = transformers
                print("  Added transformers module to resources")
            except ImportError:
                from unittest.mock import MagicMock
                self.resources["transformers"] = MagicMock()
                print("  Added mock transformers module to resources")
        
        # Setup endpoints for each model and hardware platform
        endpoint_types = ["cuda:0", "openvino:0", "cpu:0"]
        endpoint_count = 0
        
        # Handle both list and dictionary resource structures
        if isinstance(self.resources["local_endpoints"], list):
            # Convert list to empty dictionary to prepare for structured data
            self.resources["local_endpoints"] = {}
            self.resources["tokenizer"] = {}
            
        # Add required resource dictionaries for init_endpoints
        required_queue_keys = ["queue", "queues", "batch_sizes", "consumer_tasks", "caches"]
        for key in required_queue_keys:
            if key not in self.resources:
                self.resources[key] = {}
            
        print("Setting up endpoints for each model...")
        if "models" in self.metadata and self.metadata["models"]:
            if "endpoints" not in self.resources:
                self.resources["endpoints"] = {}
                
            if "local_endpoints" not in self.resources["endpoints"]:
                self.resources["endpoints"]["local_endpoints"] = {}
                
            # Make sure we have a direct local_endpoints reference for backward compatibility
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or not self.ipfs_accelerate_py.endpoints:
                self.ipfs_accelerate_py.endpoints = {
                    "local_endpoints": {},
                    "api_endpoints": {},
                    "libp2p_endpoints": {}
                }
            
            for model in self.metadata["models"]:
                # Create model entry in endpoints dictionary
                if model not in self.resources["local_endpoints"]:
                    self.resources["local_endpoints"][model] = []
                
                # Create model entry inside endpoints structure
                if model not in self.resources["endpoints"]["local_endpoints"]:
                    self.resources["endpoints"]["local_endpoints"][model] = []
                
                # Also initialize in the ipfs_accelerate_py.endpoints structure
                if model not in self.ipfs_accelerate_py.endpoints["local_endpoints"]:
                    self.ipfs_accelerate_py.endpoints["local_endpoints"][model] = []
                
                # Create model entry in tokenizer and endpoint_handler dictionaries
                if model not in self.resources["tokenizer"]:
                    self.resources["tokenizer"][model] = {}
                    
                if model not in self.resources["endpoint_handler"]:
                    self.resources["endpoint_handler"][model] = {}
                
                # Make sure ipfs_accelerate_py has the same entries in its resources
                if not hasattr(self.ipfs_accelerate_py, "resources"):
                    self.ipfs_accelerate_py.resources = {}
                
                for resource_key in ["tokenizer", "endpoint_handler", "queue", "queues", "batch_sizes"]:
                    if resource_key not in self.ipfs_accelerate_py.resources:
                        self.ipfs_accelerate_py.resources[resource_key] = {}
                    
                    if model not in self.ipfs_accelerate_py.resources[resource_key]:
                        self.ipfs_accelerate_py.resources[resource_key][model] = {} if resource_key != "queue" else asyncio.Queue(128)
                
                for endpoint in endpoint_types:
                    # Create endpoint info (model, endpoint, context_length)
                    endpoint_info = [model, endpoint, 32768]
                    
                    # Avoid duplicate entries in resources["local_endpoints"]
                    if endpoint_info not in self.resources["local_endpoints"][model]:
                        self.resources["local_endpoints"][model].append(endpoint_info)
                    
                    # Also add to resources["endpoints"]["local_endpoints"]
                    if endpoint_info not in self.resources["endpoints"]["local_endpoints"][model]:
                        self.resources["endpoints"]["local_endpoints"][model].append(endpoint_info)
                    
                    # Add to ipfs_accelerate_py.endpoints["local_endpoints"]
                    if endpoint_info not in self.ipfs_accelerate_py.endpoints["local_endpoints"][model]:
                        self.ipfs_accelerate_py.endpoints["local_endpoints"][model].append(endpoint_info)
                    
                    # Add tokenizer entry for this model-endpoint combination
                    if endpoint not in self.resources["tokenizer"][model]:
                        self.resources["tokenizer"][model][endpoint] = None
                    
                    # Add tokenizer to ipfs_accelerate_py.resources
                    if endpoint not in self.ipfs_accelerate_py.resources["tokenizer"][model]:
                        self.ipfs_accelerate_py.resources["tokenizer"][model][endpoint] = None
                    
                    # Add endpoint handler entry
                    if endpoint not in self.resources["endpoint_handler"][model]:
                        self.resources["endpoint_handler"][model][endpoint] = None
                    
                    # Add endpoint handler to ipfs_accelerate_py.resources
                    if endpoint not in self.ipfs_accelerate_py.resources["endpoint_handler"][model]:
                        self.ipfs_accelerate_py.resources["endpoint_handler"][model][endpoint] = None
                        
                        # Create a mock handler directly in ipfs_accelerate_py
                        if hasattr(self.ipfs_accelerate_py, "_create_mock_handler"):
                            try:
                                self.ipfs_accelerate_py._create_mock_handler(model, endpoint)
                                print(f"  Created mock handler for {model} with {endpoint}")
                            except Exception as e:
                                print(f"  Error creating mock handler: {str(e)}")
                    
                    # Track for reporting
                    endpoint_count += 1
            
            print(f"  Added {endpoint_count} endpoints for {len(self.metadata['models'])} models")
            
            # Create properly structured resources dictionary with all required keys
            self.resources["structured_resources"] = {
                "tokenizer": self.resources["tokenizer"],
                "endpoint_handler": self.resources["endpoint_handler"],
                "queue": self.resources.get("queue", {}),  # Add queue dictionary - required by init_endpoints
                "queues": self.resources.get("queues", {}), # Add queues dictionary
                "batch_sizes": self.resources.get("batch_sizes", {}), # Add batch_sizes dictionary
                "consumer_tasks": self.resources.get("consumer_tasks", {}), # Add consumer_tasks dictionary
                "caches": self.resources.get("caches", {}), # Add caches dictionary
                "endpoints": {
                    "local_endpoints": self.resources["local_endpoints"],
                    "api_endpoints": self.resources.get("tei_endpoints", {}),
                    "libp2p_endpoints": self.resources.get("libp2p_endpoints", {})
                }
            }
            
            # Debugging: Print structure information
            print(f"  Endpoints structure: Dictionary with {len(self.resources['local_endpoints'])} models")
            print(f"  Tokenizer structure: Dictionary with {len(self.resources['tokenizer'])} models")
            print(f"  Endpoint handler structure: Dictionary with {len(self.resources['endpoint_handler'])} models")
        else:
            print("  Warning: No models found in metadata")
        
        # Prepare test results structure
        test_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_date": start_time,
                "status": "Running",
                "test_phases_completed": 0,
                "total_test_phases": 4 if mapped_models else 2
            },
            "models_tested": {
                "global_models": len(self.metadata.get("models", [])),
                "mapped_models": len(mapped_models)
            },
            "configuration": {
                "endpoint_types": endpoint_types,
                "model_count": len(self.metadata.get("models", [])),
                "endpoints_per_model": len(endpoint_types)
            }
        }
        
        # Run the tests in phases
        try:
            # Phase 1: Test with models in global metadata
            print("\n=== PHASE 1: Testing with global metadata models ===")
            if not self.metadata.get("models"):
                print("No models in global metadata, skipping Phase 1")
                test_results["phase1_global_models"] = {"status": "Skipped", "reason": "No models in global metadata"}
            else:
                # Limit to first 2 models to avoid timeouts
                original_models = self.metadata.get('models', []).copy()
                self.metadata["models"] = self.metadata.get("models", [])[:2]
                print(f"Testing {len(self.metadata.get('models', []))} models from global metadata (limited to first 2 for speed)")
                test_results["phase1_global_models"] = await self.test()
                self.metadata["models"] = original_models  # Restore the original list
                test_results["metadata"]["test_phases_completed"] += 1
                print(f"Phase 1 completed with status: {test_results['phase1_global_models'].get('status', 'Unknown')}")
            
            # Phase 2: Test with mapped models from JSON file
            print("\n=== PHASE 2: Testing with mapped models ===")
            if not mapped_models:
                print("No mapped models found, skipping Phase 2")
                test_results["phase2_mapped_models"] = {"status": "Skipped", "reason": "No mapped models found"}
            else:
                # Save original models list
                original_models = self.metadata.get("models", [])
                
                # Update metadata to use a limited subset of mapped models (first 2)
                limited_models = mapped_models_values[:2]
                print(f"Testing {len(limited_models)} models from mapped_models.json (limited to first 2 for speed)")
                self.metadata["models"] = limited_models
                test_results["phase2_mapped_models"] = await self.test()
                
                # Restore original models list
                self.metadata["models"] = original_models
                test_results["metadata"]["test_phases_completed"] += 1
                print(f"Phase 2 completed with status: {test_results['phase2_mapped_models'].get('status', 'Unknown')}")
            
            # Phase 3: Analyze test results
            print("\n=== PHASE 3: Analyzing test results ===")
            analysis = {
                "model_coverage": {},
                "platform_performance": {
                    "cuda": {"success": 0, "failure": 0, "success_rate": "0%"},
                    "openvino": {"success": 0, "failure": 0, "success_rate": "0%"}
                },
                "implementation_types": {
                    "REAL": 0,
                    "MOCK": 0,
                    "Unknown": 0
                }
            }
            
            # Analyze Phase 1 results
            if "phase1_global_models" in test_results and "ipfs_accelerate_tests" in test_results["phase1_global_models"]:
                phase1_results = test_results["phase1_global_models"]["ipfs_accelerate_tests"]
                if isinstance(phase1_results, dict) and "summary" in phase1_results:
                    # Process model results
                    for model, model_results in phase1_results.items():
                        if model == "summary":
                            continue
                            
                        # Track model success/failure
                        if model not in analysis["model_coverage"]:
                            analysis["model_coverage"][model] = {"status": model_results.get("status", "Unknown")}
                            
                        # Track platform performance
                        for platform in ["cuda", "openvino"]:
                            if platform in model_results and "status" in model_results[platform]:
                                if model_results[platform]["status"] == "Success":
                                    analysis["platform_performance"][platform]["success"] += 1
                                else:
                                    analysis["platform_performance"][platform]["failure"] += 1
                                    
                        # Track implementation types
                        for platform in ["cuda", "openvino"]:
                            if platform in model_results and "implementation_type" in model_results[platform]:
                                impl_type = model_results[platform]["implementation_type"]
                                if "REAL" in impl_type:
                                    analysis["implementation_types"]["REAL"] += 1
                                elif "MOCK" in impl_type:
                                    analysis["implementation_types"]["MOCK"] += 1
                                else:
                                    analysis["implementation_types"]["Unknown"] += 1
            
            # Calculate success rates
            for platform in ["cuda", "openvino"]:
                platform_data = analysis["platform_performance"][platform]
                total = platform_data["success"] + platform_data["failure"]
                if total > 0:
                    platform_data["success_rate"] = f"{(platform_data['success'] / total) * 100:.1f}%"
            
            # Add analysis to test results
            test_results["phase3_analysis"] = analysis
            test_results["metadata"]["test_phases_completed"] += 1
            print("Analysis completed")
            
            # Phase 4: Generate test report
            print("\n=== PHASE 4: Generating test report ===")
            
            # Create test report summary
            report = {
                "summary": {
                    "test_date": start_time,
                    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "models_tested": test_results["models_tested"],
                    "phases_completed": test_results["metadata"]["test_phases_completed"],
                    "platform_performance": analysis["platform_performance"],
                    "implementation_breakdown": analysis["implementation_types"]
                },
                "recommendations": []
            }
            
            # Add recommendations based on analysis
            if analysis["implementation_types"]["MOCK"] > analysis["implementation_types"]["REAL"]:
                report["recommendations"].append("Focus on implementing more REAL implementations to replace MOCK implementations")
                
            if analysis["platform_performance"]["cuda"]["success_rate"] < "50%":
                report["recommendations"].append("Improve CUDA platform support for better performance")
                
            if analysis["platform_performance"]["openvino"]["success_rate"] < "50%":
                report["recommendations"].append("Improve OpenVINO platform support for better compatibility")
            
            # Add report to test results
            test_results["phase4_report"] = report
            test_results["metadata"]["test_phases_completed"] += 1
            print("Test report generated")
            
            # Update overall test status
            if (test_results["metadata"]["test_phases_completed"] == 
                test_results["metadata"]["total_test_phases"]):
                test_results["metadata"]["status"] = "Success"
            else:
                test_results["metadata"]["status"] = "Partial Success"
                
        except Exception as e:
            error = {
                "status": "Error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            test_results["error"] = error
            test_results["metadata"]["status"] = "Failed"
            print(f"Error running tests: {str(e)}")
            print(traceback.format_exc())
        
        # Save test results to file
        print("\nSaving test results...")
        this_file = os.path.abspath(sys.modules[__name__].__file__)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_log = os.path.join(os.path.dirname(this_file), f"test_results_{timestamp}.json")
        
        try:
            with open(test_log, "w") as f:
                json.dump(test_results, f, indent=4)
            print(f"Saved detailed test results to {test_log}")
                
            # Also save to standard test_results.json for backward compatibility
            standard_log = os.path.join(os.path.dirname(this_file), "test_results.json")
            with open(standard_log, "w") as f:
                json.dump(test_results, f, indent=4)
            print(f"Saved test results to {standard_log}")
        except Exception as e:
            print(f"Error saving test results: {str(e)}")
        
        print(f"\nTest suite completed with status: {test_results['metadata']['status']}")
        return test_results


if __name__ == "__main__":
    """
    Main entry point for the test_ipfs_accelerate script.
    
    This will initialize the test class with a list of models to test,
    setup the necessary resources, and run the test suite.
    """
    # Define metadata including models to test
    # For quick testing, we'll use just 2 small models
    metadata = {
        "dataset": "laion/gpt4v-dataset",
        "namespace": "laion/gpt4v-dataset",
        "column": "link",
        "role": "master",
        "split": "train",
        "models": [
            # Just use two small embedding models that are fast to test
            "BAAI/bge-small-en-v1.5",
            "prajjwal1/bert-tiny"
        ],
        "chunk_settings": {},
        "path": "/storage/gpt4v-dataset/data",
        "dst_path": "/storage/gpt4v-dataset/data",
    }
    
    # Initialize resources with proper dictionary structures
    resources = {
        "local_endpoints": {},
        "tei_endpoints": {},
        "tokenizer": {},
        "endpoint_handler": {}
    }
    
    # Define endpoint types and initialize with dictionary structure
    endpoint_types = ["cuda:0", "openvino:0", "cpu:0"]
    for model in metadata["models"]:
        # Initialize model dictionaries
        resources["local_endpoints"][model] = []
        resources["tokenizer"][model] = {}
        resources["endpoint_handler"][model] = {}
        
        # Add endpoints for each model and endpoint type
        for endpoint in endpoint_types:
            # Add endpoint entry
            resources["local_endpoints"][model].append([model, endpoint, 32768])
            
            # Initialize tokenizer and endpoint handler entries
            resources["tokenizer"][model][endpoint] = None
            resources["endpoint_handler"][model][endpoint] = None
    
    # Create properly structured resources for ipfs_accelerate_py.init_endpoints
    resources["structured_resources"] = {
        "tokenizer": resources["tokenizer"],
        "endpoint_handler": resources["endpoint_handler"],
        "queue": {},  # Add queue dictionary - required by init_endpoints
        "queues": {}, # Add queues dictionary
        "batch_sizes": {}, # Add batch_sizes dictionary
        "consumer_tasks": {}, # Add consumer_tasks dictionary
        "caches": {}, # Add caches dictionary
        "endpoints": {
            "local_endpoints": resources["local_endpoints"],
            "api_endpoints": resources.get("tei_endpoints", {}),
            "libp2p_endpoints": resources.get("libp2p_endpoints", {})
        }
    }

    print(f"Starting test for {len(metadata['models'])} models with {len(endpoint_types)} endpoint types")
    
    # Create test instance and run tests
    tester = test_ipfs_accelerate(resources, metadata)
    
    # Run test asynchronously
    print("Running tests...")
    asyncio.run(tester.__test__(resources, metadata))
    
    print("Test complete")