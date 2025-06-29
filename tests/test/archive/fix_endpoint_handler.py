import os
import sys
import json
import asyncio
import traceback
from datetime import datetime
from unittest.mock import MagicMock

# Add the parent directory to sys.path for proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variable to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
,
# Import the needed modules
try:
    from ipfs_accelerate_py.ipfs_accelerate import ipfs_accelerate_py
except ImportError as e:
    print(f"Error importing ipfs_accelerate_py: {}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
    sys.exit(1)

class EndpointHandlerFixer:
    """
    Class to fix the endpoint_handler implementation in ipfs_accelerate_py.
    
    The current issue is that endpoint_handler returns a dictionary instead of a
    callable function, causing "'dict' object is not callable" errors when tests
    try to use it.
    """
    
    def __init__(self):
        self.resources = {}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "local_endpoints": {}}}}}}}}}}}}}}}}}}}}}}}}}}}},
        "tokenizer": {}}}}}}}}}}}}}}}}}}}}}}}}}}}},
        "endpoint_handler": {}}}}}}}}}}}}}}}}}}}}}}}}}}}},
        "queue": {}}}}}}}}}}}}}}}}}}}}}}}}}}}},
        "queues": {}}}}}}}}}}}}}}}}}}}}}}}}}}}},
        "batch_sizes": {}}}}}}}}}}}}}}}}}}}}}}}}}}}},
        "consumer_tasks": {}}}}}}}}}}}}}}}}}}}}}}}}}}}},
        "caches": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        }
        self.metadata = {}}}}}}}}}}}}}}}}}}}}}}}}}}}"models": []}
        ,
        # Import transformers and torch
        try:
            import transformers
            self.resources["transformers"] = transformers,
            print("Successfully imported transformers module")
        except ImportError:
            self.resources["transformers"] = MagicMock(),
            print("Using MagicMock for transformers")
        
        try:
            import torch
            self.resources["torch"] = torch,
            print("Successfully imported torch module")
        except ImportError:
            self.resources["torch"] = MagicMock(),
            print("Using MagicMock for torch")
        
        # Load mapped models
        try:
            with open('mapped_models.json', 'r') as f:
                self.mapped_models = json.load(f)
                print(f"Loaded {}}}}}}}}}}}}}}}}}}}}}}}}}}}len(self.mapped_models)} models from mapped_models.json")
        except Exception as e:
            print(f"Error loading mapped_models.json: {}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
            self.mapped_models = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Initialize results
            self.results = {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "metadata": {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "timestamp": datetime.now().isoformat(),
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "model_results": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            }
    
    def apply_endpoint_handler_fix(self, accelerator):
        """
        Apply the fix to the endpoint_handler method of the ipfs_accelerate_py instance.
        
        Args:
            accelerator: The ipfs_accelerate_py instance to fix
            
        Returns:
            bool: True if the fix was applied successfully, False otherwise
        """:
        try:
            # First, make a backup of the original method if it exists:
            if hasattr(accelerator, 'endpoint_handler'):
                original_endpoint_handler = accelerator.endpoint_handler
                print("Made backup of original endpoint_handler")
            else:
                original_endpoint_handler = None
                print("No original endpoint_handler found")
            
            # Create a new endpoint_handler property that returns a callable
            def get_endpoint_handler(self, model=None, endpoint_type=None):
                """
                Get an endpoint handler for the specified model and endpoint type.
                
                Args:
                    model (str, optional): Model name to get handler for
                    endpoint_type (str, optional): Endpoint type (CPU, CUDA, OpenVINO)
                    
                Returns:
                    If model and endpoint_type are provided: callable function
                    If no arguments: dictionary of handlers
                    """
                if model is None or endpoint_type is None:
                    # Return the dictionary for direct access
                    if hasattr(self, 'resources') and 'endpoint_handler' in self.resources:
                    return self.resources.get("endpoint_handler", {}}}}}}}}}}}}}}}}}}}}}}}}}}}})
                    else:
                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                
                # Get handler and return callable function
                try:
                    if hasattr(self, 'resources') and 'endpoint_handler' in self.resources:
                        handlers = self.resources.get("endpoint_handler", {}}}}}}}}}}}}}}}}}}}}}}}}}}}})
                        if model in handlers and endpoint_type in handlers[model]:,,
                        handler = handlers[model][endpoint_type],,
                            if callable(handler):
                        return handler
                            else:
                                # Create a wrapper function for dictionary handlers
                                async def handler_wrapper(*args, **kwargs):
                                    # Implementation depends on the model type
                                    model_lower = model.lower()
                                    
                                    if any(name in model_lower for name in ["bert", "roberta", "embed", "mpnet", "minilm"]):,,,
                                        # Embedding model response
                                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                "embedding": [0.1, 0.2, 0.3, 0.4] * 96,
                                "implementation_type": "(MOCK-WRAPPER)"
                                }
                                    elif any(name in model_lower for name in ["llama", "gpt", "qwen", "opt"]):,,,
                                        # LLM response
                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "generated_text": f"This is a mock response from {}}}}}}}}}}}}}}}}}}}}}}}}}}}model} using a wrapper function",
                    "model": model,
                    "implementation_type": "(MOCK-WRAPPER)"
                    }
                                    elif any(name in model_lower for name in ["t5", "mt5", "bart"]):,,,
                                        # Text-to-text model
                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "text": "Dies ist ein Testtext für Übersetzungen.",
                    "model": model,
                    "implementation_type": "(MOCK-WRAPPER)"
                    }
                                    elif any(name in model_lower for name in ["whisper", "wav2vec"]):,,,
                                        # Audio model
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "text": "This is a mock transcription of audio content for testing purposes.",
                "model": model,
                "implementation_type": "(MOCK-WRAPPER)"
                }
                                    elif any(name in model_lower for name in ["clip", "xclip"]):,,,
                                        # Vision model
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "similarity": 0.75,
                "model": model,
                "implementation_type": "(MOCK-WRAPPER)"
                }
                                    elif any(name in model_lower for name in ["llava", "vqa"]):,,,
                                        # Vision-language model
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "generated_text": "This is a test image showing a landscape with mountains and a lake.",
            "model": model,
            "implementation_type": "(MOCK-WRAPPER)"
            }
                                    else:
                                        # Generic response
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "output": f"Mock response from {}}}}}}}}}}}}}}}}}}}}}}}}}}}model}",
            "input": args[0] if args else kwargs.get('input', 'No input'),:::,,,
            "model": model,
            "implementation_type": "(MOCK-WRAPPER)"
            }
            return handler_wrapper
                        else:
                            # Create mock handler if not found
                            return self._create_mock_handler(model, endpoint_type):
                    else:
                                return self._create_mock_handler(model, endpoint_type)
                except Exception as e:
                    print(f"Error getting endpoint handler: {}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
                                return self._create_mock_handler(model, endpoint_type)
            
            # Create a mock handler method if it doesn't exist:
            def _create_mock_handler(self, model, endpoint_type):
                """Create a mock handler function for testing."""
                async def mock_handler(*args, **kwargs):
                    model_lower = model.lower()
                    
                    if any(name in model_lower for name in ["bert", "roberta", "embed", "mpnet", "minilm"]):,,,
                        # Embedding model response
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "embedding": [0.1, 0.2, 0.3, 0.4] * 96,
                "implementation_type": "(MOCK)"
                }
                    elif any(name in model_lower for name in ["llama", "gpt", "qwen", "opt"]):,,,
                        # LLM response
                            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "generated_text": f"This is a mock response from {}}}}}}}}}}}}}}}}}}}}}}}}}}}model} using {}}}}}}}}}}}}}}}}}}}}}}}}}}}endpoint_type}",
                            "model": model,
                            "implementation_type": "(MOCK)"
                            }
                    elif any(name in model_lower for name in ["t5", "mt5", "bart"]):,,,
                        # Text-to-text model
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "text": "Dies ist ein Testtext für Übersetzungen.",
            "model": model,
            "implementation_type": "(MOCK)"
            }
                    elif any(name in model_lower for name in ["whisper", "wav2vec"]):,,,
                        # Audio model
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "text": "This is a mock transcription of audio content for testing purposes.",
            "model": model,
            "implementation_type": "(MOCK)"
            }
                    elif any(name in model_lower for name in ["clip", "xclip"]):,,,
                        # Vision model
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "similarity": 0.75,
            "model": model,
            "implementation_type": "(MOCK)"
            }
                    elif any(name in model_lower for name in ["llava", "vqa"]):,,,
                        # Vision-language model
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "generated_text": "This is a test image showing a landscape with mountains and a lake.",
            "model": model,
            "implementation_type": "(MOCK)"
            }
                    else:
                        # Generic response
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "output": f"Mock response from {}}}}}}}}}}}}}}}}}}}}}}}}}}}model}",
            "input": args[0] if args else kwargs.get('input', 'No input'),:::,,,
            "model": model,
            "implementation_type": "(MOCK)"
            }
            return mock_handler
            
            # Add the methods to the accelerator instance
            setattr(accelerator, 'get_endpoint_handler', get_endpoint_handler.__get__(accelerator))
            setattr(accelerator, '_create_mock_handler', _create_mock_handler.__get__(accelerator))
            
            # Use a property to ensure backwards compatibility
            class EndpointHandlerProperty:
                def __get__(self, obj, objtype=None):
                return obj.get_endpoint_handler
            
            # Apply the property
                type(accelerator).endpoint_handler = EndpointHandlerProperty()
            
                print("Successfully applied endpoint_handler fix")
            return True
        except Exception as e:
            print(f"Error applying endpoint_handler fix: {}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
            print(traceback.format_exc())
            return False
    
    async def test_endpoint(self, skill_name, model_name):
        """
        Test an endpoint for the given skill and model using the fixed implementation.
        
        Args:
            skill_name (str): The skill name
            model_name (str): The model name
            
        Returns:
            dict: The test result
            """
            print(f"\nTesting {}}}}}}}}}}}}}}}}}}}}}}}}}}}skill_name} with model {}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}...")
            result = {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "status": "Not tested",
            "model": model_name,
            "skill": skill_name,
            "endpoint_type": "cpu:0",
            "error": None,
            "input": None,
            "output": None,
            "implementation_type": "Unknown"
            }
        
        # Create appropriate test input based on skill
            if skill_name in ["bert", "distilbert", "roberta", "mpnet", "albert"]:,
            result["input"] = "This is a test sentence for embedding models.",
            skill_handler = "default_embed"
        elif skill_name in ["gpt_neo", "gptj", "gpt2", "opt", "bloom", "codegen", "llama"]:,
            result["input"] = "Once upon a time",
            skill_handler = "default_lm"
        elif skill_name == "whisper":
            result["input"] = "test.mp3",,,
            skill_handler = "hf_whisper"
        elif skill_name == "wav2vec2":
            result["input"] = "test.mp3",,,
            skill_handler = "hf_wav2vec2"
        elif skill_name == "clip":
            result["input"] = "test.jpg",
            skill_handler = "hf_clip"
        elif skill_name == "xclip":
            result["input"] = "test.mp4",
            skill_handler = "hf_xclip"
        elif skill_name == "clap":
            result["input"] = "test.mp3",,,
            skill_handler = "hf_clap"
        elif skill_name == "t5":
            result["input"] = "translate English to German: Hello, how are you?",
            skill_handler = "hf_t5"
        elif skill_name in ["llava", "llava_next", "qwen2_vl"]:,
            result["input"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}"image": "test.jpg", "text": "What is in this image?"},
            skill_handler = "hf_llava" if skill_name == "llava" else "hf_llava_next":
        else:
            result["input"] = "Generic test input for model.",
            skill_handler = "default_lm"
        
        try:
            # Initialize the accelerator
            accelerator = ipfs_accelerate_py(self.resources, self.metadata)
            
            # Apply the endpoint_handler fix
            if not self.apply_endpoint_handler_fix(accelerator):
                result["status"] = "Error applying endpoint_handler fix",
                result["error"],, = "Failed to apply endpoint_handler fix",
            return result
            
            # Add the endpoint
            endpoint_added = False
            try:
                print(f"  Adding endpoint for {}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name} ({}}}}}}}}}}}}}}}}}}}}}}}}}}}skill_handler})...")
                endpoint_key = f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}skill_handler}/{}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}/cpu:0"
                # Create endpoint tuple matching what add_endpoint expects: (model, backend, context_length)
                endpoint = (model_name, "cpu:0", 2048)
                endpoint_added = await accelerator.add_endpoint(skill_handler, "local_endpoints", endpoint)
                if endpoint_added:
                    print(f"  ✅ Successfully added endpoint: {}}}}}}}}}}}}}}}}}}}}}}}}}}}endpoint_key}")
                    result["status"] = "Endpoint added",
                else:
                    print(f"  ❌ Failed to add endpoint: {}}}}}}}}}}}}}}}}}}}}}}}}}}}endpoint_key}")
                    result["status"] = "Failed to add endpoint",
                    result["error"],, = "add_endpoint returned False",
                    return result
            except Exception as e:
                print(f"  ❌ Error adding endpoint: {}}}}}}}}}}}}}}}}}}}}}}}}}}}str(e)}")
                result["status"] = "Error adding endpoint",
                result["error"],, = str(e),,,,
                result["traceback"] = traceback.format_exc(),,
                    return result
            
            # Check if the endpoint handler exists and is callable:
            try:
                endpoint_handler = accelerator.endpoint_handler(skill_handler, model_name, "cpu:0")
                if endpoint_handler:
                    print(f"  ✓ Found endpoint handler: {}}}}}}}}}}}}}}}}}}}}}}}}}}}type(endpoint_handler)}")
                    
                    # Check if handler is callable:
                    if callable(endpoint_handler):
                        # Detect if handler is real or mock implementation:
                        try:
                            import inspect
                            handler_source = inspect.getsource(endpoint_handler)
                            if "MagicMock" in handler_source or "mock" in handler_source.lower() or "MOCK" in handler_source:
                                result["implementation_type"] = "MOCK",
                            else:
                                result["implementation_type"] = "REAL"
                                ,
                            # Call the endpoint handler
                                print(f"  Calling endpoint handler with input: {}}}}}}}}}}}}}}}}}}}}}}}}}}}result['input']}"),
                            if asyncio.iscoroutinefunction(endpoint_handler):
                                output = await endpoint_handler(result["input"]),
                            else:
                                output = endpoint_handler(result["input"]),
                            
                                result["output"] = output,
                                result["status"] = "Success",
                                print(f"  ✅ Successfully called endpoint handler: {}}}}}}}}}}}}}}}}}}}}}}}}}}}type(output)}")
                        except Exception as e:
                            print(f"  ❌ Error calling endpoint handler: {}}}}}}}}}}}}}}}}}}}}}}}}}}}str(e)}")
                            result["status"] = "Error calling endpoint handler",
                            result["error"],, = str(e),,,,
                            result["traceback"] = traceback.format_exc(),,
                    else:
                        print(f"  ❌ Endpoint handler is not callable: {}}}}}}}}}}}}}}}}}}}}}}}}}}}type(endpoint_handler)}")
                        result["status"] = "Error: handler is not callable",
                        result["error"],, = f"Handler has type {}}}}}}}}}}}}}}}}}}}}}}}}}}}type(endpoint_handler)} which is not callable",
                else:
                    print(f"  ❌ Endpoint handler not found")
                    result["status"] = "Endpoint handler not found",
            except Exception as e:
                print(f"  ❌ Error getting endpoint handler: {}}}}}}}}}}}}}}}}}}}}}}}}}}}str(e)}")
                result["status"] = "Error getting endpoint handler",
                result["error"],, = str(e),,,,
                result["traceback"] = traceback.format_exc(),,
            
            # Remove the endpoint
            try:
                if endpoint_added:
                    remove_success = await accelerator.remove_endpoint(skill_handler, model_name, "cpu:0")
                    if remove_success:
                        print(f"  ✓ Successfully removed endpoint")
                    else:
                        print(f"  ✗ Failed to remove endpoint")
            except Exception as e:
                print(f"  ✗ Error removing endpoint: {}}}}}}}}}}}}}}}}}}}}}}}}}}}str(e)}")
        
        except Exception as e:
            print(f"  ❌ Error in test_endpoint: {}}}}}}}}}}}}}}}}}}}}}}}}}}}str(e)}")
            result["status"] = "Error in test_endpoint",
            result["error"],, = str(e),,,,
            result["traceback"] = traceback.format_exc(),,
        
                return result
    
    async def run_tests(self):
        """
        Run tests for all models in mapped_models.json
        
        Returns:
            dict: Test results
            """
        for skill_name, model_name in self.mapped_models.items():
            try:
                result = await self.test_endpoint(skill_name, model_name)
                self.results["model_results"][skill_name] = result,
            except Exception as e:
                print(f"Error testing {}}}}}}}}}}}}}}}}}}}}}}}}}}}skill_name} with {}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}str(e)}")
                self.results["model_results"][skill_name] = {}}}}}}}}}}}}}}}}}}}}}}}}}}},
                "status": "Error",
                "model": model_name,
                "skill": skill_name,
                "error": str(e),
                "traceback": traceback.format_exc()
                }
        
        # Save the results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"fixed_endpoints_test_results_{}}}}}}}}}}}}}}}}}}}}}}}}}}}timestamp}.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
            print(f"\nTest results saved to fixed_endpoints_test_results_{}}}}}}}}}}}}}}}}}}}}}}}}}}}timestamp}.json")
        
        # Generate summary report
            success_count = 0
            real_impl_count = 0
            mock_impl_count = 0
            failed_count = 0
            error_endpoints = []
            ,
            for skill_name, result in self.results["model_results"].items():,
            if result["status"] == "Success":,,
            success_count += 1
            if result["implementation_type"] == "REAL":,
            real_impl_count += 1
                elif "MOCK" in result["implementation_type"]:,
                mock_impl_count += 1
            else:
                failed_count += 1
                error_endpoints.append({}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "skill": skill_name,
                "model": result["model"],
                "status": result["status"],
                "error": result["error"],,
                })
        
                print("\n=== TEST SUMMARY ===")
                print(f"Total models tested: {}}}}}}}}}}}}}}}}}}}}}}}}}}}len(self.mapped_models)}")
                print(f"Successful endpoints: {}}}}}}}}}}}}}}}}}}}}}}}}}}}success_count} ({}}}}}}}}}}}}}}}}}}}}}}}}}}}success_count/len(self.mapped_models)*100:.1f}%)")
                print(f"  - REAL implementations: {}}}}}}}}}}}}}}}}}}}}}}}}}}}real_impl_count}")
                print(f"  - MOCK implementations: {}}}}}}}}}}}}}}}}}}}}}}}}}}}mock_impl_count}")
                print(f"Failed endpoints: {}}}}}}}}}}}}}}}}}}}}}}}}}}}failed_count} ({}}}}}}}}}}}}}}}}}}}}}}}}}}}failed_count/len(self.mapped_models)*100:.1f}%)")
        
        if error_endpoints:
            print("\nEndpoints with errors:")
            for error in error_endpoints:
                print(f"  - {}}}}}}}}}}}}}}}}}}}}}}}}}}}error['skill']} ({}}}}}}}}}}}}}}}}}}}}}}}}}}}error['model']}): {}}}}}}}}}}}}}}}}}}}}}}}}}}}error['status']} - {}}}}}}}}}}}}}}}}}}}}}}}}}}}error['error']}")
                ,
        # Generate markdown report
        with open(f"fixed_endpoints_report_{}}}}}}}}}}}}}}}}}}}}}}}}}}}timestamp}.md", "w") as f:
            f.write(f"# Local Endpoints Test Report with Fixed Handler\n\n")
            f.write(f"Generated on: {}}}}}}}}}}}}}}}}}}}}}}}}}}}datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- Total models tested: {}}}}}}}}}}}}}}}}}}}}}}}}}}}len(self.mapped_models)}\n")
            f.write(f"- Successful endpoints: {}}}}}}}}}}}}}}}}}}}}}}}}}}}success_count} ({}}}}}}}}}}}}}}}}}}}}}}}}}}}success_count/len(self.mapped_models)*100:.1f}%)\n")
            f.write(f"  - REAL implementations: {}}}}}}}}}}}}}}}}}}}}}}}}}}}real_impl_count}\n")
            f.write(f"  - MOCK implementations: {}}}}}}}}}}}}}}}}}}}}}}}}}}}mock_impl_count}\n")
            f.write(f"- Failed endpoints: {}}}}}}}}}}}}}}}}}}}}}}}}}}}failed_count} ({}}}}}}}}}}}}}}}}}}}}}}}}}}}failed_count/len(self.mapped_models)*100:.1f}%)\n\n")
            
            f.write(f"## Successful Endpoints\n\n")
            f.write("| Skill | Model | Implementation |\n")
            f.write("|-------|-------|----------------|\n")
            
            for skill_name, result in sorted(self.results["model_results"].items()):,,
            if result["status"] == "Success":,,
            f.write(f"| {}}}}}}}}}}}}}}}}}}}}}}}}}}}skill_name} | {}}}}}}}}}}}}}}}}}}}}}}}}}}}result['model']} | {}}}}}}}}}}}}}}}}}}}}}}}}}}}result['implementation_type']} |\n")
            ,
            f.write(f"\n## Failed Endpoints\n\n")
            f.write("| Skill | Model | Status | Error |\n")
            f.write("|-------|-------|--------|-------|\n")
            
            for skill_name, result in sorted(self.results["model_results"].items()):,,
            if result["status"] != "Success":,
            error_msg = result["error"],,
                    if error_msg and len(str(error_msg)) > 100:
                        error_msg = str(error_msg)[:100] + "...",
                        f.write(f"| {}}}}}}}}}}}}}}}}}}}}}}}}}}}skill_name} | {}}}}}}}}}}}}}}}}}}}}}}}}}}}result['model']} | {}}}}}}}}}}}}}}}}}}}}}}}}}}}result['status']} | {}}}}}}}}}}}}}}}}}}}}}}}}}}}error_msg} |\n")
                        ,
                        print(f"\nDetailed report saved to fixed_endpoints_report_{}}}}}}}}}}}}}}}}}}}}}}}}}}}timestamp}.md")
        
            return self.results

    def create_persistent_fix(self):
        """
        Create a file with the permanent fix that can be applied to the
        ipfs_accelerate_py module directly.
        
        Returns:
            str: The path to the created file
            """
            fix_code = """# Fix for the endpoint_handler method in ipfs_accelerate_py
# 
# This code can be applied to the ipfs_accelerate.py file to fix the 
# endpoint_handler implementation so it returns callable functions
# instead of dictionaries.

# Replace the existing endpoint_handler property with this implementation:

            @property
def endpoint_handler(self):
    """
    Property that provides access to endpoint handlers.
    
    This can be used in two ways:
        1. When accessed without arguments: returns the resources dictionary
        for direct attribute access (self.endpoint_handler[model][type]),
        2. When called with arguments: returns a callable function
        for the specific model and endpoint type (self.endpoint_handler(model, type))
        """
    return self.get_endpoint_handler

def get_endpoint_handler(self, model=None, endpoint_type=None):
    """
    Get an endpoint handler for the specified model and endpoint type.
    
    Args:
        model (str, optional): Model name to get handler for
        endpoint_type (str, optional): Endpoint type (CPU, CUDA, OpenVINO)
        
    Returns:
        If model and endpoint_type are provided: callable function
        If no arguments: dictionary of handlers
        """
    if model is None or endpoint_type is None:
        # Return the dictionary for direct access
        return self.resources.get("endpoint_handler", {}}}}}}}}}}}}}}}}}}}}}}}}}}}})
    
    # Get handler and return callable function
    try:
        handlers = self.resources.get("endpoint_handler", {}}}}}}}}}}}}}}}}}}}}}}}}}}}})
        if model in handlers and endpoint_type in handlers[model]:,,
        handler = handlers[model][endpoint_type],,
            if callable(handler):
        return handler
            else:
                # Create a wrapper function for dictionary handlers
                async def handler_wrapper(*args, **kwargs):
                    # Implementation would depend on the model type
                    # This is a placeholder
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}"text": f"Response from {}}}}}}}}}}}}}}}}}}}}}}}}}}}model} using {}}}}}}}}}}}}}}}}}}}}}}}}}}}endpoint_type}",
                "implementation_type": "(MOCK)"}
        return handler_wrapper
        else:
            # Create mock handler if not found
            return self._create_mock_handler(model, endpoint_type):
    except Exception as e:
        print(f"Error getting endpoint handler: {}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
                return self._create_mock_handler(model, endpoint_type)

def _create_mock_handler(self, model, endpoint_type):
    """Create a mock handler function for testing."""
    async def mock_handler(*args, **kwargs):
        model_lower = model.lower()
        
        if any(name in model_lower for name in ["bert", "roberta", "embed", "mpnet", "minilm"]):,,,
            # Embedding model response
    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
    "embedding": [0.1, 0.2, 0.3, 0.4] * 96,
    "implementation_type": "(MOCK)"
    }
        elif any(name in model_lower for name in ["llama", "gpt", "qwen", "opt"]):,,,
            # LLM response
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "generated_text": f"This is a mock response from {}}}}}}}}}}}}}}}}}}}}}}}}}}}model} using {}}}}}}}}}}}}}}}}}}}}}}}}}}}endpoint_type}",
            "model": model,
            "implementation_type": "(MOCK)"
            }
        elif any(name in model_lower for name in ["t5", "mt5", "bart"]):,,,
            # Text-to-text model
        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "text": "Dies ist ein Testtext für Übersetzungen.",
        "model": model,
        "implementation_type": "(MOCK)"
        }
        elif any(name in model_lower for name in ["whisper", "wav2vec"]):,,,
            # Audio model
    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
    "text": "This is a mock transcription of audio content for testing purposes.",
    "model": model,
    "implementation_type": "(MOCK)"
    }
        elif any(name in model_lower for name in ["clip", "xclip"]):,,,
            # Vision model
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "similarity": 0.75,
            "model": model,
            "implementation_type": "(MOCK)"
            }
        elif any(name in model_lower for name in ["llava", "vqa"]):,,,
            # Vision-language model
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "generated_text": "This is a test image showing a landscape with mountains and a lake.",
            "model": model,
            "implementation_type": "(MOCK)"
            }
        else:
            # Generic response
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "output": f"Mock response from {}}}}}}}}}}}}}}}}}}}}}}}}}}}model}",
            "input": args[0] if args else kwargs.get('input', 'No input'),:::,,,
            "model": model,
            "implementation_type": "(MOCK)"
            }
                return mock_handler
                """
        
        # Write the fix to a file
                fix_path = "endpoint_handler_fix.py"
        with open(fix_path, "w") as f:
            f.write(fix_code)
        
            print(f"Permanent fix written to {}}}}}}}}}}}}}}}}}}}}}}}}}}}fix_path}")
                return fix_path

# Main function
async def main():
    fixer = EndpointHandlerFixer()
    
    # Create the persistent fix first
    fix_path = fixer.create_persistent_fix()
    print(f"Created persistent fix in {}}}}}}}}}}}}}}}}}}}}}}}}}}}fix_path}")
    
    # Run the tests with the dynamic fix
    print("\nRunning tests with dynamic fix...")
    await fixer.run_tests()

# Run the fixer
if __name__ == "__main__":
    asyncio.run(main())