#!/usr/bin/env python
"""
Implementation of OpenAI Function Calling for ipfs_accelerate_py.
This enhances the existing OpenAI API implementation with comprehensive function calling capabilities.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional, Union, Callable
from dotenv import load_dotenv

# Load environment variables
load_dotenv()))))))))))))

# Add parent directory to path to import the ipfs_accelerate_py module
sys.path.append())))))))))))os.path.join())))))))))))os.path.dirname())))))))))))os.path.dirname())))))))))))__file__)), 'ipfs_accelerate_py'))

try:
    # Import the OpenAI API implementation to extend it
    from api_backends import openai_api
    import openai
except ImportError as e:
    print())))))))))))f"Failed to import required modules: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
    sys.exit())))))))))))1)

class OpenAIFunctionCalling:
    """
    Extension to add Function Calling capabilities to the ipfs_accelerate_py OpenAI implementation.
    This class provides methods for structured function calling, parallel function calling, and tool use.
    """
    
    def __init__())))))))))))self, api_key: Optional[]]]]]],,,,,,str] = None, resources: Dict[]]]]]],,,,,,str, Any] = None, metadata: Dict[]]]]]],,,,,,str, Any] = None):,
    """
    Initialize the OpenAI Function Calling extension.
        
        Args:
            api_key: OpenAI API key ())))))))))))optional, will use environment variable if not provided::::):
                resources: Resources dictionary for ipfs_accelerate_py
                metadata: Metadata dictionary for ipfs_accelerate_py
                """
                self.resources = resources if resources else {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                self.metadata = metadata if metadata else {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Get API key from parameters, metadata, or environment variable
                self.api_key = api_key or ())))))))))))metadata.get())))))))))))"openai_api_key") if metadata else None) or os.environ.get())))))))))))"OPENAI_API_KEY")
        :
        if not self.api_key:
            print())))))))))))"Warning: No OpenAI API key provided. Function calling will not work without an API key.")
        
        # Set up the OpenAI API client
            openai.api_key = self.api_key
        
        # Store main OpenAI API implementation for delegation
            self.openai_api_impl = openai_api())))))))))))resources=self.resources, metadata=self.metadata)
        
        # Initialize tracking for rate limiting and backoff
            self.current_requests = 0
            self.max_concurrent_requests = 5
            self.request_queue = []]]]]],,,,,,],,,,
            self.max_retries = 3
            self.base_wait_time = 1.0  # in seconds
        
        # Store registered functions
            self.registered_functions = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
            print())))))))))))"OpenAI Function Calling extension initialized")
    
    # === Function Definition and Registration ===
    
            def register_function())))))))))))self, function: Callable, name: Optional[]]]]]],,,,,,str] = None,
            description: Optional[]]]]]],,,,,,str] = None,
            parameters: Optional[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]] = None) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,,
            """
            Register a function for use with OpenAI function calling.
        
        Args:
            function: The Python function to register
            name: The name of the function ())))))))))))defaults to the function's __name__)
            description: A description of what the function does
            parameters: A JSON Schema object describing the function's parameters
            ())))))))))))if None, will attempt to generate from function signature)
            :
        Returns:
            Function registration status
            """
        try:
            import inspect
            from inspect import signature, Parameter
            
            # Get function name if not provided::::
            func_name = name or function.__name__
            
            # Get function description if not provided::::
            func_description = description:
            if not func_description and function.__doc__:
                func_description = function.__doc__.strip()))))))))))))
            elif not func_description:
                func_description = f"Function {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}func_name}"
            
            # Get function parameters if not provided::::
            func_parameters = parameters:
            if not func_parameters:
                sig = signature())))))))))))function)
                
                # Generate parameters schema from function signature
                properties = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                required = []]]]]],,,,,,],,,,
                
                for param_name, param in sig.parameters.items())))))))))))):
                    # Skip self parameter for methods
                    if param_name == 'self':
                    continue
                    
                    # Create parameter schema
                    param_schema = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"type": "string"}  # Default to string
                    
                    # Try to get a better type from annotations
                    if param.annotation != Parameter.empty:
                        annotation = param.annotation
                        if annotation == str:
                            param_schema[]]]]]],,,,,,"type"] = "string",
                        elif annotation == int:
                            param_schema[]]]]]],,,,,,"type"] = "integer",
                        elif annotation == float:
                            param_schema[]]]]]],,,,,,"type"] = "number",
                        elif annotation == bool:
                            param_schema[]]]]]],,,,,,"type"] = "boolean",
                        elif annotation == list or annotation == List:
                            param_schema[]]]]]],,,,,,"type"] = "array",
                            param_schema[]]]]]],,,,,,"items"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"type": "string"},
                        elif annotation == dict or annotation == Dict:
                            param_schema[]]]]]],,,,,,"type"] = "object"
                            ,
                    # Get parameter description from docstring if available:
                    if function.__doc__:
                        param_doc = self._extract_param_doc())))))))))))function.__doc__, param_name)
                        if param_doc:
                            param_schema[]]]]]],,,,,,"description"] = param_doc
                            ,
                            properties[]]]]]],,,,,,param_name] = param_schema
                            ,
                    # Check if parameter is required:
                    if param.default == Parameter.empty:
                        required.append())))))))))))param_name)
                
                # Create the full parameters schema
                        func_parameters = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "type": "object",
                        "properties": properties
                        }
                
                if required:
                    func_parameters[]]]]]],,,,,,"required"] = required
                    ,
            # Create the function definition
                    function_def = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "name": func_name,
                    "description": func_description,
                    "parameters": func_parameters
                    }
            
            # Store the function and its definition
                    self.registered_functions[]]]]]],,,,,,func_name] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
                    "function": function,
                    "definition": function_def
                    }
            
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "success": True,
                        "name": func_name,
                        "description": func_description,
                        "parameters": func_parameters
                        }
        
        except Exception as e:
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "success": False,
                        "error": str())))))))))))e)
                        }
    
                        def _extract_param_doc())))))))))))self, docstring: str, param_name: str) -> Optional[]]]]]],,,,,,str]:,
                        """
                        Extract parameter documentation from a docstring.
        
        Args:
            docstring: The function's docstring
            param_name: The parameter name to extract documentation for
            
        Returns:
            The parameter description or None if not found
            """
            lines = docstring.split())))))))))))'\n')
            param_line = None
        
        # Look for parameter in docstring ())))))))))))supports various styles):
        for i, line in enumerate())))))))))))lines):
            # Look for ":param param_name:" or "@param param_name" or "Args: ... param_name:"
            if f":param {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}param_name}:" in line or f"@param {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}param_name}" in line:
                param_line = line.split())))))))))))':', 2)[]]]]]],,,,,,-1].strip()))))))))))))
                ,
                # Check for multi-line descriptions
                j = i + 1
                while j < len())))))))))))lines) and ())))))))))))lines[]]]]]],,,,,,j].startswith())))))))))))' ') or lines[]]]]]],,,,,,j].startswith())))))))))))'\t')):,,
                param_line += ' ' + lines[]]]]]],,,,,,j].strip())))))))))))),
                j += 1
                    
            return param_line
            
            # Check for Args: section
            if "Args:" in line or "Arguments:" in line:
                # Look for the parameter in subsequent indented lines
                j = i + 1
                while j < len())))))))))))lines) and ())))))))))))lines[]]]]]],,,,,,j].startswith())))))))))))' ') or lines[]]]]]],,,,,,j].startswith())))))))))))'\t')):,,
                    if lines[]]]]]],,,,,,j].strip())))))))))))),.startswith())))))))))))f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}param_name}:"):
                        param_line = lines[]]]]]],,,,,,j].split())))))))))))':', 1)[]]]]]],,,,,,-1].strip()))))))))))))
                        ,
                        # Check for multi-line descriptions
                        k = j + 1
                        while k < len())))))))))))lines) and lines[]]]]]],,,,,,k].startswith())))))))))))' ' * ())))))))))))len())))))))))))lines[]]]]]],,,,,,j]) - len())))))))))))lines[]]]]]],,,,,,j].lstrip()))))))))))))) + 4)):,
                        param_line += ' ' + lines[]]]]]],,,,,,k].strip())))))))))))),
                        k += 1
                            
                return param_line
                j += 1
        
            return None
    
            def get_registered_function_definitions())))))))))))self) -> List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]]:,
            """
            Get the definitions of all registered functions for use with OpenAI.
        
        Returns:
            A list of function definitions
            """
            return []]]]]],,,,,,func_data[]]]]]],,,,,,"definition"] for func_data in self.registered_functions.values()))))))))))))]:,
            def unregister_function())))))))))))self, name: str) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,
            """
            Unregister a function by name.
        
        Args:
            name: The name of the function to unregister
            
        Returns:
            Unregistration status
            """
        if name in self.registered_functions:
            del self.registered_functions[]]]]]],,,,,,name],
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "message": f"Function '{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}name}' unregistered successfully."
            }
        else:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": f"Function '{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}name}' is not registered."
            }
    
    # === Function Calling ===
    
            def request_with_functions())))))))))))self, messages: List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]],
            functions: Optional[]]]]]],,,,,,List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]]] = None,
            model: str = "gpt-4o",
            temperature: float = 0.7,
            max_tokens: Optional[]]]]]],,,,,,int] = None,
            tool_choice: Optional[]]]]]],,,,,,Union[]]]]]],,,,,,str, Dict[]]]]]],,,,,,str, Any]]] = None) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,,
            """
            Send a request to OpenAI with function calling capabilities.
        
        Args:
            messages: The conversation messages
            functions: Function definitions ())))))))))))if None, uses all registered functions)::::::
                model: The model to use for the request
                temperature: The sampling temperature
                max_tokens: The maximum number of tokens to generate
                tool_choice: Specifies which function to use ())))))))))))auto, required, or specific function)
            
        Returns:
            The response from OpenAI
            """
        try:
            # Check API key
            if not self.api_key:
            raise ValueError())))))))))))"No OpenAI API key provided.")
            
            # Get function definitions if not provided::::
            if functions is None:
                functions = self.get_registered_function_definitions()))))))))))))
            
            # Convert functions to tools format for API
                tools = []]]]]],,,,,,],,,,
            for func in functions:
                tools.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "type": "function",
                "function": func
                })
            
            # Create the request
                response = openai.chat.completions.create())))))))))))
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens
                )
            
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "response": response,
                "content": response.choices[]]]]]],,,,,,0].message.content,
                "tool_calls": response.choices[]]]]]],,,,,,0].message.tool_calls,
                "finish_reason": response.choices[]]]]]],,,,,,0].finish_reason,,
                }
        
        except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": str())))))))))))e)
                }
    
                def execute_function_call())))))))))))self, tool_call) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,
                """
                Execute a function call from an OpenAI response.
        
        Args:
            tool_call: The tool_call object from the OpenAI response
            
        Returns:
            The result of the function execution
            """
        try:
            # Extract function call information
            func_name = tool_call.function.name
            func_args = json.loads())))))))))))tool_call.function.arguments)
            
            # Check if the function is registered:
            if func_name not in self.registered_functions:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": f"Function '{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}func_name}' is not registered."
            }
            
            # Get the Python function to call
            func = self.registered_functions[]]]]]],,,,,,func_name][]]]]]],,,,,,"function"]
            ,
            # Call the function with the provided arguments
            result = func())))))))))))**func_args)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "function_name": func_name,
            "arguments": func_args,
            "result": result
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))))))))e),
            "function_name": func_name if 'func_name' in locals())))))))))))) else "unknown"
            }
    :
        def process_with_function_calling())))))))))))self, messages: List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]],
        functions: Optional[]]]]]],,,,,,List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]]] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[]]]]]],,,,,,int] = None,
        tool_choice: Optional[]]]]]],,,,,,Union[]]]]]],,,,,,str, Dict[]]]]]],,,,,,str, Any]]] = None,
        auto_execute: bool = True) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,
        """
        Process a conversation with function calling, optionally executing the functions.
        
        Args:
            messages: The conversation messages
            functions: Function definitions ())))))))))))if None, uses all registered functions)::::::
                model: The model to use for the request
                temperature: The sampling temperature
                max_tokens: The maximum number of tokens to generate
                tool_choice: Specifies which function to use ())))))))))))auto, required, or specific function)
                auto_execute: Whether to automatically execute the called functions
            
        Returns:
            The processed response with function results
            """
        try:
            # Make the initial request with functions
            response = self.request_with_functions())))))))))))
            messages=messages,
            functions=functions,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice
            )
            
            if not response[]]]]]],,,,,,"success"],,:,,
            return response
            
            # Check if the model wants to call a function:
            if response[]]]]]],,,,,,"finish_reason"],, == "tool_calls" and response[]]]]]],,,,,,"tool_calls"],,:,
            tool_calls = response[]]]]]],,,,,,"tool_calls"],,
            function_results = []]]]]],,,,,,],,,,
                
                # Execute each function call if auto_execute is enabled::
                for tool_call in tool_calls:
                    if tool_call.type == "function":
                        if auto_execute:
                            # Execute the function call
                            result = self.execute_function_call())))))))))))tool_call)
                            
                            # Add the result
                            function_results.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "tool_call_id": tool_call.id,
                            "function_name": tool_call.function.name,
                            "result": result
                            })
                        else:
                            # Just include the function call information without executing
                            function_results.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "tool_call_id": tool_call.id,
                            "function_name": tool_call.function.name,
                            "arguments": json.loads())))))))))))tool_call.function.arguments),
                            "executed": False
                            })
                
                # Prepare the response with function results
                            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "success": True,
                            "initial_response": response[]]]]]],,,,,,"response"],
                            "tool_calls": tool_calls,
                            "function_results": function_results,
                            "executed": auto_execute
                            }
            
            # No function calls, just return the regular response
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "success": True,
                        "response": response[]]]]]],,,,,,"response"],
                        "content": response[]]]]]],,,,,,"content"],
                        "finish_reason": response[]]]]]],,,,,,"finish_reason"],,
                        }
        
        except Exception as e:
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "success": False,
                        "error": str())))))))))))e)
                        }
    
                        def complete_function_conversation())))))))))))self, messages: List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]],
                        functions: Optional[]]]]]],,,,,,List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]]] = None,
                        model: str = "gpt-4o",
                        temperature: float = 0.7,
                        max_tokens: Optional[]]]]]],,,,,,int] = None,
                        max_function_calls: int = 10) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,
                        """
                        Complete a full conversation with multiple back-and-forth function calls.
        
        Args:
            messages: The initial conversation messages
            functions: Function definitions ())))))))))))if None, uses all registered functions)::::::
                model: The model to use for the request
                temperature: The sampling temperature
                max_tokens: The maximum number of tokens to generate
                max_function_calls: Maximum number of function calls to prevent infinite loops
            
        Returns:
            The complete conversation with all function calls and responses
            """
        try:
            # Create a copy of the messages to work with
            conversation = messages.copy()))))))))))))
            
            # Get function definitions if not provided::::
            if functions is None:
                functions = self.get_registered_function_definitions()))))))))))))
            
            # Initialize tracking variables
                function_call_count = 0
                all_function_calls = []]]]]],,,,,,],,,,
            
            while function_call_count < max_function_calls:
                # Make the request with functions
                response = self.request_with_functions())))))))))))
                messages=conversation,
                functions=functions,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
                )
                
                if not response[]]]]]],,,,,,"success"],,:,,
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": response[]]]]]],,,,,,"error"],
                "conversation": conversation,
                "function_calls": all_function_calls
                }
                
                # Add the assistant's response to the conversation
                assistant_message = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "role": "assistant",
                "content": response[]]]]]],,,,,,"content"] if response[]]]]]],,,,,,"content"] else None,,
                }
                
                # Check if the model wants to call a function::
                if response[]]]]]],,,,,,"finish_reason"],, == "tool_calls" and response[]]]]]],,,,,,"tool_calls"],,:,
                tool_calls = response[]]]]]],,,,,,"tool_calls"],,
                    
                    # Add tool calls to the assistant message
                assistant_message[]]]]]],,,,,,"tool_calls"],, = []]]]]],,,,,,],,,,
                    for tool_call in tool_calls:
                        if tool_call.type == "function":
                            assistant_message[]]]]]],,,,,,"tool_calls"],,.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "id": tool_call.id,
                            "type": "function",
                            "function": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                            }
                            })
                    
                    # Add the assistant's message to the conversation
                            conversation.append())))))))))))assistant_message)
                    
                    # Process each function call
                    for tool_call in tool_calls:
                        if tool_call.type == "function":
                            function_call_count += 1
                            
                            # Execute the function
                            result = self.execute_function_call())))))))))))tool_call)
                            
                            # Convert the result to a string
                            if isinstance())))))))))))result.get())))))))))))"result"), ())))))))))))dict, list)):
                                result_str = json.dumps())))))))))))result[]]]]]],,,,,,"result"]),
                            else:
                                result_str = str())))))))))))result[]]]]]],,,,,,"result"]),
                            
                            # Add the function call to tracking
                                all_function_calls.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                "function_name": tool_call.function.name,
                                "arguments": json.loads())))))))))))tool_call.function.arguments),
                                "result": result[]]]]]],,,,,,"result"],
                                "success": result[]]]]]],,,,,,"success"],,
                                })
                            
                            # Add the function result as a tool response message
                                conversation.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": result_str
                                })
                    
                    # Continue the conversation with another API call
                                continue
                
                # No function calls, add the message and we're done
                                conversation.append())))))))))))assistant_message)
                            break
            
            # Return the complete conversation
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "success": True,
                        "conversation": conversation,
                        "function_calls": all_function_calls,
                        "function_call_count": function_call_count
                        }
        
        except Exception as e:
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "success": False,
                        "error": str())))))))))))e)
                        }
    
    # === Parallel Function Calling ===
    
                        def request_with_parallel_functions())))))))))))self, messages: List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]],
                        functions: Optional[]]]]]],,,,,,List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]]] = None,
                        model: str = "gpt-4o",
                        temperature: float = 0.7,
                        max_tokens: Optional[]]]]]],,,,,,int] = None) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,,,
                        """
                        Send a request to OpenAI with parallel function calling capabilities.
                        This forces the model to call multiple functions in parallel.
        
        Args:
            messages: The conversation messages
            functions: Function definitions ())))))))))))if None, uses all registered functions)::::::
                model: The model to use for the request
                temperature: The sampling temperature
                max_tokens: The maximum number of tokens to generate
            
        Returns:
            The response from OpenAI with parallel function calls
            """
        try:
            # Check API key
            if not self.api_key:
            raise ValueError())))))))))))"No OpenAI API key provided.")
            
            # Get function definitions if not provided::::
            if functions is None:
                functions = self.get_registered_function_definitions()))))))))))))
            
            # Convert functions to tools format for API
                tools = []]]]]],,,,,,],,,,
            for func in functions:
                tools.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "type": "function",
                "function": func
                })
            
            # Create the request with parallel function calling
                response = openai.chat.completions.create())))))))))))
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",  # Use auto to allow multiple function calls
                temperature=temperature,
                max_tokens=max_tokens,
                parallel_tool_calls=True  # Enable parallel function calling
                )
            
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "response": response,
                "content": response.choices[]]]]]],,,,,,0].message.content,
                "tool_calls": response.choices[]]]]]],,,,,,0].message.tool_calls,
                "finish_reason": response.choices[]]]]]],,,,,,0].finish_reason,,
                }
        
        except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": str())))))))))))e)
                }
    
                def execute_parallel_function_calls())))))))))))self, tool_calls) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,
                """
                Execute multiple function calls in parallel.
        
        Args:
            tool_calls: The tool_calls from the OpenAI response
            
        Returns:
            Results of all function executions
            """
        try:
            import concurrent.futures
            
            # Initialize results
            results = []]]]]],,,,,,],,,,
            
            # Use thread pool to execute functions in parallel
            with concurrent.futures.ThreadPoolExecutor())))))))))))) as executor:
                # Submit all function calls to the executor
                future_to_tool_call = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                executor.submit())))))))))))self.execute_function_call, tool_call): tool_call
                for tool_call in tool_calls if tool_call.type == "function"
                }
                
                # Collect results as they complete:
                for future in concurrent.futures.as_completed())))))))))))future_to_tool_call):
                    tool_call = future_to_tool_call[]]]]]],,,,,,future],
                    try:
                        result = future.result()))))))))))))
                        results.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "tool_call_id": tool_call.id,
                        "function_name": tool_call.function.name,
                        "result": result,
                        "success": result[]]]]]],,,,,,"success"],,
                        })
                    except Exception as e:
                        results.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "tool_call_id": tool_call.id,
                        "function_name": tool_call.function.name,
                        "error": str())))))))))))e),
                        "success": False
                        })
            
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "success": True,
                        "results": results
                        }
        
        except Exception as e:
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "success": False,
                        "error": str())))))))))))e)
                        }
    
                        def process_with_parallel_functions())))))))))))self, messages: List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]],
                        functions: Optional[]]]]]],,,,,,List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]]] = None,
                        model: str = "gpt-4o",
                        temperature: float = 0.7,
                        max_tokens: Optional[]]]]]],,,,,,int] = None,
                        auto_execute: bool = True) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,
                        """
                        Process a conversation with parallel function calling, optionally executing the functions.
        
        Args:
            messages: The conversation messages
            functions: Function definitions ())))))))))))if None, uses all registered functions)::::::
                model: The model to use for the request
                temperature: The sampling temperature
                max_tokens: The maximum number of tokens to generate
                auto_execute: Whether to automatically execute the called functions
            
        Returns:
            The processed response with parallel function results
            """
        try:
            # Make the request with parallel function calling
            response = self.request_with_parallel_functions())))))))))))
            messages=messages,
            functions=functions,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
            )
            
            if not response[]]]]]],,,,,,"success"],,:,,
            return response
            
            # Check if the model wants to call functions:
            if response[]]]]]],,,,,,"finish_reason"],, == "tool_calls" and response[]]]]]],,,,,,"tool_calls"],,:,
            tool_calls = response[]]]]]],,,,,,"tool_calls"],,
                
                # Execute functions in parallel if auto_execute is enabled::
                if auto_execute:
                    execution_results = self.execute_parallel_function_calls())))))))))))tool_calls)
                    
                    if not execution_results[]]]]]],,,,,,"success"],,:,,
            return execution_results
                    
            function_results = execution_results[]]]]]],,,,,,"results"],
                else:
                    # Just include the function call information without executing
                    function_results = []]]]]],,,,,,],,,,
                    for tool_call in tool_calls:
                        if tool_call.type == "function":
                            function_results.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "tool_call_id": tool_call.id,
                            "function_name": tool_call.function.name,
                            "arguments": json.loads())))))))))))tool_call.function.arguments),
                            "executed": False
                            })
                
                # Prepare the response with parallel function results
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "success": True,
                        "initial_response": response[]]]]]],,,,,,"response"],
                        "tool_calls": tool_calls,
                        "function_results": function_results,
                        "executed": auto_execute,
                        "parallel": True
                        }
            
            # No function calls, just return the regular response
                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "success": True,
                    "response": response[]]]]]],,,,,,"response"],
                    "content": response[]]]]]],,,,,,"content"],
                    "finish_reason": response[]]]]]],,,,,,"finish_reason"],,
                    }
        
        except Exception as e:
                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "success": False,
                    "error": str())))))))))))e)
                    }
    
                    def complete_parallel_function_conversation())))))))))))self, messages: List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]],
                    functions: Optional[]]]]]],,,,,,List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]]] = None,
                    model: str = "gpt-4o",
                    temperature: float = 0.7,
                    max_tokens: Optional[]]]]]],,,,,,int] = None,
                    max_function_calls: int = 10) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,
                    """
                    Complete a full conversation with multiple rounds of parallel function calls.
        
        Args:
            messages: The initial conversation messages
            functions: Function definitions ())))))))))))if None, uses all registered functions)::::::
                model: The model to use for the request
                temperature: The sampling temperature
                max_tokens: The maximum number of tokens to generate
                max_function_calls: Maximum number of function call rounds to prevent infinite loops
            
        Returns:
            The complete conversation with all parallel function calls and responses
            """
        try:
            # Create a copy of the messages to work with
            conversation = messages.copy()))))))))))))
            
            # Get function definitions if not provided::::
            if functions is None:
                functions = self.get_registered_function_definitions()))))))))))))
            
            # Initialize tracking variables
                function_call_rounds = 0
                all_function_calls = []]]]]],,,,,,],,,,
            
            while function_call_rounds < max_function_calls:
                # Make the request with parallel functions
                response = self.request_with_parallel_functions())))))))))))
                messages=conversation,
                functions=functions,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
                )
                
                if not response[]]]]]],,,,,,"success"],,:,,
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": response[]]]]]],,,,,,"error"],
                "conversation": conversation,
                "function_calls": all_function_calls
                }
                
                # Add the assistant's response to the conversation
                assistant_message = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "role": "assistant",
                "content": response[]]]]]],,,,,,"content"] if response[]]]]]],,,,,,"content"] else None,,
                }
                
                # Check if the model wants to call functions::
                if response[]]]]]],,,,,,"finish_reason"],, == "tool_calls" and response[]]]]]],,,,,,"tool_calls"],,:,
                tool_calls = response[]]]]]],,,,,,"tool_calls"],,
                function_call_rounds += 1
                    
                    # Add tool calls to the assistant message
                assistant_message[]]]]]],,,,,,"tool_calls"],, = []]]]]],,,,,,],,,,
                    for tool_call in tool_calls:
                        if tool_call.type == "function":
                            assistant_message[]]]]]],,,,,,"tool_calls"],,.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "id": tool_call.id,
                            "type": "function",
                            "function": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                            }
                            })
                    
                    # Add the assistant's message to the conversation
                            conversation.append())))))))))))assistant_message)
                    
                    # Execute functions in parallel
                            execution_results = self.execute_parallel_function_calls())))))))))))tool_calls)
                    
                            if not execution_results[]]]]]],,,,,,"success"],,:,,
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "success": False,
                        "error": execution_results[]]]]]],,,,,,"error"],
                        "conversation": conversation,
                        "function_calls": all_function_calls
                        }
                    
                    # Add the results to tracking
                    for result in execution_results[]]]]]],,,,,,"results"],:
                        tool_call = next())))))))))))())))))))))))tc for tc in tool_calls if tc.id == result[]]]]]],,,,,,"tool_call_id"]), None),
                        :
                        if tool_call:
                            all_function_calls.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "function_name": tool_call.function.name,
                            "arguments": json.loads())))))))))))tool_call.function.arguments),
                            "result": result.get())))))))))))"result", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get())))))))))))"result", None),
                            "success": result.get())))))))))))"success", False)
                            })
                            
                            # Convert the result to a string
                            result_data = result.get())))))))))))"result", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get())))))))))))"result", None)
                            if isinstance())))))))))))result_data, ())))))))))))dict, list)):
                                result_str = json.dumps())))))))))))result_data)
                            else:
                                result_str = str())))))))))))result_data) if result_data is not None else ""
                            
                            # Add the function result as a tool response message
                            conversation.append()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": result_str
                                })
                    
                    # Continue the conversation with another API call
                                continue
                
                # No function calls, add the message and we're done
                                conversation.append())))))))))))assistant_message)
                                break
            
            # Return the complete conversation
                            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "success": True,
                            "conversation": conversation,
                            "function_calls": all_function_calls,
                            "function_call_rounds": function_call_rounds
                            }
        
        except Exception as e:
                            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "success": False,
                            "error": str())))))))))))e)
                            }
    
    # === Tool Use ())))))))))))Code Interpreter, etc.) ===
    
                            def request_with_code_interpreter())))))))))))self, messages: List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]],
                            model: str = "gpt-4o",
                            temperature: float = 0.7,
                            max_tokens: Optional[]]]]]],,,,,,int] = None) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,,,
                            """
                            Send a request to OpenAI with code interpreter tool.
        
        Args:
            messages: The conversation messages
            model: The model to use for the request
            temperature: The sampling temperature
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            The response from OpenAI
            """
        try:
            # Check API key
            if not self.api_key:
            raise ValueError())))))))))))"No OpenAI API key provided.")
            
            # Create the request with code interpreter
            response = openai.chat.completions.create())))))))))))
            model=model,
            messages=messages,
            tools=[]]]]]],,,,,,
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "type": "code_interpreter"
            }
            ],
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "response": response,
            "content": response.choices[]]]]]],,,,,,0].message.content,
            "tool_calls": response.choices[]]]]]],,,,,,0].message.tool_calls,
            "finish_reason": response.choices[]]]]]],,,,,,0].finish_reason,,
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))))))))e)
            }
    
            def request_with_retrieval())))))))))))self, messages: List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]],
            model: str = "gpt-4o",
            temperature: float = 0.7,
            max_tokens: Optional[]]]]]],,,,,,int] = None) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,,,
            """
            Send a request to OpenAI with retrieval tool.
        
        Args:
            messages: The conversation messages
            model: The model to use for the request
            temperature: The sampling temperature
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            The response from OpenAI
            """
        try:
            # Check API key
            if not self.api_key:
            raise ValueError())))))))))))"No OpenAI API key provided.")
            
            # Create the request with retrieval
            response = openai.chat.completions.create())))))))))))
            model=model,
            messages=messages,
            tools=[]]]]]],,,,,,
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "type": "retrieval"
            }
            ],
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "response": response,
            "content": response.choices[]]]]]],,,,,,0].message.content,
            "tool_calls": response.choices[]]]]]],,,,,,0].message.tool_calls,
            "finish_reason": response.choices[]]]]]],,,,,,0].finish_reason,,
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))))))))e)
            }
    
            def request_with_file_search())))))))))))self, messages: List[]]]]]],,,,,,Dict[]]]]]],,,,,,str, Any]],
            file_ids: List[]]]]]],,,,,,str],
            model: str = "gpt-4o",
            temperature: float = 0.7,
            max_tokens: Optional[]]]]]],,,,,,int] = None) -> Dict[]]]]]],,,,,,str, Any]:,,,,,,,,,
            """
            Send a request to OpenAI with file search tool.
        
        Args:
            messages: The conversation messages
            file_ids: List of file IDs to search
            model: The model to use for the request
            temperature: The sampling temperature
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            The response from OpenAI
            """
        try:
            # Check API key
            if not self.api_key:
            raise ValueError())))))))))))"No OpenAI API key provided.")
            
            # Create the request with file search
            response = openai.chat.completions.create())))))))))))
            model=model,
            messages=messages,
            tools=[]]]]]],,,,,,
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "type": "file_search"
            }
            ],
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
            file_ids=file_ids
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "response": response,
            "content": response.choices[]]]]]],,,,,,0].message.content,
            "tool_calls": response.choices[]]]]]],,,,,,0].message.tool_calls,
            "finish_reason": response.choices[]]]]]],,,,,,0].finish_reason,,
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))))))))e)
            }

# === Example Usage ===

# Define some example functions for testing
def get_weather())))))))))))location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and state, e.g. "San Francisco, CA"
        unit: The unit of temperature, either "celsius" or "fahrenheit"
        
    Returns:
        A string with the weather information
        """
    # This is a mock implementation
        import random
        temp = random.randint())))))))))))0, 30) if unit == "celsius" else random.randint())))))))))))32, 86)
        conditions = random.choice())))))))))))[]]]]]],,,,,,"sunny", "cloudy", "rainy", "snowy"])
        return f"The current weather in {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}location} is {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}temp}°{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}unit[]]]]]],,,,,,0].upper()))))))))))))} and {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}conditions}."
:
def calculate_mortgage())))))))))))principal: float, interest_rate: float, years: int) -> dict:
    """
    Calculate monthly mortgage payment.
    
    Args:
        principal: The loan amount
        interest_rate: Annual interest rate ())))))))))))percentage)
        years: Term of the mortgage in years
        
    Returns:
        Dictionary with payment info
        """
    # Convert annual interest rate to monthly and decimal form
        monthly_rate = interest_rate / 100 / 12
    
    # Calculate number of payments
        payments = years * 12
    
    # Calculate monthly payment using formula
    if monthly_rate == 0:
        monthly_payment = principal / payments
    else:
        monthly_payment = principal * ())))))))))))monthly_rate * ())))))))))))1 + monthly_rate) ** payments) / ())))))))))))())))))))))))1 + monthly_rate) ** payments - 1)
    
    # Calculate total cost
        total_cost = monthly_payment * payments
    
        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "monthly_payment": round())))))))))))monthly_payment, 2),
        "total_cost": round())))))))))))total_cost, 2),
        "total_interest": round())))))))))))total_cost - principal, 2)
        }

def find_restaurants())))))))))))cuisine: str, location: str, price_range: str = "moderate") -> list:
    """
    Find restaurants based on criteria.
    
    Args:
        cuisine: Type of food ())))))))))))e.g., Italian, Chinese)
        location: City or neighborhood
        price_range: Price level ())))))))))))budget, moderate, expensive)
        
    Returns:
        List of restaurant recommendations
        """
    # Mock implementation
        restaurants = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "Italian": []]]]]],,,,,,"Pasta Paradise", "Mamma Mia", "The Italian Job"],
        "Chinese": []]]]]],,,,,,"Golden Dragon", "Wok This Way", "Dim Sum Palace"],
        "Mexican": []]]]]],,,,,,"Taco Time", "Guacamole Grill", "Burrito Bros"],
        "Indian": []]]]]],,,,,,"Curry House", "Spice Garden", "Taj Mahal"],
        "Japanese": []]]]]],,,,,,"Sushi Star", "Ramen House", "Tokyo Express"]
        }
    
    if cuisine in restaurants:
        return []]]]]],,,,,,{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": name,
        "cuisine": cuisine,
        "location": location,
        "price_range": price_range
        } for name in restaurants.get())))))))))))cuisine, []]]]]],,,,,,],,,,)]
    else:
        return []]]]]],,,,,,],,,,

def main())))))))))))):
    """
    Example usage of the OpenAI Function Calling extension.
    """
    # Check if we have an API key
    api_key = os.environ.get())))))))))))"OPENAI_API_KEY"):
    if not api_key:
        print())))))))))))"No OpenAI API key found in environment variables.")
        print())))))))))))"Please set OPENAI_API_KEY in your .env file.")
        return
    
    # Initialize the Function Calling extension
        function_api = OpenAIFunctionCalling())))))))))))api_key=api_key)
    
    # Register example functions
        function_api.register_function())))))))))))get_weather)
        function_api.register_function())))))))))))calculate_mortgage)
        function_api.register_function())))))))))))find_restaurants)
    
        print())))))))))))"\nRegistered functions:")
    for name, data in function_api.registered_functions.items())))))))))))):
        print())))))))))))f"- {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}name}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}data[]]]]]],,,,,,'definition'][]]]]]],,,,,,'description']}")
    
    # Example 1: Basic function calling
        print())))))))))))"\nExample 1: Basic function calling")
        print())))))))))))"Asking about weather in San Francisco...")
    
        response = function_api.process_with_function_calling())))))))))))
        messages=[]]]]]],,,,,,
        {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "system", "content": "You are a helpful assistant."},
        {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        temperature=0.7
        )
    
        if response[]]]]]],,,,,,"success"],,:,,
        if "function_results" in response:
            print())))))))))))"\nFunction called:")
            for result in response[]]]]]],,,,,,"function_results"]:
                print())))))))))))f"- Function: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}result[]]]]]],,,,,,'function_name']}")
                print())))))))))))f"- Result: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}result.get())))))))))))'result', {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get())))))))))))'result', 'N/A')}")
        else:
            print())))))))))))"\nResponse ())))))))))))no function called):")
            print())))))))))))response[]]]]]],,,,,,"content"])
    else:
        print())))))))))))f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}response.get())))))))))))'error')}")
    
    # Example 2: Parallel function calling
        print())))))))))))"\nExample 2: Parallel function calling")
        print())))))))))))"Asking about weather and restaurants in multiple cities...")
    
        response = function_api.process_with_parallel_functions())))))))))))
        messages=[]]]]]],,,,,,
        {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "system", "content": "You are a helpful assistant."},
        {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "user", "content": "What's the weather like in New York and Tokyo? Also, can you recommend some Italian restaurants in Chicago?"}
        ],
        temperature=0.7
        )
    
        if response[]]]]]],,,,,,"success"],,:,,
        if "function_results" in response:
            print())))))))))))"\nFunctions called in parallel:")
            for result in response[]]]]]],,,,,,"function_results"]:
                print())))))))))))f"- Function: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}result[]]]]]],,,,,,'function_name']}")
                print())))))))))))f"- Result: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}result.get())))))))))))'result', {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get())))))))))))'result', 'N/A')}")
        else:
            print())))))))))))"\nResponse ())))))))))))no functions called):")
            print())))))))))))response[]]]]]],,,,,,"content"])
    else:
        print())))))))))))f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}response.get())))))))))))'error')}")
    
    # Example 3: Complete conversation with multiple function calls
        print())))))))))))"\nExample 3: Complete conversation with function calls")
        print())))))))))))"Having a conversation about mortgages and weather...")
    
        response = function_api.complete_function_conversation())))))))))))
        messages=[]]]]]],,,,,,
        {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "system", "content": "You are a helpful financial and travel assistant."},
        {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "user", "content": "I'm planning to buy a house for $350,000. What would my monthly payment be with a 4.5% interest rate for 30 years? Also, what's the weather like in Miami right now?"}
        ],
        temperature=0.7,
        max_function_calls=5
        )
    
        if response[]]]]]],,,,,,"success"],,:,,
        print())))))))))))"\nComplete conversation:")
        conversation = response[]]]]]],,,,,,"conversation"]
        for message in conversation:
            role = message[]]]]]],,,,,,"role"]
            content = message.get())))))))))))"content", "")
            
            if role == "tool":
                print())))))))))))f"\n{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}role.upper()))))))))))))} ()))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}message.get())))))))))))'name', 'unknown')}): {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}content}")
            else:
                print())))))))))))f"\n{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}role.upper()))))))))))))}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}content if content else '[]]]]]],,,,,,No content, using function call]'}")
                
                # Print tool calls if present:
                if "tool_calls" in message:
                    for tool_call in message[]]]]]],,,,,,"tool_calls"],,:
                        print())))))))))))f"  Function: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}tool_call[]]]]]],,,,,,'function'][]]]]]],,,,,,'name']}")
                        print())))))))))))f"  Arguments: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}tool_call[]]]]]],,,,,,'function'][]]]]]],,,,,,'arguments']}")
    else:
        print())))))))))))f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}response.get())))))))))))'error')}")

if __name__ == "__main__":
    main()))))))))))))