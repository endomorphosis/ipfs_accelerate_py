#!/usr/bin/env python3
"""
API Backend Converter - Python to TypeScript

This script converts Python API backends from ipfs_accelerate_py to TypeScript backends
for ipfs_accelerate_js. It generates TypeScript class implementations that follow the
established patterns and extend the BaseApiBackend class.
"""

import os
import re
import ast
import argparse
import textwrap
import logging
from typing import Dict, List, Optional, Set, Tuple, Any

# Configuration
PYTHON_BASE_DIR = "ipfs_accelerate_py/api_backends"
TS_BASE_DIR = "ipfs_accelerate_js/src/api_backends"

# TypeScript type mappings
TYPE_MAPPINGS = {
    "str": "string",
    "int": "number",
    "float": "number",
    "bool": "boolean",
    "dict": "Record<string, any>",
    "list": "Array<any>",
    "Dict": "Record<string, any>",
    "List": "Array<any>",
    "None": "null",
    "Optional": "Optional",
    "Union": "Union",
    "Any": "any",
    "Tuple": "Tuple",
    "Set": "Set",
}

# Method mappings from Python to TypeScript
METHOD_MAPPINGS = {
    "create_endpoint_handler": "createEndpointHandler",
    "test_endpoint": "testEndpoint",
    "make_post_request": "makePostRequest",
    "make_stream_request": "makeStreamRequest",
    "get_api_key": "getApiKey",
    "get_default_model": "getDefaultModel",
    "is_compatible_model": "isCompatibleModel",
    "process_queue": "_processQueue",
}

class APIBackendConverter:
    def __init__(self, python_file: str, output_dir: str, dry_run: bool = False):
        """Initialize the converter with source file and output directory."""
        self.python_file = python_file
        self.output_dir = output_dir
        self.dry_run = dry_run
        
        # Parse backend name from filename
        self.backend_name = os.path.splitext(os.path.basename(python_file))[0]
        self.backend_class_name = self._to_camel_case(self.backend_name)
        
        # Output files
        self.ts_dir = os.path.join(output_dir, self.backend_name)
        self.main_file = os.path.join(self.ts_dir, f"{self.backend_name}.ts")
        self.index_file = os.path.join(self.ts_dir, "index.ts")
        self.types_file = os.path.join(self.ts_dir, "types.ts")
        
        # AST nodes and metadata
        self.ast = None
        self.class_node = None
        self.imports = set()
        self.methods = []
        self.constructor_params = []
        self.class_properties = []
        self.api_specific_types = []
        
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        # Handle special cases for known backends
        known_backends = {
            "ollama": "Ollama",
            "groq": "Groq",
            "hf_tei": "HfTei",
            "hf_tgi": "HfTgi",
            "gemini": "Gemini",
            "vllm": "VLLM",
            "opea": "OPEA",
            "ovms": "OVMS",
            "openai": "OpenAI",
            "claude": "Claude",
        }
        
        if snake_str.lower() in known_backends:
            return known_backends[snake_str.lower()]
            
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)
    
    def _to_camel_case_lower(self, snake_str: str) -> str:
        """Convert snake_case to camelCase (first letter lowercase)."""
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])
    
    def parse_python_file(self) -> bool:
        """Parse the Python file and extract relevant information."""
        print(f"Parsing {self.python_file}...")
        
        try:
            with open(self.python_file, "r") as f:
                source = f.read()
                
            # Clean up common syntax issues in the source files
            source = self._clean_source(source)
            
            # Parse Python code into an AST
            self.ast = ast.parse(source)
            
            # Find the main class in the file
            for node in ast.walk(self.ast):
                if isinstance(node, ast.ClassDef):
                    # Assume the main class is named after the file or contains backend name
                    if node.name.lower() == self.backend_name.lower() or self.backend_name.lower() in node.name.lower():
                        self.class_node = node
                        break
            
            if not self.class_node:
                print(f"Warning: Could not find an exact class match in {self.python_file}")
                # Fall back to any class if available
                for node in ast.walk(self.ast):
                    if isinstance(node, ast.ClassDef):
                        self.class_node = node
                        print(f"Using class {node.name} as fallback")
                        break
                        
            if not self.class_node:
                print(f"Error: Could not find any class in {self.python_file}")
                return False

            # Extract methods, constructor params, and class properties
            self._extract_class_info()
            
            return True
            
        except SyntaxError as e:
            print(f"Syntax error in {self.python_file}: {e}")
            print("Consider manually fixing syntax errors in the Python file before conversion.")
            return False
        except Exception as e:
            print(f"Error parsing {self.python_file}: {e}")
            return False
            
    def _clean_source(self, source: str) -> str:
        """Clean up common syntax issues in the source."""
        # Remove extra closing parentheses (a common error in the files)
        source = re.sub(r'\){2,}', ')', source)
        
        # Fix indentation issues
        lines = source.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix excessive indentation on import statements
            if re.match(r'^\s{4,}(import|from)', line):
                fixed_line = re.sub(r'^\s{4,}', '', line)
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
                
        return '\n'.join(fixed_lines)
    
    def _extract_class_info(self):
        """Extract methods, constructor params, and class properties from the class node."""
        for node in self.class_node.body:
            if isinstance(node, ast.FunctionDef):
                self._extract_method_info(node)
            elif isinstance(node, ast.Assign):
                self._extract_property_info(node)
    
    def _extract_method_info(self, node: ast.FunctionDef):
        """Extract information about a method."""
        method_name = node.name
        
        # Skip private methods (starting with underscore) except _process_queue
        if method_name.startswith("_") and method_name != "_process_queue":
            return
        
        # Extract parameters
        params = []
        for arg in node.args.args:
            if arg.arg != "self":
                param_name = arg.arg
                param_type = "any"
                
                # Try to extract type annotation if available
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        raw_type = arg.annotation.id
                        param_type = TYPE_MAPPINGS.get(raw_type, raw_type)
                    elif isinstance(arg.annotation, ast.Subscript):
                        # Handle more complex types like List[str]
                        param_type = self._convert_complex_type(arg.annotation)
                
                # Handle default values
                default_value = None
                if len(node.args.defaults) > 0:
                    defaults_offset = len(node.args.args) - len(node.args.defaults)
                    if arg_index := node.args.args.index(arg) >= defaults_offset:
                        default_idx = arg_index - defaults_offset
                        if default_idx < len(node.args.defaults):
                            default_node = node.args.defaults[default_idx]
                            default_value = self._get_default_value(default_node)
                
                params.append({
                    "name": param_name,
                    "type": param_type,
                    "default": default_value
                })
        
        # Extract return type if available
        return_type = "any"
        if node.returns:
            if isinstance(node.returns, ast.Name):
                raw_type = node.returns.id
                return_type = TYPE_MAPPINGS.get(raw_type, raw_type)
            elif isinstance(node.returns, ast.Subscript):
                return_type = self._convert_complex_type(node.returns)
                
                # Special case for async generators
                if "AsyncGenerator" in str(node.returns) or "Generator" in str(node.returns):
                    return_type = "AsyncGenerator<StreamChunk>"
                    self.imports.add("StreamChunk")
            elif isinstance(node.returns, ast.Constant) and node.returns.value is None:
                return_type = "void"
        
        # Check for async methods
        is_async = isinstance(node, ast.AsyncFunctionDef) or any(
            isinstance(n, ast.Await) for n in ast.walk(node)
        )
        
        # Check for generator methods
        is_generator = any(
            isinstance(n, ast.Yield) or isinstance(n, ast.YieldFrom)
            for n in ast.walk(node)
        )
        
        # Convert snake_case to camelCase for method names
        ts_method_name = METHOD_MAPPINGS.get(method_name, self._to_camel_case_lower(method_name))
        
        self.methods.append({
            "name": ts_method_name,
            "original_name": method_name,
            "params": params,
            "return_type": return_type,
            "is_async": is_async or "async" in method_name.lower(),
            "is_generator": is_generator or "stream" in method_name.lower(),
            "is_constructor": method_name == "__init__",
        })
        
        # If this is the constructor, save its parameters
        if method_name == "__init__":
            self.constructor_params = params
    
    def _convert_complex_type(self, node):
        """Convert complex Python type annotations to TypeScript."""
        if isinstance(node, ast.Subscript):
            value = node.value
            if isinstance(value, ast.Name):
                container_type = value.id
                ts_container = TYPE_MAPPINGS.get(container_type, container_type)
                
                # Extract inner type
                if isinstance(node.slice, ast.Index):  # Python 3.8 and below
                    inner_type = self._convert_complex_type(node.slice.value)
                else:  # Python 3.9+
                    inner_type = self._convert_complex_type(node.slice)
                
                # Special case for Optional
                if container_type == "Optional":
                    return f"{inner_type} | null | undefined"
                
                # Special case for Union
                if container_type == "Union":
                    if isinstance(inner_type, list):
                        return " | ".join(inner_type)
                    return inner_type
                
                return f"{ts_container}<{inner_type}>"
            
        elif isinstance(node, ast.Name):
            return TYPE_MAPPINGS.get(node.id, node.id)
        
        elif isinstance(node, ast.Tuple):
            # For tuple types like Tuple[str, int]
            elts = []
            for elt in node.elts:
                elts.append(self._convert_complex_type(elt))
            return f"[{', '.join(elts)}]"
        
        return "any"
    
    def _get_default_value(self, node):
        """Convert Python AST default value to TypeScript string."""
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "null"
            elif isinstance(node.value, bool):
                return str(node.value).lower()
            elif isinstance(node.value, (int, float)):
                return str(node.value)
            elif isinstance(node.value, str):
                return f'"{node.value}"'
        elif isinstance(node, ast.Dict):
            return "{}"
        elif isinstance(node, ast.List):
            return "[]"
        elif isinstance(node, ast.Name) and node.id == "None":
            return "null"
        return None
    
    def _extract_property_info(self, node: ast.Assign):
        """Extract information about class properties."""
        # Only process class-level assignments
        for target in node.targets:
            if isinstance(target, ast.Name):
                property_name = target.id
                property_value = "null"
                
                if isinstance(node.value, ast.Constant):
                    if node.value.value is None:
                        property_value = "null"
                    elif isinstance(node.value.value, bool):
                        property_value = str(node.value.value).lower()
                    elif isinstance(node.value.value, (int, float)):
                        property_value = str(node.value.value)
                    elif isinstance(node.value.value, str):
                        property_value = f'"{node.value.value}"'
                elif isinstance(node.value, ast.Dict):
                    property_value = "{}"
                elif isinstance(node.value, ast.List):
                    property_value = "[]"
                
                self.class_properties.append({
                    "name": property_name,
                    "value": property_value
                })
    
    def _analyze_api_specific_types(self):
        """Analyze the code to extract API-specific type definitions."""
        # Look for patterns suggesting custom types
        # This is a simplified approach - a more comprehensive analysis would require
        # deeper semantic analysis of the Python code
        
        # Example: Find data structures that might need API-specific types
        response_patterns = [
            r'response\s*=\s*{([^}]*)}',
            r'data\s*=\s*{([^}]*)}',
            r'request_data\s*=\s*{([^}]*)}',
        ]
        
        with open(self.python_file, "r") as f:
            source = f.read()
        
        # Extract potential fields for response types
        response_fields = set()
        for pattern in response_patterns:
            matches = re.findall(pattern, source, re.DOTALL)
            for match in matches:
                # Extract field names from the dictionary
                field_matches = re.findall(r'[\'"]([a-zA-Z_]+)[\'"]', match)
                response_fields.update(field_matches)
        
        # Create response type with backend-specific customizations
        response_type_fields = [
            {"name": "id", "type": "string"},
            {"name": "object", "type": "string"},
            {"name": "created", "type": "number"},
            {"name": "model", "type": "string"},
            {"name": "choices", "type": "Array<{\n    index: number;\n    message?: {\n      role: string;\n      content: string;\n    };\n    delta?: {\n      content: string;\n    };\n    finish_reason: string;\n  }>"},
            {"name": "usage", "type": "{\n    prompt_tokens: number;\n    completion_tokens: number;\n    total_tokens: number;\n  }"},
        ]
        
        # Add backend-specific response fields
        if self.backend_name.lower() == "claude":
            response_type_fields.extend([
                {"name": "stop_reason", "type": "string", "optional": True},
                {"name": "stop_sequence", "type": "string", "optional": True},
                {"name": "input_tokens", "type": "number", "optional": True},
                {"name": "output_tokens", "type": "number", "optional": True}
            ])
        elif self.backend_name.lower() == "gemini":
            response_type_fields.extend([
                {"name": "candidates", "type": "Array<{\n    content: {\n      parts: Array<{\n        text: string;\n      }>;\n    };\n    finishReason: string;\n    safetyRatings: Array<{\n      category: string;\n      probability: string;\n    }>;\n  }>", "optional": True},
                {"name": "promptFeedback", "type": "any", "optional": True}
            ])
        elif self.backend_name.lower() == "hf_tei":
            response_type_fields = [
                {"name": "embeddings", "type": "number[][]"},
                {"name": "model", "type": "string"},
                {"name": "dimensions", "type": "number"}
            ]
        
        self.api_specific_types.append({
            "name": f"{self.backend_class_name}Response",
            "fields": response_type_fields
        })
        
        # Create a request type with backend-specific customizations
        request_type_fields = [
            {"name": "model", "type": "string"},
            {"name": "messages", "type": "Message[]", "optional": True},
            {"name": "prompt", "type": "string", "optional": True},
            {"name": "stream", "type": "boolean", "optional": True},
            {"name": "max_tokens", "type": "number", "optional": True},
            {"name": "temperature", "type": "number", "optional": True},
            {"name": "top_p", "type": "number", "optional": True},
        ]
        
        # Add backend-specific request fields
        if self.backend_name.lower() == "claude":
            request_type_fields.extend([
                {"name": "system", "type": "string", "optional": True},
                {"name": "stop_sequences", "type": "string[]", "optional": True},
                {"name": "anthropic_version", "type": "string", "optional": True}
            ])
        elif self.backend_name.lower() == "openai":
            request_type_fields.extend([
                {"name": "functions", "type": "Array<{\n    name: string;\n    description?: string;\n    parameters: any;\n  }>", "optional": True},
                {"name": "function_call", "type": "string | object", "optional": True},
                {"name": "n", "type": "number", "optional": True},
                {"name": "stop", "type": "string | string[]", "optional": True},
                {"name": "presence_penalty", "type": "number", "optional": True},
                {"name": "frequency_penalty", "type": "number", "optional": True},
                {"name": "logit_bias", "type": "Record<string, number>", "optional": True},
                {"name": "user", "type": "string", "optional": True}
            ])
        elif self.backend_name.lower() == "hf_tei":
            request_type_fields = [
                {"name": "inputs", "type": "string | string[]"},
                {"name": "model", "type": "string", "optional": True}
            ]
        
        self.api_specific_types.append({
            "name": f"{self.backend_class_name}Request",
            "fields": request_type_fields
        })
        
        self.imports.add("Message")
    
    def generate_typescript_files(self) -> bool:
        """Generate TypeScript files based on the parsed Python code."""
        try:
            # Create output directory if it doesn't exist
            if not self.dry_run:
                os.makedirs(self.ts_dir, exist_ok=True)
            
            # Analyze API-specific types
            self._analyze_api_specific_types()
            
            # Generate main implementation file
            main_content = self._generate_main_file()
            if not self.dry_run:
                with open(self.main_file, "w") as f:
                    f.write(main_content)
            else:
                print(f"\n--- {self.main_file} ---\n")
                print(main_content)
            
            # Generate index file
            index_content = self._generate_index_file()
            if not self.dry_run:
                with open(self.index_file, "w") as f:
                    f.write(index_content)
            else:
                print(f"\n--- {self.index_file} ---\n")
                print(index_content)
            
            # Generate types file if needed
            if self.api_specific_types:
                types_content = self._generate_types_file()
                if not self.dry_run:
                    with open(self.types_file, "w") as f:
                        f.write(types_content)
                else:
                    print(f"\n--- {self.types_file} ---\n")
                    print(types_content)
            
            print(f"Successfully generated TypeScript files for {self.backend_name}")
            return True
            
        except Exception as e:
            print(f"Error generating TypeScript files: {e}")
            return False
    
    def _generate_main_file(self) -> str:
        """Generate the main TypeScript implementation file."""
        # Add required imports
        imports = [
            "import { BaseApiBackend } from '../base';",
            "import { ApiMetadata, ApiRequestOptions, Message, ChatCompletionResponse, StreamChunk } from '../types';"
        ]
        
        # Import API-specific types if any
        if self.api_specific_types:
            imports.append(f"import {{ {', '.join(t['name'] for t in self.api_specific_types)} }} from './types';")
        
        # Generate class declaration
        class_dec = f"export class {self.backend_class_name} extends BaseApiBackend {{"
        
        # Generate class properties
        properties = []
        for prop in self.class_properties:
            properties.append(f"  private {prop['name']}: any = {prop['value']};")
        
        # Add API endpoint property if not already defined
        if not any(p['name'] == 'apiEndpoint' for p in self.class_properties):
            # Handle special cases for backend endpoints
            if self.backend_name.lower() == 'ollama':
                properties.append(f"  private apiEndpoint: string = 'http://localhost:11434/api/chat';")
            elif self.backend_name.lower() == 'openai':
                properties.append(f"  private apiEndpoint: string = 'https://api.openai.com/v1/chat/completions';")
            elif self.backend_name.lower() == 'claude':
                properties.append(f"  private apiEndpoint: string = 'https://api.anthropic.com/v1/messages';")
            elif self.backend_name.lower() == 'gemini':
                properties.append(f"  private apiEndpoint: string = 'https://generativelanguage.googleapis.com/v1/models';")
            elif self.backend_name.lower() == 'groq':
                properties.append(f"  private apiEndpoint: string = 'https://api.groq.com/openai/v1/chat/completions';")
            elif self.backend_name.lower() == 'hf_tei':
                properties.append(f"  private apiEndpoint: string = 'http://localhost:8080/';")
            elif self.backend_name.lower() == 'hf_tgi':
                properties.append(f"  private apiEndpoint: string = 'http://localhost:8080/generate';")
            else:
                api_name = self.backend_name.replace('_', '')
                properties.append(f"  private apiEndpoint: string = 'https://api.{api_name}.com/v1/chat';")
                
            # Add API version property for specific backends
            if self.backend_name.lower() == 'claude':
                properties.append(f"  private apiVersion: string = '2023-06-01';")
            elif self.backend_name.lower() == 'gemini':
                properties.append(f"  private apiVersion: string = 'v1';")
                
            # Add base URL property for specific backends
            if self.backend_name.lower() in ['hf_tei', 'hf_tgi', 'ollama', 'vllm', 'ovms']:
                properties.append(f"  private apiBase: string = '';")
                properties.append(f"  private useLocalDeployment: boolean = true;")
        
        # If there are no constructor params, add default constructor
        constructor = []
        if not any(m["is_constructor"] for m in self.methods):
            constructor = [
                "  constructor(resources: Record<string, any> = {}, metadata: ApiMetadata = {}) {",
                "    super(resources, metadata);",
                "  }"
            ]
        else:
            # Find the constructor method
            for method in self.methods:
                if method["is_constructor"]:
                    param_strings = []
                    for param in method["params"]:
                        param_str = f"{param['name']}: any"
                        if param["default"] is not None:
                            param_str += f" = {param['default']}"
                        param_strings.append(param_str)
                    
                    constructor = [
                        f"  constructor(resources: Record<string, any> = {{}}, metadata: ApiMetadata = {{}}) {{",
                        "    super(resources, metadata);",
                        "    // Initialize from resources and metadata",
                        "  }"
                    ]
        
        # Generate methods
        methods = []
        for method in self.methods:
            # Skip constructor as we handle it separately
            if method["is_constructor"]:
                continue
            
            # Format parameters
            param_strings = []
            for param in method["params"]:
                param_str = f"{param['name']}: {param['type']}"
                if param["default"] is not None:
                    param_str += f" = {param['default']}"
                param_strings.append(param_str)
            
            # Handle special method cases
            if method["name"] == "getApiKey":
                methods.append(self._generate_get_api_key_method())
                continue
            elif method["name"] == "getDefaultModel":
                methods.append(self._generate_get_default_model_method())
                continue
            elif method["name"] == "isCompatibleModel":
                methods.append(self._generate_is_compatible_model_method())
                continue
            
            # Method signature
            method_prefix = "async " if method["is_async"] or method["is_generator"] or method["name"] in ["makePostRequest", "chat", "testEndpoint"] else ""
            generator_prefix = "*" if method["is_generator"] or "stream" in method["name"].lower() else ""
            
            # Improve return types
            if method["name"] == "makePostRequest":
                return_type = "Promise<any>"
            elif method["name"] == "makeStreamRequest":
                return_type = "AsyncGenerator<StreamChunk>"
            elif method["name"] == "chat":
                return_type = "Promise<ChatCompletionResponse>"
            elif method["name"] == "streamChat":
                return_type = "AsyncGenerator<StreamChunk>"
            elif method["name"] == "testEndpoint":
                return_type = "Promise<boolean>"
            elif method["name"] == "createEndpointHandler":
                return_type = "(data: any) => Promise<any>"
            else:
                return_type = method["return_type"]
            
            method_signature = f"  {method_prefix}{generator_prefix}{method['name']}({', '.join(param_strings)}): {return_type} {{"
            
            # Method implementation (simplified template)
            implementation = []
            if method["name"] == "createEndpointHandler":
                implementation = self._generate_create_endpoint_handler_method()
            elif method["name"] == "testEndpoint":
                implementation = self._generate_test_endpoint_method()
            elif method["name"] == "makePostRequest":
                implementation = self._generate_make_post_request_method()
            elif method["name"] == "makeStreamRequest":
                implementation = self._generate_make_stream_request_method()
            elif method["name"] == "chat":
                implementation = self._generate_chat_method()
            elif method["name"] == "streamChat":
                implementation = self._generate_stream_chat_method()
            else:
                implementation = [
                    "    // TODO: Implement this method based on the Python source",
                    f"    // Original Python method: {method['original_name']}",
                    "    throw new Error('Method not implemented');"
                ]
            
            methods.append(method_signature)
            methods.extend(implementation)
            methods.append("  }")
            methods.append("")
        
        # Combine all sections
        sections = [
            "\n".join(imports),
            "",
            class_dec,
            "",
            "\n".join(properties),
            "",
            "\n".join(constructor),
            "",
            "\n".join(methods),
            "}"
        ]
        
        return "\n".join(sections)
    
    def _generate_get_api_key_method(self) -> str:
        """Generate the getApiKey method."""
        api_key_env = f"{self.backend_name.upper()}_API_KEY"
        api_key_snake = f"{self.backend_name}_api_key"
        api_key_camel = self._to_camel_case_lower(f"{self.backend_name}_api_key")
        
        return textwrap.dedent(f"""
          protected getApiKey(metadata: ApiMetadata): string {{
            return metadata.{api_key_snake} || 
                   metadata.{api_key_camel} || 
                   (typeof process !== 'undefined' ? process.env.{api_key_env} || '' : '');
          }}
        """).strip()
    
    def _generate_get_default_model_method(self) -> str:
        """Generate the getDefaultModel method."""
        # Use appropriate default models for known backends
        default_models = {
            "ollama": "llama2",
            "openai": "gpt-3.5-turbo",
            "claude": "claude-3-sonnet-20240229",
            "groq": "llama2-70b-4096",
            "gemini": "gemini-pro",
            "hf_tei": "BAAI/bge-small-en-v1.5",
            "hf_tgi": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }
        
        default_model = default_models.get(self.backend_name.lower(), f"{self.backend_name}-model")
        
        return textwrap.dedent(f"""
          protected getDefaultModel(): string {{
            return "{default_model}";
          }}
        """).strip()
    
    def _generate_is_compatible_model_method(self) -> str:
        """Generate the isCompatibleModel method."""
        # Use appropriate model compatibility checks for known backends
        if self.backend_name.lower() == "ollama":
            return textwrap.dedent("""
              isCompatibleModel(model: string): boolean {
                // Ollama supports various models
                return (
                  model.startsWith('llama') || 
                  model.startsWith('mistral') || 
                  model.startsWith('gemma') || 
                  model.toLowerCase().includes('ollama:')
                );
              }
            """).strip()
        
        elif self.backend_name.lower() == "openai":
            return textwrap.dedent("""
              isCompatibleModel(model: string): boolean {
                // OpenAI models
                return (
                  model.startsWith('gpt-') || 
                  model.startsWith('dall-e') || 
                  model.startsWith('text-embedding') ||
                  model.toLowerCase().includes('openai')
                );
              }
            """).strip()
            
        elif self.backend_name.lower() == "claude":
            return textwrap.dedent("""
              isCompatibleModel(model: string): boolean {
                // Claude models
                return (
                  model.startsWith('claude-') || 
                  model.toLowerCase().includes('anthropic') ||
                  model.toLowerCase().includes('claude')
                );
              }
            """).strip()
            
        elif self.backend_name.lower() == "groq":
            return textwrap.dedent("""
              isCompatibleModel(model: string): boolean {
                // Groq models - typically using either OpenAI compatible models or Llama variants
                return (
                  model.startsWith('llama') || 
                  model.startsWith('mistral') || 
                  model.startsWith('mixtral') || 
                  model.toLowerCase().includes('groq')
                );
              }
            """).strip()
            
        elif self.backend_name.lower() == "hf_tei":
            return textwrap.dedent("""
              isCompatibleModel(model: string): boolean {
                // HuggingFace embedding models
                return (
                  model.includes('/') || // Most HF models have a namespace/model format
                  model.toLowerCase().includes('embed') ||
                  model.toLowerCase().includes('sentence') ||
                  model.toLowerCase().includes('text-embedding')
                );
              }
            """).strip()
            
        elif self.backend_name.lower() == "hf_tgi":
            return textwrap.dedent("""
              isCompatibleModel(model: string): boolean {
                // HuggingFace text generation models
                return (
                  model.includes('/') || // Most HF models have a namespace/model format
                  model.startsWith('llama') ||
                  model.startsWith('mistral') ||
                  model.startsWith('gpt-') ||
                  model.startsWith('falcon')
                );
              }
            """).strip()
            
        else:
            return textwrap.dedent(f"""
              isCompatibleModel(model: string): boolean {{
                // Return true if this model is compatible with this backend
                return model.toLowerCase().includes('{self.backend_name.lower()}');
              }}
            """).strip()
    
    def _generate_create_endpoint_handler_method(self) -> List[str]:
        """Generate the createEndpointHandler method."""
        return [
            "    return async (data: any) => {",
            "      try {",
            "        return await this.makePostRequest(data);",
            "      } catch (error) {",
            "        throw this.createApiError(`${this.constructor.name} endpoint error: ${error.message}`, 500);",
            "      }",
            "    };"
        ]
    
    def _generate_test_endpoint_method(self) -> List[str]:
        """Generate the testEndpoint method."""
        return [
            "    try {",
            "      const apiKey = this.getApiKey(this.metadata);",
            "      if (!apiKey) {",
            "        throw this.createApiError('API key is required', 401, 'authentication_error');",
            "      }",
            "",
            "      // Make a minimal request to verify the endpoint works",
            "      const testRequest = {",
            "        model: this.getDefaultModel(),",
            "        messages: [{ role: 'user', content: 'Hello' }],",
            "        max_tokens: 5",
            "      };",
            "",
            "      await this.makePostRequest(testRequest, apiKey, { timeoutMs: 5000 });",
            "      return true;",
            "    } catch (error) {",
            "      console.error(`${this.constructor.name} endpoint test failed:`, error);",
            "      return false;",
            "    }"
        ]
    
    def _generate_make_post_request_method(self) -> List[str]:
        """Generate the makePostRequest method."""
        return [
            "    const apiKey = apiKey || this.getApiKey(this.metadata);",
            "    if (!apiKey) {",
            "      throw this.createApiError('API key is required', 401, 'authentication_error');",
            "    }",
            "",
            "    // Process with queue and circuit breaker",
            "    return this.processWithQueueAndBackoff(async () => {",
            "      // Prepare request headers",
            "      const headers = {",
            "        'Content-Type': 'application/json',",
            "        'Authorization': `Bearer ${apiKey}`",
            "      };",
            "",
            "      // Prepare request body",
            "      const requestBody = JSON.stringify(data);",
            "",
            "      // Set up timeout",
            "      const timeoutMs = options?.timeoutMs || 30000;",
            "      const controller = new AbortController();",
            "      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);",
            "",
            "      try {",
            "        // Make the request",
            "        const response = await fetch(this.apiEndpoint, {",
            "          method: 'POST',",
            "          headers,",
            "          body: requestBody,",
            "          signal: controller.signal",
            "        });",
            "",
            "        // Check for errors",
            "        if (!response.ok) {",
            "          const errorData = await response.json().catch(() => ({}));",
            "          throw this.createApiError(",
            "            errorData.error?.message || `HTTP error ${response.status}`,",
            "            response.status,",
            "            errorData.error?.type || 'api_error'",
            "          );",
            "        }",
            "",
            "        // Parse response",
            "        const responseData = await response.json();",
            "        return responseData;",
            "      } catch (error) {",
            "        if (error.name === 'AbortError') {",
            "          throw this.createApiError(`Request timed out after ${timeoutMs}ms`, 408, 'timeout_error');",
            "        }",
            "        throw error;",
            "      } finally {",
            "        clearTimeout(timeoutId);",
            "      }",
            "    }, options);"
        ]
    
    def _generate_make_stream_request_method(self) -> List[str]:
        """Generate the makeStreamRequest method."""
        return [
            "    const apiKey = this.getApiKey(this.metadata);",
            "    if (!apiKey) {",
            "      throw this.createApiError('API key is required', 401, 'authentication_error');",
            "    }",
            "",
            "    // Ensure stream option is set",
            "    const streamData = { ...data, stream: true };",
            "",
            "    // Prepare request headers",
            "    const headers = {",
            "      'Content-Type': 'application/json',",
            "      'Authorization': `Bearer ${apiKey}`",
            "    };",
            "",
            "    // Prepare request body",
            "    const requestBody = JSON.stringify(streamData);",
            "",
            "    // Set up timeout",
            "    const timeoutMs = options?.timeoutMs || 30000;",
            "    const controller = new AbortController();",
            "    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);",
            "",
            "    try {",
            "      // Make the request",
            "      const response = await fetch(this.apiEndpoint, {",
            "        method: 'POST',",
            "        headers,",
            "        body: requestBody,",
            "        signal: controller.signal",
            "      });",
            "",
            "      // Check for errors",
            "      if (!response.ok) {",
            "        const errorData = await response.json().catch(() => ({}));",
            "        throw this.createApiError(",
            "          errorData.error?.message || `HTTP error ${response.status}`,",
            "          response.status,",
            "          errorData.error?.type || 'api_error'",
            "        );",
            "      }",
            "",
            "      if (!response.body) {",
            "        throw this.createApiError('Response body is null', 500, 'stream_error');",
            "      }",
            "",
            "      // Process the stream",
            "      const reader = response.body.getReader();",
            "      const decoder = new TextDecoder();",
            "      let buffer = '';",
            "",
            "      while (true) {",
            "        const { done, value } = await reader.read();",
            "        if (done) break;",
            "",
            "        buffer += decoder.decode(value, { stream: true });",
            "        const lines = buffer.split('\\n');",
            "        buffer = lines.pop() || ''; // Keep the last incomplete line in the buffer",
            "",
            "        for (const line of lines) {",
            "          if (line.trim() === '') continue;",
            "          if (line.startsWith('data: ')) {",
            "            const data = line.slice(6);",
            "            if (data === '[DONE]') continue;",
            "",
            "            try {",
            "              const parsed = JSON.parse(data);",
            "              yield {",
            "                content: parsed.choices?.[0]?.delta?.content || '',",
            "                type: 'delta'",
            "              };",
            "            } catch (e) {",
            "              console.warn('Failed to parse stream data:', data);",
            "            }",
            "          }",
            "        }",
            "      }",
            "",
            "      // Handle any remaining data in the buffer",
            "      if (buffer.trim() !== '') {",
            "        if (buffer.startsWith('data: ') && buffer !== 'data: [DONE]') {",
            "          try {",
            "            const data = buffer.slice(6);",
            "            const parsed = JSON.parse(data);",
            "            yield {",
            "              content: parsed.choices?.[0]?.delta?.content || '',",
            "              type: 'delta'",
            "            };",
            "          } catch (e) {",
            "            console.warn('Failed to parse final stream data:', buffer);",
            "          }",
            "        }",
            "      }",
            "    } catch (error) {",
            "      if (error.name === 'AbortError') {",
            "        throw this.createApiError(`Stream request timed out after ${timeoutMs}ms`, 408, 'timeout_error');",
            "      }",
            "      throw error;",
            "    } finally {",
            "      clearTimeout(timeoutId);",
            "    }"
        ]
    
    def _generate_chat_method(self) -> List[str]:
        """Generate the chat method."""
        return [
            "    // Prepare request data",
            "    const modelName = options?.model || this.getDefaultModel();",
            "    ",
            f"    const requestData: {self.backend_class_name}Request = {{",
            "      model: modelName,",
            "      messages,",
            "      max_tokens: options?.maxTokens,",
            "      temperature: options?.temperature,",
            "      top_p: options?.topP",
            "    };",
            "",
            "    // Make the request",
            "    const response = await this.makePostRequest(requestData, undefined, options);",
            "",
            "    // Convert to standard format",
            "    return {",
            "      id: response.id || '',",
            "      model: response.model || model,",
            "      content: response.choices?.[0]?.message?.content || '',",
            "      created: response.created || Date.now(),",
            "      usage: response.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }",
            "    };"
        ]
    
    def _generate_stream_chat_method(self) -> List[str]:
        """Generate the streamChat method."""
        return [
            "    // Prepare request data",
            "    const modelName = options?.model || this.getDefaultModel();",
            "    ",
            f"    const requestData: {self.backend_class_name}Request = {{",
            "      model: modelName,",
            "      messages,",
            "      stream: true,",
            "      max_tokens: options?.maxTokens,",
            "      temperature: options?.temperature,",
            "      top_p: options?.topP",
            "    };",
            "",
            "    // Make the stream request",
            "    const stream = this.makeStreamRequest(requestData, options);",
            "    ",
            "    // Pass through the stream chunks",
            "    return stream;"
        ]
    
    def _generate_index_file(self) -> str:
        """Generate the index.ts file."""
        return f"export * from './{self.backend_name}';\n"
    
    def _generate_types_file(self) -> str:
        """Generate the types.ts file with API-specific types."""
        content = ["import { Message } from '../types';\n"]
        for t in self.api_specific_types:
            # Interface declaration
            content.append(f"export interface {t['name']} {{")
            
            # Fields
            for field in t["fields"]:
                optional = "?" if field.get("optional") else ""
                content.append(f"  {field['name']}{optional}: {field['type']};")
            
            # Close interface
            content.append("}")
            content.append("")
        
        return "\n".join(content)

def main():
    parser = argparse.ArgumentParser(description="Convert Python API backends to TypeScript")
    parser.add_argument("--backend", help="Specific backend to convert (e.g. 'groq')")
    parser.add_argument("--all", action="store_true", help="Convert all available backends")
    parser.add_argument("--python-dir", default=PYTHON_BASE_DIR, help=f"Python API backends directory (default: {PYTHON_BASE_DIR})")
    parser.add_argument("--ts-dir", default=TS_BASE_DIR, help=f"TypeScript output directory (default: {TS_BASE_DIR})")
    parser.add_argument("--dry-run", action="store_true", help="Print output without writing files")
    parser.add_argument("--fix-source", action="store_true", help="Try to automatically fix Python source syntax errors")
    parser.add_argument("--force", action="store_true", help="Force conversion even if parsing has issues")
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    args = parser.parse_args()
    
    if not args.backend and not args.all:
        parser.error("Either --backend or --all must be specified")
    
    if args.all:
        # Get all .py files in the Python directory
        backends = []
        for filename in os.listdir(args.python_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                backends.append(os.path.splitext(filename)[0])
    else:
        backends = [args.backend]
    
    # Set up logging based on verbosity
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    print(f"Converting {len(backends)} API backends...")
    
    successes = 0
    failures = 0
    
    for backend in backends:
        python_file = os.path.join(args.python_dir, f"{backend}.py")
        if not os.path.exists(python_file):
            print(f"Error: {python_file} does not exist")
            failures += 1
            continue
        
        # If fix-source is enabled, try to automatically fix common syntax errors
        if args.fix_source:
            try:
                with open(python_file, "r") as f:
                    source = f.read()
                
                # Apply common fixes
                fixed_source = re.sub(r'\){2,}', ')', source)  # Remove extra closing parentheses
                fixed_source = re.sub(r':\s*:{1,}', ':', fixed_source)  # Fix double colons
                
                # Fix indentation issues
                lines = fixed_source.split('\n')
                fixed_lines = []
                
                for line in lines:
                    # Fix excessive indentation on import statements
                    if re.match(r'^\s{4,}(import|from)', line):
                        fixed_line = re.sub(r'^\s{4,}', '', line)
                        fixed_lines.append(fixed_line)
                    else:
                        fixed_lines.append(line)
                
                fixed_source = '\n'.join(fixed_lines)
                
                # Create a backup of the original file
                if source != fixed_source:
                    print(f"Creating backup of {python_file} as {python_file}.bak")
                    with open(f"{python_file}.bak", "w") as f:
                        f.write(source)
                    
                    # Write the fixed source
                    with open(python_file, "w") as f:
                        f.write(fixed_source)
                    
                    print(f"Applied automatic syntax fixes to {python_file}")
            except Exception as e:
                print(f"Error while trying to fix {python_file}: {e}")
        
        # Run the converter
        converter = APIBackendConverter(python_file, args.ts_dir, args.dry_run)
        success = converter.parse_python_file()
        
        if success or args.force:
            if not success and args.force:
                print(f"Warning: Forcing conversion of {backend} despite parsing issues")
            
            result = converter.generate_typescript_files()
            if result:
                successes += 1
            else:
                failures += 1
        else:
            failures += 1
        
        print()
    
    print(f"Conversion completed with {successes} successes and {failures} failures.")
    
    # If there were failures but some conversions succeeded, suggest running with --force
    if failures > 0 and successes > 0 and not args.force:
        print("Tip: Run with --force to attempt conversion on backends with parsing issues.")
        
    # If all conversions failed, suggest fixing the source files
    if failures == len(backends) and not args.fix_source:
        print("Tip: Run with --fix-source to automatically fix common syntax issues in the Python files.")

if __name__ == "__main__":
    main()