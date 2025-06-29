#!/usr/bin/env python3
"""
Test Generator for TypeScript API Backends

This script generates Jest test files for the TypeScript API backends created by the converter.
"""

import os
import argparse
import json
from typing import Dict, Any, List, Optional


class ApiBackendTestGenerator:
    """Generator for TypeScript API Backend tests"""
    
    def __init__(self, backend_name: str, ts_dir: str, test_dir: str):
        """Initialize the test generator"""
        self.backend_name = backend_name
        self.backend_class_name = self._to_camel_case(backend_name)
        self.ts_dir = os.path.join(ts_dir, backend_name)
        self.test_dir = test_dir
        
        # Output file path
        self.test_file = os.path.join(test_dir, f"{backend_name}.test.ts")
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase"""
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
    
    def generate_test_file(self) -> bool:
        """Generate test file for the backend"""
        # Create the test directory if it doesn't exist
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Generate test content
        test_content = self._generate_test_content()
        
        # Write the test file
        try:
            with open(self.test_file, "w") as f:
                f.write(test_content)
            
            print(f"Successfully generated test file: {self.test_file}")
            return True
        except Exception as e:
            print(f"Error generating test file: {e}")
            return False
    
    def _generate_test_content(self) -> str:
        """Generate test content for the backend"""
        template = f"""// Tests for {self.backend_class_name} API Backend
import {{ {self.backend_class_name} }} from '../src/api_backends/{self.backend_name}';
import {{ ApiMetadata, Message }} from '../src/api_backends/types';

// Mock fetch for testing
global.fetch = jest.fn();

describe('{self.backend_class_name} API Backend', () => {{
  let backend: {self.backend_class_name};
  let mockMetadata: ApiMetadata;
  
  beforeEach(() => {{
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock successful fetch response
    (global.fetch as jest.Mock).mockResolvedValue({{
      ok: true,
      json: jest.fn().mockResolvedValue({{
        id: 'test-id',
        model: 'test-model',
        choices: [
          {{
            message: {{
              role: 'assistant',
              content: 'Hello, I am an AI assistant.'
            }},
            finish_reason: 'stop'
          }}
        ],
        usage: {{
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30
        }}
      }})
    }});
    
    // Set up test data
    mockMetadata = {{
      {self.backend_name}_api_key: 'test-api-key'
    }};
    
    // Create backend instance
    backend = new {self.backend_class_name}({{}}, mockMetadata);
  }});
  
  test('should initialize correctly', () => {{
    expect(backend).toBeDefined();
  }});
  
  test('should get API key from metadata', () => {{
    // @ts-ignore - Testing protected method
    const apiKey = backend.getApiKey(mockMetadata);
    expect(apiKey).toBe('test-api-key');
  }});
  
  test('should get default model', () => {{
    // @ts-ignore - Testing protected method
    const model = backend.getDefaultModel();
    expect(model).toBeDefined();
    expect(typeof model).toBe('string');
  }});
  
  test('should create endpoint handler', () => {{
    const handler = backend.createEndpointHandler();
    expect(handler).toBeDefined();
    expect(typeof handler).toBe('function');
  }});
  
  test('should test endpoint', async () => {{
    const result = await backend.testEndpoint();
    expect(result).toBe(true);
    expect(global.fetch).toHaveBeenCalled();
  }});
  
  test('should handle chat completion', async () => {{
    const messages: Message[] = [
      {{ role: 'user', content: 'Hello' }}
    ];
    
    const response = await backend.chat(messages);
    
    expect(response).toBeDefined();
    expect(response.id).toBe('test-id');
    expect(response.model).toBe('test-model');
    expect(response.content).toBe('Hello, I am an AI assistant.');
    expect(global.fetch).toHaveBeenCalled();
  }});
  
  test('should handle API errors', async () => {{
    // Mock error response
    (global.fetch as jest.Mock).mockResolvedValue({{
      ok: false,
      status: 401,
      json: jest.fn().mockResolvedValue({{
        error: {{
          message: 'Invalid API key',
          type: 'authentication_error'
        }}
      }})
    }});
    
    const messages: Message[] = [
      {{ role: 'user', content: 'Hello' }}
    ];
    
    await expect(backend.chat(messages)).rejects.toThrow();
  }});
  
  test('should check model compatibility', () => {{
    // Test with compatible model
    const compatibleModel = backend.isCompatibleModel('sample-model');
    expect(compatibleModel).toBeDefined();
    
    // Type of result should be boolean
    expect(typeof compatibleModel).toBe('boolean');
  }});
}});
"""
        return template


def main():
    """Main function to run the test generator"""
    parser = argparse.ArgumentParser(description="Generate tests for TypeScript API backends")
    parser.add_argument("--backend", help="Specific backend to generate tests for (e.g. 'ollama')")
    parser.add_argument("--ts-dir", default="ipfs_accelerate_js/src/api_backends", 
                      help="TypeScript API backends directory")
    parser.add_argument("--test-dir", default="ipfs_accelerate_js/test/api_backends", 
                      help="Output directory for test files")
    parser.add_argument("--all", action="store_true", 
                      help="Generate tests for all available backends")
    args = parser.parse_args()
    
    if not args.backend and not args.all:
        parser.error("Either --backend or --all must be specified")
    
    if args.all:
        # Get all subdirectories in the TypeScript directory
        backends = []
        try:
            for item in os.listdir(args.ts_dir):
                if os.path.isdir(os.path.join(args.ts_dir, item)) and not item.startswith("__"):
                    backends.append(item)
        except FileNotFoundError:
            print(f"Error: {args.ts_dir} does not exist")
            return
    else:
        backends = [args.backend]
    
    print(f"Generating tests for {len(backends)} API backends...")
    
    # Create test output directory if it doesn't exist
    os.makedirs(args.test_dir, exist_ok=True)
    
    # Generate tests for each backend
    successes = 0
    failures = 0
    
    for backend in backends:
        generator = ApiBackendTestGenerator(backend, args.ts_dir, args.test_dir)
        if generator.generate_test_file():
            successes += 1
        else:
            failures += 1
    
    print(f"Test generation completed with {successes} successes and {failures} failures.")


if __name__ == "__main__":
    main()