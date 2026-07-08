# API Backend Converter - Python to TypeScript

This tool automates the conversion of Python API backends from `ipfs_accelerate_py` to TypeScript backends for `ipfs_accelerate_js`. It generates TypeScript class implementations that follow the established patterns and extend the `BaseApiBackend` class.

## Quick Start

```bash
# Install: No installation required, script is ready to use

# Basic Usage: Convert a single backend
./convert_api_backends.py --backend ollama

# Complete Conversion: Convert all backends with auto-fixing
./convert_api_backends.py --all --fix-source --force

# View Help
./convert_api_backends.py --help
```

## Getting Started

1. Identify the Python API backends you want to convert
2. Try automatic conversion first:
   ```bash
   ./convert_api_backends.py --backend ollama
   ```
3. If syntax errors occur, try with fixing enabled:
   ```bash 
   ./convert_api_backends.py --backend ollama --fix-source
   ```
4. Force conversion even with syntax errors:
   ```bash
   ./convert_api_backends.py --backend ollama --force
   ```
5. Review generated TypeScript files in `ipfs_accelerate_js/src/api_backends/`
6. Complete any missing method implementations
7. Add the backend to the registry in `ipfs_accelerate_js/src/api_backends/index.ts`
8. Create and run tests for the new TypeScript implementation

## Usage

```bash
# Convert a specific backend
./convert_api_backends.py --backend ollama

# Convert all available backends
./convert_api_backends.py --all

# Test without writing files (dry run)
./convert_api_backends.py --backend ollama --dry-run

# Specify custom directories
./convert_api_backends.py --all --python-dir /path/to/python/backends --ts-dir /path/to/typescript/output

# Attempt to fix syntax errors in the Python source
./convert_api_backends.py --backend ollama --fix-source

# Force conversion even if parsing has issues
./convert_api_backends.py --backend ollama --force

# Show detailed logs
./convert_api_backends.py --backend ollama --verbose

# Complete conversion with all options
./convert_api_backends.py --all --fix-source --force --verbose
```

## Handling Syntax Errors

Many of the Python files in the repository have syntax errors that prevent proper parsing. The converter provides several options to handle these issues:

1. **Fix Source (`--fix-source`)**: Attempts to automatically fix common syntax errors in the Python files
   - Creates a backup of the original file (*.bak)
   - Fixes extra closing parentheses and improper indentation
   - Tries to repair common issues

2. **Force Conversion (`--force`)**: Proceeds with conversion even if parsing fails
   - Creates a minimal class structure based on available information
   - May produce incomplete TypeScript files that require manual editing

3. **Manual Cleanup**: For files with more complex syntax issues
   - Manually edit the Python source files to fix syntax errors
   - Run the converter again once files are fixed

## Common Python Syntax Issues

The most common syntax issues encountered in the API backend files include:

1. Extra closing parentheses (e.g., `logger.getLogger())))))))"ollama_api"`)
2. Incorrect indentation in import statements
3. Double colons (e.g., `try::` instead of `try:`)
4. Missing parentheses in method definitions

## Features

- Parses Python API backend files using AST (Abstract Syntax Tree)
- Extracts methods, parameters, and class properties
- Converts Python types to TypeScript types with smart mapping
- Generates TypeScript class implementations extending BaseApiBackend
- Creates API-specific type definitions with custom fields for each backend
- Maintains proper method signatures and implementations
- Handles async methods and generators appropriately
- Creates consistent directory structure for TypeScript backends
- Provides smart defaults for API endpoints and models based on backend type
- Implements backend-specific compatibility checks
- Supports automatic syntax error fixing in Python files
- Includes comprehensive error handling and recovery

## Output Structure

For each API backend, the tool generates the following files:

```
ipfs_accelerate_js/src/api_backends/{backend_name}/
├── index.ts          # Simple re-export file
├── {backend_name}.ts # Main implementation
└── types.ts          # Backend-specific type definitions
```

## Example Generated Output

The converter will generate TypeScript files for each API backend with the following patterns:

### Main Implementation File (ollama.ts)

```typescript
import { BaseApiBackend } from '../base';
import { ApiMetadata, ApiRequestOptions, Message, ChatCompletionResponse, StreamChunk } from '../types';
import { OllamaResponse, OllamaRequest } from './types';

export class Ollama extends BaseApiBackend {
  private apiEndpoint: string = 'http://localhost:11434/api/chat';
  private apiBase: string = '';
  private useLocalDeployment: boolean = true;
  
  constructor(resources: Record<string, any> = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
  }
  
  protected getApiKey(metadata: ApiMetadata): string {
    return metadata.ollama_api_key || 
           metadata.ollamaApiKey || 
           (typeof process !== 'undefined' ? process.env.OLLAMA_API_KEY || '' : '');
  }
  
  protected getDefaultModel(): string {
    return "llama2";
  }
  
  isCompatibleModel(model: string): boolean {
    // Ollama supports various models
    return (
      model.startsWith('llama') || 
      model.startsWith('mistral') || 
      model.startsWith('gemma') || 
      model.toLowerCase().includes('ollama:')
    );
  }
  
  createEndpointHandler(): (data: any) => Promise<any> {
    return async (data: any) => {
      try {
        return await this.makePostRequest(data);
      } catch (error) {
        throw this.createApiError(`${this.constructor.name} endpoint error: ${error.message}`, 500);
      }
    };
  }
  
  async testEndpoint(): Promise<boolean> {
    try {
      const apiKey = this.getApiKey(this.metadata);
      // Ollama doesn't actually require an API key but we keep the check for interface consistency
      
      // Make a minimal request to verify the endpoint works
      const testRequest = {
        model: this.getDefaultModel(),
        messages: [{ role: 'user', content: 'Hello' }],
        max_tokens: 5
      };
      
      await this.makePostRequest(testRequest, apiKey, { timeoutMs: 5000 });
      return true;
    } catch (error) {
      console.error(`${this.constructor.name} endpoint test failed:`, error);
      return false;
    }
  }
  
  // ... other method implementations ...
  
  async chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse> {
    // Prepare request data
    const modelName = options?.model || this.getDefaultModel();
    
    const requestData: OllamaRequest = {
      model: modelName,
      messages,
      max_tokens: options?.maxTokens,
      temperature: options?.temperature,
      top_p: options?.topP
    };
    
    // Make the request
    const response = await this.makePostRequest(requestData, undefined, options);
    
    // Convert to standard format
    return {
      id: response.id || '',
      model: response.model || modelName,
      content: response.choices?.[0]?.message?.content || '',
      created: response.created || Date.now(),
      usage: response.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };
  }
}
```

### Type Definitions File (types.ts)

```typescript
import { Message } from '../types';

export interface OllamaResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message?: {
      role: string;
      content: string;
    };
    delta?: {
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface OllamaRequest {
  model: string;
  messages?: Message[];
  prompt?: string;
  stream?: boolean;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
}
```

### Export File (index.ts)

```typescript
export * from './ollama';
```

## Post-Processing

After generating the TypeScript files, you may need to:

1. Review the converted code for any issues or missing functionality
2. Add additional API-specific customizations
3. Update the main API backend registry in `ipfs_accelerate_js/src/api_backends/index.ts`
4. Create comprehensive tests for the new TypeScript backend

## Backend-Specific Customizations

The converter has built-in knowledge of common API backends and provides specialized behavior for each:

### API Endpoints

- **Claude**: `https://api.anthropic.com/v1/messages`
- **OpenAI**: `https://api.openai.com/v1/chat/completions`
- **Groq**: `https://api.groq.com/openai/v1/chat/completions`
- **Gemini**: `https://generativelanguage.googleapis.com/v1/models`
- **Ollama**: `http://localhost:11434/api/chat`
- **HF TGI**: `http://localhost:8080/generate`
- **HF TEI**: `http://localhost:8080/`

### Default Models

- **Claude**: `claude-3-sonnet-20240229`
- **OpenAI**: `gpt-3.5-turbo`
- **Groq**: `llama2-70b-4096`
- **Gemini**: `gemini-pro`
- **Ollama**: `llama2`
- **HF TGI**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **HF TEI**: `BAAI/bge-small-en-v1.5`

### Backend-Specific Types

Each backend has customized type definitions for requests and responses:

- **Claude**: Includes system messages, anthropic version, and stop sequences
- **OpenAI**: Includes functions, function calls, and presence/frequency penalties
- **Gemini**: Uses a different message format with content parts
- **HF TEI**: Specialized for embedding models with inputs and dimensions
- **HF TGI**: Text generation specific parameters

## Manual Completion Steps

After generating the TypeScript files, you may need to perform these additional steps:

1. **Complete Implementation**: Add any missing method implementations or enhance generated code
2. **Register Backends**: Update main API registry in `ipfs_accelerate_js/src/api_backends/index.ts`:
   ```typescript
   // Add new backend to the registry
   import { Ollama } from './ollama';
   
   export const apiBackends = {
     // Existing backends
     ollama: Ollama,
     // More backends
   };
   ```

3. **Testing**: Create and run tests for each backend:
   ```typescript
   // Sample test for Ollama backend
   import { Ollama } from '../src/api_backends/ollama';
   
   describe('Ollama Backend', () => {
     test('should create an endpoint handler', async () => {
       const backend = new Ollama();
       const handler = backend.createEndpointHandler();
       expect(handler).toBeDefined();
     });
   });
   ```

## Known Limitations

- The converter can't handle severe syntax errors in Python files without manual fixes
- Method implementations may need completion if parsing fails
- Python language features without TypeScript equivalents may require manual conversion
- Python-specific patterns (like threading) need manual adaptation to async/await
- Type annotations derived from non-typed Python code may be suboptimal

## Best Practices for Using the Converter

For best results with the API backend converter, follow these guidelines:

1. **Pre-Clean Python Files**: When possible, fix major syntax errors in Python files before conversion
   - Ensure proper parentheses matching
   - Fix indentation issues
   - Make sure all methods have proper signatures

2. **Use Force Option for Partial Results**: The `--force` option is useful for generating a skeleton when parsing fails
   - Run with `--fix-source` first to attempt automatic fixes
   - Use `--force` to generate at least the class structure
   - Manually complete the implementation

3. **Consider Backend Special Cases**: Be aware of backend-specific implementations
   - Check API endpoint URLs for correctness
   - Verify default model names match your usage
   - Review request/response type definitions for accuracy

4. **Post-Processing Review**: Always review and test the generated code
   - Complete any missing method implementations
   - Verify type definitions match API documentation
   - Ensure error handling is appropriate
   - Test with actual API calls

## Troubleshooting

### Common Issues and Solutions

1. **"Error parsing file"**
   - Use `--fix-source` option to attempt automatic fixes
   - Manually check the Python file for syntax errors
   - Use `--force` to generate a partial implementation

2. **Missing Methods in Output**
   - The converter may not have been able to parse all methods
   - Check for syntax errors in the specific method
   - Manually implement missing methods

3. **Type Definition Issues**
   - Review and update type definitions in the generated `types.ts` file
   - Consult API documentation for accurate request/response structures
   - Implement more specific types for better type safety

4. **Incompatible API Endpoints**
   - Verify the generated endpoint URLs match documentation
   - Update endpoint URLs if they don't match
   - Add any required headers or authentication logic

5. **Incomplete Implementations**
   - Generated files may have template methods with "not implemented" errors
   - Use Python implementation as a reference to complete these methods
   - Convert Python-specific patterns to TypeScript appropriately

## Future Improvements

- Further enhance parser resilience to severe syntax errors in Python files
- Add more comprehensive type mapping for complex Python types
- Generate test files automatically based on Python code usage patterns
- Enhance API-specific parameter handling with more model-specific logic
- Add source documentation copying to maintain API documentation
- Support Python decorators conversion to TypeScript equivalents
- Generate more complete implementations for streaming methods
- Add conversion metrics and validation reports
- Support cross-file dependencies and imports detection
- Add code quality metrics for generated code
- Implement automatic test generation based on Python implementation