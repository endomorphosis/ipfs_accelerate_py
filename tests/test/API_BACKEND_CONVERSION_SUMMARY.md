# API Backend Conversion: Python to TypeScript

This document summarizes the conversion of API backends from Python to TypeScript as part of the migration to a unified JavaScript SDK.

## Overview

We have successfully converted the following API backends from Python to TypeScript:

- **Groq** (`groq.ts`): Implemented with streaming support and OpenAI-compatible API
- **Gemini** (`gemini.ts`): Implemented with Google AI-specific request/response format conversion
- **HF TEI** (`hf_tei.ts`): Hugging Face Text Embedding Inference with embedding-specific functionality
- **HF TGI** (`hf_tgi.ts`): Hugging Face Text Generation Inference with streaming and chat template support
- **HF TEI Unified** (`hf_tei_unified/hf_tei_unified.ts`): Unified Hugging Face Text Embedding client with both API and container modes
- **HF TGI Unified** (`hf_tgi_unified/hf_tgi_unified.ts`): Unified Hugging Face Text Generation client with API and container modes
- **Ollama** (`ollama.ts`): Local LLM server with streaming, circuit breaker, and queuing support
- **OpenAI** (`openai.ts`): Full-featured OpenAI API client with all capabilities and streaming
- **OpenAI Mini** (`openai_mini/openai_mini.ts`): Lightweight OpenAI API client with core functionality and optimized implementation
- **Claude** (`claude.ts`): Anthropic Claude API client with streaming and content formatting
- **Sample Backend** (`sample_backend.ts`): Reference implementation with basic functionality
- **OVMS** (`ovms/ovms.ts`): OpenVINO Model Server with tensor-based inference
- **VLLM** (`vllm/vllm.ts`): vLLM high-performance inference server with LoRA adapter support
- **OPEA** (`opea/opea.ts`): OpenAI Platform Extensions API for self-hosted OpenAI-compatible APIs
- **S3 Kit** (`s3_kit/s3_kit.ts`): S3-compatible storage API with connection multiplexing and endpoint management
- **LLVM** (`llvm/llvm.ts`): LLVM-based inference server API with model management and queue handling

Each backend extends the `BaseApiBackend` class and implements a consistent API for:
- Making API requests
- Streaming responses
- Error handling
- Chat and streaming interfaces
- Model compatibility checks

## Implementation Details

### Core Components

1. **Base API Backend**: All backends extend the `BaseApiBackend` class that provides:
   - Request queueing
   - Retry and backoff logic
   - Error handling
   - Circuit breaker pattern
   - Resource tracking
   - Model compatibility checks

2. **Types Interface**: Each backend has a corresponding `types.ts` file with:
   - Backend-specific request types
   - Backend-specific response types

3. **Common API**: All backends implement the following methods:
   - `getApiKey`: Retrieve API key from metadata or environment
   - `getDefaultModel`: Get the default model for this backend
   - `isCompatibleModel`: Check if a model is compatible with this backend
   - `createEndpointHandler`: Create an endpoint handler for API requests
   - `testEndpoint`: Test the API endpoint
   - `makePostRequest`: Make a POST request to the API
   - `makeStreamRequest`: Make a streaming request to the API
   - `chat`: Generate a chat completion
   - `streamChat`: Generate a streaming chat completion

### Backend-Specific Features

Each backend includes specialized functionality:

#### Groq Backend
- OpenAI-compatible API interface
- Specialized model compatibility for Llama and Mistral models

#### Gemini Backend
- Google AI-specific request/response format conversion
- Direct endpoint URL construction with API key
- Content formatting for Google's content structure

#### HF TEI Backend (Text Embedding)
- Support for embedding generation
- Model namespace format compatibility check
- Embedding-specific response processing

#### HF TEI Unified Backend
- Unified interface for both direct API and container-based deployment
- Docker container management for local deployment
- Comprehensive benchmark system with performance metrics
- Smart mode switching between API and container modes
- Batch embedding optimization with performance analysis
- Container configuration with GPU and environment settings

#### HF TGI Backend (Text Generation)
- Chat template formatting for HF models
- Support for both streaming and non-streaming responses
- Model parameter control (temperature, top-p, etc.)

#### HF TGI Unified Backend
- Unified interface for both API and container-based text generation
- Docker container management with GPU and environment configuration
- Advanced prompt template system for different model architectures
- Streaming text generation with chunk processing and buffering
- Chat formatting with support for multiple chat formats (Llama, Mistral, ChatML)
- Performance benchmarking and optimization
- Smart mode switching between API and container modes
- Comprehensive error handling with specific error codes
- Container health monitoring and lifecycle management

#### Ollama Backend
- Local deployment support
- Local model compatibility checks
- Ollama-specific streaming response processing
- Circuit breaker pattern for fault tolerance
- Request queue with priority levels
- Exponential backoff for error handling

#### OpenAI Backend
- Full support for all OpenAI API endpoints:
  - Chat completions with streaming
  - Embeddings generation
  - Image generation (DALL-E)
  - Content moderation
  - Text-to-speech
  - Speech-to-text
- Tool calling and function calling support
- Multimodal content support
- Comprehensive error handling

#### OpenAI Mini Backend
- Lightweight implementation focused on core functionality:
  - Chat completions with streaming
  - Audio transcription with Whisper models
  - Text-to-speech with TTS models
  - Image generation with DALL-E models
  - File upload and management
- Circuit breaker pattern for fault tolerance
- Compatible with the full OpenAI API but with optimized implementation
- Reduced dependencies and simplified codebase

#### Claude Backend
- Content formatting for Anthropic's message structure
- System prompt support
- Streaming response support with event parsing
- Claude-specific model compatibility checks

#### OVMS Backend (OpenVINO Model Server)
- Support for tensor-based inference with OpenVINO Model Server
- Input formatting for various data types and shapes
- Model version management and selection
- Server and model health monitoring capabilities
- Advanced features like model reloading and metadata retrieval
- Support for model explanation and prediction interpretation
- Quantization configuration management

#### VLLM Backend (vLLM Server)
- High-performance inference with vLLM server
- LoRA adapter management (listing, loading)
- Complete batch processing support with metrics
- Streaming generation with customizable parameters
- Model status and statistics monitoring
- Specialized streaming response handling with SSE parsing
- Quantization configuration management
- Model architecture-based compatibility detection

#### OPEA Backend (OpenAI Platform Extensions API)
- Compatible interface for self-hosted OpenAI-compatible APIs
- OpenAI format compatibility with custom endpoints
- Standardized streaming response handling with buffer management
- Unified error handling across different API implementations
- Support for multiple model types through single implementation

#### S3 Kit Backend
- Multi-endpoint management for different S3-compatible storage services
- Connection multiplexing with round-robin and least-loaded selection strategies
- Circuit breaker pattern for resilient endpoint handling
- Queue management with priority levels (HIGH, NORMAL, LOW)
- Core S3 operations: upload, download, list, delete
- Request retry mechanism with exponential backoff
- Per-endpoint configuration for credentials, concurrency, and circuit breaker settings
- Thread-safe operation with proper locking mechanisms

#### LLVM Backend
- Complete LLVM-based inference server client
- Model listing and information retrieval
- Configurable inference options with parameter passing
- Queue management with request priority levels
- Retry mechanism with exponential backoff
- Environment variable fallback for configuration
- API key management and authentication
- Chat interface adaptation for text-based inference

### Index File

The main `index.ts` file exports:
- All backend implementations
- Registry of available backends
- Utility functions for backend creation and discovery:
  - `getBackend`: Get a backend class by name
  - `createApiBackend`: Create a backend instance by name
  - `getAvailableBackends`: Get a list of available backends
  - `findCompatibleBackend`: Find a compatible backend for a model

## Testing Infrastructure

A comprehensive testing infrastructure has been created:

1. **Unit Tests**: Tests for the converter itself (`test_api_backend_converter.py`)
2. **Integration Tests**: End-to-end tests for converting real backends (`test_api_backend_converter_integration.py`)
3. **TypeScript Tests**: Generated Jest tests for each backend in `ipfs_accelerate_js/test/api_backends/`
4. **Test Runner**: Script to run all tests (`run_api_converter_tests.py`)

## How to Use the Converted Backends

### Basic Usage

```typescript
import { Groq } from '../api_backends/groq';
import { ApiMetadata } from '../api_backends/types';

// Create backend instance
const metadata: ApiMetadata = {
  groq_api_key: 'your-api-key'
};
const groq = new Groq({}, metadata);

// Chat completion
const response = await groq.chat([
  { role: 'user', content: 'Hello, how are you?' }
]);
console.log(response.content);

// Streaming chat completion
const stream = groq.streamChat([
  { role: 'user', content: 'Tell me a story about a robot.' }
]);

for await (const chunk of stream) {
  process.stdout.write(chunk.content);
}
```

### Dynamic Backend Creation

```typescript
import { createApiBackend, findCompatibleBackend } from '../api_backends';

// Create by name
const backend = createApiBackend('gemini', {}, { 
  gemini_api_key: 'your-api-key' 
});

// Find by model compatibility
const model = 'mistral-7b';
const compatibleBackend = findCompatibleBackend(model, {}, {
  groq_api_key: 'your-groq-key',
  ollama_api_key: 'your-ollama-key'
});

if (compatibleBackend) {
  const response = await compatibleBackend.chat([
    { role: 'user', content: 'Hello!' }
  ], { model });
  console.log(response.content);
}
```

## Conversion Process

The conversion was performed using the `convert_api_backends.py` script, which:

1. Parses Python files into AST (Abstract Syntax Tree)
2. Extracts class methods, properties, and types
3. Converts Python types to TypeScript types
4. Generates TypeScript implementation files
5. Creates appropriate TypeScript interfaces

### Challenges and Solutions

1. **Python Syntax Errors**: Several Python files had syntax errors
   - Solution: Implemented auto-fixing with `--fix-source` flag
   - Solution: Used `--force` flag to continue despite parsing errors

2. **Type Conversion**: Python's dynamic typing vs. TypeScript's static typing
   - Solution: Created type mappings dictionary
   - Solution: Implemented complex type converter for generic types

3. **Backend-Specific APIs**: Each backend has unique API structures
   - Solution: Implemented specialized formatters for each backend
   - Solution: Created custom request/response interfaces

4. **Streaming Implementation**: Differences in streaming patterns
   - Solution: Implemented AsyncGenerator pattern in TypeScript
   - Solution: Created unified StreamChunk interface

## Next Steps

1. ✅ **Manual Verification**: All backend implementations have been reviewed for correctness
2. ✅ **Basic Implementation**: All required backends have been implemented
   - Groq, Gemini, HF TEI, HF TGI, Ollama, OpenAI, Claude, Sample Backend, OVMS, VLLM, OPEA
3. 🔄 **Integration Testing**: Test each backend with real API calls
   - ✅ Basic test files created for Groq, Gemini, HF TEI, HF TGI, Ollama, OpenAI, Claude, etc.
   - ✅ Comprehensive test for HF TGI Unified backend implemented
   - ✅ Comprehensive test for HF TEI Unified backend implemented
   - ✅ Comprehensive test for VLLM Unified backend implemented
   - ✅ Comprehensive test for OVMS Unified backend implemented
   - 🔄 Continue adding tests for remaining unified backends
4. **Documentation Updates**: Add JSDoc comments to backend methods
5. **Usage Examples**: Create examples for each backend
6. **CI Integration**: Add TypeScript tests to CI pipeline

### Implementation Status

All planned API backends have now been successfully implemented:

| Backend | Status | Implementation | Features |
|---------|--------|----------------|----------|
| Groq | ✅ Complete | `groq.ts` | Streaming, OpenAI-compatible |
| Gemini | ✅ Complete | `gemini.ts` | Google AI format conversion |
| HF TEI | ✅ Complete | `hf_tei.ts` | Embedding generation |
| HF TGI | ✅ Complete | `hf_tgi.ts` | Text generation, streaming |
| HF TEI Unified | ✅ Complete | `hf_tei_unified/hf_tei_unified.ts` | API and container modes |
| HF TGI Unified | ✅ Complete | `hf_tgi_unified/hf_tgi_unified.ts` | Text generation with container |
| Ollama | ✅ Complete | `ollama.ts` | Local LLM, circuit breaker |
| OpenAI | ✅ Complete | `openai.ts` | Full API support, multimodal |
| OpenAI Mini | ✅ Complete | `openai_mini/openai_mini.ts` | Lightweight, core functionality |
| Claude | ✅ Complete | `claude.ts` | System prompts, streaming |
| OVMS | ✅ Complete | `ovms/ovms.ts` | Tensor-based inference |
| VLLM | ✅ Complete | `vllm/vllm.ts` | High-performance, LoRA |
| OPEA | ✅ Complete | `opea/opea.ts` | OpenAI-compatible APIs |
| S3 Kit | ✅ Complete | `s3_kit/s3_kit.ts` | S3 storage, multiplexing |
| LLVM | ✅ Complete | `llvm/llvm.ts` | LLVM inference server |
| Sample | ✅ Complete | `sample_backend.ts` | Reference implementation |

## References

- [Base API Backend Implementation](ipfs_accelerate_js/src/api_backends/base.ts)
- [API Backend Converter](./convert_api_backends.py)
- [API Backend Converter Testing](./README_API_CONVERTER_TESTING.md)
- [TypeScript Backend Tests](ipfs_accelerate_js/test/api_backends/)