# API Backends Documentation

This directory contains documentation for all API backends in the IPFS Accelerate JavaScript SDK.

## Available API Backends

| Backend | Documentation | Description |
|---------|---------------|-------------|
| **Common API Interface** | [API_BACKEND_INTERFACE_REFERENCE.md](./API_BACKEND_INTERFACE_REFERENCE.md) | **Reference documentation for the BaseApiBackend interface** |
| **Developer Guide** | [API_BACKEND_DEVELOPMENT.md](./API_BACKEND_DEVELOPMENT.md) | **Guide for implementing new API backends** |
| VLLM Unified | [VLLM_UNIFIED_USAGE.md](./VLLM_UNIFIED_USAGE.md) | Advanced VLLM integration with batching, streaming, model management, and more |
| OpenAI | [OPENAI_BACKEND_USAGE.md](./OPENAI_BACKEND_USAGE.md) | OpenAI API client for GPT models |
| OpenAI Mini | [OPENAI_MINI_BACKEND_USAGE.md](./OPENAI_MINI_BACKEND_USAGE.md) | Lightweight OpenAI API client with optimized implementation |
| Claude | [CLAUDE_BACKEND_USAGE.md](./CLAUDE_BACKEND_USAGE.md) | Anthropic Claude API client |
| Gemini | [GEMINI_BACKEND_USAGE.md](./GEMINI_BACKEND_USAGE.md) | Google Gemini API client |
| Groq | [GROQ_BACKEND_USAGE.md](./GROQ_BACKEND_USAGE.md) | Groq API integration |
| HF TGI | [HF_TGI_BACKEND_USAGE.md](./HF_TGI_BACKEND_USAGE.md) | HuggingFace Text Generation Inference |
| HF TEI | [HF_TEI_BACKEND_USAGE.md](./HF_TEI_BACKEND_USAGE.md) | HuggingFace Text Embedding Inference |
| Ollama | [OLLAMA_BACKEND_USAGE.md](./OLLAMA_BACKEND_USAGE.md) | Ollama local model integration |
| Ollama Clean | [OLLAMA_CLEAN_BACKEND_USAGE.md](./OLLAMA_CLEAN_BACKEND_USAGE.md) | OpenAI-compatible interface for Ollama models |
| OVMS | [OVMS_BACKEND_USAGE.md](./OVMS_BACKEND_USAGE.md) | OpenVINO Model Server integration |
| S3 Kit | [S3_KIT_BACKEND_USAGE.md](./S3_KIT_BACKEND_USAGE.md) | S3-compatible object storage client with multiple endpoint support |
| LLVM | [LLVM_BACKEND_USAGE.md](./LLVM_BACKEND_USAGE.md) | LLVM-based inference server integration |

## TypeScript Migration Status

The following backends have been migrated from Python to TypeScript:

- ✅ Claude (Tested & Documented)
- ✅ Gemini (Tested & Documented)
- ✅ Groq (Tested & Documented)
- ✅ HF TEI (Tested & Documented)
- ✅ HF TGI (Tested & Documented)
- ✅ HF TEI Unified (Tested & Documented)
- ✅ HF TGI Unified (Tested & Documented)
- ✅ Ollama (Tested & Documented)
- ✅ Ollama Clean (Tested & Documented)
- ✅ OPEA (Tested & Documented)
- ✅ OpenAI (Tested & Documented)
- ✅ OpenAI Mini (Tested & Documented)
- ✅ OVMS (Tested & Documented)
- ✅ S3 Kit (Tested & Documented)
- ✅ VLLM (Tested & Documented)
- ✅ VLLM Unified (Tested & Documented)
- ✅ LLVM (Tested & Documented)

Overall migration progress: **100%** complete

## API Backend Interface

All API backends implement the `BaseApiBackend` interface, which includes:

- Authentication and API key management
- Request handling and formatting
- Error handling and retries
- Streaming support
- Circuit breaker pattern
- Resource pooling

For complete details on the API Backend interface, see the [API Backend Interface Reference](./API_BACKEND_INTERFACE_REFERENCE.md).

For guidance on implementing new API backends, see the [API Backend Development Guide](./API_BACKEND_DEVELOPMENT.md).