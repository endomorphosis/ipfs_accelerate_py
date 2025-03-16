# VLLM Unified API Backend

This module provides a TypeScript implementation of the VLLM unified API backend, offering enhanced functionality and integration with the Hardware Abstraction Layer.

## Overview

The VLLM Unified backend is part of the ongoing TypeScript migration effort and provides a comprehensive client for VLLM servers with advanced features:

- Standard and streaming text generation
- Batch processing
- LoRA adapter support
- Quantization configuration
- Comprehensive error handling
- Resource pooling and circuit breaking

## Usage

```typescript
import { VllmUnified } from '../../src/api_backends/vllm_unified';

const vllm = new VllmUnified({}, {
  vllm_api_url: 'http://localhost:8000',
  vllm_model: 'meta-llama/Llama-2-7b-chat-hf'
});

// Basic usage
const result = await vllm.makeRequest(
  'http://localhost:8000',
  'Tell me a story about dragons',
  'llama-7b'
);

// Chat completion
const chatResult = await vllm.chat(
  [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Tell me about dragons.' }
  ],
  { model: 'llama-7b-chat' }
);

// Batch processing
const batchResults = await vllm.processBatch(
  'http://localhost:8000',
  ['Question 1', 'Question 2', 'Question 3'],
  'llama-7b'
);
```

## Files

- `types.ts`: TypeScript type definitions for the VLLM API
- `vllm_unified.ts`: Main implementation of the VLLM Unified backend
- `index.ts`: Module exports

## API Documentation

For complete documentation, see the [VLLM Unified Usage Guide](../../../docs/api_backends/VLLM_UNIFIED_USAGE.md).