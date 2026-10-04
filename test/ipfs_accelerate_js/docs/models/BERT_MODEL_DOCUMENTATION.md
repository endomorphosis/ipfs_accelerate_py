# BERT Model Implementation Documentation

This document describes the BERT (Bidirectional Encoder Representations from Transformers) model implementation in the IPFS Accelerate JS SDK with hardware acceleration support.

## Overview

The BERT model is a transformer-based machine learning model for natural language processing, designed to understand the context of words in search queries. Our implementation provides a hardware-accelerated version of BERT that can run on WebGPU and WebNN backends, enabling high-performance inference directly in the browser.

## Features

- **Hardware Acceleration**: Uses WebGPU and WebNN backends for accelerated inference
- **Optimized Operations**: Implements browser-specific optimizations for matrix operations
- **Memory Efficient**: Careful tensor management with proper cleanup
- **Cross-Model Tensor Sharing**: Supports sharing embedding tensors between models
- **Configurable**: Customizable model parameters (number of layers, hidden size, etc.)

## Usage

### Basic Usage

```typescript
import { createBertModel, WebGPUBackend } from 'ipfs-accelerate-js';

// Create hardware backend
const hardware = new WebGPUBackend();
await hardware.initialize();

// Create BERT model
const bert = createBertModel(hardware, {
  modelId: 'bert-base-uncased',
  useOptimizedOps: true
});

// Initialize model (loads weights)
await bert.initialize();

// Tokenize input text
const tokenizedInput = await bert.tokenize("Hello world");

// Run inference
const output = await bert.process(tokenizedInput);

// Use the output
console.log("Pooled output:", output.pooledOutput);
console.log("Last hidden state shape:", 
  `[${output.lastHiddenState.length}, ${output.lastHiddenState[0].length}]`);

// Clean up resources
await bert.dispose();
```

### With Cross-Model Tensor Sharing

```typescript
import { createBertModel, WebGPUBackend } from 'ipfs-accelerate-js';

const hardware = new WebGPUBackend();
await hardware.initialize();

// Create first BERT model
const bert1 = createBertModel(hardware);
await bert1.initialize();

// Process first input
const input1 = await bert1.tokenize("First text input");
const output1 = await bert1.process(input1);

// Create a shared tensor from the output
const sequenceOutput = hardware.tensorFromArray(output1.lastHiddenState);
const sharedTensor = await bert1.createSharedTensor(sequenceOutput, 'text_embedding');

// Create second BERT model that can use the shared tensor
const bert2 = createBertModel(hardware);
await bert2.initialize();

// Get the shared tensor (for example, to avoid recomputing embeddings)
const existingTensor = bert2.getSharedTensor('text_embedding');
if (existingTensor) {
  console.log("Using shared tensor from previous model");
  // Use the shared tensor in downstream tasks
}

// Clean up resources
await bert1.dispose();
await bert2.dispose();
```

## Model Configuration

The BERT model can be configured with the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| modelId | string | 'bert-base-uncased' | Model identifier |
| vocabSize | number | 30522 | Vocabulary size |
| hiddenSize | number | 768 | Hidden layer size |
| numLayers | number | 12 | Number of transformer layers |
| numHeads | number | 12 | Number of attention heads |
| intermediateSize | number | 3072 | Intermediate layer size |
| maxPositions | number | 512 | Maximum sequence length |
| layerNormEps | number | 1e-12 | Layer normalization epsilon |
| backendPreference | array | ['webgpu', 'webnn', 'cpu'] | Backend preference order |
| useOptimizedOps | boolean | true | Whether to use optimized operations |

## Implementation Details

### Architecture

The BERT implementation follows the original architecture described in the paper ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805) by Jacob Devlin et al.

Key components include:

1. **Embeddings Layer**: Combines token embeddings, position embeddings, and token type embeddings
2. **Encoder Layers**: Multi-head self-attention followed by feed-forward networks
3. **Pooler**: Transforms the [CLS] token representation for classification tasks

### Hardware Acceleration

The implementation leverages hardware acceleration through:

- **WebGPU Backend**: Uses compute shaders for matrix operations, with browser-specific optimizations
- **WebNN Backend**: Leverages neural network accelerators when available
- **Optimized Matrix Operations**: Uses browser-optimized operations for matrix multiplication and other intensive calculations

### Performance Considerations

- **Tensor Management**: Careful creation and disposal of tensors to minimize memory usage
- **Operation Fusion**: Support for operation fusion to reduce memory transfers
- **Browser Detection**: Adapts to the specific browser for optimal performance
- **Shared Tensor Pool**: Allows reuse of tensors across models to avoid redundant computations

## Examples

See the included examples for interactive demonstrations:

- `examples/browser/models/bert_example.html`: Interactive browser demo for testing BERT with different hardware backends
- `examples/browser/basic/bert_basic_usage.ts`: Simple usage example
- `examples/browser/advanced/bert_cross_model_sharing.ts`: Example of cross-model tensor sharing

## API Reference

See the TypeScript interface definitions for complete API documentation:

- `BertConfig`: Configuration interface for BERT models
- `BertInput`: Input format for tokenized text
- `BertOutput`: Output format with embeddings and metadata
- `Bert`: Main model class implementation
- `createBertModel()`: Factory function for creating model instances