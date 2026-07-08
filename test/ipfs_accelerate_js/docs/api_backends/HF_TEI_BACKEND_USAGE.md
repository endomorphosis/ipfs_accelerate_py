# HuggingFace Text Embedding Inference (HF TEI) API Backend - Usage Guide

This document provides instructions on how to use the HuggingFace Text Embedding Inference (HF TEI) API backend in the IPFS Accelerate JS framework.

## Introduction

The HF TEI API backend provides access to HuggingFace's text embedding models through their Inference API. Text embeddings convert text into numeric vector representations that capture semantic meaning, making them essential for various NLP tasks such as semantic search, clustering, classification, and recommendation systems.

This backend supports both the hosted HuggingFace Inference API and self-hosted deployments, with features including single and batch embedding generation, similarity calculations, multiple endpoint management, and more.

## Prerequisites

To use the HF TEI API backend, you'll need:

1. A HuggingFace API key (get one from [HuggingFace](https://huggingface.co/settings/tokens))
2. Node.js v16 or higher
3. IPFS Accelerate JS framework

## Installation

The HF TEI API backend is included in the IPFS Accelerate JS package, so no additional installation is required.

```bash
npm install @ipfs-accelerate/js
```

## Basic Usage

### Initialize the Backend

```typescript
import { HfTei } from '@ipfs-accelerate/js/api_backends';

// Initialize with API key and default model
const hfTei = new HfTei({}, {
  hf_tei_api_key: 'your-api-key-here',
  model_id: 'sentence-transformers/all-MiniLM-L6-v2',
  // Optional configuration
  max_retries: 3,
  initial_retry_delay: 1000,
  backoff_factor: 2,
  timeout: 30000
});

// Or use environment variable HF_TEI_API_KEY or HF_API_KEY
// export HF_TEI_API_KEY=your-api-key-here
const hfTeiWithEnvKey = new HfTei();
```

### Generating Text Embeddings

```typescript
// Example: Generate embedding for a single text
async function generateSingleEmbedding() {
  const text = "This is a sample text for embedding generation.";
  
  try {
    // Generate embedding
    const embedding = await hfTei.generateEmbedding(
      'sentence-transformers/all-MiniLM-L6-v2', // Model ID
      text                                       // Input text
    );
    
    console.log(`Text: "${text}"`);
    console.log(`Embedding dimensions: ${embedding.length}`);
    console.log(`First few dimensions: ${embedding.slice(0, 5)}`);
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### Batch Processing Multiple Texts

```typescript
// Example: Generate embeddings for multiple texts in one request
async function generateBatchEmbeddings() {
  const texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models transform input data into useful representations.",
    "Vector embeddings capture semantic meaning in a numerical format."
  ];
  
  try {
    // Generate embeddings for all texts in one batch request
    const embeddings = await hfTei.batchEmbed(
      'sentence-transformers/all-MiniLM-L6-v2', // Model ID
      texts                                     // Array of input texts
    );
    
    console.log(`Number of texts: ${texts.length}`);
    console.log(`Number of embeddings: ${embeddings.length}`);
    console.log(`Embedding dimensions: ${embeddings[0].length}`);
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### Simplified Embedding Interface

```typescript
// Example: Use the simplified embedText interface
async function useEmbedText() {
  const text = "This is a sample text for the simplified interface.";
  
  try {
    // Generate embedding using the simplified interface
    const response = await hfTei.embedText(text, {
      model: 'sentence-transformers/all-MiniLM-L6-v2'
    });
    
    console.log(`Text: "${text}"`);
    console.log(`Model used: ${response.model}`);
    console.log(`Embedding: ${response.content[0].slice(0, 5)}...`);
  } catch (error) {
    console.error('Error:', error);
  }
}

// Example: Batch embedding with simplified interface
async function batchEmbedText() {
  const texts = [
    "First example text.",
    "Second example text.",
    "Third example text."
  ];
  
  try {
    // Generate embeddings for multiple texts
    const response = await hfTei.embedText(texts, {
      model: 'sentence-transformers/all-MiniLM-L6-v2'
    });
    
    console.log(`Number of texts: ${texts.length}`);
    console.log(`Number of embeddings: ${response.content.length}`);
  } catch (error) {
    console.error('Error:', error);
  }
}
```

## Advanced Usage

### Calculating Similarity Between Texts

```typescript
async function calculateTextSimilarity() {
  const text1 = "Artificial intelligence is revolutionizing technology.";
  const text2 = "AI is transforming how we use computers and software.";
  const text3 = "The weather today is sunny with clear skies.";
  
  try {
    // Generate embeddings for each text
    const embedding1 = await hfTei.generateEmbedding('sentence-transformers/all-MiniLM-L6-v2', text1);
    const embedding2 = await hfTei.generateEmbedding('sentence-transformers/all-MiniLM-L6-v2', text2);
    const embedding3 = await hfTei.generateEmbedding('sentence-transformers/all-MiniLM-L6-v2', text3);
    
    // Calculate similarities (cosine similarity, range 0-1)
    const similarity1_2 = hfTei.calculateSimilarity(embedding1, embedding2);
    const similarity1_3 = hfTei.calculateSimilarity(embedding1, embedding3);
    
    console.log(`Similarity between text 1 and 2: ${similarity1_2.toFixed(4)}`); // Should be high (related)
    console.log(`Similarity between text 1 and 3: ${similarity1_3.toFixed(4)}`); // Should be low (unrelated)
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### Testing Endpoint Availability

```typescript
async function testConnection() {
  const isAvailable = await hfTei.testEndpoint();
  console.log(`HuggingFace TEI API is available: ${isAvailable}`);
  
  // Test a specific endpoint
  const customEndpointAvailable = await hfTei.testEndpoint(
    'https://custom-endpoint-url/api/models/my-model',
    'custom-api-key',
    'my-custom-model'
  );
  console.log(`Custom endpoint is available: ${customEndpointAvailable}`);
}
```

### Creating and Managing Multiple Endpoints

```typescript
function manageEndpoints() {
  // Create an endpoint for a general-purpose embedding model
  const generalEndpointId = hfTei.createEndpoint({
    id: 'general-embedding',
    apiKey: 'your-api-key',
    model: 'sentence-transformers/all-MiniLM-L6-v2',
    maxConcurrentRequests: 5
  });
  
  // Create an endpoint for a specialized model
  const specializedEndpointId = hfTei.createEndpoint({
    id: 'specialized-embedding',
    apiKey: 'your-second-api-key',
    model: 'BAAI/bge-large-en-v1.5',
    maxConcurrentRequests: 2,
    timeout: 10000
  });
  
  // Update an endpoint's settings
  hfTei.updateEndpoint(generalEndpointId, {
    timeout: 15000,
    maxConcurrentRequests: 10
  });
  
  // Get statistics for a specific endpoint
  const stats = hfTei.getStats(generalEndpointId);
  console.log('Endpoint stats:', stats);
  
  // Make a request using a specific endpoint
  const data = { inputs: 'Text to embed' };
  hfTei.makeRequestWithEndpoint(generalEndpointId, data)
    .then(response => console.log('Response:', response))
    .catch(error => console.error('Error:', error));
}
```

### Normalizing Embeddings

```typescript
function normalizeVectors() {
  // Create a non-normalized vector
  const vector = [1.0, 2.0, 3.0, 4.0];
  console.log('Original vector:', vector);
  
  // Normalize it
  const normalizedVector = hfTei.normalizeEmbedding(vector);
  console.log('Normalized vector:', normalizedVector);
  
  // Verify the magnitude is 1.0
  const magnitude = Math.sqrt(normalizedVector.reduce((sum, val) => sum + val * val, 0));
  console.log('Magnitude:', magnitude); // Should be very close to 1.0
}
```

### Managing Usage Statistics

```typescript
function manageStats() {
  // Get global statistics across all endpoints
  const globalStats = hfTei.getStats();
  console.log('Global stats:', globalStats);
  
  // Reset statistics for a specific endpoint
  hfTei.resetStats('endpoint-id');
  
  // Reset all statistics
  hfTei.resetStats();
}
```

## Error Handling

The backend implements proper error handling with retries and backoff:

```typescript
try {
  const embedding = await hfTei.generateEmbedding(
    'sentence-transformers/all-MiniLM-L6-v2', 
    'Test text'
  );
  console.log(embedding);
} catch (error) {
  if (error.isAuthError) {
    console.error('Authentication error - check your API key');
  } else if (error.isRateLimitError) {
    console.error('Rate limit exceeded, try again later');
  } else if (error.isTimeout) {
    console.error('Request timed out');
  } else if (error.isTransientError) {
    console.error('Temporary server error, try again later');
  } else {
    console.error('Unknown error:', error.message);
  }
}
```

## Best Practices

1. **API Key Security**: Never hardcode your API key in your application code. Use environment variables or a secure configuration management system.

2. **Model Selection**: Choose the appropriate model for your use case:
   - `sentence-transformers/all-MiniLM-L6-v2`: Balanced model (384 dimensions) - good default choice
   - `BAAI/bge-small-en-v1.5`: Optimized for English (384 dimensions)
   - `BAAI/bge-large-en-v1.5`: Higher quality, larger model (1024 dimensions)
   - `sentence-transformers/all-mpnet-base-v2`: High quality but larger (768 dimensions)
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: Good for multilingual use (384 dimensions)

3. **Batch Processing**: When embedding multiple texts, use `batchEmbed()` instead of multiple `generateEmbedding()` calls to reduce API calls and improve performance.

4. **Normalization**: Embeddings are normalized by default. Keep them normalized when comparing or when using cosine similarity.

5. **Error Handling**: Implement proper error handling with retries for transient errors.

6. **Caching**: Consider caching embeddings for frequently used texts to reduce API usage and improve response times.

7. **Endpoint Management**: Create separate endpoints for different models or use cases to better manage resources and monitor usage.

## Supported Models

The backend supports all HuggingFace embedding models, including:

- **Sentence Transformers family**:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `sentence-transformers/all-mpnet-base-v2`
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - And many other sentence-transformers models

- **BGE (BAAI) family**:
  - `BAAI/bge-small-en-v1.5`
  - `BAAI/bge-base-en-v1.5`
  - `BAAI/bge-large-en-v1.5`
  - Multi-lingual variants like `BAAI/bge-small-zh-v1.5`

- **E5 family**:
  - `intfloat/e5-small`
  - `intfloat/e5-base`
  - `intfloat/e5-large`

- **Other embedding models**:
  - `thenlper/gte-small`
  - `thenlper/gte-base`
  - And other dense retrieval models

## Additional Resources

- [HuggingFace Inference API Documentation](https://huggingface.co/docs/api-inference/detailed_parameters#feature-extraction-task)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [IPFS Accelerate JS Documentation](https://github.com/your-org/ipfs-accelerate-js)