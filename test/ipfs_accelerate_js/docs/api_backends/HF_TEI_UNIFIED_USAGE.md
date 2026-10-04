# HuggingFace Text Embedding Inference Unified Backend Usage Guide

This guide explains how to use the HuggingFace Text Embedding Inference (HF TEI) Unified backend for generating text embeddings with both the HuggingFace Inference API and self-hosted container deployments.

## Introduction

The HF TEI Unified backend provides a versatile interface for working with embedding models:

- **Dual-mode operation**: Works with both the hosted HuggingFace API and container-based deployments
- **Batched embedding generation**: Optimized for processing multiple texts efficiently
- **Container management**: Built-in Docker container deployment and management
- **Performance benchmarking**: Tools for evaluating embedding model performance
- **Hardware acceleration support**: GPU-accelerated embedding generation

## Prerequisites

Before using the HF TEI Unified backend, you'll need:

1. For API mode:
   - A HuggingFace account
   - A HuggingFace API key (for accessing hosted models)

2. For container mode:
   - Docker installed on your system
   - Sufficient disk space for model downloads (1-2GB per model)
   - (Optional) NVIDIA GPU with CUDA support for acceleration

### Obtaining an API Key

1. Create an account on [HuggingFace](https://huggingface.co/join)
2. Navigate to [Settings > API Tokens](https://huggingface.co/settings/tokens)
3. Create a new API token
4. Save the token securely

## Installation

The HF TEI Unified backend is included in the IPFS Accelerate JS package:

```bash
npm install ipfs-accelerate
```

Or if you're installing from source:

```bash
git clone https://github.com/your-org/ipfs-accelerate-js.git
cd ipfs-accelerate-js
npm install
```

## Basic Usage

### Initializing the Backend

```typescript
import { HfTeiUnified } from 'ipfs-accelerate/api_backends/hf_tei_unified';
import { HfTeiUnifiedOptions, HfTeiUnifiedApiMetadata } from 'ipfs-accelerate/api_backends/hf_tei_unified/types';

// Set up configuration options
const options: HfTeiUnifiedOptions = {
  apiUrl: 'https://api-inference.huggingface.co/models',
  maxRetries: 3,
  requestTimeout: 30000,
  useRequestQueue: true,
  debug: false
};

// Set up metadata with API key
const metadata: HfTeiUnifiedApiMetadata = {
  hf_api_key: 'YOUR_API_KEY', // Can also use process.env.HF_API_KEY
  model_id: 'BAAI/bge-small-en-v1.5' // Default embedding model
};

// Create the backend instance
const hfTeiUnified = new HfTeiUnified(options, metadata);
```

You can also provide the API key as an environment variable:

```bash
export HF_API_KEY=your_api_key
```

### Generating Embeddings

```typescript
// Generate embeddings for a single text
const text = "This is a sample text for embedding generation.";
const embedding = await hfTeiUnified.generateEmbeddings(text);
console.log(`Generated embedding with ${embedding[0].length} dimensions`);

// Generate embeddings for multiple texts
const texts = [
  "The quick brown fox jumps over the lazy dog.",
  "Machine learning models can process natural language.",
  "Embeddings represent text as dense numerical vectors."
];

const batchEmbeddings = await hfTeiUnified.batchEmbeddings(texts);
console.log(`Generated ${batchEmbeddings.length} embeddings`);
```

### Working with Embeddings

```typescript
// Generate embeddings with custom options
const options = {
  normalize: true,
  maxTokens: 512,
  priority: 'HIGH'
};

const embeddings = await hfTeiUnified.generateEmbeddings("Calculate text similarity with embeddings", options);

// Get embedding dimensions
const dimensions = embeddings[0].length;
console.log(`Embedding dimensions: ${dimensions}`);
```

## Container Mode

The HF TEI Unified backend can manage Docker containers running the HuggingFace Text Embedding Inference server.

### Switching to Container Mode

```typescript
// Switch to container mode
hfTeiUnified.setMode(true);
console.log(`Current mode: ${hfTeiUnified.getMode()}`); // Should output 'container'
```

### Starting a Container

```typescript
import { DeploymentConfig } from 'ipfs-accelerate/api_backends/hf_tei_unified/types';

// Define container deployment configuration
const deployConfig: DeploymentConfig = {
  dockerRegistry: 'ghcr.io/huggingface/text-embeddings-inference',
  containerTag: 'latest',
  gpuDevice: '0', // GPU device ID, use empty string for CPU-only
  modelId: 'BAAI/bge-small-en-v1.5',
  port: 8080,
  env: {
    'HF_API_TOKEN': 'your_api_key' // Optional for accessing gated models
  },
  volumes: ['./cache:/cache'], // Optional volume mounts
  network: 'bridge' // Docker network
};

// Start the container
const containerInfo = await hfTeiUnified.startContainer(deployConfig);
console.log('Container started:', containerInfo);

// Now embeddings are generated using the container
const embeddings = await hfTeiUnified.generateEmbeddings("This is processed by the container.");
```

### Stopping the Container

```typescript
// Stop and remove the container when done
const stopped = await hfTeiUnified.stopContainer();
console.log(`Container stopped: ${stopped}`);
```

## Performance Benchmarking

The backend includes tools for benchmarking embedding performance:

```typescript
// Run a benchmark
const benchmarkOptions = {
  iterations: 10, // Number of iterations to run
  batchSize: 5,   // Batch size for measuring batch performance
  model: 'BAAI/bge-small-en-v1.5' // Model to benchmark
};

const benchmarkResults = await hfTeiUnified.runBenchmark(benchmarkOptions);

console.log('Benchmark results:');
console.log(`- Single embedding time: ${benchmarkResults.singleEmbeddingTime.toFixed(2)} ms`);
console.log(`- Batch embedding time: ${benchmarkResults.batchEmbeddingTime.toFixed(2)} ms`);
console.log(`- Sentences per second: ${benchmarkResults.sentencesPerSecond.toFixed(2)}`);
console.log(`- Batch speedup factor: ${benchmarkResults.batchSpeedupFactor.toFixed(2)}x`);
```

## Advanced Features

### Model Information

```typescript
// Get model details
const modelInfo = await hfTeiUnified.getModelInfo();

console.log('Model information:');
console.log(`- Model ID: ${modelInfo.model_id}`);
console.log(`- Embedding dimensions: ${modelInfo.dim}`);
console.log(`- Status: ${modelInfo.status}`);
if (modelInfo.revision) console.log(`- Revision: ${modelInfo.revision}`);
if (modelInfo.framework) console.log(`- Framework: ${modelInfo.framework}`);
if (modelInfo.quantized !== undefined) console.log(`- Quantized: ${modelInfo.quantized}`);
```

### Testing Endpoint Availability

```typescript
// Check if the endpoint is available
const isAvailable = await hfTeiUnified.testEndpoint();
console.log(`Endpoint available: ${isAvailable}`);
```

### Model Compatibility Checking

```typescript
// Check if models are compatible with the backend
const models = [
  'BAAI/bge-small-en-v1.5',
  'sentence-transformers/all-MiniLM-L6-v2',
  'intfloat/e5-base-v2',
  'thenlper/gte-base',
  'random-model-name'
];

for (const model of models) {
  const isCompatible = hfTeiUnified.isCompatibleModel(model);
  console.log(`${model}: ${isCompatible ? 'Compatible' : 'Not compatible'}`);
}
```

## Error Handling

The HF TEI Unified backend includes comprehensive error handling:

```typescript
try {
  const embeddings = await hfTeiUnified.generateEmbeddings("Test text");
  console.log(`Generated embedding with ${embeddings[0].length} dimensions`);
} catch (error) {
  if (error.message.includes('not found')) {
    console.error('Model not found. Check the model ID and try again.');
  } else if (error.message.includes('Authorization')) {
    console.error('API key error. Check your API key and permissions.');
  } else if (error.message.includes('loading')) {
    console.error('Model is still loading. Try again in a few moments.');
  } else {
    console.error('Error generating embeddings:', error);
  }
}
```

Common error scenarios:
- Invalid API key
- Model not found
- Rate limiting
- Server errors
- Model loading delays
- Container startup issues

## Best Practices

### Environment Variables

Store your API key in environment variables:

```typescript
const apiKey = process.env.HF_API_KEY;
const hfTeiUnified = new HfTeiUnified({}, { hf_api_key: apiKey });
```

### Batch Processing

Use batch processing for multiple texts:

```typescript
// Process multiple texts efficiently
const texts = [];
for (let i = 0; i < 100; i++) {
  texts.push(`Sample text ${i} for embedding generation.`);
}

// More efficient than calling generateEmbeddings 100 times
const batchEmbeddings = await hfTeiUnified.batchEmbeddings(texts);
```

### Container Resources

When using container mode:

1. Ensure sufficient GPU memory for the model
2. Consider setting volume mounts for caching downloaded models
3. Monitor container resource usage for production deployments
4. Stop containers when not in use to free resources

```typescript
// Use container with appropriate resources
const deployConfig = {
  dockerRegistry: 'ghcr.io/huggingface/text-embeddings-inference',
  containerTag: 'latest',
  gpuDevice: '0',
  modelId: 'BAAI/bge-small-en-v1.5',
  port: 8080,
  env: {},
  volumes: ['./cache:/cache'], // Cache downloaded models
  network: 'bridge'
};
```

### Multi-Modal Applications

Use different embedding models for different content types:

```typescript
// Create multiple specialized backend instances
const textBackend = new HfTeiUnified({}, { 
  hf_api_key: apiKey, 
  model_id: 'BAAI/bge-base-en-v1.5' 
});

const codeBackend = new HfTeiUnified({}, { 
  hf_api_key: apiKey, 
  model_id: 'microsoft/unixcoder-base' 
});

// Use appropriate backend for the content type
const textEmbedding = await textBackend.generateEmbeddings("Natural language text");
const codeEmbedding = await codeBackend.generateEmbeddings("function calculateSum(a, b) { return a + b; }");
```

## Compatibility

The HF TEI Unified backend is compatible with:

- A wide range of embedding models on HuggingFace Hub
- Docker containers for self-hosted deployments
- CPU and GPU hardware acceleration
- Various embedding use cases (semantic search, clustering, etc.)

Popular compatible models:
- `BAAI/bge-small-en-v1.5`: Small, efficient English embedding model
- `BAAI/bge-base-en-v1.5`: Base-sized English embedding model
- `BAAI/bge-large-en-v1.5`: Large, high-quality English embedding model
- `sentence-transformers/all-MiniLM-L6-v2`: Compact multilingual model
- `sentence-transformers/all-mpnet-base-v2`: High-quality multilingual model
- `intfloat/e5-base-v2`: Strong general-purpose embedding model
- `thenlper/gte-base`: High-performance embedding model
- `jinaai/jina-embeddings-v2-base-en`: Modern English embedding model

## Examples

### Basic Embedding Example

```typescript
import { HfTeiUnified } from 'ipfs-accelerate/api_backends/hf_tei_unified';

async function generateEmbeddings() {
  const backend = new HfTeiUnified({}, {
    hf_api_key: process.env.HF_API_KEY,
    model_id: 'BAAI/bge-small-en-v1.5'
  });
  
  const texts = [
    "Climate change is a global challenge requiring immediate action.",
    "Renewable energy sources are essential for sustainable development.",
    "Conservation efforts help preserve biodiversity and ecosystems."
  ];
  
  const embeddings = await backend.batchEmbeddings(texts, { normalize: true });
  console.log(`Generated ${embeddings.length} embeddings, each with ${embeddings[0].length} dimensions`);
  
  return embeddings;
}

generateEmbeddings().catch(console.error);
```

### Container Deployment Example

```typescript
import { HfTeiUnified } from 'ipfs-accelerate/api_backends/hf_tei_unified';
import { DeploymentConfig } from 'ipfs-accelerate/api_backends/hf_tei_unified/types';

async function runEmbeddingContainer() {
  try {
    // Initialize the backend in container mode
    const backend = new HfTeiUnified({
      useContainer: true,
      containerUrl: 'http://localhost:8080',
      debug: true
    }, {
      model_id: 'BAAI/bge-small-en-v1.5'
    });
    
    // Configure and start the container
    const config: DeploymentConfig = {
      dockerRegistry: 'ghcr.io/huggingface/text-embeddings-inference',
      containerTag: 'latest',
      gpuDevice: '0',
      modelId: 'BAAI/bge-small-en-v1.5',
      port: 8080,
      env: {},
      volumes: ['./cache:/cache'],
      network: 'bridge'
    };
    
    const containerInfo = await backend.startContainer(config);
    console.log('Container started:', containerInfo);
    
    // Generate embeddings using the container
    const text = "Using a self-hosted container for embeddings generation.";
    const embeddings = await backend.generateEmbeddings(text);
    console.log(`Generated embedding with ${embeddings[0].length} dimensions`);
    
    // Stop the container when done
    await backend.stopContainer();
    console.log('Container stopped');
  } catch (error) {
    console.error('Error:', error);
  }
}

runEmbeddingContainer().catch(console.error);
```

## Advanced Integration Patterns

### Semantic Similarity with Cosine Similarity

Embeddings can be used to calculate semantic similarity between texts:

```typescript
import { HfTeiUnified } from 'ipfs-accelerate/api_backends/hf_tei_unified';

// Helper function to calculate cosine similarity
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

async function semanticSearch() {
  // Initialize the backend
  const backend = new HfTeiUnified({}, {
    hf_api_key: process.env.HF_API_KEY,
    model_id: 'BAAI/bge-small-en-v1.5'
  });
  
  // Query and document corpus
  const query = "How do neural networks learn?";
  const documents = [
    "Neural networks learn by adjusting weights through backpropagation.",
    "Machine learning algorithms improve with more training data.",
    "Transformer models use attention mechanisms for language understanding.",
    "The weather forecast predicts rain tomorrow afternoon."
  ];
  
  // Generate embeddings with normalization (important for cosine similarity)
  const queryEmbedding = (await backend.generateEmbeddings(query, { normalize: true }))[0];
  const documentEmbeddings = await backend.batchEmbeddings(documents, { normalize: true });
  
  // Calculate similarities and rank documents
  const similarities = documentEmbeddings.map(docEmb => cosineSimilarity(queryEmb, docEmb));
  
  // Create ranked results
  const rankedResults = documents.map((doc, i) => ({
    document: doc,
    similarity: similarities[i]
  })).sort((a, b) => b.similarity - a.similarity);
  
  // Output results
  console.log(`Query: "${query}"`);
  console.log("Ranked results:");
  rankedResults.forEach((result, i) => {
    console.log(`${i+1}. [${result.similarity.toFixed(4)}] ${result.document}`);
  });
}

semanticSearch().catch(console.error);
```

### Document Clustering with k-means

Cluster documents based on their semantic meaning:

```typescript
import { HfTeiUnified } from 'ipfs-accelerate/api_backends/hf_tei_unified';

// Simple k-means clustering implementation
function kMeansClustering(vectors: number[][], k: number, iterations: number = 10): number[] {
  // Initialize centroids randomly
  const centroids: number[][] = [];
  for (let i = 0; i < k; i++) {
    centroids.push(vectors[Math.floor(Math.random() * vectors.length)]);
  }
  
  let assignments: number[] = [];
  
  // Run iterations
  for (let iter = 0; iter < iterations; iter++) {
    // Assign points to nearest centroid
    assignments = vectors.map(vector => {
      const distances = centroids.map(centroid => {
        // Euclidean distance
        return Math.sqrt(
          vector.reduce((sum, val, i) => sum + Math.pow(val - centroid[i], 2), 0)
        );
      });
      return distances.indexOf(Math.min(...distances));
    });
    
    // Update centroids
    for (let i = 0; i < k; i++) {
      const clusterPoints = vectors.filter((_, idx) => assignments[idx] === i);
      if (clusterPoints.length > 0) {
        centroids[i] = clusterPoints[0].map((_, dimIdx) => {
          const sum = clusterPoints.reduce((s, p) => s + p[dimIdx], 0);
          return sum / clusterPoints.length;
        });
      }
    }
  }
  
  return assignments;
}

async function clusterDocuments() {
  // Initialize the backend
  const backend = new HfTeiUnified({}, {
    hf_api_key: process.env.HF_API_KEY,
    model_id: 'BAAI/bge-small-en-v1.5'
  });
  
  // Sample documents for clustering
  const documents = [
    "Machine learning models can process text data efficiently.",
    "Neural networks are used for many natural language tasks.",
    "Deep learning has revolutionized the field of AI.",
    "The weather forecast predicts rain today.",
    "Temperatures will be around 65 degrees Fahrenheit.",
    "Expect thunderstorms throughout the evening.",
    "Embedding models convert text into vectors.",
    "Vector representations enable semantic similarity computation."
  ];
  
  // Generate embeddings for all documents
  const embeddings = await backend.batchEmbeddings(documents, { normalize: true });
  
  // Apply k-means clustering with 3 clusters
  const k = 3;
  const clusters = kMeansClustering(embeddings, k);
  
  // Group documents by cluster
  const groupedDocuments: Record<number, string[]> = {};
  clusters.forEach((cluster, index) => {
    if (!groupedDocuments[cluster]) {
      groupedDocuments[cluster] = [];
    }
    groupedDocuments[cluster].push(documents[index]);
  });
  
  // Output the clusters
  Object.entries(groupedDocuments).forEach(([cluster, docs]) => {
    console.log(`Cluster ${parseInt(cluster) + 1}:`);
    docs.forEach(doc => console.log(`  - ${doc}`));
    console.log();
  });
}

clusterDocuments().catch(console.error);
```

### Efficient Embedding Cache

Improve performance by caching embeddings for repeated texts:

```typescript
import { HfTeiUnified } from 'ipfs-accelerate/api_backends/hf_tei_unified';

class EmbeddingCache {
  private cache = new Map<string, number[]>();
  private backend: HfTeiUnified;
  
  constructor(apiKey: string, modelId: string = 'BAAI/bge-small-en-v1.5') {
    this.backend = new HfTeiUnified({}, {
      hf_api_key: apiKey,
      model_id: modelId
    });
  }
  
  // Generate a consistent cache key
  private getCacheKey(text: string): string {
    return text.trim().toLowerCase();
  }
  
  // Get embedding with caching
  async getEmbedding(text: string): Promise<number[]> {
    const key = this.getCacheKey(text);
    
    // Check cache first
    if (this.cache.has(key)) {
      return this.cache.get(key)!;
    }
    
    // Generate embedding if not cached
    const embedding = (await this.backend.generateEmbeddings(text))[0];
    
    // Store in cache
    this.cache.set(key, embedding);
    
    return embedding;
  }
  
  // Get embeddings for multiple texts, using cache where possible
  async batchGetEmbeddings(texts: string[]): Promise<number[][]> {
    const result: number[][] = [];
    const textsToFetch: string[] = [];
    const indices: number[] = [];
    
    // Check which texts are in cache
    for (let i = 0; i < texts.length; i++) {
      const key = this.getCacheKey(texts[i]);
      
      if (this.cache.has(key)) {
        result[i] = this.cache.get(key)!;
      } else {
        textsToFetch.push(texts[i]);
        indices.push(i);
      }
    }
    
    // Fetch missing embeddings
    if (textsToFetch.length > 0) {
      const newEmbeddings = await this.backend.batchEmbeddings(textsToFetch);
      
      // Store new embeddings in cache and result
      for (let i = 0; i < newEmbeddings.length; i++) {
        const key = this.getCacheKey(textsToFetch[i]);
        this.cache.set(key, newEmbeddings[i]);
        result[indices[i]] = newEmbeddings[i];
      }
    }
    
    return result;
  }
  
  // Get cache statistics
  getCacheStats() {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys())
    };
  }
  
  // Clear the cache
  clearCache() {
    this.cache.clear();
  }
}

async function demoEmbeddingCache() {
  const apiKey = process.env.HF_API_KEY || 'your_api_key';
  const cache = new EmbeddingCache(apiKey);
  
  console.log("First request (cache miss)");
  const startTime1 = performance.now();
  const embedding1 = await cache.getEmbedding("This is a test of the embedding cache system.");
  const endTime1 = performance.now();
  console.log(`Time: ${(endTime1 - startTime1).toFixed(2)}ms`);
  
  console.log("\nSecond request (cache hit)");
  const startTime2 = performance.now();
  const embedding2 = await cache.getEmbedding("This is a test of the embedding cache system.");
  const endTime2 = performance.now();
  console.log(`Time: ${(endTime2 - startTime2).toFixed(2)}ms`);
  
  console.log(`\nCache speedup: ${((endTime1 - startTime1) / (endTime2 - startTime2)).toFixed(2)}x`);
  
  // Batch processing
  console.log("\nBatch processing with mixed cache hits/misses");
  const texts = [
    "This is a test of the embedding cache system.", // Cache hit
    "A second example for the cache demonstration.",  // Cache miss
    "A third example that wasn't seen before."        // Cache miss
  ];
  
  const batchEmbeddings = await cache.batchGetEmbeddings(texts);
  console.log(`Processed ${batchEmbeddings.length} texts`);
  console.log(`Cache statistics: ${cache.getCacheStats().size} entries`);
}

demoEmbeddingCache().catch(console.error);
```

### Memory-Efficient Processing of Large Datasets

Process large datasets in smaller batches to manage memory:

```typescript
import { HfTeiUnified } from 'ipfs-accelerate/api_backends/hf_tei_unified';
import fs from 'fs';

async function processLargeDataset(
  filePath: string, 
  outputPath: string,
  batchSize: number = 20
) {
  // Initialize the backend
  const backend = new HfTeiUnified({}, {
    hf_api_key: process.env.HF_API_KEY,
    model_id: 'BAAI/bge-small-en-v1.5'
  });
  
  // Read dataset file
  const data = fs.readFileSync(filePath, 'utf8');
  const lines = data.split('\n').filter(line => line.trim() !== '');
  
  console.log(`Processing ${lines.length} documents in batches of ${batchSize}`);
  
  // Prepare output stream
  const outputStream = fs.createWriteStream(outputPath);
  
  // Process in batches
  const batches = Math.ceil(lines.length / batchSize);
  
  for (let i = 0; i < batches; i++) {
    const start = i * batchSize;
    const end = Math.min(start + batchSize, lines.length);
    const batch = lines.slice(start, end);
    
    console.log(`Processing batch ${i+1}/${batches} (${batch.length} documents)`);
    
    try {
      // Generate embeddings for the batch
      const batchEmbeddings = await backend.batchEmbeddings(batch, { normalize: true });
      
      // Write embeddings to output file
      for (let j = 0; j < batch.length; j++) {
        const embedding = batchEmbeddings[j];
        outputStream.write(JSON.stringify({
          text: batch[j],
          embedding: embedding
        }) + '\n');
      }
      
      console.log(`Batch ${i+1} completed`);
    } catch (error) {
      console.error(`Error processing batch ${i+1}:`, error);
    }
    
    // Add a small delay between batches to avoid rate limiting
    if (i < batches - 1) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  outputStream.end();
  console.log(`Processing complete. Results saved to ${outputPath}`);
}

// Example usage
processLargeDataset('dataset.txt', 'embeddings.jsonl', 20)
  .catch(console.error);
```

## Advanced Feature Example: Cross-Model Comparison

Compare different embedding models to select the best one for your task:

```typescript
import { HfTeiUnified } from 'ipfs-accelerate/api_backends/hf_tei_unified';
import { HfTeiUnifiedOptions, HfTeiUnifiedApiMetadata } from 'ipfs-accelerate/api_backends/hf_tei_unified/types';

// Function to calculate cosine similarity
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

async function compareEmbeddingModels() {
  const apiKey = process.env.HF_API_KEY || 'your_api_key';
  
  // Define models to compare
  const models = {
    'bge-small': 'BAAI/bge-small-en-v1.5',
    'bge-base': 'BAAI/bge-base-en-v1.5',
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'e5-base': 'intfloat/e5-base-v2'
  };
  
  // Define evaluation pairs (similar and dissimilar sentences)
  const evaluationPairs = [
    {
      text1: "Machine learning algorithms improve with more data.",
      text2: "Neural networks perform better with larger datasets.",
      expectedSimilarity: "high"
    },
    {
      text1: "The economic impact of climate change is significant.",
      text2: "Global warming affects economies worldwide.",
      expectedSimilarity: "high"
    },
    {
      text1: "Regular exercise improves cardiovascular health.",
      text2: "The stock market showed gains in tech sectors today.",
      expectedSimilarity: "low"
    },
    {
      text1: "Renewable energy sources are becoming more affordable.",
      text2: "The history of ancient Rome spans many centuries.",
      expectedSimilarity: "low"
    }
  ];
  
  // Results for each model
  const results: Record<string, any> = {};
  
  // Get embeddings and calculate similarities for each model
  for (const [modelName, modelId] of Object.entries(models)) {
    console.log(`\nEvaluating model: ${modelName} (${modelId})`);
    
    // Initialize backend with current model
    const backend = new HfTeiUnified({}, {
      hf_api_key: apiKey,
      model_id: modelId
    });
    
    const modelResults = [];
    
    // Process each evaluation pair
    for (const pair of evaluationPairs) {
      // Get embeddings for each text
      const embedding1 = (await backend.generateEmbeddings(pair.text1, { normalize: true }))[0];
      const embedding2 = (await backend.generateEmbeddings(pair.text2, { normalize: true }))[0];
      
      // Calculate similarity
      const similarity = cosineSimilarity(embedding1, embedding2);
      
      modelResults.push({
        pair,
        similarity,
        correct: (pair.expectedSimilarity === "high" && similarity > 0.5) || 
                (pair.expectedSimilarity === "low" && similarity < 0.5)
      });
      
      console.log(`- Texts: "${pair.text1.slice(0, 20)}..." and "${pair.text2.slice(0, 20)}..."`);
      console.log(`  Similarity: ${similarity.toFixed(4)}, Expected: ${pair.expectedSimilarity}`);
    }
    
    // Calculate accuracy for this model
    const accuracy = modelResults.filter(r => r.correct).length / modelResults.length;
    console.log(`Accuracy: ${(accuracy * 100).toFixed(2)}%`);
    
    // Store results
    results[modelName] = {
      modelId,
      pairs: modelResults,
      accuracy,
      avgSimilarityForHigh: modelResults
        .filter(r => r.pair.expectedSimilarity === "high")
        .reduce((sum, r) => sum + r.similarity, 0) / 
        modelResults.filter(r => r.pair.expectedSimilarity === "high").length,
      avgSimilarityForLow: modelResults
        .filter(r => r.pair.expectedSimilarity === "low")
        .reduce((sum, r) => sum + r.similarity, 0) / 
        modelResults.filter(r => r.pair.expectedSimilarity === "low").length
    };
  }
  
  // Find the best model based on accuracy
  const bestModel = Object.entries(results)
    .sort((a, b) => b[1].accuracy - a[1].accuracy)[0];
  
  console.log(`\nBest model: ${bestModel[0]} (${bestModel[1].modelId})`);
  console.log(`Accuracy: ${(bestModel[1].accuracy * 100).toFixed(2)}%`);
  console.log(`Avg similarity for expected high: ${bestModel[1].avgSimilarityForHigh.toFixed(4)}`);
  console.log(`Avg similarity for expected low: ${bestModel[1].avgSimilarityForLow.toFixed(4)}`);
  
  return results;
}

compareEmbeddingModels().catch(console.error);
```

## Additional Resources

- [HuggingFace Inference API Documentation](https://huggingface.co/docs/api-inference/index)
- [Text Embeddings Inference GitHub Repository](https://github.com/huggingface/text-embeddings-inference)
- [HuggingFace Embedding Models](https://huggingface.co/models?pipeline_tag=feature-extraction)
- [IPFS Accelerate JS Documentation](https://github.com/your-org/ipfs-accelerate-js)
- [Docker Installation Guide](https://docs.docker.com/get-docker/)
- [Comprehensive Example](https://github.com/your-org/ipfs-accelerate-js/blob/main/examples/hf_tei_unified_comprehensive_example.ts)