/**
 * HuggingFace Text Embedding Inference Unified API Backend - Comprehensive Example
 * 
 * This comprehensive example demonstrates the full capabilities of the HF TEI Unified backend:
 * 
 * 1. Configuration options and initialization with best practices
 * 2. Multiple model support and model management
 * 3. Single and batch embedding generation with optimization
 * 4. Performance benchmarking and optimization
 * 5. Robust error handling and retry strategies
 * 6. Container-based deployment and management
 * 7. Advanced integration patterns
 * 8. Circuit breaker pattern for API resilience
 * 9. Custom batching strategies
 * 10. Memory-efficient processing
 */

// Import the HfTeiUnified class and types
import { HfTeiUnified } from '../src/api_backends/hf_tei_unified/hf_tei_unified';
import {
  HfTeiUnifiedOptions,
  HfTeiUnifiedApiMetadata,
  EmbeddingOptions,
  DeploymentConfig,
  PerformanceMetrics,
  ContainerInfo,
  HfTeiModelInfo
} from '../src/api_backends/hf_tei_unified/types';
import path from 'path';
import fs from 'fs';

// Polyfill for performance.now() if not available in environment
if (typeof performance === 'undefined') {
  const { performance } = require('perf_hooks');
  global.performance = performance;
}

/**
 * Comprehensive example for the HF TEI Unified backend
 */
async function main() {
  console.log('\n----------------------------------------------------------');
  console.log('HF TEI UNIFIED BACKEND - COMPREHENSIVE EXAMPLE');
  console.log('----------------------------------------------------------\n');
  
  const results: Record<string, any> = {};
  
  try {
    // -------------------------------------------------------------------------------
    // SECTION 1: INITIALIZATION AND CONFIGURATION
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 1: INITIALIZATION AND CONFIGURATION');
    
    // 1.1 Environment and API Key Management
    console.log('\n📋 1.1 Environment and API Key Management:');
    
    // Get API key from environment variable, parameter, or configuration file
    const envApiKey = process.env.HF_API_KEY;
    let apiKey: string | undefined = undefined;
    
    if (envApiKey) {
      console.log('   ✓ Using API key from environment variable');
      apiKey = envApiKey;
    } else {
      // Check for API key in configuration file
      const configPath = path.join(process.cwd(), 'hf_config.json');
      if (fs.existsSync(configPath)) {
        try {
          const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
          if (config.api_key) {
            console.log('   ✓ Using API key from configuration file');
            apiKey = config.api_key;
          }
        } catch (error) {
          console.log('   × Failed to read configuration file');
        }
      }
      
      // Fallback to demo mode
      if (!apiKey) {
        console.log('   ⚠ No API key found, running in demo mode with limited functionality');
        console.log('   ℹ To use full functionality, set HF_API_KEY environment variable');
      }
    }
    
    // 1.2 Advanced Configuration Options
    console.log('\n📋 1.2 Advanced Configuration Options:');
    
    // Set up advanced configuration options
    const options: HfTeiUnifiedOptions = {
      apiUrl: 'https://api-inference.huggingface.co/models',
      containerUrl: 'http://localhost:8080',
      maxRetries: 3,                // Number of retries for failed requests
      requestTimeout: 30000,        // 30 second timeout
      useRequestQueue: true,        // Use request queue for rate limiting
      debug: false,                 // Enable/disable debug logging
      useContainer: false,          // Start in API mode (not container mode)
      dockerRegistry: 'ghcr.io/huggingface/text-embeddings-inference',
      containerTag: 'latest',
      gpuDevice: '0'                // GPU device ID for container mode
    };
    
    console.log('   ✓ Configured with timeout:', options.requestTimeout, 'ms');
    console.log('   ✓ Retries:', options.maxRetries);
    console.log('   ✓ Request queue:', options.useRequestQueue ? 'enabled' : 'disabled');
    console.log('   ✓ Mode:', options.useContainer ? 'container' : 'API');
    
    // 1.3 Metadata Configuration
    console.log('\n📋 1.3 Metadata Configuration:');
    
    // Define multiple models to demonstrate model management
    const defaultModel = 'BAAI/bge-small-en-v1.5';
    const availableModels = {
      'small-768': 'BAAI/bge-small-en-v1.5',         // 768-dim embeddings, fast
      'base-768': 'BAAI/bge-base-en-v1.5',           // 768-dim embeddings, better quality
      'large-1024': 'BAAI/bge-large-en-v1.5',        // 1024-dim embeddings, best quality
      'minilm-384': 'sentence-transformers/all-MiniLM-L6-v2', // 384-dim embeddings, very fast
      'e5-768': 'intfloat/e5-base-v2',              // 768-dim embeddings, good for search
      'jina-512': 'jinaai/jina-embeddings-v2-small-en' // 512-dim embeddings, good balance
    };
    
    // Set up metadata with API key and default model
    const metadata: HfTeiUnifiedApiMetadata = {
      hf_api_key: apiKey,
      model_id: defaultModel
    };
    
    console.log('   ✓ Default model:', defaultModel);
    console.log('   ✓ Available models:', Object.keys(availableModels).length);
    
    // 1.4 Backend Initialization
    console.log('\n📋 1.4 Backend Initialization:');
    
    // Create the HF TEI Unified backend instance
    const hfTeiUnified = new HfTeiUnified(options, metadata);
    
    console.log('   ✓ HF TEI Unified backend initialized successfully');
    console.log('   ✓ Current mode:', hfTeiUnified.getMode());
    console.log('   ✓ Default model:', hfTeiUnified.getDefaultModel());
    
    // Store results for reporting
    results.initialization = {
      mode: hfTeiUnified.getMode(),
      defaultModel: hfTeiUnified.getDefaultModel(),
      apiKeyAvailable: !!apiKey
    };
    
    // -------------------------------------------------------------------------------
    // SECTION 2: MODEL COMPATIBILITY AND INFORMATION
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 2: MODEL COMPATIBILITY AND INFORMATION');
    
    // 2.1 Model Compatibility Check
    console.log('\n📋 2.1 Model Compatibility Check:');
    
    // Check compatibility for various models
    const modelsToCheck = [
      ...Object.values(availableModels),
      'openai/clip-base',          // Vision model (should be incompatible)
      'facebook/bart-large',       // Text generation model (should be incompatible)
      'microsoft/unixcoder-base',  // Code embedding model (should be compatible)
      'random-model-name'          // Unknown model (should be incompatible)
    ];
    
    const compatibilityResults: Record<string, boolean> = {};
    for (const model of modelsToCheck) {
      const isCompatible = hfTeiUnified.isCompatibleModel(model);
      compatibilityResults[model] = isCompatible;
      console.log(`   ${isCompatible ? '✓' : '×'} ${model}: ${isCompatible ? 'Compatible' : 'Not compatible'}`);
    }
    
    results.modelCompatibility = compatibilityResults;
    
    // 2.2 Model Information
    console.log('\n📋 2.2 Model Information:');
    
    // Attempt to get model information for the default model
    try {
      const modelInfo = await hfTeiUnified.getModelInfo();
      console.log('   ✓ Model information retrieved successfully:');
      console.log(`     - Model ID: ${modelInfo.model_id}`);
      console.log(`     - Embedding dimensions: ${modelInfo.dim}`);
      console.log(`     - Status: ${modelInfo.status}`);
      
      if (modelInfo.revision) console.log(`     - Revision: ${modelInfo.revision}`);
      if (modelInfo.framework) console.log(`     - Framework: ${modelInfo.framework}`);
      if (modelInfo.quantized !== undefined) console.log(`     - Quantized: ${modelInfo.quantized}`);
      
      results.modelInfo = modelInfo;
    } catch (error) {
      console.log('   × Error retrieving model information:');
      console.log(`     ${error instanceof Error ? error.message : String(error)}`);
      console.log('     Continuing with limited model information');
      
      // Use default values for dimensions based on model
      const defaultDimensions: Record<string, number> = {
        'BAAI/bge-small-en-v1.5': 768,
        'BAAI/bge-base-en-v1.5': 768,
        'BAAI/bge-large-en-v1.5': 1024,
        'sentence-transformers/all-MiniLM-L6-v2': 384,
        'intfloat/e5-base-v2': 768,
        'jinaai/jina-embeddings-v2-small-en': 512
      };
      
      const estimatedDims = defaultDimensions[hfTeiUnified.getDefaultModel()] || 768;
      console.log(`     - Estimated dimensions: ${estimatedDims}`);
      
      results.modelInfo = {
        model_id: hfTeiUnified.getDefaultModel(),
        dim: estimatedDims,
        status: 'unknown',
        estimated: true
      };
    }
    
    // 2.3 API Endpoint Verification
    console.log('\n📋 2.3 API Endpoint Verification:');
    
    try {
      const isEndpointAvailable = await hfTeiUnified.testEndpoint();
      console.log(`   ${isEndpointAvailable ? '✓' : '×'} Endpoint available: ${isEndpointAvailable}`);
      
      if (!isEndpointAvailable && !apiKey) {
        console.log('   ℹ Endpoint availability may be limited without an API key');
      }
      
      results.endpointAvailable = isEndpointAvailable;
    } catch (error) {
      console.log('   × Error testing endpoint:', error);
      results.endpointAvailable = false;
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 3: BASIC EMBEDDING GENERATION
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 3: BASIC EMBEDDING GENERATION');
    
    // 3.1 Generate Single Embedding
    console.log('\n📋 3.1 Generate Single Embedding:');
    
    try {
      const text = "This is a sample text for embedding generation.";
      console.log(`   Input text: "${text}"`);
      
      const embeddings = await hfTeiUnified.generateEmbeddings(text);
      
      console.log(`   ✓ Generated embedding with ${embeddings[0].length} dimensions`);
      console.log('   ✓ First 5 values:', embeddings[0].slice(0, 5).map(v => v.toFixed(6)));
      
      results.singleEmbedding = {
        dimensions: embeddings[0].length,
        sample: embeddings[0].slice(0, 5)
      };
    } catch (error) {
      console.log('   × Error generating embedding:', error);
      console.log('   Using mock data for demonstration...');
      
      // Use mock embeddings for demonstration
      const mockEmbedding = Array(results.modelInfo.dim || 768).fill(0).map(() => Math.random() * 2 - 1);
      console.log(`   Generated mock embedding with ${mockEmbedding.length} dimensions`);
      console.log('   First 5 values:', mockEmbedding.slice(0, 5).map(v => v.toFixed(6)));
      
      results.singleEmbedding = {
        dimensions: mockEmbedding.length,
        sample: mockEmbedding.slice(0, 5),
        mock: true
      };
    }
    
    // 3.2 Generate Embeddings with Options
    console.log('\n📋 3.2 Generate Embeddings with Options:');
    
    try {
      const text = "Embedding options provide control over normalization and token limits.";
      
      const options: EmbeddingOptions = {
        normalize: true,      // Apply L2 normalization
        maxTokens: 512,       // Limit to 512 tokens
        priority: 'HIGH'      // Process with high priority
      };
      
      console.log(`   Input text: "${text}"`);
      console.log('   Options:');
      console.log(`     - normalize: ${options.normalize}`);
      console.log(`     - maxTokens: ${options.maxTokens}`);
      console.log(`     - priority: ${options.priority}`);
      
      const embeddings = await hfTeiUnified.generateEmbeddings(text, options);
      
      console.log(`   ✓ Generated embedding with ${embeddings[0].length} dimensions`);
      
      // Verify normalization (L2 norm should be close to 1.0)
      const norm = Math.sqrt(embeddings[0].reduce((sum, val) => sum + val * val, 0));
      console.log(`   ✓ L2 Norm: ${norm.toFixed(6)} (should be ~1.0 for normalized vectors)`);
      
      results.embeddingWithOptions = {
        dimensions: embeddings[0].length,
        norm: norm
      };
    } catch (error) {
      console.log('   × Error generating embedding with options:', error);
      results.embeddingWithOptions = { error: String(error) };
    }
    
    // 3.3 Generate Embeddings with Different Models
    console.log('\n📋 3.3 Generate Embeddings with Different Models:');
    
    // Try with different models if available
    const modelResults: Record<string, any> = {};
    
    for (const [modelKey, modelId] of Object.entries(availableModels).slice(0, 3)) {
      try {
        console.log(`   Model: ${modelKey} (${modelId})`);
        
        const text = "Testing different embedding models for comparison.";
        const options: EmbeddingOptions = {
          model: modelId,
          normalize: true
        };
        
        const startTime = performance.now();
        const embeddings = await hfTeiUnified.generateEmbeddings(text, options);
        const endTime = performance.now();
        
        const dimensions = embeddings[0].length;
        const duration = endTime - startTime;
        
        console.log(`   ✓ Generated ${dimensions}-dimensional embedding in ${duration.toFixed(2)}ms`);
        modelResults[modelKey] = {
          dimensions,
          duration,
          model: modelId
        };
      } catch (error) {
        console.log(`   × Error with model ${modelKey}:`, error);
        modelResults[modelKey] = { error: String(error) };
      }
    }
    
    results.modelComparison = modelResults;
    
    // -------------------------------------------------------------------------------
    // SECTION 4: BATCH PROCESSING
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 4: BATCH PROCESSING');
    
    // 4.1 Basic Batch Processing
    console.log('\n📋 4.1 Basic Batch Processing:');
    
    try {
      const texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can process natural language.",
        "Embeddings represent text as dense numerical vectors."
      ];
      
      console.log(`   Processing batch of ${texts.length} texts`);
      
      const options: EmbeddingOptions = {
        normalize: true,
        maxTokens: 512,
        priority: 'NORMAL'
      };
      
      const startTime = performance.now();
      const batchEmbeddings = await hfTeiUnified.batchEmbeddings(texts, options);
      const endTime = performance.now();
      
      console.log(`   ✓ Generated ${batchEmbeddings.length} embeddings in ${(endTime - startTime).toFixed(2)}ms`);
      console.log(`   ✓ Each embedding has ${batchEmbeddings[0].length} dimensions`);
      
      results.batchProcessing = {
        count: batchEmbeddings.length,
        dimensions: batchEmbeddings[0].length,
        duration: endTime - startTime
      };
    } catch (error) {
      console.log('   × Error generating batch embeddings:', error);
      results.batchProcessing = { error: String(error) };
    }
    
    // 4.2 Custom Batching for Large Datasets
    console.log('\n📋 4.2 Custom Batching for Large Datasets:');
    
    // Helper function to generate dummy texts
    const generateDummyTexts = (count: number): string[] => {
      const templates = [
        "This is example text number {n} for embedding testing.",
        "Sample sentence {n} demonstrates embedding capabilities.",
        "Testing batch processing with example {n} in the sequence.",
        "Embedding model handles text number {n} efficiently."
      ];
      
      return Array.from({ length: count }, (_, i) => {
        const template = templates[i % templates.length];
        return template.replace('{n}', (i + 1).toString());
      });
    };
    
    // Create a larger dataset
    const largeDataset = generateDummyTexts(20);
    console.log(`   Created dataset with ${largeDataset.length} texts`);
    
    // Process in smaller batches for memory efficiency
    const batchSize = 5;
    const batches = Math.ceil(largeDataset.length / batchSize);
    console.log(`   Processing in ${batches} batches of size ${batchSize}`);
    
    const allEmbeddings: number[][] = [];
    let totalDuration = 0;
    
    try {
      for (let i = 0; i < batches; i++) {
        const start = i * batchSize;
        const end = Math.min(start + batchSize, largeDataset.length);
        const batchTexts = largeDataset.slice(start, end);
        
        console.log(`   Processing batch ${i + 1}/${batches} with ${batchTexts.length} texts`);
        
        const startTime = performance.now();
        const batchResults = await hfTeiUnified.batchEmbeddings(batchTexts, { normalize: true });
        const endTime = performance.now();
        
        const batchDuration = endTime - startTime;
        totalDuration += batchDuration;
        
        console.log(`   ✓ Batch ${i + 1} processed in ${batchDuration.toFixed(2)}ms`);
        allEmbeddings.push(...batchResults);
      }
      
      console.log(`   ✓ All batches completed successfully`);
      console.log(`   ✓ Generated ${allEmbeddings.length} embeddings in ${totalDuration.toFixed(2)}ms`);
      console.log(`   ✓ Average time per text: ${(totalDuration / largeDataset.length).toFixed(2)}ms`);
      
      results.customBatching = {
        totalTexts: largeDataset.length,
        batches,
        batchSize,
        totalDuration,
        averageTimePerText: totalDuration / largeDataset.length
      };
    } catch (error) {
      console.log('   × Error in custom batching:', error);
      results.customBatching = { error: String(error) };
    }
    
    // 4.3 Parallel Processing (Limited by API Rate Limits)
    console.log('\n📋 4.3 Parallel Processing (with caution):');
    console.log('   ⚠ Note: Parallel processing may trigger API rate limits');
    console.log('   ⚠ Use with caution and respect API provider policies');
    
    // Create test data for parallel processing
    const parallelTexts = [
      "Text for parallel processing example 1.",
      "Text for parallel processing example 2.",
      "Text for parallel processing example 3."
    ];
    
    try {
      console.log(`   Processing ${parallelTexts.length} texts in parallel (demonstration)`);
      
      // Process texts in parallel (use with caution due to rate limits)
      const startTime = performance.now();
      
      // With useRequestQueue=true (default), this will be automatically rate-limited
      const promises = parallelTexts.map(text => 
        hfTeiUnified.generateEmbeddings(text, { normalize: true })
      );
      
      const parallelResults = await Promise.all(promises);
      const endTime = performance.now();
      
      console.log(`   ✓ Parallel processing completed in ${(endTime - startTime).toFixed(2)}ms`);
      console.log(`   ✓ Generated ${parallelResults.length} embeddings`);
      
      results.parallelProcessing = {
        count: parallelResults.length,
        duration: endTime - startTime,
        averageTimePerText: (endTime - startTime) / parallelTexts.length
      };
    } catch (error) {
      console.log('   × Error in parallel processing:', error);
      results.parallelProcessing = { error: String(error) };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 5: BENCHMARKING AND PERFORMANCE
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 5: BENCHMARKING AND PERFORMANCE');
    
    // 5.1 Basic Benchmark
    console.log('\n📋 5.1 Basic Benchmark:');
    
    try {
      const benchmarkOptions = {
        iterations: 3,           // Number of iterations for reliable results
        batchSize: 5,            // Batch size for measuring throughput
        model: defaultModel      // Model to benchmark
      };
      
      console.log(`   Running benchmark with ${benchmarkOptions.iterations} iterations`);
      console.log(`   Batch size: ${benchmarkOptions.batchSize}`);
      console.log(`   Model: ${benchmarkOptions.model}`);
      
      const benchmarkResults = await hfTeiUnified.runBenchmark(benchmarkOptions);
      
      console.log('   ✓ Benchmark results:');
      console.log(`     - Single embedding time: ${benchmarkResults.singleEmbeddingTime.toFixed(2)} ms`);
      console.log(`     - Batch embedding time: ${benchmarkResults.batchEmbeddingTime.toFixed(2)} ms`);
      console.log(`     - Sentences per second: ${benchmarkResults.sentencesPerSecond.toFixed(2)}`);
      console.log(`     - Batch speedup factor: ${benchmarkResults.batchSpeedupFactor.toFixed(2)}x`);
      
      results.basicBenchmark = benchmarkResults;
    } catch (error) {
      console.log('   × Error running benchmark:', error);
      results.basicBenchmark = { error: String(error) };
    }
    
    // 5.2 Model Comparison Benchmark
    console.log('\n📋 5.2 Model Comparison Benchmark:');
    
    const modelBenchmarks: Record<string, PerformanceMetrics> = {};
    const modelsToCompare = Object.entries(availableModels).slice(0, 2); // Limit to 2 models for demo
    
    try {
      for (const [modelKey, modelId] of modelsToCompare) {
        console.log(`   Benchmarking model: ${modelKey} (${modelId})`);
        
        try {
          const benchmarkOptions = {
            iterations: 2,
            batchSize: 3,
            model: modelId
          };
          
          const result = await hfTeiUnified.runBenchmark(benchmarkOptions);
          
          console.log(`   ✓ ${modelKey}: ${result.sentencesPerSecond.toFixed(2)} sentences/sec`);
          modelBenchmarks[modelKey] = result;
        } catch (error) {
          console.log(`   × Error benchmarking ${modelKey}:`, error);
        }
      }
      
      // Find the best performing model
      if (Object.keys(modelBenchmarks).length > 0) {
        const bestModel = Object.entries(modelBenchmarks).reduce(
          (best, [model, metrics]) => 
            metrics.sentencesPerSecond > best.metrics.sentencesPerSecond 
              ? { model, metrics } 
              : best,
          { model: '', metrics: { sentencesPerSecond: 0 } as PerformanceMetrics }
        );
        
        console.log(`   ✓ Best performing model: ${bestModel.model} with ${bestModel.metrics.sentencesPerSecond.toFixed(2)} sentences/sec`);
      }
      
      results.modelBenchmarks = modelBenchmarks;
    } catch (error) {
      console.log('   × Error in model comparison benchmark:', error);
      results.modelBenchmarks = { error: String(error) };
    }
    
    // 5.3 Custom Performance Metrics
    console.log('\n📋 5.3 Custom Performance Metrics:');
    
    // Track memory usage if available
    let memoryUsage = null;
    try {
      if (process && process.memoryUsage) {
        const memory = process.memoryUsage();
        memoryUsage = {
          heapUsed: Math.round(memory.heapUsed / 1024 / 1024),
          heapTotal: Math.round(memory.heapTotal / 1024 / 1024),
          rss: Math.round(memory.rss / 1024 / 1024)
        };
        
        console.log('   ✓ Memory usage:');
        console.log(`     - Heap used: ${memoryUsage.heapUsed} MB`);
        console.log(`     - Heap total: ${memoryUsage.heapTotal} MB`);
        console.log(`     - RSS: ${memoryUsage.rss} MB`);
      } else {
        console.log('   ℹ Memory usage metrics not available in this environment');
      }
    } catch (error) {
      console.log('   × Error getting memory usage:', error);
    }
    
    // Calculate token rate (estimated)
    try {
      const text = "This is a performance test for token rate calculation.";
      const tokens = text.split(' ').length; // Rough estimate
      
      const iterations = 5;
      const startTime = performance.now();
      
      for (let i = 0; i < iterations; i++) {
        await hfTeiUnified.generateEmbeddings(text);
      }
      
      const endTime = performance.now();
      const duration = endTime - startTime;
      const tokensPerSecond = (tokens * iterations) / (duration / 1000);
      
      console.log(`   ✓ Token rate: ${tokensPerSecond.toFixed(2)} tokens/second (estimated)`);
      
      results.customMetrics = {
        tokensPerSecond,
        memoryUsage
      };
    } catch (error) {
      console.log('   × Error calculating token rate:', error);
      results.customMetrics = { error: String(error), memoryUsage };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 6: ERROR HANDLING
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 6: ERROR HANDLING');
    
    // 6.1 Basic Error Handling
    console.log('\n📋 6.1 Basic Error Handling:');
    
    // Example: Invalid model error
    try {
      console.log('   Testing error handling with invalid model');
      
      await hfTeiUnified.generateEmbeddings("Test text", {
        model: 'invalid-model-id'
      });
      
      console.log('   ✗ Expected error was not thrown');
    } catch (error) {
      console.log(`   ✓ Successfully caught error: ${error instanceof Error ? error.message : String(error)}`);
      results.errorHandling = { modelError: String(error) };
    }
    
    // 6.2 Circuit Breaker Pattern
    console.log('\n📋 6.2 Circuit Breaker Pattern:');
    console.log('   ℹ The HF TEI Unified backend uses a circuit breaker pattern');
    console.log('   ℹ This prevents repeated API calls when the service is experiencing issues');
    console.log('   ℹ After consecutive failures, the circuit "opens" to prevent further calls');
    console.log('   ℹ The circuit will "close" after a reset timeout, allowing calls again');
    
    // 6.3 Graceful Error Recovery
    console.log('\n📋 6.3 Graceful Error Recovery:');
    
    try {
      // Simulate a temporary error situation with an invalid model
      console.log('   Demonstrating recovery from temporary error');
      
      // First, a failing call with invalid model
      try {
        await hfTeiUnified.generateEmbeddings("Error recovery test", {
          model: 'non-existent-model'
        });
      } catch (error) {
        console.log('   ✓ Error triggered as expected (controlled test)');
      }
      
      // Now recover by using a valid model
      const recoveryResult = await hfTeiUnified.generateEmbeddings("Error recovery test");
      
      console.log('   ✓ Successfully recovered from error');
      console.log(`   ✓ Generated embedding with ${recoveryResult[0].length} dimensions`);
      
      results.errorRecovery = { successful: true, dimensions: recoveryResult[0].length };
    } catch (error) {
      console.log('   × Recovery failed:', error);
      results.errorRecovery = { successful: false, error: String(error) };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 7: CONTAINER MANAGEMENT
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 7: CONTAINER MANAGEMENT');
    console.log('   ⚠ Container operations are shown for demonstration only');
    console.log('   ⚠ Actual container start/stop is commented out to prevent unintended effects');
    
    // 7.1 Container Configuration
    console.log('\n📋 7.1 Container Configuration:');
    
    const deployConfig: DeploymentConfig = {
      dockerRegistry: 'ghcr.io/huggingface/text-embeddings-inference',
      containerTag: 'latest',
      gpuDevice: '0',
      modelId: defaultModel,
      port: 8080,
      env: {
        'HF_API_TOKEN': apiKey || ''
      },
      volumes: ['./cache:/cache'],
      network: 'bridge'
    };
    
    console.log('   ✓ Container configuration prepared:');
    console.log(`     - Image: ${deployConfig.dockerRegistry}:${deployConfig.containerTag}`);
    console.log(`     - Model: ${deployConfig.modelId}`);
    console.log(`     - Port: ${deployConfig.port}`);
    console.log(`     - GPU: ${deployConfig.gpuDevice}`);
    
    results.containerConfig = {
      image: `${deployConfig.dockerRegistry}:${deployConfig.containerTag}`,
      port: deployConfig.port,
      gpu: deployConfig.gpuDevice,
      model: deployConfig.modelId
    };
    
    // 7.2 Container Mode Switching
    console.log('\n📋 7.2 Container Mode Switching:');
    
    // Switch to container mode
    console.log('   Switching to container mode');
    hfTeiUnified.setMode(true);
    console.log(`   ✓ Current mode: ${hfTeiUnified.getMode()}`);
    
    // Switch back to API mode
    console.log('   Switching back to API mode');
    hfTeiUnified.setMode(false);
    console.log(`   ✓ Current mode: ${hfTeiUnified.getMode()}`);
    
    results.containerModeSwitching = {
      finalMode: hfTeiUnified.getMode()
    };
    
    // 7.3 Container Operations (Demonstration Only)
    console.log('\n📋 7.3 Container Operations (Demonstration Only):');
    
    // Show container operation example (without actually executing)
    console.log('   ℹ Container start/stop operations are commented out');
    console.log('   ℹ In a real application, you would use:');
    console.log('     const containerInfo = await hfTeiUnified.startContainer(deployConfig);');
    console.log('     // Use container for embeddings');
    console.log('     const stopped = await hfTeiUnified.stopContainer();');
    
    // Output the container command that would be executed
    try {
      const imageName = `${deployConfig.dockerRegistry}:${deployConfig.containerTag}`;
      const volumes = deployConfig.volumes?.length 
        ? deployConfig.volumes.map(v => `-v ${v}`).join(' ') 
        : '';
      
      const envVars = Object.entries(deployConfig.env || {})
        .map(([key, value]) => `-e ${key}=${value}`)
        .join(' ');
      
      const gpuArgs = deployConfig.gpuDevice 
        ? `--gpus device=${deployConfig.gpuDevice}` 
        : '';
      
      const networkArg = deployConfig.network 
        ? `--network=${deployConfig.network}` 
        : '';
      
      const containerName = `hf-tei-example`;
      const command = `docker run -d --name ${containerName} \
        -p ${deployConfig.port}:80 \
        ${gpuArgs} \
        ${envVars} \
        ${volumes} \
        ${networkArg} \
        ${imageName} \
        --model-id ${deployConfig.modelId}`;
      
      console.log('   ℹ Example container command:');
      console.log(`     ${command.replace(/\n\s+/g, ' ')}`);
      
      results.containerCommand = command.replace(/\n\s+/g, ' ');
    } catch (error) {
      console.log('   × Error generating container command:', error);
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 8: ADVANCED INTEGRATION PATTERNS
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 8: ADVANCED INTEGRATION PATTERNS');
    
    // 8.1 Semantic Similarity
    console.log('\n📋 8.1 Semantic Similarity:');
    
    // Helper function to calculate cosine similarity
    const cosineSimilarity = (vecA: number[], vecB: number[]): number => {
      const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
      const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
      const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
      return dotProduct / (magnitudeA * magnitudeB);
    };
    
    try {
      const queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What's the weather like today?"
      ];
      
      const documents = [
        "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.",
        "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
        "The forecast predicts rain and thunderstorms throughout the day with temperatures around 65°F."
      ];
      
      console.log('   Computing semantic similarities between queries and documents');
      
      // Generate embeddings for queries and documents
      const queryEmbeddings = await hfTeiUnified.batchEmbeddings(queries, { normalize: true });
      const docEmbeddings = await hfTeiUnified.batchEmbeddings(documents, { normalize: true });
      
      // Compute similarity matrix
      const similarityMatrix: number[][] = [];
      
      for (let i = 0; i < queryEmbeddings.length; i++) {
        const similarities: number[] = [];
        
        for (let j = 0; j < docEmbeddings.length; j++) {
          const similarity = cosineSimilarity(queryEmbeddings[i], docEmbeddings[j]);
          similarities.push(similarity);
        }
        
        similarityMatrix.push(similarities);
      }
      
      // Find best matches
      for (let i = 0; i < queries.length; i++) {
        const bestMatchIndex = similarityMatrix[i].indexOf(Math.max(...similarityMatrix[i]));
        console.log(`   Query: "${queries[i]}"`);
        console.log(`   Best match: "${documents[bestMatchIndex].slice(0, 70)}..."`);
        console.log(`   Similarity: ${similarityMatrix[i][bestMatchIndex].toFixed(4)}`);
        console.log('');
      }
      
      results.semanticSimilarity = {
        similarityMatrix,
        queryCount: queries.length,
        documentCount: documents.length
      };
    } catch (error) {
      console.log('   × Error calculating semantic similarity:', error);
      results.semanticSimilarity = { error: String(error) };
    }
    
    // 8.2 Document Clustering
    console.log('\n📋 8.2 Document Clustering (Simple K-Means):');
    
    // Simple k-means clustering implementation
    const kMeansClustering = (vectors: number[][], k: number, iterations: number = 10): number[] => {
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
    };
    
    try {
      const documents = [
        "Machine learning models can process text data efficiently.",
        "Neural networks are used for many natural language tasks.",
        "Deep learning has revolutionized the field of AI.",
        "The weather forecast predicts rain today.",
        "Temperatures will be around 65 degrees Fahrenheit.",
        "Expect thunderstorms throughout the evening.",
        "Embedding models convert text into vectors.",
        "Vector representations enable semantic similarity computation.",
      ];
      
      console.log(`   Clustering ${documents.length} documents into 3 groups`);
      
      // Generate embeddings for documents
      const embeddings = await hfTeiUnified.batchEmbeddings(documents, { normalize: true });
      
      // Apply k-means clustering
      const k = 3;
      const clusters = kMeansClustering(embeddings, k);
      
      // Display clusters
      for (let i = 0; i < k; i++) {
        const clusterDocs = documents.filter((_, idx) => clusters[idx] === i);
        console.log(`   Cluster ${i + 1} (${clusterDocs.length} documents):`);
        clusterDocs.forEach(doc => console.log(`     - "${doc}"`));
        console.log('');
      }
      
      results.documentClustering = {
        clusters,
        documentCount: documents.length,
        k
      };
    } catch (error) {
      console.log('   × Error performing document clustering:', error);
      results.documentClustering = { error: String(error) };
    }
    
    // 8.3 Embeddings Cache for Efficiency
    console.log('\n📋 8.3 Embeddings Cache for Efficiency:');
    
    try {
      // Create a simple embedding cache
      const embeddingCache = new Map<string, number[]>();
      
      // Function to get embeddings with cache
      const getCachedEmbedding = async (text: string): Promise<number[]> => {
        // Normalize text for consistent cache keys
        const normalizedText = text.trim().toLowerCase();
        
        // Check cache first
        if (embeddingCache.has(normalizedText)) {
          return embeddingCache.get(normalizedText)!;
        }
        
        // Generate embedding if not in cache
        const embeddings = await hfTeiUnified.generateEmbeddings(text);
        const embedding = embeddings[0];
        
        // Store in cache
        embeddingCache.set(normalizedText, embedding);
        
        return embedding;
      };
      
      console.log('   Testing embedding cache with repeated queries');
      
      // First query - should generate new embedding
      const text = "This is a test of the embedding cache system.";
      console.log(`   Query: "${text}"`);
      
      console.log('   First request (cache miss)');
      const startTime1 = performance.now();
      const embedding1 = await getCachedEmbedding(text);
      const endTime1 = performance.now();
      console.log(`   ✓ Generated embedding with ${embedding1.length} dimensions`);
      console.log(`   ✓ Time: ${(endTime1 - startTime1).toFixed(2)}ms`);
      
      // Second query - should use cache
      console.log('   Second request (cache hit)');
      const startTime2 = performance.now();
      const embedding2 = await getCachedEmbedding(text);
      const endTime2 = performance.now();
      console.log(`   ✓ Retrieved embedding with ${embedding2.length} dimensions`);
      console.log(`   ✓ Time: ${(endTime2 - startTime2).toFixed(2)}ms`);
      
      // Slight variation - should not use cache
      const text2 = text + " With a small addition.";
      console.log('   Third request with variation (cache miss)');
      const startTime3 = performance.now();
      const embedding3 = await getCachedEmbedding(text2);
      const endTime3 = performance.now();
      console.log(`   ✓ Generated embedding with ${embedding3.length} dimensions`);
      console.log(`   ✓ Time: ${(endTime3 - startTime3).toFixed(2)}ms`);
      
      // Display cache statistics
      console.log(`   ✓ Cache size: ${embeddingCache.size} entries`);
      console.log(`   ✓ Cache hit speedup: ${((endTime1 - startTime1) / (endTime2 - startTime2)).toFixed(2)}x`);
      
      results.embeddingCache = {
        cacheSize: embeddingCache.size,
        cacheHitTime: endTime2 - startTime2,
        cacheMissTime: endTime1 - startTime1,
        speedup: (endTime1 - startTime1) / (endTime2 - startTime2)
      };
    } catch (error) {
      console.log('   × Error testing embedding cache:', error);
      results.embeddingCache = { error: String(error) };
    }
    
    // -------------------------------------------------------------------------------
    // CONCLUSION
    // -------------------------------------------------------------------------------
    console.log('\n==========================================================');
    console.log('COMPREHENSIVE EXAMPLE COMPLETED SUCCESSFULLY');
    console.log('==========================================================');
    
    console.log('\nSummary of Results:');
    console.log(`✅ Model Used: ${hfTeiUnified.getDefaultModel()}`);
    console.log(`✅ Mode: ${hfTeiUnified.getMode()}`);
    console.log(`✅ Embedding Dimensions: ${results.modelInfo?.dim || 'Unknown'}`);
    
    if (results.basicBenchmark && !results.basicBenchmark.error) {
      console.log(`✅ Performance: ${results.basicBenchmark.sentencesPerSecond.toFixed(2)} sentences/sec`);
    }
    
    console.log('\nDemonstrated Features:');
    console.log('✅ Single and Batch Embedding Generation');
    console.log('✅ Performance Benchmarking');
    console.log('✅ Error Handling and Recovery');
    console.log('✅ Container Configuration and Management');
    console.log('✅ Advanced Integration Patterns');
    console.log('✅ Semantic Similarity and Document Clustering');
    
    console.log('\nFor detailed API documentation, refer to:');
    console.log('- HF_TEI_UNIFIED_USAGE.md in the docs directory');
    
  } catch (error) {
    console.error('\n❌ ERROR IN COMPREHENSIVE EXAMPLE:', error);
    console.error('Please check your API key and network connectivity');
  }
}

// Call the main function
if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error in main:', error);
    process.exit(1);
  });
}

// Export for testing or module usage
export default main;