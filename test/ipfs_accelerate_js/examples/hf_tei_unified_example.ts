/**
 * HuggingFace Text Embedding Inference Unified API Backend Example
 * 
 * This example demonstrates how to use the HF TEI Unified backend for:
 * 1. Basic embedding generation
 * 2. Batch embedding processing
 * 3. Benchmarking embedding models
 * 4. Container-based deployment and management
 * 5. Model information retrieval
 * 6. Switching between API and container modes
 */

// Import the HfTeiUnified class and types
import { HfTeiUnified } from '../src/api_backends/hf_tei_unified/hf_tei_unified';
import {
  HfTeiUnifiedOptions,
  HfTeiUnifiedApiMetadata,
  EmbeddingOptions,
  DeploymentConfig,
  PerformanceMetrics
} from '../src/api_backends/hf_tei_unified/types';

async function main() {
  console.log('HuggingFace Text Embedding Inference Unified API Backend Example');
  
  try {
    // -------------------------------------------------------------------------------
    // Example 1: Initialize the HF TEI Unified backend with API key
    // -------------------------------------------------------------------------------
    const apiKey = process.env.HF_API_KEY || 'your_api_key'; // Replace with your API key or set env var
    
    // Set up configuration options
    const options: HfTeiUnifiedOptions = {
      apiUrl: 'https://api-inference.huggingface.co/models',
      maxRetries: 3,
      requestTimeout: 30000,
      useRequestQueue: true,
      debug: true
    };
    
    // Set up metadata
    const metadata: HfTeiUnifiedApiMetadata = {
      hf_api_key: apiKey,
      model_id: 'BAAI/bge-small-en-v1.5' // Default embedding model
    };
    
    // Create the backend instance
    const hfTeiUnified = new HfTeiUnified(options, metadata);
    
    console.log('\n1. HF TEI Unified backend initialized successfully in API mode');
    console.log('   Current mode:', hfTeiUnified.getMode());
    console.log('   Default model:', hfTeiUnified.getDefaultModel());
    
    // -------------------------------------------------------------------------------
    // Example 2: Check model compatibility
    // -------------------------------------------------------------------------------
    console.log('\n2. Model compatibility check:');
    
    const compatibleModels = [
      'BAAI/bge-small-en-v1.5',
      'sentence-transformers/all-MiniLM-L6-v2',
      'intfloat/e5-base-v2',
      'thenlper/gte-base',
      'random-model-name'
    ];
    
    for (const model of compatibleModels) {
      const isCompatible = hfTeiUnified.isCompatibleModel(model);
      console.log(`   ${model}: ${isCompatible ? 'Compatible' : 'Not compatible'}`);
    }

    // -------------------------------------------------------------------------------
    // Example 3: Generate embeddings for a single text
    // -------------------------------------------------------------------------------
    console.log('\n3. Generate embeddings for a single text:');
    
    try {
      const text = "This is a sample text for embedding generation.";
      const embeddings = await hfTeiUnified.generateEmbeddings(text);
      
      console.log(`   Generated embedding with ${embeddings[0].length} dimensions`);
      console.log('   First 5 values:', embeddings[0].slice(0, 5));
    } catch (error) {
      console.log('   Error generating embeddings:', error);
      console.log('   Continuing with mock embeddings for demonstration...');
    }
    
    // -------------------------------------------------------------------------------
    // Example 4: Generate embeddings for a batch of texts
    // -------------------------------------------------------------------------------
    console.log('\n4. Generate embeddings for a batch of texts:');
    
    try {
      const texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can process natural language.",
        "Embeddings represent text as dense numerical vectors."
      ];
      
      const options: EmbeddingOptions = {
        normalize: true,
        maxTokens: 512,
        priority: 'HIGH'
      };
      
      const batchEmbeddings = await hfTeiUnified.batchEmbeddings(texts, options);
      
      console.log(`   Generated ${batchEmbeddings.length} embeddings`);
      console.log(`   Each embedding has ${batchEmbeddings[0].length} dimensions`);
    } catch (error) {
      console.log('   Error generating batch embeddings:', error);
      console.log('   Continuing with mock data for demonstration...');
    }
    
    // -------------------------------------------------------------------------------
    // Example 5: Get model information
    // -------------------------------------------------------------------------------
    console.log('\n5. Get model information:');
    
    try {
      const modelInfo = await hfTeiUnified.getModelInfo();
      
      console.log('   Model information:');
      console.log(`   - Model ID: ${modelInfo.model_id}`);
      console.log(`   - Embedding dimensions: ${modelInfo.dim}`);
      console.log(`   - Status: ${modelInfo.status}`);
      if (modelInfo.revision) console.log(`   - Revision: ${modelInfo.revision}`);
      if (modelInfo.framework) console.log(`   - Framework: ${modelInfo.framework}`);
      if (modelInfo.quantized !== undefined) console.log(`   - Quantized: ${modelInfo.quantized}`);
    } catch (error) {
      console.log('   Error getting model information:', error);
      console.log('   Continuing with demonstration...');
    }
    
    // -------------------------------------------------------------------------------
    // Example 6: Run a benchmark
    // -------------------------------------------------------------------------------
    console.log('\n6. Run a benchmark:');
    
    try {
      const benchmarkOptions = {
        iterations: 3,
        batchSize: 5,
        model: 'BAAI/bge-small-en-v1.5'
      };
      
      const benchmarkResults = await hfTeiUnified.runBenchmark(benchmarkOptions);
      
      console.log('   Benchmark results:');
      console.log(`   - Single embedding time: ${benchmarkResults.singleEmbeddingTime.toFixed(2)} ms`);
      console.log(`   - Batch embedding time: ${benchmarkResults.batchEmbeddingTime.toFixed(2)} ms`);
      console.log(`   - Sentences per second: ${benchmarkResults.sentencesPerSecond.toFixed(2)}`);
      console.log(`   - Batch speedup factor: ${benchmarkResults.batchSpeedupFactor.toFixed(2)}x`);
    } catch (error) {
      console.log('   Error running benchmark:', error);
      console.log('   Continuing with demonstration...');
    }
    
    // -------------------------------------------------------------------------------
    // Example 7: Switch to container mode
    // -------------------------------------------------------------------------------
    console.log('\n7. Switch to container mode:');
    
    // Switch to container mode
    hfTeiUnified.setMode(true);
    console.log(`   Current mode: ${hfTeiUnified.getMode()}`);
    
    // -------------------------------------------------------------------------------
    // Example 8: Container management (with warning)
    // -------------------------------------------------------------------------------
    console.log('\n8. Container management (demonstration only):');
    
    console.log('   NOTE: This would start a real Docker container if run with proper configuration');
    console.log('   For this example, we\'ll just demonstrate the API without actually starting a container');
    
    /*
    // Uncomment to actually start a container
    try {
      const deployConfig: DeploymentConfig = {
        dockerRegistry: 'ghcr.io/huggingface/text-embeddings-inference',
        containerTag: 'latest',
        gpuDevice: '0',
        modelId: 'BAAI/bge-small-en-v1.5',
        port: 8080,
        env: {
          'HF_API_TOKEN': apiKey
        },
        volumes: ['./cache:/cache'],
        network: 'bridge'
      };
      
      const containerInfo = await hfTeiUnified.startContainer(deployConfig);
      
      console.log('   Container started:', containerInfo);
      
      // Generate embeddings using the container
      const text = "Testing the container-based embedding service.";
      const embeddings = await hfTeiUnified.generateEmbeddings(text);
      
      console.log(`   Generated embedding with ${embeddings[0].length} dimensions`);
      
      // Stop the container
      const stopped = await hfTeiUnified.stopContainer();
      console.log(`   Container stopped: ${stopped}`);
    } catch (error) {
      console.log('   Error with container operations:', error);
    }
    */
    
    // -------------------------------------------------------------------------------
    // Example 9: Switch back to API mode
    // -------------------------------------------------------------------------------
    console.log('\n9. Switch back to API mode:');
    
    // Switch back to API mode for the rest of the examples
    hfTeiUnified.setMode(false);
    console.log(`   Current mode: ${hfTeiUnified.getMode()}`);
    
    // -------------------------------------------------------------------------------
    // Example 10: Test endpoint availability
    // -------------------------------------------------------------------------------
    console.log('\n10. Test endpoint availability:');
    
    try {
      const isAvailable = await hfTeiUnified.testEndpoint();
      console.log(`   Endpoint available: ${isAvailable}`);
    } catch (error) {
      console.log('   Error testing endpoint:', error);
    }
    
    console.log('\nExample completed successfully.');
  } catch (error) {
    console.error('Example failed with error:', error);
  }
}

// Call the main function
if (require.main === module) {
  main().catch(error => {
    console.error('Error in main:', error);
    process.exit(1);
  });
}