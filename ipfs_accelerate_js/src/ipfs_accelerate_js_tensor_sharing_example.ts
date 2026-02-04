/**
 * Example of using Cross-Model Tensor Sharing Integration
 * This example demonstrates sharing tensors between different model types
 * with both in-memory sharing and persistent storage
 */

import TensorSharingIntegration from './ipfs_accelerate_js_tensor_sharing_integration';
import { WebNNBackend } from './ipfs_accelerate_js_webnn_backend';

/**
 * Run a demonstration of the tensor sharing integration
 */
async function runTensorSharingIntegrationDemo() {
  console.log("Cross-Model Tensor Sharing Integration Demo");
  console.log("==========================================\n");
  
  // Create a WebNN backend for the demonstration
  const webnnBackend = new WebNNBackend({
    enableLogging: true,
    preferredDeviceType: 'cpu'
  });
  
  // Initialize the tensor sharing integration
  const integration = new TensorSharingIntegration({
    enablePersistence: true,
    enableLogging: true,
    dbName: 'tensor-sharing-demo',
    maxMemoryMb: 1024 // 1GB
  });
  
  console.log("Initializing tensor sharing integration...");
  const initialized = await integration.initialize(webnnBackend);
  
  if (!initialized) {
    console.error("Failed to initialize tensor sharing integration");
    return null;
  }
  
  // Step 1: Create and register some shared tensors
  console.log("\nStep 1: Creating shared tensors");
  
  // Create a simple text embedding tensor (simulating BERT output)
  const textEmbeddingShape = [1, 768]; // [batch_size, embedding_size]
  const textEmbeddingData = new Float32Array(textEmbeddingShape[0] * textEmbeddingShape[1]);
  // Fill with some demo values (normally would be actual embedding values)
  for (let i = 0; i < textEmbeddingData.length; i++) {
    textEmbeddingData[i] = Math.random() * 2 - 1; // Random values between -1 and 1
  }
  const textEmbedding = await integration.registerSharedTensor(
    "text_embedding",
    textEmbeddingShape,
    textEmbeddingData,
    "cpu",
    "bert-base-uncased", // Producer model
    null // No initial consumers
  );
  console.log(`- Created text embedding tensor: ${textEmbeddingShape[0]}x${textEmbeddingShape[1]}`);
  
  // Create a simple vision embedding tensor (simulating ViT output)
  const visionEmbeddingShape = [1, 1024]; // [batch_size, embedding_size]
  const visionEmbeddingData = new Float32Array(visionEmbeddingShape[0] * visionEmbeddingShape[1]);
  // Fill with some demo values
  for (let i = 0; i < visionEmbeddingData.length; i++) {
    visionEmbeddingData[i] = Math.random() * 2 - 1;
  }
  const visionEmbedding = await integration.registerSharedTensor(
    "vision_embedding",
    visionEmbeddingShape,
    visionEmbeddingData,
    "webgpu", // Store on GPU (simulated)
    "vit-base-patch16-224", // Producer model
    null // No initial consumers
  );
  console.log(`- Created vision embedding tensor: ${visionEmbeddingShape[0]}x${visionEmbeddingShape[1]}`);
  
  // Step 2: Share tensors with multiple models
  console.log("\nStep 2: Sharing tensors with multiple models");
  
  // Share text embedding with T5 model
  const t5Result = await integration.shareTensorBetweenModels(
    "text_embedding",
    "bert-base-uncased", // Source model
    ["t5-base"] // Target model
  );
  console.log(`- ${t5Result.message}`);
  
  // Share vision embedding with CLIP model
  const clipResult = await integration.shareTensorBetweenModels(
    "vision_embedding",
    "vit-base-patch16-224", // Source model
    ["clip-vit-base-patch16"] // Target model
  );
  console.log(`- ${clipResult.message}`);
  
  // Create a view for a smaller model (simulate DistilBERT using half of BERT embeddings)
  const embeddingView = await integration.createTensorView(
    "text_embedding", // Parent tensor
    "text_embedding_half", // View name
    [0, 0], // Start offset
    [1, 384], // Half the embedding size
    "distilbert-base-uncased" // Model using this view
  );
  
  if (embeddingView) {
    console.log(`- Created embedding view for DistilBERT: [1, 384] (half of original embedding)`);
  }
  
  // Step 3: Demonstrate retrieval from cache
  console.log("\nStep 3: Retrieving shared tensors for models");
  
  // Get tensor for T5 (should be a cache hit)
  const t5Tensor = await integration.getSharedTensor("text_embedding", "t5-base");
  console.log(`- Retrieved text embedding for T5: ${t5Tensor ? "Success" : "Failed"}`);
  
  // Get tensor for CLIP (should be a cache hit)
  const clipTensor = await integration.getSharedTensor("vision_embedding", "clip-vit-base-patch16");
  console.log(`- Retrieved vision embedding for CLIP: ${clipTensor ? "Success" : "Failed"}`);
  
  // Step 4: Analyze sharing opportunities
  console.log("\nStep 4: Analyzing sharing opportunities");
  
  const opportunities = integration.analyzeSharingOpportunities();
  for (const [tensorType, models] of Object.entries(opportunities)) {
    console.log(`- ${tensorType} can be shared among models: ${models.join(", ")}`);
  }
  
  // Step 5: Check memory usage
  console.log("\nStep 5: Memory usage statistics");
  
  const tensorMemoryUsage = integration.getTensorMemoryUsage();
  for (const [name, usage] of Object.entries(tensorMemoryUsage)) {
    console.log(`- ${name}: ${(usage.memory_mb as number).toFixed(2)} MB`);
  }
  
  // Check model memory usage
  const modelMemoryUsage = integration.getModelMemoryUsage();
  for (const [modelName, usage] of Object.entries(modelMemoryUsage)) {
    console.log(`- ${modelName}: ${(usage.total_memory_mb as number).toFixed(2)} MB`);
  }
  
  // Step 6: Get optimization recommendations
  console.log("\nStep 6: Optimization recommendations");
  
  const recommendations = integration.getOptimizationRecommendations();
  console.log(`- Largest tensors: ${recommendations.largest_tensors.map(t => t.name).join(", ")}`);
  console.log(`- Potential memory savings: ${recommendations.potential_savings_mb.toFixed(2)} MB`);
  
  // Step 7: Create WebNN tensors from shared tensors
  console.log("\nStep 7: Creating WebNN tensors from shared tensors");
  
  const webnnTensors = await integration.createWebNNTensorsFromShared(
    ["text_embedding"], 
    "llama-7b" // New model using the tensor
  );
  
  console.log(`- Created WebNN tensors: ${webnnTensors ? webnnTensors.size : 0} tensors`);
  
  // Step 8: Save and load shared tensors as WebNN model
  console.log("\nStep 8: Saving shared tensors as WebNN model");
  
  const modelId = "shared_tensors_demo";
  const modelName = "Shared Tensors Demo";
  
  const saveResult = await integration.saveAsWebNNModel(
    modelId,
    modelName,
    ["text_embedding", "vision_embedding"]
  );
  
  console.log(`- Saved as WebNN model: ${saveResult ? "Success" : "Failed"}`);
  
  // Load the tensors back as shared tensors with a new producer
  const loadedTensors = await integration.loadFromWebNNModel(
    modelId,
    "multilingual-model" // New producer model
  );
  
  console.log(`- Loaded ${loadedTensors.length} tensors from WebNN model`);
  
  // Step 9: Synchronize with persistent storage
  console.log("\nStep 9: Synchronizing with persistent storage");
  
  const syncResult = await integration.synchronizePersistentStorage();
  console.log(`- ${syncResult.message}`);
  
  // Step 10: Release tensors used by a model
  console.log("\nStep 10: Releasing model tensors");
  
  const releasedCount = await integration.releaseModelTensors("distilbert-base-uncased");
  console.log(`- Released ${releasedCount} tensors used by DistilBERT`);
  
  // Step 11: Optimize memory usage
  console.log("\nStep 11: Optimizing memory usage");
  
  const optimizationResult = await integration.optimizeMemoryUsage();
  console.log(`- Initial memory: ${(optimizationResult.initial_memory_bytes / (1024 * 1024)).toFixed(2)} MB`);
  console.log(`- Current memory: ${(optimizationResult.current_memory_bytes / (1024 * 1024)).toFixed(2)} MB`);
  console.log(`- Memory reduction: ${optimizationResult.memory_reduction_percent.toFixed(2)}%`);
  console.log(`- Freed tensors: ${optimizationResult.freed_tensors_count}`);
  
  // Step 12: Get final statistics
  console.log("\nStep 12: Final statistics");
  
  const stats = integration.getStats();
  console.log(`- Total tensors: ${stats.total_tensors}`);
  console.log(`- Total models: ${stats.total_models}`);
  console.log(`- Memory usage: ${stats.memory_usage_mb.toFixed(2)} MB`);
  console.log(`- Cache hit rate: ${(stats.hit_rate * 100).toFixed(2)}%`);
  console.log(`- Persistent tensor count: ${stats.persistentTensorCount || 0}`);
  
  // Return the integration for further testing/demo
  return integration;
}

// Run the demo if this file is executed directly
if (typeof window !== 'undefined') {
  runTensorSharingIntegrationDemo().catch(error => {
    console.error("Demo error:", error);
  });
} else {
  console.log("This demo is designed to run in a browser environment.");
}

export { runTensorSharingIntegrationDemo };