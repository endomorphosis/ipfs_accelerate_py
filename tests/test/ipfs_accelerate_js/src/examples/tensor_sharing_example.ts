/**
 * Example of how to use the SharedTensor and TensorSharingManager for 
 * cross-model tensor sharing
 */

import { TensorSharingManager, SharedTensor } from '../tensor/shared_tensor';

/**
 * Example of using the tensor sharing manager to optimize memory usage
 * when running multiple models that could share tensors.
 */
export function runTensorSharingExample() {
  console.log("Cross-Model Tensor Sharing Example");
  console.log("==================================");
  
  // Create the tensor sharing manager
  const manager = new TensorSharingManager(1024); // 1GB max memory
  console.log("Created tensor sharing manager with 1GB memory limit");
  
  // Create a BERT embedding tensor
  const bertEmbedding = manager.registerSharedTensor(
    "bert_embedding",
    [1, 768],  // [batch_size, embedding_size]
    "cpu",
    "bert-base-uncased",
    null,      // No initial consumers
    "float32"
  );
  console.log("Created BERT embedding tensor:", bertEmbedding.toString());
  
  // Create a ViT (Vision Transformer) embedding tensor
  const vitEmbedding = manager.registerSharedTensor(
    "vit_embedding",
    [1, 1024],  // [batch_size, embedding_size]
    "webgpu",   // Store on GPU for efficiency
    "vit-base-patch16",
    null,
    "float32"
  );
  console.log("Created ViT embedding tensor:", vitEmbedding.toString());
  
  // Simulate loading models that can share embeddings
  console.log("\nSimulating multiple models sharing tensors:");
  
  // T5 model can use BERT embeddings
  const t5Model = "t5-base";
  const t5Embedding = manager.getSharedTensor("bert_embedding", t5Model);
  console.log(`- Model ${t5Model} is using the BERT embedding`);
  
  // CLIP model can use ViT embeddings
  const clipModel = "clip-vit-base";
  const clipEmbedding = manager.getSharedTensor("vit_embedding", clipModel);
  console.log(`- Model ${clipModel} is using the ViT embedding`);
  
  // Create a view of the BERT embedding for a smaller model
  const embeddingView = manager.createTensorView(
    "bert_embedding",
    "bert_embedding_half",
    [0, 0],        // Start offset
    [1, 384],      // Half the embedding size
    "distilbert"   // Model using the view
  );
  console.log(`- Created a view for DistilBERT: ${embeddingView?.toString()}`);
  
  // Share tensors with other models
  manager.shareTensorBetweenModels(
    "bert_embedding",
    "bert-base-uncased",
    ["bart", "roberta"]
  );
  console.log("- Shared BERT embedding with BART and RoBERTa models");
  
  // Analyze opportunities for tensor sharing
  console.log("\nAnalyzing tensor sharing opportunities:");
  const opportunities = manager.analyzeSharingOpportunities();
  for (const [tensorType, models] of Object.entries(opportunities)) {
    console.log(`- ${tensorType} can be shared among: ${models.join(", ")}`);
  }
  
  // Print memory usage by model
  console.log("\nMemory usage by model:");
  const modelMemory = manager.getModelMemoryUsage();
  for (const [model, memInfo] of Object.entries(modelMemory)) {
    console.log(`- ${model}: ${memInfo.total_memory_mb.toFixed(2)} MB (${memInfo.tensor_count} tensors)`);
  }
  
  // Get optimization recommendations
  console.log("\nOptimization recommendations:");
  const recommendations = manager.getOptimizationRecommendations();
  
  // Show largest tensors
  console.log("- Largest tensors:");
  for (const tensor of recommendations.largest_tensors) {
    console.log(`  * ${tensor.name}: ${tensor.memory_mb.toFixed(2)} MB`);
  }
  
  // Show sharing opportunities
  console.log("- Potential memory savings: " + 
    recommendations.potential_savings_mb.toFixed(2) + " MB");
  
  // Simulate releasing models to free memory
  console.log("\nReleasing models:");
  const releasedCount = manager.releaseModelTensors("distilbert");
  console.log(`- Released ${releasedCount} tensors for DistilBERT`);
  
  // Run memory optimization
  console.log("\nOptimizing memory usage:");
  const result = manager.optimizeMemoryUsage();
  console.log(`- Initial memory: ${(result.initial_memory_bytes / (1024 * 1024)).toFixed(2)} MB`);
  console.log(`- Current memory: ${(result.current_memory_bytes / (1024 * 1024)).toFixed(2)} MB`);
  console.log(`- Memory reduction: ${result.memory_reduction_percent.toFixed(2)}%`);
  console.log(`- Freed tensors: ${result.freed_tensors_count}`);
  
  // Get final statistics
  console.log("\nFinal statistics:");
  const stats = manager.getStats();
  console.log(`- Total tensors: ${stats.total_tensors}`);
  console.log(`- Total models: ${stats.total_models}`);
  console.log(`- Memory usage: ${stats.memory_usage_mb.toFixed(2)} MB`);
  console.log(`- Cache hit rate: ${(stats.hit_rate * 100).toFixed(2)}%`);
  
  return {
    manager,
    result,
    stats
  };
}

// Run the example if this file is executed directly
if (typeof window !== 'undefined') {
  runTensorSharingExample();
}