/**
 * WebNN Graph Builder Example
 * Demonstrates using the WebNN graph builder for neural network operations
 */

import { WebNNGraphBuilder } from './ipfs_accelerate_js_webnn_graph_builder';
import { WebNNBackend } from './ipfs_accelerate_js_webnn_backend';
import { Tensor } from './ipfs_accelerate_js_tensor';

/**
 * Simple Multi-Layer Perceptron (MLP) example using WebNN
 */
export async function mlpExample(): Promise<void> {
  try {
    console.log('Running MLP example with WebNN...');
    
    // Initialize backend
    const backend = new WebNNBackend();
    if (!backend.isAvailable) {
      console.log('WebNN is not available in this environment');
      return;
    }
    
    await backend.initialize();
    
    // Create graph builder
    const graphBuilder = new WebNNGraphBuilder(backend);
    await graphBuilder.initialize();
    
    // Create input tensor with shape [1, 10] (batch_size, input_features)
    const inputData = new Array(10).fill(0).map(() => Math.random());
    const inputTensor = new Tensor([1, 10], inputData, { dataType: 'float32' });
    
    // Define network architecture
    const hiddenSize = 20;
    const outputSize = 5;
    
    // Create input node
    const inputNode = graphBuilder.input('input', inputTensor);
    
    // Create sequential model with two layers
    const model = graphBuilder.sequential(inputNode, [
      {
        // Hidden layer: 10 -> 20 with ReLU activation
        weights: new Tensor(
          [10, hiddenSize],
          new Array(10 * hiddenSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.1),
          { dataType: 'float32' }
        ),
        bias: new Tensor(
          [hiddenSize],
          new Array(hiddenSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.1),
          { dataType: 'float32' }
        ),
        activation: 'relu'
      },
      {
        // Output layer: 20 -> 5 with Softmax (applied later)
        weights: new Tensor(
          [hiddenSize, outputSize],
          new Array(hiddenSize * outputSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.1),
          { dataType: 'float32' }
        ),
        bias: new Tensor(
          [outputSize],
          new Array(outputSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.1),
          { dataType: 'float32' }
        ),
        activation: 'none'
      }
    ]);
    
    // Apply softmax to output
    const outputNode = graphBuilder.softmax(model);
    graphBuilder.output('output', outputNode);
    
    // Execute model
    console.log('Executing MLP model...');
    const results = await graphBuilder.execute({ 'input': inputTensor });
    
    // Display results
    console.log('Input:', inputData);
    console.log('Output (class probabilities):', results.output.data);
    console.log('Predicted class:', results.output.data.indexOf(Math.max(...results.output.data)));
    
    // Clean up
    graphBuilder.dispose();
    backend.dispose();
    
    console.log('MLP example completed successfully');
  } catch (error) {
    console.error('Error in MLP example:', error);
  }
}

/**
 * Convolutional Neural Network (CNN) example using WebNN
 */
export async function cnnExample(): Promise<void> {
  try {
    console.log('Running CNN example with WebNN...');
    
    // Initialize backend
    const backend = new WebNNBackend();
    if (!backend.isAvailable) {
      console.log('WebNN is not available in this environment');
      return;
    }
    
    await backend.initialize();
    
    // Create graph builder
    const graphBuilder = new WebNNGraphBuilder(backend);
    await graphBuilder.initialize();
    
    // Create input tensor with shape [1, 28, 28, 1] (batch, height, width, channels)
    // This would be a grayscale image for MNIST dataset
    const inputSize = 1 * 28 * 28 * 1;
    const inputData = new Array(inputSize).fill(0).map(() => Math.random());
    const inputTensor = new Tensor([1, 28, 28, 1], inputData, { dataType: 'float32' });
    
    // Create input node
    const inputNode = graphBuilder.input('input', inputTensor);
    
    // Define CNN architecture
    
    // First convolution layer: 1 -> 16 channels, 3x3 filter, ReLU activation
    const conv1FilterData = new Array(3 * 3 * 1 * 16).fill(0).map(() => (Math.random() * 2 - 1) * 0.1);
    const conv1Filter = new Tensor([3, 3, 1, 16], conv1FilterData, { dataType: 'float32' });
    const conv1FilterNode = graphBuilder.constant(conv1Filter);
    
    const conv1 = graphBuilder.conv2d(inputNode, conv1FilterNode, {
      strides: [1, 1],
      padding: [1, 1, 1, 1], // Same padding
      activation: 'relu'
    });
    
    // First pooling layer: 2x2 max pooling with stride 2
    const pool1 = graphBuilder.pool(conv1, 'max', {
      windowDimensions: [2, 2],
      strides: [2, 2]
    });
    
    // Second convolution layer: 16 -> 32 channels, 3x3 filter, ReLU activation
    const conv2FilterData = new Array(3 * 3 * 16 * 32).fill(0).map(() => (Math.random() * 2 - 1) * 0.1);
    const conv2Filter = new Tensor([3, 3, 16, 32], conv2FilterData, { dataType: 'float32' });
    const conv2FilterNode = graphBuilder.constant(conv2Filter);
    
    const conv2 = graphBuilder.conv2d(pool1, conv2FilterNode, {
      strides: [1, 1],
      padding: [1, 1, 1, 1], // Same padding
      activation: 'relu'
    });
    
    // Second pooling layer: 2x2 max pooling with stride 2
    const pool2 = graphBuilder.pool(conv2, 'max', {
      windowDimensions: [2, 2],
      strides: [2, 2]
    });
    
    // Flatten: [1, 7, 7, 32] -> [1, 1568]
    // After 2 pooling layers with stride 2, we have 28/2/2 = 7 spatial dimensions
    const flattened = graphBuilder.reshape(pool2, [1, 7 * 7 * 32]);
    
    // Fully connected layer: 1568 -> 10 (output classes for MNIST)
    const fcWeightsData = new Array(7 * 7 * 32 * 10).fill(0).map(() => (Math.random() * 2 - 1) * 0.01);
    const fcWeights = new Tensor([7 * 7 * 32, 10], fcWeightsData, { dataType: 'float32' });
    
    const fcBiasData = new Array(10).fill(0).map(() => (Math.random() * 2 - 1) * 0.01);
    const fcBias = new Tensor([10], fcBiasData, { dataType: 'float32' });
    
    const fc = graphBuilder.layer(flattened, fcWeights, fcBias, 'none');
    
    // Apply softmax to get class probabilities
    const outputNode = graphBuilder.softmax(fc);
    graphBuilder.output('output', outputNode);
    
    // Execute model
    console.log('Executing CNN model...');
    const results = await graphBuilder.execute({ 'input': inputTensor });
    
    // Display results
    console.log('Output (class probabilities):', results.output.data);
    console.log('Predicted class:', results.output.data.indexOf(Math.max(...results.output.data)));
    
    // Clean up
    graphBuilder.dispose();
    backend.dispose();
    
    console.log('CNN example completed successfully');
  } catch (error) {
    console.error('Error in CNN example:', error);
  }
}

/**
 * BERT Embedding Layer example using WebNN
 * Simplified version of BERT token embedding + position embedding + token type embedding
 */
export async function bertEmbeddingExample(): Promise<void> {
  try {
    console.log('Running BERT Embedding example with WebNN...');
    
    // Initialize backend
    const backend = new WebNNBackend();
    if (!backend.isAvailable) {
      console.log('WebNN is not available in this environment');
      return;
    }
    
    await backend.initialize();
    
    // Create graph builder
    const graphBuilder = new WebNNGraphBuilder(backend);
    await graphBuilder.initialize();
    
    // Parameters
    const batchSize = 1;
    const seqLength = 16;
    const hiddenSize = 128;
    const vocabSize = 30522; // Full BERT vocab size
    const maxPositions = 512; // Max positions in BERT
    const typeVocabSize = 2; // Segment types (0 and 1)
    
    // Create input tensors for token IDs, position IDs, and token type IDs
    // For this example, we'll use random IDs in the valid range
    const tokenIds = new Tensor(
      [batchSize, seqLength],
      new Array(batchSize * seqLength).fill(0).map(() => Math.floor(Math.random() * 1000)), // Random token IDs
      { dataType: 'int32' }
    );
    
    const positionIds = new Tensor(
      [batchSize, seqLength],
      Array.from({ length: batchSize * seqLength }, (_, i) => i % seqLength), // Sequential positions
      { dataType: 'int32' }
    );
    
    const tokenTypeIds = new Tensor(
      [batchSize, seqLength],
      new Array(batchSize * seqLength).fill(0).map(() => Math.floor(Math.random() * 2)), // 0 or 1
      { dataType: 'int32' }
    );
    
    // Create input nodes
    const tokenIdNode = graphBuilder.input('token_ids', tokenIds);
    const positionIdNode = graphBuilder.input('position_ids', positionIds);
    const tokenTypeIdNode = graphBuilder.input('token_type_ids', tokenTypeIds);
    
    // Create embedding tables with random weights
    // In a real model, these would be loaded from a pre-trained model
    const wordEmbeddingTable = new Tensor(
      [vocabSize, hiddenSize],
      new Array(vocabSize * hiddenSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.02),
      { dataType: 'float32' }
    );
    
    const positionEmbeddingTable = new Tensor(
      [maxPositions, hiddenSize],
      new Array(maxPositions * hiddenSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.02),
      { dataType: 'float32' }
    );
    
    const tokenTypeEmbeddingTable = new Tensor(
      [typeVocabSize, hiddenSize],
      new Array(typeVocabSize * hiddenSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.02),
      { dataType: 'float32' }
    );
    
    // In a real implementation, we would use gather operations on the embedding tables
    // For this simplified example, we'll just create pre-gathered embeddings
    // This is a simplification - in practice, we'd implement gather operations
    
    // Create pre-gathered embeddings for this example
    const wordEmbeddings = new Tensor(
      [batchSize, seqLength, hiddenSize],
      new Array(batchSize * seqLength * hiddenSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.1),
      { dataType: 'float32' }
    );
    
    const positionEmbeddings = new Tensor(
      [batchSize, seqLength, hiddenSize],
      new Array(batchSize * seqLength * hiddenSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.1),
      { dataType: 'float32' }
    );
    
    const tokenTypeEmbeddings = new Tensor(
      [batchSize, seqLength, hiddenSize],
      new Array(batchSize * seqLength * hiddenSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.1),
      { dataType: 'float32' }
    );
    
    // Create constant nodes for the embeddings
    const wordEmbNode = graphBuilder.constant(wordEmbeddings);
    const posEmbNode = graphBuilder.constant(positionEmbeddings);
    const typeEmbNode = graphBuilder.constant(tokenTypeEmbeddings);
    
    // Add the three embeddings together
    const embSum1 = graphBuilder.add(wordEmbNode, posEmbNode);
    const embSum2 = graphBuilder.add(embSum1, typeEmbNode);
    
    // Layer normalization parameters
    const gamma = new Tensor(
      [hiddenSize],
      new Array(hiddenSize).fill(1.0), // Initialize to ones
      { dataType: 'float32' }
    );
    
    const beta = new Tensor(
      [hiddenSize],
      new Array(hiddenSize).fill(0.0), // Initialize to zeros
      { dataType: 'float32' }
    );
    
    // For layer normalization, we'd need mean and variance
    // In this simplified example, we'll just create dummy tensors
    // In practice, these would be computed from the input
    const mean = new Tensor(
      [batchSize, seqLength, 1],
      new Array(batchSize * seqLength).fill(0),
      { dataType: 'float32' }
    );
    
    const variance = new Tensor(
      [batchSize, seqLength, 1],
      new Array(batchSize * seqLength).fill(1),
      { dataType: 'float32' }
    );
    
    const meanNode = graphBuilder.constant(mean);
    const varianceNode = graphBuilder.constant(variance);
    const gammaNode = graphBuilder.constant(gamma);
    const betaNode = graphBuilder.constant(beta);
    
    // Apply layer normalization
    // This is a simplified approach - in practice we would compute mean and variance
    const embLayerNorm = graphBuilder.batchNormalization(
      embSum2,
      meanNode,
      varianceNode,
      gammaNode,
      betaNode,
      1e-12 // Epsilon (BERT uses 1e-12)
    );
    
    graphBuilder.output('embeddings', embLayerNorm);
    
    // Execute model
    console.log('Executing BERT Embedding model...');
    const results = await graphBuilder.execute({
      'token_ids': tokenIds,
      'position_ids': positionIds,
      'token_type_ids': tokenTypeIds
    });
    
    // Display results
    console.log('Output embedding shape:', results.embeddings.shape);
    console.log('First few values of the embedding:');
    console.log(results.embeddings.data.slice(0, 10));
    
    // Clean up
    graphBuilder.dispose();
    backend.dispose();
    
    console.log('BERT Embedding example completed successfully');
  } catch (error) {
    console.error('Error in BERT Embedding example:', error);
  }
}

/**
 * Run the examples
 */
export async function runWebNNGraphExamples(): Promise<void> {
  // Check if WebNN is available
  const isWebNNAvailable = typeof navigator !== 'undefined' && 'ml' in navigator;
  
  if (!isWebNNAvailable) {
    console.log('WebNN is not available in this environment. Examples will not run correctly.');
    return;
  }
  
  console.log('Starting WebNN Graph examples...');
  
  // Run examples
  await mlpExample();
  console.log('\n');
  
  await cnnExample();
  console.log('\n');
  
  await bertEmbeddingExample();
  
  console.log('\nAll WebNN Graph examples completed.');
}

// Uncomment to run examples directly
// runWebNNGraphExamples().catch(console.error);