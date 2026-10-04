/**
 * Example usage of the HuggingFace Text Embedding Inference (HF TEI) API Backend
 * 
 * This example demonstrates how to use the HF TEI backend for various tasks
 * including generating embeddings, batch processing, and similarity calculations.
 */

import { HfTei } from '../src/api_backends/hf_tei/hf_tei';
import { ApiMetadata, ApiResources } from '../src/api_backends/types';

// Environment variables can be used for configuration:
// process.env.HF_TEI_API_KEY = 'your_api_key';
// process.env.HF_API_KEY = 'your_api_key'; // Fallback

async function main() {
  console.log('HuggingFace Text Embedding Inference (HF TEI) API Backend Example');
  console.log('=================================================================\n');

  // 1. Initialize the HF TEI backend
  console.log('1. Initializing HF TEI backend...');
  
  const resources: ApiResources = {};
  const metadata: ApiMetadata = {
    hf_tei_api_key: 'demo_api_key',  // Replace with your actual HF TEI API key
    model_id: 'sentence-transformers/all-MiniLM-L6-v2'  // Default embedding model
  };
  
  const tei = new HfTei(resources, metadata);
  console.log('HF TEI backend initialized with the following settings:');
  console.log(`  - API Key: ${metadata.hf_tei_api_key ? '****' + metadata.hf_tei_api_key.slice(-4) : 'Not provided'}`);
  console.log(`  - Default Model: ${tei.getDefaultModel ? tei.getDefaultModel() : 'sentence-transformers/all-MiniLM-L6-v2'}`);
  
  // 2. Test the endpoint connection
  console.log('\n2. Testing endpoint connection...');
  try {
    const isConnected = await tei.testEndpoint();
    console.log(`  Connection test: ${isConnected ? 'SUCCESS' : 'FAILED'}`);
    
    if (!isConnected) {
      console.log('  Unable to connect to HF TEI API. Please check your API key.');
      return;
    }
  } catch (error) {
    console.error('  Error testing endpoint:', error);
    return;
  }
  
  // 3. Generate a single embedding
  console.log('\n3. Generating a single text embedding...');
  try {
    const text = "This is a sample text for embedding generation.";
    
    const embedding = await tei.generateEmbedding(
      'sentence-transformers/all-MiniLM-L6-v2',
      text
    );
    
    console.log(`  Input text: "${text}"`);
    console.log(`  Embedding dimensions: ${embedding.length}`);
    console.log(`  First 5 dimensions: [${embedding.slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`);
    console.log(`  Embedding is normalized: ${isNormalized(embedding) ? 'Yes' : 'No'}`);
  } catch (error) {
    console.error('  Error generating embedding:', error);
  }
  
  // 4. Generate batch embeddings
  console.log('\n4. Generating batch embeddings for multiple texts...');
  try {
    const texts = [
      "The quick brown fox jumps over the lazy dog.",
      "Machine learning models transform input data into useful representations.",
      "Vector embeddings capture semantic meaning in a numerical format."
    ];
    
    const embeddings = await tei.batchEmbed(
      'sentence-transformers/all-MiniLM-L6-v2',
      texts
    );
    
    console.log(`  Number of texts: ${texts.length}`);
    console.log(`  Number of embeddings: ${embeddings.length}`);
    console.log(`  Embedding dimensions: ${embeddings[0].length}`);
    console.log(`  All embeddings have the same dimension: ${embeddings.every(e => e.length === embeddings[0].length) ? 'Yes' : 'No'}`);
  } catch (error) {
    console.error('  Error generating batch embeddings:', error);
  }
  
  // 5. Calculate similarity between texts
  console.log('\n5. Calculating similarity between texts...');
  try {
    const text1 = "Artificial intelligence is revolutionizing technology.";
    const text2 = "AI is transforming how we use computers and software.";
    const text3 = "The weather today is sunny with clear skies.";
    
    // Generate embeddings for each text
    const embedding1 = await tei.generateEmbedding('sentence-transformers/all-MiniLM-L6-v2', text1);
    const embedding2 = await tei.generateEmbedding('sentence-transformers/all-MiniLM-L6-v2', text2);
    const embedding3 = await tei.generateEmbedding('sentence-transformers/all-MiniLM-L6-v2', text3);
    
    // Calculate similarities
    const similarity1_2 = tei.calculateSimilarity(embedding1, embedding2);
    const similarity1_3 = tei.calculateSimilarity(embedding1, embedding3);
    const similarity2_3 = tei.calculateSimilarity(embedding2, embedding3);
    
    console.log(`  Text 1: "${text1}"`);
    console.log(`  Text 2: "${text2}"`);
    console.log(`  Text 3: "${text3}"`);
    console.log(`  Similarity between text 1 and 2: ${similarity1_2.toFixed(4)} (semantically related)`);
    console.log(`  Similarity between text 1 and 3: ${similarity1_3.toFixed(4)} (unrelated)`);
    console.log(`  Similarity between text 2 and 3: ${similarity2_3.toFixed(4)} (unrelated)`);
  } catch (error) {
    console.error('  Error calculating similarities:', error);
  }
  
  // 6. Creating and using multiple endpoints
  console.log('\n6. Creating and using multiple endpoints...');
  try {
    // Create a default endpoint for general use
    const generalEndpointId = tei.createEndpoint({
      id: 'general-endpoint',
      apiKey: 'general-key',  // Replace with actual key
      model: 'sentence-transformers/all-MiniLM-L6-v2'
    });
    
    // Create a specialized endpoint for another embedding model
    const specializedEndpointId = tei.createEndpoint({
      id: 'specialized-endpoint',
      apiKey: 'specialized-key',  // Replace with actual key
      model: 'BAAI/bge-small-en-v1.5',  // Different model for specific use cases
      maxConcurrentRequests: 2,
      timeout: 10000
    });
    
    console.log(`  Created general endpoint: ${generalEndpointId}`);
    console.log(`  Created specialized endpoint: ${specializedEndpointId}`);
    
    // Get endpoint statistics
    const generalStats = tei.getStats(generalEndpointId);
    console.log(`  General endpoint stats: ${JSON.stringify(generalStats, null, 2)}`);
    
    const specializedStats = tei.getStats(specializedEndpointId);
    console.log(`  Specialized endpoint stats: ${JSON.stringify(specializedStats, null, 2)}`);
    
    // Update an endpoint's configuration
    tei.updateEndpoint(generalEndpointId, {
      maxConcurrentRequests: 5,
      timeout: 15000
    });
    
    console.log(`  Updated general endpoint configuration`);
    
    // Make a request using a specific endpoint
    /*
    // In a real application:
    const response = await tei.makeRequestWithEndpoint(
      generalEndpointId,
      { inputs: "Test text for the general endpoint" }
    );
    console.log(`  Response from general endpoint: ${JSON.stringify(response)}`);
    */
    
    console.log('  Endpoint creation and management demonstrated successfully');
  } catch (error) {
    console.error('  Error with endpoints:', error);
  }
  
  // 7. Normalize embeddings
  console.log('\n7. Normalizing embeddings...');
  try {
    // Create a non-normalized vector
    const vector = [1.0, 2.0, 3.0, 4.0];
    console.log(`  Original vector: [${vector.join(', ')}]`);
    console.log(`  Original vector magnitude: ${calculateMagnitude(vector).toFixed(4)}`);
    
    // Normalize it
    const normalizedVector = tei.normalizeEmbedding(vector);
    console.log(`  Normalized vector: [${normalizedVector.map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`  Normalized vector magnitude: ${calculateMagnitude(normalizedVector).toFixed(4)}`);
  } catch (error) {
    console.error('  Error normalizing vector:', error);
  }
  
  // 8. Using embedText for simpler interface
  console.log('\n8. Using embedText for a simpler interface...');
  try {
    const text = "This is a simple text for the embedText interface.";
    
    const response = await tei.embedText(text, {
      model: 'sentence-transformers/all-MiniLM-L6-v2'
    });
    
    console.log(`  Input text: "${text}"`);
    console.log(`  Model used: ${response.model}`);
    console.log(`  Response has embeddings: ${Array.isArray(response.content) ? 'Yes' : 'No'}`);
    if (Array.isArray(response.content) && response.content.length > 0 && Array.isArray(response.content[0])) {
      console.log(`  Embedding dimensions: ${response.content[0].length}`);
      console.log(`  First 5 dimensions: [${response.content[0].slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`);
    }
  } catch (error) {
    console.error('  Error using embedText:', error);
  }
  
  // 9. Batch embedding with embedText
  console.log('\n9. Batch embedding with embedText...');
  try {
    const texts = [
      "First text for batch embedding.",
      "Second text for batch embedding.",
      "Third text for batch embedding."
    ];
    
    const response = await tei.embedText(texts, {
      model: 'sentence-transformers/all-MiniLM-L6-v2'
    });
    
    console.log(`  Number of input texts: ${texts.length}`);
    console.log(`  Model used: ${response.model}`);
    console.log(`  Response has embeddings: ${Array.isArray(response.content) ? 'Yes' : 'No'}`);
    console.log(`  Number of embeddings: ${Array.isArray(response.content) ? response.content.length : 0}`);
  } catch (error) {
    console.error('  Error with batch embedText:', error);
  }
  
  console.log('\nExample completed successfully!');
}

// Utility function to check if a vector is normalized (magnitude = 1)
function isNormalized(vector: number[]): boolean {
  const magnitude = calculateMagnitude(vector);
  return Math.abs(magnitude - 1.0) < 0.0001; // Allow for floating point precision
}

// Utility function to calculate vector magnitude
function calculateMagnitude(vector: number[]): number {
  return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
}

// Run the example
main().catch(error => {
  console.error('Example failed with error:', error);
});