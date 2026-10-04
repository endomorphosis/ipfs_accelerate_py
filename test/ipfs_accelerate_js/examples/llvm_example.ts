/**
 * Example usage of the LLVM backend
 * 
 * This example demonstrates how to use the LLVM API backend
 * for various operations including listing models, getting model info,
 * running inference, and using the chat interface.
 */

import { LLVM } from '../src/api_backends/llvm/llvm';
import { LlvmOptions } from '../src/api_backends/llvm/types';
import { ApiMetadata, ChatMessage } from '../src/api_backends/types';

// Environment variables can be used to configure the backend:
// process.env.LLVM_API_KEY = 'your_api_key';
// process.env.LLVM_BASE_URL = 'http://your-llvm-server:8000';
// process.env.LLVM_DEFAULT_MODEL = 'llvm-default-model';

async function main() {
  console.log('LLVM Backend Example');
  console.log('====================\n');

  // 1. Initialize the LLVM backend
  console.log('1. Initializing LLVM backend...');
  
  const options: LlvmOptions = {
    base_url: 'http://localhost:8000',  // Replace with your LLVM server URL
    max_concurrent_requests: 5,
    max_retries: 3,
    retry_delay: 1000,
    queue_size: 50
  };
  
  const metadata: ApiMetadata = {
    llvm_api_key: 'demo_api_key',  // Replace with your API key
    llvm_default_model: 'llvm-model-1'  // Replace with your default model
  };
  
  const llvmBackend = new LLVM(options, metadata);
  console.log('Backend initialized with options:', options);
  console.log('Using API key:', metadata.llvm_api_key ? '****' + metadata.llvm_api_key.slice(-4) : 'None');
  console.log('Default model:', metadata.llvm_default_model || 'None');
  
  // 2. Test the endpoint connection
  console.log('\n2. Testing endpoint connection...');
  try {
    const isConnected = await llvmBackend.testEndpoint();
    console.log('Endpoint connection test:', isConnected ? 'SUCCESS' : 'FAILED');
  } catch (error) {
    console.error('Endpoint test error:', error);
  }
  
  // 3. List available models
  console.log('\n3. Listing available models...');
  try {
    const models = await llvmBackend.listModels();
    console.log(`Found ${models.models.length} models:`);
    models.models.forEach(model => console.log(`  - ${model}`));
  } catch (error) {
    console.error('Error listing models:', error);
  }
  
  // 4. Get model information
  console.log('\n4. Getting model information...');
  const modelId = metadata.llvm_default_model || 'llvm-model-1';
  try {
    const modelInfo = await llvmBackend.getModelInfo(modelId);
    console.log(`Model info for ${modelId}:`);
    console.log(`  Status: ${modelInfo.status}`);
    if (modelInfo.details) {
      console.log('  Details:');
      Object.entries(modelInfo.details).forEach(([key, value]) => {
        console.log(`    ${key}: ${JSON.stringify(value)}`);
      });
    }
  } catch (error) {
    console.error(`Error getting model info for ${modelId}:`, error);
  }
  
  // 5. Run inference with text input
  console.log('\n5. Running inference with text input...');
  try {
    const inferenceOptions = {
      max_tokens: 100,
      temperature: 0.7,
      top_p: 0.95
    };
    
    const textInput = "Translate the following English text to French: 'Hello, how are you today?'";
    const inferenceResult = await llvmBackend.runInference(modelId, textInput, inferenceOptions);
    
    console.log('Inference result:');
    console.log(`  Model: ${inferenceResult.model_id}`);
    console.log(`  Output: ${inferenceResult.outputs}`);
    if (inferenceResult.metadata) {
      console.log('  Metadata:');
      Object.entries(inferenceResult.metadata).forEach(([key, value]) => {
        console.log(`    ${key}: ${JSON.stringify(value)}`);
      });
    }
  } catch (error) {
    console.error('Error running inference:', error);
  }
  
  // 6. Run inference with structured input
  console.log('\n6. Running inference with structured input...');
  try {
    const structuredInput = {
      text: "What is the capital of France?",
      options: {
        format: "json",
        include_references: true
      }
    };
    
    const inferenceResult = await llvmBackend.runInference(
      modelId, 
      structuredInput, 
      { 
        max_tokens: 150 
      }
    );
    
    console.log('Structured inference result:');
    console.log(`  Model: ${inferenceResult.model_id}`);
    console.log(`  Output: ${JSON.stringify(inferenceResult.outputs, null, 2)}`);
  } catch (error) {
    console.error('Error running structured inference:', error);
  }
  
  // 7. Using the chat interface
  console.log('\n7. Using the chat interface...');
  try {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'What is machine learning?' }
    ];
    
    const chatResponse = await llvmBackend.chat(modelId, messages, {
      max_tokens: 200,
      temperature: 0.8
    });
    
    console.log('Chat response:');
    console.log(`  ID: ${chatResponse.id}`);
    console.log(`  Model: ${chatResponse.model}`);
    console.log(`  Created: ${new Date(chatResponse.created).toISOString()}`);
    console.log(`  Content: ${chatResponse.content}`);
  } catch (error) {
    console.error('Error using chat interface:', error);
  }
  
  // 8. Error handling demonstration
  console.log('\n8. Error handling demonstration...');
  try {
    // Attempt to use a non-existent model
    await llvmBackend.getModelInfo('non-existent-model');
  } catch (error) {
    console.log('Expected error caught successfully:');
    console.log(`  ${error}`);
  }
  
  // 9. API key management
  console.log('\n9. API key management...');
  console.log('Current API key:', llvmBackend.getApiKey() ? '****' + llvmBackend.getApiKey()?.slice(-4) : 'None');
  llvmBackend.setApiKey('new_api_key_example');
  console.log('Updated API key:', '****' + llvmBackend.getApiKey()?.slice(-4));
  
  // 10. Model compatibility check
  console.log('\n10. Model compatibility check...');
  const modelNames = [
    'llvm-text-model',
    'text-model',
    'text-model-llvm',
    'other-model'
  ];
  
  modelNames.forEach(model => {
    const isCompatible = llvmBackend.isCompatibleModel(model);
    console.log(`Model "${model}" is ${isCompatible ? 'compatible' : 'not compatible'} with LLVM backend`);
  });
}

// Run the example
main().catch(error => {
  console.error('Example failed with error:', error);
});