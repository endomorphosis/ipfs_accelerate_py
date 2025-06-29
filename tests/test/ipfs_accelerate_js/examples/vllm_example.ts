/**
 * Example usage of the VLLM API Backend
 * 
 * This example demonstrates how to use the VLLM API backend for various tasks
 * including running inference, chat completions, batch processing, and
 * working with LoRA adapters and model quantization.
 */

import { VLLM } from '../src/api_backends/vllm/vllm';
import { ApiMetadata, ApiResources, Message } from '../src/api_backends/types';
import { VLLMQuantizationConfig } from '../src/api_backends/vllm/types';

// Environment variables can be used for configuration:
// process.env.VLLM_API_URL = 'http://your-vllm-server:8000';
// process.env.VLLM_MODEL = 'your-model-name';
// process.env.VLLM_API_KEY = 'your-api-key';

async function main() {
  console.log('VLLM API Backend Example');
  console.log('=========================\n');

  // 1. Initialize the VLLM backend
  console.log('1. Initializing VLLM backend...');
  
  const resources: ApiResources = {};
  const metadata: ApiMetadata = {
    vllm_api_url: 'http://localhost:8000',  // Replace with your VLLM server URL
    vllm_model: 'meta-llama/Llama-2-7b-chat-hf',  // Replace with your model name
    vllm_api_key: 'demo_api_key'  // Replace with your API key if needed
  };
  
  const vllm = new VLLM(resources, metadata);
  console.log('VLLM backend initialized with the following settings:');
  console.log(`  - API URL: ${metadata.vllm_api_url}`);
  console.log(`  - Model: ${metadata.vllm_model}`);
  console.log(`  - API Key: ${metadata.vllm_api_key ? '****' + metadata.vllm_api_key.slice(-4) : 'Not provided'}`);
  
  // 2. Test the endpoint connection
  console.log('\n2. Testing endpoint connection...');
  try {
    const isConnected = await vllm.testEndpoint();
    console.log(`  Connection test: ${isConnected ? 'SUCCESS' : 'FAILED'}`);
    
    if (!isConnected) {
      console.log('  Unable to connect to VLLM server. Please check your server URL and make sure the server is running.');
      return;
    }
  } catch (error) {
    console.error('  Error testing endpoint:', error);
    return;
  }
  
  // 3. Get model information
  console.log('\n3. Getting model information...');
  try {
    const modelInfo = await vllm.getModelInfo();
    console.log('  Model information:');
    console.log(`  - Model: ${modelInfo.model}`);
    console.log(`  - Maximum model length: ${modelInfo.max_model_len} tokens`);
    console.log(`  - Number of GPUs: ${modelInfo.num_gpu}`);
    console.log(`  - Data type: ${modelInfo.dtype}`);
    console.log(`  - GPU memory utilization: ${modelInfo.gpu_memory_utilization * 100}%`);
    
    if (modelInfo.quantization) {
      console.log('  - Quantization:');
      console.log(`    - Enabled: ${modelInfo.quantization.enabled}`);
      if (modelInfo.quantization.method) {
        console.log(`    - Method: ${modelInfo.quantization.method}`);
      }
    }
    
    if (modelInfo.lora_adapters && modelInfo.lora_adapters.length > 0) {
      console.log('  - Active LoRA adapters:');
      modelInfo.lora_adapters.forEach(adapter => {
        console.log(`    - ${adapter}`);
      });
    }
  } catch (error) {
    console.error('  Error getting model information:', error);
  }
  
  // 4. Get model statistics
  console.log('\n4. Getting model statistics...');
  try {
    const stats = await vllm.getModelStatistics();
    console.log('  Model statistics:');
    console.log(`  - Requests processed: ${stats.statistics.requests_processed || 'N/A'}`);
    console.log(`  - Tokens generated: ${stats.statistics.tokens_generated || 'N/A'}`);
    console.log(`  - Average tokens per request: ${stats.statistics.avg_tokens_per_request || 'N/A'}`);
    console.log(`  - Maximum tokens per request: ${stats.statistics.max_tokens_per_request || 'N/A'}`);
    console.log(`  - Average generation time: ${stats.statistics.avg_generation_time || 'N/A'} seconds`);
    console.log(`  - Throughput: ${stats.statistics.throughput || 'N/A'} tokens/second`);
    console.log(`  - Errors: ${stats.statistics.errors || 'N/A'}`);
    console.log(`  - Uptime: ${stats.statistics.uptime || 'N/A'} seconds`);
  } catch (error) {
    console.error('  Error getting model statistics:', error);
  }
  
  // 5. List LoRA adapters
  console.log('\n5. Listing LoRA adapters...');
  try {
    const adapters = await vllm.listLoraAdapters();
    if (adapters.length === 0) {
      console.log('  No LoRA adapters found.');
    } else {
      console.log(`  Found ${adapters.length} LoRA adapters:`);
      adapters.forEach(adapter => {
        console.log(`  - ${adapter.name} (${adapter.id})`);
        console.log(`    Base model: ${adapter.base_model}`);
        console.log(`    Size: ${adapter.size_mb} MB`);
        console.log(`    Active: ${adapter.active}`);
      });
    }
  } catch (error) {
    console.error('  Error listing LoRA adapters:', error);
  }
  
  // 6. Load a LoRA adapter (example - actual adapter paths would depend on your setup)
  console.log('\n6. Loading a LoRA adapter (example)...');
  try {
    const adapterData = {
      adapter_name: 'example-adapter',
      adapter_path: '/path/to/lora_adapter',
      base_model: 'meta-llama/Llama-2-7b-chat-hf'
    };
    
    // Note: This is just an example. You would need an actual adapter path on your server.
    // Uncomment to test with a real adapter:
    // const result = await vllm.loadLoraAdapter(adapterData);
    // console.log('  LoRA adapter loaded successfully:');
    // console.log(JSON.stringify(result, null, 2).replace(/^/gm, '  '));
    
    console.log('  (Skipping actual adapter loading in this example)');
  } catch (error) {
    console.error('  Error loading LoRA adapter:', error);
  }
  
  // 7. Set quantization configuration
  console.log('\n7. Setting quantization configuration (example)...');
  try {
    const quantConfig: VLLMQuantizationConfig = {
      enabled: true,
      method: 'awq',
      bits: 4
    };
    
    // Note: This would actually change your model's quantization settings.
    // Uncomment to test with a real model:
    // const result = await vllm.setQuantization(undefined, quantConfig);
    // console.log('  Quantization configuration set successfully:');
    // console.log(JSON.stringify(result, null, 2).replace(/^/gm, '  '));
    
    console.log('  (Skipping actual quantization in this example)');
  } catch (error) {
    console.error('  Error setting quantization configuration:', error);
  }
  
  // 8. Run inference with a simple prompt
  console.log('\n8. Running inference with a simple prompt...');
  try {
    const prompt = "Explain the concept of quantum computing in simple terms.";
    const data = {
      prompt,
      model: metadata.vllm_model,
      max_tokens: 100,
      temperature: 0.7,
      top_p: 0.95
    };
    
    const result = await vllm.makePostRequest(data);
    
    console.log('  Prompt:', prompt);
    console.log('  Response:');
    
    if (result.choices && result.choices.length > 0) {
      console.log(result.choices[0].text || 
                 (result.choices[0].message && result.choices[0].message.content) || 
                 'No response text');
    } else if (result.text) {
      console.log(result.text);
    } else {
      console.log('  No response text found in the result');
    }
    
    if (result.usage) {
      console.log('  Usage statistics:');
      console.log(`  - Prompt tokens: ${result.usage.prompt_tokens}`);
      console.log(`  - Completion tokens: ${result.usage.completion_tokens}`);
      console.log(`  - Total tokens: ${result.usage.total_tokens}`);
    }
  } catch (error) {
    console.error('  Error running inference:', error);
  }
  
  // 9. Using the chat interface
  console.log('\n9. Using the chat interface...');
  try {
    const messages: Message[] = [
      { role: 'system', content: 'You are a helpful AI assistant specializing in quantum physics.' },
      { role: 'user', content: 'What is quantum entanglement?' }
    ];
    
    const chatResult = await vllm.chat(messages, {
      model: metadata.vllm_model,
      max_tokens: 150,
      temperature: 0.7
    });
    
    console.log('  User message: What is quantum entanglement?');
    console.log('  Assistant response:');
    console.log(chatResult.content);
  } catch (error) {
    console.error('  Error using chat interface:', error);
  }
  
  // 10. Streaming chat completion (this requires an actual VLLM server)
  console.log('\n10. Streaming chat completion example (simulated)...');
  try {
    /*
    // In a real application with a real VLLM server:
    const messages: Message[] = [
      { role: 'system', content: 'You are a helpful AI assistant.' },
      { role: 'user', content: 'Write a short poem about artificial intelligence.' }
    ];
    
    console.log('  User message: Write a short poem about artificial intelligence.');
    console.log('  Assistant response (streaming):');
    
    let fullResponse = '';
    
    for await (const chunk of await vllm.streamChat(messages)) {
      process.stdout.write(chunk.content);
      fullResponse += chunk.content;
      
      if (chunk.done) {
        console.log('\n  (Stream completed)');
        break;
      }
    }
    */
    
    // Since we're using a simulated environment, we'll just show the concept
    console.log('  In a real application, you would see the response appear token by token.');
    console.log('  Simulated response:');
    console.log('  Silicon dreams in neural nets,');
    console.log('  Learning patterns, no regrets.');
    console.log('  Algorithms grow and thrive,');
    console.log('  In data\'s ocean, they\'re alive.');
  } catch (error) {
    console.error('  Error in streaming chat completion:', error);
  }
  
  // 11. Process a batch of prompts
  console.log('\n11. Processing a batch of prompts...');
  try {
    const batchPrompts = [
      "What is the capital of France?",
      "List three benefits of exercise.",
      "Explain how photosynthesis works."
    ];
    
    // Uncomment for actual batch processing:
    // const batchResults = await vllm.processBatch(
    //   `${metadata.vllm_api_url}/v1/completions`,
    //   batchPrompts,
    //   metadata.vllm_model,
    //   { max_tokens: 50 }
    // );
    
    console.log('  In a real application, the following batch of prompts would be processed:');
    batchPrompts.forEach((prompt, index) => {
      console.log(`  [${index + 1}] ${prompt}`);
    });
    
    console.log('  (Skipping actual batch processing in this example)');
  } catch (error) {
    console.error('  Error processing batch:', error);
  }
  
  // 12. Process a batch with metrics
  console.log('\n12. Processing a batch with metrics...');
  try {
    const batchPrompts = [
      "What are three interesting facts about the moon?",
      "Give me a recipe for chocolate cake."
    ];
    
    // Uncomment for actual batch processing with metrics:
    // const [batchResults, metrics] = await vllm.processBatchWithMetrics(
    //   `${metadata.vllm_api_url}/v1/completions`,
    //   batchPrompts,
    //   metadata.vllm_model,
    //   { max_tokens: 100 }
    // );
    
    // console.log('  Batch processing metrics:');
    // console.log(`  - Total time: ${metrics.total_time_ms} ms`);
    // console.log(`  - Average time per item: ${metrics.average_time_per_item_ms} ms`);
    // console.log(`  - Batch size: ${metrics.batch_size}`);
    // console.log(`  - Successful items: ${metrics.successful_items}`);
    
    console.log('  In a real application, a batch of prompts would be processed with metrics:');
    batchPrompts.forEach((prompt, index) => {
      console.log(`  [${index + 1}] ${prompt}`);
    });
    
    console.log('  (Skipping actual batch processing in this example)');
  } catch (error) {
    console.error('  Error processing batch with metrics:', error);
  }
  
  // 13. Stream generation (requires a real VLLM server)
  console.log('\n13. Stream generation example (simulated)...');
  try {
    /*
    // In a real application with a real VLLM server:
    const prompt = "Once upon a time in a land far away";
    
    console.log('  Prompt: Once upon a time in a land far away');
    console.log('  Streamed continuation:');
    
    let fullResponse = '';
    
    for await (const chunk of await vllm.streamGeneration(
      `${metadata.vllm_api_url}/v1/completions`,
      prompt,
      metadata.vllm_model,
      { max_tokens: 50 }
    )) {
      process.stdout.write(chunk);
      fullResponse += chunk;
    }
    
    console.log('\n  (Stream completed)');
    */
    
    // Since we're using a simulated environment, we'll just show the concept
    console.log('  In a real application, you would see the response appear token by token.');
    console.log('  Simulated response:');
    console.log('  Once upon a time in a land far away, there lived a wise old dragon');
    console.log('  who guarded an ancient library filled with the world\'s knowledge.');
    console.log('  The dragon had scales that shimmered like emeralds in the sunlight...');
  } catch (error) {
    console.error('  Error in stream generation:', error);
  }
  
  // 14. Model compatibility checking
  console.log('\n14. Checking model compatibility...');
  const modelsToCheck = [
    'meta-llama/Llama-2-7b-chat-hf',
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'EleutherAI/pythia-12b',
    'databricks/dolly-v2-12b',
    'gpt-3.5-turbo',
    'random-model-name'
  ];
  
  modelsToCheck.forEach(model => {
    const isCompatible = vllm.isCompatibleModel(model);
    console.log(`  - ${model}: ${isCompatible ? 'COMPATIBLE' : 'NOT COMPATIBLE'}`);
  });
  
  console.log('\nExample completed successfully!');
}

// Run the example
main().catch(error => {
  console.error('Example failed with error:', error);
});