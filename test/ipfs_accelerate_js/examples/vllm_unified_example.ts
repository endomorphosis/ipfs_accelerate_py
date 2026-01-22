/**
 * VLLM Unified API Backend Example
 * 
 * This example demonstrates how to use the VLLM Unified API backend for text generation,
 * batched inference, streaming, and model management.
 * 
 * To run this example:
 * 1. Build the SDK: npm run build
 * 2. Run the example: node dist/examples/vllm_unified_example.js
 */

import { VllmUnified } from '../src/api_backends/vllm_unified';

async function runVllmUnifiedExample() {
  console.log("VLLM Unified API Backend Example");
  console.log("================================\n");

  // Initialize the VLLM Unified API backend
  const vllm = new VllmUnified(
    {}, // resources
    {
      vllm_api_url: process.env.VLLM_API_URL || 'http://localhost:8000',
      vllm_model: process.env.VLLM_MODEL || 'meta-llama/Llama-2-7b-chat-hf'
    }
  );

  try {
    // Test the endpoint
    console.log("Testing VLLM endpoint...");
    const isAvailable = await vllm.testEndpoint();
    console.log(`Endpoint available: ${isAvailable}\n`);

    if (!isAvailable) {
      console.log("This example uses a mock implementation since the VLLM server is not available.");
      console.log("For a real demo, please start a VLLM server first or provide correct VLLM_API_URL.\n");
      
      // Mock the API for demonstration
      mockVllmApi(vllm);
    }

    // Basic text generation
    console.log("1. Basic Text Generation");
    console.log("----------------------");
    const genResult = await vllm.makeRequest(
      process.env.VLLM_API_URL || 'http://localhost:8000',
      'Tell me a very short story about a dragon.',
      process.env.VLLM_MODEL || 'meta-llama/Llama-2-7b-chat-hf'
    );
    console.log("Result:");
    console.log(genResult.text);
    console.log();

    // Chat completion
    console.log("2. Chat Completion");
    console.log("----------------");
    const chatResult = await vllm.chat(
      [
        { role: 'system', content: 'You are a helpful assistant that gives very short responses.' },
        { role: 'user', content: 'What is the capital of France?' }
      ],
      {
        model: process.env.VLLM_MODEL || 'meta-llama/Llama-2-7b-chat-hf',
        temperature: 0.7
      }
    );
    console.log("Chat Result:");
    console.log(chatResult.content);
    console.log();

    // Batch processing
    console.log("3. Batch Processing");
    console.log("----------------");
    const prompts = [
      'What is the capital of France?',
      'What is the capital of Italy?',
      'What is the capital of Germany?'
    ];
    const batchResults = await vllm.processBatch(
      process.env.VLLM_API_URL || 'http://localhost:8000',
      prompts,
      process.env.VLLM_MODEL || 'meta-llama/Llama-2-7b-chat-hf',
      { temperature: 0.1, max_tokens: 10 }
    );
    console.log("Batch Results:");
    for (let i = 0; i < prompts.length; i++) {
      console.log(`Question: ${prompts[i]}`);
      console.log(`Answer: ${batchResults[i]}`);
      console.log();
    }

    // Demonstrate streaming (using a mock for the example)
    console.log("4. Streaming Generation (simulated)");
    console.log("-------------------------------");
    console.log("Streaming: ");
    simulateStreaming();

    // Custom endpoint handler
    console.log("\n5. Custom Endpoint Handler");
    console.log("----------------------");
    const handler = vllm.createVllmEndpointHandler(
      process.env.VLLM_API_URL || 'http://localhost:8000',
      process.env.VLLM_MODEL || 'meta-llama/Llama-2-7b-chat-hf'
    );
    
    const handlerResult = await handler({ prompt: 'What is the capital of Spain?' });
    console.log("Handler Result:");
    console.log(handlerResult.text);
    console.log();

    // Model information (mock for demonstration)
    console.log("6. Model Information (simulated)");
    console.log("----------------------------");
    // In a real implementation, this would be:
    // const modelInfo = await vllm.getModelInfo(process.env.VLLM_API_URL || 'http://localhost:8000', process.env.VLLM_MODEL);
    const modelInfo = {
      model: process.env.VLLM_MODEL || 'meta-llama/Llama-2-7b-chat-hf',
      max_model_len: 4096,
      num_gpu: 1,
      dtype: 'float16',
      gpu_memory_utilization: 0.75,
      quantization: {
        enabled: false,
        method: null
      }
    };
    console.log("Model Information:");
    console.log(JSON.stringify(modelInfo, null, 2));
    console.log();

    console.log("VLLM Unified API Example Completed Successfully!");

  } catch (error) {
    console.error("Error running VLLM example:", error);
  }
}

// Helper function to mock the API for demonstration purposes
function mockVllmApi(vllm: VllmUnified) {
  // Mock the makePostRequestVllm method
  vllm.makePostRequestVllm = async () => {
    return {
      text: "Once upon a time, there was a magnificent dragon with scales that shimmered like emeralds. The dragon lived in a cave at the top of a mountain and was known for its wisdom. Many brave souls would climb the mountain to seek advice from the ancient creature. The dragon was kind to those who approached with respect, but fiercely protective of its home.",
      metadata: {
        finish_reason: "length",
        model: "llama-7b",
        usage: {
          prompt_tokens: 10,
          completion_tokens: 70,
          total_tokens: 80
        }
      }
    };
  };

  // Mock the processBatch method
  vllm.processBatch = async (_endpointUrl: string, prompts: string[]) => {
    const responses = [
      "Paris is the capital of France.",
      "Rome is the capital of Italy.",
      "Berlin is the capital of Germany."
    ];
    return responses.slice(0, prompts.length);
  };
}

// Helper function to simulate streaming for the example
function simulateStreaming() {
  const response = "The dragon breathed fire into the night sky, illuminating the mountains around.";
  let index = 0;
  
  const interval = setInterval(() => {
    if (index < response.length) {
      process.stdout.write(response[index]);
      index++;
    } else {
      clearInterval(interval);
    }
  }, 50);
}

// Run the example
runVllmUnifiedExample().catch(console.error);