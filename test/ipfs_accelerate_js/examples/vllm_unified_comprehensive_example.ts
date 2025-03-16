/**
 * VLLM Unified API Backend Comprehensive Example
 * 
 * This example demonstrates the full range of capabilities offered by the VLLM Unified API backend,
 * including text generation, chat interfaces, streaming, batch processing, LoRA adapter management,
 * quantization settings, and advanced error handling.
 * 
 * VLLM is a high-performance inference engine for large language models that provides:
 * - Continuous batching for efficient processing
 * - PagedAttention for memory-efficient KV cache management
 * - Tensor parallelism for distributed inference across multiple GPUs
 * - Quantization support (AWQ, SqueezeLLM, etc.)
 * - Efficient streaming for responsive UIs
 * 
 * The Unified API backend adds enhanced functionality:
 * - Dual-mode operation (API and container modes)
 * - Circuit breaker pattern for resilience
 * - Resource pooling and connection management
 * - Advanced error handling and monitoring
 * - Performance benchmarking and optimization tools
 * 
 * To run this example:
 * 1. Build the SDK: npm run build
 * 2. Run the example: node dist/examples/vllm_unified_comprehensive_example.js
 * 
 * If you don't have a running VLLM server, the example will use a mock implementation
 * for demonstration purposes.
 */

import { VllmUnified } from '../src/api_backends/vllm_unified';
import { VllmRequest, VllmUnifiedResponse, VllmLoraAdapter, VllmQuantizationConfig } from '../src/api_backends/vllm_unified/types';
import { Message } from '../src/api_backends/types';

// API URLs and model configuration
const DEFAULT_API_URL = process.env.VLLM_API_URL || 'http://localhost:8000';
const DEFAULT_MODEL = process.env.VLLM_MODEL || 'meta-llama/Llama-2-7b-chat-hf';
const SMALL_MODEL = 'facebook/opt-125m'; // For quick, lightweight examples

/**
 * Sleep utility for demonstration purposes
 */
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Format the output of an example section
 */
const formatExampleOutput = (title: string, output: any) => {
  console.log(`\n${title}`);
  console.log('='.repeat(title.length));
  
  if (typeof output === 'string') {
    console.log(output);
  } else {
    console.log(JSON.stringify(output, null, 2));
  }
};

/**
 * Mock the VLLM API functions for demonstration purposes
 * This allows the example to run without a real server
 */
function setupMockImplementation(vllm: VllmUnified) {
  console.log("\n[USING MOCK IMPLEMENTATION]");
  console.log("This example is running with mock data since a real VLLM server is not available.");
  console.log("For a complete demonstration, please start a VLLM server or provide the VLLM_API_URL environment variable.\n");

  // Mock basic request
  vllm.makePostRequestVllm = async (endpoint: string, data: any) => {
    const request = data as VllmRequest;
    
    // Default mock response
    let responseText = "This is a mock response from the VLLM Unified backend. In a real implementation, this would be generated text from a language model.";
    
    // Customize based on prompt if available
    if (request.prompt) {
      if (request.prompt.toLowerCase().includes('summarize')) {
        responseText = "Summary: The document discusses artificial intelligence developments and their impact on society.";
      } else if (request.prompt.toLowerCase().includes('translate')) {
        responseText = "Bonjour, comment ça va? Je suis un modèle de langage.";
      } else if (request.prompt.toLowerCase().includes('dragon')) {
        responseText = "Once upon a time, there was a magnificent dragon with scales that shimmered like emeralds. The dragon lived in a cave at the top of a mountain and was known for its wisdom.";
      }
    }
    
    // If messages are provided (chat format)
    if (request.messages && request.messages.length > 0) {
      const lastMessage = request.messages[request.messages.length - 1];
      if (lastMessage.content.toLowerCase().includes('capital')) {
        responseText = "Paris is the capital of France.";
      } else if (lastMessage.content.toLowerCase().includes('recipe')) {
        responseText = "Here's a simple pasta recipe: Cook pasta according to package instructions. In a pan, sauté garlic in olive oil. Add tomatoes and basil. Toss with pasta and serve with grated cheese.";
      }
    }
    
    // Construct response
    const response: VllmUnifiedResponse = {
      text: responseText,
      metadata: {
        finish_reason: "length",
        model: request.model || DEFAULT_MODEL,
        is_streaming: false,
        usage: {
          prompt_tokens: 15,
          completion_tokens: 30,
          total_tokens: 45
        }
      }
    };
    
    // Simulate network delay
    await sleep(500);
    return response;
  };
  
  // Mock batch processing
  vllm.processBatch = async (endpointUrl: string, prompts: string[], model?: string) => {
    // Create mock responses for each prompt
    const responses = prompts.map(prompt => {
      if (prompt.toLowerCase().includes('capital')) {
        if (prompt.toLowerCase().includes('france')) return "Paris is the capital of France.";
        if (prompt.toLowerCase().includes('italy')) return "Rome is the capital of Italy.";
        if (prompt.toLowerCase().includes('germany')) return "Berlin is the capital of Germany.";
        if (prompt.toLowerCase().includes('japan')) return "Tokyo is the capital of Japan.";
        return "The capital of that country is [capital city].";
      }
      return `Mock response for: ${prompt}`;
    });
    
    // Simulate network delay
    await sleep(800);
    return responses;
  };
  
  // Mock streaming generator
  vllm.makeStreamRequestVllm = async function* (endpointUrl: string, data: any) {
    const mockResponse = "This is a mock streaming response from the VLLM Unified backend. In a real implementation, this would be generated token by token from a language model.";
    const tokens = mockResponse.split(" ");
    
    for (let i = 0; i < tokens.length; i++) {
      // Yield each token as a chunk
      yield {
        text: i === 0 ? tokens[i] : " " + tokens[i],
        metadata: {
          finish_reason: i === tokens.length - 1 ? "stop" : null,
          is_streaming: i < tokens.length - 1
        }
      };
      
      // Simulate token generation delay
      await sleep(100);
    }
  };
  
  // Mock model info
  vllm.getModelInfo = async () => {
    return {
      model: DEFAULT_MODEL,
      max_model_len: 4096,
      num_gpu: 1,
      dtype: "float16",
      gpu_memory_utilization: 0.75,
      quantization: {
        enabled: false
      }
    };
  };
  
  // Mock model stats
  vllm.getModelStatistics = async () => {
    return {
      model: DEFAULT_MODEL,
      statistics: {
        requests_processed: 1250,
        tokens_generated: 78500,
        avg_tokens_per_request: 62.8,
        max_tokens_per_request: 512,
        avg_generation_time: 1.2,
        throughput: 250,
        errors: 5,
        uptime: 86400
      }
    };
  };
  
  // Mock LoRA adapters
  vllm.listLoraAdapters = async () => {
    return [
      {
        id: "adapter1",
        name: "finance-tuned",
        base_model: DEFAULT_MODEL,
        size_mb: 125,
        active: true
      },
      {
        id: "adapter2",
        name: "medical-domain",
        base_model: DEFAULT_MODEL,
        size_mb: 145,
        active: false
      }
    ];
  };
  
  // Mock LoRA adapter loading
  vllm.loadLoraAdapter = async () => {
    return {
      success: true,
      adapter_id: "adapter3",
      message: "Adapter loaded successfully"
    };
  };
  
  // Mock quantization
  vllm.setQuantization = async (endpointUrl: string, model: string, config: VllmQuantizationConfig) => {
    return {
      success: true,
      message: "Quantization applied successfully",
      model: model,
      quantization: config
    };
  };
}

/**
 * Main example runner function
 */
async function runVllmUnifiedComprehensiveExample() {
  console.log("VLLM Unified API Backend - Comprehensive Example");
  console.log("===============================================\n");

  // Initialize the VLLM Unified API backend
  const vllm = new VllmUnified(
    {}, // resources
    {
      vllm_api_url: DEFAULT_API_URL,
      vllm_model: DEFAULT_MODEL,
      timeout: 30000,
      maxRetries: 3
    }
  );

  try {
    // Test the endpoint
    console.log("Testing VLLM endpoint connection...");
    const isAvailable = await vllm.testEndpoint();
    console.log(`Endpoint available: ${isAvailable}\n`);

    // If the endpoint is not available, set up mock implementations
    if (!isAvailable) {
      setupMockImplementation(vllm);
    }

    // -------------------------------------------------------------------------
    // 1. Basic Text Generation
    // -------------------------------------------------------------------------
    console.log("\n1. Basic Text Generation");
    console.log("------------------------");
    
    // 1.1 Simple text generation with string input
    console.log("1.1 Simple text generation with a prompt string");
    const simplePrompt = "Tell me a short story about a dragon.";
    console.log(`Prompt: "${simplePrompt}"`);
    
    const simpleResult = await vllm.makeRequest(
      DEFAULT_API_URL,
      simplePrompt,
      DEFAULT_MODEL
    );
    
    console.log("Response:");
    console.log(simpleResult.text);
    console.log("\nMetadata:");
    console.log(JSON.stringify(simpleResult.metadata, null, 2));
    
    // 1.2 Text generation with parameters
    console.log("\n1.2 Text generation with custom parameters");
    const paramPrompt = {
      prompt: "Summarize the benefits of exercise in 3 points.",
      max_tokens: 150,
      temperature: 0.3,  // Lower temperature for more focused output
      top_p: 0.9,
      repetition_penalty: 1.2
    };
    console.log(`Prompt with parameters: ${JSON.stringify(paramPrompt, null, 2)}`);
    
    const paramResult = await vllm.makeRequest(
      DEFAULT_API_URL,
      paramPrompt,
      DEFAULT_MODEL
    );
    
    console.log("Response:");
    console.log(paramResult.text);
    
    // 1.3 Using a custom endpoint handler
    console.log("\n1.3 Using a custom endpoint handler");
    const customHandler = vllm.createVllmEndpointHandler(
      DEFAULT_API_URL,
      SMALL_MODEL
    );
    
    const handlerResult = await customHandler({
      prompt: "Translate this to French: Hello, how are you?",
      max_tokens: 50
    });
    
    console.log("Response from custom handler:");
    console.log(handlerResult.text);
    
    // 1.4 Using a handler with predefined parameters
    console.log("\n1.4 Using a handler with predefined parameters");
    const paramHandler = vllm.createVllmEndpointHandlerWithParams(
      DEFAULT_API_URL,
      SMALL_MODEL,
      {
        temperature: 0.7,
        top_p: 0.95,
        max_tokens: 100
      }
    );
    
    const paramHandlerResult = await paramHandler({
      prompt: "Give me a recipe for pasta."
    });
    
    console.log("Response from parameter handler:");
    console.log(paramHandlerResult.text);

    // -------------------------------------------------------------------------
    // 2. Chat Interface
    // -------------------------------------------------------------------------
    console.log("\n2. Chat Interface");
    console.log("----------------");
    
    // 2.1 Basic chat completion
    console.log("2.1 Basic chat completion");
    const messages: Message[] = [
      { role: 'system', content: 'You are a helpful assistant that gives concise responses.' },
      { role: 'user', content: 'What is the capital of France?' }
    ];
    
    console.log("Messages:");
    console.log(JSON.stringify(messages, null, 2));
    
    const chatResult = await vllm.chat(
      messages,
      {
        model: DEFAULT_MODEL,
        temperature: 0.7,
        maxTokens: 100
      }
    );
    
    console.log("Chat Response:");
    console.log(chatResult.content);
    console.log("Tokens used:", chatResult.usage);
    
    // 2.2 Multi-turn conversation
    console.log("\n2.2 Multi-turn conversation");
    const conversation: Message[] = [
      { role: 'system', content: 'You are a helpful cooking assistant.' },
      { role: 'user', content: 'I want to make pasta.' },
      { role: 'assistant', content: 'Great! Pasta is a versatile dish. Do you have a specific type of pasta or sauce in mind?' },
      { role: 'user', content: 'I have tomatoes and garlic. What can I make?' }
    ];
    
    console.log("Conversation:");
    conversation.forEach((msg, i) => {
      if (msg.role !== 'system') {
        console.log(`${msg.role}: ${msg.content}`);
      }
    });
    
    const multiTurnResult = await vllm.chat(
      conversation,
      {
        model: DEFAULT_MODEL,
        temperature: 0.8,
        maxTokens: 150
      }
    );
    
    console.log("\nAssistant:", multiTurnResult.content);

    // -------------------------------------------------------------------------
    // 3. Streaming Generation
    // -------------------------------------------------------------------------
    console.log("\n3. Streaming Generation");
    console.log("----------------------");
    
    // 3.1 Basic streaming text generation
    console.log("3.1 Basic streaming text generation");
    console.log("Prompt: 'Tell me about artificial intelligence.'");
    console.log("Streaming response:");
    
    const stream = vllm.streamGeneration(
      DEFAULT_API_URL,
      "Tell me about artificial intelligence.",
      DEFAULT_MODEL,
      { max_tokens: 100, temperature: 0.7 }
    );
    
    let streamedText = "";
    for await (const chunk of stream) {
      process.stdout.write(chunk.text);
      streamedText += chunk.text;
    }
    
    console.log("\n\nFinal streamed text length:", streamedText.length);
    
    // 3.2 Streaming chat generation
    console.log("\n3.2 Streaming chat generation");
    
    const chatMessages: Message[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Explain quantum computing briefly.' }
    ];
    
    console.log("Prompt: 'Explain quantum computing briefly.'");
    console.log("Streaming response:");
    
    const chatStream = vllm.streamChat(
      chatMessages,
      {
        model: DEFAULT_MODEL,
        temperature: 0.5,
        maxTokens: 150
      }
    );
    
    let streamedChatText = "";
    for await (const chunk of chatStream) {
      process.stdout.write(chunk.content);
      streamedChatText += chunk.content;
      
      if (chunk.done) {
        console.log("\n\nStreaming completed.");
      }
    }
    
    console.log("Final streamed chat text length:", streamedChatText.length);

    // -------------------------------------------------------------------------
    // 4. Batch Processing
    // -------------------------------------------------------------------------
    console.log("\n4. Batch Processing");
    console.log("------------------");
    
    // 4.1 Basic batch processing
    console.log("4.1 Basic batch processing");
    const prompts = [
      "What is the capital of France?",
      "What is the capital of Italy?",
      "What is the capital of Germany?",
      "What is the capital of Japan?"
    ];
    
    console.log("Batch prompts:");
    prompts.forEach((prompt, i) => console.log(`${i + 1}. ${prompt}`));
    
    const batchResults = await vllm.processBatch(
      DEFAULT_API_URL,
      prompts,
      DEFAULT_MODEL,
      { temperature: 0.1, max_tokens: 20 }
    );
    
    console.log("\nBatch results:");
    batchResults.forEach((result, i) => {
      console.log(`${i + 1}. ${prompts[i]}`);
      console.log(`   ${result}`);
    });
    
    // 4.2 Batch processing with metrics
    console.log("\n4.2 Batch processing with metrics");
    
    const [resultsWithMetrics, metrics] = await vllm.processBatchWithMetrics(
      DEFAULT_API_URL,
      prompts,
      DEFAULT_MODEL,
      { temperature: 0.1, max_tokens: 20 }
    );
    
    console.log("Batch metrics:");
    console.log(JSON.stringify(metrics, null, 2));

    // -------------------------------------------------------------------------
    // 5. Model Information and Statistics
    // -------------------------------------------------------------------------
    console.log("\n5. Model Information and Statistics");
    console.log("----------------------------------");
    
    // 5.1 Get model information
    console.log("5.1 Get model information");
    const modelInfo = await vllm.getModelInfo(
      DEFAULT_API_URL,
      DEFAULT_MODEL
    );
    
    formatExampleOutput("Model Information", modelInfo);
    
    // 5.2 Get model statistics
    console.log("\n5.2 Get model statistics");
    const modelStats = await vllm.getModelStatistics(
      DEFAULT_API_URL,
      DEFAULT_MODEL
    );
    
    formatExampleOutput("Model Statistics", modelStats);

    // -------------------------------------------------------------------------
    // 6. LoRA Adapters Management
    // -------------------------------------------------------------------------
    console.log("\n6. LoRA Adapters Management");
    console.log("---------------------------");
    
    // 6.1 List LoRA adapters
    console.log("6.1 List LoRA adapters");
    const adapters = await vllm.listLoraAdapters(DEFAULT_API_URL);
    
    formatExampleOutput("LoRA Adapters", adapters);
    
    // 6.2 Load a LoRA adapter
    console.log("\n6.2 Load a LoRA adapter");
    const adapterData = {
      adapter_name: "finance-domain",
      adapter_path: "/path/to/adapter",
      adapter_config: {
        r: 16,
        lora_alpha: 32,
        target_modules: ["q_proj", "v_proj"]
      }
    };
    
    console.log("Loading adapter with data:");
    console.log(JSON.stringify(adapterData, null, 2));
    
    const loadResult = await vllm.loadLoraAdapter(
      DEFAULT_API_URL,
      adapterData
    );
    
    formatExampleOutput("Load Result", loadResult);

    // -------------------------------------------------------------------------
    // 7. Quantization Settings
    // -------------------------------------------------------------------------
    console.log("\n7. Quantization Settings");
    console.log("-----------------------");
    
    // 7.1 Set quantization configuration
    console.log("7.1 Set quantization configuration");
    const quantConfig: VllmQuantizationConfig = {
      enabled: true,
      method: "awq",
      bits: 4
    };
    
    console.log("Setting quantization config:");
    console.log(JSON.stringify(quantConfig, null, 2));
    
    const quantResult = await vllm.setQuantization(
      DEFAULT_API_URL,
      DEFAULT_MODEL,
      quantConfig
    );
    
    formatExampleOutput("Quantization Result", quantResult);

    // -------------------------------------------------------------------------
    // 8. Error Handling and Resilience
    // -------------------------------------------------------------------------
    console.log("\n8. Error Handling and Resilience");
    console.log("-------------------------------");
    
    // 8.1 Handle non-existent model
    console.log("8.1 Handle non-existent model");
    try {
      const invalidModelResult = await vllm.makeRequest(
        DEFAULT_API_URL,
        "This should fail",
        "non-existent-model"
      );
      console.log("This should not execute - model should fail");
      console.log(invalidModelResult);
    } catch (error) {
      console.log("Expected error caught successfully:");
      console.log(`Error message: ${error.message}`);
      console.log(`Status code: ${error.statusCode || 'N/A'}`);
      console.log(`Error type: ${error.type || 'N/A'}`);
    }
    
    // 8.2 Retry mechanism demonstration
    console.log("\n8.2 Retry mechanism demonstration (simulated)");
    
    // Mock implementation to simulate retries
    const originalMakePostRequestVllm = vllm.makePostRequestVllm;
    let attempts = 0;
    
    vllm.makePostRequestVllm = async (endpoint, data, requestId, options) => {
      attempts++;
      if (attempts <= 2) {
        console.log(`Attempt ${attempts}: Simulating a failure (429 rate limit)`);
        const error = vllm.createApiError("Rate limit exceeded", 429, "rate_limit_error");
        error.retryAfter = 1;
        error.isRateLimitError = true;
        throw error;
      }
      
      console.log(`Attempt ${attempts}: Successful request after retries`);
      return {
        text: "This response came after successful retries.",
        metadata: {
          finish_reason: "length",
          model: "retry-test-model",
          usage: { prompt_tokens: 5, completion_tokens: 10, total_tokens: 15 }
        }
      };
    };
    
    try {
      const retryResult = await vllm.makeRequest(
        DEFAULT_API_URL,
        "Test retry mechanism",
        "retry-test-model",
        { maxRetries: 3 }
      );
      
      console.log("Result after retries:");
      console.log(retryResult.text);
    } catch (error) {
      console.log("Retry mechanism failed:", error);
    } finally {
      // Restore original implementation
      vllm.makePostRequestVllm = originalMakePostRequestVllm;
    }
    
    // 8.3 Circuit breaker pattern demonstration
    console.log("\n8.3 Circuit breaker pattern demonstration (simulated)");
    console.log("The circuit breaker prevents cascading failures by disabling an API endpoint after consecutive errors.");
    console.log("This is a simulated demonstration - in production, the circuit breaker is activated after multiple failures.");
    
    // Simulate circuit breaker
    console.log("Status: Circuit open (healthy)");
    console.log("3 consecutive failures detected...");
    console.log("Status: Circuit closed (unhealthy)");
    console.log("Requests temporarily blocked to prevent cascading failures");
    console.log("Cooling down period (30 seconds)...");
    console.log("Testing endpoint health...");
    console.log("Status: Circuit half-open (testing)");
    console.log("Test request successful");
    console.log("Status: Circuit open (healthy)");
    console.log("Normal operation resumed\n");
    
    // 8.4 Timeout handling
    console.log("8.4 Timeout handling (simulated)");
    
    // Mock implementation to simulate timeout
    vllm.makePostRequestVllm = async () => {
      console.log("Simulating a request that takes too long...");
      throw vllm.createApiError("Request timed out after 30000ms", 408, "timeout_error");
    };
    
    try {
      const timeoutResult = await vllm.makeRequest(
        DEFAULT_API_URL,
        "This should timeout",
        DEFAULT_MODEL,
        { timeout: 30000 }
      );
      console.log("This should not execute - request should timeout");
    } catch (error) {
      console.log("Expected timeout error caught successfully:");
      console.log(`Error message: ${error.message}`);
      console.log(`Status code: ${error.statusCode || 'N/A'}`);
      console.log(`Error type: ${error.type || 'N/A'}`);
    } finally {
      // Restore original implementation
      vllm.makePostRequestVllm = originalMakePostRequestVllm;
    }

    // -------------------------------------------------------------------------
    // 9. Performance Considerations
    // -------------------------------------------------------------------------
    console.log("\n9. Performance Considerations");
    console.log("----------------------------");
    
    console.log("9.1 Generation parameter optimization");
    console.log(`
  Performance can be significantly affected by generation parameters:
  
  - Lower temperature (0.1-0.3): More deterministic, faster responses
  - Higher temperature (0.7-1.0): More creative, potentially slower
  - Lower max_tokens: Faster responses, less content
  - Use stop sequences: End generation early when appropriate
  - top_k and top_p: Balance between speed and quality
    `);
    
    console.log("9.2 Hardware considerations");
    console.log(`
  VLLM performance scales with hardware:
  
  - CPU: Limited performance, suitable for small models only
  - Single GPU: Good performance for most models
  - Multiple GPUs: Required for largest models (>20B parameters)
  - VRAM requirements:
    - 7B parameter models: ~14GB VRAM
    - 13B parameter models: ~26GB VRAM
    - 70B parameter models: Multiple GPUs required
    `);
    
    console.log("9.3 Container mode performance advantages");
    console.log(`
  Container mode provides significant performance benefits:
  
  - Lower latency (no API overhead)
  - Higher throughput with continuous batching
  - No rate limits
  - Local tensor parallelism
  - Customizable quantization
  - Persistent model loading
    `);

    // -------------------------------------------------------------------------
    // 10. Advanced API Usage with Endpoints
    // -------------------------------------------------------------------------
    console.log("\n10. Advanced API Usage with Endpoints");
    console.log("------------------------------------");
    
    // 10.1 Create and use multiple endpoints
    console.log("10.1 Create and use multiple endpoints");
    
    // Register endpoints
    const endpointId1 = vllm.createEndpoint({
      name: "primary",
      apiKey: "key1",
      maxConcurrentRequests: 10,
      priority: "HIGH"
    });
    
    const endpointId2 = vllm.createEndpoint({
      name: "secondary",
      apiKey: "key2",
      maxConcurrentRequests: 5,
      priority: "MEDIUM"
    });
    
    console.log(`Created endpoints: ${endpointId1}, ${endpointId2}`);
    
    // Make requests with specific endpoints
    try {
      console.log("\nMaking request with primary endpoint...");
      const primaryResult = await vllm.makeRequestWithEndpoint(
        endpointId1,
        "What is machine learning?",
        DEFAULT_MODEL
      );
      
      console.log("Primary endpoint response:");
      console.log(primaryResult.text);
      
      console.log("\nMaking request with secondary endpoint...");
      const secondaryResult = await vllm.makeRequestWithEndpoint(
        endpointId2,
        "What is deep learning?",
        DEFAULT_MODEL
      );
      
      console.log("Secondary endpoint response:");
      console.log(secondaryResult.text);
      
      // Check endpoint statistics
      const stats1 = vllm.getEndpointStats(endpointId1);
      const stats2 = vllm.getEndpointStats(endpointId2);
      
      console.log("\nEndpoint statistics:");
      console.log("Primary endpoint:", stats1);
      console.log("Secondary endpoint:", stats2);
      
    } catch (error) {
      console.log("Error using endpoints:", error);
    }

    // -------------------------------------------------------------------------
    // 11. Best Practices and Recommendations
    // -------------------------------------------------------------------------
    console.log("\n11. Best Practices and Recommendations");
    console.log("--------------------------------------");
    
    console.log(`
VLLM Unified API Backend Best Practices:

1. Error Handling:
   - Always implement robust error handling with typed error classification
   - Use the retry mechanism for transient errors
   - Implement proper circuit breaker pattern for resilience

2. Performance Optimization:
   - Use smaller models for less complex tasks
   - Adjust temperature and other parameters based on requirements
   - Consider container mode for high-throughput applications
   - Implement request batching for similar requests

3. Resource Management:
   - Monitor token usage to control costs
   - Implement timeouts to prevent hanging requests
   - Use streaming for responsive UIs and early termination

4. Model Selection:
   - Choose the smallest model that meets requirements
   - Consider quantized models for better performance
   - Use LoRA adapters for domain-specific tasks

5. Security:
   - Store API keys securely
   - Validate and sanitize user inputs
   - Implement rate limiting and request throttling
    `);

    console.log("\nVLLM Unified API Comprehensive Example Completed Successfully!");

  } catch (error) {
    console.error("Error running VLLM Unified Comprehensive Example:", error);
  }
}

// Run the example
runVllmUnifiedComprehensiveExample().catch(console.error);