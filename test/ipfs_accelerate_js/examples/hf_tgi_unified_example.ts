/**
 * Example demonstrating comprehensive usage of the HF TGI Unified API Backend
 * 
 * This example shows how to use the HF TGI Unified API client for text generation
 * with various configuration options and features.
 */
import { HfTgiUnified } from '../src/api_backends/hf_tgi_unified';
import { ApiMetadata } from '../src/api_backends/types';

/**
 * A complete example of using the HF TGI Unified API backend with all available features
 */
async function runHfTgiUnifiedExample() {
  console.log('Starting HF TGI Unified API Example');

  // Create an instance of the HF TGI Unified API backend
  const hfTgiUnified = new HfTgiUnified(
    {}, // resources
    {
      hf_tgi_api_key: process.env.HF_API_KEY || 'your-api-key', 
      hf_tgi_model: 'TinyLlama/TinyLlama-1.1B-Chat-v0.1',
      hf_tgi_api_url: process.env.HF_TGI_API_URL || 'https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v0.1'
    }
  );

  try {
    // Test the endpoint to ensure connectivity
    const isEndpointWorking = await hfTgiUnified.testEndpoint();
    console.log(`HF TGI endpoint working: ${isEndpointWorking}`);
    
    if (\!isEndpointWorking) {
      throw new Error('HF TGI endpoint is not available');
    }

    // Basic chat example
    const basicChatResult = await hfTgiUnified.chat(
      [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is the capital of France?' }
      ]
    );
    
    console.log('Basic chat result:');
    console.log(basicChatResult.content);
    console.log(`Model: ${basicChatResult.model}`);
    console.log('---');

    // Chat with parameters
    const parameterizedChatResult = await hfTgiUnified.chat(
      [
        { role: 'system', content: 'You are a creative assistant who responds with unique ideas.' },
        { role: 'user', content: 'Suggest a name for a pet robot.' }
      ],
      {
        temperature: 0.9,
        maxTokens: 150,
        topP: 0.8,
        topK: 40,
        repetitionPenalty: 1.2,
        doSample: true
      }
    );
    
    console.log('Chat with parameters result:');
    console.log(parameterizedChatResult.content);
    console.log(`Model: ${parameterizedChatResult.model}`);
    console.log('---');

    // Streaming chat example
    console.log('Streaming chat result:');
    
    const streamingMessages = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Write a short poem about coding.' }
    ];
    
    const stream = hfTgiUnified.streamChat(streamingMessages, { temperature: 0.7 });
    
    let streamedContent = '';
    for await (const chunk of await stream) {
      streamedContent += chunk.content;
      // In a real application, you would update the UI incrementally
      process.stdout.write(chunk.content);
    }
    
    console.log('\n---');

    // Chat with different models
    // Use Mistral 7B model
    const mistralClient = new HfTgiUnified(
      {}, 
      {
        hf_tgi_api_key: process.env.HF_API_KEY || 'your-api-key',
        hf_tgi_model: 'mistralai/Mistral-7B-Instruct-v0.1'
      }
    );
    
    if (await mistralClient.testEndpoint()) {
      console.log('Using Mistral model:');
      
      const mistralResult = await mistralClient.chat(
        [
          { role: 'user', content: 'Explain how transformers work in one paragraph.' }
        ],
        { maxTokens: 200 }
      );
      
      console.log(mistralResult.content);
      console.log('---');
    }

    // Advanced TGI features
    console.log('Advanced TGI features:');
    
    // Using stop sequences
    const stopSequencesResult = await hfTgiUnified.chat(
      [
        { role: 'user', content: 'Create a list of 5 best programming languages' }
      ],
      {
        stop: ['5.', '5)', '5:'],  // Stop after the 5th item
        maxTokens: 200
      }
    );
    
    console.log('Chat with stop sequences:');
    console.log(stopSequencesResult.content);
    console.log('---');
    
    // Using watermarking
    const watermarkResult = await hfTgiUnified.makePostRequest({
      inputs: 'Tell me about artificial intelligence',
      parameters: {
        max_new_tokens: 100,
        watermark: true,  // Enable watermarking
        details: true     // Return detailed token information
      }
    });
    
    console.log('Response with watermarking:');
    console.log('Generated text:', watermarkResult.generated_text);
    if (watermarkResult.details) {
      console.log('Tokens:', watermarkResult.details.generated_tokens);
      console.log('Finish reason:', watermarkResult.details.finish_reason);
    }
    console.log('---');
    
    // Error handling example
    try {
      await hfTgiUnified.chat(
        [{ role: 'user', content: 'Hello' }],
        { model: 'non-existent-model' }
      );
    } catch (error) {
      console.log('Error handling example:');
      console.log(`Caught error: ${error.message}`);
      console.log('---');
    }

    // Advanced queue and circuit breaker configuration
    const advancedHfTgi = new HfTgiUnified(
      {}, 
      {
        hf_tgi_api_key: process.env.HF_API_KEY || 'your-api-key',
        hf_tgi_model: 'TinyLlama/TinyLlama-1.1B-Chat-v0.1',
        queue_size: 10,
        circuit_breaker_threshold: 5,
        circuit_breaker_timeout_ms: 30000,
        retry_count: 3,
        retry_delay_ms: 1000
      }
    );
    
    console.log('Created HF TGI Unified instance with advanced configuration');
    console.log('---');

    console.log('HF TGI Unified example completed successfully');
  } catch (error) {
    console.error('HF TGI Unified example failed:', error);
  }
}

// Run the example if executed directly
if (require.main === module) {
  runHfTgiUnifiedExample().catch(console.error);
}

// Export for use in other examples
export { runHfTgiUnifiedExample };
