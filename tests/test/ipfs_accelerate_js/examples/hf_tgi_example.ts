/**
 * HuggingFace Text Generation Inference API Backend Example
 * 
 * This example demonstrates how to use the HF TGI backend for:
 * 1. Basic text generation
 * 2. Text generation with custom parameters
 * 3. Streaming text generation
 * 4. Chat API with structured messages
 * 5. Managing multiple endpoints
 * 6. Endpoint testing and stats tracking
 */

// Import the HfTgi class and types
import { HfTgi } from '../src/api_backends/hf_tgi/hf_tgi';
import { Message } from '../src/api_backends/types';
import { HfTgiChatOptions, HfTgiEndpoint } from '../src/api_backends/hf_tgi/types';

async function main() {
  console.log('HuggingFace Text Generation Inference API Backend Example');
  
  try {
    // -------------------------------------------------------------------------------
    // Example 1: Initialize the HF TGI backend with API key
    // -------------------------------------------------------------------------------
    const apiKey = process.env.HF_API_KEY || 'your_api_key'; // Replace with your API key or set env var
    
    const hfTgi = new HfTgi(
      {}, // No specific resources required
      {
        hf_api_key: apiKey, 
        model_id: 'gpt2', // Default model to use
      }
    );
    
    console.log('\n1. HF TGI backend initialized successfully');
    
    // -------------------------------------------------------------------------------
    // Example 2: Basic text generation
    // -------------------------------------------------------------------------------
    console.log('\n2. Basic text generation:');
    
    try {
      const response = await hfTgi.generateText(
        'gpt2', // Model ID
        'Once upon a time in a land far away', // Input text
        { max_new_tokens: 20 } // Parameters
      );
      
      console.log('Generated text:');
      console.log(response.generated_text);
    } catch (error) {
      console.error('Error during text generation:', error);
    }

    // -------------------------------------------------------------------------------
    // Example 3: Text generation with custom parameters
    // -------------------------------------------------------------------------------
    console.log('\n3. Text generation with custom parameters:');
    
    try {
      const response = await hfTgi.generateText(
        'gpt2',
        'The future of artificial intelligence is',
        {
          max_new_tokens: 30,
          temperature: 0.7,
          top_p: 0.9,
          top_k: 50,
          repetition_penalty: 1.2,
          do_sample: true
        }
      );
      
      console.log('Generated text with custom parameters:');
      console.log(response.generated_text);
    } catch (error) {
      console.error('Error during text generation with custom parameters:', error);
    }
    
    // -------------------------------------------------------------------------------
    // Example 4: Streaming text generation
    // -------------------------------------------------------------------------------
    console.log('\n4. Streaming text generation:');
    
    try {
      const stream = hfTgi.streamGenerate(
        'gpt2',
        'The most important invention of the 21st century',
        {
          max_new_tokens: 30,
          temperature: 0.8
        }
      );
      
      console.log('Streaming response:');
      let fullText = '';
      
      for await (const chunk of stream) {
        process.stdout.write(chunk.content);
        fullText += chunk.content;
      }
      
      console.log('\nFull streaming text: ', fullText);
    } catch (error) {
      console.error('Error during streaming text generation:', error);
    }
    
    // -------------------------------------------------------------------------------
    // Example 5: Using the chat API
    // -------------------------------------------------------------------------------
    console.log('\n5. Using the chat API:');
    
    try {
      const messages: Message[] = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Tell me about the origins of machine learning.' }
      ];
      
      const chatOptions: HfTgiChatOptions = {
        model: 'gpt2',
        max_new_tokens: 100,
        temperature: 0.7
      };
      
      const chatResponse = await hfTgi.chat(messages, chatOptions);
      
      console.log('Chat API response:');
      console.log(chatResponse.text);
      console.log('Token usage:', chatResponse.usage);
    } catch (error) {
      console.error('Error during chat API usage:', error);
    }
    
    // -------------------------------------------------------------------------------
    // Example 6: Streaming chat API
    // -------------------------------------------------------------------------------
    console.log('\n6. Streaming chat API:');
    
    try {
      const messages: Message[] = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Explain quantum computing in simple terms.' }
      ];
      
      const chatOptions: HfTgiChatOptions = {
        model: 'gpt2',
        max_new_tokens: 50,
        temperature: 0.7
      };
      
      const chatStream = hfTgi.streamChat(messages, chatOptions);
      
      console.log('Streaming chat response:');
      let fullChatText = '';
      
      for await (const chunk of chatStream) {
        process.stdout.write(chunk.content);
        fullChatText += chunk.content;
      }
      
      console.log('\nFull chat text: ', fullChatText);
    } catch (error) {
      console.error('Error during streaming chat API usage:', error);
    }
    
    // -------------------------------------------------------------------------------
    // Example 7: Managing multiple endpoints
    // -------------------------------------------------------------------------------
    console.log('\n7. Managing multiple endpoints:');
    
    // Create a named endpoint
    const endpointId = hfTgi.createEndpoint({
      id: 'gpt2-endpoint',
      apiKey: apiKey,
      model: 'gpt2',
      maxConcurrentRequests: 5,
      queueSize: 10,
      timeout: 30000
    });
    
    console.log(`Created endpoint with ID: ${endpointId}`);
    
    // Get the endpoint
    const endpoint = hfTgi.getEndpoint(endpointId) as HfTgiEndpoint;
    console.log(`Endpoint model: ${endpoint.model_id}`);
    
    // Make a request with a specific endpoint
    try {
      const response = await hfTgi.makeRequestWithEndpoint(
        endpointId,
        {
          inputs: 'Artificial intelligence will transform',
          parameters: {
            max_new_tokens: 20,
            temperature: 0.8
          }
        }
      );
      
      console.log('Response from specific endpoint:');
      console.log(response.generated_text);
    } catch (error) {
      console.error('Error during endpoint-specific request:', error);
    }
    
    // -------------------------------------------------------------------------------
    // Example 8: Testing an endpoint
    // -------------------------------------------------------------------------------
    console.log('\n8. Testing an endpoint:');
    
    try {
      const endpointUrl = `https://api-inference.huggingface.co/models/gpt2`;
      const isEndpointWorking = await hfTgi.testEndpoint(endpointUrl, apiKey, 'gpt2');
      
      console.log(`Endpoint test result: ${isEndpointWorking ? 'Working' : 'Not working'}`);
    } catch (error) {
      console.error('Error during endpoint testing:', error);
    }
    
    // -------------------------------------------------------------------------------
    // Example 9: Getting usage statistics
    // -------------------------------------------------------------------------------
    console.log('\n9. Getting usage statistics:');
    
    // Get stats for a specific endpoint
    const endpointStats = hfTgi.getStats(endpointId);
    console.log('Endpoint stats:', endpointStats);
    
    // Get global stats
    const globalStats = hfTgi.getStats();
    console.log('Global stats:', globalStats);
    
    // Reset stats for an endpoint
    hfTgi.resetStats(endpointId);
    console.log('Stats reset for endpoint');
    
    // -------------------------------------------------------------------------------
    // Example 10: Advanced - Custom endpoint with specific URL
    // -------------------------------------------------------------------------------
    console.log('\n10. Advanced - Custom endpoint with specific URL:');
    
    // Create a custom endpoint for a self-hosted TGI server
    const customEndpointId = hfTgi.createEndpoint({
      id: 'custom-tgi-server',
      model_id: 'mistral-7b-instruct',
      api_key: apiKey,
      endpoint_url: 'http://your-custom-tgi-server:8080/generate', // Replace with actual URL
      timeout: 60000
    });
    
    console.log(`Created custom endpoint with ID: ${customEndpointId}`);
    
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