/**
 * Ollama API Backend Example
 * 
 * This example demonstrates how to use the Ollama backend for:
 * 1. Basic chat generation
 * 2. Streaming chat responses
 * 3. Model listing and compatibility
 * 4. Direct text generation
 * 5. Circuit breaker pattern for error handling
 * 
 * Ollama is a local LLM runtime that allows you to run open models like Llama3 and Mistral
 * on your own hardware.
 */

// Import the Ollama class and dependencies
import { Ollama } from '../src/api_backends/ollama/ollama';
import { Message } from '../src/api_backends/types';
import { OllamaOptions } from '../src/api_backends/ollama/types';

async function main() {
  console.log('Ollama API Backend Example');
  
  try {
    // -------------------------------------------------------------------------------
    // Example 1: Initialize the Ollama backend
    // -------------------------------------------------------------------------------
    // By default, Ollama connects to http://localhost:11434/api
    // You can override with OLLAMA_API_URL environment variable
    const ollama = new Ollama(
      {}, // Resources (not needed for this example)
      {
        // You can override the default URL
        ollama_api_url: process.env.OLLAMA_API_URL || 'http://localhost:11434/api',
        // Default model if not specified in requests
        model: process.env.OLLAMA_MODEL || 'llama3'
      }
    );
    
    console.log('\n1. Ollama backend initialized');
    console.log('   Backend info:', ollama.getBackendInfo());
    
    // -------------------------------------------------------------------------------
    // Example 2: Test Ollama endpoint
    // -------------------------------------------------------------------------------
    console.log('\n2. Testing Ollama endpoint:');
    
    try {
      const endpointWorking = await ollama.testOllamaEndpoint();
      console.log(`   Endpoint working: ${endpointWorking}`);
      
      if (!endpointWorking) {
        console.log('   Ollama endpoint is not responding. Is Ollama running?');
        console.log('   Please start Ollama and try again.');
        console.log('   For installation instructions, visit: https://ollama.ai/');
        console.log('   After installation, run: ollama serve');
        console.log('   This example will continue with simulated responses.');
      }
    } catch (error) {
      console.log('   Error testing endpoint:', error);
      console.log('   Continuing with simulated responses...');
    }
    
    // -------------------------------------------------------------------------------
    // Example 3: List available models
    // -------------------------------------------------------------------------------
    console.log('\n3. Listing available models:');
    
    try {
      const models = await ollama.listModels();
      console.log('   Available models:');
      
      if (models.length === 0) {
        console.log('   No models found. To pull a model, run: ollama pull llama3');
      } else {
        models.forEach((model: any, index: number) => {
          console.log(`   ${index + 1}. ${model.name} (${model.size})`);
        });
      }
    } catch (error) {
      console.log('   Error listing models:', error);
      console.log('   Continuing with example...');
    }
    
    // -------------------------------------------------------------------------------
    // Example 4: Basic chat generation
    // -------------------------------------------------------------------------------
    console.log('\n4. Basic chat generation:');
    
    try {
      // Define messages for the chat
      const messages: Message[] = [
        { role: 'system', content: 'You are a helpful assistant that provides concise answers.' },
        { role: 'user', content: 'What is machine learning?' }
      ];
      
      // Request options
      const options = {
        temperature: 0.7,
        max_tokens: 150
      };
      
      // The model name to use ('llama3' is used as default if omitted)
      const model = 'llama3'; // Change to a model you have pulled
      
      console.log('   Generating chat response:');
      console.log('   System: You are a helpful assistant that provides concise answers.');
      console.log('   User: What is machine learning?');
      
      const response = await ollama.chat(model, messages, options);
      
      console.log('   Assistant:', response.text);
      
      if (response.usage) {
        console.log('   Usage statistics:');
        console.log(`   - Prompt tokens: ${response.usage.prompt_tokens}`);
        console.log(`   - Completion tokens: ${response.usage.completion_tokens}`);
        console.log(`   - Total tokens: ${response.usage.total_tokens}`);
      }
    } catch (error) {
      console.log('   Error generating chat:', error);
      console.log('   Continuing with example...');
    }
    
    // -------------------------------------------------------------------------------
    // Example 5: Streaming chat responses
    // -------------------------------------------------------------------------------
    console.log('\n5. Streaming chat responses:');
    
    try {
      // Define messages for the streaming chat
      const messages: Message[] = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Write a short poem about artificial intelligence.' }
      ];
      
      // Request options (same format as regular chat)
      const options = {
        temperature: 0.8,
        max_tokens: 200
      };
      
      console.log('   System: You are a helpful assistant.');
      console.log('   User: Write a short poem about artificial intelligence.');
      console.log('   Assistant: ');
      
      // Get streaming response
      const streamingResponse = ollama.streamChat('llama3', messages, options);
      
      // Process each chunk as it arrives
      let fullResponse = '';
      process.stdout.write('   ');
      
      for await (const chunk of streamingResponse) {
        process.stdout.write(chunk.text);
        fullResponse += chunk.text;
        
        if (chunk.done) {
          console.log('\n   Streaming completed.');
        }
      }
      
      console.log('\n   Full response length:', fullResponse.length, 'characters');
    } catch (error) {
      console.log('   Error streaming chat:', error);
      console.log('   Continuing with example...');
    }
    
    // -------------------------------------------------------------------------------
    // Example 6: Generate text completion (simpler than chat)
    // -------------------------------------------------------------------------------
    console.log('\n6. Generate text completion:');
    
    try {
      // Simple prompt for text generation
      const prompt = 'List three benefits of exercise:';
      
      // Same options format as chat
      const options = {
        temperature: 0.5,
        max_tokens: 100
      };
      
      console.log('   Prompt:', prompt);
      
      const response = await ollama.generate('llama3', prompt, options);
      
      console.log('   Generated response:', response.text);
    } catch (error) {
      console.log('   Error generating text:', error);
      console.log('   Continuing with example...');
    }
    
    // -------------------------------------------------------------------------------
    // Example 7: Model compatibility check
    // -------------------------------------------------------------------------------
    console.log('\n7. Model compatibility check:');
    
    const modelsToCheck = [
      'llama3',
      'mistral',
      'llama2',
      'gemma',
      'phi3',
      'gpt4',
      'random-model',
      'mistral:latest'
    ];
    
    modelsToCheck.forEach(model => {
      const isCompatible = ollama.isCompatibleModel(model);
      console.log(`   ${model}: ${isCompatible ? 'Compatible' : 'Not compatible'}`);
    });
    
    // -------------------------------------------------------------------------------
    // Example 8: Usage statistics
    // -------------------------------------------------------------------------------
    console.log('\n8. Get usage statistics:');
    console.log('   Backend info:', ollama.getBackendInfo());
    console.log('   Resetting usage statistics...');
    ollama.resetUsageStats();
    
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