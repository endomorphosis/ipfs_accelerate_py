/**
 * OPEA (OpenAI Proxy API Extension) Backend Example
 * 
 * This example demonstrates how to use the OPEA backend to interact with OpenAI-compatible APIs,
 * including self-hosted solutions that implement the OpenAI API interface.
 */

import { OPEA } from '../src/api_backends/opea';

// For real usage, install from npm:
// import { OPEA } from 'ipfs-accelerate/api_backends';

/**
 * Basic usage with chat completions
 */
async function basicChatExample() {
  console.log('\n=== Basic Chat Example ===');
  
  // Create an OPEA instance with configuration
  const opea = new OPEA({}, {
    // URL of your OpenAI-compatible API
    opea_api_url: process.env.OPEA_API_URL || 'http://localhost:8000',
    
    // API key for your service (if required)
    opea_api_key: process.env.OPEA_API_KEY || 'your-api-key-here',
    
    // Default model to use
    opea_model: 'gpt-3.5-turbo'
  });
  
  try {
    // Define chat messages
    const messages = [
      { role: 'system', content: 'You are a helpful assistant specializing in JavaScript.' },
      { role: 'user', content: 'Write a simple function to calculate the factorial of a number.' }
    ];
    
    // Generate a response
    const response = await opea.chat(messages, {
      temperature: 0.7,
      max_tokens: 300
    });
    
    console.log('Response:', response.content);
    console.log('Model:', response.model);
    console.log('Usage:', response.usage);
  } catch (error) {
    console.error('Error:', error);
  }
}

/**
 * Streaming chat example
 */
async function streamingChatExample() {
  console.log('\n=== Streaming Chat Example ===');
  
  // Create an OPEA instance
  const opea = new OPEA({}, {
    opea_api_url: process.env.OPEA_API_URL || 'http://localhost:8000',
    opea_api_key: process.env.OPEA_API_KEY || 'your-api-key-here',
    opea_model: 'gpt-3.5-turbo'
  });
  
  try {
    // Define chat messages
    const messages = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Write a short poem about artificial intelligence.' }
    ];
    
    // Stream the response
    console.log('Streaming response:');
    const stream = opea.streamChat(messages, {
      temperature: 0.8,
      max_tokens: 200
    });
    
    // Process the stream
    let response = '';
    for await (const chunk of stream) {
      // Print each chunk as it arrives
      process.stdout.write(chunk.content);
      response += chunk.content;
      
      // Break if done
      if (chunk.done && chunk.type === 'complete') {
        break;
      }
    }
    
    console.log('\n\nFinal response length:', response.length);
  } catch (error) {
    console.error('Error:', error);
  }
}

/**
 * Custom endpoint example
 */
async function customEndpointExample() {
  console.log('\n=== Custom Endpoint Example ===');
  
  // Create an OPEA instance
  const opea = new OPEA({}, {
    opea_api_url: 'http://localhost:8000', // Base URL
    opea_api_key: 'your-api-key-here'
  });
  
  try {
    // Create a custom endpoint handler
    const customEndpoint = 'http://localhost:8000/v1/custom/endpoint';
    const handler = opea.createEndpointHandler(customEndpoint);
    
    // Use the handler directly
    const response = await handler({
      messages: [
        { role: 'user', content: 'What is the meaning of life?' }
      ],
      model: 'gpt-3.5-turbo',
      temperature: 0.7
    });
    
    console.log('Custom endpoint response:', response);
  } catch (error) {
    console.error('Error with custom endpoint:', error);
  }
}

/**
 * Environment variable and configuration example
 */
async function environmentConfigExample() {
  console.log('\n=== Environment Configuration Example ===');
  
  // This example shows how to use environment variables for configuration
  console.log('You can set these environment variables:');
  console.log('- OPEA_API_URL: URL of your OpenAI-compatible API');
  console.log('- OPEA_API_KEY: API key for your service');
  console.log('- OPEA_MODEL: Default model to use');
  console.log('- OPEA_TIMEOUT: Request timeout in seconds (default: 30)');
  
  // Create an OPEA instance that uses environment variables
  const opea = new OPEA();
  
  // Display the current configuration
  console.log('Current configuration:');
  console.log('- API URL:', (opea as any).apiUrl);
  console.log('- Default model:', (opea as any).model);
  console.log('- Timeout:', (opea as any).timeout, 'ms');
}

/**
 * Testing endpoint availability
 */
async function testEndpointExample() {
  console.log('\n=== Test Endpoint Example ===');
  
  // Create an OPEA instance
  const opea = new OPEA({}, {
    opea_api_url: process.env.OPEA_API_URL || 'http://localhost:8000',
    opea_api_key: process.env.OPEA_API_KEY || 'your-api-key-here'
  });
  
  // Test if the endpoint is available
  try {
    const isAvailable = await opea.testEndpoint();
    console.log('Endpoint available:', isAvailable);
    
    // Test a different endpoint
    const customEndpoint = 'http://localhost:8000/v1/completions';
    const isCustomAvailable = await opea.testEndpoint(customEndpoint);
    console.log('Custom endpoint available:', isCustomAvailable);
  } catch (error) {
    console.error('Error testing endpoint:', error);
  }
}

/**
 * Error handling example
 */
async function errorHandlingExample() {
  console.log('\n=== Error Handling Example ===');
  
  // Create an OPEA instance with an invalid URL
  const opea = new OPEA({}, {
    opea_api_url: 'http://invalid-url-that-does-not-exist.example',
    opea_api_key: 'invalid-key'
  });
  
  try {
    // This should fail
    const response = await opea.chat([
      { role: 'user', content: 'Hello' }
    ]);
    
    console.log('Response:', response);
  } catch (error: any) {
    console.error('Error caught successfully!');
    console.error('Error message:', error.message);
    
    // Check for specific error types
    if (error.isAuthError) {
      console.error('Authentication error - check your API key');
    } else if (error.isRateLimitError) {
      console.error('Rate limit exceeded - slow down requests');
    } else if (error.isTimeout) {
      console.error('Request timed out - check server availability');
    } else if (error.isTransientError) {
      console.error('Temporary server error - retry later');
    }
  }
}

/**
 * Model compatibility check example
 */
function modelCompatibilityExample() {
  console.log('\n=== Model Compatibility Example ===');
  
  const opea = new OPEA();
  
  // List of models to test compatibility
  const models = [
    'gpt-3.5-turbo',
    'gpt-4',
    'text-embedding-ada-002',
    'claude-2',
    'llama-7b',
    'mistral-7b-instruct',
    'custom-model-name'
  ];
  
  // Check each model for compatibility
  console.log('Model compatibility results:');
  models.forEach(model => {
    const isCompatible = opea.isCompatibleModel(model);
    console.log(`- ${model}: ${isCompatible ? 'Compatible' : 'Not compatible'}`);
  });
}

// Run all examples
async function main() {
  console.log('=== OPEA Backend Examples ===');
  console.log('NOTE: These examples are designed to show API usage.');
  console.log('They will not work without a valid OpenAI-compatible API endpoint.');
  
  // Comment out examples you don't want to run
  await basicChatExample().catch(console.error);
  await streamingChatExample().catch(console.error);
  await customEndpointExample().catch(console.error);
  await environmentConfigExample().catch(console.error);
  await testEndpointExample().catch(console.error);
  await errorHandlingExample().catch(console.error);
  modelCompatibilityExample();
}

// Run the examples
if (require.main === module) {
  main().catch(console.error);
}

export {
  basicChatExample,
  streamingChatExample,
  customEndpointExample,
  environmentConfigExample,
  testEndpointExample,
  errorHandlingExample,
  modelCompatibilityExample
};