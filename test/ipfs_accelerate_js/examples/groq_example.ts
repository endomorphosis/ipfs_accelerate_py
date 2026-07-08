/**
 * Groq API Backend Example
 * 
 * This example demonstrates how to use the Groq backend to interact with the Groq API
 * for LLM-powered chat completions using models like Llama and Mixtral.
 */

import { Groq } from '../src/api_backends/groq';

// For real usage, install from npm:
// import { Groq } from 'ipfs-accelerate/api_backends';

/**
 * Basic chat completion example
 */
async function basicChatExample() {
  console.log('\n=== Basic Chat Example ===');
  
  // Create a Groq instance with configuration
  const groq = new Groq({}, {
    // API key from environment variable or directly provided
    groq_api_key: process.env.GROQ_API_KEY || 'your-api-key-here',
  });
  
  try {
    // Define chat messages
    const messages = [
      { role: 'system', content: 'You are a helpful assistant with expertise in quantum physics.' },
      { role: 'user', content: 'Explain quantum entanglement in simple terms.' }
    ];
    
    // Generate a response with Llama 3 70B model
    const response = await groq.chat(messages, {
      model: 'llama3-70b-8192',
      temperature: 0.7,
      maxTokens: 200
    });
    
    console.log('Response:', response.content);
    console.log('Model:', response.model);
    console.log('Token usage:', response.usage);
  } catch (error) {
    console.error('Error:', error);
  }
}

/**
 * Streaming chat completion example
 */
async function streamingChatExample() {
  console.log('\n=== Streaming Chat Example ===');
  
  // Create a Groq instance
  const groq = new Groq({}, {
    groq_api_key: process.env.GROQ_API_KEY || 'your-api-key-here',
  });
  
  try {
    // Define chat messages
    const messages = [
      { role: 'system', content: 'You are a creative writing assistant.' },
      { role: 'user', content: 'Write a short poem about artificial intelligence.' }
    ];
    
    // Stream the response using Mixtral model
    console.log('Streaming response:');
    const stream = groq.streamChat(messages, {
      model: 'mixtral-8x7b-32768',
      temperature: 0.9,
      maxTokens: 150
    });
    
    // Process the stream
    let fullResponse = '';
    
    for await (const chunk of stream) {
      // Print each chunk as it arrives
      process.stdout.write(chunk.content);
      fullResponse += chunk.content;
    }
    
    console.log('\n\nFull response length:', fullResponse.length);
  } catch (error) {
    console.error('Error:', error);
  }
}

/**
 * Model selection example
 */
async function modelSelectionExample() {
  console.log('\n=== Model Selection Example ===');
  
  // Create a Groq instance
  const groq = new Groq({}, {
    groq_api_key: process.env.GROQ_API_KEY || 'your-api-key-here',
  });
  
  try {
    const query = 'What are the advantages and disadvantages of different energy sources?';
    const messages = [{ role: 'user', content: query }];
    
    // List of models to try
    const models = [
      { name: 'llama3-8b-8192', description: 'Llama 3 8B (smaller, faster)' },
      { name: 'llama3-70b-8192', description: 'Llama 3 70B (larger, more capable)' },
      { name: 'mixtral-8x7b-32768', description: 'Mixtral 8x7B (mixture of experts)' },
      { name: 'gemma2-9b-it', description: 'Gemma 2 9B (Google model)' }
    ];
    
    // Print available models
    console.log('Available models:');
    models.forEach((model, i) => {
      console.log(`${i+1}. ${model.name}: ${model.description}`);
    });
    
    // Example of using the Llama 3 8B model (faster, but less capable)
    console.log(`\nUsing the ${models[0].name} model:`);
    const response1 = await groq.chat(messages, {
      model: models[0].name,
      temperature: 0.7,
      maxTokens: 100
    });
    
    console.log(`Response: "${response1.content.substring(0, 100)}..."`);
    
    // Example of using the Llama 3 70B model (slower, but more capable)
    console.log(`\nUsing the ${models[1].name} model:`);
    const response2 = await groq.chat(messages, {
      model: models[1].name,
      temperature: 0.7,
      maxTokens: 100
    });
    
    console.log(`Response: "${response2.content.substring(0, 100)}..."`);
  } catch (error) {
    console.error('Error:', error);
  }
}

/**
 * Advanced parameters example
 */
async function advancedParametersExample() {
  console.log('\n=== Advanced Parameters Example ===');
  
  // Create a Groq instance
  const groq = new Groq({}, {
    groq_api_key: process.env.GROQ_API_KEY || 'your-api-key-here',
  });
  
  try {
    const messages = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'What is the capital of France?' }
    ];
    
    // Different temperature settings
    console.log('Testing different temperature settings:');
    
    console.log('\nTemperature = 0.0 (deterministic):');
    const response1 = await groq.chat(messages, {
      model: 'llama3-8b-8192',
      temperature: 0.0,
      maxTokens: 30
    });
    console.log(`Response: "${response1.content}"`);
    
    console.log('\nTemperature = 0.7 (balanced):');
    const response2 = await groq.chat(messages, {
      model: 'llama3-8b-8192',
      temperature: 0.7,
      maxTokens: 30
    });
    console.log(`Response: "${response2.content}"`);
    
    console.log('\nTemperature = 1.2 (more creative):');
    const response3 = await groq.chat(messages, {
      model: 'llama3-8b-8192',
      temperature: 1.2,
      maxTokens: 30
    });
    console.log(`Response: "${response3.content}"`);
    
    // Testing top_p
    console.log('\nTesting top_p = 0.5 (more focused):');
    const response4 = await groq.chat(messages, {
      model: 'llama3-8b-8192',
      temperature: 0.7,
      topP: 0.5,
      maxTokens: 30
    });
    console.log(`Response: "${response4.content}"`);
  } catch (error) {
    console.error('Error:', error);
  }
}

/**
 * Error handling example
 */
async function errorHandlingExample() {
  console.log('\n=== Error Handling Example ===');
  
  try {
    // Create a Groq instance with an invalid API key
    const groq = new Groq({}, {
      groq_api_key: 'invalid-api-key'
    });
    
    console.log('Attempting to chat with invalid API key...');
    
    await groq.chat([
      { role: 'user', content: 'Hello' }
    ]);
  } catch (error: any) {
    console.error('Error caught successfully!');
    console.error('Error message:', error.message);
    
    // Check error properties
    if (error.status === 401) {
      console.error('Authentication error - invalid API key');
    } else if (error.status === 429) {
      console.error('Rate limit exceeded - too many requests');
    } else if (error.type === 'timeout_error') {
      console.error('Request timed out - check your internet connection');
    } else {
      console.error('Other error detected, status:', error.status);
    }
  }
  
  // Testing timeout handling
  try {
    console.log('\nTesting timeout handling...');
    
    const groq = new Groq({}, {
      groq_api_key: process.env.GROQ_API_KEY || 'your-api-key'
    });
    
    // Mock a timeout by setting timeout to 1ms (unrealistic)
    await groq.chat(
      [{ role: 'user', content: 'Hello' }],
      { timeout: 1 }
    );
  } catch (error: any) {
    console.error('Timeout error caught successfully!');
    console.error('Error message:', error.message);
    
    if (error.type === 'timeout_error') {
      console.error('Request timed out as expected');
    }
  }
}

/**
 * Environment variables and configuration
 */
function environmentConfigExample() {
  console.log('\n=== Environment Configuration Example ===');
  
  console.log('To use Groq API with environment variables:');
  console.log('1. Set GROQ_API_KEY environment variable with your API key');
  console.log('2. Create a Groq instance without explicit API key');
  
  console.log('\nExample:');
  console.log('export GROQ_API_KEY=your-api-key-here');
  console.log('const groq = new Groq();');
  
  // Create a Groq instance that uses environment variables
  const groq = new Groq();
  
  console.log('\nDefault configuration:');
  console.log('- Default model:', (groq as any).getDefaultModel());
}

// Run all examples
async function main() {
  console.log('=== Groq API Backend Examples ===');
  console.log('NOTE: These examples require a valid Groq API key.');
  console.log('To get a key, sign up at https://console.groq.com/\n');
  
  if (!process.env.GROQ_API_KEY) {
    console.log('GROQ_API_KEY environment variable not set.');
    console.log('Examples will run but may fail with authentication errors.');
    console.log('You can provide your API key by running:');
    console.log('  export GROQ_API_KEY=your-api-key-here\n');
  }
  
  // Comment out examples you don't want to run
  await basicChatExample().catch(console.error);
  await streamingChatExample().catch(console.error);
  await modelSelectionExample().catch(console.error);
  await advancedParametersExample().catch(console.error);
  await errorHandlingExample().catch(console.error);
  environmentConfigExample();
}

// Run the examples
if (require.main === module) {
  main().catch(console.error);
}

export {
  basicChatExample,
  streamingChatExample,
  modelSelectionExample,
  advancedParametersExample,
  errorHandlingExample,
  environmentConfigExample
};