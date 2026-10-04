/**
 * Example usage of the Gemini API Backend
 * 
 * This example demonstrates how to use the Gemini API backend for various tasks
 * including chat completions and streaming responses.
 */

import { Gemini } from '../src/api_backends/gemini/gemini';
import { ApiMetadata, ApiResources, Message } from '../src/api_backends/types';

// Environment variables can be used for configuration:
// process.env.GEMINI_API_KEY = 'your_api_key';

async function main() {
  console.log('Gemini API Backend Example');
  console.log('===========================\n');

  // 1. Initialize the Gemini backend
  console.log('1. Initializing Gemini backend...');
  
  const resources: ApiResources = {};
  const metadata: ApiMetadata = {
    gemini_api_key: 'demo_api_key'  // Replace with your actual Gemini API key
  };
  
  const gemini = new Gemini(resources, metadata);
  console.log('Gemini backend initialized with the following settings:');
  console.log(`  - API Key: ${metadata.gemini_api_key ? '****' + metadata.gemini_api_key.slice(-4) : 'Not provided'}`);
  console.log(`  - Default Model: ${gemini.getDefaultModel ? gemini.getDefaultModel() : 'gemini-pro'}`);
  
  // 2. Test the endpoint connection
  console.log('\n2. Testing endpoint connection...');
  try {
    const isConnected = await gemini.testEndpoint();
    console.log(`  Connection test: ${isConnected ? 'SUCCESS' : 'FAILED'}`);
    
    if (!isConnected) {
      console.log('  Unable to connect to Gemini API. Please check your API key.');
      return;
    }
  } catch (error) {
    console.error('  Error testing endpoint:', error);
    return;
  }
  
  // 3. Basic chat completion
  console.log('\n3. Running a basic chat completion...');
  try {
    const messages: Message[] = [
      { role: 'user', content: 'What are the key features of quantum computing?' }
    ];
    
    const response = await gemini.chat(messages, {
      model: 'gemini-pro',
      maxTokens: 200,
      temperature: 0.7
    });
    
    console.log('  User message: What are the key features of quantum computing?');
    console.log('  Gemini response:');
    console.log(`  ${response.content}`);
    
    if (response.usage) {
      console.log('  Usage statistics:');
      console.log(`  - Prompt tokens: ${response.usage.prompt_tokens}`);
      console.log(`  - Completion tokens: ${response.usage.completion_tokens}`);
      console.log(`  - Total tokens: ${response.usage.total_tokens}`);
    }
  } catch (error) {
    console.error('  Error in chat completion:', error);
  }
  
  // 4. Multi-turn conversation
  console.log('\n4. Multi-turn conversation...');
  try {
    const messages: Message[] = [
      { role: 'user', content: 'Tell me about renewable energy sources.' },
      { role: 'assistant', content: 'Renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass. These sources are sustainable and produce minimal greenhouse gas emissions...' },
      { role: 'user', content: 'Which renewable energy source has seen the most growth in recent years?' }
    ];
    
    const response = await gemini.chat(messages, {
      model: 'gemini-pro',
      maxTokens: 300,
      temperature: 0.5
    });
    
    console.log('  User: Tell me about renewable energy sources.');
    console.log('  Gemini: Renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass. These sources are sustainable and produce minimal greenhouse gas emissions...');
    console.log('  User: Which renewable energy source has seen the most growth in recent years?');
    console.log('  Gemini response:');
    console.log(`  ${response.content}`);
  } catch (error) {
    console.error('  Error in multi-turn conversation:', error);
  }
  
  // 5. Streaming chat completion (this requires an actual Gemini API key)
  console.log('\n5. Streaming chat completion example (simulated)...');
  try {
    /*
    // In a real application with a valid Gemini API key:
    const messages: Message[] = [
      { role: 'user', content: 'Explain machine learning algorithms in simple terms.' }
    ];
    
    console.log('  User message: Explain machine learning algorithms in simple terms.');
    console.log('  Gemini response (streaming):');
    
    let fullResponse = '';
    
    for await (const chunk of await gemini.streamChat(messages, { 
      model: 'gemini-pro',
      maxTokens: 200 
    })) {
      if (chunk.content) {
        process.stdout.write(chunk.content);
        fullResponse += chunk.content;
      }
    }
    
    console.log('\n  (Stream completed)');
    */
    
    // Since we're using a simulated environment, we'll just show the concept
    console.log('  In a real application, you would see the response appear incrementally as Gemini generates it.');
    console.log('  Simulated response:');
    console.log('  Machine learning algorithms are computer programs that can learn from data without being explicitly');
    console.log('  programmed. Think of them as recipes that help computers recognize patterns and make decisions.');
    console.log('  Common types include:');
    console.log('  1. Supervised learning: Learning from labeled examples (like a student learning with an answer key)');
    console.log('  2. Unsupervised learning: Finding patterns without labels (like grouping similar items)');
    console.log('  3. Reinforcement learning: Learning through trial and error with rewards (like training a dog)');
  } catch (error) {
    console.error('  Error in streaming chat completion:', error);
  }
  
  // 6. Error handling example
  console.log('\n6. Error handling example...');
  try {
    // Try to use a non-existent model
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    // Proper error handling
    try {
      const response = await gemini.chat(messages, {
        model: 'non-existent-model',
        maxTokens: 10
      });
      console.log('  Response:', response);
    } catch (error) {
      console.log('  Successfully caught error:');
      console.log(`  ${error}`);
    }
  } catch (error) {
    console.error('  Error in error handling example:', error);
  }
  
  // 7. Model compatibility checking
  console.log('\n7. Checking model compatibility...');
  const modelsToCheck = [
    'gemini-pro',
    'gemini-pro-vision',
    'gemini-1.5-pro',
    'gemini-1.5-flash',
    'palm-2',
    'gpt-4',
    'llama-2-70b'
  ];
  
  modelsToCheck.forEach(model => {
    const isCompatible = gemini.isCompatibleModel(model);
    console.log(`  - ${model}: ${isCompatible ? 'COMPATIBLE' : 'NOT COMPATIBLE'}`);
  });
  
  console.log('\nExample completed successfully!');
}

// Run the example
main().catch(error => {
  console.error('Example failed with error:', error);
});