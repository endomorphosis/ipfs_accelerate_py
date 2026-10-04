/**
 * Example usage of the Claude (Anthropic) API Backend
 * 
 * This example demonstrates how to use the Claude API backend for various tasks
 * including chat completions and streaming responses.
 */

import { Claude } from '../src/api_backends/claude/claude';
import { ApiMetadata, ApiResources, Message } from '../src/api_backends/types';

// Environment variables can be used for configuration:
// process.env.ANTHROPIC_API_KEY = 'your_api_key';

async function main() {
  console.log('Claude (Anthropic) API Backend Example');
  console.log('=====================================\n');

  // 1. Initialize the Claude backend
  console.log('1. Initializing Claude backend...');
  
  const resources: ApiResources = {};
  const metadata: ApiMetadata = {
    claude_api_key: 'demo_api_key'  // Replace with your actual Claude API key
  };
  
  const claude = new Claude(resources, metadata);
  console.log('Claude backend initialized with the following settings:');
  console.log(`  - API Key: ${metadata.claude_api_key ? '****' + metadata.claude_api_key.slice(-4) : 'Not provided'}`);
  console.log(`  - Default Model: ${claude.getDefaultModel ? claude.getDefaultModel() : 'claude-3-haiku-20240307'}`);
  
  // 2. Test the endpoint connection
  console.log('\n2. Testing endpoint connection...');
  try {
    const isConnected = await claude.testEndpoint();
    console.log(`  Connection test: ${isConnected ? 'SUCCESS' : 'FAILED'}`);
    
    if (!isConnected) {
      console.log('  Unable to connect to Claude API. Please check your API key.');
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
      { role: 'user', content: 'What is quantum computing in simple terms?' }
    ];
    
    const response = await claude.chat(messages, {
      model: 'claude-3-haiku-20240307',
      max_tokens: 200,
      temperature: 0.7
    });
    
    console.log('  User message: What is quantum computing in simple terms?');
    console.log('  Claude response:');
    
    if (Array.isArray(response.content)) {
      // Content could be an array of content blocks in newer Claude API versions
      for (const block of response.content) {
        if (block.type === 'text') {
          console.log(`  ${block.text}`);
        }
      }
    } else {
      console.log(`  ${response.content}`);
    }
    
    if (response.usage) {
      console.log('  Usage statistics:');
      console.log(`  - Input tokens: ${response.usage.inputTokens}`);
      console.log(`  - Output tokens: ${response.usage.outputTokens}`);
    }
  } catch (error) {
    console.error('  Error in chat completion:', error);
  }
  
  // 4. Chat completion with system message
  console.log('\n4. Chat completion with system message...');
  try {
    const messages: Message[] = [
      { role: 'user', content: 'Write a short poem about artificial intelligence.' }
    ];
    
    const response = await claude.chat(messages, {
      model: 'claude-3-haiku-20240307',
      system: 'You are a brilliant poet who specializes in creating concise and thoughtful poetry.',
      max_tokens: 200,
      temperature: 0.9
    });
    
    console.log('  User message: Write a short poem about artificial intelligence.');
    console.log('  System prompt: You are a brilliant poet who specializes in creating concise and thoughtful poetry.');
    console.log('  Claude response:');
    
    if (Array.isArray(response.content)) {
      // Content could be an array of content blocks in newer Claude API versions
      for (const block of response.content) {
        if (block.type === 'text') {
          console.log(`  ${block.text}`);
        }
      }
    } else {
      console.log(`  ${response.content}`);
    }
  } catch (error) {
    console.error('  Error in chat completion with system message:', error);
  }
  
  // 5. Multi-turn conversation
  console.log('\n5. Multi-turn conversation...');
  try {
    const messages: Message[] = [
      { role: 'user', content: 'Hello, I want to learn about renewable energy.' },
      { role: 'assistant', content: 'Hello! I\'d be happy to help you learn about renewable energy. What specific aspect of renewable energy are you interested in exploring?' },
      { role: 'user', content: 'What are the main types of renewable energy?' }
    ];
    
    const response = await claude.chat(messages, {
      model: 'claude-3-haiku-20240307',
      max_tokens: 300
    });
    
    console.log('  User: Hello, I want to learn about renewable energy.');
    console.log('  Claude: Hello! I\'d be happy to help you learn about renewable energy. What specific aspect of renewable energy are you interested in exploring?');
    console.log('  User: What are the main types of renewable energy?');
    console.log('  Claude response:');
    
    if (Array.isArray(response.content)) {
      // Content could be an array of content blocks in newer Claude API versions
      for (const block of response.content) {
        if (block.type === 'text') {
          console.log(`  ${block.text}`);
        }
      }
    } else {
      console.log(`  ${response.content}`);
    }
  } catch (error) {
    console.error('  Error in multi-turn conversation:', error);
  }
  
  // 6. Streaming chat completion (this requires an actual Claude API key)
  console.log('\n6. Streaming chat completion example (simulated)...');
  try {
    /*
    // In a real application with a valid Claude API key:
    const messages: Message[] = [
      { role: 'user', content: 'Explain the process of photosynthesis briefly.' }
    ];
    
    console.log('  User message: Explain the process of photosynthesis briefly.');
    console.log('  Claude response (streaming):');
    
    let fullResponse = '';
    
    for await (const chunk of await claude.streamChat(messages, { 
      model: 'claude-3-haiku-20240307',
      max_tokens: 200 
    })) {
      if (chunk.content) {
        process.stdout.write(chunk.content);
        fullResponse += chunk.content;
      }
      
      if (chunk.type === 'stop') {
        console.log('\n  (Stream completed)');
        break;
      }
    }
    */
    
    // Since we're using a simulated environment, we'll just show the concept
    console.log('  In a real application, you would see the response appear incrementally as Claude generates it.');
    console.log('  Simulated response:');
    console.log('  Photosynthesis is the process where green plants, algae, and some bacteria convert sunlight');
    console.log('  into chemical energy. This process involves capturing light energy using chlorophyll,');
    console.log('  a green pigment found in chloroplasts. The light energy is used to combine carbon dioxide');
    console.log('  from the air and water from the soil to produce glucose (sugar) and oxygen.');
    console.log('  The simplified equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂');
  } catch (error) {
    console.error('  Error in streaming chat completion:', error);
  }
  
  // 7. Error handling example
  console.log('\n7. Error handling example...');
  try {
    // Try to use a model that doesn't exist
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    // Proper error handling
    try {
      const response = await claude.chat(messages, {
        model: 'non-existent-model',
        max_tokens: 10
      });
      console.log('  Response:', response);
    } catch (error) {
      console.log('  Successfully caught error:');
      console.log(`  ${error}`);
    }
  } catch (error) {
    console.error('  Error in error handling example:', error);
  }
  
  // 8. Model compatibility checking
  console.log('\n8. Checking model compatibility...');
  const modelsToCheck = [
    'claude-3-haiku-20240307',
    'claude-3-opus-20240229',
    'claude-2.1',
    'anthropic-claude-instant-1',
    'gpt-4',
    'llama-2-70b'
  ];
  
  modelsToCheck.forEach(model => {
    const isCompatible = claude.isCompatibleModel(model);
    console.log(`  - ${model}: ${isCompatible ? 'COMPATIBLE' : 'NOT COMPATIBLE'}`);
  });
  
  console.log('\nExample completed successfully!');
}

// Run the example
main().catch(error => {
  console.error('Example failed with error:', error);
});