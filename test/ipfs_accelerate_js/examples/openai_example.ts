/**
 * OpenAI API Backend Example
 * 
 * This example demonstrates how to use the OpenAI API backend for various tasks:
 * - Basic chat completions
 * - Streaming chat completions
 * - Function calling (tools)
 * - Embeddings generation
 * - Text moderation
 * - Image generation
 * - Text-to-speech generation
 * - Speech-to-text transcription
 */

import { OpenAI } from '../src/api_backends/openai';
import { Message } from '../src/api_backends/types';
import * as fs from 'fs';
import * as path from 'path';

// Create a new OpenAI instance
// Note: You should set OPENAI_API_KEY in your environment, or pass it in metadata
const openai = new OpenAI({}, {
  openai_api_key: process.env.OPENAI_API_KEY,
});

/**
 * Basic Chat Example
 */
async function basicChatExample() {
  console.log('Running basic chat example...');
  
  const messages: Message[] = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is the capital of France?' }
  ];
  
  try {
    const response = await openai.chat(messages);
    console.log('Response:', response.content);
    console.log('Model:', response.model);
    console.log('Tokens used:', response.usage.total_tokens);
  } catch (error) {
    console.error('Error in basic chat example:', error);
  }
}

/**
 * Chat with Custom Parameters Example
 */
async function chatWithParametersExample() {
  console.log('\nRunning chat with custom parameters example...');
  
  const messages: Message[] = [
    { role: 'system', content: 'You are a creative story writer.' },
    { role: 'user', content: 'Write the first sentence of a story about a mysterious island.' }
  ];
  
  try {
    const response = await openai.chat(messages, {
      model: 'gpt-4-turbo',
      temperature: 0.8,
      maxTokens: 50,
      topP: 0.9
    });
    
    console.log('Creative Response:', response.content);
    console.log('Model:', response.model);
  } catch (error) {
    console.error('Error in chat with parameters example:', error);
  }
}

/**
 * Streaming Chat Example
 */
async function streamingChatExample() {
  console.log('\nRunning streaming chat example...');
  
  const messages: Message[] = [
    { role: 'user', content: 'Explain quantum computing in simple terms.' }
  ];
  
  try {
    console.log('Streaming response:');
    const stream = openai.streamChat(messages);
    
    let fullResponse = '';
    
    for await (const chunk of stream) {
      process.stdout.write(chunk.content);
      fullResponse += chunk.content;
    }
    
    console.log('\n\nFull streamed response:', fullResponse);
  } catch (error) {
    console.error('Error in streaming chat example:', error);
  }
}

/**
 * Function Calling (Tools) Example
 */
async function functionCallingExample() {
  console.log('\nRunning function calling example...');
  
  const messages: Message[] = [
    { role: 'user', content: 'What\'s the weather like in New York?' }
  ];
  
  // Define tools (functions)
  const tools = [
    {
      type: 'function',
      function: {
        name: 'get_weather',
        description: 'Get the current weather in a location',
        parameters: {
          type: 'object',
          properties: {
            location: {
              type: 'string',
              description: 'The city and state, e.g. San Francisco, CA'
            },
            unit: {
              type: 'string',
              enum: ['celsius', 'fahrenheit'],
              description: 'The temperature unit to use'
            }
          },
          required: ['location']
        }
      }
    }
  ];
  
  try {
    const response = await openai.chat(messages, {
      tools,
      toolChoice: 'auto'
    });
    
    console.log('Response has tool calls:', !!response.tool_calls);
    
    if (response.tool_calls && response.tool_calls.length > 0) {
      const toolCall = response.tool_calls[0];
      console.log('Tool call function name:', toolCall.function.name);
      console.log('Tool call arguments:', toolCall.function.arguments);
      
      // Parse the arguments
      const args = JSON.parse(toolCall.function.arguments);
      console.log('Parsed location:', args.location);
      
      // In a real implementation, you would call your actual function here
      // and then send the result back to the API
      const weatherResult = `It's 72°F and sunny in ${args.location}`;
      
      // Add the tool call and result to the conversation
      const updatedMessages = [
        ...messages,
        {
          role: 'assistant',
          content: null,
          tool_calls: [toolCall]
        },
        {
          role: 'tool',
          tool_call_id: toolCall.id,
          content: weatherResult
        }
      ];
      
      // Get the final response with the function results incorporated
      const finalResponse = await openai.chat(updatedMessages);
      console.log('Final response after tool call:', finalResponse.content);
    }
  } catch (error) {
    console.error('Error in function calling example:', error);
  }
}

/**
 * Embeddings Example
 */
async function embeddingsExample() {
  console.log('\nRunning embeddings example...');
  
  try {
    // Get embedding for a single text
    const singleEmbedding = await openai.embedding('The quick brown fox jumps over the lazy dog.');
    console.log('Single embedding length:', singleEmbedding[0].length);
    console.log('First 5 values of embedding:', singleEmbedding[0].slice(0, 5));
    
    // Get embeddings for multiple texts
    const multipleEmbeddings = await openai.embedding([
      'The quick brown fox jumps over the lazy dog.',
      'Hello, world!'
    ]);
    
    console.log('Number of embeddings:', multipleEmbeddings.length);
    console.log('Second embedding length:', multipleEmbeddings[1].length);
  } catch (error) {
    console.error('Error in embeddings example:', error);
  }
}

/**
 * Moderation Example
 */
async function moderationExample() {
  console.log('\nRunning moderation example...');
  
  try {
    // Test with benign content
    const safeResult = await openai.moderation('The sky is blue and the weather is nice today.');
    console.log('Safe content flagged:', safeResult.results[0].flagged);
    
    // Test with potentially problematic content
    const unsafeResult = await openai.moderation('I want to harm someone.');
    console.log('Unsafe content flagged:', unsafeResult.results[0].flagged);
    
    if (unsafeResult.results[0].flagged) {
      console.log('Flagged categories:', Object.entries(unsafeResult.results[0].categories)
        .filter(([_, value]) => value)
        .map(([key, _]) => key)
        .join(', '));
    }
  } catch (error) {
    console.error('Error in moderation example:', error);
  }
}

/**
 * Image Generation Example
 */
async function imageGenerationExample() {
  console.log('\nRunning image generation example...');
  
  try {
    const imageResponse = await openai.textToImage('A serene mountain lake at sunset with reflections in the water');
    
    if (imageResponse.data && imageResponse.data.length > 0) {
      console.log('Image URL:', imageResponse.data[0].url);
      console.log('Revised prompt:', imageResponse.data[0].revised_prompt);
      
      // In a real application, you might download or display the image
      // For this example, we'll just show the URL
    }
  } catch (error) {
    console.error('Error in image generation example:', error);
  }
}

/**
 * Text to Speech Example
 */
async function textToSpeechExample() {
  console.log('\nRunning text to speech example...');
  
  try {
    const speechBuffer = await openai.textToSpeech(
      'Hello, this is a test of the OpenAI text to speech API.',
      'alloy'
    );
    
    // Save the audio file
    const outputPath = path.join(__dirname, 'openai_tts_output.mp3');
    fs.writeFileSync(outputPath, Buffer.from(speechBuffer));
    
    console.log('Speech generated and saved to:', outputPath);
  } catch (error) {
    console.error('Error in text to speech example:', error);
  }
}

/**
 * Speech to Text Example
 */
async function speechToTextExample() {
  console.log('\nRunning speech to text example...');
  
  try {
    // This example requires an audio file
    const audioPath = path.join(__dirname, 'sample_audio.mp3');
    
    // Check if the file exists
    if (!fs.existsSync(audioPath)) {
      console.log('Sample audio file not found. Skipping speech to text example.');
      return;
    }
    
    // Read the audio file
    const audioBlob = new Blob([fs.readFileSync(audioPath)]);
    
    // Transcribe the audio
    const transcription = await openai.speechToText(audioBlob, {
      language: 'en'
    });
    
    console.log('Transcription:', transcription.text);
  } catch (error) {
    console.error('Error in speech to text example:', error);
  }
}

/**
 * Error Handling Example
 */
async function errorHandlingExample() {
  console.log('\nRunning error handling example...');
  
  // Create a client with an invalid API key
  const invalidClient = new OpenAI({}, {
    openai_api_key: 'invalid-api-key'
  });
  
  try {
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    await invalidClient.chat(messages);
  } catch (error) {
    console.log('Caught error with invalid API key:');
    console.log('- Message:', error.message);
    console.log('- Status code:', error.statusCode);
    console.log('- Is auth error:', error.isAuthError);
  }
}

/**
 * Environment Variables Example
 */
function environmentVariablesExample() {
  console.log('\nEnvironment variables usage:');
  console.log('- Set OPENAI_API_KEY for automatic API key detection');
  console.log('- Example: export OPENAI_API_KEY=your-api-key');
  console.log('- Or use a .env file with a library like dotenv');
  
  // In a real application with dotenv:
  // require('dotenv').config();
  // const openai = new OpenAI(); // Will automatically use process.env.OPENAI_API_KEY
}

/**
 * Testing Endpoint Availability
 */
async function testEndpointExample() {
  console.log('\nTesting endpoint availability...');
  
  const isAvailable = await openai.testEndpoint();
  console.log('OpenAI API endpoint available:', isAvailable);
}

/**
 * Run all examples
 */
async function runExamples() {
  // Basic usage
  await basicChatExample();
  await chatWithParametersExample();
  await streamingChatExample();
  
  // Advanced features
  await functionCallingExample();
  await embeddingsExample();
  await moderationExample();
  await imageGenerationExample();
  
  // Audio features (may require file setup)
  if (process.env.RUN_AUDIO_EXAMPLES === 'true') {
    await textToSpeechExample();
    await speechToTextExample();
  } else {
    console.log('\nSkipping audio examples. Set RUN_AUDIO_EXAMPLES=true to run them.');
  }
  
  // Utility examples
  await errorHandlingExample();
  environmentVariablesExample();
  await testEndpointExample();
  
  console.log('\nAll examples completed.');
}

// Run examples if this file is executed directly
if (require.main === module) {
  runExamples().catch(error => {
    console.error('Error running examples:', error);
  });
}