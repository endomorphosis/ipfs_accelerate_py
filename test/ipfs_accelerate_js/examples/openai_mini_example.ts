/**
 * OpenAI Mini API Backend Example
 * 
 * This is a lightweight version of the OpenAI API client with core functionality.
 * It's designed to be more efficient and lightweight than the full OpenAI client,
 * while still supporting all essential features.
 */

import { OpenAiMini } from '../src/api_backends/openai_mini';
import { ChatMessage } from '../src/api_backends/types';
import * as fs from 'fs';
import * as path from 'path';

// Create a new OpenAI Mini instance
// Note: You should set OPENAI_API_KEY in your environment, or pass it in metadata
const client = new OpenAiMini({
  // Optional configuration
  maxRetries: 3,
  requestTimeout: 30000,
  useRequestQueue: true,
  debug: false
}, {
  openai_mini_api_key: process.env.OPENAI_API_KEY,
});

/**
 * Basic Chat Example
 */
async function basicChatExample() {
  console.log('Running basic chat example...');
  
  const messages: ChatMessage[] = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is the capital of Japan?' }
  ];
  
  try {
    const response = await client.chat(messages);
    console.log('Response:', response.content);
    console.log('Response metadata:', response.meta);
  } catch (error) {
    console.error('Error in basic chat example:', error);
  }
}

/**
 * Streaming Chat Example
 */
async function streamingChatExample() {
  console.log('\nRunning streaming chat example...');
  
  const messages: ChatMessage[] = [
    { role: 'user', content: 'List the top 3 planets in our solar system by size.' }
  ];
  
  try {
    console.log('Streaming response:');
    const stream = client.streamChat(messages);
    
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
 * Chat with Custom Parameters Example
 */
async function customParametersExample() {
  console.log('\nRunning chat with custom parameters example...');
  
  const messages: ChatMessage[] = [
    { role: 'system', content: 'You are a creative poet.' },
    { role: 'user', content: 'Write a haiku about programming.' }
  ];
  
  try {
    const response = await client.chat(messages, {
      model: 'gpt-4',
      temperature: 0.8,
      top_p: 0.9,
      max_tokens: 50,
      stop: ['END'],
      priority: 'HIGH'
    });
    
    console.log('Response:', response.content);
  } catch (error) {
    console.error('Error in custom parameters example:', error);
  }
}

/**
 * File Upload Example
 */
async function fileUploadExample() {
  console.log('\nRunning file upload example...');
  
  try {
    // Check if the file exists
    const filePath = path.join(__dirname, 'example_data.jsonl');
    if (!fs.existsSync(filePath)) {
      console.log('Example file not found. Creating a sample file for upload...');
      
      // Create a simple JSONL file
      const sampleData = [
        '{"prompt": "What is AI?", "completion": "Artificial Intelligence is the simulation of human intelligence by machines."}',
        '{"prompt": "What is machine learning?", "completion": "Machine learning is a subset of AI that enables systems to learn and improve from experience."}'
      ].join('\n');
      
      fs.writeFileSync(filePath, sampleData);
      console.log('Created sample file at:', filePath);
    }
    
    // Upload the file
    const fileResult = await client.uploadFile(filePath, {
      purpose: 'fine-tune',
      fileName: 'custom_filename.jsonl'
    });
    
    console.log('File uploaded successfully:');
    console.log('- File ID:', fileResult.id);
    console.log('- Filename:', fileResult.filename);
    console.log('- Purpose:', fileResult.purpose);
    console.log('- Size:', fileResult.bytes, 'bytes');
    console.log('- Status:', fileResult.status);
  } catch (error) {
    console.error('Error in file upload example:', error);
  }
}

/**
 * Text-to-Speech Example
 */
async function textToSpeechExample() {
  console.log('\nRunning text-to-speech example...');
  
  try {
    const text = 'Hello world! This is a test of the text to speech API using OpenAI Mini.';
    
    const audioBuffer = await client.textToSpeech(text, {
      model: 'tts-1',
      voice: 'alloy',
      speed: 1.0,
      response_format: 'mp3'
    });
    
    // Save the audio to a file
    const outputPath = path.join(__dirname, 'openai_mini_tts_output.mp3');
    fs.writeFileSync(outputPath, audioBuffer);
    
    console.log('Speech generated and saved to:', outputPath);
  } catch (error) {
    console.error('Error in text-to-speech example:', error);
  }
}

/**
 * Speech-to-Text Transcription Example
 */
async function speechToTextExample() {
  console.log('\nRunning speech-to-text example...');
  
  try {
    // Path to an audio file (should be .mp3, .mp4, .mpeg, .mpga, .m4a, .wav, or .webm format)
    const audioPath = path.join(__dirname, 'sample_audio.mp3');
    
    // Check if the file exists
    if (!fs.existsSync(audioPath)) {
      console.log('Sample audio file not found. Skipping transcription example.');
      return;
    }
    
    const transcription = await client.transcribeAudio(audioPath, {
      model: 'whisper-1',
      language: 'en',
      prompt: 'This is a clear recording of someone speaking.',
      response_format: 'text',
      temperature: 0.2
    });
    
    console.log('Transcription:', transcription);
  } catch (error) {
    console.error('Error in speech-to-text example:', error);
  }
}

/**
 * Image Generation Example
 */
async function imageGenerationExample() {
  console.log('\nRunning image generation example...');
  
  try {
    const prompt = 'A futuristic city with flying cars and tall skyscrapers at sunset';
    
    const imageResult = await client.generateImage(prompt, {
      model: 'dall-e-3',
      size: '1024x1024',
      quality: 'standard',
      style: 'vivid',
      n: 1
    });
    
    if (imageResult.data && imageResult.data.length > 0) {
      console.log('Generated image URL:', imageResult.data[0].url);
      console.log('Revised prompt:', imageResult.data[0].revised_prompt);
    }
  } catch (error) {
    console.error('Error in image generation example:', error);
  }
}

/**
 * Error Handling Example
 */
async function errorHandlingExample() {
  console.log('\nRunning error handling example...');
  
  // Create a client with an invalid API key
  const invalidClient = new OpenAiMini({}, {
    openai_mini_api_key: 'invalid-api-key'
  });
  
  try {
    await invalidClient.chat([{ role: 'user', content: 'Hello' }]);
  } catch (error) {
    console.log('Caught API error:');
    console.log('- Error Message:', error.message);
    console.log('- Error contains authentication details:', error.message.includes('authentication'));
  }
}

/**
 * API Endpoint Testing Example
 */
async function testEndpointExample() {
  console.log('\nTesting API endpoint availability...');
  
  const isAvailable = await client.testEndpoint();
  console.log('OpenAI API endpoint available:', isAvailable);
  
  if (!isAvailable) {
    console.log('API endpoint is not available. Check your API key and internet connection.');
  }
}

/**
 * Environment Variables Example
 */
function environmentVariablesExample() {
  console.log('\nEnvironment variables configuration:');
  console.log('- Set OPENAI_API_KEY for automatic API key detection');
  console.log('- Example: export OPENAI_API_KEY=your-api-key');
  console.log('- Or use a .env file with a library like dotenv');
  
  // In a real application with dotenv:
  // require('dotenv').config();
  // const client = new OpenAiMini(); // Will automatically use process.env.OPENAI_API_KEY
}

/**
 * Debug Mode Example
 */
function debugModeExample() {
  console.log('\nDebug mode configuration:');
  console.log('Create a client with debug mode enabled:');
  console.log('const client = new OpenAiMini({ debug: true });');
  console.log('This will output detailed logs for API requests and responses');
}

/**
 * Performance Configuration Example
 */
function performanceConfigExample() {
  console.log('\nPerformance configuration options:');
  console.log('const client = new OpenAiMini({');
  console.log('  maxRetries: 3,            // Number of retries for failed requests');
  console.log('  requestTimeout: 30000,    // Request timeout in milliseconds');
  console.log('  useRequestQueue: true,    // Queue and rate limit requests');
  console.log('});');
}

/**
 * Run all examples
 */
async function runExamples() {
  // Basic examples
  await basicChatExample();
  await streamingChatExample();
  await customParametersExample();
  
  // Advanced features
  if (process.env.RUN_ADVANCED_EXAMPLES === 'true') {
    await fileUploadExample();
    await textToSpeechExample();
    await speechToTextExample();
    await imageGenerationExample();
  } else {
    console.log('\nSkipping advanced examples that require API calls.');
    console.log('Set RUN_ADVANCED_EXAMPLES=true to run them.');
  }
  
  // Utility examples
  await errorHandlingExample();
  await testEndpointExample();
  environmentVariablesExample();
  debugModeExample();
  performanceConfigExample();
  
  console.log('\nAll examples completed.');
}

// Run examples if this file is executed directly
if (require.main === module) {
  runExamples().catch(error => {
    console.error('Error running examples:', error);
  });
}