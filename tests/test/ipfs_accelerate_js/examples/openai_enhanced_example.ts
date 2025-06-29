/**
 * Enhanced OpenAI API Features Example
 * 
 * This example demonstrates the new features added to the OpenAI API backend:
 * - GPT-4o-mini model support
 * - Enhanced voice agent capabilities
 * - Improved metrics collection
 * - Audio transcription with timestamps
 * - Parallel function calling
 */

import { OpenAI } from '../src/api_backends/openai';
import { OpenAIVoiceType, OpenAIAudioFormat, OpenAITranscriptionFormat } from '../src/api_backends/openai/types';
import { Message } from '../src/api_backends/types';
import * as fs from 'fs';
import * as path from 'path';

// Create a new OpenAI instance
// Note: You should set OPENAI_API_KEY in your environment, or pass it in metadata
const openai = new OpenAI({}, {
  openai_api_key: process.env.OPENAI_API_KEY,
});

/**
 * GPT-4o-mini Example
 */
async function gpt4oMiniExample() {
  console.log('Running GPT-4o-mini example...');
  
  const messages: Message[] = [
    { role: 'system', content: 'You are a helpful assistant. Be concise.' },
    { role: 'user', content: 'Explain what makes GPT-4o-mini unique compared to other models?' }
  ];
  
  try {
    const response = await openai.chat(messages, {
      model: 'gpt-4o-mini',
      temperature: 0.7
    });
    
    console.log('Response:', response.content);
    console.log('Model:', response.model);
    console.log('Tokens used:', response.usage.total_tokens);
  } catch (error) {
    console.error('Error in GPT-4o-mini example:', error);
  }
}

/**
 * Model Information Example
 */
async function modelInformationExample() {
  console.log('\nRunning model information example...');
  
  console.log('GPT-4o max tokens:', openai.getModelMaxTokens('gpt-4o'));
  console.log('GPT-4o-mini max tokens:', openai.getModelMaxTokens('gpt-4o-mini'));
  console.log('GPT-3.5-turbo max tokens:', openai.getModelMaxTokens('gpt-3.5-turbo'));
  
  console.log('Does GPT-4o support vision?', openai.supportsVision('gpt-4o'));
  console.log('Does GPT-4o-mini support vision?', openai.supportsVision('gpt-4o-mini'));
}

/**
 * Enhanced Voice Types Example
 */
async function voiceTypesExample() {
  console.log('\nRunning voice types example...');
  
  const text = "Hello, I'm demonstrating the different voice types available in the OpenAI text-to-speech API.";
  
  // Loop through all voice types
  for (const voice of Object.values(OpenAIVoiceType)) {
    try {
      console.log(`Generating speech with voice: ${voice}`);
      
      const speechBuffer = await openai.textToSpeech(
        text,
        voice,
        {
          model: 'tts-1',
          responseFormat: OpenAIAudioFormat.MP3
        }
      );
      
      // Save the audio file
      const outputPath = path.join(__dirname, `openai_voice_${voice}.mp3`);
      fs.writeFileSync(outputPath, Buffer.from(speechBuffer));
      
      console.log(`Speech with voice ${voice} saved to:`, outputPath);
    } catch (error) {
      console.error(`Error generating speech with voice ${voice}:`, error);
    }
  }
}

/**
 * Voice Agent Example
 */
async function voiceAgentExample() {
  console.log('\nRunning voice agent example...');
  
  // Create a voice agent
  const voiceAgent = openai.createVoiceAgent(
    "You are a helpful assistant with expertise in technology. Be concise and informative.",
    {
      voice: OpenAIVoiceType.NOVA,
      model: 'tts-1-hd',
      speed: 1.1
    },
    {
      chatModel: 'gpt-4o-mini',
      temperature: 0.7
    }
  );
  
  try {
    // Process a text query
    console.log('Processing text query...');
    const result = await voiceAgent.processText(
      "What are the key differences between TypeScript and JavaScript?",
      { speed: 1.2 }
    );
    
    // Save the audio response
    const outputPath = path.join(__dirname, 'voice_agent_response.mp3');
    fs.writeFileSync(outputPath, Buffer.from(result.audioResponse));
    
    console.log('Text response:', result.textResponse);
    console.log('Audio response saved to:', outputPath);
    
    // Get conversation history
    console.log('Conversation messages:', voiceAgent.getMessages().length);
    
    // Process a follow-up query
    console.log('\nProcessing follow-up query...');
    const followUpResult = await voiceAgent.processText(
      "What are some key TypeScript features developers should learn first?"
    );
    
    // Save the follow-up audio response
    const followUpPath = path.join(__dirname, 'voice_agent_followup.mp3');
    fs.writeFileSync(followUpPath, Buffer.from(followUpResult.audioResponse));
    
    console.log('Follow-up response:', followUpResult.textResponse);
    console.log('Follow-up audio saved to:', followUpPath);
    console.log('Updated conversation messages:', voiceAgent.getMessages().length);
    
    // Reset the agent
    voiceAgent.reset("You are a coding teacher focusing on best practices.");
    console.log('Agent reset. New conversation messages:', voiceAgent.getMessages().length);
  } catch (error) {
    console.error('Error in voice agent example:', error);
  }
}

/**
 * Speech to Text with Timestamps Example
 */
async function speechToTextWithTimestampsExample() {
  console.log('\nRunning speech to text with timestamps example...');
  
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
    
    // Transcribe the audio with word-level timestamps
    const transcription = await openai.speechToText(audioBlob, {
      model: 'whisper-1',
      responseFormat: OpenAITranscriptionFormat.VERBOSE_JSON,
      timestamp_granularities: ['word']
    });
    
    console.log('Transcription text:', transcription.text);
    
    if (transcription.words && transcription.words.length > 0) {
      console.log('\nWord timestamps:');
      transcription.words.slice(0, 5).forEach(word => {
        console.log(`- "${word.word}": ${word.start}s to ${word.end}s (probability: ${word.probability.toFixed(2)})`);
      });
      console.log(`... and ${transcription.words.length - 5} more words`);
    }
  } catch (error) {
    console.error('Error in speech to text with timestamps example:', error);
  }
}

/**
 * Audio Translation Example
 */
async function audioTranslationExample() {
  console.log('\nRunning audio translation example...');
  
  try {
    // This example requires an audio file in a non-English language
    const audioPath = path.join(__dirname, 'non_english_sample.mp3');
    
    // Check if the file exists
    if (!fs.existsSync(audioPath)) {
      console.log('Non-English audio file not found. Skipping translation example.');
      return;
    }
    
    // Read the audio file
    const audioBlob = new Blob([fs.readFileSync(audioPath)]);
    
    // Translate the audio to English
    const translation = await openai.translateAudio(audioBlob, {
      model: 'whisper-1'
    });
    
    console.log('Translation:', translation.text);
  } catch (error) {
    console.error('Error in audio translation example:', error);
  }
}

/**
 * Parallel Function Calling Example
 */
async function parallelFunctionCallingExample() {
  console.log('\nRunning parallel function calling example...');
  
  // Define the messages
  const messages: Message[] = [
    { role: 'user', content: 'What\'s the weather and news headline for New York City and San Francisco?' }
  ];
  
  // Define the functions (these would make actual API calls in a real application)
  const functions = {
    get_weather: async (args: { location: string }) => {
      console.log(`Getting weather for ${args.location}...`);
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 500));
      // Return mock weather data
      return {
        location: args.location,
        temperature: Math.round(70 + Math.random() * 20),
        conditions: ['sunny', 'partly cloudy', 'cloudy', 'rainy'][Math.floor(Math.random() * 4)]
      };
    },
    get_news: async (args: { location: string, category?: string }) => {
      console.log(`Getting news for ${args.location} (category: ${args.category || 'general'})...`);
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 800));
      // Return mock news data
      return {
        location: args.location,
        headline: `Major development announced in ${args.location}`,
        source: 'News API',
        category: args.category || 'general'
      };
    }
  };
  
  try {
    // Call the combined method that handles the entire conversation
    const response = await openai.chatWithFunctions(messages, functions, {
      model: 'gpt-4o',
      temperature: 0.7,
      maxRounds: 3,
      functionTimeout: 5000
    });
    
    console.log('Final response after function calls:', response.content);
  } catch (error) {
    console.error('Error in parallel function calling example:', error);
  }
}

/**
 * Metrics Collection Example
 */
async function metricsCollectionExample() {
  console.log('\nRunning metrics collection example...');
  
  // Make several different API calls
  try {
    await openai.chat([{ role: 'user', content: 'Hello' }], { model: 'gpt-4o-mini' });
    await openai.embedding('Test embedding', { model: 'text-embedding-3-small' });
    
    try {
      // Purposely make an error
      await openai.chat([{ role: 'user', content: 'Test error' }], { model: 'non-existent-model' });
    } catch (e) {
      console.log('Expected error with invalid model');
    }
    
    // Get metrics data
    const metrics = (openai as any).recentRequests;
    console.log(`Number of tracked requests: ${Object.keys(metrics).length}`);
    
    // Show sample of metrics
    if (Object.keys(metrics).length > 0) {
      const sampleKey = Object.keys(metrics)[0];
      console.log('Sample metrics entry:');
      console.log(metrics[sampleKey]);
    }
  } catch (error) {
    console.error('Error in metrics collection example:', error);
  }
}

/**
 * Run all examples
 */
async function runExamples() {
  // Start with model info to show available models
  await modelInformationExample();
  
  // GPT-4o-mini example
  await gpt4oMiniExample();
  
  // Voice-related examples
  if (process.env.RUN_AUDIO_EXAMPLES === 'true') {
    await voiceTypesExample();
    await voiceAgentExample();
    await speechToTextWithTimestampsExample();
    await audioTranslationExample();
  } else {
    console.log('\nSkipping audio examples. Set RUN_AUDIO_EXAMPLES=true to run them.');
  }
  
  // Function calling example
  await parallelFunctionCallingExample();
  
  // Metrics example
  await metricsCollectionExample();
  
  console.log('\nAll examples completed.');
}

// Run examples if this file is executed directly
if (require.main === module) {
  runExamples().catch(error => {
    console.error('Error running examples:', error);
  });
}