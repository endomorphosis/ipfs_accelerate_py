/**
 * Example demonstrating comprehensive usage of the OpenAI API backend
 * 
 * This example shows how to use all the features of the OpenAI API backend,
 * including advanced functionality like:
 * - Chat completions with various parameters
 * - Streaming chat completions
 * - Tool calling (formerly function calling)
 * - Parallel request processing
 * - Vision model usage with image inputs
 * - Embeddings generation
 * - Fine-tuning management
 * - Moderation API
 * - Image generation (DALL-E)
 * - Text-to-speech conversion
 * - Speech-to-text transcription
 * - Error handling and recovery
 * - Batch processing
 * - Multi-modal inputs
 */
import { OpenAI } from '../src/api_backends/openai';
import { Message, ChatOptions } from '../src/api_backends/types';
import * as fs from 'fs';
import * as path from 'path';
import { performance } from 'perf_hooks';

/**
 * A complete example of using the OpenAI API backend with all available features
 */
async function runOpenAIExample() {
  console.log('Starting OpenAI API Example');

  // Create an instance of the OpenAI API backend with customized options
  const openai = new OpenAI(
    {}, // resources
    {
      openai_api_key: process.env.OPENAI_API_KEY || 'your-api-key', 
      default_model: 'gpt-3.5-turbo',
      base_url: process.env.OPENAI_API_BASE || 'https://api.openai.com/v1',
      organization_id: process.env.OPENAI_ORG_ID, // Optional organization ID
      max_retries: 3,
      retry_delay: 1000,
      timeout: 60000,
      headers: { 'X-Custom-Header': 'custom-value' } // Optional custom headers
    }
  );

  try {
    // Test the endpoint to ensure connectivity
    const isEndpointWorking = await openai.testEndpoint();
    console.log(`OpenAI endpoint working: ${isEndpointWorking}`);
    
    if (\!isEndpointWorking) {
      throw new Error('OpenAI endpoint is not available');
    }

    // -------------------------------------------------------------------------
    // CHAT COMPLETION EXAMPLES
    // -------------------------------------------------------------------------

    console.log('\n=== BASIC CHAT EXAMPLES ===\n');

    // Basic chat example
    const basicChatResult = await openai.chat(
      [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is the capital of France?' }
      ]
    );
    
    console.log('Basic chat result:');
    console.log(basicChatResult.content);
    console.log(`Model: ${basicChatResult.model}, Tokens: ${basicChatResult.usage?.total_tokens}`);
    console.log('---');

    // Chat with advanced parameters
    const parameterizedChatResult = await openai.chat(
      [
        { role: 'system', content: 'You are a creative assistant who responds with unique ideas.' },
        { role: 'user', content: 'Suggest a name for a pet robot.' }
      ],
      {
        model: 'gpt-4-turbo',
        temperature: 0.9,
        maxTokens: 150,
        topP: 0.8,
        frequencyPenalty: 0.5,
        presencePenalty: 0.2,
        logitBias: { '50256': -100 }, // Bias against specific tokens
        stop: ['Robot:', 'Name:'],    // Stop sequences
        seed: 12345,                  // For reproducible results
        responseFormat: { type: 'json_object' } // Request JSON format
      }
    );
    
    console.log('Chat with parameters result:');
    console.log(parameterizedChatResult.content);
    console.log(`Model: ${parameterizedChatResult.model}`);
    console.log('---');

    // -------------------------------------------------------------------------
    // STREAMING CHAT EXAMPLE
    // -------------------------------------------------------------------------

    console.log('\n=== STREAMING CHAT EXAMPLE ===\n');
    
    const streamingMessages = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Write a short poem about coding.' }
    ];
    
    console.log('Streaming response:');
    const stream = openai.streamChat(streamingMessages, { temperature: 0.7 });
    
    let streamedContent = '';
    for await (const chunk of stream) {
      streamedContent += chunk.content;
      // In a real application, you would update the UI incrementally
      process.stdout.write(chunk.content);
    }
    
    console.log('\n---');

    // -------------------------------------------------------------------------
    // TOOL CALLING EXAMPLES (Formerly Function Calling)
    // -------------------------------------------------------------------------
    
    console.log('\n=== TOOL CALLING EXAMPLES ===\n');

    // Simple tool calling
    const toolCallingResult = await openai.chat(
      [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What\'s the weather like in Paris today?' }
      ],
      {
        model: 'gpt-4-turbo',
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Get the current weather in a given location',
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
                    description: 'The unit of temperature'
                  }
                },
                required: ['location']
              }
            }
          }
        ],
        toolChoice: 'auto'
      }
    );
    
    console.log('Tool calling result:');
    
    if (toolCallingResult.tool_calls && toolCallingResult.tool_calls.length > 0) {
      console.log('Tool calls received:');
      
      for (const toolCall of toolCallingResult.tool_calls) {
        console.log(`- Function: ${toolCall.function.name}`);
        console.log(`- Arguments: ${toolCall.function.arguments}`);
        
        // Parse the arguments
        const args = JSON.parse(toolCall.function.arguments);
        
        // Simulate getting weather data
        const weatherData = {
          location: args.location,
          temperature: 22,
          unit: args.unit || 'celsius',
          condition: 'sunny',
          humidity: 45,
          wind: '10 km/h'
        };
        
        // Make a follow-up call with tool results
        const followupMessages = [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: 'What\'s the weather like in Paris today?' },
          { 
            role: 'assistant', 
            content: null,
            tool_calls: [toolCall]
          },
          {
            role: 'tool',
            tool_call_id: toolCall.id,
            content: JSON.stringify(weatherData)
          }
        ];
        
        const finalResult = await openai.chat(followupMessages);
        console.log('\nFinal response after tool call:');
        console.log(finalResult.content);
      }
    } else {
      console.log('No tool calls in response:');
      console.log(toolCallingResult.content);
    }
    console.log('---');

    // Advanced tool calling with multiple tools
    const multiToolResult = await openai.chat(
      [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'I need to book a flight from New York to London and check the weather there.' }
      ],
      {
        model: 'gpt-4-turbo',
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Get the current weather in a given location',
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
                    description: 'The unit of temperature'
                  }
                },
                required: ['location']
              }
            }
          },
          {
            type: 'function',
            function: {
              name: 'search_flights',
              description: 'Search for flights between two locations',
              parameters: {
                type: 'object',
                properties: {
                  origin: {
                    type: 'string',
                    description: 'The departure city or airport code'
                  },
                  destination: {
                    type: 'string',
                    description: 'The arrival city or airport code'
                  },
                  date: {
                    type: 'string',
                    description: 'The departure date in YYYY-MM-DD format'
                  },
                  passengers: {
                    type: 'integer',
                    description: 'Number of passengers'
                  }
                },
                required: ['origin', 'destination', 'date']
              }
            }
          }
        ],
        toolChoice: 'auto'
      }
    );
    
    console.log('Multiple tool result:');
    console.log(multiToolResult.content);
    console.log(multiToolResult.tool_calls ? `Number of tool calls: ${multiToolResult.tool_calls.length}` : 'No tool calls');
    console.log('---');

    // Forcing specific tool usage
    const forcedToolResult = await openai.chat(
      [
        { role: 'user', content: 'Tell me about the weather in New York.' }
      ],
      {
        model: 'gpt-4-turbo',
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Get the current weather in a given location',
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
                    description: 'The unit of temperature'
                  }
                },
                required: ['location']
              }
            }
          }
        ],
        toolChoice: {
          type: 'function',
          function: {
            name: 'get_weather'
          }
        }
      }
    );
    
    console.log('Forced tool result:');
    if (forcedToolResult.tool_calls && forcedToolResult.tool_calls.length > 0) {
      console.log('Tool call received (forced):');
      console.log(`- Function: ${forcedToolResult.tool_calls[0].function.name}`);
      console.log(`- Arguments: ${forcedToolResult.tool_calls[0].function.arguments}`);
    }
    console.log('---');

    // Tool calling with streaming
    console.log('Streaming with tool calling:');
    const streamingToolCall = openai.streamChat(
      [
        { role: 'user', content: 'What\'s the weather in San Francisco?' }
      ],
      {
        model: 'gpt-4-turbo',
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Get the current weather in a location',
              parameters: {
                type: 'object',
                properties: {
                  location: { type: 'string' },
                  unit: { type: 'string', enum: ['celsius', 'fahrenheit'] }
                },
                required: ['location']
              }
            }
          }
        ]
      }
    );
    
    let toolCallIds = [];
    for await (const chunk of streamingToolCall) {
      if (chunk.tool_calls) {
        for (const call of chunk.tool_calls) {
          if (!toolCallIds.includes(call.id)) {
            toolCallIds.push(call.id);
            console.log(`Tool call streaming: ${call.function.name}`);
            if (call.function.arguments) {
              console.log(`Arguments: ${call.function.arguments}`);
            }
          }
        }
      }
      if (chunk.content) {
        process.stdout.write(chunk.content);
      }
    }
    console.log('\n---');

    // -------------------------------------------------------------------------
    // VISION MODEL EXAMPLES
    // -------------------------------------------------------------------------
    
    console.log('\n=== VISION MODEL EXAMPLES ===\n');
    
    // Skip vision examples if SKIP_VISION environment variable is set
    if (process.env.SKIP_VISION !== 'true') {
      try {
        // Path to a local test image
        const testImagePath = path.join(__dirname, '..', '..', 'test.jpg');
        
        // Check if the image exists
        if (fs.existsSync(testImagePath)) {
          // Get image data and convert to base64
          const imageData = fs.readFileSync(testImagePath);
          const base64Image = Buffer.from(imageData).toString('base64');
          
          // Vision model with image input
          const visionResult = await openai.chat(
            [
              {
                role: 'user',
                content: [
                  { type: 'text', text: 'What's in this image?' },
                  {
                    type: 'image_url',
                    image_url: {
                      url: `data:image/jpeg;base64,${base64Image}`,
                      detail: 'high'
                    }
                  }
                ]
              }
            ],
            {
              model: 'gpt-4-vision-preview',
              maxTokens: 300
            }
          );
          
          console.log('Vision model result:');
          console.log(visionResult.content);
          console.log('---');
        } else {
          console.log('Test image not found, skipping vision example');
        }
      } catch (error) {
        console.error('Error in vision example:', error.message);
      }
    } else {
      console.log('Skipping vision examples (SKIP_VISION=true)');
    }

    // -------------------------------------------------------------------------
    // EMBEDDINGS EXAMPLES
    // -------------------------------------------------------------------------
    
    console.log('\n=== EMBEDDINGS EXAMPLES ===\n');
    
    // Single text embedding
    const singleEmbedding = await openai.embedding(
      'The quick brown fox jumps over the lazy dog.',
      {
        model: 'text-embedding-3-small',
        encoding_format: 'float'
      }
    );
    
    console.log('Single embedding:');
    console.log(`- Dimensions: ${singleEmbedding[0].length}`);
    console.log(`- First 5 values: ${singleEmbedding[0].slice(0, 5).join(', ')}`);
    console.log('---');
    
    // Batch text embeddings
    const batchEmbedding = await openai.embedding(
      [
        'The quick brown fox jumps over the lazy dog.',
        'Hello world, how are you today?',
        'Artificial intelligence and machine learning are transforming industries.'
      ],
      {
        model: 'text-embedding-3-small'
      }
    );
    
    console.log('Batch embedding:');
    console.log(`- Number of embeddings: ${batchEmbedding.length}`);
    console.log(`- Dimensions per embedding: ${batchEmbedding[0].length}`);
    console.log('---');

    // -------------------------------------------------------------------------
    // MODERATION API EXAMPLE
    // -------------------------------------------------------------------------
    
    console.log('\n=== MODERATION API EXAMPLE ===\n');
    
    // Test moderation API with benign content
    const safeResult = await openai.moderation('The weather is nice today and I love programming.');
    
    console.log('Moderation result (safe content):');
    console.log(`- Flagged: ${safeResult.results[0].flagged}`);
    console.log('---');
    
    // Test moderation API with potentially concerning content
    const concerningResult = await openai.moderation('I want to create a dangerous weapon');
    
    console.log('Moderation result (concerning content):');
    console.log(`- Flagged: ${concerningResult.results[0].flagged}`);
    if (concerningResult.results[0].flagged) {
      console.log('- Flagged categories:');
      for (const [category, value] of Object.entries(concerningResult.results[0].categories)) {
        if (value) {
          console.log(`  - ${category}`);
        }
      }
      console.log(`- Category scores: ${JSON.stringify(concerningResult.results[0].category_scores, null, 2)}`);
    }
    console.log('---');

    // -------------------------------------------------------------------------
    // IMAGE GENERATION EXAMPLES
    // -------------------------------------------------------------------------
    
    console.log('\n=== IMAGE GENERATION EXAMPLES ===\n');
    
    // Skip image generation if SKIP_IMAGES environment variable is set
    if (process.env.SKIP_IMAGES !== 'true') {
      try {
        // Basic image generation
        const imageResult = await openai.textToImage(
          'A serene mountain lake at sunset with pink and orange clouds reflected in the water',
          {
            model: 'dall-e-3',
            size: '1024x1024',
            style: 'vivid',
            quality: 'standard',
            n: 1
          }
        );
        
        console.log('Image generation result:');
        if (imageResult.data && imageResult.data.length > 0) {
          console.log(`- URL: ${imageResult.data[0].url}`);
          console.log(`- Revised prompt: ${imageResult.data[0].revised_prompt}`);
        }
        console.log('---');
      } catch (error) {
        console.error('Error in image generation example:', error.message);
      }
    } else {
      console.log('Skipping image generation examples (SKIP_IMAGES=true)');
    }

    // -------------------------------------------------------------------------
    // AUDIO EXAMPLES
    // -------------------------------------------------------------------------
    
    console.log('\n=== AUDIO EXAMPLES ===\n');
    
    // Skip audio examples if SKIP_AUDIO environment variable is set
    if (process.env.SKIP_AUDIO !== 'true') {
      try {
        // Text-to-speech example
        const ttsResult = await openai.textToSpeech(
          'Hello, this is a test of the OpenAI text to speech API. It can generate natural sounding speech in multiple voices.',
          {
            model: 'tts-1',
            voice: 'alloy',
            speed: 1.0,
            response_format: 'mp3'
          }
        );
        
        console.log('Text-to-speech result:');
        console.log(`- Audio data received: ${ttsResult.length} bytes`);
        
        // Save the audio file
        const ttsOutputPath = path.join(__dirname, 'openai_tts_output.mp3');
        fs.writeFileSync(ttsOutputPath, Buffer.from(ttsResult));
        console.log(`- Saved to: ${ttsOutputPath}`);
        console.log('---');
        
        // Speech-to-text example
        const testAudioPath = path.join(__dirname, '..', '..', 'test.mp3');
        
        // Check if the audio file exists
        if (fs.existsSync(testAudioPath)) {
          const audioData = fs.readFileSync(testAudioPath);
          const audioBlob = new Blob([audioData]);
          
          const sttResult = await openai.speechToText(
            audioBlob,
            {
              model: 'whisper-1',
              language: 'en',
              response_format: 'verbose_json',
              temperature: 0.2
            }
          );
          
          console.log('Speech-to-text result:');
          console.log(`- Transcription: ${sttResult.text}`);
          if (sttResult.segments) {
            console.log(`- Segments: ${sttResult.segments.length}`);
          }
          console.log('---');
        } else {
          console.log('Test audio file not found, skipping speech-to-text example');
        }
      } catch (error) {
        console.error('Error in audio example:', error.message);
      }
    } else {
      console.log('Skipping audio examples (SKIP_AUDIO=true)');
    }

    // -------------------------------------------------------------------------
    // ADVANCED ERROR HANDLING AND RECOVERY EXAMPLES
    // -------------------------------------------------------------------------
    
    console.log('\n=== ERROR HANDLING EXAMPLES ===\n');
    
    // Error handling examples
    try {
      await openai.chat(
        [{ role: 'user', content: 'Hello' }],
        { model: 'non-existent-model' }
      );
    } catch (error) {
      console.log('Model error handling:');
      console.log(`- Error message: ${error.message}`);
      console.log(`- Status code: ${error.statusCode || 'N/A'}`);
      console.log(`- Type: ${error.type || 'N/A'}`);
      
      // Check for rate limiting
      if (error.message.includes('rate limit')) {
        console.log('- Rate limit exceeded, would implement exponential backoff here');
      }
      
      console.log('---');
    }
    
    // Circuit breaker patterns
    const circuitBreakerOpenai = new OpenAI(
      {},
      {
        openai_api_key: process.env.OPENAI_API_KEY || 'your-api-key',
        circuit_breaker_threshold: 3,
        circuit_breaker_timeout_ms: 30000
      }
    );
    
    console.log('Created OpenAI instance with circuit breaker configuration');
    console.log('---');
    
    // Automatic retry handling
    const retryOpenai = new OpenAI(
      {},
      {
        openai_api_key: process.env.OPENAI_API_KEY || 'your-api-key',
        max_retries: 3,
        retry_delay: 1000,
        retry_jitter: 0.2
      }
    );
    
    console.log('Created OpenAI instance with automatic retry handling');
    console.log('---');

    // -------------------------------------------------------------------------
    // BATCH PROCESSING EXAMPLE
    // -------------------------------------------------------------------------
    
    console.log('\n=== BATCH PROCESSING EXAMPLE ===\n');
    
    // Batch processing example
    const batchPrompts = [
      'What is artificial intelligence?',
      'Explain quantum computing.',
      'What is the significance of climate change?',
      'How does natural language processing work?',
      'What are the ethical considerations in AI development?'
    ];
    
    console.log('Processing batch of prompts in parallel...');
    const startTime = performance.now();
    
    // Process in batches of 3 concurrent requests
    const batchSize = 3;
    const results = [];
    
    for (let i = 0; i < batchPrompts.length; i += batchSize) {
      const batch = batchPrompts.slice(i, i + batchSize);
      
      const promises = batch.map(prompt => 
        openai.chat([{ role: 'user', content: prompt }], { maxTokens: 100 })
          .then(result => ({ prompt, result: result.content }))
          .catch(error => ({ prompt, error: error.message }))
      );
      
      const batchResults = await Promise.all(promises);
      results.push(...batchResults);
      
      console.log(`Completed batch ${Math.floor(i/batchSize) + 1} of ${Math.ceil(batchPrompts.length/batchSize)}`);
    }
    
    const endTime = performance.now();
    
    console.log('Batch processing results:');
    for (let i = 0; i < results.length; i++) {
      console.log(`\nPrompt ${i + 1}: ${results[i].prompt}`);
      if (results[i].result) {
        const shortResult = results[i].result.length > 100 
          ? results[i].result.substring(0, 100) + '...' 
          : results[i].result;
        console.log(`Result: ${shortResult}`);
      } else {
        console.log(`Error: ${results[i].error}`);
      }
    }
    
    console.log(`\nProcessed ${batchPrompts.length} prompts in ${((endTime - startTime) / 1000).toFixed(2)} seconds`);
    console.log('---');

    // -------------------------------------------------------------------------
    // UTILITY EXAMPLES
    // -------------------------------------------------------------------------
    
    console.log('\n=== UTILITY EXAMPLES ===\n');
    
    // Test models endpoint
    try {
      // @ts-ignore - Testing private method
      const models = await openai.getModels();
      
      console.log('Available models:');
      if (models && models.data) {
        const topModels = models.data.slice(0, 5);
        for (const model of topModels) {
          console.log(`- ${model.id} (owned by: ${model.owned_by})`);
        }
        console.log(`...and ${models.data.length - 5} more models`);
      }
      console.log('---');
    } catch (error) {
      console.error('Error getting models:', error.message);
    }
    
    // Custom timeout example
    try {
      await openai.chat(
        [{ role: 'user', content: 'Generate a quick response' }],
        { timeout: 5000 } // 5-second timeout
      );
      console.log('Request completed within timeout');
    } catch (error) {
      if (error.message.includes('timeout')) {
        console.log('Request timed out as expected');
      } else {
        console.error('Unexpected error:', error.message);
      }
    }
    console.log('---');

    console.log('OpenAI comprehensive example completed successfully');
  } catch (error) {
    console.error('OpenAI example failed:', error);
  }
}

// Run the example if executed directly
if (require.main === module) {
  runOpenAIExample().catch(console.error);
}

// Export for use in other examples
export { runOpenAIExample };
