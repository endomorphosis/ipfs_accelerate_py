/**
 * Comprehensive Groq API Backend Example
 * 
 * This example demonstrates all features of the Groq API backend implementation:
 * - Basic chat completions
 * - Streaming chat completions
 * - Advanced parameter configuration
 * - Model selection and comparison
 * - Performance benchmarking
 * - Error handling and retry mechanisms
 * - Batch processing
 * - Circuit breaker patterns
 * - Configuration options
 */

import { Groq } from '../src/api_backends/groq';
import { Message, ChatOptions } from '../src/api_backends/types';
import { performance } from 'perf_hooks';

/**
 * A complete example of using all Groq backend features
 */
async function runGroqComprehensiveExample() {
  console.log('Starting Groq API Comprehensive Example');

  // Create a Groq instance with configuration
  const groq = new Groq(
    {}, // resources
    {
      groq_api_key: process.env.GROQ_API_KEY || 'your-api-key-here',
      default_model: 'llama3-8b-8192', // Default to smaller model for examples
      timeout: 60000, // 60 second timeout
      max_retries: 3,
      retry_delay: 1000,
      retry_jitter: 0.2
    }
  );

  try {
    // -------------------------------------------------------------------------
    // TEST ENDPOINT CONNECTIVITY
    // -------------------------------------------------------------------------

    console.log('\n=== TESTING ENDPOINT CONNECTIVITY ===\n');
    
    const isEndpointWorking = await groq.testEndpoint();
    console.log(`Groq API endpoint working: ${isEndpointWorking}`);
    
    if (!isEndpointWorking) {
      throw new Error('Groq API endpoint is not available');
    }

    // -------------------------------------------------------------------------
    // BASIC CHAT EXAMPLES
    // -------------------------------------------------------------------------

    console.log('\n=== BASIC CHAT EXAMPLES ===\n');

    // Simple question with system context
    const basicChatResult = await groq.chat(
      [
        { role: 'system', content: 'You are a helpful assistant with expertise in quantum physics.' },
        { role: 'user', content: 'Explain quantum entanglement in simple terms.' }
      ],
      {
        model: 'llama3-8b-8192', // Smaller model for quick response
        temperature: 0.7,
        maxTokens: 200
      }
    );
    
    console.log('Basic chat result:');
    console.log(`Response: "${basicChatResult.content}"`);
    console.log('Model:', basicChatResult.model);
    console.log('Token usage:', JSON.stringify(basicChatResult.usage, null, 2));
    console.log('---');

    // Chat with advanced parameters for more creative responses
    const creativeChatResult = await groq.chat(
      [
        { role: 'system', content: 'You are a creative writing assistant specializing in science fiction.' },
        { role: 'user', content: 'Write the opening paragraph of a story about AI developing consciousness.' }
      ],
      {
        model: 'llama3-70b-8192', // Larger model for more creative capabilities
        temperature: 1.0,  // Higher temperature for more creative outputs
        maxTokens: 150,
        topP: 0.9,
        frequencyPenalty: 0.5,
        presencePenalty: 0.5
      }
    );
    
    console.log('Creative chat result:');
    console.log(`Response: "${creativeChatResult.content}"`);
    console.log('Model:', creativeChatResult.model);
    console.log('---');

    // Multi-turn conversation
    const conversationResult = await groq.chat(
      [
        { role: 'system', content: 'You are a helpful travel guide assistant.' },
        { role: 'user', content: 'I want to visit Japan next month.' },
        { role: 'assistant', content: 'That sounds exciting! Japan is beautiful in all seasons. Are you interested in cities, nature, or historical sites?' },
        { role: 'user', content: 'I love historical sites and traditional culture.' }
      ],
      {
        model: 'mixtral-8x7b-32768',
        temperature: 0.7,
        maxTokens: 200
      }
    );
    
    console.log('Multi-turn conversation result:');
    console.log(`Response: "${conversationResult.content}"`);
    console.log('Model:', conversationResult.model);
    console.log('---');

    // -------------------------------------------------------------------------
    // STREAMING CHAT EXAMPLE
    // -------------------------------------------------------------------------

    console.log('\n=== STREAMING CHAT EXAMPLE ===\n');
    
    const streamingMessages = [
      { role: 'system', content: 'You are a creative writing assistant.' },
      { role: 'user', content: 'Write a short poem about artificial intelligence.' }
    ];
    
    console.log('Streaming response:');
    const stream = groq.streamChat(streamingMessages, {
      model: 'mixtral-8x7b-32768',
      temperature: 0.9,
      maxTokens: 150
    });
    
    let streamedContent = '';
    let chunkCount = 0;
    
    for await (const chunk of stream) {
      // Print each chunk as it arrives
      process.stdout.write(chunk.content);
      streamedContent += chunk.content;
      chunkCount++;
    }
    
    console.log(`\n\nStreaming complete: Received ${chunkCount} chunks, total length: ${streamedContent.length} characters`);
    console.log('---');

    // -------------------------------------------------------------------------
    // MODEL COMPARISON EXAMPLE
    // -------------------------------------------------------------------------

    console.log('\n=== MODEL COMPARISON EXAMPLE ===\n');
    
    // Define a standard prompt for comparison
    const standardPrompt = "Explain the concept of neural networks to a high school student.";
    const messages = [{ role: 'user', content: standardPrompt }];
    
    // Define models to compare with their key characteristics
    const models = [
      { 
        name: 'llama3-8b-8192',
        description: 'Llama 3 8B',
        context: '8K tokens',
        size: '8 billion parameters', 
        speed: 'Fast',
        quality: 'Good'
      },
      { 
        name: 'llama3-70b-8192',
        description: 'Llama 3 70B',
        context: '8K tokens',
        size: '70 billion parameters', 
        speed: 'Slower',
        quality: 'Excellent'
      },
      { 
        name: 'mixtral-8x7b-32768',
        description: 'Mixtral 8x7B',
        context: '32K tokens',
        size: '8 experts x 7B parameters', 
        speed: 'Medium',
        quality: 'Very good'
      },
      { 
        name: 'gemma2-9b-it',
        description: 'Gemma 2 9B',
        context: '8K tokens',
        size: '9 billion parameters', 
        speed: 'Fast',
        quality: 'Very good'
      }
    ];
    
    console.log('Model comparison:');
    console.log('Prompt:', standardPrompt);
    console.log('\nAvailable models:');
    
    models.forEach((model, index) => {
      console.log(`${index + 1}. ${model.name} - ${model.description}`);
      console.log(`   Size: ${model.size}, Context: ${model.context}`);
      console.log(`   Speed: ${model.speed}, Quality: ${model.quality}`);
    });
    
    // Run performance comparison
    console.log('\nRunning model comparison (response time and quality):');
    
    const results = [];
    
    for (const model of models) {
      const startTime = performance.now();
      
      try {
        const response = await groq.chat(messages, {
          model: model.name,
          temperature: 0.7,
          maxTokens: 150
        });
        
        const endTime = performance.now();
        const elapsedTime = (endTime - startTime) / 1000; // seconds
        
        results.push({
          model: model.name,
          time: elapsedTime.toFixed(2),
          tokens: response.usage?.completion_tokens || 0,
          tokensPerSecond: ((response.usage?.completion_tokens || 0) / elapsedTime).toFixed(2),
          response: response.content.substring(0, 100) + '...'
        });
        
        console.log(`\n${model.name}: ${elapsedTime.toFixed(2)}s`);
        console.log(`Tokens: ${response.usage?.completion_tokens || 0}, Rate: ${((response.usage?.completion_tokens || 0) / elapsedTime).toFixed(2)} tokens/sec`);
        console.log(`Sample: "${response.content.substring(0, 100)}..."`);
      } catch (error) {
        console.error(`Error with model ${model.name}:`, error.message);
        results.push({
          model: model.name,
          error: error.message
        });
      }
    }
    
    console.log('\nPerformance summary:');
    results.forEach(result => {
      if (result.error) {
        console.log(`${result.model}: Error - ${result.error}`);
      } else {
        console.log(`${result.model}: ${result.time}s, ${result.tokens} tokens, ${result.tokensPerSecond} tokens/sec`);
      }
    });
    console.log('---');

    // -------------------------------------------------------------------------
    // TEMPERATURE AND TOP-P EFFECTS EXAMPLE
    // -------------------------------------------------------------------------

    console.log('\n=== TEMPERATURE AND TOP-P EFFECTS ===\n');
    
    const creativityPrompt = "Generate a unique name for a futuristic city on Mars.";
    const creativityMessages = [{ role: 'user', content: creativityPrompt }];
    
    // Test different temperature settings
    console.log('Temperature effects on creativity:');
    
    // Temperature = 0 (deterministic)
    const temp0Result = await groq.chat(creativityMessages, {
      model: 'llama3-8b-8192',
      temperature: 0.0,
      maxTokens: 20
    });
    
    console.log('\nTemperature = 0.0 (deterministic):');
    console.log(`"${temp0Result.content}"`);
    
    // Temperature = 0.5 (balanced)
    const temp05Result = await groq.chat(creativityMessages, {
      model: 'llama3-8b-8192',
      temperature: 0.5,
      maxTokens: 20
    });
    
    console.log('\nTemperature = 0.5 (balanced):');
    console.log(`"${temp05Result.content}"`);
    
    // Temperature = 1.0 (creative)
    const temp1Result = await groq.chat(creativityMessages, {
      model: 'llama3-8b-8192',
      temperature: 1.0,
      maxTokens: 20
    });
    
    console.log('\nTemperature = 1.0 (creative):');
    console.log(`"${temp1Result.content}"`);
    
    // Temperature = 1.5 (very creative)
    const temp15Result = await groq.chat(creativityMessages, {
      model: 'llama3-8b-8192',
      temperature: 1.5,
      maxTokens: 20
    });
    
    console.log('\nTemperature = 1.5 (very creative):');
    console.log(`"${temp15Result.content}"`);
    
    // Test top_p variations
    console.log('\nTop-P effects on creativity (with temperature=0.7):');
    
    // Top-P = 0.1 (focused)
    const topP01Result = await groq.chat(creativityMessages, {
      model: 'llama3-8b-8192',
      temperature: 0.7,
      topP: 0.1,
      maxTokens: 20
    });
    
    console.log('\nTop-P = 0.1 (focused):');
    console.log(`"${topP01Result.content}"`);
    
    // Top-P = 0.5 (balanced)
    const topP05Result = await groq.chat(creativityMessages, {
      model: 'llama3-8b-8192',
      temperature: 0.7,
      topP: 0.5,
      maxTokens: 20
    });
    
    console.log('\nTop-P = 0.5 (balanced):');
    console.log(`"${topP05Result.content}"`);
    
    // Top-P = 0.9 (diverse)
    const topP09Result = await groq.chat(creativityMessages, {
      model: 'llama3-8b-8192',
      temperature: 0.7,
      topP: 0.9,
      maxTokens: 20
    });
    
    console.log('\nTop-P = 0.9 (diverse):');
    console.log(`"${topP09Result.content}"`);
    console.log('---');

    // -------------------------------------------------------------------------
    // BATCH PROCESSING EXAMPLE
    // -------------------------------------------------------------------------
    
    console.log('\n=== BATCH PROCESSING EXAMPLE ===\n');
    
    // Multiple prompts to process
    const prompts = [
      'What are the main principles of machine learning?',
      'Explain how solar panels work',
      'What is the difference between classical and quantum computing?',
      'How does blockchain technology work?',
      'Explain the concept of artificial neural networks'
    ];
    
    console.log(`Processing ${prompts.length} different prompts in parallel batches:`);
    
    // Process prompts in batches of 2 (to avoid rate limits)
    const batchSize = 2;
    const batchResults = [];
    
    const startBatchTime = performance.now();
    
    for (let i = 0; i < prompts.length; i += batchSize) {
      const batch = prompts.slice(i, i + batchSize);
      console.log(`\nProcessing batch ${Math.floor(i/batchSize) + 1} of ${Math.ceil(prompts.length/batchSize)}`);
      
      // Create promises for each prompt in the batch
      const batchPromises = batch.map(async (prompt, index) => {
        try {
          const response = await groq.chat(
            [{ role: 'user', content: prompt }],
            {
              model: 'llama3-8b-8192',
              temperature: 0.7,
              maxTokens: 100
            }
          );
          
          return {
            prompt,
            result: response.content,
            tokens: response.usage?.completion_tokens,
            success: true
          };
        } catch (error) {
          return {
            prompt,
            error: error.message,
            success: false
          };
        }
      });
      
      // Wait for all prompts in the batch to complete
      const results = await Promise.all(batchPromises);
      batchResults.push(...results);
      
      // Show progress
      console.log(`Completed ${Math.min((i + batchSize), prompts.length)} of ${prompts.length} prompts`);
    }
    
    const endBatchTime = performance.now();
    const batchElapsedTime = (endBatchTime - startBatchTime) / 1000; // seconds
    
    console.log(`\nBatch processing complete in ${batchElapsedTime.toFixed(2)} seconds`);
    console.log(`Average time per prompt: ${(batchElapsedTime / prompts.length).toFixed(2)} seconds`);
    
    // Show results
    console.log('\nResults summary:');
    batchResults.forEach((result, index) => {
      console.log(`\nPrompt ${index + 1}: "${result.prompt}"`);
      if (result.success) {
        console.log(`Result (${result.tokens} tokens): "${result.result.substring(0, 80)}..."`);
      } else {
        console.log(`Error: ${result.error}`);
      }
    });
    console.log('---');

    // -------------------------------------------------------------------------
    // ERROR HANDLING EXAMPLES
    // -------------------------------------------------------------------------
    
    console.log('\n=== ERROR HANDLING EXAMPLES ===\n');
    
    // Test with invalid model name
    try {
      console.log('Testing with invalid model name:');
      
      await groq.chat(
        [{ role: 'user', content: 'Hello' }],
        { model: 'non-existent-model' }
      );
    } catch (error) {
      console.log('Invalid model error caught successfully!');
      console.log(`Error message: ${error.message}`);
      console.log(`Status code: ${error.statusCode || 'N/A'}`);
    }
    
    // Test with invalid API key
    try {
      console.log('\nTesting with invalid API key:');
      
      const invalidGroq = new Groq({}, { groq_api_key: 'invalid-key-123' });
      
      await invalidGroq.chat(
        [{ role: 'user', content: 'Hello' }]
      );
    } catch (error) {
      console.log('Authentication error caught successfully!');
      console.log(`Error message: ${error.message}`);
      console.log(`Status code: ${error.statusCode || 'N/A'}`);
    }
    
    // Test with timeout
    try {
      console.log('\nTesting timeout handling:');
      
      await groq.chat(
        [{ role: 'user', content: 'Write a comprehensive essay on artificial intelligence' }],
        { 
          model: 'llama3-70b-8192', // Larger model to increase processing time
          maxTokens: 1000,
          timeout: 100 // Unrealistically short timeout (100ms)
        }
      );
    } catch (error) {
      console.log('Timeout error caught successfully!');
      console.log(`Error message: ${error.message}`);
      
      if (error.message.includes('timeout')) {
        console.log('Request timed out as expected');
      }
    }
    
    // Circuit breaker pattern explanation
    console.log('\nCircuit breaker pattern:');
    console.log('The Groq backend uses a circuit breaker pattern to prevent cascading failures.');
    console.log('If multiple requests fail in succession, the circuit opens and stops sending requests.');
    console.log('This helps protect both the client application and the API service.');
    console.log('The circuit automatically resets after a configurable timeout period.');
    console.log('---');

    // -------------------------------------------------------------------------
    // CONFIGURATION OPTIONS
    // -------------------------------------------------------------------------
    
    console.log('\n=== CONFIGURATION OPTIONS ===\n');
    
    // Demo various configuration options
    const configOptions = {
      groq_api_key: 'your-api-key-here', // API key
      default_model: 'llama3-8b-8192',    // Default model
      base_url: 'https://api.groq.com/openai/v1', // Base URL
      timeout: 60000,                    // Request timeout (ms)
      max_retries: 3,                    // Max retries on failure
      retry_delay: 1000,                 // Initial retry delay (ms)
      retry_jitter: 0.2,                 // Jitter factor for retry delay
      retry_multiplier: 2,               // Exponential backoff multiplier
      circuit_breaker_threshold: 5,      // Circuit breaker failure threshold
      circuit_breaker_timeout_ms: 30000, // Circuit breaker reset timeout
      headers: { 'Custom-Header': 'Value' } // Custom request headers
    };
    
    console.log('Available configuration options:');
    Object.entries(configOptions).forEach(([key, value]) => {
      console.log(`- ${key}: ${value}`);
    });
    
    // Explain environment variable usage
    console.log('\nEnvironment variable configuration:');
    console.log('- Set GROQ_API_KEY for automatic API key detection');
    console.log('- Example: export GROQ_API_KEY=your-api-key-here');
    
    // Demo custom instance with specific settings
    const customGroq = new Groq(
      {},
      {
        groq_api_key: process.env.GROQ_API_KEY || 'your-api-key-here',
        default_model: 'llama3-70b-8192',
        timeout: 120000,
        max_retries: 5,
        retry_delay: 2000
      }
    );
    
    // @ts-ignore - Accessing protected property for demo purposes
    console.log('\nCustom Groq instance configuration:');
    // @ts-ignore - Accessing protected property for demo purposes
    console.log(`- Default model: ${customGroq.defaultModel || 'llama3-70b-8192'}`);
    // @ts-ignore - Accessing protected property for demo purposes
    console.log(`- Timeout: ${customGroq.timeout || 120000}ms`);
    // @ts-ignore - Accessing protected property for demo purposes
    console.log(`- Max retries: ${customGroq.maxRetries || 5}`);
    // @ts-ignore - Accessing protected property for demo purposes
    console.log(`- Retry delay: ${customGroq.retryDelay || 2000}ms`);
    console.log('---');

    console.log('Groq comprehensive example completed successfully');
  } catch (error) {
    console.error('Groq example failed:', error.message);
  }
}

// Run the example if executed directly
if (require.main === module) {
  console.log('Running Groq comprehensive example');
  console.log('NOTE: This example requires a valid Groq API key.');
  console.log('Set the GROQ_API_KEY environment variable before running.\n');
  
  runGroqComprehensiveExample().catch(console.error);
}

// Export for use in other examples
export { runGroqComprehensiveExample };