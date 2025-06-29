/**
 * HuggingFace Text Generation Inference Unified Backend - Comprehensive Example
 * 
 * This comprehensive example demonstrates the full capabilities of the HF TGI Unified backend:
 * 
 * 1. Configuration options and initialization with best practices
 * 2. Text generation with various parameters
 * 3. Streaming text generation
 * 4. Chat interface with different model families
 * 5. Container management and deployment
 * 6. Advanced prompt engineering
 * 7. Performance benchmarking
 * 8. Error handling and recovery
 * 9. Circuit breaker and request queue usage
 * 10. Model family-specific optimizations
 */

// Import the HfTgiUnified class and types
import { HfTgiUnified } from '../src/api_backends/hf_tgi_unified';
import {
  HfTgiUnifiedOptions,
  HfTgiUnifiedApiMetadata,
  TextGenerationOptions,
  ChatGenerationOptions,
  DeploymentConfig,
  PerformanceMetrics,
  HfTgiModelInfo
} from '../src/api_backends/hf_tgi_unified/types';
import { ChatMessage, PriorityLevel } from '../src/api_backends/types';
import path from 'path';
import fs from 'fs';

// Polyfill for performance.now() if not available in environment
if (typeof performance === 'undefined') {
  const { performance } = require('perf_hooks');
  global.performance = performance;
}

/**
 * Comprehensive example for the HF TGI Unified backend
 */
async function main() {
  console.log('\n----------------------------------------------------------');
  console.log('HF TGI UNIFIED BACKEND - COMPREHENSIVE EXAMPLE');
  console.log('----------------------------------------------------------\n');
  
  const results: Record<string, any> = {};
  
  try {
    // -------------------------------------------------------------------------------
    // SECTION 1: INITIALIZATION AND CONFIGURATION
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 1: INITIALIZATION AND CONFIGURATION');
    
    // 1.1 Environment and API Key Management
    console.log('\n📋 1.1 Environment and API Key Management:');
    
    // Get API key from environment variable, parameter, or configuration file
    const envApiKey = process.env.HF_API_KEY;
    let apiKey: string | undefined = undefined;
    
    if (envApiKey) {
      console.log('   ✓ Using API key from environment variable');
      apiKey = envApiKey;
    } else {
      // Check for API key in configuration file
      const configPath = path.join(process.cwd(), 'hf_config.json');
      if (fs.existsSync(configPath)) {
        try {
          const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
          if (config.api_key) {
            console.log('   ✓ Using API key from configuration file');
            apiKey = config.api_key;
          }
        } catch (error) {
          console.log('   × Failed to read configuration file');
        }
      }
      
      // Fallback to demo mode
      if (!apiKey) {
        console.log('   ⚠ No API key found, running in demo mode with limited functionality');
        console.log('   ℹ To use full functionality, set HF_API_KEY environment variable');
      }
    }
    
    // 1.2 Advanced Configuration Options
    console.log('\n📋 1.2 Advanced Configuration Options:');
    
    // Set up advanced configuration options
    const options: HfTgiUnifiedOptions = {
      apiUrl: 'https://api-inference.huggingface.co/models',
      containerUrl: 'http://localhost:8080',
      maxRetries: 3,                // Number of retries for failed requests
      requestTimeout: 60000,        // 60 second timeout
      useRequestQueue: true,        // Use request queue for rate limiting
      debug: false,                 // Enable/disable debug logging
      useContainer: false,          // Start in API mode (not container mode)
      dockerRegistry: 'ghcr.io/huggingface/text-generation-inference',
      containerTag: 'latest',
      gpuDevice: '0',               // GPU device ID for container mode
      // Default generation parameters
      maxTokens: 100,               // Default max tokens to generate
      temperature: 0.7,             // Default temperature
      topP: 0.95,                   // Default top-p sampling
      topK: 50,                     // Default top-k sampling
      repetitionPenalty: 1.1        // Default repetition penalty
    };
    
    console.log('   ✓ Configured with timeout:', options.requestTimeout, 'ms');
    console.log('   ✓ Retries:', options.maxRetries);
    console.log('   ✓ Request queue:', options.useRequestQueue ? 'enabled' : 'disabled');
    console.log('   ✓ Mode:', options.useContainer ? 'container' : 'API');
    console.log('   ✓ Default max tokens:', options.maxTokens);
    console.log('   ✓ Default temperature:', options.temperature);
    
    // 1.3 Metadata Configuration
    console.log('\n📋 1.3 Metadata Configuration:');
    
    // Define multiple models to demonstrate model selection
    const availableModels = {
      'small': 'google/flan-t5-small',           // Small model (250M parameters)
      'base': 'google/flan-t5-base',             // Base model (580M parameters)
      'opt-small': 'facebook/opt-125m',          // Small OPT model
      'bloom': 'bigscience/bloom-560m',          // Multilingual model
      'phi': 'microsoft/phi-2',                  // Efficient small model
      'mistral': 'mistralai/Mistral-7B-Instruct-v0.1'  // High-quality model (requires GPU)
    };
    
    // Default to a small model for this example
    const defaultModel = 'google/flan-t5-small';
    
    // Set up metadata with API key and default model
    const metadata: HfTgiUnifiedApiMetadata = {
      hf_api_key: apiKey,
      model_id: defaultModel
    };
    
    console.log('   ✓ Default model:', defaultModel);
    console.log('   ✓ Available models:', Object.keys(availableModels).length);
    
    // 1.4 Backend Initialization
    console.log('\n📋 1.4 Backend Initialization:');
    
    // Create the HF TGI Unified backend instance
    const hfTgiUnified = new HfTgiUnified(options, metadata);
    
    console.log('   ✓ HF TGI Unified backend initialized successfully');
    console.log('   ✓ Current mode:', hfTgiUnified.getMode());
    console.log('   ✓ Default model:', hfTgiUnified.getDefaultModel());
    
    // Store results for reporting
    results.initialization = {
      mode: hfTgiUnified.getMode(),
      defaultModel: hfTgiUnified.getDefaultModel(),
      apiKeyAvailable: !!apiKey
    };
    
    // 1.5 Test Endpoint Availability
    console.log('\n📋 1.5 Test Endpoint Availability:');
    
    // Check if the endpoint is available
    try {
      const isEndpointAvailable = await hfTgiUnified.testEndpoint();
      console.log(`   ${isEndpointAvailable ? '✓' : '×'} Endpoint available: ${isEndpointAvailable}`);
      
      if (!isEndpointAvailable && !apiKey) {
        console.log('   ℹ Endpoint availability may be limited without an API key');
      }
      
      results.endpointAvailable = isEndpointAvailable;
    } catch (error) {
      console.log('   × Error testing endpoint:', error);
      results.endpointAvailable = false;
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 2: MODEL COMPATIBILITY AND INFORMATION
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 2: MODEL COMPATIBILITY AND INFORMATION');
    
    // 2.1 Model Compatibility Check
    console.log('\n📋 2.1 Model Compatibility Check:');
    
    // Check compatibility for various models
    const modelsToCheck = [
      ...Object.values(availableModels),
      'openai/clip-base',          // Vision model (should be incompatible)
      'facebook/bart-large',       // Sequence-to-sequence model (should be incompatible)
      'microsoft/phi-1_5',         // Compatible text generation model
      'random-model-name'          // Unknown model (should be incompatible)
    ];
    
    const compatibilityResults: Record<string, boolean> = {};
    for (const model of modelsToCheck) {
      const isCompatible = hfTgiUnified.isCompatibleModel(model);
      compatibilityResults[model] = isCompatible;
      console.log(`   ${isCompatible ? '✓' : '×'} ${model}: ${isCompatible ? 'Compatible' : 'Not compatible'}`);
    }
    
    results.modelCompatibility = compatibilityResults;
    
    // 2.2 Model Information
    console.log('\n📋 2.2 Model Information:');
    
    // Attempt to get model information for the default model
    let modelInfo: HfTgiModelInfo | null = null;
    try {
      modelInfo = await hfTgiUnified.getModelInfo();
      console.log('   ✓ Model information retrieved successfully:');
      console.log(`     - Model ID: ${modelInfo.model_id}`);
      console.log(`     - Status: ${modelInfo.status}`);
      
      if (modelInfo.revision) console.log(`     - Revision: ${modelInfo.revision}`);
      if (modelInfo.framework) console.log(`     - Framework: ${modelInfo.framework}`);
      if (modelInfo.max_input_length) console.log(`     - Max input length: ${modelInfo.max_input_length}`);
      if (modelInfo.max_total_tokens) console.log(`     - Max total tokens: ${modelInfo.max_total_tokens}`);
      if (modelInfo.parameters) console.log(`     - Parameters: ${modelInfo.parameters.join(', ')}`);
      
      results.modelInfo = modelInfo;
    } catch (error) {
      console.log('   × Error retrieving model information:');
      console.log(`     ${error instanceof Error ? error.message : String(error)}`);
      console.log('     Continuing with limited model information');
      
      // Create default model info
      modelInfo = {
        model_id: hfTgiUnified.getDefaultModel(),
        status: 'unknown',
        revision: 'unknown',
        framework: 'unknown',
        parameters: []
      };
      
      results.modelInfo = {
        ...modelInfo,
        estimated: true
      };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 3: BASIC TEXT GENERATION
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 3: BASIC TEXT GENERATION');
    
    // 3.1 Generate Text with Default Parameters
    console.log('\n📋 3.1 Generate Text with Default Parameters:');
    
    try {
      const prompt = "Write a short poem about artificial intelligence.";
      console.log(`   Input prompt: "${prompt}"`);
      
      const startTime = performance.now();
      const generatedText = await hfTgiUnified.generateText(prompt);
      const endTime = performance.now();
      
      console.log('   ✓ Generated text:');
      console.log(`     "${generatedText}"`);
      console.log(`   ✓ Generation time: ${(endTime - startTime).toFixed(2)}ms`);
      
      results.basicGeneration = {
        prompt,
        output: generatedText,
        time: endTime - startTime
      };
    } catch (error) {
      console.log('   × Error generating text:', error);
      console.log('   Using mock data for demonstration...');
      
      // Use mock text for demonstration
      const mockText = "AI thinks with silicon minds,\nLearning patterns, connecting lines.\nSilent wisdom grows each day,\nGuiding us along the way.";
      console.log('   Generated mock text:');
      console.log(`     "${mockText}"`);
      
      results.basicGeneration = {
        prompt: "Write a short poem about artificial intelligence.",
        output: mockText,
        mock: true
      };
    }
    
    // 3.2 Generate Text with Custom Parameters
    console.log('\n📋 3.2 Generate Text with Custom Parameters:');
    
    try {
      const prompt = "Explain the concept of quantum computing.";
      
      const options: TextGenerationOptions = {
        maxTokens: 150,           // Generate more tokens
        temperature: 0.8,         // Slightly more creative
        topP: 0.92,               // Slightly more diverse
        topK: 40,                 // Consider fewer tokens
        repetitionPenalty: 1.2,   // Stronger repetition penalty
        priority: 'HIGH'          // High priority request
      };
      
      console.log(`   Input prompt: "${prompt}"`);
      console.log('   Generation parameters:');
      console.log(`     - maxTokens: ${options.maxTokens}`);
      console.log(`     - temperature: ${options.temperature}`);
      console.log(`     - topP: ${options.topP}`);
      console.log(`     - topK: ${options.topK}`);
      console.log(`     - repetitionPenalty: ${options.repetitionPenalty}`);
      console.log(`     - priority: ${options.priority}`);
      
      const startTime = performance.now();
      const generatedText = await hfTgiUnified.generateText(prompt, options);
      const endTime = performance.now();
      
      console.log('   ✓ Generated text:');
      console.log(`     "${generatedText}"`);
      console.log(`   ✓ Generation time: ${(endTime - startTime).toFixed(2)}ms`);
      
      results.customGeneration = {
        prompt,
        parameters: options,
        output: generatedText,
        time: endTime - startTime
      };
    } catch (error) {
      console.log('   × Error generating text with custom parameters:', error);
      results.customGeneration = { error: String(error) };
    }
    
    // 3.3 Generate Text with Different Models
    console.log('\n📋 3.3 Generate Text with Different Models:');
    
    // Try with different models if available
    const modelResults: Record<string, any> = {};
    
    // Only try the first 2 models to save time
    for (const [modelKey, modelId] of Object.entries(availableModels).slice(0, 2)) {
      try {
        console.log(`   Model: ${modelKey} (${modelId})`);
        
        const prompt = "Summarize the benefits of regular exercise.";
        const options: TextGenerationOptions = {
          model: modelId,
          maxTokens: 100,
          temperature: 0.7
        };
        
        const startTime = performance.now();
        const generatedText = await hfTgiUnified.generateText(prompt, options);
        const endTime = performance.now();
        
        console.log(`   ✓ Generated text (${(endTime - startTime).toFixed(2)}ms):`);
        console.log(`     "${generatedText}"`);
        
        modelResults[modelKey] = {
          modelId,
          prompt,
          output: generatedText,
          time: endTime - startTime
        };
      } catch (error) {
        console.log(`   × Error with model ${modelKey}:`, error);
        modelResults[modelKey] = { error: String(error) };
      }
    }
    
    results.modelComparison = modelResults;
    
    // -------------------------------------------------------------------------------
    // SECTION 4: STREAMING TEXT GENERATION
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 4: STREAMING TEXT GENERATION');
    
    // 4.1 Basic Streaming
    console.log('\n📋 4.1 Basic Streaming:');
    
    try {
      const prompt = "List 3 benefits of cloud computing.";
      console.log(`   Input prompt: "${prompt}"`);
      
      const options: TextGenerationOptions = {
        maxTokens: 100,
        temperature: 0.7,
        stream: true
      };
      
      console.log('   Streaming output:');
      process.stdout.write('     "');
      
      const startTime = performance.now();
      let fullText = '';
      
      const streamGenerator = hfTgiUnified.streamGenerateText(prompt, options);
      
      for await (const chunk of streamGenerator) {
        process.stdout.write(chunk.text);
        fullText += chunk.text;
        
        if (chunk.done) {
          process.stdout.write('"');
          console.log();
        }
      }
      
      const endTime = performance.now();
      console.log(`   ✓ Streaming completed in ${(endTime - startTime).toFixed(2)}ms`);
      
      results.basicStreaming = {
        prompt,
        output: fullText,
        time: endTime - startTime
      };
    } catch (error) {
      console.log('   × Error streaming text:', error);
      results.basicStreaming = { error: String(error) };
    }
    
    // 4.2 Streaming with Callback
    console.log('\n📋 4.2 Streaming with Callback:');
    
    try {
      const prompt = "Write a short story about a robot learning to cook.";
      console.log(`   Input prompt: "${prompt}"`);
      
      // Set up a callback to track streaming progress
      const streamProgress: {chunks: number, tokens: number, text: string} = {
        chunks: 0,
        tokens: 0,
        text: ''
      };
      
      const options: TextGenerationOptions = {
        maxTokens: 120,
        temperature: 0.8,
        stream: true,
        streamCallback: (chunk) => {
          streamProgress.chunks++;
          streamProgress.tokens += chunk.text.split(/\s+/).length; // Rough estimate
          streamProgress.text += chunk.text;
        }
      };
      
      console.log('   Streaming with callback (processing each chunk as it arrives):');
      
      const startTime = performance.now();
      let fullText = '';
      
      const streamGenerator = hfTgiUnified.streamGenerateText(prompt, options);
      
      for await (const chunk of streamGenerator) {
        // Just collect the text, the callback is doing the tracking
        fullText += chunk.text;
        
        if (chunk.done) {
          break;
        }
      }
      
      const endTime = performance.now();
      
      console.log(`   ✓ Streaming stats:`);
      console.log(`     - Chunks received: ${streamProgress.chunks}`);
      console.log(`     - Estimated tokens: ${streamProgress.tokens}`);
      console.log(`     - Total time: ${(endTime - startTime).toFixed(2)}ms`);
      console.log(`   ✓ Generated text: "${fullText}"`);
      
      results.callbackStreaming = {
        prompt,
        chunks: streamProgress.chunks,
        tokens: streamProgress.tokens,
        output: fullText,
        time: endTime - startTime
      };
    } catch (error) {
      console.log('   × Error streaming with callback:', error);
      results.callbackStreaming = { error: String(error) };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 5: CHAT INTERFACE
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 5: CHAT INTERFACE');
    
    // 5.1 Basic Chat
    console.log('\n📋 5.1 Basic Chat:');
    
    try {
      // Set up a simple conversation
      const messages: ChatMessage[] = [
        { role: 'system', content: 'You are a helpful AI assistant.' },
        { role: 'user', content: 'What is the capital of France?' }
      ];
      
      console.log('   System: You are a helpful AI assistant.');
      console.log('   User: What is the capital of France?');
      
      const startTime = performance.now();
      const response = await hfTgiUnified.chat(messages);
      const endTime = performance.now();
      
      console.log(`   Assistant: ${response.text}`);
      console.log(`   ✓ Chat response generated in ${(endTime - startTime).toFixed(2)}ms`);
      
      results.basicChat = {
        messages,
        response: response.text,
        time: endTime - startTime
      };
    } catch (error) {
      console.log('   × Error in basic chat:', error);
      results.basicChat = { error: String(error) };
    }
    
    // 5.2 Chat with Parameters
    console.log('\n📋 5.2 Chat with Parameters:');
    
    try {
      // Set up a conversation with more turns
      const messages: ChatMessage[] = [
        { role: 'system', content: 'You are a creative storyteller.' },
        { role: 'user', content: 'Tell me a short story about a magic library.' },
        { role: 'assistant', content: 'Once upon a time, there was a library where books would whisper their stories to visitors. The librarian was a mysterious old woman who could speak to the books.' },
        { role: 'user', content: 'What happened to someone who borrowed a book and never returned it?' }
      ];
      
      const chatOptions: ChatGenerationOptions = {
        maxTokens: 150,
        temperature: 0.8,
        topP: 0.92,
        promptTemplate: 'instruction', // Use instruction prompt template
        systemMessage: 'You are a creative storyteller who specializes in fantasy tales.'
      };
      
      console.log('   Multi-turn conversation with system message and prompt template:');
      console.log('   System: You are a creative storyteller who specializes in fantasy tales.');
      console.log('   User: Tell me a short story about a magic library.');
      console.log('   Assistant: Once upon a time, there was a library where books would whisper their stories to visitors. The librarian was a mysterious old woman who could speak to the books.');
      console.log('   User: What happened to someone who borrowed a book and never returned it?');
      
      const startTime = performance.now();
      const response = await hfTgiUnified.chat(messages, chatOptions);
      const endTime = performance.now();
      
      console.log(`   Assistant: ${response.text}`);
      console.log(`   ✓ Chat response generated in ${(endTime - startTime).toFixed(2)}ms`);
      console.log(`   ✓ Using prompt template: ${chatOptions.promptTemplate}`);
      
      results.chatWithParameters = {
        messages,
        options: chatOptions,
        response: response.text,
        time: endTime - startTime
      };
    } catch (error) {
      console.log('   × Error in chat with parameters:', error);
      results.chatWithParameters = { error: String(error) };
    }
    
    // 5.3 Streaming Chat
    console.log('\n📋 5.3 Streaming Chat:');
    
    try {
      // Simple conversation for streaming
      const messages: ChatMessage[] = [
        { role: 'system', content: 'You are a helpful AI assistant.' },
        { role: 'user', content: 'Write a short poem about coding.' }
      ];
      
      const chatOptions: ChatGenerationOptions = {
        maxTokens: 100,
        temperature: 0.7
      };
      
      console.log('   System: You are a helpful AI assistant.');
      console.log('   User: Write a short poem about coding.');
      console.log('   Assistant (streaming): ');
      process.stdout.write('     ');
      
      const startTime = performance.now();
      let fullResponse = '';
      
      const chatStream = hfTgiUnified.streamChat(messages, chatOptions);
      
      for await (const chunk of chatStream) {
        process.stdout.write(chunk.text);
        fullResponse += chunk.text;
        
        if (chunk.done) {
          console.log();
        }
      }
      
      const endTime = performance.now();
      
      console.log(`   ✓ Streaming chat completed in ${(endTime - startTime).toFixed(2)}ms`);
      
      results.streamingChat = {
        messages,
        response: fullResponse,
        time: endTime - startTime
      };
    } catch (error) {
      console.log('   × Error in streaming chat:', error);
      results.streamingChat = { error: String(error) };
    }
    
    // 5.4 Different Prompt Templates
    console.log('\n📋 5.4 Different Prompt Templates:');
    
    // Test with different prompt templates
    const promptTemplates = ['default', 'instruction', 'chat', 'llama2', 'mistral', 'chatml'];
    const templateResults: Record<string, any> = {};
    
    // Only test the first few templates to save time
    for (const template of promptTemplates.slice(0, 3)) {
      try {
        console.log(`   Template: ${template}`);
        
        const messages: ChatMessage[] = [
          { role: 'system', content: 'You are a helpful AI assistant.' },
          { role: 'user', content: 'What is cloud computing?' }
        ];
        
        const chatOptions: ChatGenerationOptions = {
          maxTokens: 80,
          temperature: 0.7,
          promptTemplate: template
        };
        
        const startTime = performance.now();
        const response = await hfTgiUnified.chat(messages, chatOptions);
        const endTime = performance.now();
        
        console.log(`   ✓ Response (${(endTime - startTime).toFixed(2)}ms): "${response.text.substring(0, 100)}${response.text.length > 100 ? '...' : ''}"`);
        
        templateResults[template] = {
          response: response.text,
          time: endTime - startTime
        };
      } catch (error) {
        console.log(`   × Error with template ${template}:`, error);
        templateResults[template] = { error: String(error) };
      }
    }
    
    results.promptTemplates = templateResults;
    
    // -------------------------------------------------------------------------------
    // SECTION 6: CONTAINER MANAGEMENT (DEMONSTRATION MODE)
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 6: CONTAINER MANAGEMENT (DEMONSTRATION MODE)');
    console.log('   ⚠ Container operations are shown for demonstration only');
    console.log('   ⚠ Actual container start/stop is commented out to prevent unintended effects');
    
    // 6.1 Container Configuration
    console.log('\n📋 6.1 Container Configuration:');
    
    const deployConfig: DeploymentConfig = {
      dockerRegistry: 'ghcr.io/huggingface/text-generation-inference',
      containerTag: 'latest',
      gpuDevice: '0',
      modelId: 'facebook/opt-125m', // Use a small model for demonstration
      port: 8080,
      env: {
        'HF_API_TOKEN': apiKey || ''
      },
      volumes: ['./cache:/data'],
      network: 'bridge',
      parameters: ['--max-batch-size=32'],
      maxInputLength: 1024,
      disableGpu: false
    };
    
    console.log('   ✓ Container configuration prepared:');
    console.log(`     - Image: ${deployConfig.dockerRegistry}:${deployConfig.containerTag}`);
    console.log(`     - Model: ${deployConfig.modelId}`);
    console.log(`     - Port: ${deployConfig.port}`);
    console.log(`     - GPU: ${deployConfig.gpuDevice}`);
    console.log(`     - Max input length: ${deployConfig.maxInputLength}`);
    console.log(`     - Additional parameters: ${deployConfig.parameters.join(' ')}`);
    
    results.containerConfig = {
      image: `${deployConfig.dockerRegistry}:${deployConfig.containerTag}`,
      port: deployConfig.port,
      gpu: deployConfig.gpuDevice,
      model: deployConfig.modelId,
      parameters: deployConfig.parameters
    };
    
    // 6.2 Container Mode Switching
    console.log('\n📋 6.2 Container Mode Switching:');
    
    // Switch to container mode
    console.log('   Switching to container mode');
    hfTgiUnified.setMode(true);
    console.log(`   ✓ Current mode: ${hfTgiUnified.getMode()}`);
    
    // Switch back to API mode
    console.log('   Switching back to API mode');
    hfTgiUnified.setMode(false);
    console.log(`   ✓ Current mode: ${hfTgiUnified.getMode()}`);
    
    results.containerModeSwitching = {
      finalMode: hfTgiUnified.getMode()
    };
    
    // 6.3 Container Operations (Demonstration Only)
    console.log('\n📋 6.3 Container Operations (Demonstration Only):');
    
    // Show container operation example (without actually executing)
    console.log('   ℹ Container start/stop operations are commented out');
    console.log('   ℹ In a real application, you would use:');
    console.log('     const containerInfo = await hfTgiUnified.startContainer(deployConfig);');
    console.log('     // Generate text using container');
    console.log('     const text = await hfTgiUnified.generateText("Hello, world!");');
    console.log('     // Stop the container when done');
    console.log('     const stopped = await hfTgiUnified.stopContainer();');
    
    // Output the container command that would be executed
    try {
      const imageName = `${deployConfig.dockerRegistry}:${deployConfig.containerTag}`;
      const volumes = deployConfig.volumes?.length 
        ? deployConfig.volumes.map(v => `-v ${v}`).join(' ') 
        : '';
      
      const envVars = Object.entries(deployConfig.env || {})
        .map(([key, value]) => `-e ${key}=${value}`)
        .join(' ');
      
      const gpuArgs = deployConfig.disableGpu ? '' : 
        (deployConfig.gpuDevice ? `--gpus device=${deployConfig.gpuDevice}` : '--gpus all');
      
      const networkArg = deployConfig.network 
        ? `--network=${deployConfig.network}` 
        : '';
      
      const additionalParams = deployConfig.parameters?.length
        ? deployConfig.parameters.join(' ')
        : '';
      
      const maxInputLengthArg = deployConfig.maxInputLength
        ? `--max-input-length ${deployConfig.maxInputLength}`
        : '';
      
      const containerName = `hf-tgi-example`;
      const command = `docker run -d --name ${containerName} \
        -p ${deployConfig.port}:80 \
        ${gpuArgs} \
        ${envVars} \
        ${volumes} \
        ${networkArg} \
        ${imageName} \
        --model-id ${deployConfig.modelId} \
        ${maxInputLengthArg} \
        ${additionalParams}`;
      
      console.log('   ℹ Example container command:');
      console.log(`     ${command.replace(/\n\s+/g, ' ')}`);
      
      results.containerCommand = command.replace(/\n\s+/g, ' ');
    } catch (error) {
      console.log('   × Error generating container command:', error);
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 7: ADVANCED FEATURES
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 7: ADVANCED FEATURES');
    
    // 7.1 Stop Sequences
    console.log('\n📋 7.1 Stop Sequences:');
    
    try {
      const prompt = "List the top 5 programming languages in 2025:";
      
      const options: TextGenerationOptions = {
        maxTokens: 200,
        temperature: 0.7,
        stopSequences: ['5.', '5)', '5:'] // Stop after item 5
      };
      
      console.log(`   Input prompt: "${prompt}"`);
      console.log(`   Stop sequences: ${JSON.stringify(options.stopSequences)}`);
      
      const response = await hfTgiUnified.generateText(prompt, options);
      
      console.log('   ✓ Generated text with stop sequences:');
      console.log(`     "${response}"`);
      
      results.stopSequences = {
        prompt,
        stopSequences: options.stopSequences,
        response
      };
    } catch (error) {
      console.log('   × Error using stop sequences:', error);
      results.stopSequences = { error: String(error) };
    }
    
    // 7.2 Low Temperature Comparison
    console.log('\n📋 7.2 Low vs High Temperature Comparison:');
    
    try {
      const prompt = "Write a tagline for a new smartphone.";
      console.log(`   Input prompt: "${prompt}"`);
      
      // Low temperature (more deterministic)
      const lowTempOptions: TextGenerationOptions = {
        maxTokens: 30,
        temperature: 0.1,
        topP: 0.95
      };
      
      const highTempOptions: TextGenerationOptions = {
        maxTokens: 30,
        temperature: 0.9,
        topP: 0.95
      };
      
      const lowTempResponse = await hfTgiUnified.generateText(prompt, lowTempOptions);
      const highTempResponse = await hfTgiUnified.generateText(prompt, highTempOptions);
      
      console.log('   ✓ Low temperature (0.1) response:');
      console.log(`     "${lowTempResponse}"`);
      console.log('   ✓ High temperature (0.9) response:');
      console.log(`     "${highTempResponse}"`);
      
      results.temperatureComparison = {
        prompt,
        lowTemp: {
          temperature: lowTempOptions.temperature,
          response: lowTempResponse
        },
        highTemp: {
          temperature: highTempOptions.temperature,
          response: highTempResponse
        }
      };
    } catch (error) {
      console.log('   × Error in temperature comparison:', error);
      results.temperatureComparison = { error: String(error) };
    }
    
    // 7.3 Model Family-Specific Formatting
    console.log('\n📋 7.3 Model Family-Specific Formatting:');
    
    try {
      // Demonstrate how different model families use different prompt templates
      const modelFamilies = {
        't5': { model: 'google/flan-t5-small', template: 'instruction' },
        'llama': { model: 'meta-llama/Llama-2-7b-chat-hf', template: 'llama2' },
        'mistral': { model: 'mistralai/Mistral-7B-Instruct-v0.1', template: 'mistral' },
        'falcon': { model: 'tiiuae/falcon-7b', template: 'falcon' }
      };
      
      console.log('   Automatic model family detection and prompt formatting:');
      
      // Only test the first model to save time
      const family = Object.entries(modelFamilies)[0];
      const [familyName, { model, template }] = family;
      
      console.log(`   Testing family: ${familyName} with model ${model}`);
      console.log(`   Expected template: ${template}`);
      
      const messages: ChatMessage[] = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is machine learning?' }
      ];
      
      // Let the backend auto-detect the template
      const chatOptions: ChatGenerationOptions = {
        model,
        maxTokens: 50,
        temperature: 0.7
      };
      
      try {
        // This may fail if the model is not available, but we're just demonstrating the concept
        // In practice, you would choose a model that you have access to
        const response = await hfTgiUnified.chat(messages, chatOptions);
        
        console.log(`   ✓ Response: "${response.text.substring(0, 100)}${response.text.length > 100 ? '...' : ''}"`);
        
        results.modelFamilyFormatting = {
          family: familyName,
          model,
          expectedTemplate: template,
          response: response.text
        };
      } catch (error) {
        console.log(`   ℹ Could not test with actual model (expected in demo mode)`);
        console.log(`   ℹ In practice, the backend automatically selects the right template for each model family`);
        
        results.modelFamilyFormatting = {
          family: familyName,
          model,
          expectedTemplate: template,
          demo: true
        };
      }
    } catch (error) {
      console.log('   × Error in model family formatting:', error);
      results.modelFamilyFormatting = { error: String(error) };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 8: PERFORMANCE BENCHMARKING
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 8: PERFORMANCE BENCHMARKING');
    
    // 8.1 Basic Benchmark
    console.log('\n📋 8.1 Basic Benchmark:');
    
    try {
      const benchmarkOptions = {
        iterations: 3,           // Number of iterations for reliable results
        model: defaultModel,     // Model to benchmark
        maxTokens: 50            // Maximum tokens to generate
      };
      
      console.log(`   Running benchmark with ${benchmarkOptions.iterations} iterations`);
      console.log(`   Model: ${benchmarkOptions.model}`);
      console.log(`   Max tokens: ${benchmarkOptions.maxTokens}`);
      
      const benchmarkResults = await hfTgiUnified.runBenchmark(benchmarkOptions);
      
      console.log('   ✓ Benchmark results:');
      console.log(`     - Single generation time: ${benchmarkResults.singleGenerationTime.toFixed(2)} ms`);
      console.log(`     - Tokens per second: ${benchmarkResults.tokensPerSecond.toFixed(2)}`);
      console.log(`     - Generated tokens: ${benchmarkResults.generatedTokens.toFixed(2)}`);
      console.log(`     - Input tokens: ${benchmarkResults.inputTokens}`);
      
      results.basicBenchmark = benchmarkResults;
    } catch (error) {
      console.log('   × Error running benchmark:', error);
      results.basicBenchmark = { error: String(error) };
    }
    
    // 8.2 Model Comparison Benchmark
    console.log('\n📋 8.2 Model Comparison Benchmark:');
    
    const modelBenchmarks: Record<string, PerformanceMetrics> = {};
    const modelsToCompare = Object.entries(availableModels).slice(0, 2); // Limit to 2 models for demo
    
    try {
      for (const [modelKey, modelId] of modelsToCompare) {
        console.log(`   Benchmarking model: ${modelKey} (${modelId})`);
        
        try {
          const benchmarkOptions = {
            iterations: 2,
            model: modelId,
            maxTokens: 30
          };
          
          const result = await hfTgiUnified.runBenchmark(benchmarkOptions);
          
          console.log(`   ✓ ${modelKey}: ${result.tokensPerSecond.toFixed(2)} tokens/sec`);
          modelBenchmarks[modelKey] = result;
        } catch (error) {
          console.log(`   × Error benchmarking ${modelKey}:`, error);
        }
      }
      
      // Find the best performing model
      if (Object.keys(modelBenchmarks).length > 0) {
        const bestModel = Object.entries(modelBenchmarks).reduce(
          (best, [model, metrics]) => 
            metrics.tokensPerSecond > best.metrics.tokensPerSecond 
              ? { model, metrics } 
              : best,
          { model: '', metrics: { tokensPerSecond: 0 } as PerformanceMetrics }
        );
        
        console.log(`   ✓ Best performing model: ${bestModel.model} with ${bestModel.metrics.tokensPerSecond.toFixed(2)} tokens/sec`);
      }
      
      results.modelBenchmarks = modelBenchmarks;
    } catch (error) {
      console.log('   × Error in model comparison benchmark:', error);
      results.modelBenchmarks = { error: String(error) };
    }
    
    // 8.3 Performance Optimization Suggestions
    console.log('\n📋 8.3 Performance Optimization Suggestions:');
    
    // Based on benchmarks, provide optimization suggestions
    console.log('   Performance optimization suggestions:');
    console.log('   1. Use container mode for production deployments:');
    console.log('      - Lower latency due to avoiding API overhead');
    console.log('      - More consistent performance without rate limits');
    console.log('      - Better for high-volume applications');
    console.log('   2. Choose models based on your performance requirements:');
    console.log('      - Smaller models (flan-t5-small, opt-125m) for low latency');
    console.log('      - Larger models (llama-2-7b, mistral-7b) for better quality');
    console.log('   3. Optimize generation parameters:');
    console.log('      - Lower max_tokens for faster response times');
    console.log('      - Use appropriate temperature for your use case');
    console.log('      - Consider using stop sequences to end generation early');
    console.log('   4. Hardware considerations for container mode:');
    console.log('      - Use GPU acceleration for larger models');
    console.log('      - Ensure sufficient memory for your chosen model');
    console.log('      - Consider model quantization for efficiency');
    
    // -------------------------------------------------------------------------------
    // SECTION 9: ERROR HANDLING
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 9: ERROR HANDLING');
    
    // 9.1 Common Errors
    console.log('\n📋 9.1 Common Errors:');
    
    console.log('   Common error types and how to handle them:');
    console.log('   1. Model not found error:');
    
    try {
      await hfTgiUnified.generateText("Test prompt", { model: 'non-existent-model' });
    } catch (error) {
      console.log(`      ✓ ${error instanceof Error ? error.message : String(error)}`);
      console.log('      ✓ Handle by checking model compatibility before use');
    }
    
    console.log('   2. API key errors:');
    console.log('      ✓ "Authorization error: Invalid API key or insufficient permissions"');
    console.log('      ✓ Handle by validating API key and using environment variables');
    
    console.log('   3. Rate limiting errors:');
    console.log('      ✓ "The model is currently loading or busy. Try again later."');
    console.log('      ✓ Handle with exponential backoff and retry strategy');
    
    console.log('   4. Input validation errors:');
    console.log('      ✓ "Chat messages array cannot be empty"');
    console.log('      ✓ Handle by validating inputs before calling API');
    
    // 9.2 Robust Error Handling Pattern
    console.log('\n📋 9.2 Robust Error Handling Pattern:');
    
    try {
      // Example of robust error handling
      const robustGenerateText = async (prompt: string, options: TextGenerationOptions = {}, retries = 3): Promise<string> => {
        try {
          return await hfTgiUnified.generateText(prompt, options);
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          
          // Check for retriable errors
          if (
            errorMessage.includes('loading') || 
            errorMessage.includes('busy') || 
            errorMessage.includes('rate limit') ||
            errorMessage.includes('timeout')
          ) {
            if (retries > 0) {
              // Exponential backoff
              const delay = 1000 * Math.pow(2, 3 - retries);
              console.log(`      Retrying after ${delay}ms (${retries} retries left)...`);
              await new Promise(resolve => setTimeout(resolve, delay));
              return robustGenerateText(prompt, options, retries - 1);
            }
          }
          
          // Re-throw non-retriable errors or when out of retries
          throw error;
        }
      };
      
      console.log('   ✓ Implemented robust error handling with:');
      console.log('      - Typed error classification');
      console.log('      - Exponential backoff for retriable errors');
      console.log('      - Maximum retry limit');
      
      results.errorHandling = {
        robustPattern: true
      };
    } catch (error) {
      console.log('   × Error in robust error handling pattern:', error);
      results.errorHandling = { error: String(error) };
    }
    
    // 9.3 Circuit Breaker Pattern
    console.log('\n📋 9.3 Circuit Breaker Pattern:');
    
    try {
      console.log('   The HF TGI Unified backend uses a circuit breaker pattern:');
      console.log('   ✓ Prevents cascading failures by stopping requests after consecutive errors');
      console.log('   ✓ Automatically resets after a cooldown period');
      console.log('   ✓ Configurable failure threshold and reset timeout');
      console.log('   ✓ Used internally by the backend for API resilience');
      
      console.log('   Circuit breaker configuration:');
      console.log('   - Failure threshold: 3 consecutive failures');
      console.log('   - Reset timeout: 30000ms (30 seconds)');
      
      results.circuitBreaker = {
        enabled: true,
        failureThreshold: 3,
        resetTimeout: 30000
      };
    } catch (error) {
      console.log('   × Error in circuit breaker pattern:', error);
      results.circuitBreaker = { error: String(error) };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 10: ADVANCED USE CASES
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 10: ADVANCED USE CASES');
    
    // 10.1 Question Answering System
    console.log('\n📋 10.1 Question Answering System:');
    
    try {
      const buildQASystem = async (context: string, question: string): Promise<string> => {
        const prompt = `
Context: ${context}

Question: ${question}

Answer:
`;
        
        const options: TextGenerationOptions = {
          maxTokens: 100,
          temperature: 0.3, // Lower temperature for more factual answers
          topP: 0.95,
          repetitionPenalty: 1.2
        };
        
        return await hfTgiUnified.generateText(prompt, options);
      };
      
      const context = `
The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. 
The U.S. National Aeronautics and Space Administration (NASA) led development of the telescope in collaboration 
with the European Space Agency (ESA) and the Canadian Space Agency (CSA). The telescope is named after James E. Webb, 
who was the administrator of NASA from 1961 to 1968 during the Mercury, Gemini, and Apollo programs.
The telescope was launched on December 25, 2021, on an Ariane 5 rocket from Kourou, French Guiana, 
and arrived at the Sun–Earth L2 Lagrange point in January 2022.
`;
      
      const question = "When was the James Webb Space Telescope launched?";
      
      console.log('   Building a question answering system:');
      console.log(`   Question: ${question}`);
      
      const answer = await buildQASystem(context, question);
      
      console.log(`   Answer: ${answer}`);
      
      results.questionAnswering = {
        context: context.trim(),
        question,
        answer
      };
    } catch (error) {
      console.log('   × Error in question answering system:', error);
      results.questionAnswering = { error: String(error) };
    }
    
    // 10.2 Multi-Turn Conversation Agent
    console.log('\n📋 10.2 Multi-Turn Conversation Agent:');
    
    try {
      interface ConversationAgent {
        name: string;
        persona: string;
        conversation: ChatMessage[];
        addUserMessage: (message: string) => void;
        getResponse: () => Promise<string>;
        getConversationHistory: () => ChatMessage[];
      }
      
      // Create a conversation agent
      const createAgent = (name: string, persona: string): ConversationAgent => {
        const agent: ConversationAgent = {
          name,
          persona,
          conversation: [
            { role: 'system', content: persona }
          ],
          addUserMessage(message: string) {
            this.conversation.push({ role: 'user', content: message });
          },
          async getResponse() {
            // Generate a response from the model
            const chatOptions: ChatGenerationOptions = {
              maxTokens: 150,
              temperature: 0.7,
              topP: 0.9,
              systemMessage: this.persona
            };
            
            const response = await hfTgiUnified.chat(this.conversation, chatOptions);
            
            // Add the response to the conversation history
            this.conversation.push({ role: 'assistant', content: response.text });
            
            return response.text;
          },
          getConversationHistory() {
            return this.conversation;
          }
        };
        
        return agent;
      };
      
      // Create a travel agent
      const travelAgent = createAgent(
        'TravelBot',
        'You are a helpful travel assistant. You provide concise and specific travel recommendations based on user preferences.'
      );
      
      console.log('   Creating a travel assistant conversation agent:');
      console.log(`   Agent: ${travelAgent.name}`);
      console.log(`   Persona: ${travelAgent.persona}`);
      
      // First user message
      travelAgent.addUserMessage("I want to visit a warm place with beaches in December.");
      console.log('   User: I want to visit a warm place with beaches in December.');
      
      // Get first response
      const response1 = await travelAgent.getResponse();
      console.log(`   ${travelAgent.name}: ${response1}`);
      
      // Second user message
      travelAgent.addUserMessage("I prefer places where English is commonly spoken. What do you recommend?");
      console.log('   User: I prefer places where English is commonly spoken. What do you recommend?');
      
      // Get second response
      const response2 = await travelAgent.getResponse();
      console.log(`   ${travelAgent.name}: ${response2}`);
      
      results.conversationAgent = {
        agent: travelAgent.name,
        persona: travelAgent.persona,
        conversation: travelAgent.getConversationHistory()
      };
    } catch (error) {
      console.log('   × Error in multi-turn conversation agent:', error);
      results.conversationAgent = { error: String(error) };
    }
    
    // 10.3 Document Summarization
    console.log('\n📋 10.3 Document Summarization:');
    
    try {
      const summarizeDocument = async (document: string, maxLength: number = 150): Promise<string> => {
        const prompt = `
Summarize the following document in a concise way, highlighting the key points. 
Keep the summary to around ${maxLength} words.

Document:
${document}

Summary:
`;
        
        const options: TextGenerationOptions = {
          maxTokens: maxLength * 2, // Allow enough tokens for a good summary
          temperature: 0.5,
          topP: 0.95
        };
        
        return await hfTgiUnified.generateText(prompt, options);
      };
      
      const document = `
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by animals including humans. 
AI applications include advanced web search engines, recommendation systems, language translation, autonomous driving, and creating art.

Machine learning is a subset of AI where computers can learn and improve from experience without being explicitly programmed. 
In recent years, deep learning, a form of machine learning based on artificial neural networks, has led to significant breakthroughs 
in image recognition, natural language processing, and reinforcement learning.

The field of AI raises serious ethical concerns. The development of superintelligent AI systems could pose existential risks, 
while increasing automation may lead to significant economic disruption. Other AI safety considerations include preventing AI 
from being programmed for destructive uses like autonomous weapons, ensuring AI systems don't develop harmful emergent behaviors, 
and addressing issues of AI alignment with human values.

Despite these concerns, AI has the potential to solve many of humanity's most pressing challenges, from climate change to disease. 
The responsible development of AI systems represents one of the most important technological frontiers of our time.
`;
      
      console.log('   Document summarization example:');
      console.log('   Original document length:', document.split(/\s+/).length, 'words');
      
      const summary = await summarizeDocument(document.trim(), 75);
      
      console.log(`   Summary (target ~75 words): ${summary}`);
      console.log('   Summary length:', summary.split(/\s+/).length, 'words');
      
      results.documentSummarization = {
        documentLength: document.split(/\s+/).length,
        summary,
        summaryLength: summary.split(/\s+/).length
      };
    } catch (error) {
      console.log('   × Error in document summarization:', error);
      results.documentSummarization = { error: String(error) };
    }
    
    // -------------------------------------------------------------------------------
    // CONCLUSION
    // -------------------------------------------------------------------------------
    console.log('\n==========================================================');
    console.log('COMPREHENSIVE EXAMPLE COMPLETED SUCCESSFULLY');
    console.log('==========================================================');
    
    console.log('\nSummary of Results:');
    console.log(`✅ Model Used: ${hfTgiUnified.getDefaultModel()}`);
    console.log(`✅ Mode: ${hfTgiUnified.getMode()}`);
    
    if (results.basicBenchmark && !results.basicBenchmark.error) {
      console.log(`✅ Performance: ${results.basicBenchmark.tokensPerSecond.toFixed(2)} tokens/second`);
    }
    
    console.log('\nDemonstrated Features:');
    console.log('✅ Text Generation with Various Parameters');
    console.log('✅ Streaming Text Generation');
    console.log('✅ Chat Interface with Different Templates');
    console.log('✅ Container Management');
    console.log('✅ Performance Benchmarking');
    console.log('✅ Error Handling and Circuit Breaker Pattern');
    console.log('✅ Advanced Use Cases for AI Applications');
    
    console.log('\nFor detailed API documentation, refer to:');
    console.log('- HF_TGI_UNIFIED_USAGE.md in the docs/api_backends directory');
    
  } catch (error) {
    console.error('\n❌ ERROR IN COMPREHENSIVE EXAMPLE:', error);
    console.error('Please check your API key and network connectivity');
  }
}

// Call the main function
if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error in main:', error);
    process.exit(1);
  });
}

// Export for testing or module usage
export default main;