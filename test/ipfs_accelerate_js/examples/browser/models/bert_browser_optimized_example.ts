/**
 * Example of using BERT with browser-specific optimizations
 */

import { WebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { Bert, BertConfig } from '../../../src/model/transformers/bert';
import { BrowserType, detectBrowserType } from '../../../src/hardware/webgpu/browser_optimized_operations';

/**
 * Run BERT model with browser-specific optimizations
 */
async function runBrowserOptimizedBert() {
  try {
    // Initialize WebGPU backend
    console.log('Initializing WebGPU backend...');
    const backend = new WebGPUBackend();
    await backend.initialize();
    
    // Detect browser type
    const browserType = detectBrowserType();
    const browserNames = {
      [BrowserType.CHROME]: 'Google Chrome',
      [BrowserType.FIREFOX]: 'Mozilla Firefox',
      [BrowserType.SAFARI]: 'Apple Safari',
      [BrowserType.EDGE]: 'Microsoft Edge'
    };
    console.log(`Detected browser: ${browserNames[browserType] || 'Unknown'}`);
    
    // Configure BERT with browser-specific optimizations
    const config: BertConfig = {
      modelId: 'bert-base-uncased',
      vocabSize: 30522,
      hiddenSize: 768,
      numLayers: 12,
      numHeads: 12,
      intermediateSize: 3072,
      maxPositions: 512,
      layerNormEps: 1e-12,
      backendPreference: ['webgpu', 'webnn', 'cpu'],
      useOptimizedOps: true,
      useBrowserOptimizations: true,  // Enable browser-specific optimizations
      browserType: browserType,       // Use detected browser type
      useOperationFusion: true,       // Enable operation fusion for better performance
      attentionDropout: 0.1
    };
    
    console.log('Creating BERT model with the following configuration:');
    console.log(JSON.stringify(config, null, 2));
    
    // Create BERT model
    const bert = new Bert(backend, config);
    
    // Initialize model (load weights)
    console.log('Initializing BERT model...');
    await bert.initialize();
    
    // Example input text
    const inputText = "Hello, world! This is a test of BERT with browser-specific optimizations.";
    
    // Tokenize input text
    console.log(`Tokenizing input text: "${inputText}"`);
    const tokenizedInput = await bert.tokenize(inputText);
    
    console.log(`Tokenized with ${tokenizedInput.inputIds.length} tokens`);
    
    // Run inference
    console.log('Running inference with browser-specific optimizations...');
    const startTime = performance.now();
    const output = await bert.process(tokenizedInput);
    const endTime = performance.now();
    
    console.log(`Inference completed in ${(endTime - startTime).toFixed(2)} ms`);
    console.log(`Model: ${output.model}`);
    console.log(`Backend: ${output.backend}`);
    
    // Display output dimensions
    console.log(`Output shape: [${output.lastHiddenState.length}, ${output.lastHiddenState[0].length}]`);
    
    // Display pooled output (CLS token representation) dimensions
    console.log(`Pooled output shape: [${output.pooledOutput.length}]`);
    
    // Cleanup
    await bert.dispose();
    await backend.dispose();
    
  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the example
runBrowserOptimizedBert();

// Add DOM output for browser environment
if (typeof document !== 'undefined') {
  const outputElement = document.getElementById('output');
  if (outputElement) {
    // Override console.log to also output to the DOM
    const originalLog = console.log;
    const originalError = console.error;
    
    console.log = function(...args) {
      originalLog.apply(console, args);
      outputElement.innerHTML += args.join(' ') + '<br>';
    };
    
    console.error = function(...args) {
      originalError.apply(console, args);
      outputElement.innerHTML += '<span style="color:red">' + args.join(' ') + '</span><br>';
    };
  }
}