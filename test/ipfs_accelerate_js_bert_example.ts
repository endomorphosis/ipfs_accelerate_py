/**
 * Example demonstrating the browser-optimized WebGPU accelerated BERT model
 * 
 * This example shows how to load and run inference with the BERT model
 * using browser-specific WebGPU optimizations for maximum performance.
 */

import { WebGPUTensorSharing } from './ipfs_accelerate_js_webgpu_tensor_sharing.ts';
import { StorageManager } from './ipfs_accelerate_js_storage_manager.ts';
import { WebGPUOptimizedBERT, BERTConfig, BERTEmbedding } from './ipfs_accelerate_js_bert_optimized.ts';
import { getOptimizedShader, BrowserCapabilities } from './ipfs_accelerate_js_browser_optimized_shaders.ts';

/**
 * Set up the WebGPU device and capabilities
 */
async function setupWebGPU(): Promise<GPUDevice | null> {
  if (!navigator.gpu) {
    console.error('WebGPU not supported in this browser');
    return null;
  }
  
  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });
    
    if (!adapter) {
      console.error('No WebGPU adapter found');
      return null;
    }
    
    const device = await adapter.requestDevice({
      requiredFeatures: ['shader-f16'],
      requiredLimits: {
        maxBufferSize: 1 << 30, // 1GB
        maxStorageBufferBindingSize: 1 << 30, // 1GB
      }
    });
    
    return device;
  } catch (error) {
    console.error('Error setting up WebGPU:', error);
    return null;
  }
}

/**
 * Basic BERT example for text embeddings with browser-optimized WebGPU acceleration
 */
export async function runBasicBERTExample(text: string): Promise<Float32Array> {
  // Set up WebGPU
  const device = await setupWebGPU();
  if (!device) {
    throw new Error('WebGPU not supported or failed to initialize');
  }
  
  console.log('WebGPU device initialized');
  
  // Initialize tensor sharing system
  const tensorSharing = new WebGPUTensorSharing();
  await tensorSharing.initialize(device);
  
  // Get browser capabilities for logging
  const capabilities = await tensorSharing.getBrowserCapabilities();
  console.log('Browser capabilities detected:', capabilities);
  
  // Initialize storage manager
  const storageManager = new StorageManager('bert-models');
  await storageManager.initialize();
  
  // Check if model is already in storage, otherwise fetch and store
  const modelId = 'bert-base-uncased';
  const modelExists = await storageManager.modelExists(modelId);
  
  if (!modelExists) {
    console.log('Model not found in storage, downloading...');
    // In a real application, you would fetch the model weights here
    // and store them using storageManager.storeModelWeights()
    
    // For this example, we'll assume the model is already in storage
    throw new Error('Model not found in storage. Please download and store the model first.');
  }
  
  // Create BERT model configuration
  const bertConfig: BERTConfig = {
    vocabSize: 30522,
    hiddenSize: 768,
    numLayers: 12,
    numHeads: 12,
    intermediateSize: 3072,
    maxSequenceLength: 512,
    quantization: {
      enabled: true,
      bits: 8,
      blockSize: 32
    },
    useOptimizedAttention: true,
    modelId: modelId,
    taskType: 'embedding'
  };
  
  // Initialize BERT model
  console.log('Initializing BERT model with browser-optimized WebGPU acceleration');
  const model = new WebGPUOptimizedBERT(bertConfig, tensorSharing, storageManager);
  await model.initialize();
  
  // Show model info
  const modelInfo = model.getModelInfo();
  console.log('Model info:', modelInfo);
  
  // Run inference
  console.log('Running inference with browser-optimized WebGPU acceleration');
  console.time('inference');
  const embedding = await model.predict(text);
  console.timeEnd('inference');
  
  // Show performance metrics
  const metrics = model.getPerformanceMetrics();
  console.log('Performance metrics:', metrics);
  
  // Clean up resources
  await model.dispose();
  await tensorSharing.dispose();
  
  return embedding as Float32Array;
}

/**
 * Compare browser-optimized WebGPU implementation with standard implementation
 */
export async function runPerformanceComparison(text: string): Promise<{
  optimizedTime: number;
  standardTime: number;
  speedup: number;
  browserInfo: BrowserCapabilities;
  operationBreakdown: Record<string, { optimized: number; standard: number; improvement: number }>
}> {
  // Set up WebGPU
  const device = await setupWebGPU();
  if (!device) {
    throw new Error('WebGPU not supported or failed to initialize');
  }
  
  // Initialize tensor sharing system with browser optimizations
  const optimizedTensorSharing = new WebGPUTensorSharing();
  await optimizedTensorSharing.initialize(device, {
    enableOptimizations: true,
    precompileShaders: true,
    optimizationLevel: 'maximum'
  });
  
  // Initialize tensor sharing system without browser optimizations
  const standardTensorSharing = new WebGPUTensorSharing();
  await standardTensorSharing.initialize(device, {
    enableOptimizations: false,
    precompileShaders: false,
    optimizationLevel: 'minimum'
  });
  
  // Get browser capabilities for reporting
  const browserCapabilities = await optimizedTensorSharing.getBrowserCapabilities();
  
  // Initialize storage manager
  const storageManager = new StorageManager('bert-models');
  await storageManager.initialize();
  
  // Create BERT model configuration
  const bertConfig: BERTConfig = {
    vocabSize: 30522,
    hiddenSize: 768,
    numLayers: 12,
    numHeads: 12,
    intermediateSize: 3072,
    maxSequenceLength: 512,
    quantization: {
      enabled: true,
      bits: 8
    },
    useOptimizedAttention: true,
    modelId: 'bert-base-uncased',
    taskType: 'embedding'
  };
  
  // Initialize optimized BERT model
  const optimizedModel = new WebGPUOptimizedBERT(
    bertConfig, 
    optimizedTensorSharing, 
    storageManager
  );
  await optimizedModel.initialize();
  
  // Initialize standard BERT model 
  const standardModel = new WebGPUOptimizedBERT(
    { ...bertConfig, useOptimizedAttention: false },
    standardTensorSharing,
    storageManager
  );
  await standardModel.initialize();
  
  // Warm-up runs
  console.log('Performing warm-up runs...');
  await optimizedModel.predict(text);
  await standardModel.predict(text);
  
  // Number of iterations for timing
  const iterations = 5;
  
  // Time optimized model
  console.log(`Running ${iterations} iterations with browser-optimized implementation...`);
  const optimizedStartTime = performance.now();
  for (let i = 0; i < iterations; i++) {
    await optimizedModel.predict(text);
  }
  const optimizedEndTime = performance.now();
  const optimizedTime = (optimizedEndTime - optimizedStartTime) / iterations;
  
  // Get operation breakdown for optimized model
  const optimizedMetrics = optimizedModel.getPerformanceMetrics();
  
  // Time standard model
  console.log(`Running ${iterations} iterations with standard implementation...`);
  const standardStartTime = performance.now();
  for (let i = 0; i < iterations; i++) {
    await standardModel.predict(text);
  }
  const standardEndTime = performance.now();
  const standardTime = (standardEndTime - standardStartTime) / iterations;
  
  // Get operation breakdown for standard model
  const standardMetrics = standardModel.getPerformanceMetrics();
  
  // Calculate speedup
  const speedup = standardTime / optimizedTime;
  
  // Calculate operation breakdown comparison
  const operationBreakdown: Record<string, { 
    optimized: number; 
    standard: number; 
    improvement: number 
  }> = {};
  
  // Collect all operation keys
  const allOperationKeys = new Set<string>();
  Object.keys(optimizedMetrics).forEach(key => allOperationKeys.add(key));
  Object.keys(standardMetrics).forEach(key => allOperationKeys.add(key));
  
  // Calculate improvements for each operation
  for (const key of allOperationKeys) {
    const optimizedTime = optimizedMetrics[key]?.avg || 0;
    const standardTime = standardMetrics[key]?.avg || 0;
    
    // Skip operations that don't exist in both implementations
    if (optimizedTime === 0 || standardTime === 0) continue;
    
    const improvement = standardTime / optimizedTime;
    operationBreakdown[key] = {
      optimized: optimizedTime,
      standard: standardTime,
      improvement
    };
  }
  
  // Clean up resources
  await optimizedModel.dispose();
  await standardModel.dispose();
  await optimizedTensorSharing.dispose();
  await standardTensorSharing.dispose();
  
  // Return performance results
  return {
    optimizedTime,
    standardTime,
    speedup,
    browserInfo: browserCapabilities,
    operationBreakdown
  };
}

/**
 * Run BERT for question answering task
 */
export async function runQuestionAnsweringExample(
  context: string, 
  question: string
): Promise<{ answer: string; score: number }> {
  // Set up WebGPU
  const device = await setupWebGPU();
  if (!device) {
    throw new Error('WebGPU not supported or failed to initialize');
  }
  
  // Initialize tensor sharing system
  const tensorSharing = new WebGPUTensorSharing();
  await tensorSharing.initialize(device);
  
  // Initialize storage manager
  const storageManager = new StorageManager('bert-qa-models');
  await storageManager.initialize();
  
  // Create BERT model configuration for question answering
  const bertConfig: BERTConfig = {
    vocabSize: 30522,
    hiddenSize: 768,
    numLayers: 12,
    numHeads: 12,
    intermediateSize: 3072,
    maxSequenceLength: 384, // Shorter for QA
    quantization: {
      enabled: true,
      bits: 8
    },
    useOptimizedAttention: true,
    modelId: 'bert-large-uncased-whole-word-masking-finetuned-squad',
    taskType: 'question_answering'
  };
  
  // Initialize BERT model
  const model = new WebGPUOptimizedBERT(bertConfig, tensorSharing, storageManager);
  await model.initialize();
  
  // In a real implementation, we would use a proper tokenizer with WordPiece
  // Here, we'll use a simplified approach with mock tokenization
  
  // Combine question and context with special tokens:
  // [CLS] Question [SEP] Context [SEP]
  const combined = `${question} [SEP] ${context}`;
  
  // Run inference
  const output = await model.predict(combined) as { 
    start_logits: Float32Array; 
    end_logits: Float32Array; 
  };
  
  // Process the output to find the answer span
  // This is a simplified implementation - in a real system we would map
  // the logits back to token positions and extract the text
  
  // Find the position with the highest start and end logits
  let maxStartIdx = 0;
  let maxStartLogit = output.start_logits[0];
  for (let i = 1; i < output.start_logits.length; i++) {
    if (output.start_logits[i] > maxStartLogit) {
      maxStartLogit = output.start_logits[i];
      maxStartIdx = i;
    }
  }
  
  let maxEndIdx = 0;
  let maxEndLogit = output.end_logits[0];
  for (let i = 1; i < output.end_logits.length; i++) {
    if (output.end_logits[i] > maxEndLogit) {
      maxEndLogit = output.end_logits[i];
      maxEndIdx = i;
    }
  }
  
  // Ensure end comes after start
  if (maxEndIdx < maxStartIdx) {
    [maxStartIdx, maxEndIdx] = [maxEndIdx, maxStartIdx];
  }
  
  // Calculate score (sum of logits)
  const score = maxStartLogit + maxEndLogit;
  
  // In a real implementation, we would use token-to-text mapping
  // Here, we'll just return a placeholder answer
  const answer = `Answer span from position ${maxStartIdx} to ${maxEndIdx}`;
  
  // Clean up resources
  await model.dispose();
  await tensorSharing.dispose();
  
  return { answer, score };
}

/**
 * Run BERT for text classification task
 */
export async function runClassificationExample(
  text: string
): Promise<{ label: string; probability: number }> {
  // Set up WebGPU
  const device = await setupWebGPU();
  if (!device) {
    throw new Error('WebGPU not supported or failed to initialize');
  }
  
  // Initialize tensor sharing system
  const tensorSharing = new WebGPUTensorSharing();
  await tensorSharing.initialize(device);
  
  // Initialize storage manager
  const storageManager = new StorageManager('bert-classification-models');
  await storageManager.initialize();
  
  // Create BERT model configuration for classification
  const bertConfig: BERTConfig = {
    vocabSize: 30522,
    hiddenSize: 768,
    numLayers: 12,
    numHeads: 12,
    intermediateSize: 3072,
    maxSequenceLength: 128, // Shorter for classification
    quantization: {
      enabled: true,
      bits: 8
    },
    useOptimizedAttention: true,
    modelId: 'bert-base-uncased-finetuned-sst2',
    taskType: 'sequence_classification',
    numLabels: 2 // Binary sentiment: positive/negative
  };
  
  // Initialize BERT model
  const model = new WebGPUOptimizedBERT(bertConfig, tensorSharing, storageManager);
  await model.initialize();
  
  // Run inference
  const output = await model.predict(text) as { logits: Float32Array };
  
  // Process logits with softmax to get probabilities
  const logits = output.logits;
  
  // Apply softmax
  const maxLogit = Math.max(...Array.from(logits));
  const expValues = Array.from(logits).map(x => Math.exp(x - maxLogit));
  const sumExp = expValues.reduce((a, b) => a + b, 0);
  const probabilities = expValues.map(x => x / sumExp);
  
  // Find highest probability class
  let maxIdx = 0;
  let maxProb = probabilities[0];
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIdx = i;
    }
  }
  
  // Map index to label
  const labels = ['negative', 'positive'];
  const label = labels[maxIdx];
  
  // Clean up resources
  await model.dispose();
  await tensorSharing.dispose();
  
  return { label, probability: maxProb };
}

/**
 * Create an interactive demonstration of browser-optimized WebGPU BERT model
 */
export async function createInteractiveBERTDemo(
  containerElement: HTMLElement
): Promise<{ 
  runEmbedding: (text: string) => Promise<Float32Array>;
  runClassification: (text: string) => Promise<{ label: string; probability: number }>;
  detectBrowserCapabilities: () => Promise<BrowserCapabilities | null>;
  benchmarkPerformance: (text: string) => Promise<void>;
}> {
  // Create UI components
  containerElement.innerHTML = `
    <div class="bert-demo">
      <h2>Browser-Optimized BERT Demo</h2>
      
      <div class="browser-info-section card">
        <h3>Browser Capabilities</h3>
        <div id="browser-capabilities">
          <button id="detect-capabilities-btn" class="btn">Detect Browser Capabilities</button>
          <div id="capabilities-result" class="capabilities-result"></div>
        </div>
      </div>
      
      <div class="tabs">
        <div class="tab active" data-tab="embedding">Text Embedding</div>
        <div class="tab" data-tab="classification">Sentiment Analysis</div>
        <div class="tab" data-tab="benchmark">Performance Benchmark</div>
      </div>
      
      <div class="tab-content active" id="embedding">
        <div class="card">
          <h3>Text Embedding</h3>
          <p>Generate embeddings from text using BERT with browser-optimized WebGPU acceleration</p>
          
          <div class="input-section">
            <textarea id="embedding-text" placeholder="Enter text for embedding" rows="4">This is an example sentence to generate BERT embeddings.</textarea>
            <button id="run-embedding-btn" class="btn">Generate Embedding</button>
          </div>
          
          <div id="embedding-status" class="status-message status-info">
            Enter text and click "Generate Embedding" to create embeddings
          </div>
          
          <div id="embedding-result" style="display: none;">
            <h4>Embedding Vector (first 10 dimensions)</h4>
            <div id="embedding-preview" class="code-block"></div>
            <div id="embedding-time" class="status-message status-success"></div>
          </div>
        </div>
      </div>
      
      <div class="tab-content" id="classification">
        <div class="card">
          <h3>Sentiment Analysis</h3>
          <p>Classify text sentiment with BERT using browser-optimized WebGPU acceleration</p>
          
          <div class="input-section">
            <textarea id="classification-text" placeholder="Enter text for sentiment analysis" rows="4">This movie was fantastic! I really enjoyed it and would recommend it to everyone.</textarea>
            <button id="run-classification-btn" class="btn">Analyze Sentiment</button>
          </div>
          
          <div id="classification-status" class="status-message status-info">
            Enter text and click "Analyze Sentiment" to classify the sentiment
          </div>
          
          <div id="classification-result" style="display: none;">
            <div class="sentiment-display">
              <div class="sentiment-label">Sentiment: <span id="sentiment-value">Positive</span></div>
              <div class="sentiment-gauge-container">
                <div class="sentiment-gauge-label sentiment-negative">Negative</div>
                <div class="sentiment-gauge">
                  <div id="sentiment-gauge-fill" class="sentiment-gauge-fill"></div>
                </div>
                <div class="sentiment-gauge-label sentiment-positive">Positive</div>
              </div>
              <div class="sentiment-probability">Confidence: <span id="sentiment-probability">95%</span></div>
            </div>
            <div id="classification-time" class="status-message status-success"></div>
          </div>
        </div>
      </div>
      
      <div class="tab-content" id="benchmark">
        <div class="card">
          <h3>Performance Benchmark</h3>
          <p>Compare optimized vs. standard implementations</p>
          
          <div class="input-section">
            <textarea id="benchmark-text" placeholder="Enter text for benchmarking" rows="4">This is an example text that will be used to benchmark the BERT model with browser-optimized WebGPU acceleration compared to a standard implementation.</textarea>
            <button id="run-benchmark-btn" class="btn">Run Benchmark</button>
          </div>
          
          <div id="benchmark-status" class="status-message status-info">
            Enter text and click "Run Benchmark" to compare optimized vs. standard implementations
          </div>
          
          <div id="benchmark-results" style="display: none;">
            <div id="benchmark-info"></div>
            
            <div class="benchmark-chart-container">
              <h4>Overall Performance</h4>
              <div class="benchmark-bars">
                <div class="benchmark-bar-group">
                  <div class="benchmark-bar optimized-bar" id="optimized-bar">
                    <div class="benchmark-value" id="optimized-value">0 ms</div>
                  </div>
                  <div class="benchmark-label">Optimized</div>
                </div>
                
                <div class="benchmark-bar-group">
                  <div class="benchmark-bar standard-bar" id="standard-bar">
                    <div class="benchmark-value" id="standard-value">0 ms</div>
                  </div>
                  <div class="benchmark-label">Standard</div>
                </div>
              </div>
              <div id="speedup-info" class="status-message status-success"></div>
            </div>
            
            <div class="operation-breakdown">
              <h4>Operation Breakdown</h4>
              <div id="operation-table"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <style>
      .bert-demo {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
      }
      
      .card {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
      }
      
      .btn {
        background: #0078d7;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
      }
      
      .btn:hover {
        background: #0062ab;
      }
      
      .input-section {
        margin-bottom: 20px;
      }
      
      textarea {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-family: inherit;
        font-size: 14px;
        margin-bottom: 10px;
      }
      
      .status-message {
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 15px;
      }
      
      .status-info {
        background-color: rgba(0, 120, 215, 0.1);
        border: 1px solid rgba(0, 120, 215, 0.3);
        color: #0078d7;
      }
      
      .status-error {
        background-color: rgba(232, 17, 35, 0.1);
        border: 1px solid rgba(232, 17, 35, 0.3);
        color: #e81123;
      }
      
      .status-success {
        background-color: rgba(16, 124, 16, 0.1);
        border: 1px solid rgba(16, 124, 16, 0.3);
        color: #107c10;
      }
      
      .capabilities-result {
        margin-top: 15px;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 14px;
        background: #222;
        color: #fff;
        padding: 15px;
        border-radius: 4px;
        max-height: 300px;
        overflow: auto;
      }
      
      .code-block {
        font-family: monospace;
        background: #f8f8f8;
        padding: 15px;
        border-radius: 4px;
        border: 1px solid #ddd;
        overflow-x: auto;
      }
      
      .tabs {
        display: flex;
        margin-bottom: 20px;
        border-bottom: 1px solid #ddd;
      }
      
      .tab {
        padding: 10px 20px;
        cursor: pointer;
        border-bottom: 2px solid transparent;
        font-weight: 500;
      }
      
      .tab.active {
        border-bottom-color: #0078d7;
        color: #0078d7;
      }
      
      .tab-content {
        display: none;
      }
      
      .tab-content.active {
        display: block;
      }
      
      .sentiment-display {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
      }
      
      .sentiment-label {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
      }
      
      .sentiment-gauge-container {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
      }
      
      .sentiment-gauge {
        flex: 1;
        height: 20px;
        background: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0 10px;
      }
      
      .sentiment-gauge-fill {
        height: 100%;
        background: linear-gradient(to right, #e81123, #107c10);
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
        width: 50%;
      }
      
      .sentiment-gauge-label {
        font-weight: 500;
        width: 80px;
        text-align: center;
      }
      
      .sentiment-negative {
        color: #e81123;
      }
      
      .sentiment-positive {
        color: #107c10;
      }
      
      .sentiment-probability {
        text-align: center;
        font-weight: 500;
      }
      
      .benchmark-chart-container {
        margin-top: 20px;
        margin-bottom: 30px;
      }
      
      .benchmark-bars {
        display: flex;
        height: 300px;
        align-items: flex-end;
        padding: 20px;
        background: white;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
      }
      
      .benchmark-bar-group {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 0 20px;
      }
      
      .benchmark-bar {
        width: 100%;
        max-width: 100px;
        transition: height 1s ease-in-out;
        border-radius: 4px 4px 0 0;
        position: relative;
      }
      
      .optimized-bar {
        background-color: #0078d7;
      }
      
      .standard-bar {
        background-color: #666;
      }
      
      .benchmark-value {
        position: absolute;
        top: -25px;
        left: 0;
        right: 0;
        text-align: center;
        font-weight: 500;
      }
      
      .benchmark-label {
        margin-top: 10px;
        font-weight: 500;
      }
      
      .operation-breakdown {
        margin-top: 30px;
      }
      
      .operation-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
      }
      
      .operation-table th, .operation-table td {
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #ddd;
      }
      
      .operation-table th {
        background-color: #f5f5f5;
        font-weight: 500;
      }
      
      .improvement-positive {
        color: #107c10;
      }
      
      .improvement-neutral {
        color: #0078d7;
      }
      
      .improvement-negative {
        color: #e81123;
      }
      
      @media (max-width: 768px) {
        .benchmark-bars {
          flex-direction: column;
          height: auto;
        }
        
        .benchmark-bar-group {
          width: 100%;
          margin: 15px 0;
        }
        
        .benchmark-bar {
          height: 40px !important;
          max-width: none;
        }
        
        .benchmark-value {
          top: 50%;
          transform: translateY(-50%);
          color: white;
        }
      }
    </style>
  `;
  
  // Get UI elements
  const detectCapabilitiesBtn = containerElement.querySelector('#detect-capabilities-btn') as HTMLButtonElement;
  const capabilitiesResult = containerElement.querySelector('#capabilities-result') as HTMLDivElement;
  
  const embeddingText = containerElement.querySelector('#embedding-text') as HTMLTextAreaElement;
  const runEmbeddingBtn = containerElement.querySelector('#run-embedding-btn') as HTMLButtonElement;
  const embeddingStatus = containerElement.querySelector('#embedding-status') as HTMLDivElement;
  const embeddingResult = containerElement.querySelector('#embedding-result') as HTMLDivElement;
  const embeddingPreview = containerElement.querySelector('#embedding-preview') as HTMLDivElement;
  const embeddingTime = containerElement.querySelector('#embedding-time') as HTMLDivElement;
  
  const classificationText = containerElement.querySelector('#classification-text') as HTMLTextAreaElement;
  const runClassificationBtn = containerElement.querySelector('#run-classification-btn') as HTMLButtonElement;
  const classificationStatus = containerElement.querySelector('#classification-status') as HTMLDivElement;
  const classificationResult = containerElement.querySelector('#classification-result') as HTMLDivElement;
  const sentimentValue = containerElement.querySelector('#sentiment-value') as HTMLSpanElement;
  const sentimentGaugeFill = containerElement.querySelector('#sentiment-gauge-fill') as HTMLDivElement;
  const sentimentProbability = containerElement.querySelector('#sentiment-probability') as HTMLSpanElement;
  const classificationTime = containerElement.querySelector('#classification-time') as HTMLDivElement;
  
  const benchmarkText = containerElement.querySelector('#benchmark-text') as HTMLTextAreaElement;
  const runBenchmarkBtn = containerElement.querySelector('#run-benchmark-btn') as HTMLButtonElement;
  const benchmarkStatus = containerElement.querySelector('#benchmark-status') as HTMLDivElement;
  const benchmarkResults = containerElement.querySelector('#benchmark-results') as HTMLDivElement;
  const benchmarkInfo = containerElement.querySelector('#benchmark-info') as HTMLDivElement;
  const optimizedBar = containerElement.querySelector('#optimized-bar') as HTMLDivElement;
  const standardBar = containerElement.querySelector('#standard-bar') as HTMLDivElement;
  const optimizedValue = containerElement.querySelector('#optimized-value') as HTMLDivElement;
  const standardValue = containerElement.querySelector('#standard-value') as HTMLDivElement;
  const speedupInfo = containerElement.querySelector('#speedup-info') as HTMLDivElement;
  const operationTable = containerElement.querySelector('#operation-table') as HTMLDivElement;
  
  const tabs = containerElement.querySelectorAll('.tab');
  const tabContents = containerElement.querySelectorAll('.tab-content');
  
  // Function to switch tabs
  function switchTab(tabId: string) {
    tabs.forEach(tab => {
      tab.classList.toggle('active', (tab as HTMLElement).dataset.tab === tabId);
    });
    
    tabContents.forEach(content => {
      content.classList.toggle('active', content.id === tabId);
    });
  }
  
  // Add tab click event listeners
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const tabId = (tab as HTMLElement).dataset.tab;
      if (tabId) {
        switchTab(tabId);
      }
    });
  });
  
  /**
   * Detect browser capabilities
   */
  async function detectBrowserCapabilities(): Promise<BrowserCapabilities | null> {
    try {
      capabilitiesResult.textContent = 'Detecting browser capabilities...';
      
      // Set up WebGPU
      const device = await setupWebGPU();
      if (!device) {
        capabilitiesResult.textContent = 'WebGPU not supported in this browser';
        return null;
      }
      
      // Initialize tensor sharing system
      const tensorSharing = new WebGPUTensorSharing();
      await tensorSharing.initialize(device);
      
      // Get browser capabilities
      const capabilities = await tensorSharing.getBrowserCapabilities();
      
      // Display capabilities
      const formattedCapabilities = JSON.stringify(capabilities, null, 2);
      capabilitiesResult.textContent = formattedCapabilities;
      
      // Clean up
      await tensorSharing.dispose();
      
      return capabilities;
    } catch (error) {
      console.error('Error detecting capabilities:', error);
      capabilitiesResult.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
      return null;
    }
  }
  
  /**
   * Run text embedding with BERT
   */
  async function runEmbedding(text: string): Promise<Float32Array> {
    try {
      embeddingStatus.className = 'status-message status-info';
      embeddingStatus.textContent = 'Running BERT embedding...';
      embeddingResult.style.display = 'none';
      
      const startTime = performance.now();
      
      // Run the BERT model
      const embedding = await runBasicBERTExample(text);
      
      const endTime = performance.now();
      const timeElapsed = endTime - startTime;
      
      // Display results
      const embeddingSize = embedding.length;
      const previewSize = Math.min(10, embeddingSize);
      
      // Format embedding preview
      let preview = `// Embedding vector with ${embeddingSize} dimensions (showing first ${previewSize})\n`;
      preview += '[\n';
      
      for (let i = 0; i < previewSize; i++) {
        preview += `  ${embedding[i].toFixed(6)}${i < previewSize - 1 ? ',' : ''}\n`;
      }
      
      if (previewSize < embeddingSize) {
        preview += '  // ... additional dimensions not shown\n';
      }
      
      preview += ']';
      
      embeddingPreview.textContent = preview;
      embeddingTime.textContent = `Inference time: ${timeElapsed.toFixed(2)}ms`;
      
      embeddingStatus.className = 'status-message status-success';
      embeddingStatus.textContent = 'Embedding generated successfully';
      embeddingResult.style.display = 'block';
      
      return embedding;
    } catch (error) {
      console.error('Error running embedding:', error);
      
      embeddingStatus.className = 'status-message status-error';
      embeddingStatus.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
      
      return new Float32Array();
    }
  }
  
  /**
   * Run sentiment analysis with BERT
   */
  async function runClassification(text: string): Promise<{ label: string; probability: number }> {
    try {
      classificationStatus.className = 'status-message status-info';
      classificationStatus.textContent = 'Analyzing sentiment...';
      classificationResult.style.display = 'none';
      
      const startTime = performance.now();
      
      // Run the classification
      const result = await runClassificationExample(text);
      
      const endTime = performance.now();
      const timeElapsed = endTime - startTime;
      
      // Display results
      sentimentValue.textContent = result.label.charAt(0).toUpperCase() + result.label.slice(1);
      sentimentProbability.textContent = `${(result.probability * 100).toFixed(1)}%`;
      
      // Update gauge
      let fillValue = 50; // Default to middle
      
      if (result.label === 'positive') {
        fillValue = 50 + (result.probability * 50);
      } else {
        fillValue = 50 - (result.probability * 50);
      }
      
      sentimentGaugeFill.style.width = `${fillValue}%`;
      
      classificationTime.textContent = `Inference time: ${timeElapsed.toFixed(2)}ms`;
      
      classificationStatus.className = 'status-message status-success';
      classificationStatus.textContent = 'Sentiment analysis completed successfully';
      classificationResult.style.display = 'block';
      
      return result;
    } catch (error) {
      console.error('Error running classification:', error);
      
      classificationStatus.className = 'status-message status-error';
      classificationStatus.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
      
      return { label: '', probability: 0 };
    }
  }
  
  /**
   * Run benchmark comparison
   */
  async function benchmarkPerformance(text: string): Promise<void> {
    try {
      benchmarkStatus.className = 'status-message status-info';
      benchmarkStatus.textContent = 'Running benchmark comparison...';
      benchmarkResults.style.display = 'none';
      
      // Run performance comparison
      const results = await runPerformanceComparison(text);
      
      // Format browser info
      benchmarkInfo.innerHTML = `
        <div><strong>Browser:</strong> ${results.browserInfo.browserType.charAt(0).toUpperCase() + 
                                       results.browserInfo.browserType.slice(1)} ${results.browserInfo.browserVersion || ''}</div>
        <div><strong>GPU:</strong> ${results.browserInfo.gpuVendor} ${results.browserInfo.gpuModel || ''}</div>
        <div><strong>Optimized time:</strong> ${results.optimizedTime.toFixed(2)}ms</div>
        <div><strong>Standard time:</strong> ${results.standardTime.toFixed(2)}ms</div>
        <div><strong>Overall speedup:</strong> ${results.speedup.toFixed(2)}x</div>
      `;
      
      // Calculate bar heights
      const maxTime = Math.max(results.optimizedTime, results.standardTime);
      const optimizedHeight = (results.optimizedTime / maxTime * 100).toFixed(1);
      const standardHeight = (results.standardTime / maxTime * 100).toFixed(1);
      
      // Update chart bars
      optimizedBar.style.height = `${optimizedHeight}%`;
      standardBar.style.height = `${standardHeight}%`;
      
      optimizedValue.textContent = `${results.optimizedTime.toFixed(1)}ms`;
      standardValue.textContent = `${results.standardTime.toFixed(1)}ms`;
      
      // Update speedup message
      speedupInfo.textContent = `${results.speedup.toFixed(2)}x faster with browser optimizations`;
      
      // Create operation breakdown table
      let tableHTML = `
        <table class="operation-table">
          <thead>
            <tr>
              <th>Operation</th>
              <th>Optimized (ms)</th>
              <th>Standard (ms)</th>
              <th>Improvement</th>
            </tr>
          </thead>
          <tbody>
      `;
      
      // Sort operations by improvement (highest first)
      const sortedOperations = Object.entries(results.operationBreakdown)
        .sort(([, a], [, b]) => b.improvement - a.improvement);
      
      for (const [operation, data] of sortedOperations) {
        const improvementClass = 
          data.improvement >= 2.0 ? 'improvement-positive' :
          data.improvement >= 1.2 ? 'improvement-neutral' : 'improvement-negative';
        
        tableHTML += `
          <tr>
            <td>${operation}</td>
            <td>${data.optimized.toFixed(2)}ms</td>
            <td>${data.standard.toFixed(2)}ms</td>
            <td class="${improvementClass}">${data.improvement.toFixed(2)}x</td>
          </tr>
        `;
      }
      
      tableHTML += `
          </tbody>
        </table>
      `;
      
      operationTable.innerHTML = tableHTML;
      
      // Update status and show results
      benchmarkStatus.className = 'status-message status-success';
      benchmarkStatus.textContent = 'Benchmark completed successfully';
      benchmarkResults.style.display = 'block';
    } catch (error) {
      console.error('Error running benchmark:', error);
      
      benchmarkStatus.className = 'status-message status-error';
      benchmarkStatus.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
    }
  }
  
  // Add event listeners
  detectCapabilitiesBtn.addEventListener('click', () => {
    detectBrowserCapabilities();
  });
  
  runEmbeddingBtn.addEventListener('click', () => {
    const text = embeddingText.value.trim();
    if (text) {
      runEmbedding(text);
    } else {
      embeddingStatus.className = 'status-message status-error';
      embeddingStatus.textContent = 'Please enter some text';
    }
  });
  
  runClassificationBtn.addEventListener('click', () => {
    const text = classificationText.value.trim();
    if (text) {
      runClassification(text);
    } else {
      classificationStatus.className = 'status-message status-error';
      classificationStatus.textContent = 'Please enter some text';
    }
  });
  
  runBenchmarkBtn.addEventListener('click', () => {
    const text = benchmarkText.value.trim();
    if (text) {
      benchmarkPerformance(text);
    } else {
      benchmarkStatus.className = 'status-message status-error';
      benchmarkStatus.textContent = 'Please enter some text';
    }
  });
  
  // Return API for external use
  return {
    runEmbedding,
    runClassification,
    detectBrowserCapabilities,
    benchmarkPerformance
  };
}

// Example usage in browser:
// 
// document.addEventListener('DOMContentLoaded', async () => {
//   const demoContainer = document.getElementById('bert-demo-container');
//   if (demoContainer) {
//     const demo = await createInteractiveBERTDemo(demoContainer);
//     // You can programmatically interact with the demo through the API:
//     // const capabilities = await demo.detectBrowserCapabilities();
//     // const embedding = await demo.runEmbedding('Example text');
//     // const sentiment = await demo.runClassification('This is great!');
//     // await demo.benchmarkPerformance('Example text for benchmarking');
//   }
// });