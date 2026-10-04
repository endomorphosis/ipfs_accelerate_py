/**
 * Example demonstrating the Hardware Abstracted BERT model
 * 
 * This example shows how to load and run inference with the BERT model
 * using the Hardware Abstraction Layer (HAL) for automatic backend selection
 * and optimization across WebGPU, WebNN, and CPU.
 */

import { StorageManager } from './ipfs_accelerate_js_storage_manager';
import { HardwareAbstractedBERT, BERTConfig, BERTEmbedding } from './ipfs_accelerate_js_bert_hardware_abstraction';
import { HardwareAbstraction, createHardwareAbstraction } from './ipfs_accelerate_js_hardware_abstraction';

/**
 * Basic BERT example for text embeddings with HAL-based acceleration
 */
export async function runBasicBERTExample(text: string): Promise<Float32Array> {
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
  
  // Initialize BERT model with HAL
  console.log('Initializing BERT model with Hardware Abstraction Layer');
  const model = new HardwareAbstractedBERT(bertConfig, storageManager);
  await model.initialize();
  
  // Show model info
  const modelInfo = model.getModelInfo();
  console.log('Model info:', modelInfo);
  
  // Run inference
  console.log('Running inference with HAL-accelerated BERT');
  console.time('inference');
  const embedding = await model.predict(text);
  console.timeEnd('inference');
  
  // Show performance metrics
  const metrics = model.getPerformanceMetrics();
  console.log('Performance metrics:', metrics);
  
  // Show backend-specific metrics
  const backendMetrics = model.getBackendMetrics();
  console.log('Backend metrics:', backendMetrics);
  
  // Clean up resources
  await model.dispose();
  
  return embedding as Float32Array;
}

/**
 * Compare BERT inference across all available backends
 */
export async function runCrossBackendPerformanceComparison(text: string): Promise<{
  results: Record<string, number>;
  bestBackend: string;
  worstBackend: string;
  speedup: number;
  backendDetails: Record<string, any>;
}> {
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
  
  // Initialize BERT model with HAL
  console.log('Initializing BERT model with Hardware Abstraction Layer');
  const model = new HardwareAbstractedBERT(bertConfig, storageManager);
  await model.initialize();

  // Get available backends
  const modelInfo = model.getModelInfo();
  const availableBackends = modelInfo.availableBackends;
  console.log(`Available backends: ${availableBackends.join(', ')}`);
  console.log(`Selected backend for text models: ${modelInfo.selectedBackend}`);
  
  // Warm-up run
  console.log('Performing warm-up run...');
  await model.predict(text);
  
  // Compare backends
  console.log('Comparing performance across backends...');
  const results = await model.compareBackends(text);
  
  // Find best and worst backends
  let bestBackend = '';
  let worstBackend = '';
  let bestTime = Number.MAX_VALUE;
  let worstTime = 0;
  
  for (const [backend, time] of Object.entries(results)) {
    if (time > 0) { // Skip backends that failed (time < 0)
      if (time < bestTime) {
        bestTime = time;
        bestBackend = backend;
      }
      if (time > worstTime) {
        worstTime = time;
        worstBackend = backend;
      }
    }
  }
  
  // Calculate speedup
  const speedup = worstTime > 0 ? worstTime / bestTime : 0;
  
  // Get backend-specific metrics for more details
  const backendMetrics = model.getBackendMetrics();
  
  // Clean up resources
  await model.dispose();
  
  return {
    results,
    bestBackend,
    worstBackend,
    speedup,
    backendDetails: backendMetrics
  };
}

/**
 * Run BERT for sentiment analysis using HAL-based acceleration
 */
export async function runSentimentAnalysisExample(text: string): Promise<{
  label: string;
  probability: number;
  inferenceTime: number;
  backendUsed: string;
}> {
  // Initialize storage manager
  const storageManager = new StorageManager('bert-classification-models');
  await storageManager.initialize();
  
  // Create BERT model configuration for sentiment analysis
  const bertConfig: BERTConfig = {
    vocabSize: 30522,
    hiddenSize: 768,
    numLayers: 12,
    numHeads: 12,
    intermediateSize: 3072,
    maxSequenceLength: 128,
    quantization: {
      enabled: true,
      bits: 8
    },
    useOptimizedAttention: true,
    modelId: 'bert-base-uncased-finetuned-sst2',
    taskType: 'sequence_classification',
    numLabels: 2 // Binary sentiment: positive/negative
  };
  
  // Initialize BERT model with HAL
  console.log('Initializing BERT model with Hardware Abstraction Layer');
  const model = new HardwareAbstractedBERT(bertConfig, storageManager);
  await model.initialize();
  
  // Get selected backend
  const modelInfo = model.getModelInfo();
  const backendUsed = modelInfo.selectedBackend;
  
  // Run inference
  console.log(`Running sentiment analysis with ${backendUsed} backend`);
  const startTime = performance.now();
  const output = await model.predict(text) as { logits: Float32Array };
  const endTime = performance.now();
  const inferenceTime = endTime - startTime;
  
  // Process logits to get predicted label and probability
  const logits = output.logits;
  
  // Apply softmax to get probabilities
  const maxLogit = Math.max(logits[0], logits[1]);
  const expValues = [Math.exp(logits[0] - maxLogit), Math.exp(logits[1] - maxLogit)];
  const sumExp = expValues[0] + expValues[1];
  const probabilities = [expValues[0] / sumExp, expValues[1] / sumExp];
  
  // Get predicted label
  const positiveIdx = probabilities[1] > probabilities[0] ? 1 : 0;
  const label = positiveIdx === 1 ? 'positive' : 'negative';
  const probability = probabilities[positiveIdx];
  
  // Show performance metrics
  const metrics = model.getPerformanceMetrics();
  console.log('Performance metrics:', metrics);
  
  // Clean up resources
  await model.dispose();
  
  return {
    label,
    probability,
    inferenceTime,
    backendUsed
  };
}

/**
 * Run BERT for question answering using HAL-based acceleration
 */
export async function runQuestionAnsweringExample(
  context: string,
  question: string
): Promise<{
  answer: string;
  confidence: number;
  inferenceTime: number;
  backendUsed: string;
}> {
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
    maxSequenceLength: 384,
    quantization: {
      enabled: true,
      bits: 8
    },
    useOptimizedAttention: true,
    modelId: 'bert-large-uncased-whole-word-masking-finetuned-squad',
    taskType: 'question_answering'
  };
  
  // Initialize BERT model with HAL
  console.log('Initializing BERT model with Hardware Abstraction Layer');
  const model = new HardwareAbstractedBERT(bertConfig, storageManager);
  await model.initialize();
  
  // Get selected backend
  const modelInfo = model.getModelInfo();
  const backendUsed = modelInfo.selectedBackend;
  
  // Combine question and context
  // In a real implementation, we would use the tokenizer with proper handling
  const combined = `${question} [SEP] ${context}`;
  
  // Run inference
  console.log(`Running question answering with ${backendUsed} backend`);
  const startTime = performance.now();
  const output = await model.predict(combined) as {
    start_logits: Float32Array;
    end_logits: Float32Array;
  };
  const endTime = performance.now();
  const inferenceTime = endTime - startTime;
  
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
  
  // Calculate confidence score
  const confidence = Math.tanh((maxStartLogit + maxEndLogit) / 2);
  
  // In a real implementation, we would use token-to-text mapping
  // Here, we'll just return a placeholder answer
  const answer = `Answer span from position ${maxStartIdx} to ${maxEndIdx}`;
  
  // Show performance metrics
  const metrics = model.getPerformanceMetrics();
  console.log('Performance metrics:', metrics);
  
  // Clean up resources
  await model.dispose();
  
  return {
    answer,
    confidence,
    inferenceTime,
    backendUsed
  };
}

/**
 * Create an interactive demonstration of HAL-accelerated BERT model
 */
export async function createInteractiveBERTDemo(
  containerElement: HTMLElement
): Promise<{
  detectHardware: () => Promise<Record<string, any>>;
  runEmbedding: (text: string) => Promise<Float32Array>;
  runSentimentAnalysis: (text: string) => Promise<{ label: string; probability: number }>;
  runQuestionAnswering: (context: string, question: string) => Promise<{ answer: string; confidence: number }>;
  compareBackends: (text: string) => Promise<Record<string, number>>;
}> {
  // Create UI components
  containerElement.innerHTML = `
    <div class="bert-hal-demo">
      <h2>Hardware Abstracted BERT Demo</h2>
      
      <div class="hardware-info-section card">
        <h3>Hardware Capabilities</h3>
        <div id="hardware-capabilities">
          <button id="detect-hardware-btn" class="btn">Detect Hardware Capabilities</button>
          <div id="hardware-result" class="hardware-result"></div>
        </div>
      </div>
      
      <div class="tabs">
        <div class="tab active" data-tab="embedding">Text Embedding</div>
        <div class="tab" data-tab="sentiment">Sentiment Analysis</div>
        <div class="tab" data-tab="qa">Question Answering</div>
        <div class="tab" data-tab="compare">Backend Comparison</div>
      </div>
      
      <div class="tab-content active" id="embedding">
        <div class="card">
          <h3>Text Embedding</h3>
          <p>Generate embeddings from text using BERT with HAL-based acceleration</p>
          
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
            <div id="embedding-backend" class="backend-info"></div>
          </div>
        </div>
      </div>
      
      <div class="tab-content" id="sentiment">
        <div class="card">
          <h3>Sentiment Analysis</h3>
          <p>Classify text sentiment with BERT using HAL-based acceleration</p>
          
          <div class="input-section">
            <textarea id="sentiment-text" placeholder="Enter text for sentiment analysis" rows="4">This movie was fantastic! I really enjoyed it and would recommend it to everyone.</textarea>
            <button id="run-sentiment-btn" class="btn">Analyze Sentiment</button>
          </div>
          
          <div id="sentiment-status" class="status-message status-info">
            Enter text and click "Analyze Sentiment" to classify the sentiment
          </div>
          
          <div id="sentiment-result" style="display: none;">
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
            <div id="sentiment-time" class="status-message status-success"></div>
            <div id="sentiment-backend" class="backend-info"></div>
          </div>
        </div>
      </div>
      
      <div class="tab-content" id="qa">
        <div class="card">
          <h3>Question Answering</h3>
          <p>Answer questions based on context using BERT with HAL-based acceleration</p>
          
          <div class="input-section">
            <textarea id="qa-context" placeholder="Enter context" rows="6">The Hardware Abstraction Layer (HAL) provides a unified interface for using different hardware backends (WebGPU, WebNN, CPU) for AI model acceleration. It automatically selects the optimal backend based on the model type and available hardware capabilities.</textarea>
            <textarea id="qa-question" placeholder="Enter question" rows="2">What is HAL?</textarea>
            <button id="run-qa-btn" class="btn">Answer Question</button>
          </div>
          
          <div id="qa-status" class="status-message status-info">
            Enter context and question, then click "Answer Question"
          </div>
          
          <div id="qa-result" style="display: none;">
            <div class="qa-display">
              <div class="qa-answer-label">Answer:</div>
              <div id="qa-answer" class="qa-answer"></div>
              <div class="qa-confidence">Confidence: <span id="qa-confidence">85%</span></div>
            </div>
            <div id="qa-time" class="status-message status-success"></div>
            <div id="qa-backend" class="backend-info"></div>
          </div>
        </div>
      </div>
      
      <div class="tab-content" id="compare">
        <div class="card">
          <h3>Backend Comparison</h3>
          <p>Compare performance across all available backends</p>
          
          <div class="input-section">
            <textarea id="compare-text" placeholder="Enter text for comparison" rows="4">This is a sample text that will be used to compare the performance of different backends (WebGPU, WebNN, CPU) for BERT inference.</textarea>
            <button id="run-compare-btn" class="btn">Compare Backends</button>
          </div>
          
          <div id="compare-status" class="status-message status-info">
            Enter text and click "Compare Backends" to run the comparison
          </div>
          
          <div id="compare-result" style="display: none;">
            <h4>Performance Comparison</h4>
            <div id="compare-chart" class="compare-chart"></div>
            <div id="compare-summary" class="status-message status-success"></div>
            <div id="compare-details" class="compare-details"></div>
          </div>
        </div>
      </div>
    </div>
    
    <style>
      .bert-hal-demo {
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
      
      .hardware-result {
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
      
      .backend-info {
        padding: 10px;
        background-color: rgba(0, 120, 215, 0.05);
        border-radius: 4px;
        margin-top: 10px;
        font-family: monospace;
        font-size: 14px;
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
      
      .qa-display {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
      }
      
      .qa-answer-label {
        font-weight: bold;
        margin-bottom: 10px;
      }
      
      .qa-answer {
        background: #f8f8f8;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
        border: 1px solid #ddd;
      }
      
      .qa-confidence {
        text-align: right;
        font-weight: 500;
      }
      
      .compare-chart {
        display: flex;
        height: 250px;
        align-items: flex-end;
        padding: 20px;
        background: white;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
      }
      
      .compare-bar-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 0 10px;
      }
      
      .compare-bar {
        width: 60px;
        background-color: #0078d7;
        border-radius: 4px 4px 0 0;
        position: relative;
        transition: height 1s ease-in-out;
      }
      
      .compare-bar.best {
        background-color: #107c10;
      }
      
      .compare-bar.worst {
        background-color: #e81123;
      }
      
      .compare-bar-value {
        position: absolute;
        top: -25px;
        left: 0;
        right: 0;
        text-align: center;
        font-weight: 500;
      }
      
      .compare-bar-label {
        margin-top: 10px;
        font-weight: 500;
        text-align: center;
      }
      
      .compare-details {
        margin-top: 20px;
        font-family: monospace;
        font-size: 14px;
        background: #f8f8f8;
        padding: 15px;
        border-radius: 4px;
        border: 1px solid #ddd;
        max-height: 300px;
        overflow: auto;
      }
    </style>
  `;
  
  // Get UI elements
  const detectHardwareBtn = containerElement.querySelector('#detect-hardware-btn') as HTMLButtonElement;
  const hardwareResult = containerElement.querySelector('#hardware-result') as HTMLDivElement;
  
  const embeddingText = containerElement.querySelector('#embedding-text') as HTMLTextAreaElement;
  const runEmbeddingBtn = containerElement.querySelector('#run-embedding-btn') as HTMLButtonElement;
  const embeddingStatus = containerElement.querySelector('#embedding-status') as HTMLDivElement;
  const embeddingResult = containerElement.querySelector('#embedding-result') as HTMLDivElement;
  const embeddingPreview = containerElement.querySelector('#embedding-preview') as HTMLDivElement;
  const embeddingTime = containerElement.querySelector('#embedding-time') as HTMLDivElement;
  const embeddingBackend = containerElement.querySelector('#embedding-backend') as HTMLDivElement;
  
  const sentimentText = containerElement.querySelector('#sentiment-text') as HTMLTextAreaElement;
  const runSentimentBtn = containerElement.querySelector('#run-sentiment-btn') as HTMLButtonElement;
  const sentimentStatus = containerElement.querySelector('#sentiment-status') as HTMLDivElement;
  const sentimentResult = containerElement.querySelector('#sentiment-result') as HTMLDivElement;
  const sentimentValue = containerElement.querySelector('#sentiment-value') as HTMLSpanElement;
  const sentimentGaugeFill = containerElement.querySelector('#sentiment-gauge-fill') as HTMLDivElement;
  const sentimentProbability = containerElement.querySelector('#sentiment-probability') as HTMLSpanElement;
  const sentimentTime = containerElement.querySelector('#sentiment-time') as HTMLDivElement;
  const sentimentBackend = containerElement.querySelector('#sentiment-backend') as HTMLDivElement;
  
  const qaContext = containerElement.querySelector('#qa-context') as HTMLTextAreaElement;
  const qaQuestion = containerElement.querySelector('#qa-question') as HTMLTextAreaElement;
  const runQaBtn = containerElement.querySelector('#run-qa-btn') as HTMLButtonElement;
  const qaStatus = containerElement.querySelector('#qa-status') as HTMLDivElement;
  const qaResult = containerElement.querySelector('#qa-result') as HTMLDivElement;
  const qaAnswer = containerElement.querySelector('#qa-answer') as HTMLDivElement;
  const qaConfidence = containerElement.querySelector('#qa-confidence') as HTMLSpanElement;
  const qaTime = containerElement.querySelector('#qa-time') as HTMLDivElement;
  const qaBackend = containerElement.querySelector('#qa-backend') as HTMLDivElement;
  
  const compareText = containerElement.querySelector('#compare-text') as HTMLTextAreaElement;
  const runCompareBtn = containerElement.querySelector('#run-compare-btn') as HTMLButtonElement;
  const compareStatus = containerElement.querySelector('#compare-status') as HTMLDivElement;
  const compareResult = containerElement.querySelector('#compare-result') as HTMLDivElement;
  const compareChart = containerElement.querySelector('#compare-chart') as HTMLDivElement;
  const compareSummary = containerElement.querySelector('#compare-summary') as HTMLDivElement;
  const compareDetails = containerElement.querySelector('#compare-details') as HTMLDivElement;
  
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
   * Detect hardware capabilities
   */
  async function detectHardware(): Promise<Record<string, any>> {
    try {
      hardwareResult.textContent = 'Detecting hardware capabilities...';
      
      // Initialize HAL to detect hardware capabilities
      const hal = await createHardwareAbstraction();
      const capabilities = hal.getCapabilities();
      
      // Format capabilities for display
      const formattedCapabilities = JSON.stringify(capabilities, null, 2);
      hardwareResult.textContent = formattedCapabilities;
      
      // Clean up
      hal.dispose();
      
      return capabilities;
    } catch (error) {
      console.error('Error detecting hardware:', error);
      hardwareResult.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
      return {};
    }
  }
  
  /**
   * Run text embedding with BERT and HAL
   */
  async function runEmbedding(text: string): Promise<Float32Array> {
    try {
      embeddingStatus.className = 'status-message status-info';
      embeddingStatus.textContent = 'Running BERT embedding with HAL...';
      embeddingResult.style.display = 'none';
      
      const startTime = performance.now();
      
      // Run the BERT model with HAL
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
      
      // Get HAL model info to display backend used
      const storageManager = new StorageManager('bert-models');
      await storageManager.initialize();
      
      const bertConfig: BERTConfig = {
        vocabSize: 30522,
        hiddenSize: 768,
        numLayers: 12,
        numHeads: 12,
        intermediateSize: 3072,
        maxSequenceLength: 512,
        modelId: 'bert-base-uncased',
        taskType: 'embedding'
      };
      
      const model = new HardwareAbstractedBERT(bertConfig, storageManager);
      await model.initialize();
      const modelInfo = model.getModelInfo();
      await model.dispose();
      
      embeddingBackend.textContent = `Backend used: ${modelInfo.selectedBackend} (available: ${modelInfo.availableBackends.join(', ')})`;
      
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
   * Run sentiment analysis with BERT and HAL
   */
  async function runSentimentAnalysis(text: string): Promise<{ label: string; probability: number }> {
    try {
      sentimentStatus.className = 'status-message status-info';
      sentimentStatus.textContent = 'Analyzing sentiment with HAL...';
      sentimentResult.style.display = 'none';
      
      // Run the sentiment analysis
      const result = await runSentimentAnalysisExample(text);
      
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
      
      sentimentTime.textContent = `Inference time: ${result.inferenceTime.toFixed(2)}ms`;
      sentimentBackend.textContent = `Backend used: ${result.backendUsed}`;
      
      sentimentStatus.className = 'status-message status-success';
      sentimentStatus.textContent = 'Sentiment analysis completed successfully';
      sentimentResult.style.display = 'block';
      
      return {
        label: result.label,
        probability: result.probability
      };
    } catch (error) {
      console.error('Error running sentiment analysis:', error);
      
      sentimentStatus.className = 'status-message status-error';
      sentimentStatus.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
      
      return { label: '', probability: 0 };
    }
  }
  
  /**
   * Run question answering with BERT and HAL
   */
  async function runQuestionAnswering(
    context: string,
    question: string
  ): Promise<{ answer: string; confidence: number }> {
    try {
      qaStatus.className = 'status-message status-info';
      qaStatus.textContent = 'Answering question with HAL...';
      qaResult.style.display = 'none';
      
      // Run question answering
      const result = await runQuestionAnsweringExample(context, question);
      
      // Display results
      qaAnswer.textContent = result.answer;
      qaConfidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
      qaTime.textContent = `Inference time: ${result.inferenceTime.toFixed(2)}ms`;
      qaBackend.textContent = `Backend used: ${result.backendUsed}`;
      
      qaStatus.className = 'status-message status-success';
      qaStatus.textContent = 'Question answering completed successfully';
      qaResult.style.display = 'block';
      
      return {
        answer: result.answer,
        confidence: result.confidence
      };
    } catch (error) {
      console.error('Error running question answering:', error);
      
      qaStatus.className = 'status-message status-error';
      qaStatus.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
      
      return { answer: '', confidence: 0 };
    }
  }
  
  /**
   * Compare backends performance
   */
  async function compareBackends(text: string): Promise<Record<string, number>> {
    try {
      compareStatus.className = 'status-message status-info';
      compareStatus.textContent = 'Comparing backends performance...';
      compareResult.style.display = 'none';
      
      // Run comparison
      const comparison = await runCrossBackendPerformanceComparison(text);
      
      // Display results
      compareChart.innerHTML = '';
      
      // Create bars for each backend
      for (const [backend, time] of Object.entries(comparison.results)) {
        if (time <= 0) continue; // Skip backends that failed
        
        const barContainer = document.createElement('div');
        barContainer.className = 'compare-bar-container';
        
        const bar = document.createElement('div');
        bar.className = 'compare-bar';
        if (backend === comparison.bestBackend) {
          bar.classList.add('best');
        } else if (backend === comparison.worstBackend) {
          bar.classList.add('worst');
        }
        
        // Calculate height percentage (inverse of time)
        const maxTime = Math.max(...Object.values(comparison.results).filter(t => t > 0));
        const heightPercent = 100 - (time / maxTime * 95); // Min height 5%
        bar.style.height = `${Math.max(5, heightPercent)}%`;
        
        const barValue = document.createElement('div');
        barValue.className = 'compare-bar-value';
        barValue.textContent = `${time.toFixed(1)}ms`;
        
        const barLabel = document.createElement('div');
        barLabel.className = 'compare-bar-label';
        barLabel.textContent = backend;
        
        bar.appendChild(barValue);
        barContainer.appendChild(bar);
        barContainer.appendChild(barLabel);
        compareChart.appendChild(barContainer);
      }
      
      // Update summary
      compareSummary.textContent = `${comparison.bestBackend} was ${comparison.speedup.toFixed(2)}x faster than ${comparison.worstBackend}`;
      
      // Update details
      compareDetails.innerHTML = `
        <h4>Backend Details</h4>
        <pre>${JSON.stringify(comparison.backendDetails, null, 2)}</pre>
      `;
      
      compareStatus.className = 'status-message status-success';
      compareStatus.textContent = 'Backend comparison completed successfully';
      compareResult.style.display = 'block';
      
      return comparison.results;
    } catch (error) {
      console.error('Error comparing backends:', error);
      
      compareStatus.className = 'status-message status-error';
      compareStatus.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
      
      return {};
    }
  }
  
  // Add event listeners
  detectHardwareBtn.addEventListener('click', () => {
    detectHardware();
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
  
  runSentimentBtn.addEventListener('click', () => {
    const text = sentimentText.value.trim();
    if (text) {
      runSentimentAnalysis(text);
    } else {
      sentimentStatus.className = 'status-message status-error';
      sentimentStatus.textContent = 'Please enter some text';
    }
  });
  
  runQaBtn.addEventListener('click', () => {
    const context = qaContext.value.trim();
    const question = qaQuestion.value.trim();
    
    if (context && question) {
      runQuestionAnswering(context, question);
    } else {
      qaStatus.className = 'status-message status-error';
      qaStatus.textContent = 'Please enter both context and question';
    }
  });
  
  runCompareBtn.addEventListener('click', () => {
    const text = compareText.value.trim();
    if (text) {
      compareBackends(text);
    } else {
      compareStatus.className = 'status-message status-error';
      compareStatus.textContent = 'Please enter some text';
    }
  });
  
  // Return API for external use
  return {
    detectHardware,
    runEmbedding,
    runSentimentAnalysis,
    runQuestionAnswering,
    compareBackends
  };
}

// Example usage:
// 
// document.addEventListener('DOMContentLoaded', async () => {
//   const demoContainer = document.getElementById('bert-hal-demo-container');
//   if (demoContainer) {
//     const demo = await createInteractiveBERTDemo(demoContainer);
//     // You can programmatically interact with the demo through the API:
//     // const hardware = await demo.detectHardware();
//     // const embedding = await demo.runEmbedding('Example text');
//     // const sentiment = await demo.runSentimentAnalysis('This is great!');
//     // const answer = await demo.runQuestionAnswering('HAL is a hardware abstraction layer', 'What is HAL?');
//     // const comparison = await demo.compareBackends('Example text for comparison');
//   }
// });