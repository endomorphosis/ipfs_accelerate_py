/**
 * WebNN Tensor Example
 * Demonstrates using the WebNN backend for neural network acceleration
 */

import { Tensor, random } from '../tensor/tensor';
import { WebNNBackend } from '../hardware/webnn/backend';
import { detectWebNNFeatures, hasNeuralProcessor } from '../hardware/webnn/capabilities';

/**
 * WebNN Tensor Example
 * Runs basic tensor operations on the WebNN backend
 */
export async function runWebNNTensorExample() {
  try {
    // Check for WebNN support
    const features = await detectWebNNFeatures();
    
    if (!features.supported) {
      console.error('WebNN is not available in this browser');
      return {
        supported: false,
        error: 'WebNN not available',
        features,
        results: null
      };
    }
    
    console.log('WebNN features detected:', features);
    
    // Initialize the WebNN backend
    const backend = new WebNNBackend();
    await backend.initialize();
    
    console.log('WebNN backend initialized successfully');
    console.log('Capabilities:', backend.capabilities);
    
    // Check for NPU
    const hasNPU = hasNeuralProcessor();
    console.log(`Neural processor detected: ${hasNPU ? 'Yes' : 'No'}`);
    
    // Create tensors for operations
    const matrixSize = 512; // Using 512x512 for benchmark (adjust based on browser capabilities)
    
    console.log(`Creating ${matrixSize}x${matrixSize} matrices...`);
    
    // Create input matrices
    const matrixA = random(
      [matrixSize, matrixSize],
      -1,
      1,
      { backend: 'webnn' }
    );
    
    const matrixB = random(
      [matrixSize, matrixSize],
      -1,
      1,
      { backend: 'webnn' }
    );
    
    // Create tensors for testing basic operations
    const tensorA = random([1000], -10, 10, { backend: 'webnn' });
    const tensorB = random([1000], -10, 10, { backend: 'webnn' });
    
    console.log('Running tensor operations on WebNN...');
    
    // Time the operations
    const results = {
      add: 0,
      multiply: 0,
      matmul: 0,
      relu: 0,
      sigmoid: 0,
      tanh: 0,
      supported: true
    };
    
    // Test element-wise addition
    console.log('Testing element-wise addition...');
    const startAdd = performance.now();
    const addResult = await backend.add(tensorA, tensorB);
    results.add = performance.now() - startAdd;
    
    // Test element-wise multiplication
    console.log('Testing element-wise multiplication...');
    const startMul = performance.now();
    const mulResult = await backend.multiply(tensorA, tensorB);
    results.multiply = performance.now() - startMul;
    
    // Test matrix multiplication
    console.log('Testing matrix multiplication...');
    const startMatmul = performance.now();
    const matmulResult = await backend.matmul(matrixA, matrixB);
    results.matmul = performance.now() - startMatmul;
    
    // Test ReLU activation
    console.log('Testing ReLU activation...');
    const startRelu = performance.now();
    const reluResult = await backend.relu(tensorA);
    results.relu = performance.now() - startRelu;
    
    // Test sigmoid activation
    console.log('Testing sigmoid activation...');
    const startSigmoid = performance.now();
    const sigmoidResult = await backend.sigmoid(tensorA);
    results.sigmoid = performance.now() - startSigmoid;
    
    // Test tanh activation
    console.log('Testing tanh activation...');
    const startTanh = performance.now();
    const tanhResult = await backend.tanh(tensorA);
    results.tanh = performance.now() - startTanh;
    
    // Print results
    console.log('WebNN Operation Benchmarks:');
    console.log(`- Element-wise addition: ${results.add.toFixed(2)}ms`);
    console.log(`- Element-wise multiplication: ${results.multiply.toFixed(2)}ms`);
    console.log(`- Matrix multiplication (${matrixSize}x${matrixSize}): ${results.matmul.toFixed(2)}ms`);
    console.log(`- ReLU activation: ${results.relu.toFixed(2)}ms`);
    console.log(`- Sigmoid activation: ${results.sigmoid.toFixed(2)}ms`);
    console.log(`- Tanh activation: ${results.tanh.toFixed(2)}ms`);
    
    // Clean up
    backend.dispose();
    
    return {
      supported: true,
      features,
      results,
      hasNPU
    };
  } catch (error) {
    console.error('WebNN example failed:', error);
    return {
      supported: false,
      error: error instanceof Error ? error.message : String(error),
      features: await detectWebNNFeatures(),
      results: null
    };
  }
}

// Run the example if in browser environment
if (typeof window !== 'undefined') {
  // Create output element
  const outputElement = document.createElement('div');
  outputElement.id = 'webnn-output';
  document.body.appendChild(outputElement);
  
  // Set initial content
  outputElement.innerHTML = `
    <h2>WebNN Tensor Operations</h2>
    <p>Detecting WebNN support...</p>
  `;
  
  // Run the example and update the output
  runWebNNTensorExample().then(result => {
    if (result.supported && result.results) {
      // Create features table based on detected features
      const featureRows = [];
      if (result.features) {
        featureRows.push(`<tr><td>API Supported</td><td>${result.features.supported ? '✅' : '❌'}</td></tr>`);
        featureRows.push(`<tr><td>Hardware Accelerated</td><td>${result.features.hardwareAccelerated ? '✅' : '❌'}</td></tr>`);
        featureRows.push(`<tr><td>Acceleration Type</td><td>${result.features.accelerationType || 'Unknown'}</td></tr>`);
        featureRows.push(`<tr><td>Neural Processor</td><td>${result.hasNPU ? '✅' : '❌'}</td></tr>`);
        featureRows.push(`<tr><td>Browser</td><td>${result.features.browser.name} ${result.features.browser.version}</td></tr>`);
        
        // Operation support
        if (result.features.supportedOperations) {
          const ops = result.features.supportedOperations;
          featureRows.push(`<tr><td>Basic Operations</td><td>${ops.basic ? '✅' : '❌'}</td></tr>`);
          featureRows.push(`<tr><td>Convolution</td><td>${ops.conv2d ? '✅' : '❌'}</td></tr>`);
          featureRows.push(`<tr><td>Pooling</td><td>${ops.pool ? '✅' : '❌'}</td></tr>`);
          featureRows.push(`<tr><td>Normalization</td><td>${ops.normalization ? '✅' : '❌'}</td></tr>`);
          featureRows.push(`<tr><td>Recurrent</td><td>${ops.recurrent ? '✅' : '❌'}</td></tr>`);
          featureRows.push(`<tr><td>Transformer</td><td>${ops.transformer ? '✅' : '❌'}</td></tr>`);
        }
      }
      
      // Create performance results HTML
      outputElement.innerHTML = `
        <h2>WebNN Tensor Operations</h2>
        <p>WebNN is supported in this browser!</p>
        
        <h3>WebNN Features</h3>
        <table border="1" cellpadding="5" cellspacing="0">
          <tr><th>Feature</th><th>Status</th></tr>
          ${featureRows.join('')}
        </table>
        
        <h3>Performance Results:</h3>
        <ul>
          <li>Element-wise addition: ${result.results.add.toFixed(2)}ms</li>
          <li>Element-wise multiplication: ${result.results.multiply.toFixed(2)}ms</li>
          <li>Matrix multiplication (512x512): ${result.results.matmul.toFixed(2)}ms</li>
          <li>ReLU activation: ${result.results.relu.toFixed(2)}ms</li>
          <li>Sigmoid activation: ${result.results.sigmoid.toFixed(2)}ms</li>
          <li>Tanh activation: ${result.results.tanh.toFixed(2)}ms</li>
        </ul>
        
        <p><strong>Acceleration Type:</strong> ${result.features?.accelerationType || 'Unknown'}</p>
        <p><strong>Neural Processor:</strong> ${result.hasNPU ? 'Detected' : 'Not detected'}</p>
      `;
    } else {
      outputElement.innerHTML = `
        <h2>WebNN Tensor Operations</h2>
        <p>WebNN is not supported in this browser.</p>
        <p>Error: ${result.error}</p>
        
        <h3>WebNN Feature Detection Results:</h3>
        <pre>${JSON.stringify(result.features, null, 2)}</pre>
        
        <p>WebNN requires a modern browser like:</p>
        <ul>
          <li>Chrome 113+ or Edge 113+</li>
          <li>Safari 17.4+</li>
          <li>Firefox with flags enabled (experimental)</li>
        </ul>
      `;
    }
  });
}