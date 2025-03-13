/**
 * Basic BERT example
 * 
 * This example shows how to use the IPFS Accelerate JS SDK to run a BERT model.
 */

import { initialize } from 'ipfs-accelerate-js';

async function runBertExample() {
  console.log('Initializing IPFS Accelerate JS SDK...');
  
  // Initialize the SDK
  const sdk = await initialize({
    logging: true,
    preferredBackends: ['webgpu', 'webnn', 'cpu'],
    enableCache: true
  });
  
  // Get hardware capabilities
  const hardware = sdk.hardware;
  const capabilities = hardware.getCapabilities();
  
  console.log('Hardware capabilities:');
  console.log('- WebGPU support:', capabilities.webgpu.supported);
  if (capabilities.webgpu.supported) {
    console.log('  - Vendor:', capabilities.webgpu.adapterInfo?.vendor);
    console.log('  - Device:', capabilities.webgpu.adapterInfo?.device);
    console.log('  - Simulated:', capabilities.webgpu.isSimulated);
  }
  
  console.log('- WebNN support:', capabilities.webnn.supported);
  if (capabilities.webnn.supported) {
    console.log('  - Device type:', capabilities.webnn.deviceType);
    console.log('  - Device name:', capabilities.webnn.deviceName);
    console.log('  - Simulated:', capabilities.webnn.isSimulated);
  }
  
  console.log('- Recommended backend:', capabilities.optimalBackend);
  
  // Create a BERT model
  console.log('Loading BERT model...');
  const bertModel = await sdk.createModel('bert-base-uncased', {
    preferredBackend: capabilities.optimalBackend
  });
  
  // Get model info
  const modelType = bertModel.getType();
  const modelName = bertModel.getName();
  const modelMetadata = bertModel.getMetadata();
  
  console.log('Model loaded:');
  console.log('- Type:', modelType);
  console.log('- Name:', modelName);
  console.log('- Hidden size:', modelMetadata.hiddenSize);
  console.log('- Layers:', modelMetadata.numLayers);
  console.log('- Vocab size:', modelMetadata.vocabSize);
  
  // Run inference
  console.log('Running inference...');
  const input = {
    input: 'Hello, world! This is a test of the BERT model.'
  };
  
  const result = await bertModel.predict(input);
  
  // Get dimensions of the output
  const lastHiddenState = result.lastHiddenState;
  const pooledOutput = result.pooledOutput;
  
  console.log('Inference complete:');
  console.log('- Last hidden state shape:', lastHiddenState.getDimensions());
  console.log('- Pooled output shape:', pooledOutput.getDimensions());
  
  // Extract embedding from pooled output
  const embedding = pooledOutput.getData();
  console.log('- First 5 values of embedding:', Array.from(embedding.slice(0, 5)));
  
  // Clean up
  lastHiddenState.dispose();
  pooledOutput.dispose();
  bertModel.dispose();
  
  console.log('Example complete!');
}

// Run the example
runBertExample().catch(error => {
  console.error('Error running example:', error);
});