/**
 * Example usage of the OpenVINO Model Server (OVMS) API Backend
 * 
 * This example demonstrates how to use the OVMS backend for various tasks
 * including model inference, batch processing, and configuration.
 */

import { OVMS } from '../src/api_backends/ovms/ovms';
import { ApiMetadata, ApiResources, Message } from '../src/api_backends/types';
import { OVMSModelConfig, OVMSQuantizationConfig } from '../src/api_backends/ovms/types';

// Environment variables can be used for configuration:
// process.env.OVMS_API_URL = 'http://your-ovms-server:9000';
// process.env.OVMS_MODEL = 'your-model-name';
// process.env.OVMS_VERSION = 'your-model-version';
// process.env.OVMS_API_KEY = 'your-api-key';

async function main() {
  console.log('OpenVINO Model Server (OVMS) Backend Example');
  console.log('============================================\n');

  // 1. Initialize the OVMS backend
  console.log('1. Initializing OVMS backend...');
  
  const resources: ApiResources = {};
  const metadata: ApiMetadata = {
    ovms_api_url: 'http://localhost:9000',  // Replace with your OVMS server URL
    ovms_model: 'bert-base-uncased',        // Replace with your model name
    ovms_version: 'latest',                 // Replace with your model version
    ovms_precision: 'FP32',                 // Precision (FP32, FP16, INT8)
    ovms_api_key: 'demo_api_key'            // Replace with your API key if needed
  };
  
  const ovms = new OVMS(resources, metadata);
  console.log('OVMS backend initialized with the following settings:');
  console.log(`  - API URL: ${metadata.ovms_api_url}`);
  console.log(`  - Model: ${metadata.ovms_model}`);
  console.log(`  - Version: ${metadata.ovms_version}`);
  console.log(`  - Precision: ${metadata.ovms_precision}`);
  console.log(`  - API Key: ${metadata.ovms_api_key ? '****' + metadata.ovms_api_key.slice(-4) : 'Not provided'}`);
  
  // 2. Test the endpoint connection
  console.log('\n2. Testing endpoint connection...');
  try {
    const isConnected = await ovms.testEndpoint();
    console.log(`  Connection test: ${isConnected ? 'SUCCESS' : 'FAILED'}`);
    
    if (!isConnected) {
      console.log('  Unable to connect to OVMS server. Please check your server URL and make sure the server is running.');
      return;
    }
  } catch (error) {
    console.error('  Error testing endpoint:', error);
    return;
  }
  
  // 3. Get model information
  console.log('\n3. Getting model information...');
  try {
    // @ts-ignore - Using method not in interface but in implementation
    const modelInfo = await ovms.getModelInfo();
    console.log('  Model information:');
    console.log(`  - Name: ${modelInfo.name}`);
    console.log(`  - Versions: ${modelInfo.versions.join(', ')}`);
    console.log(`  - Platform: ${modelInfo.platform}`);
    
    if (modelInfo.inputs) {
      console.log('  - Inputs:');
      modelInfo.inputs.forEach(input => {
        console.log(`    - ${input.name} (${input.datatype}, shape: [${input.shape.join(', ')}])`);
      });
    }
    
    if (modelInfo.outputs) {
      console.log('  - Outputs:');
      modelInfo.outputs.forEach(output => {
        console.log(`    - ${output.name} (${output.datatype}, shape: [${output.shape.join(', ')}])`);
      });
    }
  } catch (error) {
    console.error('  Error getting model information:', error);
  }
  
  // 4. Get model versions
  console.log('\n4. Getting model versions...');
  try {
    // @ts-ignore - Using method not in interface but in implementation
    const versions = await ovms.getModelVersions();
    console.log(`  Available versions: ${versions.join(', ')}`);
  } catch (error) {
    console.error('  Error getting model versions:', error);
  }
  
  // 5. Get model status
  console.log('\n5. Getting model status...');
  try {
    // @ts-ignore - Using method not in interface but in implementation
    const status = await ovms.getModelStatus();
    console.log('  Model status:');
    console.log(JSON.stringify(status, null, 2).replace(/^/gm, '  '));
  } catch (error) {
    console.error('  Error getting model status:', error);
  }
  
  // 6. Run inference with a simple input
  console.log('\n6. Running inference with a simple input...');
  try {
    const input = [1, 2, 3, 4, 5];  // Example input array
    
    // @ts-ignore - Using method not in interface but in implementation
    const result = await ovms.infer(metadata.ovms_model, input);
    console.log('  Inference result:');
    console.log(JSON.stringify(result, null, 2).replace(/^/gm, '  '));
  } catch (error) {
    console.error('  Error running inference:', error);
  }
  
  // 7. Run inference with structured input
  console.log('\n7. Running inference with structured input...');
  try {
    // Example structured input for a text classification model
    const structuredInput = {
      text: "OpenVINO is a great tool for inference optimization.",
      options: {
        max_length: 128,
        return_tensors: true
      }
    };
    
    // @ts-ignore - Using method not in interface but in implementation
    const result = await ovms.infer(metadata.ovms_model, structuredInput);
    console.log('  Inference result with structured input:');
    console.log(JSON.stringify(result, null, 2).replace(/^/gm, '  '));
  } catch (error) {
    console.error('  Error running inference with structured input:', error);
  }
  
  // 8. Run batch inference
  console.log('\n8. Running batch inference...');
  try {
    // Batch of inputs
    const batchInputs = [
      [1, 2, 3, 4, 5],
      [6, 7, 8, 9, 10],
      [11, 12, 13, 14, 15]
    ];
    
    // @ts-ignore - Using method not in interface but in implementation
    const batchResults = await ovms.batchInfer(metadata.ovms_model, batchInputs);
    console.log(`  Batch inference results (${batchResults.length} results):`);
    console.log(JSON.stringify(batchResults, null, 2).replace(/^/gm, '  '));
  } catch (error) {
    console.error('  Error running batch inference:', error);
  }
  
  // 9. Set model configuration
  console.log('\n9. Setting model configuration...');
  try {
    const config: OVMSModelConfig = {
      batch_size: 4,              // Maximum batch size
      instance_count: 2,          // Number of model instances to load
      execution_mode: 'throughput' // Optimize for throughput (alternative: 'latency')
    };
    
    // @ts-ignore - Using method not in interface but in implementation
    const configResponse = await ovms.setModelConfig(metadata.ovms_model, config);
    console.log('  Configuration set successfully:');
    console.log(JSON.stringify(configResponse, null, 2).replace(/^/gm, '  '));
  } catch (error) {
    console.error('  Error setting model configuration:', error);
  }
  
  // 10. Set execution mode
  console.log('\n10. Setting execution mode to optimize for latency...');
  try {
    // @ts-ignore - Using method not in interface but in implementation
    const modeResponse = await ovms.setExecutionMode(metadata.ovms_model, 'latency');
    console.log('  Execution mode set successfully:');
    console.log(JSON.stringify(modeResponse, null, 2).replace(/^/gm, '  '));
  } catch (error) {
    console.error('  Error setting execution mode:', error);
  }
  
  // 11. Get server statistics
  console.log('\n11. Getting server statistics...');
  try {
    // @ts-ignore - Using method not in interface but in implementation
    const stats = await ovms.getServerStatistics();
    console.log('  Server statistics:');
    console.log(`  - Uptime: ${stats.server_uptime} seconds`);
    console.log(`  - Server version: ${stats.server_version}`);
    console.log(`  - Active models: ${stats.active_models}`);
    console.log(`  - Total requests: ${stats.total_requests}`);
    console.log(`  - Requests per second: ${stats.requests_per_second}`);
    console.log(`  - Average inference time: ${stats.avg_inference_time} ms`);
    console.log(`  - CPU usage: ${stats.cpu_usage}%`);
    console.log(`  - Memory usage: ${stats.memory_usage} MB`);
  } catch (error) {
    console.error('  Error getting server statistics:', error);
  }
  
  // 12. Inference with specific version
  console.log('\n12. Running inference with specific model version...');
  try {
    const input = [1, 2, 3, 4, 5];
    const version = '1';  // Replace with a valid version from your model
    
    // @ts-ignore - Using method not in interface but in implementation
    const result = await ovms.inferWithVersion(metadata.ovms_model, version, input);
    console.log(`  Inference result with version ${version}:`);
    console.log(JSON.stringify(result, null, 2).replace(/^/gm, '  '));
  } catch (error) {
    console.error('  Error running inference with specific version:', error);
  }
  
  // 13. Get model metadata with shapes
  console.log('\n13. Getting detailed model metadata with shapes...');
  try {
    // @ts-ignore - Using method not in interface but in implementation
    const detailedMetadata = await ovms.getModelMetadataWithShapes();
    console.log('  Detailed model metadata:');
    
    if (detailedMetadata.inputs) {
      console.log('  - Inputs:');
      detailedMetadata.inputs.forEach(input => {
        console.log(`    - ${input.name} (${input.datatype})`);
        console.log(`      Shape: [${input.shape.join(', ')}]`);
        if (input.layout) {
          console.log(`      Layout: ${input.layout}`);
        }
      });
    }
    
    if (detailedMetadata.outputs) {
      console.log('  - Outputs:');
      detailedMetadata.outputs.forEach(output => {
        console.log(`    - ${output.name} (${output.datatype})`);
        console.log(`      Shape: [${output.shape.join(', ')}]`);
        if (output.layout) {
          console.log(`      Layout: ${output.layout}`);
        }
      });
    }
  } catch (error) {
    console.error('  Error getting detailed model metadata:', error);
  }
  
  // 14. Set quantization configuration
  console.log('\n14. Setting quantization configuration...');
  try {
    const quantConfig: OVMSQuantizationConfig = {
      enabled: true,
      method: 'MinMax',
      bits: 8
    };
    
    // @ts-ignore - Using method not in interface but in implementation
    const quantResponse = await ovms.setQuantization(metadata.ovms_model, quantConfig);
    console.log('  Quantization configuration set successfully:');
    console.log(JSON.stringify(quantResponse, null, 2).replace(/^/gm, '  '));
  } catch (error) {
    console.error('  Error setting quantization configuration:', error);
  }
  
  // 15. Reload model
  console.log('\n15. Reloading model...');
  try {
    // @ts-ignore - Using method not in interface but in implementation
    const reloadResponse = await ovms.reloadModel(metadata.ovms_model);
    console.log('  Model reloaded successfully:');
    console.log(JSON.stringify(reloadResponse, null, 2).replace(/^/gm, '  '));
  } catch (error) {
    console.error('  Error reloading model:', error);
  }
  
  // 16. Using the chat interface (adaption to non-chat model)
  console.log('\n16. Using the chat interface with OVMS (adaptation)...');
  try {
    const messages: Message[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'What is the capital of France?' }
    ];
    
    const chatResponse = await ovms.chat(messages);
    console.log('  Chat response (adapted from non-chat model):');
    console.log(`  - Role: ${chatResponse.role}`);
    console.log(`  - Model: ${chatResponse.model}`);
    console.log(`  - Content: ${chatResponse.content}`);
  } catch (error) {
    console.error('  Error using chat interface:', error);
  }
  
  // 17. Explain model prediction (if available)
  console.log('\n17. Getting explanation for model prediction...');
  try {
    const input = [1, 2, 3, 4, 5];
    
    // @ts-ignore - Using method not in interface but in implementation
    const explanation = await ovms.explainPrediction(metadata.ovms_model, input);
    console.log('  Prediction explanation:');
    console.log(JSON.stringify(explanation, null, 2).replace(/^/gm, '  '));
  } catch (error) {
    console.error('  Error getting prediction explanation:', error);
    console.log('  Note: Explanation feature may not be available for all models or OVMS server configurations.');
  }
  
  console.log('\nExample completed successfully!');
}

// Run the example
main().catch(error => {
  console.error('Example failed with error:', error);
});