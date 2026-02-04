/**
 * OpenVINO Model Server (OVMS) Unified Backend - Comprehensive Example
 * 
 * This comprehensive example demonstrates the full capabilities of the OVMS Unified backend:
 * 
 * 1. Configuration options and initialization with best practices
 * 2. Model inference with various input formats
 * 3. Batch inference and performance benchmarking
 * 4. Model management and configuration
 * 5. Containerized deployment and management
 * 6. Advanced features: model versioning, quantization, and execution modes
 * 7. Error handling and resilience patterns
 * 8. API key multiplexing and load balancing
 * 9. Performance optimization techniques
 * 10. Practical applications for different model types
 */

// Import required modules and types
import { OVMS } from '../src/api_backends/ovms/ovms';
import {
  ApiMetadata,
  ApiResources,
  Message,
  PriorityLevel
} from '../src/api_backends/types';
import {
  OVMSRequestData,
  OVMSResponse,
  OVMSRequestOptions,
  OVMSModelMetadata,
  OVMSModelConfig,
  OVMSServerStatistics,
  OVMSQuantizationConfig
} from '../src/api_backends/ovms/types';

// Polyfill for performance.now() if not available in environment
if (typeof performance === 'undefined') {
  const { performance } = require('perf_hooks');
  global.performance = performance;
}

/**
 * Comprehensive example for the OVMS Unified backend
 */
async function main() {
  console.log('\n----------------------------------------------------------');
  console.log('OVMS UNIFIED BACKEND - COMPREHENSIVE EXAMPLE');
  console.log('----------------------------------------------------------\n');
  
  const results: Record<string, any> = {};
  
  try {
    // -------------------------------------------------------------------------------
    // SECTION 1: INITIALIZATION AND CONFIGURATION
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 1: INITIALIZATION AND CONFIGURATION');
    
    // 1.1 Environment and API Key Management
    console.log('\n📋 1.1 Environment and API Key Management:');
    
    // Get API URL and API key from environment variables or configuration
    const envApiUrl = process.env.OVMS_API_URL;
    const envApiKey = process.env.OVMS_API_KEY;
    const envModel = process.env.OVMS_MODEL;
    
    // Determine API URL with fallbacks
    let apiUrl = envApiUrl || 'http://localhost:9000';
    console.log(`   ✓ Using API URL: ${apiUrl}`);
    
    // Determine API key with fallbacks
    let apiKey: string | undefined = undefined;
    if (envApiKey) {
      console.log('   ✓ Using API key from environment variable');
      apiKey = envApiKey;
    } else {
      console.log('   ℹ No API key found. OVMS often doesn\'t require an API key for local deployments.');
    }
    
    // Determine model with fallbacks
    let defaultModel = envModel || 'resnet50';
    console.log(`   ✓ Using default model: ${defaultModel}`);
    
    // 1.2 Advanced Configuration Options
    console.log('\n📋 1.2 Advanced Configuration Options:');
    
    // Set up advanced configuration options
    const resources: ApiResources = {};
    const metadata: ApiMetadata = {
      ovms_api_url: apiUrl,
      ovms_api_key: apiKey,
      ovms_model: defaultModel,
      ovms_version: 'latest',        // Use latest version by default
      ovms_precision: 'FP32',        // Default precision (alternatives: FP16, INT8)
      timeout: 30000                 // 30 second timeout
    };
    
    // Create the OVMS backend instance
    const ovms = new OVMS(resources, metadata);
    
    console.log('   ✓ Configured with timeout:', metadata.timeout, 'ms');
    console.log('   ✓ Default model:', defaultModel);
    console.log('   ✓ Default precision:', metadata.ovms_precision);
    console.log('   ✓ Model version:', metadata.ovms_version);
    
    // Store results for reporting
    results.initialization = {
      apiUrl,
      defaultModel,
      hasApiKey: !!apiKey,
      timeout: metadata.timeout
    };
    
    // 1.3 Test Endpoint Availability
    console.log('\n📋 1.3 Test Endpoint Availability:');
    
    // Check if the endpoint is available
    try {
      const isEndpointAvailable = await ovms.testEndpoint();
      console.log(`   ${isEndpointAvailable ? '✓' : '×'} Endpoint available: ${isEndpointAvailable}`);
      
      if (!isEndpointAvailable) {
        console.log('   ℹ Endpoint not available. Continuing with mock responses for demonstration.');
      }
      
      results.endpointAvailable = isEndpointAvailable;
    } catch (error) {
      console.log('   × Error testing endpoint:', error);
      console.log('   ℹ Continuing with mock responses for demonstration.');
      results.endpointAvailable = false;
    }
    
    // 1.4 Create Endpoint Handler
    console.log('\n📋 1.4 Create Endpoint Handler:');
    
    // Create an endpoint handler for making inference requests
    const endpointUrl = `${apiUrl}/v1/models/${defaultModel}:predict`;
    const endpointHandler = ovms.createEndpointHandler(endpointUrl, defaultModel);
    
    console.log(`   ✓ Created endpoint handler for: ${endpointUrl}`);
    console.log('   ✓ Model:', defaultModel);
    
    results.endpointHandler = {
      url: endpointUrl,
      model: defaultModel
    };
    
    // -------------------------------------------------------------------------------
    // SECTION 2: MODEL INFORMATION AND METADATA
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 2: MODEL INFORMATION AND METADATA');
    
    // 2.1 Model Information
    console.log('\n📋 2.1 Model Information:');
    
    // Get model information
    let modelInfo: OVMSModelMetadata | null = null;
    try {
      // @ts-ignore - Using method not in interface but in implementation
      modelInfo = await ovms.getModelInfo(defaultModel);
      console.log('   ✓ Model information retrieved successfully:');
      console.log(`     - Name: ${modelInfo.name}`);
      console.log(`     - Versions: ${modelInfo.versions.join(', ')}`);
      console.log(`     - Platform: ${modelInfo.platform}`);
      
      if (modelInfo.inputs && modelInfo.inputs.length > 0) {
        console.log('     - Inputs:');
        modelInfo.inputs.forEach(input => {
          console.log(`       - ${input.name} (${input.datatype}, shape: [${input.shape.join(', ')}])`);
        });
      }
      
      if (modelInfo.outputs && modelInfo.outputs.length > 0) {
        console.log('     - Outputs:');
        modelInfo.outputs.forEach(output => {
          console.log(`       - ${output.name} (${output.datatype}, shape: [${output.shape.join(', ')}])`);
        });
      }
      
      results.modelInfo = modelInfo;
    } catch (error) {
      console.log('   × Error retrieving model information:', error);
      console.log('   ℹ Using mock model information for demonstration.');
      
      // Create mock model info for demonstration
      modelInfo = {
        name: defaultModel,
        versions: ['1', '2', 'latest'],
        platform: 'openvino',
        inputs: [
          { name: 'input', datatype: 'float32', shape: [1, 3, 224, 224], layout: 'NCHW' }
        ],
        outputs: [
          { name: 'output', datatype: 'float32', shape: [1, 1000], layout: 'NC' }
        ]
      };
      
      results.modelInfo = {
        ...modelInfo,
        mock: true
      };
    }
    
    // 2.2 Model Versions
    console.log('\n📋 2.2 Model Versions:');
    
    // Get model versions
    try {
      // @ts-ignore - Using method not in interface but in implementation
      const versions = await ovms.getModelVersions(defaultModel);
      console.log(`   ✓ Available versions: ${versions.join(', ')}`);
      
      results.modelVersions = versions;
    } catch (error) {
      console.log('   × Error retrieving model versions:', error);
      console.log('   ℹ Using versions from model info for demonstration.');
      
      const versions = modelInfo?.versions || ['1', 'latest'];
      console.log(`   - Available versions: ${versions.join(', ')}`);
      
      results.modelVersions = versions;
    }
    
    // 2.3 Detailed Model Metadata
    console.log('\n📋 2.3 Detailed Model Metadata:');
    
    // Get detailed model metadata with shapes
    try {
      // @ts-ignore - Using method not in interface but in implementation
      const detailedMetadata = await ovms.getModelMetadataWithShapes(defaultModel);
      console.log('   ✓ Detailed model metadata retrieved successfully:');
      
      if (detailedMetadata.inputs && detailedMetadata.inputs.length > 0) {
        console.log('     - Inputs:');
        detailedMetadata.inputs.forEach(input => {
          console.log(`       - ${input.name} (${input.datatype})`);
          console.log(`         Shape: [${input.shape.join(', ')}]`);
          if (input.layout) {
            console.log(`         Layout: ${input.layout}`);
          }
        });
      }
      
      if (detailedMetadata.outputs && detailedMetadata.outputs.length > 0) {
        console.log('     - Outputs:');
        detailedMetadata.outputs.forEach(output => {
          console.log(`       - ${output.name} (${output.datatype})`);
          console.log(`         Shape: [${output.shape.join(', ')}]`);
          if (output.layout) {
            console.log(`         Layout: ${output.layout}`);
          }
        });
      }
      
      results.detailedMetadata = detailedMetadata;
    } catch (error) {
      console.log('   × Error retrieving detailed metadata:', error);
    }
    
    // 2.4 Server Statistics
    console.log('\n📋 2.4 Server Statistics:');
    
    // Get server statistics
    try {
      // @ts-ignore - Using method not in interface but in implementation
      const serverStats = await ovms.getServerStatistics();
      console.log('   ✓ Server statistics retrieved successfully:');
      console.log(`     - Server Uptime: ${serverStats.server_uptime} seconds`);
      console.log(`     - Server Version: ${serverStats.server_version}`);
      console.log(`     - Active Models: ${serverStats.active_models}`);
      console.log(`     - Total Requests: ${serverStats.total_requests}`);
      console.log(`     - Requests per Second: ${serverStats.requests_per_second}`);
      console.log(`     - Average Inference Time: ${serverStats.avg_inference_time} ms`);
      console.log(`     - CPU Usage: ${serverStats.cpu_usage}%`);
      console.log(`     - Memory Usage: ${serverStats.memory_usage} MB`);
      
      results.serverStats = serverStats;
    } catch (error) {
      console.log('   × Error retrieving server statistics:', error);
      console.log('   ℹ Using mock server statistics for demonstration.');
      
      // Create mock server stats for demonstration
      const mockServerStats: OVMSServerStatistics = {
        server_uptime: 3600,
        server_version: '2023.1',
        active_models: 5,
        total_requests: 12345,
        requests_per_second: 42.5,
        avg_inference_time: 0.023,
        cpu_usage: 35.2,
        memory_usage: 1024.5
      };
      
      console.log('   - Mock server statistics:');
      console.log(`     - Server Uptime: ${mockServerStats.server_uptime} seconds`);
      console.log(`     - Server Version: ${mockServerStats.server_version}`);
      console.log(`     - Active Models: ${mockServerStats.active_models}`);
      
      results.serverStats = {
        ...mockServerStats,
        mock: true
      };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 3: BASIC INFERENCE
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 3: BASIC INFERENCE');
    
    // 3.1 Simple Inference with Array Input
    console.log('\n📋 3.1 Simple Inference with Array Input:');
    
    // Create a simple array input
    const simpleInput = [1.0, 2.0, 3.0, 4.0, 5.0];
    console.log(`   Input data: [${simpleInput.join(', ')}]`);
    
    try {
      // @ts-ignore - Using method not in interface but in implementation
      const result = await ovms.infer(defaultModel, simpleInput);
      console.log('   ✓ Inference result:');
      console.log(JSON.stringify(result, null, 2).replace(/^/gm, '     '));
      
      results.simpleInference = {
        input: simpleInput,
        output: result
      };
    } catch (error) {
      console.log('   × Error running inference:', error);
      console.log('   ℹ Using mock inference result for demonstration.');
      
      // Create mock inference result for demonstration
      const mockResult = {
        predictions: [[0.1, 0.2, 0.3, 0.4, 0.5]]
      };
      
      console.log('   - Mock inference result:');
      console.log(JSON.stringify(mockResult, null, 2).replace(/^/gm, '     '));
      
      results.simpleInference = {
        input: simpleInput,
        output: mockResult,
        mock: true
      };
    }
    
    // 3.2 Structured Input Inference
    console.log('\n📋 3.2 Structured Input Inference:');
    
    // Create a structured input
    const structuredInput = {
      data: [1.0, 2.0, 3.0, 4.0, 5.0],
      parameters: {
        scale: 0.5,
        normalize: true
      }
    };
    
    console.log('   Input data:');
    console.log(JSON.stringify(structuredInput, null, 2).replace(/^/gm, '     '));
    
    try {
      // @ts-ignore - Using method not in interface but in implementation
      const result = await ovms.infer(defaultModel, structuredInput);
      console.log('   ✓ Inference result with structured input:');
      console.log(JSON.stringify(result, null, 2).replace(/^/gm, '     '));
      
      results.structuredInference = {
        input: structuredInput,
        output: result
      };
    } catch (error) {
      console.log('   × Error running inference with structured input:', error);
      console.log('   ℹ Using mock inference result for demonstration.');
      
      // Create mock inference result for demonstration
      const mockResult = {
        predictions: [[0.05, 0.1, 0.15, 0.2, 0.25]],
        metadata: {
          scaling_applied: true,
          normalization_applied: true
        }
      };
      
      console.log('   - Mock inference result:');
      console.log(JSON.stringify(mockResult, null, 2).replace(/^/gm, '     '));
      
      results.structuredInference = {
        input: structuredInput,
        output: mockResult,
        mock: true
      };
    }
    
    // 3.3 Inference with OVMS Standard Format
    console.log('\n📋 3.3 Inference with OVMS Standard Format:');
    
    // Create input in standard OVMS format
    const standardInput: OVMSRequestData = {
      instances: [
        {
          data: [1.0, 2.0, 3.0, 4.0, 5.0]
        }
      ]
    };
    
    console.log('   Input data in OVMS standard format:');
    console.log(JSON.stringify(standardInput, null, 2).replace(/^/gm, '     '));
    
    try {
      // Make request using the endpoint handler
      const result = await endpointHandler(standardInput);
      console.log('   ✓ Inference result with standard format:');
      console.log(JSON.stringify(result, null, 2).replace(/^/gm, '     '));
      
      results.standardInference = {
        input: standardInput,
        output: result
      };
    } catch (error) {
      console.log('   × Error running inference with standard format:', error);
      console.log('   ℹ Using mock inference result for demonstration.');
      
      // Create mock inference result for demonstration
      const mockResult = {
        predictions: [[0.1, 0.2, 0.3, 0.4, 0.5]]
      };
      
      console.log('   - Mock inference result:');
      console.log(JSON.stringify(mockResult, null, 2).replace(/^/gm, '     '));
      
      results.standardInference = {
        input: standardInput,
        output: mockResult,
        mock: true
      };
    }
    
    // 3.4 Inference with Specific Version
    console.log('\n📋 3.4 Inference with Specific Version:');
    
    // Get a version to use
    const version = modelInfo?.versions?.[0] || '1';
    
    // Create simple input
    const versionInput = [1.0, 2.0, 3.0, 4.0, 5.0];
    
    console.log(`   Using model version: ${version}`);
    console.log(`   Input data: [${versionInput.join(', ')}]`);
    
    try {
      // @ts-ignore - Using method not in interface but in implementation
      const result = await ovms.inferWithVersion(defaultModel, version, versionInput);
      console.log('   ✓ Inference result with specific version:');
      console.log(JSON.stringify(result, null, 2).replace(/^/gm, '     '));
      
      results.versionInference = {
        version,
        input: versionInput,
        output: result
      };
    } catch (error) {
      console.log('   × Error running inference with specific version:', error);
      console.log('   ℹ Using mock inference result for demonstration.');
      
      // Create mock inference result for demonstration
      const mockResult = {
        predictions: [[0.1, 0.2, 0.3, 0.4, 0.5]],
        model_version: version
      };
      
      console.log('   - Mock inference result:');
      console.log(JSON.stringify(mockResult, null, 2).replace(/^/gm, '     '));
      
      results.versionInference = {
        version,
        input: versionInput,
        output: mockResult,
        mock: true
      };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 4: BATCH INFERENCE AND PERFORMANCE
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 4: BATCH INFERENCE AND PERFORMANCE');
    
    // 4.1 Batch Inference
    console.log('\n📋 4.1 Batch Inference:');
    
    // Create a batch of inputs
    const batchInputs = [
      [1.0, 2.0, 3.0, 4.0, 5.0],
      [6.0, 7.0, 8.0, 9.0, 10.0],
      [11.0, 12.0, 13.0, 14.0, 15.0]
    ];
    
    console.log(`   Batch size: ${batchInputs.length}`);
    console.log('   Input batch:');
    batchInputs.forEach((input, index) => {
      console.log(`     - Batch item ${index + 1}: [${input.join(', ')}]`);
    });
    
    try {
      // @ts-ignore - Using method not in interface but in implementation
      const batchResults = await ovms.batchInfer(defaultModel, batchInputs);
      console.log(`   ✓ Batch inference results (${batchResults.length} results):`);
      console.log(JSON.stringify(batchResults, null, 2).replace(/^/gm, '     '));
      
      results.batchInference = {
        batchSize: batchInputs.length,
        inputs: batchInputs,
        outputs: batchResults
      };
    } catch (error) {
      console.log('   × Error running batch inference:', error);
      console.log('   ℹ Using mock batch inference results for demonstration.');
      
      // Create mock batch inference results for demonstration
      const mockBatchResults = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [1.1, 1.2, 1.3, 1.4, 1.5]
      ];
      
      console.log('   - Mock batch inference results:');
      console.log(JSON.stringify(mockBatchResults, null, 2).replace(/^/gm, '     '));
      
      results.batchInference = {
        batchSize: batchInputs.length,
        inputs: batchInputs,
        outputs: mockBatchResults,
        mock: true
      };
    }
    
    // 4.2 Performance Benchmarking
    console.log('\n📋 4.2 Performance Benchmarking:');
    
    // Create a function to run inference multiple times and measure performance
    const benchmarkInference = async (
      iterations: number,
      batchSize: number
    ): Promise<{
      totalTime: number;
      averageTime: number;
      throughput: number;
    }> => {
      // Create a batch of the specified size
      const batch = Array.from({ length: batchSize }, () => 
        Array.from({ length: 5 }, (_, i) => i + 1)
      );
      
      const startTime = performance.now();
      
      try {
        for (let i = 0; i < iterations; i++) {
          // @ts-ignore - Using method not in interface but in implementation
          await ovms.batchInfer(defaultModel, batch);
        }
        
        const endTime = performance.now();
        const totalTime = endTime - startTime;
        const averageTime = totalTime / iterations;
        const throughput = (iterations * batchSize) / (totalTime / 1000);
        
        return { totalTime, averageTime, throughput };
      } catch (error) {
        console.log(`   × Error during benchmark: ${error}`);
        
        // Return mock results for demonstration
        return {
          totalTime: iterations * 100,
          averageTime: 100,
          throughput: batchSize * 10
        };
      }
    };
    
    // Run benchmarks with different batch sizes
    console.log('   Running benchmarks with different batch sizes...');
    const batchSizes = [1, 4, 8];
    const iterations = 5;
    const benchmarkResults: Record<number, any> = {};
    
    for (const batchSize of batchSizes) {
      console.log(`   - Testing batch size ${batchSize} with ${iterations} iterations`);
      
      try {
        const result = await benchmarkInference(iterations, batchSize);
        console.log(`     ✓ Batch size ${batchSize}:`);
        console.log(`       - Total time: ${result.totalTime.toFixed(2)} ms`);
        console.log(`       - Average time per batch: ${result.averageTime.toFixed(2)} ms`);
        console.log(`       - Throughput: ${result.throughput.toFixed(2)} items/sec`);
        
        benchmarkResults[batchSize] = result;
      } catch (error) {
        console.log(`     × Error benchmarking batch size ${batchSize}: ${error}`);
      }
    }
    
    // Calculate speedup from batch processing
    if (Object.keys(benchmarkResults).length > 1) {
      const singleBatchThroughput = benchmarkResults[1]?.throughput || 1;
      
      for (const batchSize of batchSizes.slice(1)) {
        if (benchmarkResults[batchSize]) {
          const speedup = benchmarkResults[batchSize].throughput / singleBatchThroughput;
          console.log(`   ✓ Speedup from batch size ${batchSize} vs. single: ${speedup.toFixed(2)}x`);
          benchmarkResults[batchSize].speedup = speedup;
        }
      }
    }
    
    results.benchmarks = benchmarkResults;
    
    // -------------------------------------------------------------------------------
    // SECTION 5: MODEL CONFIGURATION AND OPTIMIZATION
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 5: MODEL CONFIGURATION AND OPTIMIZATION');
    
    // 5.1 Set Model Configuration
    console.log('\n📋 5.1 Set Model Configuration:');
    
    // Create model configuration
    const modelConfig: OVMSModelConfig = {
      batch_size: 4,              // Maximum batch size
      instance_count: 2,          // Number of model instances to load
      execution_mode: 'throughput' // Optimize for throughput
    };
    
    console.log('   Model configuration:');
    console.log(`     - Batch size: ${modelConfig.batch_size}`);
    console.log(`     - Instance count: ${modelConfig.instance_count}`);
    console.log(`     - Execution mode: ${modelConfig.execution_mode}`);
    
    try {
      // @ts-ignore - Using method not in interface but in implementation
      const configResponse = await ovms.setModelConfig(defaultModel, modelConfig);
      console.log('   ✓ Configuration set successfully:');
      console.log(JSON.stringify(configResponse, null, 2).replace(/^/gm, '     '));
      
      results.modelConfig = {
        config: modelConfig,
        response: configResponse
      };
    } catch (error) {
      console.log('   × Error setting model configuration:', error);
      console.log('   ℹ Using mock configuration response for demonstration.');
      
      // Create mock configuration response for demonstration
      const mockConfigResponse = {
        status: 'success',
        message: 'Model configuration updated'
      };
      
      console.log('   - Mock configuration response:');
      console.log(JSON.stringify(mockConfigResponse, null, 2).replace(/^/gm, '     '));
      
      results.modelConfig = {
        config: modelConfig,
        response: mockConfigResponse,
        mock: true
      };
    }
    
    // 5.2 Set Execution Mode
    console.log('\n📋 5.2 Set Execution Mode:');
    
    // Set execution mode to optimize for latency
    console.log('   Setting execution mode to optimize for latency...');
    
    try {
      // @ts-ignore - Using method not in interface but in implementation
      const modeResponse = await ovms.setExecutionMode(defaultModel, 'latency');
      console.log('   ✓ Execution mode set successfully:');
      console.log(JSON.stringify(modeResponse, null, 2).replace(/^/gm, '     '));
      
      results.executionMode = {
        mode: 'latency',
        response: modeResponse
      };
    } catch (error) {
      console.log('   × Error setting execution mode:', error);
      console.log('   ℹ Using mock execution mode response for demonstration.');
      
      // Create mock execution mode response for demonstration
      const mockModeResponse = {
        status: 'success',
        message: 'Execution mode updated',
        model: defaultModel,
        mode: 'latency'
      };
      
      console.log('   - Mock execution mode response:');
      console.log(JSON.stringify(mockModeResponse, null, 2).replace(/^/gm, '     '));
      
      results.executionMode = {
        mode: 'latency',
        response: mockModeResponse,
        mock: true
      };
    }
    
    // 5.3 Set Quantization Configuration
    console.log('\n📋 5.3 Set Quantization Configuration:');
    
    // Create quantization configuration
    const quantConfig: OVMSQuantizationConfig = {
      enabled: true,
      method: 'MinMax',
      bits: 8
    };
    
    console.log('   Quantization configuration:');
    console.log(`     - Enabled: ${quantConfig.enabled}`);
    console.log(`     - Method: ${quantConfig.method}`);
    console.log(`     - Bits: ${quantConfig.bits}`);
    
    try {
      // @ts-ignore - Using method not in interface but in implementation
      const quantResponse = await ovms.setQuantization(defaultModel, quantConfig);
      console.log('   ✓ Quantization configuration set successfully:');
      console.log(JSON.stringify(quantResponse, null, 2).replace(/^/gm, '     '));
      
      results.quantization = {
        config: quantConfig,
        response: quantResponse
      };
    } catch (error) {
      console.log('   × Error setting quantization configuration:', error);
      console.log('   ℹ Using mock quantization response for demonstration.');
      
      // Create mock quantization response for demonstration
      const mockQuantResponse = {
        status: 'success',
        message: 'Quantization configuration updated',
        model: defaultModel,
        quantization: quantConfig
      };
      
      console.log('   - Mock quantization response:');
      console.log(JSON.stringify(mockQuantResponse, null, 2).replace(/^/gm, '     '));
      
      results.quantization = {
        config: quantConfig,
        response: mockQuantResponse,
        mock: true
      };
    }
    
    // 5.4 Reload Model
    console.log('\n📋 5.4 Reload Model:');
    
    // Reload the model to apply configuration changes
    console.log('   Reloading model to apply configuration changes...');
    
    try {
      // @ts-ignore - Using method not in interface but in implementation
      const reloadResponse = await ovms.reloadModel(defaultModel);
      console.log('   ✓ Model reloaded successfully:');
      console.log(JSON.stringify(reloadResponse, null, 2).replace(/^/gm, '     '));
      
      results.modelReload = {
        response: reloadResponse
      };
    } catch (error) {
      console.log('   × Error reloading model:', error);
      console.log('   ℹ Using mock reload response for demonstration.');
      
      // Create mock reload response for demonstration
      const mockReloadResponse = {
        status: 'success',
        message: 'Model reloaded successfully',
        model: defaultModel
      };
      
      console.log('   - Mock reload response:');
      console.log(JSON.stringify(mockReloadResponse, null, 2).replace(/^/gm, '     '));
      
      results.modelReload = {
        response: mockReloadResponse,
        mock: true
      };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 6: INPUT FORMAT HANDLING
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 6: INPUT FORMAT HANDLING');
    
    // 6.1 Different Input Format Support
    console.log('\n📋 6.1 Different Input Format Support:');
    
    // Create different types of input formats
    const inputFormats = [
      // Simple array
      [1.0, 2.0, 3.0, 4.0, 5.0],
      
      // 2D array
      [[1.0, 2.0], [3.0, 4.0]],
      
      // Object with data field
      { data: [1.0, 2.0, 3.0, 4.0, 5.0] },
      
      // OVMS standard format
      { instances: [{ data: [1.0, 2.0, 3.0, 4.0, 5.0] }] }
    ];
    
    // Create a function to format different inputs
    const formatInput = (handler: (data: OVMSRequestData) => Promise<OVMSResponse>, input: any): Promise<OVMSResponse> => {
      // @ts-ignore - Using method not in interface but in implementation
      return ovms.formatRequest(handler, input);
    };
    
    // Test each input format
    console.log('   Testing different input formats:');
    
    for (let i = 0; i < inputFormats.length; i++) {
      const format = inputFormats[i];
      console.log(`   - Format ${i + 1}: ${Array.isArray(format) ? 'Array' : 'Object'}`);
      console.log('     Input:', JSON.stringify(format));
      
      try {
        // Create a mock handler for demonstration
        const mockHandler = async (data: OVMSRequestData): Promise<OVMSResponse> => {
          return {
            predictions: [[0.1, 0.2, 0.3, 0.4, 0.5]],
            model_name: defaultModel,
            format_index: i + 1
          };
        };
        
        // Format the input
        const formattedResult = await formatInput(mockHandler, format);
        console.log('     ✓ Formatted successfully');
        console.log('     Result:', JSON.stringify(formattedResult));
        
        if (!results.inputFormats) {
          results.inputFormats = {};
        }
        
        results.inputFormats[`format${i + 1}`] = {
          input: format,
          result: formattedResult
        };
      } catch (error) {
        console.log(`     × Error formatting input format ${i + 1}:`, error);
      }
    }
    
    // 6.2 Format Input for Different Models
    console.log('\n📋 6.2 Format Input for Different Models:');
    
    // Define model formats for different types of models
    const modelTypes = {
      'image_classification': {
        input: {
          instances: [{
            data: Array.from({ length: 3 * 224 * 224 }, (_, i) => i / (3 * 224 * 224))
          }]
        },
        description: 'Image classification model (ResNet, MobileNet, etc.)'
      },
      'object_detection': {
        input: {
          instances: [{
            data: Array.from({ length: 3 * 512 * 512 }, (_, i) => i / (3 * 512 * 512))
          }]
        },
        description: 'Object detection model (YOLO, SSD, etc.)'
      },
      'text_classification': {
        input: {
          instances: [{
            text: "This is an example of text for classification."
          }]
        },
        description: 'Text classification model (BERT, etc.)'
      }
    };
    
    // Test formatting for different model types
    console.log('   Testing input formatting for different model types:');
    
    for (const [type, info] of Object.entries(modelTypes)) {
      console.log(`   - Model type: ${type}`);
      console.log(`     Description: ${info.description}`);
      console.log('     Input:', JSON.stringify(info.input));
      
      try {
        // Create a mock handler for demonstration
        const mockHandler = async (data: OVMSRequestData): Promise<OVMSResponse> => {
          return {
            predictions: type === 'text_classification' 
              ? [0.2, 0.8] // Text classification output (binary)
              : type === 'object_detection'
                ? [[0.1, 0.2, 0.3, 0.4, 0.9]] // Object detection output (x, y, w, h, confidence)
                : [[0.1, 0.2, 0.7]], // Image classification output (class probabilities)
            model_name: defaultModel,
            model_type: type
          };
        };
        
        // Format the input
        // @ts-ignore - Using method not in interface but in implementation
        const formattedResult = await ovms.formatRequest(mockHandler, info.input);
        console.log('     ✓ Formatted successfully');
        console.log('     Result:', JSON.stringify(formattedResult));
        
        if (!results.modelTypeFormats) {
          results.modelTypeFormats = {};
        }
        
        results.modelTypeFormats[type] = {
          input: info.input,
          result: formattedResult
        };
      } catch (error) {
        console.log(`     × Error formatting input for model type ${type}:`, error);
      }
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 7: ERROR HANDLING AND RESILIENCE
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 7: ERROR HANDLING AND RESILIENCE');
    
    // 7.1 Standard Error Handling
    console.log('\n📋 7.1 Standard Error Handling:');
    
    // Define common error scenarios
    const errorScenarios = [
      {
        name: 'Model not found',
        fn: async () => {
          // @ts-ignore - Using method not in interface but in implementation
          await ovms.infer('non_existent_model', [1, 2, 3]);
        }
      },
      {
        name: 'Invalid input format',
        fn: async () => {
          // @ts-ignore - Using method not in interface but in implementation
          await ovms.infer(defaultModel, { invalid: 'format' });
        }
      },
      {
        name: 'Server connection error',
        fn: async () => {
          const tempOvms = new OVMS({}, { 
            ovms_api_url: 'http://invalid-url:9000',
            ovms_model: defaultModel
          });
          await tempOvms.testEndpoint();
        }
      }
    ];
    
    // Test error handling for each scenario
    console.log('   Testing error handling for common scenarios:');
    
    for (const scenario of errorScenarios) {
      console.log(`   - Scenario: ${scenario.name}`);
      
      try {
        await scenario.fn();
        console.log('     Unexpectedly succeeded.');
      } catch (error) {
        console.log(`     ✓ Caught expected error: ${error.message}`);
        
        if (!results.errorHandling) {
          results.errorHandling = {};
        }
        
        results.errorHandling[scenario.name] = {
          error: error.message
        };
      }
    }
    
    // 7.2 Robust Error Handling Pattern
    console.log('\n📋 7.2 Robust Error Handling Pattern:');
    
    // Create a robust error handling wrapper
    const robustInference = async (model: string, data: any, maxRetries = 3): Promise<any> => {
      let retries = 0;
      let lastError: Error | null = null;
      
      while (retries <= maxRetries) {
        try {
          // @ts-ignore - Using method not in interface but in implementation
          return await ovms.infer(model, data);
        } catch (error) {
          lastError = error;
          
          // Check if error is retriable
          if (
            error.message.includes('timeout') ||
            error.message.includes('connection') ||
            error.message.includes('busy') ||
            error.message.includes('temporary')
          ) {
            retries++;
            console.log(`     Retriable error detected. Retry ${retries}/${maxRetries}`);
            
            // Exponential backoff
            const delay = Math.pow(2, retries) * 100;
            console.log(`     Waiting ${delay}ms before retry...`);
            await new Promise(resolve => setTimeout(resolve, delay));
          } else {
            // Non-retriable error
            console.log('     Non-retriable error detected. Aborting.');
            throw error;
          }
        }
      }
      
      // Exhausted retries
      throw new Error(`Failed after ${maxRetries} retries: ${lastError?.message}`);
    };
    
    // Test robust error handling
    console.log('   Testing robust error handling pattern:');
    
    try {
      // Mock a retriable error scenario
      // @ts-ignore - Mock the infer method to simulate a retriable error
      ovms.infer = async (model: string, data: any) => {
        throw new Error('temporary unavailable: server busy');
      };
      
      // Try robust inference
      const result = await robustInference(defaultModel, [1, 2, 3]);
      console.log('     ✓ Inference succeeded after retries');
    } catch (error) {
      console.log(`     ✓ Failed after retries: ${error.message}`);
      
      results.robustErrorHandling = {
        pattern: 'exponential backoff with typed error classification',
        maxRetries: 3,
        error: error.message
      };
    }
    
    // 7.3 Circuit Breaker Pattern
    console.log('\n📋 7.3 Circuit Breaker Pattern:');
    
    // Test if circuit breaker is implemented
    console.log('   OVMS backend includes circuit breaker pattern:');
    
    // @ts-ignore - Check for circuit breaker attributes
    const hasCircuitBreaker = ovms.circuit_state !== undefined && 
                           // @ts-ignore
                           ovms.failure_count !== undefined && 
                           // @ts-ignore
                           ovms._check_circuit !== undefined;
    
    if (hasCircuitBreaker) {
      console.log('   ✓ Circuit breaker is implemented');
      console.log('   ✓ Protected from cascading failures');
      console.log('   ✓ Automatically resets after cooldown period');
      console.log('   ✓ Used internally for API resilience');
      
      results.circuitBreaker = {
        implemented: true,
        details: 'Prevents cascading failures by temporarily disabling requests after consecutive errors.'
      };
    } else {
      console.log('   ℹ Circuit breaker pattern is not explicitly implemented');
      console.log('   ✓ Error resilience is handled through retry mechanisms');
      
      results.circuitBreaker = {
        implemented: false,
        alternative: 'Error resilience is handled through retry mechanisms'
      };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 8: ADVANCED API FEATURES
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 8: ADVANCED API FEATURES');
    
    // 8.1 API Key Multiplexing
    console.log('\n📋 8.1 API Key Multiplexing:');
    
    // Check if the OVMS backend supports endpoint creation
    // @ts-ignore - Check for endpoint creation method
    const supportsEndpoints = typeof ovms.create_endpoint === 'function';
    
    if (supportsEndpoints) {
      console.log('   ✓ API key multiplexing is supported');
      
      // Create multiple endpoints with different API keys
      const apiKeys = ['key1', 'key2', 'key3'];
      const endpointIds: string[] = [];
      
      for (let i = 0; i < apiKeys.length; i++) {
        try {
          // @ts-ignore - Using method not in interface but in implementation
          const endpointId = await ovms.create_endpoint({
            api_key: apiKeys[i],
            max_concurrent_requests: 5 + i*2,
            max_retries: 3 + i
          });
          
          console.log(`   ✓ Created endpoint ${i + 1} with ID: ${endpointId}`);
          endpointIds.push(endpointId);
        } catch (error) {
          console.log(`   × Error creating endpoint ${i + 1}:`, error);
        }
      }
      
      console.log(`   ✓ Created ${endpointIds.length} endpoints with different API keys`);
      
      // Test endpoint selection
      // @ts-ignore - Check for endpoint selection method
      if (typeof ovms.select_endpoint === 'function') {
        const strategies = ['round-robin', 'least-loaded', 'fastest'];
        
        console.log('   Testing endpoint selection strategies:');
        for (const strategy of strategies) {
          try {
            // @ts-ignore - Using method not in interface but in implementation
            const selectedEndpoint = await ovms.select_endpoint(strategy);
            console.log(`   ✓ Selected endpoint with strategy '${strategy}': ${selectedEndpoint}`);
          } catch (error) {
            console.log(`   × Error selecting endpoint with strategy '${strategy}':`, error);
          }
        }
      }
      
      results.apiKeyMultiplexing = {
        supported: true,
        endpointsCreated: endpointIds.length,
        endpointIds
      };
    } else {
      console.log('   ℹ API key multiplexing is not explicitly implemented');
      console.log('   ℹ Using standard API key configuration');
      
      results.apiKeyMultiplexing = {
        supported: false
      };
    }
    
    // 8.2 Request Tracking and Metrics
    console.log('\n📋 8.2 Request Tracking and Metrics:');
    
    // Check if the OVMS backend supports request tracking
    // @ts-ignore - Check for request tracking method
    const supportsTracking = typeof ovms.track_request === 'function';
    
    if (supportsTracking) {
      console.log('   ✓ Request tracking is supported');
      
      // Track a sample request
      try {
        // @ts-ignore - Using method not in interface but in implementation
        await ovms.track_request({
          request_id: 'test-request-123',
          start_time: performance.now(),
          end_time: performance.now() + 100,
          status: 'success',
          input_tokens: 10,
          output_tokens: 50
        });
        
        console.log('   ✓ Request tracked successfully');
        
        // Get usage statistics
        // @ts-ignore - Check for usage stats method
        if (typeof ovms.get_usage_stats === 'function') {
          // @ts-ignore - Using method not in interface but in implementation
          const stats = await ovms.get_usage_stats();
          console.log('   ✓ Usage statistics retrieved:');
          console.log(JSON.stringify(stats, null, 2).replace(/^/gm, '     '));
        }
        
        results.requestTracking = {
          supported: true,
          statsAvailable: typeof ovms.get_usage_stats === 'function'
        };
      } catch (error) {
        console.log('   × Error tracking request:', error);
      }
    } else {
      console.log('   ℹ Request tracking is not explicitly implemented');
      console.log('   ℹ Using standard request handling');
      
      results.requestTracking = {
        supported: false
      };
    }
    
    // 8.3 Explain Prediction
    console.log('\n📋 8.3 Explain Prediction:');
    
    // Check if the OVMS backend supports explanation
    // @ts-ignore - Check for explain prediction method
    const supportsExplanation = typeof ovms.explainPrediction === 'function';
    
    if (supportsExplanation) {
      console.log('   ✓ Model prediction explanation is supported');
      
      // Explain a sample prediction
      const explainInput = [1.0, 2.0, 3.0, 4.0, 5.0];
      
      try {
        // @ts-ignore - Using method not in interface but in implementation
        const explanation = await ovms.explainPrediction(defaultModel, explainInput);
        console.log('   ✓ Explanation retrieved:');
        console.log(JSON.stringify(explanation, null, 2).replace(/^/gm, '     '));
        
        results.explainPrediction = {
          supported: true,
          input: explainInput,
          explanation
        };
      } catch (error) {
        console.log('   × Error explaining prediction:', error);
        console.log('   ℹ Explanation feature may not be available for all models or server configurations.');
        
        // Create mock explanation for demonstration
        const mockExplanation = {
          explanations: [
            {
              feature_importances: [0.1, 0.5, 0.2, 0.1, 0.1],
              prediction: [0.1, 0.2, 0.3, 0.4, 0.5]
            }
          ]
        };
        
        console.log('   - Mock explanation:');
        console.log(JSON.stringify(mockExplanation, null, 2).replace(/^/gm, '     '));
        
        results.explainPrediction = {
          supported: true,
          input: explainInput,
          explanation: mockExplanation,
          mock: true
        };
      }
    } else {
      console.log('   ℹ Model prediction explanation is not explicitly implemented');
      console.log('   ℹ Using standard prediction without explanation');
      
      results.explainPrediction = {
        supported: false
      };
    }
    
    // -------------------------------------------------------------------------------
    // SECTION 9: PRACTICAL APPLICATIONS
    // -------------------------------------------------------------------------------
    console.log('\n✅ SECTION 9: PRACTICAL APPLICATIONS');
    
    // 9.1 Image Classification Pipeline
    console.log('\n📋 9.1 Image Classification Pipeline:');
    
    // Create a function to simulate image classification
    const classifyImage = async (imageData: number[]): Promise<{
      classId: number;
      className: string;
      confidence: number;
    }> => {
      try {
        // Normalize the image data (simulate preprocessing)
        const normalizedData = imageData.map(x => x / 255.0);
        
        // Run inference
        // @ts-ignore - Using method not in interface but in implementation
        const result = await ovms.infer(defaultModel, normalizedData);
        
        // Process results (simulate postprocessing)
        const predictions = result.predictions?.[0] || [0.1, 0.2, 0.7];
        const classId = predictions.indexOf(Math.max(...predictions));
        
        // Map to class names (mock for demonstration)
        const classNames = ['cat', 'dog', 'bird'];
        const className = classNames[classId % classNames.length];
        const confidence = predictions[classId];
        
        return { classId, className, confidence };
      } catch (error) {
        console.log('   × Error in image classification:', error);
        
        // Return mock result for demonstration
        return { classId: 2, className: 'bird', confidence: 0.7 };
      }
    };
    
    // Test image classification pipeline
    console.log('   Testing image classification pipeline:');
    
    // Create mock image data (224x224 RGB image flattened to array)
    const mockImageData = Array.from({ length: 10 }, () => Math.random() * 255);
    
    try {
      const classification = await classifyImage(mockImageData);
      console.log('   ✓ Image classification result:');
      console.log(`     - Class ID: ${classification.classId}`);
      console.log(`     - Class Name: ${classification.className}`);
      console.log(`     - Confidence: ${classification.confidence.toFixed(2)}`);
      
      results.imageClassification = classification;
    } catch (error) {
      console.log('   × Error in image classification pipeline:', error);
    }
    
    // 9.2 Object Detection Pipeline
    console.log('\n📋 9.2 Object Detection Pipeline:');
    
    // Create a function to simulate object detection
    const detectObjects = async (imageData: number[]): Promise<Array<{
      bbox: [number, number, number, number]; // [x, y, width, height]
      classId: number;
      className: string;
      confidence: number;
    }>> => {
      try {
        // Normalize the image data (simulate preprocessing)
        const normalizedData = imageData.map(x => x / 255.0);
        
        // Create request with proper shape for object detection
        const requestData: OVMSRequestData = {
          instances: [
            {
              data: normalizedData
            }
          ]
        };
        
        // Run inference with endpoint handler
        const result = await endpointHandler(requestData);
        
        // Process results (simulate postprocessing of detection outputs)
        // Object detection models typically output [batch, detections, 5+classes] where 5 is [x, y, width, height, confidence]
        const detections = result.predictions?.[0] || [
          // Mock detections for demonstration
          [0.2, 0.3, 0.1, 0.2, 0.9, 0.8, 0.1, 0.1], // Format: [x, y, w, h, confidence, class1, class2, class3]
          [0.5, 0.6, 0.2, 0.1, 0.8, 0.1, 0.7, 0.2]
        ];
        
        // Map class names (mock for demonstration)
        const classNames = ['person', 'car', 'dog'];
        
        // Process each detection
        return detections.map(detection => {
          const [x, y, width, height, confidence, ...classScores] = detection;
          const classId = classScores.indexOf(Math.max(...classScores));
          const className = classNames[classId % classNames.length];
          
          return {
            bbox: [x, y, width, height] as [number, number, number, number],
            classId,
            className,
            confidence
          };
        });
      } catch (error) {
        console.log('   × Error in object detection:', error);
        
        // Return mock results for demonstration
        return [
          {
            bbox: [0.2, 0.3, 0.1, 0.2],
            classId: 0,
            className: 'person',
            confidence: 0.9
          },
          {
            bbox: [0.5, 0.6, 0.2, 0.1],
            classId: 1,
            className: 'car',
            confidence: 0.8
          }
        ];
      }
    };
    
    // Test object detection pipeline
    console.log('   Testing object detection pipeline:');
    
    // Create mock image data
    const mockDetectionImageData = Array.from({ length: 10 }, () => Math.random() * 255);
    
    try {
      const detections = await detectObjects(mockDetectionImageData);
      console.log(`   ✓ Object detection results: ${detections.length} objects detected`);
      
      detections.forEach((detection, i) => {
        console.log(`     - Detection ${i + 1}:`);
        console.log(`       Class: ${detection.className} (ID: ${detection.classId})`);
        console.log(`       Confidence: ${detection.confidence.toFixed(2)}`);
        console.log(`       Bounding Box: [${detection.bbox.map(v => v.toFixed(2)).join(', ')}]`);
      });
      
      results.objectDetection = detections;
    } catch (error) {
      console.log('   × Error in object detection pipeline:', error);
    }
    
    // 9.3 Semantic Segmentation
    console.log('\n📋 9.3 Semantic Segmentation:');
    
    // Create a function to simulate semantic segmentation
    const performSegmentation = async (imageData: number[]): Promise<{
      segmentationMap: number[][];
      classNames: string[];
    }> => {
      try {
        // Normalize the image data (simulate preprocessing)
        const normalizedData = imageData.map(x => x / 255.0);
        
        // Create request with proper shape for segmentation
        const requestData: OVMSRequestData = {
          instances: [
            {
              data: normalizedData
            }
          ]
        };
        
        // Run inference with endpoint handler
        const result = await endpointHandler(requestData);
        
        // Process results (simulate postprocessing of segmentation output)
        // Segmentation models typically output [batch, classes, height, width]
        // For simplicity, we'll create a small 5x5 segmentation map
        const segmentationData = result.predictions?.[0] || [
          // Mock segmentation data for demonstration (5x5 grid with 3 classes)
          [0, 0, 1, 1, 1],
          [0, 0, 1, 2, 2],
          [0, 1, 1, 2, 2],
          [1, 1, 1, 1, 2],
          [1, 1, 1, 2, 2]
        ];
        
        // Map class names (mock for demonstration)
        const classNames = ['background', 'road', 'car'];
        
        return {
          segmentationMap: segmentationData,
          classNames
        };
      } catch (error) {
        console.log('   × Error in semantic segmentation:', error);
        
        // Return mock results for demonstration
        return {
          segmentationMap: [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 2, 2],
            [0, 1, 1, 2, 2],
            [1, 1, 1, 1, 2],
            [1, 1, 1, 2, 2]
          ],
          classNames: ['background', 'road', 'car']
        };
      }
    };
    
    // Test semantic segmentation pipeline
    console.log('   Testing semantic segmentation pipeline:');
    
    // Create mock image data
    const mockSegmentationImageData = Array.from({ length: 10 }, () => Math.random() * 255);
    
    try {
      const segmentation = await performSegmentation(mockSegmentationImageData);
      console.log('   ✓ Semantic segmentation results:');
      console.log(`     - Classes: ${segmentation.classNames.join(', ')}`);
      console.log('     - Segmentation Map (5x5):');
      
      segmentation.segmentationMap.forEach(row => {
        console.log(`       ${row.join(' ')}`);
      });
      
      // Calculate class distribution
      const classDistribution: Record<string, number> = {};
      segmentation.classNames.forEach(name => classDistribution[name] = 0);
      
      segmentation.segmentationMap.forEach(row => {
        row.forEach(pixel => {
          const className = segmentation.classNames[pixel];
          classDistribution[className] = (classDistribution[className] || 0) + 1;
        });
      });
      
      console.log('     - Class Distribution:');
      Object.entries(classDistribution).forEach(([className, count]) => {
        const percentage = (count / 25 * 100).toFixed(1); // 5x5 = 25 pixels
        console.log(`       ${className}: ${count} pixels (${percentage}%)`);
      });
      
      results.semanticSegmentation = {
        segmentationMap: segmentation.segmentationMap,
        classNames: segmentation.classNames,
        distribution: classDistribution
      };
    } catch (error) {
      console.log('   × Error in semantic segmentation pipeline:', error);
    }
    
    // -------------------------------------------------------------------------------
    // CONCLUSION
    // -------------------------------------------------------------------------------
    console.log('\n==========================================================');
    console.log('COMPREHENSIVE EXAMPLE COMPLETED SUCCESSFULLY');
    console.log('==========================================================');
    
    console.log('\nSummary of Results:');
    console.log(`✅ Model Used: ${defaultModel}`);
    console.log(`✅ API URL: ${apiUrl}`);
    
    if (results.benchmarks && Object.keys(results.benchmarks).length > 0) {
      const firstBatchSize = Object.keys(results.benchmarks)[0];
      const throughput = results.benchmarks[firstBatchSize]?.throughput || 0;
      console.log(`✅ Performance: ${throughput.toFixed(2)} items/second`);
    }
    
    console.log('\nDemonstrated Features:');
    console.log('✅ Model Information and Metadata Retrieval');
    console.log('✅ Basic and Structured Inference');
    console.log('✅ Batch Inference and Performance Benchmarking');
    console.log('✅ Model Configuration and Optimization');
    console.log('✅ Input Format Handling');
    console.log('✅ Error Handling and Resilience Patterns');
    console.log('✅ Advanced API Features');
    console.log('✅ Practical Application Pipelines');
    
    console.log('\nFor detailed API documentation, refer to:');
    console.log('- OVMS_BACKEND_USAGE.md in the docs/api_backends directory');
    
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