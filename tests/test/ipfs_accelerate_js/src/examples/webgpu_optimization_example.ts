/**
 * WebGPU Optimization Example
 * Demonstrates the performance improvements from WebGPU optimizations
 */

import { Tensor, random } from '../tensor/tensor';
import { WebGPUBackend } from '../hardware/webgpu/backend';
import { FusionOpType } from '../hardware/webgpu/optimizations/operation_fusion';

/**
 * Runs benchmark with optimizations enabled vs disabled
 */
export async function runWebGPUOptimizationBenchmark() {
  // Check if WebGPU is available
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    console.error('WebGPU is not available in this environment');
    return {
      supported: false,
      error: 'WebGPU not available',
      results: null
    };
  }
  
  try {
    console.log('Running WebGPU optimization benchmark...');
    
    // Create two WebGPU backends - one with optimizations, one without
    const optimizedBackend = new WebGPUBackend({
      enableOperationFusion: true,
      enableSpecializedShaders: true,
      enableBrowserOptimizations: true
    });
    
    const standardBackend = new WebGPUBackend({
      enableOperationFusion: false,
      enableSpecializedShaders: false,
      enableBrowserOptimizations: false
    });
    
    // Initialize both backends
    await Promise.all([
      optimizedBackend.initialize(),
      standardBackend.initialize()
    ]);
    
    console.log('WebGPU backends initialized successfully');
    
    // Define the benchmark scenarios
    const scenarios = [
      {
        name: 'Small Matrix Multiplication (128x128)',
        size: 128,
        operation: 'matmul'
      },
      {
        name: 'Medium Matrix Multiplication (512x512)',
        size: 512,
        operation: 'matmul'
      },
      {
        name: 'Large Matrix Multiplication (1024x1024)',
        size: 1024,
        operation: 'matmul'
      },
      {
        name: 'ReLU Activation (100,000 elements)',
        size: 100000,
        operation: 'relu'
      },
      {
        name: 'Sigmoid Activation (100,000 elements)',
        size: 100000,
        operation: 'sigmoid'
      },
      {
        name: 'Element-wise Addition (100,000 elements)',
        size: 100000,
        operation: 'add'
      }
    ];
    
    // Add fusion benchmark scenarios
    const fusionScenarios = [
      {
        name: 'MatMul + ReLU Fusion (512x512)',
        size: 512,
        operations: ['matmul', 'relu'],
        type: 'fusion'
      },
      {
        name: 'Add + ReLU Fusion (100,000 elements)',
        size: 100000,
        operations: ['add', 'relu'],
        type: 'fusion'
      },
      {
        name: 'MatMul + Sigmoid Fusion (512x512)',
        size: 512,
        operations: ['matmul', 'sigmoid'],
        type: 'fusion'
      },
      {
        name: 'Element-wise Chain (Add + Multiply) (100,000 elements)',
        size: 100000,
        operations: ['add', 'multiply'],
        type: 'fusion'
      }
    ];
    
    // Combine all scenarios
    const allScenarios = [...scenarios, ...fusionScenarios];
    
    // Run benchmarks
    const results = [];
    
    for (const scenario of allScenarios) {
      console.log(`Running benchmark: ${scenario.name}`);
      
      let standardTensor1, standardTensor2, standardTensor3, optimizedTensor1, optimizedTensor2, optimizedTensor3;
      
      // Create tensors based on operation type
      if (scenario.operation === 'matmul' || (scenario.type === 'fusion' && scenario.operations[0] === 'matmul')) {
        // For matmul, create square matrices
        standardTensor1 = random([scenario.size, scenario.size], -1, 1, { backend: 'webgpu' });
        standardTensor2 = random([scenario.size, scenario.size], -1, 1, { backend: 'webgpu' });
        
        optimizedTensor1 = random([scenario.size, scenario.size], -1, 1, { backend: 'webgpu' });
        optimizedTensor2 = random([scenario.size, scenario.size], -1, 1, { backend: 'webgpu' });
      } else if (scenario.operation === 'relu' || scenario.operation === 'sigmoid') {
        // For activations, create 1D tensors
        standardTensor1 = random([scenario.size], -10, 10, { backend: 'webgpu' });
        optimizedTensor1 = random([scenario.size], -10, 10, { backend: 'webgpu' });
      } else if (scenario.operation === 'add' || (scenario.type === 'fusion' && scenario.operations[0] === 'add')) {
        // For element-wise operations
        standardTensor1 = random([scenario.size], -10, 10, { backend: 'webgpu' });
        standardTensor2 = random([scenario.size], -10, 10, { backend: 'webgpu' });
        
        optimizedTensor1 = random([scenario.size], -10, 10, { backend: 'webgpu' });
        optimizedTensor2 = random([scenario.size], -10, 10, { backend: 'webgpu' });
        
        // For element-wise chain (Add + Multiply), need a third tensor
        if (scenario.type === 'fusion' && scenario.operations.length > 1 && scenario.operations[1] === 'multiply') {
          standardTensor3 = random([scenario.size], -10, 10, { backend: 'webgpu' });
          optimizedTensor3 = random([scenario.size], -10, 10, { backend: 'webgpu' });
        }
      }
      
      // Warmup iterations
      if (scenario.type !== 'fusion') {
        if (scenario.operation === 'matmul') {
          await standardBackend.matmul(standardTensor1, standardTensor2);
          await optimizedBackend.matmul(optimizedTensor1, optimizedTensor2);
        } else if (scenario.operation === 'relu') {
          await standardBackend.relu(standardTensor1);
          await optimizedBackend.relu(optimizedTensor1);
        } else if (scenario.operation === 'sigmoid') {
          await standardBackend.sigmoid(standardTensor1);
          await optimizedBackend.sigmoid(optimizedTensor1);
        } else if (scenario.operation === 'add') {
          await standardBackend.add(standardTensor1, standardTensor2);
          await optimizedBackend.add(optimizedTensor1, optimizedTensor2);
        }
      } else {
        // Warmup for fusion scenarios (sequential operations)
        if (scenario.operations[0] === 'matmul') {
          const stdTemp = await standardBackend.matmul(standardTensor1, standardTensor2);
          const optTemp = await optimizedBackend.matmul(optimizedTensor1, optimizedTensor2);
          
          if (scenario.operations[1] === 'relu') {
            await standardBackend.relu(stdTemp);
            await optimizedBackend.relu(optTemp);
          } else if (scenario.operations[1] === 'sigmoid') {
            await standardBackend.sigmoid(stdTemp);
            await optimizedBackend.sigmoid(optTemp);
          }
        } else if (scenario.operations[0] === 'add') {
          const stdTemp = await standardBackend.add(standardTensor1, standardTensor2);
          const optTemp = await optimizedBackend.add(optimizedTensor1, optimizedTensor2);
          
          if (scenario.operations[1] === 'relu') {
            await standardBackend.relu(stdTemp);
            await optimizedBackend.relu(optTemp);
          } else if (scenario.operations[1] === 'multiply') {
            await standardBackend.multiply(stdTemp, standardTensor3);
            await optimizedBackend.multiply(optTemp, optimizedTensor3);
          }
        }
      }
      
      // Benchmark standard implementation
      console.log(`  Running standard implementation...`);
      const standardStartTime = performance.now();
      
      // Run standard implementation
      let standardResult;
      if (scenario.type !== 'fusion') {
        // Single operation benchmark
        if (scenario.operation === 'matmul') {
          standardResult = await standardBackend.matmul(standardTensor1, standardTensor2);
        } else if (scenario.operation === 'relu') {
          standardResult = await standardBackend.relu(standardTensor1);
        } else if (scenario.operation === 'sigmoid') {
          standardResult = await standardBackend.sigmoid(standardTensor1);
        } else if (scenario.operation === 'add') {
          standardResult = await standardBackend.add(standardTensor1, standardTensor2);
        }
      } else {
        // Fusion scenarios - run operations sequentially
        if (scenario.operations[0] === 'matmul') {
          standardResult = await standardBackend.matmul(standardTensor1, standardTensor2);
          
          if (scenario.operations[1] === 'relu') {
            standardResult = await standardBackend.relu(standardResult);
          } else if (scenario.operations[1] === 'sigmoid') {
            standardResult = await standardBackend.sigmoid(standardResult);
          }
        } else if (scenario.operations[0] === 'add') {
          standardResult = await standardBackend.add(standardTensor1, standardTensor2);
          
          if (scenario.operations[1] === 'relu') {
            standardResult = await standardBackend.relu(standardResult);
          } else if (scenario.operations[1] === 'multiply') {
            standardResult = await standardBackend.multiply(standardResult, standardTensor3);
          }
        }
      }
      
      const standardTime = performance.now() - standardStartTime;
      
      // Benchmark optimized implementation
      console.log(`  Running optimized implementation...`);
      const optimizedStartTime = performance.now();
      
      // Run optimized implementation
      let optimizedResult;
      if (scenario.type !== 'fusion') {
        // Single operation benchmark
        if (scenario.operation === 'matmul') {
          optimizedResult = await optimizedBackend.matmul(optimizedTensor1, optimizedTensor2);
        } else if (scenario.operation === 'relu') {
          optimizedResult = await optimizedBackend.relu(optimizedTensor1);
        } else if (scenario.operation === 'sigmoid') {
          optimizedResult = await optimizedBackend.sigmoid(optimizedTensor1);
        } else if (scenario.operation === 'add') {
          optimizedResult = await optimizedBackend.add(optimizedTensor1, optimizedTensor2);
        }
      } else {
        // Fusion scenarios - optimized backend should use fusion
        
        // For matmul + activation fusion
        if (scenario.operations[0] === 'matmul') {
          // First operation creates temporary result - but fusion should happen inside the backend
          const temp = await optimizedBackend.matmul(optimizedTensor1, optimizedTensor2);
          
          if (scenario.operations[1] === 'relu') {
            optimizedResult = await optimizedBackend.relu(temp);
          } else if (scenario.operations[1] === 'sigmoid') {
            optimizedResult = await optimizedBackend.sigmoid(temp);
          }
        } else if (scenario.operations[0] === 'add') {
          // First operation creates temporary result - but fusion should happen inside the backend
          const temp = await optimizedBackend.add(optimizedTensor1, optimizedTensor2);
          
          if (scenario.operations[1] === 'relu') {
            optimizedResult = await optimizedBackend.relu(temp);
          } else if (scenario.operations[1] === 'multiply') {
            optimizedResult = await optimizedBackend.multiply(temp, optimizedTensor3);
          }
        }
      }
      
      const optimizedTime = performance.now() - optimizedStartTime;
      
      // Verify correctness of results (simplified)
      if (standardResult && optimizedResult) {
        const stdShape = standardResult.shape;
        const optShape = optimizedResult.shape;
        
        let shapesMatch = true;
        if (stdShape.length === optShape.length) {
          for (let i = 0; i < stdShape.length; i++) {
            if (stdShape[i] !== optShape[i]) {
              shapesMatch = false;
              break;
            }
          }
        } else {
          shapesMatch = false;
        }
        
        if (!shapesMatch) {
          console.warn(`  Result shapes don't match: ${stdShape} vs ${optShape}`);
        }
      }
      
      // Calculate improvement
      const improvement = (standardTime / optimizedTime) * 100 - 100;
      
      // Add to results
      results.push({
        name: scenario.name,
        operation: scenario.type === 'fusion' ? scenario.operations.join('->') : scenario.operation,
        size: scenario.size,
        standardTime,
        optimizedTime,
        improvement: improvement.toFixed(2)
      });
      
      console.log(`  Standard: ${standardTime.toFixed(2)}ms, Optimized: ${optimizedTime.toFixed(2)}ms, Improvement: ${improvement.toFixed(2)}%`);
    }
    
    // Clean up
    optimizedBackend.dispose();
    standardBackend.dispose();
    
    return {
      supported: true,
      results
    };
  } catch (error) {
    console.error('WebGPU optimization benchmark failed:', error);
    return {
      supported: false,
      error: error instanceof Error ? error.message : String(error),
      results: null
    };
  }
}

/**
 * Run a test of operation fusion specifically
 */
export async function testOperationFusion() {
  // Check if WebGPU is available
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    console.error('WebGPU is not available in this environment');
    return {
      supported: false,
      error: 'WebGPU not available',
      results: null
    };
  }
  
  try {
    console.log('Testing operation fusion...');
    
    // Create WebGPU backend with optimizations
    const backend = new WebGPUBackend({
      enableOperationFusion: true,
      enableSpecializedShaders: true,
      enableBrowserOptimizations: true
    });
    
    // Initialize backend
    await backend.initialize();
    
    console.log('WebGPU backend initialized successfully');
    
    // Define fusion test cases
    const testCases = [
      {
        name: 'MatMul + ReLU',
        operations: ['matmul', 'relu'] as FusionOpType[],
        createInputs: () => {
          const size = 512;
          return [
            random([size, size], -1, 1, { backend: 'webgpu' }),
            random([size, size], -1, 1, { backend: 'webgpu' })
          ];
        }
      },
      {
        name: 'Add + ReLU',
        operations: ['add', 'relu'] as FusionOpType[],
        createInputs: () => {
          const size = 10000;
          return [
            random([size], -1, 1, { backend: 'webgpu' }),
            random([size], -1, 1, { backend: 'webgpu' })
          ];
        }
      },
      {
        name: 'MatMul + Sigmoid',
        operations: ['matmul', 'sigmoid'] as FusionOpType[],
        createInputs: () => {
          const size = 512;
          return [
            random([size, size], -1, 1, { backend: 'webgpu' }),
            random([size, size], -1, 1, { backend: 'webgpu' })
          ];
        }
      },
      {
        name: 'Add + Multiply + ReLU',
        operations: ['add', 'multiply', 'relu'] as FusionOpType[],
        createInputs: () => {
          const size = 10000;
          return [
            random([size], -1, 1, { backend: 'webgpu' }),
            random([size], -1, 1, { backend: 'webgpu' }),
            random([size], -1, 1, { backend: 'webgpu' })
          ];
        }
      }
    ];
    
    // Run tests
    const results = [];
    
    for (const testCase of testCases) {
      console.log(`Testing fusion: ${testCase.name}`);
      
      // Create inputs
      const inputs = testCase.createInputs();
      
      try {
        // Use the optimizer directly to attempt fusion
        const optimizer = (backend as any).optimizer;
        
        // Check if fusion is possible
        const canFuse = optimizer.fusionEngine.canFuse(testCase.operations);
        console.log(`  Can fuse: ${canFuse}`);
        
        if (canFuse) {
          // Time the fused operation
          const startTime = performance.now();
          const fusedResult = await optimizer.tryExecuteFusion(testCase.operations, inputs);
          const fusionTime = performance.now() - startTime;
          
          if (fusedResult) {
            console.log(`  Fusion successful. Time: ${fusionTime.toFixed(2)}ms`);
            
            // Now run the operations sequentially for comparison
            const sequentialStartTime = performance.now();
            
            let sequentialResult = inputs[0];
            
            if (testCase.operations[0] === 'matmul') {
              sequentialResult = await backend.matmul(inputs[0], inputs[1]);
            } else if (testCase.operations[0] === 'add') {
              sequentialResult = await backend.add(inputs[0], inputs[1]);
            }
            
            for (let i = 1; i < testCase.operations.length; i++) {
              const op = testCase.operations[i];
              
              if (op === 'relu') {
                sequentialResult = await backend.relu(sequentialResult);
              } else if (op === 'sigmoid') {
                sequentialResult = await backend.sigmoid(sequentialResult);
              } else if (op === 'multiply') {
                sequentialResult = await backend.multiply(sequentialResult, inputs[i+1]);
              }
            }
            
            const sequentialTime = performance.now() - sequentialStartTime;
            
            console.log(`  Sequential operations. Time: ${sequentialTime.toFixed(2)}ms`);
            
            // Calculate improvement
            const improvement = (sequentialTime / fusionTime) * 100 - 100;
            
            results.push({
              name: testCase.name,
              operations: testCase.operations.join('->'),
              fusionTime,
              sequentialTime,
              improvement: improvement.toFixed(2)
            });
            
            console.log(`  Improvement from fusion: ${improvement.toFixed(2)}%`);
          } else {
            console.log(`  Fusion execution failed`);
            results.push({
              name: testCase.name,
              operations: testCase.operations.join('->'),
              fusionTime: null,
              sequentialTime: null,
              improvement: 'Failed'
            });
          }
        } else {
          console.log(`  Cannot fuse operations: ${testCase.operations.join(', ')}`);
          results.push({
            name: testCase.name,
            operations: testCase.operations.join('->'),
            fusionTime: null,
            sequentialTime: null,
            improvement: 'Not supported'
          });
        }
      } catch (error) {
        console.error(`  Error testing fusion ${testCase.name}:`, error);
        results.push({
          name: testCase.name,
          operations: testCase.operations.join('->'),
          fusionTime: null,
          sequentialTime: null,
          error: error instanceof Error ? error.message : String(error),
          improvement: 'Error'
        });
      }
    }
    
    // Clean up
    backend.dispose();
    
    return {
      supported: true,
      results
    };
  } catch (error) {
    console.error('Operation fusion test failed:', error);
    return {
      supported: false,
      error: error instanceof Error ? error.message : String(error),
      results: null
    };
  }
}

/**
 * Run comprehensive WebGPU optimization benchmark
 * Tests memory layout optimizations, fusion patterns, and specialized shaders
 */
export async function runComprehensiveWebGPUBenchmark() {
  // Check if WebGPU is available
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    console.error('WebGPU is not available in this environment');
    return {
      supported: false,
      error: 'WebGPU not available',
      results: null
    };
  }
  
  try {
    console.log('Running comprehensive WebGPU optimization benchmark...');
    
    // Create WebGPU backend with all optimizations enabled
    const backend = new WebGPUBackend({
      enableOperationFusion: true,
      enableSpecializedShaders: true,
      enableBrowserOptimizations: true,
      enableMemoryOptimizations: true
    });
    
    // Initialize the backend
    await backend.initialize();
    
    console.log('WebGPU backend initialized successfully');
    
    // Get the optimizer from the backend (internal access)
    const optimizer = (backend as any).optimizer;
    
    // Section 1: Basic Operation Benchmarks
    console.log('Section 1: Basic Operations');
    const basicResults = await benchmarkBasicOperations(backend, optimizer);
    
    // Section 2: Fusion Patterns Benchmarks
    console.log('Section 2: Fusion Patterns');
    const fusionResults = await benchmarkFusionPatterns(backend, optimizer);
    
    // Section 3: Memory Layout Optimizations
    console.log('Section 3: Memory Layout Optimizations');
    const layoutResults = await benchmarkMemoryLayouts(backend, optimizer);
    
    // Section 4: Browser-Specific Optimizations
    console.log('Section 4: Browser-Specific Optimizations');
    const browserResults = await benchmarkBrowserOptimizations(backend, optimizer);
    
    // Clean up
    backend.dispose();
    
    // Combine all results
    return {
      supported: true,
      browser: optimizer.browserType,
      results: {
        basicOperations: basicResults,
        fusionPatterns: fusionResults,
        memoryLayouts: layoutResults,
        browserOptimizations: browserResults
      }
    };
  } catch (error) {
    console.error('Comprehensive WebGPU optimization benchmark failed:', error);
    return {
      supported: false,
      error: error instanceof Error ? error.message : String(error),
      results: null
    };
  }
}

/**
 * Benchmark basic operations (matmul, element-wise, etc.)
 */
async function benchmarkBasicOperations(backend: any, optimizer: any) {
  const results = [];
  
  // Matrix multiplication with different sizes
  for (const size of [128, 512, 1024, 2048]) {
    console.log(`  Testing MatMul (${size}x${size})...`);
    
    // Create tensors
    const a = random([size, size], -1, 1, { backend: 'webgpu' });
    const b = random([size, size], -1, 1, { backend: 'webgpu' });
    
    // Warm-up
    await backend.matmul(a, b);
    
    // Benchmark
    const startTime = performance.now();
    const output = await backend.matmul(a, b);
    const endTime = performance.now();
    
    results.push({
      operation: 'matmul',
      size: `${size}x${size}`,
      time: endTime - startTime,
      outputShape: output.shape
    });
  }
  
  // Element-wise operations
  for (const op of ['add', 'multiply', 'relu', 'sigmoid']) {
    for (const size of [10000, 100000, 1000000]) {
      console.log(`  Testing ${op} (${size} elements)...`);
      
      // Create tensors
      const a = random([size], -1, 1, { backend: 'webgpu' });
      const b = op === 'add' || op === 'multiply' 
        ? random([size], -1, 1, { backend: 'webgpu' })
        : null;
      
      // Warm-up
      if (op === 'add') await backend.add(a, b);
      else if (op === 'multiply') await backend.multiply(a, b);
      else if (op === 'relu') await backend.relu(a);
      else if (op === 'sigmoid') await backend.sigmoid(a);
      
      // Benchmark
      const startTime = performance.now();
      let output;
      
      if (op === 'add') output = await backend.add(a, b);
      else if (op === 'multiply') output = await backend.multiply(a, b);
      else if (op === 'relu') output = await backend.relu(a);
      else if (op === 'sigmoid') output = await backend.sigmoid(a);
      
      const endTime = performance.now();
      
      results.push({
        operation: op,
        size: size,
        time: endTime - startTime,
        outputShape: output.shape
      });
    }
  }
  
  return results;
}

/**
 * Benchmark fusion patterns
 */
async function benchmarkFusionPatterns(backend: any, optimizer: any) {
  const results = [];
  
  // Test different fusion patterns
  const fusionPatterns = [
    {
      name: 'MatMul + ReLU',
      operations: ['matmul', 'relu'],
      sizes: [512, 1024]
    },
    {
      name: 'Add + ReLU',
      operations: ['add', 'relu'],
      sizes: [10000, 100000]
    },
    {
      name: 'MatMul + Sigmoid',
      operations: ['matmul', 'sigmoid'],
      sizes: [512, 1024]
    },
    {
      name: 'MatMul + MatMul (Chain)',
      operations: ['matmul', 'matmul'],
      sizes: [256, 512]
    },
    {
      name: 'Add + Multiply + ReLU',
      operations: ['add', 'multiply', 'relu'],
      sizes: [10000, 100000]
    }
  ];
  
  for (const pattern of fusionPatterns) {
    for (const size of pattern.sizes) {
      console.log(`  Testing fusion pattern: ${pattern.name} (size ${size})...`);
      
      try {
        // Create input tensors
        let inputs = [];
        
        if (pattern.operations.includes('matmul')) {
          // For matrix operations
          inputs.push(random([size, size], -1, 1, { backend: 'webgpu' }));
          inputs.push(random([size, size], -1, 1, { backend: 'webgpu' }));
          
          // For matrix chain, add a third matrix
          if (pattern.operations.length > 1 && pattern.operations[1] === 'matmul') {
            inputs.push(random([size, size], -1, 1, { backend: 'webgpu' }));
          }
        } else {
          // For element-wise operations
          inputs.push(random([size], -1, 1, { backend: 'webgpu' }));
          inputs.push(random([size], -1, 1, { backend: 'webgpu' }));
          
          // For longer chains, add more inputs
          if (pattern.operations.length > 2) {
            inputs.push(random([size], -1, 1, { backend: 'webgpu' }));
          }
        }
        
        // First run standard operations sequentially
        const standardStartTime = performance.now();
        
        let intermediate = null;
        if (pattern.operations[0] === 'matmul') {
          intermediate = await backend.matmul(inputs[0], inputs[1]);
        } else if (pattern.operations[0] === 'add') {
          intermediate = await backend.add(inputs[0], inputs[1]);
        }
        
        for (let i = 1; i < pattern.operations.length; i++) {
          const op = pattern.operations[i];
          
          if (op === 'matmul') {
            intermediate = await backend.matmul(intermediate, inputs[i+1]);
          } else if (op === 'relu') {
            intermediate = await backend.relu(intermediate);
          } else if (op === 'sigmoid') {
            intermediate = await backend.sigmoid(intermediate);
          } else if (op === 'multiply') {
            intermediate = await backend.multiply(intermediate, inputs[i+1]);
          }
        }
        
        const standardTime = performance.now() - standardStartTime;
        
        // Now try with fusion
        const fusionStartTime = performance.now();
        
        // Try to execute fused operations
        const fusionResult = await optimizer.tryExecuteFusion(
          pattern.operations,
          inputs
        );
        
        const fusionTime = performance.now() - fusionStartTime;
        
        // Calculate improvement
        const improvement = fusionResult 
          ? ((standardTime / fusionTime) * 100 - 100).toFixed(2) + '%'
          : 'N/A (Fusion failed)';
        
        results.push({
          pattern: pattern.name,
          size: size,
          operations: pattern.operations.join('->'),
          standardTime,
          fusionTime: fusionResult ? fusionTime : null,
          improvement
        });
        
        console.log(`    Standard: ${standardTime.toFixed(2)}ms, Fusion: ${fusionResult ? fusionTime.toFixed(2) + 'ms' : 'Failed'}, Improvement: ${improvement}`);
      } catch (error) {
        console.error(`  Error testing fusion pattern ${pattern.name}:`, error);
        results.push({
          pattern: pattern.name,
          size: size,
          operations: pattern.operations.join('->'),
          standardTime: null,
          fusionTime: null,
          improvement: 'Error',
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }
  }
  
  return results;
}

/**
 * Benchmark memory layout optimizations
 */
async function benchmarkMemoryLayouts(backend: any, optimizer: any) {
  const results = [];
  
  // Only test if the optimizer supports memory layout optimizations
  if (!optimizer.getOptimalMemoryLayout) {
    console.log('  Memory layout optimizations not available');
    return [{ status: 'Not implemented' }];
  }
  
  // Test different memory layouts for matrix multiplication
  const sizes = [1024, 2048];
  
  for (const size of sizes) {
    console.log(`  Testing memory layouts for matmul (${size}x${size})...`);
    
    // Create standard row-major layout tensors
    const rowMajorA = random([size, size], -1, 1, { backend: 'webgpu' });
    const rowMajorB = random([size, size], -1, 1, { backend: 'webgpu' });
    
    // Get optimal layout for this browser and operation
    const optimalLayout = optimizer.getOptimalMemoryLayout('matmul', [size, size]);
    console.log(`    Optimal layout for this browser: ${optimalLayout.rowMajor ? 'Row-major' : 'Column-major'}, alignment: ${optimalLayout.alignment}`);
    
    // Run with standard row-major layout
    const rowMajorStartTime = performance.now();
    await backend.matmul(rowMajorA, rowMajorB);
    const rowMajorTime = performance.now() - rowMajorStartTime;
    
    // Create tensors with the optimal layout (would normally be handled by the backend)
    // This is simplified - in a real implementation, we would create tensors with the proper memory layout
    
    results.push({
      operation: 'matmul',
      size: `${size}x${size}`,
      rowMajorTime,
      optimalLayout: `${optimalLayout.rowMajor ? 'Row-major' : 'Column-major'}, alignment: ${optimalLayout.alignment}`
    });
    
    console.log(`    Row-major time: ${rowMajorTime.toFixed(2)}ms`);
    console.log(`    Note: Real layout optimization would require deep tensor memory layout modifications`);
  }
  
  return results;
}

/**
 * Benchmark browser-specific optimizations
 */
async function benchmarkBrowserOptimizations(backend: any, optimizer: any) {
  const results = [];
  
  const browser = optimizer.browserType || 'unknown';
  console.log(`  Browser detected: ${browser}`);
  
  // Test browser-optimized shaders for standard operations
  const operations = ['matmul', 'add', 'relu', 'sigmoid'];
  
  for (const operation of operations) {
    try {
      console.log(`  Testing browser-optimized shader for ${operation}...`);
      
      // For matrix operations
      if (operation === 'matmul') {
        const size = 1024;
        const a = random([size, size], -1, 1, { backend: 'webgpu' });
        const b = random([size, size], -1, 1, { backend: 'webgpu' });
        
        // Run the operation
        const startTime = performance.now();
        await backend.matmul(a, b);
        const endTime = performance.now();
        
        results.push({
          browser,
          operation,
          size: `${size}x${size}`,
          time: endTime - startTime
        });
      } else {
        // For element-wise operations
        const size = 100000;
        const a = random([size], -1, 1, { backend: 'webgpu' });
        const b = operation === 'add' ? random([size], -1, 1, { backend: 'webgpu' }) : null;
        
        // Run the operation
        const startTime = performance.now();
        
        if (operation === 'add') await backend.add(a, b);
        else if (operation === 'relu') await backend.relu(a);
        else if (operation === 'sigmoid') await backend.sigmoid(a);
        
        const endTime = performance.now();
        
        results.push({
          browser,
          operation,
          size,
          time: endTime - startTime
        });
      }
    } catch (error) {
      console.error(`  Error benchmarking browser-optimized shader for ${operation}:`, error);
      results.push({
        browser,
        operation,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }
  
  return results;
}

// Run the example if in browser environment
if (typeof window !== 'undefined') {
  // Create output element
  const outputElement = document.createElement('div');
  outputElement.id = 'webgpu-optimization-output';
  document.body.appendChild(outputElement);
  
  // Set initial content
  outputElement.innerHTML = `
    <h2>WebGPU Optimization Benchmark</h2>
    <p>Select a benchmark type to run:</p>
    
    <div style="margin-top: 20px;">
      <button id="run-basic-benchmark" style="padding: 10px; margin-right: 10px;">Run Standard Benchmark</button>
      <button id="run-comprehensive-benchmark" style="padding: 10px; margin-right: 10px;">Run Comprehensive Benchmark</button>
      <button id="run-fusion-test" style="padding: 10px;">Test Operation Fusion</button>
    </div>
    
    <div id="results-container" style="margin-top: 20px;">
      <p>Select a benchmark to run from the options above.</p>
    </div>
  `;
  
  // Add event listeners to buttons
  document.getElementById('run-basic-benchmark')?.addEventListener('click', () => {
    // Show loading message
    const resultsContainer = document.getElementById('results-container');
    if (resultsContainer) {
      resultsContainer.innerHTML = '<p>Running standard benchmark...</p>';
    }
    
    // Run the standard benchmark
    runWebGPUOptimizationBenchmark().then(result => {
      displayBasicResults(result, resultsContainer);
    });
  });
  
  document.getElementById('run-comprehensive-benchmark')?.addEventListener('click', () => {
    // Show loading message
    const resultsContainer = document.getElementById('results-container');
    if (resultsContainer) {
      resultsContainer.innerHTML = '<p>Running comprehensive benchmark (this may take a while)...</p>';
    }
    
    // Run the comprehensive benchmark
    runComprehensiveWebGPUBenchmark().then(result => {
      displayComprehensiveResults(result, resultsContainer);
    });
  });
  
  document.getElementById('run-fusion-test')?.addEventListener('click', () => {
    // Show loading message
    const resultsContainer = document.getElementById('results-container');
    if (resultsContainer) {
      resultsContainer.innerHTML = '<p>Testing operation fusion...</p>';
    }
    
    // Run the fusion test
    testOperationFusion().then(result => {
      displayFusionResults(result, resultsContainer);
    });
  });
  
  // Helper function to display basic benchmark results
  function displayBasicResults(result: any, container: HTMLElement | null) {
    if (!container) return;
    
    if (result.supported && result.results) {
    if (result.supported && result.results) {
      // Create HTML table for results
      const tableRows = result.results.map(r => `
        <tr>
          <td>${r.name}</td>
          <td>${r.standardTime.toFixed(2)}ms</td>
          <td>${r.optimizedTime.toFixed(2)}ms</td>
          <td>${r.improvement}%</td>
        </tr>
      `).join('');
      
      // Create bar chart data
      const chartData = result.results.map((r, i) => {
        return {
          name: r.name.split(' ')[0] + ' ' + r.name.split(' ')[1],
          standard: r.standardTime,
          optimized: r.optimizedTime
        };
      });
      
      // Update output
      outputElement.innerHTML = `
        <h2>WebGPU Optimization Benchmark</h2>
        <p>Comparing standard vs. optimized WebGPU implementation</p>
        
        <h3>Results</h3>
        <table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%;">
          <tr>
            <th>Operation</th>
            <th>Standard Time</th>
            <th>Optimized Time</th>
            <th>Improvement</th>
          </tr>
          ${tableRows}
        </table>
        
        <h3>Performance Comparison</h3>
        <div id="chart-container" style="height: 400px; margin-top: 20px;"></div>
        
        <h3>Optimization Techniques</h3>
        <ul>
          <li><strong>Operation Fusion</strong>: Combines multiple operations into a single GPU pass</li>
          <li><strong>Specialized Shaders</strong>: Uses optimized shaders based on tensor size and shape</li>
          <li><strong>Browser-Specific Optimizations</strong>: Adapts to different browser WebGPU implementations</li>
          <li><strong>Memory Optimizations</strong>: Efficiently reuses GPU memory buffers</li>
        </ul>
        
        <script>
          // Simple chart rendering using DOM elements
          function renderChart() {
            const container = document.getElementById('chart-container');
            container.style.display = 'flex';
            container.style.flexDirection = 'column';
            container.style.justifyContent = 'space-between';
            
            const data = ${JSON.stringify(chartData)};
            
            // Find max value for scaling
            const maxValue = Math.max(...data.flatMap(d => [d.standard, d.optimized])) * 1.1;
            
            // Create bars for each operation
            data.forEach(item => {
              const barGroup = document.createElement('div');
              barGroup.style.display = 'flex';
              barGroup.style.flexDirection = 'column';
              barGroup.style.marginBottom = '20px';
              
              const label = document.createElement('div');
              label.textContent = item.name;
              label.style.fontWeight = 'bold';
              label.style.marginBottom = '5px';
              
              const barContainer = document.createElement('div');
              barContainer.style.display = 'flex';
              barContainer.style.flexDirection = 'column';
              barContainer.style.gap = '5px';
              
              // Standard bar
              const standardBarContainer = document.createElement('div');
              standardBarContainer.style.display = 'flex';
              standardBarContainer.style.alignItems = 'center';
              
              const standardLabel = document.createElement('div');
              standardLabel.textContent = 'Standard:';
              standardLabel.style.width = '80px';
              
              const standardBarWrapper = document.createElement('div');
              standardBarWrapper.style.flex = '1';
              standardBarWrapper.style.backgroundColor = '#eee';
              standardBarWrapper.style.position = 'relative';
              standardBarWrapper.style.height = '20px';
              
              const standardBar = document.createElement('div');
              standardBar.style.width = \`\${(item.standard / maxValue) * 100}%\`;
              standardBar.style.backgroundColor = '#f44336';
              standardBar.style.height = '100%';
              standardBar.style.position = 'absolute';
              standardBar.style.left = '0';
              
              const standardValue = document.createElement('div');
              standardValue.textContent = \`\${item.standard.toFixed(2)}ms\`;
              standardValue.style.position = 'absolute';
              standardValue.style.right = '5px';
              standardValue.style.color = 'white';
              standardValue.style.fontSize = '12px';
              standardValue.style.lineHeight = '20px';
              
              standardBarWrapper.appendChild(standardBar);
              standardBar.appendChild(standardValue);
              standardBarContainer.appendChild(standardLabel);
              standardBarContainer.appendChild(standardBarWrapper);
              
              // Optimized bar
              const optimizedBarContainer = document.createElement('div');
              optimizedBarContainer.style.display = 'flex';
              optimizedBarContainer.style.alignItems = 'center';
              
              const optimizedLabel = document.createElement('div');
              optimizedLabel.textContent = 'Optimized:';
              optimizedLabel.style.width = '80px';
              
              const optimizedBarWrapper = document.createElement('div');
              optimizedBarWrapper.style.flex = '1';
              optimizedBarWrapper.style.backgroundColor = '#eee';
              optimizedBarWrapper.style.position = 'relative';
              optimizedBarWrapper.style.height = '20px';
              
              const optimizedBar = document.createElement('div');
              optimizedBar.style.width = \`\${(item.optimized / maxValue) * 100}%\`;
              optimizedBar.style.backgroundColor = '#2196F3';
              optimizedBar.style.height = '100%';
              optimizedBar.style.position = 'absolute';
              optimizedBar.style.left = '0';
              
              const optimizedValue = document.createElement('div');
              optimizedValue.textContent = \`\${item.optimized.toFixed(2)}ms\`;
              optimizedValue.style.position = 'absolute';
              optimizedValue.style.right = '5px';
              optimizedValue.style.color = 'white';
              optimizedValue.style.fontSize = '12px';
              optimizedValue.style.lineHeight = '20px';
              
              optimizedBarWrapper.appendChild(optimizedBar);
              optimizedBar.appendChild(optimizedValue);
              optimizedBarContainer.appendChild(optimizedLabel);
              optimizedBarContainer.appendChild(optimizedBarWrapper);
              
              barContainer.appendChild(standardBarContainer);
              barContainer.appendChild(optimizedBarContainer);
              
              barGroup.appendChild(label);
              barGroup.appendChild(barContainer);
              
              container.appendChild(barGroup);
            });
          }
          
          // Render chart when DOM is ready
          document.addEventListener('DOMContentLoaded', renderChart);
          
          // If DOM is already loaded, render immediately
          if (document.readyState === 'interactive' || document.readyState === 'complete') {
            renderChart();
          }
        </script>
      `;
    } else {
      outputElement.innerHTML = `
        <h2>WebGPU Optimization Benchmark</h2>
        <p>WebGPU is not supported in this browser or an error occurred.</p>
        <p>Error: ${result.error}</p>
      `;
    }
  });
}