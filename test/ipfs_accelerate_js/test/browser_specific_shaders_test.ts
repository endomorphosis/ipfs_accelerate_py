/**
 * Browser-Specific Shader Optimizations Test
 * Tests the generation and performance of browser-specific shaders
 */

import { WebGPUBackend } from '../src/hardware/webgpu/backend';
import { Tensor } from '../src/tensor/tensor';
import { 
  BrowserType, 
  detectBrowserType 
} from '../src/hardware/webgpu/browser_optimized_operations';
import {
  getOptimalWorkgroupSize,
  getOptimalTileSize,
  generateBrowserOptimizedMatmulShader,
  generateBrowserOptimizedElementwiseShader,
  generateBrowserOptimizedQuantizedMatmulShader,
  generateBrowserOptimizedAttentionShader
} from '../src/hardware/webgpu/optimizations/browser_specific_shaders';
import {
  loadBrowserShader,
  getBrowserShaderSync,
  ShaderType
} from '../src/hardware/webgpu/optimizations/browser_shader_loader';

/**
 * Detects browser and shows optimal configurations
 */
function printBrowserOptimizations() {
  const browserType = detectBrowserType();
  const browserNames = {
    [BrowserType.CHROME]: 'Google Chrome',
    [BrowserType.FIREFOX]: 'Mozilla Firefox',
    [BrowserType.SAFARI]: 'Apple Safari',
    [BrowserType.EDGE]: 'Microsoft Edge',
    [BrowserType.UNKNOWN]: 'Unknown Browser'
  };
  
  console.log(`Detected browser: ${browserNames[browserType]}`);
  
  console.log('\nOptimal workgroup sizes:');
  console.log(`MatMul: ${getOptimalWorkgroupSize(browserType, 'matmul')}`);
  console.log(`Conv2D: ${getOptimalWorkgroupSize(browserType, 'conv2d')}`);
  console.log(`Elementwise: ${getOptimalWorkgroupSize(browserType, 'elementwise')}`);
  console.log(`Attention: ${getOptimalWorkgroupSize(browserType, 'attention')}`);
  
  console.log('\nOptimal tile sizes:');
  console.log(`Small matrices: ${getOptimalTileSize(browserType, 64)}`);
  console.log(`Medium matrices: ${getOptimalTileSize(browserType, 256)}`);
  console.log(`Large matrices: ${getOptimalTileSize(browserType, 1024)}`);
}

/**
 * Generates and prints shader code for the current browser
 */
function generateBrowserOptimizedShaders() {
  const browserType = detectBrowserType();
  const browserNames = {
    [BrowserType.CHROME]: 'Google Chrome',
    [BrowserType.FIREFOX]: 'Mozilla Firefox',
    [BrowserType.SAFARI]: 'Apple Safari',
    [BrowserType.EDGE]: 'Microsoft Edge',
    [BrowserType.UNKNOWN]: 'Unknown Browser'
  };
  
  console.log(`Generating optimized shaders for ${browserNames[browserType]}...`);
  
  // Generate and show matmul shader
  console.log('\n===== Matrix Multiplication Shader =====');
  const matmulShader = generateBrowserOptimizedMatmulShader({
    browserType: browserType,
    useSharedMemory: true
  });
  console.log(matmulShader.slice(0, 500) + '...\n[shader truncated]');
  
  // Generate and show elementwise shader
  console.log('\n===== Elementwise ReLU Shader =====');
  const reluShader = generateBrowserOptimizedElementwiseShader('relu', {
    browserType: browserType,
    useFastMath: true
  });
  console.log(reluShader.slice(0, 500) + '...\n[shader truncated]');
  
  // Generate and show quantized shader
  console.log('\n===== 4-bit Quantized Matmul Shader =====');
  const quantizedShader = generateBrowserOptimizedQuantizedMatmulShader(4, {
    browserType: browserType,
    optimizeMemoryAccess: true
  });
  console.log(quantizedShader.slice(0, 500) + '...\n[shader truncated]');
}

/**
 * Compare performance of browser-specific vs. generic shaders
 */
async function benchmarkShaderPerformance() {
  // Skip benchmark if WebGPU is not available
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    console.log('WebGPU is not available. Skipping performance benchmark.');
    return;
  }
  
  try {
    console.log('Initializing WebGPU backend...');
    const backend = new WebGPUBackend();
    await backend.initialize();
    
    console.log('Performing shader benchmark...');
    const browserType = detectBrowserType();
    
    // Create test tensors
    const matrixSize = 512;
    const matrixA = new Tensor<number>(
      [matrixSize, matrixSize], 
      Array(matrixSize * matrixSize).fill(0).map(() => Math.random() * 2 - 1),
      { dataType: 'float32' }
    );
    
    const matrixB = new Tensor<number>(
      [matrixSize, matrixSize], 
      Array(matrixSize * matrixSize).fill(0).map(() => Math.random() * 2 - 1),
      { dataType: 'float32' }
    );
    
    // Run standard matmul
    console.log('Running standard matmul...');
    const startTimeStandard = performance.now();
    const standardResult = await backend.matmul(matrixA, matrixB, { 
      useBrowserOptimizations: false 
    });
    const endTimeStandard = performance.now();
    const standardTime = endTimeStandard - startTimeStandard;
    
    // Run optimized matmul with browser optimizations
    console.log('Running browser-optimized matmul...');
    const startTimeOptimized = performance.now();
    const optimizedResult = await backend.matmul(matrixA, matrixB, { 
      useBrowserOptimizations: true,
      browserType: browserType
    });
    const endTimeOptimized = performance.now();
    const optimizedTime = endTimeOptimized - startTimeOptimized;
    
    // Calculate speedup
    const speedup = standardTime / optimizedTime;
    
    console.log('\nPerformance Results:');
    console.log(`Standard Shader: ${standardTime.toFixed(2)}ms`);
    console.log(`Optimized Shader: ${optimizedTime.toFixed(2)}ms`);
    console.log(`Speedup: ${speedup.toFixed(2)}x`);
    
    console.log('\nNow testing quantized operations...');
    
    // Run standard quantized matmul
    console.log('Running standard quantized matmul (4-bit)...');
    const startTimeQuant = performance.now();
    const quantResult = await backend.matmul(matrixA, matrixB, { 
      useQuantization: true,
      bitsPerWeight: 4,
      useBrowserOptimizations: false 
    });
    const endTimeQuant = performance.now();
    const quantTime = endTimeQuant - startTimeQuant;
    
    // Run optimized quantized matmul
    console.log('Running browser-optimized quantized matmul (4-bit)...');
    const startTimeQuantOpt = performance.now();
    const quantOptResult = await backend.matmul(matrixA, matrixB, { 
      useQuantization: true,
      bitsPerWeight: 4,
      useBrowserOptimizations: true,
      browserType: browserType
    });
    const endTimeQuantOpt = performance.now();
    const quantOptTime = endTimeQuantOpt - startTimeQuantOpt;
    
    // Calculate quantized speedup
    const quantSpeedup = quantTime / quantOptTime;
    
    console.log('\nQuantized Performance Results:');
    console.log(`Standard Quantized Shader: ${quantTime.toFixed(2)}ms`);
    console.log(`Optimized Quantized Shader: ${quantOptTime.toFixed(2)}ms`);
    console.log(`Speedup: ${quantSpeedup.toFixed(2)}x`);
    
    console.log('\nMemory Usage Comparison:');
    console.log(`Standard FP32: ${matrixA.size * 4 / (1024 * 1024)} MB`);
    console.log(`4-bit Quantized: ${Math.ceil(matrixA.size * 4 / 8) / (1024 * 1024)} MB`);
    console.log(`Memory Reduction: ${(1 - Math.ceil(matrixA.size * 4 / 8) / (matrixA.size * 4)) * 100}%`);
    
    // Cleanup
    await backend.dispose();
    
  } catch (error) {
    console.error('Error during benchmark:', error);
  }
}

/**
 * Test shader loading from the browser_shader_loader module
 */
async function testShaderLoaderModule() {
  console.log('\n=== Testing Shader Loader Module ===');
  
  try {
    const browserType = detectBrowserType();
    
    console.log(`Loading MATMUL shader for current browser (${browserType})...`);
    const matmulShader = await loadBrowserShader(ShaderType.MATMUL);
    console.log('✅ Successfully loaded MATMUL shader via loadBrowserShader()');
    
    console.log('Loading QUANTIZED_MATMUL shader for current browser...');
    const quantizedShader = await loadBrowserShader(ShaderType.QUANTIZED_MATMUL);
    console.log('✅ Successfully loaded QUANTIZED_MATMUL shader via loadBrowserShader()');
    
    // Test sync loading
    console.log('Testing synchronous shader loading...');
    const syncShader = getBrowserShaderSync(ShaderType.MATMUL);
    console.log('✅ Successfully loaded shader via getBrowserShaderSync()');
    
    // Test loading for different browsers
    console.log('\nTesting shader loading for different browsers:');
    const browsers = [BrowserType.CHROME, BrowserType.FIREFOX, BrowserType.SAFARI, BrowserType.EDGE];
    
    for (const browser of browsers) {
      const browserName = {
        [BrowserType.CHROME]: 'Chrome',
        [BrowserType.FIREFOX]: 'Firefox',
        [BrowserType.SAFARI]: 'Safari',
        [BrowserType.EDGE]: 'Edge',
        [BrowserType.UNKNOWN]: 'Unknown'
      }[browser];
      
      try {
        const shader = getBrowserShaderSync(ShaderType.MATMUL, browser);
        console.log(`✅ ${browserName}: Successfully loaded shader`);
      } catch (error) {
        console.error(`❌ ${browserName}: Failed to load shader`, error);
      }
    }
    
  } catch (error) {
    console.error('Error testing shader loader:', error);
  }
}

/**
 * Main function to run all tests
 */
async function main() {
  console.log('Browser-Specific Shader Optimizations Test');
  console.log('==========================================');
  
  // Print browser optimizations
  printBrowserOptimizations();
  
  // Generate and show optimized shaders
  generateBrowserOptimizedShaders();
  
  // Test shader loader module
  await testShaderLoaderModule();
  
  // Run performance benchmark
  await benchmarkShaderPerformance();
}

// Check if WebGPU is available and run tests
if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
  console.log('WebGPU is available. Running tests...');
  main();
} else {
  console.log('WebGPU is not supported in this environment. Running shader generation only...');
  printBrowserOptimizations();
  generateBrowserOptimizedShaders();
}