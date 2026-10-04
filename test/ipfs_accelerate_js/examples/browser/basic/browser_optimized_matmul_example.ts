/**
 * Example demonstrating browser-optimized matrix multiplication with WebGPU
 */

import { createWebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { Tensor } from '../../../src/tensor/tensor';
import { BrowserType } from '../../../src/hardware/webgpu/browser_optimized_operations';

/**
 * Main example function
 */
async function runExample() {
  console.log('Browser-Optimized Matrix Multiplication Example');
  console.log('----------------------------------------------');
  
  // Check if WebGPU is supported
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    console.error('WebGPU is not supported in this browser.');
    return;
  }
  
  try {
    // Initialize WebGPU backend
    console.log('Initializing WebGPU backend...');
    const backend = await createWebGPUBackend();
    
    // Get browser information
    const browserType = backend.getBrowserType();
    const capabilities = backend.getBrowserCapabilities();
    
    console.log(`Browser detected: ${browserType}`);
    console.log(`Performance tier: ${capabilities?.performanceTier}`);
    console.log(`Hardware vendor: ${capabilities?.hardware.vendor}`);
    console.log(`Hardware architecture: ${capabilities?.hardware.architecture}`);
    console.log('Optimal parameters:', {
      matmulWorkgroupSize: capabilities?.optimalWorkgroupSizes.matmul,
      matmulTileSize: capabilities?.optimalTileSizes.matmul,
      useSharedMemory: capabilities?.optimizationFlags.useSharedMemory,
      useMicroTiling: capabilities?.optimizationFlags.useMicroTiling,
      loopUnrollingLevel: capabilities?.optimizationFlags.loopUnrollingLevel,
    });
    
    // Create sample matrices
    console.log('\nCreating test matrices...');
    const SIZE = 512; // Matrix size for benchmark
    
    const matrixA = new Tensor<number>(
      [SIZE, SIZE],
      new Float32Array(SIZE * SIZE).fill(1.0),
      { dataType: 'float32' }
    );
    
    const matrixB = new Tensor<number>(
      [SIZE, SIZE],
      new Float32Array(SIZE * SIZE).fill(1.0),
      { dataType: 'float32' }
    );
    
    // Run with browser optimizations
    console.log('\nRunning matrix multiplication with browser optimizations...');
    backend.setBrowserOptimizationsEnabled(true);
    
    const startWithOpt = performance.now();
    const resultWithOpt = await backend.matmul(matrixA, matrixB);
    const endWithOpt = performance.now();
    
    // Run without browser optimizations
    console.log('Running matrix multiplication without browser optimizations...');
    backend.setBrowserOptimizationsEnabled(false);
    
    const startWithoutOpt = performance.now();
    const resultWithoutOpt = await backend.matmul(matrixA, matrixB);
    const endWithoutOpt = performance.now();
    
    // Re-enable optimizations
    backend.setBrowserOptimizationsEnabled(true);
    
    // Report performance
    const timeWithOpt = endWithOpt - startWithOpt;
    const timeWithoutOpt = endWithoutOpt - startWithoutOpt;
    const speedup = timeWithoutOpt / timeWithOpt;
    
    console.log('\nPerformance Results:');
    console.log(`Matrix size: ${SIZE}x${SIZE}`);
    console.log(`With browser optimizations: ${timeWithOpt.toFixed(2)}ms`);
    console.log(`Without browser optimizations: ${timeWithoutOpt.toFixed(2)}ms`);
    console.log(`Speedup: ${speedup.toFixed(2)}x`);
    
    // Validate results
    console.log('\nValidating results...');
    const resultWithOptData = resultWithOpt.getData() as Float32Array;
    const resultWithoutOptData = resultWithoutOpt.getData() as Float32Array;
    
    let maxDiff = 0;
    for (let i = 0; i < resultWithOptData.length; i++) {
      const diff = Math.abs(resultWithOptData[i] - resultWithoutOptData[i]);
      if (diff > maxDiff) {
        maxDiff = diff;
      }
    }
    
    console.log(`Maximum difference between results: ${maxDiff.toExponential(2)}`);
    console.log(`Results match: ${maxDiff < 1e-6 ? 'Yes' : 'No'}`);
    
    // Browser-specific recommendations
    console.log('\nBrowser-Specific Recommendations:');
    switch (browserType) {
      case BrowserType.CHROME:
        console.log('- Chrome excels at vision model processing');
        console.log('- Consider using larger batch sizes for optimal throughput');
        console.log('- Enable shader precompilation for better startup performance');
        break;
      case BrowserType.FIREFOX:
        console.log('- Firefox excels at audio model processing');
        console.log('- Consider using WebGPU for audio feature extraction');
        console.log('- Enable memory coalescing for better performance');
        break;
      case BrowserType.SAFARI:
        console.log('- Safari works best with smaller workgroup sizes');
        console.log('- Consider using WebNN on Apple Silicon when available');
        console.log('- Enable Metal-specific optimizations if available');
        break;
      case BrowserType.EDGE:
        console.log('- Edge has excellent WebNN support');
        console.log('- Consider using WebNN for inference when possible');
        console.log('- Similar WebGPU performance profile to Chrome');
        break;
      default:
        console.log('- Use WebGPU feature detection to ensure compatibility');
        console.log('- Consider running benchmarks to optimize for this browser');
    }
    
    console.log('\nExample completed successfully.');
  } catch (error) {
    console.error('Error running example:', error);
  }
}

// Run the example when loaded
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    const outputElement = document.getElementById('output');
    if (outputElement) {
      // Redirect console logs to the output element
      const originalConsoleLog = console.log;
      const originalConsoleError = console.error;
      
      console.log = (...args) => {
        originalConsoleLog(...args);
        outputElement.innerHTML += args.join(' ') + '<br>';
      };
      
      console.error = (...args) => {
        originalConsoleError(...args);
        outputElement.innerHTML += '<span style="color: red;">' + args.join(' ') + '</span><br>';
      };
    }
    
    // Run the example
    runExample();
  });
}