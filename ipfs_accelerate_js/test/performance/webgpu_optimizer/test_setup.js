/**
 * Setup file for WebGPU optimizer tests
 * 
 * This file sets up the mock environment for WebGPU tests when running in Node.js
 */

// Mock WebGPU device for testing purposes
global.GPUDevice = class GPUDevice {
  createShaderModule() { return {}; }
  createBuffer() { return {}; }
  createBindGroupLayout() { return {}; }
  createPipelineLayout() { return {}; }
  createBindGroup() { return {}; }
  createComputePipeline() { return {}; }
  createCommandEncoder() { 
    return {
      beginComputePass() {
        return {
          setPipeline() {},
          setBindGroup() {},
          dispatchWorkgroups() {},
          end() {}
        };
      },
      copyBufferToBuffer() {},
      finish() { return {}; }
    };
  }
  queue = {
    submit() {},
    writeBuffer() {}
  };
};

// Mock WebGPU API for testing purposes
global.navigator = {
  gpu: {
    requestAdapter() {
      return Promise.resolve({
        requestDevice() {
          return Promise.resolve(new GPUDevice());
        }
      });
    }
  }
};

// Mock browser detection
global.navigator.userAgent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36';

// Helper for benchmarking in Node environment
global.performance = {
  now: () => Date.now(),
  mark: () => {},
  measure: () => {},
  getEntriesByName: () => []
};

// Mock browser memory API
if (!global.performance.memory) {
  global.performance.memory = {
    jsHeapSizeLimit: 2147483648,
    totalJSHeapSize: 50000000,
    usedJSHeapSize: 25000000
  };
}

// Console override to reduce noise during tests
const originalConsoleLog = console.log;
console.log = (...args) => {
  // Filter out less important logs during tests
  if (process.env.VERBOSE === 'true') {
    originalConsoleLog(...args);
  } else if (args[0] && typeof args[0] === 'string') {
    // Only log errors and important messages
    if (
      args[0].includes('Error') || 
      args[0].includes('error') || 
      args[0].includes('Failed') ||
      args[0].includes('✓') ||
      args[0].includes('✗')
    ) {
      originalConsoleLog(...args);
    }
  }
};

// Setup environment variables for tests
process.env.BENCHMARK_ITERATIONS = process.env.BENCHMARK_ITERATIONS || '5';
process.env.BENCHMARK_WARMUP_ITERATIONS = process.env.BENCHMARK_WARMUP_ITERATIONS || '2';