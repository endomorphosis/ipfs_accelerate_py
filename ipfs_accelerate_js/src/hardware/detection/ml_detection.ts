/**
 * Jest Test Setup File
 * 
 * This file sets up the testing environment before Jest runs tests.
 */

// Import any polyfills needed for testing

// Set up IndexedDB mock
require('fake-indexeddb/auto');

// Mock browser WebGPU implementation
class MockGPUAdapter {
  async requestDevice() {
    return new MockGPUDevice();
  }
  
  async requestAdapterInfo() {
    return {
      vendor: 'Test Vendor',
      architecture: 'Test Architecture',
      device: 'Test Device',
      description: 'Test WebGPU Device for Testing'
    };
  }
  
  get features() {
    return new Set(['texture-compression-bc']);
  }
  
  get limits() {
    return {
      maxBindGroups: 4,
      maxBindingsPerBindGroup: 16,
      maxBufferSize: 1 << 30,
      maxDynamicUniformBuffersPerPipelineLayout: 8,
      maxDynamicStorageBuffersPerPipelineLayout: 4,
      maxSampledTexturesPerShaderStage: 16,
      maxSamplersPerShaderStage: 16,
      maxStorageBuffersPerShaderStage: 8,
      maxStorageTexturesPerShaderStage: 4,
      maxUniformBuffersPerShaderStage: 12
    };
  }
}

class MockGPUDevice {
  constructor() {
    this.features = new Set(['texture-compression-bc']);
    this.limits = {
      maxBindGroups: 4,
      maxBindingsPerBindGroup: 16,
      maxBufferSize: 1 << 30,
      maxDynamicUniformBuffersPerPipelineLayout: 8,
      maxDynamicStorageBuffersPerPipelineLayout: 4,
      maxSampledTexturesPerShaderStage: 16,
      maxSamplersPerShaderStage: 16,
      maxStorageBuffersPerShaderStage: 8,
      maxStorageTexturesPerShaderStage: 4,
      maxUniformBuffersPerShaderStage: 12
    };
    this.queue = new MockGPUQueue();
  }
  
  createShaderModule({ code }) {
    return { code };
  }
  
  createBuffer({ size, usage, mappedAtCreation }) {
    return new MockGPUBuffer(size, usage, mappedAtCreation);
  }
  
  createBindGroupLayout() {
    return {};
  }
  
  createPipelineLayout() {
    return {};
  }
  
  createComputePipeline() {
    return {
      getBindGroupLayout: () => ({})
    };
  }
  
  createBindGroup() {
    return {};
  }
  
  createCommandEncoder() {
    return new MockGPUCommandEncoder();
  }
  
  destroy() {}
}

class MockGPUBuffer {
  constructor(size, usage, mappedAtCreation) {
    this.size = size;
    this.usage = usage;
    this.mapState = mappedAtCreation ? 'mapped' : 'unmapped';
    this.data = new ArrayBuffer(size);
  }
  
  getMappedRange() {
    return this.data;
  }
  
  unmap() {
    this.mapState = 'unmapped';
  }
  
  destroy() {}
}

class MockGPUCommandEncoder {
  beginComputePass() {
    return new MockGPUComputePass();
  }
  
  copyBufferToBuffer() {}
  
  finish() {
    return {};
  }
}

class MockGPUComputePass {
  setPipeline() {}
  setBindGroup() {}
  dispatchWorkgroups() {}
  end() {}
}

class MockGPUQueue {
  submit() {}
  writeBuffer() {}
  onSubmittedWorkDone() {
    return Promise.resolve();
  }
}

// Attach WebGPU mock to the global/window object
const mockGPU = {
  requestAdapter: async () => new MockGPUAdapter()
};

// Mock WebNN
class MockMLContext {
  createOperand(descriptor, bufferView) {
    return {
      descriptor,
      data: bufferView
    };
  }
}

class MockMLGraphBuilder {
  constructor(context) {
    this.context = context;
  }
  
  input(name, descriptor) {
    return { name, descriptor };
  }
  
  constant(descriptor, bufferView) {
    return { descriptor, data: bufferView };
  }
  
  relu(input) { return { op: 'relu', input }; }
  sigmoid(input) { return { op: 'sigmoid', input }; }
  tanh(input) { return { op: 'tanh', input }; }
  add(a, b) { return { op: 'add', inputs: [a, b] }; }
  matmul(a, b) { return { op: 'matmul', inputs: [a, b] }; }
  
  async build({ inputs, outputs }) {
    return {
      inputs,
      outputs,
      async compute(inputs, outputs) {
        return outputs;
      }
    };
  }
}

const mockML = {
  createContext: async () => new MockMLContext()
};

// Define window if we're in Node.js environment (for test environment)
if (typeof window === 'undefined') {
  (global as any).window = {};
}

// Attach mocks to global/window object
if (typeof window !== 'undefined') {
  (window as any).gpu = mockGPU;
  (window as any).navigator = window.navigator || {};
  (window as any).navigator.gpu = mockGPU;
  (window as any).navigator.ml = mockML;
  (window as any).MLGraphBuilder = MockMLGraphBuilder;
  
  // Mock AudioContext
  (window as any).AudioContext = class AudioContext {
    constructor() {
      this.sampleRate = 44100;
    }
    
    close() {}
  };
  
  // Mock WebGL
  (window as any).WebGLRenderingContext = class WebGLRenderingContext {
    getExtension(name: string) {
      if (name === 'WEBGL_debug_renderer_info') {
        return {
          UNMASKED_RENDERER_WEBGL: 'test-renderer',
          UNMASKED_VENDOR_WEBGL: 'test-vendor'
        };
      }
      return null;
    }
    
    getParameter(param: any) {
      return 'test-value';
    }
  };
  
  // Mock canvas element and context
  (window as any).HTMLCanvasElement.prototype.getContext = function(contextType: string) {
    if (contextType === 'webgl') {
      return new WebGLRenderingContext();
    }
    return null;
  };
} else {
  (global as any).navigator = {
    gpu: mockGPU,
    ml: mockML,
    userAgent: 'Jest Test Environment'
  };
  (global as any).MLGraphBuilder = MockMLGraphBuilder;
  
  // Node.js specific mocks for file system operations
  const mockFS = {
    existsSync: () => true,
    mkdirSync: () => {},
    readFileSync: () => '{}',
    writeFileSync: () => {},
    readdirSync: () => [],
    statSync: () => ({ size: 1000, mtime: new Date() }),
    unlinkSync: () => {}
  };
  
  const mockPath = {
    join: (...args: string[]) => args.join('/'),
    resolve: (...args: string[]) => args.join('/')
  };
  
  jest.mock('fs', () => mockFS);
  jest.mock('path', () => mockPath);
}

// Increase Jest timeout for complex tests
jest.setTimeout(10000);

// Console spy to silence expected warnings
beforeAll(() => {
  jest.spyOn(console, 'warn').mockImplementation(() => {});
  jest.spyOn(console, 'error').mockImplementation(() => {});
});

afterAll(() => {
  jest.restoreAllMocks();
});
