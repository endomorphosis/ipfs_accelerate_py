// jest.setup.js - This file contains setup code that will be run before each test

// Mock WebNN API for Jest tests
// This is necessary because the WebNN API is not available in JSDOM
if (!global.navigator) {
  global.navigator = {};
}

if (!global.navigator.ml) {
  // Create a mock implementation of the WebNN API
  class MockMLOperand {}

  class MockMLGraph {
    constructor() {
      this.inputs = null;
      this.outputs = null;
    }

    async compute(inputs, outputs) {
      // Return the outputs directly for testing
      return outputs;
    }
  }

  class MockMLGraphBuilder {
    constructor(context) {
      this.context = context;
    }

    constant(descriptor, bufferView) {
      return new MockMLOperand();
    }

    input(name, descriptor) {
      return new MockMLOperand();
    }

    relu(input) {
      return new MockMLOperand();
    }

    sigmoid(input) {
      return new MockMLOperand();
    }

    tanh(input) {
      return new MockMLOperand();
    }

    exp(input) {
      return new MockMLOperand();
    }

    add(a, b) {
      return new MockMLOperand();
    }

    sub(a, b) {
      return new MockMLOperand();
    }

    mul(a, b) {
      return new MockMLOperand();
    }

    div(a, b) {
      return new MockMLOperand();
    }

    matmul(a, b) {
      return new MockMLOperand();
    }

    reshape(input, newShape) {
      return new MockMLOperand();
    }

    concat(inputs, axis) {
      return new MockMLOperand();
    }

    transpose(input, permutation) {
      return new MockMLOperand();
    }

    conv2d(input, filter, options) {
      return new MockMLOperand();
    }

    async build(options) {
      const graph = new MockMLGraph();
      graph.inputs = options.inputs;
      graph.outputs = options.outputs;
      return graph;
    }
  }

  class MockMLContext {
    constructor(options) {
      this.deviceType = options?.deviceType || 'cpu';
    }

    createOperand(descriptor, bufferView) {
      return new MockMLOperand();
    }

    async readOperand(operand, bufferView) {
      // Fill the buffer with some test data
      if (bufferView instanceof Float32Array) {
        for (let i = 0; i < bufferView.length; i++) {
          bufferView[i] = i + 1;
        }
      } else if (bufferView instanceof Int32Array) {
        for (let i = 0; i < bufferView.length; i++) {
          bufferView[i] = i + 1;
        }
      } else if (bufferView instanceof Uint8Array) {
        for (let i = 0; i < bufferView.length; i++) {
          bufferView[i] = i + 1;
        }
      }
      return bufferView;
    }
  }

  // Set up the global ml object
  global.navigator.ml = {
    async createContext(options) {
      return new MockMLContext(options);
    },
    MLGraphBuilder: MockMLGraphBuilder
  };

  // Add MLGraphBuilder to global scope for tests
  global.MLGraphBuilder = MockMLGraphBuilder;
}

// Mock WebGPU API for Jest tests
if (!global.navigator.gpu) {
  class MockGPUAdapter {
    constructor() {
      this.isFallbackAdapter = false;
    }

    async requestAdapterInfo() {
      return {
        device: 'Mock GPU Device',
        description: 'Mock GPU for testing',
        vendor: 'Jest Test'
      };
    }

    async requestDevice(options) {
      return new MockGPUDevice();
    }
  }

  class MockGPUBuffer {
    constructor(options) {
      this.label = options.label;
      this.size = options.size;
      this.usage = options.usage;
      this.mappedAtCreation = options.mappedAtCreation;
      this._isMapped = options.mappedAtCreation;
      this._data = new ArrayBuffer(options.size);
    }

    getMappedRange() {
      return this._data;
    }

    unmap() {
      this._isMapped = false;
    }

    async mapAsync() {
      this._isMapped = true;
      return Promise.resolve();
    }

    destroy() {
      this._data = null;
    }
  }

  class MockGPUCommandEncoder {
    constructor(options) {
      this.label = options?.label;
    }

    beginComputePass(options) {
      return new MockGPUComputePassEncoder();
    }

    copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
      // Mock implementation
    }

    finish() {
      return new MockGPUCommandBuffer();
    }
  }

  class MockGPUCommandBuffer {}

  class MockGPUComputePassEncoder {
    setPipeline(pipeline) {}
    setBindGroup(index, bindGroup) {}
    dispatchWorkgroups(x, y, z) {}
    end() {}
  }

  class MockGPUComputePipeline {
    getBindGroupLayout(index) {
      return new MockGPUBindGroupLayout();
    }
  }

  class MockGPUBindGroupLayout {}

  class MockGPUBindGroup {}

  class MockGPUShaderModule {}

  class MockGPUDevice {
    constructor() {
      this.limits = new Map([
        ['maxBindGroups', 8],
        ['maxBindingsPerBindGroup', 16],
        ['maxBufferSize', 1024 * 1024 * 1024],
        ['maxComputeWorkgroupSizeX', 256],
        ['maxComputeWorkgroupSizeY', 256],
        ['maxComputeWorkgroupSizeZ', 64],
        ['maxComputeWorkgroupsPerDimension', 65535],
        ['maxStorageBufferBindingSize', 128 * 1024 * 1024]
      ]);
      this.features = new Set(['shader-f16']);
    }

    createBuffer(options) {
      return new MockGPUBuffer(options);
    }

    createBindGroup(options) {
      return new MockGPUBindGroup();
    }

    createShaderModule(options) {
      return new MockGPUShaderModule();
    }

    createComputePipeline(options) {
      return new MockGPUComputePipeline();
    }

    createCommandEncoder(options) {
      return new MockGPUCommandEncoder(options);
    }

    get queue() {
      return {
        submit: (commandBuffers) => {},
        writeBuffer: (buffer, offset, data) => {}
      };
    }

    destroy() {}
  }

  // Set up the global gpu object
  global.navigator.gpu = {
    async requestAdapter(options) {
      return new MockGPUAdapter();
    }
  };
}

// Add other global mocks as needed
global.fetch = jest.fn().mockImplementation((url) => {
  return Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(''),
    blob: () => Promise.resolve(new Blob()),
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(0)),
  });
});

// Mock performance.now() for timing tests
const originalNow = performance.now;
global.performance.now = jest.fn(() => 1000);

// Add a cleanup to restore original values after tests
afterAll(() => {
  global.performance.now = originalNow;
});

// Suppress console errors during tests
const originalConsoleError = console.error;
console.error = jest.fn();

// Restore console.error after tests
afterAll(() => {
  console.error = originalConsoleError;
});