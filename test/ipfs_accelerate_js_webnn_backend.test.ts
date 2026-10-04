/**
 * Tests for WebNN Backend Implementation
 */

import { WebNNBackend } from './webnn_backend';
import { HardwareBackend } from './hardware_abstraction';

describe('WebNNBackend', () => {
  let webnnBackend: WebNNBackend;

  beforeEach(() => {
    webnnBackend = new WebNNBackend({
      deviceType: 'gpu',
      powerPreference: 'high-performance',
      enableLogging: false
    });
  });

  afterEach(() => {
    webnnBackend.dispose();
  });

  describe('initialization', () => {
    it('should check if WebNN is supported', async () => {
      const isSupported = await webnnBackend.isSupported();
      expect(isSupported).toBeDefined();
    });

    it('should initialize the backend', async () => {
      const result = await webnnBackend.initialize();
      expect(result).toBe(true);
    });

    it('should get capabilities after initialization', async () => {
      await webnnBackend.initialize();
      const capabilities = await webnnBackend.getCapabilities();
      
      expect(capabilities).toBeDefined();
      expect(capabilities.deviceType).toBeDefined();
      expect(capabilities.operations).toBeDefined();
      expect(Array.isArray(capabilities.operations)).toBe(true);
    });

    it('should correctly identify the backend type', () => {
      expect(webnnBackend.type).toBe('webnn');
    });

    it('should report realistic browser detection', async () => {
      // Mock navigator.userAgent to simulate different browser environments
      const originalUserAgent = navigator.userAgent;
      
      // Test Edge detection (best WebNN support)
      Object.defineProperty(navigator, 'userAgent', {
        value: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.63',
        configurable: true
      });
      
      await webnnBackend.initialize();
      const edgeCapabilities = await webnnBackend.getCapabilities();
      
      // Reset and dispose
      webnnBackend.dispose();
      Object.defineProperty(navigator, 'userAgent', { value: originalUserAgent, configurable: true });
      
      // We can't assert specific browser detection properties here since our mock doesn't change,
      // but we can verify the capabilities structure is consistent
      expect(edgeCapabilities.deviceType).toBeDefined();
    });
  });

  describe('tensor operations', () => {
    beforeEach(async () => {
      await webnnBackend.initialize();
    });

    it('should create a tensor from data', async () => {
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const dataType = 'float32';
      
      const tensor = await webnnBackend.createTensor(data, shape, dataType);
      
      expect(tensor).toBeDefined();
      expect(tensor.tensor).toBeDefined();
      expect(tensor.shape).toEqual(shape);
      expect(tensor.dataType).toBe(dataType);
      expect(tensor.id).toBeDefined();
      expect(tensor.size).toBeGreaterThan(0);
    });

    it('should throw an error when creating tensor with mismatched data type', async () => {
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const dataType = 'int32'; // Mismatched type
      
      await expect(webnnBackend.createTensor(data, shape, dataType)).rejects.toThrow();
    });

    it('should read tensor data back to CPU', async () => {
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const dataType = 'float32';
      
      const tensor = await webnnBackend.createTensor(data, shape, dataType);
      const result = await webnnBackend.readTensor(tensor.tensor, shape, dataType);
      
      expect(result).toBeDefined();
      expect(result.length).toBe(data.length);
      // In our mock implementation, readOperand fills the array with sequential values
      expect(result instanceof Float32Array).toBe(true);
    });

    it('should handle int32 tensors correctly', async () => {
      const data = new Int32Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const dataType = 'int32';
      
      const tensor = await webnnBackend.createTensor(data, shape, dataType);
      const result = await webnnBackend.readTensor(tensor.tensor, shape, dataType);
      
      expect(result).toBeDefined();
      expect(result.length).toBe(data.length);
      expect(result instanceof Int32Array).toBe(true);
    });

    it('should handle uint8 tensors correctly', async () => {
      const data = new Uint8Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const dataType = 'uint8';
      
      const tensor = await webnnBackend.createTensor(data, shape, dataType);
      const result = await webnnBackend.readTensor(tensor.tensor, shape, dataType);
      
      expect(result).toBeDefined();
      expect(result.length).toBe(data.length);
      expect(result instanceof Uint8Array).toBe(true);
    });

    it('should handle different tensor shapes', async () => {
      // Test different tensor shapes
      const testCases = [
        { data: new Float32Array([1, 2, 3, 4]), shape: [4], dataType: 'float32' },
        { data: new Float32Array([1, 2, 3, 4, 5, 6]), shape: [2, 3], dataType: 'float32' },
        { data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), shape: [2, 2, 2], dataType: 'float32' }
      ];
      
      for (const testCase of testCases) {
        const tensor = await webnnBackend.createTensor(testCase.data, testCase.shape, testCase.dataType);
        expect(tensor).toBeDefined();
        expect(tensor.shape).toEqual(testCase.shape);
        
        const result = await webnnBackend.readTensor(tensor.tensor, testCase.shape, testCase.dataType);
        expect(result.length).toBe(testCase.data.length);
      }
    });
  });

  describe('execution operations', () => {
    let inputTensorA: any;
    let inputTensorB: any;
    let filterTensor: any;

    beforeEach(async () => {
      await webnnBackend.initialize();
      
      // Create test tensors
      inputTensorA = await webnnBackend.createTensor(
        new Float32Array([1, 2, 3, 4]), 
        [2, 2], 
        'float32'
      );
      
      inputTensorB = await webnnBackend.createTensor(
        new Float32Array([5, 6, 7, 8]), 
        [2, 2], 
        'float32'
      );
      
      filterTensor = await webnnBackend.createTensor(
        new Float32Array([1, 1, 1, 0, 0, 0, -1, -1, -1]), 
        [1, 1, 3, 3], 
        'float32'
      );
    });

    it('should execute matrix multiplication operation', async () => {
      const result = await webnnBackend.execute('matmul', {
        a: inputTensorA,
        b: inputTensorB
      }, {
        transposeA: false,
        transposeB: false
      });
      
      expect(result).toBeDefined();
      expect(result.tensor).toBeDefined();
      expect(result.shape).toEqual([2, 2]);
      expect(result.dataType).toBe('float32');
    });

    it('should execute matrix multiplication with transposed inputs', async () => {
      // Test transposed matmul operations
      const transposeTests = [
        { transposeA: true, transposeB: false },
        { transposeA: false, transposeB: true },
        { transposeA: true, transposeB: true }
      ];
      
      for (const { transposeA, transposeB } of transposeTests) {
        const result = await webnnBackend.execute('matmul', {
          a: inputTensorA,
          b: inputTensorB
        }, {
          transposeA,
          transposeB
        });
        
        expect(result).toBeDefined();
        expect(result.tensor).toBeDefined();
        expect(result.dataType).toBe('float32');
      }
    });

    it('should execute elementwise operations', async () => {
      const operations = ['relu', 'sigmoid', 'tanh'];
      
      for (const operation of operations) {
        const result = await webnnBackend.execute('elementwise', {
          input: inputTensorA
        }, {
          operation
        });
        
        expect(result).toBeDefined();
        expect(result.tensor).toBeDefined();
        expect(result.shape).toEqual(inputTensorA.shape);
        expect(result.dataType).toBe('float32');
      }
    });

    it('should execute softmax operation', async () => {
      const result = await webnnBackend.execute('softmax', {
        input: inputTensorA
      }, {
        axis: -1
      });
      
      expect(result).toBeDefined();
      expect(result.tensor).toBeDefined();
      expect(result.shape).toEqual(inputTensorA.shape);
      expect(result.dataType).toBe('float32');
    });

    it('should execute softmax with different axis values', async () => {
      const axisValues = [-1, 0, 1];
      
      for (const axis of axisValues) {
        const result = await webnnBackend.execute('softmax', {
          input: inputTensorA
        }, {
          axis
        });
        
        expect(result).toBeDefined();
        expect(result.tensor).toBeDefined();
        expect(result.shape).toEqual(inputTensorA.shape);
      }
    });

    it('should execute 2D convolution operation', async () => {
      const result = await webnnBackend.execute('conv2d', {
        input: inputTensorA,
        filter: filterTensor
      }, {
        padding: [1, 1, 1, 1],
        strides: [1, 1]
      });
      
      expect(result).toBeDefined();
      expect(result.tensor).toBeDefined();
      expect(result.shape).toBeDefined();
      expect(result.dataType).toBe('float32');
    });

    it('should execute 2D convolution with different parameters', async () => {
      const convParams = [
        { padding: [0, 0, 0, 0], strides: [1, 1] },
        { padding: [1, 1, 1, 1], strides: [2, 2] },
        { padding: [2, 2, 2, 2], strides: [1, 1], dilations: [2, 2] }
      ];
      
      for (const params of convParams) {
        const result = await webnnBackend.execute('conv2d', {
          input: inputTensorA,
          filter: filterTensor
        }, params);
        
        expect(result).toBeDefined();
        expect(result.tensor).toBeDefined();
        expect(result.dataType).toBe('float32');
      }
    });

    it('should throw error for unsupported operation', async () => {
      await expect(webnnBackend.execute('unsupported_operation', {}, {})).rejects.toThrow();
    });
  });

  describe('graph computation', () => {
    beforeEach(async () => {
      await webnnBackend.initialize();
    });

    it('should cache compiled graphs for better performance', async () => {
      // This test will be implementation-specific based on how graph caching is handled
      // Here we're just testing that the function runs without errors
      const inputTensorA = await webnnBackend.createTensor(
        new Float32Array([1, 2, 3, 4]), 
        [2, 2], 
        'float32'
      );
      
      const inputTensorB = await webnnBackend.createTensor(
        new Float32Array([5, 6, 7, 8]), 
        [2, 2], 
        'float32'
      );
      
      // Run the same operation twice to test caching
      const result1 = await webnnBackend.execute('matmul', {
        a: inputTensorA,
        b: inputTensorB
      });
      
      const result2 = await webnnBackend.execute('matmul', {
        a: inputTensorA,
        b: inputTensorB
      });
      
      expect(result1).toBeDefined();
      expect(result2).toBeDefined();
    });

    it('should handle complex computation graphs', async () => {
      // Create input tensor
      const inputTensor = await webnnBackend.createTensor(
        new Float32Array([1, 2, 3, 4]), 
        [2, 2], 
        'float32'
      );
      
      // Chain multiple operations to test graph building
      // 1. Apply ReLU
      const reluResult = await webnnBackend.execute('elementwise', {
        input: inputTensor
      }, {
        operation: 'relu'
      });
      
      // 2. Apply matrix multiplication with itself
      const matmulResult = await webnnBackend.execute('matmul', {
        a: reluResult,
        b: reluResult
      });
      
      // 3. Apply softmax
      const softmaxResult = await webnnBackend.execute('softmax', {
        input: matmulResult
      });
      
      expect(softmaxResult).toBeDefined();
      expect(softmaxResult.tensor).toBeDefined();
      expect(softmaxResult.shape).toEqual([2, 2]);
    });

    it('should handle computation with different input shapes', async () => {
      // Create tensors with different shapes
      const shapes = [
        [1, 4],
        [2, 2],
        [4, 1]
      ];
      
      for (const shape of shapes) {
        const tensor = await webnnBackend.createTensor(
          new Float32Array([1, 2, 3, 4]), 
          shape, 
          'float32'
        );
        
        const result = await webnnBackend.execute('elementwise', {
          input: tensor
        }, {
          operation: 'relu'
        });
        
        expect(result).toBeDefined();
        expect(result.shape).toEqual(shape);
      }
    });
  });

  describe('memory management', () => {
    beforeEach(async () => {
      await webnnBackend.initialize();
    });

    it('should dispose of resources', () => {
      webnnBackend.dispose();
      // Test that dispose runs without errors
      expect(true).toBe(true);
    });

    it('should handle garbage collection of tensors', async () => {
      // Create many tensors to trigger garbage collection
      const tensors = [];
      for (let i = 0; i < 20; i++) {
        const tensor = await webnnBackend.createTensor(
          new Float32Array([1, 2, 3, 4]), 
          [2, 2], 
          'float32'
        );
        tensors.push(tensor);
      }
      
      // Test that we can still create and use tensors after GC
      const newTensor = await webnnBackend.createTensor(
        new Float32Array([1, 2, 3, 4]), 
        [2, 2], 
        'float32'
      );
      
      expect(newTensor).toBeDefined();
      expect(newTensor.tensor).toBeDefined();
    });

    it('should handle creating and releasing many tensors', async () => {
      // Create and use many tensors in sequence
      for (let i = 0; i < 50; i++) {
        const tensor = await webnnBackend.createTensor(
          new Float32Array([i, i+1, i+2, i+3]), 
          [2, 2], 
          'float32'
        );
        
        // Use the tensor in a simple operation
        const result = await webnnBackend.execute('elementwise', {
          input: tensor
        }, {
          operation: 'relu'
        });
        
        expect(result).toBeDefined();
      }
      
      // Memory usage should be managed by the backend
      // We can't directly test memory metrics, but we can ensure the backend still works
      const finalTensor = await webnnBackend.createTensor(
        new Float32Array([100, 200, 300, 400]), 
        [2, 2], 
        'float32'
      );
      
      expect(finalTensor).toBeDefined();
    });
  });

  describe('error handling', () => {
    it('should handle initialization errors gracefully', async () => {
      // Temporarily mock navigator.ml to simulate an error
      const originalMl = navigator.ml;
      (navigator as any).ml = {
        async createContext() {
          throw new Error('Simulated error');
        }
      };
      
      const result = await webnnBackend.initialize();
      expect(result).toBe(false);
      
      // Restore navigator.ml
      (navigator as any).ml = originalMl;
    });

    it('should handle tensor creation errors', async () => {
      await webnnBackend.initialize();
      
      // Mock context to throw an error
      (webnnBackend as any).context = {
        createOperand: () => {
          throw new Error('Simulated error');
        }
      };
      
      await expect(webnnBackend.createTensor(
        new Float32Array([1, 2, 3, 4]), 
        [2, 2], 
        'float32'
      )).rejects.toThrow();
    });

    it('should handle invalid tensor shape', async () => {
      await webnnBackend.initialize();
      
      // Test with invalid shape (mismatch between shape and data length)
      await expect(webnnBackend.createTensor(
        new Float32Array([1, 2, 3, 4]), 
        [3, 3], // 9 elements expected but only 4 provided
        'float32'
      )).rejects.toThrow();
    });

    it('should handle execution errors', async () => {
      await webnnBackend.initialize();
      
      const inputTensor = await webnnBackend.createTensor(
        new Float32Array([1, 2, 3, 4]), 
        [2, 2], 
        'float32'
      );
      
      // Mock graph builder to throw an error
      (webnnBackend as any).graphBuilder = {
        relu: () => {
          throw new Error('Simulated execution error');
        }
      };
      
      await expect(webnnBackend.execute('elementwise', {
        input: inputTensor
      }, {
        operation: 'relu'
      })).rejects.toThrow();
    });
  });

  describe('hardware abstraction layer compatibility', () => {
    it('should implement the HardwareBackend interface', () => {
      // This test ensures that WebNNBackend correctly implements the HardwareBackend interface
      const backend: HardwareBackend = webnnBackend;
      
      // Check that all required methods are implemented
      expect(typeof backend.initialize).toBe('function');
      expect(typeof backend.isSupported).toBe('function');
      expect(typeof backend.getCapabilities).toBe('function');
      expect(typeof backend.createTensor).toBe('function');
      expect(typeof backend.readTensor).toBe('function');
      expect(typeof backend.execute).toBe('function');
      expect(typeof backend.dispose).toBe('function');
      expect(backend.type).toBe('webnn');
    });
  });

  describe('browser-specific behavior', () => {
    it('should handle simulation detection', async () => {
      // Test simulation detection logic
      const originalContext = (webnnBackend as any).context;
      
      // Mock a simulated implementation (CPU device type)
      (webnnBackend as any).context = { deviceType: 'cpu' };
      (webnnBackend as any).detectDeviceInfo();
      
      expect((webnnBackend as any).deviceInfo.isSimulated).toBe(true);
      
      // Mock a hardware implementation (GPU device type)
      (webnnBackend as any).context = { deviceType: 'gpu' };
      (webnnBackend as any).deviceInfo.deviceName = 'NVIDIA GeForce RTX 3080';
      (webnnBackend as any).detectDeviceInfo();
      
      expect((webnnBackend as any).deviceInfo.isSimulated).toBe(false);
      
      // Mock a known software renderer
      (webnnBackend as any).context = { deviceType: 'gpu' };
      (webnnBackend as any).deviceInfo.deviceName = 'Microsoft Basic Renderer';
      (webnnBackend as any).detectDeviceInfo();
      
      expect((webnnBackend as any).deviceInfo.isSimulated).toBe(true);
      
      // Restore original context
      (webnnBackend as any).context = originalContext;
    });
  });
});