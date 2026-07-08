/**
 * CPU Backend Tests
 * Tests for the CPU backend implementation
 */

import { CPUBackend } from './ipfs_accelerate_js_cpu_backend';

describe('CPUBackend', () => {
  let backend: CPUBackend;

  beforeEach(() => {
    backend = new CPUBackend();
  });

  afterEach(() => {
    backend.dispose();
  });

  describe('initialization', () => {
    it('should initialize successfully', async () => {
      const result = await backend.initialize();
      expect(result).toBe(true);
    });

    it('should report being supported', async () => {
      const supported = await backend.isSupported();
      expect(supported).toBe(true);
    });

    it('should return capabilities', async () => {
      await backend.initialize();
      const capabilities = await backend.getCapabilities();
      expect(capabilities).toHaveProperty('typedArrays');
      expect(capabilities).toHaveProperty('optimizations');
      expect(capabilities).toHaveProperty('cache');
    });
  });

  describe('tensor creation', () => {
    it('should create float32 tensor', async () => {
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const result = await backend.createTensor(data, shape, 'float32');
      
      expect(result.data).toEqual(data);
      expect(result.shape).toEqual(shape);
      expect(result.dataType).toBe('float32');
    });
    
    it('should create int32 tensor', async () => {
      const data = new Int32Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const result = await backend.createTensor(data, shape, 'int32');
      
      expect(result.data).toEqual(data);
      expect(result.shape).toEqual(shape);
      expect(result.dataType).toBe('int32');
    });
    
    it('should create uint8 tensor', async () => {
      const data = new Uint8Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const result = await backend.createTensor(data, shape, 'uint8');
      
      expect(result.data).toEqual(data);
      expect(result.shape).toEqual(shape);
      expect(result.dataType).toBe('uint8');
    });
  });

  describe('basic operations', () => {
    it('should execute matmul operation', async () => {
      await backend.initialize();
      
      const a = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [2, 2], 'float32');
      const b = await backend.createTensor(new Float32Array([5, 6, 7, 8]), [2, 2], 'float32');
      
      const result = await backend.execute('matmul', { a, b });
      
      // Expected result:
      // [1, 2] * [5, 6] = [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
      // [3, 4] * [7, 8] = [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
      const expected = new Float32Array([19, 22, 43, 50]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([2, 2]);
    });
    
    it('should execute matmul with transpose options', async () => {
      await backend.initialize();
      
      const a = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [2, 2], 'float32');
      const b = await backend.createTensor(new Float32Array([5, 6, 7, 8]), [2, 2], 'float32');
      
      // Test transposeA
      const resultA = await backend.execute('matmul', { a, b }, { transposeA: true });
      
      // Expected result with a transposed:
      // [1, 3] * [5, 6] = [1*5 + 3*7, 1*6 + 3*8] = [26, 30]
      // [2, 4] * [7, 8] = [2*5 + 4*7, 2*6 + 4*8] = [38, 44]
      const expectedA = new Float32Array([26, 30, 38, 44]);
      
      expect(resultA.data).toEqual(expectedA);
      expect(resultA.shape).toEqual([2, 2]);
      
      // Test transposeB
      const resultB = await backend.execute('matmul', { a, b }, { transposeB: true });
      
      // Expected result with b transposed:
      // [1, 2] * [5, 7] = [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
      // [3, 4] * [6, 8] = [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
      const expectedB = new Float32Array([17, 23, 39, 53]);
      
      expect(resultB.data).toEqual(expectedB);
      expect(resultB.shape).toEqual([2, 2]);
    });
    
    it('should execute relu operation', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([-1, 0, 1, 2]), [2, 2], 'float32');
      
      // Test elementwise with relu
      const result = await backend.execute('elementwise', { input }, { operation: 'relu' });
      
      // Expected result:
      // relu([-1, 0, 1, 2]) = [0, 0, 1, 2]
      const expected = new Float32Array([0, 0, 1, 2]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([2, 2]);
    });
    
    it('should execute sigmoid operation', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([-1, 0, 1, 2]), [2, 2], 'float32');
      
      // Test elementwise with sigmoid
      const result = await backend.execute('elementwise', { input }, { operation: 'sigmoid' });
      
      // Expected result:
      // sigmoid(x) = 1 / (1 + exp(-x))
      // sigmoid([-1, 0, 1, 2]) ≈ [0.269, 0.5, 0.731, 0.881]
      
      // Check with reasonable tolerance
      expect(Math.abs(result.data[0] - 0.269)).toBeLessThan(0.001);
      expect(Math.abs(result.data[1] - 0.5)).toBeLessThan(0.001);
      expect(Math.abs(result.data[2] - 0.731)).toBeLessThan(0.001);
      expect(Math.abs(result.data[3] - 0.881)).toBeLessThan(0.001);
      expect(result.shape).toEqual([2, 2]);
    });
    
    it('should execute softmax operation', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [2, 2], 'float32');
      
      // Test softmax
      const result = await backend.execute('softmax', { input }, { axis: 1 });
      
      // Expected result:
      // For row 1: softmax([1, 2]) = [exp(1), exp(2)] / (exp(1) + exp(2)) ≈ [0.269, 0.731]
      // For row 2: softmax([3, 4]) = [exp(3), exp(4)] / (exp(3) + exp(4)) ≈ [0.269, 0.731]
      
      // Check first row
      expect(Math.abs(result.data[0] - 0.269)).toBeLessThan(0.001);
      expect(Math.abs(result.data[1] - 0.731)).toBeLessThan(0.001);
      
      // Check second row
      expect(Math.abs(result.data[2] - 0.269)).toBeLessThan(0.001);
      expect(Math.abs(result.data[3] - 0.731)).toBeLessThan(0.001);
      
      expect(result.shape).toEqual([2, 2]);
    });
  });

  describe('binary operations', () => {
    it('should execute add operation', async () => {
      await backend.initialize();
      
      const a = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [2, 2], 'float32');
      const b = await backend.createTensor(new Float32Array([5, 6, 7, 8]), [2, 2], 'float32');
      
      const result = await backend.execute('add', { a, b });
      
      const expected = new Float32Array([6, 8, 10, 12]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([2, 2]);
    });
    
    it('should execute sub operation', async () => {
      await backend.initialize();
      
      const a = await backend.createTensor(new Float32Array([5, 6, 7, 8]), [2, 2], 'float32');
      const b = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [2, 2], 'float32');
      
      const result = await backend.execute('sub', { a, b });
      
      const expected = new Float32Array([4, 4, 4, 4]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([2, 2]);
    });
    
    it('should execute mul operation', async () => {
      await backend.initialize();
      
      const a = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [2, 2], 'float32');
      const b = await backend.createTensor(new Float32Array([5, 6, 7, 8]), [2, 2], 'float32');
      
      const result = await backend.execute('mul', { a, b });
      
      const expected = new Float32Array([5, 12, 21, 32]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([2, 2]);
    });
    
    it('should execute div operation', async () => {
      await backend.initialize();
      
      const a = await backend.createTensor(new Float32Array([10, 12, 14, 16]), [2, 2], 'float32');
      const b = await backend.createTensor(new Float32Array([2, 3, 2, 4]), [2, 2], 'float32');
      
      const result = await backend.execute('div', { a, b });
      
      const expected = new Float32Array([5, 4, 7, 4]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([2, 2]);
    });
    
    it('should handle broadcasting in binary operations', async () => {
      await backend.initialize();
      
      const a = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [2, 2], 'float32');
      const b = await backend.createTensor(new Float32Array([10, 20]), [2], 'float32');
      
      const result = await backend.execute('add', { a, b });
      
      // Broadcasting b to [2, 2] makes it [[10, 20], [10, 20]]
      const expected = new Float32Array([11, 22, 13, 24]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([2, 2]);
    });
  });
  
  describe('unary operations', () => {
    it('should execute exp operation', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([0, 1, Math.log(2), 2]), [2, 2], 'float32');
      
      const result = await backend.execute('exp', { input });
      
      // Expected: [exp(0), exp(1), exp(log(2)), exp(2)] = [1, e, 2, e²]
      const expected = new Float32Array([1, Math.exp(1), 2, Math.exp(2)]);
      
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(result.data[i] - expected[i])).toBeLessThan(0.0001);
      }
      
      expect(result.shape).toEqual([2, 2]);
    });
    
    it('should execute log operation', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([1, Math.E, 10, 100]), [2, 2], 'float32');
      
      const result = await backend.execute('log', { input });
      
      // Expected: [log(1), log(e), log(10), log(100)] = [0, 1, log(10), log(100)]
      const expected = new Float32Array([0, 1, Math.log(10), Math.log(100)]);
      
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(result.data[i] - expected[i])).toBeLessThan(0.0001);
      }
      
      expect(result.shape).toEqual([2, 2]);
    });
    
    it('should execute sqrt operation', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([0, 1, 4, 9]), [2, 2], 'float32');
      
      const result = await backend.execute('sqrt', { input });
      
      // Expected: [sqrt(0), sqrt(1), sqrt(4), sqrt(9)] = [0, 1, 2, 3]
      const expected = new Float32Array([0, 1, 2, 3]);
      
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(result.data[i] - expected[i])).toBeLessThan(0.0001);
      }
      
      expect(result.shape).toEqual([2, 2]);
    });
  });
  
  describe('tensor manipulation', () => {
    it('should execute reshape operation', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3], 'float32');
      
      // Reshape to [3, 2]
      const result = await backend.execute('reshape', { input }, { newShape: [3, 2] });
      
      expect(result.data).toEqual(input.data);
      expect(result.shape).toEqual([3, 2]);
    });
    
    it('should execute transpose operation', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3], 'float32');
      
      // Transpose to [3, 2]
      const result = await backend.execute('transpose', { input }, { permutation: [1, 0] });
      
      // Expected:
      // Input:  [1, 2, 3]
      //         [4, 5, 6]
      // 
      // Output: [1, 4]
      //         [2, 5]
      //         [3, 6]
      
      const expected = new Float32Array([1, 4, 2, 5, 3, 6]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([3, 2]);
    });
    
    it('should execute slice operation', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]), [3, 3], 'float32');
      
      // Slice [1:3, 0:2]
      const result = await backend.execute('slice', { input }, { begin: [1, 0], size: [2, 2] });
      
      // Expected:
      // Input:  [1, 2, 3]
      //         [4, 5, 6]
      //         [7, 8, 9]
      // 
      // Slice:  [4, 5]
      //         [7, 8]
      
      const expected = new Float32Array([4, 5, 7, 8]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([2, 2]);
    });
    
    it('should execute concat operation', async () => {
      await backend.initialize();
      
      const a = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [2, 2], 'float32');
      const b = await backend.createTensor(new Float32Array([5, 6, 7, 8]), [2, 2], 'float32');
      
      // Concat along axis 0
      const result = await backend.execute('concat', { inputs: [a, b] }, { axis: 0 });
      
      // Expected:
      // A:  [1, 2]    B:  [5, 6]    Result:  [1, 2]
      //     [3, 4]        [7, 8]             [3, 4]
      //                                       [5, 6]
      //                                       [7, 8]
      
      const expected = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([4, 2]);
    });
  });
  
  describe('conv2d and pooling', () => {
    it('should execute conv2d operation', async () => {
      await backend.initialize();
      
      // Simple 1x1 convolution test (dot product)
      // Input: 2x2x1, Filter: 1x1x1x1 (outputChannels, inputChannels, height, width)
      const input = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [1, 2, 2, 1], 'float32');
      const filter = await backend.createTensor(new Float32Array([2]), [1, 1, 1, 1], 'float32');
      
      const result = await backend.execute('conv2d', { input, filter });
      
      // Expected:
      // Input:  [1, 2]    Filter: [2]    Result:  [2, 4]
      //         [3, 4]                            [6, 8]
      
      const expected = new Float32Array([2, 4, 6, 8]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([1, 2, 2, 1]);
    });
    
    it('should execute maxpool operation', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [1, 2, 2, 1], 'float32');
      
      const result = await backend.execute('maxpool', { input }, { windowDimensions: [2, 2], strides: [1, 1] });
      
      // Expected:
      // Input:  [1, 2]    Result:  [4]
      //         [3, 4]              
      
      const expected = new Float32Array([4]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([1, 1, 1, 1]);
    });
    
    it('should execute avgpool operation', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [1, 2, 2, 1], 'float32');
      
      const result = await backend.execute('avgpool', { input }, { windowDimensions: [2, 2], strides: [1, 1] });
      
      // Expected:
      // Input:  [1, 2]    Result:  [2.5]
      //         [3, 4]              
      
      const expected = new Float32Array([2.5]);
      
      expect(result.data).toEqual(expected);
      expect(result.shape).toEqual([1, 1, 1, 1]);
    });
  });
  
  describe('quantization', () => {
    it('should execute quantize and dequantize operations', async () => {
      await backend.initialize();
      
      const input = await backend.createTensor(new Float32Array([-1, -0.5, 0, 0.5, 1]), [5], 'float32');
      
      // Quantize to int8
      const quantized = await backend.execute('quantize', { input });
      
      // Max value is 1.0, so scale should be 127
      expect(quantized.scale[0]).toBeCloseTo(127);
      
      // Check quantized values
      expect(quantized.data[0]).toBeCloseTo(-127);
      expect(quantized.data[1]).toBeCloseTo(-64);
      expect(quantized.data[2]).toBe(0);
      expect(quantized.data[3]).toBeCloseTo(64);
      expect(quantized.data[4]).toBeCloseTo(127);
      
      // Dequantize back to float32
      const dequantized = await backend.execute('dequantize', { input: quantized, scale: { data: quantized.scale } });
      
      // Check dequantized values (should be close to the original)
      expect(dequantized.data[0]).toBeCloseTo(-1);
      expect(dequantized.data[1]).toBeCloseTo(-0.5);
      expect(dequantized.data[2]).toBeCloseTo(0);
      expect(dequantized.data[3]).toBeCloseTo(0.5);
      expect(dequantized.data[4]).toBeCloseTo(1);
    });
  });
  
  describe('result caching', () => {
    it('should cache results when enabled', async () => {
      // Create backend with caching enabled
      const cachedBackend = new CPUBackend({ cacheResults: true });
      await cachedBackend.initialize();
      
      const a = await cachedBackend.createTensor(new Float32Array([1, 2, 3, 4]), [2, 2], 'float32');
      const b = await cachedBackend.createTensor(new Float32Array([5, 6, 7, 8]), [2, 2], 'float32');
      
      // First execution
      const result1 = await cachedBackend.execute('matmul', { a, b });
      
      // Mock spying on the executeMatmul method
      const spy = jest.spyOn(cachedBackend as any, 'executeMatmul');
      
      // Second execution with same inputs and operation should use cache
      const result2 = await cachedBackend.execute('matmul', { a, b });
      
      // Check that executeMatmul wasn't called for the second execution
      expect(spy).not.toHaveBeenCalled();
      
      // Results should be the same
      expect(result2.data).toEqual(result1.data);
      expect(result2.shape).toEqual(result1.shape);
      
      // Clean up
      cachedBackend.dispose();
      spy.mockRestore();
    });
  });
});