/**
 * WebNN Backend Additional Operations Tests
 */

import {
  executePoolingOperation,
  executeNormalizationOperation,
  executeElementwiseOperation,
  executeTensorManipulationOperation
} from './ipfs_accelerate_js_webnn_operations';

// Mock WebNN backend
const createMockBackend = () => {
  const graphBuilderMock = {
    // Mock pooling operations
    maxPool2d: jest.fn().mockReturnValue('maxpool_result'),
    averagePool2d: jest.fn().mockReturnValue('avgpool_result'),
    
    // Mock normalization operations
    batchNormalization: jest.fn().mockReturnValue('batchnorm_result'),
    layerNormalization: jest.fn().mockReturnValue('layernorm_result'),
    
    // Mock elementwise operations
    add: jest.fn().mockReturnValue('add_result'),
    sub: jest.fn().mockReturnValue('sub_result'),
    mul: jest.fn().mockReturnValue('mul_result'),
    div: jest.fn().mockReturnValue('div_result'),
    pow: jest.fn().mockReturnValue('pow_result'),
    min: jest.fn().mockReturnValue('min_result'),
    max: jest.fn().mockReturnValue('max_result'),
    exp: jest.fn().mockReturnValue('exp_result'),
    log: jest.fn().mockReturnValue('log_result'),
    sqrt: jest.fn().mockReturnValue('sqrt_result'),
    
    // Mock tensor manipulation operations
    reshape: jest.fn().mockReturnValue('reshape_result'),
    transpose: jest.fn().mockReturnValue('transpose_result'),
    concat: jest.fn().mockReturnValue('concat_result'),
    slice: jest.fn().mockReturnValue('slice_result'),
    pad: jest.fn().mockReturnValue('pad_result'),
    
    // Mock helpers for layer normalization
    reduceMean: jest.fn().mockReturnValue('reducemean_result'),
    sub: jest.fn().mockReturnValue('sub_result'),
    mul: jest.fn().mockReturnValue('mul_result'),
    sqrt: jest.fn().mockReturnValue('sqrt_result'),
    div: jest.fn().mockReturnValue('div_result'),
    constant: jest.fn().mockReturnValue('constant_result'),
    add: jest.fn().mockReturnValue('add_result')
  };
  
  return {
    graphBuilder: graphBuilderMock,
    runGraphComputation: jest.fn().mockResolvedValue({
      output: 'graph_result'
    })
  };
};

// Mock MLOperand and shapes
const createMockTensor = (shape = [2, 2]) => ({
  tensor: 'mock_tensor',
  shape
});

describe('WebNN Additional Operations', () => {
  describe('Pooling Operations', () => {
    test('executePoolingOperation - maxpool', async () => {
      const mockBackend = createMockBackend();
      const input = createMockTensor([1, 4, 4, 3]);
      
      const result = await executePoolingOperation(
        mockBackend as any,
        'max',
        { input },
        { windowDimensions: [2, 2] }
      );
      
      expect(mockBackend.graphBuilder.maxPool2d).toHaveBeenCalledWith(
        input.tensor,
        expect.objectContaining({
          windowDimensions: [2, 2]
        })
      );
      
      expect(result).toEqual({
        tensor: 'graph_result',
        shape: [1, 3, 3, 3], // Output shape calculation
        dataType: 'float32'
      });
    });
    
    test('executePoolingOperation - avgpool', async () => {
      const mockBackend = createMockBackend();
      const input = createMockTensor([1, 4, 4, 3]);
      
      const result = await executePoolingOperation(
        mockBackend as any,
        'average',
        { input },
        { 
          windowDimensions: [2, 2],
          strides: [2, 2],
          padding: [1, 1, 1, 1] 
        }
      );
      
      expect(mockBackend.graphBuilder.averagePool2d).toHaveBeenCalledWith(
        input.tensor,
        expect.objectContaining({
          windowDimensions: [2, 2],
          strides: [2, 2],
          padding: [1, 1, 1, 1]
        })
      );
      
      expect(result.dataType).toBe('float32');
    });
  });
  
  describe('Normalization Operations', () => {
    test('executeNormalizationOperation - batchnorm', async () => {
      const mockBackend = createMockBackend();
      const input = createMockTensor([1, 3, 3, 16]);
      const mean = createMockTensor([16]);
      const variance = createMockTensor([16]);
      const scale = createMockTensor([16]);
      const bias = createMockTensor([16]);
      
      const result = await executeNormalizationOperation(
        mockBackend as any,
        'batch',
        { input, mean, variance, scale, bias },
        { epsilon: 1e-5 }
      );
      
      expect(mockBackend.graphBuilder.batchNormalization).toHaveBeenCalledWith(
        input.tensor,
        mean.tensor,
        variance.tensor,
        scale.tensor,
        bias.tensor,
        expect.objectContaining({ epsilon: 1e-5 })
      );
      
      expect(result).toEqual({
        tensor: 'graph_result',
        shape: [1, 3, 3, 16],
        dataType: 'float32'
      });
    });
    
    test('executeNormalizationOperation - layernorm with direct support', async () => {
      const mockBackend = createMockBackend();
      const input = createMockTensor([1, 768]);
      const scale = createMockTensor([768]);
      const bias = createMockTensor([768]);
      
      const result = await executeNormalizationOperation(
        mockBackend as any,
        'layer',
        { input, scale, bias },
        { axes: [-1] }
      );
      
      expect(mockBackend.graphBuilder.layerNormalization).toHaveBeenCalledWith(
        input.tensor,
        scale.tensor,
        bias.tensor,
        expect.objectContaining({ 
          epsilon: 1e-5,
          axes: [-1]
        })
      );
      
      expect(result.shape).toEqual([1, 768]);
    });
  });
  
  describe('Elementwise Operations', () => {
    test('executeElementwiseOperation - add', async () => {
      const mockBackend = createMockBackend();
      const a = createMockTensor([2, 3]);
      const b = createMockTensor([2, 3]);
      
      const result = await executeElementwiseOperation(
        mockBackend as any,
        'add',
        { a, b }
      );
      
      expect(mockBackend.graphBuilder.add).toHaveBeenCalledWith(
        a.tensor,
        b.tensor
      );
      
      expect(result.shape).toEqual([2, 3]);
    });
    
    test('executeElementwiseOperation - unary operation (exp)', async () => {
      const mockBackend = createMockBackend();
      const a = createMockTensor([2, 3]);
      
      const result = await executeElementwiseOperation(
        mockBackend as any,
        'exp',
        { a }
      );
      
      expect(mockBackend.graphBuilder.exp).toHaveBeenCalledWith(a.tensor);
      expect(result.shape).toEqual([2, 3]);
    });
    
    test('executeElementwiseOperation - broadcasting shapes', async () => {
      const mockBackend = createMockBackend();
      const a = createMockTensor([4, 1]);
      const b = createMockTensor([1, 3]);
      
      const result = await executeElementwiseOperation(
        mockBackend as any,
        'mul',
        { a, b }
      );
      
      expect(mockBackend.graphBuilder.mul).toHaveBeenCalledWith(
        a.tensor,
        b.tensor
      );
      
      // Should broadcast to [4, 3]
      expect(result.shape).toEqual([4, 3]);
    });
  });
  
  describe('Tensor Manipulation Operations', () => {
    test('executeTensorManipulationOperation - reshape', async () => {
      const mockBackend = createMockBackend();
      const input = createMockTensor([2, 3]);
      
      const result = await executeTensorManipulationOperation(
        mockBackend as any,
        'reshape',
        { input },
        { newShape: [6, 1] }
      );
      
      expect(mockBackend.graphBuilder.reshape).toHaveBeenCalledWith(
        input.tensor,
        [6, 1]
      );
      
      expect(result.shape).toEqual([6, 1]);
    });
    
    test('executeTensorManipulationOperation - transpose', async () => {
      const mockBackend = createMockBackend();
      const input = createMockTensor([2, 3]);
      
      const result = await executeTensorManipulationOperation(
        mockBackend as any,
        'transpose',
        { input },
        { permutation: [1, 0] }
      );
      
      expect(mockBackend.graphBuilder.transpose).toHaveBeenCalledWith(
        input.tensor,
        [1, 0]
      );
      
      // Transposed shape should be [3, 2]
      expect(result.shape).toEqual([3, 2]);
    });
    
    test('executeTensorManipulationOperation - concat', async () => {
      const mockBackend = createMockBackend();
      const input = createMockTensor([2, 3]);
      const inputs = [
        createMockTensor([2, 3]),
        createMockTensor([2, 3])
      ];
      
      const result = await executeTensorManipulationOperation(
        mockBackend as any,
        'concat',
        { input, inputs },
        { axis: 1 }
      );
      
      expect(mockBackend.graphBuilder.concat).toHaveBeenCalledWith(
        inputs.map(i => i.tensor),
        1
      );
      
      // Concatenated shape along axis 1 should be [2, 6]
      expect(result.shape).toEqual([2, 6]);
    });
    
    test('executeTensorManipulationOperation - slice', async () => {
      const mockBackend = createMockBackend();
      const input = createMockTensor([4, 4]);
      
      const result = await executeTensorManipulationOperation(
        mockBackend as any,
        'slice',
        { input },
        { starts: [1, 1], sizes: [2, 2] }
      );
      
      expect(mockBackend.graphBuilder.slice).toHaveBeenCalledWith(
        input.tensor,
        [1, 1],
        [2, 2]
      );
      
      expect(result.shape).toEqual([2, 2]);
    });
  });
});