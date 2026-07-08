/**
 * WebNN Graph Builder Tests
 * Tests the WebNN graph builder functionality
 */

import { WebNNGraphBuilder } from './ipfs_accelerate_js_webnn_graph_builder';
import { WebNNBackend } from './ipfs_accelerate_js_webnn_backend';
import { Tensor } from './ipfs_accelerate_js_tensor';

// Mock WebNN backend for testing
jest.mock('./ipfs_accelerate_js_webnn_backend');

describe('WebNNGraphBuilder', () => {
  let mockBackend: jest.Mocked<WebNNBackend>;
  let graphBuilder: WebNNGraphBuilder;
  let mockMLContext: any;
  let mockGraphBuilder: any;
  
  // Mock WebNN graph creation
  const mockGraph = {
    name: 'MockGraph',
    compute: jest.fn().mockResolvedValue(undefined)
  };
  
  // Mock WebNN operands
  const mockInputOperand = { name: 'input_operand' };
  const mockConstantOperand = { name: 'constant_operand' };
  const mockOutputOperand = { name: 'output_operand' };
  
  // Setup global navigator mock
  const mockCreateGraphBuilder = jest.fn().mockResolvedValue({
    input: jest.fn().mockReturnValue(mockInputOperand),
    constant: jest.fn().mockReturnValue(mockConstantOperand),
    matmul: jest.fn().mockReturnValue(mockOutputOperand),
    add: jest.fn().mockReturnValue(mockOutputOperand),
    sub: jest.fn().mockReturnValue(mockOutputOperand),
    mul: jest.fn().mockReturnValue(mockOutputOperand),
    div: jest.fn().mockReturnValue(mockOutputOperand),
    relu: jest.fn().mockReturnValue(mockOutputOperand),
    sigmoid: jest.fn().mockReturnValue(mockOutputOperand),
    tanh: jest.fn().mockReturnValue(mockOutputOperand),
    softmax: jest.fn().mockReturnValue(mockOutputOperand),
    reshape: jest.fn().mockReturnValue(mockOutputOperand),
    transpose: jest.fn().mockReturnValue(mockOutputOperand),
    concat: jest.fn().mockReturnValue(mockOutputOperand),
    conv2d: jest.fn().mockReturnValue(mockOutputOperand),
    maxPool2d: jest.fn().mockReturnValue(mockOutputOperand),
    averagePool2d: jest.fn().mockReturnValue(mockOutputOperand),
    batchNormalization: jest.fn().mockReturnValue(mockOutputOperand),
    gemm: jest.fn().mockReturnValue(mockOutputOperand),
    max: jest.fn().mockReturnValue(mockOutputOperand),
    build: jest.fn().mockResolvedValue(mockGraph)
  });
  
  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Setup mock WebNN context
    mockMLContext = {
      compute: jest.fn().mockImplementation((graph, inputs, outputs) => {
        // Fill output with dummy data for testing
        for (const key of Object.keys(outputs)) {
          const output = outputs[key];
          if (output instanceof Float32Array) {
            for (let i = 0; i < output.length; i++) {
              output[i] = 0.5;
            }
          }
        }
        return Promise.resolve();
      })
    };
    
    // Setup mock graphBuilder
    mockGraphBuilder = {
      input: jest.fn().mockReturnValue(mockInputOperand),
      constant: jest.fn().mockReturnValue(mockConstantOperand),
      matmul: jest.fn().mockReturnValue(mockOutputOperand),
      add: jest.fn().mockReturnValue(mockOutputOperand),
      sub: jest.fn().mockReturnValue(mockOutputOperand),
      mul: jest.fn().mockReturnValue(mockOutputOperand),
      div: jest.fn().mockReturnValue(mockOutputOperand),
      relu: jest.fn().mockReturnValue(mockOutputOperand),
      sigmoid: jest.fn().mockReturnValue(mockOutputOperand),
      tanh: jest.fn().mockReturnValue(mockOutputOperand),
      softmax: jest.fn().mockReturnValue(mockOutputOperand),
      reshape: jest.fn().mockReturnValue(mockOutputOperand),
      transpose: jest.fn().mockReturnValue(mockOutputOperand),
      concat: jest.fn().mockReturnValue(mockOutputOperand),
      conv2d: jest.fn().mockReturnValue(mockOutputOperand),
      maxPool2d: jest.fn().mockReturnValue(mockOutputOperand),
      averagePool2d: jest.fn().mockReturnValue(mockOutputOperand),
      batchNormalization: jest.fn().mockReturnValue(mockOutputOperand),
      gemm: jest.fn().mockReturnValue(mockOutputOperand),
      max: jest.fn().mockReturnValue(mockOutputOperand),
      build: jest.fn().mockResolvedValue(mockGraph)
    };
    
    // Setup mock backend
    mockBackend = {
      id: 'mock-webnn',
      type: 'webnn',
      isAvailable: true,
      initialize: jest.fn().mockResolvedValue(undefined),
      isInitialized: jest.fn().mockReturnValue(true),
      allocateTensor: jest.fn().mockResolvedValue(undefined),
      releaseTensor: jest.fn().mockReturnValue(undefined),
      add: jest.fn().mockResolvedValue(new Tensor([2, 2], [1, 2, 3, 4])),
      subtract: jest.fn().mockResolvedValue(new Tensor([2, 2], [1, 2, 3, 4])),
      multiply: jest.fn().mockResolvedValue(new Tensor([2, 2], [1, 2, 3, 4])),
      divide: jest.fn().mockResolvedValue(new Tensor([2, 2], [1, 2, 3, 4])),
      matmul: jest.fn().mockResolvedValue(new Tensor([2, 2], [1, 2, 3, 4])),
      transpose: jest.fn().mockResolvedValue(new Tensor([2, 2], [1, 2, 3, 4])),
      relu: jest.fn().mockResolvedValue(new Tensor([2, 2], [1, 2, 3, 4])),
      sigmoid: jest.fn().mockResolvedValue(new Tensor([2, 2], [1, 2, 3, 4])),
      tanh: jest.fn().mockResolvedValue(new Tensor([2, 2], [1, 2, 3, 4])),
      softmax: jest.fn().mockResolvedValue(new Tensor([2, 2], [1, 2, 3, 4])),
      reshape: jest.fn().mockResolvedValue(new Tensor([2, 2], [1, 2, 3, 4])),
      sync: jest.fn().mockResolvedValue(undefined),
      dispose: jest.fn().mockReturnValue(undefined)
    } as unknown as jest.Mocked<WebNNBackend>;
    
    // Add internal properties for mocking
    (mockBackend as any).context = mockMLContext;
    (mockBackend as any).graphBuilder = mockGraphBuilder;
    
    // Setup mock navigator
    global.navigator = {
      ...global.navigator,
      ml: {
        createContext: jest.fn().mockResolvedValue(mockMLContext),
        createGraphBuilder: mockCreateGraphBuilder
      }
    } as any;
    
    // Create the graph builder
    graphBuilder = new WebNNGraphBuilder(mockBackend);
  });
  
  describe('Initialization', () => {
    it('should initialize the graph builder correctly', async () => {
      await graphBuilder.initialize();
      
      expect(mockBackend.isInitialized).toHaveBeenCalled();
      expect(mockCreateGraphBuilder).toHaveBeenCalledWith(mockMLContext);
    });
    
    it('should initialize the backend if it is not initialized', async () => {
      // Set backend as not initialized
      mockBackend.isInitialized.mockReturnValue(false);
      
      await graphBuilder.initialize();
      
      expect(mockBackend.initialize).toHaveBeenCalled();
      expect(mockCreateGraphBuilder).toHaveBeenCalledWith(mockMLContext);
    });
    
    it('should not reinitialize if already initialized', async () => {
      // First initialization
      await graphBuilder.initialize();
      
      // Reset mocks to check second call
      mockCreateGraphBuilder.mockClear();
      
      // Second initialization
      await graphBuilder.initialize();
      
      // Verify createGraphBuilder not called again
      expect(mockCreateGraphBuilder).not.toHaveBeenCalled();
    });
  });
  
  describe('Graph Building', () => {
    beforeEach(async () => {
      await graphBuilder.initialize();
    });
    
    it('should create input nodes correctly', () => {
      const inputTensor = new Tensor([2, 3], [1, 2, 3, 4, 5, 6], { dataType: 'float32' });
      const inputNode = graphBuilder.input('input1', inputTensor);
      
      expect(inputNode).toBeDefined();
      expect(inputNode.operationType).toBe('input');
      expect(inputNode.shape).toEqual([2, 3]);
      expect(inputNode.dataType).toBe('float32');
      expect(mockGraphBuilder.input).toHaveBeenCalledWith('input1', {
        type: 'float32',
        dimensions: [2, 3]
      });
    });
    
    it('should create constant nodes correctly', () => {
      const constTensor = new Tensor([2, 2], [1, 2, 3, 4], { dataType: 'float32' });
      const constNode = graphBuilder.constant(constTensor);
      
      expect(constNode).toBeDefined();
      expect(constNode.operationType).toBe('constant');
      expect(constNode.shape).toEqual([2, 2]);
      expect(constNode.dataType).toBe('float32');
      expect(mockGraphBuilder.constant).toHaveBeenCalled();
    });
    
    it('should set output nodes correctly', () => {
      const inputTensor = new Tensor([2, 3], [1, 2, 3, 4, 5, 6], { dataType: 'float32' });
      const inputNode = graphBuilder.input('input1', inputTensor);
      
      graphBuilder.output('output1', inputNode);
      
      // Test output by building graph
      return graphBuilder.buildAndCompile().then(() => {
        expect(mockGraphBuilder.build).toHaveBeenCalledWith({
          output1: mockInputOperand
        });
      });
    });
    
    it('should build and compile the graph correctly', async () => {
      const inputTensor = new Tensor([2, 3], [1, 2, 3, 4, 5, 6], { dataType: 'float32' });
      const inputNode = graphBuilder.input('input1', inputTensor);
      graphBuilder.output('output1', inputNode);
      
      const graph = await graphBuilder.buildAndCompile();
      
      expect(graph).toBe(mockGraph);
      expect(mockGraphBuilder.build).toHaveBeenCalledWith({
        output1: mockInputOperand
      });
    });
    
    it('should throw error when building graph without outputs', async () => {
      await expect(graphBuilder.buildAndCompile()).rejects.toThrow('Graph must have at least one output');
    });
  });
  
  describe('Graph Execution', () => {
    beforeEach(async () => {
      await graphBuilder.initialize();
    });
    
    it('should execute the graph correctly', async () => {
      // Create simple graph
      const inputTensor = new Tensor([2, 2], [1, 2, 3, 4], { dataType: 'float32' });
      const inputNode = graphBuilder.input('input1', inputTensor);
      graphBuilder.output('output1', inputNode);
      
      // Execute graph
      const results = await graphBuilder.execute({ 'input1': inputTensor });
      
      expect(results).toBeDefined();
      expect(results.output1).toBeDefined();
      expect(results.output1.shape).toEqual([2, 2]);
      expect(mockMLContext.compute).toHaveBeenCalled();
    });
    
    it('should execute with caching correctly', async () => {
      // Create simple graph
      const inputTensor = new Tensor([2, 2], [1, 2, 3, 4], { dataType: 'float32' });
      const inputNode = graphBuilder.input('input1', inputTensor);
      graphBuilder.output('output1', inputNode);
      
      // First execution should build the graph
      await graphBuilder.executeWithCache({ 'input1': inputTensor }, 'test_graph');
      
      // Reset build mock to check if it's called again
      mockGraphBuilder.build.mockClear();
      
      // Second execution should use cached graph
      await graphBuilder.executeWithCache({ 'input1': inputTensor }, 'test_graph');
      
      // Verify build not called again
      expect(mockGraphBuilder.build).not.toHaveBeenCalled();
    });
  });
  
  describe('Neural Network Operations', () => {
    let inputNode: any;
    let filterNode: any;
    let biasNode: any;
    let inputTensor: Tensor<number>;
    
    beforeEach(async () => {
      await graphBuilder.initialize();
      
      // Create common tensors and nodes
      inputTensor = new Tensor([1, 28, 28, 3], Array(1 * 28 * 28 * 3).fill(0.5), { dataType: 'float32' });
      inputNode = graphBuilder.input('input', inputTensor);
      
      // Filter for convolution: [3, 3, 3, 16] - [height, width, inChannels, outChannels]
      const filterTensor = new Tensor([3, 3, 3, 16], Array(3 * 3 * 3 * 16).fill(0.1), { dataType: 'float32' });
      filterNode = graphBuilder.constant(filterTensor);
      
      // Bias: [16]
      const biasTensor = new Tensor([16], Array(16).fill(0.01), { dataType: 'float32' });
      biasNode = graphBuilder.constant(biasTensor);
    });
    
    it('should create convolution correctly', () => {
      const convNode = graphBuilder.conv2d(inputNode, filterNode, {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        activation: 'relu'
      });
      
      expect(convNode).toBeDefined();
      expect(convNode.operationType).toBe('conv2d');
      expect(convNode.shape).toEqual([1, 28, 28, 16]); // Output shape with padding
      expect(mockGraphBuilder.conv2d).toHaveBeenCalled();
      expect(mockGraphBuilder.relu).toHaveBeenCalled();
    });
    
    it('should create pooling correctly', () => {
      const poolNode = graphBuilder.pool(inputNode, 'max', {
        windowDimensions: [2, 2],
        strides: [2, 2]
      });
      
      expect(poolNode).toBeDefined();
      expect(poolNode.operationType).toBe('maxPool');
      expect(poolNode.shape).toEqual([1, 14, 14, 3]); // Pooled output shape
      expect(mockGraphBuilder.maxPool2d).toHaveBeenCalled();
    });
    
    it('should create matrix multiplication correctly', () => {
      const matrix1 = new Tensor([2, 3], [1, 2, 3, 4, 5, 6], { dataType: 'float32' });
      const matrix2 = new Tensor([3, 2], [7, 8, 9, 10, 11, 12], { dataType: 'float32' });
      
      const m1Node = graphBuilder.input('m1', matrix1);
      const m2Node = graphBuilder.constant(matrix2);
      
      const matmulNode = graphBuilder.matmul(m1Node, m2Node);
      
      expect(matmulNode).toBeDefined();
      expect(matmulNode.operationType).toBe('matmul');
      expect(matmulNode.shape).toEqual([2, 2]); // Matmul output shape
      expect(mockGraphBuilder.matmul).toHaveBeenCalled();
    });
    
    it('should create fully connected layer correctly', () => {
      // Input tensor with shape [1, 10] (batch_size, features)
      const fcInputTensor = new Tensor([1, 10], Array(10).fill(0.5), { dataType: 'float32' });
      const fcInputNode = graphBuilder.input('fcInput', fcInputTensor);
      
      // Weights with shape [10, 5] (input_features, output_features)
      const weightsTensor = new Tensor([10, 5], Array(10 * 5).fill(0.1), { dataType: 'float32' });
      const weightsNode = graphBuilder.constant(weightsTensor);
      
      // Bias with shape [5] (output_features)
      const biasTensor = new Tensor([5], Array(5).fill(0.01), { dataType: 'float32' });
      const biasNode = graphBuilder.constant(biasTensor);
      
      const fcNode = graphBuilder.fullyConnected(fcInputNode, weightsNode, biasNode, 'relu');
      
      expect(fcNode).toBeDefined();
      expect(fcNode.operationType).toBe('fullyConnected');
      expect(fcNode.shape).toEqual([1, 5]); // FC output shape
      expect(mockGraphBuilder.matmul).toHaveBeenCalled();
      expect(mockGraphBuilder.add).toHaveBeenCalled();
      expect(mockGraphBuilder.relu).toHaveBeenCalled();
    });
    
    it('should create layer with weights and bias correctly', () => {
      // Input tensor with shape [1, 10] (batch_size, features)
      const fcInputTensor = new Tensor([1, 10], Array(10).fill(0.5), { dataType: 'float32' });
      const fcInputNode = graphBuilder.input('fcInput', fcInputTensor);
      
      // Weights with shape [10, 5] (input_features, output_features)
      const weightsTensor = new Tensor([10, 5], Array(10 * 5).fill(0.1), { dataType: 'float32' });
      
      // Bias with shape [5] (output_features)
      const biasTensor = new Tensor([5], Array(5).fill(0.01), { dataType: 'float32' });
      
      const layerNode = graphBuilder.layer(fcInputNode, weightsTensor, biasTensor, 'sigmoid');
      
      expect(layerNode).toBeDefined();
      expect(layerNode.operationType).toBe('fullyConnected');
      expect(layerNode.shape).toEqual([1, 5]); // Layer output shape
      expect(mockGraphBuilder.constant).toHaveBeenCalledTimes(2); // For weights and bias
      expect(mockGraphBuilder.matmul).toHaveBeenCalled();
      expect(mockGraphBuilder.add).toHaveBeenCalled();
      expect(mockGraphBuilder.sigmoid).toHaveBeenCalled();
    });
    
    it('should create sequential model correctly', () => {
      // Input tensor with shape [1, 10] (batch_size, features)
      const inputTensor = new Tensor([1, 10], Array(10).fill(0.5), { dataType: 'float32' });
      const inputNode = graphBuilder.input('seqInput', inputTensor);
      
      // Define layers
      const layers = [
        {
          weights: new Tensor([10, 20], Array(10 * 20).fill(0.1), { dataType: 'float32' }),
          bias: new Tensor([20], Array(20).fill(0.01), { dataType: 'float32' }),
          activation: 'relu' as const
        },
        {
          weights: new Tensor([20, 10], Array(20 * 10).fill(0.05), { dataType: 'float32' }),
          bias: new Tensor([10], Array(10).fill(0.005), { dataType: 'float32' }),
          activation: 'sigmoid' as const
        }
      ];
      
      const modelNode = graphBuilder.sequential(inputNode, layers);
      
      expect(modelNode).toBeDefined();
      expect(modelNode.shape).toEqual([1, 10]); // Final layer output shape
      expect(mockGraphBuilder.constant).toHaveBeenCalledTimes(4); // 2 weights + 2 biases
      expect(mockGraphBuilder.matmul).toHaveBeenCalledTimes(2);
      expect(mockGraphBuilder.add).toHaveBeenCalledTimes(2);
      expect(mockGraphBuilder.relu).toHaveBeenCalledTimes(1);
      expect(mockGraphBuilder.sigmoid).toHaveBeenCalledTimes(1);
    });
    
    it('should create batch normalization correctly', () => {
      // Create mean and variance tensors
      const meanTensor = new Tensor([3], [0, 0, 0], { dataType: 'float32' });
      const varianceTensor = new Tensor([3], [1, 1, 1], { dataType: 'float32' });
      const scaleTensor = new Tensor([3], [1, 1, 1], { dataType: 'float32' });
      const biasTensor = new Tensor([3], [0, 0, 0], { dataType: 'float32' });
      
      const meanNode = graphBuilder.constant(meanTensor);
      const varianceNode = graphBuilder.constant(varianceTensor);
      const scaleNode = graphBuilder.constant(scaleTensor);
      const biasNode = graphBuilder.constant(biasTensor);
      
      const bnNode = graphBuilder.batchNormalization(
        inputNode,
        meanNode,
        varianceNode,
        scaleNode,
        biasNode,
        1e-5
      );
      
      expect(bnNode).toBeDefined();
      expect(bnNode.operationType).toBe('batchNormalization');
      expect(bnNode.shape).toEqual([1, 28, 28, 3]); // Same as input shape
      expect(mockGraphBuilder.batchNormalization).toHaveBeenCalled();
    });
    
    it('should create residual block correctly', () => {
      // Input tensor with shape [1, 10] (batch_size, features)
      const inputTensor = new Tensor([1, 10], Array(10).fill(0.5), { dataType: 'float32' });
      const inputNode = graphBuilder.input('resInput', inputTensor);
      
      // Layer 1 weights and bias
      const fc1Weights = new Tensor([10, 10], Array(10 * 10).fill(0.1), { dataType: 'float32' });
      const fc1Bias = new Tensor([10], Array(10).fill(0.01), { dataType: 'float32' });
      
      // Layer 2 weights and bias
      const fc2Weights = new Tensor([10, 10], Array(10 * 10).fill(0.05), { dataType: 'float32' });
      const fc2Bias = new Tensor([10], Array(10).fill(0.005), { dataType: 'float32' });
      
      const resBlock = graphBuilder.residualBlock(
        inputNode,
        fc1Weights,
        fc1Bias,
        fc2Weights,
        fc2Bias,
        'relu'
      );
      
      expect(resBlock).toBeDefined();
      expect(resBlock.shape).toEqual([1, 10]); // Same as input shape
      expect(mockGraphBuilder.matmul).toHaveBeenCalledTimes(2);
      expect(mockGraphBuilder.add).toHaveBeenCalledTimes(3); // 2 bias adds + 1 residual
      expect(mockGraphBuilder.relu).toHaveBeenCalledTimes(2); // Once for first layer, once after residual
    });
  });
  
  describe('Resource Management', () => {
    beforeEach(async () => {
      await graphBuilder.initialize();
    });
    
    it('should reset the graph builder correctly', async () => {
      // Create some nodes
      const inputTensor = new Tensor([2, 2], [1, 2, 3, 4], { dataType: 'float32' });
      const inputNode = graphBuilder.input('input1', inputTensor);
      graphBuilder.output('output1', inputNode);
      
      // Reset the builder
      graphBuilder.reset();
      
      // Try to build - should fail because outputs were cleared
      await expect(graphBuilder.buildAndCompile()).rejects.toThrow('Graph must have at least one output');
    });
    
    it('should clear the cache correctly', async () => {
      // Create simple graph
      const inputTensor = new Tensor([2, 2], [1, 2, 3, 4], { dataType: 'float32' });
      const inputNode = graphBuilder.input('input1', inputTensor);
      graphBuilder.output('output1', inputNode);
      
      // First execution should build the graph
      await graphBuilder.executeWithCache({ 'input1': inputTensor }, 'test_graph');
      
      // Clear the cache
      graphBuilder.clearCache();
      
      // Second execution should build again
      mockGraphBuilder.build.mockClear();
      await graphBuilder.executeWithCache({ 'input1': inputTensor }, 'test_graph');
      
      // Verify build was called again
      expect(mockGraphBuilder.build).toHaveBeenCalled();
    });
    
    it('should dispose resources correctly', async () => {
      // Create some nodes
      const inputTensor = new Tensor([2, 2], [1, 2, 3, 4], { dataType: 'float32' });
      const inputNode = graphBuilder.input('input1', inputTensor);
      graphBuilder.output('output1', inputNode);
      
      // Execute to populate cache
      await graphBuilder.executeWithCache({ 'input1': inputTensor }, 'test_graph');
      
      // Dispose
      graphBuilder.dispose();
      
      // Verify initialized is false
      expect((graphBuilder as any).initialized).toBe(false);
      
      // Try to use - should throw
      expect(() => graphBuilder.input('test', inputTensor)).toThrow('WebNN graph builder not initialized');
    });
  });
});