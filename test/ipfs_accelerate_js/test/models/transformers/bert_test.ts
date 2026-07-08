/**
 * BERT model tests
 */

import { Bert, BertConfig, createBertModel } from '../../../src/model/transformers/bert';
import { WebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { HardwareBackend } from '../../../src/hardware/interfaces/hardware_backend';

// Mock hardware backend for testing
class MockHardwareBackend implements HardwareBackend {
  private initialized: boolean = false;
  private mockTensors = new Map<string, any>();
  private nextTensorId = 1;
  
  async initialize(): Promise<void> {
    this.initialized = true;
    return Promise.resolve();
  }
  
  isInitialized(): boolean {
    return this.initialized;
  }
  
  getBackendType(): string {
    return 'mock';
  }
  
  async createTensor(options: any): Promise<any> {
    const id = `tensor_${this.nextTensorId++}`;
    const tensor = {
      id,
      dimensions: options.dimensions,
      data: options.data,
      dtype: options.dtype,
      toArray: async () => {
        // Return zeros for simplicity
        const size = options.dimensions.reduce((a: number, b: number) => a * b, 1);
        if (options.dimensions.length === 1) {
          return new Array(size).fill(0);
        } else if (options.dimensions.length === 2) {
          return new Array(options.dimensions[0]).fill(0).map(() => 
            new Array(options.dimensions[1]).fill(0));
        } else if (options.dimensions.length === 3) {
          return new Array(options.dimensions[0]).fill(0).map(() => 
            new Array(options.dimensions[1]).fill(0).map(() => 
              new Array(options.dimensions[2]).fill(0)));
        }
        return new Array(size).fill(0);
      }
    };
    
    this.mockTensors.set(id, tensor);
    return Promise.resolve(tensor);
  }
  
  async releaseTensor(tensor: any): Promise<void> {
    if (tensor && tensor.id) {
      this.mockTensors.delete(tensor.id);
    }
    return Promise.resolve();
  }
  
  async gatherEmbedding(embeddings: any, indices: any): Promise<any> {
    // Return a tensor with the same shape as indices but last dimension from embeddings
    const lastDim = embeddings.dimensions[embeddings.dimensions.length - 1];
    const newDimensions = [...indices.dimensions, lastDim];
    
    return this.createTensor({
      dimensions: newDimensions,
      data: new Float32Array(newDimensions.reduce((a, b) => a * b, 1)),
      dtype: 'float32'
    });
  }
  
  async reshape(tensor: any, newDimensions: number[]): Promise<any> {
    return this.createTensor({
      dimensions: newDimensions,
      data: new Float32Array(newDimensions.reduce((a, b) => a * b, 1)),
      dtype: tensor.dtype
    });
  }
  
  async transpose(tensor: any, permutation: number[]): Promise<any> {
    // Calculate new dimensions based on permutation
    const newDimensions = permutation.map(i => tensor.dimensions[i]);
    
    return this.createTensor({
      dimensions: newDimensions,
      data: new Float32Array(newDimensions.reduce((a, b) => a * b, 1)),
      dtype: tensor.dtype
    });
  }
  
  async mul(tensor: any, scalar: any): Promise<any> {
    return this.createTensor({
      dimensions: tensor.dimensions,
      data: new Float32Array(tensor.dimensions.reduce((a, b) => a * b, 1)),
      dtype: tensor.dtype
    });
  }
  
  async sub(tensor1: any, tensor2: any): Promise<any> {
    return this.createTensor({
      dimensions: tensor1.dimensions,
      data: new Float32Array(tensor1.dimensions.reduce((a, b) => a * b, 1)),
      dtype: tensor1.dtype
    });
  }
  
  async slice(tensor: any, starts: number[], sizes: number[]): Promise<any> {
    return this.createTensor({
      dimensions: sizes,
      data: new Float32Array(sizes.reduce((a, b) => a * b, 1)),
      dtype: tensor.dtype
    });
  }
  
  async tanh(tensor: any): Promise<any> {
    return this.createTensor({
      dimensions: tensor.dimensions,
      data: new Float32Array(tensor.dimensions.reduce((a, b) => a * b, 1)),
      dtype: tensor.dtype
    });
  }
  
  async dispose(): Promise<void> {
    this.mockTensors.clear();
    this.initialized = false;
    return Promise.resolve();
  }
}

describe('BERT Model', () => {
  let mockHardware: MockHardwareBackend;
  
  beforeEach(() => {
    mockHardware = new MockHardwareBackend();
  });
  
  afterEach(async () => {
    await mockHardware.dispose();
  });
  
  test('should create BERT model with default config', () => {
    const bert = createBertModel(mockHardware);
    expect(bert).toBeInstanceOf(Bert);
  });
  
  test('should create BERT model with custom config', () => {
    const config: Partial<BertConfig> = {
      modelId: 'bert-custom',
      vocabSize: 10000,
      hiddenSize: 512,
      numLayers: 8,
      numHeads: 8
    };
    
    const bert = createBertModel(mockHardware, config);
    expect(bert).toBeInstanceOf(Bert);
  });
  
  test('should initialize model successfully', async () => {
    const bert = createBertModel(mockHardware);
    await expect(bert.initialize()).resolves.not.toThrow();
  });
  
  test('should tokenize input text', async () => {
    const bert = createBertModel(mockHardware);
    await bert.initialize();
    
    const input = "Hello world";
    const tokenized = await bert.tokenize(input);
    
    expect(tokenized).toBeDefined();
    expect(tokenized.inputIds).toBeDefined();
    expect(tokenized.inputIds.length).toBeGreaterThan(0);
    expect(tokenized.attentionMask).toBeDefined();
    expect(tokenized.tokenTypeIds).toBeDefined();
  });
  
  test('should process tokenized input', async () => {
    const bert = createBertModel(mockHardware);
    await bert.initialize();
    
    const input = "Test input";
    const tokenized = await bert.tokenize(input);
    const output = await bert.process(tokenized);
    
    expect(output).toBeDefined();
    expect(output.model).toBe('bert-base-uncased');
    expect(output.backend).toBe('mock');
    expect(output.lastHiddenState).toBeDefined();
    expect(output.pooledOutput).toBeDefined();
  });
  
  test('should create and retrieve shared tensors', async () => {
    const bert = createBertModel(mockHardware);
    await bert.initialize();
    
    // Mock sequence output tensor
    const sequenceOutput = await mockHardware.createTensor({
      dimensions: [1, 10, 768],
      data: new Float32Array(1 * 10 * 768),
      dtype: 'float32'
    });
    
    // Create shared tensor
    const sharedTensor = await bert.createSharedTensor(sequenceOutput, 'text_embedding');
    expect(sharedTensor).toBeDefined();
    
    // Retrieve shared tensor
    const retrievedTensor = bert.getSharedTensor('text_embedding');
    expect(retrievedTensor).toBeDefined();
    expect(retrievedTensor).toBe(sharedTensor);
  });
  
  test('should dispose resources properly', async () => {
    const bert = createBertModel(mockHardware);
    await bert.initialize();
    
    // Process something to create internal tensors
    const input = "Test input";
    const tokenized = await bert.tokenize(input);
    await bert.process(tokenized);
    
    // Dispose model
    await expect(bert.dispose()).resolves.not.toThrow();
  });
});