/**
 * Integration Tests for Hardware Abstracted CLIP model
 */

import { createHardwareAbstractedCLIP, HardwareAbstractedCLIPConfig } from '../../src/model/hardware/clip';
import { createCPUBackend } from '../../src/hardware/cpu/backend';
import { createInMemoryStorageManager } from '../../src/storage/in_memory_storage_manager';
import { mockWebGPUBackend } from '../utils/mock_webgpu_backend';
import { mockWebNNBackend } from '../utils/mock_webnn_backend';
import { HardwareBackend } from '../../src/hardware/interfaces/hardware_backend';

describe('Hardware Abstracted CLIP', () => {
  let cpuBackend: HardwareBackend;
  let storageManager: any;
  let mockImageData: any;
  
  beforeAll(async () => {
    // Set up CPU backend for testing
    cpuBackend = await createCPUBackend();
    await cpuBackend.initialize();
    
    // Create storage manager
    storageManager = createInMemoryStorageManager();
    await storageManager.initialize();
    
    // Mock image data
    mockImageData = {
      imageData: new Uint8Array(224 * 224 * 4),
      width: 224,
      height: 224
    };
    
    // Fill mock image data with random values
    for (let i = 0; i < mockImageData.imageData.length; i++) {
      mockImageData.imageData[i] = Math.floor(Math.random() * 256);
    }
  });
  
  afterAll(async () => {
    await cpuBackend.dispose();
  });
  
  test('should initialize with CPU backend', async () => {
    // Create CLIP model
    const clip = createHardwareAbstractedCLIP({
      modelId: 'openai/clip-vit-base-patch32',
      imageSize: 224,
      taskType: 'similarity',
      allowFallback: true,
      collectMetrics: true
    }, storageManager);
    
    // Mock the initializeHardware method to use CPU backend
    (clip as any).initializeHardware = jest.fn().mockImplementation(async function(this: any) {
      this.hardware = cpuBackend;
      this.selectedBackend = 'cpu';
      this.availableBackends = ['cpu'];
    });
    
    // Mock the CLIPImplementation class
    (clip as any).modelImpl = {
      initialize: jest.fn().mockResolvedValue(undefined),
      process: jest.fn().mockImplementation(async (input: any) => {
        // Mock behavior
        if (input.image) {
          return {
            imageEmbeddings: new Float32Array(512).fill(0.1),
            model: 'openai/clip-vit-base-patch32',
            backend: 'cpu'
          };
        } else if (input.text) {
          return {
            textEmbeddings: new Float32Array(512).fill(0.2),
            model: 'openai/clip-vit-base-patch32',
            backend: 'cpu'
          };
        } else if (input.image && input.text) {
          return {
            imageEmbeddings: new Float32Array(512).fill(0.1),
            textEmbeddings: new Float32Array(512).fill(0.2),
            similarity: 0.75,
            model: 'openai/clip-vit-base-patch32',
            backend: 'cpu'
          };
        }
        
        return {};
      }),
      getSharedTensor: jest.fn().mockImplementation((type: string) => {
        if (type === 'vision_embedding') {
          return {
            tensor: {
              data: new Float32Array(512).fill(0.1)
            },
            type: 'vision_embedding',
            modelId: 'openai/clip-vit-base-patch32',
            backendType: 'cpu'
          };
        } else if (type === 'text_embedding') {
          return {
            tensor: {
              data: new Float32Array(512).fill(0.2)
            },
            type: 'text_embedding',
            modelId: 'openai/clip-vit-base-patch32',
            backendType: 'cpu'
          };
        }
        
        return null;
      }),
      dispose: jest.fn().mockResolvedValue(undefined)
    };
    
    // Initialize model
    await clip.initialize();
    
    // Check if model is initialized
    expect((clip as any).initialized).toBe(true);
    expect((clip as any).selectedBackend).toBe('cpu');
    
    // Clean up
    await clip.dispose();
  });
  
  test('should encode image', async () => {
    // Create CLIP model
    const clip = createHardwareAbstractedCLIP({
      modelId: 'openai/clip-vit-base-patch32',
      imageSize: 224,
      taskType: 'similarity',
      allowFallback: true,
      collectMetrics: true
    }, storageManager);
    
    // Mock the initializeHardware method to use CPU backend
    (clip as any).initializeHardware = jest.fn().mockImplementation(async function(this: any) {
      this.hardware = cpuBackend;
      this.selectedBackend = 'cpu';
      this.availableBackends = ['cpu'];
    });
    
    // Mock the CLIPImplementation class
    (clip as any).modelImpl = {
      initialize: jest.fn().mockResolvedValue(undefined),
      process: jest.fn().mockImplementation(async (input: any) => {
        if (input.image) {
          return {
            imageEmbeddings: new Float32Array(512).fill(0.1),
            model: 'openai/clip-vit-base-patch32',
            backend: 'cpu'
          };
        }
        
        return {};
      }),
      getSharedTensor: jest.fn().mockReturnValue({
        tensor: { data: new Float32Array(512).fill(0.1) },
        type: 'vision_embedding',
        modelId: 'openai/clip-vit-base-patch32',
        backendType: 'cpu'
      }),
      dispose: jest.fn().mockResolvedValue(undefined)
    };
    
    // Initialize model
    await clip.initialize();
    
    // Encode image
    const embeddings = await clip.encodeImage(mockImageData);
    
    // Check embeddings
    expect(embeddings).toBeInstanceOf(Float32Array);
    expect(embeddings.length).toBe(512);
    expect((clip as any).modelImpl.process).toHaveBeenCalledWith({ image: mockImageData });
    
    // Check metrics
    const metrics = clip.getPerformanceMetrics();
    expect(metrics).toBeTruthy();
    expect(Object.keys(metrics).length).toBeGreaterThan(0);
    
    // Clean up
    await clip.dispose();
  });
  
  test('should encode text', async () => {
    // Create CLIP model
    const clip = createHardwareAbstractedCLIP({
      modelId: 'openai/clip-vit-base-patch32',
      imageSize: 224,
      taskType: 'similarity',
      allowFallback: true,
      collectMetrics: true
    }, storageManager);
    
    // Mock the initializeHardware method to use CPU backend
    (clip as any).initializeHardware = jest.fn().mockImplementation(async function(this: any) {
      this.hardware = cpuBackend;
      this.selectedBackend = 'cpu';
      this.availableBackends = ['cpu'];
    });
    
    // Mock the CLIPImplementation class
    (clip as any).modelImpl = {
      initialize: jest.fn().mockResolvedValue(undefined),
      process: jest.fn().mockImplementation(async (input: any) => {
        if (input.text) {
          return {
            textEmbeddings: new Float32Array(512).fill(0.2),
            model: 'openai/clip-vit-base-patch32',
            backend: 'cpu'
          };
        }
        
        return {};
      }),
      getSharedTensor: jest.fn().mockReturnValue({
        tensor: { data: new Float32Array(512).fill(0.2) },
        type: 'text_embedding',
        modelId: 'openai/clip-vit-base-patch32',
        backendType: 'cpu'
      }),
      dispose: jest.fn().mockResolvedValue(undefined)
    };
    
    // Initialize model
    await clip.initialize();
    
    // Encode text
    const text = 'a photo of a dog';
    const embeddings = await clip.encodeText(text);
    
    // Check embeddings
    expect(embeddings).toBeInstanceOf(Float32Array);
    expect(embeddings.length).toBe(512);
    expect((clip as any).modelImpl.process).toHaveBeenCalledWith({ text: { text } });
    
    // Check metrics
    const metrics = clip.getPerformanceMetrics();
    expect(metrics).toBeTruthy();
    expect(Object.keys(metrics).length).toBeGreaterThan(0);
    
    // Clean up
    await clip.dispose();
  });
  
  test('should compute similarity', async () => {
    // Create CLIP model
    const clip = createHardwareAbstractedCLIP({
      modelId: 'openai/clip-vit-base-patch32',
      imageSize: 224,
      taskType: 'similarity',
      allowFallback: true,
      collectMetrics: true
    }, storageManager);
    
    // Mock the initializeHardware method to use CPU backend
    (clip as any).initializeHardware = jest.fn().mockImplementation(async function(this: any) {
      this.hardware = cpuBackend;
      this.selectedBackend = 'cpu';
      this.availableBackends = ['cpu'];
    });
    
    // Mock the CLIPImplementation class
    (clip as any).modelImpl = {
      initialize: jest.fn().mockResolvedValue(undefined),
      process: jest.fn().mockImplementation(async (input: any) => {
        if (input.image && input.text) {
          return {
            imageEmbeddings: new Float32Array(512).fill(0.1),
            textEmbeddings: new Float32Array(512).fill(0.2),
            similarity: 0.75,
            model: 'openai/clip-vit-base-patch32',
            backend: 'cpu'
          };
        }
        
        return {};
      }),
      getSharedTensor: jest.fn().mockReturnValue(null),
      dispose: jest.fn().mockResolvedValue(undefined)
    };
    
    // Initialize model
    await clip.initialize();
    
    // Compute similarity
    const text = 'a photo of a dog';
    const similarity = await clip.computeSimilarity(mockImageData, text);
    
    // Check similarity
    expect(similarity).toBe(0.75);
    expect((clip as any).modelImpl.process).toHaveBeenCalledWith({
      image: mockImageData,
      text: { text }
    });
    
    // Check metrics
    const metrics = clip.getPerformanceMetrics();
    expect(metrics).toBeTruthy();
    expect(Object.keys(metrics).length).toBeGreaterThan(0);
    
    // Clean up
    await clip.dispose();
  });
  
  test('should classify image', async () => {
    // Create CLIP model
    const clip = createHardwareAbstractedCLIP({
      modelId: 'openai/clip-vit-base-patch32',
      imageSize: 224,
      taskType: 'zero_shot_classification',
      allowFallback: true,
      collectMetrics: true
    }, storageManager);
    
    // Mock the initializeHardware method to use CPU backend
    (clip as any).initializeHardware = jest.fn().mockImplementation(async function(this: any) {
      this.hardware = cpuBackend;
      this.selectedBackend = 'cpu';
      this.availableBackends = ['cpu'];
    });
    
    // Mock the CLIPImplementation class with a stateful handler to return different values for different classes
    const processHandler = jest.fn().mockImplementation(async (input: any) => {
      if (input.image && !input.text) {
        // Image-only process call
        return {
          imageEmbeddings: new Float32Array(512).fill(0.1),
          model: 'openai/clip-vit-base-patch32',
          backend: 'cpu'
        };
      } else if (input.text && !input.image) {
        // Text-only process call
        return {
          textEmbeddings: new Float32Array(512).fill(0.2),
          model: 'openai/clip-vit-base-patch32',
          backend: 'cpu'
        };
      } else if (input.image && input.text) {
        // Combined process call
        const text = input.text.text;
        let similarity = 0.5; // Default
        
        if (text.includes('dog')) {
          similarity = 0.9;
        } else if (text.includes('cat')) {
          similarity = 0.7;
        } else if (text.includes('car')) {
          similarity = 0.3;
        }
        
        return {
          imageEmbeddings: new Float32Array(512).fill(0.1),
          textEmbeddings: new Float32Array(512).fill(0.2),
          similarity,
          model: 'openai/clip-vit-base-patch32',
          backend: 'cpu'
        };
      }
      
      return {};
    });
    
    (clip as any).modelImpl = {
      initialize: jest.fn().mockResolvedValue(undefined),
      process: processHandler,
      getSharedTensor: jest.fn().mockReturnValue(null),
      dispose: jest.fn().mockResolvedValue(undefined)
    };
    
    // Initialize model
    await clip.initialize();
    
    // Classify image
    const classes = ['dog', 'cat', 'car'];
    const classifications = await clip.classifyImage(mockImageData, classes);
    
    // Check classifications
    expect(classifications).toBeInstanceOf(Array);
    expect(classifications.length).toBe(3);
    expect(classifications[0].label).toBe('dog');
    expect(classifications[0].score).toBe(0.9);
    expect(classifications[1].label).toBe('cat');
    expect(classifications[1].score).toBe(0.7);
    expect(classifications[2].label).toBe('car');
    expect(classifications[2].score).toBe(0.3);
    
    // Check metrics
    const metrics = clip.getPerformanceMetrics();
    expect(metrics).toBeTruthy();
    expect(Object.keys(metrics).length).toBeGreaterThan(0);
    
    // Clean up
    await clip.dispose();
  });
  
  test('should handle backend fallback', async () => {
    // Create CLIP model with mock WebGPU preference that will fail
    const clip = createHardwareAbstractedCLIP({
      modelId: 'openai/clip-vit-base-patch32',
      imageSize: 224,
      taskType: 'similarity',
      backendPreference: ['webgpu', 'cpu'], // WebGPU will fail, should fall back to CPU
      allowFallback: true,
      collectMetrics: true
    }, storageManager);
    
    // Store original method to restore later
    const originalInitializeHardware = (clip as any).initializeHardware;
    
    // Mock the initializeHardware method to simulate WebGPU failure and CPU fallback
    (clip as any).initializeHardware = jest.fn().mockImplementation(async function(this: any) {
      // Simulate trying WebGPU and failing
      try {
        throw new Error('WebGPU not available');
      } catch (error) {
        if (this.config.allowFallback) {
          // Fallback to CPU
          this.hardware = cpuBackend;
          this.selectedBackend = 'cpu';
          this.availableBackends = ['cpu'];
          console.log('Falling back to CPU backend');
        } else {
          throw error;
        }
      }
    });
    
    // Mock the CLIPImplementation class
    (clip as any).modelImpl = {
      initialize: jest.fn().mockResolvedValue(undefined),
      process: jest.fn().mockImplementation(async (input: any) => {
        if (input.image) {
          return {
            imageEmbeddings: new Float32Array(512).fill(0.1),
            model: 'openai/clip-vit-base-patch32',
            backend: 'cpu'
          };
        }
        
        return {};
      }),
      getSharedTensor: jest.fn().mockReturnValue(null),
      dispose: jest.fn().mockResolvedValue(undefined)
    };
    
    // Initialize model
    await clip.initialize();
    
    // Check if model is initialized with CPU fallback
    expect((clip as any).initialized).toBe(true);
    expect((clip as any).selectedBackend).toBe('cpu');
    
    // Test fallback behavior by encoding an image
    const embeddings = await clip.encodeImage(mockImageData);
    expect(embeddings).toBeInstanceOf(Float32Array);
    
    // Clean up
    await clip.dispose();
    
    // Restore original method
    (clip as any).initializeHardware = originalInitializeHardware;
  });
  
  test('should expose proper metrics and model info', async () => {
    // Create CLIP model
    const clip = createHardwareAbstractedCLIP({
      modelId: 'openai/clip-vit-base-patch32',
      imageSize: 224,
      taskType: 'similarity',
      allowFallback: true,
      collectMetrics: true
    }, storageManager);
    
    // Mock the initializeHardware method to use CPU backend
    (clip as any).initializeHardware = jest.fn().mockImplementation(async function(this: any) {
      this.hardware = cpuBackend;
      this.selectedBackend = 'cpu';
      this.availableBackends = ['cpu'];
      
      // Mock backend metrics
      this.backendMetrics = {
        type: 'cpu',
        capabilities: { tensorOps: true },
        isAvailable: true
      };
    });
    
    // Mock the CLIPImplementation class
    (clip as any).modelImpl = {
      initialize: jest.fn().mockResolvedValue(undefined),
      process: jest.fn().mockResolvedValue({
        imageEmbeddings: new Float32Array(512).fill(0.1)
      }),
      getSharedTensor: jest.fn().mockReturnValue(null),
      dispose: jest.fn().mockResolvedValue(undefined)
    };
    
    // Initialize model
    await clip.initialize();
    
    // Add some metrics
    (clip as any).updateMetric('initialization', 100);
    (clip as any).updateMetric('inference', 200);
    (clip as any).updateMetric('inference', 300);
    
    // Get metrics
    const metrics = clip.getPerformanceMetrics();
    expect(metrics).toBeTruthy();
    expect(metrics.initialization).toBeDefined();
    expect(metrics.initialization.avg).toBe(100);
    expect(metrics.inference).toBeDefined();
    expect(metrics.inference.avg).toBe(250); // Average of 200 and 300
    expect(metrics.inference.min).toBe(200);
    expect(metrics.inference.max).toBe(300);
    expect(metrics.inference.count).toBe(2);
    
    // Get backend metrics
    const backendMetrics = clip.getBackendMetrics();
    expect(backendMetrics).toBeTruthy();
    expect(backendMetrics.type).toBe('cpu');
    expect(backendMetrics.capabilities).toBeDefined();
    expect(backendMetrics.isAvailable).toBe(true);
    
    // Get model info
    const modelInfo = clip.getModelInfo();
    expect(modelInfo).toBeTruthy();
    expect(modelInfo.modelId).toBe('openai/clip-vit-base-patch32');
    expect(modelInfo.modelType).toBe('CLIP');
    expect(modelInfo.taskType).toBe('similarity');
    expect(modelInfo.selectedBackend).toBe('cpu');
    expect(modelInfo.imageSize).toBe(224);
    
    // Reset metrics
    clip.resetMetrics();
    const resetMetrics = clip.getPerformanceMetrics();
    expect(Object.keys(resetMetrics).length).toBe(0);
    
    // Clean up
    await clip.dispose();
  });
  
  test('should getSharedTensor from model', async () => {
    // Create CLIP model
    const clip = createHardwareAbstractedCLIP({
      modelId: 'openai/clip-vit-base-patch32',
      imageSize: 224,
      taskType: 'similarity',
      allowFallback: true,
      collectMetrics: true
    }, storageManager);
    
    // Mock the initializeHardware method to use CPU backend
    (clip as any).initializeHardware = jest.fn().mockImplementation(async function(this: any) {
      this.hardware = cpuBackend;
      this.selectedBackend = 'cpu';
      this.availableBackends = ['cpu'];
    });
    
    // Mock the CLIPImplementation class with getSharedTensor
    const mockSharedTensor = {
      tensor: { data: new Float32Array(512).fill(0.1) },
      type: 'vision_embedding',
      modelId: 'openai/clip-vit-base-patch32',
      backendType: 'cpu',
      release: jest.fn().mockResolvedValue(undefined)
    };
    
    (clip as any).modelImpl = {
      initialize: jest.fn().mockResolvedValue(undefined),
      process: jest.fn().mockResolvedValue({
        imageEmbeddings: new Float32Array(512).fill(0.1)
      }),
      getSharedTensor: jest.fn().mockImplementation((type: string) => {
        if (type === 'vision_embedding') {
          return mockSharedTensor;
        }
        return null;
      }),
      dispose: jest.fn().mockResolvedValue(undefined)
    };
    
    // Initialize model
    await clip.initialize();
    
    // Get shared tensor
    const sharedTensor = clip.getSharedTensor('vision_embedding');
    expect(sharedTensor).toBe(mockSharedTensor);
    
    // Try to get non-existent shared tensor
    const nullTensor = clip.getSharedTensor('non_existent');
    expect(nullTensor).toBeNull();
    
    // Clean up
    await clip.dispose();
  });
  
  test('should compare backends', async () => {
    // Create CLIP model
    const clip = createHardwareAbstractedCLIP({
      modelId: 'openai/clip-vit-base-patch32',
      imageSize: 224,
      taskType: 'similarity',
      allowFallback: true,
      collectMetrics: true
    }, storageManager);
    
    // Mock the initializeHardware method to use CPU backend
    (clip as any).initializeHardware = jest.fn().mockImplementation(async function(this: any) {
      this.hardware = cpuBackend;
      this.selectedBackend = 'cpu';
      this.availableBackends = ['cpu', 'webgpu', 'webnn'];
    });
    
    // Mock getAvailableBackends method to return consistent results
    (clip as any).getAvailableBackends = jest.fn().mockReturnValue(['cpu', 'webgpu', 'webnn']);
    
    // Mock the createOptimalBackend function to return a mock backend
    const mockCreateOptimalBackend = jest.fn().mockImplementation(async (options: any) => {
      const forceBackend = options.forceBackend;
      
      if (forceBackend === 'webgpu') {
        return mockWebGPUBackend();
      } else if (forceBackend === 'webnn') {
        return mockWebNNBackend();
      } else {
        return cpuBackend;
      }
    });
    
    // Apply the mock to the original function
    const originalCreateOptimalBackend = require('../../src/hardware/index').createOptimalBackend;
    require('../../src/hardware/index').createOptimalBackend = mockCreateOptimalBackend;
    
    // Mock CLIPImplementation class
    (clip as any).modelImpl = {
      initialize: jest.fn().mockResolvedValue(undefined),
      process: jest.fn().mockResolvedValue({
        similarity: 0.75
      }),
      getSharedTensor: jest.fn().mockReturnValue(null),
      dispose: jest.fn().mockResolvedValue(undefined)
    };
    
    // Initialize model
    await clip.initialize();
    
    // Mock createOptimalBackend behavior for different backends
    const mockCPUTime = 500;
    const mockWebGPUTime = 100;
    const mockWebNNTime = 300;
    
    // Create mock implementation for CLIPImplementation class
    const mockCLIPImplementation = jest.fn().mockImplementation(() => ({
      initialize: jest.fn().mockResolvedValue(undefined),
      process: jest.fn().mockImplementation(async () => {
        // Simulate different processing times for different backends
        // We're assuming different backends would have different performance
        if ((clip as any).selectedBackend === 'webgpu') {
          return new Promise(resolve => {
            setTimeout(() => {
              resolve({ similarity: 0.75 });
            }, mockWebGPUTime / 10); // Simulate WebGPU being fastest
          });
        } else if ((clip as any).selectedBackend === 'webnn') {
          return new Promise(resolve => {
            setTimeout(() => {
              resolve({ similarity: 0.75 });
            }, mockWebNNTime / 10); // Simulate WebNN being in the middle
          });
        } else {
          return new Promise(resolve => {
            setTimeout(() => {
              resolve({ similarity: 0.75 });
            }, mockCPUTime / 10); // Simulate CPU being slowest
          });
        }
      }),
      dispose: jest.fn().mockResolvedValue(undefined)
    }));
    
    // Mock the CLIPImplementation class constructor
    const originalCLIPImplementation = require('../../src/model/vision/clip').Clip;
    require('../../src/model/vision/clip').Clip = mockCLIPImplementation;
    
    // Mock performance.now() to get consistent results
    const originalPerformanceNow = performance.now;
    let mockTime = 0;
    performance.now = jest.fn().mockImplementation(() => {
      mockTime += 100;
      return mockTime;
    });
    
    // Run backend comparison
    const text = 'a photo of a dog';
    const comparisonResults = await clip.compareBackends(mockImageData, text);
    
    // Check comparison results - mocked values
    expect(comparisonResults).toBeTruthy();
    expect(Object.keys(comparisonResults)).toEqual(['cpu', 'webgpu', 'webnn']);
    
    // Restore mocks
    require('../../src/hardware/index').createOptimalBackend = originalCreateOptimalBackend;
    require('../../src/model/vision/clip').Clip = originalCLIPImplementation;
    performance.now = originalPerformanceNow;
    
    // Clean up
    await clip.dispose();
  });
});