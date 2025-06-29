/**
 * Tests for the Vision Transformer (ViT) model implementation
 */

import { ViT, createVitModel, ViTConfig, ViTInput } from '../../../src/model/vision/vit';
import { createMockHardwareBackend } from '../../utils/mock_hardware';

// Mock image data (3x3 RGB image)
const createMockImageData = (size: number = 224): Uint8Array => {
  // Create a checkerboard pattern for testing
  const data = new Uint8Array(size * size * 3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const isWhite = (Math.floor(x / 16) + Math.floor(y / 16)) % 2 === 0;
      const idx = (y * size + x) * 3;
      // RGB values
      data[idx] = isWhite ? 255 : 0;     // R
      data[idx + 1] = isWhite ? 255 : 0; // G
      data[idx + 2] = isWhite ? 255 : 0; // B
    }
  }
  return data;
};

describe('ViT Model', () => {
  let mockHardware: any;
  let model: ViT;
  let mockConfig: Partial<ViTConfig>;
  
  beforeEach(async () => {
    // Create mock hardware backend
    mockHardware = createMockHardwareBackend();
    await mockHardware.initialize();
    
    // Configure model
    mockConfig = {
      modelId: 'google/vit-base-patch16-224',
      imageSize: 224,
      patchSize: 16,
      hiddenSize: 768,
      numLayers: 2, // Reduced for testing
      numHeads: 12,
      intermediateSize: 3072,
      numClasses: 1000,
      backendPreference: ['cpu'],
      useOptimizedOps: false
    };
    
    // Create model
    model = createVitModel(mockHardware, mockConfig);
  });
  
  afterEach(async () => {
    // Clean up
    if (model) {
      await model.dispose();
    }
  });
  
  test('Model initialization', async () => {
    // Initialize the model
    await model.initialize();
    
    // Check that model is initialized
    expect(mockHardware.isInitialized()).toBe(true);
    expect(mockHardware.createTensor).toHaveBeenCalled();
  });
  
  test('Image preprocessing', async () => {
    // Initialize the model
    await model.initialize();
    
    // Create mock image input
    const imageData = createMockImageData();
    const input: ViTInput = {
      imageData,
      width: 224,
      height: 224
    };
    
    // Process image
    const output = await model.process(input);
    
    // Check basic properties of output
    expect(output).toBeDefined();
    expect(output.model).toBe(mockConfig.modelId);
    expect(output.backend).toBe(mockHardware.getBackendType());
    expect(output.logits).toBeDefined();
    expect(output.probabilities).toBeDefined();
    expect(output.classId).toBeDefined();
    
    // Check that probabilities sum to approximately 1
    const sum = output.probabilities.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 1);
  });
  
  test('Shared tensor creation', async () => {
    // Initialize the model
    await model.initialize();
    
    // Create mock image input
    const imageData = createMockImageData();
    const input: ViTInput = {
      imageData,
      width: 224,
      height: 224
    };
    
    // Process image
    await model.process(input);
    
    // Get shared tensor
    const sharedTensor = model.getSharedTensor('vision_embedding');
    
    // Check shared tensor properties
    expect(sharedTensor).toBeDefined();
    if (sharedTensor) {
      expect(sharedTensor.type).toBe('vision_embedding');
      expect(sharedTensor.model).toBe(mockConfig.modelId);
      expect(sharedTensor.backend).toBe(mockHardware.getBackendType());
    }
  });
  
  test('Model disposal', async () => {
    // Initialize the model
    await model.initialize();
    
    // Dispose the model
    await model.dispose();
    
    // Check that resources were released
    expect(mockHardware.releaseTensor).toHaveBeenCalled();
  });
});