/**
 * Tests for WebGPU Matrix Operations
 * 
 * These tests validate the different matrix multiplication strategies
 * and browser-specific optimizations.
 */

import { WebGPUMatrixMultiplication, MatrixMultiplyStrategy, BrowserOptimizedMatrixOperations } from './ipfs_accelerate_js_matrix_operations';
import { GPUBufferUtils } from './ipfs_accelerate_js_webgpu_backend';

// Mock GPU device and buffer utils for testing
const mockDevice = {
  createShaderModule: jest.fn(() => ({ })),
  createBindGroupLayout: jest.fn(() => ({ })),
  createPipelineLayout: jest.fn(() => ({ })),
  createComputePipelineAsync: jest.fn(async () => ({ })),
  createBuffer: jest.fn(() => ({
    destroy: jest.fn()
  })),
  createCommandEncoder: jest.fn(() => ({
    beginComputePass: jest.fn(() => ({
      setPipeline: jest.fn(),
      setBindGroup: jest.fn(),
      dispatchWorkgroups: jest.fn(),
      end: jest.fn()
    })),
    copyBufferToBuffer: jest.fn(),
    finish: jest.fn(() => ({}))
  })),
  queue: {
    writeBuffer: jest.fn(),
    submit: jest.fn()
  },
  createBindGroup: jest.fn(() => ({}))
} as unknown as GPUDevice;

const mockBufferUtils = {
  createBuffer: jest.fn((data, usage, label) => ({
    destroy: jest.fn()
  }))
} as unknown as GPUBufferUtils;

// Helper to create matrices and validate multiplication results
function setupMatrixMultiplyTest(M: number, N: number, K: number) {
  // Create input matrices
  const matrixA = new Float32Array(M * K);
  const matrixB = new Float32Array(K * N);
  
  // Fill with test data
  for (let i = 0; i < M * K; i++) {
    matrixA[i] = i / (M * K);
  }
  
  for (let i = 0; i < K * N; i++) {
    matrixB[i] = i / (K * N);
  }
  
  // Create expected result using CPU
  const expected = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += matrixA[i * K + k] * matrixB[k * N + j];
      }
      expected[i * N + j] = sum;
    }
  }
  
  return { matrixA, matrixB, expected };
}

// Mock for browser environment features
const getBrowserInfo = () => ({
  browserType: 'chrome',
  browserVersion: '120.0.0',
  gpuVendor: 'nvidia'
});

describe('WebGPU Matrix Operations', () => {
  let matrixOps: WebGPUMatrixMultiplication;
  
  beforeEach(() => {
    jest.clearAllMocks();
    matrixOps = new WebGPUMatrixMultiplication(mockDevice, mockBufferUtils);
  });
  
  test('Should initialize compute pipelines correctly', async () => {
    await matrixOps.initialize();
    
    // Check that shader modules were created
    expect(mockDevice.createShaderModule).toHaveBeenCalledTimes(3);
    
    // Check that bind group layout was created
    expect(mockDevice.createBindGroupLayout).toHaveBeenCalledTimes(1);
    
    // Check that compute pipelines were created
    expect(mockDevice.createComputePipelineAsync).toHaveBeenCalledTimes(3);
  });
  
  test('Should perform matrix multiplication with correct buffer setup', async () => {
    // Mock stagingBuffer.mapAsync and getMappedRange
    const mockResult = new Float32Array(4);
    mockResult.fill(1.0);
    
    const stagingBuffer = {
      mapAsync: jest.fn().mockResolvedValue(undefined),
      getMappedRange: jest.fn().mockReturnValue(mockResult),
      unmap: jest.fn()
    };
    
    mockDevice.createBuffer = jest.fn((options) => {
      if (options.usage & GPUBufferUsage.MAP_READ) {
        return stagingBuffer;
      }
      return { destroy: jest.fn() };
    });
    
    const { matrixA, matrixB } = setupMatrixMultiplyTest(2, 2, 2);
    
    const result = await matrixOps.matmul(matrixA, matrixB, 2, 2, 2, MatrixMultiplyStrategy.SIMPLE);
    
    // Check buffer creation
    expect(mockBufferUtils.createBuffer).toHaveBeenCalledTimes(2);
    expect(mockDevice.createBuffer).toHaveBeenCalledTimes(3); // Output buffer, uniform buffer, staging buffer
    
    // Check buffer writes
    expect(mockDevice.queue.writeBuffer).toHaveBeenCalledTimes(1);
    
    // Check compute pass dispatch
    const commandEncoder = mockDevice.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    expect(passEncoder.dispatchWorkgroups).toHaveBeenCalledWith(1, 1);
    
    // Check result handling
    expect(stagingBuffer.mapAsync).toHaveBeenCalled();
    expect(stagingBuffer.getMappedRange).toHaveBeenCalled();
    expect(stagingBuffer.unmap).toHaveBeenCalled();
    
    // Check buffer cleanup
    expect(result).toHaveLength(4);
  });
  
  test('Should select correct strategy based on matrix size', () => {
    // Use the private method directly by type coercion
    const selectStrategy = (matrixOps as any).selectOptimalStrategy.bind(matrixOps);
    
    // Test small matrices
    expect(selectStrategy(32, 32, 32)).toBe(MatrixMultiplyStrategy.SIMPLE);
    
    // Test medium matrices
    expect(selectStrategy(128, 128, 128)).toBe(MatrixMultiplyStrategy.TILED);
    
    // Test large matrices
    expect(selectStrategy(1024, 1024, 1024)).toBe(MatrixMultiplyStrategy.MICRO_TILED);
    
    // Test mixed dimensions
    expect(selectStrategy(32, 1024, 32)).toBe(MatrixMultiplyStrategy.MICRO_TILED);
  });
});

describe('Browser Optimized Matrix Operations', () => {
  let browserOps: BrowserOptimizedMatrixOperations;
  
  beforeEach(() => {
    jest.clearAllMocks();
    browserOps = new BrowserOptimizedMatrixOperations(
      mockDevice, 
      mockBufferUtils,
      getBrowserInfo()
    );
  });
  
  test('Should initialize matrix operations correctly', async () => {
    await browserOps.initialize();
    
    // Check that shader modules were created
    expect(mockDevice.createShaderModule).toHaveBeenCalledTimes(3);
    
    // Check that bind group layout was created
    expect(mockDevice.createBindGroupLayout).toHaveBeenCalledTimes(1);
    
    // Check that compute pipelines were created
    expect(mockDevice.createComputePipelineAsync).toHaveBeenCalledTimes(3);
  });
  
  test('Should select Chrome-specific optimizations for large matrices', async () => {
    // Mock the matrixOps.matmul method
    const mockMatmul = jest.fn().mockResolvedValue(new Float32Array(4));
    (browserOps as any).matrixOps.matmul = mockMatmul;
    
    const { matrixA, matrixB } = setupMatrixMultiplyTest(2, 2, 2);
    
    await browserOps.matmul(matrixA, matrixB, 512, 512, 512);
    
    // Chrome with large matrices should use micro-tiled strategy
    expect(mockMatmul).toHaveBeenCalledWith(
      matrixA, matrixB, 512, 512, 512, MatrixMultiplyStrategy.MICRO_TILED
    );
  });
  
  test('Should use browser-specific strategy based on matrix size', async () => {
    // Mock the matrixOps.matmul method
    const mockMatmul = jest.fn().mockResolvedValue(new Float32Array(4));
    (browserOps as any).matrixOps.matmul = mockMatmul;
    (browserOps as any).browserType = 'firefox';
    
    const { matrixA, matrixB } = setupMatrixMultiplyTest(2, 2, 2);
    
    await browserOps.matmul(matrixA, matrixB, 256, 256, 256);
    
    // Firefox with medium matrices should use tiled strategy
    expect(mockMatmul).toHaveBeenCalledWith(
      matrixA, matrixB, 256, 256, 256, MatrixMultiplyStrategy.TILED
    );
  });
});

describe('Strategy Selection Logic', () => {
  test('Should select strategies based on browser type', () => {
    // Create instances with different browser types
    const chromeOps = new BrowserOptimizedMatrixOperations(
      mockDevice, mockBufferUtils, 
      { browserType: 'chrome', browserVersion: '120.0.0', gpuVendor: 'nvidia' }
    );
    
    const firefoxOps = new BrowserOptimizedMatrixOperations(
      mockDevice, mockBufferUtils, 
      { browserType: 'firefox', browserVersion: '120.0.0', gpuVendor: 'nvidia' }
    );
    
    const safariOps = new BrowserOptimizedMatrixOperations(
      mockDevice, mockBufferUtils, 
      { browserType: 'safari', browserVersion: '17.0.0', gpuVendor: 'apple' }
    );
    
    // Test with different matrix sizes
    const selectStrategyChrome = (chromeOps as any).selectOptimalStrategy.bind(chromeOps);
    const selectStrategyFirefox = (firefoxOps as any).selectOptimalStrategy.bind(firefoxOps);
    const selectStrategySafari = (safariOps as any).selectOptimalStrategy.bind(safariOps);
    
    // Chrome transitions to micro-tiled earlier
    expect(selectStrategyChrome(300, 300, 300)).toBe(MatrixMultiplyStrategy.MICRO_TILED);
    
    // Firefox stays with tiled longer
    expect(selectStrategyFirefox(300, 300, 300)).toBe(MatrixMultiplyStrategy.TILED);
    
    // Safari transitions between strategies at different boundaries
    expect(selectStrategySafari(150, 150, 150)).toBe(MatrixMultiplyStrategy.SIMPLE);
    expect(selectStrategySafari(200, 200, 200)).toBe(MatrixMultiplyStrategy.TILED);
  });
});