/**
 * Browser-Optimized WebGPU Operations
 * 
 * Provides browser-specific optimizations for WebGPU matrix operations to maximize performance
 * across different browsers and hardware configurations.
 */

import { WebGPUBackend } from './backend';
import { WebGPUMatrixMultiplication, MatrixOperationOptions } from './matrix_operations';
import { Tensor } from '../../tensor/tensor';

/**
 * Browser type identification
 */
export enum BrowserType {
  CHROME = 'chrome',
  FIREFOX = 'firefox',
  SAFARI = 'safari',
  EDGE = 'edge',
  UNKNOWN = 'unknown'
}

/**
 * Browser capabilities and optimization settings
 */
export interface BrowserCapabilities {
  /**
   * Browser type
   */
  browserType: BrowserType;
  
  /**
   * Browser version
   */
  version: string;
  
  /**
   * WebGPU feature support
   */
  features: {
    /**
     * Supports 16-bit storage
     */
    storage16Bit: boolean;
    
    /**
     * Supports 8-bit storage
     */
    storage8Bit: boolean;
    
    /**
     * Supports timestamp queries
     */
    timestampQuery: boolean;
    
    /**
     * Supports indirect dispatch
     */
    indirectDispatch: boolean;
  };
  
  /**
   * Hardware info
   */
  hardware: {
    /**
     * GPU vendor name
     */
    vendor: string;
    
    /**
     * GPU architecture
     */
    architecture: string;
    
    /**
     * Is integrated GPU
     */
    isIntegrated: boolean;
    
    /**
     * Has unified memory architecture
     */
    hasUnifiedMemory: boolean;
  };
  
  /**
   * Performance tier (1-5, with 5 being highest)
   */
  performanceTier: number;
  
  /**
   * Optimal workgroup sizes
   */
  optimalWorkgroupSizes: {
    /**
     * Matrix multiplication workgroup size
     */
    matmul: [number, number, number];
    
    /**
     * Elementwise operations workgroup size
     */
    elementwise: [number, number, number];
    
    /**
     * Convolution workgroup size
     */
    conv2d: [number, number, number];
    
    /**
     * Batch matrix multiplication workgroup size
     */
    batchMatmul: [number, number, number];
  };
  
  /**
   * Optimal tile sizes
   */
  optimalTileSizes: {
    /**
     * Matrix multiplication tile size
     */
    matmul: number;
    
    /**
     * Convolution tile size
     */
    conv2d: number;
  };
  
  /**
   * Browser-specific optimization flags
   */
  optimizationFlags: {
    /**
     * Use shared memory for matrix multiplication
     */
    useSharedMemory: boolean;
    
    /**
     * Use micro-tiling for large matrices
     */
    useMicroTiling: boolean;
    
    /**
     * Use vectorization for elementwise operations
     */
    useVectorization: boolean;
    
    /**
     * Use fast math approximations
     */
    useFastMath: boolean;
    
    /**
     * Use memory layout optimizations
     */
    useLayoutOptimizations: boolean;
    
    /**
     * Aggressiveness of loop unrolling (0-3)
     */
    loopUnrollingLevel: number;
    
    /**
     * Use predication optimization
     */
    usePredication: boolean;
  };
}

/**
 * Detects current browser type
 */
export function detectBrowserType(): BrowserType {
  const userAgent = navigator.userAgent.toLowerCase();
  
  if (userAgent.includes('edg/')) {
    return BrowserType.EDGE;
  } else if (userAgent.includes('chrome/')) {
    return BrowserType.CHROME;
  } else if (userAgent.includes('firefox/')) {
    return BrowserType.FIREFOX;
  } else if (userAgent.includes('safari/') && !userAgent.includes('chrome/')) {
    return BrowserType.SAFARI;
  }
  
  return BrowserType.UNKNOWN;
}

/**
 * Detects browser version
 */
export function detectBrowserVersion(): string {
  const browserType = detectBrowserType();
  const userAgent = navigator.userAgent.toLowerCase();
  let match;
  
  switch (browserType) {
    case BrowserType.CHROME:
      match = userAgent.match(/chrome\/(\d+\.\d+)/);
      break;
    case BrowserType.FIREFOX:
      match = userAgent.match(/firefox\/(\d+\.\d+)/);
      break;
    case BrowserType.SAFARI:
      match = userAgent.match(/version\/(\d+\.\d+)/);
      break;
    case BrowserType.EDGE:
      match = userAgent.match(/edg\/(\d+\.\d+)/);
      break;
  }
  
  return match ? match[1] : 'unknown';
}

/**
 * Get browser-specific capabilities and optimizations
 */
export async function getBrowserCapabilities(device: GPUDevice): Promise<BrowserCapabilities> {
  const browserType = detectBrowserType();
  const version = detectBrowserVersion();
  
  // Get adapter info
  let adapterInfo: GPUAdapterInfo;
  try {
    // @ts-ignore - TypeScript may not have updated types for this yet
    adapterInfo = await device.adapter?.requestAdapterInfo();
  } catch (e) {
    adapterInfo = { vendor: 'unknown', architecture: 'unknown' };
  }
  
  // Default capabilities
  const defaultCapabilities: BrowserCapabilities = {
    browserType,
    version,
    features: {
      storage16Bit: false,
      storage8Bit: false,
      timestampQuery: false,
      indirectDispatch: false,
    },
    hardware: {
      vendor: adapterInfo.vendor || 'unknown',
      architecture: adapterInfo.architecture || 'unknown',
      isIntegrated: false,
      hasUnifiedMemory: false
    },
    performanceTier: 3,
    optimalWorkgroupSizes: {
      matmul: [16, 16, 1],
      elementwise: [256, 1, 1],
      conv2d: [8, 8, 1],
      batchMatmul: [16, 16, 1],
    },
    optimalTileSizes: {
      matmul: 16,
      conv2d: 8,
    },
    optimizationFlags: {
      useSharedMemory: true,
      useMicroTiling: true,
      useVectorization: true,
      useFastMath: true,
      useLayoutOptimizations: true,
      loopUnrollingLevel: 2,
      usePredication: false,
    }
  };
  
  // Check for features
  defaultCapabilities.features.storage16Bit = device.features.has('shader-f16');
  
  // Browser-specific optimizations
  switch (browserType) {
    case BrowserType.CHROME:
      // Chrome typically performs well with larger workgroup sizes
      defaultCapabilities.optimalWorkgroupSizes.matmul = [16, 16, 1];
      defaultCapabilities.optimalWorkgroupSizes.elementwise = [256, 1, 1];
      defaultCapabilities.optimalWorkgroupSizes.conv2d = [8, 8, 1];
      defaultCapabilities.optimalWorkgroupSizes.batchMatmul = [16, 16, 1];
      defaultCapabilities.optimalTileSizes.matmul = 16;
      defaultCapabilities.optimalTileSizes.conv2d = 8;
      defaultCapabilities.optimizationFlags.loopUnrollingLevel = 3;
      defaultCapabilities.performanceTier = 4;
      break;
      
    case BrowserType.FIREFOX:
      // Firefox works best with workgroup sizes that are multiples of 64
      defaultCapabilities.optimalWorkgroupSizes.matmul = [8, 8, 1]; 
      defaultCapabilities.optimalWorkgroupSizes.elementwise = [64, 1, 1];
      defaultCapabilities.optimalWorkgroupSizes.conv2d = [8, 8, 1];
      defaultCapabilities.optimalWorkgroupSizes.batchMatmul = [8, 8, 1];
      defaultCapabilities.optimalTileSizes.matmul = 8;
      defaultCapabilities.optimalTileSizes.conv2d = 4;
      // Firefox is particularly good at compute shaders with optimized barriers
      defaultCapabilities.optimizationFlags.usePredication = true;
      defaultCapabilities.performanceTier = 3;
      break;
      
    case BrowserType.SAFARI:
      // Safari benefits from smaller workgroup sizes on some hardware
      defaultCapabilities.optimalWorkgroupSizes.matmul = [8, 8, 1];
      defaultCapabilities.optimalWorkgroupSizes.elementwise = [128, 1, 1];
      defaultCapabilities.optimalWorkgroupSizes.conv2d = [4, 4, 1];
      defaultCapabilities.optimalWorkgroupSizes.batchMatmul = [8, 8, 1];
      defaultCapabilities.optimalTileSizes.matmul = 8;
      defaultCapabilities.optimalTileSizes.conv2d = 4;
      defaultCapabilities.optimizationFlags.useMicroTiling = false; // Can cause issues on Safari
      defaultCapabilities.optimizationFlags.loopUnrollingLevel = 1;
      defaultCapabilities.performanceTier = 3;
      break;
      
    case BrowserType.EDGE:
      // Edge/Chrome similar optimization patterns
      defaultCapabilities.optimalWorkgroupSizes.matmul = [16, 16, 1];
      defaultCapabilities.optimalWorkgroupSizes.elementwise = [256, 1, 1];
      defaultCapabilities.optimalWorkgroupSizes.conv2d = [8, 8, 1];
      defaultCapabilities.optimalWorkgroupSizes.batchMatmul = [16, 16, 1];
      defaultCapabilities.optimalTileSizes.matmul = 16;
      defaultCapabilities.optimalTileSizes.conv2d = 8;
      defaultCapabilities.optimizationFlags.loopUnrollingLevel = 3;
      defaultCapabilities.performanceTier = 4;
      break;
      
    default:
      // Conservative defaults for unknown browsers
      defaultCapabilities.optimalWorkgroupSizes.matmul = [8, 8, 1];
      defaultCapabilities.optimalWorkgroupSizes.elementwise = [64, 1, 1];
      defaultCapabilities.optimalWorkgroupSizes.conv2d = [4, 4, 1];
      defaultCapabilities.optimalWorkgroupSizes.batchMatmul = [8, 8, 1];
      defaultCapabilities.optimalTileSizes.matmul = 8;
      defaultCapabilities.optimalTileSizes.conv2d = 4;
      defaultCapabilities.optimizationFlags.loopUnrollingLevel = 1;
      defaultCapabilities.optimizationFlags.useMicroTiling = false;
      defaultCapabilities.performanceTier = 2;
  }
  
  // Hardware-specific adjustments
  if (adapterInfo.vendor) {
    const vendorLower = adapterInfo.vendor.toLowerCase();
    
    // Apple GPUs (Metal)
    if (vendorLower.includes('apple')) {
      defaultCapabilities.hardware.hasUnifiedMemory = true;
      defaultCapabilities.optimalTileSizes.matmul = 8; // Apple GPUs work better with smaller tiles
      
      // M1/M2 processors are very powerful
      if (browserType === BrowserType.SAFARI) {
        defaultCapabilities.performanceTier = 5;
      }
    }
    
    // NVIDIA GPUs
    else if (vendorLower.includes('nvidia')) {
      defaultCapabilities.hardware.isIntegrated = false;
      defaultCapabilities.optimalWorkgroupSizes.matmul = [16, 16, 1]; // NVIDIA GPUs typically perform well with larger workgroups
      defaultCapabilities.optimalTileSizes.matmul = 16;
      defaultCapabilities.optimizationFlags.useMicroTiling = true;
      defaultCapabilities.performanceTier = 5;
    }
    
    // AMD GPUs
    else if (vendorLower.includes('amd') || vendorLower.includes('ati')) {
      defaultCapabilities.hardware.isIntegrated = false;
      // AMD typically works better with smaller tiles
      defaultCapabilities.optimalWorkgroupSizes.matmul = [8, 8, 1];
      defaultCapabilities.optimalTileSizes.matmul = 8;
      defaultCapabilities.performanceTier = 4;
    }
    
    // Intel GPUs (typically integrated)
    else if (vendorLower.includes('intel')) {
      defaultCapabilities.hardware.isIntegrated = true;
      // Intel integrated GPUs work better with moderate workgroup size
      defaultCapabilities.optimalWorkgroupSizes.matmul = [8, 8, 1];
      defaultCapabilities.optimalTileSizes.matmul = 8;
      defaultCapabilities.optimizationFlags.useMicroTiling = false;
      defaultCapabilities.performanceTier = 2;
      
      // Recent Intel GPUs are much improved
      if (adapterInfo.architecture && 
         (adapterInfo.architecture.includes('Xe') || 
          adapterInfo.architecture.includes('Arc'))) {
        defaultCapabilities.performanceTier = 3;
        defaultCapabilities.optimizationFlags.useMicroTiling = true;
      }
    }
    
    // Qualcomm (mobile)
    else if (vendorLower.includes('qualcomm') || vendorLower.includes('adreno')) {
      defaultCapabilities.hardware.isIntegrated = true;
      defaultCapabilities.hardware.hasUnifiedMemory = true;
      defaultCapabilities.optimalWorkgroupSizes.matmul = [4, 4, 1]; // Mobile GPUs typically need smaller workgroups
      defaultCapabilities.optimalWorkgroupSizes.elementwise = [64, 1, 1];
      defaultCapabilities.optimalTileSizes.matmul = 4;
      defaultCapabilities.optimizationFlags.useMicroTiling = false;
      defaultCapabilities.performanceTier = 2;
    }
    
    // ARM (mobile)
    else if (vendorLower.includes('arm') || vendorLower.includes('mali')) {
      defaultCapabilities.hardware.isIntegrated = true;
      defaultCapabilities.hardware.hasUnifiedMemory = true;
      defaultCapabilities.optimalWorkgroupSizes.matmul = [4, 4, 1];
      defaultCapabilities.optimalWorkgroupSizes.elementwise = [64, 1, 1];
      defaultCapabilities.optimalTileSizes.matmul = 4;
      defaultCapabilities.optimizationFlags.useMicroTiling = false;
      defaultCapabilities.performanceTier = 2;
    }
  }
  
  return defaultCapabilities;
}

/**
 * Convert browser capabilities to matrix operation options
 */
export function capabilitiesToOptions(capabilities: BrowserCapabilities): MatrixOperationOptions {
  return {
    workgroupSize: capabilities.optimalWorkgroupSizes.matmul[0],
    tileSize: capabilities.optimalTileSizes.matmul,
    useSharedMemory: capabilities.optimizationFlags.useSharedMemory,
    browserOptimization: capabilities.browserType,
    useFastMath: capabilities.optimizationFlags.useFastMath,
    useLayoutOptimizations: capabilities.optimizationFlags.useLayoutOptimizations,
  };
}

/**
 * Browser-optimized matrix operations
 */
export class BrowserOptimizedMatrixOperations {
  /** WebGPU backend reference */
  private backend: WebGPUBackend;
  
  /** GPU device reference */
  private device: GPUDevice;
  
  /** Matrix multiplication implementation */
  private matrixMultiplication: WebGPUMatrixMultiplication;
  
  /** Browser capabilities */
  private capabilities: BrowserCapabilities;
  
  /** Default options based on browser */
  private defaultOptions: MatrixOperationOptions;
  
  /** Cache of optimal options for different matrix sizes */
  private optionsCache: Map<string, MatrixOperationOptions> = new Map();
  
  /**
   * Constructor
   * @param backend WebGPU backend instance
   * @param matrixMultiplication Matrix multiplication implementation
   */
  constructor(
    backend: WebGPUBackend,
    matrixMultiplication: WebGPUMatrixMultiplication,
    capabilities: BrowserCapabilities
  ) {
    this.backend = backend;
    this.device = (backend as any).device;
    this.matrixMultiplication = matrixMultiplication;
    this.capabilities = capabilities;
    this.defaultOptions = capabilitiesToOptions(capabilities);
    
    if (!this.device) {
      throw new Error('WebGPU backend not properly initialized');
    }
  }
  
  /**
   * Execute matrix multiplication with browser-specific optimizations
   * @param a Matrix A (M×K)
   * @param b Matrix B (K×N)
   * @param customOptions Additional operation options
   * @returns Result matrix C (M×N)
   */
  async multiply<T>(a: Tensor<T>, b: Tensor<T>, customOptions: Partial<MatrixOperationOptions> = {}): Promise<Tensor<T>> {
    // Get optimal options for these matrix dimensions
    const options = this.getOptimalOptions(a.shape, b.shape, customOptions);
    
    // Execute matrix multiplication with optimal options
    return this.matrixMultiplication.multiply(a, b, options);
  }
  
  /**
   * Execute batch matrix multiplication with browser-specific optimizations
   * @param a Batch of matrices A [batchSize, M, K]
   * @param b Batch of matrices B [batchSize, K, N]
   * @param customOptions Additional operation options
   * @returns Batch of result matrices C [batchSize, M, N]
   */
  async batchMultiply<T>(a: Tensor<T>, b: Tensor<T>, customOptions: Partial<MatrixOperationOptions> = {}): Promise<Tensor<T>> {
    // Get optimal options for these batch matrix dimensions
    const options = this.getOptimalBatchOptions(a.shape, b.shape, customOptions);
    
    // Execute batch matrix multiplication with optimal options
    return this.matrixMultiplication.batchMultiply(a, b, options);
  }
  
  /**
   * Execute 2D convolution with browser-specific optimizations
   * @param input Input tensor [batchSize, inputHeight, inputWidth, inputChannels]
   * @param filters Filter tensor [filterHeight, filterWidth, inputChannels, outputChannels]
   * @param strides Stride of the convolution [strideHeight, strideWidth]
   * @param padding Padding mode: 'same' or 'valid'
   * @param customOptions Additional operation options
   * @returns Output tensor [batchSize, outputHeight, outputWidth, outputChannels]
   */
  async conv2d<T>(
    input: Tensor<T>,
    filters: Tensor<T>,
    strides: [number, number] = [1, 1],
    padding: 'same' | 'valid' = 'valid',
    customOptions: Partial<MatrixOperationOptions> = {}
  ): Promise<Tensor<T>> {
    // Get optimal options for convolution
    const options = this.getOptimalConvOptions(input.shape, filters.shape, customOptions);
    
    // Execute convolution with optimal options
    return this.matrixMultiplication.conv2d(input, filters, strides, padding, options);
  }
  
  /**
   * Get optimal options for matrix multiplication based on matrix dimensions and browser
   * @param aShape Shape of matrix A
   * @param bShape Shape of matrix B
   * @param customOptions Custom options to override defaults
   * @returns Optimized operation options
   */
  private getOptimalOptions(
    aShape: readonly number[],
    bShape: readonly number[],
    customOptions: Partial<MatrixOperationOptions> = {}
  ): MatrixOperationOptions {
    if (aShape.length !== 2 || bShape.length !== 2) {
      throw new Error(`Matrix shapes must be 2D, got ${aShape} and ${bShape}`);
    }
    
    const M = aShape[0];
    const K = aShape[1];
    const N = bShape[1];
    
    // Check if we have cached options for this size
    const sizeKey = `matmul_${M}_${K}_${N}`;
    if (this.optionsCache.has(sizeKey)) {
      const cachedOptions = this.optionsCache.get(sizeKey)!;
      // Merge with custom options
      return { ...cachedOptions, ...customOptions };
    }
    
    // Start with browser-specific defaults
    const options: MatrixOperationOptions = { ...this.defaultOptions };
    
    // Adjust based on matrix dimensions
    if (M * K * N > 1000000) {
      // Large matrices benefit from different optimizations
      options.useSharedMemory = this.capabilities.optimizationFlags.useSharedMemory;
      
      if (this.capabilities.optimizationFlags.useMicroTiling) {
        options.workgroupSize = Math.min(16, this.capabilities.optimalWorkgroupSizes.matmul[0]);
        options.tileSize = this.capabilities.optimalTileSizes.matmul;
      } else {
        // If micro-tiling isn't supported, use regular tiling
        options.workgroupSize = this.capabilities.optimalWorkgroupSizes.matmul[0];
        options.tileSize = this.capabilities.optimalTileSizes.matmul;
      }
    } else if (M * K * N > 10000) {
      // Medium-sized matrices
      options.useSharedMemory = this.capabilities.optimizationFlags.useSharedMemory;
      options.workgroupSize = this.capabilities.optimalWorkgroupSizes.matmul[0];
      options.tileSize = this.capabilities.optimalTileSizes.matmul;
    } else {
      // Small matrices
      options.useSharedMemory = false;
      options.workgroupSize = Math.min(8, this.capabilities.optimalWorkgroupSizes.matmul[0]);
    }
    
    // Browser-specific adjustments
    switch (this.capabilities.browserType) {
      case BrowserType.FIREFOX:
        // Firefox benefits from workgroup sizes divisible by 8
        options.workgroupSize = options.workgroupSize - (options.workgroupSize % 8);
        options.workgroupSize = Math.max(8, options.workgroupSize);
        break;
        
      case BrowserType.SAFARI:
        // Safari performs better with powers of 2
        options.workgroupSize = options.workgroupSize > 8 ? 8 : 4;
        options.tileSize = options.tileSize > 8 ? 8 : 4;
        break;
    }
    
    // Cache the computed options
    this.optionsCache.set(sizeKey, { ...options });
    
    // Merge with custom options
    return { ...options, ...customOptions };
  }
  
  /**
   * Get optimal options for batch matrix multiplication
   * @param aShape Shape of batch matrix A [batchSize, M, K]
   * @param bShape Shape of batch matrix B [batchSize, K, N]
   * @param customOptions Custom options to override defaults
   * @returns Optimized operation options
   */
  private getOptimalBatchOptions(
    aShape: readonly number[],
    bShape: readonly number[],
    customOptions: Partial<MatrixOperationOptions> = {}
  ): MatrixOperationOptions {
    if (aShape.length !== 3 || bShape.length !== 3) {
      throw new Error(`Batch matrix shapes must be 3D, got ${aShape} and ${bShape}`);
    }
    
    const batchSize = aShape[0];
    const M = aShape[1];
    const K = aShape[2];
    const N = bShape[2];
    
    // Check if we have cached options for this size
    const sizeKey = `batch_matmul_${batchSize}_${M}_${K}_${N}`;
    if (this.optionsCache.has(sizeKey)) {
      const cachedOptions = this.optionsCache.get(sizeKey)!;
      // Merge with custom options
      return { ...cachedOptions, ...customOptions };
    }
    
    // Start with browser-specific defaults
    const options: MatrixOperationOptions = { ...this.defaultOptions };
    
    // For batch operations, we use the batch matmul workgroup sizes
    options.workgroupSize = this.capabilities.optimalWorkgroupSizes.batchMatmul[0];
    options.tileSize = this.capabilities.optimalTileSizes.matmul;
    
    // Adjust based on batch size and matrix dimensions
    if (batchSize > 8) {
      // Large batch sizes benefit from smaller workgroups
      options.workgroupSize = Math.min(options.workgroupSize, 8);
    }
    
    if (M * K * N > 100000) {
      // Large matrices in the batch
      options.useSharedMemory = this.capabilities.optimizationFlags.useSharedMemory;
    } else {
      // Smaller matrices in the batch
      options.useSharedMemory = false;
    }
    
    // Browser-specific adjustments
    switch (this.capabilities.browserType) {
      case BrowserType.FIREFOX:
        // Firefox benefits from workgroup sizes divisible by 8
        options.workgroupSize = options.workgroupSize - (options.workgroupSize % 8);
        options.workgroupSize = Math.max(8, options.workgroupSize);
        break;
        
      case BrowserType.SAFARI:
        // Safari performs better with powers of 2
        options.workgroupSize = options.workgroupSize > 8 ? 8 : 4;
        options.tileSize = options.tileSize > 8 ? 8 : 4;
        break;
    }
    
    // Cache the computed options
    this.optionsCache.set(sizeKey, { ...options });
    
    // Merge with custom options
    return { ...options, ...customOptions };
  }
  
  /**
   * Get optimal options for 2D convolution
   * @param inputShape Shape of input tensor [batchSize, inputHeight, inputWidth, inputChannels]
   * @param filterShape Shape of filter tensor [filterHeight, filterWidth, inputChannels, outputChannels]
   * @param customOptions Custom options to override defaults
   * @returns Optimized operation options
   */
  private getOptimalConvOptions(
    inputShape: readonly number[],
    filterShape: readonly number[],
    customOptions: Partial<MatrixOperationOptions> = {}
  ): MatrixOperationOptions {
    if (inputShape.length !== 4 || filterShape.length !== 4) {
      throw new Error(`Conv2D shapes must be 4D, got ${inputShape} and ${filterShape}`);
    }
    
    const batchSize = inputShape[0];
    const inputHeight = inputShape[1];
    const inputWidth = inputShape[2];
    const inputChannels = inputShape[3];
    const filterHeight = filterShape[0];
    const filterWidth = filterShape[1];
    const outputChannels = filterShape[3];
    
    // Check if we have cached options for this size
    const sizeKey = `conv2d_${batchSize}_${inputHeight}x${inputWidth}_${inputChannels}_${filterHeight}x${filterWidth}_${outputChannels}`;
    if (this.optionsCache.has(sizeKey)) {
      const cachedOptions = this.optionsCache.get(sizeKey)!;
      // Merge with custom options
      return { ...cachedOptions, ...customOptions };
    }
    
    // Start with browser-specific defaults
    const options: MatrixOperationOptions = { ...this.defaultOptions };
    
    // For convolution, we use the conv2d workgroup sizes
    options.workgroupSize = this.capabilities.optimalWorkgroupSizes.conv2d[0];
    options.tileSize = this.capabilities.optimalTileSizes.conv2d;
    
    // Adjust based on tensor and filter dimensions
    if (filterHeight * filterWidth > 25) {
      // Large filters benefit from smaller workgroups
      options.workgroupSize = Math.min(options.workgroupSize, 4);
    }
    
    if (inputChannels * outputChannels > 1024) {
      // Many channels benefit from shared memory
      options.useSharedMemory = this.capabilities.optimizationFlags.useSharedMemory;
    } else {
      // Fewer channels might be faster without shared memory overhead
      options.useSharedMemory = false;
    }
    
    // Browser-specific adjustments
    switch (this.capabilities.browserType) {
      case BrowserType.FIREFOX:
        // Firefox benefits from workgroup sizes divisible by 4 for convolution
        options.workgroupSize = options.workgroupSize - (options.workgroupSize % 4);
        options.workgroupSize = Math.max(4, options.workgroupSize);
        break;
        
      case BrowserType.SAFARI:
        // Safari performs better with small workgroups for convolution
        options.workgroupSize = Math.min(options.workgroupSize, 4);
        options.tileSize = Math.min(options.tileSize, 4);
        break;
    }
    
    // Cache the computed options
    this.optionsCache.set(sizeKey, { ...options });
    
    // Merge with custom options
    return { ...options, ...customOptions };
  }
  
  /**
   * Benchmark different parameter settings to find optimal configuration
   * @param a Test matrix A
   * @param b Test matrix B
   * @returns Promise resolving to optimized options
   */
  async benchmarkAndOptimize<T>(a: Tensor<T>, b: Tensor<T>): Promise<MatrixOperationOptions> {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error(`Matrix shapes must be 2D for benchmarking, got ${a.shape} and ${b.shape}`);
    }
    
    const M = a.shape[0];
    const K = a.shape[1];
    const N = b.shape[1];
    
    // Check if we already have cached options for this size
    const sizeKey = `matmul_${M}_${K}_${N}`;
    if (this.optionsCache.has(sizeKey)) {
      return this.optionsCache.get(sizeKey)!;
    }
    
    console.log(`Benchmarking matrix multiplication for size ${M}×${K} × ${K}×${N} on ${this.capabilities.browserType}...`);
    
    // Parameters to test
    const workgroupSizes = [4, 8, 16];
    const tileSizes = [4, 8, 16];
    const sharedMemoryOptions = [true, false];
    
    let bestOptions: MatrixOperationOptions = this.getOptimalOptions(a.shape, b.shape);
    let bestTime = Number.MAX_VALUE;
    
    // Run tests for different parameter combinations
    for (const workgroupSize of workgroupSizes) {
      for (const tileSize of tileSizes) {
        for (const useSharedMemory of sharedMemoryOptions) {
          // Skip invalid combinations
          if (tileSize < workgroupSize) continue;
          
          const testOptions: MatrixOperationOptions = {
            workgroupSize,
            tileSize,
            useSharedMemory,
            browserOptimization: this.capabilities.browserType,
            useFastMath: this.capabilities.optimizationFlags.useFastMath,
            useLayoutOptimizations: this.capabilities.optimizationFlags.useLayoutOptimizations,
          };
          
          // Run benchmark
          const time = await this.benchmarkMultiply(a, b, testOptions);
          
          console.log(`Options: workgroupSize=${workgroupSize}, tileSize=${tileSize}, useSharedMemory=${useSharedMemory}, time=${time.toFixed(2)}ms`);
          
          // Update best options if this is faster
          if (time < bestTime) {
            bestTime = time;
            bestOptions = { ...testOptions };
          }
        }
      }
    }
    
    console.log(`Best options: workgroupSize=${bestOptions.workgroupSize}, tileSize=${bestOptions.tileSize}, useSharedMemory=${bestOptions.useSharedMemory}, time=${bestTime.toFixed(2)}ms`);
    
    // Cache the best options
    this.optionsCache.set(sizeKey, bestOptions);
    
    return bestOptions;
  }
  
  /**
   * Benchmark matrix multiplication with given options
   * @param a Matrix A
   * @param b Matrix B
   * @param options Operation options
   * @returns Promise resolving to execution time in milliseconds
   */
  private async benchmarkMultiply<T>(a: Tensor<T>, b: Tensor<T>, options: MatrixOperationOptions): Promise<number> {
    // Warm-up run
    await this.matrixMultiplication.multiply(a, b, options);
    
    // Timed runs
    const NUM_RUNS = 5;
    const times: number[] = [];
    
    for (let i = 0; i < NUM_RUNS; i++) {
      const start = performance.now();
      await this.matrixMultiplication.multiply(a, b, options);
      const end = performance.now();
      times.push(end - start);
    }
    
    // Return average time (excluding the slowest run)
    times.sort((a, b) => a - b);
    const avgTime = times.slice(0, NUM_RUNS - 1).reduce((sum, time) => sum + time, 0) / (NUM_RUNS - 1);
    
    return avgTime;
  }
  
  /**
   * Clear cached options
   */
  clearCache(): void {
    this.optionsCache.clear();
  }
  
  /**
   * Dispose resources
   */
  dispose(): void {
    this.clearCache();
  }
}

/**
 * Create browser-optimized matrix operations
 * @param backend WebGPU backend
 * @param matrixMultiplication Matrix multiplication implementation
 * @returns Promise resolving to browser-optimized matrix operations
 */
export async function createBrowserOptimizedOperations(
  backend: WebGPUBackend,
  matrixMultiplication: WebGPUMatrixMultiplication
): Promise<BrowserOptimizedMatrixOperations> {
  const device = (backend as any).device;
  const capabilities = await getBrowserCapabilities(device);
  
  const operations = new BrowserOptimizedMatrixOperations(
    backend,
    matrixMultiplication,
    capabilities
  );
  
  return operations;
}