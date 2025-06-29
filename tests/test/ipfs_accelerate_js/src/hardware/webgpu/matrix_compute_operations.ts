/**
 * WebGPU Matrix Compute Operations
 * 
 * Advanced matrix operations using specialized WebGPU compute shaders
 * with optimizations for different browsers and hardware configurations.
 */

import { WebGPUBackend } from './backend';
import { WebGPUBufferManager } from './buffer_manager';
import { Tensor } from '../../tensor/tensor';
import { BrowserType, getBrowserCapabilities } from './browser_optimized_operations';

/**
 * Operation type for matrix compute operations
 */
export enum MatrixComputeOperationType {
  // Element-wise operations
  ADD = 'add',
  SUBTRACT = 'subtract',
  MULTIPLY = 'multiply',
  DIVIDE = 'divide',
  
  // Advanced operations
  MATMUL = 'matmul',
  TRANSPOSE = 'transpose',
  INVERSE = 'inverse',
  
  // Decompositions
  LU = 'lu',
  QR = 'qr',
  SVD = 'svd',
  
  // Statistical operations
  MEAN = 'mean',
  VARIANCE = 'variance',
  NORM = 'norm',
  
  // Special operations
  FFT = 'fft',
  CONV = 'conv'
}

/**
 * Configuration options for matrix compute operations
 */
export interface MatrixComputeOptions {
  /**
   * Workgroup size for compute operations
   */
  workgroupSize?: number;
  
  /**
   * Tile size for tiled operations
   */
  tileSize?: number;
  
  /**
   * Whether to use shared memory optimizations
   */
  useSharedMemory?: boolean;
  
  /**
   * Browser optimization target
   */
  browserOptimization?: BrowserType;
  
  /**
   * Whether to use fast math approximations
   */
  useFastMath?: boolean;
  
  /**
   * Whether to use vectorized operations (vec4)
   */
  useVectorization?: boolean;
  
  /**
   * Whether to use memory layout optimizations
   */
  useLayoutOptimizations?: boolean;
  
  /**
   * Precision level (high, medium, low)
   */
  precision?: 'high' | 'medium' | 'low';
  
  /**
   * Whether to use asynchronous execution
   */
  useAsync?: boolean;
}

/**
 * Default matrix compute options
 */
const DEFAULT_COMPUTE_OPTIONS: MatrixComputeOptions = {
  workgroupSize: 16,
  tileSize: 16,
  useSharedMemory: true,
  useFastMath: true,
  useVectorization: true,
  useLayoutOptimizations: true,
  precision: 'medium',
  useAsync: true
};

/**
 * WebGPU Matrix Compute Operations class
 * Provides hardware-accelerated matrix operations using optimized compute shaders
 */
export class WebGPUMatrixComputeOperations {
  /** WebGPU backend reference */
  private backend: WebGPUBackend;
  
  /** GPU device reference */
  private device: GPUDevice;
  
  /** Buffer manager for GPU memory */
  private bufferManager: WebGPUBufferManager;
  
  /** Cache for compute pipelines */
  private pipelineCache: Map<string, GPUComputePipeline> = new Map();
  
  /** Cache for bind group layouts */
  private bindGroupLayoutCache: Map<string, GPUBindGroupLayout> = new Map();
  
  /** Browser capabilities */
  private browserCapabilities: any = null;
  
  /** Whether browser optimizations are enabled */
  private browserOptimizationsEnabled: boolean = true;
  
  /**
   * Constructor
   * @param backend WebGPU backend
   */
  constructor(backend: WebGPUBackend) {
    this.backend = backend;
    this.device = (backend as any).device;
    this.bufferManager = (backend as any).bufferManager;
    
    if (!this.device || !this.bufferManager) {
      throw new Error('WebGPU backend not properly initialized');
    }
  }
  
  /**
   * Initialize browser capabilities
   */
  async initialize(): Promise<void> {
    this.browserCapabilities = await getBrowserCapabilities(this.device);
  }
  
  /**
   * Set whether browser optimizations are enabled
   * @param enabled Whether browser optimizations are enabled
   */
  setBrowserOptimizationsEnabled(enabled: boolean): void {
    this.browserOptimizationsEnabled = enabled;
  }
  
  /**
   * Get optimized compute options based on operation type and browser
   * @param type Operation type
   * @param options User-provided options
   * @returns Optimized options
   */
  private getOptimizedOptions(
    type: MatrixComputeOperationType,
    options: MatrixComputeOptions = {}
  ): MatrixComputeOptions {
    const result = { ...DEFAULT_COMPUTE_OPTIONS, ...options };
    
    if (!this.browserOptimizationsEnabled || !this.browserCapabilities) {
      return result;
    }
    
    // Apply browser-specific optimizations
    switch (this.browserCapabilities.browserType) {
      case BrowserType.CHROME:
        // Chrome generally works well with larger workgroups and vectorization
        result.workgroupSize = options.workgroupSize || 16;
        result.useVectorization = options.useVectorization !== undefined ? 
          options.useVectorization : true;
        break;
        
      case BrowserType.FIREFOX:
        // Firefox works best with workgroup sizes divisible by 8
        result.workgroupSize = options.workgroupSize || 8;
        // Firefox has excellent compute shader performance but benefits from specific optimizations
        result.useVectorization = options.useVectorization !== undefined ? 
          options.useVectorization : true;
        result.useSharedMemory = options.useSharedMemory !== undefined ? 
          options.useSharedMemory : true;
        break;
        
      case BrowserType.SAFARI:
        // Safari works better with smaller workgroups and less aggressive optimizations
        result.workgroupSize = options.workgroupSize || 8;
        result.useVectorization = options.useVectorization !== undefined ? 
          options.useVectorization : false;
        break;
        
      case BrowserType.EDGE:
        // Edge is similar to Chrome with some specific optimizations
        result.workgroupSize = options.workgroupSize || 16;
        result.useVectorization = options.useVectorization !== undefined ? 
          options.useVectorization : true;
        break;
    }
    
    // Apply operation-specific optimizations
    switch (type) {
      case MatrixComputeOperationType.MATMUL:
        // Matrix multiplication benefits from shared memory and tiling
        result.useSharedMemory = options.useSharedMemory !== undefined ? 
          options.useSharedMemory : true;
        result.tileSize = options.tileSize || 
          this.browserCapabilities.optimalTileSizes.matmul;
        break;
        
      case MatrixComputeOperationType.TRANSPOSE:
        // Transpose operations benefit from specific workgroup sizes
        result.workgroupSize = options.workgroupSize || 16;
        result.useSharedMemory = options.useSharedMemory !== undefined ? 
          options.useSharedMemory : true;
        break;
        
      case MatrixComputeOperationType.INVERSE:
      case MatrixComputeOperationType.LU:
      case MatrixComputeOperationType.QR:
      case MatrixComputeOperationType.SVD:
        // Numerical algorithms benefit from higher precision
        result.precision = options.precision || 'high';
        result.useFastMath = options.useFastMath !== undefined ? 
          options.useFastMath : false;
        break;
        
      case MatrixComputeOperationType.FFT:
        // FFT benefits from specific optimizations
        result.useVectorization = options.useVectorization !== undefined ? 
          options.useVectorization : true;
        result.workgroupSize = options.workgroupSize || 8;
        break;
        
      case MatrixComputeOperationType.CONV:
        // Convolution benefits from larger tiles but smaller workgroups
        result.tileSize = options.tileSize || 
          this.browserCapabilities.optimalTileSizes.conv2d;
        result.workgroupSize = options.workgroupSize || 
          this.browserCapabilities.optimalWorkgroupSizes.conv2d[0];
        break;
        
      // Element-wise operations generally benefit from vectorization and larger workgroups
      case MatrixComputeOperationType.ADD:
      case MatrixComputeOperationType.SUBTRACT:
      case MatrixComputeOperationType.MULTIPLY:
      case MatrixComputeOperationType.DIVIDE:
        result.useVectorization = options.useVectorization !== undefined ? 
          options.useVectorization : true;
        result.workgroupSize = options.workgroupSize || 256;
        result.useSharedMemory = options.useSharedMemory !== undefined ? 
          options.useSharedMemory : false; // Typically not needed for element-wise ops
        break;
    }
    
    return result;
  }
  
  /**
   * Helper to upload tensor to GPU if needed
   * @param tensor Tensor to upload
   * @returns GPU buffer
   */
  private async uploadTensorIfNeeded<T>(tensor: Tensor<T>): Promise<GPUBuffer> {
    // If already in GPU memory, retrieve the buffer
    if (tensor.backend === 'webgpu' && tensor.device === this.backend.id) {
      const tensorId = `tensor_${tensor.dataType}_${tensor.shape.join('x')}_${Date.now()}`;
      const existingBuffer = this.bufferManager.getTensorBuffer(tensorId);
      
      if (existingBuffer) {
        return existingBuffer;
      }
    }
    
    // Otherwise upload to GPU
    return await this.bufferManager.uploadTensor(tensor);
  }
  
  /**
   * Execute matrix transpose operation using compute shaders
   * @param input Input tensor [M, N]
   * @param options Operation options
   * @returns Output tensor [N, M]
   */
  async transpose<T>(input: Tensor<T>, options: MatrixComputeOptions = {}): Promise<Tensor<T>> {
    if (input.shape.length !== 2) {
      throw new Error(`Transpose requires 2D tensor, got shape: ${input.shape}`);
    }
    
    const M = input.shape[0];
    const N = input.shape[1];
    
    // Get optimized options
    const opts = this.getOptimizedOptions(MatrixComputeOperationType.TRANSPOSE, options);
    
    // Ensure tensor is on GPU
    const inputBuffer = await this.uploadTensorIfNeeded(input);
    
    // Create output tensor
    const outputShape = [N, M];
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: input.dataType,
        backend: 'webgpu',
        device: this.backend.id
      }
    );
    
    // Create output buffer
    const outputBuffer = this.bufferManager.createOutputBuffer(
      outputShape,
      input.dataType
    );
    
    // Generate shader for transpose
    const shaderCode = this.generateTransposeShader(M, N, opts);
    
    // Create or get the pipeline
    const pipelineKey = `transpose_${M}_${N}_${opts.workgroupSize}_${opts.useSharedMemory}`;
    let pipeline: GPUComputePipeline;
    
    if (this.pipelineCache.has(pipelineKey)) {
      pipeline = this.pipelineCache.get(pipelineKey)!;
    } else {
      // Create shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderCode,
        label: `transpose_shader_${pipelineKey}`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      
      if (this.bindGroupLayoutCache.has('transpose')) {
        bindGroupLayout = this.bindGroupLayoutCache.get('transpose')!;
      } else {
        bindGroupLayout = this.device.createBindGroupLayout({
          entries: [
            { // Input matrix
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Output matrix
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Dimensions
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' }
            }
          ],
          label: 'transpose_bind_group_layout'
        });
        
        this.bindGroupLayoutCache.set('transpose', bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `transpose_pipeline_layout_${pipelineKey}`
      });
      
      // Create compute pipeline
      pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `transpose_pipeline_${pipelineKey}`
      });
      
      // Cache pipeline for future use
      this.pipelineCache.set(pipelineKey, pipeline);
    }
    
    // Create uniform buffer for dimensions
    const uniformBuffer = this.device.createBuffer({
      size: 8, // 2 x 4 bytes (u32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'transpose_dimensions'
    });
    
    // Write dimensions
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([M, N])
    );
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } }
      ],
      label: `transpose_bind_group_${pipelineKey}`
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: `transpose_encoder_${pipelineKey}`
    });
    
    // Compute pass
    const pass = encoder.beginComputePass({
      label: `transpose_pass_${pipelineKey}`
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate workgroup counts
    const workgroupSize = opts.workgroupSize || 16;
    const workgroupCountX = Math.ceil(N / workgroupSize);
    const workgroupCountY = Math.ceil(M / workgroupSize);
    
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    return outputTensor;
  }
  
  /**
   * Execute matrix inversion using compute shaders
   * @param input Input tensor [N, N] (must be square)
   * @param options Operation options
   * @returns Output tensor [N, N] containing the inverse
   */
  async inverse<T>(input: Tensor<T>, options: MatrixComputeOptions = {}): Promise<Tensor<T>> {
    if (input.shape.length !== 2) {
      throw new Error(`Matrix inverse requires 2D tensor, got shape: ${input.shape}`);
    }
    
    if (input.shape[0] !== input.shape[1]) {
      throw new Error(`Matrix inverse requires square matrix, got shape: ${input.shape}`);
    }
    
    const N = input.shape[0];
    
    // Get optimized options
    const opts = this.getOptimizedOptions(MatrixComputeOperationType.INVERSE, options);
    
    // Ensure tensor is on GPU
    const inputBuffer = await this.uploadTensorIfNeeded(input);
    
    // Create output tensor
    const outputShape = [N, N];
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: input.dataType,
        backend: 'webgpu',
        device: this.backend.id
      }
    );
    
    // Create output buffer
    const outputBuffer = this.bufferManager.createOutputBuffer(
      outputShape,
      input.dataType
    );
    
    // Generate shader for matrix inversion
    const shaderCode = this.generateInverseShader(N, opts);
    
    // Create or get the pipeline
    const pipelineKey = `inverse_${N}_${opts.precision}`;
    let pipeline: GPUComputePipeline;
    
    if (this.pipelineCache.has(pipelineKey)) {
      pipeline = this.pipelineCache.get(pipelineKey)!;
    } else {
      // Create shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderCode,
        label: `inverse_shader_${pipelineKey}`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      
      if (this.bindGroupLayoutCache.has('inverse')) {
        bindGroupLayout = this.bindGroupLayoutCache.get('inverse')!;
      } else {
        bindGroupLayout = this.device.createBindGroupLayout({
          entries: [
            { // Input matrix
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Output matrix
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Dimensions
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' }
            }
          ],
          label: 'inverse_bind_group_layout'
        });
        
        this.bindGroupLayoutCache.set('inverse', bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `inverse_pipeline_layout_${pipelineKey}`
      });
      
      // Create compute pipeline
      pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `inverse_pipeline_${pipelineKey}`
      });
      
      // Cache pipeline for future use
      this.pipelineCache.set(pipelineKey, pipeline);
    }
    
    // Create uniform buffer for dimensions
    const uniformBuffer = this.device.createBuffer({
      size: 4, // 1 x 4 bytes (u32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'inverse_dimensions'
    });
    
    // Write dimensions
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([N])
    );
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } }
      ],
      label: `inverse_bind_group_${pipelineKey}`
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: `inverse_encoder_${pipelineKey}`
    });
    
    // Compute pass
    const pass = encoder.beginComputePass({
      label: `inverse_pass_${pipelineKey}`
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // For small matrices, use a single workgroup
    if (N <= 16) {
      pass.dispatchWorkgroups(1);
    } else {
      // For larger matrices, use block-based inversion
      // We dispatch N workgroups, one for each column of the output
      pass.dispatchWorkgroups(N);
    }
    
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    return outputTensor;
  }
  
  /**
   * Execute element-wise operation using compute shaders
   * @param a First input tensor
   * @param b Second input tensor
   * @param operationType Type of element-wise operation to perform
   * @param options Operation options
   * @returns Output tensor with the result of the element-wise operation
   */
  async elementWiseOp<T>(
    a: Tensor<T>, 
    b: Tensor<T>, 
    operationType: MatrixComputeOperationType,
    options: MatrixComputeOptions = {}
  ): Promise<Tensor<T>> {
    // Validate operation type
    if (![
      MatrixComputeOperationType.ADD,
      MatrixComputeOperationType.SUBTRACT,
      MatrixComputeOperationType.MULTIPLY,
      MatrixComputeOperationType.DIVIDE
    ].includes(operationType)) {
      throw new Error(`Invalid element-wise operation type: ${operationType}`);
    }
    
    // Validate shapes
    if (a.shape.length !== b.shape.length) {
      throw new Error(`Element-wise operations require tensors with same number of dimensions, got ${a.shape} and ${b.shape}`);
    }
    
    for (let i = 0; i < a.shape.length; i++) {
      if (a.shape[i] !== b.shape[i]) {
        throw new Error(`Element-wise operations require tensors with same shape, got ${a.shape} and ${b.shape}`);
      }
    }
    
    // Get optimized options
    const opts = this.getOptimizedOptions(operationType, options);
    
    // Ensure tensors are on GPU
    const aBuffer = await this.uploadTensorIfNeeded(a);
    const bBuffer = await this.uploadTensorIfNeeded(b);
    
    // Create output tensor
    const outputTensor = new Tensor<T>(
      a.shape,
      null,
      {
        dataType: a.dataType,
        backend: 'webgpu',
        device: this.backend.id
      }
    );
    
    // Create output buffer
    const outputBuffer = this.bufferManager.createOutputBuffer(
      a.shape,
      a.dataType
    );
    
    // Calculate total size
    const totalElements = a.shape.reduce((prod, dim) => prod * dim, 1);
    
    // Generate shader for element-wise operation
    const shaderCode = this.generateElementWiseShader(
      operationType,
      a.shape,
      opts
    );
    
    // Create or get the pipeline
    const dimKey = a.shape.join('x');
    const pipelineKey = `elementwise_${operationType}_${dimKey}_${opts.useVectorization}`;
    let pipeline: GPUComputePipeline;
    
    if (this.pipelineCache.has(pipelineKey)) {
      pipeline = this.pipelineCache.get(pipelineKey)!;
    } else {
      // Create shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderCode,
        label: `elementwise_shader_${pipelineKey}`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      
      if (this.bindGroupLayoutCache.has('elementwise')) {
        bindGroupLayout = this.bindGroupLayoutCache.get('elementwise')!;
      } else {
        bindGroupLayout = this.device.createBindGroupLayout({
          entries: [
            { // First input tensor
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Second input tensor
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Output tensor
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Dimensions and parameters
              binding: 3,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' }
            }
          ],
          label: 'elementwise_bind_group_layout'
        });
        
        this.bindGroupLayoutCache.set('elementwise', bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `elementwise_pipeline_layout_${pipelineKey}`
      });
      
      // Create compute pipeline
      pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `elementwise_pipeline_${pipelineKey}`
      });
      
      // Cache pipeline for future use
      this.pipelineCache.set(pipelineKey, pipeline);
    }
    
    // Create uniform buffer for dimensions and parameters
    const uniformBuffer = this.device.createBuffer({
      size: (a.shape.length + 1) * 4, // (ndims + 1) x 4 bytes (u32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'elementwise_parameters'
    });
    
    // Create uniform data including total size and shape dimensions
    const uniformData = new Uint32Array(a.shape.length + 1);
    uniformData[0] = totalElements;
    for (let i = 0; i < a.shape.length; i++) {
      uniformData[i + 1] = a.shape[i];
    }
    
    // Write dimensions and parameters
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      uniformData
    );
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } }
      ],
      label: `elementwise_bind_group_${pipelineKey}`
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: `elementwise_encoder_${pipelineKey}`
    });
    
    // Compute pass
    const pass = encoder.beginComputePass({
      label: `elementwise_pass_${pipelineKey}`
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate workgroup dispatch size
    const workgroupSize = opts.workgroupSize || 256;
    const useVectorization = opts.useVectorization || false;
    
    // For vectorized operations, we process 4 elements at once
    const elementsPerThread = useVectorization ? 4 : 1;
    const dispatchSize = Math.ceil(totalElements / (workgroupSize * elementsPerThread));
    
    pass.dispatchWorkgroups(dispatchSize);
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    return outputTensor;
  }
  
  /**
   * Generate transpose shader
   * @param M Number of rows in input
   * @param N Number of columns in input
   * @param options Operation options
   * @returns WGSL shader code
   */
  private generateTransposeShader(
    M: number,
    N: number,
    options: MatrixComputeOptions
  ): string {
    const workgroupSize = options.workgroupSize || 16;
    const useSharedMemory = options.useSharedMemory !== false;
    
    if (useSharedMemory) {
      // Use shared memory for efficient transposition
      return /* wgsl */`
// Matrix transpose using shared memory
// Input: [M, N] matrix
// Output: [N, M] matrix

struct Dimensions {
  M: u32,  // Rows in input
  N: u32,  // Columns in input
};

@group(0) @binding(0) var<storage, read> inputMatrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputMatrix: array<f32>;
@group(0) @binding(2) var<uniform> dimensions: Dimensions;

// Shared memory tile for efficient transposition
var<workgroup> tile: array<array<f32, ${workgroupSize}>, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize}, ${workgroupSize})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  // Calculate the base row and column for this workgroup
  let base_row = workgroup_id.x * ${workgroupSize};
  let base_col = workgroup_id.y * ${workgroupSize};
  
  // Calculate this thread's input coordinates
  let row = base_row + local_id.x;
  let col = base_col + local_id.y;
  
  // Load data into shared memory, with bounds checking
  if (row < dimensions.M && col < dimensions.N) {
    tile[local_id.x][local_id.y] = inputMatrix[row * dimensions.N + col];
  }
  
  // Ensure all threads have loaded their data
  workgroupBarrier();
  
  // Calculate this thread's output coordinates (transposed)
  let out_row = base_col + local_id.x;
  let out_col = base_row + local_id.y;
  
  // Store transposed data to output, with bounds checking
  if (out_row < dimensions.N && out_col < dimensions.M) {
    // Note the indices are swapped for tile access to perform the transpose
    outputMatrix[out_row * dimensions.M + out_col] = tile[local_id.y][local_id.x];
  }
}`;
    } else {
      // Simple direct transpose (less efficient but works for any size)
      return /* wgsl */`
// Simple direct matrix transpose
// Input: [M, N] matrix
// Output: [N, M] matrix

struct Dimensions {
  M: u32,  // Rows in input
  N: u32,  // Columns in input
};

@group(0) @binding(0) var<storage, read> inputMatrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputMatrix: array<f32>;
@group(0) @binding(2) var<uniform> dimensions: Dimensions;

@compute @workgroup_size(${workgroupSize}, ${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  
  // Bounds check
  if (row < dimensions.N && col < dimensions.M) {
    // Read from input at [col, row] and write to output at [row, col]
    outputMatrix[row * dimensions.M + col] = inputMatrix[col * dimensions.N + row];
  }
}`;
    }
  }
  
  /**
   * Generate matrix inversion shader
   * @param N Size of square matrix
   * @param options Operation options
   * @returns WGSL shader code
   */
  private generateInverseShader(
    N: number,
    options: MatrixComputeOptions
  ): string {
    // For small matrices, we can use a direct Gauss-Jordan elimination approach
    if (N <= 16) {
      return this.generateSmallMatrixInverseShader(N, options);
    } else {
      // For larger matrices, use a block-based approach or LU decomposition
      return this.generateLargeMatrixInverseShader(N, options);
    }
  }
  
  /**
   * Generate inverse shader for small matrices using Gauss-Jordan elimination
   * @param N Size of square matrix
   * @param options Operation options
   * @returns WGSL shader code
   */
  private generateSmallMatrixInverseShader(
    N: number,
    options: MatrixComputeOptions
  ): string {
    const precision = options.precision || 'medium';
    
    // Define precision-specific values
    const epsilon = precision === 'high' ? '1e-10' : precision === 'medium' ? '1e-7' : '1e-5';
    
    return /* wgsl */`
// Matrix inversion using Gauss-Jordan elimination
// Input: [N, N] matrix
// Output: [N, N] matrix (inverse)

struct Dimensions {
  N: u32,  // Matrix size
};

@group(0) @binding(0) var<storage, read> inputMatrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputMatrix: array<f32>;
@group(0) @binding(2) var<uniform> dimensions: Dimensions;

// Small matrix size allows for direct inversion within a single workgroup
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let N = dimensions.N;
  
  // Create a combined matrix [A|I] for Gauss-Jordan elimination
  var augmented: array<array<f32, ${N * 2}>, ${N}>;
  
  // Initialize augmented matrix [A|I]
  for (var i: u32 = 0u; i < N; i = i + 1u) {
    for (var j: u32 = 0u; j < N; j = j + 1u) {
      // Left side: copy input matrix A
      augmented[i][j] = inputMatrix[i * N + j];
      
      // Right side: identity matrix I
      if (i == j) {
        augmented[i][j + N] = 1.0;
      } else {
        augmented[i][j + N] = 0.0;
      }
    }
  }
  
  // Perform Gauss-Jordan elimination
  for (var k: u32 = 0u; k < N; k = k + 1u) {
    // Find pivot
    var maxRow = k;
    var maxVal = abs(augmented[k][k]);
    
    // Partial pivoting to improve numerical stability
    for (var i: u32 = k + 1u; i < N; i = i + 1u) {
      let absVal = abs(augmented[i][k]);
      if (absVal > maxVal) {
        maxVal = absVal;
        maxRow = i;
      }
    }
    
    // Check for singular matrix
    if (maxVal < ${epsilon}) {
      // Matrix is singular or nearly singular, set result to identity
      for (var i: u32 = 0u; i < N; i = i + 1u) {
        for (var j: u32 = 0u; j < N; j = j + 1u) {
          if (i == j) {
            outputMatrix[i * N + j] = 1.0;
          } else {
            outputMatrix[i * N + j] = 0.0;
          }
        }
      }
      return;
    }
    
    // Swap rows if needed
    if (maxRow != k) {
      for (var j: u32 = 0u; j < 2u * N; j = j + 1u) {
        let temp = augmented[k][j];
        augmented[k][j] = augmented[maxRow][j];
        augmented[maxRow][j] = temp;
      }
    }
    
    // Scale row k to have a unit diagonal
    let pivot = augmented[k][k];
    if (abs(pivot) > ${epsilon}) {
      for (var j: u32 = 0u; j < 2u * N; j = j + 1u) {
        augmented[k][j] = augmented[k][j] / pivot;
      }
    }
    
    // Eliminate other rows
    for (var i: u32 = 0u; i < N; i = i + 1u) {
      if (i != k) {
        let factor = augmented[i][k];
        for (var j: u32 = 0u; j < 2u * N; j = j + 1u) {
          augmented[i][j] = augmented[i][j] - factor * augmented[k][j];
        }
      }
    }
  }
  
  // Extract the inverse matrix from the right side of the augmented matrix
  for (var i: u32 = 0u; i < N; i = i + 1u) {
    for (var j: u32 = 0u; j < N; j = j + 1u) {
      outputMatrix[i * N + j] = augmented[i][j + N];
    }
  }
}`;
  }
  
  /**
   * Generate inverse shader for large matrices using block-based approach
   * @param N Size of square matrix
   * @param options Operation options
   * @returns WGSL shader code
   */
  private generateLargeMatrixInverseShader(
    N: number,
    options: MatrixComputeOptions
  ): string {
    const precision = options.precision || 'medium';
    
    // Define precision-specific values
    const epsilon = precision === 'high' ? '1e-10' : precision === 'medium' ? '1e-7' : '1e-5';
    
    // For larger matrices, we compute one column of the inverse at a time
    // This avoids using too much memory in a single workgroup
    return /* wgsl */`
// Matrix inversion for large matrices - one column at a time
// Input: [N, N] matrix
// Output: [N, N] matrix (inverse)

struct Dimensions {
  N: u32,  // Matrix size
};

@group(0) @binding(0) var<storage, read> inputMatrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputMatrix: array<f32>;
@group(0) @binding(2) var<uniform> dimensions: Dimensions;

// Compute one column of the inverse matrix at a time
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let N = dimensions.N;
  let col = global_id.x; // Which column of the inverse to compute
  
  // Skip if outside matrix bounds
  if (col >= N) {
    return;
  }
  
  // Make a copy of the input matrix for LU decomposition
  var A: array<f32, ${N * N}>;
  for (var i: u32 = 0u; i < N * N; i = i + 1u) {
    A[i] = inputMatrix[i];
  }
  
  // Create vector b for the linear system Ax = b
  // For the inverse, b is a unit vector with 1 at position col
  var b: array<f32, ${N}>;
  for (var i: u32 = 0u; i < N; i = i + 1u) {
    b[i] = 0.0;
  }
  b[col] = 1.0;
  
  // Create permutation array for pivoting
  var perm: array<u32, ${N}>;
  for (var i: u32 = 0u; i < N; i = i + 1u) {
    perm[i] = i;
  }
  
  // Perform LU decomposition with partial pivoting
  for (var k: u32 = 0u; k < N - 1u; k = k + 1u) {
    // Find pivot
    var maxIndex = k;
    var maxVal = abs(A[perm[k] * N + k]);
    
    for (var i: u32 = k + 1u; i < N; i = i + 1u) {
      let val = abs(A[perm[i] * N + k]);
      if (val > maxVal) {
        maxVal = val;
        maxIndex = i;
      }
    }
    
    // Check for singular matrix
    if (maxVal < ${epsilon}) {
      // Set output to identity matrix
      for (var i: u32 = 0u; i < N; i = i + 1u) {
        outputMatrix[i * N + col] = 0.0;
      }
      outputMatrix[col * N + col] = 1.0;
      return;
    }
    
    // Swap pivot rows
    if (maxIndex != k) {
      let temp = perm[k];
      perm[k] = perm[maxIndex];
      perm[maxIndex] = temp;
    }
    
    // Compute multipliers
    for (var i: u32 = k + 1u; i < N; i = i + 1u) {
      let Aik = A[perm[i] * N + k] / A[perm[k] * N + k];
      A[perm[i] * N + k] = Aik;
      
      // Update remaining submatrix
      for (var j: u32 = k + 1u; j < N; j = j + 1u) {
        A[perm[i] * N + j] = A[perm[i] * N + j] - Aik * A[perm[k] * N + j];
      }
    }
  }
  
  // Solve Ly = b
  var y: array<f32, ${N}>;
  for (var i: u32 = 0u; i < N; i = i + 1u) {
    y[i] = b[perm[i]];
    for (var j: u32 = 0u; j < i; j = j + 1u) {
      y[i] = y[i] - A[perm[i] * N + j] * y[j];
    }
  }
  
  // Solve Ux = y
  var x: array<f32, ${N}>;
  for (var i: i32 = i32(N) - 1; i >= 0; i = i - 1) {
    let ui = u32(i);
    x[ui] = y[ui];
    for (var j: u32 = ui + 1u; j < N; j = j + 1u) {
      x[ui] = x[ui] - A[perm[ui] * N + j] * x[j];
    }
    x[ui] = x[ui] / A[perm[ui] * N + ui];
  }
  
  // Write the solution (one column of the inverse) to the output matrix
  for (var i: u32 = 0u; i < N; i = i + 1u) {
    outputMatrix[i * N + col] = x[i];
  }
}`;
  }
  
  /**
   * Generate element-wise operation shader
   * @param operationType Type of element-wise operation
   * @param shape Shape of input tensors
   * @param options Operation options
   * @returns WGSL shader code
   */
  private generateElementWiseShader(
    operationType: MatrixComputeOperationType,
    shape: readonly number[],
    options: MatrixComputeOptions
  ): string {
    const workgroupSize = options.workgroupSize || 256;
    const useVectorization = options.useVectorization !== false;
    const ndims = shape.length;
    
    // Get operation code
    let opCode: string;
    switch (operationType) {
      case MatrixComputeOperationType.ADD:
        opCode = 'a + b';
        break;
      case MatrixComputeOperationType.SUBTRACT:
        opCode = 'a - b';
        break;
      case MatrixComputeOperationType.MULTIPLY:
        opCode = 'a * b';
        break;
      case MatrixComputeOperationType.DIVIDE:
        opCode = 'a / b';
        break;
      default:
        throw new Error(`Unsupported element-wise operation: ${operationType}`);
    }
    
    if (useVectorization) {
      // Vectorized implementation for better performance
      return /* wgsl */`
// Element-wise operation (${operationType}) using vectorization
// Input: tensors of shape ${shape}
// Output: tensor of same shape

struct Parameters {
  total_elements: u32,
  ${Array.from({length: ndims}, (_, i) => `dim_${i}: u32,`).join('\n  ')}
};

@group(0) @binding(0) var<storage, read> inputA: array<f32>;
@group(0) @binding(1) var<storage, read> inputB: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Parameters;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let vec_idx = idx * 4u;
  
  // Process 4 elements at once
  if (vec_idx + 3u < params.total_elements) {
    // Process 4 elements at once using vec4
    let a = vec4<f32>(
      inputA[vec_idx],
      inputA[vec_idx + 1u],
      inputA[vec_idx + 2u],
      inputA[vec_idx + 3u]
    );
    
    let b = vec4<f32>(
      inputB[vec_idx],
      inputB[vec_idx + 1u],
      inputB[vec_idx + 2u],
      inputB[vec_idx + 3u]
    );
    
    // Apply element-wise operation
    let result = ${opCode.replace(/a/g, 'a').replace(/b/g, 'b')};
    
    // Store results
    output[vec_idx] = result[0];
    output[vec_idx + 1u] = result[1];
    output[vec_idx + 2u] = result[2];
    output[vec_idx + 3u] = result[3];
  } else if (vec_idx < params.total_elements) {
    // Handle remaining elements individually
    for (var i: u32 = 0u; i < 4u && vec_idx + i < params.total_elements; i = i + 1u) {
      let a = inputA[vec_idx + i];
      let b = inputB[vec_idx + i];
      output[vec_idx + i] = ${opCode};
    }
  }
}`;
    } else {
      // Simple implementation without vectorization
      return /* wgsl */`
// Element-wise operation (${operationType})
// Input: tensors of shape ${shape}
// Output: tensor of same shape

struct Parameters {
  total_elements: u32,
  ${Array.from({length: ndims}, (_, i) => `dim_${i}: u32,`).join('\n  ')}
};

@group(0) @binding(0) var<storage, read> inputA: array<f32>;
@group(0) @binding(1) var<storage, read> inputB: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Parameters;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  if (idx < params.total_elements) {
    let a = inputA[idx];
    let b = inputB[idx];
    output[idx] = ${opCode};
  }
}`;
    }
  }
  
  /**
   * Dispose resources
   */
  dispose(): void {
    this.pipelineCache.clear();
    this.bindGroupLayoutCache.clear();
  }
}

/**
 * Create a WebGPU matrix compute operations instance
 * @param backend WebGPU backend
 * @returns WebGPU matrix compute operations instance
 */
export async function createMatrixComputeOperations(
  backend: WebGPUBackend
): Promise<WebGPUMatrixComputeOperations> {
  const matrixComputeOps = new WebGPUMatrixComputeOperations(backend);
  await matrixComputeOps.initialize();
  return matrixComputeOps;
}