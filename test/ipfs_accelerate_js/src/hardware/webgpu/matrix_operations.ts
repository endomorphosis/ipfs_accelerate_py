/**
 * WebGPU Matrix Operations
 * High-performance implementations of matrix operations using WebGPU compute shaders
 */

import { Tensor } from '../../tensor/tensor';
import { WebGPUBackend } from './backend';
import { WebGPUBufferManager } from './buffer_manager';
import { detectBrowserType, BrowserType } from './browser_optimized_operations';
import { loadBrowserShader, getBrowserShaderSync, ShaderType } from './optimizations/browser_shader_loader';

/**
 * Matrix operation configuration options
 */
export interface MatrixOperationOptions {
  /** Workgroup size for compute operations */
  workgroupSize?: number;
  
  /** Tile size for tiled operations */
  tileSize?: number;
  
  /** Whether to use shared memory optimizations */
  useSharedMemory?: boolean;
  
  /** Browser optimization target */
  browserOptimization?: 'chrome' | 'firefox' | 'safari' | 'edge';
  
  /** Whether to use fast math approximations */
  useFastMath?: boolean;
  
  /** Whether to use memory layout optimizations */
  useLayoutOptimizations?: boolean;
}

/**
 * Default matrix operation options
 */
const DEFAULT_OPTIONS: MatrixOperationOptions = {
  workgroupSize: 16,
  tileSize: 16,
  useSharedMemory: true,
  useFastMath: true,
  useLayoutOptimizations: true
};

/**
 * Advanced Matrix Multiplication with optimized compute shaders
 */
export class WebGPUMatrixMultiplication {
  /** WebGPU backend reference */
  private backend: WebGPUBackend;
  
  /** Buffer manager for GPU memory */
  private bufferManager: WebGPUBufferManager;
  
  /** GPU device reference */
  private device: GPUDevice;
  
  /** Cache for compute pipelines */
  private pipelineCache: Map<string, GPUComputePipeline> = new Map();
  
  /** Cache for bind group layouts */
  private bindGroupLayoutCache: Map<string, GPUBindGroupLayout> = new Map();
  
  /**
   * Constructor
   * @param backend WebGPU backend instance
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
   * Execute matrix multiplication with optimized compute shaders
   * C = A × B
   * @param a Matrix A (M×K)
   * @param b Matrix B (K×N)
   * @param options Operation configuration options
   * @returns Result matrix C (M×N)
   */
  async multiply<T>(a: Tensor<T>, b: Tensor<T>, options: MatrixOperationOptions = {}): Promise<Tensor<T>> {
    const opts = { ...DEFAULT_OPTIONS, ...options };
    
    // Validate input tensor shapes
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error(`Matrix multiplication requires 2D tensors, got shapes: ${a.shape} and ${b.shape}`);
    }
    
    const M = a.shape[0]; // Rows in A
    const K = a.shape[1]; // Cols in A = Rows in B
    const N = b.shape[1]; // Cols in B
    
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(`Matrix dimensions mismatch: ${a.shape} and ${b.shape}`);
    }
    
    // Ensure tensors are on GPU
    const aBuffer = await this.uploadTensorIfNeeded(a);
    const bBuffer = await this.uploadTensorIfNeeded(b);
    
    // Create output tensor
    const outputShape = [M, N];
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: a.dataType,
        backend: 'webgpu',
        device: this.backend.id
      }
    );
    
    // Create output buffer
    const outputBuffer = this.bufferManager.createOutputBuffer(
      outputShape,
      a.dataType
    );
    
    // Select the appropriate shader based on matrix dimensions and options
    let shaderCode: string;
    
    // For very large matrices, use the advanced tiled shared memory implementation
    if (M * K * N > 1000000 && opts.useSharedMemory) {
      shaderCode = this.generateAdvancedTiledMatmulShader(M, K, N, opts);
    } 
    // For medium-sized matrices, use the simple tiled implementation
    else if (M * K * N > 10000 && opts.useSharedMemory) {
      shaderCode = this.generateTiledMatmulShader(M, K, N, opts);
    } 
    // For small matrices, use the simple implementation
    else {
      shaderCode = this.generateSimpleMatmulShader(M, K, N, opts);
    }
    
    // Create or get the pipeline
    const pipelineKey = `matmul_${M}_${K}_${N}_${opts.useSharedMemory}_${opts.tileSize}`;
    let pipeline: GPUComputePipeline;
    
    if (this.pipelineCache.has(pipelineKey)) {
      pipeline = this.pipelineCache.get(pipelineKey)!;
    } else {
      // Create shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderCode,
        label: `matmul_shader_${pipelineKey}`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      
      if (this.bindGroupLayoutCache.has('matmul')) {
        bindGroupLayout = this.bindGroupLayoutCache.get('matmul')!;
      } else {
        bindGroupLayout = this.device.createBindGroupLayout({
          entries: [
            { // Matrix A
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Matrix B
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Output matrix
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Uniform dimensions
              binding: 3,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' }
            }
          ],
          label: 'matmul_bind_group_layout'
        });
        
        this.bindGroupLayoutCache.set('matmul', bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `matmul_pipeline_layout_${pipelineKey}`
      });
      
      // Create compute pipeline
      pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `matmul_pipeline_${pipelineKey}`
      });
      
      // Cache pipeline for future use
      this.pipelineCache.set(pipelineKey, pipeline);
    }
    
    // Create uniform buffer for matrix dimensions
    const uniformBuffer = this.device.createBuffer({
      size: 16, // 4 x 4 bytes (u32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'matmul_dimensions'
    });
    
    // Write matrix dimensions
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([M, K, N, opts.tileSize || 16])
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
      label: `matmul_bind_group_${pipelineKey}`
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: `matmul_encoder_${pipelineKey}`
    });
    
    // Compute pass
    const pass = encoder.beginComputePass({
      label: `matmul_pass_${pipelineKey}`
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate workgroup sizes
    // This depends on the shader implementation being used
    let workgroupSizeX, workgroupSizeY;
    const tileSize = opts.tileSize || 16;
    
    if (opts.useSharedMemory) {
      // For tiled implementations, we dispatch one workgroup per tile
      workgroupSizeX = Math.ceil(M / tileSize);
      workgroupSizeY = Math.ceil(N / tileSize);
    } else {
      // For simple implementation, default to 16x16 workgroups
      const workgroupSize = opts.workgroupSize || 16;
      workgroupSizeX = Math.ceil(M / workgroupSize);
      workgroupSizeY = Math.ceil(N / workgroupSize);
    }
    
    pass.dispatchWorkgroups(workgroupSizeX, workgroupSizeY);
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    return outputTensor;
  }
  
  /**
   * Execute batched matrix multiplication
   * C[i] = A[i] × B[i] for each i in the batch
   * @param a Batch of matrices A, shape [batchSize, M, K]
   * @param b Batch of matrices B, shape [batchSize, K, N]
   * @param options Operation options
   * @returns Batch of result matrices C, shape [batchSize, M, N]
   */
  async batchMultiply<T>(a: Tensor<T>, b: Tensor<T>, options: MatrixOperationOptions = {}): Promise<Tensor<T>> {
    const opts = { ...DEFAULT_OPTIONS, ...options };
    
    // Validate input tensor shapes
    if (a.shape.length !== 3 || b.shape.length !== 3) {
      throw new Error(`Batch matrix multiplication requires 3D tensors, got shapes: ${a.shape} and ${b.shape}`);
    }
    
    if (a.shape[0] !== b.shape[0]) {
      throw new Error(`Batch sizes must match: ${a.shape[0]} vs ${b.shape[0]}`);
    }
    
    if (a.shape[2] !== b.shape[1]) {
      throw new Error(`Matrix dimensions mismatch: ${a.shape} and ${b.shape}`);
    }
    
    const batchSize = a.shape[0];
    const M = a.shape[1]; // Rows in each A
    const K = a.shape[2]; // Cols in each A = Rows in each B
    const N = b.shape[2]; // Cols in each B
    
    // Ensure tensors are on GPU
    const aBuffer = await this.uploadTensorIfNeeded(a);
    const bBuffer = await this.uploadTensorIfNeeded(b);
    
    // Create output tensor
    const outputShape = [batchSize, M, N];
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: a.dataType,
        backend: 'webgpu',
        device: this.backend.id
      }
    );
    
    // Create output buffer
    const outputBuffer = this.bufferManager.createOutputBuffer(
      outputShape,
      a.dataType
    );
    
    // Generate batch matmul shader
    const shaderCode = this.generateBatchMatmulShader(batchSize, M, K, N, opts);
    
    // Create or get the pipeline
    const pipelineKey = `batch_matmul_${batchSize}_${M}_${K}_${N}`;
    let pipeline: GPUComputePipeline;
    
    if (this.pipelineCache.has(pipelineKey)) {
      pipeline = this.pipelineCache.get(pipelineKey)!;
    } else {
      // Create shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderCode,
        label: `batch_matmul_shader_${pipelineKey}`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      
      if (this.bindGroupLayoutCache.has('batch_matmul')) {
        bindGroupLayout = this.bindGroupLayoutCache.get('batch_matmul')!;
      } else {
        bindGroupLayout = this.device.createBindGroupLayout({
          entries: [
            { // Batch of matrices A
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Batch of matrices B
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Output batch of matrices C
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Uniform dimensions
              binding: 3,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' }
            }
          ],
          label: 'batch_matmul_bind_group_layout'
        });
        
        this.bindGroupLayoutCache.set('batch_matmul', bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `batch_matmul_pipeline_layout_${pipelineKey}`
      });
      
      // Create compute pipeline
      pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `batch_matmul_pipeline_${pipelineKey}`
      });
      
      // Cache pipeline for future use
      this.pipelineCache.set(pipelineKey, pipeline);
    }
    
    // Create uniform buffer for dimensions
    const uniformBuffer = this.device.createBuffer({
      size: 16, // 4 x 4 bytes (u32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'batch_matmul_dimensions'
    });
    
    // Write dimensions
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([batchSize, M, K, N])
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
      label: `batch_matmul_bind_group_${pipelineKey}`
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: `batch_matmul_encoder_${pipelineKey}`
    });
    
    // Compute pass
    const pass = encoder.beginComputePass({
      label: `batch_matmul_pass_${pipelineKey}`
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate workgroup counts
    // Dispatch one workgroup per batch item and divide the matrix by workgroup size
    const workgroupSize = opts.workgroupSize || 16;
    const workgroupCountX = Math.ceil(M / workgroupSize);
    const workgroupCountY = Math.ceil(N / workgroupSize);
    const workgroupCountZ = batchSize;
    
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    return outputTensor;
  }
  
  /**
   * Execute matrix convolution for 2D data (for neural networks)
   * @param input Input tensor of shape [batchSize, inputHeight, inputWidth, inputChannels]
   * @param filters Filter tensor of shape [filterHeight, filterWidth, inputChannels, outputChannels]
   * @param strides Stride of the convolution [strideHeight, strideWidth]
   * @param padding Padding mode: 'same' or 'valid'
   * @param options Operation options
   * @returns Output tensor of shape [batchSize, outputHeight, outputWidth, outputChannels]
   */
  async conv2d<T>(
    input: Tensor<T>, 
    filters: Tensor<T>, 
    strides: [number, number] = [1, 1], 
    padding: 'same' | 'valid' = 'valid',
    options: MatrixOperationOptions = {}
  ): Promise<Tensor<T>> {
    const opts = { ...DEFAULT_OPTIONS, ...options };
    
    // Validate input tensor shapes
    if (input.shape.length !== 4) {
      throw new Error(`Conv2D requires input tensor of shape [batch, height, width, channels], got: ${input.shape}`);
    }
    
    if (filters.shape.length !== 4) {
      throw new Error(`Conv2D requires filter tensor of shape [filterHeight, filterWidth, inputChannels, outputChannels], got: ${filters.shape}`);
    }
    
    if (input.shape[3] !== filters.shape[2]) {
      throw new Error(`Input channels (${input.shape[3]}) must match filter input channels (${filters.shape[2]})`);
    }
    
    const batchSize = input.shape[0];
    const inputHeight = input.shape[1];
    const inputWidth = input.shape[2];
    const inputChannels = input.shape[3];
    
    const filterHeight = filters.shape[0];
    const filterWidth = filters.shape[1];
    const outputChannels = filters.shape[3];
    
    const strideHeight = strides[0];
    const strideWidth = strides[1];
    
    // Calculate output dimensions based on padding mode
    let outputHeight: number;
    let outputWidth: number;
    let padTop: number = 0;
    let padLeft: number = 0;
    
    if (padding === 'same') {
      outputHeight = Math.ceil(inputHeight / strideHeight);
      outputWidth = Math.ceil(inputWidth / strideWidth);
      
      // Calculate padding needed for 'same' output
      const padHeightTotal = Math.max(0, (outputHeight - 1) * strideHeight + filterHeight - inputHeight);
      const padWidthTotal = Math.max(0, (outputWidth - 1) * strideWidth + filterWidth - inputWidth);
      
      // Divide padding evenly (with extra padding at the bottom/right if needed)
      padTop = Math.floor(padHeightTotal / 2);
      padLeft = Math.floor(padWidthTotal / 2);
    } else { // 'valid' padding
      outputHeight = Math.ceil((inputHeight - filterHeight + 1) / strideHeight);
      outputWidth = Math.ceil((inputWidth - filterWidth + 1) / strideWidth);
    }
    
    // Ensure dimensions are valid
    if (outputHeight <= 0 || outputWidth <= 0) {
      throw new Error(`Invalid output dimensions: ${outputHeight}x${outputWidth}`);
    }
    
    // Ensure tensors are on GPU
    const inputBuffer = await this.uploadTensorIfNeeded(input);
    const filtersBuffer = await this.uploadTensorIfNeeded(filters);
    
    // Create output tensor
    const outputShape = [batchSize, outputHeight, outputWidth, outputChannels];
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
    
    // Generate convolution shader
    const shaderCode = this.generateConv2dShader(
      batchSize, inputHeight, inputWidth, inputChannels,
      filterHeight, filterWidth, outputChannels,
      strideHeight, strideWidth, 
      padTop, padLeft,
      opts
    );
    
    // Create or get the pipeline
    const pipelineKey = `conv2d_${batchSize}_${inputHeight}_${inputWidth}_${inputChannels}_${filterHeight}_${filterWidth}_${outputChannels}_${strideHeight}_${strideWidth}_${padding}`;
    let pipeline: GPUComputePipeline;
    
    if (this.pipelineCache.has(pipelineKey)) {
      pipeline = this.pipelineCache.get(pipelineKey)!;
    } else {
      // Create shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderCode,
        label: `conv2d_shader_${pipelineKey}`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      
      if (this.bindGroupLayoutCache.has('conv2d')) {
        bindGroupLayout = this.bindGroupLayoutCache.get('conv2d')!;
      } else {
        bindGroupLayout = this.device.createBindGroupLayout({
          entries: [
            { // Input tensor
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Filter tensor
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Output tensor
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Uniform dimensions
              binding: 3,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' }
            }
          ],
          label: 'conv2d_bind_group_layout'
        });
        
        this.bindGroupLayoutCache.set('conv2d', bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `conv2d_pipeline_layout_${pipelineKey}`
      });
      
      // Create compute pipeline
      pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `conv2d_pipeline_${pipelineKey}`
      });
      
      // Cache pipeline for future use
      this.pipelineCache.set(pipelineKey, pipeline);
    }
    
    // Create uniform buffer for dimensions and parameters
    const uniformBuffer = this.device.createBuffer({
      size: 48, // 12 x 4 bytes (u32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'conv2d_params'
    });
    
    // Write convolution parameters
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        batchSize, inputHeight, inputWidth, inputChannels,
        filterHeight, filterWidth, outputChannels,
        outputHeight, outputWidth,
        strideHeight, strideWidth,
        padding === 'same' ? 1 : 0
      ])
    );
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: filtersBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } }
      ],
      label: `conv2d_bind_group_${pipelineKey}`
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: `conv2d_encoder_${pipelineKey}`
    });
    
    // Compute pass
    const pass = encoder.beginComputePass({
      label: `conv2d_pass_${pipelineKey}`
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate workgroup counts
    // Each workgroup processes a block of the output tensor
    const workgroupSize = 8; // 8x8 is common for convolution
    const workgroupCountX = Math.ceil(outputWidth / workgroupSize);
    const workgroupCountY = Math.ceil(outputHeight / workgroupSize);
    const workgroupCountZ = batchSize * outputChannels;
    
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    return outputTensor;
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
   * Generate a simple direct matrix multiplication shader
   * @param M Rows in A
   * @param K Columns in A / Rows in B
   * @param N Columns in B 
   * @param options Shader configuration options
   * @returns WGSL shader code
   */
  private generateSimpleMatmulShader(
    M: number, 
    K: number, 
    N: number, 
    options: MatrixOperationOptions
  ): string {
    // Try to use browser-specific shaders if enabled
    if (options.browserOptimization !== 'none') {
      try {
        // Detect browser or use provided browser override
        const browserType = options.browserOptimization ?
          this.getBrowserTypeFromString(options.browserOptimization) :
          detectBrowserType();
        
        // Get browser-specific shader from our optimized shader collection
        return getBrowserShaderSync(ShaderType.MATMUL, browserType);
      } catch (error) {
        console.warn('Failed to load browser-specific shader, falling back to simple generic implementation', error);
      }
    }
    const workgroupSize = options.workgroupSize || 16;
    
    return /* wgsl */`
// Binding group layout:
// binding 0: Input matrix A (M×K)
// binding 1: Input matrix B (K×N)
// binding 2: Output matrix C (M×N)
// binding 3: Uniform buffer with matrix dimensions

struct Dimensions {
  M: u32,  // Rows in A
  K: u32,  // Columns in A / Rows in B
  N: u32,  // Columns in B
  tileSize: u32, // Not used in simple implementation
};

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: Dimensions;

@compute @workgroup_size(${workgroupSize}, ${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  
  // Bounds check
  if (row >= dimensions.M || col >= dimensions.N) {
    return;
  }
  
  var sum: f32 = 0.0;
  
  // Compute the dot product for this output element
  for (var k: u32 = 0u; k < dimensions.K; k = k + 1u) {
    let a_index = row * dimensions.K + k;
    let b_index = k * dimensions.N + col;
    sum = sum + matrix_a[a_index] * matrix_b[b_index];
  }
  
  // Write the result
  let c_index = row * dimensions.N + col;
  matrix_c[c_index] = sum;
}`;
  }
  
  /**
   * Generate a tiled matrix multiplication shader using workgroup shared memory
   * @param M Rows in A
   * @param K Columns in A / Rows in B
   * @param N Columns in B 
   * @param options Shader configuration options
   * @returns WGSL shader code
   */
  private generateTiledMatmulShader(
    M: number, 
    K: number, 
    N: number, 
    options: MatrixOperationOptions
  ): string {
    // Try to use browser-specific shaders if enabled
    if (options.browserOptimization !== 'none') {
      try {
        // Detect browser or use provided browser override
        const browserType = options.browserOptimization ?
          this.getBrowserTypeFromString(options.browserOptimization) :
          detectBrowserType();
        
        // Get browser-specific shader from our optimized shader collection
        return getBrowserShaderSync(ShaderType.MATMUL, browserType);
      } catch (error) {
        console.warn('Failed to load browser-specific shader, falling back to generic implementation', error);
      }
    }
    
    // Fall back to generic implementation if browser-specific shader fails or is disabled
    const tileSize = options.tileSize || 16;
    
    return /* wgsl */`
// Binding group layout:
// binding 0: Input matrix A (M×K)
// binding 1: Input matrix B (K×N)
// binding 2: Output matrix C (M×N)
// binding 3: Uniform buffer with matrix dimensions

struct Dimensions {
  M: u32,  // Rows in A
  K: u32,  // Columns in A / Rows in B
  N: u32,  // Columns in B
  tileSize: u32, // Tile size for shared memory
};

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: Dimensions;

// Shared memory for tiled multiplication
var<workgroup> tile_a: array<f32, ${tileSize * tileSize}>;
var<workgroup> tile_b: array<f32, ${tileSize * tileSize}>;

@compute @workgroup_size(${tileSize}, ${tileSize})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let row = workgroup_id.x * ${tileSize} + local_id.x;
  let col = workgroup_id.y * ${tileSize} + local_id.y;
  
  // Initialize accumulator
  var sum: f32 = 0.0;
  
  // Loop over tiles
  let num_tiles = (dimensions.K + ${tileSize} - 1) / ${tileSize};
  
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    // Load tile from matrix A
    let a_row = row;
    let a_col = t * ${tileSize} + local_id.y;
    
    if (a_row < dimensions.M && a_col < dimensions.K) {
      tile_a[local_id.x * ${tileSize} + local_id.y] = matrix_a[a_row * dimensions.K + a_col];
    } else {
      tile_a[local_id.x * ${tileSize} + local_id.y] = 0.0;
    }
    
    // Load tile from matrix B
    let b_row = t * ${tileSize} + local_id.x;
    let b_col = col;
    
    if (b_row < dimensions.K && b_col < dimensions.N) {
      tile_b[local_id.x * ${tileSize} + local_id.y] = matrix_b[b_row * dimensions.N + b_col];
    } else {
      tile_b[local_id.x * ${tileSize} + local_id.y] = 0.0;
    }
    
    // Synchronize to ensure tiles are loaded
    workgroupBarrier();
    
    // Perform multiplication within the tile
    for (var k: u32 = 0u; k < ${tileSize}; k = k + 1u) {
      sum = sum + tile_a[local_id.x * ${tileSize} + k] * tile_b[k * ${tileSize} + local_id.y];
    }
    
    // Synchronize before loading the next tiles
    workgroupBarrier();
  }
  
  // Write the result
  if (row < dimensions.M && col < dimensions.N) {
    matrix_c[row * dimensions.N + col] = sum;
  }
}`;
  }
  
  /**
   * Generate an advanced tiled matrix multiplication shader with optimizations
   * @param M Rows in A
   * @param K Columns in A / Rows in B
   * @param N Columns in B 
   * @param options Shader configuration options
   * @returns WGSL shader code
   */
  private generateAdvancedTiledMatmulShader(
    M: number, 
    K: number, 
    N: number, 
    options: MatrixOperationOptions
  ): string {
    // Try to use browser-specific shaders if enabled
    if (options.browserOptimization !== 'none') {
      try {
        // Detect browser or use provided browser override
        const browserType = options.browserOptimization ?
          this.getBrowserTypeFromString(options.browserOptimization) :
          detectBrowserType();
        
        // Get browser-specific shader from our optimized shader collection
        return getBrowserShaderSync(ShaderType.MATMUL, browserType);
      } catch (error) {
        console.warn('Failed to load browser-specific shader, falling back to advanced generic implementation', error);
      }
    }
    const tileSize = options.tileSize || 16;
    // For advanced implementation, we use a micro-tile approach where each thread
    // computes multiple output elements for better instruction-level parallelism
    const microTileSize = 4; // Each thread computes a 4x4 block
    
    // Adjusted workgroup size for micro-tiling
    const workgroupSize = tileSize / microTileSize;
    
    return /* wgsl */`
// Binding group layout:
// binding 0: Input matrix A (M×K)
// binding 1: Input matrix B (K×N)
// binding 2: Output matrix C (M×N)
// binding 3: Uniform buffer with matrix dimensions

struct Dimensions {
  M: u32,  // Rows in A
  K: u32,  // Columns in A / Rows in B
  N: u32,  // Columns in B
  tileSize: u32, // Tile size for shared memory
};

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: Dimensions;

// Shared memory for tiled multiplication
var<workgroup> tile_a: array<f32, ${tileSize * tileSize}>;
var<workgroup> tile_b: array<f32, ${tileSize * tileSize}>;

@compute @workgroup_size(${workgroupSize}, ${workgroupSize})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  // Calculate base position for this thread
  let baseRow = workgroup_id.x * ${tileSize} + local_id.x * ${microTileSize};
  let baseCol = workgroup_id.y * ${tileSize} + local_id.y * ${microTileSize};
  
  // Initialize accumulators for the micro-tile (4x4 block)
  var sums: array<array<f32, ${microTileSize}>, ${microTileSize}>;
  
  // Initialize the accumulators to zero
  for (var mi = 0u; mi < ${microTileSize}u; mi = mi + 1u) {
    for (var mj = 0u; mj < ${microTileSize}u; mj = mj + 1u) {
      sums[mi][mj] = 0.0;
    }
  }
  
  // Loop over tiles in K dimension
  let num_tiles = (dimensions.K + ${tileSize} - 1) / ${tileSize};
  
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    // Collaboratively load tiles from global memory to shared memory
    // Each thread loads ${microTileSize}x${microTileSize} elements
    
    // Load elements from matrix A into shared memory
    for (var mi = 0u; mi < ${microTileSize}u; mi = mi + 1u) {
      for (var mj = 0u; mj < ${microTileSize}u; mj = mj + 1u) {
        let a_row = baseRow + mi;
        let a_col = t * ${tileSize} + local_id.y * ${microTileSize} + mj;
        
        let shared_idx = (local_id.x * ${microTileSize} + mi) * ${tileSize} + 
                         (local_id.y * ${microTileSize} + mj);
        
        if (a_row < dimensions.M && a_col < dimensions.K) {
          tile_a[shared_idx] = matrix_a[a_row * dimensions.K + a_col];
        } else {
          tile_a[shared_idx] = 0.0;
        }
      }
    }
    
    // Load elements from matrix B into shared memory
    for (var mi = 0u; mi < ${microTileSize}u; mi = mi + 1u) {
      for (var mj = 0u; mj < ${microTileSize}u; mj = mj + 1u) {
        let b_row = t * ${tileSize} + local_id.x * ${microTileSize} + mi;
        let b_col = baseCol + mj;
        
        let shared_idx = (local_id.x * ${microTileSize} + mi) * ${tileSize} + 
                         (local_id.y * ${microTileSize} + mj);
        
        if (b_row < dimensions.K && b_col < dimensions.N) {
          tile_b[shared_idx] = matrix_b[b_row * dimensions.N + b_col];
        } else {
          tile_b[shared_idx] = 0.0;
        }
      }
    }
    
    // Synchronize to ensure all threads have loaded their tiles
    workgroupBarrier();
    
    // Compute matrix multiplication for this tile
    // Each thread computes a ${microTileSize}x${microTileSize} micro-tile of the output
    for (var k = 0u; k < ${tileSize}u; k = k + 1u) {
      // Load A values for this iteration
      var a_values: array<f32, ${microTileSize}>;
      for (var mi = 0u; mi < ${microTileSize}u; mi = mi + 1u) {
        a_values[mi] = tile_a[(local_id.x * ${microTileSize} + mi) * ${tileSize} + k];
      }
      
      // Load B values for this iteration
      var b_values: array<f32, ${microTileSize}>;
      for (var mj = 0u; mj < ${microTileSize}u; mj = mj + 1u) {
        b_values[mj] = tile_b[k * ${tileSize} + local_id.y * ${microTileSize} + mj];
      }
      
      // Update accumulators
      for (var mi = 0u; mi < ${microTileSize}u; mi = mi + 1u) {
        for (var mj = 0u; mj < ${microTileSize}u; mj = mj + 1u) {
          sums[mi][mj] = sums[mi][mj] + a_values[mi] * b_values[mj];
        }
      }
    }
    
    // Synchronize before moving to the next tile
    workgroupBarrier();
  }
  
  // Write results to global memory
  for (var mi = 0u; mi < ${microTileSize}u; mi = mi + 1u) {
    for (var mj = 0u; mj < ${microTileSize}u; mj = mj + 1u) {
      let row = baseRow + mi;
      let col = baseCol + mj;
      
      if (row < dimensions.M && col < dimensions.N) {
        matrix_c[row * dimensions.N + col] = sums[mi][mj];
      }
    }
  }
}`;
  }
  
  /**
   * Generate a batch matrix multiplication shader
   * @param batchSize Number of matrices in the batch
   * @param M Rows in each A
   * @param K Columns in each A / Rows in each B
   * @param N Columns in each B
   * @param options Shader configuration options 
   * @returns WGSL shader code
   */
  private generateBatchMatmulShader(
    batchSize: number,
    M: number,
    K: number,
    N: number,
    options: MatrixOperationOptions
  ): string {
    const workgroupSize = options.workgroupSize || 16;
    
    return /* wgsl */`
// Binding group layout:
// binding 0: Batch of input matrices A [batchSize, M, K]
// binding 1: Batch of input matrices B [batchSize, K, N]
// binding 2: Batch of output matrices C [batchSize, M, N]
// binding 3: Uniform buffer with dimensions

struct Dimensions {
  batchSize: u32,  // Number of matrices in the batch
  M: u32,          // Rows in each A
  K: u32,          // Columns in each A / Rows in each B
  N: u32,          // Columns in each B
};

@group(0) @binding(0) var<storage, read> batch_a: array<f32>;
@group(0) @binding(1) var<storage, read> batch_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> batch_c: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: Dimensions;

@compute @workgroup_size(${workgroupSize}, ${workgroupSize}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  let batch = global_id.z;
  
  // Bounds check
  if (row >= dimensions.M || col >= dimensions.N || batch >= dimensions.batchSize) {
    return;
  }
  
  // Calculate strides for 3D tensors
  let strideA_batch = dimensions.M * dimensions.K;
  let strideB_batch = dimensions.K * dimensions.N;
  let strideC_batch = dimensions.M * dimensions.N;
  
  // Calculate base indices for this batch item
  let baseA = batch * strideA_batch;
  let baseB = batch * strideB_batch;
  let baseC = batch * strideC_batch;
  
  var sum: f32 = 0.0;
  
  // Compute the dot product for this output element
  for (var k: u32 = 0u; k < dimensions.K; k = k + 1u) {
    let a_index = baseA + row * dimensions.K + k;
    let b_index = baseB + k * dimensions.N + col;
    sum = sum + batch_a[a_index] * batch_b[b_index];
  }
  
  // Write the result
  let c_index = baseC + row * dimensions.N + col;
  batch_c[c_index] = sum;
}`;
  }
  
  /**
   * Generate a 2D convolution shader for neural networks
   * @param batchSize Batch size
   * @param inputHeight Input tensor height
   * @param inputWidth Input tensor width
   * @param inputChannels Input tensor channels
   * @param filterHeight Filter height
   * @param filterWidth Filter width
   * @param outputChannels Output tensor channels
   * @param strideHeight Vertical stride
   * @param strideWidth Horizontal stride
   * @param padTop Top padding
   * @param padLeft Left padding
   * @param options Shader configuration options
   * @returns WGSL shader code
   */
  private generateConv2dShader(
    batchSize: number,
    inputHeight: number,
    inputWidth: number,
    inputChannels: number,
    filterHeight: number,
    filterWidth: number,
    outputChannels: number,
    strideHeight: number,
    strideWidth: number,
    padTop: number,
    padLeft: number,
    options: MatrixOperationOptions
  ): string {
    // Use a smaller workgroup size for convolution as each thread does more work
    const workgroupSize = 8;
    
    return /* wgsl */`
// Binding group layout:
// binding 0: Input tensor [batchSize, inputHeight, inputWidth, inputChannels]
// binding 1: Filter tensor [filterHeight, filterWidth, inputChannels, outputChannels]
// binding 2: Output tensor [batchSize, outputHeight, outputWidth, outputChannels]
// binding 3: Uniform buffer with dimensions and parameters

struct Params {
  batchSize: u32,       // Number of items in batch
  inputHeight: u32,     // Input tensor height
  inputWidth: u32,      // Input tensor width
  inputChannels: u32,   // Input tensor channels
  filterHeight: u32,    // Filter height
  filterWidth: u32,     // Filter width
  outputChannels: u32,  // Output tensor channels
  outputHeight: u32,    // Output tensor height
  outputWidth: u32,     // Output tensor width
  strideHeight: u32,    // Vertical stride
  strideWidth: u32,     // Horizontal stride
  useSamePadding: u32,  // 1 for 'same' padding, 0 for 'valid'
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> filter: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Helper function to compute the input index
fn get_input_index(b: u32, h: i32, w: i32, c: u32) -> u32 {
  // Get tensor dimensions
  let height = params.inputHeight;
  let width = params.inputWidth;
  let channels = params.inputChannels;
  
  // Calculate strides
  let stride_b = height * width * channels;
  let stride_h = width * channels;
  let stride_w = channels;
  
  return b * stride_b + u32(h) * stride_h + u32(w) * stride_w + c;
}

// Helper function to compute the filter index
fn get_filter_index(h: u32, w: u32, ic: u32, oc: u32) -> u32 {
  // Get filter dimensions
  let height = params.filterHeight;
  let width = params.filterWidth;
  let inChannels = params.inputChannels;
  
  // Calculate strides
  let stride_h = width * inChannels * params.outputChannels;
  let stride_w = inChannels * params.outputChannels;
  let stride_ic = params.outputChannels;
  
  return h * stride_h + w * stride_w + ic * stride_ic + oc;
}

// Helper function to compute the output index
fn get_output_index(b: u32, h: u32, w: u32, c: u32) -> u32 {
  // Get tensor dimensions
  let height = params.outputHeight;
  let width = params.outputWidth;
  let channels = params.outputChannels;
  
  // Calculate strides
  let stride_b = height * width * channels;
  let stride_h = width * channels;
  let stride_w = channels;
  
  return b * stride_b + h * stride_h + w * stride_w + c;
}

@compute @workgroup_size(${workgroupSize}, ${workgroupSize}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // Each thread computes one output element
  let x = global_id.x; // Output width dimension
  let y = global_id.y; // Output height dimension
  let z = global_id.z; // Batch * output channels
  
  // Decode z into batch and output channel
  let batch = z / params.outputChannels;
  let outChannel = z % params.outputChannels;
  
  // Bounds check
  if (x >= params.outputWidth || y >= params.outputHeight || batch >= params.batchSize) {
    return;
  }
  
  // Compute the output position in the input tensor
  let inY = i32(y * params.strideHeight) - ${padTop};
  let inX = i32(x * params.strideWidth) - ${padLeft};
  
  // Compute convolution
  var sum: f32 = 0.0;
  
  for (var fh: u32 = 0u; fh < params.filterHeight; fh = fh + 1u) {
    for (var fw: u32 = 0u; fw < params.filterWidth; fw = fw + 1u) {
      // Calculate input position
      let iy = inY + i32(fh);
      let ix = inX + i32(fw);
      
      // Skip if outside input bounds
      if (iy >= 0 && iy < i32(params.inputHeight) && ix >= 0 && ix < i32(params.inputWidth)) {
        for (var ic: u32 = 0u; ic < params.inputChannels; ic = ic + 1u) {
          // Get input and filter values
          let inputVal = input[get_input_index(batch, iy, ix, ic)];
          let filterVal = filter[get_filter_index(fh, fw, ic, outChannel)];
          
          // Accumulate result
          sum = sum + inputVal * filterVal;
        }
      }
    }
  }
  
  // Write output
  output[get_output_index(batch, y, x, outChannel)] = sum;
}`;
  }
  
  /**
   * Convert browser name string to BrowserType enum
   * @param browserName Browser name
   * @returns Browser type enum value
   */
  private getBrowserTypeFromString(browserName: string): BrowserType {
    switch (browserName.toLowerCase()) {
      case 'chrome':
        return BrowserType.CHROME;
      case 'firefox':
        return BrowserType.FIREFOX;
      case 'safari':
        return BrowserType.SAFARI;
      case 'edge':
        return BrowserType.EDGE;
      default:
        return BrowserType.UNKNOWN;
    }
  }
  
  /**
   * Dispose resources used by this class
   */
  dispose(): void {
    // Clear caches to free memory
    this.pipelineCache.clear();
    this.bindGroupLayoutCache.clear();
  }
}

/**
 * Export the WebGPU Matrix Operations functionality
 */
export async function createMatrixOperations(backend: WebGPUBackend): Promise<WebGPUMatrixMultiplication> {
  const matrixOps = new WebGPUMatrixMultiplication(backend);
  return matrixOps;
}