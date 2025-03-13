/**
 * WebGPU Backend Implementation
 * Provides hardware acceleration for tensor operations using WebGPU
 */

import { HardwareBackend, HardwareCapabilities } from '../interfaces/hardware_backend';
import { Tensor } from '../../tensor/tensor';
import { WebGPUBufferManager } from './buffer_manager';
import { getShader } from './shaders';

/**
 * WebGPU operation type enumeration
 */
export enum WebGPUOperationType {
  Add = 0,
  Subtract = 1,
  Multiply = 2,
  Divide = 3,
  Power = 4,
  Exp = 5,
  Log = 6,
  Sqrt = 7,
  Relu = 8,
  Sigmoid = 9,
  Tanh = 10
}

/**
 * WebGPU reduction operation type enumeration
 */
export enum WebGPUReductionType {
  Sum = 0,
  Mean = 1,
  Max = 2,
  Min = 3
}

/**
 * Map for tracking tensor-to-GPU-buffer associations
 */
interface TensorBufferMap {
  [tensorId: string]: {
    buffer: GPUBuffer;
    timestamp: number;
  };
}

/**
 * WebGPU backend implementation
 */
export class WebGPUBackend implements HardwareBackend {
  /** Unique identifier for this backend */
  readonly id: string = 'webgpu';
  
  /** Type of backend */
  readonly type: string = 'webgpu';
  
  /** WebGPU adapter */
  private adapter: GPUAdapter | null = null;
  
  /** WebGPU device */
  private device: GPUDevice | null = null;
  
  /** Whether the backend is initialized */
  private initialized: boolean = false;
  
  /** Buffer manager for GPU memory */
  private bufferManager: WebGPUBufferManager | null = null;
  
  /** Map of shader pipeline cache */
  private pipelineCache: Map<string, GPUComputePipeline> = new Map();
  
  /** Map of shader bind group layout cache */
  private bindGroupLayoutCache: Map<string, GPUBindGroupLayout> = new Map();
  
  /** Map of tensor to GPU buffer */
  private tensorBuffers: TensorBufferMap = {};
  
  /** WebGPU backend capabilities */
  readonly capabilities: HardwareCapabilities = {
    maxDimensions: 4,
    maxMatrixSize: 16384,
    supportedDataTypes: ['float32'],
    supportsAsync: true,
    supportedOperations: {
      basicArithmetic: true,
      matrixMultiplication: true,
      convolution: false,
      reduction: true,
      activation: true
    }
  };
  
  /**
   * Constructor
   */
  constructor() {
    // Generate a unique ID
    this.id = `webgpu_${Date.now()}`;
  }
  
  /**
   * Check if WebGPU is available in the current environment
   */
  get isAvailable(): boolean {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
  }
  
  /**
   * Initialize the WebGPU backend
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    
    if (!this.isAvailable) {
      throw new Error('WebGPU is not available in this environment');
    }
    
    try {
      // Request adapter
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });
      
      if (!this.adapter) {
        throw new Error('Failed to get WebGPU adapter');
      }
      
      // Request device
      this.device = await this.adapter.requestDevice({
        label: 'IPFS Accelerate WebGPU Device'
      });
      
      // Create buffer manager
      this.bufferManager = new WebGPUBufferManager(this.device);
      
      // Update capabilities with adapter limits
      this.updateCapabilities();
      
      this.initialized = true;
      
      console.log('WebGPU backend initialized successfully');
    } catch (error) {
      console.error('Failed to initialize WebGPU backend:', error);
      throw error;
    }
  }
  
  /**
   * Check if the backend is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }
  
  /**
   * Update capabilities based on adapter information
   */
  private updateCapabilities(): void {
    if (!this.adapter) {
      return;
    }
    
    // Update capabilities with adapter limits
    const limits = this.adapter.limits;
    this.capabilities.maxMatrixSize = Math.min(
      limits.maxComputeWorkgroupSizeX,
      16384
    );
    
    // Update available memory if possible
    if ('maxBufferSize' in limits) {
      this.capabilities.availableMemory = limits.maxBufferSize as number;
    }
  }
  
  /**
   * Allocate a tensor on the GPU
   * @param tensor Tensor to allocate
   */
  async allocateTensor<T>(tensor: Tensor<T>): Promise<void> {
    if (!this.initialized || !this.bufferManager || !this.device) {
      throw new Error('WebGPU backend not initialized');
    }
    
    // Generate a unique tensor ID
    const tensorId = this.getTensorId(tensor);
    
    // If already allocated, do nothing
    if (tensorId in this.tensorBuffers) {
      // Update timestamp
      this.tensorBuffers[tensorId].timestamp = Date.now();
      return;
    }
    
    // Upload tensor data to GPU
    const buffer = await this.bufferManager.uploadTensor(tensor);
    
    // Store in tensor buffer map
    this.tensorBuffers[tensorId] = {
      buffer,
      timestamp: Date.now()
    };
  }
  
  /**
   * Release a tensor from the GPU
   * @param tensor Tensor to release
   */
  releaseTensor<T>(tensor: Tensor<T>): void {
    if (!this.initialized || !this.bufferManager) {
      return;
    }
    
    const tensorId = this.getTensorId(tensor);
    
    if (tensorId in this.tensorBuffers) {
      this.bufferManager.releaseTensor(tensor);
      delete this.tensorBuffers[tensorId];
    }
  }
  
  /**
   * Generate a unique tensor ID
   * @param tensor Tensor to generate ID for
   * @returns Unique tensor ID
   */
  private getTensorId<T>(tensor: Tensor<T>): string {
    return `tensor_${tensor.shape.join('x')}_${tensor.dataType}_${tensor.backend}`;
  }
  
  /**
   * Get a cached compute pipeline or create a new one
   * @param operationName Name of the operation
   * @param shaderCode WGSL shader code
   * @returns Compute pipeline
   */
  private getOrCreatePipeline(
    operationName: string,
    shaderCode: string
  ): GPUComputePipeline {
    if (!this.device) {
      throw new Error('WebGPU backend not initialized');
    }
    
    const cacheKey = `${operationName}`;
    
    // Check cache first
    if (this.pipelineCache.has(cacheKey)) {
      return this.pipelineCache.get(cacheKey)!;
    }
    
    // Create a new shader module
    const shaderModule = this.device.createShaderModule({
      code: shaderCode,
      label: `${operationName}_shader`
    });
    
    // Create bind group layout based on operation
    let bindGroupLayout: GPUBindGroupLayout;
    
    if (this.bindGroupLayoutCache.has(operationName)) {
      bindGroupLayout = this.bindGroupLayoutCache.get(operationName)!;
    } else {
      // Create appropriate bind group layout based on operation
      switch (operationName) {
        case 'elementwise': {
          bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
              { // Input buffer A
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' }
              },
              { // Input buffer B
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' }
              },
              { // Output buffer
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' }
              },
              { // Uniform parameters
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' }
              }
            ],
            label: `${operationName}_bind_group_layout`
          });
          break;
        }
        case 'matmul': {
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
            label: `${operationName}_bind_group_layout`
          });
          break;
        }
        case 'transpose': {
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
              { // Uniform dimensions
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' }
              }
            ],
            label: `${operationName}_bind_group_layout`
          });
          break;
        }
        case 'softmax':
        case 'reduction': {
          bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
              { // Input tensor
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' }
              },
              { // Output tensor
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' }
              },
              { // Uniform parameters
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' }
              }
            ],
            label: `${operationName}_bind_group_layout`
          });
          break;
        }
        default: {
          throw new Error(`Unsupported operation: ${operationName}`);
        }
      }
      
      // Cache the bind group layout
      this.bindGroupLayoutCache.set(operationName, bindGroupLayout);
    }
    
    // Create pipeline layout
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
      label: `${operationName}_pipeline_layout`
    });
    
    // Create compute pipeline
    const pipeline = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      },
      label: `${operationName}_pipeline`
    });
    
    // Cache the pipeline
    this.pipelineCache.set(cacheKey, pipeline);
    
    return pipeline;
  }
  
  /**
   * Execute tensor addition
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async add<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    return this.elementwiseOperation(a, b, WebGPUOperationType.Add);
  }
  
  /**
   * Execute tensor subtraction
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async subtract<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    return this.elementwiseOperation(a, b, WebGPUOperationType.Subtract);
  }
  
  /**
   * Execute tensor multiplication (element-wise)
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async multiply<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    return this.elementwiseOperation(a, b, WebGPUOperationType.Multiply);
  }
  
  /**
   * Execute tensor division
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async divide<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    return this.elementwiseOperation(a, b, WebGPUOperationType.Divide);
  }
  
  /**
   * Execute a generic element-wise operation
   * @param a First tensor
   * @param b Second tensor (optional for unary operations)
   * @param operation Operation type
   * @param alpha Optional alpha parameter
   * @param beta Optional beta parameter
   * @returns Resulting tensor
   */
  private async elementwiseOperation<T>(
    a: Tensor<T>, 
    b: Tensor<T> | null = null,
    operation: WebGPUOperationType,
    alpha: number = 1.0,
    beta: number = 0.0
  ): Promise<Tensor<T>> {
    if (!this.initialized || !this.device || !this.bufferManager) {
      throw new Error('WebGPU backend not initialized');
    }
    
    // Ensure tensors are allocated on GPU
    await this.allocateTensor(a);
    if (b) {
      await this.allocateTensor(b);
    }
    
    // Get shader
    const shader = getShader('elementwise');
    
    // Create pipeline
    const pipeline = this.getOrCreatePipeline('elementwise', shader);
    
    // Get tensor buffers
    const tensorIdA = this.getTensorId(a);
    const bufferA = this.tensorBuffers[tensorIdA].buffer;
    
    let bufferB: GPUBuffer;
    if (b) {
      const tensorIdB = this.getTensorId(b);
      bufferB = this.tensorBuffers[tensorIdB].buffer;
    } else {
      // For unary operations, use the same buffer for input B
      bufferB = bufferA;
    }
    
    // Calculate output shape
    const outputShape = [...a.shape];
    
    // Create output tensor
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: a.dataType,
        backend: 'webgpu',
        device: this.id
      }
    );
    
    // Create output buffer
    const outputBuffer = this.bufferManager.createOutputBuffer(
      outputShape,
      a.dataType
    );
    
    // Create uniform buffer with operation parameters
    const uniformBuffer = this.device.createBuffer({
      size: 24, // 6 x 4 bytes (u32, u32, u32, u32, f32, f32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'elementwise_params'
    });
    
    // Calculate dimensions
    const outputSize = outputShape.reduce((a, b) => a * b, 1);
    
    // Write uniform data
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        operation,          // Operation type
        a.shape.length,     // Dimension count
        a.data.length,      // Input length
        outputSize,         // Output size
      ])
    );
    
    // Write alpha and beta
    this.device.queue.writeBuffer(
      uniformBuffer,
      16, // Offset after the 4 uint32 values
      new Float32Array([alpha, beta])
    );
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } }
      ],
      label: 'elementwise_bind_group'
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: 'elementwise_encoder'
    });
    
    // Compute pass
    const pass = encoder.beginComputePass({
      label: 'elementwise_pass'
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate workgroup count (divide by 256 and round up)
    const workgroupCount = Math.ceil(outputSize / 256);
    pass.dispatchWorkgroups(workgroupCount);
    
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result to the output tensor
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    return outputTensor;
  }
  
  /**
   * Execute matrix multiplication
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async matmul<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (!this.initialized || !this.device || !this.bufferManager) {
      throw new Error('WebGPU backend not initialized');
    }
    
    // Validate tensor shapes for matrix multiplication
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error('Matrix multiplication requires 2D tensors');
    }
    
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(`Matrix dimensions mismatch: ${a.shape} and ${b.shape}`);
    }
    
    // Ensure tensors are allocated on GPU
    await this.allocateTensor(a);
    await this.allocateTensor(b);
    
    // Get shader
    const shader = getShader('matmul');
    
    // Create pipeline
    const pipeline = this.getOrCreatePipeline('matmul', shader);
    
    // Get tensor buffers
    const tensorIdA = this.getTensorId(a);
    const bufferA = this.tensorBuffers[tensorIdA].buffer;
    
    const tensorIdB = this.getTensorId(b);
    const bufferB = this.tensorBuffers[tensorIdB].buffer;
    
    // Calculate output shape [M, N]
    const M = a.shape[0];
    const K = a.shape[1];
    const N = b.shape[1];
    const outputShape = [M, N];
    
    // Create output tensor
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: a.dataType,
        backend: 'webgpu',
        device: this.id
      }
    );
    
    // Create output buffer
    const outputBuffer = this.bufferManager.createOutputBuffer(
      outputShape,
      a.dataType
    );
    
    // Create uniform buffer with matrix dimensions
    const uniformBuffer = this.device.createBuffer({
      size: 12, // 3 x 4 bytes (u32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'matmul_dimensions'
    });
    
    // Write matrix dimensions
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([M, K, N])
    );
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } }
      ],
      label: 'matmul_bind_group'
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: 'matmul_encoder'
    });
    
    // Compute pass
    const pass = encoder.beginComputePass({
      label: 'matmul_pass'
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate workgroup count
    // Each workgroup is 16x16, so divide dimensions by 16 and round up
    const workgroupCountX = Math.ceil(M / 16);
    const workgroupCountY = Math.ceil(N / 16);
    
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result to the output tensor
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    return outputTensor;
  }
  
  /**
   * Execute transpose operation
   * @param tensor Input tensor
   * @returns Transposed tensor
   */
  async transpose<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (!this.initialized || !this.device || !this.bufferManager) {
      throw new Error('WebGPU backend not initialized');
    }
    
    // Validate tensor shape
    if (tensor.shape.length !== 2) {
      throw new Error('Transpose requires a 2D tensor');
    }
    
    // Ensure tensor is allocated on GPU
    await this.allocateTensor(tensor);
    
    // Get shader
    const shader = getShader('transpose');
    
    // Create pipeline
    const pipeline = this.getOrCreatePipeline('transpose', shader);
    
    // Get tensor buffer
    const tensorId = this.getTensorId(tensor);
    const buffer = this.tensorBuffers[tensorId].buffer;
    
    // Calculate output shape by swapping dimensions
    const outputShape = [tensor.shape[1], tensor.shape[0]];
    
    // Create output tensor
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: tensor.dataType,
        backend: 'webgpu',
        device: this.id
      }
    );
    
    // Create output buffer
    const outputBuffer = this.bufferManager.createOutputBuffer(
      outputShape,
      tensor.dataType
    );
    
    // Create uniform buffer with matrix dimensions
    const uniformBuffer = this.device.createBuffer({
      size: 8, // 2 x 4 bytes (u32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'transpose_dimensions'
    });
    
    // Write matrix dimensions
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([tensor.shape[0], tensor.shape[1]])
    );
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } }
      ],
      label: 'transpose_bind_group'
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: 'transpose_encoder'
    });
    
    // Compute pass
    const pass = encoder.beginComputePass({
      label: 'transpose_pass'
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate workgroup count
    // Each workgroup is 16x16, so divide dimensions by 16 and round up
    const workgroupCountX = Math.ceil(tensor.shape[0] / 16);
    const workgroupCountY = Math.ceil(tensor.shape[1] / 16);
    
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result to the output tensor
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    return outputTensor;
  }
  
  /**
   * Execute ReLU activation function
   * @param tensor Input tensor
   * @returns Tensor with ReLU applied
   */
  async relu<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    return this.elementwiseOperation(tensor, null, WebGPUOperationType.Relu);
  }
  
  /**
   * Execute sigmoid activation function
   * @param tensor Input tensor
   * @returns Tensor with sigmoid applied
   */
  async sigmoid<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    return this.elementwiseOperation(tensor, null, WebGPUOperationType.Sigmoid);
  }
  
  /**
   * Execute tanh activation function
   * @param tensor Input tensor
   * @returns Tensor with tanh applied
   */
  async tanh<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    return this.elementwiseOperation(tensor, null, WebGPUOperationType.Tanh);
  }
  
  /**
   * Execute softmax activation function
   * @param tensor Input tensor
   * @param axis Axis to apply softmax on
   * @returns Tensor with softmax applied
   */
  async softmax<T>(tensor: Tensor<T>, axis: number = -1): Promise<Tensor<T>> {
    if (!this.initialized || !this.device || !this.bufferManager) {
      throw new Error('WebGPU backend not initialized');
    }
    
    // Handle negative axis
    if (axis < 0) {
      axis = tensor.shape.length + axis;
    }
    
    // Validate axis
    if (axis < 0 || axis >= tensor.shape.length) {
      throw new Error(`Invalid softmax axis: ${axis}`);
    }
    
    // Ensure tensor is allocated on GPU
    await this.allocateTensor(tensor);
    
    // Get shader
    const shader = getShader('softmax');
    
    // Create pipeline
    const pipeline = this.getOrCreatePipeline('softmax', shader);
    
    // Get tensor buffer
    const tensorId = this.getTensorId(tensor);
    const buffer = this.tensorBuffers[tensorId].buffer;
    
    // Calculate dimensions
    const totalElements = tensor.size;
    const innerDim = tensor.shape[axis];
    
    // Calculate outer dim (product of dims before axis)
    let outerDim = 1;
    for (let i = 0; i < axis; i++) {
      outerDim *= tensor.shape[i];
    }
    
    // Calculate stride (product of dims after axis)
    let stride = 1;
    for (let i = axis + 1; i < tensor.shape.length; i++) {
      stride *= tensor.shape[i];
    }
    
    // Create output tensor with same shape
    const outputTensor = new Tensor<T>(
      tensor.shape,
      null,
      {
        dataType: tensor.dataType,
        backend: 'webgpu',
        device: this.id
      }
    );
    
    // Create output buffer
    const outputBuffer = this.bufferManager.createOutputBuffer(
      tensor.shape,
      tensor.dataType
    );
    
    // Create uniform buffer with softmax parameters
    const uniformBuffer = this.device.createBuffer({
      size: 16, // 4 x 4 bytes (u32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'softmax_params'
    });
    
    // Write softmax parameters
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([totalElements, innerDim, outerDim, stride])
    );
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } }
      ],
      label: 'softmax_bind_group'
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: 'softmax_encoder'
    });
    
    // Compute pass
    const pass = encoder.beginComputePass({
      label: 'softmax_pass'
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Dispatch one workgroup per outer dimension
    pass.dispatchWorkgroups(outerDim);
    
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result to the output tensor
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    return outputTensor;
  }
  
  /**
   * Execute tensor reshape
   * @param tensor Input tensor
   * @param newShape New shape
   * @returns Reshaped tensor
   */
  async reshape<T>(tensor: Tensor<T>, newShape: number[]): Promise<Tensor<T>> {
    // Validate new shape
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== tensor.size) {
      throw new Error(`Cannot reshape tensor of size ${tensor.size} to shape ${newShape} (size ${newSize})`);
    }
    
    // For reshape, we can just create a new tensor with the same data but different shape
    return new Tensor<T>(
      newShape,
      tensor.data,
      {
        dataType: tensor.dataType,
        backend: 'webgpu',
        device: this.id
      }
    );
  }
  
  /**
   * Synchronize backend execution
   * Ensures all queued operations are complete
   */
  async sync(): Promise<void> {
    if (!this.initialized || !this.device) {
      throw new Error('WebGPU backend not initialized');
    }
    
    try {
      await this.device.queue.onSubmittedWorkDone();
    } catch (error) {
      console.error('Error synchronizing WebGPU queue:', error);
      throw error;
    }
  }
  
  /**
   * Get buffer allocation statistics
   * @returns Current buffer statistics
   */
  getStats() {
    if (!this.bufferManager) {
      return {};
    }
    
    return this.bufferManager.getStats();
  }
  
  /**
   * Garbage collect unused resources
   * @param maxBuffersPerSize Maximum number of unused buffers to keep per size
   */
  garbageCollect(maxBuffersPerSize: number = 5): void {
    if (!this.bufferManager) {
      return;
    }
    
    this.bufferManager.garbageCollect(maxBuffersPerSize);
  }
  
  /**
   * Free all resources associated with this backend
   */
  dispose(): void {
    if (!this.initialized) {
      return;
    }
    
    // Release all buffers
    if (this.bufferManager) {
      this.bufferManager.releaseAll();
    }
    
    // Clear pipeline cache
    this.pipelineCache.clear();
    
    // Clear bind group layout cache
    this.bindGroupLayoutCache.clear();
    
    // Clear tensor buffers map
    this.tensorBuffers = {};
    
    // Reset initialization flag
    this.initialized = false;
  }
}

/**
 * Create and initialize a WebGPU backend
 * @returns Initialized WebGPU backend
 */
export async function createWebGPUBackend(): Promise<WebGPUBackend> {
  const backend = new WebGPUBackend();
  await backend.initialize();
  return backend;
}