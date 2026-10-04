/**
 * WebGPU Backend Implementation
 * Provides hardware acceleration for tensor operations using WebGPU
 */

import { HardwareBackend, HardwareCapabilities } from '../interfaces/hardware_backend';
import { Tensor } from '../../tensor/tensor';
import { WebGPUBufferManager } from './buffer_manager';
import { getShader } from './shaders';
import { WebGPUOptimizer, WebGPUOptimizerConfig } from './optimizations/webgpu_optimizer';
import { WebGPUOperationFusion, FusionOpType, FusionPattern, FusionConfig } from './optimizations/operation_fusion';
import { WebGPUMatrixMultiplication, MatrixOperationOptions } from './matrix_operations';
import { 
  BrowserOptimizedMatrixOperations, 
  detectBrowserType, 
  getBrowserCapabilities, 
  BrowserType,
  BrowserCapabilities
} from './browser_optimized_operations';
import { loadBrowserShader, getBrowserShaderSync, ShaderType } from './optimizations/browser_shader_loader';

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
  
  /** Operation fusion utility */
  private operationFusion: WebGPUOperationFusion | null = null;
  
  /** Advanced matrix operations */
  private matrixOperations: WebGPUMatrixMultiplication | null = null;
  
  /** Browser-optimized matrix operations */
  private browserOptimizedOperations: BrowserOptimizedMatrixOperations | null = null;
  
  /** Browser type detection */
  private browserType: BrowserType = BrowserType.UNKNOWN;
  
  /** Browser capabilities */
  private browserCapabilities: BrowserCapabilities | null = null;
  
  /** Map of shader pipeline cache */
  private pipelineCache: Map<string, GPUComputePipeline> = new Map();
  
  /** Map of shader bind group layout cache */
  private bindGroupLayoutCache: Map<string, GPUBindGroupLayout> = new Map();
  
  /** Map of tensor to GPU buffer */
  private tensorBuffers: TensorBufferMap = {};
  
  /** Optimizer for advanced performance optimizations */
  private optimizer: WebGPUOptimizer | null = null;
  
  /** Whether advanced optimizations are enabled */
  private optimizationsEnabled: boolean = true;
  
  /** Whether browser-specific optimizations are enabled */
  private browserOptimizationsEnabled: boolean = true;
  
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
   * @param optimizerConfig Configuration for performance optimizations
   */
  constructor(optimizerConfig: WebGPUOptimizerConfig = {}) {
    // Generate a unique ID
    this.id = `webgpu_${Date.now()}`;
    
    // Set whether optimizations are enabled
    this.optimizationsEnabled = optimizerConfig.enableOperationFusion !== false &&
                              optimizerConfig.enableSpecializedShaders !== false;
    
    // Set whether browser optimizations are enabled
    this.browserOptimizationsEnabled = optimizerConfig.enableBrowserOptimizations !== false;
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
      
      // Initialize the operation fusion utility
      this.operationFusion = new WebGPUOperationFusion(this);
      
      // Initialize advanced matrix operations
      this.matrixOperations = new WebGPUMatrixMultiplication(this);
      
      // Initialize browser-specific optimizations and detection first
      // so optimizer can use browser information
      if (this.browserOptimizationsEnabled) {
        // Detect browser type
        this.browserType = detectBrowserType();
        
        // Get browser capabilities
        this.browserCapabilities = await getBrowserCapabilities(this.device);
        
        // Initialize browser-optimized matrix operations
        this.browserOptimizedOperations = await createBrowserOptimizedOperations(
          this,
          this.matrixOperations
        );
        
        console.log(`Browser-optimized operations initialized for ${this.browserType}`);
      }
      
      // Initialize the optimizer if optimizations are enabled
      if (this.optimizationsEnabled) {
        // Create optimizer config based on browser information
        const optimizerConfig: WebGPUOptimizerConfig = {
          enableOperationFusion: true,
          enableSpecializedShaders: true,
          enableBrowserOptimizations: this.browserOptimizationsEnabled,
          enableMemoryOptimizations: true,
          fusionConfig: {
            maxFusionLength: 10, // Default fusion length
            enableAutoFusion: true,
            enabledPatterns: [
              FusionPattern.LinearActivation,
              FusionPattern.ElementWiseChain,
              FusionPattern.BinaryUnary,
              FusionPattern.ReshapeOp,
              FusionPattern.ActivationChain,
              FusionPattern.NormActivation,
              FusionPattern.AttentionPattern,
              FusionPattern.MatrixChain
            ]
          },
          shaderOptions: {
            workgroupSize: 256,
            useSpecializedLayout: true,
            useFastMath: true,
            browserOptimized: this.browserType as any
          }
        };
        
        // Adjust optimization settings based on browser
        if (this.browserType === BrowserType.FIREFOX) {
          optimizerConfig.fusionConfig!.maxFusionLength = 3; // Smaller fusion chains for Firefox
          optimizerConfig.shaderOptions!.workgroupSize = 128; // Smaller workgroups for Firefox
        } else if (this.browserType === BrowserType.SAFARI) {
          optimizerConfig.fusionConfig!.maxFusionLength = 8; // Longer chains on Safari/Apple Silicon
          optimizerConfig.shaderOptions!.workgroupSize = 512; // Larger workgroups for Apple GPUs
          optimizerConfig.shaderOptions!.useFastMath = false; // Higher precision for Apple GPUs
        }
        
        // Create the optimizer with the appropriate config
        this.optimizer = new WebGPUOptimizer(this, optimizerConfig);
        
        console.log('WebGPU optimizer initialized with browser-specific settings');
      }
      
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
    // Try optimized implementation
    const optimizedResult = await this.tryOptimizedImplementation('add', [a, b]);
    if (optimizedResult) {
      return optimizedResult;
    }
    
    return this.elementwiseOperation(a, b, WebGPUOperationType.Add);
  }
  
  /**
   * Execute tensor subtraction
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async subtract<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    // Try optimized implementation
    const optimizedResult = await this.tryOptimizedImplementation('subtract', [a, b]);
    if (optimizedResult) {
      return optimizedResult;
    }
    
    return this.elementwiseOperation(a, b, WebGPUOperationType.Subtract);
  }
  
  /**
   * Execute tensor multiplication (element-wise)
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async multiply<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    // Try optimized implementation
    const optimizedResult = await this.tryOptimizedImplementation('multiply', [a, b]);
    if (optimizedResult) {
      return optimizedResult;
    }
    
    return this.elementwiseOperation(a, b, WebGPUOperationType.Multiply);
  }
  
  /**
   * Execute tensor division
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async divide<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    // Try optimized implementation
    const optimizedResult = await this.tryOptimizedImplementation('divide', [a, b]);
    if (optimizedResult) {
      return optimizedResult;
    }
    
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
    
    // Get optimal workgroup size based on operation and browser
    const optimalWorkgroup = this.getOptimalWorkgroupSize('elementwise');
    const workgroupSize = optimalWorkgroup.x;
    
    // Calculate workgroup count based on optimal size
    const workgroupCount = Math.ceil(outputSize / workgroupSize);
    pass.dispatchWorkgroups(workgroupCount);
    
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result to the output tensor
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    // Track the operation in the optimizer for future optimization
    if (this.optimizationsEnabled && this.optimizer) {
      // Map WebGPUOperationType to FusionOpType
      let opName: FusionOpType;
      switch (operation) {
        case WebGPUOperationType.Add: opName = 'add'; break;
        case WebGPUOperationType.Subtract: opName = 'subtract'; break;
        case WebGPUOperationType.Multiply: opName = 'multiply'; break;
        case WebGPUOperationType.Divide: opName = 'divide'; break;
        case WebGPUOperationType.Relu: opName = 'relu'; break;
        case WebGPUOperationType.Sigmoid: opName = 'sigmoid'; break;
        case WebGPUOperationType.Tanh: opName = 'tanh'; break;
        default: opName = 'unknown' as FusionOpType;
      }
      
      // Track operation
      const aId = this.getTensorId(a);
      const bId = b ? this.getTensorId(b) : '';
      const outputId = this.getTensorId(outputTensor);
      
      const inputIds = b ? [aId, bId] : [aId];
      this.optimizer.trackOperation(opName, inputIds, outputId);
    }
    
    return outputTensor;
  }
  
  /**
   * Execute matrix multiplication
   * @param a First tensor
   * @param b Second tensor
   * @param options Optional matrix operation configuration
   * @returns Resulting tensor
   */
  async matmul<T>(a: Tensor<T>, b: Tensor<T>, options?: MatrixOperationOptions): Promise<Tensor<T>> {
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
    
    // Calculate output shape [M, N]
    const M = a.shape[0];
    const K = a.shape[1];
    const N = b.shape[1];
    
    // Use browser-optimized matrix operations if available and enabled
    if (this.browserOptimizedOperations && this.browserOptimizationsEnabled) {
      try {
        return await this.browserOptimizedOperations.multiply(a, b, options);
      } catch (error) {
        console.warn('Browser-optimized matrix multiply failed, falling back to standard implementation:', error);
      }
    }
    
    // Use the advanced matrix operations if available
    if (this.matrixOperations && this.optimizationsEnabled) {
      try {
        return await this.matrixOperations.multiply(a, b, options);
      } catch (error) {
        console.warn('Advanced matrix multiply failed, falling back to standard implementation:', error);
      }
    }
    
    // Try to use optimizer if available
    const optimizedResult = await this.tryOptimizedImplementation(
      'matmul',
      [a, b],
      { M, K, N, options }
    );
    
    if (optimizedResult) {
      return optimizedResult;
    }
    
    // Standard implementation - proceed with regular WebGPU execution
    const outputShape = [M, N];
    
    // Try to optimize memory layout for matrix multiplication
    if (this.optimizationsEnabled && this.optimizer) {
      try {
        // Get optimal memory layouts for matmul
        a = await this.optimizeMemoryLayout(a, 'matmul');
        b = await this.optimizeMemoryLayout(b, 'matmul');
      } catch (error) {
        console.warn('Memory layout optimization failed:', error);
      }
    }
    
    // Ensure tensors are allocated on GPU
    await this.allocateTensor(a);
    await this.allocateTensor(b);
    
    // Get shader - if optimizer is available, try to get a specialized shader
    let shader;
    if (this.optimizationsEnabled && this.optimizer) {
      try {
        shader = this.optimizer.getSpecializedShader('matmul', { M, K, N });
      } catch (error) {
        // Fall back to standard shader
        shader = getShader('matmul');
      }
    } else {
      shader = getShader('matmul');
    }
    
    // Create pipeline
    const pipeline = this.getOrCreatePipeline('matmul', shader);
    
    // Get tensor buffers
    const tensorIdA = this.getTensorId(a);
    const bufferA = this.tensorBuffers[tensorIdA].buffer;
    
    const tensorIdB = this.getTensorId(b);
    const bufferB = this.tensorBuffers[tensorIdB].buffer;
    
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
    
    // Get optimal workgroup size based on operation and browser
    const optimalWorkgroup = this.getOptimalWorkgroupSize('matmul', { M, K, N });
    const workgroupSizeX = optimalWorkgroup.x;
    const workgroupSizeY = optimalWorkgroup.y;
    
    // Calculate workgroup counts
    const workgroupCountX = Math.ceil(M / workgroupSizeX);
    const workgroupCountY = Math.ceil(N / workgroupSizeY);
    
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    
    pass.end();
    
    // Submit command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Download result to the output tensor
    await this.bufferManager.downloadTensor(outputBuffer, outputTensor);
    
    // Track the operation in the optimizer for future optimization
    if (this.optimizationsEnabled && this.optimizer) {
      // Track that we performed a matmul operation
      const aId = this.getTensorId(a);
      const bId = this.getTensorId(b);
      const outputId = this.getTensorId(outputTensor);
      
      // Track operation with shape information for better optimization
      this.optimizer.trackOperation('matmul', [aId, bId], outputId);
      
      // Check if this matmul can be part of a fusion pattern
      // For example, matmul followed by relu/sigmoid is common in neural networks
      const shapeMeta = { M, K, N };
      this.optimizer.optimizeOperation('matmul', [a, b], shapeMeta)
        .catch(e => {
          // Silently ignore errors since this is just a speculative optimization
        });
    }
    
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
    
    // Try to use optimized implementation
    const optimizedResult = await this.tryOptimizedImplementation(
      'transpose',
      [tensor],
      { shape: tensor.shape }
    );
    
    if (optimizedResult) {
      return optimizedResult;
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
    
    // Get optimal workgroup size based on operation and browser
    const optimalWorkgroup = this.getOptimalWorkgroupSize('transpose', { 
      rows: tensor.shape[0], 
      cols: tensor.shape[1] 
    });
    
    // Calculate workgroup count based on optimal size
    const workgroupCountX = Math.ceil(tensor.shape[0] / optimalWorkgroup.x);
    const workgroupCountY = Math.ceil(tensor.shape[1] / optimalWorkgroup.y);
    
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
    // Try to use optimized implementation
    const optimizedResult = await this.tryOptimizedImplementation('relu', [tensor]);
    if (optimizedResult) {
      return optimizedResult;
    }
    
    // Standard implementation
    return this.elementwiseOperation(tensor, null, WebGPUOperationType.Relu);
  }
  
  /**
   * Execute sigmoid activation function
   * @param tensor Input tensor
   * @returns Tensor with sigmoid applied
   */
  async sigmoid<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    // Try to use optimized implementation
    const optimizedResult = await this.tryOptimizedImplementation('sigmoid', [tensor]);
    if (optimizedResult) {
      return optimizedResult;
    }
    
    // Standard implementation
    return this.elementwiseOperation(tensor, null, WebGPUOperationType.Sigmoid);
  }
  
  /**
   * Execute tanh activation function
   * @param tensor Input tensor
   * @returns Tensor with tanh applied
   */
  async tanh<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    // Try to use optimized implementation
    const optimizedResult = await this.tryOptimizedImplementation('tanh', [tensor]);
    if (optimizedResult) {
      return optimizedResult;
    }
    
    // Standard implementation
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
    
    // Try to use optimized implementation
    const optimizedResult = await this.tryOptimizedImplementation(
      'softmax', 
      [tensor], 
      { axis }
    );
    
    if (optimizedResult) {
      return optimizedResult;
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
    
    // Get optimal workgroup size based on operation and browser
    const optimalWorkgroup = this.getOptimalWorkgroupSize('softmax', { outerDim, innerDim });
    
    // Dispatch workgroups - one per outer dimension, with optimal size for reductions
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
   * Execute a sequence of fused operations
   * @param inputs Input tensors
   * @param operations Operations to fuse
   * @returns Output tensor from fused operations
   */
  async executeFusedOperations<T>(
    inputs: Tensor<T>[],
    operations: FusionOpType[]
  ): Promise<Tensor<T>> {
    if (!this.initialized || !this.operationFusion) {
      // Initialize operation fusion if needed
      if (this.initialized && !this.operationFusion && this.device) {
        this.operationFusion = new WebGPUOperationFusion(this);
      } else {
        throw new Error('WebGPU backend not initialized');
      }
    }
    
    // Check if operations can be fused
    if (!this.operationFusion.canFuse(operations)) {
      throw new Error(`Cannot fuse operations: ${operations.join(', ')}`);
    }
    
    // If optimizations are enabled, check if the optimizer can optimize this sequence
    if (this.optimizationsEnabled && this.optimizer) {
      try {
        // Get shape information to help with optimization
        const shapeInfo: any = {};
        
        // Extract shape information for various operations
        if (operations.includes('matmul')) {
          if (operations[0] === 'matmul' && inputs.length >= 2) {
            shapeInfo.M = inputs[0].shape[0];
            shapeInfo.K = inputs[0].shape[1];
            shapeInfo.N = inputs[1].shape[1];
          }
        }
        
        // Try optimal memory layout if available
        if (operations.length > 0 && inputs.length > 0) {
          const layout = this.optimizer.getOptimalMemoryLayout(operations[0], inputs[0].shape);
          shapeInfo.rowMajor = layout.rowMajor;
          shapeInfo.alignment = layout.alignment;
          if (layout.paddedShape) {
            shapeInfo.paddedShape = layout.paddedShape;
          }
        }
        
        // Try to optimize this operation sequence
        const result = await this.optimizer.tryExecuteFusion(operations, inputs, shapeInfo);
        if (result) {
          console.log(`Using optimized fusion for: ${operations.join(', ')}`);
          return result;
        }
      } catch (error) {
        console.warn('Optimization failed for fusion sequence, falling back to standard execution:', error);
      }
    }
    
    // Allocate inputs on GPU if needed
    for (const input of inputs) {
      await this.allocateTensor(input);
    }
    
    // Execute fused operations using standard implementation
    const output = await this.operationFusion.executeFusedOperations(inputs, operations);
    
    // Track the sequence for future optimization
    if (this.optimizationsEnabled && this.optimizer) {
      const inputIds = inputs.map(t => this.getTensorId(t));
      const outputId = this.getTensorId(output);
      
      // Track each operation in the sequence
      for (let i = 0; i < operations.length; i++) {
        // For each operation in the sequence, track the appropriate inputs and outputs
        // For simplicity, we'll just track the first operation with all inputs
        // and intermediate operations with their preceding operation's output
        const opInputs = i === 0 ? inputIds : [outputId];
        this.optimizer.trackOperation(operations[i], opInputs, outputId);
      }
    }
    
    return output;
  }

  /**
   * Create a fusion pattern with custom operations
   * @param operations Operations to fuse
   * @param id Optional pattern identifier
   * @returns Whether the pattern is valid
   */
  createFusionPattern(operations: FusionOpType[], id?: string): boolean {
    if (!this.initialized || !this.operationFusion) {
      // Initialize operation fusion if needed
      if (this.initialized && !this.operationFusion && this.device) {
        this.operationFusion = new WebGPUOperationFusion(this);
      } else {
        throw new Error('WebGPU backend not initialized');
      }
    }
    
    // Check if operations can be fused
    return this.operationFusion.canFuse(operations);
  }
  
  /**
   * Enable or disable optimizations
   * @param enabled Whether optimizations should be enabled
   */
  setOptimizationsEnabled(enabled: boolean): void {
    this.optimizationsEnabled = enabled;
  }
  
  /**
   * Get the current optimization status
   * @returns Whether optimizations are enabled
   */
  areOptimizationsEnabled(): boolean {
    return this.optimizationsEnabled;
  }
  
  /**
   * Enable or disable browser-specific optimizations
   * @param enabled Whether browser-specific optimizations should be enabled
   */
  setBrowserOptimizationsEnabled(enabled: boolean): void {
    this.browserOptimizationsEnabled = enabled;
  }
  
  /**
   * Get the current browser optimization status
   * @returns Whether browser-specific optimizations are enabled
   */
  areBrowserOptimizationsEnabled(): boolean {
    return this.browserOptimizationsEnabled;
  }
  
  /**
   * Get the detected browser type
   * @returns Detected browser type
   */
  getBrowserType(): BrowserType {
    return this.browserType;
  }
  
  /**
   * Get browser capabilities
   * @returns Browser capabilities or null if not detected
   */
  getBrowserCapabilities(): BrowserCapabilities | null {
    return this.browserCapabilities;
  }
  
  /**
   * Get the optimizer instance
   * @returns WebGPU optimizer instance or null if not available
   */
  getOptimizer(): WebGPUOptimizer | null {
    return this.optimizer;
  }
  
  /**
   * Analyze neural network layer for fusion opportunities
   * This can be used to automatically identify optimization opportunities in common neural network patterns
   * @param operations List of operations in the layer 
   * @param tensors Tensors used by the operations
   * @returns List of fusion opportunities identified
   */
  analyzeLayerForFusionOpportunities(
    operations: {type: FusionOpType, inputs: string[], output: string}[],
    tensors: {[id: string]: {shape: number[], dataType: string}}
  ): {
    fusionPatterns: {pattern: FusionPattern, operations: FusionOpType[], benefit: 'high' | 'medium' | 'low'}[],
    optimizationTips: string[]
  } {
    if (!this.optimizationsEnabled || !this.optimizer || !this.operationFusion) {
      return { fusionPatterns: [], optimizationTips: [] };
    }
    
    const fusionPatterns: {
      pattern: FusionPattern, 
      operations: FusionOpType[], 
      benefit: 'high' | 'medium' | 'low'
    }[] = [];
    
    const optimizationTips: string[] = [];
    
    // Check for common patterns in the operations list
    
    // Linear layer: MatMul + Add (bias) + Activation
    const matmulIndices = operations
      .map((op, i) => op.type === 'matmul' ? i : -1)
      .filter(i => i !== -1);
    
    for (const matmulIdx of matmulIndices) {
      // Look for bias add after matmul
      const biasAddIdx = operations.findIndex((op, i) => 
        i > matmulIdx && 
        op.type === 'add' && 
        op.inputs.includes(operations[matmulIdx].output)
      );
      
      // Look for activation after matmul (or bias add if present)
      const checkAfterIdx = biasAddIdx !== -1 ? biasAddIdx : matmulIdx;
      const activationIdx = operations.findIndex((op, i) => 
        i > checkAfterIdx && 
        ['relu', 'sigmoid', 'tanh', 'gelu'].includes(op.type) && 
        op.inputs.includes(operations[checkAfterIdx].output)
      );
      
      if (biasAddIdx !== -1 && activationIdx !== -1) {
        // MatMul + Add + Activation pattern
        fusionPatterns.push({
          pattern: FusionPattern.LinearActivation,
          operations: [
            operations[matmulIdx].type,
            operations[biasAddIdx].type,
            operations[activationIdx].type
          ],
          benefit: 'high'
        });
        
        optimizationTips.push(
          `Detected linear layer with activation. This pattern can benefit from fusion optimization: ` +
          `${operations[matmulIdx].type} → ${operations[biasAddIdx].type} → ${operations[activationIdx].type}`
        );
      } else if (activationIdx !== -1) {
        // MatMul + Activation pattern
        fusionPatterns.push({
          pattern: FusionPattern.LinearActivation,
          operations: [
            operations[matmulIdx].type,
            operations[activationIdx].type
          ],
          benefit: 'high'
        });
        
        optimizationTips.push(
          `Detected matrix multiplication with activation. This pattern can benefit from fusion optimization: ` +
          `${operations[matmulIdx].type} → ${operations[activationIdx].type}`
        );
      }
    }
    
    // Check for element-wise chains
    let elementWiseOps = [];
    let currentChain = [];
    
    for (let i = 0; i < operations.length; i++) {
      const op = operations[i].type;
      const isElementWise = ['add', 'subtract', 'multiply', 'divide'].includes(op);
      
      if (isElementWise) {
        // Check if this continues the current chain
        if (currentChain.length > 0) {
          const prevOp = operations[i - 1];
          const currOp = operations[i];
          
          // If current operation uses previous operation's output, add to chain
          if (currOp.inputs.includes(prevOp.output)) {
            currentChain.push(op);
          } else {
            // Chain broken, save if long enough
            if (currentChain.length >= 2) {
              elementWiseOps.push([...currentChain]);
            }
            // Start new chain
            currentChain = [op];
          }
        } else {
          // Start new chain
          currentChain.push(op);
        }
      } else {
        // End of element-wise operations, save chain if long enough
        if (currentChain.length >= 2) {
          elementWiseOps.push([...currentChain]);
        }
        currentChain = [];
      }
    }
    
    // Add final chain if it exists and is long enough
    if (currentChain.length >= 2) {
      elementWiseOps.push(currentChain);
    }
    
    // Add element-wise chains to fusion patterns
    for (const chain of elementWiseOps) {
      fusionPatterns.push({
        pattern: FusionPattern.ElementWiseChain,
        operations: chain as FusionOpType[],
        benefit: chain.length > 3 ? 'high' : 'medium'
      });
      
      optimizationTips.push(
        `Detected element-wise operation chain (${chain.join(' → ')}). ` +
        `This can be fused into a single shader for better performance.`
      );
    }
    
    // Look for attention patterns in transformer models: MatMul + Scale + Softmax + MatMul
    for (let i = 0; i < operations.length - 3; i++) {
      if (operations[i].type === 'matmul' &&
          (operations[i+1].type === 'multiply' || operations[i+1].type === 'scale') &&
          operations[i+2].type === 'softmax' &&
          operations[i+3].type === 'matmul') {
        
        // Verify connections between operations
        if (operations[i+1].inputs.includes(operations[i].output) &&
            operations[i+2].inputs.includes(operations[i+1].output) &&
            operations[i+3].inputs.includes(operations[i+2].output)) {
          
          fusionPatterns.push({
            pattern: FusionPattern.AttentionPattern,
            operations: [
              operations[i].type,
              operations[i+1].type,
              operations[i+2].type,
              operations[i+3].type
            ],
            benefit: 'high'
          });
          
          optimizationTips.push(
            `Detected attention pattern in transformer model. This can be highly optimized with a specialized implementation.`
          );
        }
      }
    }
    
    // Look for normalization + activation patterns
    const normIndices = operations
      .map((op, i) => ['layer_norm', 'batch_norm'].includes(op.type) ? i : -1)
      .filter(i => i !== -1);
    
    for (const normIdx of normIndices) {
      // Look for activation after normalization
      const activationIdx = operations.findIndex((op, i) => 
        i > normIdx && 
        ['relu', 'sigmoid', 'tanh', 'gelu'].includes(op.type) && 
        op.inputs.includes(operations[normIdx].output)
      );
      
      if (activationIdx !== -1) {
        fusionPatterns.push({
          pattern: FusionPattern.NormActivation,
          operations: [
            operations[normIdx].type,
            operations[activationIdx].type
          ],
          benefit: 'medium'
        });
        
        optimizationTips.push(
          `Detected normalization layer with activation. This pattern can benefit from fusion: ` +
          `${operations[normIdx].type} → ${operations[activationIdx].type}`
        );
      }
    }
    
    // Add browser-specific optimization tips
    if (this.browserOptimizationsEnabled) {
      if (this.browserType === BrowserType.FIREFOX) {
        optimizationTips.push(
          `Running on Firefox. For optimal performance with WebGPU, audio processing operations will be prioritized and smaller workgroups used.`
        );
      } else if (this.browserType === BrowserType.SAFARI) {
        optimizationTips.push(
          `Running on Safari. For optimal performance on Apple Silicon, large batch sizes and specific memory layouts will be used.`
        );
      } else if (this.browserType === BrowserType.CHROME) {
        optimizationTips.push(
          `Running on Chrome. Standard WebGPU optimizations will be applied with large workgroups for matrix operations.`
        );
      }
    }
    
    // Add memory optimization tips
    if (this.optimizationsEnabled && this.optimizer) {
      // Look for large matmul operations that could benefit from memory layout optimization
      const largeMatMuls = operations.filter(op => 
        op.type === 'matmul' && 
        tensors[op.inputs[0]]?.shape[0] > 1024 && 
        tensors[op.inputs[1]]?.shape[1] > 1024
      );
      
      if (largeMatMuls.length > 0) {
        optimizationTips.push(
          `Detected ${largeMatMuls.length} large matrix multiplications. These operations will benefit from memory layout optimization.`
        );
      }
    }
    
    return { fusionPatterns, optimizationTips };
  }
  
  /**
   * Transform tensor memory layout for optimal performance
   * This transforms between row-major and column-major layouts based on operation type and browser
   * @param tensor Input tensor
   * @param operation Target operation for optimization
   * @returns Transformed tensor or original if no transformation needed
   */
  async optimizeMemoryLayout<T>(
    tensor: Tensor<T>,
    operation: FusionOpType
  ): Promise<Tensor<T>> {
    if (!this.optimizer || !this.optimizationsEnabled) {
      return tensor;
    }
    
    try {
      // Get optimal memory layout for this tensor and operation
      const layout = this.optimizer.getOptimalMemoryLayout(operation, tensor.shape);
      
      // If layout is already optimal, return original tensor
      const currentLayout = (tensor as any).rowMajor !== false; // Default to row-major if not specified
      
      if (layout.rowMajor === currentLayout) {
        // Layout already optimal, no transformation needed
        return tensor;
      }
      
      console.log(`Transforming tensor memory layout for ${operation}: 
        ${currentLayout ? 'row-major → column-major' : 'column-major → row-major'}`);
      
      // Create new tensor with transformed layout
      let transformedTensor: Tensor<T>;
      
      // For 2D tensors, we can use transpose to switch between row/column major
      if (tensor.shape.length === 2) {
        // Transpose to change layout
        const transposed = await this.transpose(tensor);
        
        // Transpose again to get original shape but with different memory layout
        transformedTensor = await this.transpose(transposed);
        
        // Set layout property
        (transformedTensor as any).rowMajor = layout.rowMajor;
        
        return transformedTensor;
      } else {
        // For non-2D tensors, we need to reshape, transpose relevant dims, and reshape back
        // This is a simplified approach - a complete implementation would be more complex
        
        // For now, just copy the tensor and set the layout property
        transformedTensor = new Tensor<T>(
          tensor.shape,
          tensor.data ? Array.from(tensor.data as any) as any : null,
          {
            dataType: tensor.dataType,
            backend: 'webgpu',
            device: this.id
          }
        );
        
        // Set layout property
        (transformedTensor as any).rowMajor = layout.rowMajor;
        
        return transformedTensor;
      }
    } catch (error) {
      console.warn('Memory layout optimization failed:', error);
      return tensor;
    }
  }
  
  /**
   * Get optimal workgroup size based on operation and browser
   * @param operation Operation type
   * @param dims Dimensions for the operation (optional)
   * @returns Optimal workgroup configuration
   */
  getOptimalWorkgroupSize(
    operation: 'matmul' | 'elementwise' | 'reduction' | 'transpose' | 'softmax',
    dims?: {[key: string]: number}
  ): {x: number, y: number, z: number} {
    // Default workgroup sizes
    const defaultSizes = {
      matmul: {x: 16, y: 16, z: 1},
      elementwise: {x: 256, y: 1, z: 1},
      reduction: {x: 256, y: 1, z: 1},
      transpose: {x: 16, y: 16, z: 1},
      softmax: {x: 256, y: 1, z: 1}
    };
    
    // If optimizations are disabled or no optimizer, return defaults
    if (!this.optimizationsEnabled || !this.optimizer) {
      return defaultSizes[operation];
    }
    
    // Browser-specific optimizations
    if (this.browserOptimizationsEnabled) {
      // Chrome optimizations
      if (this.browserType === BrowserType.CHROME) {
        // Chrome works well with larger workgroups
        switch (operation) {
          case 'matmul':
            // For large matrices, use larger workgroups
            if (dims && dims.M > 2048 && dims.N > 2048) {
              return {x: 32, y: 32, z: 1};
            }
            return {x: 16, y: 16, z: 1};
          
          case 'elementwise':
          case 'reduction':
          case 'softmax':
            return {x: 256, y: 1, z: 1};
            
          case 'transpose':
            return {x: 16, y: 16, z: 1};
        }
      }
      
      // Firefox optimizations
      else if (this.browserType === BrowserType.FIREFOX) {
        // Firefox often works better with smaller workgroups
        switch (operation) {
          case 'matmul':
            return {x: 8, y: 8, z: 1};
            
          case 'elementwise':
          case 'reduction':
          case 'softmax':
            return {x: 128, y: 1, z: 1};
            
          case 'transpose':
            return {x: 8, y: 8, z: 1};
        }
      }
      
      // Safari optimizations
      else if (this.browserType === BrowserType.SAFARI) {
        // Safari/Apple Silicon benefits from larger workgroups
        switch (operation) {
          case 'matmul':
            return {x: 32, y: 32, z: 1};
            
          case 'elementwise':
          case 'reduction':
          case 'softmax':
            return {x: 512, y: 1, z: 1};
            
          case 'transpose':
            return {x: 32, y: 16, z: 1};
        }
      }
    }
    
    // Default fallback
    return defaultSizes[operation];
  }

  /**
   * Create an optimized WebGPU tensor from any tensor source with optimal memory layout
   * @param tensor Source tensor (can be from CPU or another backend)
   * @param targetOperation Target operation for optimizing layout
   * @returns Optimized WebGPU tensor
   */
  async createOptimizedTensor<T>(
    tensor: Tensor<T>,
    targetOperation?: FusionOpType
  ): Promise<Tensor<T>> {
    if (!this.initialized) {
      throw new Error('WebGPU backend not initialized');
    }
    
    // If tensor is already on WebGPU, try to optimize layout
    if (tensor.backend === 'webgpu' && tensor.device === this.id) {
      // If we have a target operation and optimizer is enabled, optimize layout
      if (targetOperation && this.optimizationsEnabled && this.optimizer) {
        return this.optimizeMemoryLayout(tensor, targetOperation);
      }
      return tensor;
    }
    
    // Create a new WebGPU tensor
    const webgpuTensor = new Tensor<T>(
      tensor.shape,
      tensor.data ? Array.from(tensor.data as any) as any : null,
      {
        dataType: tensor.dataType,
        backend: 'webgpu',
        device: this.id
      }
    );
    
    // If we have a target operation and optimizer, set optimal layout property
    if (targetOperation && this.optimizer && this.optimizationsEnabled) {
      const layout = this.optimizer.getOptimalMemoryLayout(targetOperation, tensor.shape);
      (webgpuTensor as any).rowMajor = layout.rowMajor;
      
      // If padding is needed, we could pad the data here
      if (layout.paddedShape) {
        // In a real implementation, we would pad the data to match the padded shape
        console.log(`Tensor padding recommended: ${tensor.shape} → ${layout.paddedShape}`);
      }
    }
    
    // Allocate on device
    await this.allocateTensor(webgpuTensor);
    
    return webgpuTensor;
  }

  /**
   * Try to use optimized implementation for an operation
   * @param operation Operation type
   * @param inputs Input tensors
   * @param shapeInfo Additional shape information for optimization
   * @returns Optimized result tensor or null if optimization not possible
   */
  private async tryOptimizedImplementation<T>(
    operation: FusionOpType,
    inputs: Tensor<T>[],
    shapeInfo: any = {}
  ): Promise<Tensor<T> | null> {
    // Check if optimizations are enabled
    if (!this.optimizationsEnabled || !this.optimizer) {
      return null;
    }
    
    try {
      // Try to use an optimized implementation
      const optimizedResult = await this.optimizer.optimizeOperation<T>(
        operation,
        inputs,
        shapeInfo
      );
      
      // If optimization was successful, return the result
      if (optimizedResult) {
        return optimizedResult;
      }
    } catch (error) {
      // If optimization fails, we log the error and let the caller fall back
      console.warn(`Optimized ${operation} failed, falling back to standard implementation:`, error);
    }
    
    return null;
  }
  
  /**
   * Execute batch matrix multiplication on a batch of matrices
   * Computes C[i] = A[i] × B[i] for each matrix in the batch
   * @param a Batch of matrices A, shape [batchSize, M, K]
   * @param b Batch of matrices B, shape [batchSize, K, N]
   * @param options Optional matrix operation configuration
   * @returns Resulting batch of matrices C, shape [batchSize, M, N]
   */
  async batchMatmul<T>(a: Tensor<T>, b: Tensor<T>, options?: MatrixOperationOptions): Promise<Tensor<T>> {
    if (!this.initialized || !this.device || !this.bufferManager) {
      throw new Error('WebGPU backend not initialized');
    }
    
    // Validate tensor shapes
    if (a.shape.length !== 3 || b.shape.length !== 3) {
      throw new Error(`Batch matrix multiplication requires 3D tensors, got shapes: ${a.shape} and ${b.shape}`);
    }
    
    if (a.shape[0] !== b.shape[0]) {
      throw new Error(`Batch sizes must match: ${a.shape[0]} vs ${b.shape[0]}`);
    }
    
    if (a.shape[2] !== b.shape[1]) {
      throw new Error(`Matrix dimensions mismatch: ${a.shape} and ${b.shape}`);
    }
    
    // Use browser-optimized matrix operations if available and enabled
    if (this.browserOptimizedOperations && this.browserOptimizationsEnabled) {
      try {
        return await this.browserOptimizedOperations.batchMultiply(a, b, options);
      } catch (error) {
        console.warn('Browser-optimized batch matrix multiply failed, falling back to standard implementation:', error);
      }
    }
    
    // Use the advanced matrix operations if available
    if (this.matrixOperations && this.optimizationsEnabled) {
      return await this.matrixOperations.batchMultiply(a, b, options);
    }
    
    throw new Error('Batch matrix multiplication requires advanced matrix operations');
  }
  
  /**
   * Execute 2D convolution for neural networks
   * @param input Input tensor of shape [batchSize, inputHeight, inputWidth, inputChannels]
   * @param filters Filter tensor of shape [filterHeight, filterWidth, inputChannels, outputChannels]
   * @param strides Stride of the convolution [strideHeight, strideWidth]
   * @param padding Padding mode: 'same' or 'valid'
   * @param options Optional matrix operation configuration
   * @returns Output tensor of shape [batchSize, outputHeight, outputWidth, outputChannels]
   */
  async conv2d<T>(
    input: Tensor<T>, 
    filters: Tensor<T>, 
    strides: [number, number] = [1, 1], 
    padding: 'same' | 'valid' = 'valid',
    options?: MatrixOperationOptions
  ): Promise<Tensor<T>> {
    if (!this.initialized || !this.device || !this.bufferManager) {
      throw new Error('WebGPU backend not initialized');
    }
    
    // Use browser-optimized matrix operations if available and enabled
    if (this.browserOptimizedOperations && this.browserOptimizationsEnabled) {
      try {
        return await this.browserOptimizedOperations.conv2d(input, filters, strides, padding, options);
      } catch (error) {
        console.warn('Browser-optimized convolution failed, falling back to standard implementation:', error);
      }
    }
    
    // Use the advanced matrix operations if available
    if (this.matrixOperations && this.optimizationsEnabled) {
      return await this.matrixOperations.conv2d(input, filters, strides, padding, options);
    }
    
    throw new Error('Convolution operation requires advanced matrix operations');
  }
  
  /**
   * Perform garbage collection of GPU resources
   * @param maxBuffersPerSize Maximum number of unused buffers to keep per size
   * @param aggressiveMode Whether to perform aggressive garbage collection
   */
  garbageCollect(maxBuffersPerSize: number = 5, aggressiveMode: boolean = false): void {
    if (!this.initialized) {
      return;
    }
    
    console.log(`Performing garbage collection (aggressive: ${aggressiveMode})`);
    
    // Garbage collect buffer cache
    if (this.bufferManager) {
      this.bufferManager.garbageCollect(aggressiveMode ? 1 : maxBuffersPerSize);
    }
    
    // Garbage collect optimizer caches
    if (this.optimizer) {
      this.optimizer.garbageCollect();
    }
    
    // Garbage collect fusion engine if available
    if (this.operationFusion) {
      // Clear shader caches in operation fusion
      (this.operationFusion as any).shaderCache?.clear();
      (this.operationFusion as any).bindGroupLayoutCache?.clear();
    }
    
    // Limit pipeline cache size or clear completely in aggressive mode
    if (aggressiveMode || this.pipelineCache.size > 100) {
      console.log(`Clearing pipeline cache (size: ${this.pipelineCache.size})`);
      this.pipelineCache.clear();
    }
    
    // Limit bind group layout cache size or clear completely in aggressive mode
    if (aggressiveMode || this.bindGroupLayoutCache.size > 100) {
      console.log(`Clearing bind group layout cache (size: ${this.bindGroupLayoutCache.size})`);
      this.bindGroupLayoutCache.clear();
    }
    
    // Clean up tensor buffers that haven't been used recently
    if (aggressiveMode) {
      const currentTime = Date.now();
      const bufferTimeout = 60000; // 1 minute
      
      let releasedCount = 0;
      for (const tensorId in this.tensorBuffers) {
        const entry = this.tensorBuffers[tensorId];
        
        // If buffer hasn't been used in a while, release it
        if (currentTime - entry.timestamp > bufferTimeout) {
          if (this.bufferManager) {
            this.bufferManager.releaseBuffer(entry.buffer);
          }
          delete this.tensorBuffers[tensorId];
          releasedCount++;
        }
      }
      
      if (releasedCount > 0) {
        console.log(`Released ${releasedCount} stale tensor buffers`);
      }
    }
  }
  
  /**
   * Free all resources associated with this backend
   */
  dispose(): void {
    if (!this.initialized) {
      return;
    }
    
    console.log('Disposing WebGPU backend resources...');
    
    // Release all buffers
    if (this.bufferManager) {
      this.bufferManager.releaseAll();
      console.log('Released all WebGPU buffers');
    }
    
    // Dispose browser-optimized operations if initialized
    if (this.browserOptimizedOperations) {
      this.browserOptimizedOperations.dispose();
      this.browserOptimizedOperations = null;
      console.log('Disposed browser-optimized operations');
    }
    
    // Dispose matrix operations if initialized
    if (this.matrixOperations) {
      this.matrixOperations.dispose();
      this.matrixOperations = null;
      console.log('Disposed WebGPU matrix operations');
    }
    
    // Release optimizer resources if available
    if (this.optimizer) {
      this.optimizer.garbageCollect();
      (this.optimizer as any).shaderCache?.clear();
      this.optimizer = null;
      console.log('Cleaned up optimizer resources');
    }
    
    // Dispose operation fusion resources if available
    if (this.operationFusion) {
      (this.operationFusion as any).shaderCache?.clear();
      (this.operationFusion as any).bindGroupLayoutCache?.clear();
      this.operationFusion = null;
      console.log('Cleaned up operation fusion resources');
    }
    
    // Clear pipeline cache
    this.pipelineCache.clear();
    console.log('Cleared WebGPU pipeline cache');
    
    // Clear bind group layout cache
    this.bindGroupLayoutCache.clear();
    console.log('Cleared WebGPU bind group layout cache');
    
    // Clear tensor buffers map
    this.tensorBuffers = {};
    console.log('Cleared WebGPU tensor buffers mapping');
    
    // Reset browser capabilities
    this.browserCapabilities = null;
    
    // Reset device and adapter
    this.device = null;
    this.adapter = null;
    
    // Reset initialization flag
    this.initialized = false;
    
    console.log('WebGPU backend successfully disposed');
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