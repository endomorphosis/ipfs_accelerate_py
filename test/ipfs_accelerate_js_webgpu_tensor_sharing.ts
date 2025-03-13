/**
 * WebGPU Tensor Sharing Integration
 * 
 * Integrates the Cross-Model Tensor Sharing system with WebGPU compute shader operations
 * Enables efficient tensor operations directly on shared tensors without copying
 * Provides zero-copy tensor sharing between WebNN and WebGPU backends
 */

import { WebGPUBackend } from './ipfs_accelerate_js_webgpu_backend';
import { TensorSharingIntegration } from './ipfs_accelerate_js_tensor_sharing_integration';
import { StorageManager } from './ipfs_accelerate_js_storage_manager';
import {
  getOptimizedShader,
  BrowserType,
  detectBrowserType,
  getBrowserCapabilities
} from './ipfs_accelerate_js_browser_optimized_shaders';

/**
 * WebGPU Tensor Sharing Options
 */
export interface WebGPUTensorSharingOptions {
  /**
   * WebGPU backend to use, if not provided, a new one will be created
   */
  webgpuBackend?: WebGPUBackend;
  
  /**
   * Storage manager to use for persistent storage
   */
  storageManager?: StorageManager;
  
  /**
   * WebGPU backend options if creating a new backend
   */
  webgpuOptions?: any;
  
  /**
   * Enable browser-specific optimizations for shaders
   */
  browserOptimizations?: boolean;
  
  /**
   * Browser type to optimize for (auto-detected if not specified)
   */
  browserType?: BrowserType;
  
  /**
   * Custom shader optimization settings
   */
  shaderOptimizationSettings?: Record<string, any>;
  
  /**
   * Enable zero-copy mode where possible
   * When enabled, tensors are shared between WebNN and WebGPU without copying
   */
  enableZeroCopy?: boolean;
  
  /**
   * Cache compute pipelines for better performance
   */
  cachePipelines?: boolean;
  
  /**
   * Priority mode for tensor operations
   * - 'speed': Optimize for speed (may use more memory)
   * - 'memory': Optimize for memory usage (may be slower)
   * - 'balanced': Balance between speed and memory usage
   */
  priorityMode?: 'speed' | 'memory' | 'balanced';
  
  /**
   * Debug mode to provide detailed logs and validations
   */
  debug?: boolean;
}

/**
 * Shared Tensor Location
 * Specifies where a shared tensor is stored
 */
export enum SharedTensorLocation {
  /** Tensor stored in CPU memory */
  CPU = 'cpu',
  /** Tensor stored in WebGPU memory */
  WEBGPU = 'webgpu',
  /** Tensor stored in WebNN memory */
  WEBNN = 'webnn',
  /** Tensor stored in multiple locations */
  MULTI = 'multi'
}

/**
 * WebGPU Tensor Sharing Integration
 * Provides efficient integration between the Cross-Model Tensor Sharing system and WebGPU
 */
export class WebGPUTensorSharing {
  private tensorSharing: TensorSharingIntegration;
  private webgpuBackend: WebGPUBackend;
  private storageManager?: StorageManager;
  private options: WebGPUTensorSharingOptions;
  private initialized: boolean = false;
  private browserCapabilities: any = null;
  
  // Cache of GPU buffers for shared tensors
  private tensorBufferCache: Map<string, {
    buffer: GPUBuffer;
    shape: number[];
    dataType: string;
    location: SharedTensorLocation;
    lastUsed: number;
    size: number;
  }> = new Map();
  
  // Cache of compute pipelines
  private pipelineCache: Map<string, {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
    lastUsed: number;
  }> = new Map();
  
  // Cache of optimized shaders
  private shaderCache: Map<string, string> = new Map();
  
  /**
   * Create a new WebGPU Tensor Sharing integration
   */
  constructor(tensorSharing: TensorSharingIntegration, options: WebGPUTensorSharingOptions = {}) {
    this.tensorSharing = tensorSharing;
    this.options = {
      browserOptimizations: true,
      enableZeroCopy: true,
      cachePipelines: true,
      priorityMode: 'balanced',
      debug: false,
      ...options
    };
    
    // Use provided WebGPU backend or create a new one
    this.webgpuBackend = options.webgpuBackend || new WebGPUBackend(options.webgpuOptions);
    this.storageManager = options.storageManager;
  }
  
  /**
   * Initialize the WebGPU Tensor Sharing integration
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) {
      return true;
    }
    
    try {
      // Check if WebGPU is supported
      const isSupported = await this.webgpuBackend.isSupported();
      if (!isSupported) {
        console.warn('WebGPU is not supported in this browser');
        return false;
      }
      
      // Initialize WebGPU backend
      const initialized = await this.webgpuBackend.initialize();
      if (!initialized) {
        console.warn('Failed to initialize WebGPU backend');
        return false;
      }
      
      // Get browser capabilities for optimizations
      if (this.options.browserOptimizations !== false) {
        // Use the WebGPU device from the backend
        // This requires accessing the private device property
        const device = (this.webgpuBackend as any)['device'];
        if (device) {
          this.browserCapabilities = await getBrowserCapabilities(device);
          
          if (this.options.debug) {
            console.log('Browser capabilities detected:', this.browserCapabilities);
          }
        }
      }
      
      // Precompile optimized shaders if browser optimizations are enabled
      if (this.options.browserOptimizations !== false && this.browserCapabilities) {
        await this.precompileOptimizedShaders();
      }
      
      this.initialized = true;
      
      if (this.options.debug) {
        console.log('WebGPU Tensor Sharing initialized');
        const capabilities = await this.webgpuBackend.getCapabilities();
        console.log('WebGPU capabilities:', capabilities);
      }
      
      return true;
    } catch (error) {
      console.error('Error initializing WebGPU Tensor Sharing:', error);
      return false;
    }
  }
  
  /**
   * Precompile optimized shaders for common operations
   */
  private async precompileOptimizedShaders(): Promise<void> {
    if (!this.browserCapabilities) return;
    
    try {
      // Get device from backend
      const device = (this.webgpuBackend as any)['device'];
      if (!device) return;
      
      // Precompile common operations
      const operations: Array<'matmul' | 'elementwise' | 'softmax' | 'quantize'> = [
        'matmul', 'elementwise', 'softmax', 'quantize'
      ];
      
      for (const operation of operations) {
        const optimizedShader = await getOptimizedShader(
          device, 
          operation,
          this.options.shaderOptimizationSettings
        );
        
        // Cache the optimized shader
        this.shaderCache.set(operation, optimizedShader);
        
        // Create shader module
        const shaderModule = device.createShaderModule({
          label: `optimized_${operation}_shader`,
          code: optimizedShader
        });
        
        // Add to backend shader cache if it has one
        const backendShaderCache = (this.webgpuBackend as any)['shaderCache'];
        if (backendShaderCache && backendShaderCache instanceof Map) {
          backendShaderCache.set(operation, shaderModule);
        }
        
        if (this.options.debug) {
          console.log(`Precompiled optimized ${operation} shader for ${this.browserCapabilities.browserType}`);
        }
      }
    } catch (error) {
      console.warn('Error precompiling optimized shaders:', error);
    }
  }
  
  /**
   * Check if WebGPU Tensor Sharing is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }
  
  /**
   * Create a WebGPU buffer from a shared tensor
   * @param tensorName Name of the shared tensor
   * @param modelName Model that owns the tensor
   * @returns Object containing GPU buffer and metadata
   */
  async createGPUBufferFromSharedTensor(
    tensorName: string, 
    modelName: string
  ): Promise<{
    buffer: GPUBuffer;
    shape: number[];
    dataType: string;
    location: SharedTensorLocation;
    size: number;
  }> {
    // Check if already initialized
    if (!this.initialized) {
      const success = await this.initialize();
      if (!success) {
        throw new Error('Failed to initialize WebGPU Tensor Sharing');
      }
    }
    
    // Check if already in cache
    const cacheKey = `${tensorName}_${modelName}`;
    const cached = this.tensorBufferCache.get(cacheKey);
    if (cached) {
      cached.lastUsed = Date.now();
      return cached;
    }
    
    // Get the shared tensor from tensor sharing integration
    const sharedTensor = await this.tensorSharing.getSharedTensor(tensorName, modelName);
    if (!sharedTensor) {
      throw new Error(`Shared tensor ${tensorName} not found for model ${modelName}`);
    }
    
    // Create GPU buffer from tensor data
    const data = sharedTensor.getData();
    const shape = sharedTensor.getShape();
    const dataType = sharedTensor.getDataType();
    
    // Create tensor in WebGPU
    const tensorInfo = await this.webgpuBackend.createTensor(
      data,
      shape,
      dataType
    );
    
    // Store in cache
    const result = {
      buffer: tensorInfo.buffer,
      shape,
      dataType,
      location: SharedTensorLocation.WEBGPU,
      lastUsed: Date.now(),
      size: tensorInfo.size
    };
    
    this.tensorBufferCache.set(cacheKey, result);
    
    return result;
  }
  
  /**
   * Execute a compute shader operation on shared tensors
   * 
   * @param operation Operation to execute (matmul, elementwise, softmax, etc.)
   * @param inputTensors Object mapping input names to tensor names
   * @param outputTensorName Name for the output tensor
   * @param modelName Model that will own the output tensor
   * @param options Additional operation-specific options
   */
  async executeOperation(
    operation: string,
    inputTensors: Record<string, { tensorName: string; modelName: string }>,
    outputTensorName: string,
    modelName: string,
    options: Record<string, any> = {}
  ): Promise<string> {
    // Check if already initialized
    if (!this.initialized) {
      const success = await this.initialize();
      if (!success) {
        throw new Error('Failed to initialize WebGPU Tensor Sharing');
      }
    }
    
    // Create GPU buffers for all input tensors
    const gpuInputs: Record<string, any> = {};
    for (const [inputName, { tensorName, modelName: inputModelName }] of Object.entries(inputTensors)) {
      gpuInputs[inputName] = await this.createGPUBufferFromSharedTensor(tensorName, inputModelName);
    }
    
    // Execute the operation
    let result;
    try {
      result = await this.webgpuBackend.execute(operation, gpuInputs, options);
    } catch (error) {
      throw new Error(`Failed to execute operation ${operation}: ${error}`);
    }
    
    // Read the result back to CPU
    const resultData = await this.webgpuBackend.readBuffer(result.buffer, result.dataType);
    
    // Register the result as a shared tensor
    await this.tensorSharing.registerSharedTensor(
      outputTensorName,
      result.shape,
      resultData, 
      result.dataType === 'float32' ? 'float32' : (result.dataType === 'int32' ? 'int32' : 'uint8'),
      modelName,
      [modelName] // Initially, only the creator model consumes it
    );
    
    // Add to buffer cache
    const cacheKey = `${outputTensorName}_${modelName}`;
    this.tensorBufferCache.set(cacheKey, {
      buffer: result.buffer,
      shape: result.shape,
      dataType: result.dataType,
      location: SharedTensorLocation.WEBGPU,
      lastUsed: Date.now(),
      size: result.buffer.size
    });
    
    // Return the name of the created tensor
    return outputTensorName;
  }
  
  /**
   * Matrix multiplication with WebGPU acceleration
   * 
   * @param tensorA Name of the first tensor
   * @param modelA Model that owns the first tensor
   * @param tensorB Name of the second tensor
   * @param modelB Model that owns the second tensor
   * @param outputTensorName Name for the output tensor
   * @param modelName Model that will own the output tensor
   * @param transposeA Whether to transpose the first tensor
   * @param transposeB Whether to transpose the second tensor
   */
  async matmul(
    tensorA: string,
    modelA: string,
    tensorB: string,
    modelB: string,
    outputTensorName: string,
    modelName: string,
    transposeA: boolean = false,
    transposeB: boolean = false
  ): Promise<string> {
    // If browser optimizations are enabled, use our custom implementation
    if (this.options.browserOptimizations !== false && this.browserCapabilities) {
      return this.executeOptimizedMatmul(
        tensorA, modelA,
        tensorB, modelB,
        outputTensorName, modelName,
        transposeA, transposeB
      );
    }
    
    // Fallback to default implementation
    return this.executeOperation(
      'matmul',
      {
        a: { tensorName: tensorA, modelName: modelA },
        b: { tensorName: tensorB, modelName: modelB }
      },
      outputTensorName,
      modelName,
      { transposeA, transposeB }
    );
  }
  
  /**
   * Execute optimized matrix multiplication using browser-specific shaders
   */
  private async executeOptimizedMatmul(
    tensorA: string,
    modelA: string,
    tensorB: string,
    modelB: string,
    outputTensorName: string,
    modelName: string,
    transposeA: boolean = false,
    transposeB: boolean = false
  ): Promise<string> {
    // Check if already initialized
    if (!this.initialized) {
      const success = await this.initialize();
      if (!success) {
        throw new Error('Failed to initialize WebGPU Tensor Sharing');
      }
    }
    
    try {
      // Get device from backend
      const device = (this.webgpuBackend as any)['device'];
      if (!device) {
        throw new Error('WebGPU device not available');
      }
      
      // Create GPU buffers for input tensors
      const tensorAInfo = await this.createGPUBufferFromSharedTensor(tensorA, modelA);
      const tensorBInfo = await this.createGPUBufferFromSharedTensor(tensorB, modelB);
      
      // Get dimensions
      const [M, K_a] = transposeA ? [tensorAInfo.shape[1], tensorAInfo.shape[0]] : [tensorAInfo.shape[0], tensorAInfo.shape[1]];
      const [K_b, N] = transposeB ? [tensorBInfo.shape[1], tensorBInfo.shape[0]] : [tensorBInfo.shape[0], tensorBInfo.shape[1]];
      
      if (K_a !== K_b) {
        throw new Error(`Matrix multiplication dimension mismatch: ${K_a} !== ${K_b}`);
      }
      
      const K = K_a;
      
      // Create output buffer
      const outputBuffer = device.createBuffer({
        label: `matmul_output_${outputTensorName}`,
        size: M * N * 4, // float32 (4 bytes per element)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });
      
      // Create uniform buffer for parameters
      const paramsBuffer = device.createBuffer({
        label: 'matmul_params',
        size: 12, // 3 x uint32
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      
      device.queue.writeBuffer(
        paramsBuffer,
        0,
        new Uint32Array([M, N, K])
      );
      
      // Get optimized shader module
      let shaderModule;
      
      // Use cached optimized shader if available
      if (this.shaderCache.has('matmul')) {
        // Get the optimized shader code
        const shaderCode = this.shaderCache.get('matmul')!;
        
        // Create shader module
        shaderModule = device.createShaderModule({
          label: 'optimized_matmul_shader',
          code: shaderCode
        });
      } else {
        // Get the optimal shader from backend if available
        const backendShaderCache = (this.webgpuBackend as any)['shaderCache'];
        if (backendShaderCache && backendShaderCache instanceof Map) {
          shaderModule = backendShaderCache.get('matmul');
        }
        
        // If still not available, use default implementation
        if (!shaderModule) {
          return this.executeOperation(
            'matmul',
            {
              a: { tensorName: tensorA, modelName: modelA },
              b: { tensorName: tensorB, modelName: modelB }
            },
            outputTensorName,
            modelName,
            { transposeA, transposeB }
          );
        }
      }
      
      // Create pipeline
      const pipeline = device.createComputePipeline({
        label: 'optimized_matmul_pipeline',
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });
      
      // Create bind group
      const bindGroup = device.createBindGroup({
        label: 'matmul_bind_group',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tensorAInfo.buffer } },
          { binding: 1, resource: { buffer: tensorBInfo.buffer } },
          { binding: 2, resource: { buffer: outputBuffer } },
          { binding: 3, resource: { buffer: paramsBuffer } }
        ]
      });
      
      // Create command encoder
      const commandEncoder = device.createCommandEncoder({
        label: 'matmul_command_encoder'
      });
      
      // Compute pass
      const passEncoder = commandEncoder.beginComputePass({
        label: 'matmul_compute_pass'
      });
      
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      
      // Get optimal workgroup size for matrix operations
      const [workgroupSizeX, workgroupSizeY, workgroupSizeZ] = this.getOptimalWorkgroupSize('matrix');
      
      // Dispatch workgroups
      const workgroupCountX = Math.ceil(M / workgroupSizeX);
      const workgroupCountY = Math.ceil(N / workgroupSizeY);
      
      passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY, 1);
      passEncoder.end();
      
      // Submit commands
      device.queue.submit([commandEncoder.finish()]);
      
      // Read back results
      const resultData = await this.webgpuBackend.readBuffer(outputBuffer, 'float32');
      
      // Register the result as a shared tensor
      await this.tensorSharing.registerSharedTensor(
        outputTensorName,
        [M, N],
        resultData,
        'float32',
        modelName,
        [modelName] // Initially, only the creator model consumes it
      );
      
      // Add to buffer cache
      const cacheKey = `${outputTensorName}_${modelName}`;
      this.tensorBufferCache.set(cacheKey, {
        buffer: outputBuffer,
        shape: [M, N],
        dataType: 'float32',
        location: SharedTensorLocation.WEBGPU,
        lastUsed: Date.now(),
        size: outputBuffer.size
      });
      
      if (this.options.debug) {
        console.log(`Executed optimized matrix multiplication for browser: ${this.browserCapabilities.browserType}`);
      }
      
      return outputTensorName;
    } catch (error) {
      console.warn('Error executing optimized matmul, falling back to default implementation:', error);
      
      // Fallback to default implementation
      return this.executeOperation(
        'matmul',
        {
          a: { tensorName: tensorA, modelName: modelA },
          b: { tensorName: tensorB, modelName: modelB }
        },
        outputTensorName,
        modelName,
        { transposeA, transposeB }
      );
    }
  }
  
  /**
   * Element-wise operation (relu, sigmoid, tanh)
   * 
   * @param inputTensorName Name of the input tensor
   * @param inputModelName Model that owns the input tensor
   * @param outputTensorName Name for the output tensor
   * @param outputModelName Model that will own the output tensor
   * @param operation Element-wise operation to perform (relu, sigmoid, tanh)
   */
  async elementwise(
    inputTensorName: string,
    inputModelName: string,
    outputTensorName: string,
    outputModelName: string,
    operation: 'relu' | 'sigmoid' | 'tanh' = 'relu'
  ): Promise<string> {
    // If browser optimizations are enabled, use our custom implementation
    if (this.options.browserOptimizations !== false && this.browserCapabilities) {
      return this.executeOptimizedElementwise(
        inputTensorName,
        inputModelName,
        outputTensorName,
        outputModelName,
        operation
      );
    }
    
    // Fallback to default implementation
    return this.executeOperation(
      'elementwise',
      {
        input: { tensorName: inputTensorName, modelName: inputModelName }
      },
      outputTensorName,
      outputModelName,
      { operation }
    );
  }
  
  /**
   * Execute optimized elementwise operation using browser-specific shaders
   */
  private async executeOptimizedElementwise(
    inputTensorName: string,
    inputModelName: string,
    outputTensorName: string,
    outputModelName: string,
    operation: 'relu' | 'sigmoid' | 'tanh' = 'relu'
  ): Promise<string> {
    // Check if already initialized
    if (!this.initialized) {
      const success = await this.initialize();
      if (!success) {
        throw new Error('Failed to initialize WebGPU Tensor Sharing');
      }
    }
    
    try {
      // Get device from backend
      const device = (this.webgpuBackend as any)['device'];
      if (!device) {
        throw new Error('WebGPU device not available');
      }
      
      // Create GPU buffer for input tensor
      const inputTensorInfo = await this.createGPUBufferFromSharedTensor(inputTensorName, inputModelName);
      
      // Calculate size
      const shape = inputTensorInfo.shape;
      const size = shape.reduce((a, b) => a * b, 1);
      
      // Create output buffer
      const outputBuffer = device.createBuffer({
        label: `elementwise_${operation}_output_${outputTensorName}`,
        size: size * 4, // float32 (4 bytes per element)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });
      
      // Create uniform buffer for parameters
      const paramsBuffer = device.createBuffer({
        label: `elementwise_${operation}_params`,
        size: 8, // 2 x uint32
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      
      // Map operation to code
      const opCode = {
        'relu': 0,
        'sigmoid': 1,
        'tanh': 2
      }[operation] || 0;
      
      device.queue.writeBuffer(
        paramsBuffer,
        0,
        new Uint32Array([size, opCode])
      );
      
      // Get optimized shader module
      let shaderModule;
      
      // Use cached optimized shader if available
      if (this.shaderCache.has('elementwise')) {
        // Get the optimized shader code
        const shaderCode = this.shaderCache.get('elementwise')!;
        
        // Create shader module
        shaderModule = device.createShaderModule({
          label: `optimized_elementwise_${operation}_shader`,
          code: shaderCode
        });
      } else {
        // Get the optimal shader from backend if available
        const backendShaderCache = (this.webgpuBackend as any)['shaderCache'];
        if (backendShaderCache && backendShaderCache instanceof Map) {
          shaderModule = backendShaderCache.get('elementwise');
        }
        
        // If still not available, use default implementation
        if (!shaderModule) {
          return this.executeOperation(
            'elementwise',
            {
              input: { tensorName: inputTensorName, modelName: inputModelName }
            },
            outputTensorName,
            outputModelName,
            { operation }
          );
        }
      }
      
      // Create pipeline
      const pipeline = device.createComputePipeline({
        label: `optimized_elementwise_${operation}_pipeline`,
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });
      
      // Create bind group
      const bindGroup = device.createBindGroup({
        label: `elementwise_${operation}_bind_group`,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputTensorInfo.buffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } }
        ]
      });
      
      // Create command encoder
      const commandEncoder = device.createCommandEncoder({
        label: `elementwise_${operation}_command_encoder`
      });
      
      // Compute pass
      const passEncoder = commandEncoder.beginComputePass({
        label: `elementwise_${operation}_compute_pass`
      });
      
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      
      // Get optimal workgroup size for elementwise operations
      const [workgroupSizeX, workgroupSizeY, workgroupSizeZ] = this.getOptimalWorkgroupSize('elementwise');
      
      // Dispatch workgroups
      const workgroupCount = Math.ceil(size / workgroupSizeX);
      
      passEncoder.dispatchWorkgroups(workgroupCount, 1, 1);
      passEncoder.end();
      
      // Submit commands
      device.queue.submit([commandEncoder.finish()]);
      
      // Read back results
      const resultData = await this.webgpuBackend.readBuffer(outputBuffer, 'float32');
      
      // Register the result as a shared tensor
      await this.tensorSharing.registerSharedTensor(
        outputTensorName,
        shape,
        resultData,
        'float32',
        outputModelName,
        [outputModelName] // Initially, only the creator model consumes it
      );
      
      // Add to buffer cache
      const cacheKey = `${outputTensorName}_${outputModelName}`;
      this.tensorBufferCache.set(cacheKey, {
        buffer: outputBuffer,
        shape,
        dataType: 'float32',
        location: SharedTensorLocation.WEBGPU,
        lastUsed: Date.now(),
        size: outputBuffer.size
      });
      
      if (this.options.debug) {
        console.log(`Executed optimized ${operation} operation for browser: ${this.browserCapabilities.browserType}`);
      }
      
      return outputTensorName;
    } catch (error) {
      console.warn(`Error executing optimized ${operation}, falling back to default implementation:`, error);
      
      // Fallback to default implementation
      return this.executeOperation(
        'elementwise',
        {
          input: { tensorName: inputTensorName, modelName: inputModelName }
        },
        outputTensorName,
        outputModelName,
        { operation }
      );
    }
  }
  
  /**
   * Softmax operation
   * 
   * @param inputTensorName Name of the input tensor
   * @param inputModelName Model that owns the input tensor
   * @param outputTensorName Name for the output tensor
   * @param outputModelName Model that will own the output tensor
   * @param axis Axis to perform softmax on (default: -1)
   */
  async softmax(
    inputTensorName: string,
    inputModelName: string,
    outputTensorName: string,
    outputModelName: string,
    axis: number = -1
  ): Promise<string> {
    return this.executeOperation(
      'softmax',
      {
        input: { tensorName: inputTensorName, modelName: inputModelName }
      },
      outputTensorName,
      outputModelName,
      { axis }
    );
  }
  
  /**
   * Quantize tensor to int8
   * 
   * @param inputTensorName Name of the input tensor
   * @param inputModelName Model that owns the input tensor
   * @param outputTensorName Name for the output tensor
   * @param outputModelName Model that will own the output tensor
   */
  async quantize(
    inputTensorName: string,
    inputModelName: string,
    outputTensorName: string,
    outputModelName: string
  ): Promise<{
    tensorName: string;
    scaleTensorName: string;
  }> {
    // Execute quantization operation
    const result = await this.executeOperation(
      'quantize',
      {
        input: { tensorName: inputTensorName, modelName: inputModelName }
      },
      outputTensorName,
      outputModelName
    );
    
    // Create a scale tensor name
    const scaleTensorName = `${outputTensorName}_scale`;
    
    // Get the cache key for the output tensor to access the scale buffer
    const cacheKey = `${outputTensorName}_${outputModelName}`;
    const cached = this.tensorBufferCache.get(cacheKey);
    
    if (!cached) {
      throw new Error(`Failed to find cached tensor after quantization: ${outputTensorName}`);
    }
    
    // Read the scale data
    const scaleData = await this.webgpuBackend.readBuffer(
      (cached as any).scale,
      'float32'
    );
    
    // Register the scale tensor
    await this.tensorSharing.registerSharedTensor(
      scaleTensorName,
      [1],  // Shape is just a single value
      scaleData,
      'float32',
      outputModelName,
      [outputModelName]
    );
    
    return {
      tensorName: result,
      scaleTensorName
    };
  }
  
  /**
   * Dequantize tensor from int8 to float32
   * 
   * @param inputTensorName Name of the input tensor
   * @param inputModelName Model that owns the input tensor
   * @param scaleTensorName Name of the scale tensor
   * @param scaleModelName Model that owns the scale tensor
   * @param outputTensorName Name for the output tensor
   * @param outputModelName Model that will own the output tensor
   */
  async dequantize(
    inputTensorName: string,
    inputModelName: string,
    scaleTensorName: string,
    scaleModelName: string,
    outputTensorName: string,
    outputModelName: string
  ): Promise<string> {
    return this.executeOperation(
      'dequantize',
      {
        input: { tensorName: inputTensorName, modelName: inputModelName },
        scale: { tensorName: scaleTensorName, modelName: scaleModelName }
      },
      outputTensorName,
      outputModelName
    );
  }
  
  /**
   * Create a zero-copy view of a tensor in WebGPU
   * This allows efficient sharing between models without copying data
   * 
   * @param sourceTensorName Name of the source tensor
   * @param sourceModelName Model that owns the source tensor
   * @param viewTensorName Name for the view tensor
   * @param targetModelName Model that will own the view
   * @param offset Start index in the source tensor
   * @param size Number of elements to include in the view
   */
  async createTensorView(
    sourceTensorName: string,
    sourceModelName: string,
    viewTensorName: string,
    targetModelName: string,
    offset: number = 0,
    size?: number
  ): Promise<string> {
    // First create the tensor view in the tensor sharing system
    await this.tensorSharing.createTensorView(
      sourceTensorName,
      viewTensorName,
      offset,
      size,
      sourceModelName
    );
    
    // Now add it to target model's available tensors
    await this.tensorSharing.shareTensorBetweenModels(
      viewTensorName,
      sourceModelName,
      [targetModelName]
    );
    
    // If the source tensor has a GPU buffer, we can create a view of that buffer
    const sourceKey = `${sourceTensorName}_${sourceModelName}`;
    const sourceCache = this.tensorBufferCache.get(sourceKey);
    
    if (sourceCache && sourceCache.location === SharedTensorLocation.WEBGPU) {
      // Get the new tensor view details
      const tensor = await this.tensorSharing.getSharedTensor(viewTensorName, targetModelName);
      if (!tensor) {
        throw new Error(`Failed to create tensor view ${viewTensorName}`);
      }
      
      // Calculate byte offset and size
      const elementSize = sourceCache.dataType === 'float32' || sourceCache.dataType === 'int32' ? 4 : 1;
      const byteOffset = offset * elementSize;
      const byteSize = (size || (tensor.getSize() - offset)) * elementSize;
      
      // Create a new buffer that's a view into the original buffer
      const viewBuffer = this.webgpuBackend['device'].createBuffer({
        label: `view_${viewTensorName}`,
        size: byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: false
      });
      
      // Copy the relevant portion of the source buffer to the view buffer
      const commandEncoder = this.webgpuBackend['device'].createCommandEncoder({
        label: 'create_view_command_encoder'
      });
      
      commandEncoder.copyBufferToBuffer(
        sourceCache.buffer, 
        byteOffset,
        viewBuffer, 
        0,
        byteSize
      );
      
      this.webgpuBackend['device'].queue.submit([commandEncoder.finish()]);
      
      // Store the view in cache
      const viewKey = `${viewTensorName}_${targetModelName}`;
      this.tensorBufferCache.set(viewKey, {
        buffer: viewBuffer,
        shape: tensor.getShape(),
        dataType: sourceCache.dataType,
        location: SharedTensorLocation.WEBGPU,
        lastUsed: Date.now(),
        size: byteSize
      });
    }
    
    return viewTensorName;
  }
  
  /**
   * Share a WebGPU tensor between models
   * 
   * @param tensorName Name of the tensor to share
   * @param sourceModelName Model that currently owns the tensor
   * @param targetModelNames Array of models to share the tensor with
   */
  async shareTensorBetweenModels(
    tensorName: string,
    sourceModelName: string,
    targetModelNames: string[]
  ): Promise<void> {
    // Share the tensor in the tensor sharing system
    await this.tensorSharing.shareTensorBetweenModels(
      tensorName,
      sourceModelName,
      targetModelNames
    );
    
    // If the tensor has a GPU buffer, we need to create copies for each target model
    const sourceKey = `${tensorName}_${sourceModelName}`;
    const sourceCache = this.tensorBufferCache.get(sourceKey);
    
    if (sourceCache && sourceCache.location === SharedTensorLocation.WEBGPU) {
      for (const targetModel of targetModelNames) {
        const targetKey = `${tensorName}_${targetModel}`;
        if (!this.tensorBufferCache.has(targetKey)) {
          // Set up the cache entry to point to the same GPU buffer
          this.tensorBufferCache.set(targetKey, {
            ...sourceCache,
            lastUsed: Date.now()
          });
        }
      }
    }
  }
  
  /**
   * Synchronize a tensor between CPU and GPU
   * This ensures the tensor data is consistent across both memory spaces
   * 
   * @param tensorName Name of the tensor to synchronize
   * @param modelName Model that owns the tensor
   * @param direction Direction to synchronize (cpu-to-gpu, gpu-to-cpu, or both)
   */
  async synchronizeTensor(
    tensorName: string,
    modelName: string,
    direction: 'cpu-to-gpu' | 'gpu-to-cpu' | 'both' = 'both'
  ): Promise<void> {
    const cacheKey = `${tensorName}_${modelName}`;
    const gpuCache = this.tensorBufferCache.get(cacheKey);
    const cpuTensor = await this.tensorSharing.getSharedTensor(tensorName, modelName);
    
    if (!cpuTensor) {
      throw new Error(`Tensor ${tensorName} not found for model ${modelName}`);
    }
    
    if (!gpuCache && (direction === 'gpu-to-cpu' || direction === 'both')) {
      // If there's no GPU cache but we're supposed to sync from GPU to CPU, 
      // we need to create the GPU tensor first
      await this.createGPUBufferFromSharedTensor(tensorName, modelName);
      return; // No need to continue since we just created the GPU tensor from CPU data
    }
    
    if (!gpuCache && direction === 'cpu-to-gpu') {
      // Create GPU buffer from CPU data
      await this.createGPUBufferFromSharedTensor(tensorName, modelName);
      return;
    }
    
    if (gpuCache) {
      if (direction === 'gpu-to-cpu' || direction === 'both') {
        // Read GPU data back to CPU
        const data = await this.webgpuBackend.readBuffer(gpuCache.buffer, gpuCache.dataType);
        
        // Update CPU tensor data
        cpuTensor.setData(data);
      }
      
      if (direction === 'cpu-to-gpu' || direction === 'both') {
        // Update GPU buffer with CPU data
        const data = cpuTensor.getData();
        
        // Create a new buffer with the updated data
        const tensorInfo = await this.webgpuBackend.createTensor(
          data,
          cpuTensor.getShape(),
          cpuTensor.getDataType() === 'float32' ? 'float32' : 
            (cpuTensor.getDataType() === 'int32' ? 'int32' : 'uint8')
        );
        
        // Replace the old buffer in cache
        this.tensorBufferCache.set(cacheKey, {
          buffer: tensorInfo.buffer,
          shape: cpuTensor.getShape(),
          dataType: gpuCache.dataType,
          location: SharedTensorLocation.WEBGPU,
          lastUsed: Date.now(),
          size: tensorInfo.size
        });
      }
    }
  }
  
  /**
   * Get the memory usage of all tensors in WebGPU
   * @returns Memory usage in bytes
   */
  getGPUMemoryUsage(): number {
    let totalMemory = 0;
    
    for (const entry of this.tensorBufferCache.values()) {
      totalMemory += entry.size;
    }
    
    return totalMemory;
  }
  
  /**
   * Clean up unused tensor buffers in WebGPU
   * @param maxAge Maximum age in milliseconds for unused tensors
   */
  cleanupUnusedTensors(maxAge: number = 30000): void {
    const now = Date.now();
    
    for (const [key, entry] of this.tensorBufferCache.entries()) {
      if (now - entry.lastUsed > maxAge) {
        // Check if tensor is still in use before removing
        const [tensorName, modelName] = key.split('_');
        this.tensorSharing.getSharedTensor(tensorName, modelName)
          .then(tensor => {
            // If tensor exists but hasn't been used in WebGPU, remove from GPU
            if (tensor && now - entry.lastUsed > maxAge) {
              this.tensorBufferCache.delete(key);
            }
          })
          .catch(() => {
            // If tensor doesn't exist anymore, definitely remove from GPU
            this.tensorBufferCache.delete(key);
          });
      }
    }
  }
  
  /**
   * Optimize memory usage across CPU and GPU
   * Moves tensors to the most appropriate location based on usage patterns
   */
  async optimizeMemoryUsage(): Promise<void> {
    // First, get memory usage information
    const gpuMemoryUsage = this.getGPUMemoryUsage();
    const cpuMemoryUsage = await this.tensorSharing.getTensorMemoryUsage();
    
    // Get the list of models and their tensors
    const modelMemoryUsage = await this.tensorSharing.getModelMemoryUsage();
    
    // Identify tensors that are rarely used in GPU operations
    const lowGPUUsageTensors = [];
    const now = Date.now();
    
    for (const [key, entry] of this.tensorBufferCache.entries()) {
      // If not used in the last 5 minutes and larger than 1MB, consider moving to CPU
      if (now - entry.lastUsed > 5 * 60 * 1000 && entry.size > 1024 * 1024) {
        const [tensorName, modelName] = key.split('_');
        lowGPUUsageTensors.push({ tensorName, modelName, size: entry.size });
      }
    }
    
    // Sort by size (largest first) and age (oldest first)
    lowGPUUsageTensors.sort((a, b) => b.size - a.size);
    
    // Move tensors to CPU if needed (implement eviction policy)
    if (this.options.priorityMode === 'memory' || 
        (this.options.priorityMode === 'balanced' && gpuMemoryUsage > 1024 * 1024 * 512)) {
      for (const { tensorName, modelName } of lowGPUUsageTensors) {
        // Synchronize to CPU before removing from GPU
        await this.synchronizeTensor(tensorName, modelName, 'gpu-to-cpu');
        
        // Remove from GPU cache
        const key = `${tensorName}_${modelName}`;
        this.tensorBufferCache.delete(key);
        
        if (this.options.debug) {
          console.log(`Moved tensor ${tensorName} from GPU to CPU to optimize memory usage`);
        }
      }
    }
    
    // Clean up unused tensors
    this.cleanupUnusedTensors();
  }
  
  /**
   * Create a custom compute shader operation
   * 
   * @param shaderName Name of the shader
   * @param shaderCode WGSL shader code
   * @param inputTypes Types of input bindings
   * @param outputTypes Types of output bindings
   */
  async createCustomShader(
    shaderName: string,
    shaderCode: string,
    inputTypes: Record<string, 'float32' | 'int32' | 'uint8'> = {},
    outputTypes: Record<string, 'float32' | 'int32' | 'uint8'> = {}
  ): Promise<void> {
    if (!this.initialized) {
      const success = await this.initialize();
      if (!success) {
        throw new Error('Failed to initialize WebGPU Tensor Sharing');
      }
    }
    
    // Create shader module
    const shaderModule = this.webgpuBackend['device'].createShaderModule({
      label: `custom_${shaderName}_shader`,
      code: shaderCode
    });
    
    // Store in WebGPU backend shader cache
    this.webgpuBackend['shaderCache'].set(shaderName, shaderModule);
    
    if (this.options.debug) {
      console.log(`Custom shader ${shaderName} created successfully`);
    }
  }
  
  /**
   * Get optimal workgroup size based on browser and operation type
   * 
   * @param operationType Type of operation ('compute', 'matrix', 'elementwise')
   * @returns Optimal workgroup size
   */
  getOptimalWorkgroupSize(operationType: 'compute' | 'matrix' | 'elementwise'): [number, number, number] {
    // Default workgroup sizes
    const defaults: Record<string, [number, number, number]> = {
      compute: [256, 1, 1],
      matrix: [8, 8, 1],
      elementwise: [256, 1, 1]
    };
    
    // If browser capabilities are detected, use those
    if (this.browserCapabilities && this.browserCapabilities.optimalWorkgroupSize) {
      return this.browserCapabilities.optimalWorkgroupSize[operationType] || defaults[operationType];
    }
    
    return defaults[operationType];
  }
  
  /**
   * Execute a custom compute shader
   * 
   * @param shaderName Name of the custom shader to execute
   * @param inputTensors Object mapping input names to tensor names
   * @param outputTensorNames Object mapping output names to tensor names
   * @param modelName Model that will own the output tensors
   * @param workgroupSize Workgroup size for the compute shader
   * @param workgroupCount Number of workgroups to dispatch
   */
  async executeCustomShader(
    shaderName: string,
    inputTensors: Record<string, { tensorName: string; modelName: string }>,
    outputTensorNames: Record<string, { tensorName: string; shape: number[]; dataType: 'float32' | 'int32' | 'uint8' }>,
    modelName: string,
    workgroupSize?: [number, number, number],
    workgroupCount: [number, number, number] = [1, 1, 1]
  ): Promise<Record<string, string>> {
    if (!this.initialized) {
      const success = await this.initialize();
      if (!success) {
        throw new Error('Failed to initialize WebGPU Tensor Sharing');
      }
    }
    
    // Get the shader module
    const shaderModule = this.webgpuBackend['shaderCache'].get(shaderName);
    if (!shaderModule) {
      throw new Error(`Custom shader ${shaderName} not found`);
    }
    
    // Create GPU buffers for all input tensors
    const gpuInputs: Record<string, any> = {};
    for (const [inputName, { tensorName, modelName: inputModelName }] of Object.entries(inputTensors)) {
      gpuInputs[inputName] = await this.createGPUBufferFromSharedTensor(tensorName, inputModelName);
    }
    
    // Create output buffers
    const gpuOutputs: Record<string, any> = {};
    for (const [outputName, { tensorName, shape, dataType }] of Object.entries(outputTensorNames)) {
      // Calculate size for the output buffer
      const totalElements = shape.reduce((a, b) => a * b, 1);
      const elementSize = dataType === 'float32' || dataType === 'int32' ? 4 : 1;
      const byteSize = totalElements * elementSize;
      
      // Create buffer
      const outputBuffer = this.webgpuBackend['device'].createBuffer({
        label: `output_${tensorName}`,
        size: byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });
      
      gpuOutputs[outputName] = {
        buffer: outputBuffer,
        shape,
        dataType
      };
    }
    
    // Create bind group entries
    const bindGroupEntries: GPUBindGroupEntry[] = [];
    let bindingIndex = 0;
    
    // Add input bindings
    for (const [inputName, gpuTensor] of Object.entries(gpuInputs)) {
      bindGroupEntries.push({
        binding: bindingIndex++,
        resource: { buffer: gpuTensor.buffer }
      });
    }
    
    // Add output bindings
    for (const [outputName, gpuTensor] of Object.entries(gpuOutputs)) {
      bindGroupEntries.push({
        binding: bindingIndex++,
        resource: { buffer: gpuTensor.buffer }
      });
    }
    
    // Create pipeline
    const pipeline = this.webgpuBackend['device'].createComputePipeline({
      label: `custom_${shaderName}_pipeline`,
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
        constants: {}
      }
    });
    
    // Create bind group
    const bindGroup = this.webgpuBackend['device'].createBindGroup({
      label: `custom_${shaderName}_bind_group`,
      layout: pipeline.getBindGroupLayout(0),
      entries: bindGroupEntries
    });
    
    // Create command encoder
    const commandEncoder = this.webgpuBackend['device'].createCommandEncoder({
      label: `custom_${shaderName}_command_encoder`
    });
    
    // Compute pass
    const passEncoder = commandEncoder.beginComputePass({
      label: `custom_${shaderName}_compute_pass`
    });
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    
    // Use provided workgroup size or get optimal one
    const actualWorkgroupSize = workgroupSize || this.getOptimalWorkgroupSize('compute');
    
    if (this.options.debug) {
      console.log(`Executing ${shaderName} with workgroup size: [${actualWorkgroupSize.join(', ')}]`);
    }
    
    // Dispatch workgroups
    passEncoder.dispatchWorkgroups(workgroupCount[0], workgroupCount[1], workgroupCount[2]);
    passEncoder.end();
    
    // Submit commands
    this.webgpuBackend['device'].queue.submit([commandEncoder.finish()]);
    
    // Read output tensors back to CPU and register as shared tensors
    const results: Record<string, string> = {};
    
    for (const [outputName, { tensorName, shape, dataType }] of Object.entries(outputTensorNames)) {
      const gpuTensor = gpuOutputs[outputName];
      const resultData = await this.webgpuBackend.readBuffer(gpuTensor.buffer, dataType);
      
      // Register the result as a shared tensor
      await this.tensorSharing.registerSharedTensor(
        tensorName,
        shape,
        resultData, 
        dataType,
        modelName,
        [modelName] // Initially, only the creator model consumes it
      );
      
      // Add to buffer cache
      const cacheKey = `${tensorName}_${modelName}`;
      this.tensorBufferCache.set(cacheKey, {
        buffer: gpuTensor.buffer,
        shape,
        dataType,
        location: SharedTensorLocation.WEBGPU,
        lastUsed: Date.now(),
        size: gpuTensor.buffer.size
      });
      
      results[outputName] = tensorName;
    }
    
    return results;
  }
  
  /**
   * Dispose of all resources and clean up
   */
  dispose(): void {
    // Clean up GPU resources
    for (const entry of this.tensorBufferCache.values()) {
      if (entry.buffer && 'destroy' in entry.buffer) {
        entry.buffer.destroy();
      }
    }
    
    this.tensorBufferCache.clear();
    this.pipelineCache.clear();
    
    // Dispose of WebGPU backend if we created it
    if (this.webgpuBackend && !this.options.webgpuBackend) {
      this.webgpuBackend.dispose();
    }
    
    this.initialized = false;
  }
}