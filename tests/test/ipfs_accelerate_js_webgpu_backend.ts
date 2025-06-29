/**
 * WebGPU Backend Implementation
 * Provides hardware acceleration using the WebGPU API
 */

import { HardwareBackend, HardwareBackendType } from '../hardware_abstraction';

/**
 * WebGPU Backend Options
 */
export interface WebGPUBackendOptions {
  /**
   * Preferred GPU adapter type (high-performance or low-power)
   */
  powerPreference?: 'high-performance' | 'low-power';
  
  /**
   * Force using the fallback adapter (CPU implementation) if available
   */
  forceFallbackAdapter?: boolean;
  
  /**
   * Required features to request from the adapter
   */
  requiredFeatures?: string[];
  
  /**
   * Required limits to request from the adapter
   */
  requiredLimits?: Record<string, number>;
  
  /**
   * Shader compilation options
   */
  shaderCompilation?: {
    /**
     * Precompile shaders during initialization
     */
    precompile?: boolean;
    
    /**
     * Cache compiled shaders
     */
    cacheShaders?: boolean;
    
    /**
     * Custom WGSL shaders path
     */
    shadersPath?: string;
    
    /**
     * Browser-specific shader optimizations
     */
    browserOptimizations?: boolean;
  };
  
  /**
   * Memory limits and buffer management options
   */
  memory?: {
    /**
     * Maximum memory size to allocate (in bytes)
     */
    maxMemorySize?: number;
    
    /**
     * Enable garbage collection of unused buffers
     */
    enableGarbageCollection?: boolean;
    
    /**
     * Threshold for garbage collection (in bytes)
     */
    garbageCollectionThreshold?: number;
  };
}

/**
 * WebGPU Backend
 * Implements hardware acceleration using the WebGPU API
 */
export class WebGPUBackend implements HardwareBackend {
  readonly type: HardwareBackendType = 'webgpu';
  
  private options: WebGPUBackendOptions;
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private bufferCache: Map<string, GPUBuffer> = new Map();
  private shaderCache: Map<string, GPUShaderModule> = new Map();
  private pipelineCache: Map<string, GPUComputePipeline> = new Map();
  private memoryAllocated = 0;
  
  /**
   * Create a WebGPU backend
   */
  constructor(options: WebGPUBackendOptions = {}) {
    this.options = {
      powerPreference: 'high-performance',
      forceFallbackAdapter: false,
      shaderCompilation: {
        precompile: true,
        cacheShaders: true,
        browserOptimizations: true
      },
      memory: {
        enableGarbageCollection: true,
        garbageCollectionThreshold: 1024 * 1024 * 256 // 256MB
      },
      ...options
    };
  }
  
  /**
   * Check if WebGPU is supported in this browser
   */
  async isSupported(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        return false;
      }
      
      // Try requesting an adapter to confirm WebGPU support
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: this.options.powerPreference,
        forceFallbackAdapter: this.options.forceFallbackAdapter
      });
      
      return !!adapter;
    } catch (error) {
      console.warn('WebGPU not supported:', error);
      return false;
    }
  }
  
  /**
   * Initialize the WebGPU backend
   */
  async initialize(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        throw new Error('WebGPU not supported in this browser');
      }
      
      // Request adapter
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: this.options.powerPreference,
        forceFallbackAdapter: this.options.forceFallbackAdapter
      });
      
      if (!this.adapter) {
        throw new Error('Failed to get WebGPU adapter');
      }
      
      // Request device
      this.device = await this.adapter.requestDevice({
        requiredFeatures: this.options.requiredFeatures || [],
        requiredLimits: this.options.requiredLimits || {}
      });
      
      if (!this.device) {
        throw new Error('Failed to get WebGPU device');
      }
      
      // Add error handler
      this.device.addEventListener('uncapturederror', (event) => {
        console.error('WebGPU device error:', event);
      });
      
      // Precompile common shaders if requested
      if (this.options.shaderCompilation?.precompile) {
        await this.precompileShaders();
      }
      
      return true;
    } catch (error) {
      console.error('WebGPU initialization failed:', error);
      this.adapter = null;
      this.device = null;
      return false;
    }
  }
  
  /**
   * Get WebGPU-specific capabilities
   */
  async getCapabilities(): Promise<Record<string, any>> {
    if (!this.adapter || !this.device) {
      throw new Error('WebGPU backend not initialized');
    }
    
    const capabilities = {
      adapter: {
        isFallbackAdapter: this.adapter.isFallbackAdapter
      },
      features: Array.from(this.device.features).reduce((obj, feature) => {
        obj[feature] = true;
        return obj;
      }, {} as Record<string, boolean>),
      limits: {}
    };
    
    // Add device limits
    const limitNames = [
      'maxBindGroups',
      'maxBindingsPerBindGroup',
      'maxBufferSize',
      'maxComputeWorkgroupSizeX',
      'maxComputeWorkgroupSizeY',
      'maxComputeWorkgroupSizeZ',
      'maxComputeWorkgroupsPerDimension',
      'maxStorageBufferBindingSize'
    ];
    
    for (const limit of limitNames) {
      (capabilities.limits as any)[limit] = this.device.limits.get(limit);
    }
    
    return capabilities;
  }
  
  /**
   * Precompile common shader modules
   */
  private async precompileShaders(): Promise<void> {
    if (!this.device) return;
    
    // Common shader operations for AI workloads
    const shaders = {
      // Matrix multiplication
      matmul: `
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;
        
        struct Params {
          M: u32,
          N: u32,
          K: u32
        }
        
        @group(0) @binding(3) var<uniform> params: Params;
        
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let row = global_id.x;
          let col = global_id.y;
          
          if (row >= params.M || col >= params.N) {
            return;
          }
          
          var sum = 0.0;
          for (var k = 0u; k < params.K; k = k + 1u) {
            sum = sum + a[row * params.K + k] * b[k * params.N + col];
          }
          
          c[row * params.N + col] = sum;
        }
      `,
      
      // Element-wise operations
      elementwise: `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        
        struct Params {
          length: u32,
          op: u32    // 0: relu, 1: sigmoid, 2: tanh
        }
        
        @group(0) @binding(2) var<uniform> params: Params;
        
        @compute @workgroup_size(256, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          
          if (idx >= params.length) {
            return;
          }
          
          let x = input[idx];
          var result = x;
          
          switch(params.op) {
            case 0u: {  // ReLU
              result = max(0.0, x);
              break;
            }
            case 1u: {  // Sigmoid
              result = 1.0 / (1.0 + exp(-x));
              break;
            }
            case 2u: {  // Tanh
              result = tanh(x);
              break;
            }
            default: {
              result = x;
            }
          }
          
          output[idx] = result;
        }
      `,
      
      // Softmax
      softmax: `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        
        struct Params {
          length: u32,
          batch_size: u32
        }
        
        @group(0) @binding(2) var<uniform> params: Params;
        
        @compute @workgroup_size(256, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let batch_idx = global_id.x;
          
          if (batch_idx >= params.batch_size) {
            return;
          }
          
          let seq_len = params.length / params.batch_size;
          let start_idx = batch_idx * seq_len;
          
          // Find max for numerical stability
          var max_val = input[start_idx];
          for (var i = 1u; i < seq_len; i = i + 1u) {
            let idx = start_idx + i;
            max_val = max(max_val, input[idx]);
          }
          
          // Compute exp(x - max) and sum
          var sum = 0.0;
          for (var i = 0u; i < seq_len; i = i + 1u) {
            let idx = start_idx + i;
            let exp_val = exp(input[idx] - max_val);
            output[idx] = exp_val;
            sum = sum + exp_val;
          }
          
          // Normalize
          for (var i = 0u; i < seq_len; i = i + 1u) {
            let idx = start_idx + i;
            output[idx] = output[idx] / sum;
          }
        }
      `,
      
      // Quantization (int8)
      quantize: `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<i32>;
        @group(0) @binding(2) var<storage, read_write> scale: array<f32>;
        
        struct Params {
          length: u32,
          scale_index: u32
        }
        
        @group(0) @binding(3) var<uniform> params: Params;
        
        @compute @workgroup_size(256, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          
          if (idx >= params.length) {
            return;
          }
          
          // Find max absolute value in the block for scale
          if (idx == 0u) {
            var max_abs = 0.0;
            for (var i = 0u; i < params.length; i = i + 1u) {
              max_abs = max(max_abs, abs(input[i]));
            }
            
            // Compute scale (127.0 for int8)
            if (max_abs > 0.0) {
              scale[params.scale_index] = 127.0 / max_abs;
            } else {
              scale[params.scale_index] = 1.0;
            }
          }
          
          // Wait for scale to be computed
          workgroupBarrier();
          
          // Quantize value
          let current_scale = scale[params.scale_index];
          let value = input[idx];
          let quantized = i32(round(value * current_scale));
          
          // Clamp to int8 range (-127 to 127, saving -128 for padding)
          output[idx] = clamp(quantized, -127, 127);
        }
      `,
      
      // Dequantization (int8 to float32)
      dequantize: `
        @group(0) @binding(0) var<storage, read> input: array<i32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @group(0) @binding(2) var<storage, read> scale: array<f32>;
        
        struct Params {
          length: u32,
          scale_index: u32
        }
        
        @group(0) @binding(3) var<uniform> params: Params;
        
        @compute @workgroup_size(256, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          
          if (idx >= params.length) {
            return;
          }
          
          let current_scale = scale[params.scale_index];
          let value = input[idx];
          
          // Dequantize
          output[idx] = f32(value) / current_scale;
        }
      `
    };
    
    // Compile and cache all shaders
    for (const [name, code] of Object.entries(shaders)) {
      const shaderModule = this.device.createShaderModule({
        label: `${name}_shader`,
        code
      });
      
      this.shaderCache.set(name, shaderModule);
    }
  }
  
  /**
   * Create a tensor from data
   */
  async createTensor(data: Float32Array | Uint8Array | Int32Array, shape: number[], dataType = 'float32'): Promise<{
    buffer: GPUBuffer;
    shape: number[];
    dataType: string;
    size: number;
  }> {
    if (!this.device) {
      throw new Error('WebGPU backend not initialized');
    }
    
    // Calculate size in bytes
    const elementSize = dataType === 'float32' || dataType === 'int32' ? 4 : 1;
    const size = data.byteLength;
    
    // Create buffer
    const buffer = this.device.createBuffer({
      label: `tensor_${shape.join('x')}_${dataType}`,
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true
    });
    
    // Write data to buffer
    const arrayBuffer = buffer.getMappedRange();
    
    if (dataType === 'float32') {
      new Float32Array(arrayBuffer).set(data as Float32Array);
    } else if (dataType === 'int32') {
      new Int32Array(arrayBuffer).set(data as Int32Array);
    } else {
      new Uint8Array(arrayBuffer).set(data as Uint8Array);
    }
    
    buffer.unmap();
    
    // Update memory tracking
    this.memoryAllocated += size;
    this.maybeGarbageCollect();
    
    return {
      buffer,
      shape,
      dataType,
      size
    };
  }
  
  /**
   * Execute operation on WebGPU device
   */
  async execute(
    operation: string,
    inputs: Record<string, any>,
    options: Record<string, any> = {}
  ): Promise<any> {
    if (!this.device) {
      throw new Error('WebGPU backend not initialized');
    }
    
    switch (operation) {
      case 'matmul':
        return this.executeMatmul(inputs, options);
        
      case 'elementwise':
        return this.executeElementwise(inputs, options);
        
      case 'softmax':
        return this.executeSoftmax(inputs, options);
        
      case 'quantize':
        return this.executeQuantize(inputs, options);
        
      case 'dequantize':
        return this.executeDequantize(inputs, options);
        
      default:
        throw new Error(`Unsupported operation: ${operation}`);
    }
  }
  
  /**
   * Matrix multiplication operation
   */
  private async executeMatmul(
    inputs: {
      a: { buffer: GPUBuffer; shape: number[] };
      b: { buffer: GPUBuffer; shape: number[] };
    },
    options: {
      transposeA?: boolean;
      transposeB?: boolean;
    } = {}
  ): Promise<{
    buffer: GPUBuffer;
    shape: number[];
    dataType: string;
  }> {
    const { a, b } = inputs;
    const { transposeA = false, transposeB = false } = options;
    
    // Get dimensions
    const [M, K_a] = transposeA ? [a.shape[1], a.shape[0]] : [a.shape[0], a.shape[1]];
    const [K_b, N] = transposeB ? [b.shape[1], b.shape[0]] : [b.shape[0], b.shape[1]];
    
    if (K_a !== K_b) {
      throw new Error(`Matrix multiplication dimension mismatch: ${K_a} != ${K_b}`);
    }
    
    const K = K_a;
    
    // Create output buffer
    const outputBuffer = this.device!.createBuffer({
      label: 'matmul_output',
      size: M * N * 4, // float32 (4 bytes per element)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    // Create uniform buffer for parameters
    const paramsBuffer = this.device!.createBuffer({
      label: 'matmul_params',
      size: 12, // 3 x uint32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    this.device!.queue.writeBuffer(
      paramsBuffer,
      0,
      new Uint32Array([M, N, K])
    );
    
    // Get or create shader module
    let shaderModule = this.shaderCache.get('matmul');
    if (!shaderModule) {
      const shaderCode = `
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;
        
        struct Params {
          M: u32,
          N: u32,
          K: u32
        }
        
        @group(0) @binding(3) var<uniform> params: Params;
        
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let row = global_id.x;
          let col = global_id.y;
          
          if (row >= params.M || col >= params.N) {
            return;
          }
          
          var sum = 0.0;
          for (var k = 0u; k < params.K; k = k + 1u) {
            sum = sum + a[row * params.K + k] * b[k * params.N + col];
          }
          
          c[row * params.N + col] = sum;
        }
      `;
      
      shaderModule = this.device!.createShaderModule({
        label: 'matmul_shader',
        code: shaderCode
      });
      
      this.shaderCache.set('matmul', shaderModule);
    }
    
    // Create pipeline
    const pipeline = this.device!.createComputePipeline({
      label: 'matmul_pipeline',
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
    
    // Create bind group
    const bindGroup = this.device!.createBindGroup({
      label: 'matmul_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a.buffer } },
        { binding: 1, resource: { buffer: b.buffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } }
      ]
    });
    
    // Create command encoder
    const commandEncoder = this.device!.createCommandEncoder({
      label: 'matmul_command_encoder'
    });
    
    // Compute pass
    const passEncoder = commandEncoder.beginComputePass({
      label: 'matmul_compute_pass'
    });
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    
    // Dispatch workgroups
    const workgroupSizeX = 8;
    const workgroupSizeY = 8;
    const workgroupCountX = Math.ceil(M / workgroupSizeX);
    const workgroupCountY = Math.ceil(N / workgroupSizeY);
    
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    passEncoder.end();
    
    // Submit commands
    this.device!.queue.submit([commandEncoder.finish()]);
    
    // Return result buffer with shape
    return {
      buffer: outputBuffer,
      shape: [M, N],
      dataType: 'float32'
    };
  }
  
  /**
   * Element-wise operations (relu, sigmoid, tanh)
   */
  private async executeElementwise(
    inputs: {
      input: { buffer: GPUBuffer; shape: number[] };
    },
    options: {
      operation?: 'relu' | 'sigmoid' | 'tanh';
    } = {}
  ): Promise<{
    buffer: GPUBuffer;
    shape: number[];
    dataType: string;
  }> {
    const { input } = inputs;
    const { operation = 'relu' } = options;
    
    // Map operation to code
    const opCode = {
      'relu': 0,
      'sigmoid': 1,
      'tanh': 2
    }[operation] || 0;
    
    // Calculate size
    const size = input.shape.reduce((a, b) => a * b, 1);
    
    // Create output buffer
    const outputBuffer = this.device!.createBuffer({
      label: `${operation}_output`,
      size: size * 4, // float32 (4 bytes per element)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    // Create uniform buffer for parameters
    const paramsBuffer = this.device!.createBuffer({
      label: `${operation}_params`,
      size: 8, // 2 x uint32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    this.device!.queue.writeBuffer(
      paramsBuffer,
      0,
      new Uint32Array([size, opCode])
    );
    
    // Get or create shader module
    let shaderModule = this.shaderCache.get('elementwise');
    if (!shaderModule) {
      // Shader module is precompiled in initialize()
      shaderModule = this.shaderCache.get('elementwise')!;
    }
    
    // Create pipeline
    const pipeline = this.device!.createComputePipeline({
      label: `${operation}_pipeline`,
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
    
    // Create bind group
    const bindGroup = this.device!.createBindGroup({
      label: `${operation}_bind_group`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } }
      ]
    });
    
    // Create command encoder
    const commandEncoder = this.device!.createCommandEncoder({
      label: `${operation}_command_encoder`
    });
    
    // Compute pass
    const passEncoder = commandEncoder.beginComputePass({
      label: `${operation}_compute_pass`
    });
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    
    // Dispatch workgroups
    const workgroupSize = 256;
    const workgroupCount = Math.ceil(size / workgroupSize);
    
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();
    
    // Submit commands
    this.device!.queue.submit([commandEncoder.finish()]);
    
    // Return result buffer with shape
    return {
      buffer: outputBuffer,
      shape: input.shape,
      dataType: 'float32'
    };
  }
  
  /**
   * Softmax operation
   */
  private async executeSoftmax(
    inputs: {
      input: { buffer: GPUBuffer; shape: number[] };
    },
    options: {
      axis?: number;
    } = {}
  ): Promise<{
    buffer: GPUBuffer;
    shape: number[];
    dataType: string;
  }> {
    const { input } = inputs;
    const { axis = -1 } = options;
    
    // Calculate dimensions
    const shape = input.shape;
    const actualAxis = axis < 0 ? shape.length + axis : axis;
    
    if (actualAxis < 0 || actualAxis >= shape.length) {
      throw new Error(`Invalid softmax axis: ${axis}`);
    }
    
    // Calculate batch size and sequence length
    let batchSize = 1;
    let seqLen = 1;
    
    for (let i = 0; i < shape.length; i++) {
      if (i === actualAxis) {
        seqLen = shape[i];
      } else {
        batchSize *= shape[i];
      }
    }
    
    const size = batchSize * seqLen;
    
    // Create output buffer
    const outputBuffer = this.device!.createBuffer({
      label: 'softmax_output',
      size: size * 4, // float32 (4 bytes per element)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    // Create uniform buffer for parameters
    const paramsBuffer = this.device!.createBuffer({
      label: 'softmax_params',
      size: 8, // 2 x uint32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    this.device!.queue.writeBuffer(
      paramsBuffer,
      0,
      new Uint32Array([size, batchSize])
    );
    
    // Get or create shader module
    let shaderModule = this.shaderCache.get('softmax');
    if (!shaderModule) {
      // Shader module is precompiled in initialize()
      shaderModule = this.shaderCache.get('softmax')!;
    }
    
    // Create pipeline
    const pipeline = this.device!.createComputePipeline({
      label: 'softmax_pipeline',
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
    
    // Create bind group
    const bindGroup = this.device!.createBindGroup({
      label: 'softmax_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } }
      ]
    });
    
    // Create command encoder
    const commandEncoder = this.device!.createCommandEncoder({
      label: 'softmax_command_encoder'
    });
    
    // Compute pass
    const passEncoder = commandEncoder.beginComputePass({
      label: 'softmax_compute_pass'
    });
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    
    // Dispatch workgroups - one workgroup per batch
    const workgroupCount = batchSize;
    
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();
    
    // Submit commands
    this.device!.queue.submit([commandEncoder.finish()]);
    
    // Return result buffer with shape
    return {
      buffer: outputBuffer,
      shape: input.shape,
      dataType: 'float32'
    };
  }
  
  /**
   * Quantize to int8 operation
   */
  private async executeQuantize(
    inputs: {
      input: { buffer: GPUBuffer; shape: number[] };
    },
    options: {
      scaleIndex?: number;
    } = {}
  ): Promise<{
    buffer: GPUBuffer;
    scale: GPUBuffer;
    shape: number[];
    dataType: string;
  }> {
    const { input } = inputs;
    const { scaleIndex = 0 } = options;
    
    // Calculate size
    const size = input.shape.reduce((a, b) => a * b, 1);
    
    // Create output buffer
    const outputBuffer = this.device!.createBuffer({
      label: 'quantize_output',
      size: size * 4, // int32 (4 bytes per element)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    // Create scale buffer
    const scaleBuffer = this.device!.createBuffer({
      label: 'quantize_scale',
      size: 4, // float32 (4 bytes per element)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    // Create uniform buffer for parameters
    const paramsBuffer = this.device!.createBuffer({
      label: 'quantize_params',
      size: 8, // 2 x uint32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    this.device!.queue.writeBuffer(
      paramsBuffer,
      0,
      new Uint32Array([size, scaleIndex])
    );
    
    // Get or create shader module
    let shaderModule = this.shaderCache.get('quantize');
    if (!shaderModule) {
      // Shader module is precompiled in initialize()
      shaderModule = this.shaderCache.get('quantize')!;
    }
    
    // Create pipeline
    const pipeline = this.device!.createComputePipeline({
      label: 'quantize_pipeline',
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
    
    // Create bind group
    const bindGroup = this.device!.createBindGroup({
      label: 'quantize_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: scaleBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } }
      ]
    });
    
    // Create command encoder
    const commandEncoder = this.device!.createCommandEncoder({
      label: 'quantize_command_encoder'
    });
    
    // Compute pass
    const passEncoder = commandEncoder.beginComputePass({
      label: 'quantize_compute_pass'
    });
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    
    // Dispatch workgroups
    const workgroupSize = 256;
    const workgroupCount = Math.ceil(size / workgroupSize);
    
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();
    
    // Submit commands
    this.device!.queue.submit([commandEncoder.finish()]);
    
    // Return result buffer with shape
    return {
      buffer: outputBuffer,
      scale: scaleBuffer,
      shape: input.shape,
      dataType: 'int32'
    };
  }
  
  /**
   * Dequantize from int8 operation
   */
  private async executeDequantize(
    inputs: {
      input: { buffer: GPUBuffer; shape: number[] };
      scale: { buffer: GPUBuffer };
    },
    options: {
      scaleIndex?: number;
    } = {}
  ): Promise<{
    buffer: GPUBuffer;
    shape: number[];
    dataType: string;
  }> {
    const { input, scale } = inputs;
    const { scaleIndex = 0 } = options;
    
    // Calculate size
    const size = input.shape.reduce((a, b) => a * b, 1);
    
    // Create output buffer
    const outputBuffer = this.device!.createBuffer({
      label: 'dequantize_output',
      size: size * 4, // float32 (4 bytes per element)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    // Create uniform buffer for parameters
    const paramsBuffer = this.device!.createBuffer({
      label: 'dequantize_params',
      size: 8, // 2 x uint32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    this.device!.queue.writeBuffer(
      paramsBuffer,
      0,
      new Uint32Array([size, scaleIndex])
    );
    
    // Get or create shader module
    let shaderModule = this.shaderCache.get('dequantize');
    if (!shaderModule) {
      // Shader module is precompiled in initialize()
      shaderModule = this.shaderCache.get('dequantize')!;
    }
    
    // Create pipeline
    const pipeline = this.device!.createComputePipeline({
      label: 'dequantize_pipeline',
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
    
    // Create bind group
    const bindGroup = this.device!.createBindGroup({
      label: 'dequantize_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: scale.buffer } },
        { binding: 3, resource: { buffer: paramsBuffer } }
      ]
    });
    
    // Create command encoder
    const commandEncoder = this.device!.createCommandEncoder({
      label: 'dequantize_command_encoder'
    });
    
    // Compute pass
    const passEncoder = commandEncoder.beginComputePass({
      label: 'dequantize_compute_pass'
    });
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    
    // Dispatch workgroups
    const workgroupSize = 256;
    const workgroupCount = Math.ceil(size / workgroupSize);
    
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();
    
    // Submit commands
    this.device!.queue.submit([commandEncoder.finish()]);
    
    // Return result buffer with shape
    return {
      buffer: outputBuffer,
      shape: input.shape,
      dataType: 'float32'
    };
  }
  
  /**
   * Read data from a GPU buffer
   */
  async readBuffer(buffer: GPUBuffer, dataType = 'float32'): Promise<Float32Array | Int32Array | Uint8Array> {
    if (!this.device) {
      throw new Error('WebGPU backend not initialized');
    }
    
    // Create a staging buffer for reading
    const stagingBuffer = this.device.createBuffer({
      label: 'read_staging_buffer',
      size: buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    
    // Create command encoder
    const commandEncoder = this.device.createCommandEncoder({
      label: 'read_command_encoder'
    });
    
    // Copy data from the buffer to the staging buffer
    commandEncoder.copyBufferToBuffer(
      buffer, 0,
      stagingBuffer, 0,
      buffer.size
    );
    
    // Submit commands
    this.device.queue.submit([commandEncoder.finish()]);
    
    // Map the staging buffer for reading
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    
    // Get the mapped range data
    const arrayBuffer = stagingBuffer.getMappedRange();
    
    // Create a typed array view based on the data type
    let data: Float32Array | Int32Array | Uint8Array;
    
    if (dataType === 'float32') {
      data = new Float32Array(arrayBuffer.slice(0));
    } else if (dataType === 'int32') {
      data = new Int32Array(arrayBuffer.slice(0));
    } else {
      data = new Uint8Array(arrayBuffer.slice(0));
    }
    
    // Unmap the buffer
    stagingBuffer.unmap();
    
    // Destroy the staging buffer
    stagingBuffer.destroy();
    
    return data;
  }
  
  /**
   * Run garbage collection if memory usage is above threshold
   */
  private maybeGarbageCollect(): void {
    if (!this.options.memory?.enableGarbageCollection) {
      return;
    }
    
    const threshold = this.options.memory.garbageCollectionThreshold || 1024 * 1024 * 256;
    
    if (this.memoryAllocated > threshold) {
      this.garbageCollect();
    }
  }
  
  /**
   * Force garbage collection of unused buffers
   */
  private garbageCollect(): void {
    // Clear caches
    this.bufferCache.forEach((buffer) => {
      buffer.destroy();
    });
    
    this.bufferCache.clear();
    
    // Explicitly run V8 garbage collection if available
    if (typeof global !== 'undefined' && global.gc) {
      global.gc();
    }
    
    // Reset memory tracking
    this.memoryAllocated = 0;
  }
  
  /**
   * Release all resources
   */
  dispose(): void {
    // Destroy all cached buffers
    this.bufferCache.forEach((buffer) => {
      buffer.destroy();
    });
    
    this.bufferCache.clear();
    
    // Clear shader and pipeline caches
    this.shaderCache.clear();
    this.pipelineCache.clear();
    
    // Destroy device
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
    
    // Reset memory tracking
    this.memoryAllocated = 0;
    
    // Reset adapter reference
    this.adapter = null;
  }
}