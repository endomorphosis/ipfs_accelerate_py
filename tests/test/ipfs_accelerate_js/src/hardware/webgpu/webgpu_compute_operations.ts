/**
 * WebGPU Compute Operations
 * High-performance implementation of tensor operations using WebGPU compute shaders
 */

import { Tensor } from '../../tensor/tensor';
import { WebGPUBackend } from './backend';
import { WebGPUBufferManager } from './buffer_manager';
import { WebGPUMatrixMultiplication, MatrixOperationOptions } from './matrix_operations';
import { BrowserOptimizedMatrixOperations, BrowserType, detectBrowserType } from './browser_optimized_operations';
import { 
  generateQuantizeShader, 
  generateDequantizeShader, 
  generateQuantizedMatmulShader,
  generateKVCacheQuantizeShader,
  generateKVCacheDequantizeShader,
  generateUltraLowPrecisionQuantizeShader,
  generateUltraLowPrecisionDequantizeShader,
  QuantizationShaderOptions
} from './optimizations/quantization_shaders';

/**
 * Quantization parameters for tensor operations
 */
export interface QuantizationParams {
  /** Number of bits per weight (1, 2, 3, 4, 8) */
  bitsPerWeight: 1 | 2 | 3 | 4 | 8;
  
  /** Whether to use symmetric quantization */
  symmetric?: boolean;
  
  /** Whether to use per-channel quantization */
  perChannel?: boolean;
  
  /** Channel size (for per-channel quantization) */
  channelSize?: number;
  
  /** Number of channels (for per-channel quantization) */
  numChannels?: number;
}

/**
 * Configuration options for WebGPU compute operations
 */
export interface ComputeOperationOptions {
  /** Browser optimization target */
  browserOptimization?: boolean;
  
  /** Whether to use fast math approximations */
  useFastMath?: boolean;
  
  /** Quantization parameters */
  quantization?: QuantizationParams;
}

/**
 * WebGPU compute operations for tensor operations with hardware acceleration
 */
export class WebGPUComputeOperations {
  /** WebGPU backend reference */
  private backend: WebGPUBackend;
  
  /** Buffer manager for GPU memory */
  private bufferManager: WebGPUBufferManager;
  
  /** Matrix operations for standard matrix operations */
  private matrixOperations: WebGPUMatrixMultiplication;
  
  /** Browser-optimized matrix operations */
  private optimizedMatrixOps: BrowserOptimizedMatrixOperations | null;
  
  /** Detected browser type */
  private browserType: BrowserType;
  
  /** GPU device reference */
  private device: GPUDevice;
  
  /** Pipeline cache for compute shaders */
  private pipelineCache: Map<string, GPUComputePipeline> = new Map();
  
  /** Bind group layout cache */
  private bindGroupLayoutCache: Map<string, GPUBindGroupLayout> = new Map();
  
  /**
   * Constructor
   * @param backend WebGPU backend
   */
  constructor(backend: WebGPUBackend) {
    this.backend = backend;
    this.device = (backend as any).device;
    this.bufferManager = (backend as any).bufferManager;
    this.browserType = detectBrowserType();
    
    // Create matrix operations
    this.matrixOperations = new WebGPUMatrixMultiplication(backend);
    
    // Create browser-optimized operations if possible
    try {
      this.optimizedMatrixOps = new BrowserOptimizedMatrixOperations(backend, this.matrixOperations);
    } catch (e) {
      console.warn('Failed to initialize browser-optimized operations:', e);
      this.optimizedMatrixOps = null;
    }
    
    if (!this.device || !this.bufferManager) {
      throw new Error('WebGPU backend not properly initialized');
    }
  }
  
  /**
   * Quantize a tensor to reduced bit precision
   * @param tensor Tensor to quantize
   * @param options Quantization options
   * @returns Quantized tensor data and parameters needed for dequantization
   */
  async quantizeTensor<T>(
    tensor: Tensor<T>, 
    options: ComputeOperationOptions = {}
  ): Promise<{
    quantizedData: Uint32Array;
    scales: Float32Array;
    zeroPoints: Float32Array;
  }> {
    // Default to 4-bit quantization if not specified
    const quantParams = options.quantization || { bitsPerWeight: 4 };
    
    // Apply browser-specific optimizations
    const browserOptimized = options.browserOptimization !== false;
    
    // Create shader options
    const shaderOptions: QuantizationShaderOptions = {
      bitsPerWeight: quantParams.bitsPerWeight,
      useSymmetricQuantization: quantParams.symmetric,
      usePerChannelQuantization: quantParams.perChannel !== false,
      browserType: browserOptimized ? this.browserType : BrowserType.UNKNOWN,
      useFastMath: options.useFastMath !== false
    };
    
    // Choose appropriate shader based on bit depth
    let shaderCode: string;
    if (quantParams.bitsPerWeight <= 3) {
      // Use ultra-low precision shader for 1-3 bits
      shaderCode = generateUltraLowPrecisionQuantizeShader(shaderOptions);
    } else {
      // Use standard quantization shader for 4-8 bits
      shaderCode = generateQuantizeShader(shaderOptions);
    }
    
    // Get tensor shape and data
    const shape = tensor.shape;
    const tensorSize = tensor.size;
    
    // Calculate channel info for per-channel quantization
    let channelSize = tensorSize;
    let numChannels = 1;
    
    if (quantParams.perChannel && shape.length > 1) {
      // Default: assume last dimension is channels for per-channel quantization
      numChannels = shape[shape.length - 1];
      channelSize = tensorSize / numChannels;
      
      // Override with explicit values if provided
      if (quantParams.channelSize) channelSize = quantParams.channelSize;
      if (quantParams.numChannels) numChannels = quantParams.numChannels;
    }
    
    // Calculate output size based on bit depth
    const valuesPerU32 = 32 / quantParams.bitsPerWeight;
    const outputSize = Math.ceil(tensorSize / valuesPerU32);
    
    // Create buffers
    const inputBuffer = await this.uploadTensorIfNeeded(tensor);
    
    const outputBuffer = this.device.createBuffer({
      size: outputSize * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'Quantized output buffer'
    });
    
    const scalesBuffer = this.device.createBuffer({
      size: numChannels * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'Quantization scales buffer'
    });
    
    const zeroPointsBuffer = this.device.createBuffer({
      size: numChannels * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'Quantization zero points buffer'
    });
    
    // Create uniform buffer
    const uniformBuffer = this.device.createBuffer({
      size: 6 * Uint32Array.BYTES_PER_ELEMENT, // 6 u32 values
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'Quantization params buffer'
    });
    
    // Write uniform data
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        tensorSize,                             // tensor_size
        channelSize,                            // channel_size
        numChannels,                            // num_channels
        quantParams.bitsPerWeight,              // bits_per_weight
        quantParams.symmetric ? 1 : 0,          // symmetric_quant
        quantParams.perChannel !== false ? 1 : 0 // per_channel_quant
      ])
    );
    
    // Create or get pipeline
    const pipelineKey = `quantize_${quantParams.bitsPerWeight}_${quantParams.symmetric}_${quantParams.perChannel}`;
    let pipeline: GPUComputePipeline;
    
    if (this.pipelineCache.has(pipelineKey)) {
      pipeline = this.pipelineCache.get(pipelineKey)!;
    } else {
      // Create shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderCode,
        label: `Quantization shader ${pipelineKey}`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      
      if (this.bindGroupLayoutCache.has('quantize')) {
        bindGroupLayout = this.bindGroupLayoutCache.get('quantize')!;
      } else {
        bindGroupLayout = this.device.createBindGroupLayout({
          entries: [
            { // Input tensor
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Output quantized tensor
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Scales
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Zero points
              binding: 3,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Params
              binding: 4,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' }
            }
          ],
          label: 'Quantization bind group layout'
        });
        
        this.bindGroupLayoutCache.set('quantize', bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `Quantization pipeline layout ${pipelineKey}`
      });
      
      // Create compute pipeline
      pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `Quantization pipeline ${pipelineKey}`
      });
      
      // Cache pipeline
      this.pipelineCache.set(pipelineKey, pipeline);
    }
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: scalesBuffer } },
        { binding: 3, resource: { buffer: zeroPointsBuffer } },
        { binding: 4, resource: { buffer: uniformBuffer } }
      ],
      label: `Quantization bind group ${pipelineKey}`
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: `Quantization encoder ${pipelineKey}`
    });
    
    // Create compute pass
    const pass = encoder.beginComputePass({
      label: `Quantization pass ${pipelineKey}`
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate workgroup counts based on output size
    // Each thread processes one u32 of output (multiple values based on bit depth)
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(outputSize / workgroupSize);
    
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    
    // Create staging buffers for reading results
    const outputStagingBuffer = this.device.createBuffer({
      size: outputSize * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'Quantized output staging buffer'
    });
    
    const scalesStagingBuffer = this.device.createBuffer({
      size: numChannels * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'Scales staging buffer'
    });
    
    const zeroPointsStagingBuffer = this.device.createBuffer({
      size: numChannels * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'Zero points staging buffer'
    });
    
    // Copy results to staging buffers
    encoder.copyBufferToBuffer(
      outputBuffer, 0,
      outputStagingBuffer, 0,
      outputSize * Uint32Array.BYTES_PER_ELEMENT
    );
    
    encoder.copyBufferToBuffer(
      scalesBuffer, 0,
      scalesStagingBuffer, 0,
      numChannels * Float32Array.BYTES_PER_ELEMENT
    );
    
    encoder.copyBufferToBuffer(
      zeroPointsBuffer, 0,
      zeroPointsStagingBuffer, 0,
      numChannels * Float32Array.BYTES_PER_ELEMENT
    );
    
    // Submit commands
    this.device.queue.submit([encoder.finish()]);
    
    // Read results back to CPU
    await outputStagingBuffer.mapAsync(GPUMapMode.READ);
    await scalesStagingBuffer.mapAsync(GPUMapMode.READ);
    await zeroPointsStagingBuffer.mapAsync(GPUMapMode.READ);
    
    const quantizedData = new Uint32Array(outputStagingBuffer.getMappedRange().slice(0));
    const scales = new Float32Array(scalesStagingBuffer.getMappedRange().slice(0));
    const zeroPoints = new Float32Array(zeroPointsStagingBuffer.getMappedRange().slice(0));
    
    // Cleanup
    outputStagingBuffer.unmap();
    scalesStagingBuffer.unmap();
    zeroPointsStagingBuffer.unmap();
    
    // Destroy buffers
    outputBuffer.destroy();
    scalesBuffer.destroy();
    zeroPointsBuffer.destroy();
    uniformBuffer.destroy();
    outputStagingBuffer.destroy();
    scalesStagingBuffer.destroy();
    zeroPointsStagingBuffer.destroy();
    
    return {
      quantizedData,
      scales,
      zeroPoints
    };
  }
  
  /**
   * Dequantize data from reduced bit precision to full precision
   * @param quantizedData Quantized tensor data
   * @param shape Output tensor shape
   * @param scales Scale factors for dequantization
   * @param zeroPoints Zero points for dequantization
   * @param options Quantization options
   * @returns Dequantized tensor
   */
  async dequantizeTensor<T>(
    quantizedData: Uint32Array,
    shape: number[],
    scales: Float32Array,
    zeroPoints: Float32Array,
    options: ComputeOperationOptions = {}
  ): Promise<Tensor<T>> {
    // Default to 4-bit quantization if not specified
    const quantParams = options.quantization || { bitsPerWeight: 4 };
    
    // Apply browser-specific optimizations
    const browserOptimized = options.browserOptimization !== false;
    
    // Create shader options
    const shaderOptions: QuantizationShaderOptions = {
      bitsPerWeight: quantParams.bitsPerWeight,
      useSymmetricQuantization: quantParams.symmetric,
      usePerChannelQuantization: quantParams.perChannel !== false,
      browserType: browserOptimized ? this.browserType : BrowserType.UNKNOWN,
      useFastMath: options.useFastMath !== false
    };
    
    // Choose appropriate shader based on bit depth
    let shaderCode: string;
    if (quantParams.bitsPerWeight <= 3) {
      // Use ultra-low precision shader for 1-3 bits
      shaderCode = generateUltraLowPrecisionDequantizeShader(shaderOptions);
    } else {
      // Use standard dequantization shader for 4-8 bits
      shaderCode = generateDequantizeShader(shaderOptions);
    }
    
    // Calculate output tensor size
    const tensorSize = shape.reduce((a, b) => a * b, 1);
    
    // Calculate channel info for per-channel quantization
    let channelSize = tensorSize;
    let numChannels = 1;
    
    if (quantParams.perChannel && shape.length > 1) {
      // Default: assume last dimension is channels for per-channel quantization
      numChannels = shape[shape.length - 1];
      channelSize = tensorSize / numChannels;
      
      // Override with explicit values if provided
      if (quantParams.channelSize) channelSize = quantParams.channelSize;
      if (quantParams.numChannels) numChannels = quantParams.numChannels;
    }
    
    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: quantizedData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'Quantized input buffer'
    });
    
    const outputBuffer = this.device.createBuffer({
      size: tensorSize * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'Dequantized output buffer'
    });
    
    const scalesBuffer = this.device.createBuffer({
      size: scales.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'Scales buffer'
    });
    
    const zeroPointsBuffer = this.device.createBuffer({
      size: zeroPoints.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'Zero points buffer'
    });
    
    // Create uniform buffer
    const uniformBuffer = this.device.createBuffer({
      size: 6 * Uint32Array.BYTES_PER_ELEMENT, // 6 u32 values
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'Dequantization params buffer'
    });
    
    // Write data to buffers
    this.device.queue.writeBuffer(inputBuffer, 0, quantizedData);
    this.device.queue.writeBuffer(scalesBuffer, 0, scales);
    this.device.queue.writeBuffer(zeroPointsBuffer, 0, zeroPoints);
    
    // Write uniform data
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        tensorSize,                             // tensor_size
        channelSize,                            // channel_size
        numChannels,                            // num_channels
        quantParams.bitsPerWeight,              // bits_per_weight
        quantParams.symmetric ? 1 : 0,          // symmetric_quant
        quantParams.perChannel !== false ? 1 : 0 // per_channel_quant
      ])
    );
    
    // Create or get pipeline
    const pipelineKey = `dequantize_${quantParams.bitsPerWeight}_${quantParams.symmetric}_${quantParams.perChannel}`;
    let pipeline: GPUComputePipeline;
    
    if (this.pipelineCache.has(pipelineKey)) {
      pipeline = this.pipelineCache.get(pipelineKey)!;
    } else {
      // Create shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderCode,
        label: `Dequantization shader ${pipelineKey}`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      
      if (this.bindGroupLayoutCache.has('dequantize')) {
        bindGroupLayout = this.bindGroupLayoutCache.get('dequantize')!;
      } else {
        bindGroupLayout = this.device.createBindGroupLayout({
          entries: [
            { // Input quantized tensor
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Output tensor
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Scales
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Zero points
              binding: 3,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Params
              binding: 4,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' }
            }
          ],
          label: 'Dequantization bind group layout'
        });
        
        this.bindGroupLayoutCache.set('dequantize', bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `Dequantization pipeline layout ${pipelineKey}`
      });
      
      // Create compute pipeline
      pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `Dequantization pipeline ${pipelineKey}`
      });
      
      // Cache pipeline
      this.pipelineCache.set(pipelineKey, pipeline);
    }
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: scalesBuffer } },
        { binding: 3, resource: { buffer: zeroPointsBuffer } },
        { binding: 4, resource: { buffer: uniformBuffer } }
      ],
      label: `Dequantization bind group ${pipelineKey}`
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: `Dequantization encoder ${pipelineKey}`
    });
    
    // Create compute pass
    const pass = encoder.beginComputePass({
      label: `Dequantization pass ${pipelineKey}`
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate workgroup counts
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(tensorSize / workgroupSize);
    
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    
    // Create output tensor
    const outputTensor = new Tensor<T>(
      shape,
      null,
      {
        dataType: 'float32',
        backend: 'webgpu',
        device: this.backend.id
      }
    );
    
    // Create staging buffer for reading results
    const stagingBuffer = this.device.createBuffer({
      size: tensorSize * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'Dequantized output staging buffer'
    });
    
    // Copy output to staging buffer
    encoder.copyBufferToBuffer(
      outputBuffer, 0,
      stagingBuffer, 0,
      tensorSize * Float32Array.BYTES_PER_ELEMENT
    );
    
    // Submit commands
    this.device.queue.submit([encoder.finish()]);
    
    // Read results back to CPU
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const dequantizedData = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    
    // Copy data to output tensor
    (outputTensor as any).data = Array.from(dequantizedData);
    
    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    scalesBuffer.destroy();
    zeroPointsBuffer.destroy();
    uniformBuffer.destroy();
    stagingBuffer.destroy();
    
    return outputTensor;
  }
  
  /**
   * Perform matrix multiplication with a quantized weight matrix
   * This is especially useful for large models where weights can be stored in 4-bit precision
   * @param activations Input activation tensor (not quantized)
   * @param quantizedWeights Quantized weight data
   * @param weightShape Shape of the weight matrix [M, K]
   * @param scales Scales for dequantization
   * @param zeroPoints Zero points for dequantization
   * @param options Operation options
   * @returns Result of matrix multiplication
   */
  async quantizedMatmul<T>(
    activations: Tensor<T>,
    quantizedWeights: Uint32Array,
    weightShape: [number, number], // [M, K]
    scales: Float32Array,
    zeroPoints: Float32Array,
    options: ComputeOperationOptions = {}
  ): Promise<Tensor<T>> {
    // Default to 4-bit quantization for weights
    const quantParams = options.quantization || { bitsPerWeight: 4 };
    const bitsPerWeight = quantParams.bitsPerWeight;
    
    // Apply browser-specific optimizations
    const browserOptimized = options.browserOptimization !== false;
    
    // Create shader options
    const shaderOptions: QuantizationShaderOptions = {
      bitsPerWeight: bitsPerWeight,
      browserType: browserOptimized ? this.browserType : BrowserType.UNKNOWN,
      useFastMath: options.useFastMath !== false
    };
    
    // Get specialized matmul shader for quantized weights
    const shaderCode = generateQuantizedMatmulShader(shaderOptions);
    
    // Validate inputs
    if (activations.shape.length !== 2) {
      throw new Error(`Quantized matmul requires 2D activation tensor, got shape: ${activations.shape}`);
    }
    
    const [M, K] = weightShape;
    const [K_act, N] = activations.shape;
    
    if (K !== K_act) {
      throw new Error(`Dimension mismatch: weightShape[1]=${K} must match activations.shape[0]=${K_act}`);
    }
    
    // Create buffers
    const activationsBuffer = await this.uploadTensorIfNeeded(activations);
    
    const weightsBuffer = this.device.createBuffer({
      size: quantizedWeights.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'Quantized weights buffer'
    });
    
    const scalesBuffer = this.device.createBuffer({
      size: scales.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'Scales buffer'
    });
    
    const zeroPointsBuffer = this.device.createBuffer({
      size: zeroPoints.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'Zero points buffer'
    });
    
    const outputBuffer = this.device.createBuffer({
      size: M * N * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'Output buffer'
    });
    
    // Create dimensions uniform buffer
    const uniformBuffer = this.device.createBuffer({
      size: 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'Dimensions buffer'
    });
    
    // Write data to buffers
    this.device.queue.writeBuffer(weightsBuffer, 0, quantizedWeights);
    this.device.queue.writeBuffer(scalesBuffer, 0, scales);
    this.device.queue.writeBuffer(zeroPointsBuffer, 0, zeroPoints);
    
    // Write dimensions
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([M, K, N, bitsPerWeight])
    );
    
    // Create or get pipeline
    const pipelineKey = `quantized_matmul_${bitsPerWeight}`;
    let pipeline: GPUComputePipeline;
    
    if (this.pipelineCache.has(pipelineKey)) {
      pipeline = this.pipelineCache.get(pipelineKey)!;
    } else {
      // Create shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderCode,
        label: `Quantized matmul shader ${pipelineKey}`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      
      if (this.bindGroupLayoutCache.has('quantized_matmul')) {
        bindGroupLayout = this.bindGroupLayoutCache.get('quantized_matmul')!;
      } else {
        bindGroupLayout = this.device.createBindGroupLayout({
          entries: [
            { // Quantized weights
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Activations
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Output
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Dimensions
              binding: 3,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' }
            },
            { // Scales
              binding: 4,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Zero points
              binding: 5,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            }
          ],
          label: 'Quantized matmul bind group layout'
        });
        
        this.bindGroupLayoutCache.set('quantized_matmul', bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `Quantized matmul pipeline layout ${pipelineKey}`
      });
      
      // Create compute pipeline
      pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `Quantized matmul pipeline ${pipelineKey}`
      });
      
      // Cache pipeline
      this.pipelineCache.set(pipelineKey, pipeline);
    }
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: weightsBuffer } },
        { binding: 1, resource: { buffer: activationsBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
        { binding: 4, resource: { buffer: scalesBuffer } },
        { binding: 5, resource: { buffer: zeroPointsBuffer } }
      ],
      label: `Quantized matmul bind group ${pipelineKey}`
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: `Quantized matmul encoder ${pipelineKey}`
    });
    
    // Create compute pass
    const pass = encoder.beginComputePass({
      label: `Quantized matmul pass ${pipelineKey}`
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Calculate workgroup counts
    // Default to 8x32 workgroups for matrix multiplication
    const workgroupSizeX = 8;
    const workgroupSizeY = 32;
    
    const workgroupCountX = Math.ceil(M / workgroupSizeX);
    const workgroupCountY = Math.ceil(N / workgroupSizeY);
    
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    pass.end();
    
    // Create output tensor
    const outputTensor = new Tensor<T>(
      [M, N],
      null,
      {
        dataType: 'float32',
        backend: 'webgpu',
        device: this.backend.id
      }
    );
    
    // Create staging buffer for reading results
    const stagingBuffer = this.device.createBuffer({
      size: M * N * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'Output staging buffer'
    });
    
    // Copy output to staging buffer
    encoder.copyBufferToBuffer(
      outputBuffer, 0,
      stagingBuffer, 0,
      M * N * Float32Array.BYTES_PER_ELEMENT
    );
    
    // Submit commands
    this.device.queue.submit([encoder.finish()]);
    
    // Read results back to CPU
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    
    // Copy data to output tensor
    (outputTensor as any).data = Array.from(resultData);
    
    // Cleanup
    weightsBuffer.destroy();
    scalesBuffer.destroy();
    zeroPointsBuffer.destroy();
    outputBuffer.destroy();
    uniformBuffer.destroy();
    stagingBuffer.destroy();
    
    return outputTensor;
  }
  
  /**
   * Optimize KV cache memory usage by quantizing to 4-bit precision
   * @param kvCache KV cache tensor to quantize
   * @param batchSize Number of sequences
   * @param numLayers Number of transformer layers
   * @param numHeads Number of attention heads
   * @param headDim Dimension of each attention head
   * @param options Operation options
   * @returns Quantized KV cache, scales, and zero points
   */
  async quantizeKVCache<T>(
    kvCache: Tensor<T>,
    batchSize: number,
    numLayers: number,
    numHeads: number,
    headDim: number,
    options: ComputeOperationOptions = {}
  ): Promise<{
    quantizedData: Uint32Array;
    scales: Float32Array;
    zeroPoints: Float32Array;
  }> {
    // Default to 4-bit quantization for KV cache
    const quantParams = options.quantization || { bitsPerWeight: 4 };
    const bitsPerWeight = quantParams.bitsPerWeight;
    
    // Apply browser-specific optimizations
    const browserOptimized = options.browserOptimization !== false;
    
    // Create shader options
    const shaderOptions: QuantizationShaderOptions = {
      bitsPerWeight: bitsPerWeight,
      browserType: browserOptimized ? this.browserType : BrowserType.UNKNOWN,
      useFastMath: options.useFastMath !== false
    };
    
    // Get specialized KV cache quantization shader
    const shaderCode = generateKVCacheQuantizeShader(shaderOptions);
    
    // Validate input shape
    const seqLen = kvCache.shape[1]; // Assuming shape is [batchSize, seqLen, numLayers * numHeads, headDim]
    
    // Calculate total elements and output size
    const totalElements = kvCache.size;
    const totalHeads = batchSize * numLayers * numHeads;
    const valuesPerU32 = 32 / bitsPerWeight;
    const outputSize = Math.ceil(totalElements / valuesPerU32);
    
    // Create buffers
    const inputBuffer = await this.uploadTensorIfNeeded(kvCache);
    
    const outputBuffer = this.device.createBuffer({
      size: outputSize * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'Quantized KV cache buffer'
    });
    
    const scalesBuffer = this.device.createBuffer({
      size: totalHeads * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'KV cache scales buffer'
    });
    
    const zeroPointsBuffer = this.device.createBuffer({
      size: totalHeads * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'KV cache zero points buffer'
    });
    
    // Create uniform buffer
    const uniformBuffer = this.device.createBuffer({
      size: 6 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'KV cache params buffer'
    });
    
    // Write uniform data
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        batchSize,
        numLayers,
        numHeads,
        headDim,
        seqLen,
        bitsPerWeight
      ])
    );
    
    // Create or get pipeline
    const pipelineKey = `kv_cache_quantize_${bitsPerWeight}`;
    let pipeline: GPUComputePipeline;
    
    if (this.pipelineCache.has(pipelineKey)) {
      pipeline = this.pipelineCache.get(pipelineKey)!;
    } else {
      // Create shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderCode,
        label: `KV cache quantization shader ${pipelineKey}`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      
      if (this.bindGroupLayoutCache.has('kv_cache_quantize')) {
        bindGroupLayout = this.bindGroupLayoutCache.get('kv_cache_quantize')!;
      } else {
        bindGroupLayout = this.device.createBindGroupLayout({
          entries: [
            { // Input KV cache
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Output quantized KV cache
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Scales
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Zero points
              binding: 3,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Params
              binding: 4,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' }
            }
          ],
          label: 'KV cache quantization bind group layout'
        });
        
        this.bindGroupLayoutCache.set('kv_cache_quantize', bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `KV cache quantization pipeline layout ${pipelineKey}`
      });
      
      // Create compute pipeline
      pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `KV cache quantization pipeline ${pipelineKey}`
      });
      
      // Cache pipeline
      this.pipelineCache.set(pipelineKey, pipeline);
    }
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: scalesBuffer } },
        { binding: 3, resource: { buffer: zeroPointsBuffer } },
        { binding: 4, resource: { buffer: uniformBuffer } }
      ],
      label: `KV cache quantization bind group ${pipelineKey}`
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: `KV cache quantization encoder ${pipelineKey}`
    });
    
    // Create compute pass
    const pass = encoder.beginComputePass({
      label: `KV cache quantization pass ${pipelineKey}`
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Dispatch workgroups - one thread per head
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(totalHeads / workgroupSize);
    
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    
    // Create staging buffers for reading results
    const outputStagingBuffer = this.device.createBuffer({
      size: outputSize * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'Quantized KV cache staging buffer'
    });
    
    const scalesStagingBuffer = this.device.createBuffer({
      size: totalHeads * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'KV cache scales staging buffer'
    });
    
    const zeroPointsStagingBuffer = this.device.createBuffer({
      size: totalHeads * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'KV cache zero points staging buffer'
    });
    
    // Copy results to staging buffers
    encoder.copyBufferToBuffer(
      outputBuffer, 0,
      outputStagingBuffer, 0,
      outputSize * Uint32Array.BYTES_PER_ELEMENT
    );
    
    encoder.copyBufferToBuffer(
      scalesBuffer, 0,
      scalesStagingBuffer, 0,
      totalHeads * Float32Array.BYTES_PER_ELEMENT
    );
    
    encoder.copyBufferToBuffer(
      zeroPointsBuffer, 0,
      zeroPointsStagingBuffer, 0,
      totalHeads * Float32Array.BYTES_PER_ELEMENT
    );
    
    // Submit commands
    this.device.queue.submit([encoder.finish()]);
    
    // Read results back to CPU
    await outputStagingBuffer.mapAsync(GPUMapMode.READ);
    await scalesStagingBuffer.mapAsync(GPUMapMode.READ);
    await zeroPointsStagingBuffer.mapAsync(GPUMapMode.READ);
    
    const quantizedData = new Uint32Array(outputStagingBuffer.getMappedRange().slice(0));
    const scales = new Float32Array(scalesStagingBuffer.getMappedRange().slice(0));
    const zeroPoints = new Float32Array(zeroPointsStagingBuffer.getMappedRange().slice(0));
    
    // Cleanup
    outputStagingBuffer.unmap();
    scalesStagingBuffer.unmap();
    zeroPointsStagingBuffer.unmap();
    
    // Destroy buffers
    outputBuffer.destroy();
    scalesBuffer.destroy();
    zeroPointsBuffer.destroy();
    uniformBuffer.destroy();
    outputStagingBuffer.destroy();
    scalesStagingBuffer.destroy();
    zeroPointsStagingBuffer.destroy();
    
    return {
      quantizedData,
      scales,
      zeroPoints
    };
  }
  
  /**
   * Dequantize KV cache for use in attention computation
   * @param quantizedData Quantized KV cache data
   * @param scales Scale factors for dequantization
   * @param zeroPoints Zero points for dequantization
   * @param batchSize Number of sequences
   * @param numLayers Number of transformer layers
   * @param numHeads Number of attention heads
   * @param headDim Dimension of each attention head
   * @param seqLen Sequence length
   * @param options Operation options
   * @returns Dequantized KV cache tensor
   */
  async dequantizeKVCache<T>(
    quantizedData: Uint32Array,
    scales: Float32Array,
    zeroPoints: Float32Array,
    batchSize: number,
    numLayers: number,
    numHeads: number,
    headDim: number,
    seqLen: number,
    options: ComputeOperationOptions = {}
  ): Promise<Tensor<T>> {
    // Default to 4-bit quantization for KV cache
    const quantParams = options.quantization || { bitsPerWeight: 4 };
    const bitsPerWeight = quantParams.bitsPerWeight;
    
    // Apply browser-specific optimizations
    const browserOptimized = options.browserOptimization !== false;
    
    // Create shader options
    const shaderOptions: QuantizationShaderOptions = {
      bitsPerWeight: bitsPerWeight,
      browserType: browserOptimized ? this.browserType : BrowserType.UNKNOWN,
      useFastMath: options.useFastMath !== false
    };
    
    // Get specialized KV cache dequantization shader
    const shaderCode = generateKVCacheDequantizeShader(shaderOptions);
    
    // Calculate total elements and output shape
    const totalElements = batchSize * numLayers * numHeads * headDim * seqLen;
    const outputShape = [batchSize, seqLen, numLayers * numHeads, headDim];
    
    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: quantizedData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'Quantized KV cache buffer'
    });
    
    const outputBuffer = this.device.createBuffer({
      size: totalElements * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'Dequantized KV cache buffer'
    });
    
    const scalesBuffer = this.device.createBuffer({
      size: scales.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'KV cache scales buffer'
    });
    
    const zeroPointsBuffer = this.device.createBuffer({
      size: zeroPoints.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'KV cache zero points buffer'
    });
    
    // Create uniform buffer
    const uniformBuffer = this.device.createBuffer({
      size: 6 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'KV cache params buffer'
    });
    
    // Write data to buffers
    this.device.queue.writeBuffer(inputBuffer, 0, quantizedData);
    this.device.queue.writeBuffer(scalesBuffer, 0, scales);
    this.device.queue.writeBuffer(zeroPointsBuffer, 0, zeroPoints);
    
    // Write uniform data
    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        batchSize,
        numLayers,
        numHeads,
        headDim,
        seqLen,
        bitsPerWeight
      ])
    );
    
    // Create or get pipeline
    const pipelineKey = `kv_cache_dequantize_${bitsPerWeight}`;
    let pipeline: GPUComputePipeline;
    
    if (this.pipelineCache.has(pipelineKey)) {
      pipeline = this.pipelineCache.get(pipelineKey)!;
    } else {
      // Create shader module
      const shaderModule = this.device.createShaderModule({
        code: shaderCode,
        label: `KV cache dequantization shader ${pipelineKey}`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      
      if (this.bindGroupLayoutCache.has('kv_cache_dequantize')) {
        bindGroupLayout = this.bindGroupLayoutCache.get('kv_cache_dequantize')!;
      } else {
        bindGroupLayout = this.device.createBindGroupLayout({
          entries: [
            { // Quantized KV cache
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Output dequantized KV cache
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            },
            { // Scales
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Zero points
              binding: 3,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            { // Params
              binding: 4,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' }
            }
          ],
          label: 'KV cache dequantization bind group layout'
        });
        
        this.bindGroupLayoutCache.set('kv_cache_dequantize', bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `KV cache dequantization pipeline layout ${pipelineKey}`
      });
      
      // Create compute pipeline
      pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `KV cache dequantization pipeline ${pipelineKey}`
      });
      
      // Cache pipeline
      this.pipelineCache.set(pipelineKey, pipeline);
    }
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: scalesBuffer } },
        { binding: 3, resource: { buffer: zeroPointsBuffer } },
        { binding: 4, resource: { buffer: uniformBuffer } }
      ],
      label: `KV cache dequantization bind group ${pipelineKey}`
    });
    
    // Create command encoder
    const encoder = this.device.createCommandEncoder({
      label: `KV cache dequantization encoder ${pipelineKey}`
    });
    
    // Create compute pass
    const pass = encoder.beginComputePass({
      label: `KV cache dequantization pass ${pipelineKey}`
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // Dispatch workgroups - one thread per element
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(totalElements / workgroupSize);
    
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    
    // Create output tensor
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: 'float32',
        backend: 'webgpu',
        device: this.backend.id
      }
    );
    
    // Create staging buffer for reading results
    const stagingBuffer = this.device.createBuffer({
      size: totalElements * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'Dequantized KV cache staging buffer'
    });
    
    // Copy output to staging buffer
    encoder.copyBufferToBuffer(
      outputBuffer, 0,
      stagingBuffer, 0,
      totalElements * Float32Array.BYTES_PER_ELEMENT
    );
    
    // Submit commands
    this.device.queue.submit([encoder.finish()]);
    
    // Read results back to CPU
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const dequantizedData = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    
    // Copy data to output tensor
    (outputTensor as any).data = Array.from(dequantizedData);
    
    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    scalesBuffer.destroy();
    zeroPointsBuffer.destroy();
    uniformBuffer.destroy();
    stagingBuffer.destroy();
    
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
   * Dispose resources used by this class
   */
  dispose(): void {
    this.pipelineCache.clear();
    this.bindGroupLayoutCache.clear();
    
    if (this.matrixOperations) {
      (this.matrixOperations as any).dispose?.();
    }
  }
}

/**
 * Factory function to create WebGPU compute operations
 * @param backend WebGPU backend
 * @returns WebGPU compute operations instance
 */
export async function createWebGPUComputeOperations(
  backend: WebGPUBackend
): Promise<WebGPUComputeOperations> {
  return new WebGPUComputeOperations(backend);
}