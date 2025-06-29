/**
 * WebGPU Buffer Manager
 * Handles efficient allocation, reuse, and management of GPU buffers
 */

import { Tensor } from '../../tensor/tensor';

/**
 * Buffer allocation options
 */
export interface BufferAllocationOptions {
  /** Label for the buffer (for debugging) */
  label?: string;
  
  /** Usage flags for the buffer */
  usage?: GPUBufferUsageFlags;
  
  /** Whether to map the buffer at creation */
  mapped?: boolean;
}

/**
 * WebGPU Buffer Cache for reusing buffers of similar sizes
 */
export class WebGPUBufferCache {
  /** Map from size to list of available buffers */
  private availableBuffers: Map<number, GPUBuffer[]> = new Map();
  
  /** Map from tensor ID to allocated buffer */
  private allocatedBuffers: Map<string, GPUBuffer> = new Map();
  
  /** Reference to the device */
  private device: GPUDevice;
  
  /** Statistics for buffer usage */
  private stats = {
    totalAllocated: 0,
    currentlyAllocated: 0,
    reused: 0,
    created: 0
  };
  
  /**
   * Constructor
   * @param device WebGPU device
   */
  constructor(device: GPUDevice) {
    this.device = device;
  }
  
  /**
   * Allocate a buffer of the specified size
   * @param byteSize Size in bytes
   * @param options Allocation options
   * @returns Allocated or reused buffer
   */
  allocateBuffer(byteSize: number, options: BufferAllocationOptions = {}): GPUBuffer {
    // Round size up to a multiple of 4 bytes for alignment
    const alignedSize = Math.ceil(byteSize / 4) * 4;
    
    // Try to find an available buffer of the same size
    const availableOfSize = this.availableBuffers.get(alignedSize) || [];
    if (availableOfSize.length > 0) {
      const buffer = availableOfSize.pop()!;
      this.stats.reused++;
      return buffer;
    }
    
    // Create a new buffer if none available
    const usage = options.usage || 
      (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    
    const buffer = this.device.createBuffer({
      size: alignedSize,
      usage,
      label: options.label || `tensor_buffer_${alignedSize}`,
      mappedAtCreation: options.mapped || false
    });
    
    this.stats.created++;
    this.stats.totalAllocated += alignedSize;
    this.stats.currentlyAllocated += alignedSize;
    
    return buffer;
  }
  
  /**
   * Associate a buffer with a tensor for future reference
   * @param tensorId Unique ID for the tensor
   * @param buffer Buffer to associate
   */
  registerTensorBuffer(tensorId: string, buffer: GPUBuffer): void {
    this.allocatedBuffers.set(tensorId, buffer);
  }
  
  /**
   * Get the buffer associated with a tensor
   * @param tensorId Unique ID for the tensor
   * @returns Associated buffer or undefined if not found
   */
  getTensorBuffer(tensorId: string): GPUBuffer | undefined {
    return this.allocatedBuffers.get(tensorId);
  }
  
  /**
   * Release a buffer back to the pool for reuse
   * @param tensorId Unique ID for the tensor
   * @returns True if the buffer was released, false if not found
   */
  releaseBuffer(tensorId: string): boolean {
    const buffer = this.allocatedBuffers.get(tensorId);
    if (!buffer) {
      return false;
    }
    
    // Remove from allocated map
    this.allocatedBuffers.delete(tensorId);
    
    // Add to available map
    const size = buffer.size;
    if (!this.availableBuffers.has(size)) {
      this.availableBuffers.set(size, []);
    }
    this.availableBuffers.get(size)!.push(buffer);
    
    this.stats.currentlyAllocated -= size;
    
    return true;
  }
  
  /**
   * Get buffer allocation statistics
   * @returns Current buffer statistics
   */
  getStats() {
    return { ...this.stats };
  }
  
  /**
   * Garbage collect unused buffers to free GPU memory
   * @param maxBuffersPerSize Maximum number of unused buffers to keep per size
   */
  garbageCollect(maxBuffersPerSize: number = 5): void {
    for (const [size, buffers] of this.availableBuffers.entries()) {
      if (buffers.length > maxBuffersPerSize) {
        // Keep only the specified number of buffers
        const buffersToDelete = buffers.splice(maxBuffersPerSize);
        
        // Destroy excess buffers
        for (const buffer of buffersToDelete) {
          buffer.destroy();
          this.stats.totalAllocated -= size;
        }
      }
    }
  }
  
  /**
   * Release all buffers (for shutdown)
   */
  releaseAll(): void {
    // Destroy all allocated buffers
    for (const buffer of this.allocatedBuffers.values()) {
      buffer.destroy();
    }
    this.allocatedBuffers.clear();
    
    // Destroy all available buffers
    for (const buffers of this.availableBuffers.values()) {
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
    this.availableBuffers.clear();
    
    // Reset stats
    this.stats.currentlyAllocated = 0;
    this.stats.totalAllocated = 0;
  }
}

/**
 * WebGPU Buffer Manager
 * Manages the transfer of tensor data to and from GPU memory
 */
export class WebGPUBufferManager {
  /** WebGPU device */
  private device: GPUDevice;
  
  /** Buffer cache for efficient reuse */
  private bufferCache: WebGPUBufferCache;
  
  /** Default usage flags for buffers */
  private defaultUsage: GPUBufferUsageFlags;
  
  /**
   * Constructor
   * @param device WebGPU device
   */
  constructor(device: GPUDevice) {
    this.device = device;
    this.bufferCache = new WebGPUBufferCache(device);
    this.defaultUsage = GPUBufferUsage.STORAGE | 
                        GPUBufferUsage.COPY_SRC | 
                        GPUBufferUsage.COPY_DST;
  }
  
  /**
   * Upload a tensor to the GPU
   * @param tensor Tensor to upload
   * @param usage Buffer usage flags
   * @returns GPU buffer containing the tensor data
   */
  async uploadTensor<T>(
    tensor: Tensor<T>, 
    usage: GPUBufferUsageFlags = this.defaultUsage
  ): Promise<GPUBuffer> {
    // Generate a unique ID for the tensor
    const tensorId = `tensor_${tensor.dataType}_${tensor.shape.join('x')}_${Date.now()}`;
    
    // Calculate byte length based on data type
    const bytesPerElement = this.getBytesPerElement(tensor.dataType);
    const byteLength = tensor.size * bytesPerElement;
    
    // Allocate a GPU buffer for the tensor
    const buffer = this.bufferCache.allocateBuffer(byteLength, {
      usage,
      label: `tensor_${tensor.shape.join('x')}`
    });
    
    // Register the buffer with the tensor ID
    this.bufferCache.registerTensorBuffer(tensorId, buffer);
    
    // Convert tensor data to appropriate typed array
    const typedArray = this.convertToTypedArray(tensor);
    
    // Write data to the buffer
    this.device.queue.writeBuffer(buffer, 0, typedArray);
    
    return buffer;
  }
  
  /**
   * Download tensor data from the GPU
   * @param buffer GPU buffer containing tensor data
   * @param tensor Tensor to download data into
   */
  async downloadTensor<T>(buffer: GPUBuffer, tensor: Tensor<T>): Promise<void> {
    // Calculate byte length
    const bytesPerElement = this.getBytesPerElement(tensor.dataType);
    const byteLength = tensor.size * bytesPerElement;
    
    // Create a staging buffer for the download
    const stagingBuffer = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'download_staging_buffer'
    });
    
    // Create a command encoder
    const encoder = this.device.createCommandEncoder({
      label: 'download_encoder'
    });
    
    // Copy from the source buffer to the staging buffer
    encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, byteLength);
    
    // Submit the command buffer
    this.device.queue.submit([encoder.finish()]);
    
    // Map the staging buffer to read the data
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const mappedRange = stagingBuffer.getMappedRange();
    
    // Copy the data from the staging buffer to the tensor
    this.copyDataToTensor(mappedRange, tensor);
    
    // Clean up
    stagingBuffer.unmap();
    stagingBuffer.destroy();
  }
  
  /**
   * Release a tensor buffer
   * @param tensor Tensor whose buffer should be released
   */
  releaseTensor<T>(tensor: Tensor<T>): void {
    const tensorId = `tensor_${tensor.dataType}_${tensor.shape.join('x')}_${Date.now()}`;
    this.bufferCache.releaseBuffer(tensorId);
  }
  
  /**
   * Create a new GPU buffer with the tensor shape but without data
   * @param shape Tensor shape
   * @param dataType Data type
   * @param usage Buffer usage flags
   * @returns GPU buffer
   */
  createOutputBuffer(
    shape: number[],
    dataType: string,
    usage: GPUBufferUsageFlags = this.defaultUsage
  ): GPUBuffer {
    // Calculate size from shape
    const size = shape.reduce((a, b) => a * b, 1);
    
    // Calculate byte length based on data type
    const bytesPerElement = this.getBytesPerElement(dataType);
    const byteLength = size * bytesPerElement;
    
    // Allocate a GPU buffer for the output
    return this.bufferCache.allocateBuffer(byteLength, {
      usage,
      label: `output_${shape.join('x')}`
    });
  }
  
  /**
   * Get the number of bytes per element for a data type
   * @param dataType Data type string
   * @returns Bytes per element
   */
  private getBytesPerElement(dataType: string): number {
    switch (dataType) {
      case 'float32':
      case 'int32':
        return 4;
      case 'float64':
      case 'int64':
        return 8;
      case 'uint8':
      case 'bool':
        return 1;
      default:
        return 4; // Default to float32
    }
  }
  
  /**
   * Convert tensor data to an appropriate typed array
   * @param tensor Tensor to convert
   * @returns Typed array containing tensor data
   */
  private convertToTypedArray<T>(tensor: Tensor<T>): ArrayBufferView {
    switch (tensor.dataType) {
      case 'float32':
        return new Float32Array(tensor.data as unknown as ArrayLike<number>);
      case 'int32':
        return new Int32Array(tensor.data as unknown as ArrayLike<number>);
      case 'float64':
        return new Float64Array(tensor.data as unknown as ArrayLike<number>);
      case 'int64':
        // JavaScript doesn't have Int64Array, so we use BigInt64Array
        // This requires conversion from regular numbers to BigInt
        const bigIntArray = new BigInt64Array(tensor.size);
        for (let i = 0; i < tensor.size; i++) {
          bigIntArray[i] = BigInt(Number(tensor.data[i]));
        }
        return bigIntArray;
      case 'uint8':
        return new Uint8Array(tensor.data as unknown as ArrayLike<number>);
      case 'bool':
        // Convert boolean values to 0 and 1
        const boolArray = new Uint8Array(tensor.size);
        for (let i = 0; i < tensor.size; i++) {
          boolArray[i] = tensor.data[i] ? 1 : 0;
        }
        return boolArray;
      default:
        // Default to float32
        return new Float32Array(tensor.data as unknown as ArrayLike<number>);
    }
  }
  
  /**
   * Copy data from a GPU buffer to a tensor
   * @param buffer GPU buffer mapped range
   * @param tensor Tensor to copy data into
   */
  private copyDataToTensor<T>(buffer: ArrayBuffer, tensor: Tensor<T>): void {
    switch (tensor.dataType) {
      case 'float32': {
        const typedArray = new Float32Array(buffer);
        for (let i = 0; i < tensor.size; i++) {
          tensor.data[i] = typedArray[i] as unknown as T;
        }
        break;
      }
      case 'int32': {
        const typedArray = new Int32Array(buffer);
        for (let i = 0; i < tensor.size; i++) {
          tensor.data[i] = typedArray[i] as unknown as T;
        }
        break;
      }
      case 'float64': {
        const typedArray = new Float64Array(buffer);
        for (let i = 0; i < tensor.size; i++) {
          tensor.data[i] = typedArray[i] as unknown as T;
        }
        break;
      }
      case 'int64': {
        const typedArray = new BigInt64Array(buffer);
        for (let i = 0; i < tensor.size; i++) {
          tensor.data[i] = Number(typedArray[i]) as unknown as T;
        }
        break;
      }
      case 'uint8': {
        const typedArray = new Uint8Array(buffer);
        for (let i = 0; i < tensor.size; i++) {
          tensor.data[i] = typedArray[i] as unknown as T;
        }
        break;
      }
      case 'bool': {
        const typedArray = new Uint8Array(buffer);
        for (let i = 0; i < tensor.size; i++) {
          tensor.data[i] = (typedArray[i] !== 0) as unknown as T;
        }
        break;
      }
      default: {
        // Default to float32
        const typedArray = new Float32Array(buffer);
        for (let i = 0; i < tensor.size; i++) {
          tensor.data[i] = typedArray[i] as unknown as T;
        }
      }
    }
  }
  
  /**
   * Get buffer allocation statistics
   * @returns Current buffer statistics
   */
  getStats() {
    return this.bufferCache.getStats();
  }
  
  /**
   * Garbage collect unused buffers
   * @param maxBuffersPerSize Maximum number of unused buffers to keep per size
   */
  garbageCollect(maxBuffersPerSize: number = 5): void {
    this.bufferCache.garbageCollect(maxBuffersPerSize);
  }
  
  /**
   * Release all buffers (for shutdown)
   */
  releaseAll(): void {
    this.bufferCache.releaseAll();
  }
}