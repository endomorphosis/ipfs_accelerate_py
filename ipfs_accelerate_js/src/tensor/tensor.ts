/**
 * Tensor implementation
 * 
 * This file provides a cross-platform tensor implementation with support
 * for different hardware backends and memory layouts.
 */

import { ITensor, TensorDataType } from '../core/interfaces';

/**
 * Tensor storage types
 */
export type TensorStorage = 
  | 'cpu'      // CPU memory (ArrayBuffer)
  | 'webgpu'   // WebGPU buffer
  | 'webnn'    // WebNN operand
  | 'shared';  // Shared between CPU and GPU

/**
 * Tensor memory layout
 */
export interface TensorLayout {
  /** Number of dimensions */
  dims: number[];
  /** Strides for each dimension */
  strides?: number[];
  /** Byte offset in the buffer */
  offset?: number;
  /** Total size in elements */
  size: number;
  /** Data type */
  dataType: TensorDataType;
}

/**
 * Tensor descriptor for creation
 */
export interface TensorDescriptor {
  /** Tensor dimensions */
  dims: number[];
  /** Tensor data type */
  dataType: TensorDataType;
  /** Tensor storage location */
  storage?: TensorStorage;
  /** Optional name */
  name?: string;
}

/**
 * Cross-platform tensor implementation
 */
export class Tensor implements ITensor {
  private dims: number[];
  private strides: number[];
  private dataType: TensorDataType;
  private size: number;
  private cpuData: ArrayBuffer | null = null;
  private gpuData: GPUBuffer | null = null;
  private webnnData: any | null = null;
  private ownsData: boolean = true;
  private storage: TensorStorage;
  private name: string;
  private refCount: number = 1;

  /**
   * Create a new tensor
   */
  constructor(
    descriptor: TensorDescriptor, 
    data?: ArrayBufferView | GPUBuffer | any
  ) {
    this.dims = [...descriptor.dims];
    this.dataType = descriptor.dataType;
    this.storage = descriptor.storage || 'cpu';
    this.name = descriptor.name || '';
    
    // Calculate size
    this.size = this.dims.reduce((a, b) => a * b, 1);
    
    // Calculate strides
    this.strides = new Array(this.dims.length).fill(1);
    for (let i = this.dims.length - 2; i >= 0; i--) {
      this.strides[i] = this.strides[i + 1] * this.dims[i + 1];
    }
    
    // Initialize data storage
    if (data) {
      if (this.storage === 'cpu' && ArrayBuffer.isView(data)) {
        this.cpuData = data.buffer;
      } else if (this.storage === 'webgpu' && 'usage' in data) {
        this.gpuData = data as GPUBuffer;
        this.ownsData = false;
      } else if (this.storage === 'webnn') {
        this.webnnData = data;
        this.ownsData = false;
      } else {
        throw new Error(`Incompatible data type for ${this.storage} storage`);
      }
    } else {
      // Create new data storage
      this.allocateStorage();
    }
  }

  /**
   * Get tensor dimensions
   */
  getDimensions(): number[] {
    return [...this.dims];
  }

  /**
   * Get tensor data type
   */
  getDataType(): TensorDataType {
    return this.dataType;
  }

  /**
   * Get tensor size (total number of elements)
   */
  getSize(): number {
    return this.size;
  }

  /**
   * Get tensor name
   */
  getName(): string {
    return this.name;
  }

  /**
   * Set tensor name
   */
  setName(name: string): void {
    this.name = name;
  }

  /**
   * Get tensor data
   */
  getData<T extends ArrayBufferView>(): T {
    if (!this.cpuData) {
      this.syncToCPU();
    }
    
    if (!this.cpuData) {
      throw new Error('Failed to get CPU data');
    }
    
    // Create a view of the appropriate type
    let dataView: ArrayBufferView;
    switch (this.dataType) {
      case 'float32':
        dataView = new Float32Array(this.cpuData);
        break;
      case 'float16':
        // Float16Array not available in all browsers, use Uint16Array
        dataView = new Uint16Array(this.cpuData);
        break;
      case 'int32':
        dataView = new Int32Array(this.cpuData);
        break;
      case 'int16':
        dataView = new Int16Array(this.cpuData);
        break;
      case 'int8':
        dataView = new Int8Array(this.cpuData);
        break;
      case 'uint8':
        dataView = new Uint8Array(this.cpuData);
        break;
      default:
        throw new Error(`Unsupported data type: ${this.dataType}`);
    }
    
    return dataView as T;
  }

  /**
   * Get the WebGPU buffer for this tensor
   */
  getGPUBuffer(): GPUBuffer | null {
    if (this.storage !== 'webgpu' && this.storage !== 'shared') {
      throw new Error(`Cannot get GPU buffer for tensor with ${this.storage} storage`);
    }
    
    return this.gpuData;
  }

  /**
   * Get the WebNN operand for this tensor
   */
  getWebNNOperand(): any | null {
    if (this.storage !== 'webnn') {
      throw new Error(`Cannot get WebNN operand for tensor with ${this.storage} storage`);
    }
    
    return this.webnnData;
  }
  
  /**
   * Copy data to the tensor from CPU
   */
  copyFromCPU(data: ArrayBufferView): void {
    if (data.length !== this.size) {
      throw new Error(`Data size mismatch: expected ${this.size}, got ${data.length}`);
    }
    
    if (this.storage === 'cpu' || this.storage === 'shared') {
      // Direct copy to CPU storage
      if (!this.cpuData) {
        this.cpuData = new ArrayBuffer(data.byteLength);
      }
      
      const targetView = new Uint8Array(this.cpuData);
      const sourceView = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      targetView.set(sourceView);
    } else if (this.storage === 'webgpu' && this.gpuData) {
      // Copy to GPU
      // Note: This would typically be done through a GPUQueue
      // In a real implementation, we'd need a device and queue reference
      // For this example, we'll just note that it needs to be implemented
      console.warn('copyFromCPU to WebGPU needs queue reference');
    } else if (this.storage === 'webnn' && this.webnnData) {
      console.warn('copyFromCPU to WebNN not directly supported');
    }
  }

  /**
   * Increment reference count
   */
  addReference(): void {
    this.refCount++;
  }

  /**
   * Decrement reference count
   */
  releaseReference(): boolean {
    this.refCount--;
    return this.refCount <= 0;
  }

  /**
   * Get current reference count
   */
  getReferenceCount(): number {
    return this.refCount;
  }

  /**
   * Synchronize data to CPU if needed
   */
  private syncToCPU(): void {
    if (this.cpuData) {
      return; // Already on CPU
    }
    
    if (this.storage === 'webgpu' && this.gpuData) {
      // Need to implement GPU to CPU sync
      // Would need device and queue reference
      console.warn('syncToCPU from WebGPU needs implementation');
    } else if (this.storage === 'webnn' && this.webnnData) {
      console.warn('syncToCPU from WebNN needs implementation');
    }
  }

  /**
   * Allocate storage for tensor data
   */
  private allocateStorage(): void {
    const byteSize = this.getByteSize();
    
    switch (this.storage) {
      case 'cpu':
      case 'shared':
        this.cpuData = new ArrayBuffer(byteSize);
        break;
      case 'webgpu':
        // Would need device reference
        console.warn('GPU storage allocation needs device reference');
        break;
      case 'webnn':
        // Would need ML context reference
        console.warn('WebNN storage allocation needs context reference');
        break;
    }
  }

  /**
   * Calculate byte size based on data type and element count
   */
  private getByteSize(): number {
    let elementSize: number;
    
    switch (this.dataType) {
      case 'float32':
        elementSize = 4;
        break;
      case 'float16':
        elementSize = 2;
        break;
      case 'int32':
        elementSize = 4;
        break;
      case 'int16':
        elementSize = 2;
        break;
      case 'int8':
      case 'uint8':
        elementSize = 1;
        break;
      default:
        throw new Error(`Unsupported data type: ${this.dataType}`);
    }
    
    return this.size * elementSize;
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    if (this.ownsData) {
      if (this.gpuData) {
        this.gpuData.destroy();
        this.gpuData = null;
      }
      
      // WebNN doesn't have explicit dispose
      this.webnnData = null;
    }
    
    this.cpuData = null;
  }
}

/**
 * Create a tensor from CPU data
 */
export function createTensor(
  descriptor: TensorDescriptor,
  data?: ArrayBufferView
): Tensor {
  return new Tensor(descriptor, data);
}

/**
 * Create a tensor with zeros
 */
export function zeros(
  dims: number[],
  options: {
    dataType?: TensorDataType,
    storage?: TensorStorage,
    name?: string
  } = {}
): Tensor {
  const descriptor: TensorDescriptor = {
    dims,
    dataType: options.dataType || 'float32',
    storage: options.storage || 'cpu',
    name: options.name
  };
  
  const tensor = new Tensor(descriptor);
  
  // Initialize with zeros (CPU only)
  if (descriptor.storage === 'cpu' || descriptor.storage === 'shared') {
    const data = tensor.getData<ArrayBufferView>();
    if (data instanceof Float32Array) {
      data.fill(0);
    } else if (data instanceof Int32Array) {
      data.fill(0);
    } else if (data instanceof Uint8Array) {
      data.fill(0);
    } else if (data instanceof Int8Array) {
      data.fill(0);
    } else if (data instanceof Int16Array) {
      data.fill(0);
    } else if (data instanceof Uint16Array) {
      data.fill(0);
    }
  }
  
  return tensor;
}

/**
 * Create a tensor with ones
 */
export function ones(
  dims: number[],
  options: {
    dataType?: TensorDataType,
    storage?: TensorStorage,
    name?: string
  } = {}
): Tensor {
  const descriptor: TensorDescriptor = {
    dims,
    dataType: options.dataType || 'float32',
    storage: options.storage || 'cpu',
    name: options.name
  };
  
  const tensor = new Tensor(descriptor);
  
  // Initialize with ones (CPU only)
  if (descriptor.storage === 'cpu' || descriptor.storage === 'shared') {
    const data = tensor.getData<ArrayBufferView>();
    if (data instanceof Float32Array) {
      data.fill(1);
    } else if (data instanceof Int32Array) {
      data.fill(1);
    } else if (data instanceof Uint8Array) {
      data.fill(1);
    } else if (data instanceof Int8Array) {
      data.fill(1);
    } else if (data instanceof Int16Array) {
      data.fill(1);
    } else if (data instanceof Uint16Array) {
      data.fill(1);
    }
  }
  
  return tensor;
}