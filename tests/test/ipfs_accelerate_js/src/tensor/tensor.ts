/**
 * Core Tensor implementation
 * Provides a unified interface for tensor operations across different backends
 */

/**
 * Options for tensor creation
 */
export interface TensorOptions {
  dataType?: 'float32' | 'int32' | 'float64' | 'int64' | 'uint8' | 'bool';
  backend?: 'cpu' | 'webgpu' | 'webnn' | 'wasm';
  device?: string;
  requiresGrad?: boolean;
}

/**
 * Tensor class representing n-dimensional arrays with type information
 * Supports operations with different hardware backends (CPU, WebGPU, WebNN)
 */
export class Tensor<T = number> {
  /** Shape of the tensor (dimensions) */
  readonly shape: number[];
  
  /** Data array holding tensor elements */
  readonly data: T[];
  
  /** Data type of tensor elements */
  readonly dataType: string;
  
  /** Backend used for tensor operations */
  readonly backend: string;
  
  /** Whether tensor requires gradient computation */
  readonly requiresGrad: boolean;
  
  /** Device the tensor is stored on */
  readonly device: string;
  
  /**
   * Creates a new Tensor
   * @param shape Shape of the tensor
   * @param data Data for the tensor (optional, initialized to zeros if not provided)
   * @param options Additional options
   */
  constructor(
    shape: number[],
    data: T[] | null = null,
    options: TensorOptions = {}
  ) {
    this.shape = [...shape];
    this.dataType = options.dataType || 'float32';
    this.backend = options.backend || 'cpu';
    this.device = options.device || 'default';
    this.requiresGrad = options.requiresGrad || false;
    
    // Calculate total size from shape
    const size = shape.reduce((a, b) => a * b, 1);
    
    // Initialize data array
    if (data) {
      // Validate data length
      if (data.length !== size) {
        throw new Error(`Data length ${data.length} does not match shape ${shape} (expected ${size} elements)`);
      }
      this.data = [...data];
    } else {
      // Create array filled with zeros
      this.data = Array(size).fill(0) as T[];
    }
  }
  
  /**
   * Get total number of elements in the tensor
   */
  get size(): number {
    return this.data.length;
  }
  
  /**
   * Get number of dimensions of the tensor
   */
  get rank(): number {
    return this.shape.length;
  }
  
  /**
   * Creates a copy of the tensor
   * @returns A new tensor with the same data
   */
  clone(): Tensor<T> {
    return new Tensor<T>(
      this.shape,
      this.data,
      {
        dataType: this.dataType,
        backend: this.backend,
        device: this.device,
        requiresGrad: this.requiresGrad
      }
    );
  }
  
  /**
   * Creates a tensor of the same shape filled with zeros
   * @returns A new zero-filled tensor with the same shape
   */
  zeros(): Tensor<T> {
    return new Tensor<T>(
      this.shape,
      null,
      {
        dataType: this.dataType,
        backend: this.backend,
        device: this.device,
        requiresGrad: this.requiresGrad
      }
    );
  }
  
  /**
   * Creates a tensor of the same shape filled with ones
   * @returns A new one-filled tensor with the same shape
   */
  ones(): Tensor<T> {
    const data = Array(this.size).fill(1) as T[];
    return new Tensor<T>(
      this.shape,
      data,
      {
        dataType: this.dataType,
        backend: this.backend,
        device: this.device,
        requiresGrad: this.requiresGrad
      }
    );
  }
  
  /**
   * Converts tensor to a string representation
   */
  toString(): string {
    const shapeStr = `[${this.shape.join(', ')}]`;
    let dataStr = '';
    
    if (this.size <= 100) {
      dataStr = `[${this.data.join(', ')}]`;
    } else {
      dataStr = `[${this.data.slice(0, 3).join(', ')}, ..., ${this.data.slice(-3).join(', ')}]`;
    }
    
    return `Tensor(${shapeStr}, ${this.dataType}, ${dataStr})`;
  }
  
  /**
   * Gets a value at the specified indices
   * @param indices Indices to get value from
   * @returns Tensor value at the specified indices
   */
  get(...indices: number[]): T {
    if (indices.length !== this.shape.length) {
      throw new Error(`Expected ${this.shape.length} indices but got ${indices.length}`);
    }
    
    // Validate indices
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= this.shape[i]) {
        throw new Error(`Index ${indices[i]} is out of bounds for dimension ${i} with size ${this.shape[i]}`);
      }
    }
    
    // Calculate flat index using row-major layout
    let flatIndex = 0;
    let stride = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      flatIndex += indices[i] * stride;
      stride *= this.shape[i];
    }
    
    return this.data[flatIndex];
  }
  
  /**
   * Sets a value at the specified indices
   * @param value Value to set
   * @param indices Indices to set value at
   */
  set(value: T, ...indices: number[]): void {
    if (indices.length !== this.shape.length) {
      throw new Error(`Expected ${this.shape.length} indices but got ${indices.length}`);
    }
    
    // Validate indices
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= this.shape[i]) {
        throw new Error(`Index ${indices[i]} is out of bounds for dimension ${i} with size ${this.shape[i]}`);
      }
    }
    
    // Calculate flat index using row-major layout
    let flatIndex = 0;
    let stride = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      flatIndex += indices[i] * stride;
      stride *= this.shape[i];
    }
    
    this.data[flatIndex] = value;
  }
}

/**
 * Creates a tensor filled with zeros
 * @param shape Shape of the tensor
 * @param options Additional options
 * @returns A new zero-filled tensor
 */
export function zeros<T>(
  shape: number[],
  options: TensorOptions = {}
): Tensor<T> {
  return new Tensor<T>(shape, null, options);
}

/**
 * Creates a tensor filled with ones
 * @param shape Shape of the tensor
 * @param options Additional options
 * @returns A new one-filled tensor
 */
export function ones<T>(
  shape: number[],
  options: TensorOptions = {}
): Tensor<T> {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = Array(size).fill(1) as T[];
  return new Tensor<T>(shape, data, options);
}

/**
 * Creates a tensor with a range of values
 * @param start Start value (inclusive)
 * @param end End value (exclusive)
 * @param step Step size (default: 1)
 * @param options Additional options
 * @returns A new tensor with the range of values
 */
export function range(
  start: number,
  end: number,
  step: number = 1,
  options: TensorOptions = {}
): Tensor<number> {
  const size = Math.ceil((end - start) / step);
  const data = Array(size).fill(0).map((_, i) => start + i * step);
  return new Tensor<number>([size], data, options);
}

/**
 * Creates a tensor with random values
 * @param shape Shape of the tensor
 * @param min Minimum value (default: 0)
 * @param max Maximum value (default: 1)
 * @param options Additional options
 * @returns A new tensor with random values
 */
export function random(
  shape: number[],
  min: number = 0,
  max: number = 1,
  options: TensorOptions = {}
): Tensor<number> {
  const size = shape.reduce((a, b) => a * b, 1);
  const range = max - min;
  const data = Array(size).fill(0).map(() => min + Math.random() * range);
  return new Tensor<number>(shape, data, options);
}