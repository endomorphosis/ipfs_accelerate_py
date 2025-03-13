/**
 * Type definitions for tensor operations
 */

/**
 * Supported tensor data types
 */
export type TensorDataType = 
  | 'float32'  // 32-bit floating point
  | 'float16'  // 16-bit floating point
  | 'int32'    // 32-bit integer
  | 'int16'    // 16-bit integer
  | 'int8'     // 8-bit integer
  | 'uint8';   // 8-bit unsigned integer

/**
 * Tensor storage types
 */
export type TensorStorage = 
  | 'cpu'      // CPU memory
  | 'webgpu'   // WebGPU buffer
  | 'webnn'    // WebNN operand
  | 'shared';  // Shared between CPU and GPU

/**
 * Tensor descriptor for creation
 */
export interface TensorDescriptor {
  /** Dimensions of the tensor */
  dims: number[];
  
  /** Data type of the tensor */
  dataType: TensorDataType;
  
  /** Storage type of the tensor */
  storage: TensorStorage;
  
  /** Optional name for the tensor */
  name?: string;
}

/**
 * Tensor memory layout
 */
export interface TensorLayout {
  /** Dimensions of the tensor */
  dims: number[];
  
  /** Strides for each dimension */
  strides: number[];
  
  /** Byte size of each element */
  elementByteSize: number;
  
  /** Total size in bytes */
  totalByteSize: number;
  
  /** Data type of the tensor */
  dataType: TensorDataType;
  
  /** Tensor storage type */
  storage: TensorStorage;
  
  /** Optional name for the tensor */
  name?: string;
}

/**
 * Base tensor interface
 */
export interface ITensor {
  /** Get tensor dimensions */
  getDimensions(): number[];
  
  /** Get tensor data type */
  getDataType(): TensorDataType;
  
  /** Get tensor size (total number of elements) */
  getSize(): number;
  
  /** Get tensor name */
  getName(): string;
  
  /** Set tensor name */
  setName(name: string): void;
  
  /** Get data as appropriate TypedArray */
  getData<T extends ArrayBufferView>(): T;
  
  /** Get WebGPU buffer if available */
  getGPUBuffer(): GPUBuffer | null;
  
  /** Get WebNN operand if available */
  getWebNNOperand(): any | null;
  
  /** Copy data from CPU TypedArray */
  copyFromCPU(data: ArrayBufferView): void;
  
  /** Reference counting: add reference */
  addReference(): void;
  
  /** Reference counting: release reference */
  releaseReference(): boolean;
  
  /** Get current reference count */
  getReferenceCount(): number;
  
  /** Dispose of tensor resources */
  dispose(): void;
}

/**
 * Shape-related utilities for tensors
 */

/**
 * Broadcast two shapes to a compatible shape
 * @param shape1 First shape
 * @param shape2 Second shape
 * @returns Compatible broadcast shape or null if not broadcastable
 */
export function broadcastShapes(shape1: number[], shape2: number[]): number[] | null {
  // Result has rank of max(rank1, rank2)
  const resultRank = Math.max(shape1.length, shape2.length);
  const resultShape = new Array(resultRank);
  
  // Right-align shapes
  for (let i = 0; i < resultRank; i++) {
    const dim1 = i < shape1.length ? shape1[shape1.length - 1 - i] : 1;
    const dim2 = i < shape2.length ? shape2[shape2.length - 1 - i] : 1;
    
    // Dimensions must be equal or one of them must be 1
    if (dim1 === dim2 || dim1 === 1 || dim2 === 1) {
      resultShape[resultRank - 1 - i] = Math.max(dim1, dim2);
    } else {
      // Not broadcastable
      return null;
    }
  }
  
  return resultShape;
}

/**
 * Check if shape is broadcastable to target shape
 * @param shape Source shape
 * @param targetShape Target shape
 * @returns Whether shape is broadcastable to target
 */
export function isBroadcastableTo(shape: number[], targetShape: number[]): boolean {
  // Source rank must be <= target rank
  if (shape.length > targetShape.length) {
    return false;
  }
  
  // Right-align shapes
  for (let i = 0; i < shape.length; i++) {
    const sourceDim = shape[shape.length - 1 - i];
    const targetDim = targetShape[targetShape.length - 1 - i];
    
    // Source dim must be 1 or equal to target dim
    if (sourceDim !== 1 && sourceDim !== targetDim) {
      return false;
    }
  }
  
  return true;
}

/**
 * Calculate strides for a shape
 * @param shape Shape
 * @returns Strides array
 */
export function calculateStrides(shape: number[]): number[] {
  const rank = shape.length;
  const strides = new Array(rank);
  
  // Last dimension has stride 1
  strides[rank - 1] = 1;
  
  // Calculate rest of strides
  for (let i = rank - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  
  return strides;
}

/**
 * Calculate flat index from multi-dimensional indices
 * @param indices Multi-dimensional indices
 * @param strides Strides for each dimension
 * @returns Flat index
 */
export function indicesToIndex(indices: number[], strides: number[]): number {
  let index = 0;
  for (let i = 0; i < indices.length; i++) {
    index += indices[i] * strides[i];
  }
  return index;
}

/**
 * Convert flat index to multi-dimensional indices
 * @param index Flat index
 * @param shape Shape
 * @param strides Strides
 * @returns Multi-dimensional indices
 */
export function indexToIndices(index: number, shape: number[], strides: number[]): number[] {
  const indices = new Array(shape.length);
  for (let i = 0; i < shape.length; i++) {
    indices[i] = Math.floor(index / strides[i]);
    index -= indices[i] * strides[i];
  }
  return indices;
}