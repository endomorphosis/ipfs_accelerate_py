/**
 * Matrix operations for tensors
 * 
 * This module provides matrix operations for tensors, including:
 * - Matrix multiplication
 * - Transpose
 * - Reshape
 */

import { Tensor } from '../tensor';

/**
 * Performs matrix multiplication between two tensors
 * 
 * @param a First tensor [M, K]
 * @param b Second tensor [K, N]
 * @returns Result tensor [M, N]
 */
export function matmul<T extends number>(a: Tensor<T>, b: Tensor<T>): Tensor<T> {
  // Validate tensor dimensions
  if (a.rank !== 2 || b.rank !== 2) {
    throw new Error(`Matrix multiplication requires 2D tensors, got ${a.rank}D and ${b.rank}D`);
  }
  
  if (a.shape[1] !== b.shape[0]) {
    throw new Error(
      `Matrix dimensions do not match for multiplication: [${a.shape}] and [${b.shape}]`
    );
  }
  
  const M = a.shape[0];
  const K = a.shape[1];
  const N = b.shape[1];
  
  // Create result tensor
  const result = new Tensor<T>(
    [M, N],
    null,
    {
      dataType: a.dataType,
      backend: a.backend,
      device: a.device,
      requiresGrad: a.requiresGrad || b.requiresGrad
    }
  );
  
  // Perform matrix multiplication - CPU implementation
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += (a.get(i, k) as number) * (b.get(k, j) as number);
      }
      result.set(sum as T, i, j);
    }
  }
  
  return result;
}

/**
 * Transposes a tensor along specified dimensions
 * 
 * @param tensor Input tensor
 * @param dims Dimensions to transpose (default: reverse all dimensions)
 * @returns Transposed tensor
 */
export function transpose<T>(tensor: Tensor<T>, dims?: number[]): Tensor<T> {
  const rank = tensor.rank;
  
  // Default to reversing dimensions
  if (!dims) {
    dims = Array.from({ length: rank }, (_, i) => rank - 1 - i);
  }
  
  // Validate dimensions
  if (dims.length !== rank) {
    throw new Error(`Transpose dimensions must match tensor rank: ${dims.length} vs ${rank}`);
  }
  
  // Check for duplicate dimensions
  const dimSet = new Set(dims);
  if (dimSet.size !== rank) {
    throw new Error(`Transpose dimensions must not contain duplicates: ${dims}`);
  }
  
  // Check dimension bounds
  for (const dim of dims) {
    if (dim < 0 || dim >= rank) {
      throw new Error(`Transpose dimension out of bounds: ${dim} for rank ${rank}`);
    }
  }
  
  // Create new shape
  const newShape = dims.map(d => tensor.shape[d]);
  
  // Create result tensor
  const result = new Tensor<T>(
    newShape,
    null,
    {
      dataType: tensor.dataType,
      backend: tensor.backend,
      device: tensor.device,
      requiresGrad: tensor.requiresGrad
    }
  );
  
  // Helper to convert flat index to coordinates
  const flatToCoords = (index: number, shape: number[]): number[] => {
    const coords = [];
    let remaining = index;
    let divisor = 1;
    
    for (let i = shape.length - 1; i >= 0; i--) {
      const size = shape[i];
      const coord = Math.floor(remaining / divisor) % size;
      coords.unshift(coord);
      divisor *= size;
    }
    
    return coords;
  };
  
  // Helper to convert coordinates to flat index
  const coordsToFlat = (coords: number[], shape: number[]): number => {
    let index = 0;
    let multiplier = 1;
    
    for (let i = shape.length - 1; i >= 0; i--) {
      index += coords[i] * multiplier;
      multiplier *= shape[i];
    }
    
    return index;
  };
  
  // Perform the transpose - CPU implementation
  for (let i = 0; i < tensor.size; i++) {
    const coords = flatToCoords(i, tensor.shape);
    const transposedCoords = dims.map(d => coords[d]);
    const transposedIndex = coordsToFlat(transposedCoords, newShape);
    
    result.data[transposedIndex] = tensor.data[i];
  }
  
  return result;
}

/**
 * Reshapes a tensor to a new shape with the same number of elements
 * 
 * @param tensor Input tensor
 * @param newShape New shape for the tensor
 * @returns Reshaped tensor
 */
export function reshape<T>(tensor: Tensor<T>, newShape: number[]): Tensor<T> {
  // Calculate total elements in new shape
  const newSize = newShape.reduce((a, b) => a * b, 1);
  
  // Validate size
  if (newSize !== tensor.size) {
    throw new Error(
      `Cannot reshape tensor of size ${tensor.size} to shape [${newShape}] with ${newSize} elements`
    );
  }
  
  // Create result tensor (share the same data array)
  const result = new Tensor<T>(
    newShape,
    tensor.data,
    {
      dataType: tensor.dataType,
      backend: tensor.backend,
      device: tensor.device,
      requiresGrad: tensor.requiresGrad
    }
  );
  
  return result;
}

/**
 * Slices a tensor along specified dimensions
 * 
 * @param tensor Input tensor
 * @param starts Start indices for each dimension
 * @param ends End indices for each dimension (exclusive)
 * @returns Sliced tensor
 */
export function slice<T>(tensor: Tensor<T>, starts: number[], ends: number[]): Tensor<T> {
  const rank = tensor.rank;
  
  // Validate input
  if (starts.length !== rank || ends.length !== rank) {
    throw new Error(
      `Slice dimensions must match tensor rank: ${starts.length}/${ends.length} vs ${rank}`
    );
  }
  
  // Calculate new shape
  const newShape = [];
  for (let i = 0; i < rank; i++) {
    // Validate bounds
    if (starts[i] < 0 || starts[i] >= tensor.shape[i]) {
      throw new Error(`Slice start index out of bounds: ${starts[i]} for dimension ${i}`);
    }
    if (ends[i] < starts[i] || ends[i] > tensor.shape[i]) {
      throw new Error(`Slice end index out of bounds: ${ends[i]} for dimension ${i}`);
    }
    
    newShape.push(ends[i] - starts[i]);
  }
  
  // Create result tensor
  const result = new Tensor<T>(
    newShape,
    null,
    {
      dataType: tensor.dataType,
      backend: tensor.backend,
      device: tensor.device,
      requiresGrad: tensor.requiresGrad
    }
  );
  
  // Helper to convert coordinates to flat index
  const coordsToFlat = (coords: number[], shape: number[]): number => {
    let index = 0;
    let multiplier = 1;
    
    for (let i = shape.length - 1; i >= 0; i--) {
      index += coords[i] * multiplier;
      multiplier *= shape[i];
    }
    
    return index;
  };
  
  // Helper to convert flat index to coordinates
  const flatToCoords = (index: number, shape: number[]): number[] => {
    const coords = [];
    let remaining = index;
    let divisor = 1;
    
    for (let i = shape.length - 1; i >= 0; i--) {
      const size = shape[i];
      const coord = Math.floor(remaining / divisor) % size;
      coords.unshift(coord);
      divisor *= size;
    }
    
    return coords;
  };
  
  // Perform the slice - CPU implementation
  for (let i = 0; i < result.size; i++) {
    const resultCoords = flatToCoords(i, newShape);
    const tensorCoords = resultCoords.map((c, j) => c + starts[j]);
    const tensorIndex = coordsToFlat(tensorCoords, tensor.shape);
    
    result.data[i] = tensor.data[tensorIndex];
  }
  
  return result;
}

/**
 * Concatenates tensors along a specified dimension
 * 
 * @param tensors Array of tensors to concatenate
 * @param dim Dimension along which to concatenate (default: 0)
 * @returns Concatenated tensor
 */
export function concat<T>(tensors: Tensor<T>[], dim: number = 0): Tensor<T> {
  if (tensors.length === 0) {
    throw new Error('Cannot concatenate empty tensor array');
  }
  
  const rank = tensors[0].rank;
  
  // Validate dimension
  if (dim < 0 || dim >= rank) {
    throw new Error(`Concat dimension out of bounds: ${dim} for rank ${rank}`);
  }
  
  // Validate tensor shapes
  for (let i = 1; i < tensors.length; i++) {
    if (tensors[i].rank !== rank) {
      throw new Error(`All tensors must have the same rank to concatenate`);
    }
    
    for (let j = 0; j < rank; j++) {
      if (j !== dim && tensors[i].shape[j] !== tensors[0].shape[j]) {
        throw new Error(
          `Tensor shape mismatch for concatenation at dimension ${j}: ` +
          `${tensors[i].shape} vs ${tensors[0].shape}`
        );
      }
    }
  }
  
  // Calculate new shape
  const newShape = [...tensors[0].shape];
  newShape[dim] = tensors.reduce((sum, t) => sum + t.shape[dim], 0);
  
  // Create result tensor
  const result = new Tensor<T>(
    newShape,
    null,
    {
      dataType: tensors[0].dataType,
      backend: tensors[0].backend,
      device: tensors[0].device,
      requiresGrad: tensors.some(t => t.requiresGrad)
    }
  );
  
  // Helper to convert coordinates to flat index
  const coordsToFlat = (coords: number[], shape: number[]): number => {
    let index = 0;
    let multiplier = 1;
    
    for (let i = shape.length - 1; i >= 0; i--) {
      index += coords[i] * multiplier;
      multiplier *= shape[i];
    }
    
    return index;
  };
  
  // Helper to convert flat index to coordinates
  const flatToCoords = (index: number, shape: number[]): number[] => {
    const coords = [];
    let remaining = index;
    let divisor = 1;
    
    for (let i = shape.length - 1; i >= 0; i--) {
      const size = shape[i];
      const coord = Math.floor(remaining / divisor) % size;
      coords.unshift(coord);
      divisor *= size;
    }
    
    return coords;
  };
  
  // Perform concatenation - CPU implementation
  for (let i = 0; i < result.size; i++) {
    const resultCoords = flatToCoords(i, newShape);
    
    // Find which input tensor this coordinate belongs to
    let offset = resultCoords[dim];
    let tensorIndex = 0;
    
    for (let t = 0; t < tensors.length; t++) {
      if (offset < tensors[t].shape[dim]) {
        tensorIndex = t;
        break;
      }
      offset -= tensors[t].shape[dim];
    }
    
    // Translate to source tensor coordinates
    const sourceCoords = [...resultCoords];
    sourceCoords[dim] = offset;
    
    // Copy value
    const sourceFlatIndex = coordsToFlat(sourceCoords, tensors[tensorIndex].shape);
    result.data[i] = tensors[tensorIndex].data[sourceFlatIndex];
  }
  
  return result;
}

/**
 * Calculates the dot product of two vectors (1D tensors)
 * 
 * @param a First vector
 * @param b Second vector
 * @returns Scalar dot product
 */
export function dot<T extends number>(a: Tensor<T>, b: Tensor<T>): number {
  // Validate input
  if (a.rank !== 1 || b.rank !== 1) {
    throw new Error(`Dot product requires 1D tensors, got ${a.rank}D and ${b.rank}D`);
  }
  
  if (a.shape[0] !== b.shape[0]) {
    throw new Error(
      `Vector dimensions do not match for dot product: ${a.shape[0]} and ${b.shape[0]}`
    );
  }
  
  // Calculate dot product
  let sum = 0;
  for (let i = 0; i < a.shape[0]; i++) {
    sum += (a.get(i) as number) * (b.get(i) as number);
  }
  
  return sum;
}

/**
 * Creates an identity matrix of specified size
 * 
 * @param size Size of the identity matrix
 * @param options Tensor options
 * @returns Identity matrix [size, size]
 */
export function eye(size: number, options: Record<string, any> = {}): Tensor<number> {
  const result = new Tensor<number>(
    [size, size],
    null,
    options
  );
  
  for (let i = 0; i < size; i++) {
    result.set(1, i, i);
  }
  
  return result;
}

/**
 * Creates a diagonal matrix from a vector
 * 
 * @param vector Vector to use for diagonal
 * @returns Diagonal matrix
 */
export function diag<T extends number>(vector: Tensor<T>): Tensor<T> {
  // Validate input
  if (vector.rank !== 1) {
    throw new Error(`Diag requires a 1D tensor, got ${vector.rank}D`);
  }
  
  const size = vector.shape[0];
  
  // Create result tensor
  const result = new Tensor<T>(
    [size, size],
    null,
    {
      dataType: vector.dataType,
      backend: vector.backend,
      device: vector.device,
      requiresGrad: vector.requiresGrad
    }
  );
  
  // Set diagonal elements
  for (let i = 0; i < size; i++) {
    result.set(vector.get(i), i, i);
  }
  
  return result;
}