/**
 * Tensor broadcasting utilities
 * 
 * This module provides utilities for tensor broadcasting, which allows operations
 * on tensors with different shapes according to NumPy-style broadcasting rules.
 */

/**
 * Convert a flat index to coordinates within a given shape
 * 
 * @param index Flat index
 * @param shape Shape of the tensor
 * @returns Coordinates
 */
export function flatToCoords(index: number, shape: number[]): number[] {
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
}

/**
 * Convert coordinates to a flat index within a given shape
 * 
 * @param coords Coordinates
 * @param shape Shape of the tensor
 * @returns Flat index
 */
export function coordsToFlat(coords: number[], shape: number[]): number {
  let index = 0;
  let multiplier = 1;
  
  for (let i = shape.length - 1; i >= 0; i--) {
    index += coords[i] * multiplier;
    multiplier *= shape[i];
  }
  
  return index;
}

/**
 * Get broadcasted coordinates according to broadcasting rules
 * 
 * @param coords Coordinates in the result shape
 * @param shape Shape of the tensor to broadcast from
 * @returns Broadcasted coordinates
 */
export function getBroadcastCoords(coords: number[], shape: number[]): number[] {
  const broadcastedCoords = [];
  
  // Apply broadcasting rules
  for (let i = 0; i < coords.length; i++) {
    const offset = coords.length - shape.length;
    if (i < offset) {
      // Dimensions not present in the shape (implicit dimensions of size 1)
      broadcastedCoords.push(0);
    } else {
      // Existing dimensions
      const shapeIndex = i - offset;
      if (shape[shapeIndex] === 1) {
        // Dimension of size 1 gets broadcasted
        broadcastedCoords.push(0);
      } else {
        // Regular dimension
        broadcastedCoords.push(coords[i]);
      }
    }
  }
  
  return broadcastedCoords;
}

/**
 * Get the shape that would result from broadcasting two shapes together
 * 
 * @param shapeA First shape
 * @param shapeB Second shape
 * @returns Broadcast shape
 * @throws Error if shapes are not compatible for broadcasting
 */
export function getBroadcastShape(shapeA: number[], shapeB: number[]): number[] {
  const resultShape = [];
  
  // Align shapes to the same length
  const lenA = shapeA.length;
  const lenB = shapeB.length;
  const maxLen = Math.max(lenA, lenB);
  
  // Pad with 1s
  const paddedA = Array(maxLen - lenA).fill(1).concat(shapeA);
  const paddedB = Array(maxLen - lenB).fill(1).concat(shapeB);
  
  // Calculate result shape
  for (let i = 0; i < maxLen; i++) {
    if (paddedA[i] === paddedB[i]) {
      resultShape.push(paddedA[i]);
    } else if (paddedA[i] === 1) {
      resultShape.push(paddedB[i]);
    } else if (paddedB[i] === 1) {
      resultShape.push(paddedA[i]);
    } else {
      throw new Error(
        `Shapes are not compatible for broadcasting: [${shapeA}] and [${shapeB}]`
      );
    }
  }
  
  return resultShape;
}

/**
 * Check if two shapes are compatible for broadcasting
 * 
 * @param shapeA First shape
 * @param shapeB Second shape
 * @returns True if shapes are compatible for broadcasting
 */
export function canBroadcast(shapeA: number[], shapeB: number[]): boolean {
  try {
    getBroadcastShape(shapeA, shapeB);
    return true;
  } catch (error) {
    return false;
  }
}

/**
 * Creates an iterator for broadcasting multiple tensors together
 * 
 * @param shapes Array of shapes to broadcast together
 * @returns Object with result shape and a function to get coordinates
 * @throws Error if shapes are not compatible for broadcasting
 */
export function broadcastShapes(shapes: number[][]): {
  resultShape: number[];
  getIndices: (resultIndex: number) => number[];
} {
  if (shapes.length === 0) {
    throw new Error("No shapes provided for broadcasting");
  }
  
  // Start with the first shape
  let resultShape = [...shapes[0]];
  
  // Broadcast with each subsequent shape
  for (let i = 1; i < shapes.length; i++) {
    resultShape = getBroadcastShape(resultShape, shapes[i]);
  }
  
  // Calculate the total size of the result
  const resultSize = resultShape.reduce((a, b) => a * b, 1);
  
  // Return an object with the result shape and a function to get indices
  return {
    resultShape,
    getIndices: (resultIndex: number): number[] => {
      // Convert the flat index to coordinates in the result shape
      const resultCoords = flatToCoords(resultIndex, resultShape);
      
      // Get the corresponding indices for each input tensor
      return shapes.map(shape => {
        const coords = getBroadcastCoords(resultCoords, shape);
        return coordsToFlat(coords, shape);
      });
    }
  };
}

/**
 * Normalizes negative dimensions to positive
 * 
 * @param dim Dimension index (can be negative)
 * @param rank Rank of the tensor (number of dimensions)
 * @returns Normalized dimension index
 */
export function normalizeDim(dim: number, rank: number): number {
  if (dim < 0) {
    dim = rank + dim;
  }
  
  if (dim < 0 || dim >= rank) {
    throw new Error(`Dimension out of bounds: ${dim} for tensor of rank ${rank}`);
  }
  
  return dim;
}

/**
 * Get the strides for a given shape (number of elements to skip to move by 1 in each dimension)
 * 
 * @param shape Shape of the tensor
 * @returns Strides for each dimension
 */
export function getStrides(shape: number[]): number[] {
  const rank = shape.length;
  const strides = new Array(rank);
  
  let stride = 1;
  for (let i = rank - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  
  return strides;
}

/**
 * Check if a given shape is a valid shape (all dimensions are positive integers)
 * 
 * @param shape Shape to check
 * @returns True if shape is valid
 */
export function isValidShape(shape: number[]): boolean {
  if (!Array.isArray(shape)) {
    return false;
  }
  
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0) {
      return false;
    }
  }
  
  return true;
}

/**
 * Calculates the total number of elements in a tensor with the given shape
 * 
 * @param shape Shape of the tensor
 * @returns Total number of elements
 */
export function shapeSize(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}