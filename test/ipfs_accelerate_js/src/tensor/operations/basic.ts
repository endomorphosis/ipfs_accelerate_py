/**
 * Basic tensor operations
 * 
 * This module provides element-wise operations for tensors, including:
 * - Arithmetic operations (add, subtract, multiply, divide)
 * - Comparison operations (equal, greater, less)
 * - Unary operations (neg, abs, exp, log)
 */

import { Tensor } from '../tensor';

/**
 * Element-wise addition of two tensors (supports broadcasting)
 * 
 * @param a First tensor
 * @param b Second tensor
 * @returns Result tensor
 */
export function add<T extends number>(a: Tensor<T>, b: Tensor<T>): Tensor<T> {
  // Check if shapes are compatible for broadcasting
  const resultShape = getBroadcastShape(a.shape, b.shape);
  
  // Create result tensor
  const result = new Tensor<T>(
    resultShape,
    null,
    {
      dataType: a.dataType,
      backend: a.backend,
      device: a.device,
      requiresGrad: a.requiresGrad || b.requiresGrad
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
  
  // Helper to get broadcasted coordinates
  const getBroadcastCoords = (coords: number[], shape: number[]): number[] => {
    const broadcastedCoords = [];
    
    // Apply broadcasting rules
    for (let i = 0; i < coords.length; i++) {
      if (i >= shape.length) {
        broadcastedCoords.push(0);
      } else if (shape[i] === 1) {
        broadcastedCoords.push(0);
      } else {
        broadcastedCoords.push(coords[i]);
      }
    }
    
    return broadcastedCoords;
  };
  
  // Perform element-wise addition
  for (let i = 0; i < result.size; i++) {
    const coords = flatToCoords(i, resultShape);
    
    const aCoords = getBroadcastCoords(coords, a.shape);
    const bCoords = getBroadcastCoords(coords, b.shape);
    
    const aIndex = coordsToFlat(aCoords, a.shape);
    const bIndex = coordsToFlat(bCoords, b.shape);
    
    result.data[i] = (a.data[aIndex] as number + b.data[bIndex] as number) as T;
  }
  
  return result;
}

/**
 * Element-wise subtraction of two tensors (supports broadcasting)
 * 
 * @param a First tensor
 * @param b Second tensor
 * @returns Result tensor
 */
export function subtract<T extends number>(a: Tensor<T>, b: Tensor<T>): Tensor<T> {
  // Check if shapes are compatible for broadcasting
  const resultShape = getBroadcastShape(a.shape, b.shape);
  
  // Create result tensor
  const result = new Tensor<T>(
    resultShape,
    null,
    {
      dataType: a.dataType,
      backend: a.backend,
      device: a.device,
      requiresGrad: a.requiresGrad || b.requiresGrad
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
  
  // Helper to get broadcasted coordinates
  const getBroadcastCoords = (coords: number[], shape: number[]): number[] => {
    const broadcastedCoords = [];
    
    // Apply broadcasting rules
    for (let i = 0; i < coords.length; i++) {
      if (i >= shape.length) {
        broadcastedCoords.push(0);
      } else if (shape[i] === 1) {
        broadcastedCoords.push(0);
      } else {
        broadcastedCoords.push(coords[i]);
      }
    }
    
    return broadcastedCoords;
  };
  
  // Perform element-wise subtraction
  for (let i = 0; i < result.size; i++) {
    const coords = flatToCoords(i, resultShape);
    
    const aCoords = getBroadcastCoords(coords, a.shape);
    const bCoords = getBroadcastCoords(coords, b.shape);
    
    const aIndex = coordsToFlat(aCoords, a.shape);
    const bIndex = coordsToFlat(bCoords, b.shape);
    
    result.data[i] = (a.data[aIndex] as number - b.data[bIndex] as number) as T;
  }
  
  return result;
}

/**
 * Element-wise multiplication of two tensors (supports broadcasting)
 * 
 * @param a First tensor
 * @param b Second tensor
 * @returns Result tensor
 */
export function multiply<T extends number>(a: Tensor<T>, b: Tensor<T>): Tensor<T> {
  // Check if shapes are compatible for broadcasting
  const resultShape = getBroadcastShape(a.shape, b.shape);
  
  // Create result tensor
  const result = new Tensor<T>(
    resultShape,
    null,
    {
      dataType: a.dataType,
      backend: a.backend,
      device: a.device,
      requiresGrad: a.requiresGrad || b.requiresGrad
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
  
  // Helper to get broadcasted coordinates
  const getBroadcastCoords = (coords: number[], shape: number[]): number[] => {
    const broadcastedCoords = [];
    
    // Apply broadcasting rules
    for (let i = 0; i < coords.length; i++) {
      if (i >= shape.length) {
        broadcastedCoords.push(0);
      } else if (shape[i] === 1) {
        broadcastedCoords.push(0);
      } else {
        broadcastedCoords.push(coords[i]);
      }
    }
    
    return broadcastedCoords;
  };
  
  // Perform element-wise multiplication
  for (let i = 0; i < result.size; i++) {
    const coords = flatToCoords(i, resultShape);
    
    const aCoords = getBroadcastCoords(coords, a.shape);
    const bCoords = getBroadcastCoords(coords, b.shape);
    
    const aIndex = coordsToFlat(aCoords, a.shape);
    const bIndex = coordsToFlat(bCoords, b.shape);
    
    result.data[i] = (a.data[aIndex] as number * b.data[bIndex] as number) as T;
  }
  
  return result;
}

/**
 * Element-wise division of two tensors (supports broadcasting)
 * 
 * @param a First tensor
 * @param b Second tensor
 * @param eps Small constant to avoid division by zero (default: 1e-7)
 * @returns Result tensor
 */
export function divide<T extends number>(a: Tensor<T>, b: Tensor<T>, eps: number = 1e-7): Tensor<T> {
  // Check if shapes are compatible for broadcasting
  const resultShape = getBroadcastShape(a.shape, b.shape);
  
  // Create result tensor
  const result = new Tensor<T>(
    resultShape,
    null,
    {
      dataType: a.dataType,
      backend: a.backend,
      device: a.device,
      requiresGrad: a.requiresGrad || b.requiresGrad
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
  
  // Helper to get broadcasted coordinates
  const getBroadcastCoords = (coords: number[], shape: number[]): number[] => {
    const broadcastedCoords = [];
    
    // Apply broadcasting rules
    for (let i = 0; i < coords.length; i++) {
      if (i >= shape.length) {
        broadcastedCoords.push(0);
      } else if (shape[i] === 1) {
        broadcastedCoords.push(0);
      } else {
        broadcastedCoords.push(coords[i]);
      }
    }
    
    return broadcastedCoords;
  };
  
  // Perform element-wise division
  for (let i = 0; i < result.size; i++) {
    const coords = flatToCoords(i, resultShape);
    
    const aCoords = getBroadcastCoords(coords, a.shape);
    const bCoords = getBroadcastCoords(coords, b.shape);
    
    const aIndex = coordsToFlat(aCoords, a.shape);
    const bIndex = coordsToFlat(bCoords, b.shape);
    
    // Avoid division by zero
    const denominator = (b.data[bIndex] as number) || eps;
    result.data[i] = (a.data[aIndex] as number / denominator) as T;
  }
  
  return result;
}

/**
 * Element-wise power operation (supports broadcasting)
 * 
 * @param a Base tensor
 * @param b Exponent tensor
 * @returns Result tensor
 */
export function pow<T extends number>(a: Tensor<T>, b: Tensor<T>): Tensor<T> {
  // Check if shapes are compatible for broadcasting
  const resultShape = getBroadcastShape(a.shape, b.shape);
  
  // Create result tensor
  const result = new Tensor<T>(
    resultShape,
    null,
    {
      dataType: a.dataType,
      backend: a.backend,
      device: a.device,
      requiresGrad: a.requiresGrad || b.requiresGrad
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
  
  // Helper to get broadcasted coordinates
  const getBroadcastCoords = (coords: number[], shape: number[]): number[] => {
    const broadcastedCoords = [];
    
    // Apply broadcasting rules
    for (let i = 0; i < coords.length; i++) {
      if (i >= shape.length) {
        broadcastedCoords.push(0);
      } else if (shape[i] === 1) {
        broadcastedCoords.push(0);
      } else {
        broadcastedCoords.push(coords[i]);
      }
    }
    
    return broadcastedCoords;
  };
  
  // Perform element-wise power operation
  for (let i = 0; i < result.size; i++) {
    const coords = flatToCoords(i, resultShape);
    
    const aCoords = getBroadcastCoords(coords, a.shape);
    const bCoords = getBroadcastCoords(coords, b.shape);
    
    const aIndex = coordsToFlat(aCoords, a.shape);
    const bIndex = coordsToFlat(bCoords, b.shape);
    
    result.data[i] = (Math.pow(a.data[aIndex] as number, b.data[bIndex] as number)) as T;
  }
  
  return result;
}

/**
 * Negate a tensor element-wise
 * 
 * @param tensor Input tensor
 * @returns Negated tensor
 */
export function neg<T extends number>(tensor: Tensor<T>): Tensor<T> {
  const result = new Tensor<T>(
    tensor.shape,
    null,
    {
      dataType: tensor.dataType,
      backend: tensor.backend,
      device: tensor.device,
      requiresGrad: tensor.requiresGrad
    }
  );
  
  for (let i = 0; i < tensor.size; i++) {
    result.data[i] = (-(tensor.data[i] as number)) as T;
  }
  
  return result;
}

/**
 * Calculate absolute value of a tensor element-wise
 * 
 * @param tensor Input tensor
 * @returns Absolute value tensor
 */
export function abs<T extends number>(tensor: Tensor<T>): Tensor<T> {
  const result = new Tensor<T>(
    tensor.shape,
    null,
    {
      dataType: tensor.dataType,
      backend: tensor.backend,
      device: tensor.device,
      requiresGrad: tensor.requiresGrad
    }
  );
  
  for (let i = 0; i < tensor.size; i++) {
    result.data[i] = (Math.abs(tensor.data[i] as number)) as T;
  }
  
  return result;
}

/**
 * Calculate exponential (e^x) of a tensor element-wise
 * 
 * @param tensor Input tensor
 * @returns Exponential tensor
 */
export function exp<T extends number>(tensor: Tensor<T>): Tensor<T> {
  const result = new Tensor<T>(
    tensor.shape,
    null,
    {
      dataType: tensor.dataType,
      backend: tensor.backend,
      device: tensor.device,
      requiresGrad: tensor.requiresGrad
    }
  );
  
  for (let i = 0; i < tensor.size; i++) {
    // Clamp input values to avoid overflow
    const value = Math.max(-30, Math.min(30, tensor.data[i] as number));
    result.data[i] = (Math.exp(value)) as T;
  }
  
  return result;
}

/**
 * Calculate natural logarithm (ln) of a tensor element-wise
 * 
 * @param tensor Input tensor
 * @param eps Small constant to avoid log(0) (default: 1e-7)
 * @returns Logarithm tensor
 */
export function log<T extends number>(tensor: Tensor<T>, eps: number = 1e-7): Tensor<T> {
  const result = new Tensor<T>(
    tensor.shape,
    null,
    {
      dataType: tensor.dataType,
      backend: tensor.backend,
      device: tensor.device,
      requiresGrad: tensor.requiresGrad
    }
  );
  
  for (let i = 0; i < tensor.size; i++) {
    // Avoid log(0)
    const value = Math.max(eps, tensor.data[i] as number);
    result.data[i] = (Math.log(value)) as T;
  }
  
  return result;
}

/**
 * Calculate square root of a tensor element-wise
 * 
 * @param tensor Input tensor
 * @param eps Small constant to avoid sqrt of negative numbers (default: 0)
 * @returns Square root tensor
 */
export function sqrt<T extends number>(tensor: Tensor<T>, eps: number = 0): Tensor<T> {
  const result = new Tensor<T>(
    tensor.shape,
    null,
    {
      dataType: tensor.dataType,
      backend: tensor.backend,
      device: tensor.device,
      requiresGrad: tensor.requiresGrad
    }
  );
  
  for (let i = 0; i < tensor.size; i++) {
    // Avoid sqrt of negative numbers
    const value = Math.max(eps, tensor.data[i] as number);
    result.data[i] = (Math.sqrt(value)) as T;
  }
  
  return result;
}

/**
 * Calculate element-wise maximum between two tensors (supports broadcasting)
 * 
 * @param a First tensor
 * @param b Second tensor
 * @returns Maximum tensor
 */
export function maximum<T extends number>(a: Tensor<T>, b: Tensor<T>): Tensor<T> {
  // Check if shapes are compatible for broadcasting
  const resultShape = getBroadcastShape(a.shape, b.shape);
  
  // Create result tensor
  const result = new Tensor<T>(
    resultShape,
    null,
    {
      dataType: a.dataType,
      backend: a.backend,
      device: a.device,
      requiresGrad: a.requiresGrad || b.requiresGrad
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
  
  // Helper to get broadcasted coordinates
  const getBroadcastCoords = (coords: number[], shape: number[]): number[] => {
    const broadcastedCoords = [];
    
    // Apply broadcasting rules
    for (let i = 0; i < coords.length; i++) {
      if (i >= shape.length) {
        broadcastedCoords.push(0);
      } else if (shape[i] === 1) {
        broadcastedCoords.push(0);
      } else {
        broadcastedCoords.push(coords[i]);
      }
    }
    
    return broadcastedCoords;
  };
  
  // Perform element-wise maximum
  for (let i = 0; i < result.size; i++) {
    const coords = flatToCoords(i, resultShape);
    
    const aCoords = getBroadcastCoords(coords, a.shape);
    const bCoords = getBroadcastCoords(coords, b.shape);
    
    const aIndex = coordsToFlat(aCoords, a.shape);
    const bIndex = coordsToFlat(bCoords, b.shape);
    
    result.data[i] = (Math.max(a.data[aIndex] as number, b.data[bIndex] as number)) as T;
  }
  
  return result;
}

/**
 * Calculate element-wise minimum between two tensors (supports broadcasting)
 * 
 * @param a First tensor
 * @param b Second tensor
 * @returns Minimum tensor
 */
export function minimum<T extends number>(a: Tensor<T>, b: Tensor<T>): Tensor<T> {
  // Check if shapes are compatible for broadcasting
  const resultShape = getBroadcastShape(a.shape, b.shape);
  
  // Create result tensor
  const result = new Tensor<T>(
    resultShape,
    null,
    {
      dataType: a.dataType,
      backend: a.backend,
      device: a.device,
      requiresGrad: a.requiresGrad || b.requiresGrad
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
  
  // Helper to get broadcasted coordinates
  const getBroadcastCoords = (coords: number[], shape: number[]): number[] => {
    const broadcastedCoords = [];
    
    // Apply broadcasting rules
    for (let i = 0; i < coords.length; i++) {
      if (i >= shape.length) {
        broadcastedCoords.push(0);
      } else if (shape[i] === 1) {
        broadcastedCoords.push(0);
      } else {
        broadcastedCoords.push(coords[i]);
      }
    }
    
    return broadcastedCoords;
  };
  
  // Perform element-wise minimum
  for (let i = 0; i < result.size; i++) {
    const coords = flatToCoords(i, resultShape);
    
    const aCoords = getBroadcastCoords(coords, a.shape);
    const bCoords = getBroadcastCoords(coords, b.shape);
    
    const aIndex = coordsToFlat(aCoords, a.shape);
    const bIndex = coordsToFlat(bCoords, b.shape);
    
    result.data[i] = (Math.min(a.data[aIndex] as number, b.data[bIndex] as number)) as T;
  }
  
  return result;
}

/**
 * Clip tensor values between min and max
 * 
 * @param tensor Input tensor
 * @param min Minimum value
 * @param max Maximum value
 * @returns Clipped tensor
 */
export function clip<T extends number>(tensor: Tensor<T>, min: number, max: number): Tensor<T> {
  const result = new Tensor<T>(
    tensor.shape,
    null,
    {
      dataType: tensor.dataType,
      backend: tensor.backend,
      device: tensor.device,
      requiresGrad: tensor.requiresGrad
    }
  );
  
  for (let i = 0; i < tensor.size; i++) {
    const value = tensor.data[i] as number;
    result.data[i] = (Math.max(min, Math.min(max, value))) as T;
  }
  
  return result;
}

/**
 * Calculate the sum of all elements in the tensor
 * 
 * @param tensor Input tensor
 * @returns Sum of all elements
 */
export function sum<T extends number>(tensor: Tensor<T>): number {
  let result = 0;
  
  for (let i = 0; i < tensor.size; i++) {
    result += tensor.data[i] as number;
  }
  
  return result;
}

/**
 * Calculate the mean of all elements in the tensor
 * 
 * @param tensor Input tensor
 * @returns Mean of all elements
 */
export function mean<T extends number>(tensor: Tensor<T>): number {
  if (tensor.size === 0) {
    return 0;
  }
  
  return sum(tensor) / tensor.size;
}

/**
 * Calculate the sum along the specified dimension
 * 
 * @param tensor Input tensor
 * @param dim Dimension to sum along
 * @param keepDim Whether to keep the dimension as size 1 (default: false)
 * @returns Result tensor
 */
export function sumDim<T extends number>(tensor: Tensor<T>, dim: number, keepDim: boolean = false): Tensor<T> {
  // Convert negative dimension to positive
  if (dim < 0) {
    dim = tensor.shape.length + dim;
  }
  
  // Validate dimension
  if (dim < 0 || dim >= tensor.shape.length) {
    throw new Error(`Sum dimension out of bounds: ${dim} for tensor of rank ${tensor.shape.length}`);
  }
  
  // Calculate new shape
  const newShape = [...tensor.shape];
  newShape.splice(dim, 1);
  
  if (keepDim) {
    newShape.splice(dim, 0, 1);
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
  
  // Fill result with zeros
  for (let i = 0; i < result.size; i++) {
    result.data[i] = 0 as T;
  }
  
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
  
  // Sum along the specified dimension
  for (let i = 0; i < tensor.size; i++) {
    const coords = flatToCoords(i, tensor.shape);
    
    // Create result coordinates
    const resultCoords = [...coords];
    resultCoords.splice(dim, 1);
    
    if (keepDim) {
      resultCoords.splice(dim, 0, 0);
    }
    
    const resultIndex = coordsToFlat(resultCoords, result.shape);
    result.data[resultIndex] = (result.data[resultIndex] as number + tensor.data[i] as number) as T;
  }
  
  return result;
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