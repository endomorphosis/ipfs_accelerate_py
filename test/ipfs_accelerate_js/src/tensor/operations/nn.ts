/**
 * Neural network operations for tensors
 * 
 * This module provides neural network-specific operations, including:
 * - Activation functions (ReLU, Sigmoid, Tanh, LeakyReLU)
 * - Normalization functions (Softmax, LayerNorm)
 * - Loss functions (MSE, CrossEntropy)
 */

import { Tensor } from '../tensor';

/**
 * ReLU activation function: f(x) = max(0, x)
 * 
 * @param tensor Input tensor
 * @returns Output tensor with ReLU applied
 */
export function relu<T extends number>(tensor: Tensor<T>): Tensor<T> {
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
    result.data[i] = Math.max(0, tensor.data[i] as number) as T;
  }
  
  return result;
}

/**
 * Leaky ReLU activation function: f(x) = x if x > 0, alpha * x otherwise
 * 
 * @param tensor Input tensor
 * @param alpha Slope for negative inputs (default: 0.01)
 * @returns Output tensor with LeakyReLU applied
 */
export function leakyRelu<T extends number>(tensor: Tensor<T>, alpha: number = 0.01): Tensor<T> {
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
    result.data[i] = (value > 0 ? value : alpha * value) as T;
  }
  
  return result;
}

/**
 * Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
 * 
 * @param tensor Input tensor
 * @returns Output tensor with Sigmoid applied
 */
export function sigmoid<T extends number>(tensor: Tensor<T>): Tensor<T> {
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
    // Clamp input values to avoid overflow
    const clampedValue = Math.max(-30, Math.min(30, value));
    result.data[i] = (1 / (1 + Math.exp(-clampedValue))) as T;
  }
  
  return result;
}

/**
 * Tanh activation function: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * 
 * @param tensor Input tensor
 * @returns Output tensor with Tanh applied
 */
export function tanh<T extends number>(tensor: Tensor<T>): Tensor<T> {
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
    // Use Math.tanh if available, otherwise compute manually
    result.data[i] = Math.tanh ? Math.tanh(value) as T : ((Math.exp(2 * value) - 1) / (Math.exp(2 * value) + 1)) as T;
  }
  
  return result;
}

/**
 * Applies the GELU activation function: f(x) = x * Φ(x)
 * where Φ(x) is the cumulative distribution function of the standard normal distribution
 * 
 * Approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 * 
 * @param tensor Input tensor
 * @returns Output tensor with GELU applied
 */
export function gelu<T extends number>(tensor: Tensor<T>): Tensor<T> {
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
  
  const sqrtTwoPi = Math.sqrt(2 / Math.PI);
  
  for (let i = 0; i < tensor.size; i++) {
    const x = tensor.data[i] as number;
    const xCubed = x * x * x;
    const inner = sqrtTwoPi * (x + 0.044715 * xCubed);
    result.data[i] = (0.5 * x * (1 + Math.tanh(inner))) as T;
  }
  
  return result;
}

/**
 * Softmax activation function along the specified dimension
 * 
 * @param tensor Input tensor
 * @param dim Dimension to apply softmax along (default: -1, which means the last dimension)
 * @returns Output tensor with Softmax applied
 */
export function softmax<T extends number>(tensor: Tensor<T>, dim: number = -1): Tensor<T> {
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
  
  // Convert negative dimension to positive
  if (dim < 0) {
    dim = tensor.shape.length + dim;
  }
  
  // Validate dimension
  if (dim < 0 || dim >= tensor.shape.length) {
    throw new Error(`Softmax dimension out of bounds: ${dim} for tensor of rank ${tensor.shape.length}`);
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
  
  // Calculate stride for the given dimension
  const stride = tensor.shape[dim];
  
  // Iterate through each slice along the softmax dimension
  const batchSize = tensor.size / stride;
  
  for (let batch = 0; batch < batchSize; batch++) {
    // Get coordinates for the start of this batch
    const baseCoords = flatToCoords(batch * stride, tensor.shape.slice(0, dim).concat(tensor.shape.slice(dim + 1)));
    
    // Find max value for numerical stability
    let maxValue = -Infinity;
    for (let i = 0; i < stride; i++) {
      const coords = [...baseCoords];
      coords.splice(dim, 0, i);
      const index = coordsToFlat(coords, tensor.shape);
      maxValue = Math.max(maxValue, tensor.data[index] as number);
    }
    
    // Calculate the exponentials and sum
    let sum = 0;
    const expValues = new Array(stride);
    
    for (let i = 0; i < stride; i++) {
      const coords = [...baseCoords];
      coords.splice(dim, 0, i);
      const index = coordsToFlat(coords, tensor.shape);
      
      const value = Math.exp((tensor.data[index] as number) - maxValue);
      expValues[i] = value;
      sum += value;
    }
    
    // Normalize by the sum
    for (let i = 0; i < stride; i++) {
      const coords = [...baseCoords];
      coords.splice(dim, 0, i);
      const index = coordsToFlat(coords, tensor.shape);
      
      result.data[index] = (expValues[i] / sum) as T;
    }
  }
  
  return result;
}

/**
 * Layer normalization
 * 
 * @param tensor Input tensor
 * @param eps Small constant for numerical stability (default: 1e-5)
 * @param dim Dimension to normalize along (default: -1, which means the last dimension)
 * @returns Normalized tensor
 */
export function layerNorm<T extends number>(tensor: Tensor<T>, eps: number = 1e-5, dim: number = -1): Tensor<T> {
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
  
  // Convert negative dimension to positive
  if (dim < 0) {
    dim = tensor.shape.length + dim;
  }
  
  // Validate dimension
  if (dim < 0 || dim >= tensor.shape.length) {
    throw new Error(`Layer norm dimension out of bounds: ${dim} for tensor of rank ${tensor.shape.length}`);
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
  
  // Calculate stride for the given dimension
  const stride = tensor.shape[dim];
  
  // Iterate through each slice along the normalization dimension
  const batchSize = tensor.size / stride;
  
  for (let batch = 0; batch < batchSize; batch++) {
    // Get coordinates for the start of this batch
    const baseCoords = flatToCoords(batch * stride, tensor.shape.slice(0, dim).concat(tensor.shape.slice(dim + 1)));
    
    // Calculate mean
    let mean = 0;
    for (let i = 0; i < stride; i++) {
      const coords = [...baseCoords];
      coords.splice(dim, 0, i);
      const index = coordsToFlat(coords, tensor.shape);
      mean += tensor.data[index] as number;
    }
    mean /= stride;
    
    // Calculate variance
    let variance = 0;
    for (let i = 0; i < stride; i++) {
      const coords = [...baseCoords];
      coords.splice(dim, 0, i);
      const index = coordsToFlat(coords, tensor.shape);
      const diff = (tensor.data[index] as number) - mean;
      variance += diff * diff;
    }
    variance /= stride;
    
    // Normalize
    const invStd = 1 / Math.sqrt(variance + eps);
    for (let i = 0; i < stride; i++) {
      const coords = [...baseCoords];
      coords.splice(dim, 0, i);
      const index = coordsToFlat(coords, tensor.shape);
      result.data[index] = (((tensor.data[index] as number) - mean) * invStd) as T;
    }
  }
  
  return result;
}

/**
 * Mean Squared Error (MSE) loss function
 * 
 * @param predictions Predicted values
 * @param targets Target values
 * @param reduction Reduction method ('mean' or 'sum', default: 'mean')
 * @returns MSE loss value
 */
export function mseLoss<T extends number>(
  predictions: Tensor<T>,
  targets: Tensor<T>,
  reduction: 'mean' | 'sum' = 'mean'
): number {
  // Validate shapes
  if (predictions.shape.length !== targets.shape.length) {
    throw new Error(
      `Predictions and targets must have the same rank: ${predictions.shape.length} vs ${targets.shape.length}`
    );
  }
  
  for (let i = 0; i < predictions.shape.length; i++) {
    if (predictions.shape[i] !== targets.shape[i]) {
      throw new Error(
        `Predictions and targets shape mismatch at dimension ${i}: ${predictions.shape[i]} vs ${targets.shape[i]}`
      );
    }
  }
  
  // Calculate squared differences
  let sum = 0;
  for (let i = 0; i < predictions.size; i++) {
    const diff = (predictions.data[i] as number) - (targets.data[i] as number);
    sum += diff * diff;
  }
  
  // Apply reduction
  if (reduction === 'mean') {
    return sum / predictions.size;
  } else {
    return sum;
  }
}

/**
 * Binary Cross Entropy loss function
 * 
 * @param predictions Predicted probabilities (between 0 and 1)
 * @param targets Target values (0 or 1)
 * @param reduction Reduction method ('mean' or 'sum', default: 'mean')
 * @param eps Small constant for numerical stability (default: 1e-7)
 * @returns Binary cross entropy loss value
 */
export function binaryCrossEntropyLoss<T extends number>(
  predictions: Tensor<T>,
  targets: Tensor<T>,
  reduction: 'mean' | 'sum' = 'mean',
  eps: number = 1e-7
): number {
  // Validate shapes
  if (predictions.shape.length !== targets.shape.length) {
    throw new Error(
      `Predictions and targets must have the same rank: ${predictions.shape.length} vs ${targets.shape.length}`
    );
  }
  
  for (let i = 0; i < predictions.shape.length; i++) {
    if (predictions.shape[i] !== targets.shape[i]) {
      throw new Error(
        `Predictions and targets shape mismatch at dimension ${i}: ${predictions.shape[i]} vs ${targets.shape[i]}`
      );
    }
  }
  
  // Calculate binary cross entropy
  let sum = 0;
  for (let i = 0; i < predictions.size; i++) {
    // Clamp predictions to avoid numerical issues
    const pred = Math.max(eps, Math.min(1 - eps, predictions.data[i] as number));
    const target = targets.data[i] as number;
    
    sum += -(target * Math.log(pred) + (1 - target) * Math.log(1 - pred));
  }
  
  // Apply reduction
  if (reduction === 'mean') {
    return sum / predictions.size;
  } else {
    return sum;
  }
}

/**
 * Dropout function - randomly zeros elements with probability p during training
 * 
 * @param tensor Input tensor
 * @param p Dropout probability (default: 0.5)
 * @param training Whether in training mode (default: true)
 * @returns Output tensor with dropout applied
 */
export function dropout<T extends number>(
  tensor: Tensor<T>,
  p: number = 0.5,
  training: boolean = true
): Tensor<T> {
  // If not in training mode, just return the tensor
  if (!training || p <= 0) {
    return tensor.clone();
  }
  
  // Clamp p to valid range
  p = Math.min(1, Math.max(0, p));
  
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
  
  const scale = 1 / (1 - p);
  
  for (let i = 0; i < tensor.size; i++) {
    if (Math.random() >= p) {
      result.data[i] = (tensor.data[i] as number * scale) as T;
    } else {
      result.data[i] = 0 as T;
    }
  }
  
  return result;
}