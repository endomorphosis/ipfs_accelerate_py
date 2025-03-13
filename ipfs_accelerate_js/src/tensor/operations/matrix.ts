/**
 * Matrix operations for tensor computation
 * 
 * This file provides implementations of common matrix operations (matmul, transpose, reshape)
 * with CPU backend. WebGPU implementations will be provided in separate modules.
 */

import { Tensor } from '../tensor';
import { TensorDataType, TensorStorage } from '../types';

/**
 * Calculate matrix multiplication between two 2D tensors
 * A: [M, K], B: [K, N] => C: [M, N]
 */
export function matmul(a: Tensor, b: Tensor): Tensor {
  // Check dimensions
  const aDims = a.getDimensions();
  const bDims = b.getDimensions();
  
  if (aDims.length !== 2 || bDims.length !== 2) {
    throw new Error(`matmul expects 2D tensors, got ${aDims.length}D and ${bDims.length}D`);
  }
  
  if (aDims[1] !== bDims[0]) {
    throw new Error(`matmul dimension mismatch: ${aDims} and ${bDims}`);
  }
  
  const M = aDims[0];
  const K = aDims[1];
  const N = bDims[1];
  
  // Create output tensor
  const cDims = [M, N];
  const cTensor = new Tensor({
    dims: cDims,
    dataType: a.getDataType(),
    storage: 'cpu',
    name: 'matmul_result'
  });
  
  // Get data views
  const aData = a.getData<Float32Array>();
  const bData = b.getData<Float32Array>();
  const cData = cTensor.getData<Float32Array>();
  
  // Perform matrix multiplication
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += aData[m * K + k] * bData[k * N + n];
      }
      cData[m * N + n] = sum;
    }
  }
  
  return cTensor;
}

/**
 * Transpose a 2D tensor
 * A: [M, N] => A^T: [N, M]
 */
export function transpose(a: Tensor): Tensor {
  // Check dimensions
  const aDims = a.getDimensions();
  
  if (aDims.length !== 2) {
    throw new Error(`transpose expects a 2D tensor, got ${aDims.length}D`);
  }
  
  const M = aDims[0];
  const N = aDims[1];
  
  // Create output tensor
  const transposedDims = [N, M];
  const transposedTensor = new Tensor({
    dims: transposedDims,
    dataType: a.getDataType(),
    storage: 'cpu',
    name: 'transpose_result'
  });
  
  // Get data views
  const aData = a.getData<Float32Array>();
  const transposedData = transposedTensor.getData<Float32Array>();
  
  // Perform transpose
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      transposedData[n * M + m] = aData[m * N + n];
    }
  }
  
  return transposedTensor;
}

/**
 * Reshape a tensor to new dimensions
 * Total element count must remain the same
 */
export function reshape(a: Tensor, newDims: number[]): Tensor {
  // Check dimensions
  const aDims = a.getDimensions();
  const aSize = a.getSize();
  const newSize = newDims.reduce((prod, dim) => prod * dim, 1);
  
  if (aSize !== newSize) {
    throw new Error(`reshape size mismatch: ${aSize} != ${newSize}`);
  }
  
  // Create output tensor
  const reshapedTensor = new Tensor({
    dims: newDims,
    dataType: a.getDataType(),
    storage: 'cpu',
    name: 'reshape_result'
  });
  
  // Get data views
  const aData = a.getData<Float32Array>();
  const reshapedData = reshapedTensor.getData<Float32Array>();
  
  // Copy data (no reordering needed for reshape)
  reshapedData.set(aData.subarray(0, aSize));
  
  return reshapedTensor;
}

/**
 * Create an identity matrix of the specified size
 */
export function eye(size: number, options: {
  dataType?: TensorDataType,
  storage?: TensorStorage,
  name?: string
} = {}): Tensor {
  // Create output tensor with zeros
  const tensor = new Tensor({
    dims: [size, size],
    dataType: options.dataType || 'float32',
    storage: options.storage || 'cpu',
    name: options.name || 'identity_matrix'
  });
  
  // Get data view
  const data = tensor.getData<Float32Array>();
  
  // Set diagonal elements to 1
  for (let i = 0; i < size; i++) {
    data[i * size + i] = 1;
  }
  
  return tensor;
}

/**
 * Diagonal matrix from a 1D tensor
 */
export function diag(a: Tensor): Tensor {
  // Check dimensions
  const aDims = a.getDimensions();
  
  if (aDims.length !== 1) {
    throw new Error(`diag expects a 1D tensor, got ${aDims.length}D`);
  }
  
  const size = aDims[0];
  
  // Create output tensor with zeros
  const diagTensor = new Tensor({
    dims: [size, size],
    dataType: a.getDataType(),
    storage: 'cpu',
    name: 'diag_result'
  });
  
  // Get data views
  const aData = a.getData<Float32Array>();
  const diagData = diagTensor.getData<Float32Array>();
  
  // Set diagonal elements
  for (let i = 0; i < size; i++) {
    diagData[i * size + i] = aData[i];
  }
  
  return diagTensor;
}