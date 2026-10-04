/**
 * Example of tensor matrix operations
 */

import { Tensor } from '../tensor/tensor';
import { 
  matmul, 
  transpose, 
  reshape, 
  eye, 
  diag
} from '../tensor/operations/matrix';
import {
  add,
  subtract,
  multiply,
  exp,
  log,
  sqrt
} from '../tensor/operations/basic';
import {
  relu,
  sigmoid,
  tanh,
  softmax,
  layerNorm
} from '../tensor/operations/nn';

/**
 * Run matrix operation examples
 */
export function runMatrixExamples() {
  console.log("Tensor Matrix Operations Examples");
  console.log("================================");
  
  // Create some tensors
  const a = new Tensor<number>([2, 3], [1, 2, 3, 4, 5, 6]);
  const b = new Tensor<number>([3, 2], [7, 8, 9, 10, 11, 12]);
  
  console.log("\nOriginal Tensors:");
  console.log("A:", a.toString());
  console.log("B:", b.toString());
  
  // Matrix multiplication
  const c = matmul(a, b);
  console.log("\nMatrix Multiplication (A * B):");
  console.log(c.toString());
  
  // Transpose
  const aT = transpose(a);
  console.log("\nTranspose of A:");
  console.log(aT.toString());
  
  // Reshape
  const aReshaped = reshape(a, [3, 2]);
  console.log("\nReshape A to [3, 2]:");
  console.log(aReshaped.toString());
  
  // Identity matrix
  const identity = eye(3);
  console.log("\nIdentity Matrix (3x3):");
  console.log(identity.toString());
  
  // Diagonal matrix
  const diagTensor = diag(new Tensor<number>([3], [5, 6, 7]));
  console.log("\nDiagonal Matrix from [5, 6, 7]:");
  console.log(diagTensor.toString());
  
  return {
    a,
    b,
    c,
    aT,
    aReshaped,
    identity,
    diagTensor
  };
}

/**
 * Run element-wise operation examples
 */
export function runElementWiseExamples() {
  console.log("\nTensor Element-wise Operations Examples");
  console.log("=====================================");
  
  // Create some tensors
  const a = new Tensor<number>([2, 2], [1, 2, 3, 4]);
  const b = new Tensor<number>([2, 2], [5, 6, 7, 8]);
  
  console.log("\nOriginal Tensors:");
  console.log("A:", a.toString());
  console.log("B:", b.toString());
  
  // Addition
  const sum = add(a, b);
  console.log("\nA + B:");
  console.log(sum.toString());
  
  // Subtraction
  const diff = subtract(a, b);
  console.log("\nA - B:");
  console.log(diff.toString());
  
  // Multiplication
  const prod = multiply(a, b);
  console.log("\nA * B (element-wise):");
  console.log(prod.toString());
  
  // Exponential
  const expA = exp(a);
  console.log("\nexp(A):");
  console.log(expA.toString());
  
  // Natural logarithm
  const logB = log(b);
  console.log("\nlog(B):");
  console.log(logB.toString());
  
  // Square root
  const sqrtB = sqrt(b);
  console.log("\nsqrt(B):");
  console.log(sqrtB.toString());
  
  return {
    a,
    b,
    sum,
    diff,
    prod,
    expA,
    logB,
    sqrtB
  };
}

/**
 * Run neural network operation examples
 */
export function runNNExamples() {
  console.log("\nTensor Neural Network Operations Examples");
  console.log("=======================================");
  
  // Create some tensors
  const a = new Tensor<number>([2, 3], [-1, 0, 2, 3, -2, 1]);
  const b = new Tensor<number>([2, 3], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
  
  console.log("\nOriginal Tensors:");
  console.log("A:", a.toString());
  console.log("B:", b.toString());
  
  // ReLU
  const reluA = relu(a);
  console.log("\nReLU(A):");
  console.log(reluA.toString());
  
  // Sigmoid
  const sigmoidA = sigmoid(a);
  console.log("\nSigmoid(A):");
  console.log(sigmoidA.toString());
  
  // Tanh
  const tanhA = tanh(a);
  console.log("\nTanh(A):");
  console.log(tanhA.toString());
  
  // Softmax
  const softmaxB = softmax(b, 1);
  console.log("\nSoftmax(B, dim=1):");
  console.log(softmaxB.toString());
  
  // Layer Normalization
  const layerNormB = layerNorm(b, 1e-5, 1);
  console.log("\nLayerNorm(B, dim=1):");
  console.log(layerNormB.toString());
  
  return {
    a,
    b,
    reluA,
    sigmoidA,
    tanhA,
    softmaxB,
    layerNormB
  };
}

/**
 * Run all tensor operation examples
 */
export function runAllExamples() {
  const matrixResults = runMatrixExamples();
  const elementWiseResults = runElementWiseExamples();
  const nnResults = runNNExamples();
  
  return {
    matrix: matrixResults,
    elementWise: elementWiseResults,
    nn: nnResults
  };
}

// Run examples if this file is executed directly
if (typeof window !== 'undefined') {
  runAllExamples();
}