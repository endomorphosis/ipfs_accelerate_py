/**
 * Neural network operations for tensor computation
 * 
 * This file provides implementations of common neural network operations 
 * (relu, sigmoid, etc.) with CPU backend. WebGPU implementations will be
 * provided in separate modules.
 */

import { Tensor } from '../tensor';

/**
 * Rectified Linear Unit (ReLU) activation function
 * f(x) = max(0, x)
 */
export function relu(a: Tensor): Tensor {
  // Create output tensor with same dimensions
  const outputTensor = new Tensor({
    dims: a.getDimensions(),
    dataType: a.getDataType(),
    storage: 'cpu',
    name: 'relu_result'
  });
  
  // Get data views
  const aData = a.getData<Float32Array>();
  const outputData = outputTensor.getData<Float32Array>();
  
  // Apply ReLU
  for (let i = 0; i < aData.length; i++) {
    outputData[i] = Math.max(0, aData[i]);
  }
  
  return outputTensor;
}

/**
 * Sigmoid activation function
 * f(x) = 1 / (1 + exp(-x))
 */
export function sigmoid(a: Tensor): Tensor {
  // Create output tensor with same dimensions
  const outputTensor = new Tensor({
    dims: a.getDimensions(),
    dataType: a.getDataType(),
    storage: 'cpu',
    name: 'sigmoid_result'
  });
  
  // Get data views
  const aData = a.getData<Float32Array>();
  const outputData = outputTensor.getData<Float32Array>();
  
  // Apply sigmoid
  for (let i = 0; i < aData.length; i++) {
    outputData[i] = 1 / (1 + Math.exp(-aData[i]));
  }
  
  return outputTensor;
}

/**
 * Hyperbolic tangent (tanh) activation function
 * f(x) = tanh(x)
 */
export function tanh(a: Tensor): Tensor {
  // Create output tensor with same dimensions
  const outputTensor = new Tensor({
    dims: a.getDimensions(),
    dataType: a.getDataType(),
    storage: 'cpu',
    name: 'tanh_result'
  });
  
  // Get data views
  const aData = a.getData<Float32Array>();
  const outputData = outputTensor.getData<Float32Array>();
  
  // Apply tanh
  for (let i = 0; i < aData.length; i++) {
    outputData[i] = Math.tanh(aData[i]);
  }
  
  return outputTensor;
}

/**
 * Gaussian Error Linear Unit (GELU) activation function
 * Approximation: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
export function gelu(a: Tensor): Tensor {
  // Create output tensor with same dimensions
  const outputTensor = new Tensor({
    dims: a.getDimensions(),
    dataType: a.getDataType(),
    storage: 'cpu',
    name: 'gelu_result'
  });
  
  // Get data views
  const aData = a.getData<Float32Array>();
  const outputData = outputTensor.getData<Float32Array>();
  
  // Constants for approximation
  const sqrt2OverPi = Math.sqrt(2 / Math.PI);
  const alpha = 0.044715;
  
  // Apply GELU approximation
  for (let i = 0; i < aData.length; i++) {
    const x = aData[i];
    const x3 = x * x * x;
    outputData[i] = 0.5 * x * (1 + Math.tanh(sqrt2OverPi * (x + alpha * x3)));
  }
  
  return outputTensor;
}

/**
 * 2D Convolution operation
 * Simple implementation for demonstration purposes
 */
export function conv2d(
  input: Tensor, 
  filters: Tensor, 
  options: {
    strides?: [number, number],
    padding?: 'valid' | 'same',
  } = {}
): Tensor {
  // Default options
  const strides = options.strides || [1, 1];
  const padding = options.padding || 'valid';
  
  // Get dimensions
  const inputDims = input.getDimensions(); // [batch, height, width, channels]
  const filterDims = filters.getDimensions(); // [filterHeight, filterWidth, inChannels, outChannels]
  
  if (inputDims.length !== 4) {
    throw new Error(`conv2d expects input with 4 dimensions [batch, height, width, channels], got ${inputDims.length}`);
  }
  
  if (filterDims.length !== 4) {
    throw new Error(`conv2d expects filters with 4 dimensions [filterHeight, filterWidth, inChannels, outChannels], got ${filterDims.length}`);
  }
  
  const [batchSize, inputHeight, inputWidth, inChannels] = inputDims;
  const [filterHeight, filterWidth, filterInChannels, outChannels] = filterDims;
  
  if (inChannels !== filterInChannels) {
    throw new Error(`conv2d channel mismatch: input has ${inChannels} channels, filter expects ${filterInChannels}`);
  }
  
  // Calculate output dimensions
  let outputHeight: number;
  let outputWidth: number;
  
  if (padding === 'valid') {
    outputHeight = Math.floor((inputHeight - filterHeight) / strides[0]) + 1;
    outputWidth = Math.floor((inputWidth - filterWidth) / strides[1]) + 1;
  } else { // 'same'
    outputHeight = Math.ceil(inputHeight / strides[0]);
    outputWidth = Math.ceil(inputWidth / strides[1]);
  }
  
  // Create output tensor
  const outputTensor = new Tensor({
    dims: [batchSize, outputHeight, outputWidth, outChannels],
    dataType: input.getDataType(),
    storage: 'cpu',
    name: 'conv2d_result'
  });
  
  // Get data views
  const inputData = input.getData<Float32Array>();
  const filterData = filters.getData<Float32Array>();
  const outputData = outputTensor.getData<Float32Array>();
  
  // Padding values for 'same' padding
  let padTop = 0;
  let padBottom = 0;
  let padLeft = 0;
  let padRight = 0;
  
  if (padding === 'same') {
    const padAlongHeight = Math.max(0, (outputHeight - 1) * strides[0] + filterHeight - inputHeight);
    const padAlongWidth = Math.max(0, (outputWidth - 1) * strides[1] + filterWidth - inputWidth);
    padTop = Math.floor(padAlongHeight / 2);
    padBottom = padAlongHeight - padTop;
    padLeft = Math.floor(padAlongWidth / 2);
    padRight = padAlongWidth - padLeft;
  }
  
  // Perform convolution
  for (let b = 0; b < batchSize; b++) {
    for (let oh = 0; oh < outputHeight; oh++) {
      for (let ow = 0; ow < outputWidth; ow++) {
        for (let oc = 0; oc < outChannels; oc++) {
          let sum = 0;
          
          for (let fh = 0; fh < filterHeight; fh++) {
            for (let fw = 0; fw < filterWidth; fw++) {
              for (let ic = 0; ic < inChannels; ic++) {
                const ih = oh * strides[0] + fh - padTop;
                const iw = ow * strides[1] + fw - padLeft;
                
                // Skip if outside input boundaries
                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                  const inputIdx = b * (inputHeight * inputWidth * inChannels) +
                                  ih * (inputWidth * inChannels) +
                                  iw * inChannels +
                                  ic;
                  
                  const filterIdx = fh * (filterWidth * inChannels * outChannels) +
                                   fw * (inChannels * outChannels) +
                                   ic * outChannels +
                                   oc;
                  
                  sum += inputData[inputIdx] * filterData[filterIdx];
                }
              }
            }
          }
          
          const outputIdx = b * (outputHeight * outputWidth * outChannels) +
                           oh * (outputWidth * outChannels) +
                           ow * outChannels +
                           oc;
          
          outputData[outputIdx] = sum;
        }
      }
    }
  }
  
  return outputTensor;
}

/**
 * Softmax function
 * f(x_i) = exp(x_i) / sum(exp(x_j))
 * Applied to the last dimension
 */
export function softmax(a: Tensor): Tensor {
  // Create output tensor with same dimensions
  const outputTensor = new Tensor({
    dims: a.getDimensions(),
    dataType: a.getDataType(),
    storage: 'cpu',
    name: 'softmax_result'
  });
  
  // Get data views
  const aData = a.getData<Float32Array>();
  const outputData = outputTensor.getData<Float32Array>();
  
  // Get dimensions
  const dims = a.getDimensions();
  const lastDimSize = dims[dims.length - 1];
  const outerSize = a.getSize() / lastDimSize;
  
  // Apply softmax along last dimension
  for (let i = 0; i < outerSize; i++) {
    const offset = i * lastDimSize;
    
    // Find max for numerical stability
    let max = -Infinity;
    for (let j = 0; j < lastDimSize; j++) {
      max = Math.max(max, aData[offset + j]);
    }
    
    // Calculate exp(x_i - max) for each element
    let sum = 0;
    for (let j = 0; j < lastDimSize; j++) {
      const expValue = Math.exp(aData[offset + j] - max);
      outputData[offset + j] = expValue;
      sum += expValue;
    }
    
    // Normalize by sum
    for (let j = 0; j < lastDimSize; j++) {
      outputData[offset + j] /= sum;
    }
  }
  
  return outputTensor;
}