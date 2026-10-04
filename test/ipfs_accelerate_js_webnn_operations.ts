/**
 * WebNN Backend Additional Operations
 * Implements advanced operations for the WebNN backend
 */

import { WebNNBackend } from './ipfs_accelerate_js_webnn_backend';

/**
 * Pooling operation options
 */
export interface PoolingOptions {
  /**
   * Window dimensions [height, width]
   */
  windowDimensions: [number, number];
  
  /**
   * Padding values [top, right, bottom, left]
   */
  padding?: [number, number, number, number];
  
  /**
   * Strides [vertical, horizontal]
   */
  strides?: [number, number];
  
  /**
   * Dilations [vertical, horizontal]
   */
  dilations?: [number, number];
  
  /**
   * Layout type of the input
   */
  layout?: 'nchw' | 'nhwc';
}

/**
 * Normalization operation options
 */
export interface NormalizationOptions {
  /**
   * Epsilon value for numerical stability
   */
  epsilon?: number;
  
  /**
   * Axis or axes to normalize over
   */
  axes?: number[];
}

/**
 * ExecutePoolingOperation - Implements pooling operations (max, average)
 * 
 * @param backend - The WebNN backend instance
 * @param type - Type of pooling operation ('max' or 'average')
 * @param inputs - Input tensor and metadata
 * @param options - Pooling operation options
 * @returns Promise with the result tensor
 */
export async function executePoolingOperation(
  backend: WebNNBackend,
  type: 'max' | 'average',
  inputs: {
    input: { tensor: any; shape: number[] };
  },
  options: PoolingOptions
): Promise<{
  tensor: any;
  shape: number[];
  dataType: string;
}> {
  if (!backend['graphBuilder']) {
    throw new Error('WebNN backend not initialized');
  }
  
  const graphBuilder = backend['graphBuilder'];
  const { input } = inputs;
  const {
    windowDimensions,
    padding = [0, 0, 0, 0],
    strides = [1, 1],
    dilations = [1, 1],
    layout = 'nhwc'
  } = options;
  
  // Build pooling options
  const poolingOptions: Record<string, any> = {
    windowDimensions,
    padding,
    strides,
    dilations,
    layout
  };
  
  // Determine the pooling function
  let outputTensor;
  if (type === 'max') {
    outputTensor = graphBuilder.maxPool2d(input.tensor, poolingOptions);
  } else if (type === 'average') {
    outputTensor = graphBuilder.averagePool2d(input.tensor, poolingOptions);
  } else {
    throw new Error(`Unsupported pooling type: ${type}`);
  }
  
  // Calculate output shape
  // For NHWC layout: [batch, height, width, channels]
  // For NCHW layout: [batch, channels, height, width]
  let inputHeight, inputWidth, channels, batch;
  
  if (layout === 'nhwc') {
    [batch, inputHeight, inputWidth, channels] = input.shape;
  } else {
    [batch, channels, inputHeight, inputWidth] = input.shape;
  }
  
  // Calculate output dimensions
  const outputHeight = Math.floor(
    (inputHeight - windowDimensions[0] + padding[0] + padding[2]) / strides[0] + 1
  );
  const outputWidth = Math.floor(
    (inputWidth - windowDimensions[1] + padding[1] + padding[3]) / strides[1] + 1
  );
  
  // Output shape matches the layout of input
  const outputShape = layout === 'nhwc'
    ? [batch, outputHeight, outputWidth, channels]
    : [batch, channels, outputHeight, outputWidth];
  
  // Create a unique graph key
  const graphKey = `${type}Pool_${input.shape.join('x')}_window${windowDimensions.join('x')}_strides${strides.join('x')}`;
  
  // Build and execute the graph
  const result = await backend['runGraphComputation'](
    graphKey,
    { input: input.tensor },
    { output: outputTensor }
  );
  
  // Return the result
  return {
    tensor: result.output,
    shape: outputShape,
    dataType: 'float32'
  };
}

/**
 * ExecuteNormalizationOperation - Implements normalization operations (batch, layer)
 * 
 * @param backend - The WebNN backend instance
 * @param type - Type of normalization ('batch' or 'layer')
 * @param inputs - Input tensor, scale, bias, mean, variance
 * @param options - Normalization options
 * @returns Promise with the result tensor
 */
export async function executeNormalizationOperation(
  backend: WebNNBackend,
  type: 'batch' | 'layer',
  inputs: {
    input: { tensor: any; shape: number[] };
    scale?: { tensor: any; shape: number[] };
    bias?: { tensor: any; shape: number[] };
    mean?: { tensor: any; shape: number[] };
    variance?: { tensor: any; shape: number[] };
  },
  options: NormalizationOptions = {}
): Promise<{
  tensor: any;
  shape: number[];
  dataType: string;
}> {
  if (!backend['graphBuilder']) {
    throw new Error('WebNN backend not initialized');
  }
  
  const graphBuilder = backend['graphBuilder'];
  const { input, scale, bias, mean, variance } = inputs;
  const { epsilon = 1e-5, axes } = options;
  
  let outputTensor;
  
  if (type === 'batch') {
    // BatchNormalization requires mean and variance
    if (!mean || !variance) {
      throw new Error('BatchNormalization requires mean and variance tensors');
    }
    
    outputTensor = graphBuilder.batchNormalization(
      input.tensor,
      mean.tensor,
      variance.tensor,
      scale ? scale.tensor : null,
      bias ? bias.tensor : null,
      { epsilon }
    );
  } else if (type === 'layer') {
    // If direct LayerNormalization is available in WebNN
    if (graphBuilder.layerNormalization) {
      outputTensor = graphBuilder.layerNormalization(
        input.tensor,
        scale ? scale.tensor : null,
        bias ? bias.tensor : null,
        { epsilon, axes: axes || [-1] }
      );
    } else {
      // Implement layer normalization manually if not available
      // This is a simplified implementation
      outputTensor = implementLayerNormalization(
        graphBuilder,
        input.tensor,
        scale ? scale.tensor : null,
        bias ? bias.tensor : null,
        axes || [-1],
        epsilon
      );
    }
  } else {
    throw new Error(`Unsupported normalization type: ${type}`);
  }
  
  // Create a unique graph key
  const graphKey = `${type}Norm_${input.shape.join('x')}_eps${epsilon}`;
  
  // Build and execute the graph
  const result = await backend['runGraphComputation'](
    graphKey,
    {
      input: input.tensor,
      ...(scale && { scale: scale.tensor }),
      ...(bias && { bias: bias.tensor }),
      ...(mean && { mean: mean.tensor }),
      ...(variance && { variance: variance.tensor })
    },
    { output: outputTensor }
  );
  
  // Return the result with the same shape as input
  return {
    tensor: result.output,
    shape: input.shape,
    dataType: 'float32'
  };
}

/**
 * Implements layer normalization manually using low-level operations
 * Layer normalization normalizes across the specified axes
 */
function implementLayerNormalization(
  graphBuilder: any,
  input: any,
  scale: any | null,
  bias: any | null,
  axes: number[],
  epsilon: number
): any {
  // Calculate mean across normalization axes
  const mean = graphBuilder.reduceMean(input, { axes, keepDimensions: true });
  
  // Calculate variance: mean((x - mean)²)
  const diff = graphBuilder.sub(input, mean);
  const sqr = graphBuilder.mul(diff, diff);
  const variance = graphBuilder.reduceMean(sqr, { axes, keepDimensions: true });
  
  // Add epsilon for numerical stability
  const epsilonTensor = graphBuilder.constant(
    { type: 'float32', dimensions: [1] },
    new Float32Array([epsilon])
  );
  const varianceStable = graphBuilder.add(variance, epsilonTensor);
  
  // Calculate standard deviation
  const stdDev = graphBuilder.sqrt(varianceStable);
  
  // Normalize: (x - mean) / stdDev
  const normalized = graphBuilder.div(diff, stdDev);
  
  // Apply scale and bias if provided
  let result = normalized;
  
  if (scale) {
    result = graphBuilder.mul(result, scale);
  }
  
  if (bias) {
    result = graphBuilder.add(result, bias);
  }
  
  return result;
}

/**
 * ExecuteElementwiseOperation - Implements additional elementwise operations
 * 
 * @param backend - The WebNN backend instance
 * @param operation - Type of elementwise operation
 * @param inputs - Input tensors
 * @returns Promise with the result tensor
 */
export async function executeElementwiseOperation(
  backend: WebNNBackend,
  operation: 'add' | 'sub' | 'mul' | 'div' | 'pow' | 'min' | 'max' | 'exp' | 'log' | 'sqrt',
  inputs: {
    a: { tensor: any; shape: number[] };
    b?: { tensor: any; shape: number[] };
  }
): Promise<{
  tensor: any;
  shape: number[];
  dataType: string;
}> {
  if (!backend['graphBuilder']) {
    throw new Error('WebNN backend not initialized');
  }
  
  const graphBuilder = backend['graphBuilder'];
  const { a, b } = inputs;
  
  let outputTensor;
  
  // Single input operations
  if (['exp', 'log', 'sqrt'].includes(operation)) {
    if (!b) {
      switch (operation) {
        case 'exp':
          outputTensor = graphBuilder.exp(a.tensor);
          break;
        case 'log':
          outputTensor = graphBuilder.log(a.tensor);
          break;
        case 'sqrt':
          outputTensor = graphBuilder.sqrt(a.tensor);
          break;
      }
    } else {
      throw new Error(`Operation ${operation} expects only one input tensor`);
    }
  } 
  // Binary operations
  else if (b) {
    switch (operation) {
      case 'add':
        outputTensor = graphBuilder.add(a.tensor, b.tensor);
        break;
      case 'sub':
        outputTensor = graphBuilder.sub(a.tensor, b.tensor);
        break;
      case 'mul':
        outputTensor = graphBuilder.mul(a.tensor, b.tensor);
        break;
      case 'div':
        outputTensor = graphBuilder.div(a.tensor, b.tensor);
        break;
      case 'pow':
        outputTensor = graphBuilder.pow(a.tensor, b.tensor);
        break;
      case 'min':
        outputTensor = graphBuilder.min(a.tensor, b.tensor);
        break;
      case 'max':
        outputTensor = graphBuilder.max(a.tensor, b.tensor);
        break;
      default:
        throw new Error(`Unsupported elementwise operation: ${operation}`);
    }
  } else {
    throw new Error(`Operation ${operation} requires two input tensors`);
  }
  
  // Determine output shape (accounting for broadcasting)
  const outputShape = determineOutputShape(a.shape, b?.shape);
  
  // Create a unique graph key
  const graphKey = `elementwise_${operation}_${a.shape.join('x')}_${b?.shape?.join('x') || ''}`;
  
  // Build and execute the graph
  const inputs = { a: a.tensor };
  if (b) inputs['b'] = b.tensor;
  
  const result = await backend['runGraphComputation'](
    graphKey,
    inputs,
    { output: outputTensor }
  );
  
  // Return the result
  return {
    tensor: result.output,
    shape: outputShape,
    dataType: 'float32'
  };
}

/**
 * ExecuteTensorManipulationOperation - Implements tensor manipulation operations
 * 
 * @param backend - The WebNN backend instance
 * @param operation - Type of manipulation operation
 * @param inputs - Input tensor
 * @param options - Operation-specific options
 * @returns Promise with the result tensor
 */
export async function executeTensorManipulationOperation(
  backend: WebNNBackend,
  operation: 'reshape' | 'transpose' | 'concat' | 'slice' | 'pad',
  inputs: {
    input: { tensor: any; shape: number[] };
    inputs?: Array<{ tensor: any; shape: number[] }>;
  },
  options: Record<string, any> = {}
): Promise<{
  tensor: any;
  shape: number[];
  dataType: string;
}> {
  if (!backend['graphBuilder']) {
    throw new Error('WebNN backend not initialized');
  }
  
  const graphBuilder = backend['graphBuilder'];
  const { input, inputs } = inputs;
  
  let outputTensor;
  let outputShape;
  
  switch (operation) {
    case 'reshape':
      if (!options.newShape) {
        throw new Error('Reshape operation requires newShape option');
      }
      outputTensor = graphBuilder.reshape(input.tensor, options.newShape);
      outputShape = options.newShape;
      break;
      
    case 'transpose':
      const permutation = options.permutation || generateDefaultPermutation(input.shape.length);
      outputTensor = graphBuilder.transpose(input.tensor, permutation);
      outputShape = permuteDimensions(input.shape, permutation);
      break;
      
    case 'concat':
      if (!inputs || !Array.isArray(inputs) || inputs.length < 2) {
        throw new Error('Concat operation requires at least two input tensors');
      }
      const axis = options.axis || 0;
      outputTensor = graphBuilder.concat(
        inputs.map(i => i.tensor),
        axis
      );
      outputShape = calculateConcatShape(
        inputs.map(i => i.shape),
        axis
      );
      break;
      
    case 'slice':
      if (!options.starts || !options.sizes) {
        throw new Error('Slice operation requires starts and sizes options');
      }
      outputTensor = graphBuilder.slice(
        input.tensor,
        options.starts,
        options.sizes
      );
      outputShape = options.sizes;
      break;
      
    case 'pad':
      if (!options.padding) {
        throw new Error('Pad operation requires padding option');
      }
      const padValue = options.padValue !== undefined 
        ? options.padValue 
        : 0;
      
      // Create the constant for pad value
      const padValueTensor = graphBuilder.constant(
        { type: 'float32', dimensions: [1] },
        new Float32Array([padValue])
      );
      
      outputTensor = graphBuilder.pad(
        input.tensor,
        options.padding,
        padValueTensor
      );
      outputShape = calculatePaddedShape(input.shape, options.padding);
      break;
      
    default:
      throw new Error(`Unsupported tensor manipulation operation: ${operation}`);
  }
  
  // Create a unique graph key
  const graphKey = `manipulation_${operation}_${input.shape.join('x')}`;
  
  // Build and execute the graph
  const result = await backend['runGraphComputation'](
    graphKey,
    { input: input.tensor },
    { output: outputTensor }
  );
  
  // Return the result
  return {
    tensor: result.output,
    shape: outputShape,
    dataType: 'float32'
  };
}

// ======== Helper Functions ========

/**
 * Determine the output shape after broadcasting
 */
function determineOutputShape(shapeA: number[], shapeB?: number[]): number[] {
  if (!shapeB) return shapeA;
  
  const result = [];
  const maxLength = Math.max(shapeA.length, shapeB.length);
  
  // Pad shapes with 1s if needed
  const paddedA = Array(maxLength - shapeA.length).fill(1).concat(shapeA);
  const paddedB = Array(maxLength - shapeB.length).fill(1).concat(shapeB);
  
  // Calculate output shape
  for (let i = 0; i < maxLength; i++) {
    result.push(Math.max(paddedA[i], paddedB[i]));
  }
  
  return result;
}

/**
 * Generate default permutation for transpose
 */
function generateDefaultPermutation(rank: number): number[] {
  const result = [];
  for (let i = 0; i < rank; i++) {
    result.push(rank - 1 - i);
  }
  return result;
}

/**
 * Permute dimensions based on permutation array
 */
function permuteDimensions(shape: number[], permutation: number[]): number[] {
  return permutation.map(p => shape[p]);
}

/**
 * Calculate shape after concatenation
 */
function calculateConcatShape(shapes: number[][], axis: number): number[] {
  if (shapes.length === 0) return [];
  
  const baseShape = [...shapes[0]];
  let concatDimSize = baseShape[axis];
  
  for (let i = 1; i < shapes.length; i++) {
    concatDimSize += shapes[i][axis];
  }
  
  baseShape[axis] = concatDimSize;
  return baseShape;
}

/**
 * Calculate shape after padding
 */
function calculatePaddedShape(shape: number[], padding: number[][]): number[] {
  return shape.map((dim, i) => {
    const padPair = padding[i] || [0, 0];
    return dim + padPair[0] + padPair[1];
  });
}