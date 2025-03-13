/**
 * WebGPU matrix operations for accelerated tensor computation
 * 
 * This file provides WebGPU implementations of common matrix operations
 * (matmul, transpose, etc.) using compute shaders for GPU acceleration.
 */

import { Tensor } from '../../../tensor/tensor';

// WebGPU specific types
interface GPUComputeImplementation {
  execute(inputBuffers: GPUBuffer[], outputBuffers: GPUBuffer[]): Promise<void>;
  dispose(): void;
}

/**
 * WGSL Shader code for matrix multiplication
 */
const MATMUL_SHADER = `
  struct MatrixDims {
    M: u32,
    K: u32,
    N: u32,
  };

  @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
  @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
  @group(0) @binding(2) var<storage, write> matrixC: array<f32>;
  @group(0) @binding(3) var<uniform> dims: MatrixDims;

  @compute @workgroup_size(8, 8)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= dims.M || col >= dims.N) {
      return;
    }

    var sum = 0.0;
    for (var k = 0u; k < dims.K; k = k + 1u) {
      sum = sum + matrixA[row * dims.K + k] * matrixB[k * dims.N + col];
    }

    matrixC[row * dims.N + col] = sum;
  }
`;

/**
 * WGSL Shader code for matrix transpose
 */
const TRANSPOSE_SHADER = `
  struct MatrixDims {
    M: u32,
    N: u32,
  };

  @group(0) @binding(0) var<storage, read> inputMatrix: array<f32>;
  @group(0) @binding(1) var<storage, write> outputMatrix: array<f32>;
  @group(0) @binding(2) var<uniform> dims: MatrixDims;

  @compute @workgroup_size(8, 8)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= dims.M || col >= dims.N) {
      return;
    }

    // Transposed: output[col, row] = input[row, col]
    outputMatrix[col * dims.M + row] = inputMatrix[row * dims.N + col];
  }
`;

/**
 * Create a WebGPU matrix multiplication implementation
 */
export async function createMatMulGPU(
  device: GPUDevice,
  M: number,
  K: number,
  N: number
): Promise<GPUComputeImplementation> {
  // Create shader module
  const shaderModule = device.createShaderModule({
    code: MATMUL_SHADER
  });

  // Create uniform buffer for dimensions
  const uniformBuffer = device.createBuffer({
    size: 3 * 4, // 3 u32 values (M, K, N)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Upload dimensions
  const uniformData = new Uint32Array([M, K, N]);
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // Create pipeline
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });

  // Return implementation
  return {
    async execute(inputBuffers: GPUBuffer[], outputBuffers: GPUBuffer[]): Promise<void> {
      // Create bind group
      const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffers[0] } }, // Matrix A
          { binding: 1, resource: { buffer: inputBuffers[1] } }, // Matrix B
          { binding: 2, resource: { buffer: outputBuffers[0] } }, // Matrix C (output)
          { binding: 3, resource: { buffer: uniformBuffer } }, // Matrix dimensions
        ],
      });

      // Create command encoder
      const commandEncoder = device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();

      // Dispatch compute work
      computePass.setPipeline(computePipeline);
      computePass.setBindGroup(0, bindGroup);
      
      // Calculate dispatch size (ceiling division)
      const workgroupSizeX = 8;
      const workgroupSizeY = 8;
      const dispatchX = Math.ceil(M / workgroupSizeX);
      const dispatchY = Math.ceil(N / workgroupSizeY);
      
      computePass.dispatchWorkgroups(dispatchX, dispatchY);
      computePass.end();

      // Submit commands and wait for completion
      const commandBuffer = commandEncoder.finish();
      device.queue.submit([commandBuffer]);

      // Wait for GPU to complete work
      await device.queue.onSubmittedWorkDone();
    },

    dispose(): void {
      uniformBuffer.destroy();
    }
  };
}

/**
 * Create a WebGPU matrix transpose implementation
 */
export async function createTransposeGPU(
  device: GPUDevice,
  M: number,
  N: number
): Promise<GPUComputeImplementation> {
  // Create shader module
  const shaderModule = device.createShaderModule({
    code: TRANSPOSE_SHADER
  });

  // Create uniform buffer for dimensions
  const uniformBuffer = device.createBuffer({
    size: 2 * 4, // 2 u32 values (M, N)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Upload dimensions
  const uniformData = new Uint32Array([M, N]);
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // Create pipeline
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });

  // Return implementation
  return {
    async execute(inputBuffers: GPUBuffer[], outputBuffers: GPUBuffer[]): Promise<void> {
      // Create bind group
      const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffers[0] } }, // Input matrix
          { binding: 1, resource: { buffer: outputBuffers[0] } }, // Output matrix
          { binding: 2, resource: { buffer: uniformBuffer } }, // Matrix dimensions
        ],
      });

      // Create command encoder
      const commandEncoder = device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();

      // Dispatch compute work
      computePass.setPipeline(computePipeline);
      computePass.setBindGroup(0, bindGroup);
      
      // Calculate dispatch size (ceiling division)
      const workgroupSizeX = 8;
      const workgroupSizeY = 8;
      const dispatchX = Math.ceil(M / workgroupSizeX);
      const dispatchY = Math.ceil(N / workgroupSizeY);
      
      computePass.dispatchWorkgroups(dispatchX, dispatchY);
      computePass.end();

      // Submit commands and wait for completion
      const commandBuffer = commandEncoder.finish();
      device.queue.submit([commandBuffer]);

      // Wait for GPU to complete work
      await device.queue.onSubmittedWorkDone();
    },

    dispose(): void {
      uniformBuffer.destroy();
    }
  };
}

/**
 * Matrix multiplication using WebGPU
 */
export async function matmulGPU(
  device: GPUDevice,
  a: Tensor, 
  b: Tensor
): Promise<Tensor> {
  // Check dimensions
  const aDims = a.getDimensions();
  const bDims = b.getDimensions();
  
  if (aDims.length !== 2 || bDims.length !== 2) {
    throw new Error(`matmulGPU expects 2D tensors, got ${aDims.length}D and ${bDims.length}D`);
  }
  
  if (aDims[1] !== bDims[0]) {
    throw new Error(`matmulGPU dimension mismatch: ${aDims} and ${bDims}`);
  }
  
  const M = aDims[0];
  const K = aDims[1];
  const N = bDims[1];
  
  // Create output tensor
  const cDims = [M, N];
  const cTensor = new Tensor({
    dims: cDims,
    dataType: a.getDataType(),
    storage: 'webgpu',
    name: 'matmul_result_gpu'
  });
  
  // Get data from tensors
  const aData = a.getData<Float32Array>();
  const bData = b.getData<Float32Array>();
  
  // Create WebGPU buffers
  const aBuffer = device.createBuffer({
    size: aData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  
  const bBuffer = device.createBuffer({
    size: bData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  
  const cBuffer = device.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT * M * N,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  
  // Copy data to GPU
  device.queue.writeBuffer(aBuffer, 0, aData);
  device.queue.writeBuffer(bBuffer, 0, bData);
  
  // Create matmul implementation
  const matmulImpl = await createMatMulGPU(device, M, K, N);
  
  // Execute matmul
  await matmulImpl.execute([aBuffer, bBuffer], [cBuffer]);
  
  // Read back result
  const resultBuffer = device.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT * M * N,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(cBuffer, 0, resultBuffer, 0, Float32Array.BYTES_PER_ELEMENT * M * N);
  device.queue.submit([commandEncoder.finish()]);
  
  // Map buffer for reading
  await resultBuffer.mapAsync(GPUMapMode.READ);
  const resultData = new Float32Array(resultBuffer.getMappedRange());
  
  // Copy data to tensor
  const cData = cTensor.getData<Float32Array>();
  cData.set(resultData);
  
  // Cleanup
  resultBuffer.unmap();
  matmulImpl.dispose();
  aBuffer.destroy();
  bBuffer.destroy();
  cBuffer.destroy();
  resultBuffer.destroy();
  
  return cTensor;
}

/**
 * Matrix transpose using WebGPU
 */
export async function transposeGPU(
  device: GPUDevice,
  a: Tensor
): Promise<Tensor> {
  // Check dimensions
  const aDims = a.getDimensions();
  
  if (aDims.length !== 2) {
    throw new Error(`transposeGPU expects a 2D tensor, got ${aDims.length}D`);
  }
  
  const M = aDims[0];
  const N = aDims[1];
  
  // Create output tensor
  const transposedDims = [N, M];
  const transposedTensor = new Tensor({
    dims: transposedDims,
    dataType: a.getDataType(),
    storage: 'webgpu',
    name: 'transpose_result_gpu'
  });
  
  // Get data from tensor
  const aData = a.getData<Float32Array>();
  
  // Create WebGPU buffers
  const aBuffer = device.createBuffer({
    size: aData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  
  const transposedBuffer = device.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT * M * N,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  
  // Copy data to GPU
  device.queue.writeBuffer(aBuffer, 0, aData);
  
  // Create transpose implementation
  const transposeImpl = await createTransposeGPU(device, M, N);
  
  // Execute transpose
  await transposeImpl.execute([aBuffer], [transposedBuffer]);
  
  // Read back result
  const resultBuffer = device.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT * M * N,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(transposedBuffer, 0, resultBuffer, 0, Float32Array.BYTES_PER_ELEMENT * M * N);
  device.queue.submit([commandEncoder.finish()]);
  
  // Map buffer for reading
  await resultBuffer.mapAsync(GPUMapMode.READ);
  const resultData = new Float32Array(resultBuffer.getMappedRange());
  
  // Copy data to tensor
  const transposedData = transposedTensor.getData<Float32Array>();
  transposedData.set(resultData);
  
  // Cleanup
  resultBuffer.unmap();
  transposeImpl.dispose();
  aBuffer.destroy();
  transposedBuffer.destroy();
  resultBuffer.destroy();
  
  return transposedTensor;
}