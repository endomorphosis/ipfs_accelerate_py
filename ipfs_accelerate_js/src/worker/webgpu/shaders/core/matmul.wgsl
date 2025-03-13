/**
 * Matrix multiplication shader
 * A [M, K] * B [K, N] = C [M, N]
 */

struct MatrixDims {
  M: u32,
  K: u32,
  N: u32,
};

@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;
@group(0) @binding(2) var<storage, write> matrixC: array<f32>;
@group(0) @binding(3) var<uniform> dims: MatrixDims;

// Simple 8x8 workgroup size
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;

  // Bounds check
  if (row >= dims.M || col >= dims.N) {
    return;
  }

  // Compute matrix multiplication for this element
  var sum = 0.0;
  for (var k = 0u; k < dims.K; k = k + 1u) {
    sum = sum + matrixA[row * dims.K + k] * matrixB[k * dims.N + col];
  }

  // Write result
  matrixC[row * dims.N + col] = sum;
}