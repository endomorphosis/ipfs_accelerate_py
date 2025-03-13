/**
 * Elementwise operations shader
 * Supports various operations through operation_type uniform
 */

// Operation types
// 0: Add
// 1: Subtract
// 2: Multiply
// 3: Divide
// 4: Maximum
// 5: Minimum
// 6: Power
// 7: Exponential
// 8: Log
// 9: Sqrt
// 10: ReLU
// 11: Sigmoid
// 12: Tanh

struct Params {
  // Total size of the tensors
  size: u32,
  // Operation type
  operation_type: u32,
  // Whether this is a binary operation (needs inputB)
  is_binary: u32,
  // Additional parameter for some operations
  alpha: f32,
};

@group(0) @binding(0) var<storage, read> inputA: array<f32>;
@group(0) @binding(1) var<storage, read> inputB: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// 256 threads per workgroup is generally a good size for GPUs
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Bounds check
  if (idx >= params.size) {
    return;
  }
  
  let a = inputA[idx];
  var result: f32;
  
  // Binary operations
  if (params.is_binary == 1u) {
    let b = inputB[idx];
    
    switch(params.operation_type) {
      case 0u: { // Add
        result = a + b;
        break;
      }
      case 1u: { // Subtract
        result = a - b;
        break;
      }
      case 2u: { // Multiply
        result = a * b;
        break;
      }
      case 3u: { // Divide
        result = a / b;
        break;
      }
      case 4u: { // Maximum
        result = max(a, b);
        break;
      }
      case 5u: { // Minimum
        result = min(a, b);
        break;
      }
      case 6u: { // Power
        result = pow(a, b);
        break;
      }
      default: { // Default to addition
        result = a + b;
        break;
      }
    }
  } 
  // Unary operations
  else {
    switch(params.operation_type) {
      case 7u: { // Exponential
        result = exp(a);
        break;
      }
      case 8u: { // Log
        result = log(a);
        break;
      }
      case 9u: { // Sqrt
        result = sqrt(a);
        break;
      }
      case 10u: { // ReLU
        result = max(0.0, a);
        break;
      }
      case 11u: { // Sigmoid
        result = 1.0 / (1.0 + exp(-a));
        break;
      }
      case 12u: { // Tanh
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let exp2x = exp(2.0 * a);
        result = (exp2x - 1.0) / (exp2x + 1.0);
        break;
      }
      default: { // Identity
        result = a;
        break;
      }
    }
  }
  
  output[idx] = result;
}