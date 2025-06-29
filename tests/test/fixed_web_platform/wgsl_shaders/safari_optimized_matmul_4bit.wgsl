// Safari-optimized 4-bit matrix multiplication shader
// Optimized for Safari's WebGPU implementation with conservative optimizations:
// - Simplified shader structure
// - Higher precision operations for stability 
// - Linear compute path with minimal divergence
// - Reduced block sizes and batch sizes for better Safari compatibility

@group(0) @binding(0) var<storage, read> matrix_a: array<u32>; // 4-bit packed input matrix A
@group(0) @binding(1) var<storage, read> matrix_b: array<u32>; // 4-bit packed input matrix B
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>; // Output matrix C
@group(0) @binding(3) var<storage, read> scales_a: array<f32>; // Scales for matrix A
@group(0) @binding(4) var<storage, read> scales_b: array<f32>; // Scales for matrix B
@group(0) @binding(5) var<storage, read> zeros_a: array<f32>; // Zero points for matrix A
@group(0) @binding(6) var<storage, read> zeros_b: array<f32>; // Zero points for matrix B
@group(0) @binding(7) var<uniform> params: Params;

struct Params {
    M: u32, // Rows in A
    N: u32, // Columns in B
    K: u32, // Columns in A, Rows in B
    block_size: u32, // Quantization block size
    group_size: u32, // Workgroup size
}

// Smaller workgroup size for better Safari WebGPU compatibility
@compute @workgroup_size(4, 4)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    
    // Skip threads outside matrix dimensions
    if (row >= params.M || col >= params.N) {
        return;
    }
    
    // Initialize accumulator (using explicit fp32 for Safari stability)
    var acc: f32 = 0.0;
    
    // Safari-optimized approach: direct computation without shared memory
    // Compute directly to avoid potential shared memory issues in Safari
    for (var k: u32 = 0u; k < params.K; k += 8u) {
        // Process a small batch of 8 elements at a time
        for (var i: u32 = 0u; i < 8u; i++) {
            let k_idx = k + i;
            if (k_idx >= params.K) {
                break; // Avoid out-of-bounds access
            }
            
            // Extract values from matrix A
            let a_idx = (row * params.K + k_idx) / 8u;
            let a_offset = (row * params.K + k_idx) % 8u;
            let packed_a = matrix_a[a_idx];
            let a_value_4bit = f32((packed_a >> (a_offset * 4u)) & 0xFu);
            
            // Apply dequantization for matrix A
            let block_idx_a = ((row * params.K + k_idx) / params.block_size) % params.group_size;
            let scale_a = scales_a[block_idx_a];
            let zero_a = zeros_a[block_idx_a];
            let a_value = (a_value_4bit - zero_a) * scale_a;
            
            // Extract values from matrix B
            let b_idx = ((k_idx * params.N) + col) / 8u;
            let b_offset = ((k_idx * params.N) + col) % 8u;
            let packed_b = matrix_b[b_idx];
            let b_value_4bit = f32((packed_b >> (b_offset * 4u)) & 0xFu);
            
            // Apply dequantization for matrix B
            let block_idx_b = (((k_idx) * params.N + col) / params.block_size) % params.group_size;
            let scale_b = scales_b[block_idx_b];
            let zero_b = zeros_b[block_idx_b];
            let b_value = (b_value_4bit - zero_b) * scale_b;
            
            // Accumulate the product (using FP32 for stable math in Safari)
            acc = acc + (a_value * b_value);
        }
    }
    
    // Write final result
    matrix_c[row * params.N + col] = acc;
}