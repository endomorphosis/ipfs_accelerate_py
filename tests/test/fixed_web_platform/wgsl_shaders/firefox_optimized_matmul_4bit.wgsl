// Firefox-optimized 4-bit matrix multiplication shader
// Optimized for Firefox's WebGPU implementation with careful attention to:
// - Reduced synchronization barriers
// - Minimal control flow for better shader compilation
// - Aggressive buffer reuse to minimize memory pressure
// - Optimized uniform buffer access patterns

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

// Structure for 4-bit value extraction
struct PackedValues {
    values: array<f32, 8>, // 8 values packed in a u32
}

// Workgroup shared memory with optimized layout for Firefox
var<workgroup> shared_a: array<f32, 512>; // 8x8 workgroup x 8 values per thread
var<workgroup> shared_b: array<f32, 512>; // 8x8 workgroup x 8 values per thread

// Extract 4-bit values from a u32 (optimized for Firefox)
fn extract_4bit_values(packed: u32) -> PackedValues {
    var result: PackedValues;
    
    // Direct bitwise operations with minimal branching for Firefox
    result.values[0] = f32((packed >> 0u) & 0xFu);
    result.values[1] = f32((packed >> 4u) & 0xFu);
    result.values[2] = f32((packed >> 8u) & 0xFu);
    result.values[3] = f32((packed >> 12u) & 0xFu);
    result.values[4] = f32((packed >> 16u) & 0xFu);
    result.values[5] = f32((packed >> 20u) & 0xFu);
    result.values[6] = f32((packed >> 24u) & 0xFu);
    result.values[7] = f32((packed >> 28u) & 0xFu);
    
    return result;
}

// Apply dequantization to convert 4-bit values to floating point
fn dequantize(val: f32, scale: f32, zero: f32) -> f32 {
    return (val - zero) * scale;
}

// Main compute shader entry point
@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    
    // Skip threads outside matrix dimensions
    if (row >= params.M || col >= params.N) {
        return;
    }
    
    let wg_row = local_id.x;
    let wg_col = local_id.y;
    let wg_idx = wg_row * 8u + wg_col;
    
    // Initialize accumulator
    var acc: f32 = 0.0;
    
    // Process matrix in tiles (optimized for Firefox)
    for (var k: u32 = 0u; k < params.K; k += 8u) {
        // Collaborative loading of A and B tiles into shared memory
        // This pattern is optimized for Firefox's WebGPU implementation
        if (k + wg_col < params.K) {
            let packed_a = matrix_a[(row * params.K + k + wg_col) / 8u];
            let block_idx_a = ((row * params.K + k + wg_col) / params.block_size) % params.group_size;
            let scale_a = scales_a[block_idx_a];
            let zero_a = zeros_a[block_idx_a];
            
            let values_a = extract_4bit_values(packed_a);
            
            // Store dequantized values directly to reduce register pressure
            for (var i: u32 = 0u; i < 8u; i++) {
                if (k + wg_col + i < params.K) {
                    shared_a[wg_row * 64u + wg_col * 8u + i] = dequantize(values_a.values[i], scale_a, zero_a);
                }
            }
        }
        
        if (k + wg_row < params.K) {
            let packed_b = matrix_b[((k + wg_row) * params.N + col) / 8u];
            let block_idx_b = (((k + wg_row) * params.N + col) / params.block_size) % params.group_size;
            let scale_b = scales_b[block_idx_b];
            let zero_b = zeros_b[block_idx_b];
            
            let values_b = extract_4bit_values(packed_b);
            
            // Store dequantized values directly
            for (var i: u32 = 0u; i < 8u; i++) {
                if (col + i < params.N) {
                    shared_b[wg_row * 64u + wg_col * 8u + i] = dequantize(values_b.values[i], scale_b, zero_b);
                }
            }
        }
        
        // Synchronize to ensure all data is loaded
        workgroupBarrier();
        
        // Compute matrix multiplication for this tile
        // Unrolled loop for better performance on Firefox
        if (k < params.K) { acc += shared_a[wg_idx] * shared_b[0u * 8u + wg_col]; }
        if (k + 1u < params.K) { acc += shared_a[wg_idx + 1u] * shared_b[1u * 8u + wg_col]; }
        if (k + 2u < params.K) { acc += shared_a[wg_idx + 2u] * shared_b[2u * 8u + wg_col]; }
        if (k + 3u < params.K) { acc += shared_a[wg_idx + 3u] * shared_b[3u * 8u + wg_col]; }
        if (k + 4u < params.K) { acc += shared_a[wg_idx + 4u] * shared_b[4u * 8u + wg_col]; }
        if (k + 5u < params.K) { acc += shared_a[wg_idx + 5u] * shared_b[5u * 8u + wg_col]; }
        if (k + 6u < params.K) { acc += shared_a[wg_idx + 6u] * shared_b[6u * 8u + wg_col]; }
        if (k + 7u < params.K) { acc += shared_a[wg_idx + 7u] * shared_b[7u * 8u + wg_col]; }
        
        // Firefox-optimized: Single barrier at the end of each tile
        workgroupBarrier();
    }
    
    // Write final result
    matrix_c[row * params.N + col] = acc;
}