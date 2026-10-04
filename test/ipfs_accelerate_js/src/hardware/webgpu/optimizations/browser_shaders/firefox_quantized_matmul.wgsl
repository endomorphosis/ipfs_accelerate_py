// Firefox-optimized quantized matrix multiplication shader
// This shader is optimized for Mozilla Firefox's WebGPU implementation
// Key optimizations:
// - Uses smaller workgroup size (8x8) for better occupancy
// - Simpler memory access patterns that perform better in Firefox
// - Reduced unrolling for better Firefox shader compiler performance
// - Optimized bit extraction operations for different bit widths

// Supported bit widths: 1, 2, 3, 4, 8

// Binding layout
@group(0) @binding(0) var<storage, read> a: array<f32>; // Input matrix A
@group(0) @binding(1) var<storage, read> b: array<u32>; // Quantized input matrix B
@group(0) @binding(2) var<storage, read_write> c: array<f32>; // Output matrix
@group(0) @binding(3) var<storage, read> scales: array<f32>; // Scales for dequantization
@group(0) @binding(4) var<storage, read> zeroPoints: array<f32>; // Zero points for dequantization

// Uniforms for matrix dimensions and quantization parameters
struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    bitsPerWeight: u32,
    usePerChannelQuantization: u32,
    paddingDummy1: u32,
    paddingDummy2: u32,
    paddingDummy3: u32,
};
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

// Shared memory for tiled matrix multiplication
var<workgroup> aTile: array<f32, 8 * 8>;
var<workgroup> bTile: array<f32, 8 * 8>;

// Returns a dequantized value from a packed quantized weight
fn dequantize1Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> bitIndex) & 1u;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize2Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> (bitIndex * 2u)) & 3u;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize3Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    // For 3-bit, we pack 10 values in each 32-bit word (3*10=30 bits used)
    // For 0-9 indexes: [0-2], [3-5], [6-8], [9-11], [12-14], [15-17], [18-20], [21-23], [24-26], [27-29]
    let startBit = bitIndex * 3u;
    let mask = 7u; // 0b111
    let bitValue = (packedValue >> startBit) & mask;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize4Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> (bitIndex * 4u)) & 15u;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize8Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> (bitIndex * 8u)) & 255u;
    return f32(bitValue) * scale + zeroPoint;
}

// Main compute shader
@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;
    let bitsPerWeight = uniforms.bitsPerWeight;
    let usePerChannelQuantization = uniforms.usePerChannelQuantization;

    // Matrix dimensions validation
    if (global_id.x >= N || global_id.y >= M) {
        return;
    }

    let row = global_id.y;
    let col = global_id.x;
    
    // Calculate tile indices
    let tileRow = local_id.y;
    let tileCol = local_id.x;
    
    // Initialize accumulator
    var acc = 0.0;
    
    // Number of tiles needed to cover the K dimension
    let numTiles = (K + 7u) / 8u;
    
    // Values per 32-bit word depends on bit width
    var valuesPerWord = 32u / bitsPerWeight;
    if (bitsPerWeight == 3u) {
        valuesPerWord = 10u; // Special case for 3-bit: 10 values per 32-bit word
    }
    
    // Load appropriate scale and zero point
    var scale = 0.0;
    var zeroPoint = 0.0;
    
    if (usePerChannelQuantization == 1u) {
        scale = scales[col];
        zeroPoint = zeroPoints[col];
    } else {
        scale = scales[0];
        zeroPoint = zeroPoints[0];
    }
    
    // Process tiles
    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load tile of matrix A into shared memory
        let aRow = row;
        let aCol = t * 8u + tileCol;
        
        if (aCol < K) {
            aTile[tileRow * 8u + tileCol] = a[aRow * K + aCol];
        } else {
            aTile[tileRow * 8u + tileCol] = 0.0;
        }
        
        // Load and dequantize tile of matrix B into shared memory
        let bRow = t * 8u + tileRow;
        let bCol = col;
        
        if (bRow < K && bCol < N) {
            // Calculate the index in the packed quantized data
            let packedIdx = (bRow * N + bCol) / valuesPerWord;
            let valueIdx = (bRow * N + bCol) % valuesPerWord;
            let packedValue = b[packedIdx];
            
            // Dequantize based on bit width
            var dequantizedValue = 0.0;
            
            if (bitsPerWeight == 1u) {
                dequantizedValue = dequantize1Bit(packedValue, valueIdx, scale, zeroPoint);
            } else if (bitsPerWeight == 2u) {
                dequantizedValue = dequantize2Bit(packedValue, valueIdx, scale, zeroPoint);
            } else if (bitsPerWeight == 3u) {
                dequantizedValue = dequantize3Bit(packedValue, valueIdx, scale, zeroPoint);
            } else if (bitsPerWeight == 4u) {
                dequantizedValue = dequantize4Bit(packedValue, valueIdx, scale, zeroPoint);
            } else if (bitsPerWeight == 8u) {
                dequantizedValue = dequantize8Bit(packedValue, valueIdx, scale, zeroPoint);
            }
            
            bTile[tileRow * 8u + tileCol] = dequantizedValue;
        } else {
            bTile[tileRow * 8u + tileCol] = 0.0;
        }
        
        // Synchronize to make sure both tiles are loaded
        workgroupBarrier();
        
        // Compute partial dot product for this tile
        for (var k = 0u; k < 8u; k = k + 1u) {
            acc = acc + aTile[tileRow * 8u + k] * bTile[k * 8u + tileCol];
        }
        
        // Synchronize before loading the next tile
        workgroupBarrier();
    }
    
    // Write result
    c[row * N + col] = acc;
}