/**
 * Tiled matrix multiplication shader
 * Uses shared workgroup memory for improved performance
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

// Tile size for shared memory
const TILE_SIZE = 16u;

// Workgroup shared memory for tiled multiplication
var<workgroup> tileA: array<f32, TILE_SIZE * TILE_SIZE>;
var<workgroup> tileB: array<f32, TILE_SIZE * TILE_SIZE>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let row = global_id.x;
  let col = global_id.y;
  
  let localRow = local_id.x;
  let localCol = local_id.y;
  
  // Bounds check
  if (row >= dims.M || col >= dims.N) {
    return;
  }
  
  // Accumulator for dot product
  var sum = 0.0;
  
  // Number of tiles needed
  let numTiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;
  
  // Process each tile
  for (var t = 0u; t < numTiles; t = t + 1u) {
    // Load tiles into shared memory
    
    // Load from matrix A into tileA
    let tileARow = localRow;
    let tileACol = localCol;
    let aRow = row;
    let aCol = t * TILE_SIZE + tileACol;
    
    if (aRow < dims.M && aCol < dims.K) {
      tileA[tileARow * TILE_SIZE + tileACol] = matrixA[aRow * dims.K + aCol];
    } else {
      tileA[tileARow * TILE_SIZE + tileACol] = 0.0;
    }
    
    // Load from matrix B into tileB
    let tileBRow = localRow;
    let tileBCol = localCol;
    let bRow = t * TILE_SIZE + tileBRow;
    let bCol = col;
    
    if (bRow < dims.K && bCol < dims.N) {
      tileB[tileBRow * TILE_SIZE + tileBCol] = matrixB[bRow * dims.N + bCol];
    } else {
      tileB[tileBRow * TILE_SIZE + tileBCol] = 0.0;
    }
    
    // Wait for all threads to finish loading tiles
    workgroupBarrier();
    
    // Compute partial dot product using the tiles
    for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
      sum = sum + tileA[localRow * TILE_SIZE + k] * tileB[k * TILE_SIZE + localCol];
    }
    
    // Wait for all threads to finish using the tiles before loading next tiles
    workgroupBarrier();
  }
  
  // Write result
  if (row < dims.M && col < dims.N) {
    matrixC[row * dims.N + col] = sum;
  }
}