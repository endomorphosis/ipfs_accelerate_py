// FI: any;
 * Convert: any;
 * Conversi: any;
 * Th: any;
 * Conversi: any;
 */;

import {  HardwareAbstract: any;

// WebG: any;
/** WebG: any;

Th: any;
4-bit inference with adaptive precision. It provides optimized kernels for) {

1: a: any;
2: a: any;
3: a: any;
4: a: any;
5: a: any;

Usage) {
  import {(} fr: any;
    generate_compute_shad: any;
    get_browser_optimized_shader) { a: any;
    matmul_4bit_shad: any;
    kv_cache_adaptive_precision_sha: any;
  );
  
  // Genera: any;
  shader_code) { any) { any: any = generate_compute_shad: any;
    operation: any: any: any: any: any: any = "matmul",;"
    bits: any: any: any = 4: a: any;
    browser: any: any: any: any: any: any = "chrome",;"
    adaptive_precision: any: any: any = tr: any;
    layer_type: any: any: any: any: any: any = "attention";"
  ): any { */;

impo: any;
impo: any;
impo: any;
impo: any;
impo: any;
// Configu: any;
loggi: any;
  level: any: any: any = loggi: any;
  format: any: any = '%(asctime: a: any;'
);
logger: any: any: any = loggi: any;
;
// Functi: any;
function detect_browser_environment(): any -> Dict[ str:  any: any:  any: any, Any]) {
  /** Dete: any;
  
  Retu: any;
    Dictiona: any;
  result: any: any: any = ${$1}
  
  // Che: any;
  browser_env) { any) { any = os.(environ["BROWSER_SIMULATION"] !== undefined ? environ["BROWSER_SIMULATION"] : "") {.lower();"
  if ((((((($1) {
    result["detected"] = tru) { an) { an: any;"
    if ((($1) {
      result["browser"] = "chrome";"
      result["version"] = re.search(r"(\d+)", browser_env) { any).group(1) { any) if ((re.search(r"(\d+)", browser_env) { any) else { "113";"
    else if ((($1) {
      result["browser"] = "firefox";"
      result["version"] = re.search(r"(\d+)", browser_env) { any).group(1) { any) if ((re.search(r"(\d+)", browser_env) { any) else {"121"} else if ((($1) {"
      result["browser"] = "edge";"
      result["version"] = re.search(r"(\d+)", browser_env) { any).group(1) { any) if ((re.search(r"(\d+)", browser_env) { any) else { "113";"
    else if ((($1) {
      result["browser"] = "safari";"
      result["version"] = re.search(r"(\d+)", browser_env) { any).group(1) { any) if (re.search(r"(\d+)", browser_env) { any) else {"17";"
    return) { an) { an: any;
    }
  target_browser) { any) { any) { any: any: any: any = os.(environ["TARGET_BROWSER"] !== undefined ? environ["TARGET_BROWSER"] ) {"").lower();};"
  if (((((($1) {
    result["detected"] = tru) { an) { an: any;"
    result["browser"] = target_brows) { an: any;"
    result["version"] = os.(environ["BROWSER_VERSION"] !== undefined ? environ["BROWSER_VERSION"] ) {"latest");"
    retu: any;
  }
  try ${$1} catch(error) { any)) { any {pass}
  retu: any;

// Workgro: any;
BROWSER_WORKGROUP_CONFIG: any: any: any: any: any: any = {
  "chrome") { "
    "matmul") { ${$1},;"
    "attention") { ${$1},;"
    "kv_cache": ${$1}"
  "edge": {"
    "matmul": ${$1},;"
    "attention": ${$1},;"
    "kv_cache": ${$1}"
  "firefox": {"
    "matmul": ${$1},;"
    "attention": ${$1},;"
    "kv_cache": ${$1}"
  "safari": {"
    "matmul": ${$1},;"
    "attention": ${$1},;"
    "kv_cache": ${$1}"
  "default": {"
    "matmul": ${$1},;"
    "attention": ${$1},;"
    "kv_cache": ${$1}"
// Featu: any;
BROWSER_FEATURE_SUPPORT: any: any = {
  "chrome": ${$1},;"
  "edge": ${$1},;"
  "firefox": ${$1},;"
  "safari": ${$1},;"
  "default": ${$1}"

function get_workgroup_config($1:  string:  any: any:  any: any, $1: $2 | null: any: any = nu: any;
  /** G: any;
  ;
  Args): any {
    operation) { Operation type (matmul) { a: any;
    brow: any;
    
  Retu: any;
    Workgro: any;
  if ((((((($1) {
    browser_info) { any) { any) { any) { any = detect_browser_environmen) { an: any;
    browser: any: any = (browser_info["browser"] !== undefined ? browser_info["browser"] : ) if ((((((browser_info["detected"] !== undefined ? browser_info["detected"] ) { ) else {"default";}"
  browser) { any) { any) { any = browse) { an: any;
  if (((((($1) {
    browser) {any = "default";};"
  if (($1) {
    operation) {any = "matmul"  // Default) { an) { an: any;}"
  retur) { an: any;
;
function $1($1) { any): any { $2 | null: any: any = nu: any;
  /** G: any;
  ;
  Args) {
    browser) { Targ: any;
    
  Returns) {;
    Featu: any;
  if ((((((($1) {
    browser_info) { any) { any) { any) { any = detect_browser_environmen) { an: any;
    browser: any: any = (browser_info["browser"] !== undefined ? browser_info["browser"] : ) if ((((((browser_info["detected"] !== undefined ? browser_info["detected"] ) { ) else {"default";}"
  browser) { any) { any) { any = browse) { an: any;
  if (((((($1) {
    browser) {any = "default";}"
  return) { an) { an: any;

functio) { an: any;
  $1(;
  $1) { any): any { number: any: any: any = 4: a: any;
  $1: $2 | null: any: any: any = nu: any;
  $1: $2 | null: any: any: any = nu: any;
  workgroup_size: Record<str, int | null> = nu: any;
  $1: number: any: any: any = 1: any;
  $1: boolean: any: any: any = fal: any;
  $1: boolean: any: any: any = t: any;
) -> s: an: any;
  /** Genera: any;
  ;
  Args) {
    bits) { Precision bits (2) { a: any;
    brow: any;
    use_shared_mem: any;
    workgroup_s: any;
    block_s: any;
    per_channel) { U: any;
    symmetric) { U: any;
    
  Returns) {;
    WG: any;
  // G: any;
  if ((((((($1) {
    browser_info) { any) { any) { any) { any = detect_browser_environmen) { an: any;
    browser: any: any = (browser_info["browser"] !== undefined ? browser_info["browser"] : ) if ((((((browser_info["detected"] !== undefined ? browser_info["detected"] ) { ) else {null;};"
  if (($1) {
    workgroup_size) {any = get_workgroup_config("matmul", browser) { any) { an) { an: any;}"
  feature_support) { any: any = get_feature_suppo: any;
  
  // Determi: any;
  if (((($1) {
    use_shared_memory) {any = feature_support) { an) { an: any;}
  // Adjus) { an: any;
  workgroup_x) { any: any: any = workgroup_si: any;
  workgroup_y: any: any: any = workgroup_si: any;
  workgroup_z: any: any = (workgroup_size["z"] !== undefin: any;"
  
  // Constan: any;
  values_per_byte) { any) { any: any: any: any: any = 8 // bits if (((((bits > 0 else { 1;
  
  // Firefox) { an) { an: any;
  unroll_factor) { any) { any) { any: any: any: any = 4 if (((((browser != "firefox" && browser != "safari" else { 2;"
  
  // Create) { an) { an: any;
  shader) { any) { any) { any: any: any: any = `$1`;
  // WebG: any;
  // Configuration) { ${$1}-bit, ${$1}, ${$1}, block_size: any) { any: any: any: any: any: any: any: any: any: any: any = ${$1};
  // Optimized for ((((((${$1} browse) { an) { an) { an: any;
  ;
  struct Uniforms {${$1};
  
  @group(0) { any) { @binding(0) { any) var<uniform> uniforms) { Unif: any;
  @group(0: a: any;  // [M, K: a: any;
  @group(0: a: any; // Pack: any;
  @group(0: a: any;        // Quantizati: any;
  @group(0: any) @binding(4: any) var<storage, read> zeros: array<${$1}>; // Zero points (!used if ((((((symmetric) { any) {
  @group(0) { any) @binding(5) { any) var<storage, read_write> output_matrix) { array) { a) { an: any; // [M, N: a: any;
  /** // A: any;
  if (((($1) {
    shader += `$1`;
    var<workgroup> tile_input) { array<f16, ${$1} * ${$1}>;;
    var<workgroup> tile_weights) { array<u32, ${$1} * ${$1} * ${$1}>; */;
  
  }
  // Add) { an) { an: any;
  shader += `$1`;
  fn unpack_${$1}bit(packed_value) { any) { u32, idx) { any) { u32) -> u32 {
    let bits_per_value) { any: any: any: any: any: any: any: any: any: any: any = ${$1}u;;
    let mask: any: any: any: any: any: any = (1u << bits_per_va: any;
    ret: any;
  }
  
  fn apply_quantization(value: u32, scale: f16, {'zero: f16' if ((((((($1) {) { any { ${$1}) -> f16 {'
    ${$1}
    return scale * (f16(value) { any) - ${$1});
  }
  /** // Mai) { an: any;
  shader += `$1`;
  @compute @workgroup_size(${$1}, ${$1}, ${$1});
  f) { an: any;
    @builtin(global_invocation_id: any) global_id) { ve: any;
    @builtin(workgroup_id: any) workgroup_id) { ve: any;
    @builtin(local_invocation_id: a: any;
  ) {let M: any: any: any: any: any: any = unifo: any;;
    let N: any: any: any: any: any: any = unifo: any;
    let K: any: any: any: any: any: any = unifo: any;
    let block_size: any: any: any: any: any: any = unifo: any;}
    let row: any: any: any: any: any: any = global: any;
    let col: any: any: any: any: any: any = global: any;
    
    if (((((((row >= M || col >= N) {${$1}
    
    // Initialize) { an) { an: any;
    var acc) { any) { any) { any) { any: any: any = 0: a: an: any;
    
    // Calcula: any;
    let elements_per_u32: any: any: any: any: any: any: any: any: any: any: any = 32u / ${$1}u;
    
    // Ma: any;
  
  if ((((((($1) {
    // Version) { an) { an: any;
    shader += `$1`;
    for ((((var k_base) { any) { any) { any) { any) { any) { any) { any = 0) { a) { an: any;; k_ba) { an: any; k_base += ${$1}) {
      // Collaborati: any;
      if (((((((k_base + local_id.x < K) {
        tile_input[local_id.y * ${$1} + local_id.x] = input_matrix) {any;;}
      // Load) { an) { an: any;
      let weight_offset) { any) { any) { any: any: any: any = (k_base / elements_per_: any;
      if (((((((local_id.y < ${$1} && k_base + local_id.x * 4 < K) {
        for (((((((var i) { any) { any) { any) { any) { any) { any) { any = 0) { an) { an) { an: any; i) { a) { an: any; i += 1u) {
          let w_idx: any: any: any: any: any: any = local_id.y * ${$1} + local: any;;
          if (((((((k_base + w_idx < K) {${$1}
      workgroupBarrier) {any;}
      // Compute) { an) { an: any;
      let k_end) { any) { any) { any: any: any: any: any: any: any: any: any = min(K - k_base, ${$1}u);
      for (((((((var k_offset) { any) { any) { any) { any) { any) { any = 0: a: an: any; k_off: any; k_offset += ${$1}u) {// Unro: any;
        /**}
    // Unroll: any;
    for ((((let $1 = 0;; $1 < $2; $1++) {
      shader += `$1`;
        {
          let k) { any) { any) { any) { any) { any) { any: any: any: any: any: any = k_base + k_offset + ${$1}u;;
          if (((((((k < K) {
            let input_val) { any) { any) { any) { any) { any) { any: any: any: any: any: any = tile_input[local_id.y * ${$1} + k_offset + ${$1}u];
            
          }
            // Calcula: any;
            let block_idx) { any) {any) { any: any: any: any = k: a: an: any;}
            // Get packed weight && unpack the ${$1}-bit va: any;
            let packed_idx: any: any: any: any: any: any = k: a: an: any;
            let bit_offset: any: any: any: any: any: any = k: a: an: any;
            let packed_weight: any: any: any: any: any: any: any: any: any: any: any = tile_weights[k_offset + ${$1}u];
            let quantized: any: any: any: any: any: any = unpack_${$1}bit(packed_weight: a: any;
            
    }
            // App: any;
            let scale_idx: any: any: any: any: any: any: any: any: any: any: any = ${$1};
            let zero_idx: any: any: any: any: any: any: any: any: any: any: any = ${$1};
            let scale: any: any: any: any: any: any = sca: any;
            let ${$1};
            
            let weight_val: any: any: any: any: any: any: any = apply_quantization(quantized: any, scale, ${$1});
            
            // Accumula: any;
            acc += f: an: any;;
          } */;
    
    shader += `$1`;
      }
      
      workgroupBarr: any;;
    }
    /** } else {
    // Versi: any;
    shader += `$1`;
    for ((((var k) { any) { any) { any) { any) { any) { any = 0: a: an: any;; k: a: an: any; k += 1u) {let input_val: any: any: any: any: any: any = input_mat: any;;}
      // Calcula: any;
      let block_idx) { any) {any) { any: any: any: any = k: a: an: any;}
      // Get packed weight && unpack the ${$1}-bit va: any;
      let packed_idx: any: any: any: any: any: any = k: a: an: any;
      let bit_offset: any: any: any: any: any: any = k: a: an: any;
      let packed_weight: any: any: any: any: any: any = weight_mat: any;
      let quantized: any: any: any: any: any: any = unpack_${$1}bit(packed_weight: a: any;
      
      // App: any;
      let scale_idx: any: any: any: any: any: any: any: any: any: any: any = ${$1};
      let zero_idx: any: any: any: any: any: any: any: any: any: any: any = ${$1};
      let scale: any: any: any: any: any: any = sca: any;
      let ${$1};
      
      let weight_val: any: any: any: any: any: any: any = apply_quantization(quantized: any, scale, ${$1});
      
      // Accumula: any;
      acc += f: an: any;;
    } */;
  
  // Wri: any;
  shader += `$1`;
    // Wri: any;
    output_matrix[row * N + col] = f: an: any;;
  }
  /** retu: any;

functi: any;
  $1: number: any: any: any = 4: a: any;
  $1: $2 | null: any: any: any = nu: any;
  $1: number: any: any: any = 6: an: any;
  $1: boolean: any: any: any = tr: any;
  $1: boolean: any: any: any = tr: any;
  $1: boolean: any: any: any = t: any;
): a: any;
  Genera: any;
  
  A: any;
    b: any;
    browser) { Targ: any;
    block_size) { Blo: any;
    use_flash_attention) { U: any;
    causal_mask) { App: any;
    adaptive_precision) { Enab: any;
    
  Returns) {
    WG: any;
  /** // G: any;
  if ((((((($1) {
    browser_info) { any) { any) { any) { any = detect_browser_environment) { an) { an: any;
    browser) { any: any = (browser_info["browser"] !== undefined ? browser_info["browser"] : ) if ((((((browser_info["detected"] !== undefined ? browser_info["detected"] ) { ) else {null;}"
  workgroup_size) { any) { any) { any = get_workgroup_confi) { an: any;
  feature_support: any: any = get_feature_suppo: any;
  
  // Adju: any;
  use_shared_memory: any: any: any = feature_suppo: any;
  
  // Adju: any;
  workgroup_x: any: any: any = workgroup_si: any;
  workgroup_y: any: any: any = workgroup_si: any;
  workgroup_z: any: any = (workgroup_size["z"] !== undefin: any;"
  
  // Shad: any;
  shader: any: any: any: any: any: any = `$1`;
  // WebG: any;
  // Configuration) { ${$1}-bit, block_size: any: any: any: any: any: any: any: any = ${$1}, ${$1} FlashAttent: any;
  // ${$1}
  // ${$1}
  // Optimized for ((((((${$1} browse) { an) { an: any;
  
  struct Uniforms {${$1};
  
  @group(0) { any) { @binding(0) { any) var<uniform> uniforms) { Uniform) { a: an: any;
  @group(0: a: any;          // [batch_size, seq_len: any;
  @group(0: any) @binding(2: any) var<storage, read> key: array<${$1}>;   // Pack: any;
  @group(0: any) @binding(3: any) var<storage, read> value: array<${$1}>; // Pack: any;
  @group(0: a: any;     // K: any;
  @group(0: a: any;   // Val: any;
  @group(0: a: any;   // [batch_size, seq_len: any;
  
  // A: any;
  if (((($1) {
    shader += `$1`;
    var<workgroup> shared_q) { array<f16, ${$1} * ${$1}>;;
    var<workgroup> shared_k) { array<f16, ${$1} * ${$1}>;
    var<workgroup> shared_v) { array<f16, ${$1} * ${$1}>;
    var<workgroup> shared_s) { array<f32, ${$1} * ${$1}>;
    /** }
  // Helpe) { an: any;
  shader += `$1`;
  fn unpack_${$1}bit(packed_value) { any) { u32, idx) { any) { u32) -> u32 {
    let bits_per_value: any: any: any: any: any: any: any: any: any: any: any = ${$1}u;;
    let mask: any: any: any: any: any: any = (1u << bits_per_va: any;
    ret: any;
  }
  
  fn dequantize_${$1}bit(quantized: u32, scale: f16) -> f16 {${$1}
  
  fn masked_softmax(scores: array<f32, ${$1}>, length: u32, position: u32) -> array<f32, ${$1}> {
    var max_score: any: any: any: any: any: any: any: any: any: any: any = -1.0e9;
    var result: array<f32, ${$1}>;
    
  }
    // Fi: any;
    for ((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {
      if (((((((${$1}) {${$1}
    
    // Compute) { an) { an: any;
    var sum) { any) { any) { any) { any: any: any = 0: a: an: any;;
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {
      if (((((((${$1}) {${$1} else {${$1}
    
    // Normaliz) { an) { an: any;
    let scale) { any) { any) { any) { any: any: any = 1: a: an: any;;
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {${$1}
    
    ret: any;;
  } */;
  
  // Ma: any;
  shader += `$1`;
  @compute @workgroup_size(${$1}, ${$1}, ${$1});
  f: an: any;
    @builtin(global_invocation_id: a: any;
    @builtin(workgroup_id: a: any;
    @builtin(local_invocation_id: a: any;
  ) {let batch_idx: any: any: any: any: any: any = global: any;;
    let head_idx: any: any: any: any: any: any = global: any;
    let query_idx: any: any: any: any: any: any = global: any;}
    let batch_size: any: any: any: any: any: any = unifo: any;
    let seq_length: any: any: any: any: any: any = unifo: any;
    let num_heads: any: any: any: any: any: any = unifo: any;
    let head_size: any: any: any: any: any: any = unifo: any;
    let block_size: any: any: any: any: any: any = unifo: any;
    let precision_threshold: any: any: any: any: any: any = unifo: any;
    let kv_precision_bits: any: any: any: any: any: any = unifo: any;
    
    if (((((((batch_idx >= batch_size || head_idx >= num_heads || query_idx >= seq_length) {${$1}
    
    // Calculate) { an) { an: any;
    let elements_per_u32) { any) { any) { any) { any: any: any: any: any: any: any: any = 32u / ${$1}u;
    
    // Determi: any;
    let needs_high_precision) { any) { any) { any: any: any: any: any: any: any: any = ${$1} (;
      query_: any;
    
    // Pointe: any;
    let q_offset) { any) { any) { any: any: any: any = (batch_idx * seq_len: any;
    
    // Outp: any;
    let out_offset: any: any: any: any: any: any = (batch_idx * seq_len: any;
    
    // Initiali: any;
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {${$1}
    
    // Compu: any;
    var attn_scores) { array<f32, ${$1}>;;
    for (((((var key_pos) { any) { any) { any) { any) { any) { any = 0: a: an: any; key_: any; key_pos += 1u) {
      if (((((((${$1}) {${$1}
      // Key) { an) { an: any;
      let k_offset) { any) { any) { any) { any) { any) { any = (batch_idx * seq_leng) { an: any;;
      
      // Compu: any;
      var score) { any) { any) { any: any: any: any = 0: a: an: any;
      
      for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {let q_val: any: any: any: any: any: any = f: an: any;;}
        // Dequanti: any;
        v: any;
        if (((((((needs_high_precision || kv_precision_bits > ${$1}u) {
          // Use) { an) { an: any;
          let packed_idx) { any) { any) { any) { any) { any) { any = i) { a: an: any;
          let bit_offset: any: any: any: any: any: any = i: a: an: any;
          let packed_key: any: any: any: any: any: any = k: an: any;
          let quantized: any: any: any: any: any: any = unpack_${$1}bit(packed_key: a: any;
          let scale_idx: any: any: any: any: any: any = (head_idx * seq_len: any;
          k_val: any: any: any: any: any: any = f32(dequantize_${$1}bit(quantized: a: any;
        } else {
          // U: any;
          let packed_idx) {) { any {) { any { any: any: any: any: any: any = i: a: an: any;
          let bit_offset: any: any: any: any: any: any = i: a: an: any;
          let packed_key: any: any: any: any: any: any = k: an: any;
          let quantized: any: any: any: any: any: any = unpack_${$1}bit(packed_key: a: any;
          let scale_idx: any: any: any: any: any: any = (head_idx * seq_len: any;
          k_val: any: any: any: any: any: any = f32(dequantize_${$1}bit(quantized: a: any;
        }
        score += q_: any;;
      }
      
      // Sca: any;
      score /= s: any;
      
      // Sto: any;
      attn_scores[key_pos] = s: any;
    }
    
    // Compu: any;
    let attn_probs: any: any: any: any: any: any = masked_soft: any;
    
    // Compu: any;
    for (((((((var key_pos) { any) { any) { any) { any) { any) { any = 0: a: an: any; key_: any; key_pos += 1u) {
      if (((((((${$1}) {${$1}
      let v_offset) { any) { any) { any) { any) { any) { any = (batch_idx * seq_len: any;;
      let attn_prob: any: any: any: any: any: any = attn_pr: any;
      
      // Sk: any;
      if ((((attn_prob < 1.0e-8) {${$1}
      
      for (((((((var i) { any) { any) { any) { any) { any) { any) { any = 0) { an) { an) { an: any; i) { a) { an: any; i += 1u) {
        // Dequanti: any;
        v: any;;
        if (((((((needs_high_precision || kv_precision_bits > ${$1}u) {
          // Use) { an) { an: any;
          let packed_idx) { any) { any) { any) { any) { any) { any = i) { a: an: any;
          let bit_offset: any: any: any: any: any: any = i: a: an: any;
          let packed_value: any: any: any: any: any: any = va: any;
          let quantized: any: any: any: any: any: any = unpack_${$1}bit(packed_value: a: any;
          let scale_idx: any: any: any: any: any: any = (head_idx * seq_len: any;
          v_val: any: any: any: any: any: any = f32(dequantize_${$1}bit(quantized: a: any;
        } else {
          // U: any;
          let packed_idx) {) { any {) { any { any: any: any: any: any: any = i: a: an: any;
          let bit_offset: any: any: any: any: any: any = i: a: an: any;
          let packed_value: any: any: any: any: any: any = va: any;
          let quantized: any: any: any: any: any: any = unpack_${$1}bit(packed_value: a: any;
          let scale_idx: any: any: any: any: any: any = (head_idx * seq_len: any;
          v_val: any: any: any: any: any: any = f32(dequantize_${$1}bit(quantized: a: any;
        }
        // Weight: any;
        output[out_offset + i] += f: an: any;
      }
  /** retu: any;

functi: any;
  $1: number: any: any: any = 4: a: any;
  $1: $2 | null: any: any: any = nu: any;
  $1: boolean: any: any: any = tr: any;
  $1: boolean: any: any: any = tr: any;
  $1: number: any: any: any = 4: any;
): a: any;
  Genera: any;
  
  A: any;
    kv_cache_b: any;
    browser) { Targ: any;
    enable_variable_precision) { Enab: any;
    enable_sliding_window) { Enab: any;
    window_size) { Si: any;
    
  Returns) {;
    WG: any;
  /** // G: any;
  if ((((((($1) {
    browser_info) { any) { any) { any) { any = detect_browser_environmen) { an: any;
    browser: any: any = (browser_info["browser"] !== undefined ? browser_info["browser"] : ) if ((((((browser_info["detected"] !== undefined ? browser_info["detected"] ) { ) else {null;}"
  workgroup_size) { any) { any) { any = get_workgroup_confi) { an: any;
  feature_support: any: any = get_feature_suppo: any;
  
  // Adju: any;
  use_shared_memory: any: any: any = feature_suppo: any;
  
  // Adju: any;
  workgroup_x: any: any: any = workgroup_si: any;
  workgroup_y: any: any: any = workgroup_si: any;
  workgroup_z: any: any = (workgroup_size["z"] !== undefin: any;"
  
  // Safa: any;
  if (((((($1) {
    enable_variable_precision) {any = fals) { an) { an: any;}
  // Shade) { an: any;
  shader) { any) { any) { any: any: any: any = `$1`;
  // WebG: any;
  // Configuration) { ${$1}-bit default, ${$1}
  // ${$1}
  // Optimized for (((((${$1} browse) { an) { an: any;
  
  struct Uniforms {${$1};
  
  struct PrecisionConfig {${$1};
  
  @group(0) { any) { @binding(0) { any) var<uniform> uniforms) { Uniform) { a: an: any;
  @group(0: a: any;
  @group(0: a: any;  // Pack: any;
  @group(0: a: any;       // N: any;
  @group(0: a: any;    // Quantizati: any;
  @group(0: a: any; // Metadata for ((((((cache (precision) { any, etc.) {
  
  // Helper) { an) { an: any;
  fn pack_to_2bit(values) { any)) { any { array<f32, 16>, scales) { ptr<function, array<f32, 4>>) -> array<u32, 1> {var res: any;
    result[0] = 0: a: an: any;}
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {${$1}
    
    ret: any;;
  }
  
  fn pack_to_4bit(values: array<f32, 8>, scales: ptr<function, array<f32, 2>>) -> array<u32, 1> {var res: any;
    result[0] = 0: a: an: any;}
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {${$1}
    
    ret: any;;
  }
  
  fn pack_to_8bit(values: array<f32, 4>, scale: f32) -> array<u32, 1> {var res: any;
    result[0] = 0: a: an: any;}
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {${$1}
    
    ret: any;;
  }
  
  fn unpack_2bit(packed: u32, indices: array<u32, 4>, scale: f32) -> array<f32, 4> {var res: any;}
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {${$1}
    
    ret: any;;
  }
  
  fn unpack_4bit(packed: u32, indices: array<u32, 4>, scale: f32) -> array<f32, 4> {var res: any;}
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {${$1}
    
    ret: any;;
  }
  
  fn unpack_8bit(packed: u32, indices: array<u32, 4>, scale: f32) -> array<f32, 4> {var res: any;}
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {${$1}
    
    ret: any;;
  }
  
  // Functi: any;
  fn get_precision_for_position(position) { any) {: any {) { any { u32, current_length: u32) -> u32 {
    
      '// Fix: any;'
      if ((((((!enable_variable_precision else {
      /**;
 * 
      // Determine token recency (how far from the current token) {
      let recency) { any) { any) { any) {any) { any) { any) { any = current_lengt) { an) { an: any;}
      // Rece: any;
      if (((((((recency < precision_config.recent_token_count) { ${$1}
      // Early) { an) { an: any;
      if (((position < current_length / 4u) { ${$1}
      
      // Middle) {any;
      
 */}
    
    return ${$1}u;
  }
  
  // Main) { an) { an: any;
  @compute @workgroup_size(${$1}, ${$1}, ${$1}) {
  f) { an: any;
    @builtin(global_invocation_id) { any) global_id) { ve: any;
  ) {
    let batch_idx) { any) {any) { any: any: any: any = global: any;
    let head_idx: any: any: any: any: any: any = global: any;
    let value_idx: any: any: any: any: any: any = global: any;}
    let batch_size: any: any: any: any: any: any = unifo: any;
    let max_seq_length: any: any: any: any: any: any = unifo: any;
    let current_length: any: any: any: any: any: any = unifo: any;
    let num_heads: any: any: any: any: any: any = unifo: any;
    let head_size: any: any: any: any: any: any = unifo: any;
    
    if (((((((batch_idx >= batch_size || head_idx >= num_heads || value_idx >= head_size) {${$1}
    
    // Get) { an) { an: any;
    let new_position) { any) { any) { any) { any: any: any = current_le: any;
    
    // Hand: any;
    let effective_position) { any) { any) { any: any: any: any: any: any: any: any: any = ${$1};
    
    // Che: any;
    if ((((new_position >= max_seq_length) {${$1}
    
    // Determine) { an) { an: any;
    let precision_bits) { any) { any) { any) { any) { any) { any = get_precision_for_positi) { an: any;
    
    // Calcula: any;
    let values_per_u32: any: any: any: any: any: any = 3: an: any;
    let kv_size: any: any: any: any: any: any = head_s: any;
    
    // Offs: any;
    let kv_input_offset) { any) { any) { any: any: any: any = (batch_idx * num_he: any;
    
    // Calcula: any;
    let cache_position: any: any: any: any: any: any: any: any: any: any = (;
      (effective_position * batch_s: any;
    
    // G: any;
    let value: any: any: any: any: any: any = f: an: any;
    
    // Positi: any;
    let packed_idx: any: any: any: any: any: any = cache_posit: any;
    let bit_offset: any: any: any: any: any: any = cache_posit: any;
    
    // Calcula: any;
    let scale_position) { any) { any) { any: any: any: any: any: any: any: any = (;
      (effective_position * batch_s: any;
    
    // Proce: any;
    if (((((((value_idx % 4u) { any) { any) { any) { any = = 0u) {
      // Collec) { an: any;
      var values) { array) {any;
      values[0] = valu) { a: an: any;}
      // Group size is 4 values (could be processed in one u32 for (((((8-bit) {
      let group_size) { any) { any) { any) { any) { any) { any = m: an: any;
      
      for (((((((var i) { any) { any) { any) { any) { any) { any = 1: a: an: any; i: a: an: any; i += 1u) {
        if (((((((value_idx + i < head_size) {${$1}
      
      // Find) { an) { an: any;
      var max_abs) { any) { any) { any) { any) { any) { any = 0) { a: an: any;;
      for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {${$1}
      
      // Calcula: any;
      let max_representable: any: any: any: any: any: any = f: an: any;;
      let scale: any: any: any: any: any: any = max_: any;
      
      // Sto: any;
      scales[scale_position] = f: an: any;
      
      // Pa: any;
      var packed_value: u32: any: any: any: any: any: any = 0: a: an: any;
      
      if (((((((precision_bits == 2u) {
        var scales_array) { array) {any;
        scales_array[0] = scal) { an) { an) { an: any;
        scales_array[1] = sc) { an: any;
        scales_array[2] = s: any;
        scales_array[3] = s: any;}
        v: any;
        for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {${$1}
        
        let packed: any: any: any: any: any: any = pack_to_2: any;;
        packed_value: any: any: any: any: any: any = pac: any;
      } else if (((((((precision_bits == 4u) {
        var scales_array) { array) {any;
        scales_array[0] = scal) { an) { an) { an: any;
        scales_array[1] = sc) { an: any;}
        v: any;
        for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i: a: an: any; i += 1u) {${$1}
        
        let packed: any: any: any: any: any: any = pack_to_4: any;;
        packed_value: any: any: any: any: any: any = pac: any;
      } else {${$1}
      
      // Sto: any;
      kv_cache[packed_idx] = packed_v: any;
      
      // Sto: any;
      let metadata_idx: any: any: any: any: any: any: any: any: any: any = (;
        (effective_position * batch_s: any;
      cache_metadata[metadata_idx] = precision_: any;
    }
  
  // Ma: any;
  @compute @workgroup_size(${$1}, ${$1}, ${$1}) {
  f: an: any;
    @builtin(global_invocation_id) { any) global_id) { ve: any;
  ) {
    let batch_idx) {any: any: any: any: any: any = global: any;
    let head_idx: any: any: any: any: any: any = global: any;
    let position: any: any: any: any: any: any = global: any;}
    let batch_size: any: any: any: any: any: any = unifo: any;
    let max_seq_length: any: any: any: any: any: any = unifo: any;
    let current_length: any: any: any: any: any: any = unifo: any;
    let num_heads: any: any: any: any: any: any = unifo: any;
    let head_size: any: any: any: any: any: any = unifo: any;
    let sliding_window: any: any: any: any: any: any = unifo: any;
    let window_start: any: any: any: any: any: any = unifo: any;
    
    if (((((((batch_idx >= batch_size || head_idx >= num_heads || position >= current_length) {${$1}
    
    // Check) { an) { an: any;
    if (((${$1} && sliding_window > 0u) {
      let window_end) { any) { any) { any) { any) { any) { any = window_st: any;
      if (((((((position < window_start || position >= window_end) {${$1}
      // Map) { an) { an: any;
      position) { any) {any) { any) { any: any: any = posit: any;}
    
    // G: any;
    let metadata_idx) { any) { any) { any: any: any: any: any: any: any: any = (;
      (position * batch_s: any;
    let precision_bits: any: any: any: any: any: any = cache_metad: any;
    
    // Calcula: any;
    let values_per_u32: any: any: any: any: any: any = 3: an: any;
    
    // Proce: any;
    for (((((((var value_idx) { any) { any) { any) { any) { any) { any = 0: a: an: any; value_: any; value_idx += 4u) {
      // Calcula: any;
      let cache_position) { any) {any) { any: any: any: any: any: any: any: any = (;;
        (position * batch_s: any;}
      // Positi: any;
      let packed_idx: any: any: any: any: any: any = cache_posit: any;
      
      // G: any;
      let scale_position) { any) { any) { any: any: any: any: any: any: any: any = (;
        (position * batch_s: any;
      let scale: any: any: any: any: any: any = f: an: any;
      
      // Re: any;
      let packed_value: any: any: any: any: any: any = kv_ca: any;
      
      // Unpa: any;
      v: any;
      let indices: any: any: any: any: any: any = ar: any;
      
      if (((((((precision_bits == 2u) {${$1} else if (precision_bits == 4u) {${$1} else {${$1}
      
      // Use) { an) { an: any;
      // Thi) { an: any;
      // F: any;
      for ((((var i) { any) { any) { any) { any) { any) { any) { any = 0) { a) { an: any; i: a: an: any; i += 1u) {
        if (((((((value_idx + i < head_size) {
          // This) { an) { an: any;
          // Fo) { an: any;
          if ((((i == 0u) {${$1} */;
      }
  
  return) { an) { an: any;

functio) { an: any;
  $1) { any)) { any { number: any: any: any = 4: a: any;;
  $1: $2 | null: any: any: any = nu: any;
  $1: number: any: any: any = 1: any;
  $1: string: any: any: any: any: any: any = "silu",;"
  $1: boolean: any: any: any = t: any;
) -> s: an: any;
  /** Genera: any;
  
  A: any;
    b: any;
    browser) { Targ: any;
    block_size) { Blo: any;
    activation_fn) { Activation function (silu) { a: any;
    adaptive_precision) { Enab: any;
    
  Retu: any;
    WG: any;
  // G: any;
  if ((((((($1) {
    browser_info) { any) { any) { any) { any = detect_browser_environmen) { an: any;
    browser: any: any = (browser_info["browser"] !== undefined ? browser_info["browser"] : ) if ((((((browser_info["detected"] !== undefined ? browser_info["detected"] ) { ) else {null;}"
  workgroup_size) { any) { any) { any = get_workgroup_confi) { an: any;
  feature_support: any: any = get_feature_suppo: any;
  
  // Adju: any;
  use_shared_memory: any: any: any = feature_suppo: any;
  
  // Adju: any;
  workgroup_x: any: any: any = workgroup_si: any;
  workgroup_y: any: any: any = workgroup_si: any;
  workgroup_z: any: any = (workgroup_size["z"] !== undefin: any;"
  
  // Crea: any;
  if (((((($1) {
    activation_code) { any) { any) { any = "fn silu(x) { any)) { any { f32) -> f32 ${$1}";"
    apply_activation: any: any: any: any: any: any = "silu";"
  else if (((((((($1) {
    activation_code) { any) { any) { any = "fn gelu(x) { any)) { any { f32) -> f32 ${$1}";"
    apply_activation: any: any: any: any: any: any = "gelu";"
  } else {// relu}
    activation_code: any: any = "fn relu(x: any): any { f32) -> f32 ${$1}";"
    apply_activation: any: any: any: any: any: any = "relu";"
  
  }
  // Shad: any;
  shader: any: any: any: any: any: any = `$1`;
  // WebG: any;
  // Configuration: ${$1}-bit, block_size: any: any: any: any: any: any: any = ${$1}, activation: any: any: any: any = ${$1}
  // ${$1}
  // Optimized for ((((((${$1} browse) { an) { an: any;
  
  struct Uniforms {${$1};
  
  @group(0) { any) { @binding(0) { any) var<uniform> uniforms) { Uniform) { a: an: any;
  @group(0: a: any;         // [batch_size, seq_len: any;
  @group(0: a: any;  // Pack: any;
  @group(0: a: any;    // Pack: any;
  @group(0: a: any;  // Pack: any;
  @group(0: a: any;   // Ga: any;
  @group(0: a: any;     // U: an: any;
  @group(0: a: any;   // Do: any;
  @group(0: a: any;  // [batch_size, seq_len: any;
  
  ${$1}
  
  fn unpack_${$1}bit(packed_value: u32, idx: u32) -> u32 {
    let bits_per_value: any: any: any: any: any: any: any: any: any: any: any = ${$1}u;
    let mask: any: any: any: any: any: any = (1u << bits_per_va: any;
    ret: any;
  }
  
  fn dequantize_${$1}bit(quantized: u32, scale: f16) -> f16 {${$1}
  
  @compute @workgroup_size(${$1}, ${$1}, ${$1});
  f: an: any;
    @builtin(global_invocation_id: a: any;
    @builtin(workgroup_id: a: any;
    @builtin(local_invocation_id: a: any;
  ) {let batch_idx: any: any: any: any: any: any = global: any;
    let seq_idx: any: any: any: any: any: any = global: any;
    let hidden_idx: any: any: any: any: any: any = global: any;}
    let batch_size: any: any: any: any: any: any = unifo: any;
    let seq_length: any: any: any: any: any: any = unifo: any;
    let hidden_size: any: any: any: any: any: any = unifo: any;
    let intermediate_size: any: any: any: any: any: any = unifo: any;
    let block_size: any: any: any: any: any: any = unifo: any;
    let calibrated_scales: any: any: any: any: any: any = unifo: any;
    
    if (((((((batch_idx >= batch_size || seq_idx >= seq_length || hidden_idx >= hidden_size) {${$1}
    
    // Input) { an) { an: any;
    let input_offset) { any) { any) { any) { any: any: any = (batch_idx * seq_len: any;
    
    // Calcula: any;
    let elements_per_u32: any: any: any: any: any: any: any: any: any: any: any = 32u / ${$1}u;
    
    /** // A: any;
  if (((($1) {
    shader += `$1`;
    // Shared) { an) { an: any;
    var<workgroup> shared_gate) { array<f16, ${$1} * ${$1}>;;
    var<workgroup> shared_up) { array<f16, ${$1} * ${$1}>;
    
  }
    // Collaborativ) { an: any;
    for (((((var i) { any) { any) { any) { any) { any) { any = 0) { a) { an: any; i: a: an: any; i += ${$1}) {
      let idx: any: any: any: any: any: any = local_id.y * ${$1} + local: any;;
      if (((((((idx + i < hidden_size) {${$1}
    
    workgroupBarrier) { an) { an) { an: any; */;
  
  // Continu) { an: any;
  shader += `$1`;
    // Compu: any;
    var gate_activations) { array<f16, ${$1}>;;
    var up_activations) { array<f16, ${$1}>;
    
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i < ${$1}; i += 1u) {${$1}
    
    // Fir: any;
    for (((((((var in_idx) { any) { any) { any) { any) { any) { any = 0: a: an: any;; in_: any; in_idx += 1u) {
      ${$1}
      for (((((((var out_idx) { any) { any) { any) { any) { any) { any = 0: a: an: any;; out_idx < min(${$1}u, intermediate_s: any; out_idx += 1u): any {
        // Ga: any;
        {
          let weight_offset: any: any: any: any: any: any = in_: any;;
          let packed_idx: any: any: any: any: any: any = weight_off: any;
          let bit_offset: any: any: any: any: any: any = weight_off: any;
          let packed_weight: any: any: any: any: any: any = gate_weig: any;
          let quantized: any: any: any: any: any: any = unpack_${$1}bit(packed_weight: a: any;
          
        }
          let block_idx: any: any: any: any: any: any = (in_idx / block_s: any;
          let scale: any: any: any: any: any: any = gate_sca: any;
          
      }
          let weight_val: any: any: any: any: any: any: any = dequantize_${$1}bit(quantized: any, ${$1});
          gate_activations[out_idx] += in_: any;
        }
        
        // U: an: any;
        {
          let weight_offset: any: any: any: any: any: any = in_: any;
          let packed_idx: any: any: any: any: any: any = weight_off: any;
          let bit_offset: any: any: any: any: any: any = weight_off: any;
          let packed_weight: any: any: any: any: any: any = up_weig: any;
          let quantized: any: any: any: any: any: any = unpack_${$1}bit(packed_weight: a: any;
          
        }
          let block_idx: any: any: any: any: any: any = (in_idx / block_s: any;
          let scale: any: any: any: any: any: any = up_sca: any;
          
          let weight_val: any: any: any: any: any: any: any = dequantize_${$1}bit(quantized: any, ${$1});
          up_activations[out_idx] += in_: any;
        }
    
    // Compu: any;
    var intermediate_activations: array<f16, ${$1}>;
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i < min(${$1}u, intermediate_s: any; i += 1u): any {
      let gate_val: any: any: any: any: any: any: any: any: any: any: any = ${$1}(f32(gate_activations[i]));;
      intermediate_activations[i] = f: an: any;
    }
    
    // Seco: any;
    var result: any: any: any: any: any: any = 0: a: an: any;
    
    for (((((((var i) { any) { any) { any) { any) { any) { any = 0: a: an: any; i < min(${$1}u, intermediate_s: any; i += 1u): any {
      let weight_offset: any: any: any: any: any: any = i: a: an: any;;
      let packed_idx: any: any: any: any: any: any = weight_off: any;
      let bit_offset: any: any: any: any: any: any = weight_off: any;
      let packed_weight: any: any: any: any: any: any = down_weig: any;
      let quantized: any: any: any: any: any: any = unpack_${$1}bit(packed_weight: a: any;
      
    }
      let block_idx: any: any: any: any: any: any = (i / block_s: any;
      let scale: any: any: any: any: any: any = down_sca: any;
      
      let weight_val: any: any: any: any: any: any: any = dequantize_${$1}bit(quantized: any, ${$1});
      result += f: an: any;;
    }
    
    // Wri: any;
    let output_offset: any: any: any: any: any: any = (batch_idx * seq_len: any;
    output[output_offset] = f: an: any;
  }
  /** retu: any;

functi: any;
  $1: stri: any;
  $1: number: any: any: any = 4: a: any;
  $1: $2 | null: any: any: any = nu: any;
  $1: boolean: any: any: any = tr: any;
  $1: string: any: any: any: any: any: any = "matmul",;"
  config: Record<str, Any | null> = n: any;
): a: any;
  Genera: any;
  ;
  Args) {
    operation) { Operation type (matmul) { a: any;
    b: any;
    brow: any;
    adaptive_precis: any;
    layer_t: any;
    con: any;
    
  Retu: any;
    WG: any;
  /** if ((((((($1) {
    config) { any) { any) { any) { any = {}
  if ((((($1) {
    return) { an) { an: any;
      bits) { any)) { any { any: any: any = bi: any;
      browser: any: any: any = brows: any;
      use_shared_memory: any: any = (config["use_shared_memory"] !== undefin: any;"
      workgroup_size: any: any = (config["workgroup_size"] !== undefin: any;"
      block_size: any: any = (config["block_size"] !== undefin: any;"
      per_channel: any: any = (config["per_channel"] !== undefin: any;"
      symmetric: any: any = (config["symmetric"] !== undefin: any;"
    );
  else if ((((((($1) {
    return) { an) { an: any;
      bits) { any)) { any {any = bi: any;
      browser: any: any: any = brows: any;
      block_size: any: any = (config["block_size"] !== undefin: any;"
      use_flash_attention: any: any = (config["use_flash_attention"] !== undefin: any;"
      causal_mask: any: any = (config["causal_mask"] !== undefin: any;"
      adaptive_precision: any: any: any = adaptive_precis: any;
    );} else if ((((((($1) {
    return) { an) { an: any;
      kv_cache_bits) { any)) { any { any: any: any = bi: any;
      browser) {any = brows: any;
      enable_variable_precision: any: any: any = adaptive_precisi: any;
      enable_sliding_window: any: any = (config["enable_sliding_window"] !== undefin: any;"
      window_size: any: any = (config["window_size"] !== undefin: any;"
    );} else if ((((((($1) { ${$1} else {throw new) { an) { an: any;
  }
  $1) { any)) { any {string}
  $1) {$2 | null) { any) { any: any = nu: any;}
  config: Record<str, Any | null> = n: any;
) -> Di: any;
  G: any;
  
  A: any;
    shader_t: any;
    brow: any;
    con: any;
    
  Retu: any;
    Dictiona: any;
  """;"
  if ((((((($1) {
    config) { any) { any) { any) { any = {}
  // Ge) { an: any;
  if (((((($1) {
    browser_info) { any) { any) { any) { any = detect_browser_environmen) { an: any;
    browser: any: any = (browser_info["browser"] !== undefined ? browser_info["browser"] : ) if ((((((browser_info["detected"] !== undefined ? browser_info["detected"] ) { ) else {null;}"
  // Get) { an) { an: any;
  feature_support) { any) { any = get_feature_suppo: any;
  
  // G: any;
  operation: any: any = "matmul" if (((((shader_type) { any) { any) { any) { any = = "mlp" else { shader_ty) { an: any;"
  workgroup_config: any: any = get_workgroup_conf: any;
  
  // S: any;
  default_config: any: any: any = ${$1}
  
  // Overri: any;
  shader_config: any: any: any = ${$1}
  
  // Genera: any;
  shader_code: any: any: any = generate_compute_shad: any;
    operation: any: any: any = shader_ty: any;
    bits: any: any: any = shader_conf: any;
    browser: any: any: any = brows: any;
    adaptive_precision: any: any: any = shader_conf: any;
    layer_type: any: any: any = shader_ty: any;
    config: any: any: any = shader_con: any;
  );
  ;
  return ${$1}

if (((((($1) {// Example) { an) { an: any;
  consol) { an: any;
  conso: any;
  browser) { any) { any: any: any: any: any = "chrome"  // || "firefox", "edge", "safari";"
  
  conso: any;
  shader: any: any = matmul_4bit_shader(bits=4, browser: any: any = browser, use_shared_memory: any: any: any = tr: any;
  conso: any;
  
  conso: any;
  shader: any: any = attention_with_adaptive_precision_shader(bits=4, browser: any: any: any = brows: any;
  conso: any;
  
  conso: any;
  shader: any: any = kv_cache_adaptive_precision_shader(kv_cache_bits=4, browser: any: any: any = brows: any;
  conso: any;
  
  conso: any;
  shader: any: any = mlp_with_adaptive_precision_shader(bits=4, browser: any: any: any = brows: any;
  conso: any;
  
  conso: any;
  for ((((((browser_name in ["chrome", "edge", "firefox", "safari"]) {"
    features) { any) { any) { any = get_feature_support) { an) { an: any;
    conso: any;
  
  conso: any;
  for (((((browser_name in ["chrome", "edge", "firefox", "safari"]) {"
    for (operation in ["matmul", "attention", "kv_cache"]) {"
      config) { any) { any) { any = get_workgroup_config) { an) { an: any;
      conso: any;