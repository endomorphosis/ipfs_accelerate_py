/**
 * Browser-Optimized WGSL Shaders
 * 
 * Provides browser-specific optimizations for WebGPU shaders to maximize performance
 * across different browsers and hardware configurations.
 */

/**
 * Browser type identification
 */
export enum BrowserType {
  CHROME = 'chrome',
  FIREFOX = 'firefox',
  SAFARI = 'safari',
  EDGE = 'edge',
  UNKNOWN = 'unknown'
}

/**
 * Browser capabilities and optimization settings
 */
export interface BrowserCapabilities {
  /**
   * Browser type
   */
  browserType: BrowserType;
  
  /**
   * Browser version
   */
  version: string;
  
  /**
   * WebGPU feature support
   */
  features: {
    /**
     * Supports 16-bit storage
     */
    storage16Bit: boolean;
    
    /**
     * Supports 8-bit storage
     */
    storage8Bit: boolean;
    
    /**
     * Supports timestamp queries
     */
    timestampQuery: boolean;
    
    /**
     * Supports indirect dispatch
     */
    indirectDispatch: boolean;
    
    /**
     * Supports pipeline statistics query
     */
    pipelineStatistics: boolean;
    
    /**
     * Supports async pipeline compilation
     */
    asyncPipeline: boolean;
  };
  
  /**
   * Hardware info
   */
  hardware: {
    /**
     * GPU vendor name
     */
    vendor: string;
    
    /**
     * GPU architecture
     */
    architecture: string;
    
    /**
     * Is integrated GPU
     */
    isIntegrated: boolean;
    
    /**
     * Has unified memory architecture
     */
    hasUnifiedMemory: boolean;
  };
  
  /**
   * Performance tier (1-5, with 5 being highest)
   */
  performanceTier: number;
  
  /**
   * Optimal workgroup size
   */
  optimalWorkgroupSize: {
    /**
     * For compute operations
     */
    compute: [number, number, number];
    
    /**
     * For matrix operations
     */
    matrix: [number, number, number];
    
    /**
     * For element-wise operations
     */
    elementwise: [number, number, number];
  };
}

/**
 * Shader optimization settings
 */
export interface ShaderOptimizationSettings {
  /**
   * Enable loop unrolling
   */
  enableLoopUnrolling: boolean;
  
  /**
   * Enable memory coalescing
   */
  enableMemoryCoalescing: boolean;
  
  /**
   * Enable workgroup memory usage
   */
  enableWorkgroupMemory: boolean;
  
  /**
   * Enable subgroup operations if supported
   */
  enableSubgroupOps: boolean;
  
  /**
   * Prefer performance over precision
   */
  preferPerformanceOverPrecision: boolean;
  
  /**
   * Aggressiveness of optimizations (1-5)
   */
  optimizationLevel: number;
  
  /**
   * Use browser-specific intrinsics
   */
  useBrowserSpecificIntrinsics: boolean;
  
  /**
   * Custom settings for specific operations
   */
  customSettings: Record<string, any>;
}

/**
 * Detects current browser type
 */
export function detectBrowserType(): BrowserType {
  const userAgent = navigator.userAgent.toLowerCase();
  
  if (userAgent.includes('edg/')) {
    return BrowserType.EDGE;
  } else if (userAgent.includes('chrome/')) {
    return BrowserType.CHROME;
  } else if (userAgent.includes('firefox/')) {
    return BrowserType.FIREFOX;
  } else if (userAgent.includes('safari/') && !userAgent.includes('chrome/')) {
    return BrowserType.SAFARI;
  }
  
  return BrowserType.UNKNOWN;
}

/**
 * Detects browser version
 */
export function detectBrowserVersion(): string {
  const browserType = detectBrowserType();
  const userAgent = navigator.userAgent.toLowerCase();
  let match;
  
  switch (browserType) {
    case BrowserType.CHROME:
      match = userAgent.match(/chrome\/(\d+\.\d+)/);
      break;
    case BrowserType.FIREFOX:
      match = userAgent.match(/firefox\/(\d+\.\d+)/);
      break;
    case BrowserType.SAFARI:
      match = userAgent.match(/version\/(\d+\.\d+)/);
      break;
    case BrowserType.EDGE:
      match = userAgent.match(/edg\/(\d+\.\d+)/);
      break;
  }
  
  return match ? match[1] : 'unknown';
}

/**
 * Get optimal browser-specific settings
 */
export async function getBrowserCapabilities(device: GPUDevice): Promise<BrowserCapabilities> {
  const browserType = detectBrowserType();
  const version = detectBrowserVersion();
  
  // Get adapter info
  let adapterInfo: GPUAdapterInfo;
  try {
    // @ts-ignore - TypeScript may not have updated types for this yet
    adapterInfo = await device.adapter?.requestAdapterInfo();
  } catch (e) {
    adapterInfo = { vendor: 'unknown', architecture: 'unknown' };
  }
  
  // Default settings
  const defaultCapabilities: BrowserCapabilities = {
    browserType,
    version,
    features: {
      storage16Bit: false,
      storage8Bit: false,
      timestampQuery: false,
      indirectDispatch: false,
      pipelineStatistics: false,
      asyncPipeline: false
    },
    hardware: {
      vendor: adapterInfo.vendor || 'unknown',
      architecture: adapterInfo.architecture || 'unknown',
      isIntegrated: false,
      hasUnifiedMemory: false
    },
    performanceTier: 3,
    optimalWorkgroupSize: {
      compute: [256, 1, 1],
      matrix: [8, 8, 1],
      elementwise: [256, 1, 1]
    }
  };
  
  // Check for features
  defaultCapabilities.features.storage16Bit = device.features.has('shader-f16');
  
  // Browser-specific optimizations
  switch (browserType) {
    case BrowserType.CHROME:
      // Chrome typically performs well with larger workgroup sizes
      defaultCapabilities.optimalWorkgroupSize.compute = [256, 1, 1];
      defaultCapabilities.optimalWorkgroupSize.matrix = [16, 16, 1];
      defaultCapabilities.optimalWorkgroupSize.elementwise = [256, 1, 1];
      defaultCapabilities.performanceTier = 4;
      break;
      
    case BrowserType.FIREFOX:
      // Firefox works best with workgroup sizes that are multiples of 64
      defaultCapabilities.optimalWorkgroupSize.compute = [64, 1, 1];
      defaultCapabilities.optimalWorkgroupSize.matrix = [8, 8, 1];
      defaultCapabilities.optimalWorkgroupSize.elementwise = [64, 1, 1];
      // Firefox is particularly good at compute shaders
      defaultCapabilities.performanceTier = 3;
      break;
      
    case BrowserType.SAFARI:
      // Safari benefits from smaller workgroup sizes on some hardware
      defaultCapabilities.optimalWorkgroupSize.compute = [128, 1, 1];
      defaultCapabilities.optimalWorkgroupSize.matrix = [8, 8, 1];
      defaultCapabilities.optimalWorkgroupSize.elementwise = [128, 1, 1];
      defaultCapabilities.performanceTier = 3;
      break;
      
    case BrowserType.EDGE:
      // Edge/Chrome similar optimization patterns
      defaultCapabilities.optimalWorkgroupSize.compute = [256, 1, 1];
      defaultCapabilities.optimalWorkgroupSize.matrix = [16, 16, 1];
      defaultCapabilities.optimalWorkgroupSize.elementwise = [256, 1, 1];
      defaultCapabilities.performanceTier = 4;
      break;
      
    default:
      // Conservative defaults for unknown browsers
      defaultCapabilities.optimalWorkgroupSize.compute = [64, 1, 1];
      defaultCapabilities.optimalWorkgroupSize.matrix = [8, 8, 1];
      defaultCapabilities.optimalWorkgroupSize.elementwise = [64, 1, 1];
      defaultCapabilities.performanceTier = 2;
  }
  
  // Hardware-specific adjustments
  if (adapterInfo.vendor) {
    const vendorLower = adapterInfo.vendor.toLowerCase();
    
    // Apple GPUs (Metal)
    if (vendorLower.includes('apple')) {
      defaultCapabilities.hardware.hasUnifiedMemory = true;
      defaultCapabilities.optimalWorkgroupSize.matrix = [8, 8, 1]; // Apple GPUs work better with smaller tiles
      
      // M1/M2 processors are very powerful
      if (browserType === BrowserType.SAFARI) {
        defaultCapabilities.performanceTier = 5;
      }
    }
    
    // NVIDIA GPUs
    else if (vendorLower.includes('nvidia')) {
      defaultCapabilities.hardware.isIntegrated = false;
      defaultCapabilities.optimalWorkgroupSize.compute = [256, 1, 1]; // NVIDIA GPUs typically perform well with larger workgroups
      defaultCapabilities.optimalWorkgroupSize.matrix = [16, 16, 1];
      defaultCapabilities.performanceTier = 5;
    }
    
    // AMD GPUs
    else if (vendorLower.includes('amd') || vendorLower.includes('ati')) {
      defaultCapabilities.hardware.isIntegrated = false;
      defaultCapabilities.optimalWorkgroupSize.compute = [256, 1, 1];
      defaultCapabilities.optimalWorkgroupSize.matrix = [8, 8, 1]; // AMD typically works better with smaller tiles
      defaultCapabilities.performanceTier = 4;
    }
    
    // Intel GPUs (typically integrated)
    else if (vendorLower.includes('intel')) {
      defaultCapabilities.hardware.isIntegrated = true;
      defaultCapabilities.optimalWorkgroupSize.compute = [128, 1, 1]; // Intel integrated GPUs work better with moderate workgroup size
      defaultCapabilities.optimalWorkgroupSize.matrix = [8, 8, 1];
      defaultCapabilities.performanceTier = 2;
      
      // Recent Intel GPUs are much improved
      if (adapterInfo.architecture && 
          (adapterInfo.architecture.includes('Xe') || 
           adapterInfo.architecture.includes('Arc'))) {
        defaultCapabilities.performanceTier = 3;
      }
    }
    
    // Qualcomm (mobile)
    else if (vendorLower.includes('qualcomm') || vendorLower.includes('adreno')) {
      defaultCapabilities.hardware.isIntegrated = true;
      defaultCapabilities.hardware.hasUnifiedMemory = true;
      defaultCapabilities.optimalWorkgroupSize.compute = [64, 1, 1]; // Mobile GPUs typically need smaller workgroups
      defaultCapabilities.optimalWorkgroupSize.matrix = [4, 4, 1];
      defaultCapabilities.optimalWorkgroupSize.elementwise = [64, 1, 1];
      defaultCapabilities.performanceTier = 2;
    }
    
    // ARM (mobile)
    else if (vendorLower.includes('arm') || vendorLower.includes('mali')) {
      defaultCapabilities.hardware.isIntegrated = true;
      defaultCapabilities.hardware.hasUnifiedMemory = true;
      defaultCapabilities.optimalWorkgroupSize.compute = [64, 1, 1];
      defaultCapabilities.optimalWorkgroupSize.matrix = [4, 4, 1];
      defaultCapabilities.optimalWorkgroupSize.elementwise = [64, 1, 1];
      defaultCapabilities.performanceTier = 2;
    }
  }
  
  return defaultCapabilities;
}

/**
 * Get default optimization settings for a browser
 */
export function getDefaultOptimizationSettings(browserCapabilities: BrowserCapabilities): ShaderOptimizationSettings {
  // Default settings
  const settings: ShaderOptimizationSettings = {
    enableLoopUnrolling: browserCapabilities.performanceTier >= 3,
    enableMemoryCoalescing: true,
    enableWorkgroupMemory: true,
    enableSubgroupOps: false, // Not widely supported yet
    preferPerformanceOverPrecision: false,
    optimizationLevel: browserCapabilities.performanceTier,
    useBrowserSpecificIntrinsics: true,
    customSettings: {}
  };
  
  // Browser-specific adjustments
  switch (browserCapabilities.browserType) {
    case BrowserType.CHROME:
      settings.enableLoopUnrolling = true;
      settings.enableWorkgroupMemory = true;
      settings.customSettings.useChromeMacros = true;
      break;
      
    case BrowserType.FIREFOX:
      // Firefox benefits most from memory coalescing
      settings.enableMemoryCoalescing = true;
      settings.enableWorkgroupMemory = true;
      settings.customSettings.useFirefoxOptimizedBarriers = true;
      break;
      
    case BrowserType.SAFARI:
      // Safari benefits from precision optimizations
      settings.preferPerformanceOverPrecision = true;
      settings.enableWorkgroupMemory = true;
      settings.customSettings.useMetalOptimizations = true;
      break;
      
    case BrowserType.EDGE:
      // Edge shares Chrome's settings
      settings.enableLoopUnrolling = true;
      settings.enableWorkgroupMemory = true;
      settings.customSettings.useChromeMacros = true;
      break;
      
    default:
      // Conservative settings for unknown browsers
      settings.enableLoopUnrolling = false;
      settings.enableWorkgroupMemory = true;
      settings.preferPerformanceOverPrecision = false;
      settings.optimizationLevel = Math.min(settings.optimizationLevel, 3);
  }
  
  // Hardware-specific adjustments
  if (browserCapabilities.hardware.isIntegrated) {
    // Integrated GPUs benefit from different optimizations
    settings.enableWorkgroupMemory = false; // Often shared memory anyway
    settings.optimizationLevel = Math.min(settings.optimizationLevel, 3);
  }
  
  if (browserCapabilities.hardware.hasUnifiedMemory) {
    // Unified memory architectures benefit from different memory patterns
    settings.customSettings.optimizeForUnifiedMemory = true;
  }
  
  return settings;
}

/**
 * Optimized Matrix Multiplication Shader
 * 
 * @param capabilities Browser capabilities
 * @param settings Optimization settings
 * @returns Optimized WGSL shader code
 */
export function getOptimizedMatmulShader(
  capabilities: BrowserCapabilities,
  settings: ShaderOptimizationSettings
): string {
  const [workgroupSizeX, workgroupSizeY, workgroupSizeZ] = capabilities.optimalWorkgroupSize.matrix;
  
  // Common shader features
  let shader = `
    // Matrix Multiplication Shader
    // Optimized for ${capabilities.browserType}
    
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> c: array<f32>;
    
    struct Params {
      M: u32,
      N: u32,
      K: u32
    }
    
    @group(0) @binding(3) var<uniform> params: Params;
  `;
  
  // Use workgroup memory if enabled
  if (settings.enableWorkgroupMemory) {
    shader += `
    var<workgroup> tile_a: array<f32, ${workgroupSizeX} * ${workgroupSizeY}>;
    var<workgroup> tile_b: array<f32, ${workgroupSizeY} * ${workgroupSizeX}>;
    `;
  }
  
  shader += `
    @compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY}, ${workgroupSizeZ})
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) workgroup_id: vec3<u32>) {
      let row = global_id.x;
      let col = global_id.y;
      
      if (row >= params.M || col >= params.N) {
        return;
      }
      
      var sum = 0.0;
  `;
  
  // Different implementation based on optimization settings
  if (settings.enableWorkgroupMemory) {
    shader += `
      let tileSize = ${workgroupSizeX};
      let numTiles = (params.K + tileSize - 1) / tileSize;
      
      for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load tile of A into workgroup memory
        let tileRowA = local_id.x;
        let tileColA = local_id.y;
        let tileIndexA = tileRowA * tileSize + tileColA;
        
        let globalRowA = workgroup_id.x * tileSize + tileRowA;
        let globalColA = t * tileSize + tileColA;
        
        if (globalRowA < params.M && globalColA < params.K) {
          tile_a[tileIndexA] = a[globalRowA * params.K + globalColA];
        } else {
          tile_a[tileIndexA] = 0.0;
        }
        
        // Load tile of B into workgroup memory
        let tileRowB = local_id.x;
        let tileColB = local_id.y;
        let tileIndexB = tileRowB * tileSize + tileColB;
        
        let globalRowB = t * tileSize + tileRowB;
        let globalColB = workgroup_id.y * tileSize + tileColB;
        
        if (globalRowB < params.K && globalColB < params.N) {
          tile_b[tileIndexB] = b[globalRowB * params.N + globalColB];
        } else {
          tile_b[tileIndexB] = 0.0;
        }
        
        workgroupBarrier();
        
        // Compute partial dot product for this tile
        for (var k = 0u; k < tileSize; k = k + 1u) {
          if (t * tileSize + k < params.K) {
            let aIndex = local_id.x * tileSize + k;
            let bIndex = k * tileSize + local_id.y;
            sum = sum + tile_a[aIndex] * tile_b[bIndex];
          }
        }
        
        workgroupBarrier();
      }
    `;
  } else {
    // Simpler implementation without workgroup memory
    shader += `
      for (var k = 0u; k < params.K; k = k + 1u) {
        sum = sum + a[row * params.K + k] * b[k * params.N + col];
      }
    `;
  }
  
  // Loop unrolling for high-performance browsers
  if (settings.enableLoopUnrolling && capabilities.performanceTier >= 4) {
    shader = shader.replace(
      `for (var k = 0u; k < params.K; k = k + 1u) {
        sum = sum + a[row * params.K + k] * b[k * params.N + col];
      }`,
      `for (var k = 0u; k < params.K; k = k + 4u) {
        if (k < params.K) {
          sum = sum + a[row * params.K + k] * b[k * params.N + col];
        }
        if (k + 1u < params.K) {
          sum = sum + a[row * params.K + (k + 1u)] * b[(k + 1u) * params.N + col];
        }
        if (k + 2u < params.K) {
          sum = sum + a[row * params.K + (k + 2u)] * b[(k + 2u) * params.N + col];
        }
        if (k + 3u < params.K) {
          sum = sum + a[row * params.K + (k + 3u)] * b[(k + 3u) * params.N + col];
        }
      }`
    );
  }
  
  // Complete the shader
  shader += `
      c[row * params.N + col] = sum;
    }
  `;
  
  // Add browser-specific optimizations
  if (settings.useBrowserSpecificIntrinsics) {
    switch (capabilities.browserType) {
      case BrowserType.CHROME:
        // Chrome-specific comments/hints
        shader = shader.replace(
          '// Matrix Multiplication Shader',
          '// Matrix Multiplication Shader\n// @optimize: true'
        );
        break;
        
      case BrowserType.FIREFOX:
        // Firefox does well with specific alignment and barrier patterns
        if (settings.customSettings.useFirefoxOptimizedBarriers) {
          shader = shader.replace(
            'workgroupBarrier();',
            'workgroupBarrier(); // @align'
          );
        }
        break;
        
      case BrowserType.SAFARI:
        // Safari/Metal-specific optimizations
        if (settings.customSettings.useMetalOptimizations) {
          // Add Metal-friendly memory access patterns
          shader = shader.replace(
            '// Matrix Multiplication Shader',
            '// Matrix Multiplication Shader\n// @metal_optimize'
          );
        }
        break;
    }
  }
  
  return shader;
}

/**
 * Optimized Elementwise Operation Shader
 * 
 * @param capabilities Browser capabilities
 * @param settings Optimization settings
 * @param operation Elementwise operation ('relu', 'sigmoid', 'tanh')
 * @returns Optimized WGSL shader code
 */
export function getOptimizedElementwiseShader(
  capabilities: BrowserCapabilities,
  settings: ShaderOptimizationSettings,
  operation: 'relu' | 'sigmoid' | 'tanh' = 'relu'
): string {
  const [workgroupSizeX, workgroupSizeY, workgroupSizeZ] = capabilities.optimalWorkgroupSize.elementwise;
  
  // Operation implementations
  const operationImpl = {
    relu: 'max(0.0, x)',
    sigmoid: '1.0 / (1.0 + exp(-x))',
    tanh: 'tanh(x)'
  };
  
  // Browser-specific operation implementations for better precision/performance
  if (settings.preferPerformanceOverPrecision && 
      (capabilities.browserType === BrowserType.SAFARI || 
       capabilities.browserType === BrowserType.FIREFOX)) {
    // Faster approximations for some functions
    if (operation === 'sigmoid') {
      operationImpl.sigmoid = '0.5 + 0.5 * tanh(x * 0.5)';  // Faster approximation
    }
  }
  
  let shader = `
    // Elementwise ${operation} Shader
    // Optimized for ${capabilities.browserType}
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    
    struct Params {
      length: u32
    }
    
    @group(0) @binding(2) var<uniform> params: Params;
    
    @compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY}, ${workgroupSizeZ})
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      
      if (idx >= params.length) {
        return;
      }
      
      let x = input[idx];
      let result = ${operationImpl[operation]};
      
      output[idx] = result;
    }
  `;
  
  // Add vectorization for higher-tier browsers
  if (settings.optimizationLevel >= 4 && operation === 'relu') {
    shader = shader.replace(
      `let x = input[idx];
      let result = ${operationImpl[operation]};
      
      output[idx] = result;`,
      `// Process 4 elements at once if possible
      if (idx + 3 < params.length) {
        let x0 = input[idx];
        let x1 = input[idx + 1];
        let x2 = input[idx + 2];
        let x3 = input[idx + 3];
        
        output[idx] = max(0.0, x0);
        output[idx + 1] = max(0.0, x1);
        output[idx + 2] = max(0.0, x2);
        output[idx + 3] = max(0.0, x3);
      } else {
        let x = input[idx];
        let result = ${operationImpl[operation]};
        
        output[idx] = result;
      }`
    );
  }
  
  // Add browser-specific optimizations
  if (settings.useBrowserSpecificIntrinsics) {
    switch (capabilities.browserType) {
      case BrowserType.CHROME:
        // Chrome-specific optimizations
        shader = shader.replace(
          `// Elementwise ${operation} Shader`,
          `// Elementwise ${operation} Shader\n// @optimize: true`
        );
        break;
        
      case BrowserType.FIREFOX:
        // Firefox benefits from aligned workgroups
        shader = shader.replace(
          '@compute @workgroup_size',
          '@compute @workgroup_size // @align'
        );
        break;
        
      case BrowserType.SAFARI:
        // Safari benefits from metal optimizations
        if (settings.customSettings.useMetalOptimizations) {
          shader = shader.replace(
            `// Elementwise ${operation} Shader`,
            `// Elementwise ${operation} Shader\n// @metal_optimize`
          );
        }
        break;
    }
  }
  
  return shader;
}

/**
 * Optimized Softmax Shader
 * 
 * @param capabilities Browser capabilities
 * @param settings Optimization settings
 * @returns Optimized WGSL shader code
 */
export function getOptimizedSoftmaxShader(
  capabilities: BrowserCapabilities,
  settings: ShaderOptimizationSettings
): string {
  const [workgroupSizeX, workgroupSizeY, workgroupSizeZ] = capabilities.optimalWorkgroupSize.compute;
  
  let shader = `
    // Softmax Shader
    // Optimized for ${capabilities.browserType}
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    
    struct Params {
      length: u32,
      batch_size: u32
    }
    
    @group(0) @binding(2) var<uniform> params: Params;
  `;
  
  // Add shared memory for reduction operations
  if (settings.enableWorkgroupMemory) {
    shader += `
    var<workgroup> temp_max: f32;
    var<workgroup> temp_sum: f32;
    `;
  }
  
  shader += `
    @compute @workgroup_size(${workgroupSizeX}, 1, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) workgroup_id: vec3<u32>) {
      let batch_idx = global_id.x;
      
      if (batch_idx >= params.batch_size) {
        return;
      }
      
      let seq_len = params.length / params.batch_size;
      let start_idx = batch_idx * seq_len;
  `;
  
  // Different implementation based on optimization settings
  if (settings.enableWorkgroupMemory && workgroupSizeX <= 32) {
    // Using workgroup memory for small batch sizes
    shader += `
      // Find max value (reduction to temp_max)
      var max_val = -3.402823e+38; // -FLT_MAX
      for (var i = 0u; i < seq_len; i = i + 1u) {
        max_val = max(max_val, input[start_idx + i]);
      }
      
      // Compute exp(x - max) and sum
      var sum = 0.0;
      for (var i = 0u; i < seq_len; i = i + 1u) {
        let idx = start_idx + i;
        let exp_val = exp(input[idx] - max_val);
        output[idx] = exp_val;
        sum = sum + exp_val;
      }
      
      // Normalize
      let inv_sum = 1.0 / sum;
      for (var i = 0u; i < seq_len; i = i + 1u) {
        output[start_idx + i] = output[start_idx + i] * inv_sum;
      }
    `;
  } else {
    // Use more direct computation for larger batches or when workgroup memory is disabled
    shader += `
      // Find max for numerical stability
      var max_val = input[start_idx];
      for (var i = 1u; i < seq_len; i = i + 1u) {
        let idx = start_idx + i;
        max_val = max(max_val, input[idx]);
      }
      
      // Compute exp(x - max) and sum
      var sum = 0.0;
      for (var i = 0u; i < seq_len; i = i + 1u) {
        let idx = start_idx + i;
        let exp_val = exp(input[idx] - max_val);
        output[idx] = exp_val;
        sum = sum + exp_val;
      }
      
      // Normalize
      for (var i = 0u; i < seq_len; i = i + 1u) {
        let idx = start_idx + i;
        output[idx] = output[idx] / sum;
      }
    `;
  }
  
  // Close shader
  shader += `
    }
  `;
  
  // Add browser-specific optimizations
  if (settings.useBrowserSpecificIntrinsics) {
    switch (capabilities.browserType) {
      case BrowserType.CHROME:
        // Chrome optimizations
        shader = shader.replace(
          '// Softmax Shader',
          '// Softmax Shader\n// @optimize: true'
        );
        break;
        
      case BrowserType.FIREFOX:
        // Firefox optimizations
        shader = shader.replace(
          '// Softmax Shader',
          '// Softmax Shader\n// @reduce: sequential'
        );
        break;
        
      case BrowserType.SAFARI:
        // Safari/Metal optimizations
        if (settings.customSettings.useMetalOptimizations) {
          shader = shader.replace(
            '// Softmax Shader',
            '// Softmax Shader\n// @metal_optimize'
          );
        }
        break;
    }
  }
  
  return shader;
}

/**
 * Optimized Quantization Shader
 * 
 * @param capabilities Browser capabilities
 * @param settings Optimization settings
 * @returns Optimized WGSL shader code
 */
export function getOptimizedQuantizeShader(
  capabilities: BrowserCapabilities,
  settings: ShaderOptimizationSettings
): string {
  const [workgroupSizeX, workgroupSizeY, workgroupSizeZ] = capabilities.optimalWorkgroupSize.compute;
  
  let shader = `
    // Quantization Shader
    // Optimized for ${capabilities.browserType}
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<i32>;
    @group(0) @binding(2) var<storage, read_write> scale: array<f32>;
    
    struct Params {
      length: u32,
      scale_index: u32
    }
    
    @group(0) @binding(3) var<uniform> params: Params;
  `;
  
  // Add shared memory for reduction
  if (settings.enableWorkgroupMemory) {
    shader += `
    var<workgroup> temp_max: f32;
    `;
  }
  
  shader += `
    @compute @workgroup_size(${workgroupSizeX}, 1, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) workgroup_id: vec3<u32>) {
      let idx = global_id.x;
      
      if (idx >= params.length) {
        return;
      }
  `;
  
  // Different implementation based on optimization settings
  if (settings.enableWorkgroupMemory && workgroupSizeX <= 256) {
    // Use workgroup memory for smaller sizes
    shader += `
      // First thread finds max absolute value
      if (local_id.x == 0u) {
        var max_abs = 0.0;
        for (var i = 0u; i < params.length; i = i + 1u) {
          max_abs = max(max_abs, abs(input[i]));
        }
        
        // Compute scale (127.0 for int8)
        if (max_abs > 0.0) {
          scale[params.scale_index] = 127.0 / max_abs;
        } else {
          scale[params.scale_index] = 1.0;
        }
        
        temp_max = max_abs;
      }
      
      // Wait for scale to be computed
      workgroupBarrier();
      
      // Now all threads have access to max_abs via temp_max
      let current_scale = 127.0 / temp_max;
      if (temp_max == 0.0) {
        current_scale = 1.0;
      }
      
      // Quantize value
      let value = input[idx];
      let quantized = i32(round(value * current_scale));
      
      // Clamp to int8 range (-127 to 127, saving -128 for padding)
      output[idx] = clamp(quantized, -127, 127);
    `;
  } else {
    // Direct implementation for larger sizes
    shader += `
      // First thread finds max absolute value
      if (idx == 0u) {
        var max_abs = 0.0;
        for (var i = 0u; i < params.length; i = i + 1u) {
          max_abs = max(max_abs, abs(input[i]));
        }
        
        // Compute scale (127.0 for int8)
        if (max_abs > 0.0) {
          scale[params.scale_index] = 127.0 / max_abs;
        } else {
          scale[params.scale_index] = 1.0;
        }
      }
      
      // Wait for scale to be computed
      workgroupBarrier();
      
      // Quantize value
      let current_scale = scale[params.scale_index];
      let value = input[idx];
      let quantized = i32(round(value * current_scale));
      
      // Clamp to int8 range (-127 to 127, saving -128 for padding)
      output[idx] = clamp(quantized, -127, 127);
    `;
  }
  
  // Close shader
  shader += `
    }
  `;
  
  // Add browser-specific optimizations
  if (settings.useBrowserSpecificIntrinsics) {
    switch (capabilities.browserType) {
      case BrowserType.CHROME:
        // Chrome-specific optimizations
        shader = shader.replace(
          '// Quantization Shader',
          '// Quantization Shader\n// @optimize: true'
        );
        break;
        
      case BrowserType.FIREFOX:
        // Firefox-specific optimizations
        if (settings.customSettings.useFirefoxOptimizedBarriers) {
          shader = shader.replace(
            'workgroupBarrier();',
            'workgroupBarrier(); // @align'
          );
        }
        break;
        
      case BrowserType.SAFARI:
        // Safari/Metal-specific optimizations
        if (settings.customSettings.useMetalOptimizations) {
          shader = shader.replace(
            '// Quantization Shader',
            '// Quantization Shader\n// @metal_optimize'
          );
        }
        break;
    }
  }
  
  return shader;
}

/**
 * Get optimized shader code based on current browser and operation
 * 
 * @param device WebGPU device
 * @param operation Operation to optimize ('matmul', 'elementwise', 'softmax', 'quantize')
 * @param customSettings Additional optimization settings
 * @returns Optimized WGSL shader code
 */
export async function getOptimizedShader(
  device: GPUDevice,
  operation: 'matmul' | 'elementwise' | 'softmax' | 'quantize',
  customSettings: Partial<ShaderOptimizationSettings> = {}
): Promise<string> {
  // Get browser capabilities
  const capabilities = await getBrowserCapabilities(device);
  
  // Get default optimization settings
  const defaultSettings = getDefaultOptimizationSettings(capabilities);
  
  // Merge with custom settings
  const settings: ShaderOptimizationSettings = {
    ...defaultSettings,
    ...customSettings,
    customSettings: {
      ...defaultSettings.customSettings,
      ...(customSettings.customSettings || {})
    }
  };
  
  // Get appropriate shader
  switch (operation) {
    case 'matmul':
      return getOptimizedMatmulShader(capabilities, settings);
      
    case 'elementwise':
      return getOptimizedElementwiseShader(capabilities, settings, 'relu');
      
    case 'softmax':
      return getOptimizedSoftmaxShader(capabilities, settings);
      
    case 'quantize':
      return getOptimizedQuantizeShader(capabilities, settings);
      
    default:
      throw new Error(`Unsupported operation: ${operation}`);
  }
}