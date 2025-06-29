/**
 * Hardware Detection Module
 * Detects hardware capabilities and provides recommendations for optimal backends
 */

/**
 * Hardware capabilities interface
 */
export interface HardwareCapabilities {
  /**
   * Browser name
   */
  browserName: string;
  
  /**
   * Browser version
   */
  browserVersion: string;
  
  /**
   * Platform (OS)
   */
  platform: string;
  
  /**
   * OS version
   */
  osVersion: string;
  
  /**
   * Whether the device is mobile
   */
  isMobile: boolean;
  
  /**
   * Whether WebGPU is supported
   */
  webgpuSupported: boolean;
  
  /**
   * WebGPU features supported
   */
  webgpuFeatures: string[];
  
  /**
   * Whether WebNN is supported
   */
  webnnSupported: boolean;
  
  /**
   * WebNN features supported
   */
  webnnFeatures: string[];
  
  /**
   * Whether WebAssembly is supported
   */
  wasmSupported: boolean;
  
  /**
   * WebAssembly features supported
   */
  wasmFeatures: string[];
  
  /**
   * Recommended backend for hardware acceleration
   */
  recommendedBackend: string;
  
  /**
   * Recommended browser-specific optimizations
   */
  browserOptimizations: {
    /**
     * Recommended for audio models (Firefox compute shaders)
     */
    audioOptimized: boolean;
    
    /**
     * Recommended for shader precompilation
     */
    shaderPrecompilation: boolean;
    
    /**
     * Supports parallel model loading
     */
    parallelLoading: boolean;
  };
  
  /**
   * Estimated memory limit in MB
   */
  memoryLimitMB: number;
}

/**
 * Browser detection result
 */
interface BrowserDetection {
  name: string;
  version: string;
  isChrome: boolean;
  isFirefox: boolean;
  isEdge: boolean;
  isSafari: boolean;
  isMobile: boolean;
  platform: string;
  osVersion: string;
}

/**
 * Detect browser information
 */
function detectBrowser(): BrowserDetection {
  const userAgent = navigator.userAgent;
  let browserName = 'Unknown';
  let browserVersion = '';
  let isChrome = false;
  let isFirefox = false;
  let isEdge = false;
  let isSafari = false;
  let isMobile = false;
  let platform = '';
  let osVersion = '';
  
  // Detect platform
  if (/android/i.test(userAgent)) {
    platform = 'Android';
    isMobile = true;
    const match = userAgent.match(/Android\s([0-9.]+)/);
    osVersion = match ? match[1] : '';
  } else if (/iPad|iPhone|iPod/.test(userAgent) && !(window as any).MSStream) {
    platform = 'iOS';
    isMobile = true;
    const match = userAgent.match(/OS\s([0-9_]+)/);
    osVersion = match ? match[1].replace(/_/g, '.') : '';
  } else if (/Win/.test(userAgent)) {
    platform = 'Windows';
    const match = userAgent.match(/Windows NT\s([0-9.]+)/);
    osVersion = match ? match[1] : '';
  } else if (/Mac/.test(userAgent)) {
    platform = 'macOS';
    const match = userAgent.match(/Mac OS X\s([0-9_.]+)/);
    osVersion = match ? match[1].replace(/_/g, '.') : '';
  } else if (/Linux/.test(userAgent)) {
    platform = 'Linux';
    osVersion = '';
  }
  
  // Detect browser
  if (/Edge\//.test(userAgent)) {
    // Edge Legacy
    isEdge = true;
    browserName = 'Edge';
    const match = userAgent.match(/Edge\/([0-9.]+)/);
    browserVersion = match ? match[1] : '';
  } else if (/Edg\//.test(userAgent)) {
    // Edge Chromium
    isEdge = true;
    browserName = 'Edge';
    const match = userAgent.match(/Edg\/([0-9.]+)/);
    browserVersion = match ? match[1] : '';
  } else if (/Firefox\//.test(userAgent)) {
    isFirefox = true;
    browserName = 'Firefox';
    const match = userAgent.match(/Firefox\/([0-9.]+)/);
    browserVersion = match ? match[1] : '';
  } else if (/Chrome\//.test(userAgent)) {
    isChrome = true;
    browserName = 'Chrome';
    const match = userAgent.match(/Chrome\/([0-9.]+)/);
    browserVersion = match ? match[1] : '';
  } else if (/Safari\//.test(userAgent)) {
    isSafari = true;
    browserName = 'Safari';
    const match = userAgent.match(/Version\/([0-9.]+)/);
    browserVersion = match ? match[1] : '';
  }
  
  // Additional mobile detection
  if (!isMobile) {
    isMobile = /Mobile|Android|iPhone|iPad|iPod/.test(userAgent);
  }
  
  return {
    name: browserName,
    version: browserVersion,
    isChrome,
    isFirefox,
    isEdge,
    isSafari,
    isMobile,
    platform,
    osVersion
  };
}

/**
 * Check WebGPU support
 */
async function checkWebGPUSupport(): Promise<{
  supported: boolean;
  features: string[];
  adapter: GPUAdapter | null;
}> {
  if (!navigator.gpu) {
    return { supported: false, features: [], adapter: null };
  }
  
  try {
    // Request adapter with high-performance preference
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });
    
    if (!adapter) {
      return { supported: false, features: [], adapter: null };
    }
    
    // Get supported features
    const features = Array.from(adapter.features).map(feature => feature.toString());
    
    return {
      supported: true,
      features,
      adapter
    };
  } catch (error) {
    console.warn('WebGPU check error:', error);
    return { supported: false, features: [], adapter: null };
  }
}

/**
 * Check WebNN support
 */
async function checkWebNNSupport(): Promise<{
  supported: boolean;
  features: string[];
}> {
  if (!(navigator as any).ml) {
    return { supported: false, features: [] };
  }
  
  try {
    // Try to create a context
    const context = await (navigator as any).ml.createContext();
    
    if (!context) {
      return { supported: false, features: [] };
    }
    
    // List of features to check (these are hypothetical as the WebNN API is still evolving)
    const featuresChecks = [
      { name: 'float32', check: () => true }, // Most basic feature
      { name: 'float16', check: () => true },  // Hypothetical check
      { name: 'int8', check: () => true },     // Hypothetical check
      { name: 'conv2d', check: () => true },   // Hypothetical check
      { name: 'matmul', check: () => true }    // Hypothetical check
    ];
    
    // Check each feature (this would need to be adapted as the WebNN API evolves)
    const features = featuresChecks
      .filter(feature => {
        try {
          return feature.check();
        } catch {
          return false;
        }
      })
      .map(feature => feature.name);
    
    return {
      supported: true,
      features
    };
  } catch (error) {
    console.warn('WebNN check error:', error);
    return { supported: false, features: [] };
  }
}

/**
 * Check WebAssembly support
 */
function checkWasmSupport(): {
  supported: boolean;
  features: string[];
} {
  if (typeof WebAssembly !== 'object') {
    return { supported: false, features: [] };
  }
  
  const features: string[] = [];
  
  // Check for basic WebAssembly support
  if (WebAssembly.validate) {
    features.push('basic');
  }
  
  // Check for SIMD support
  if (WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]))) {
    features.push('simd');
  }
  
  // Check for threads support
  if (typeof SharedArrayBuffer !== 'undefined') {
    features.push('threads');
  }
  
  // Check for bulk memory operations
  if (WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 5, 3, 1, 0, 1, 10, 11, 1, 9, 0, 65, 0, 65, 0, 65, 0, 252, 10, 11]))) {
    features.push('bulk-memory');
  }
  
  // Check for reference types
  if (WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 2, 7, 1, 1, 101, 3, 102, 0, 0, 3, 2, 1, 0, 7, 5, 1, 1, 102, 0, 0, 10, 3, 1, 1, 0]))) {
    features.push('reference-types');
  }
  
  return {
    supported: features.length > 0,
    features
  };
}

/**
 * Estimate available memory
 */
function estimateMemory(): number {
  // Try to get memory info from performance API
  if ((navigator as any).deviceMemory) {
    return (navigator as any).deviceMemory * 1024; // Convert GB to MB
  }
  
  // Estimate based on platform
  const browser = detectBrowser();
  
  if (browser.isMobile) {
    // Conservative estimate for mobile
    return 1024; // 1GB
  } else {
    // Desktop estimate
    return 4096; // 4GB
  }
}

/**
 * Determine browser-specific optimizations
 */
function determineBrowserOptimizations(browserInfo: BrowserDetection): {
  audioOptimized: boolean;
  shaderPrecompilation: boolean;
  parallelLoading: boolean;
} {
  const result = {
    audioOptimized: false,
    shaderPrecompilation: false,
    parallelLoading: false
  };
  
  // Firefox has excellent compute shader performance for audio models
  if (browserInfo.isFirefox) {
    result.audioOptimized = true;
  }
  
  // Chrome has good shader precompilation support
  if (browserInfo.isChrome) {
    result.shaderPrecompilation = true;
  }
  
  // Edge and Chrome have good support for parallel loading
  if (browserInfo.isEdge || browserInfo.isChrome) {
    result.parallelLoading = true;
  }
  
  return result;
}

/**
 * Detect hardware capabilities
 */
export async function detectHardwareCapabilities(): Promise<HardwareCapabilities> {
  // Detect browser
  const browserInfo = detectBrowser();
  
  // Check hardware acceleration support
  const [webgpuSupport, webnnSupport, wasmSupport] = await Promise.all([
    checkWebGPUSupport(),
    checkWebNNSupport(),
    Promise.resolve(checkWasmSupport())
  ]);
  
  // Determine browser optimizations
  const browserOptimizations = determineBrowserOptimizations(browserInfo);
  
  // Determine recommended backend
  let recommendedBackend = 'cpu';
  
  if (webgpuSupport.supported) {
    recommendedBackend = 'webgpu';
  } else if (webnnSupport.supported) {
    recommendedBackend = 'webnn';
  } else if (wasmSupport.supported) {
    recommendedBackend = 'wasm';
  }
  
  // Edge has better WebNN support
  if (browserInfo.isEdge && webnnSupport.supported) {
    recommendedBackend = 'webnn';
  }
  
  // Firefox has better WebGPU compute shader performance
  if (browserInfo.isFirefox && webgpuSupport.supported && browserOptimizations.audioOptimized) {
    recommendedBackend = 'webgpu';
  }
  
  // Estimate memory limit
  const memoryLimitMB = estimateMemory();
  
  return {
    browserName: browserInfo.name,
    browserVersion: browserInfo.version,
    platform: browserInfo.platform,
    osVersion: browserInfo.osVersion,
    isMobile: browserInfo.isMobile,
    webgpuSupported: webgpuSupport.supported,
    webgpuFeatures: webgpuSupport.features,
    webnnSupported: webnnSupport.supported,
    webnnFeatures: webnnSupport.features,
    wasmSupported: wasmSupport.supported,
    wasmFeatures: wasmSupport.features,
    recommendedBackend,
    browserOptimizations,
    memoryLimitMB
  };
}