/**
 * Browser detection utilities
 * 
 * This file provides functions for detecting browser type, version,
 * and hardware capabilities with detailed feature detection.
 */

export interface BrowserInfo {
  name: string;
  version: string;
  userAgent: string;
  isMobile: boolean;
  os: {
    name: string;
    version: string;
  };
  features: {
    webgpu: boolean;
    webnn: boolean;
    webgl2: boolean;
    webworker: boolean;
    sharedArrayBuffer: boolean;
    simd: boolean;
    gpu: boolean;
  };
  vendor: string;
}

/**
 * Detect current browser information
 */
export function detectBrowser(): BrowserInfo {
  const userAgent = navigator.userAgent;
  let name = 'unknown';
  let version = 'unknown';
  let isMobile = false;
  
  // Detect browser name and version
  if (userAgent.indexOf('Edge') > -1 || userAgent.indexOf('Edg/') > -1) {
    name = 'edge';
    const edgeMatch = userAgent.match(/Edge\/(\d+)/) || userAgent.match(/Edg\/(\d+)/);
    version = edgeMatch ? edgeMatch[1] : 'unknown';
  } else if (userAgent.indexOf('Firefox') > -1) {
    name = 'firefox';
    const firefoxMatch = userAgent.match(/Firefox\/(\d+)/);
    version = firefoxMatch ? firefoxMatch[1] : 'unknown';
  } else if (userAgent.indexOf('Chrome') > -1) {
    name = 'chrome';
    const chromeMatch = userAgent.match(/Chrome\/(\d+)/);
    version = chromeMatch ? chromeMatch[1] : 'unknown';
  } else if (userAgent.indexOf('Safari') > -1) {
    name = 'safari';
    const safariMatch = userAgent.match(/Version\/(\d+)/);
    version = safariMatch ? safariMatch[1] : 'unknown';
  } else if (userAgent.indexOf('MSIE') > -1 || userAgent.indexOf('Trident/') > -1) {
    name = 'ie';
    const ieMatch = userAgent.match(/(?:MSIE |rv:)(\d+)/);
    version = ieMatch ? ieMatch[1] : 'unknown';
  } else if (userAgent.indexOf('Opera') > -1 || userAgent.indexOf('OPR/') > -1) {
    name = 'opera';
    const operaMatch = userAgent.match(/(?:Opera|OPR)\/(\d+)/);
    version = operaMatch ? operaMatch[1] : 'unknown';
  }
  
  // Detect mobile
  if (
    /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent)
  ) {
    isMobile = true;
  }
  
  // Detect OS
  const os = detectOS(userAgent);

  // Detect features
  const features = {
    webgpu: 'gpu' in navigator,
    webnn: 'ml' in navigator,
    webgl2: detectWebGL2Support(),
    webworker: typeof Worker !== 'undefined',
    sharedArrayBuffer: typeof SharedArrayBuffer === 'function',
    simd: detectWasmSimd(),
    gpu: detectGPUAcceleration()
  };

  // Get vendor info
  const vendor = getHardwareVendorInfo();
  
  return {
    name,
    version,
    userAgent,
    isMobile,
    os,
    features,
    vendor
  };
}

/**
 * Detect operating system
 */
function detectOS(userAgent: string): { name: string; version: string } {
  let name = 'unknown';
  let version = 'unknown';
  
  if (userAgent.indexOf('Win') !== -1) {
    name = 'windows';
    const winMatch = userAgent.match(/Windows NT (\d+\.\d+)/);
    if (winMatch) {
      const winVersion = winMatch[1];
      switch (winVersion) {
        case '10.0': version = '10'; break;
        case '6.3': version = '8.1'; break;
        case '6.2': version = '8'; break;
        case '6.1': version = '7'; break;
        case '6.0': version = 'Vista'; break;
        case '5.2': version = 'XP x64'; break;
        case '5.1': version = 'XP'; break;
        default: version = winVersion;
      }
    }
  } else if (userAgent.indexOf('Mac') !== -1) {
    if (userAgent.indexOf('iPad') !== -1 || userAgent.indexOf('iPhone') !== -1) {
      name = 'ios';
      const iosMatch = userAgent.match(/OS (\d+[._]\d+)/) || userAgent.match(/OS (\d+)/);
      version = iosMatch ? iosMatch[1].replace('_', '.') : 'unknown';
    } else {
      name = 'macos';
      const macMatch = userAgent.match(/Mac OS X (\d+[._]\d+[._]\d+)/) || 
                        userAgent.match(/Mac OS X (\d+[._]\d+)/);
      version = macMatch ? macMatch[1].replace(/_/g, '.') : 'unknown';
    }
  } else if (userAgent.indexOf('Android') !== -1) {
    name = 'android';
    const androidMatch = userAgent.match(/Android (\d+(\.\d+)*)/);
    version = androidMatch ? androidMatch[1] : 'unknown';
  } else if (userAgent.indexOf('Linux') !== -1) {
    name = 'linux';
    version = 'unknown';
  } else if (userAgent.indexOf('CrOS') !== -1) {
    name = 'chromeos';
    version = 'unknown';
  }
  
  return { name, version };
}

/**
 * Detect WebGL2 support
 */
function detectWebGL2Support(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('webgl2');
    return !!ctx;
  } catch (e) {
    return false;
  }
}

/**
 * Detect WebAssembly SIMD support
 */
function detectWasmSimd(): boolean {
  try {
    // Check for WebAssembly SIMD support
    if (typeof WebAssembly !== 'object') {
      return false;
    }
    
    // v128.const is a SIMD instruction
    const simdSupported = WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3,
      2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
    ]));
    
    return simdSupported;
  } catch (e) {
    return false;
  }
}

/**
 * Detect if real GPU acceleration is available
 */
function detectGPUAcceleration(): boolean {
  // Basic check for WebGPU
  if ('gpu' in navigator) {
    // Detailed check will happen in WebGPU backend
    return true;
  }
  
  // Check WebGL for potential GPU
  try {
    const canvas = document.createElement('canvas');
    
    // Try WebGL2 first
    let gl = canvas.getContext('webgl2');
    
    // Fallback to WebGL
    if (!gl) {
      gl = canvas.getContext('webgl') || 
           canvas.getContext('experimental-webgl') as WebGLRenderingContext;
    }
    
    if (!gl) return false;
    
    // Check renderer information
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    if (debugInfo) {
      const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      
      // Check for software renderers
      const softwareRenderers = [
        'swiftshader',
        'llvmpipe',
        'software',
        'microsoft basic',
        'gdi generic',
        'mesa offscreen',
        'microsoft basic render'
      ];
      
      return !softwareRenderers.some(renderer => 
        renderer.toLowerCase().includes(renderer)
      );
    }
    
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * Get hardware vendor information
 */
function getHardwareVendorInfo(): string {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || 
               canvas.getContext('webgl') || 
               canvas.getContext('experimental-webgl') as WebGLRenderingContext;
    
    if (!gl) return 'unknown';
    
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    if (debugInfo) {
      const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) || 'unknown';
      const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || 'unknown';
      return `${vendor} (${renderer})`;
    }
    
    return 'unknown';
  } catch (e) {
    return 'unknown';
  }
}

/**
 * Get detailed WebGPU capabilities information async
 */
export async function getWebGPUCapabilities(): Promise<any | null> {
  if (!('gpu' in navigator)) {
    return null;
  }
  
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return null;
    }
    
    const adapterInfo = await adapter.requestAdapterInfo();
    const features = Array.from(adapter.features).map(feature => String(feature));
    
    // Get adapter limits
    const limits: Record<string, number> = {};
    const adapterLimits = adapter.limits;
    
    // Convert limits to a plain object
    for (const key of Object.getOwnPropertyNames(Object.getPrototypeOf(adapterLimits))) {
      if (typeof adapterLimits[key as keyof GPUSupportedLimits] === 'number') {
        limits[key] = adapterLimits[key as keyof GPUSupportedLimits] as number;
      }
    }
    
    return {
      vendor: adapterInfo.vendor || 'unknown',
      architecture: adapterInfo.architecture || 'unknown',
      device: adapterInfo.device || 'unknown',
      description: adapterInfo.description || 'unknown',
      features,
      limits
    };
  } catch (e) {
    return null;
  }
}

/**
 * Get detailed WebNN capabilities information async
 */
export async function getWebNNCapabilities(): Promise<any | null> {
  if (!('ml' in navigator)) {
    return null;
  }
  
  try {
    const context = await (navigator as any).ml.createContext();
    if (!context) {
      return null;
    }
    
    const deviceType = context.deviceType || 'unknown';
    const deviceInfo = context.deviceInfo || {};
    
    // Test for supported operations using a graph builder
    const supportedOps: string[] = [];
    try {
      const builder = new (window as any).MLGraphBuilder(context);
      
      // Create a test tensor
      const testTensor = context.createOperand(
        { type: 'float32', dimensions: [1, 1, 1, 1] },
        new Float32Array([1])
      );
      
      // Test various operations
      if (testTensor) {
        try { builder.relu(testTensor); supportedOps.push('relu'); } catch {}
        try { builder.sigmoid(testTensor); supportedOps.push('sigmoid'); } catch {}
        try { builder.tanh(testTensor); supportedOps.push('tanh'); } catch {}
        try { builder.add(testTensor, testTensor); supportedOps.push('add'); } catch {}
        try { builder.sub(testTensor, testTensor); supportedOps.push('sub'); } catch {}
        try { builder.mul(testTensor, testTensor); supportedOps.push('mul'); } catch {}
        try { builder.matmul(testTensor, testTensor); supportedOps.push('matmul'); } catch {}
      }
    } catch (e) {
      // Failed to test operations
    }
    
    return {
      deviceType,
      deviceName: deviceInfo.name || 'unknown',
      features: supportedOps
    };
  } catch (e) {
    return null;
  }
}

/**
 * Get comprehensive browser capabilities
 */
export async function getBrowserCapabilities(): Promise<{
  browser: BrowserInfo;
  webgpu: any | null;
  webnn: any | null;
}> {
  const browser = detectBrowser();
  const webgpu = await getWebGPUCapabilities();
  const webnn = await getWebNNCapabilities();
  
  return {
    browser,
    webgpu,
    webnn
  };
}