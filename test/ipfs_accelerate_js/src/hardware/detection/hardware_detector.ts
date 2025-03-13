/**
 * Hardware Detector
 * Utilities for detecting hardware capabilities and optimizing hardware selection
 */

import { HardwareCapabilities } from '../interfaces/hardware_backend';

/**
 * DetectedHardware represents available hardware acceleration capabilities
 */
export interface DetectedHardware {
  /** Whether WebGPU is available */
  hasWebGPU: boolean;
  
  /** Whether WebNN is available */
  hasWebNN: boolean;
  
  /** Whether WebGL is available */
  hasWebGL: boolean;
  
  /** Whether WASM SIMD is available */
  hasWasmSimd: boolean;
  
  /** WebGPU device name (if available) */
  webGPUDeviceName?: string;
  
  /** WebGPU device capabilities (if available) */
  webGPUCapabilities?: HardwareCapabilities;
  
  /** WebNN device capabilities (if available) */
  webNNCapabilities?: HardwareCapabilities;
  
  /** Browser information */
  browser: {
    name: string;
    version: string;
    isMobile: boolean;
  };
  
  /** OS information */
  os: {
    name: string;
    version: string;
  };
  
  /** CPU information */
  cpu: {
    cores: number;
    architecture?: string;
  };
}

/**
 * HardwarePreference represents user preferences for hardware selection
 */
export interface HardwarePreference {
  /** Preferred backend (webgpu, webnn, wasm, cpu) */
  backend?: string;
  
  /** Whether to prefer speed over memory usage */
  preferSpeed?: boolean;
  
  /** Whether to prefer low power consumption */
  preferLowPower?: boolean;
  
  /** Maximum memory usage in bytes */
  maxMemoryUsage?: number;
}

/**
 * Detect available hardware capabilities
 * @returns Promise that resolves with detected hardware
 */
export async function detectHardware(): Promise<DetectedHardware> {
  const result: DetectedHardware = {
    hasWebGPU: false,
    hasWebNN: false,
    hasWebGL: false,
    hasWasmSimd: false,
    browser: {
      name: getBrowserName(),
      version: getBrowserVersion(),
      isMobile: isMobileBrowser()
    },
    os: {
      name: getOSName(),
      version: getOSVersion()
    },
    cpu: {
      cores: getCPUCores(),
      architecture: undefined
    }
  };
  
  // Detect WebGPU
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });
      
      if (adapter) {
        result.hasWebGPU = true;
        result.webGPUDeviceName = await getWebGPUDeviceName(adapter);
        result.webGPUCapabilities = await getWebGPUCapabilities(adapter);
      }
    } catch (error) {
      console.warn('WebGPU detection failed:', error);
    }
  }
  
  // Detect WebNN
  if (typeof navigator !== 'undefined' && 'ml' in navigator) {
    try {
      // @ts-ignore - WebNN is not yet in TypeScript standard lib
      const ml = navigator.ml;
      
      if (ml && typeof ml.getNeuralNetworkContext === 'function') {
        result.hasWebNN = true;
        result.webNNCapabilities = await getWebNNCapabilities();
      }
    } catch (error) {
      console.warn('WebNN detection failed:', error);
    }
  }
  
  // Detect WebGL
  if (typeof document !== 'undefined') {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      result.hasWebGL = !!gl;
    } catch (error) {
      console.warn('WebGL detection failed:', error);
    }
  }
  
  // Detect WASM SIMD
  try {
    // Detection based on feature detect pattern
    const simdTest = WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 
      2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
    ]));
    
    result.hasWasmSimd = simdTest;
  } catch (error) {
    console.warn('WASM SIMD detection failed:', error);
  }
  
  return result;
}

/**
 * Get WebGPU device name
 * @param adapter WebGPU adapter
 * @returns Device name if available
 */
async function getWebGPUDeviceName(adapter: GPUAdapter): Promise<string | undefined> {
  try {
    if ('requestAdapterInfo' in adapter) {
      const info = await adapter.requestAdapterInfo();
      return info.description || info.name || info.device || undefined;
    }
  } catch (error) {
    console.warn('Error getting WebGPU device name:', error);
  }
  
  return undefined;
}

/**
 * Get WebGPU capabilities
 * @param adapter WebGPU adapter
 * @returns Capabilities object
 */
async function getWebGPUCapabilities(adapter: GPUAdapter): Promise<HardwareCapabilities> {
  const limits = adapter.limits;
  
  return {
    maxDimensions: 4,
    maxMatrixSize: Math.min(limits.maxComputeWorkgroupSizeX, 16384),
    supportedDataTypes: ['float32'],
    availableMemory: 'maxBufferSize' in limits ? limits.maxBufferSize as number : undefined,
    supportsAsync: true,
    supportedOperations: {
      basicArithmetic: true,
      matrixMultiplication: true,
      convolution: false,
      reduction: true,
      activation: true
    }
  };
}

/**
 * Get WebNN capabilities
 * @returns Capabilities object
 */
async function getWebNNCapabilities(): Promise<HardwareCapabilities> {
  // WebNN capabilities detection
  // This is a placeholder as WebNN API is still evolving
  return {
    maxDimensions: 4,
    maxMatrixSize: 16384,
    supportedDataTypes: ['float32'],
    supportsAsync: true,
    supportedOperations: {
      basicArithmetic: true,
      matrixMultiplication: true,
      convolution: true,
      reduction: true,
      activation: true
    }
  };
}

/**
 * Get browser name
 * @returns Browser name
 */
function getBrowserName(): string {
  if (typeof navigator === 'undefined') {
    return 'unknown';
  }
  
  const ua = navigator.userAgent;
  
  if (ua.includes('Firefox')) {
    return 'firefox';
  }
  
  if (ua.includes('Edg/')) {
    return 'edge';
  }
  
  if (ua.includes('Chrome')) {
    return 'chrome';
  }
  
  if (ua.includes('Safari') && !ua.includes('Chrome')) {
    return 'safari';
  }
  
  return 'unknown';
}

/**
 * Get browser version
 * @returns Browser version
 */
function getBrowserVersion(): string {
  if (typeof navigator === 'undefined') {
    return 'unknown';
  }
  
  const ua = navigator.userAgent;
  const browserName = getBrowserName();
  
  switch (browserName) {
    case 'firefox': {
      const match = ua.match(/Firefox\/([\d.]+)/);
      return match ? match[1] : 'unknown';
    }
    case 'edge': {
      const match = ua.match(/Edg\/([\d.]+)/);
      return match ? match[1] : 'unknown';
    }
    case 'chrome': {
      const match = ua.match(/Chrome\/([\d.]+)/);
      return match ? match[1] : 'unknown';
    }
    case 'safari': {
      const match = ua.match(/Version\/([\d.]+)/);
      return match ? match[1] : 'unknown';
    }
    default:
      return 'unknown';
  }
}

/**
 * Check if running on a mobile browser
 * @returns True if mobile browser
 */
function isMobileBrowser(): boolean {
  if (typeof navigator === 'undefined') {
    return false;
  }
  
  const ua = navigator.userAgent;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua);
}

/**
 * Get OS name
 * @returns OS name
 */
function getOSName(): string {
  if (typeof navigator === 'undefined') {
    return 'unknown';
  }
  
  const ua = navigator.userAgent;
  
  if (ua.includes('Windows')) {
    return 'windows';
  }
  
  if (ua.includes('Mac OS')) {
    return 'macos';
  }
  
  if (ua.includes('Linux')) {
    return 'linux';
  }
  
  if (ua.includes('Android')) {
    return 'android';
  }
  
  if (ua.includes('iOS') || ua.includes('iPhone') || ua.includes('iPad')) {
    return 'ios';
  }
  
  return 'unknown';
}

/**
 * Get OS version
 * @returns OS version
 */
function getOSVersion(): string {
  if (typeof navigator === 'undefined') {
    return 'unknown';
  }
  
  const ua = navigator.userAgent;
  const osName = getOSName();
  
  switch (osName) {
    case 'windows': {
      const match = ua.match(/Windows NT ([\d.]+)/);
      return match ? match[1] : 'unknown';
    }
    case 'macos': {
      const match = ua.match(/Mac OS X ([\d_.]+)/);
      return match ? match[1].replace(/_/g, '.') : 'unknown';
    }
    case 'android': {
      const match = ua.match(/Android ([\d.]+)/);
      return match ? match[1] : 'unknown';
    }
    case 'ios': {
      const match = ua.match(/OS ([\d_]+)/);
      return match ? match[1].replace(/_/g, '.') : 'unknown';
    }
    default:
      return 'unknown';
  }
}

/**
 * Get CPU core count
 * @returns Number of CPU cores
 */
function getCPUCores(): number {
  if (typeof navigator !== 'undefined' && 'hardwareConcurrency' in navigator) {
    return navigator.hardwareConcurrency || 1;
  }
  
  return 1;
}

/**
 * Optimize hardware selection based on detected hardware and preferences
 * @param detected Detected hardware
 * @param preferences User preferences
 * @returns Optimal backend type
 */
export function optimizeHardwareSelection(
  detected: DetectedHardware,
  preferences: HardwarePreference = {}
): string {
  // If user has a specific preference, use it if available
  if (preferences.backend) {
    switch (preferences.backend) {
      case 'webgpu':
        if (detected.hasWebGPU) return 'webgpu';
        break;
      case 'webnn':
        if (detected.hasWebNN) return 'webnn';
        break;
      case 'wasm':
        if (detected.hasWasmSimd) return 'wasm-simd';
        return 'wasm';
      case 'cpu':
        return 'cpu';
    }
  }
  
  // Check for low power preference
  if (preferences.preferLowPower) {
    // For low power, WebNN is often better than WebGPU if available
    if (detected.hasWebNN) return 'webnn';
    if (detected.hasWasmSimd) return 'wasm-simd';
    return 'cpu';
  }
  
  // General optimization for performance
  if (detected.hasWebGPU) {
    // Check browser-specific optimizations
    if (detected.browser.name === 'firefox' && preferences.preferSpeed) {
      // Firefox sometimes has better WebGPU compute shader performance
      return 'webgpu';
    }
    
    if (detected.browser.name === 'safari' && detected.hasWebNN) {
      // Safari often has better WebNN than WebGPU
      return 'webnn';
    }
    
    // Default to WebGPU for chrome and edge
    return 'webgpu';
  }
  
  if (detected.hasWebNN) return 'webnn';
  if (detected.hasWasmSimd) return 'wasm-simd';
  
  // Fallback to CPU
  return 'cpu';
}