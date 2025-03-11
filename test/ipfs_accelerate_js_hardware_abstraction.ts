/**
 * Hardware Abstraction Layer
 * 
 * This file provides a unified interface for accessing various hardware backends
 * (WebGPU, WebNN, WebAssembly) with automatic fallback capability.
 */

import { WebGPUBackend, isWebGPUSupported, getWebGPUInfo } from './ipfs_accelerate_js_webgpu_backend';
import { WebNNBackend, isWebNNSupported, getWebNNInfo } from './ipfs_accelerate_js_webnn_backend';

export type HardwareBackendType = 'webgpu' | 'webnn' | 'wasm' | 'cpu';

export interface HardwareCapabilities {
  webgpu: {
    supported: boolean;
    adapterInfo?: any;
    features?: string[];
    isSimulated?: boolean;
  };
  webnn: {
    supported: boolean;
    deviceType?: string;
    deviceName?: string;
    isSimulated?: boolean;
    features?: string[];
  };
  wasm: {
    supported: boolean;
    simd?: boolean;
    threads?: boolean;
  };
  optimalBackend: HardwareBackendType;
  browserName: string;
  browserVersion: string;
}

export interface HardwareAbstractionOptions {
  /** Enable logging */
  logging?: boolean;
  /** Preferred backend order */
  preferredBackends?: HardwareBackendType[];
  /** WebGPU specific options */
  webgpuOptions?: any;
  /** WebNN specific options */
  webnnOptions?: any;
  /** WebAssembly specific options */
  wasmOptions?: any;
}

/**
 * Hardware Abstraction Layer for unified access to acceleration backends
 */
export class HardwareAbstraction {
  private webgpuBackend: WebGPUBackend | null = null;
  private webnnBackend: WebNNBackend | null = null;
  private wasmBackend: any | null = null;  // To be implemented
  private capabilities: HardwareCapabilities | null = null;
  private activeBackend: HardwareBackendType | null = null;
  private options: HardwareAbstractionOptions;
  private initialized: boolean = false;

  constructor(options: HardwareAbstractionOptions = {}) {
    this.options = {
      logging: false,
      preferredBackends: ['webgpu', 'webnn', 'wasm', 'cpu'],
      ...options
    };
  }

  /**
   * Initialize the hardware abstraction layer and detect capabilities
   */
  async initialize(): Promise<boolean> {
    try {
      // Detect browser
      const browserInfo = this.detectBrowser();
      
      // Detect capabilities
      this.capabilities = {
        webgpu: {
          supported: false
        },
        webnn: {
          supported: false
        },
        wasm: {
          supported: this.detectWasmSupport()
        },
        optimalBackend: 'cpu',
        browserName: browserInfo.name,
        browserVersion: browserInfo.version
      };
      
      // Check WebGPU support
      if (await isWebGPUSupported()) {
        const webgpuInfo = await getWebGPUInfo();
        this.capabilities.webgpu = webgpuInfo;
        
        // Initialize WebGPU backend
        this.webgpuBackend = new WebGPUBackend(this.options.webgpuOptions);
        const webgpuInitialized = await this.webgpuBackend.initialize();
        
        if (!webgpuInitialized) {
          this.webgpuBackend = null;
        }
      }
      
      // Check WebNN support
      if (await isWebNNSupported()) {
        const webnnInfo = await getWebNNInfo();
        this.capabilities.webnn = webnnInfo;
        
        // Initialize WebNN backend
        this.webnnBackend = new WebNNBackend(this.options.webnnOptions);
        const webnnInitialized = await this.webnnBackend.initialize();
        
        if (!webnnInitialized) {
          this.webnnBackend = null;
        }
      }
      
      // Initialize WebAssembly backend (to be implemented)
      // This will be expanded in the actual implementation
      
      // Determine optimal backend based on capabilities and preferences
      this.activeBackend = this.determineOptimalBackend();
      this.capabilities.optimalBackend = this.activeBackend;
      
      // Log info if logging is enabled
      if (this.options.logging) {
        console.log('Hardware Capabilities:', this.capabilities);
        console.log('Active Backend:', this.activeBackend);
      }
      
      this.initialized = true;
      return true;
    } catch (error) {
      console.error('Failed to initialize hardware abstraction layer:', error);
      return false;
    }
  }

  /**
   * Determine the optimal backend based on capabilities and preferences
   */
  private determineOptimalBackend(): HardwareBackendType {
    // Start with the preferred order
    for (const backend of this.options.preferredBackends!) {
      // Check if backend is available
      switch (backend) {
        case 'webgpu':
          if (this.webgpuBackend) return 'webgpu';
          break;
        case 'webnn':
          if (this.webnnBackend) return 'webnn';
          break;
        case 'wasm':
          if (this.wasmBackend) return 'wasm';
          break;
        case 'cpu':
          return 'cpu'; // CPU is always available as a fallback
      }
    }
    
    // If none of the preferred backends are available, use the first available
    if (this.webgpuBackend) return 'webgpu';
    if (this.webnnBackend) return 'webnn';
    if (this.wasmBackend) return 'wasm';
    
    // Default to CPU
    return 'cpu';
  }

  /**
   * Get the currently active backend
   */
  getActiveBackend(): HardwareBackendType | null {
    return this.activeBackend;
  }

  /**
   * Switch to a different backend if available
   */
  async switchBackend(backend: HardwareBackendType): Promise<boolean> {
    if (!this.initialized) {
      throw new Error('Hardware abstraction layer not initialized');
    }
    
    // Check if the requested backend is available
    switch (backend) {
      case 'webgpu':
        if (!this.webgpuBackend) {
          throw new Error('WebGPU backend not available');
        }
        break;
      case 'webnn':
        if (!this.webnnBackend) {
          throw new Error('WebNN backend not available');
        }
        break;
      case 'wasm':
        if (!this.wasmBackend) {
          throw new Error('WebAssembly backend not available');
        }
        break;
      case 'cpu':
        // CPU is always available
        break;
      default:
        throw new Error(`Unknown backend: ${backend}`);
    }
    
    // Switch to the new backend
    this.activeBackend = backend;
    
    if (this.options.logging) {
      console.log(`Switched to ${backend} backend`);
    }
    
    return true;
  }

  /**
   * Get the WebGPU backend if available
   */
  getWebGPUBackend(): WebGPUBackend | null {
    return this.webgpuBackend;
  }

  /**
   * Get the WebNN backend if available
   */
  getWebNNBackend(): WebNNBackend | null {
    return this.webnnBackend;
  }

  /**
   * Get the WebAssembly backend if available
   */
  getWasmBackend(): any | null {
    return this.wasmBackend;
  }

  /**
   * Get hardware capabilities
   */
  getCapabilities(): HardwareCapabilities | null {
    return this.capabilities;
  }

  /**
   * Check if a specific backend type is supported
   */
  isBackendSupported(backend: HardwareBackendType): boolean {
    if (!this.capabilities) {
      return false;
    }
    
    switch (backend) {
      case 'webgpu':
        return this.capabilities.webgpu.supported;
      case 'webnn':
        return this.capabilities.webnn.supported;
      case 'wasm':
        return this.capabilities.wasm.supported;
      case 'cpu':
        return true; // CPU is always supported
      default:
        return false;
    }
  }

  /**
   * Get the optimal backend for a specific model type
   */
  getOptimalBackendForModel(
    modelType: 'text' | 'vision' | 'audio' | 'multimodal'
  ): HardwareBackendType {
    if (!this.capabilities) {
      return 'cpu';
    }
    
    // Browser-specific optimizations
    const browser = this.capabilities.browserName.toLowerCase();
    
    // Consider both hardware capabilities and browser-specific optimizations
    if (modelType === 'audio' && browser === 'firefox' && this.isBackendSupported('webgpu')) {
      // Firefox has optimized audio compute shaders
      return 'webgpu';
    } else if (modelType === 'text' && browser === 'edge' && this.isBackendSupported('webnn')) {
      // Edge has good WebNN text model support
      return 'webnn';
    } else if (modelType === 'vision' && this.isBackendSupported('webgpu')) {
      // Vision models generally work best on WebGPU
      return 'webgpu';
    }
    
    // Default to the active backend
    return this.activeBackend || 'cpu';
  }

  /**
   * Detect the current browser
   */
  private detectBrowser(): { name: string; version: string } {
    const userAgent = navigator.userAgent;
    let name = 'unknown';
    let version = 'unknown';
    
    // Extract browser name and version from user agent
    if (userAgent.indexOf('Edge') > -1) {
      name = 'edge';
      const edgeMatch = userAgent.match(/Edge\/(\d+)/);
      version = edgeMatch ? edgeMatch[1] : 'unknown';
    } else if (userAgent.indexOf('Edg') > -1) {
      name = 'edge';
      const edgMatch = userAgent.match(/Edg\/(\d+)/);
      version = edgMatch ? edgMatch[1] : 'unknown';
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
    }
    
    return { name, version };
  }

  /**
   * Detect WebAssembly support
   */
  private detectWasmSupport(): boolean {
    try {
      // Check basic WebAssembly support
      if (typeof WebAssembly !== 'object') {
        return false;
      }
      
      // Check SIMD support
      const simdSupported = WebAssembly.validate(new Uint8Array([
        0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3,
        2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
      ]));
      
      // Check threads support
      const threadsSupported = typeof SharedArrayBuffer === 'function';
      
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    if (this.webgpuBackend) {
      this.webgpuBackend.dispose();
      this.webgpuBackend = null;
    }
    
    if (this.webnnBackend) {
      this.webnnBackend.dispose();
      this.webnnBackend = null;
    }
    
    if (this.wasmBackend) {
      // Clean up WebAssembly backend
      this.wasmBackend = null;
    }
    
    this.activeBackend = null;
    this.initialized = false;
  }
}

/**
 * Create a hardware abstraction layer instance and initialize it
 */
export async function createHardwareAbstraction(
  options: HardwareAbstractionOptions = {}
): Promise<HardwareAbstraction> {
  const hardware = new HardwareAbstraction(options);
  await hardware.initialize();
  return hardware;
}