/**
 * Browser Interface Implementation
 * 
 * This file provides a unified interface for interacting with the browser environment,
 * including capabilities detection, optimizations, and browser-specific features.
 */

import { HardwareBackendType } from './ipfs_accelerate_js_hardware_abstraction';

export interface BrowserInfo {
  name: string;
  version: string;
  userAgent: string;
  isMobile: boolean;
  platform: string;
  isSimulated: boolean;
}

export interface BrowserCapabilities {
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
  browserInfo: BrowserInfo;
}

export interface BrowserInterfaceOptions {
  /** Enable logging */
  logging?: boolean;
  /** Cache detection results */
  useCache?: boolean;
  /** Cache expiry time in milliseconds */
  cacheExpiryMs?: number;
}

/**
 * BrowserInterface class for interacting with browser environment
 */
export class BrowserInterface {
  private capabilities: BrowserCapabilities | null = null;
  private browserInfo: BrowserInfo | null = null;
  private isNode: boolean = false;
  private options: BrowserInterfaceOptions;

  constructor(options: BrowserInterfaceOptions = {}) {
    this.options = {
      logging: false,
      useCache: true,
      cacheExpiryMs: 3600000, // 1 hour
      ...options
    };
    
    // Detect Node.js environment
    this.isNode = typeof window === 'undefined';
    
    // Detect browser info if in browser environment
    if (!this.isNode) {
      this.browserInfo = this.detectBrowserInfo();
    }
  }

  /**
   * Detect browser information
   */
  private detectBrowserInfo(): BrowserInfo {
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
    
    // Detect if mobile browser
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent);
    
    // Get platform
    const platform = navigator.platform || 'unknown';
    
    // Detect if running in an emulator or virtual machine
    // This is a best-effort approach
    const isSimulated = this.detectSimulatedEnvironment();
    
    return {
      name,
      version,
      userAgent,
      isMobile,
      platform,
      isSimulated
    };
  }

  /**
   * Detect if running in a simulated environment (emulator or VM)
   */
  private detectSimulatedEnvironment(): boolean {
    // This is a best-effort approach, not foolproof
    try {
      // Check if navigator has unusual properties or inconsistencies
      if (navigator.hardwareConcurrency <= 1) {
        return true;
      }
      
      // Check for inconsistent audio context behavior
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const sampleRate = audioContext.sampleRate;
      audioContext.close();
      
      // Some emulators use non-standard sample rates
      if (sampleRate !== 44100 && sampleRate !== 48000) {
        return true;
      }
      
      // Check GPU renderer string if available
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl');
      
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
          const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
          
          // Check for common software renderers
          if (
            renderer.indexOf('SwiftShader') !== -1 ||
            renderer.indexOf('Basic Renderer') !== -1 ||
            renderer.indexOf('llvmpipe') !== -1 ||
            vendor.indexOf('Google') !== -1
          ) {
            return true;
          }
        }
      }
      
      return false;
    } catch (error) {
      console.warn('Error detecting simulated environment:', error);
      return false;
    }
  }

  /**
   * Detect browser capabilities
   */
  async detectCapabilities(): Promise<BrowserCapabilities> {
    // Check if we have cached capabilities
    if (this.capabilities && this.options.useCache) {
      return this.capabilities;
    }
    
    if (this.isNode) {
      throw new Error('Cannot detect browser capabilities in Node.js environment');
    }
    
    try {
      // Detect WebGPU capabilities
      const webgpuCapabilities = await this.detectWebGPUCapabilities();
      
      // Detect WebNN capabilities
      const webnnCapabilities = await this.detectWebNNCapabilities();
      
      // Detect WebAssembly capabilities
      const wasmCapabilities = this.detectWasmCapabilities();
      
      // Determine optimal backend
      const optimalBackend = this.determineOptimalBackend(
        webgpuCapabilities,
        webnnCapabilities,
        wasmCapabilities
      );
      
      // Create capabilities object
      this.capabilities = {
        webgpu: webgpuCapabilities,
        webnn: webnnCapabilities,
        wasm: wasmCapabilities,
        optimalBackend,
        browserInfo: this.browserInfo!
      };
      
      // Log if enabled
      if (this.options.logging) {
        console.log('Browser capabilities detected:', this.capabilities);
      }
      
      return this.capabilities;
    } catch (error) {
      console.error('Failed to detect browser capabilities:', error);
      
      // Return default capabilities
      return {
        webgpu: { supported: false },
        webnn: { supported: false },
        wasm: { supported: false },
        optimalBackend: 'cpu',
        browserInfo: this.browserInfo!
      };
    }
  }

  /**
   * Detect WebGPU capabilities
   */
  private async detectWebGPUCapabilities(): Promise<any> {
    try {
      // Check if WebGPU is supported
      if (!('gpu' in navigator)) {
        return { supported: false };
      }
      
      // Request adapter
      const adapter = await navigator.gpu.requestAdapter();
      
      if (!adapter) {
        return { supported: false };
      }
      
      // Get adapter info
      const adapterInfo = await adapter.requestAdapterInfo();
      
      // Get adapter features
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
      
      // Try to detect if it's a simulated adapter
      const isSimulated = this.detectSimulatedAdapter(adapterInfo);
      
      return {
        supported: true,
        adapterInfo: {
          vendor: adapterInfo.vendor || 'unknown',
          architecture: adapterInfo.architecture || 'unknown',
          device: adapterInfo.device || 'unknown',
          description: adapterInfo.description || 'unknown'
        },
        features,
        limits,
        isSimulated
      };
    } catch (error) {
      console.warn('Error detecting WebGPU capabilities:', error);
      return { supported: false };
    }
  }

  /**
   * Detect WebNN capabilities
   */
  private async detectWebNNCapabilities(): Promise<any> {
    try {
      // Check if WebNN is supported
      if (!('ml' in navigator)) {
        return { supported: false };
      }
      
      // Create context
      const context = await (navigator as any).ml.createContext();
      
      if (!context) {
        return { supported: false };
      }
      
      // Get device info
      const deviceType = (context as any).deviceType || null;
      const deviceName = await this.getWebNNDeviceName(context);
      
      // Try to detect if it's a simulated device
      const isSimulated = this.detectSimulatedWebNN(deviceName);
      
      // Try to detect supported operations
      const features = await this.detectWebNNFeatures(context);
      
      return {
        supported: true,
        deviceType,
        deviceName,
        features,
        isSimulated
      };
    } catch (error) {
      console.warn('Error detecting WebNN capabilities:', error);
      return { supported: false };
    }
  }

  /**
   * Get WebNN device name
   */
  private async getWebNNDeviceName(context: any): Promise<string | null> {
    try {
      // This is a best-effort attempt as WebNN API doesn't standardize this
      
      // Try to access device name (implementation varies)
      const deviceInfo = context.deviceInfo;
      if (deviceInfo && typeof deviceInfo === 'object') {
        return deviceInfo.name || null;
      }
      
      // If WebGPU is available, we could use that as a fallback to identify hardware
      if ('gpu' in navigator) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          const adapterInfo = await adapter.requestAdapterInfo();
          return adapterInfo.device || null;
        }
      }
      
      return null;
    } catch (error) {
      console.warn('Failed to get WebNN device name:', error);
      return null;
    }
  }

  /**
   * Detect WebNN features
   */
  private async detectWebNNFeatures(context: any): Promise<string[]> {
    try {
      const features: string[] = [];
      const builder = new (window as any).MLGraphBuilder(context);
      
      // Create a small test tensor
      const desc = {
        type: 'float32',
        dimensions: [2, 2]
      };
      
      const data = new Float32Array([1, 2, 3, 4]);
      const testTensor = context.createOperand(desc, data);
      
      // Test if various operations are supported by trying to call them
      try { builder.relu(testTensor); features.push('relu'); } catch {}
      try { builder.sigmoid(testTensor); features.push('sigmoid'); } catch {}
      try { builder.tanh(testTensor); features.push('tanh'); } catch {}
      try { builder.add(testTensor, testTensor); features.push('add'); } catch {}
      try { builder.matmul(testTensor, testTensor); features.push('matmul'); } catch {}
      try { builder.conv2d(testTensor, testTensor, { padding: [0, 0, 0, 0] }); features.push('conv2d'); } catch {}
      
      return features;
    } catch (error) {
      console.warn('Error detecting WebNN features:', error);
      return [];
    }
  }

  /**
   * Detect WebAssembly capabilities
   */
  private detectWasmCapabilities(): any {
    try {
      // Check basic WebAssembly support
      if (typeof WebAssembly !== 'object') {
        return { supported: false };
      }
      
      // Check SIMD support
      const simdSupported = WebAssembly.validate(new Uint8Array([
        0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3,
        2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
      ]));
      
      // Check threads support
      const threadsSupported = typeof SharedArrayBuffer === 'function';
      
      return {
        supported: true,
        simd: simdSupported,
        threads: threadsSupported
      };
    } catch (error) {
      console.warn('Error detecting WebAssembly capabilities:', error);
      return { supported: false };
    }
  }

  /**
   * Detect if adapter is simulated
   */
  private detectSimulatedAdapter(adapterInfo: GPUAdapterInfo): boolean {
    // Common patterns for simulated/software adapters
    const softwarePatterns = [
      'swiftshader',
      'llvmpipe',
      'software',
      'basic',
      'lavapipe',
      'microsoft basic'
    ];
    
    const vendor = (adapterInfo.vendor || '').toLowerCase();
    const device = (adapterInfo.device || '').toLowerCase();
    const description = (adapterInfo.description || '').toLowerCase();
    
    // Check if any software pattern is in the adapter info
    return softwarePatterns.some(pattern => 
      vendor.includes(pattern) || 
      device.includes(pattern) || 
      description.includes(pattern)
    );
  }

  /**
   * Detect if WebNN is simulated
   */
  private detectSimulatedWebNN(deviceName: string | null): boolean {
    if (!deviceName) {
      return false;
    }
    
    // Common patterns for simulated devices
    const softwarePatterns = [
      'swiftshader',
      'llvmpipe',
      'software',
      'basic',
      'emulation',
      'reference',
      'microsoft basic'
    ];
    
    const deviceLower = deviceName.toLowerCase();
    return softwarePatterns.some(pattern => deviceLower.includes(pattern));
  }

  /**
   * Determine the optimal backend based on capabilities
   */
  private determineOptimalBackend(
    webgpuCapabilities: any,
    webnnCapabilities: any,
    wasmCapabilities: any
  ): HardwareBackendType {
    // Order of preference depends on browser
    if (!this.browserInfo) {
      // Default preference order
      if (webgpuCapabilities.supported && !webgpuCapabilities.isSimulated) {
        return 'webgpu';
      } else if (webnnCapabilities.supported && !webnnCapabilities.isSimulated) {
        return 'webnn';
      } else if (wasmCapabilities.supported) {
        return 'wasm';
      } else {
        return 'cpu';
      }
    }
    
    const browser = this.browserInfo.name;
    
    switch (browser) {
      case 'edge':
        // Edge has good WebNN support
        if (webnnCapabilities.supported && !webnnCapabilities.isSimulated) {
          return 'webnn';
        } else if (webgpuCapabilities.supported && !webgpuCapabilities.isSimulated) {
          return 'webgpu';
        } else if (wasmCapabilities.supported) {
          return 'wasm';
        }
        break;
        
      case 'chrome':
        // Chrome has good WebGPU support
        if (webgpuCapabilities.supported && !webgpuCapabilities.isSimulated) {
          return 'webgpu';
        } else if (webnnCapabilities.supported && !webnnCapabilities.isSimulated) {
          return 'webnn';
        } else if (wasmCapabilities.supported) {
          return 'wasm';
        }
        break;
        
      case 'firefox':
        // Firefox has good WebGPU support
        if (webgpuCapabilities.supported && !webgpuCapabilities.isSimulated) {
          return 'webgpu';
        } else if (wasmCapabilities.supported) {
          return 'wasm';
        } else if (webnnCapabilities.supported && !webnnCapabilities.isSimulated) {
          return 'webnn';
        }
        break;
        
      case 'safari':
        // Safari has limited WebGPU support
        if (webgpuCapabilities.supported && !webgpuCapabilities.isSimulated) {
          return 'webgpu';
        } else if (webnnCapabilities.supported && !webnnCapabilities.isSimulated) {
          return 'webnn';
        } else if (wasmCapabilities.supported) {
          return 'wasm';
        }
        break;
        
      default:
        // Default preference order
        if (webgpuCapabilities.supported && !webgpuCapabilities.isSimulated) {
          return 'webgpu';
        } else if (webnnCapabilities.supported && !webnnCapabilities.isSimulated) {
          return 'webnn';
        } else if (wasmCapabilities.supported) {
          return 'wasm';
        }
    }
    
    // Fallback to CPU
    return 'cpu';
  }

  /**
   * Get browser information
   */
  getBrowserInfo(): BrowserInfo | null {
    return this.browserInfo;
  }

  /**
   * Get optimal backend for a specific model type
   */
  async getOptimalBackend(modelType: 'text' | 'vision' | 'audio' | 'multimodal'): Promise<HardwareBackendType> {
    // Make sure capabilities are detected
    const capabilities = await this.detectCapabilities();
    
    // Browser-specific optimizations
    const browser = capabilities.browserInfo.name;
    
    // Fine-tune based on model type and browser
    if (modelType === 'audio' && browser === 'firefox' && capabilities.webgpu.supported) {
      // Firefox has optimized audio compute shaders
      return 'webgpu';
    } else if (modelType === 'text' && browser === 'edge' && capabilities.webnn.supported) {
      // Edge has good WebNN text model support
      return 'webnn';
    } else if (modelType === 'vision' && capabilities.webgpu.supported) {
      // Vision models generally work best on WebGPU
      return 'webgpu';
    }
    
    // Default to the general optimal backend
    return capabilities.optimalBackend;
  }

  /**
   * Get browser-specific optimizations
   */
  async getBrowserOptimizations(
    modelType: 'text' | 'vision' | 'audio' | 'multimodal',
    backend: HardwareBackendType
  ): Promise<any> {
    // Make sure capabilities are detected
    const capabilities = await this.detectCapabilities();
    
    const browser = capabilities.browserInfo.name;
    const result: any = {
      browser,
      modelType,
      backend,
      optimizations: {}
    };
    
    // Common optimizations
    result.optimizations.shaderPrecompilation = true;
    
    // Browser-specific optimizations
    switch (browser) {
      case 'firefox':
        // Firefox-specific optimizations
        if (backend === 'webgpu') {
          result.optimizations.useCustomWorkgroups = true;
          result.optimizations.audioComputeShaders = modelType === 'audio';
          result.optimizations.reduceBarrierSynchronization = true;
          result.optimizations.aggressiveBufferReuse = true;
          
          if (modelType === 'audio') {
            result.optimizations.preferredShaderFormat = 'firefox_optimized';
            result.optimizations.audioWorkgroupSize = [8, 8, 1];
          }
        }
        break;
        
      case 'chrome':
        // Chrome-specific optimizations
        if (backend === 'webgpu') {
          result.optimizations.useAsyncCompile = true;
          result.optimizations.batchedOperations = true;
          result.optimizations.useBindGroupLayoutCache = true;
          
          if (modelType === 'vision') {
            result.optimizations.preferredShaderFormat = 'chrome_optimized';
            result.optimizations.visionWorkgroupSize = [16, 16, 1];
          }
        }
        break;
        
      case 'edge':
        // Edge-specific optimizations
        if (backend === 'webnn') {
          result.optimizations.useOperationFusion = true;
          result.optimizations.useHardwareDetection = true;
        }
        break;
        
      case 'safari':
        // Safari-specific optimizations
        if (backend === 'webgpu') {
          result.optimizations.conservativeWorkgroupSizes = true;
          result.optimizations.simplifiedShaders = true;
          result.optimizations.powerEfficient = true;
        }
        break;
    }
    
    return result;
  }

  /**
   * Initialize WebGPU context
   */
  async initializeWebGPU(): Promise<{
    adapter: GPUAdapter;
    device: GPUDevice;
    adapterInfo: GPUAdapterInfo;
  } | null> {
    try {
      // Check if WebGPU is available
      if (!('gpu' in navigator)) {
        return null;
      }
      
      // Request adapter
      const adapter = await navigator.gpu.requestAdapter();
      
      if (!adapter) {
        return null;
      }
      
      // Get adapter info
      const adapterInfo = await adapter.requestAdapterInfo();
      
      // Request device
      const device = await adapter.requestDevice();
      
      return {
        adapter,
        device,
        adapterInfo
      };
    } catch (error) {
      console.error('Failed to initialize WebGPU:', error);
      return null;
    }
  }

  /**
   * Load a shader module for the current browser
   */
  async loadOptimizedShader(
    device: GPUDevice,
    shaderPath: string,
    modelType: string
  ): Promise<GPUShaderModule | null> {
    if (!this.browserInfo) {
      throw new Error('Browser information not available');
    }
    
    try {
      // Determine browser-specific shader path
      const browser = this.browserInfo.name;
      const browserPath = `${browser}_optimized`;
      
      // Fetch shader code from the appropriate path
      const fullPath = `${shaderPath}/${browserPath}_${modelType}.wgsl`;
      const response = await fetch(fullPath);
      
      if (!response.ok) {
        // Try fallback to generic shader
        console.warn(`Browser-specific shader not found at ${fullPath}, using generic shader`);
        const genericPath = `${shaderPath}/generic_${modelType}.wgsl`;
        const genericResponse = await fetch(genericPath);
        
        if (!genericResponse.ok) {
          throw new Error(`Failed to load shader from ${genericPath}`);
        }
        
        const shaderCode = await genericResponse.text();
        return device.createShaderModule({ code: shaderCode });
      }
      
      const shaderCode = await response.text();
      return device.createShaderModule({ code: shaderCode });
    } catch (error) {
      console.error('Failed to load optimized shader:', error);
      return null;
    }
  }

  /**
   * Get shader modification hints for the current browser
   */
  getShaderModificationHints(shaderType: string): any {
    if (!this.browserInfo) {
      return {};
    }
    
    const browser = this.browserInfo.name;
    const hints: any = {};
    
    switch (browser) {
      case 'firefox':
        hints.minimalControlFlow = true;
        hints.reduceBarrierSynchronization = true;
        hints.preferUnrolledLoops = true;
        hints.aggressiveWorkgroupSize = true;
        break;
        
      case 'chrome':
        hints.useAsyncCompile = true;
        hints.useBindGroupCache = true;
        break;
        
      case 'safari':
        hints.simplifyShaders = true;
        hints.conservativeWorkgroups = true;
        hints.avoidAtomics = true;
        break;
    }
    
    // Shader-specific hints
    if (shaderType === 'matmul_4bit') {
      switch (browser) {
        case 'firefox':
          hints.workgroupSize = [8, 8, 1];
          hints.preferDirectBitwiseOps = true;
          break;
          
        case 'chrome':
          hints.workgroupSize = [16, 16, 1];
          break;
          
        case 'safari':
          hints.workgroupSize = [4, 4, 1];
          break;
      }
    } else if (shaderType === 'audio_processing') {
      switch (browser) {
        case 'firefox':
          hints.specializedAudioPath = true;
          hints.fixedWorkgroupSize = true;
          break;
          
        case 'chrome':
          hints.optimalAudioBlockSize = 256;
          break;
      }
    }
    
    return hints;
  }
}