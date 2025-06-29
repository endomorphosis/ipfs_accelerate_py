/**
 * Hardware Abstraction Layer for IPFS Accelerate TypeScript SDK
 * Provides a unified interface for using different hardware backends
 * (WebGPU, WebNN, WebAssembly, CPU) for AI model acceleration.
 */

import { HardwareCapabilities, detectHardwareCapabilities } from './hardware_detection';

/**
 * The hardware backend type: WebGPU, WebNN, WebAssembly, or CPU
 */
export type HardwareBackendType = 'webgpu' | 'webnn' | 'wasm' | 'cpu';

/**
 * Hardware backend interface
 * Represents a hardware acceleration backend (WebGPU, WebNN, etc.)
 */
export interface HardwareBackend<T = any> {
  /**
   * Backend type identifier
   */
  readonly type: HardwareBackendType;
  
  /**
   * Initialize the backend
   */
  initialize(): Promise<boolean>;
  
  /**
   * Check if the backend is supported on this device
   */
  isSupported(): Promise<boolean>;
  
  /**
   * Get backend-specific capabilities
   */
  getCapabilities(): Promise<Record<string, any>>;
  
  /**
   * Create tensor from data (for backend-specific optimizations)
   */
  createTensor(data: T, shape: number[], dataType?: string): Promise<any>;
  
  /**
   * Execute operation on the hardware backend
   */
  execute(operation: string, inputs: Record<string, any>, options?: Record<string, any>): Promise<any>;
  
  /**
   * Release resources
   */
  dispose(): void;
}

/**
 * Hardware abstraction options
 */
export interface HardwareAbstractionOptions {
  /**
   * Preferred order of backends to try
   */
  backendOrder?: HardwareBackendType[];
  
  /**
   * Model-specific backend preferences
   */
  modelPreferences?: Record<string, HardwareBackendType | 'auto'>;
  
  /**
   * Additional backend-specific options
   */
  backendOptions?: Record<HardwareBackendType, Record<string, any>>;
  
  /**
   * Enable automatic fallback to next backend if preferred one fails
   */
  autoFallback?: boolean;
  
  /**
   * Enable automatic backend selection based on model and hardware capabilities
   */
  autoSelection?: boolean;
}

/**
 * Hardware Abstraction Layer
 * Provides a unified interface for all hardware backends
 */
export class HardwareAbstraction {
  private options: HardwareAbstractionOptions;
  private backends: Map<HardwareBackendType, HardwareBackend> = new Map();
  private capabilities: HardwareCapabilities | null = null;
  private initialized = false;
  
  /**
   * Create a hardware abstraction layer
   */
  constructor(options: HardwareAbstractionOptions = {}) {
    this.options = {
      backendOrder: ['webgpu', 'webnn', 'wasm', 'cpu'],
      autoFallback: true,
      autoSelection: true,
      ...options
    };
  }
  
  /**
   * Initialize the hardware abstraction layer
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) return true;
    
    // Detect hardware capabilities
    this.capabilities = await detectHardwareCapabilities();
    
    // Initialize available backends in priority order
    for (const backendType of this.options.backendOrder!) {
      try {
        const backend = await this.createBackend(backendType);
        if (backend && await backend.isSupported()) {
          await backend.initialize();
          this.backends.set(backendType, backend);
        }
      } catch (error) {
        console.error(`Failed to initialize ${backendType} backend:`, error);
      }
    }
    
    this.initialized = this.backends.size > 0;
    return this.initialized;
  }
  
  /**
   * Create a backend instance
   */
  private async createBackend(type: HardwareBackendType): Promise<HardwareBackend | null> {
    const options = this.options.backendOptions?.[type] || {};
    
    switch (type) {
      case 'webgpu':
        try {
          const { WebGPUBackend } = await import('./ipfs_accelerate_js_webgpu_backend');
          return new WebGPUBackend(options);
        } catch (error) {
          console.error('Failed to load WebGPU backend:', error);
          return null;
        }
        
      case 'webnn':
        try {
          const { WebNNBackend } = await import('./ipfs_accelerate_js_webnn_backend');
          return new WebNNBackend(options);
        } catch (error) {
          console.error('Failed to load WebNN backend:', error);
          return null;
        }
        
      case 'wasm':
        try {
          // WASM backend not yet implemented, will be added in future updates
          console.warn('WASM backend not yet implemented, falling back to CPU');
          const { CPUBackend } = await import('./ipfs_accelerate_js_cpu_backend');
          return new CPUBackend(options);
        } catch (error) {
          console.error('Failed to load fallback CPU backend:', error);
          return null;
        }
        
      case 'cpu':
        try {
          const { CPUBackend } = await import('./ipfs_accelerate_js_cpu_backend');
          return new CPUBackend(options);
        } catch (error) {
          console.error('Failed to load CPU backend:', error);
          return null;
        }
        
      default:
        console.error(`Unknown backend type: ${type}`);
        return null;
    }
  }
  
  /**
   * Get the best backend for a specific model type
   */
  getBestBackend(modelType: string): HardwareBackend | null {
    if (!this.initialized) {
      throw new Error('Hardware abstraction layer not initialized');
    }
    
    // Check if there's a specific preference for this model type
    const preference = this.options.modelPreferences?.[modelType];
    
    if (preference && preference !== 'auto') {
      // Use the specified backend if available
      const backend = this.backends.get(preference);
      if (backend) return backend;
    }
    
    // Auto-select based on model type and hardware capabilities
    if (this.options.autoSelection) {
      if (modelType === 'audio' && this.backends.has('webgpu')) {
        // Audio models often perform best on WebGPU (especially on Firefox)
        return this.backends.get('webgpu')!;
      } else if (modelType === 'vision' && this.backends.has('webgpu')) {
        // Vision models generally do well on WebGPU
        return this.backends.get('webgpu')!;
      } else if (modelType === 'text' && this.backends.has('webnn')) {
        // Text models can benefit from WebNN's optimized matrix operations
        return this.backends.get('webnn')!;
      }
    }
    
    // Fallback to the first available backend in order of preference
    for (const backendType of this.options.backendOrder!) {
      const backend = this.backends.get(backendType);
      if (backend) return backend;
    }
    
    // Fallback to CPU as last resort
    return this.backends.get('cpu') || null;
  }
  
  /**
   * Execute operation using the best available backend
   */
  async execute<T = any, R = any>(
    operation: string,
    inputs: T,
    options: {
      modelType?: string;
      preferredBackend?: HardwareBackendType;
      backendOptions?: Record<string, any>;
    } = {}
  ): Promise<R> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    const modelType = options.modelType || 'generic';
    const preferredBackend = options.preferredBackend;
    
    // Try preferred backend first if specified
    if (preferredBackend) {
      const backend = this.backends.get(preferredBackend);
      if (backend) {
        try {
          return await backend.execute(operation, inputs as any, options.backendOptions) as R;
        } catch (error) {
          if (!this.options.autoFallback) {
            throw error;
          }
          console.warn(`Execution failed on ${preferredBackend}, trying fallback backends`);
        }
      }
    }
    
    // Get the best backend for this model type
    const backend = this.getBestBackend(modelType);
    if (!backend) {
      throw new Error('No suitable backend available');
    }
    
    // Execute using the selected backend
    return await backend.execute(operation, inputs as any, options.backendOptions) as R;
  }
  
  /**
   * Get hardware capabilities
   */
  getCapabilities(): HardwareCapabilities | null {
    return this.capabilities;
  }
  
  /**
   * Check if a specific backend is available
   */
  hasBackend(type: HardwareBackendType): boolean {
    return this.backends.has(type);
  }
  
  /**
   * Get a specific backend instance
   */
  getBackend(type: HardwareBackendType): HardwareBackend | null {
    return this.backends.get(type) || null;
  }
  
  /**
   * Get all available backends
   */
  getAvailableBackends(): HardwareBackendType[] {
    return Array.from(this.backends.keys());
  }
  
  /**
   * Release all resources
   */
  dispose(): void {
    for (const backend of this.backends.values()) {
      backend.dispose();
    }
    this.backends.clear();
    this.initialized = false;
  }
}

/**
 * Create a hardware abstraction layer and initialize it
 */
export async function createHardwareAbstraction(
  options: HardwareAbstractionOptions = {}
): Promise<HardwareAbstraction> {
  const hal = new HardwareAbstraction(options);
  await hal.initialize();
  return hal;
}