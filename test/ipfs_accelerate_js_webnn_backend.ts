/**
 * WebNN Backend Implementation
 * 
 * This file demonstrates the initial migration of webnn_implementation.py to TypeScript
 */

export interface WebNNBackendOptions {
  /** Enable verbose logging */
  logging?: boolean;
  /** Preferred device type */
  devicePreference?: 'gpu' | 'cpu' | undefined;
  /** Power preference */
  powerPreference?: 'default' | 'high-performance' | 'low-power';
}

export interface WebNNInfo {
  supported: boolean;
  deviceType: string | null;
  deviceName: string | null;
  isSimulated: boolean;
}

/**
 * WebNN Backend implementation for hardware acceleration
 */
export class WebNNBackend {
  private context: MLContext | null = null;
  private device: string | null = null;
  private deviceName: string | null = null;
  private initialized: boolean = false;
  private options: WebNNBackendOptions;

  constructor(options: WebNNBackendOptions = {}) {
    this.options = {
      logging: false,
      devicePreference: 'gpu',
      powerPreference: 'high-performance',
      ...options
    };
  }

  /**
   * Initialize the WebNN backend and acquire context
   */
  async initialize(): Promise<boolean> {
    try {
      // Check if WebNN is supported
      if (!('ml' in navigator)) {
        throw new Error('WebNN is not supported in this browser');
      }

      // Create context with options
      this.context = await (navigator as any).ml.createContext({
        deviceType: this.options.devicePreference,
        powerPreference: this.options.powerPreference
      });

      if (!this.context) {
        throw new Error('Failed to create WebNN context');
      }

      // Get device info
      this.device = (this.context as any).deviceType || null;
      this.deviceName = await this.getDeviceName();

      // Log info if logging is enabled
      if (this.options.logging) {
        console.log('WebNN Info:', {
          device: this.device,
          deviceName: this.deviceName
        });
      }

      this.initialized = true;
      return true;
    } catch (error) {
      console.error('WebNN initialization failed:', error);
      return false;
    }
  }

  /**
   * Attempt to get the device name (implementation varies by browser)
   */
  private async getDeviceName(): Promise<string | null> {
    try {
      // This is a best-effort attempt as WebNN API doesn't standardize this
      // Edge might expose this information differently than other browsers
      
      // Try to access device name (implementation varies)
      const deviceInfo = (this.context as any).deviceInfo;
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
   * Check if the backend is using a real hardware device
   */
  isRealHardware(): boolean {
    // We don't have a standardized way to detect simulation in WebNN yet
    // This is a best-effort heuristic
    if (!this.device || !this.deviceName) {
      return false;
    }
    
    // Common patterns for simulated/software adapters
    const softwarePatterns = [
      'swiftshader',
      'llvmpipe',
      'software',
      'basic',
      'emulation',
      'reference',
      'microsoft basic'
    ];
    
    const deviceLower = this.deviceName.toLowerCase();
    return !softwarePatterns.some(pattern => deviceLower.includes(pattern));
  }

  /**
   * Get the WebNN context
   */
  getContext(): MLContext | null {
    if (!this.initialized) {
      throw new Error('WebNN backend not initialized');
    }
    return this.context;
  }

  /**
   * Get WebNN device type
   */
  getDeviceType(): string | null {
    return this.device;
  }

  /**
   * Get WebNN information
   */
  getInfo(): WebNNInfo {
    return {
      supported: this.initialized,
      deviceType: this.device,
      deviceName: this.deviceName,
      isSimulated: !this.isRealHardware()
    };
  }

  /**
   * Create a tensor with specified data and dimensions
   */
  createTensor(
    data: Float32Array | Int32Array | Uint8Array,
    dimensions: number[]
  ): MLOperand | null {
    if (!this.context) {
      throw new Error('WebNN context not available');
    }
    
    try {
      // Determine data type
      let dataType: 'float32' | 'int32' | 'uint8';
      if (data instanceof Float32Array) {
        dataType = 'float32';
      } else if (data instanceof Int32Array) {
        dataType = 'int32';
      } else if (data instanceof Uint8Array) {
        dataType = 'uint8';
      } else {
        throw new Error('Unsupported data type');
      }
      
      // Create a tensor descriptor
      const desc: MLOperandDescriptor = {
        type: dataType,
        dimensions
      };
      
      // Create tensor with data
      return this.context.createOperand(desc, data);
    } catch (error) {
      console.error('Failed to create tensor:', error);
      return null;
    }
  }

  /**
   * Build and compile a WebNN graph
   */
  async buildGraph(
    buildFunction: (builder: MLGraphBuilder) => [MLOperand[], MLOperand[]]
  ): Promise<MLGraph | null> {
    if (!this.context) {
      throw new Error('WebNN context not available');
    }
    
    try {
      // Create graph builder
      const builder = new MLGraphBuilder(this.context);
      
      // Build the graph
      const [inputs, outputs] = buildFunction(builder);
      
      // Compile the graph
      return await builder.build({
        inputs,
        outputs
      });
    } catch (error) {
      console.error('Failed to build WebNN graph:', error);
      return null;
    }
  }

  /**
   * Execute a compiled WebNN graph
   */
  async runGraph(
    graph: MLGraph,
    inputs: Record<string, MLOperand>,
    outputs: Record<string, MLOperand>
  ): Promise<Record<string, MLOperand>> {
    try {
      // Execute the graph
      const results = await graph.compute(inputs, outputs);
      return results;
    } catch (error) {
      console.error('Failed to run WebNN graph:', error);
      throw error;
    }
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.context = null;
    this.device = null;
    this.deviceName = null;
    this.initialized = false;
  }
}

/**
 * Helper function to detect if WebNN is supported
 */
export async function isWebNNSupported(): Promise<boolean> {
  try {
    if (!('ml' in navigator)) {
      return false;
    }
    
    const context = await (navigator as any).ml.createContext();
    return !!context;
  } catch (error) {
    return false;
  }
}

/**
 * Helper function to get detailed WebNN support information
 */
export async function getWebNNInfo(): Promise<{
  supported: boolean;
  deviceType?: string;
  deviceName?: string;
  isSimulated?: boolean;
  features?: string[];
}> {
  try {
    if (!('ml' in navigator)) {
      return { supported: false };
    }
    
    const backend = new WebNNBackend({ logging: false });
    const initialized = await backend.initialize();
    
    if (!initialized) {
      return { supported: false };
    }
    
    const info = backend.getInfo();
    
    // Attempt to detect supported operations (varies by implementation)
    const features: string[] = [];
    try {
      const context = backend.getContext();
      if (context) {
        const builder = new MLGraphBuilder(context);
        
        // Test if various operations are supported by trying to call them
        // This is a best-effort approach as there's no standardized capability query
        const testTensor = backend.createTensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
        if (testTensor) {
          try { builder.relu(testTensor); features.push('relu'); } catch {}
          try { builder.sigmoid(testTensor); features.push('sigmoid'); } catch {}
          try { builder.tanh(testTensor); features.push('tanh'); } catch {}
          try { builder.add(testTensor, testTensor); features.push('add'); } catch {}
          try { builder.matmul(testTensor, testTensor); features.push('matmul'); } catch {}
        }
      }
    } catch (error) {
      console.warn('Failed to detect WebNN features:', error);
    }
    
    return {
      supported: true,
      deviceType: info.deviceType || undefined,
      deviceName: info.deviceName || undefined,
      isSimulated: info.isSimulated,
      features
    };
  } catch (error) {
    return { supported: false };
  }
}

/**
 * TypeScript type definitions for WebNN API
 * Note: These are added here since TypeScript doesn't include WebNN types yet
 */
interface MLContext {
  readonly deviceType?: string;
  createOperand(descriptor: MLOperandDescriptor, bufferView?: ArrayBufferView): MLOperand;
}

interface MLOperandDescriptor {
  type: 'float32' | 'int32' | 'uint8';
  dimensions: number[];
}

interface MLOperand {}

interface MLGraph {
  compute(inputs: Record<string, MLOperand>, outputs: Record<string, MLOperand>): Promise<Record<string, MLOperand>>;
}

interface MLGraphBuilder {
  new(context: MLContext): MLGraphBuilder;
  input(name: string, descriptor: MLOperandDescriptor): MLOperand;
  constant(descriptor: MLOperandDescriptor, bufferView: ArrayBufferView): MLOperand;
  relu(input: MLOperand): MLOperand;
  sigmoid(input: MLOperand): MLOperand;
  tanh(input: MLOperand): MLOperand;
  add(a: MLOperand, b: MLOperand): MLOperand;
  sub(a: MLOperand, b: MLOperand): MLOperand;
  mul(a: MLOperand, b: MLOperand): MLOperand;
  div(a: MLOperand, b: MLOperand): MLOperand;
  matmul(a: MLOperand, b: MLOperand): MLOperand;
  build(options: { inputs: MLOperand[]; outputs: MLOperand[] }): Promise<MLGraph>;
}