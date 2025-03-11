/**
 * WebGPU Backend Implementation
 * 
 * This file demonstrates the initial migration of webgpu_implementation.py to TypeScript
 */

export interface WebGPUBackendOptions {
  /** Enable verbose logging */
  logging?: boolean;
  /** Preferred adapter features */
  requiredFeatures?: GPUFeatureName[];
  /** Optional device descriptor */
  deviceDescriptor?: GPUDeviceDescriptor;
}

export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isSimulated: boolean;
}

/**
 * WebGPU Backend implementation for hardware acceleration
 */
export class WebGPUBackend {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private adapterInfo: AdapterInfo | null = null;
  private initialized: boolean = false;
  private options: WebGPUBackendOptions;

  constructor(options: WebGPUBackendOptions = {}) {
    this.options = {
      logging: false,
      requiredFeatures: [],
      ...options
    };
  }

  /**
   * Initialize the WebGPU backend and acquire adapter and device
   */
  async initialize(): Promise<boolean> {
    try {
      // Check if WebGPU is supported
      if (!navigator.gpu) {
        throw new Error('WebGPU is not supported in this browser');
      }

      // Request adapter
      this.adapter = await navigator.gpu.requestAdapter();
      if (!this.adapter) {
        throw new Error('Failed to get WebGPU adapter');
      }

      // Get adapter info
      const adapterInfo = await this.adapter.requestAdapterInfo();
      this.adapterInfo = {
        vendor: adapterInfo.vendor || 'unknown',
        architecture: adapterInfo.architecture || 'unknown',
        device: adapterInfo.device || 'unknown',
        description: adapterInfo.description || 'unknown',
        isSimulated: this.detectSimulatedAdapter(adapterInfo)
      };

      // Log adapter info if logging is enabled
      if (this.options.logging) {
        console.log('WebGPU Adapter Info:', this.adapterInfo);
      }

      // Create device
      const deviceDescriptor: GPUDeviceDescriptor = {
        ...this.options.deviceDescriptor,
        requiredFeatures: this.options.requiredFeatures || []
      };

      this.device = await this.adapter.requestDevice(deviceDescriptor);
      
      // Setup error handling
      this.device.addEventListener('uncapturederror', (event) => {
        console.error('WebGPU device error:', event.error);
      });

      this.initialized = true;
      return true;
    } catch (error) {
      console.error('WebGPU initialization failed:', error);
      return false;
    }
  }

  /**
   * Detect if the adapter is simulated (software rendering)
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
   * Get the WebGPU adapter
   */
  getAdapter(): GPUAdapter | null {
    return this.adapter;
  }

  /**
   * Get the WebGPU device
   */
  getDevice(): GPUDevice | null {
    if (!this.initialized) {
      throw new Error('WebGPU backend not initialized');
    }
    return this.device;
  }

  /**
   * Get adapter information
   */
  getAdapterInfo(): AdapterInfo | null {
    return this.adapterInfo;
  }

  /**
   * Check if the backend is using a real hardware adapter
   */
  isRealHardware(): boolean {
    return this.adapterInfo ? !this.adapterInfo.isSimulated : false;
  }

  /**
   * Create a shader module from WGSL source
   */
  createShaderModule(code: string): GPUShaderModule | null {
    if (!this.device) {
      throw new Error('WebGPU device not available');
    }
    
    try {
      return this.device.createShaderModule({ code });
    } catch (error) {
      console.error('Failed to create shader module:', error);
      return null;
    }
  }

  /**
   * Create a buffer with specified data and usage
   */
  createBuffer(data: BufferSource, usage: GPUBufferUsageFlags): GPUBuffer | null {
    if (!this.device) {
      throw new Error('WebGPU device not available');
    }
    
    try {
      const buffer = this.device.createBuffer({
        size: data.byteLength,
        usage,
        mappedAtCreation: true
      });
      
      // Copy data to buffer
      const arrayBuffer = buffer.getMappedRange();
      new Uint8Array(arrayBuffer).set(new Uint8Array(
        data.buffer, 
        data instanceof DataView ? data.byteOffset : 0, 
        data.byteLength
      ));
      
      buffer.unmap();
      return buffer;
    } catch (error) {
      console.error('Failed to create buffer:', error);
      return null;
    }
  }

  /**
   * Create a compute pipeline from a shader module
   */
  createComputePipeline(
    shaderModule: GPUShaderModule, 
    entryPoint: string = 'main'
  ): GPUComputePipeline | null {
    if (!this.device) {
      throw new Error('WebGPU device not available');
    }
    
    try {
      return this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint
        }
      });
    } catch (error) {
      console.error('Failed to create compute pipeline:', error);
      return null;
    }
  }

  /**
   * Run a compute shader with the given buffer bindings
   */
  async runComputeShader(
    pipeline: GPUComputePipeline,
    bindings: GPUBindingResource[],
    workgroupsX: number,
    workgroupsY: number = 1,
    workgroupsZ: number = 1
  ): Promise<void> {
    if (!this.device) {
      throw new Error('WebGPU device not available');
    }
    
    try {
      // Create bind group
      const bindGroupLayout = pipeline.getBindGroupLayout(0);
      const entries = bindings.map((resource, index) => ({
        binding: index,
        resource
      }));
      
      const bindGroup = this.device.createBindGroup({
        layout: bindGroupLayout,
        entries
      });
      
      // Create command encoder
      const commandEncoder = this.device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      
      computePass.setPipeline(pipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
      computePass.end();
      
      // Submit commands
      const commands = commandEncoder.finish();
      this.device.queue.submit([commands]);
      
      // Wait for GPU to finish
      await this.device.queue.onSubmittedWorkDone();
    } catch (error) {
      console.error('Failed to run compute shader:', error);
      throw error;
    }
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.device?.destroy();
    this.device = null;
    this.adapter = null;
    this.initialized = false;
  }
}

/**
 * Helper function to detect if WebGPU is supported
 */
export async function isWebGPUSupported(): Promise<boolean> {
  try {
    if (!navigator.gpu) {
      return false;
    }
    
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch (error) {
    return false;
  }
}

/**
 * Helper function to get detailed WebGPU support information
 */
export async function getWebGPUInfo(): Promise<{
  supported: boolean;
  adapterInfo?: AdapterInfo;
  features?: string[];
  limits?: Record<string, number>;
}> {
  try {
    if (!navigator.gpu) {
      return { supported: false };
    }
    
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { supported: false };
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
      supported: true,
      adapterInfo: {
        vendor: adapterInfo.vendor || 'unknown',
        architecture: adapterInfo.architecture || 'unknown',
        device: adapterInfo.device || 'unknown',
        description: adapterInfo.description || 'unknown',
        isSimulated: new WebGPUBackend().detectSimulatedAdapter(adapterInfo)
      },
      features,
      limits
    };
  } catch (error) {
    return { supported: false };
  }
}