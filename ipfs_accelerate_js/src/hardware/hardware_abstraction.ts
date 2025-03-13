/**
 * Hardware abstraction layer for IPFS Accelerate
 */
import { HardwareBackend, HardwarePreferences, Model } from './interfaces';
import { WebGPUBackend } from './hardware/backends/webgpu_backend';
import { WebNNBackend } from './hardware/backends/webnn_backend';
import { CPUBackend } from './hardware/backends/cpu_backend';
import { detectHardwareCapabilities } from './hardware/detection/hardware_detection';

export class HardwareAbstraction {
  private backends: Map<string, HardwareBackend> = new Map();
  private preferences: HardwarePreferences;
  private backendOrder: string[] = [];

  constructor(preferences: Partial<HardwarePreferences> = {}) {
    this.preferences = {
      backendOrder: preferences.backendOrder || ['webgpu', 'webnn', 'wasm', 'cpu'],
      modelPreferences: preferences.modelPreferences || {},
      options: preferences.options || {}
    };
  }

  async initialize(): Promise<boolean> {
    try {
      // Initialize hardware detection
      const capabilities = await detectHardwareCapabilities();
      
      // Initialize backends based on available hardware
      if (capabilities.webgpuSupported) {
        const webgpuBackend = new WebGPUBackend();
        const success = await webgpuBackend.initialize();
        if (success) {
          this.backends.set('webgpu', webgpuBackend);
        }
      }
      
      if (capabilities.webnnSupported) {
        const webnnBackend = new WebNNBackend();
        const success = await webnnBackend.initialize();
        if (success) {
          this.backends.set('webnn', webnnBackend);
        }
      }
      
      // Always add CPU backend as fallback
      const cpuBackend = new CPUBackend();
      await cpuBackend.initialize();
      this.backends.set('cpu', cpuBackend);
      
      // Apply hardware preferences
      this.applyPreferences();
      
      return this.backends.size > 0;
    } catch (error) {
      console.error("Error initializing hardware abstraction:", error);
      return false;
    }
  }

  async getPreferredBackend(modelType: string): Promise<HardwareBackend | null> {
    // Implementation would determine the best backend for the model type
    // Check if we have a preference for this model type
    if (
      this.preferences &&
      this.preferences.modelPreferences &&
      this.preferences.modelPreferences[modelType]
    ) {
      const preferredBackend = this.preferences.modelPreferences[modelType];
      if (this.backends.has(preferredBackend)) {
        return this.backends.get(preferredBackend)!;
      }
    }
    
    // Try each backend in order of preference
    for (const backendName of this.backendOrder) {
      if (this.backends.has(backendName)) {
        return this.backends.get(backendName)!;
      }
    }
    
    // Fallback to any available backend
    if (this.backends.size > 0) {
      return this.backends.values().next().value;
    }
    
    return null;
  }

  async execute<T = any, U = any>(inputs: T, modelType: string): Promise<U> {
    const backend = await this.getPreferredBackend(modelType);
    if (!backend) {
      throw new Error(`No suitable backend found for model type: ${modelType}`);
    }

    if (!backend.execute) {
      throw new Error(`Backend does not implement execute method`);
    }

    return backend.execute<T, U>(inputs);
  }

  async runModel<T = any, U = any>(model: Model, inputs: T): Promise<U> {
    const backend = await this.getPreferredBackend(model.type);
    if (!backend) {
      throw new Error(`No suitable backend found for model type: ${model.type}`);
    }
    
    return model.execute(inputs) as Promise<U>;
  }

  dispose(): void {
    // Clean up resources
    for (const backend of this.backends.values()) {
      backend.destroy();
    }
    this.backends.clear();
    this.backendOrder = [];
  }
  
  private applyPreferences(): void {
    // Apply any hardware preferences from configuration
    if (this.preferences && this.preferences.backendOrder) {
      // Reorder backends based on preferences
      this.backendOrder = this.preferences.backendOrder.filter(
        backend => this.backends.has(backend)
      );
    } else {
      // Default order: WebGPU > WebNN > CPU
      this.backendOrder = ['webgpu', 'webnn', 'wasm', 'cpu'].filter(
        backend => this.backends.has(backend)
      );
    }
  }
}
