/**
 * Device Mapper utility for API backends
 * 
 * This utility helps map model parts to specific devices,
 * implement various mapping strategies, and optimize device
 * mapping based on model architecture.
 */

export interface DeviceInfo {
  type: string;
  name: string;
  memory: number; // Memory in MB
  id?: string;
  index?: number;
  isAvailable?: boolean;
}

export interface ModelLayerInfo {
  name: string;
  memoryRequirement: number; // Memory in MB
  dependencies?: string[];
  type?: string;
}

export interface MappingStrategy {
  name: string;
  description: string;
  allocateDevices: (
    layers: ModelLayerInfo[],
    availableDevices: DeviceInfo[]
  ) => Map<string, DeviceInfo>;
}

export class DeviceMapper {
  private availableDevices: DeviceInfo[] = [];
  private mappingStrategies: Map<string, MappingStrategy> = new Map();

  constructor() {
    // Register default strategies
    this.registerStrategy({
      name: 'single_device',
      description: 'Map all layers to a single device',
      allocateDevices: (layers, devices) => {
        if (devices.length === 0) {
          throw new Error('No devices available for mapping');
        }

        // Find the device with the most memory
        const device = devices.reduce((best, current) => 
          current.memory > best.memory ? current : best, devices[0]);
        
        // Map all layers to this device
        const mapping = new Map<string, DeviceInfo>();
        for (const layer of layers) {
          mapping.set(layer.name, device);
        }
        
        return mapping;
      }
    });

    this.registerStrategy({
      name: 'balanced',
      description: 'Balance layers across available devices by memory usage',
      allocateDevices: (layers, devices) => {
        if (devices.length === 0) {
          throw new Error('No devices available for mapping');
        }

        // Sort layers by memory requirement (descending)
        const sortedLayers = [...layers].sort((a, b) => 
          b.memoryRequirement - a.memoryRequirement);
        
        // Initialize device memory usage tracking
        const deviceMemoryUsage = new Map<string, number>();
        devices.forEach(device => deviceMemoryUsage.set(device.name, 0));
        
        // Map layers to devices
        const mapping = new Map<string, DeviceInfo>();
        
        for (const layer of sortedLayers) {
          // Find device with minimum current memory usage
          let minUsageDevice = devices[0];
          let minUsage = deviceMemoryUsage.get(minUsageDevice.name) || 0;
          
          for (const device of devices) {
            const usage = deviceMemoryUsage.get(device.name) || 0;
            if (usage < minUsage) {
              minUsage = usage;
              minUsageDevice = device;
            }
          }
          
          // Assign layer to device
          mapping.set(layer.name, minUsageDevice);
          
          // Update memory usage
          deviceMemoryUsage.set(
            minUsageDevice.name,
            (deviceMemoryUsage.get(minUsageDevice.name) || 0) + layer.memoryRequirement
          );
        }
        
        return mapping;
      }
    });

    this.registerStrategy({
      name: 'pipeline_parallel',
      description: 'Distribute layers across devices in pipeline fashion',
      allocateDevices: (layers, devices) => {
        if (devices.length === 0) {
          throw new Error('No devices available for mapping');
        }

        // Simple round-robin assignment for pipeline parallelism
        const mapping = new Map<string, DeviceInfo>();
        
        layers.forEach((layer, index) => {
          const deviceIndex = index % devices.length;
          mapping.set(layer.name, devices[deviceIndex]);
        });
        
        return mapping;
      }
    });
  }

  /**
   * Detect available GPU devices in the browser environment
   */
  async detectAvailableDevices(): Promise<DeviceInfo[]> {
    const devices: DeviceInfo[] = [];
    
    // For browser context, we can check for WebGPU and WebNN support
    if (typeof navigator !== 'undefined') {
      try {
        // Check for WebGPU support
        if ('gpu' in navigator) {
          const adapter = await (navigator as any).gpu.requestAdapter();
          if (adapter) {
            const adapterInfo = await adapter.requestAdapterInfo();
            const device = await adapter.requestDevice();
            if (device) {
              devices.push({
                type: 'webgpu',
                name: adapterInfo.description || 'WebGPU Device',
                memory: 1024, // Estimated memory size, not always accurate
                isAvailable: true
              });
            }
          }
        }
        
        // Check for WebNN support
        if ('ml' in navigator) {
          const contexts = await (navigator as any).ml.getNeuralNetworkContexts();
          if (contexts && contexts.length > 0) {
            contexts.forEach((context: any, index: number) => {
              devices.push({
                type: 'webnn',
                name: `WebNN Device ${index}`,
                memory: 1024, // Estimated memory size
                isAvailable: true
              });
            });
          }
        }
      } catch (error) {
        console.warn('Error detecting browser devices:', error);
      }
    }
    
    // Always add CPU as a fallback
    devices.push({
      type: 'cpu',
      name: 'CPU',
      memory: 4096, // Assumed available memory
      isAvailable: true
    });
    
    this.availableDevices = devices;
    return devices;
  }

  /**
   * Register a new mapping strategy
   */
  registerStrategy(strategy: MappingStrategy): void {
    this.mappingStrategies.set(strategy.name, strategy);
  }

  /**
   * Get a mapping strategy by name
   */
  getStrategy(name: string): MappingStrategy | undefined {
    return this.mappingStrategies.get(name);
  }

  /**
   * List all available mapping strategies
   */
  listStrategies(): MappingStrategy[] {
    return Array.from(this.mappingStrategies.values());
  }

  /**
   * Map model layers to devices using the specified strategy
   */
  mapLayersToDevices(
    layers: ModelLayerInfo[],
    strategyName: string = 'balanced',
    devices?: DeviceInfo[]
  ): Map<string, DeviceInfo> {
    const strategy = this.mappingStrategies.get(strategyName);
    if (!strategy) {
      throw new Error(`Unknown mapping strategy: ${strategyName}`);
    }
    
    const availableDevices = devices || this.availableDevices;
    if (availableDevices.length === 0) {
      throw new Error('No devices available for mapping');
    }
    
    return strategy.allocateDevices(layers, availableDevices);
  }

  /**
   * Estimate memory requirements for a layer based on its type and parameters
   */
  estimateLayerMemory(
    layerType: string,
    inputShape: number[],
    outputShape: number[],
    parameters: number
  ): number {
    // Very simplified estimation
    let memoryEstimate = 0;
    
    // Memory for parameters (4 bytes per parameter for float32)
    memoryEstimate += parameters * 4;
    
    // Memory for input and output tensors
    const inputSize = inputShape.reduce((a, b) => a * b, 1) * 4; // float32
    const outputSize = outputShape.reduce((a, b) => a * b, 1) * 4; // float32
    
    memoryEstimate += inputSize + outputSize;
    
    // Additional memory based on layer type
    switch (layerType) {
      case 'attention':
        // Attention typically needs more memory for intermediate states
        memoryEstimate += inputSize * 3; // Q, K, V projections
        break;
      case 'convolution':
        // Conv layers need memory for intermediate feature maps
        memoryEstimate += inputSize * 2;
        break;
      default:
        // Default extra buffer
        memoryEstimate += inputSize * 0.5;
    }
    
    // Return memory in MB
    return memoryEstimate / (1024 * 1024);
  }
}