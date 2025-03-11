/**
 * IPFS Accelerate JavaScript SDK
 * 
 * Main entry point that provides access to all SDK functionality.
 */

import { HardwareAbstraction, createHardwareAbstraction, HardwareBackendType } from './ipfs_accelerate_js_hardware_abstraction';
import { WebGPUBackend, isWebGPUSupported, getWebGPUInfo } from './ipfs_accelerate_js_webgpu_backend';
import { WebNNBackend, isWebNNSupported, getWebNNInfo } from './ipfs_accelerate_js_webnn_backend';
import { ModelLoader, ModelLoadOptions, Model, ModelType, ModelConfig } from './ipfs_accelerate_js_model_loader';
import { QuantizationEngine, QuantizationConfig, UltraLowPrecisionEngine } from './ipfs_accelerate_js_quantization_engine';

// Export types
export { HardwareBackendType, ModelType, ModelConfig, QuantizationConfig };

export interface WebAcceleratorOptions {
  /** Enable automatic hardware detection */
  autoDetectHardware?: boolean;
  /** Preferred backend */
  preferredBackend?: HardwareBackendType;
  /** Fallback order for backends */
  fallbackOrder?: HardwareBackendType[];
  /** Enable P2P integration */
  enableP2P?: boolean;
  /** Store benchmark results */
  storeResults?: boolean;
  /** Enable logging */
  logging?: boolean;
  /** Default model configuration */
  defaultModelConfig?: ModelConfig;
  /** Quantization configuration */
  quantizationConfig?: QuantizationConfig;
}

export interface AccelerateOptions {
  /** Model ID to accelerate */
  modelId: string;
  /** Model type */
  modelType: ModelType;
  /** Input data for the model */
  input: any;
  /** Configuration for acceleration */
  config?: ModelConfig;
}

export interface AccelerationResult {
  /** Status of acceleration */
  status: 'success' | 'error';
  /** Processing time in milliseconds */
  processingTime: number;
  /** Throughput in items per second */
  throughput: number;
  /** Hardware backend used */
  hardware: HardwareBackendType;
  /** Memory usage in MB */
  memoryUsage: number;
  /** Result from model processing */
  result: any;
}

/**
 * Main WebAccelerator class for the IPFS Accelerate JavaScript SDK
 */
export class WebAccelerator {
  private hardware: HardwareAbstraction | null = null;
  private modelLoader: ModelLoader | null = null;
  private quantizationEngine: QuantizationEngine | null = null;
  private ultraLowPrecision: UltraLowPrecisionEngine | null = null;
  private storage: any | null = null; // To be implemented
  private p2pManager: any | null = null; // To be implemented
  private options: WebAcceleratorOptions;
  private isInitialized: boolean = false;

  constructor(options: WebAcceleratorOptions = {}) {
    this.options = {
      autoDetectHardware: true,
      preferredBackend: 'webgpu',
      fallbackOrder: ['webgpu', 'webnn', 'wasm', 'cpu'],
      enableP2P: false,
      storeResults: true,
      logging: false,
      ...options
    };
  }

  /**
   * Initialize the WebAccelerator
   */
  async initialize(): Promise<boolean> {
    try {
      // Create hardware abstraction
      this.hardware = await createHardwareAbstraction({
        logging: this.options.logging,
        preferredBackends: [
          this.options.preferredBackend!,
          ...this.options.fallbackOrder!.filter(b => b !== this.options.preferredBackend)
        ]
      });
      
      // Create model loader
      this.modelLoader = new ModelLoader(this.hardware);
      await this.modelLoader.initialize();
      
      // Create quantization engine
      const webgpuBackend = this.hardware.isBackendSupported('webgpu') ? 
        this.hardware.getWebGPUBackend() : null;
        
      const webnnBackend = this.hardware.isBackendSupported('webnn') ? 
        this.hardware.getWebNNBackend() : null;
        
      this.quantizationEngine = new QuantizationEngine({
        webgpuBackend,
        webnnBackend,
        useCache: true
      });
      await this.quantizationEngine.initialize();
      
      // Create ultra-low precision engine if WebGPU is available
      if (webgpuBackend) {
        this.ultraLowPrecision = new UltraLowPrecisionEngine(
          this.quantizationEngine,
          webgpuBackend
        );
      }
      
      // Initialize storage (to be implemented)
      if (this.options.storeResults) {
        // this.storage = new StorageManager();
        // await this.storage.initialize();
      }
      
      // Initialize P2P manager if enabled (to be implemented)
      if (this.options.enableP2P) {
        // this.p2pManager = new P2PManager();
        // await this.p2pManager.initialize();
      }
      
      this.isInitialized = true;
      return true;
    } catch (error) {
      console.error('Failed to initialize WebAccelerator:', error);
      return false;
    }
  }

  /**
   * Accelerate a model with the specified input
   */
  async accelerate(options: AccelerateOptions): Promise<AccelerationResult> {
    if (!this.isInitialized) {
      throw new Error('WebAccelerator not initialized');
    }
    
    const { modelId, modelType, input, config } = options;
    
    try {
      // Start timing
      const startTime = performance.now();
      
      // Determine backend
      const backend = config?.backend || 
        this.hardware?.getOptimalBackendForModel(modelType) || 
        this.options.preferredBackend;
      
      // Merge configurations
      const modelConfig: ModelConfig = {
        ...this.options.defaultModelConfig,
        ...config,
        backend
      };
      
      // Load the model
      const model = await this.modelLoader?.loadModel({
        modelId,
        modelType,
        backend: modelConfig.backend,
        config: modelConfig
      });
      
      if (!model) {
        throw new Error(`Failed to load model ${modelId}`);
      }
      
      // Process input based on model type
      let result: any;
      
      switch (modelType) {
        case 'text':
          if (typeof input === 'string') {
            result = await model.processText(input);
          } else {
            throw new Error('Invalid input for text model');
          }
          break;
          
        case 'vision':
          result = await model.processImage(input);
          break;
          
        case 'audio':
          result = await model.processAudio(input);
          break;
          
        case 'multimodal':
          // Handle based on input type
          if (typeof input === 'string') {
            result = await model.processText(input);
          } else if (input.image) {
            result = await model.processImage(input.image);
          } else if (input.text && input.image) {
            // Special multimodal processing
            result = await model.processMultimodal(input);
          } else {
            throw new Error('Invalid input for multimodal model');
          }
          break;
          
        default:
          throw new Error(`Unsupported model type: ${modelType}`);
      }
      
      // End timing
      const endTime = performance.now();
      const processingTime = endTime - startTime;
      
      // Calculate throughput (items per second)
      const throughput = 1000 / processingTime;
      
      // Get memory usage (placeholder)
      const memoryUsage = model.getInfo().memoryUsage || 0;
      
      // Create acceleration result
      const accelerationResult: AccelerationResult = {
        status: 'success',
        processingTime,
        throughput,
        hardware: model.getBackend(),
        memoryUsage,
        result
      };
      
      // Store result if enabled
      if (this.options.storeResults && this.storage) {
        // await this.storage.storeAccelerationResult(accelerationResult);
      }
      
      return accelerationResult;
    } catch (error) {
      console.error(`Failed to accelerate model ${modelId}:`, error);
      
      return {
        status: 'error',
        processingTime: 0,
        throughput: 0,
        hardware: 'cpu',
        memoryUsage: 0,
        result: {
          error: error instanceof Error ? error.message : String(error)
        }
      };
    }
  }

  /**
   * Get hardware capabilities
   */
  getCapabilities(): any {
    if (!this.isInitialized) {
      throw new Error('WebAccelerator not initialized');
    }
    
    return this.hardware?.getCapabilities();
  }

  /**
   * Check if a specific backend is supported
   */
  isBackendSupported(backend: HardwareBackendType): boolean {
    if (!this.isInitialized) {
      throw new Error('WebAccelerator not initialized');
    }
    
    return this.hardware?.isBackendSupported(backend) || false;
  }

  /**
   * Get optimal backend for a model type
   */
  async getOptimalBackend(modelType: ModelType): Promise<HardwareBackendType> {
    if (!this.isInitialized) {
      throw new Error('WebAccelerator not initialized');
    }
    
    return this.hardware?.getOptimalBackendForModel(modelType) || 'cpu';
  }

  /**
   * Quantize a model to lower precision
   */
  async quantizeModel(options: {
    modelId: string;
    bits?: number;
    mixedPrecision?: boolean;
    calibrationData: any[];
  }): Promise<any> {
    if (!this.isInitialized) {
      throw new Error('WebAccelerator not initialized');
    }
    
    if (!this.quantizationEngine) {
      throw new Error('Quantization engine not available');
    }
    
    const { modelId, bits = 8, mixedPrecision = false, calibrationData } = options;
    
    return await this.quantizationEngine.quantize({
      modelId,
      calibrationData,
      quantizationConfig: {
        bits,
        scheme: 'symmetric',
        mixedPrecision,
        shaderOptimizations: true,
        ...this.options.quantizationConfig
      },
      targetBackend: 'webgpu'
    });
  }

  /**
   * Use ultra-low precision for a model (2-bit, 3-bit)
   */
  async useUltraLowPrecision(modelId: string, bits: 2 | 3, calibrationData: any[]): Promise<any> {
    if (!this.isInitialized) {
      throw new Error('WebAccelerator not initialized');
    }
    
    if (!this.ultraLowPrecision) {
      throw new Error('Ultra-low precision engine not available');
    }
    
    if (bits === 2) {
      return await this.ultraLowPrecision.quantize2Bit(modelId, calibrationData);
    } else {
      return await this.ultraLowPrecision.quantize3Bit(modelId, calibrationData);
    }
  }

  /**
   * Generate a benchmark report
   */
  async generateReport(options: {
    format?: 'html' | 'markdown' | 'json';
    title?: string;
    includeCharts?: boolean;
  } = {}): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('WebAccelerator not initialized');
    }
    
    if (!this.options.storeResults) {
      throw new Error('Result storage is not enabled');
    }
    
    // This would be implemented with actual storage and reporting
    // For now, return a placeholder report
    
    const format = options.format || 'html';
    const title = options.title || 'Acceleration Benchmark Report';
    
    if (format === 'html') {
      return `
        <html>
          <head>
            <title>${title}</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 20px; }
              h1 { color: #333; }
              .container { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
              .chart { width: 100%; height: 400px; background-color: #f5f5f5; }
            </style>
          </head>
          <body>
            <h1>${title}</h1>
            <div class="container">
              <h2>Hardware Information</h2>
              <p>Browser: ${this.getCapabilities()?.browserName || 'Unknown'}</p>
              <p>WebGPU: ${this.isBackendSupported('webgpu') ? 'Supported' : 'Not supported'}</p>
              <p>WebNN: ${this.isBackendSupported('webnn') ? 'Supported' : 'Not supported'}</p>
            </div>
            ${options.includeCharts ? '<div class="chart">Chart placeholder</div>' : ''}
            <div class="container">
              <h2>Results</h2>
              <p>This is a placeholder report. Actual benchmark results would be shown here.</p>
            </div>
          </body>
        </html>
      `;
    } else if (format === 'markdown') {
      return `
        # ${title}
        
        ## Hardware Information
        
        - Browser: ${this.getCapabilities()?.browserName || 'Unknown'}
        - WebGPU: ${this.isBackendSupported('webgpu') ? 'Supported' : 'Not supported'}
        - WebNN: ${this.isBackendSupported('webnn') ? 'Supported' : 'Not supported'}
        
        ## Results
        
        This is a placeholder report. Actual benchmark results would be shown here.
      `;
    } else {
      return JSON.stringify({
        title,
        hardware: this.getCapabilities(),
        results: {
          placeholder: true,
          message: 'This is a placeholder report. Actual benchmark results would be shown here.'
        }
      }, null, 2);
    }
  }

  /**
   * Clean up resources
   */
  async dispose(): Promise<void> {
    // Clean up all components
    
    if (this.modelLoader) {
      await this.modelLoader.dispose();
      this.modelLoader = null;
    }
    
    if (this.quantizationEngine) {
      await this.quantizationEngine.dispose();
      this.quantizationEngine = null;
    }
    
    this.ultraLowPrecision = null;
    
    if (this.storage) {
      // await this.storage.dispose();
      this.storage = null;
    }
    
    if (this.p2pManager) {
      // await this.p2pManager.dispose();
      this.p2pManager = null;
    }
    
    if (this.hardware) {
      this.hardware.dispose();
      this.hardware = null;
    }
    
    this.isInitialized = false;
  }
}

// Utility functions

/**
 * Detect browser capabilities for hardware acceleration
 */
export async function detectCapabilities(): Promise<{
  webgpu: any;
  webnn: any;
  wasm: any;
  optimalBackend: HardwareBackendType;
  browserName: string;
}> {
  const webgpuInfo = await getWebGPUInfo();
  const webnnInfo = await getWebNNInfo();
  
  // Detect WebAssembly support
  const wasmSupported = typeof WebAssembly === 'object';
  const wasmFeatures = {
    simd: WebAssembly.validate && WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3,
      2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
    ])),
    threads: typeof SharedArrayBuffer === 'function'
  };
  
  // Determine optimal backend
  let optimalBackend: HardwareBackendType = 'cpu';
  if (webgpuInfo.supported) {
    optimalBackend = 'webgpu';
  } else if (webnnInfo.supported) {
    optimalBackend = 'webnn';
  } else if (wasmSupported) {
    optimalBackend = 'wasm';
  }
  
  // Detect browser
  const userAgent = navigator.userAgent;
  let browserName = 'unknown';
  
  if (userAgent.indexOf('Edge') > -1 || userAgent.indexOf('Edg') > -1) {
    browserName = 'edge';
  } else if (userAgent.indexOf('Firefox') > -1) {
    browserName = 'firefox';
  } else if (userAgent.indexOf('Chrome') > -1) {
    browserName = 'chrome';
  } else if (userAgent.indexOf('Safari') > -1) {
    browserName = 'safari';
  }
  
  return {
    webgpu: webgpuInfo,
    webnn: webnnInfo,
    wasm: {
      supported: wasmSupported,
      ...wasmFeatures
    },
    optimalBackend,
    browserName
  };
}

/**
 * Create and initialize a WebAccelerator with default options
 */
export async function createAccelerator(options: WebAcceleratorOptions = {}): Promise<WebAccelerator> {
  const accelerator = new WebAccelerator(options);
  await accelerator.initialize();
  return accelerator;
}

// Export main classes
export {
  WebGPUBackend,
  WebNNBackend,
  HardwareAbstraction,
  ModelLoader,
  Model,
  QuantizationEngine,
  UltraLowPrecisionEngine
};