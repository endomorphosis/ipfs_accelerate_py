/**
 * Hardware Abstracted BERT Implementation
 * Implements BERT model with automatic hardware acceleration selection and optimization
 */

import { HardwareBackend } from '../../hardware/interfaces/hardware_backend';
import { createOptimalBackend, createMultiBackend } from '../../hardware/index';
import { Tensor } from '../../tensor/tensor';
import { SharedTensor } from '../../tensor/shared_tensor';
import { matmul, softmax } from '../../tensor/operations/matrix';
import { layerNorm, gelu } from '../../tensor/operations/nn';
import { add, mul } from '../../tensor/operations/basic';
import { Bert as BERTImplementation, BertConfig, BertInput, BertOutput } from '../transformers/bert';

/**
 * Storage Manager Interface for model weights storage
 */
export interface StorageManager {
  initialize(): Promise<void>;
  getItem(key: string): Promise<any>;
  setItem(key: string, value: any): Promise<void>;
  hasItem(key: string): Promise<boolean>;
  removeItem(key: string): Promise<void>;
}

/**
 * Configuration for Hardware Abstracted BERT
 */
export interface HardwareAbstractedBERTConfig extends BertConfig {
  /**
   * Quantization settings
   */
  quantization?: {
    enabled: boolean;
    bits: number;
    blockSize?: number;
  };
  
  /**
   * Backend preference order
   */
  backendPreference?: ('webgpu' | 'webnn' | 'wasm' | 'cpu')[];
  
  /**
   * Whether to allow automatic fallback to next backend
   */
  allowFallback?: boolean;
  
  /**
   * Whether to collect performance metrics
   */
  collectMetrics?: boolean;
  
  /**
   * Browser-specific optimizations
   */
  browserOptimizations?: boolean;
  
  /**
   * Task type - determines output format and processing
   */
  taskType?: 'embedding' | 'sequence_classification' | 'token_classification' | 'question_answering';
  
  /**
   * Number of output labels for classification tasks
   */
  numLabels?: number;
}

/**
 * Default Hardware Abstracted BERT Configuration (bert-base-uncased)
 */
export const DEFAULT_HARDWARE_ABSTRACTED_BERT_CONFIG: HardwareAbstractedBERTConfig = {
  modelId: 'bert-base-uncased',
  vocabSize: 30522,
  hiddenSize: 768,
  numLayers: 12,
  numHeads: 12,
  intermediateSize: 3072,
  maxPositions: 512,
  layerNormEps: 1e-12,
  backendPreference: ['webnn', 'webgpu', 'cpu'],
  useOptimizedOps: true,
  useBrowserOptimizations: true,
  useOperationFusion: true,
  attentionDropout: 0.1,
  allowFallback: true,
  collectMetrics: true,
  browserOptimizations: true,
  taskType: 'embedding',
  quantization: {
    enabled: false,
    bits: 8
  }
};

/**
 * Performance metric type
 */
interface PerformanceMetric {
  avg: number;
  min: number;
  max: number;
  count: number;
  total: number;
}

/**
 * Hardware Abstracted BERT class
 * Provides hardware optimization for BERT model
 */
export class HardwareAbstractedBERT {
  private config: HardwareAbstractedBERTConfig;
  private storageManager: StorageManager;
  private hardware: HardwareBackend | null = null;
  private modelImpl: BERTImplementation | null = null;
  private initialized: boolean = false;
  
  // Metrics tracking
  private performanceMetrics: Record<string, PerformanceMetric> = {};
  private backendMetrics: Record<string, any> = {};
  private selectedBackend: string = '';
  private availableBackends: string[] = [];
  
  /**
   * Constructor for Hardware Abstracted BERT
   * @param config Configuration options
   * @param storageManager Storage manager for model weights
   */
  constructor(
    config: Partial<HardwareAbstractedBERTConfig> = {},
    storageManager: StorageManager
  ) {
    this.config = { ...DEFAULT_HARDWARE_ABSTRACTED_BERT_CONFIG, ...config };
    this.storageManager = storageManager;
  }
  
  /**
   * Initialize the model
   * Sets up hardware backend and model implementation
   */
  public async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    
    try {
      const startTime = performance.now();
      
      // Initialize hardware backend
      await this.initializeHardware();
      
      // Initialize model implementation
      if (this.hardware) {
        this.modelImpl = new BERTImplementation(this.hardware, this.config);
        await this.modelImpl.initialize();
      } else {
        throw new Error('Failed to initialize hardware backend');
      }
      
      const initTime = performance.now() - startTime;
      this.updateMetric('initialization', initTime);
      
      this.initialized = true;
      console.log(`BERT model initialized (${this.selectedBackend}) in ${initTime.toFixed(2)}ms`);
    } catch (error) {
      console.error('Failed to initialize Hardware Abstracted BERT:', error);
      throw error;
    }
  }
  
  /**
   * Initialize hardware backend
   * Selects optimal backend based on config preferences
   */
  private async initializeHardware(): Promise<void> {
    // Collect available backends for metrics
    this.availableBackends = this.getAvailableBackends();
    
    try {
      // Try to initialize with optimal backend
      if (this.config.backendPreference && this.config.backendPreference.length > 0) {
        // Use multi-backend with specified preference order
        this.hardware = await createMultiBackend(this.config.backendPreference);
      } else {
        // Use automatic optimal backend selection
        this.hardware = await createOptimalBackend({
          preferNeuralOps: true, // BERT is primarily a neural network model, so prefer WebNN
          modelType: 'text'      // Specify this is a text model
        });
      }
      
      if (!this.hardware) {
        throw new Error('Failed to initialize hardware backend');
      }
      
      // Store selected backend type
      this.selectedBackend = this.hardware.type;
      
      // Record backend capabilities
      this.backendMetrics = {
        type: this.hardware.type,
        capabilities: this.hardware.capabilities,
        isAvailable: this.hardware.isAvailable
      };
      
      console.log(`Selected backend: ${this.selectedBackend}`);
    } catch (error) {
      console.error('Error initializing hardware backend:', error);
      
      if (this.config.allowFallback) {
        console.warn('Falling back to CPU backend');
        // Fallback to CPU backend
        this.hardware = await createOptimalBackend({ forceBackend: 'cpu' });
        if (this.hardware) {
          this.selectedBackend = this.hardware.type;
          console.log(`Fallback to ${this.selectedBackend} successful`);
        } else {
          throw new Error('Failed to initialize fallback CPU backend');
        }
      } else {
        throw error;
      }
    }
  }
  
  /**
   * Get list of available backends
   * Depends on browser capabilities
   */
  private getAvailableBackends(): string[] {
    const backends: string[] = ['cpu']; // CPU is always available
    
    // Check for WebGPU
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      backends.push('webgpu');
    }
    
    // Check for WebNN
    if (typeof navigator !== 'undefined' && 'ml' in navigator) {
      backends.push('webnn');
    }
    
    return backends;
  }
  
  /**
   * Preprocess input text into tokens
   * @param text Input text
   * @returns Tokenized input
   */
  public async tokenize(text: string): Promise<BertInput> {
    const startTime = performance.now();
    
    // In a real implementation, this would use a proper tokenizer
    // For now, using the implementation in the base BERT class
    if (!this.modelImpl) {
      throw new Error('Model not initialized');
    }
    
    const result = await this.modelImpl.tokenize(text);
    
    const tokenizeTime = performance.now() - startTime;
    this.updateMetric('tokenization', tokenizeTime);
    
    return result;
  }
  
  /**
   * Run prediction on input text
   * @param text Input text or pre-tokenized input
   * @returns Model output based on task type
   */
  public async predict(text: string | BertInput): Promise<BertOutput | any> {
    if (!this.initialized || !this.modelImpl) {
      throw new Error('Model not initialized. Call initialize() first.');
    }
    
    const totalStartTime = performance.now();
    
    try {
      // Convert text to tokens if needed
      let input: BertInput;
      if (typeof text === 'string') {
        input = await this.tokenize(text);
      } else {
        input = text;
      }
      
      // Run model inference
      const inferenceStartTime = performance.now();
      const output = await this.modelImpl.process(input);
      const inferenceTime = performance.now() - inferenceStartTime;
      
      this.updateMetric('inference', inferenceTime);
      
      // Process output based on task type
      let result: any;
      
      if (this.config.taskType === 'embedding') {
        // Return embeddings
        result = output.lastHiddenState;
      } else if (this.config.taskType === 'sequence_classification') {
        // Process classification output
        result = {
          logits: output.lastHiddenState[0]
        };
      } else if (this.config.taskType === 'question_answering') {
        // Process QA output (simplified)
        const startLogits = output.lastHiddenState.map(row => row[0]);
        const endLogits = output.lastHiddenState.map(row => row[1]);
        
        result = {
          start_logits: startLogits,
          end_logits: endLogits
        };
      } else {
        // Default to returning raw output
        result = output;
      }
      
      const totalTime = performance.now() - totalStartTime;
      this.updateMetric('total_processing', totalTime);
      
      return result;
    } catch (error) {
      console.error('Error during BERT prediction:', error);
      throw error;
    }
  }
  
  /**
   * Run performance comparison across all available backends
   * @param text Test input text
   * @returns Execution time by backend
   */
  public async compareBackends(text: string): Promise<Record<string, number>> {
    const results: Record<string, number> = {};
    const availableBackends = this.getAvailableBackends();
    
    // First tokenize the input (shared across backends)
    const input = await this.tokenize(text);
    
    // Test each backend
    for (const backendType of availableBackends) {
      try {
        console.log(`Testing ${backendType} backend...`);
        
        // Create backend
        const hardware = await createOptimalBackend({ forceBackend: backendType as any });
        
        if (!hardware) {
          console.warn(`Backend ${backendType} initialization failed`);
          results[backendType] = -1; // Mark as failed
          continue;
        }
        
        // Create model
        const model = new BERTImplementation(hardware, this.config);
        await model.initialize();
        
        // Warm-up run
        await model.process(input);
        
        // Timed run
        const startTime = performance.now();
        await model.process(input);
        const endTime = performance.now();
        
        // Record result
        results[backendType] = endTime - startTime;
        
        // Clean up
        try {
          await model.dispose();
          await hardware.dispose();
        } catch (e) {
          console.warn(`Error disposing resources for ${backendType}:`, e);
        }
      } catch (error) {
        console.error(`Error testing ${backendType} backend:`, error);
        results[backendType] = -1; // Mark as failed
      }
    }
    
    console.log('Backend comparison results:', results);
    return results;
  }
  
  /**
   * Get performance metrics collected during execution
   * @returns Performance metrics by operation
   */
  public getPerformanceMetrics(): Record<string, PerformanceMetric> {
    return this.performanceMetrics;
  }
  
  /**
   * Get backend-specific metrics and capabilities
   * @returns Backend metrics
   */
  public getBackendMetrics(): Record<string, any> {
    return this.backendMetrics;
  }
  
  /**
   * Get model information
   * @returns Model info
   */
  public getModelInfo(): Record<string, any> {
    return {
      modelId: this.config.modelId,
      modelType: 'BERT',
      taskType: this.config.taskType,
      hiddenSize: this.config.hiddenSize,
      numLayers: this.config.numLayers,
      numHeads: this.config.numHeads,
      selectedBackend: this.selectedBackend,
      availableBackends: this.availableBackends,
      quantization: this.config.quantization?.enabled
        ? `${this.config.quantization.bits}-bit`
        : 'none'
    };
  }
  
  /**
   * Update a performance metric with a new timing value
   * @param name Metric name
   * @param value Timing value
   */
  private updateMetric(name: string, value: number): void {
    if (!this.config.collectMetrics) {
      return;
    }
    
    if (!this.performanceMetrics[name]) {
      this.performanceMetrics[name] = {
        avg: value,
        min: value,
        max: value,
        count: 1,
        total: value
      };
      return;
    }
    
    const metric = this.performanceMetrics[name];
    metric.count++;
    metric.total += value;
    metric.avg = metric.total / metric.count;
    metric.min = Math.min(metric.min, value);
    metric.max = Math.max(metric.max, value);
  }
  
  /**
   * Release all resources
   */
  public async dispose(): Promise<void> {
    if (this.modelImpl) {
      await this.modelImpl.dispose();
      this.modelImpl = null;
    }
    
    if (this.hardware) {
      await this.hardware.dispose();
      this.hardware = null;
    }
    
    this.initialized = false;
  }
  
  /**
   * Get a shared tensor from the model
   * @param outputType Type of output to get (e.g. 'text_embedding')
   * @returns Shared tensor or null if not found
   */
  public getSharedTensor(outputType: string = 'text_embedding'): SharedTensor | null {
    if (!this.modelImpl) {
      return null;
    }
    
    return this.modelImpl.getSharedTensor(outputType);
  }
  
  /**
   * Reset performance metrics
   */
  public resetMetrics(): void {
    this.performanceMetrics = {};
  }
}

/**
 * Factory function to create a Hardware Abstracted BERT
 * @param config Configuration options
 * @param storageManager Storage manager for model weights
 * @returns HardwareAbstractedBERT instance
 */
export function createHardwareAbstractedBERT(
  config: Partial<HardwareAbstractedBERTConfig> = {},
  storageManager: StorageManager
): HardwareAbstractedBERT {
  return new HardwareAbstractedBERT(config, storageManager);
}