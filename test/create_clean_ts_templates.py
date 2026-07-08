#!/usr/bin/env python3
# create_clean_ts_templates.py
# Script to create clean TypeScript template implementations for core components

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'create_clean_ts_templates_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    TARGET_DIR = os.path.abspath("../ipfs_accelerate_js")
    TYPES = []
    FORCE = False
    CREATE_INDEX = True

def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create clean TypeScript template implementations")
    parser.add_argument("--target-dir", help="Target directory", default="../ipfs_accelerate_js")
    parser.add_argument("--types", help="Comma-separated list of template types to create", 
                      default="interfaces,webgpu,webnn,hardware,tensor,resource_pool")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing files")
    parser.add_argument("--no-index", action="store_true", help="Skip creating index.ts files")
    args = parser.parse_args()
    
    Config.TARGET_DIR = os.path.abspath(args.target_dir)
    Config.TYPES = args.types.split(",")
    Config.FORCE = args.force
    Config.CREATE_INDEX = not args.no_index
    
    logger.info(f"Target directory: {Config.TARGET_DIR}")
    logger.info(f"Template types: {Config.TYPES}")
    logger.info(f"Force overwrite: {Config.FORCE}")
    logger.info(f"Create index files: {Config.CREATE_INDEX}")

# Template for interfaces.ts
INTERFACES_TEMPLATE = """/**
 * Core interfaces for IPFS Accelerate JavaScript SDK
 */

// Hardware interfaces
export interface HardwareBackend {
  initialize(): Promise<boolean>;
  destroy(): void;
  execute<T = any, U = any>(inputs: T): Promise<U>;
}

export interface HardwarePreferences {
  backendOrder: string[];
  modelPreferences: Record<string, string>;
  options: Record<string, any>;
}

// Model interfaces
export interface ModelConfig {
  id: string;
  type: string;
  path: string;
  options: Record<string, any>;
}

export interface Model {
  id: string;
  type: string;
  execute<T = any, U = any>(inputs: T): Promise<U>;
}

// WebGPU interfaces
export interface GPUBufferDescriptor {
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}

export interface GPUShaderModuleDescriptor {
  code: string;
}

export interface GPUBindGroupDescriptor {
  layout: any;
  entries: GPUBindGroupEntry[];
}

export interface GPUBindGroupEntry {
  binding: number;
  resource: any;
}

export interface GPUComputePipelineDescriptor {
  layout: any;
  compute: {
    module: any;
    entryPoint: string;
  };
}

// WebNN interfaces
export interface MLOperandDescriptor {
  type: string;
  dimensions: number[];
}

export interface MLOperand {}

export interface MLGraph {
  compute(inputs: Record<string, any>): Promise<Record<string, any>>;
}

export interface MLContext {}

export interface MLGraphBuilder {
  constant(value: any, dimensions?: number[]): MLOperand;
  input(name: string, descriptor: MLOperandDescriptor): MLOperand;
  build(outputs: Record<string, MLOperand>): Promise<MLGraph>;
}

// Resource Pool interfaces
export interface ResourcePoolConnection {
  id: string;
  type: string;
  status: string;
  created: Date;
  resources: Record<string, any>;
}

export interface ResourcePoolOptions {
  maxConnections: number;
  browserPreferences: Record<string, string>;
  adaptiveScaling: boolean;
  enableFaultTolerance: boolean;
  recoveryStrategy: string;
  stateSyncInterval: number;
  redundancyFactor: number;
}

// Browser interfaces
export interface BrowserCapabilities {
  browserName: string;
  browserVersion: string;
  isMobile: boolean;
  platform: string;
  osVersion: string;
  webgpuSupported: boolean;
  webgpuFeatures: string[];
  webnnSupported: boolean;
  webnnFeatures: string[];
  wasmSupported: boolean;
  wasmFeatures: string[];
  metalApiSupported: boolean;
  metalApiVersion: string;
  recommendedBackend: string;
  memoryLimitMB: number;
}

// Optimization interfaces
export interface OptimizationConfig {
  memoryOptimization: boolean;
  progressiveLoading: boolean;
  useQuantization: boolean;
  precision: string;
  maxChunkSizeMB: number;
  parallelLoading: boolean;
  specialOptimizations: Record<string, any>;
}

// Tensor interfaces
export interface Tensor {
  shape: number[];
  data: Float32Array | Int32Array | Uint8Array;
  dtype: string;
}

// Shared tensor memory interface
export interface TensorSharing {
  shareableTypes: string[];
  enableSharing: boolean;
  shareTensor(id: string, tensor: Tensor): any;
  getTensor(id: string): Tensor | null;
  releaseSharedTensors(ids: string[]): void;
}
"""

# Template for WebGPU backend
WEBGPU_BACKEND_TEMPLATE = """/**
 * WebGPU backend implementation for IPFS Accelerate
 */
import { HardwareBackend } from '../interfaces';

export class WebGPUBackend implements HardwareBackend {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private initialized: boolean = false;
  private shaderModules: Map<string, GPUShaderModule> = new Map();
  private buffers: Map<string, GPUBuffer> = new Map();
  private pipelines: Map<string, GPUComputePipeline> = new Map();

  constructor() {
    this.initialized = false;
  }

  async initialize(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        console.warn("WebGPU is not supported in this browser");
        return false;
      }

      this.adapter = await navigator.gpu.requestAdapter();
      if (!this.adapter) {
        console.warn("No WebGPU adapter found");
        return false;
      }

      this.device = await this.adapter.requestDevice();
      if (!this.device) {
        console.warn("Failed to acquire WebGPU device");
        return false;
      }

      this.initialized = true;
      return true;
    } catch (error) {
      console.error("Failed to initialize WebGPU backend:", error);
      return false;
    }
  }

  async execute<T = any, U = any>(inputs: T): Promise<U> {
    if (!this.initialized || !this.device) {
      throw new Error("WebGPU backend not initialized");
    }

    // Implementation will depend on the model type and operation
    // This is a placeholder for the actual implementation
    
    return {} as U;
  }

  destroy(): void {
    // Release WebGPU resources
    for (const buffer of this.buffers.values()) {
      buffer.destroy();
    }
    this.buffers.clear();
    this.shaderModules.clear();
    this.pipelines.clear();
    
    this.device = null;
    this.adapter = null;
    this.initialized = false;
  }

  // WebGPU-specific methods
  
  async createBuffer(size: number, usage: number): Promise<GPUBuffer | null> {
    if (!this.device) {
      throw new Error("WebGPU device not initialized");
    }
    
    try {
      const buffer = this.device.createBuffer({
        size,
        usage,
        mappedAtCreation: false
      });
      
      return buffer;
    } catch (error) {
      console.error("Error creating WebGPU buffer:", error);
      return null;
    }
  }
  
  async createComputePipeline(shaderCode: string, entryPoint: string = "main"): Promise<GPUComputePipeline | null> {
    if (!this.device) {
      throw new Error("WebGPU device not initialized");
    }
    
    try {
      const shaderModule = this.device.createShaderModule({
        code: shaderCode
      });
      
      const pipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint
        }
      });
      
      return pipeline;
    } catch (error) {
      console.error("Error creating compute pipeline:", error);
      return null;
    }
  }
  
  async runComputation(
    pipeline: GPUComputePipeline,
    bindGroups: GPUBindGroup[],
    workgroupCount: [number, number, number] = [1, 1, 1]
  ): Promise<void> {
    if (!this.device) {
      throw new Error("WebGPU device not initialized");
    }
    
    try {
      const commandEncoder = this.device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      
      passEncoder.setPipeline(pipeline);
      
      for (let i = 0; i < bindGroups.length; i++) {
        passEncoder.setBindGroup(i, bindGroups[i]);
      }
      
      passEncoder.dispatchWorkgroups(
        workgroupCount[0],
        workgroupCount[1],
        workgroupCount[2]
      );
      
      passEncoder.end();
      
      const commandBuffer = commandEncoder.finish();
      this.device.queue.submit([commandBuffer]);
    } catch (error) {
      console.error("Error running computation:", error);
      throw error;
    }
  }
}
"""

# Template for WebNN backend
WEBNN_BACKEND_TEMPLATE = """/**
 * WebNN backend implementation for IPFS Accelerate
 */
import { HardwareBackend } from '../interfaces';
import { MLContext, MLGraphBuilder, MLGraph } from '../interfaces';

export class WebNNBackend implements HardwareBackend {
  private context: MLContext | null = null;
  private builder: MLGraphBuilder | null = null;
  private initialized: boolean = false;
  private graphs: Map<string, MLGraph> = new Map();

  constructor() {
    this.initialized = false;
  }

  async initialize(): Promise<boolean> {
    try {
      // Check if WebNN is supported
      if (!('ml' in navigator)) {
        console.warn("WebNN is not supported in this browser");
        return false;
      }

      // @ts-ignore - TypeScript doesn't know about navigator.ml yet
      this.context = navigator.ml?.createContext();
      
      if (!this.context) {
        console.warn("Failed to create WebNN context");
        return false;
      }

      // @ts-ignore - TypeScript doesn't know about navigator.ml yet
      this.builder = new MLGraphBuilder(this.context);
      
      if (!this.builder) {
        console.warn("Failed to create WebNN graph builder");
        return false;
      }

      this.initialized = true;
      return true;
    } catch (error) {
      console.error("Failed to initialize WebNN backend:", error);
      return false;
    }
  }

  async execute<T = any, U = any>(inputs: T): Promise<U> {
    if (!this.initialized || !this.builder) {
      throw new Error("WebNN backend not initialized");
    }

    // Implementation will depend on the model type and operation
    // This is a placeholder for the actual implementation
    
    return {} as U;
  }

  destroy(): void {
    // Release WebNN resources
    this.graphs.clear();
    this.builder = null;
    this.context = null;
    this.initialized = false;
  }
  
  // WebNN-specific methods
  
  async buildGraph(outputs: Record<string, MLOperand>): Promise<MLGraph | null> {
    if (!this.builder) {
      throw new Error("WebNN graph builder not initialized");
    }
    
    try {
      return await this.builder.build(outputs);
    } catch (error) {
      console.error("Error building WebNN graph:", error);
      return null;
    }
  }
  
  async runGraph(graph: MLGraph, inputs: Record<string, MLOperand>): Promise<Record<string, MLOperand>> {
    if (!this.initialized) {
      throw new Error("WebNN backend not initialized");
    }
    
    try {
      return await graph.compute(inputs);
    } catch (error) {
      console.error("Error running WebNN graph:", error);
      throw error;
    }
  }
}
"""

# Template for Hardware Abstraction
HARDWARE_ABSTRACTION_TEMPLATE = """/**
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
"""

# Template for Tensor operations
TENSOR_TEMPLATE = """/**
 * Tensor operations for IPFS Accelerate
 */
import { Tensor } from '../interfaces';

export class TensorOperations {
  /**
   * Creates a new tensor with the given shape and data
   */
  static createTensor(shape: number[], data: Float32Array | Int32Array | Uint8Array, dtype: string = 'float32'): Tensor {
    return {
      shape,
      data,
      dtype
    };
  }
  
  /**
   * Creates a tensor filled with zeros
   */
  static zeros(shape: number[], dtype: string = 'float32'): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    let data: Float32Array | Int32Array | Uint8Array;
    
    if (dtype === 'float32') {
      data = new Float32Array(size);
    } else if (dtype === 'int32') {
      data = new Int32Array(size);
    } else if (dtype === 'uint8') {
      data = new Uint8Array(size);
    } else {
      throw new Error(`Unsupported dtype: ${dtype}`);
    }
    
    return {
      shape,
      data,
      dtype
    };
  }
  
  /**
   * Creates a tensor filled with ones
   */
  static ones(shape: number[], dtype: string = 'float32'): Tensor {
    const tensor = this.zeros(shape, dtype);
    
    if (dtype === 'float32') {
      const data = tensor.data as Float32Array;
      data.fill(1);
    } else if (dtype === 'int32') {
      const data = tensor.data as Int32Array;
      data.fill(1);
    } else if (dtype === 'uint8') {
      const data = tensor.data as Uint8Array;
      data.fill(1);
    }
    
    return tensor;
  }
  
  /**
   * Creates a tensor filled with random values
   */
  static random(shape: number[], dtype: string = 'float32'): Tensor {
    const tensor = this.zeros(shape, dtype);
    const size = shape.reduce((a, b) => a * b, 1);
    
    if (dtype === 'float32') {
      const data = tensor.data as Float32Array;
      for (let i = 0; i < size; i++) {
        data[i] = Math.random();
      }
    } else if (dtype === 'int32') {
      const data = tensor.data as Int32Array;
      for (let i = 0; i < size; i++) {
        data[i] = Math.floor(Math.random() * 100);
      }
    } else if (dtype === 'uint8') {
      const data = tensor.data as Uint8Array;
      for (let i = 0; i < size; i++) {
        data[i] = Math.floor(Math.random() * 256);
      }
    }
    
    return tensor;
  }
  
  /**
   * Gets the size of a tensor
   */
  static size(tensor: Tensor): number {
    return tensor.shape.reduce((a, b) => a * b, 1);
  }
  
  /**
   * Reshapes a tensor to a new shape
   */
  static reshape(tensor: Tensor, newShape: number[]): Tensor {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    const oldSize = this.size(tensor);
    
    if (newSize !== oldSize) {
      throw new Error(`Cannot reshape tensor of size ${oldSize} to size ${newSize}`);
    }
    
    return {
      shape: newShape,
      data: tensor.data,
      dtype: tensor.dtype
    };
  }
}

export class TensorSharingManager {
  private sharedTensors: Map<string, Tensor> = new Map();
  private refCounts: Map<string, number> = new Map();
  
  /**
   * Shares a tensor for reuse across models
   */
  shareTensor(id: string, tensor: Tensor): void {
    this.sharedTensors.set(id, tensor);
    this.refCounts.set(id, (this.refCounts.get(id) || 0) + 1);
  }
  
  /**
   * Gets a shared tensor by ID
   */
  getTensor(id: string): Tensor | null {
    return this.sharedTensors.get(id) || null;
  }
  
  /**
   * Releases a shared tensor
   */
  releaseTensor(id: string): void {
    if (!this.refCounts.has(id)) {
      return;
    }
    
    const count = this.refCounts.get(id)! - 1;
    if (count <= 0) {
      this.sharedTensors.delete(id);
      this.refCounts.delete(id);
    } else {
      this.refCounts.set(id, count);
    }
  }
  
  /**
   * Gets all shared tensor IDs
   */
  getSharedTensorIds(): string[] {
    return Array.from(this.sharedTensors.keys());
  }
  
  /**
   * Clears all shared tensors
   */
  clearAll(): void {
    this.sharedTensors.clear();
    this.refCounts.clear();
  }
}

// Singleton instance
export const tensorSharingManager = new TensorSharingManager();
"""

# Template for Resource Pool
RESOURCE_POOL_TEMPLATE = """/**
 * Resource Pool for managing browser resources
 */
import { ResourcePoolOptions, ResourcePoolConnection, BrowserCapabilities } from '../interfaces';
import { detectBrowserCapabilities } from '../browser/optimizations/browser_capability_detection';

export class ResourcePool {
  private connections: ResourcePoolConnection[] = [];
  private activeConnections: Map<string, ResourcePoolConnection> = new Map();
  private options: ResourcePoolOptions;
  private initialized: boolean = false;
  
  constructor(options: Partial<ResourcePoolOptions> = {}) {
    this.options = {
      maxConnections: options.maxConnections || 4,
      browserPreferences: options.browserPreferences || {},
      adaptiveScaling: options.adaptiveScaling !== undefined ? options.adaptiveScaling : true,
      enableFaultTolerance: options.enableFaultTolerance !== undefined ? options.enableFaultTolerance : false,
      recoveryStrategy: options.recoveryStrategy || 'progressive',
      stateSyncInterval: options.stateSyncInterval || 5,
      redundancyFactor: options.redundancyFactor || 1
    };
  }
  
  async initialize(): Promise<boolean> {
    try {
      // Detect browser capabilities
      const capabilities = await detectBrowserCapabilities();
      
      // Create initial connections
      for (let i = 0; i < this.options.maxConnections; i++) {
        const connection = await this.createConnection(capabilities.browserName);
        if (connection) {
          this.connections.push(connection);
          this.activeConnections.set(connection.id, connection);
        }
      }
      
      this.initialized = true;
      return this.connections.length > 0;
    } catch (error) {
      console.error("Failed to initialize resource pool:", error);
      return false;
    }
  }
  
  private async createConnection(browserType: string): Promise<ResourcePoolConnection> {
    const id = `conn-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    
    // In a real implementation, this would create an actual connection to a browser instance
    const connection: ResourcePoolConnection = {
      id,
      type: browserType,
      status: 'connected',
      created: new Date(),
      resources: {}
    };
    
    return connection;
  }
  
  async getConnection(preferredType?: string): Promise<ResourcePoolConnection | null> {
    if (!this.initialized) {
      throw new Error("Resource pool not initialized");
    }
    
    // Find an available connection of the preferred type
    if (preferredType) {
      for (const connection of this.connections) {
        if (connection.type === preferredType && connection.status === 'connected') {
          return connection;
        }
      }
    }
    
    // Fall back to any available connection
    for (const connection of this.connections) {
      if (connection.status === 'connected') {
        return connection;
      }
    }
    
    // If adaptive scaling is enabled, try to create a new connection
    if (this.options.adaptiveScaling && this.connections.length < this.options.maxConnections * 2) {
      try {
        const capabilities = await detectBrowserCapabilities();
        const connection = await this.createConnection(capabilities.browserName);
        if (connection) {
          this.connections.push(connection);
          this.activeConnections.set(connection.id, connection);
          return connection;
        }
      } catch (error) {
        console.error("Failed to create new connection:", error);
      }
    }
    
    return null;
  }
  
  async releaseConnection(connectionId: string): Promise<void> {
    const connection = this.activeConnections.get(connectionId);
    if (connection) {
      connection.status = 'available';
    }
  }
  
  async closeConnection(connectionId: string): Promise<void> {
    const index = this.connections.findIndex(c => c.id === connectionId);
    if (index >= 0) {
      this.connections.splice(index, 1);
      this.activeConnections.delete(connectionId);
    }
  }
  
  getConnectionCount(): number {
    return this.connections.length;
  }
  
  getActiveConnectionCount(): number {
    return this.activeConnections.size;
  }
  
  dispose(): void {
    this.connections = [];
    this.activeConnections.clear();
    this.initialized = false;
  }
}

export class ResourcePoolBridge {
  private resourcePool: ResourcePool;
  private models: Map<string, any> = new Map();
  private initialized: boolean = false;
  
  constructor(options: Partial<ResourcePoolOptions> = {}) {
    this.resourcePool = new ResourcePool(options);
  }
  
  async initialize(): Promise<boolean> {
    try {
      const success = await this.resourcePool.initialize();
      this.initialized = success;
      return success;
    } catch (error) {
      console.error("Failed to initialize resource pool bridge:", error);
      return false;
    }
  }
  
  async getModel(modelConfig: any): Promise<any> {
    if (!this.initialized) {
      throw new Error("Resource pool bridge not initialized");
    }
    
    const modelId = modelConfig.id;
    
    // Check if model already exists
    if (this.models.has(modelId)) {
      return this.models.get(modelId);
    }
    
    // Get a connection from the resource pool
    const connection = await this.resourcePool.getConnection();
    if (!connection) {
      throw new Error("No available connections in resource pool");
    }
    
    // In a real implementation, this would load the model in the browser
    const model = {
      id: modelId,
      type: modelConfig.type,
      connectionId: connection.id,
      execute: async (inputs: any) => {
        // Placeholder implementation
        return { outputs: "Model execution placeholder" };
      }
    };
    
    this.models.set(modelId, model);
    return model;
  }
  
  async releaseModel(modelId: string): Promise<void> {
    if (this.models.has(modelId)) {
      const model = this.models.get(modelId);
      await this.resourcePool.releaseConnection(model.connectionId);
      this.models.delete(modelId);
    }
  }
  
  getModelCount(): number {
    return this.models.size;
  }
  
  dispose(): void {
    this.models.clear();
    this.resourcePool.dispose();
    this.initialized = false;
  }
}
"""

# Template mappings
TEMPLATES = {
    "interfaces": {
        "file_path": "src/interfaces.ts",
        "content": INTERFACES_TEMPLATE
    },
    "webgpu": {
        "file_path": "src/hardware/backends/webgpu_backend.ts",
        "content": WEBGPU_BACKEND_TEMPLATE
    },
    "webnn": {
        "file_path": "src/hardware/backends/webnn_backend.ts",
        "content": WEBNN_BACKEND_TEMPLATE
    },
    "hardware": {
        "file_path": "src/hardware/hardware_abstraction.ts",
        "content": HARDWARE_ABSTRACTION_TEMPLATE
    },
    "tensor": {
        "file_path": "src/tensor/tensor_operations.ts",
        "content": TENSOR_TEMPLATE
    },
    "resource_pool": {
        "file_path": "src/browser/resource_pool/resource_pool_bridge.ts",
        "content": RESOURCE_POOL_TEMPLATE
    }
}

# Index file templates
INDEX_TEMPLATES = {
    "hardware": "export * from './hardware_abstraction';\nexport * from './detection';\nexport * from './backends';\n",
    "hardware/backends": "export * from './webgpu_backend';\nexport * from './webnn_backend';\nexport * from './cpu_backend';\n",
    "hardware/detection": "export * from './hardware_detection';\nexport * from './gpu_detection';\nexport * from './ml_detection';\n",
    "browser/resource_pool": "export * from './resource_pool_bridge';\n",
    "tensor": "export * from './tensor_operations';\n",
    "browser/optimizations": "export * from './browser_capability_detection';\n"
}

def create_template_file(template_type, target_dir):
    """Create a template file"""
    if template_type not in TEMPLATES:
        logger.error(f"Unknown template type: {template_type}")
        return False
    
    template = TEMPLATES[template_type]
    file_path = os.path.join(target_dir, template["file_path"])
    directory = os.path.dirname(file_path)
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Check if file already exists
    if os.path.exists(file_path) and not Config.FORCE:
        logger.info(f"File already exists, skipping: {file_path}")
        return False
    
    # Write the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(template["content"])
    
    logger.info(f"Created template file: {file_path}")
    return True

def create_index_files(target_dir):
    """Create index.ts files for modules"""
    for module, content in INDEX_TEMPLATES.items():
        directory = os.path.join(target_dir, "src", module)
        file_path = os.path.join(directory, "index.ts")
        
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Check if file already exists
        if os.path.exists(file_path) and not Config.FORCE:
            logger.info(f"Index file already exists, skipping: {file_path}")
            continue
        
        # Write the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created index file: {file_path}")

def create_empty_cpu_backend(target_dir):
    """Create an empty CPU backend file for reference"""
    file_path = os.path.join(target_dir, "src/hardware/backends/cpu_backend.ts")
    directory = os.path.dirname(file_path)
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(file_path) and not Config.FORCE:
        logger.info(f"CPU backend file already exists, skipping: {file_path}")
        return
    
    # Write the file
    content = """/**
 * CPU backend implementation (fallback)
 */
import { HardwareBackend } from '../../interfaces';

export class CPUBackend implements HardwareBackend {
  private initialized: boolean = false;
  
  constructor() {
    this.initialized = false;
  }
  
  async initialize(): Promise<boolean> {
    // CPU backend is always available
    this.initialized = true;
    return true;
  }
  
  async execute<T = any, U = any>(inputs: T): Promise<U> {
    // Placeholder implementation for CPU execution
    console.warn("Using CPU backend (slow)");
    return {} as U;
  }
  
  destroy(): void {
    this.initialized = false;
  }
}
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Created CPU backend file: {file_path}")

def create_hardware_detection_files(target_dir):
    """Create hardware detection files"""
    # Create hardware_detection.ts
    hardware_detection_path = os.path.join(target_dir, "src/hardware/detection/hardware_detection.ts")
    directory = os.path.dirname(hardware_detection_path)
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(hardware_detection_path) and not Config.FORCE:
        logger.info(f"Hardware detection file already exists, skipping: {hardware_detection_path}")
    else:
        # Write the file
        content = """/**
 * Hardware detection utilities
 */
import { BrowserCapabilities } from '../../interfaces';
import { detectGPUCapabilities, GPUCapabilities } from './gpu_detection';
import { detectMLCapabilities, MLCapabilities } from './ml_detection';

export async function detectHardwareCapabilities(): Promise<BrowserCapabilities> {
  // Detect CPU capabilities
  const cpuCores = navigator.hardwareConcurrency || 1;
  
  // Detect GPU capabilities
  const gpuCapabilities = await detectGPUCapabilities();
  
  // Detect ML capabilities
  const mlCapabilities = await detectMLCapabilities();
  
  // Determine recommended backend
  let recommendedBackend = 'cpu';
  if (gpuCapabilities.webgpu.supported) {
    recommendedBackend = 'webgpu';
  } else if (mlCapabilities.webnn.supported) {
    recommendedBackend = 'webnn';
  } else if (gpuCapabilities.wasm.supported && gpuCapabilities.wasm.simd) {
    recommendedBackend = 'wasm';
  }
  
  return {
    browserName: gpuCapabilities.browserName || 'unknown',
    browserVersion: gpuCapabilities.browserVersion || '0',
    isMobile: gpuCapabilities.isMobile || false,
    platform: gpuCapabilities.platform || 'unknown',
    osVersion: gpuCapabilities.osVersion || 'unknown',
    webgpuSupported: gpuCapabilities.webgpu.supported,
    webgpuFeatures: gpuCapabilities.webgpu.features,
    webnnSupported: mlCapabilities.webnn.supported,
    webnnFeatures: mlCapabilities.webnn.features,
    wasmSupported: gpuCapabilities.wasm.supported,
    wasmFeatures: gpuCapabilities.wasm.features || [],
    metalApiSupported: gpuCapabilities.metalApiSupported || false,
    metalApiVersion: gpuCapabilities.metalApiVersion || '0',
    recommendedBackend,
    memoryLimitMB: 4096 // Default value, would be determined based on device
  };
}

export function isWebGPUSupported(): boolean {
  return !!navigator.gpu;
}

export function isWebNNSupported(): boolean {
  return !!navigator.ml;
}

export function isWasmSupported(): boolean {
  return typeof WebAssembly !== 'undefined';
}
"""
        with open(hardware_detection_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created hardware detection file: {hardware_detection_path}")
    
    # Create gpu_detection.ts
    gpu_detection_path = os.path.join(target_dir, "src/hardware/detection/gpu_detection.ts")
    if os.path.exists(gpu_detection_path) and not Config.FORCE:
        logger.info(f"GPU detection file already exists, skipping: {gpu_detection_path}")
    else:
        # Write the file
        content = """/**
 * GPU detection utilities
 */

export interface GPUCapabilities {
  browserName?: string;
  browserVersion?: string;
  isMobile?: boolean;
  platform?: string;
  osVersion?: string;
  metalApiSupported?: boolean;
  metalApiVersion?: string;
  gpu: {
    vendor: string;
    model: string;
    capabilities: Record<string, boolean>;
  };
  webgpu: {
    supported: boolean;
    features: string[];
  };
  wasm: {
    supported: boolean;
    simd: boolean;
    threads: boolean;
    features?: string[];
  };
}

export async function detectGPUCapabilities(): Promise<GPUCapabilities> {
  // Placeholder implementation
  const isWebGPUSupported = !!navigator.gpu;
  
  return {
    browserName: getBrowserName(),
    browserVersion: getBrowserVersion(),
    isMobile: isMobileDevice(),
    platform: getPlatform(),
    osVersion: getOSVersion(),
    gpu: {
      vendor: 'unknown',
      model: 'unknown',
      capabilities: {
        computeShaders: isWebGPUSupported,
        parallelCompilation: true
      }
    },
    webgpu: {
      supported: isWebGPUSupported,
      features: isWebGPUSupported ? ['basic', 'compute'] : []
    },
    wasm: {
      supported: typeof WebAssembly !== 'undefined',
      simd: false,
      threads: false
    }
  };
}

// Helper functions to detect browser info
function getBrowserName(): string {
  const userAgent = navigator.userAgent;
  
  if (userAgent.indexOf('Firefox') > -1) {
    return 'Firefox';
  } else if (userAgent.indexOf('Edge') > -1 || userAgent.indexOf('Edg/') > -1) {
    return 'Edge';
  } else if (userAgent.indexOf('Chrome') > -1) {
    return 'Chrome';
  } else if (userAgent.indexOf('Safari') > -1) {
    return 'Safari';
  }
  
  return 'Unknown';
}

function getBrowserVersion(): string {
  const userAgent = navigator.userAgent;
  let match;
  
  if ((match = userAgent.match(/(Firefox|Chrome|Safari|Edge|Edg)\/(\\d+\\.\\d+)/))) {
    return match[2];
  }
  
  return '0';
}

function isMobileDevice(): boolean {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

function getPlatform(): string {
  const userAgent = navigator.userAgent;
  
  if (userAgent.indexOf('Win') > -1) {
    return 'Windows';
  } else if (userAgent.indexOf('Mac') > -1) {
    return 'macOS';
  } else if (userAgent.indexOf('Linux') > -1) {
    return 'Linux';
  } else if (userAgent.indexOf('Android') > -1) {
    return 'Android';
  } else if (userAgent.indexOf('iPhone') > -1 || userAgent.indexOf('iPad') > -1) {
    return 'iOS';
  }
  
  return 'Unknown';
}

function getOSVersion(): string {
  const userAgent = navigator.userAgent;
  let match;
  
  if ((match = userAgent.match(/(Windows NT|Mac OS X|Android|iOS) ([\\d\\.]+)/))) {
    return match[2];
  }
  
  return '0';
}
"""
        with open(gpu_detection_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created GPU detection file: {gpu_detection_path}")
    
    # Create ml_detection.ts
    ml_detection_path = os.path.join(target_dir, "src/hardware/detection/ml_detection.ts")
    if os.path.exists(ml_detection_path) and not Config.FORCE:
        logger.info(f"ML detection file already exists, skipping: {ml_detection_path}")
    else:
        # Write the file
        content = """/**
 * Machine Learning capabilities detection
 */

export interface MLCapabilities {
  webnn: {
    supported: boolean;
    features: string[];
  };
}

export async function detectMLCapabilities(): Promise<MLCapabilities> {
  // Check for WebNN support
  const isWebNNSupported = 'ml' in navigator;
  
  return {
    webnn: {
      supported: isWebNNSupported,
      features: isWebNNSupported ? ['basic'] : []
    }
  };
}
"""
        with open(ml_detection_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created ML detection file: {ml_detection_path}")

def create_browser_capability_detection(target_dir):
    """Create browser capability detection file"""
    file_path = os.path.join(target_dir, "src/browser/optimizations/browser_capability_detection.ts")
    directory = os.path.dirname(file_path)
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(file_path) and not Config.FORCE:
        logger.info(f"Browser capability detection file already exists, skipping: {file_path}")
        return
    
    # Basic template for browser capability detection
    content = """/**
 * Browser capability detection
 */
import { BrowserCapabilities, OptimizationConfig } from '../../interfaces';

export function detectBrowserCapabilities(userAgent: string | null = null): BrowserCapabilities {
  // Use provided user agent or get from navigator
  const ua = userAgent || navigator.userAgent;
  
  // Default capabilities
  const capabilities: BrowserCapabilities = {
    browserName: 'Unknown',
    browserVersion: '0',
    isMobile: false,
    platform: 'Unknown',
    osVersion: '0',
    webgpuSupported: false,
    webgpuFeatures: [],
    webnnSupported: false,
    webnnFeatures: [],
    wasmSupported: true,
    wasmFeatures: [],
    metalApiSupported: false,
    metalApiVersion: '0',
    recommendedBackend: 'cpu',
    memoryLimitMB: 4096
  };
  
  // Detect browser name and version
  if (ua.indexOf('Firefox') > -1) {
    capabilities.browserName = 'Firefox';
    const match = ua.match(/Firefox\\/(\\d+)/);
    if (match) {
      capabilities.browserVersion = match[1];
    }
  } else if (ua.indexOf('Edge') > -1 || ua.indexOf('Edg/') > -1) {
    capabilities.browserName = 'Edge';
    const match = ua.match(/Edge\\/(\\d+)/) || ua.match(/Edg\\/(\\d+)/);
    if (match) {
      capabilities.browserVersion = match[1];
    }
  } else if (ua.indexOf('Chrome') > -1) {
    capabilities.browserName = 'Chrome';
    const match = ua.match(/Chrome\\/(\\d+)/);
    if (match) {
      capabilities.browserVersion = match[1];
    }
  } else if (ua.indexOf('Safari') > -1) {
    capabilities.browserName = 'Safari';
    const match = ua.match(/Version\\/(\\d+\\.\\d+)/);
    if (match) {
      capabilities.browserVersion = match[1];
    }
  }
  
  // Detect mobile
  capabilities.isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua);
  
  // Detect platform
  if (ua.indexOf('Win') > -1) {
    capabilities.platform = 'Windows';
  } else if (ua.indexOf('Mac') > -1) {
    capabilities.platform = 'macOS';
  } else if (ua.indexOf('Linux') > -1) {
    capabilities.platform = 'Linux';
  } else if (ua.indexOf('Android') > -1) {
    capabilities.platform = 'Android';
  } else if (ua.indexOf('iPhone') > -1 || ua.indexOf('iPad') > -1) {
    capabilities.platform = 'iOS';
  }
  
  // Check WebGPU support
  capabilities.webgpuSupported = !!navigator.gpu;
  
  // Check WebNN support
  capabilities.webnnSupported = 'ml' in navigator;
  
  // Check WebAssembly support
  capabilities.wasmSupported = typeof WebAssembly !== 'undefined';
  
  // Determine recommended backend
  if (capabilities.webgpuSupported) {
    capabilities.recommendedBackend = 'webgpu';
  } else if (capabilities.webnnSupported) {
    capabilities.recommendedBackend = 'webnn';
  } else if (capabilities.wasmSupported) {
    capabilities.recommendedBackend = 'wasm';
  } else {
    capabilities.recommendedBackend = 'cpu';
  }
  
  return capabilities;
}

export function getOptimizedConfig(
  modelName: string, 
  browserCapabilities: BrowserCapabilities,
  modelSizeMB: number = 0
): OptimizationConfig {
  // Determine model size if not provided
  if (modelSizeMB === 0) {
    if (modelName.includes('tiny')) {
      modelSizeMB = 100;
    } else if (modelName.includes('small')) {
      modelSizeMB = 300;
    } else if (modelName.includes('base')) {
      modelSizeMB = 500;
    } else if (modelName.includes('large')) {
      modelSizeMB = 1000;
    } else {
      modelSizeMB = 500; // Default
    }
  }
  
  // Default configuration
  const config: OptimizationConfig = {
    memoryOptimization: false,
    progressiveLoading: false,
    useQuantization: false,
    precision: 'float32',
    maxChunkSizeMB: 100,
    parallelLoading: true,
    specialOptimizations: {}
  };
  
  // Adjust based on model size
  if (modelSizeMB > 1000) {
    config.progressiveLoading = true;
    config.memoryOptimization = true;
  }
  
  if (modelSizeMB > 2000) {
    config.useQuantization = true;
    config.precision = 'int8';
  }
  
  // Adjust based on browser
  if (browserCapabilities.browserName === 'Safari') {
    config.parallelLoading = false; // Safari has issues with parallel loading
    config.maxChunkSizeMB = 50; // Smaller chunks for Safari
  }
  
  if (browserCapabilities.isMobile) {
    config.memoryOptimization = true;
    config.useQuantization = true;
    config.maxChunkSizeMB = 30;
  }
  
  return config;
}
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Created browser capability detection file: {file_path}")

def main():
    """Main function"""
    setup_args()
    
    # Create template files
    for template_type in Config.TYPES:
        if template_type in TEMPLATES:
            create_template_file(template_type, Config.TARGET_DIR)
        else:
            logger.warning(f"Unknown template type: {template_type}")
    
    # Create empty CPU backend file (for reference)
    create_empty_cpu_backend(Config.TARGET_DIR)
    
    # Create hardware detection files
    create_hardware_detection_files(Config.TARGET_DIR)
    
    # Create browser capability detection file
    create_browser_capability_detection(Config.TARGET_DIR)
    
    # Create index.ts files
    if Config.CREATE_INDEX:
        create_index_files(Config.TARGET_DIR)
    
    logger.info("Template creation completed successfully!")

if __name__ == "__main__":
    main()