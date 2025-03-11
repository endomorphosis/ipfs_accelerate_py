/**
 * Storage Manager Implementation
 * 
 * This file provides persistent storage functionality using IndexedDB for
 * browser environments and file-based storage for Node.js environments.
 */

import { openDB, DBSchema, IDBPDatabase } from 'idb'; // We'll use the 'idb' library for IndexedDB

// Define the database schema
interface AccelerateDBSchema extends DBSchema {
  // Store for acceleration results
  'acceleration-results': {
    key: string; // UUID for the result
    value: AccelerationResult;
    indexes: {
      'by-model': string; // Model ID
      'by-hardware': string; // Hardware backend
      'by-date': number; // Timestamp
    };
  };
  // Store for quantized models
  'quantized-models': {
    key: string; // Model ID + quantization config hash
    value: QuantizedModelEntry;
    indexes: {
      'by-model': string; // Original model ID
      'by-bits': number; // Number of bits
      'by-date': number; // Timestamp
    };
  };
  // Store for performance metrics
  'performance-metrics': {
    key: string; // Metric ID
    value: PerformanceMetric;
    indexes: {
      'by-model': string; // Model ID
      'by-hardware': string; // Hardware backend
      'by-browser': string; // Browser name
      'by-date': number; // Timestamp
    };
  };
  // Store for device capabilities
  'device-capabilities': {
    key: string; // Device ID (user agent hash)
    value: DeviceCapabilities;
    indexes: {
      'by-browser': string; // Browser name
      'by-date': number; // Timestamp
    };
  };
}

// Types for database entries
interface AccelerationResult {
  id: string;
  modelId: string;
  modelType: string;
  hardware: string;
  processingTime: number;
  throughput: number;
  memoryUsage: number;
  browserInfo: string;
  timestamp: number;
  inputShape?: number[];
  outputShape?: number[];
  additionalInfo?: any;
}

interface QuantizedModelEntry {
  id: string;
  originalModelId: string;
  bits: number;
  scheme: string;
  mixedPrecision: boolean;
  backend: string;
  timestamp: number;
  size: number;
  data?: ArrayBuffer;
  metadata?: any;
}

interface PerformanceMetric {
  id: string;
  modelId: string;
  hardware: string;
  browser: string;
  metric: string;
  value: number;
  timestamp: number;
  additionalInfo?: any;
}

interface DeviceCapabilities {
  id: string;
  userAgent: string;
  browser: string;
  browserVersion: string;
  webgpu: {
    supported: boolean;
    details?: any;
  };
  webnn: {
    supported: boolean;
    details?: any;
  };
  wasm: {
    supported: boolean;
    details?: any;
  };
  timestamp: number;
}

export interface StorageManagerOptions {
  /** Database name for IndexedDB */
  databaseName?: string;
  /** Database version */
  storageVersion?: number;
  /** Directory path for Node.js file storage */
  storagePath?: string;
  /** Maximum storage size in MB */
  maxStorageSize?: number;
  /** Expiration time for old entries in days */
  expirationDays?: number;
  /** Enable logging */
  logging?: boolean;
}

/**
 * Storage manager for persistent storage of acceleration results and models
 */
export class StorageManager {
  private db: IDBPDatabase<AccelerateDBSchema> | null = null;
  private isNode: boolean = false;
  private initialized: boolean = false;
  private options: StorageManagerOptions;
  private fs: any = null; // Node.js fs module, if available
  private path: any = null; // Node.js path module, if available

  constructor(options: StorageManagerOptions = {}) {
    this.options = {
      databaseName: 'acceleration-results',
      storageVersion: 1,
      storagePath: './storage',
      maxStorageSize: 500, // 500 MB
      expirationDays: 30,
      logging: false,
      ...options
    };
    
    // Detect Node.js environment
    this.isNode = typeof window === 'undefined';
    
    // Load Node.js modules if in Node environment
    if (this.isNode) {
      try {
        this.fs = require('fs');
        this.path = require('path');
      } catch (error) {
        console.warn('Failed to load Node.js modules:', error);
      }
    }
  }

  /**
   * Initialize the storage manager
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) {
      return true;
    }
    
    try {
      if (this.isNode) {
        // Initialize file-based storage for Node.js
        await this.initializeNodeStorage();
      } else {
        // Initialize IndexedDB for browsers
        await this.initializeIndexedDB();
      }
      
      this.initialized = true;
      return true;
    } catch (error) {
      console.error('Failed to initialize storage manager:', error);
      return false;
    }
  }

  /**
   * Initialize IndexedDB for browser environments
   */
  private async initializeIndexedDB(): Promise<void> {
    const { databaseName, storageVersion } = this.options;
    
    this.db = await openDB<AccelerateDBSchema>(databaseName!, storageVersion!, {
      upgrade(db, oldVersion, newVersion, transaction) {
        // Create object stores and indexes
        if (!db.objectStoreNames.contains('acceleration-results')) {
          const resultStore = db.createObjectStore('acceleration-results', { keyPath: 'id' });
          resultStore.createIndex('by-model', 'modelId');
          resultStore.createIndex('by-hardware', 'hardware');
          resultStore.createIndex('by-date', 'timestamp');
        }
        
        if (!db.objectStoreNames.contains('quantized-models')) {
          const modelStore = db.createObjectStore('quantized-models', { keyPath: 'id' });
          modelStore.createIndex('by-model', 'originalModelId');
          modelStore.createIndex('by-bits', 'bits');
          modelStore.createIndex('by-date', 'timestamp');
        }
        
        if (!db.objectStoreNames.contains('performance-metrics')) {
          const metricStore = db.createObjectStore('performance-metrics', { keyPath: 'id' });
          metricStore.createIndex('by-model', 'modelId');
          metricStore.createIndex('by-hardware', 'hardware');
          metricStore.createIndex('by-browser', 'browser');
          metricStore.createIndex('by-date', 'timestamp');
        }
        
        if (!db.objectStoreNames.contains('device-capabilities')) {
          const capabilityStore = db.createObjectStore('device-capabilities', { keyPath: 'id' });
          capabilityStore.createIndex('by-browser', 'browser');
          capabilityStore.createIndex('by-date', 'timestamp');
        }
      }
    });
    
    if (this.options.logging) {
      console.log('IndexedDB initialized:', this.db.name, this.db.version);
    }
  }

  /**
   * Initialize file-based storage for Node.js environments
   */
  private async initializeNodeStorage(): Promise<void> {
    if (!this.fs || !this.path) {
      throw new Error('Node.js modules not available for file-based storage');
    }
    
    const { storagePath } = this.options;
    
    // Create storage directory if it doesn't exist
    if (!this.fs.existsSync(storagePath!)) {
      this.fs.mkdirSync(storagePath!, { recursive: true });
    }
    
    // Create subdirectories for different data types
    const directories = [
      'acceleration-results',
      'quantized-models',
      'performance-metrics',
      'device-capabilities'
    ];
    
    for (const dir of directories) {
      const dirPath = this.path.join(storagePath!, dir);
      if (!this.fs.existsSync(dirPath)) {
        this.fs.mkdirSync(dirPath);
      }
    }
    
    if (this.options.logging) {
      console.log('File-based storage initialized:', storagePath);
    }
  }

  /**
   * Store an acceleration result
   */
  async storeAccelerationResult(result: any): Promise<string> {
    if (!this.initialized) {
      throw new Error('Storage manager not initialized');
    }
    
    const id = this.generateUUID();
    const timestamp = Date.now();
    
    // Create the result object
    const resultEntry: AccelerationResult = {
      id,
      modelId: result.modelId || 'unknown',
      modelType: result.modelType || 'unknown',
      hardware: result.hardware || 'unknown',
      processingTime: result.processingTime || 0,
      throughput: result.throughput || 0,
      memoryUsage: result.memoryUsage || 0,
      browserInfo: result.browserInfo || navigator?.userAgent || 'unknown',
      timestamp,
      inputShape: result.inputShape,
      outputShape: result.outputShape,
      additionalInfo: result.additionalInfo
    };
    
    if (this.isNode) {
      // Store in file system for Node.js
      await this.storeNodeFile(
        'acceleration-results',
        `${id}.json`,
        JSON.stringify(resultEntry)
      );
    } else {
      // Store in IndexedDB for browsers
      await this.db!.add('acceleration-results', resultEntry);
    }
    
    // Clean up old entries
    await this.cleanupOldEntries();
    
    return id;
  }

  /**
   * Store a quantized model
   */
  async storeQuantizedModel(
    originalModelId: string,
    quantizationConfig: any,
    backend: string,
    model: any
  ): Promise<string> {
    if (!this.initialized) {
      throw new Error('Storage manager not initialized');
    }
    
    const id = `${originalModelId}-${quantizationConfig.bits}bit-${this.hashConfig(quantizationConfig)}`;
    const timestamp = Date.now();
    
    // Create the model entry
    const modelEntry: QuantizedModelEntry = {
      id,
      originalModelId,
      bits: quantizationConfig.bits,
      scheme: quantizationConfig.scheme || 'symmetric',
      mixedPrecision: quantizationConfig.mixedPrecision || false,
      backend,
      timestamp,
      size: model.size || 0,
      metadata: {
        ...quantizationConfig,
        backend
      }
    };
    
    // Add model data if available
    if (model.data) {
      modelEntry.data = model.data;
    }
    
    if (this.isNode) {
      // Store in file system for Node.js
      // Store metadata and data separately
      await this.storeNodeFile(
        'quantized-models',
        `${id}.meta.json`,
        JSON.stringify({ 
          ...modelEntry,
          data: undefined // Don't include data in metadata file
        })
      );
      
      // Store model data if available
      if (model.data) {
        await this.storeNodeFile(
          'quantized-models',
          `${id}.data.bin`,
          Buffer.from(model.data)
        );
      }
    } else {
      // Store in IndexedDB for browsers
      await this.db!.put('quantized-models', modelEntry);
    }
    
    return id;
  }

  /**
   * Get a quantized model
   */
  async getQuantizedModel(
    originalModelId: string,
    quantizationConfig: any,
    backend?: string
  ): Promise<any> {
    if (!this.initialized) {
      throw new Error('Storage manager not initialized');
    }
    
    // Generate ID based on model and config
    const configHash = this.hashConfig(quantizationConfig);
    const idPrefix = `${originalModelId}-${quantizationConfig.bits}bit-${configHash}`;
    
    if (this.isNode) {
      // Get from file system for Node.js
      const dirPath = this.path.join(this.options.storagePath!, 'quantized-models');
      
      // Find matching files
      const files = this.fs.readdirSync(dirPath);
      const metaFile = files.find((file: string) => 
        file.startsWith(idPrefix) && file.endsWith('.meta.json')
      );
      
      if (!metaFile) {
        return null;
      }
      
      // Read metadata
      const metaPath = this.path.join(dirPath, metaFile);
      const metadata = JSON.parse(this.fs.readFileSync(metaPath, 'utf8'));
      
      // Check if data file exists
      const dataFile = metaFile.replace('.meta.json', '.data.bin');
      const dataPath = this.path.join(dirPath, dataFile);
      
      if (this.fs.existsSync(dataPath)) {
        // Read data
        const data = this.fs.readFileSync(dataPath);
        metadata.data = data.buffer;
      }
      
      return metadata;
    } else {
      // Get from IndexedDB for browsers
      // Get all models with this original model ID
      const models = await this.db!.getAllFromIndex(
        'quantized-models',
        'by-model',
        originalModelId
      );
      
      // Filter by bits and scheme
      const filteredModels = models.filter(model => 
        model.bits === quantizationConfig.bits &&
        model.scheme === (quantizationConfig.scheme || 'symmetric') &&
        (backend ? model.backend === backend : true)
      );
      
      if (filteredModels.length === 0) {
        return null;
      }
      
      // Return the most recent model
      return filteredModels.sort((a, b) => b.timestamp - a.timestamp)[0];
    }
  }

  /**
   * Store performance metrics
   */
  async storePerformanceMetric(metric: {
    modelId: string;
    hardware: string;
    browser: string;
    metric: string;
    value: number;
    additionalInfo?: any;
  }): Promise<string> {
    if (!this.initialized) {
      throw new Error('Storage manager not initialized');
    }
    
    const id = this.generateUUID();
    const timestamp = Date.now();
    
    // Create the metric entry
    const metricEntry: PerformanceMetric = {
      id,
      modelId: metric.modelId,
      hardware: metric.hardware,
      browser: metric.browser,
      metric: metric.metric,
      value: metric.value,
      timestamp,
      additionalInfo: metric.additionalInfo
    };
    
    if (this.isNode) {
      // Store in file system for Node.js
      await this.storeNodeFile(
        'performance-metrics',
        `${id}.json`,
        JSON.stringify(metricEntry)
      );
    } else {
      // Store in IndexedDB for browsers
      await this.db!.add('performance-metrics', metricEntry);
    }
    
    return id;
  }

  /**
   * Store device capabilities
   */
  async storeDeviceCapabilities(capabilities: any): Promise<string> {
    if (!this.initialized) {
      throw new Error('Storage manager not initialized');
    }
    
    const userAgent = capabilities.userAgent || navigator?.userAgent || 'unknown';
    const id = this.hashString(userAgent);
    const timestamp = Date.now();
    
    // Create the capability entry
    const capabilityEntry: DeviceCapabilities = {
      id,
      userAgent,
      browser: capabilities.browser || 'unknown',
      browserVersion: capabilities.browserVersion || 'unknown',
      webgpu: capabilities.webgpu || { supported: false },
      webnn: capabilities.webnn || { supported: false },
      wasm: capabilities.wasm || { supported: false },
      timestamp
    };
    
    if (this.isNode) {
      // Store in file system for Node.js
      await this.storeNodeFile(
        'device-capabilities',
        `${id}.json`,
        JSON.stringify(capabilityEntry)
      );
    } else {
      // Store in IndexedDB for browsers
      await this.db!.put('device-capabilities', capabilityEntry);
    }
    
    return id;
  }

  /**
   * Get acceleration results with filtering
   */
  async getAccelerationResults(options: {
    modelName?: string;
    hardware?: string;
    limit?: number;
    offset?: number;
    startDate?: string;
    endDate?: string;
  } = {}): Promise<AccelerationResult[]> {
    if (!this.initialized) {
      throw new Error('Storage manager not initialized');
    }
    
    const { modelName, hardware, limit = 100, offset = 0, startDate, endDate } = options;
    
    if (this.isNode) {
      // Get from file system for Node.js
      const dirPath = this.path.join(this.options.storagePath!, 'acceleration-results');
      
      // Read all files
      const files = this.fs.readdirSync(dirPath);
      const results: AccelerationResult[] = [];
      
      for (const file of files) {
        if (file.endsWith('.json')) {
          const filePath = this.path.join(dirPath, file);
          const data = JSON.parse(this.fs.readFileSync(filePath, 'utf8'));
          
          // Apply filters
          if (
            (!modelName || data.modelId === modelName) &&
            (!hardware || data.hardware === hardware) &&
            (!startDate || data.timestamp >= new Date(startDate).getTime()) &&
            (!endDate || data.timestamp <= new Date(endDate).getTime())
          ) {
            results.push(data);
          }
        }
      }
      
      // Sort by timestamp (descending)
      results.sort((a, b) => b.timestamp - a.timestamp);
      
      // Apply limit and offset
      return results.slice(offset, offset + limit);
    } else {
      // Get from IndexedDB for browsers
      let results: AccelerationResult[] = [];
      
      if (modelName) {
        // Use index for model name
        results = await this.db!.getAllFromIndex('acceleration-results', 'by-model', modelName);
      } else if (hardware) {
        // Use index for hardware
        results = await this.db!.getAllFromIndex('acceleration-results', 'by-hardware', hardware);
      } else {
        // Get all results
        results = await this.db!.getAll('acceleration-results');
      }
      
      // Apply additional filters
      if (startDate || endDate) {
        results = results.filter(result => 
          (!startDate || result.timestamp >= new Date(startDate).getTime()) &&
          (!endDate || result.timestamp <= new Date(endDate).getTime())
        );
      }
      
      // Sort by timestamp (descending)
      results.sort((a, b) => b.timestamp - a.timestamp);
      
      // Apply limit and offset
      return results.slice(offset, offset + limit);
    }
  }

  /**
   * Get aggregated statistics
   */
  async getAggregatedStats(options: {
    groupBy?: 'hardware' | 'model' | 'browser';
    metrics?: string[];
    saveToFile?: boolean;
    outputPath?: string;
  } = {}): Promise<any> {
    if (!this.initialized) {
      throw new Error('Storage manager not initialized');
    }
    
    const { groupBy = 'hardware', metrics = ['avg_latency', 'throughput'], saveToFile = false, outputPath } = options;
    
    // Get all acceleration results
    const allResults = await this.getAccelerationResults({ limit: 1000 });
    
    // Group results
    const grouped: Record<string, any[]> = {};
    
    allResults.forEach(result => {
      let key: string;
      
      switch (groupBy) {
        case 'hardware':
          key = result.hardware;
          break;
        case 'model':
          key = result.modelId;
          break;
        case 'browser':
          key = result.browserInfo.split(' ')[0]; // Simple browser extraction
          break;
        default:
          key = 'all';
      }
      
      if (!grouped[key]) {
        grouped[key] = [];
      }
      
      grouped[key].push(result);
    });
    
    // Calculate statistics
    const stats: Record<string, any> = {};
    
    for (const [key, results] of Object.entries(grouped)) {
      stats[key] = {};
      
      // Calculate metrics
      if (metrics.includes('avg_latency')) {
        const latencies = results.map(r => r.processingTime);
        stats[key].avg_latency = this.calculateAverage(latencies);
      }
      
      if (metrics.includes('throughput')) {
        const throughputs = results.map(r => r.throughput);
        stats[key].throughput = this.calculateAverage(throughputs);
      }
      
      if (metrics.includes('memory')) {
        const memories = results.map(r => r.memoryUsage);
        stats[key].memory = this.calculateAverage(memories);
      }
      
      if (metrics.includes('count')) {
        stats[key].count = results.length;
      }
    }
    
    // Save to file if requested (Node.js only)
    if (saveToFile && this.isNode && outputPath) {
      const statsJson = JSON.stringify(stats, null, 2);
      this.fs.writeFileSync(outputPath, statsJson);
    }
    
    return stats;
  }

  /**
   * Generate a report
   */
  async generateReport(options: {
    format?: 'html' | 'markdown' | 'json';
    title?: string;
    includeCharts?: boolean;
    groupBy?: string;
    reportType?: 'benchmark' | 'performance' | 'compatibility';
    browserFilter?: string[];
    outputPath?: string;
  } = {}): Promise<string> {
    if (!this.initialized) {
      throw new Error('Storage manager not initialized');
    }
    
    const { 
      format = 'html',
      title = 'Acceleration Benchmark Report',
      includeCharts = true,
      groupBy = 'hardware',
      reportType = 'benchmark',
      browserFilter,
      outputPath
    } = options;
    
    // Get data
    const results = await this.getAccelerationResults({ limit: 100 });
    const stats = await this.getAggregatedStats({ groupBy });
    const capabilities = await this.getAllDeviceCapabilities();
    
    // Filter results by browser if requested
    const filteredResults = browserFilter 
      ? results.filter(r => browserFilter.some(b => r.browserInfo.includes(b))) 
      : results;
    
    // Generate report based on format
    let report = '';
    
    if (format === 'html') {
      report = this.generateHTMLReport({
        title,
        results: filteredResults,
        stats,
        capabilities,
        includeCharts,
        reportType
      });
    } else if (format === 'markdown') {
      report = this.generateMarkdownReport({
        title,
        results: filteredResults,
        stats,
        capabilities,
        reportType
      });
    } else {
      // JSON format
      report = JSON.stringify({
        title,
        timestamp: new Date().toISOString(),
        results: filteredResults,
        stats,
        capabilities
      }, null, 2);
    }
    
    // Save to file if requested (Node.js only)
    if (this.isNode && outputPath) {
      this.fs.writeFileSync(outputPath, report);
    }
    
    return report;
  }

  /**
   * Export results to a file
   */
  async exportResults(options: {
    format?: 'json' | 'csv';
    modelNames?: string[];
    hardwareTypes?: string[];
    startDate?: string;
    endDate?: string;
    filename?: string;
    outputDir?: string;
  } = {}): Promise<any> {
    if (!this.initialized) {
      throw new Error('Storage manager not initialized');
    }
    
    const { 
      format = 'json',
      modelNames,
      hardwareTypes,
      startDate,
      endDate,
      filename,
      outputDir
    } = options;
    
    // Get all results matching the filters
    const filters: any = {};
    
    if (modelNames && modelNames.length > 0) {
      // We'll need to filter after getting results since we can't query multiple model names at once
      filters.startDate = startDate;
      filters.endDate = endDate;
    } else {
      filters.startDate = startDate;
      filters.endDate = endDate;
    }
    
    // Get results
    let results = await this.getAccelerationResults(filters);
    
    // Apply additional filters
    if (modelNames && modelNames.length > 0) {
      results = results.filter(r => modelNames.includes(r.modelId));
    }
    
    if (hardwareTypes && hardwareTypes.length > 0) {
      results = results.filter(r => hardwareTypes.includes(r.hardware));
    }
    
    // Generate export data
    let exportData;
    
    if (format === 'json') {
      exportData = JSON.stringify(results, null, 2);
    } else {
      // CSV format
      // Generate CSV header
      const header = [
        'id', 'modelId', 'modelType', 'hardware', 'processingTime', 
        'throughput', 'memoryUsage', 'timestamp'
      ].join(',');
      
      // Generate CSV rows
      const rows = results.map(r => [
        r.id,
        r.modelId,
        r.modelType,
        r.hardware,
        r.processingTime,
        r.throughput,
        r.memoryUsage,
        new Date(r.timestamp).toISOString()
      ].join(','));
      
      exportData = [header, ...rows].join('\n');
    }
    
    // In browser environment, trigger download
    if (!this.isNode) {
      const extension = format === 'json' ? 'json' : 'csv';
      const downloadFilename = filename || `acceleration-results-${new Date().toISOString().slice(0, 10)}.${extension}`;
      
      const blob = new Blob([exportData], { type: format === 'json' ? 'application/json' : 'text/csv' });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = downloadFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      return { success: true, count: results.length };
    } else {
      // In Node.js environment, save to file
      if (!outputDir) {
        throw new Error('Output directory is required for Node.js export');
      }
      
      const extension = format === 'json' ? 'json' : 'csv';
      const outputFilename = filename || `acceleration-results-${new Date().toISOString().slice(0, 10)}.${extension}`;
      const outputPath = this.path.join(outputDir, outputFilename);
      
      this.fs.writeFileSync(outputPath, exportData);
      
      return { success: true, count: results.length, path: outputPath };
    }
  }

  /**
   * Clear old data to manage storage size
   */
  async clearOldData(options: {
    olderThan?: number; // Days
    types?: ('results' | 'models' | 'metrics' | 'capabilities')[];
  } = {}): Promise<number> {
    if (!this.initialized) {
      throw new Error('Storage manager not initialized');
    }
    
    const { olderThan = 30, types = ['results', 'models', 'metrics'] } = options;
    
    const cutoffTime = Date.now() - (olderThan * 24 * 60 * 60 * 1000);
    let removedCount = 0;
    
    if (this.isNode) {
      // Clear old data from file system for Node.js
      for (const type of types) {
        let dirName: string;
        
        switch (type) {
          case 'results':
            dirName = 'acceleration-results';
            break;
          case 'models':
            dirName = 'quantized-models';
            break;
          case 'metrics':
            dirName = 'performance-metrics';
            break;
          case 'capabilities':
            dirName = 'device-capabilities';
            break;
          default:
            continue;
        }
        
        const dirPath = this.path.join(this.options.storagePath!, dirName);
        
        if (!this.fs.existsSync(dirPath)) {
          continue;
        }
        
        const files = this.fs.readdirSync(dirPath);
        
        for (const file of files) {
          const filePath = this.path.join(dirPath, file);
          
          // Skip files that don't end with .json (like .bin files)
          if (!file.endsWith('.json') && type !== 'models') {
            continue;
          }
          
          // Read file to check timestamp
          const data = JSON.parse(this.fs.readFileSync(filePath, 'utf8'));
          
          if (data.timestamp && data.timestamp < cutoffTime) {
            // Delete file
            this.fs.unlinkSync(filePath);
            
            // If this is a model, also delete the data file
            if (type === 'models' && file.endsWith('.meta.json')) {
              const dataFilePath = filePath.replace('.meta.json', '.data.bin');
              if (this.fs.existsSync(dataFilePath)) {
                this.fs.unlinkSync(dataFilePath);
              }
            }
            
            removedCount++;
          }
        }
      }
    } else {
      // Clear old data from IndexedDB for browsers
      const tx = this.db!.transaction(
        ['acceleration-results', 'quantized-models', 'performance-metrics', 'device-capabilities'],
        'readwrite'
      );
      
      // Process each store based on types
      for (const type of types) {
        let storeName: keyof AccelerateDBSchema;
        
        switch (type) {
          case 'results':
            storeName = 'acceleration-results';
            break;
          case 'models':
            storeName = 'quantized-models';
            break;
          case 'metrics':
            storeName = 'performance-metrics';
            break;
          case 'capabilities':
            storeName = 'device-capabilities';
            break;
          default:
            continue;
        }
        
        // Get all entries
        const cursor = await tx.objectStore(storeName).index('by-date').openCursor();
        
        // Iterate through cursor
        while (cursor) {
          const entry = cursor.value;
          
          if (entry.timestamp < cutoffTime) {
            // Delete entry
            await cursor.delete();
            removedCount++;
          }
          
          await cursor.continue();
        }
      }
      
      // Commit transaction
      await tx.done;
    }
    
    return removedCount;
  }

  /**
   * Get storage statistics
   */
  async getStorageStats(): Promise<{
    totalSize: number;
    itemCounts: Record<string, number>;
    oldestEntry: number;
    newestEntry: number;
  }> {
    if (!this.initialized) {
      throw new Error('Storage manager not initialized');
    }
    
    if (this.isNode) {
      // Get storage stats from file system for Node.js
      const stats: any = {
        totalSize: 0,
        itemCounts: {
          'acceleration-results': 0,
          'quantized-models': 0,
          'performance-metrics': 0,
          'device-capabilities': 0
        },
        oldestEntry: Date.now(),
        newestEntry: 0
      };
      
      const storeNames = [
        'acceleration-results', 
        'quantized-models', 
        'performance-metrics', 
        'device-capabilities'
      ];
      
      for (const storeName of storeNames) {
        const dirPath = this.path.join(this.options.storagePath!, storeName);
        
        if (!this.fs.existsSync(dirPath)) {
          continue;
        }
        
        const files = this.fs.readdirSync(dirPath);
        let storeSize = 0;
        
        for (const file of files) {
          const filePath = this.path.join(dirPath, file);
          const fileStat = this.fs.statSync(filePath);
          
          storeSize += fileStat.size;
          
          // Count items (only count JSON files for stats)
          if (file.endsWith('.json')) {
            stats.itemCounts[storeName]++;
            
            // Check timestamps for oldest/newest
            const data = JSON.parse(this.fs.readFileSync(filePath, 'utf8'));
            
            if (data.timestamp) {
              stats.oldestEntry = Math.min(stats.oldestEntry, data.timestamp);
              stats.newestEntry = Math.max(stats.newestEntry, data.timestamp);
            }
          }
        }
        
        stats.totalSize += storeSize;
      }
      
      return stats;
    } else {
      // Get storage stats from IndexedDB for browsers
      const stats: any = {
        totalSize: 0, // This is approximate
        itemCounts: {},
        oldestEntry: Date.now(),
        newestEntry: 0
      };
      
      const storeNames = [
        'acceleration-results', 
        'quantized-models', 
        'performance-metrics', 
        'device-capabilities'
      ] as const;
      
      for (const storeName of storeNames) {
        // Count items
        const count = await this.db!.count(storeName);
        stats.itemCounts[storeName] = count;
        
        // Estimate size
        const items = await this.db!.getAll(storeName);
        let storeSize = 0;
        
        for (const item of items) {
          // Approximate size calculation
          const itemStr = JSON.stringify(item);
          storeSize += itemStr.length * 2; // Rough byte estimate for UTF-16
          
          // Check timestamps for oldest/newest
          if (item.timestamp) {
            stats.oldestEntry = Math.min(stats.oldestEntry, item.timestamp);
            stats.newestEntry = Math.max(stats.newestEntry, item.timestamp);
          }
        }
        
        stats.totalSize += storeSize;
      }
      
      return stats;
    }
  }

  /**
   * Get all device capabilities
   */
  private async getAllDeviceCapabilities(): Promise<DeviceCapabilities[]> {
    if (this.isNode) {
      // Get from file system for Node.js
      const dirPath = this.path.join(this.options.storagePath!, 'device-capabilities');
      
      if (!this.fs.existsSync(dirPath)) {
        return [];
      }
      
      const files = this.fs.readdirSync(dirPath);
      const capabilities: DeviceCapabilities[] = [];
      
      for (const file of files) {
        if (file.endsWith('.json')) {
          const filePath = this.path.join(dirPath, file);
          const data = JSON.parse(this.fs.readFileSync(filePath, 'utf8'));
          capabilities.push(data);
        }
      }
      
      return capabilities;
    } else {
      // Get from IndexedDB for browsers
      return await this.db!.getAll('device-capabilities');
    }
  }

  /**
   * Clean up old entries
   */
  private async cleanupOldEntries(): Promise<void> {
    const { expirationDays } = this.options;
    
    if (!expirationDays) {
      return;
    }
    
    // Clear old data
    await this.clearOldData({
      olderThan: expirationDays,
      types: ['results', 'metrics']
    });
  }

  /**
   * Store a file in the Node.js file system
   */
  private async storeNodeFile(directory: string, filename: string, data: string | Buffer): Promise<void> {
    if (!this.fs || !this.path) {
      throw new Error('Node.js modules not available for file-based storage');
    }
    
    const dirPath = this.path.join(this.options.storagePath!, directory);
    const filePath = this.path.join(dirPath, filename);
    
    this.fs.writeFileSync(filePath, data);
  }

  /**
   * Generate an HTML report
   */
  private generateHTMLReport(options: {
    title: string;
    results: AccelerationResult[];
    stats: any;
    capabilities: DeviceCapabilities[];
    includeCharts: boolean;
    reportType: string;
  }): string {
    const { title, results, stats, capabilities, includeCharts, reportType } = options;
    
    // Simple HTML report template
    return `
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${title}</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          h1, h2, h3 { color: #333; }
          .container { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
          table { border-collapse: collapse; width: 100%; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #f2f2f2; }
          tr:nth-child(even) { background-color: #f9f9f9; }
          .chart { width: 100%; height: 400px; background-color: #f5f5f5; border: 1px solid #ddd; margin-top: 20px; }
          .timestamp { color: #666; font-size: 0.8em; }
        </style>
        ${includeCharts ? '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>' : ''}
      </head>
      <body>
        <h1>${title}</h1>
        <p class="timestamp">Generated on ${new Date().toLocaleString()}</p>
        
        <div class="container">
          <h2>Hardware Information</h2>
          <table>
            <tr>
              <th>Browser</th>
              <th>Version</th>
              <th>WebGPU</th>
              <th>WebNN</th>
              <th>WebAssembly</th>
            </tr>
            ${capabilities.map(cap => `
              <tr>
                <td>${cap.browser}</td>
                <td>${cap.browserVersion}</td>
                <td>${cap.webgpu.supported ? '✅' : '❌'}</td>
                <td>${cap.webnn.supported ? '✅' : '❌'}</td>
                <td>${cap.wasm.supported ? '✅' : '❌'}</td>
              </tr>
            `).join('')}
          </table>
        </div>
        
        <div class="container">
          <h2>Performance Statistics</h2>
          <table>
            <tr>
              <th>Group</th>
              <th>Avg. Latency (ms)</th>
              <th>Throughput (items/s)</th>
              <th>Memory Usage (MB)</th>
              <th>Count</th>
            </tr>
            ${Object.entries(stats).map(([key, value]: [string, any]) => `
              <tr>
                <td>${key}</td>
                <td>${value.avg_latency?.toFixed(2) || 'N/A'}</td>
                <td>${value.throughput?.toFixed(2) || 'N/A'}</td>
                <td>${value.memory?.toFixed(2) || 'N/A'}</td>
                <td>${value.count || results.filter(r => 
                  options.reportType === 'hardware' ? r.hardware === key : 
                  options.reportType === 'model' ? r.modelId === key : true
                ).length}</td>
              </tr>
            `).join('')}
          </table>
          
          ${includeCharts ? `
            <div class="chart">
              <canvas id="performanceChart"></canvas>
            </div>
            <script>
              // Create chart
              const ctx = document.getElementById('performanceChart').getContext('2d');
              const chart = new Chart(ctx, {
                type: 'bar',
                data: {
                  labels: ${JSON.stringify(Object.keys(stats))},
                  datasets: [
                    {
                      label: 'Avg. Latency (ms)',
                      data: ${JSON.stringify(Object.values(stats).map((v: any) => v.avg_latency || 0))},
                      backgroundColor: 'rgba(54, 162, 235, 0.5)',
                      borderColor: 'rgba(54, 162, 235, 1)',
                      borderWidth: 1
                    },
                    {
                      label: 'Throughput (items/s)',
                      data: ${JSON.stringify(Object.values(stats).map((v: any) => v.throughput || 0))},
                      backgroundColor: 'rgba(75, 192, 192, 0.5)',
                      borderColor: 'rgba(75, 192, 192, 1)',
                      borderWidth: 1
                    }
                  ]
                },
                options: {
                  responsive: true,
                  scales: {
                    y: {
                      beginAtZero: true
                    }
                  }
                }
              });
            </script>
          ` : ''}
        </div>
        
        <div class="container">
          <h2>Recent Results</h2>
          <table>
            <tr>
              <th>Model ID</th>
              <th>Hardware</th>
              <th>Processing Time (ms)</th>
              <th>Throughput (items/s)</th>
              <th>Memory Usage (MB)</th>
              <th>Date</th>
            </tr>
            ${results.slice(0, 10).map(result => `
              <tr>
                <td>${result.modelId}</td>
                <td>${result.hardware}</td>
                <td>${result.processingTime.toFixed(2)}</td>
                <td>${result.throughput.toFixed(2)}</td>
                <td>${result.memoryUsage.toFixed(2)}</td>
                <td>${new Date(result.timestamp).toLocaleString()}</td>
              </tr>
            `).join('')}
          </table>
        </div>
      </body>
      </html>
    `;
  }

  /**
   * Generate a Markdown report
   */
  private generateMarkdownReport(options: {
    title: string;
    results: AccelerationResult[];
    stats: any;
    capabilities: DeviceCapabilities[];
    reportType: string;
  }): string {
    const { title, results, stats, capabilities, reportType } = options;
    
    // Simple Markdown report template
    return `
# ${title}

*Generated on ${new Date().toLocaleString()}*

## Hardware Information

| Browser | Version | WebGPU | WebNN | WebAssembly |
|---------|---------|--------|-------|------------|
${capabilities.map(cap => `| ${cap.browser} | ${cap.browserVersion} | ${cap.webgpu.supported ? '✅' : '❌'} | ${cap.webnn.supported ? '✅' : '❌'} | ${cap.wasm.supported ? '✅' : '❌'} |`).join('\n')}

## Performance Statistics

| Group | Avg. Latency (ms) | Throughput (items/s) | Memory Usage (MB) | Count |
|-------|------------------|---------------------|------------------|-------|
${Object.entries(stats).map(([key, value]: [string, any]) => `| ${key} | ${value.avg_latency?.toFixed(2) || 'N/A'} | ${value.throughput?.toFixed(2) || 'N/A'} | ${value.memory?.toFixed(2) || 'N/A'} | ${value.count || results.filter(r => 
  reportType === 'hardware' ? r.hardware === key : 
  reportType === 'model' ? r.modelId === key : true
).length} |`).join('\n')}

## Recent Results

| Model ID | Hardware | Processing Time (ms) | Throughput (items/s) | Memory Usage (MB) | Date |
|----------|---------|---------------------|---------------------|------------------|------|
${results.slice(0, 10).map(result => `| ${result.modelId} | ${result.hardware} | ${result.processingTime.toFixed(2)} | ${result.throughput.toFixed(2)} | ${result.memoryUsage.toFixed(2)} | ${new Date(result.timestamp).toLocaleString()} |`).join('\n')}
    `;
  }

  /**
   * Generate a UUID
   */
  private generateUUID(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  /**
   * Hash a string (simple implementation)
   */
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(16);
  }

  /**
   * Hash a configuration object
   */
  private hashConfig(config: any): string {
    return this.hashString(JSON.stringify(config));
  }

  /**
   * Calculate average of an array of numbers
   */
  private calculateAverage(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  /**
   * Close the database connection
   */
  async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
    
    this.initialized = false;
  }

  /**
   * Clean up resources
   */
  async dispose(): Promise<void> {
    await this.close();
  }
}