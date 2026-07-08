/**
 * Performance Tracking System for WebGPU/WebNN Resource Pool Integration
 * 
 * This module provides comprehensive performance tracking and analysis for
 * hardware operations, enabling:
 * 
 * 1. Historical performance recording for all hardware operations
 * 2. Statistical analysis and trend detection
 * 3. Anomaly detection for performance issues
 * 4. Data-driven hardware backend recommendations
 * 5. Performance comparison across browsers and hardware
 */

import { BackendType } from './hardware_abstraction_layer';
import { BrowserType } from './webgpu/browser_optimized_operations';

/**
 * Record of a single operation's performance
 */
export interface PerformanceRecord {
  /** Timestamp when the operation was executed */
  timestamp: number;
  
  /** Name of the operation (e.g., 'matmul', 'conv2d') */
  operation: string;
  
  /** Type of backend used (e.g., 'webgpu', 'webnn', 'cpu') */
  backendType: string;
  
  /** Browser type if available */
  browserType?: BrowserType;
  
  /** Duration of the operation in milliseconds */
  durationMs: number;
  
  /** Shapes of input tensors */
  inputShapes: number[][];
  
  /** Shape of output tensor */
  outputShape: number[];
  
  /** Memory usage during operation in bytes (if available) */
  memoryUsage?: number;
  
  /** Whether the operation completed successfully */
  success: boolean;
  
  /** Type of error if operation failed */
  errorType?: string;
}

/**
 * Analysis of performance trends for an operation
 */
export interface PerformanceTrend {
  /** Name of the operation */
  operation: string;
  
  /** Mean execution time */
  mean: number;
  
  /** Median execution time */
  median: number;
  
  /** Minimum execution time */
  min: number;
  
  /** Maximum execution time */
  max: number;
  
  /** Standard deviation of execution times */
  stdDev: number;
  
  /** Overall trend of recent executions */
  trend: 'improving' | 'stable' | 'degrading';
  
  /** Performance records that deviate significantly from the mean */
  anomalies: PerformanceRecord[];
  
  /** Recommended backend for this operation (if available) */
  recommendedBackend?: BackendType;
}

/**
 * Performance tracker for hardware operations
 */
export class PerformanceTracker {
  /** Map of operation+backend to performance records */
  private records: Map<string, PerformanceRecord[]> = new Map();
  
  /** Maximum number of records to keep per operation+backend */
  private readonly maxRecordsPerOperation: number;
  
  /** Counts of operations by type */
  private operationCounts: Map<string, number> = new Map();
  
  /** Records of operation failures */
  private failureRecords: PerformanceRecord[] = [];
  
  /** 
   * Initialize the performance tracker
   * @param maxRecordsPerOperation Maximum records to keep per operation+backend combination
   */
  constructor(maxRecordsPerOperation: number = 100) {
    this.maxRecordsPerOperation = maxRecordsPerOperation;
    console.info(`PerformanceTracker initialized with maxRecordsPerOperation=${maxRecordsPerOperation}`);
  }
  
  /**
   * Track an operation's performance
   * @param record Performance record to track
   */
  trackOperation(record: PerformanceRecord): void {
    // Create a unique key for this operation + backend
    const key = `${record.operation}_${record.backendType}`;
    
    // Initialize record array if needed
    if (!this.records.has(key)) {
      this.records.set(key, []);
    }
    
    // Get the records array
    const records = this.records.get(key)!;
    
    // Add the new record
    records.push(record);
    
    // Keep only the most recent records
    if (records.length > this.maxRecordsPerOperation) {
      records.shift();
    }
    
    // Update operation count
    const countKey = record.operation;
    this.operationCounts.set(
      countKey, 
      (this.operationCounts.get(countKey) || 0) + 1
    );
    
    // Track failures separately for quick access
    if (!record.success) {
      this.failureRecords.push(record);
      
      // Limit failure records to avoid memory growth
      if (this.failureRecords.length > 1000) {
        this.failureRecords.shift();
      }
    }
    
    // Log for debugging (only in development)
    if (process.env.NODE_ENV === 'development') {
      console.debug(
        `Tracked ${record.operation} on ${record.backendType}: ` +
        `${record.durationMs.toFixed(2)}ms, success=${record.success}`
      );
    }
  }
  
  /**
   * Get performance history for a specific operation and backend
   * @param operation Name of the operation
   * @param backendType Type of backend
   * @returns Array of performance records
   */
  getOperationHistory(operation: string, backendType: string): PerformanceRecord[] {
    const key = `${operation}_${backendType}`;
    return this.records.get(key) || [];
  }
  
  /**
   * Analyze performance trend for a specific operation and backend
   * @param operation Name of the operation
   * @param backendType Type of backend
   * @returns Performance trend analysis
   */
  analyzeTrend(operation: string, backendType: string): PerformanceTrend {
    const history = this.getOperationHistory(operation, backendType);
    
    // Return default values if we don't have enough data
    if (history.length < 5) {
      return {
        operation,
        mean: 0,
        median: 0,
        min: 0,
        max: 0,
        stdDev: 0,
        trend: 'stable',
        anomalies: []
      };
    }
    
    // Filter out failures for performance calculations
    const successfulHistory = history.filter(r => r.success);
    
    if (successfulHistory.length === 0) {
      return {
        operation,
        mean: 0,
        median: 0,
        min: 0,
        max: 0,
        stdDev: 0,
        trend: 'stable',
        anomalies: []
      };
    }
    
    // Extract durations for calculations
    const durations = successfulHistory.map(r => r.durationMs);
    
    // Calculate mean
    const mean = durations.reduce((sum, val) => sum + val, 0) / durations.length;
    
    // Sort durations for median, min, max calculations
    const sortedDurations = [...durations].sort((a, b) => a - b);
    const median = sortedDurations[Math.floor(sortedDurations.length / 2)];
    const min = sortedDurations[0];
    const max = sortedDurations[sortedDurations.length - 1];
    
    // Calculate standard deviation
    const squaredDiffs = durations.map(d => Math.pow(d - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / durations.length;
    const stdDev = Math.sqrt(variance);
    
    // Detect anomalies (values more than 2 standard deviations from mean)
    const anomalies = successfulHistory.filter(r => 
      Math.abs(r.durationMs - mean) > 2 * stdDev
    );
    
    // Determine trend based on recent history
    // Get the most recent records
    const recentHistory = successfulHistory.slice(-5);
    const recentMean = recentHistory.reduce((sum, r) => sum + r.durationMs, 0) / recentHistory.length;
    
    let trend: 'improving' | 'stable' | 'degrading';
    if (recentMean < mean * 0.9) {
      trend = 'improving';
    } else if (recentMean > mean * 1.1) {
      trend = 'degrading';
    } else {
      trend = 'stable';
    }
    
    return {
      operation,
      mean,
      median,
      min,
      max,
      stdDev,
      trend,
      anomalies
    };
  }
  
  /**
   * Compare backend performance for a specific operation
   * @param operation Name of the operation
   * @returns Record mapping backend types to performance trends
   */
  compareBackendsForOperation(operation: string): Record<string, PerformanceTrend> {
    const backendTypes = new Set<string>();
    
    // Find all backends used for this operation
    for (const key of this.records.keys()) {
      if (key.startsWith(`${operation}_`)) {
        const backendType = key.split('_')[1];
        backendTypes.add(backendType);
      }
    }
    
    // Analyze trends for each backend
    const result: Record<string, PerformanceTrend> = {};
    for (const backendType of backendTypes) {
      result[backendType] = this.analyzeTrend(operation, backendType);
    }
    
    return result;
  }
  
  /**
   * Get the recommended backend for an operation based on performance
   * @param operation Name of the operation
   * @returns Recommended backend type or null if not enough data
   */
  getRecommendedBackend(operation: string): BackendType | null {
    const comparison = this.compareBackendsForOperation(operation);
    
    // Return null if we don't have any data for this operation
    if (Object.keys(comparison).length === 0) {
      return null;
    }
    
    // Find the backend with the lowest median duration
    // We use median because it's less affected by outliers
    let bestBackend: string | null = null;
    let bestMedian = Infinity;
    
    for (const [backend, trend] of Object.entries(comparison)) {
      // Skip backends with very few records
      const history = this.getOperationHistory(operation, backend);
      if (history.length < 5) {
        continue;
      }
      
      if (trend.median > 0 && trend.median < bestMedian) {
        bestMedian = trend.median;
        bestBackend = backend;
      }
    }
    
    return bestBackend as BackendType;
  }
  
  /**
   * Get recommended backends for all operations
   * @returns Record mapping operations to recommended backend types
   */
  getAllRecommendations(): Record<string, BackendType> {
    const recommendations: Record<string, BackendType> = {};
    const operations = new Set<string>();
    
    // Collect all operations
    for (const key of this.records.keys()) {
      const operation = key.split('_')[0];
      operations.add(operation);
    }
    
    // Get recommendations for each operation
    for (const operation of operations) {
      const recommended = this.getRecommendedBackend(operation);
      if (recommended) {
        recommendations[operation] = recommended;
      }
    }
    
    return recommendations;
  }
  
  /**
   * Analyze performance by browser type
   * @returns Analysis of browser-specific performance
   */
  analyzeBrowserPerformance(): Record<string, any> {
    const browserPerformance: Record<string, any> = {};
    const browserTypes = new Set<string>();
    
    // Collect all browsers
    for (const records of this.records.values()) {
      for (const record of records) {
        if (record.browserType) {
          browserTypes.add(record.browserType);
        }
      }
    }
    
    // Analyze each browser
    for (const browser of browserTypes) {
      const operations: Record<string, any> = {};
      const strengths: string[] = [];
      const weaknesses: string[] = [];
      
      // Collect operations for this browser
      const opSet = new Set<string>();
      for (const [key, records] of this.records.entries()) {
        const browserRecords = records.filter(r => r.browserType === browser);
        if (browserRecords.length > 0) {
          const operation = key.split('_')[0];
          opSet.add(operation);
        }
      }
      
      // Analyze each operation
      for (const operation of opSet) {
        // Get all backends for this operation
        const comparison = this.compareBackendsForOperation(operation);
        
        // Calculate average for this browser across backends
        let totalDuration = 0;
        let count = 0;
        
        for (const [backend, _] of Object.entries(comparison)) {
          const key = `${operation}_${backend}`;
          const records = this.records.get(key) || [];
          const browserRecords = records.filter(r => 
            r.browserType === browser && r.success
          );
          
          if (browserRecords.length > 0) {
            const avgDuration = browserRecords.reduce(
              (sum, r) => sum + r.durationMs, 0
            ) / browserRecords.length;
            
            totalDuration += avgDuration;
            count++;
          }
        }
        
        if (count > 0) {
          const avgDuration = totalDuration / count;
          operations[operation] = {
            avgDuration,
            recordCount: count
          };
        }
      }
      
      // Find strengths and weaknesses
      const allBrowsers = Array.from(browserTypes);
      for (const operation of Object.keys(operations)) {
        // Compare with other browsers
        const browserPerformances: Record<string, number> = {};
        let bestDuration = Infinity;
        let worstDuration = 0;
        let bestBrowser = '';
        
        for (const otherBrowser of allBrowsers) {
          // Get performance data for each browser
          const comparison = this.compareBackendsForOperation(operation);
          let browserTotalDuration = 0;
          let browserCount = 0;
          
          for (const [backend, _] of Object.entries(comparison)) {
            const key = `${operation}_${backend}`;
            const records = this.records.get(key) || [];
            const browserRecords = records.filter(r => 
              r.browserType === otherBrowser && r.success
            );
            
            if (browserRecords.length > 0) {
              const avgDuration = browserRecords.reduce(
                (sum, r) => sum + r.durationMs, 0
              ) / browserRecords.length;
              
              browserTotalDuration += avgDuration;
              browserCount++;
            }
          }
          
          if (browserCount > 0) {
            const avgDuration = browserTotalDuration / browserCount;
            browserPerformances[otherBrowser] = avgDuration;
            
            if (avgDuration < bestDuration) {
              bestDuration = avgDuration;
              bestBrowser = otherBrowser;
            }
            
            if (avgDuration > worstDuration) {
              worstDuration = avgDuration;
            }
          }
        }
        
        // Add to strengths if this browser is the best
        if (bestBrowser === browser) {
          strengths.push(operation);
        }
        
        // Add to weaknesses if this browser is significantly worse than the best
        if (
          browserPerformances[browser] &&
          browserPerformances[browser] > bestDuration * 1.5
        ) {
          weaknesses.push(operation);
        }
      }
      
      // Save browser analysis
      browserPerformance[browser] = {
        operations,
        strengths,
        weaknesses
      };
    }
    
    return browserPerformance;
  }
  
  /**
   * Export performance data for external analysis
   * @returns Comprehensive performance data
   */
  exportPerformanceData(): Record<string, any> {
    const operations: Record<string, any> = {};
    const browsers: Record<string, any> = {};
    let totalExecutions = 0;
    let successfulExecutions = 0;
    
    // Process operation data
    for (const [key, records] of this.records.entries()) {
      const [operation, backendType] = key.split('_');
      
      // Initialize operation object if needed
      if (!operations[operation]) {
        operations[operation] = {};
      }
      
      // Calculate statistics for this operation + backend
      const trend = this.analyzeTrend(operation, backendType);
      
      // Add to operation data
      operations[operation][backendType] = {
        trend,
        records: records.slice(-10) // Only include the 10 most recent records
      };
      
      // Update execution counts
      totalExecutions += records.length;
      successfulExecutions += records.filter(r => r.success).length;
    }
    
    // Get browser-specific performance
    const browserPerformance = this.analyzeBrowserPerformance();
    
    // Get recommendations
    const recommendations = this.getAllRecommendations();
    
    // Prepare comprehensive export
    return {
      summary: {
        totalOperations: Object.keys(operations).length,
        totalBackends: new Set(
          Array.from(this.records.keys()).map(k => k.split('_')[1])
        ).size,
        totalExecutions,
        successRate: totalExecutions > 0 ? successfulExecutions / totalExecutions : 0,
        recordCount: Array.from(this.records.values())
          .reduce((sum, arr) => sum + arr.length, 0),
        failureCount: this.failureRecords.length
      },
      operations,
      browsers: browserPerformance,
      recommendations,
      
      // Include recent failures for analysis
      recentFailures: this.failureRecords.slice(-10)
    };
  }
  
  /**
   * Clear all performance data
   */
  clearData(): void {
    this.records.clear();
    this.operationCounts.clear();
    this.failureRecords = [];
    console.info('Performance tracker data cleared');
  }
}

/**
 * Factory function to create a performance tracker
 * @param maxRecordsPerOperation Maximum records to keep per operation
 * @returns New PerformanceTracker instance
 */
export function createPerformanceTracker(
  maxRecordsPerOperation: number = 100
): PerformanceTracker {
  return new PerformanceTracker(maxRecordsPerOperation);
}