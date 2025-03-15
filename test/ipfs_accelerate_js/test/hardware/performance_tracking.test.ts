import { PerformanceTracker, PerformanceRecord } from '../../src/hardware/performance_tracking';

describe('Performance Tracking', () => {
  let tracker: PerformanceTracker;
  
  beforeEach(() => {
    tracker = new PerformanceTracker(10); // Keep 10 records per operation
  });
  
  test('should track operation performance', () => {
    const record: PerformanceRecord = {
      timestamp: Date.now(),
      operation: 'matmul',
      backendType: 'webgpu',
      browserType: 'chrome',
      durationMs: 15.5,
      inputShapes: [[1, 128], [128, 64]],
      outputShape: [1, 64],
      success: true
    };
    
    tracker.trackOperation(record);
    
    const history = tracker.getOperationHistory('matmul', 'webgpu');
    expect(history).toBeDefined();
    expect(history.length).toBe(1);
    expect(history[0]).toEqual(record);
  });
  
  test('should respect max records per operation', () => {
    // Add 15 records (with max set to 10)
    for (let i = 0; i < 15; i++) {
      const record: PerformanceRecord = {
        timestamp: Date.now() + i,
        operation: 'matmul',
        backendType: 'webgpu',
        durationMs: 10 + i,
        inputShapes: [[1, 128], [128, 64]],
        outputShape: [1, 64],
        success: true
      };
      
      tracker.trackOperation(record);
    }
    
    const history = tracker.getOperationHistory('matmul', 'webgpu');
    expect(history.length).toBe(10); // Should keep only 10 records
    
    // Should keep the most recent 10 records (timestamps 5-14)
    const timestamps = history.map(r => r.timestamp);
    expect(Math.min(...timestamps)).toBeGreaterThanOrEqual(Date.now() + 5);
  });
  
  test('should analyze trends', () => {
    // Add records with improving performance
    for (let i = 0; i < 10; i++) {
      const record: PerformanceRecord = {
        timestamp: Date.now() + i,
        operation: 'matmul',
        backendType: 'webgpu',
        durationMs: 20 - i, // Decreasing time (improving performance)
        inputShapes: [[1, 128], [128, 64]],
        outputShape: [1, 64],
        success: true
      };
      
      tracker.trackOperation(record);
    }
    
    const trend = tracker.analyzeTrend('matmul', 'webgpu');
    expect(trend).toBeDefined();
    expect(trend.operation).toBe('matmul');
    expect(trend.mean).toBeGreaterThan(0);
    expect(trend.median).toBeGreaterThan(0);
    expect(trend.trend).toBe('improving');
  });
  
  test('should detect anomalies', () => {
    // Add 9 normal records
    for (let i = 0; i < 9; i++) {
      const record: PerformanceRecord = {
        timestamp: Date.now() + i,
        operation: 'matmul',
        backendType: 'webgpu',
        durationMs: 10, // Consistent performance
        inputShapes: [[1, 128], [128, 64]],
        outputShape: [1, 64],
        success: true
      };
      
      tracker.trackOperation(record);
    }
    
    // Add 1 anomaly (much slower)
    const anomalyRecord: PerformanceRecord = {
      timestamp: Date.now() + 10,
      operation: 'matmul',
      backendType: 'webgpu',
      durationMs: 100, // 10x slower
      inputShapes: [[1, 128], [128, 64]],
      outputShape: [1, 64],
      success: true
    };
    
    tracker.trackOperation(anomalyRecord);
    
    const trend = tracker.analyzeTrend('matmul', 'webgpu');
    expect(trend.anomalies).toBeDefined();
    expect(trend.anomalies.length).toBe(1);
    expect(trend.anomalies[0].durationMs).toBe(100);
  });
  
  test('should compare backends', () => {
    // Add records for webgpu backend
    for (let i = 0; i < 5; i++) {
      const record: PerformanceRecord = {
        timestamp: Date.now() + i,
        operation: 'matmul',
        backendType: 'webgpu',
        durationMs: 10,
        inputShapes: [[1, 128], [128, 64]],
        outputShape: [1, 64],
        success: true
      };
      
      tracker.trackOperation(record);
    }
    
    // Add records for webnn backend (faster)
    for (let i = 0; i < 5; i++) {
      const record: PerformanceRecord = {
        timestamp: Date.now() + i,
        operation: 'matmul',
        backendType: 'webnn',
        durationMs: 5, // Faster than webgpu
        inputShapes: [[1, 128], [128, 64]],
        outputShape: [1, 64],
        success: true
      };
      
      tracker.trackOperation(record);
    }
    
    // Add records for cpu backend (slower)
    for (let i = 0; i < 5; i++) {
      const record: PerformanceRecord = {
        timestamp: Date.now() + i,
        operation: 'matmul',
        backendType: 'cpu',
        durationMs: 50, // Slower than both
        inputShapes: [[1, 128], [128, 64]],
        outputShape: [1, 64],
        success: true
      };
      
      tracker.trackOperation(record);
    }
    
    const comparison = tracker.compareBackendsForOperation('matmul');
    expect(comparison).toBeDefined();
    expect(Object.keys(comparison).length).toBe(3);
    
    // WebNN should be fastest
    expect(comparison['webnn'].median).toBe(5);
    expect(comparison['webgpu'].median).toBe(10);
    expect(comparison['cpu'].median).toBe(50);
    
    // Check recommended backend
    const recommended = tracker.getRecommendedBackend('matmul');
    expect(recommended).toBe('webnn');
  });
  
  test('should handle browser-specific performance', () => {
    // Chrome performance for matmul
    for (let i = 0; i < 5; i++) {
      tracker.trackOperation({
        timestamp: Date.now() + i,
        operation: 'matmul',
        backendType: 'webgpu',
        browserType: 'chrome',
        durationMs: 10,
        inputShapes: [[1, 128], [128, 64]],
        outputShape: [1, 64],
        success: true
      });
    }
    
    // Firefox performance for matmul (better)
    for (let i = 0; i < 5; i++) {
      tracker.trackOperation({
        timestamp: Date.now() + i,
        operation: 'matmul',
        backendType: 'webgpu',
        browserType: 'firefox',
        durationMs: 8,
        inputShapes: [[1, 128], [128, 64]],
        outputShape: [1, 64],
        success: true
      });
    }
    
    // Chrome performance for conv2d (better)
    for (let i = 0; i < 5; i++) {
      tracker.trackOperation({
        timestamp: Date.now() + i,
        operation: 'conv2d',
        backendType: 'webgpu',
        browserType: 'chrome',
        durationMs: 15,
        inputShapes: [[1, 28, 28, 3], [3, 3, 3, 16]],
        outputShape: [1, 26, 26, 16],
        success: true
      });
    }
    
    // Firefox performance for conv2d (worse)
    for (let i = 0; i < 5; i++) {
      tracker.trackOperation({
        timestamp: Date.now() + i,
        operation: 'conv2d',
        backendType: 'webgpu',
        browserType: 'firefox',
        durationMs: 25,
        inputShapes: [[1, 28, 28, 3], [3, 3, 3, 16]],
        outputShape: [1, 26, 26, 16],
        success: true
      });
    }
    
    const browserPerformance = tracker.analyzeBrowserPerformance();
    expect(browserPerformance).toBeDefined();
    expect(Object.keys(browserPerformance).length).toBe(2);
    
    // Firefox should be stronger for matmul
    expect(browserPerformance['firefox'].strengths).toContain('matmul');
    
    // Chrome should be stronger for conv2d
    expect(browserPerformance['chrome'].strengths).toContain('conv2d');
  });
  
  test('should export performance data', () => {
    // Add some records
    tracker.trackOperation({
      timestamp: Date.now(),
      operation: 'matmul',
      backendType: 'webgpu',
      durationMs: 10,
      inputShapes: [[1, 128], [128, 64]],
      outputShape: [1, 64],
      success: true
    });
    
    tracker.trackOperation({
      timestamp: Date.now(),
      operation: 'transpose',
      backendType: 'webgpu',
      durationMs: 5,
      inputShapes: [[1, 128]],
      outputShape: [128, 1],
      success: true
    });
    
    const data = tracker.exportPerformanceData();
    expect(data).toBeDefined();
    expect(data.summary).toBeDefined();
    expect(data.operations).toBeDefined();
    expect(data.operations.matmul).toBeDefined();
    expect(data.operations.transpose).toBeDefined();
  });
  
  test('should clear data', () => {
    // Add some records
    tracker.trackOperation({
      timestamp: Date.now(),
      operation: 'matmul',
      backendType: 'webgpu',
      durationMs: 10,
      inputShapes: [[1, 128], [128, 64]],
      outputShape: [1, 64],
      success: true
    });
    
    // Clear the data
    tracker.clearData();
    
    // Check that history is empty
    const history = tracker.getOperationHistory('matmul', 'webgpu');
    expect(history).toEqual([]);
    
    // Check that exported data has empty operations
    const data = tracker.exportPerformanceData();
    expect(Object.keys(data.operations).length).toBe(0);
  });
});