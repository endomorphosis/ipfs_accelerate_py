/**
 * WebNN Backend Standalone Implementation
 * For direct use of WebNN hardware acceleration without the full HAL
 */

import { WebNNBackend } from './webnn_backend';
export { WebNNBackend } from './webnn_backend';

/**
 * WebNN Backend Options Interface
 */
export interface WebNNBackendOptions {
  /**
   * Preferred device type for execution
   */
  deviceType?: 'gpu' | 'cpu';
  
  /**
   * Power preference (high-performance vs low-power)
   */
  powerPreference?: 'high-performance' | 'low-power' | 'default';
  
  /**
   * Enable additional logging for debugging
   */
  enableLogging?: boolean;
  
  /**
   * Default float precision
   */
  floatPrecision?: 'float32' | 'float16';
  
  /**
   * Whether to use sync execution when available
   */
  preferSyncExecution?: boolean;
  
  /**
   * Memory management options
   */
  memory?: {
    /**
     * Enable garbage collection of unused tensors
     */
    enableGarbageCollection?: boolean;
    
    /**
     * Threshold for garbage collection (in bytes)
     */
    garbageCollectionThreshold?: number;
  };
}

/**
 * WebNN Device Information
 */
export interface WebNNDeviceInfo {
  deviceType: string | null;
  deviceName: string | null;
  isSimulated: boolean;
  operations: string[];
}

/**
 * Browser recommendation for WebNN usage
 */
export interface WebNNBrowserRecommendation {
  /**
   * Best browser for WebNN support
   */
  bestBrowser: string;
  
  /**
   * Recommendation text for the user
   */
  recommendation: string;
  
  /**
   * Ranking of browsers by WebNN support quality
   */
  browserRanking: { [browser: string]: number };
  
  /**
   * Current browser detection
   */
  currentBrowser: string;
  
  /**
   * Whether the current browser is the recommended one
   */
  isUsingRecommendedBrowser: boolean;
}

/**
 * Result of the WebNN example run
 */
export interface WebNNExampleResult {
  /**
   * Whether WebNN is supported
   */
  supported: boolean;
  
  /**
   * Whether WebNN was successfully initialized
   */
  initialized: boolean;
  
  /**
   * Result data if the operation was successful
   */
  result?: number[];
  
  /**
   * Error message if the operation failed
   */
  error?: string;
  
  /**
   * Performance metrics in milliseconds
   */
  performance?: {
    /**
     * Time to initialize the backend
     */
    initializationTime: number;
    
    /**
     * Time to create the tensor
     */
    tensorCreationTime: number;
    
    /**
     * Time to execute the operation
     */
    executionTime: number;
    
    /**
     * Time to read the tensor back to CPU
     */
    readbackTime: number;
    
    /**
     * Total time for the entire operation
     */
    totalTime: number;
  };
}

/**
 * Create a WebNN backend with the specified options
 */
export async function createWebNNBackend(
  options: WebNNBackendOptions = {}
): Promise<WebNNBackend> {
  const backend = new WebNNBackend(options);
  await backend.initialize();
  return backend;
}

/**
 * Check if WebNN is supported in this browser
 */
export async function isWebNNSupported(): Promise<boolean> {
  const backend = new WebNNBackend();
  return await backend.isSupported();
}

/**
 * Get detailed WebNN device information
 */
export async function getWebNNDeviceInfo(): Promise<WebNNDeviceInfo | null> {
  const backend = new WebNNBackend({ enableLogging: false });
  if (!await backend.isSupported()) {
    return null;
  }
  
  await backend.initialize();
  const capabilities = await backend.getCapabilities();
  
  const result = {
    deviceType: capabilities.deviceType,
    deviceName: capabilities.deviceName,
    isSimulated: capabilities.isSimulated,
    operations: capabilities.operations
  };
  
  // Clean up
  backend.dispose();
  
  return result;
}

/**
 * Get browser-specific recommendations for WebNN usage
 */
export function getWebNNBrowserRecommendations(): WebNNBrowserRecommendation {
  const userAgent = navigator.userAgent.toLowerCase();
  
  // Detect browser
  const isEdge = userAgent.includes('edg');
  const isChrome = userAgent.includes('chrome') && !isEdge;
  const isFirefox = userAgent.includes('firefox');
  const isSafari = userAgent.includes('safari') && !isChrome && !isEdge;
  
  // Browser ranking for WebNN (based on current implementation status)
  const browserRanking = {
    edge: 4,     // Best WebNN support
    chrome: 3,   // Good but not as complete as Edge
    safari: 2,   // Limited support
    firefox: 1   // Limited support
  };
  
  let bestBrowser = 'edge';
  let recommendation = 'Microsoft Edge has the most complete WebNN implementation. For best results with neural network acceleration, use Edge.';
  let currentBrowser = 'unknown';
  
  if (isEdge) {
    currentBrowser = 'edge';
    recommendation = 'You are using Microsoft Edge which has the best WebNN support. Great choice!';
  } else if (isChrome) {
    currentBrowser = 'chrome';
    recommendation = 'Chrome has good but partial WebNN support. For better neural network acceleration, consider using Microsoft Edge.';
  } else if (isFirefox) {
    currentBrowser = 'firefox';
    recommendation = 'Firefox has limited WebNN support. For better neural network acceleration, consider using Microsoft Edge.';
  } else if (isSafari) {
    currentBrowser = 'safari';
    recommendation = 'Safari has experimental WebNN support. For better neural network acceleration, consider using Microsoft Edge.';
  }
  
  return {
    bestBrowser,
    recommendation,
    browserRanking,
    currentBrowser,
    isUsingRecommendedBrowser: currentBrowser === bestBrowser
  };
}

/**
 * Run a simple WebNN example to test functionality
 */
export async function runWebNNExample(
  operation: 'relu' | 'sigmoid' | 'tanh' | 'matmul' | 'softmax' | 
             'maxpool' | 'avgpool' | 'add' | 'sub' | 'mul' | 'div' | 
             'exp' | 'log' | 'sqrt' | 'reshape' | 'transpose' = 'relu'
): Promise<WebNNExampleResult> {
  try {
    const startTime = performance.now();
    
    // Create the backend
    const backend = new WebNNBackend();
    
    // Check if supported
    const isSupported = await backend.isSupported();
    if (!isSupported) {
      return {
        supported: false,
        initialized: false,
        error: 'WebNN is not supported in this browser'
      };
    }
    
    // Initialize
    const initStartTime = performance.now();
    const initialized = await backend.initialize();
    const initEndTime = performance.now();
    
    if (!initialized) {
      return {
        supported: true,
        initialized: false,
        error: 'Failed to initialize WebNN backend'
      };
    }
    
    // Create input tensor
    const tensorStartTime = performance.now();
    const inputTensor = await backend.createTensor(
      new Float32Array([1, 2, 3, 4]),
      [2, 2],
      'float32'
    );
    
    let inputTensor2;
    if (operation === 'matmul') {
      inputTensor2 = await backend.createTensor(
        new Float32Array([5, 6, 7, 8]),
        [2, 2],
        'float32'
      );
    }
    const tensorEndTime = performance.now();
    
    // Run operation
    const execStartTime = performance.now();
    let result;
    
    switch (operation) {
      case 'relu':
      case 'sigmoid':
      case 'tanh':
        result = await backend.execute('elementwise', {
          input: inputTensor
        }, {
          operation
        });
        break;
        
      case 'matmul':
        result = await backend.execute('matmul', {
          a: inputTensor,
          b: inputTensor2!
        });
        break;
        
      case 'softmax':
        result = await backend.execute('softmax', {
          input: inputTensor
        });
        break;
        
      // New elementwise operations
      case 'add':
      case 'sub':
      case 'mul':
      case 'div':
      case 'exp':
      case 'log':
      case 'sqrt':
        // Binary operations need two tensors
        if (['add', 'sub', 'mul', 'div'].includes(operation)) {
          if (!inputTensor2) {
            inputTensor2 = await backend.createTensor(
              new Float32Array([1, 1, 1, 1]),
              [2, 2],
              'float32'
            );
          }
          result = await backend.execute(operation, {
            a: inputTensor,
            b: inputTensor2
          });
        } else {
          // Unary operations need only one tensor
          result = await backend.execute(operation, {
            a: inputTensor
          });
        }
        break;
        
      // Pooling operations
      case 'maxpool':
      case 'avgpool':
        // Create a 4D tensor for pooling (batch, height, width, channels)
        const poolingTensor = await backend.createTensor(
          new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
          [1, 4, 4, 1],
          'float32'
        );
        
        result = await backend.execute(operation, {
          input: poolingTensor
        }, {
          windowDimensions: [2, 2],
          strides: [2, 2]
        });
        break;
        
      // Tensor manipulation operations
      case 'reshape':
        result = await backend.execute('reshape', {
          input: inputTensor
        }, {
          newShape: [1, 4]
        });
        break;
        
      case 'transpose':
        result = await backend.execute('transpose', {
          input: inputTensor
        }, {
          permutation: [1, 0]
        });
        break;
    }
    const execEndTime = performance.now();
    
    // Read results back to CPU
    const readStartTime = performance.now();
    const resultData = await backend.readTensor(
      result.tensor,
      result.shape,
      'float32'
    );
    const readEndTime = performance.now();
    
    // Clean up resources
    backend.dispose();
    
    const endTime = performance.now();
    
    return {
      supported: true,
      initialized: true,
      result: Array.from(resultData),
      performance: {
        initializationTime: initEndTime - initStartTime,
        tensorCreationTime: tensorEndTime - tensorStartTime,
        executionTime: execEndTime - execStartTime,
        readbackTime: readEndTime - readStartTime,
        totalTime: endTime - startTime
      }
    };
  } catch (error) {
    return {
      supported: true,
      initialized: true,
      error: error.message
    };
  }
}

/**
 * Check if this is the recommended browser for WebNN
 */
export function isRecommendedBrowserForWebNN(): boolean {
  const recommendation = getWebNNBrowserRecommendations();
  return recommendation.isUsingRecommendedBrowser;
}

/**
 * Get estimated WebNN performance tier based on device capabilities
 */
export async function getWebNNPerformanceTier(): Promise<'high' | 'medium' | 'low' | 'unsupported'> {
  try {
    const deviceInfo = await getWebNNDeviceInfo();
    
    if (!deviceInfo) {
      return 'unsupported';
    }
    
    // Check if it's a simulated implementation
    if (deviceInfo.isSimulated) {
      return 'low';
    }
    
    // Check operation support
    const coreOps = ['matmul', 'relu', 'sigmoid', 'tanh', 'conv2d'];
    const supportedCoreOps = coreOps.filter(op => deviceInfo.operations.includes(op));
    
    if (supportedCoreOps.length === coreOps.length) {
      return 'high';
    } else if (supportedCoreOps.length >= 3) {
      return 'medium';
    } else {
      return 'low';
    }
  } catch (error) {
    return 'unsupported';
  }
}

// Default export for easier imports
export default { 
  WebNNBackend,
  createWebNNBackend,
  isWebNNSupported,
  getWebNNDeviceInfo,
  getWebNNBrowserRecommendations,
  runWebNNExample,
  isRecommendedBrowserForWebNN,
  getWebNNPerformanceTier
};