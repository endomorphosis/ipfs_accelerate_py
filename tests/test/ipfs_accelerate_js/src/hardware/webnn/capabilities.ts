/**
 * WebNN Capabilities Detection
 * Utilities for detecting WebNN features and capabilities across browsers
 */

/**
 * WebNN feature support
 */
export interface WebNNFeatures {
  /** Whether WebNN is supported */
  supported: boolean;
  
  /** WebNN API version (if available) */
  version?: string;
  
  /** Whether the browser has hardware acceleration */
  hardwareAccelerated: boolean;
  
  /** Supported model formats */
  supportedModels: string[];
  
  /** Supported operations */
  supportedOperations: {
    basic: boolean;
    conv2d: boolean;
    pool: boolean;
    normalization: boolean;
    recurrent: boolean;
    transformer: boolean;
  };
  
  /** Browser information */
  browser: {
    name: string;
    version: string;
  };
  
  /** Hardware acceleration type */
  accelerationType?: 'cpu' | 'gpu' | 'npu' | 'dsp' | 'unknown';
  
  /** Preferred compute power preference */
  preferredPowerPreference?: 'default' | 'high-performance' | 'low-power';
}

/**
 * Detect WebNN features in the current browser
 * @returns Promise that resolves with WebNN features
 */
export async function detectWebNNFeatures(): Promise<WebNNFeatures> {
  const browserInfo = detectBrowser();
  
  // Default response structure for unsupported environment
  const defaultResponse: WebNNFeatures = {
    supported: false,
    hardwareAccelerated: false,
    supportedModels: [],
    supportedOperations: {
      basic: false,
      conv2d: false,
      pool: false,
      normalization: false,
      recurrent: false,
      transformer: false
    },
    browser: browserInfo
  };
  
  // Check if WebNN is available
  if (typeof navigator === 'undefined' || !('ml' in navigator)) {
    return defaultResponse;
  }
  
  try {
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const ml = navigator.ml;
    
    if (!ml || typeof ml.createContext !== 'function') {
      return defaultResponse;
    }
    
    // Try to create a context to check if it's actually working
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const context = await ml.createContext();
    
    if (!context) {
      return {
        ...defaultResponse,
        supported: true, // API exists but couldn't create context
        hardwareAccelerated: false
      };
    }
    
    // Detect features based on browser
    const features = getWebNNFeaturesForBrowser(browserInfo);
    
    // Try to detect hardware acceleration
    const accelerationType = await detectAccelerationType(context);
    
    return {
      ...features,
      supported: true,
      hardwareAccelerated: accelerationType !== 'cpu',
      accelerationType
    };
  } catch (error) {
    console.warn('Error detecting WebNN features:', error);
    return {
      ...defaultResponse,
      supported: true, // API exists but had an error
      hardwareAccelerated: false
    };
  }
}

/**
 * Detect browser information
 * @returns Browser name and version
 */
function detectBrowser(): { name: string; version: string } {
  if (typeof navigator === 'undefined') {
    return { name: 'unknown', version: 'unknown' };
  }
  
  const ua = navigator.userAgent;
  
  if (ua.includes('Firefox')) {
    const match = ua.match(/Firefox\/([\d.]+)/);
    return { name: 'firefox', version: match ? match[1] : 'unknown' };
  }
  
  if (ua.includes('Edg/')) {
    const match = ua.match(/Edg\/([\d.]+)/);
    return { name: 'edge', version: match ? match[1] : 'unknown' };
  }
  
  if (ua.includes('Chrome')) {
    const match = ua.match(/Chrome\/([\d.]+)/);
    return { name: 'chrome', version: match ? match[1] : 'unknown' };
  }
  
  if (ua.includes('Safari') && !ua.includes('Chrome')) {
    const match = ua.match(/Version\/([\d.]+)/);
    return { name: 'safari', version: match ? match[1] : 'unknown' };
  }
  
  return { name: 'unknown', version: 'unknown' };
}

/**
 * Get WebNN features based on browser
 * @param browser Browser information
 * @returns WebNN features for the browser
 */
function getWebNNFeaturesForBrowser(
  browser: { name: string; version: string }
): WebNNFeatures {
  // Default features that should work in any WebNN-supporting browser
  const defaultFeatures: WebNNFeatures = {
    supported: true,
    hardwareAccelerated: false,
    supportedModels: ['ONNX'],
    supportedOperations: {
      basic: true,
      conv2d: true,
      pool: true,
      normalization: true,
      recurrent: false,
      transformer: false
    },
    browser,
    preferredPowerPreference: 'default'
  };
  
  // Browser-specific feature detection
  switch (browser.name) {
    case 'chrome': {
      // Chrome has the most complete WebNN implementation
      const versionNumber = parseInt(browser.version.split('.')[0], 10);
      if (versionNumber >= 113) {
        return {
          ...defaultFeatures,
          hardwareAccelerated: true,
          supportedModels: ['ONNX', 'TensorFlow Lite'],
          supportedOperations: {
            basic: true,
            conv2d: true,
            pool: true,
            normalization: true,
            recurrent: true,
            transformer: versionNumber >= 121 // Transformer support in newer Chrome
          },
          preferredPowerPreference: 'high-performance'
        };
      }
      break;
    }
    
    case 'edge': {
      // Edge is based on Chromium and has good WebNN support
      const versionNumber = parseInt(browser.version.split('.')[0], 10);
      if (versionNumber >= 113) {
        return {
          ...defaultFeatures,
          hardwareAccelerated: true,
          supportedModels: ['ONNX', 'TensorFlow Lite'],
          supportedOperations: {
            basic: true,
            conv2d: true,
            pool: true,
            normalization: true,
            recurrent: true,
            transformer: versionNumber >= 121
          },
          preferredPowerPreference: 'high-performance'
        };
      }
      break;
    }
    
    case 'safari': {
      // Safari has WebNN support in newer versions
      const versionNumber = parseFloat(browser.version);
      if (versionNumber >= 17.0) {
        return {
          ...defaultFeatures,
          hardwareAccelerated: true,
          supportedModels: ['ONNX', 'CoreML'],
          supportedOperations: {
            basic: true,
            conv2d: true,
            pool: true,
            normalization: true,
            recurrent: versionNumber >= 17.4,
            transformer: versionNumber >= 17.4
          },
          preferredPowerPreference: 'high-performance'
        };
      }
      break;
    }
    
    case 'firefox': {
      // Firefox has experimental WebNN support
      const versionNumber = parseInt(browser.version.split('.')[0], 10);
      if (versionNumber >= 113) {
        return {
          ...defaultFeatures,
          hardwareAccelerated: false, // Limited hardware acceleration currently
          supportedModels: ['ONNX'],
          supportedOperations: {
            basic: true,
            conv2d: true,
            pool: true,
            normalization: true,
            recurrent: false,
            transformer: false
          },
          preferredPowerPreference: 'default'
        };
      }
      break;
    }
  }
  
  // Default for unknown or older browsers
  return defaultFeatures;
}

/**
 * Try to detect the acceleration type of WebNN
 * @param context WebNN context
 * @returns Acceleration type (gpu, cpu, npu, dsp, unknown)
 */
async function detectAccelerationType(context: any): Promise<'cpu' | 'gpu' | 'npu' | 'dsp' | 'unknown'> {
  try {
    // @ts-ignore - Using non-standard property that might be available
    if (context.deviceType) {
      // @ts-ignore
      const deviceType = context.deviceType.toLowerCase();
      
      if (deviceType.includes('gpu')) return 'gpu';
      if (deviceType.includes('cpu')) return 'cpu';
      if (deviceType.includes('npu') || deviceType.includes('neural')) return 'npu';
      if (deviceType.includes('dsp')) return 'dsp';
    }
    
    // Try to infer from the user agent
    const ua = navigator.userAgent.toLowerCase();
    
    // Check for Apple Neural Engine (NPU)
    if (ua.includes('mac') && ua.includes('safari') && parseInt(navigator.appVersion.match(/Version\/(\d+)/)?.[1] || '0', 10) >= 17) {
      return 'npu'; // Likely using ANE on newer Macs with Safari
    }
    
    // Check for mobile devices that might have NPUs
    if (ua.includes('android') && (ua.includes('snapdragon') || ua.includes('adreno'))) {
      return 'gpu'; // Likely using GPU or NPU on Snapdragon
    }
    
    // Default to GPU for Chrome/Edge as they tend to use GPU acceleration
    if (ua.includes('chrome') || ua.includes('edg')) {
      return 'gpu';
    }
    
    // Performance-based detection (run a simple benchmark)
    const accelerationType = await performanceBasedDetection(context);
    return accelerationType;
  } catch (error) {
    console.warn('Error detecting acceleration type:', error);
    return 'unknown';
  }
}

/**
 * Attempt to detect acceleration type based on performance
 * @param context WebNN context
 * @returns Acceleration type based on performance characteristics
 */
async function performanceBasedDetection(context: any): Promise<'cpu' | 'gpu' | 'npu' | 'dsp' | 'unknown'> {
  try {
    // Create a simple test graph
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const graph = await navigator.ml.createGraph();
    
    // Create large input tensors (1024x1024) to make the difference more apparent
    const input1 = graph.input('input1', {
      type: 'float32',
      dimensions: [1024, 1024]
    });
    
    const input2 = graph.input('input2', {
      type: 'float32',
      dimensions: [1024, 1024]
    });
    
    // Create a matrix multiplication operation (computationally intensive)
    const output = graph.matmul(input1, input2);
    graph.output('output', output);
    
    // Create input data
    const data1 = new Float32Array(1024 * 1024);
    const data2 = new Float32Array(1024 * 1024);
    
    // Fill with random data
    for (let i = 0; i < data1.length; i++) {
      data1[i] = Math.random();
      data2[i] = Math.random();
    }
    
    // Create output buffer
    const outputData = new Float32Array(1024 * 1024);
    
    // Start timing
    const start = performance.now();
    
    // Run the computation
    await context.compute(graph, {
      'input1': data1,
      'input2': data2
    }, {
      'output': outputData
    });
    
    // End timing
    const end = performance.now();
    const elapsed = end - start;
    
    // Analyze the results
    // Thresholds are approximate and might need adjustment
    if (elapsed < 50) {
      // Very fast execution suggests GPU or NPU
      return 'gpu';
    } else if (elapsed < 200) {
      // Moderately fast suggests GPU but could be a fast CPU
      return 'gpu';
    } else {
      // Slow execution suggests CPU
      return 'cpu';
    }
  } catch (error) {
    console.warn('Error in performance-based detection:', error);
    return 'unknown';
  }
}

/**
 * Check if the device has a neural processing unit
 * Based on hardware and browser detection
 * @returns Whether the device likely has an NPU
 */
export function hasNeuralProcessor(): boolean {
  if (typeof navigator === 'undefined') {
    return false;
  }
  
  const ua = navigator.userAgent.toLowerCase();
  
  // Apple devices with M1/M2/M3 chips have neural engines
  if (ua.includes('mac') && 
      (ua.includes('apple') || ua.includes('safari')) && 
      !ua.includes('intel')) {
    return true;
  }
  
  // Some newer Android devices have NPUs
  if (ua.includes('android')) {
    // Qualcomm Snapdragon 8xx series often have NPUs
    if (ua.includes('sm8') || ua.includes('snapdragon 8')) {
      return true;
    }
    
    // Samsung Exynos with NPUs
    if (ua.includes('exynos 9') || ua.includes('exynos 2')) {
      return true;
    }
    
    // Google Tensor chips
    if (ua.includes('tensor')) {
      return true;
    }
  }
  
  // Windows devices with Snapdragon
  if (ua.includes('windows') && ua.includes('snapdragon')) {
    return true;
  }
  
  return false;
}

/**
 * Get optimal power/performance preference for WebNN
 * @returns Optimal power preference
 */
export function getOptimalPowerPreference(): 'default' | 'high-performance' | 'low-power' {
  try {
    // Check if running on battery
    // @ts-ignore - Using non-standard API
    if (navigator.getBattery && navigator.getBattery().dischargingTime !== Infinity) {
      return 'low-power';
    }
    
    // Check if it's a mobile device
    if (/Android|iPhone|iPad|iPod|Windows Phone/i.test(navigator.userAgent)) {
      return 'low-power';
    }
    
    // Default to high performance on desktop
    return 'high-performance';
  } catch (error) {
    return 'default';
  }
}