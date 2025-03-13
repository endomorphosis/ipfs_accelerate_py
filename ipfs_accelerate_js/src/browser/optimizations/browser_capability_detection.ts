/**
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
    const match = ua.match(/Firefox\/(\d+)/);
    if (match) {
      capabilities.browserVersion = match[1];
    }
  } else if (ua.indexOf('Edge') > -1 || ua.indexOf('Edg/') > -1) {
    capabilities.browserName = 'Edge';
    const match = ua.match(/Edge\/(\d+)/) || ua.match(/Edg\/(\d+)/);
    if (match) {
      capabilities.browserVersion = match[1];
    }
  } else if (ua.indexOf('Chrome') > -1) {
    capabilities.browserName = 'Chrome';
    const match = ua.match(/Chrome\/(\d+)/);
    if (match) {
      capabilities.browserVersion = match[1];
    }
  } else if (ua.indexOf('Safari') > -1) {
    capabilities.browserName = 'Safari';
    const match = ua.match(/Version\/(\d+\.\d+)/);
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
