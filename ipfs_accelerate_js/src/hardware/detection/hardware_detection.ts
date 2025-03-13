/**
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
