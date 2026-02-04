/**
 * IPFS Accelerate JavaScript SDK
 * Provides hardware-accelerated machine learning directly in web browsers
 */

import { IPFSAccelerate, createAccelerator, AcceleratorOptions, AccelerateOptions } from './ipfs_accelerate_js_core';
import { HardwareAbstraction, createHardwareAbstraction, HardwareBackendType } from './ipfs_accelerate_js_hardware_abstraction';
import { detectHardwareCapabilities, HardwareCapabilities } from './ipfs_accelerate_js_hardware_detection';

// Re-export main components
export {
  // Core functionality
  IPFSAccelerate,
  createAccelerator,
  
  // Hardware abstraction
  HardwareAbstraction,
  createHardwareAbstraction,
  
  // Hardware detection
  detectHardwareCapabilities,
  
  // Types
  AcceleratorOptions,
  AccelerateOptions,
  HardwareCapabilities,
  HardwareBackendType
};

// Default export for easier importing
export default {
  createAccelerator,
  createHardwareAbstraction,
  detectHardwareCapabilities
};

/**
 * Example usage:
 * 
 * ```typescript
 * import { createAccelerator } from 'ipfs-accelerate';
 * 
 * async function runInference() {
 *   // Create accelerator with automatic hardware detection
 *   const accelerator = await createAccelerator({
 *     autoDetectHardware: true
 *   });
 *   
 *   // Run inference
 *   const result = await accelerator.accelerate({
 *     modelId: 'bert-base-uncased',
 *     modelType: 'text',
 *     input: 'This is a sample text for embedding.'
 *   });
 *   
 *   console.log(result);
 * }
 * 
 * runInference();
 * ```
 */
