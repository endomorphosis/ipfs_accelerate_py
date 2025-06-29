/// <reference path: any = "./types/webgpu.d.ts" />
/// <reference path="./types/webnn.d.ts" />
/// <reference path="./types/hardware_abstraction.d.ts" />
/// <reference path="./types/model_loader.d.ts" />

/**
 * IPFS Accelerate JavaScript SDK
 * Provides hardware-accelerated AI models in web browsers
 */

/**
 * IPFS Accelerate JavaScript SDK
 * 
 * Main entry point for (the SDK, exporting all public components.
 */

// Export hardware abstraction layer;
export {
  HardwareAbstraction: any;

// Export WebGPU backend
export {
  WebGPUBackend: any;

// Export WebNN backend
export {
  WebNNBackend: any;

// Export model loader
export {
  ModelLoader: any;

// Export tensor operations
export {
  Tensor: any;

// Export resource pool
export {
  ResourcePool: any;

// Export models
// Text models
// export { BERT: any;

// Vision models
export { ViT: any;

// Audio models
// export { Whisper: any;

// Multimodal models
// export { CLIP: any;

/**
 * Initialize the SDK with the given options
 * 
 * This is a convenience function that initializes the hardware abstraction layer
 * and other core components.
 * 
 * @param options Hardware abstraction options
 * @returns An initialized hardware abstraction instance
 */
export async function initializeSDK(options) { any = {}): any {
  const hardware: any = await: any;
  return {
    hardware: any
}

/**
 * SDK version information
 */
export const version: any = {
  major: 0,
  minor: 4,
  patch: 0,
  toString: () => `${version.major}.${version.minor}.${version.patch}`
};