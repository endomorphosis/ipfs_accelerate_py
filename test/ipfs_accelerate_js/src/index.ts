/**
 * IPFS Accelerate JavaScript SDK
 * Hardware-accelerated machine learning in the browser
 */

// Export tensor module and components
export * from './tensor';
export * from './tensor/shared_tensor';
export * from './tensor/operations';

// Export hardware acceleration components
export * from './hardware';
export * from './hardware/interfaces/hardware_backend';
export * from './hardware/webgpu/backend';
export * from './hardware/webnn/backend';
export * from './hardware/webnn/capabilities';
export * from './hardware/detection/hardware_detector';
export * from './hardware/hardware_abstraction_layer';
export * from './hardware/performance_tracking';
export * from './hardware/error_recovery';

// Export model implementations
export * from './model';

// Export API backends
export * from './api_backends';

// Export examples
export * from './examples/tensor_sharing_example';
export * from './examples/webgpu_tensor_example';
export * from './examples/webnn_tensor_example';

// Version information
export const VERSION = '1.0.0';
export const BUILD_DATE = '2025-03-18';