/**
 * Tensor module exports
 */
export * from './tensor';
export * from './tensor_operations';
export * from './tensor_sharing';

// Re-export tensor creation functions as the main API
import { createTensor, zeros, ones } from './tensor';
export { createTensor, zeros, ones };