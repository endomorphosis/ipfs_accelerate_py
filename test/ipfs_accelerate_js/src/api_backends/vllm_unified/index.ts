import { VllmUnified } from './vllm_unified';
export * from './types';
export { VllmUnified };

/**
 * Factory function to create a new VllmUnified instance
 */
export default function createVllmUnified(resources = {}, metadata = {}) {
  return new VllmUnified(resources, metadata);
}