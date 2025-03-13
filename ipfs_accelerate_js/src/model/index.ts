/**
 * Model module exports
 * 
 * This file exports all model implementations from the various sub-modules.
 */

// Export all transformer models
export * from './transformers';

// Vision models will be implemented and exported here
// export * from './vision';

// Audio models will be implemented and exported here
// export * from './audio';

// Factory function to create models
import { ModelOptions } from '../core/interfaces';
import { HardwareAbstraction } from '../hardware/hardware_abstraction';
import { createBertModel } from './transformers/bert';

/**
 * Create a model by name
 * 
 * @param modelName The name of the model to create
 * @param hardware The hardware abstraction layer
 * @param options Options for the model
 * @returns A promise that resolves to the created model
 */
export async function createModel(
  modelName: string,
  hardware: HardwareAbstraction,
  options: ModelOptions = {}
): Promise<any> {
  // Determine model type from name
  if (modelName.startsWith('bert')) {
    return createBertModel(modelName, hardware, options);
  }
  // Add more model types as they are implemented
  else {
    throw new Error(`Unsupported model: ${modelName}`);
  }
}