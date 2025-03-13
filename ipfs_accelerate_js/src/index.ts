/**
 * IPFS Accelerate JS SDK
 * 
 * JavaScript SDK for accelerating AI models in web browsers using WebGPU, WebNN, and IPFS.
 * 
 * @packageDocumentation
 */

// Core types and interfaces
export * from './core/interfaces';

// Hardware acceleration
export * from './hardware';

// Tensor operations
export * from './tensor';

// Models
export * from './model';

// Re-export key factory functions
import createHardwareAbstraction from './hardware';
import { createModel } from './model';
import { createTensor, zeros, ones } from './tensor';

export { 
  createHardwareAbstraction,
  createModel,
  createTensor,
  zeros,
  ones
};

// Export the version
export const version = '0.4.0';

/**
 * Initialize the SDK with default settings
 */
export async function initialize(options: {
  logging?: boolean;
  preferredBackends?: string[];
  enableCache?: boolean;
} = {}): Promise<{
  hardware: any;
  createModel: typeof createModel;
  // Additional SDK components will be added here
}> {
  // Initialize hardware
  const hardware = await createHardwareAbstraction({
    logging: options.hasOwnProperty('logging') ? options.logging : false,
    preferredBackends: options.preferredBackends,
  });
  
  // Create a model factory bound to the hardware instance
  const modelFactory = (name: string, modelOptions = {}) => {
    return createModel(name, hardware, {
      ...modelOptions,
      enableCache: options.enableCache
    });
  };
  
  return {
    hardware,
    createModel: modelFactory,
    // Additional components will be added here
  };
}