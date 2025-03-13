/**
 * Hardware Acceleration Module
 * Exports hardware backends and utilities
 */

// Interfaces
export * from './interfaces/hardware_backend';

// WebGPU backend
export * from './webgpu/backend';
export * from './webgpu/buffer_manager';
export * from './webgpu/shaders';

// Hardware detection utilities
export * from './detection/hardware_detector';

// Factory function to create the optimal backend
import { HardwareBackend } from './interfaces/hardware_backend';
import { WebGPUBackend, createWebGPUBackend } from './webgpu/backend';
import {
  detectHardware,
  optimizeHardwareSelection,
  HardwarePreference
} from './detection/hardware_detector';

/**
 * Create the optimal hardware backend based on hardware detection
 * @param preferences Optional hardware preferences
 * @returns Promise resolving to the hardware backend
 */
export async function createOptimalBackend(
  preferences: HardwarePreference = {}
): Promise<HardwareBackend> {
  try {
    // Detect available hardware
    const hardware = await detectHardware();
    
    // Select optimal backend type
    const backendType = optimizeHardwareSelection(hardware, preferences);
    
    // Create the selected backend
    switch (backendType) {
      case 'webgpu':
        return await createWebGPUBackend();
      
      // WebNN backend will be implemented next
      case 'webnn':
        // TODO: Implement WebNN backend
        console.warn('WebNN backend not yet implemented, falling back to WebGPU');
        if (hardware.hasWebGPU) {
          return await createWebGPUBackend();
        }
        throw new Error('WebNN backend not implemented');
      
      // WASM backend will be implemented next
      case 'wasm-simd':
      case 'wasm':
        // TODO: Implement WASM backend
        console.warn('WASM backend not yet implemented, falling back to CPU');
        throw new Error('WASM backend not implemented');
      
      case 'cpu':
      default:
        // TODO: Implement CPU backend
        // For now, throw error since CPU backend is not yet implemented
        throw new Error('CPU backend not implemented');
    }
  } catch (error) {
    console.error('Error creating optimal backend:', error);
    throw error;
  }
}