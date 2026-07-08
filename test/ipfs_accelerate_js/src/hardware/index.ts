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

// WebNN backend
export * from './webnn/backend';
export * from './webnn/capabilities';

// Hardware detection utilities
export * from './detection/hardware_detector';

// Export Hardware Abstraction Layer
export { 
  HardwareAbstractionLayer, 
  createHardwareAbstractionLayer,
  BackendType,
  ModelType,
  HardwareRecommendation,
  BackendSelectionCriteria,
  HardwareAbstractionLayerConfig
} from './hardware_abstraction_layer';

// Export Performance Tracking
export {
  PerformanceTracker,
  PerformanceRecord,
  PerformanceTrend,
  createPerformanceTracker
} from './performance_tracking';

// Export Error Recovery System
export {
  ErrorRecoveryManager,
  ErrorCategory,
  BackendSwitchStrategy,
  OperationFallbackStrategy,
  BrowserSpecificRecoveryStrategy,
  ParameterAdjustmentStrategy,
  RecoveryContext,
  ErrorRecoveryStrategy,
  RecoveryResult,
  createErrorRecoveryManager
} from './error_recovery';

// Factory function to create the optimal backend
import { HardwareBackend } from './interfaces/hardware_backend';
import { WebGPUBackend, createWebGPUBackend } from './webgpu/backend';
import { WebNNBackend, createWebNNBackend } from './webnn/backend';
import { detectWebNNFeatures } from './webnn/capabilities';
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
    
    // Detect WebNN features
    let webnnFeatures = { supported: false, hardwareAccelerated: false };
    if (hardware.hasWebNN) {
      webnnFeatures = await detectWebNNFeatures();
    }
    
    // Select optimal backend type
    const backendType = optimizeHardwareSelection(hardware, preferences);
    
    // Create the selected backend
    switch (backendType) {
      case 'webgpu':
        return await createWebGPUBackend();
      
      case 'webnn':
        // Check if WebNN is truly hardware accelerated
        if (webnnFeatures.supported && webnnFeatures.hardwareAccelerated) {
          try {
            return await createWebNNBackend();
          } catch (webnnError) {
            console.warn('WebNN backend initialization failed, falling back to WebGPU:', webnnError);
            if (hardware.hasWebGPU) {
              return await createWebGPUBackend();
            }
            throw webnnError;
          }
        } else if (hardware.hasWebGPU) {
          console.warn('WebNN not hardware accelerated, falling back to WebGPU');
          return await createWebGPUBackend();
        }
        throw new Error('WebNN backend not available with hardware acceleration');
      
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

/**
 * Creates a multi-backend with priority fallback
 * @param backends Array of backend types to try in order of preference
 * @returns Promise resolving to the first available backend
 */
export async function createMultiBackend(
  backends: ('webgpu' | 'webnn' | 'wasm' | 'cpu')[] = ['webnn', 'webgpu', 'wasm', 'cpu']
): Promise<HardwareBackend> {
  for (const backend of backends) {
    try {
      switch (backend) {
        case 'webgpu': {
          const hardware = await detectHardware();
          if (hardware.hasWebGPU) {
            return await createWebGPUBackend();
          }
          break;
        }
        case 'webnn': {
          const features = await detectWebNNFeatures();
          if (features.supported) {
            return await createWebNNBackend();
          }
          break;
        }
        // Additional backends will be added in the future
      }
    } catch (error) {
      console.warn(`Failed to initialize ${backend} backend:`, error);
      // Continue to next backend
    }
  }
  
  throw new Error('No suitable hardware backend available');
}