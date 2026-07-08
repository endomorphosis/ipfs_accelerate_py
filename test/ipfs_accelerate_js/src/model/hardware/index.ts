/**
 * Hardware Abstracted Models
 * Exports hardware-optimized model implementations with automatic backend selection
 */

// Hardware Abstracted BERT
export * from './bert';

// Hardware Abstracted Whisper
export * from './whisper';

// Hardware Abstracted CLIP
export * from './clip';

// Storage manager interface (common for all models)
import { StorageManager } from './bert';
export { StorageManager };

// Import Vision Transformer (ViT) implementation
import { 
  createHardwareAbstractedViT, 
  HardwareAbstractedViT, 
  HardwareAbstractedViTConfig 
} from '../vision/hardware_abstracted_vit';
export { 
  createHardwareAbstractedViT, 
  HardwareAbstractedViT, 
  HardwareAbstractedViTConfig 
};

// Factory functions for creating hardware abstracted models
import { createHardwareAbstractedBERT, HardwareAbstractedBERTConfig } from './bert';
import { createHardwareAbstractedWhisper, HardwareAbstractedWhisperConfig } from './whisper';
import { createHardwareAbstractedCLIP, HardwareAbstractedCLIPConfig } from './clip';

/**
 * Create a hardware-optimized model based on model type
 * @param modelType Type of model to create (bert, whisper, clip, vit)
 * @param config Model configuration
 * @param storageManager Storage manager for model weights
 * @param hal Hardware Abstraction Layer (required for ViT)
 * @returns Hardware abstracted model instance
 */
export function createHardwareAbstractedModel(
  modelType: 'bert' | 'whisper' | 'clip' | 'vit',
  config: any = {},
  storageManager: StorageManager,
  hal?: any
): any {
  switch (modelType) {
    case 'bert':
      return createHardwareAbstractedBERT(
        config as Partial<HardwareAbstractedBERTConfig>,
        storageManager
      );
    case 'whisper':
      return createHardwareAbstractedWhisper(
        config as Partial<HardwareAbstractedWhisperConfig>,
        storageManager
      );
    case 'clip':
      return createHardwareAbstractedCLIP(
        config as Partial<HardwareAbstractedCLIPConfig>,
        storageManager
      );
    case 'vit':
      if (!hal) {
        throw new Error('Hardware Abstraction Layer (hal) is required for ViT model');
      }
      return createHardwareAbstractedViT(
        hal,
        config as Partial<HardwareAbstractedViTConfig>
      );
    default:
      throw new Error(`Unsupported model type: ${modelType}`);
  }
}