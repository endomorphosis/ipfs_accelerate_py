/**
 * Model exports for IPFS Accelerate JS
 */

// Transformer models (base implementations)
export * from './transformers/bert';

// Audio models (base implementations)
export * from './audio/whisper';

// Vision models (base implementations)
export * from './vision/vit';

// Hardware abstracted models (with automatic backend selection)
export * from './hardware';

// Export model types grouped by category
export const MODEL_TYPES = {
  // Text models
  TEXT: {
    BERT: 'bert-base-uncased',
  },
  
  // Vision models
  VISION: {
    VIT: 'google/vit-base-patch16-224',
  },
  
  // Audio models
  AUDIO: {
    WHISPER: 'openai/whisper-tiny',
  }
};