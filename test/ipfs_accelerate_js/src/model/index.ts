/**
 * Model exports for IPFS Accelerate JS
 */

// Transformer models
export * from './transformers/bert';

// Audio models
export * from './audio/whisper';

// Vision models
export * from './vision/vit';

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