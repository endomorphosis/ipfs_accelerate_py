import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import terser from '@rollup/plugin-terser';
import serve from 'rollup-plugin-serve';
import packageJson from './package.json' assert { type: 'json' };

/**
 * Rollup configuration for bundling IPFS Accelerate examples.
 */

// Shared plugins for all configurations
const sharedPlugins = [
  // Resolve node modules
  resolve(),
  // Convert CommonJS modules to ES6
  commonjs(),
  // Compile TypeScript
  typescript({
    tsconfig: './tsconfig.json',
    sourceMap: true,
  })
];

// Development mode plugin - add dev server
const devPlugins = [
  ...sharedPlugins,
  serve({
    open: true,
    contentBase: ['dist', 'examples'],
    host: 'localhost',
    port: 10001
  })
];

// Production mode plugins - add minification
const prodPlugins = [
  ...sharedPlugins,
  terser()
];

// Use development or production plugins based on environment
const plugins = process.env.NODE_ENV === 'production' ? prodPlugins : devPlugins;

// Example configurations
const examples = [
  // Basic tensor operations
  {
    input: 'examples/basic_tensor_operations.ts',
    output: {
      file: 'dist/examples/basic_tensor_operations.js',
      format: 'iife',
      name: 'TensorExample',
      sourcemap: true
    }
  },
  
  // BERT example
  {
    input: 'examples/bert_text_classification.ts',
    output: {
      file: 'dist/examples/bert_text_classification.js',
      format: 'iife',
      name: 'BertExample',
      sourcemap: true
    }
  },
  
  // ViT example
  {
    input: 'examples/vit_image_classification.ts',
    output: {
      file: 'dist/examples/vit_image_classification.js',
      format: 'iife',
      name: 'VitExample',
      sourcemap: true
    }
  },
  
  // Whisper example
  {
    input: 'examples/whisper_transcription.ts',
    output: {
      file: 'dist/examples/whisper_transcription.js',
      format: 'iife',
      name: 'WhisperExample',
      sourcemap: true
    }
  },
  
  // CLIP example
  {
    input: 'examples/clip_image_search.ts',
    output: {
      file: 'dist/examples/clip_image_search.js',
      format: 'iife',
      name: 'ClipExample',
      sourcemap: true
    }
  },
  
  // Cross-model tensor sharing example
  {
    input: 'examples/cross_model_tensor_sharing.ts',
    output: {
      file: 'dist/examples/cross_model_tensor_sharing.js',
      format: 'iife',
      name: 'TensorSharingExample',
      sourcemap: true
    }
  },
  
  // Browser optimization comparison
  {
    input: 'examples/browser_optimization_comparison.ts',
    output: {
      file: 'dist/examples/browser_optimization_comparison.js',
      format: 'iife',
      name: 'BrowserOptimizationExample',
      sourcemap: true
    }
  },
  
  // Hardware abstraction example
  {
    input: 'examples/hardware_abstraction_example.ts',
    output: {
      file: 'dist/examples/hardware_abstraction_example.js',
      format: 'iife',
      name: 'HardwareAbstractionExample',
      sourcemap: true
    }
  },
  
  // Multimodal example with BERT and ViT
  {
    input: 'examples/multimodal_bert_vit.ts',
    output: {
      file: 'dist/examples/multimodal_bert_vit.js',
      format: 'iife',
      name: 'MultimodalExample',
      sourcemap: true
    }
  }
];

// Add plugins to all examples
const examplesWithPlugins = examples.map(example => ({
  ...example,
  plugins
}));

export default examplesWithPlugins;