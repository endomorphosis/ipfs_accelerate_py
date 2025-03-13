import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import terser from '@rollup/plugin-terser';
import dts from 'rollup-plugin-dts';
import packageJson from './package.json' assert { type: 'json' };

/**
 * Rollup configuration for bundling the IPFS Accelerate JS SDK.
 * Creates multiple bundle formats and model-specific bundles.
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
  }),
];

// Main configurations
const configs = [
  // ESM bundle (modern browsers, tree-shakable)
  {
    input: 'src/index.ts',
    output: [
      {
        file: packageJson.module,
        format: 'esm',
        sourcemap: true,
      },
    ],
    plugins: sharedPlugins,
  },
  
  // UMD bundle (older browsers, global variable)
  {
    input: 'src/index.ts',
    output: [
      {
        file: packageJson.main,
        format: 'umd',
        name: 'IPFSAccelerate',
        sourcemap: true,
      },
      // Minified UMD
      {
        file: packageJson.main.replace('.js', '.min.js'),
        format: 'umd',
        name: 'IPFSAccelerate',
        plugins: [terser()],
        sourcemap: true,
      },
    ],
    plugins: sharedPlugins,
  },
  
  // TypeScript declaration files
  {
    input: 'dist/dts/index.d.ts',
    output: {
      file: 'dist/ipfs-accelerate.d.ts',
      format: 'es',
    },
    plugins: [dts()],
  },
  
  // Model-specific bundles
  
  // BERT model bundle
  {
    input: 'src/model/transformers/bert.ts',
    output: [
      {
        file: 'dist/models/bert.esm.js',
        format: 'esm',
        sourcemap: true,
      },
      {
        file: 'dist/models/bert.umd.js',
        format: 'umd',
        name: 'IPFSAccelerateBERT',
        sourcemap: true,
      },
      {
        file: 'dist/models/bert.umd.min.js',
        format: 'umd',
        name: 'IPFSAccelerateBERT',
        plugins: [terser()],
        sourcemap: true,
      },
    ],
    plugins: sharedPlugins,
  },
  
  // Whisper model bundle
  {
    input: 'src/model/audio/whisper.ts',
    output: [
      {
        file: 'dist/models/whisper.esm.js',
        format: 'esm',
        sourcemap: true,
      },
      {
        file: 'dist/models/whisper.umd.js',
        format: 'umd',
        name: 'IPFSAccelerateWhisper',
        sourcemap: true,
      },
      {
        file: 'dist/models/whisper.umd.min.js',
        format: 'umd',
        name: 'IPFSAccelerateWhisper',
        plugins: [terser()],
        sourcemap: true,
      },
    ],
    plugins: sharedPlugins,
  },
  
  // ViT model bundle
  {
    input: 'src/model/vision/vit.ts',
    output: [
      {
        file: 'dist/models/vit.esm.js',
        format: 'esm',
        sourcemap: true,
      },
      {
        file: 'dist/models/vit.umd.js',
        format: 'umd',
        name: 'IPFSAccelerateViT',
        sourcemap: true,
      },
      {
        file: 'dist/models/vit.umd.min.js',
        format: 'umd',
        name: 'IPFSAccelerateViT',
        plugins: [terser()],
        sourcemap: true,
      },
    ],
    plugins: sharedPlugins,
  },
];

export default configs;