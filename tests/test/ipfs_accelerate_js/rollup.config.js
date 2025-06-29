import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import terser from '@rollup/plugin-terser';
import dts from 'rollup-plugin-dts';
import packageJson from './package.json' assert { type: 'json' };

/**
 * Rollup configuration for bundling the IPFS Accelerate JS SDK v1.0.0.
 * Creates multiple bundle formats and model-specific bundles.
 */

// Banner for generated files
const banner = `/**
 * IPFS Accelerate JavaScript SDK v${packageJson.version}
 * Hardware-accelerated ML in browsers with WebGPU and WebNN
 * 
 * @license MIT
 * @copyright IPFS Accelerate Team
 */`;

const minBanner = `/** IPFS Accelerate v${packageJson.version} | MIT License */`;

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
        banner,
      },
      // Minified ESM
      {
        file: 'dist/ipfs-accelerate.esm.min.js',
        format: 'esm',
        sourcemap: true,
        banner: minBanner,
        plugins: [terser()],
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
        banner,
      },
      // Minified UMD (used for CDN)
      {
        file: 'dist/ipfs-accelerate.min.js',
        format: 'umd',
        name: 'IPFSAccelerate',
        plugins: [terser()],
        sourcemap: true,
        banner: minBanner,
      },
    ],
    plugins: sharedPlugins,
  },
  
  // TypeScript declaration files
  {
    input: 'dist/types/index.d.ts',
    output: {
      file: 'dist/ipfs-accelerate.d.ts',
      format: 'es',
      banner,
    },
    plugins: [dts()],
  },
  
  // Core-only bundle (no models)
  {
    input: 'src/core/index.ts',
    output: [
      {
        file: 'dist/core/index.js',
        format: 'esm',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/core/index.min.js',
        format: 'esm',
        sourcemap: true,
        banner: minBanner,
        plugins: [terser()],
      },
    ],
    plugins: sharedPlugins,
  },
  
  // Hardware Abstraction Layer
  {
    input: 'src/hardware/hardware_abstraction_layer.ts',
    output: [
      {
        file: 'dist/hal/index.js',
        format: 'esm',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/hal/index.min.js',
        format: 'esm',
        sourcemap: true,
        banner: minBanner,
        plugins: [terser()],
      },
    ],
    plugins: sharedPlugins,
  },
  
  // WebGPU Backend
  {
    input: 'src/hardware/webgpu/index.ts',
    output: [
      {
        file: 'dist/backends/webgpu.js',
        format: 'esm',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/backends/webgpu.min.js',
        format: 'esm',
        sourcemap: true,
        banner: minBanner,
        plugins: [terser()],
      },
    ],
    plugins: sharedPlugins,
  },
  
  // WebNN Backend
  {
    input: 'src/hardware/webnn/index.ts',
    output: [
      {
        file: 'dist/backends/webnn.js',
        format: 'esm',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/backends/webnn.min.js',
        format: 'esm',
        sourcemap: true,
        banner: minBanner,
        plugins: [terser()],
      },
    ],
    plugins: sharedPlugins,
  },
  
  // Cross-Model Tensor Sharing
  {
    input: 'src/tensor/shared_tensor.ts',
    output: [
      {
        file: 'dist/tensor/shared_tensor.js',
        format: 'esm',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/tensor/shared_tensor.min.js',
        format: 'esm',
        sourcemap: true,
        banner: minBanner,
        plugins: [terser()],
      },
    ],
    plugins: sharedPlugins,
  },
  
  // Hardware-Abstracted Model Bundles
  
  // BERT model bundle
  {
    input: 'src/model/hardware/bert.ts',
    output: [
      {
        file: 'dist/models/bert.esm.js',
        format: 'esm',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/models/bert.umd.js',
        format: 'umd',
        name: 'IPFSAccelerateBERT',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/models/bert.umd.min.js',
        format: 'umd',
        name: 'IPFSAccelerateBERT',
        plugins: [terser()],
        sourcemap: true,
        banner: minBanner,
      },
    ],
    plugins: sharedPlugins,
  },
  
  // Whisper model bundle
  {
    input: 'src/model/hardware/whisper.ts',
    output: [
      {
        file: 'dist/models/whisper.esm.js',
        format: 'esm',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/models/whisper.umd.js',
        format: 'umd',
        name: 'IPFSAccelerateWhisper',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/models/whisper.umd.min.js',
        format: 'umd',
        name: 'IPFSAccelerateWhisper',
        plugins: [terser()],
        sourcemap: true,
        banner: minBanner,
      },
    ],
    plugins: sharedPlugins,
  },
  
  // ViT model bundle
  {
    input: 'src/model/vision/hardware_abstracted_vit.ts',
    output: [
      {
        file: 'dist/models/vit.esm.js',
        format: 'esm',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/models/vit.umd.js',
        format: 'umd',
        name: 'IPFSAccelerateViT',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/models/vit.umd.min.js',
        format: 'umd',
        name: 'IPFSAccelerateViT',
        plugins: [terser()],
        sourcemap: true,
        banner: minBanner,
      },
    ],
    plugins: sharedPlugins,
  },
  
  // CLIP model bundle
  {
    input: 'src/model/hardware/clip.ts',
    output: [
      {
        file: 'dist/models/clip.esm.js',
        format: 'esm',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/models/clip.umd.js',
        format: 'umd',
        name: 'IPFSAccelerateCLIP',
        sourcemap: true,
        banner,
      },
      {
        file: 'dist/models/clip.umd.min.js',
        format: 'umd',
        name: 'IPFSAccelerateCLIP',
        plugins: [terser()],
        sourcemap: true,
        banner: minBanner,
      },
    ],
    plugins: sharedPlugins,
  },
];

export default configs;