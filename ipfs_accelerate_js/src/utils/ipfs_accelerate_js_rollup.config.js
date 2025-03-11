import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import { terser } from 'rollup-plugin-terser';
import copy from 'rollup-plugin-copy';
import generatePackageJson from 'rollup-plugin-generate-package-json';
import pkg from ".\/package.json";

const production = process.env.NODE_ENV === 'production';

// Shared plugins for all builds
const sharedPlugins = [
  resolve({
    browser: true,
    preferBuiltins: false
  }),
  commonjs(),
  typescript({
    tsconfig: './tsconfig.json',
    sourceMap: !production,
    declaration: true,
    declarationDir: 'dist/types'
  })
];

// Configuration for shader file handling
const shaderPlugin = copy({
  targets: [
    { 
      src: 'src/worker/webgpu/shaders/**/*',
      dest: 'dist/shaders' 
    }
  ]
});

export default [
  // Browser-friendly UMD build (full)
  {
    input: 'src/index.ts',
    output: {
      name: 'ipfsAccelerate',
      file: pkg.main,
      format: 'umd',
      sourcemap: !production,
      globals: {
        'react': 'React'
      }
    },
    external: ['react'],
    plugins: [
      ...sharedPlugins,
      production && terser({
        ecma: 2020,
        mangle: { toplevel: true },
        compress: {
          module: true,
          toplevel: true,
          unsafe_arrows: true,
          drop_console: false,
          drop_debugger: true
        },
        output: { comments: false }
      }),
      shaderPlugin
    ]
  },

  // ESM build for modern browsers (full)
  {
    input: 'src/index.ts',
    output: {
      file: pkg.module,
      format: 'es',
      sourcemap: !production,
    },
    external: ['react'],
    plugins: [
      ...sharedPlugins,
      production && terser({
        ecma: 2020,
        mangle: { toplevel: true },
        compress: {
          module: true,
          toplevel: true,
          unsafe_arrows: true,
          drop_console: false,
          drop_debugger: true
        },
        output: { comments: false }
      }),
      shaderPlugin
    ]
  },

  // React-specific bundle
  {
    input: 'src/react/index.ts',
    output: {
      file: 'dist/react/index.js',
      format: 'es',
      sourcemap: !production
    },
    external: ['react', '../index'],
    plugins: [
      ...sharedPlugins,
      production && terser({
        ecma: 2020,
        mangle: { toplevel: true },
        compress: {
          module: true,
          toplevel: true,
          unsafe_arrows: true,
          drop_console: false,
          drop_debugger: true
        },
        output: { comments: false }
      }),
      generatePackageJson({
        baseContents: {
          name: 'react',
          private: true,
          main: './index.js',
          module: './index.js',
          types: '../types/react/index.d.ts',
          peerDependencies: {
            react: "^16.8.0 || ^17.0.0 || ^18.0.0"
          }
        }
      })
    ]
  },

  // Core-only bundle (no React, smaller size)
  {
    input: 'src/core/index.ts',
    output: {
      file: 'dist/core/index.js',
      format: 'es',
      sourcemap: !production
    },
    plugins: [
      ...sharedPlugins,
      production && terser({
        ecma: 2020,
        mangle: { toplevel: true },
        compress: {
          module: true,
          toplevel: true,
          unsafe_arrows: true,
          drop_console: false,
          drop_debugger: true
        },
        output: { comments: false }
      }),
      generatePackageJson({
        baseContents: {
          name: 'core',
          private: true,
          main: './index.js',
          module: './index.js',
          types: '../types/core/index.d.ts'
        }
      })
    ]
  },

  // WebGPU-only bundle for specialized use cases
  {
    input: 'src/hardware/backends/webgpu_standalone.ts',
    output: {
      file: 'dist/webgpu/index.js',
      format: 'es',
      sourcemap: !production
    },
    plugins: [
      ...sharedPlugins,
      production && terser({
        ecma: 2020,
        mangle: { toplevel: true },
        compress: {
          module: true,
          toplevel: true,
          unsafe_arrows: true,
          drop_console: false,
          drop_debugger: true
        },
        output: { comments: false }
      }),
      copy({
        targets: [
          { 
            src: 'src/worker/webgpu/shaders/**/*',
            dest: 'dist/webgpu/shaders' 
          }
        ]
      }),
      generatePackageJson({
        baseContents: {
          name: 'webgpu',
          private: true,
          main: './index.js',
          module: './index.js',
          types: '../types/hardware/backends/webgpu_standalone.d.ts'
        }
      })
    ]
  },

  // Node.js specific bundle
  {
    input: 'src/node/index.ts',
    output: {
      file: 'dist/node/index.js',
      format: 'cjs',
      sourcemap: !production
    },
    external: ['fs', 'path', 'os', 'child_process'],
    plugins: [
      ...sharedPlugins,
      production && terser({
        ecma: 2020,
        mangle: { toplevel: true },
        compress: {
          module: true,
          toplevel: true,
          unsafe_arrows: true,
          drop_console: false,
          drop_debugger: true
        },
        output: { comments: false }
      }),
      generatePackageJson({
        baseContents: {
          name: 'node',
          private: true,
          main: './index.js',
          types: '../types/node/index.d.ts'
        }
      })
    ]
  }
];