import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import { terser } from 'rollup-plugin-terser';
import dts from 'rollup-plugin-dts';
import pkg from './package.json';

const production = !process.env.ROLLUP_WATCH;

export default [
  {
    input: 'ipfs_accelerate_js/src/index.ts',
    output: [
      {
        file: pkg.main,
        format: 'umd',
        name: 'ipfsAccelerate',
        sourcemap: true,
        globals: {
          react: 'React'
        }
      },
      {
        file: pkg.module,
        format: 'es',
        sourcemap: true,
      }
    ],
    plugins: [
      resolve({
        browser: true
      }),
      commonjs(),
      typescript({
        tsconfig: './tsconfig.json',
        sourceMap: !production,
        inlineSources: !production
      }),
      production && terser()
    ],
    external: ['react']
  },
  {
    input: 'ipfs_accelerate_js/src/index.ts',
    output: [
      {
        file: pkg.types,
        format: 'es'
      }
    ],
    plugins: [
      dts()
    ]
  },
  // Browser-specific bundles with optimizations
  {
    input: 'ipfs_accelerate_js/src/index.ts',
    output: [
      {
        file: 'dist/ipfs-accelerate.chrome.js',
        format: 'umd',
        name: 'ipfsAccelerate',
        sourcemap: true,
        globals: {
          react: 'React'
        }
      }
    ],
    plugins: [
      resolve({
        browser: true
      }),
      commonjs(),
      typescript({
        tsconfig: './tsconfig.json',
        sourceMap: !production,
        inlineSources: !production,
        define: {
          'process.env.BROWSER': JSON.stringify('chrome')
        }
      }),
      production && terser()
    ],
    external: ['react']
  },
  {
    input: 'ipfs_accelerate_js/src/index.ts',
    output: [
      {
        file: 'dist/ipfs-accelerate.firefox.js',
        format: 'umd',
        name: 'ipfsAccelerate',
        sourcemap: true,
        globals: {
          react: 'React'
        }
      }
    ],
    plugins: [
      resolve({
        browser: true
      }),
      commonjs(),
      typescript({
        tsconfig: './tsconfig.json',
        sourceMap: !production,
        inlineSources: !production,
        define: {
          'process.env.BROWSER': JSON.stringify('firefox')
        }
      }),
      production && terser()
    ],
    external: ['react']
  },
  {
    input: 'ipfs_accelerate_js/src/index.ts',
    output: [
      {
        file: 'dist/ipfs-accelerate.safari.js',
        format: 'umd',
        name: 'ipfsAccelerate',
        sourcemap: true,
        globals: {
          react: 'React'
        }
      }
    ],
    plugins: [
      resolve({
        browser: true
      }),
      commonjs(),
      typescript({
        tsconfig: './tsconfig.json',
        sourceMap: !production,
        inlineSources: !production,
        define: {
          'process.env.BROWSER': JSON.stringify('safari')
        }
      }),
      production && terser()
    ],
    external: ['react']
  }
];