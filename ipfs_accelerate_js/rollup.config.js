import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import { terser } from 'rollup-plugin-terser';
import pkg from './package.json';

// Banner to add to the top of each file
const banner = `/**
 * IPFS Accelerate JS SDK v${pkg.version}
 * ${pkg.description}
 * 
 * @license ${pkg.license}
 * @copyright IPFS Accelerate Team
 */`;

export default [
  // Browser-friendly UMD build
  {
    input: 'src/index.ts',
    output: {
      name: 'IPFSAccelerate',
      file: pkg.main,
      format: 'umd',
      sourcemap: true,
      banner
    },
    plugins: [
      resolve(), 
      commonjs(),
      typescript({ tsconfig: './tsconfig.json' }),
      terser({
        format: {
          comments: function(node, comment) {
            return comment.type === 'comment2' && /@license/i.test(comment.value);
          }
        }
      })
    ]
  },
  
  // ESM build for modern bundlers
  {
    input: 'src/index.ts',
    output: {
      file: pkg.module,
      format: 'es',
      sourcemap: true,
      banner
    },
    plugins: [
      resolve(),
      typescript({ tsconfig: './tsconfig.json' })
    ],
    external: [...Object.keys(pkg.dependencies || {}), ...Object.keys(pkg.peerDependencies || {})]
  },
  
  // Individual model builds
  {
    input: 'src/model/transformers/bert.ts',
    output: {
      file: 'dist/models/bert.js',
      format: 'es',
      sourcemap: true,
      banner
    },
    plugins: [
      resolve(),
      typescript({ tsconfig: './tsconfig.json' })
    ],
    external: [...Object.keys(pkg.dependencies || {}), ...Object.keys(pkg.peerDependencies || {})]
  }
];