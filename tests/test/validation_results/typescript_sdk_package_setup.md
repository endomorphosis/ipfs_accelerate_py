# JavaScript SDK Package Setup Guide

This document provides detailed instructions for setting up and publishing the IPFS Accelerate JavaScript SDK package after the TypeScript migration is complete.

## Package Configuration

### Package Metadata

Create a comprehensive `package.json` file with the following structure:

```json
{
  "name": "ipfs-accelerate",
  "version": "0.1.0",
  "description": "IPFS Accelerate JavaScript SDK for AI model acceleration in web browsers using WebGPU and WebNN",
  "author": "IPFS Accelerate Team",
  "license": "MIT",
  "keywords": [
    "ai",
    "webgpu",
    "webnn",
    "llm",
    "machine-learning",
    "browser",
    "acceleration"
  ],
  "main": "dist/cjs/index.js",
  "module": "dist/esm/index.js",
  "browser": "dist/umd/ipfs-accelerate.js",
  "types": "dist/types/index.d.ts",
  "files": [
    "dist",
    "README.md",
    "LICENSE"
  ],
  "scripts": {
    "build": "rollup -c",
    "build:types": "tsc --emitDeclarationOnly",
    "test": "jest",
    "lint": "eslint src --ext .ts,.tsx",
    "docs": "typedoc",
    "clean": "rimraf dist",
    "prepare": "npm run clean && npm run build",
    "prepublishOnly": "npm run lint && npm run test"
  },
  "dependencies": {
    "tslib": "^2.5.0"
  },
  "peerDependencies": {
    "react": "^16.8.0 || ^17.0.0 || ^18.0.0"
  },
  "peerDependenciesMeta": {
    "react": {
      "optional": true
    }
  },
  "devDependencies": {
    "@rollup/plugin-commonjs": "^24.0.0",
    "@rollup/plugin-node-resolve": "^15.0.1",
    "@rollup/plugin-typescript": "^11.0.0",
    "@types/jest": "^29.5.0",
    "@types/node": "^17.0.10",
    "@types/react": "^17.0.38",
    "@typescript-eslint/eslint-plugin": "^5.47.1",
    "@typescript-eslint/parser": "^5.47.1",
    "eslint": "^8.30.0",
    "jest": "^29.5.0",
    "rimraf": "^4.1.2",
    "rollup": "^3.20.0",
    "rollup-plugin-terser": "^7.0.2",
    "ts-jest": "^29.0.5",
    "typedoc": "^0.24.0",
    "typedoc-plugin-markdown": "^3.15.0",
    "typescript": "^4.9.4"
  },
  "engines": {
    "node": ">=14.0.0"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/organization/ipfs-accelerate-js.git"
  },
  "bugs": {
    "url": "https://github.com/organization/ipfs-accelerate-js/issues"
  },
  "homepage": "https://github.com/organization/ipfs-accelerate-js#readme"
}
```

### Rollup Configuration

Create a `rollup.config.js` file to build multiple formats:

```javascript
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import { terser } from 'rollup-plugin-terser';
import pkg from './package.json';

// Common plugins for all builds
const plugins = [
  resolve(),
  commonjs(),
  typescript({
    tsconfig: './tsconfig.json',
    declaration: false,
  }),
];

export default [
  // ESM build for modern environments
  {
    input: 'src/index.ts',
    output: {
      file: pkg.module,
      format: 'esm',
      sourcemap: true,
    },
    plugins,
    external: [
      ...Object.keys(pkg.dependencies || {}),
      ...Object.keys(pkg.peerDependencies || {}),
    ],
  },
  
  // CommonJS build for Node.js
  {
    input: 'src/index.ts',
    output: {
      file: pkg.main,
      format: 'cjs',
      sourcemap: true,
      exports: 'named',
    },
    plugins,
    external: [
      ...Object.keys(pkg.dependencies || {}),
      ...Object.keys(pkg.peerDependencies || {}),
    ],
  },
  
  // UMD build for browsers
  {
    input: 'src/index.ts',
    output: {
      file: pkg.browser,
      format: 'umd',
      name: 'IpfsAccelerate',
      sourcemap: true,
      globals: {
        react: 'React',
      },
    },
    plugins: [
      ...plugins,
      terser(), // Minify UMD build
    ],
    external: Object.keys(pkg.peerDependencies || {}),
  },
];
```

### TypeScript Configuration

Update `tsconfig.json` for production:

```json
{
  "compilerOptions": {
    "target": "es2020",
    "module": "esnext",
    "moduleResolution": "node",
    "declaration": true,
    "declarationDir": "./dist/types",
    "sourceMap": true,
    "outDir": "./dist/esm",
    "strict": true,
    "esModuleInterop": true,
    "noImplicitAny": true,
    "noImplicitThis": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "lib": ["dom", "dom.iterable", "esnext", "webworker"],
    "jsx": "react"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

## Documentation Setup

### TypeDoc Configuration

Create `typedoc.json`:

```json
{
  "entryPoints": ["src/index.ts"],
  "out": "docs",
  "name": "IPFS Accelerate JavaScript SDK",
  "excludePrivate": true,
  "excludeInternal": true,
  "disableSources": false,
  "categorizeByGroup": true,
  "categoryOrder": [
    "Core",
    "Hardware",
    "Models",
    "Optimization",
    "Quantization",
    "Browser",
    "*"
  ],
  "readme": "README.md",
  "plugin": ["typedoc-plugin-markdown"]
}
```

### README Structure

Create a comprehensive `README.md`:

```markdown
# IPFS Accelerate JavaScript SDK

A powerful JavaScript SDK for AI model acceleration in web browsers using WebGPU and WebNN.

## Features

- WebGPU and WebNN hardware acceleration
- Cross-browser compatibility
- Advanced quantization techniques
- Model sharding and distributed execution
- React integration components
- Optimized tensor operations

## Installation

```bash
npm install ipfs-accelerate
```

## Quick Start

```javascript
import { createModel, detectHardware } from 'ipfs-accelerate';

// Detect available hardware acceleration
const capabilities = await detectHardware();

// Create an optimized model
const model = await createModel({
  modelName: 'bert-base-uncased',
  preferredBackend: capabilities.recommendedBackend,
  quantization: capabilities.supportsQuantization ? '8bit' : 'none'
});

// Run inference
const result = await model.infer({
  text: "Hello, world!"
});
```

## Documentation

For complete documentation, visit our [Documentation Site](https://example.com/docs).

## Examples

Check out the `/examples` directory for sample code demonstrating various features:

- Basic inference
- WebGPU acceleration
- React integration
- Browser optimizations

## Browser Compatibility

| Browser | WebGPU | WebNN | Recommended For |
|---------|--------|-------|-----------------|
| Chrome 113+ | ✅ | ✅ | General purpose |
| Edge 113+ | ✅ | ✅ | Windows optimization |
| Firefox 113+ | ✅ | ❌ | Audio models |
| Safari 17.0+ | ✅ | ✅ | macOS/iOS |

## License

MIT
```

## Testing Setup

### Jest Configuration

Create `jest.config.js`:

```javascript
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.ts?(x)', '**/?(*.)+(spec|test).ts?(x)'],
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/__tests__/**',
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov'],
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        tsconfig: 'tsconfig.test.json',
      },
    ],
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
};
```

Create `jest.setup.js`:

```javascript
// Mock browser APIs
global.WebGPU = {};
global.navigator = {
  gpu: {}
};

// Mock other browser features as needed
```

## CI/CD Setup

### GitHub Actions

Create `.github/workflows/test.yml`:

```yaml
name: Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 16
          cache: 'npm'
      - run: npm ci
      - run: npm run lint
      - run: npm test
      - run: npm run build
```

Create `.github/workflows/publish.yml`:

```yaml
name: Publish

on:
  release:
    types: [created]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 16
          registry-url: https://registry.npmjs.org/
      - run: npm ci
      - run: npm test
      - run: npm run build
      - run: npm publish
        env:
          NODE_AUTH_TOKEN: ${{secrets.NPM_TOKEN}}
```

## Publishing Process

1. **Prepare the package**:
   ```bash
   npm run prepare
   ```

2. **Check the package contents**:
   ```bash
   npm pack --dry-run
   ```

3. **Verify the build**:
   ```bash
   npm run test
   ```

4. **Publish to npm**:
   ```bash
   npm publish
   ```

## Post-Publishing Tasks

1. Create a GitHub release with detailed changelog
2. Deploy documentation to a dedicated website
3. Update examples to use the published package
4. Create a demo application showcasing key features
5. Collect feedback from early adopters

## Support and Maintenance

1. Set up GitHub issue templates for bug reports and feature requests
2. Establish version update cadence and semantic versioning policies
3. Create contribution guidelines for external developers
4. Develop a roadmap for future SDK features and enhancements