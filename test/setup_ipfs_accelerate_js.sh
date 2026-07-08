#!/bin/bash
# Script to set up the initial directory structure for ipfs_accelerate_js
# and copy the initial implementation files

# Set script to exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}  IPFS Accelerate JavaScript SDK Setup Tool  ${NC}"
echo -e "${BLUE}==============================================${NC}"
echo

# Define base directories
BASE_DIR="$(pwd)"
TARGET_DIR="${BASE_DIR}/ipfs_accelerate_js"

# Check if target directory already exists
if [ -d "$TARGET_DIR" ]; then
    echo -e "${RED}Error: Directory ${TARGET_DIR} already exists.${NC}"
    echo -e "Please remove or rename it before running this script."
    exit 1
fi

echo -e "${GREEN}Creating IPFS Accelerate JavaScript SDK directory structure...${NC}"

# Create the main directory
mkdir -p "$TARGET_DIR"

# Create src directory and subdirectories
mkdir -p "$TARGET_DIR/src/worker/webnn"
mkdir -p "$TARGET_DIR/src/worker/webgpu"
mkdir -p "$TARGET_DIR/src/worker/wasm"
mkdir -p "$TARGET_DIR/src/api_backends"
mkdir -p "$TARGET_DIR/src/hardware/backends"
mkdir -p "$TARGET_DIR/src/utils"
mkdir -p "$TARGET_DIR/src/model"
mkdir -p "$TARGET_DIR/src/optimization/techniques"
mkdir -p "$TARGET_DIR/src/quantization"
mkdir -p "$TARGET_DIR/src/benchmark"
mkdir -p "$TARGET_DIR/src/storage"
mkdir -p "$TARGET_DIR/src/react"
mkdir -p "$TARGET_DIR/src/browser/optimizations"

# Create additional directories
mkdir -p "$TARGET_DIR/dist"
mkdir -p "$TARGET_DIR/examples/browser/basic"
mkdir -p "$TARGET_DIR/examples/browser/advanced"
mkdir -p "$TARGET_DIR/examples/browser/react"
mkdir -p "$TARGET_DIR/examples/node"
mkdir -p "$TARGET_DIR/test/unit"
mkdir -p "$TARGET_DIR/test/integration"
mkdir -p "$TARGET_DIR/test/browser"
mkdir -p "$TARGET_DIR/docs/api"
mkdir -p "$TARGET_DIR/docs/examples"
mkdir -p "$TARGET_DIR/docs/guides"

# Create shaders directory
mkdir -p "$TARGET_DIR/src/worker/webgpu/shaders"
mkdir -p "$TARGET_DIR/src/worker/webgpu/shaders/chrome"
mkdir -p "$TARGET_DIR/src/worker/webgpu/shaders/firefox"
mkdir -p "$TARGET_DIR/src/worker/webgpu/shaders/edge"
mkdir -p "$TARGET_DIR/src/worker/webgpu/shaders/safari"
mkdir -p "$TARGET_DIR/src/worker/webgpu/shaders/model_specific"

echo -e "${GREEN}Directory structure created successfully.${NC}"
echo

# Copy initial implementation files
echo -e "${GREEN}Copying initial implementation files...${NC}"

# Copy package.json and tsconfig.json
cp "${BASE_DIR}/ipfs_accelerate_js_package.json" "${TARGET_DIR}/package.json"
cp "${BASE_DIR}/ipfs_accelerate_js_tsconfig.json" "${TARGET_DIR}/tsconfig.json"

# Copy core implementation files to src directory
mkdir -p "$TARGET_DIR/src/core"
cp "${BASE_DIR}/ipfs_accelerate_js_index.ts" "${TARGET_DIR}/src/index.ts"
cp "${BASE_DIR}/ipfs_accelerate_js_hardware_abstraction.ts" "${TARGET_DIR}/src/hardware/hardware_abstraction.ts"
cp "${BASE_DIR}/ipfs_accelerate_js_webgpu_backend.ts" "${TARGET_DIR}/src/hardware/backends/webgpu_backend.ts"
cp "${BASE_DIR}/ipfs_accelerate_js_webnn_backend.ts" "${TARGET_DIR}/src/hardware/backends/webnn_backend.ts"
cp "${BASE_DIR}/ipfs_accelerate_js_model_loader.ts" "${TARGET_DIR}/src/model/model_loader.ts"
cp "${BASE_DIR}/ipfs_accelerate_js_quantization_engine.ts" "${TARGET_DIR}/src/quantization/quantization_engine.ts"

# Copy React integration
cp "${BASE_DIR}/ipfs_accelerate_js_react_hooks.ts" "${TARGET_DIR}/src/react/hooks.ts"
cp "${BASE_DIR}/ipfs_accelerate_js_react_example.jsx" "${TARGET_DIR}/examples/browser/react/text_embedding_example.jsx"

# Copy WGSL shaders
cp "${BASE_DIR}/ipfs_accelerate_js_wgsl_firefox_4bit.wgsl" "${TARGET_DIR}/src/worker/webgpu/shaders/firefox/matmul_4bit.wgsl"

# Copy documentation files
cp "${BASE_DIR}/ipfs_accelerate_js_README.md" "${TARGET_DIR}/README.md"
cp "${BASE_DIR}/WEBGPU_WEBNN_MIGRATION_PLAN.md" "${TARGET_DIR}/docs/MIGRATION_PLAN.md"
cp "${BASE_DIR}/WEBGPU_WEBNN_MIGRATION_PROGRESS.md" "${TARGET_DIR}/docs/MIGRATION_PROGRESS.md"

# Create minimal gitignore file
cat > "${TARGET_DIR}/.gitignore" << EOF
# Dependencies
node_modules/
.pnp
.pnp.js

# Build and distribution
dist/
build/
coverage/

# IDE and OS files
.DS_Store
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
.vscode/
.idea/
*.swp
*.swo

# Logs
npm-debug.log*
yarn-debug.log*
yarn-error.log*
logs/
*.log

# Temporary files
tmp/
temp/

# Cache
.eslintcache
.cache/
.npm/
EOF

# Create minimal README.md file if one doesn't exist
if [ ! -f "${TARGET_DIR}/README.md" ]; then
    cat > "${TARGET_DIR}/README.md" << EOF
# IPFS Accelerate JavaScript SDK

A comprehensive toolkit for accelerating AI models in web browsers and Node.js environments using WebGPU, WebNN, and IPFS optimization.

## Installation

\`\`\`bash
npm install ipfs-accelerate-js
\`\`\`

## Documentation

See the \`docs/\` directory for full documentation.

## Development

\`\`\`bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test
\`\`\`

## License

MIT
EOF
fi

# Create a rollup.config.js file
cat > "${TARGET_DIR}/rollup.config.js" << EOF
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import { terser } from 'rollup-plugin-terser';
import pkg from './package.json';

export default [
  // Browser-friendly UMD build
  {
    input: 'src/index.ts',
    output: {
      name: 'ipfsAccelerate',
      file: pkg.main,
      format: 'umd',
      sourcemap: true
    },
    plugins: [
      resolve(),
      commonjs(),
      typescript({ tsconfig: './tsconfig.json' }),
      terser()
    ]
  },

  // ESM build for modern browsers
  {
    input: 'src/index.ts',
    output: {
      file: pkg.module,
      format: 'es',
      sourcemap: true
    },
    plugins: [
      resolve(),
      commonjs(),
      typescript({ tsconfig: './tsconfig.json' })
    ]
  }
];
EOF

# Create a simple index.html example
cat > "${TARGET_DIR}/examples/browser/basic/index.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IPFS Accelerate JS Basic Example</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    h1 { color: #333; }
    .container { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
    button { padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
    button:hover { background-color: #45a049; }
    button:disabled { background-color: #cccccc; cursor: not-allowed; }
    pre { background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }
    textarea { width: 100%; height: 100px; padding: 8px; margin-bottom: 10px; }
  </style>
</head>
<body>
  <h1>IPFS Accelerate JavaScript SDK Basic Example</h1>
  
  <div class="container">
    <h2>Hardware Capabilities</h2>
    <button id="check-capabilities">Detect Capabilities</button>
    <pre id="capabilities-output">Capabilities will appear here...</pre>
  </div>
  
  <div class="container">
    <h2>Text Embedding Example</h2>
    <textarea id="text-input" placeholder="Enter text to embed">This is a sample text for embedding demonstration.</textarea>
    <button id="run-embedding">Generate Embedding</button>
    <pre id="embedding-output">Results will appear here...</pre>
  </div>
  
  <script type="module">
    // Import from CDN for example purposes
    // In a real application, you would use your bundled version
    import { createAccelerator, detectCapabilities } from 'https://cdn.jsdelivr.net/npm/ipfs-accelerate-js@0.4.0/dist/ipfs-accelerate.esm.js';
    
    // Detect capabilities button
    document.getElementById('check-capabilities').addEventListener('click', async () => {
      const output = document.getElementById('capabilities-output');
      
      try {
        output.textContent = 'Detecting capabilities...';
        const capabilities = await detectCapabilities();
        output.textContent = JSON.stringify(capabilities, null, 2);
      } catch (error) {
        output.textContent = `Error: ${error.message}`;
      }
    });
    
    // Text embedding button
    document.getElementById('run-embedding').addEventListener('click', async () => {
      const input = document.getElementById('text-input').value;
      const output = document.getElementById('embedding-output');
      
      if (!input) {
        output.textContent = 'Please enter some text.';
        return;
      }
      
      try {
        output.textContent = 'Creating accelerator...';
        const accelerator = await createAccelerator({
          autoDetectHardware: true,
          storeResults: false
        });
        
        output.textContent = 'Running inference...';
        const result = await accelerator.accelerate({
          modelId: 'bert-base-uncased',
          modelType: 'text',
          input: input
        });
        
        output.textContent = JSON.stringify(result, null, 2);
      } catch (error) {
        output.textContent = `Error: ${error.message}`;
      }
    });
  </script>
</body>
</html>
EOF

echo -e "${GREEN}Files copied successfully.${NC}"
echo

# Set file permissions
echo -e "${GREEN}Setting file permissions...${NC}"
chmod +x "$TARGET_DIR/setup_ipfs_accelerate_js.sh" 2>/dev/null || true

echo -e "${GREEN}IPFS Accelerate JavaScript SDK setup completed successfully!${NC}"
echo
echo -e "Directory structure created at: ${BLUE}${TARGET_DIR}${NC}"
echo
echo -e "Next steps:"
echo -e "1. ${BLUE}cd ${TARGET_DIR}${NC}"
echo -e "2. ${BLUE}npm install${NC} to install dependencies"
echo -e "3. ${BLUE}npm run dev${NC} to start development server"
echo
echo -e "${BLUE}==============================================${NC}"