#\!/bin/bash
echo "Creating validation directory..."
mkdir -p typescript_validation
cd typescript_validation

echo "Initializing package.json..."
cat > package.json << EOF
{
  "name": "typescript-validation",
  "version": "1.0.0",
  "description": "Validation for TypeScript files",
  "main": "index.js",
  "scripts": {
    "validate": "tsc --noEmit"
  },
  "dependencies": {
    "typescript": "^4.5.5"
  }
}
EOF

echo "Installing dependencies..."
npm install

echo "Creating tsconfig.json..."
cat > tsconfig.json << EOF
{
  "compilerOptions": {
    "target": "es2020",
    "module": "esnext",
    "moduleResolution": "node",
    "strict": false,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "lib": ["dom", "dom.iterable", "esnext", "webworker"]
  },
  "include": [
    "../ipfs_accelerate_js/src/interfaces.ts",
    "../ipfs_accelerate_js/src/hardware/hardware_abstraction.ts",
    "../ipfs_accelerate_js/src/hardware/backends/webgpu_backend.ts", 
    "../ipfs_accelerate_js/src/hardware/backends/webnn_backend.ts"
  ]
}
EOF

echo "Running TypeScript validation..."
npx tsc --noEmit

echo "Validation complete"

