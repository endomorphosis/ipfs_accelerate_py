#\!/bin/bash
echo "Creating validation directory..."
mkdir -p typescript_validation
cd typescript_validation

echo "Copying files for validation..."
mkdir -p src/hardware/backends src/types
cp ../ipfs_accelerate_js/src/interfaces.ts src/
cp ../ipfs_accelerate_js/src/hardware/hardware_abstraction.ts src/hardware/
cp ../ipfs_accelerate_js/src/hardware/backends/webgpu_backend.ts src/hardware/backends/
cp ../ipfs_accelerate_js/src/hardware/backends/webnn_backend.ts src/hardware/backends/
cp ../ipfs_accelerate_js/src/types/webgpu.d.ts src/types/
cp ../ipfs_accelerate_js/src/types/webnn.d.ts src/types/

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
  "include": ["src/**/*.ts"]
}
EOF

echo "Running TypeScript validation..."
npx tsc --noEmit

echo "Validation complete"

