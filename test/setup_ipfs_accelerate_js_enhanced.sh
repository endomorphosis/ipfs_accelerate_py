#!/bin/bash
# Enhanced setup script for ipfs_accelerate_js
# Addresses limitations in the original script with better error handling
# and more comprehensive file migration

# Set script to exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}  Enhanced IPFS Accelerate JavaScript SDK Setup Tool  ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo

# Define base directories
BASE_DIR="$(pwd)"
PARENT_DIR="$(dirname "$BASE_DIR")"
TARGET_DIR="${PARENT_DIR}/ipfs_accelerate_js"
LOG_FILE="${PARENT_DIR}/ipfs_accelerate_js_setup.log"

# Initialize log file
echo "Setup started at $(date)" > "$LOG_FILE"

# Function to log messages
log_message() {
    echo "$1" >> "$LOG_FILE"
    echo -e "$1"
}

# Check if target directory already exists
if [ -d "$TARGET_DIR" ]; then
    log_message "${YELLOW}Warning: Directory ${TARGET_DIR} already exists.${NC}"
    read -p "Do you want to continue and update existing files? (y/n): " answer
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        log_message "${RED}Setup aborted by user.${NC}"
        exit 1
    fi
fi

log_message "${GREEN}Creating IPFS Accelerate JavaScript SDK directory structure...${NC}"

# Create the main directory structure
mkdir -p "$TARGET_DIR"

# Create src directory and subdirectories with better structure
mkdir -p "$TARGET_DIR/src/worker/webnn"
mkdir -p "$TARGET_DIR/src/worker/webgpu/shaders/chrome"
mkdir -p "$TARGET_DIR/src/worker/webgpu/shaders/firefox"
mkdir -p "$TARGET_DIR/src/worker/webgpu/shaders/edge"
mkdir -p "$TARGET_DIR/src/worker/webgpu/shaders/safari"
mkdir -p "$TARGET_DIR/src/worker/webgpu/shaders/model_specific"
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

log_message "${GREEN}Directory structure created successfully.${NC}"

# Find and collect WebGPU/WebNN related files
log_message "${GREEN}Scanning for WebGPU/WebNN related files...${NC}"

# List of key patterns to identify WebGPU/WebNN related files
patterns=(
    "webgpu"
    "webnn"
    "wgsl"
    "shader"
    "gpu.requestAdapter"
    "navigator.ml"
    "computeShader"
)

# Find files matching the patterns recursively
file_list=$(mktemp)

for pattern in "${patterns[@]}"; do
    find "$BASE_DIR" -type f -name "*.ts" -o -name "*.js" -o -name "*.tsx" -o -name "*.jsx" -o -name "*.wgsl" | \
    xargs grep -l "$pattern" 2>/dev/null >> "$file_list" || true
done

# Sort and remove duplicates
sort "$file_list" | uniq > "${file_list}.uniq"
mv "${file_list}.uniq" "$file_list"

file_count=$(wc -l < "$file_list")
log_message "${GREEN}Found ${file_count} relevant files for potential migration.${NC}"

# Create a mapping of source files to destination directories
log_message "${GREEN}Creating file migration mapping...${NC}"

# Define file type mappings
declare -A file_mappings

# Function to determine destination based on file content and name
map_file_to_destination() {
    local filename="$1"
    local basename=$(basename "$filename")
    
    # Check filename patterns
    if [[ "$basename" == *"webgpu_backend"* ]]; then
        echo "$TARGET_DIR/src/hardware/backends/webgpu_backend.ts"
    elif [[ "$basename" == *"webnn_backend"* ]]; then
        echo "$TARGET_DIR/src/hardware/backends/webnn_backend.ts"
    elif [[ "$basename" == *"hardware_abstraction"* ]]; then
        echo "$TARGET_DIR/src/hardware/hardware_abstraction.ts"
    elif [[ "$basename" == *"model_loader"* ]]; then
        echo "$TARGET_DIR/src/model/model_loader.ts"
    elif [[ "$basename" == *"quantization_engine"* ]]; then
        echo "$TARGET_DIR/src/quantization/quantization_engine.ts"
    elif [[ "$basename" == *"react_hooks"* ]]; then
        echo "$TARGET_DIR/src/react/hooks.ts"
    elif [[ "$basename" == *"index"* ]]; then
        echo "$TARGET_DIR/src/index.ts"
    elif [[ "$basename" == *"firefox"* && "$basename" == *"wgsl"* ]]; then
        echo "$TARGET_DIR/src/worker/webgpu/shaders/firefox/matmul_4bit.wgsl"
    elif [[ "$basename" == *"chrome"* && "$basename" == *"wgsl"* ]]; then
        echo "$TARGET_DIR/src/worker/webgpu/shaders/chrome/matmul_4bit.wgsl"
    elif [[ "$basename" == *"edge"* && "$basename" == *"wgsl"* ]]; then
        echo "$TARGET_DIR/src/worker/webgpu/shaders/edge/matmul_4bit.wgsl"
    elif [[ "$basename" == *"safari"* && "$basename" == *"wgsl"* ]]; then
        echo "$TARGET_DIR/src/worker/webgpu/shaders/safari/matmul_4bit.wgsl"
    elif [[ "$basename" == *"react_example"* ]]; then
        echo "$TARGET_DIR/examples/browser/react/text_embedding_example.jsx"
    else
        # Default for other files
        echo "$TARGET_DIR/src/utils/$(basename "$filename")"
    fi
}

# Copy the known key implementation files with path correction
log_message "${GREEN}Copying key implementation files...${NC}"

copy_and_fix_file() {
    local source="$1"
    local destination="$2"
    
    if [ -f "$source" ]; then
        # Create destination directory if it doesn't exist
        mkdir -p "$(dirname "$destination")"
        
        # Copy file with import path fixes
        sed 's/from .\/ipfs_accelerate_js_/from .\//g' "$source" > "$destination"
        
        log_message "Copied: $source -> $destination"
        return 0
    else
        log_message "${YELLOW}Source file not found: $source${NC}"
        return 1
    fi
}

# Copy key files
file_mappings=(
    ["$BASE_DIR/ipfs_accelerate_js_webgpu_backend.ts"]="$TARGET_DIR/src/hardware/backends/webgpu_backend.ts"
    ["$BASE_DIR/ipfs_accelerate_js_webnn_backend.ts"]="$TARGET_DIR/src/hardware/backends/webnn_backend.ts"
    ["$BASE_DIR/ipfs_accelerate_js_hardware_abstraction.ts"]="$TARGET_DIR/src/hardware/hardware_abstraction.ts"
    ["$BASE_DIR/ipfs_accelerate_js_model_loader.ts"]="$TARGET_DIR/src/model/model_loader.ts"
    ["$BASE_DIR/ipfs_accelerate_js_quantization_engine.ts"]="$TARGET_DIR/src/quantization/quantization_engine.ts"
    ["$BASE_DIR/ipfs_accelerate_js_index.ts"]="$TARGET_DIR/src/index.ts"
    ["$BASE_DIR/ipfs_accelerate_js_react_hooks.ts"]="$TARGET_DIR/src/react/hooks.ts"
    ["$BASE_DIR/ipfs_accelerate_js_react_example.jsx"]="$TARGET_DIR/examples/browser/react/text_embedding_example.jsx"
    ["$BASE_DIR/ipfs_accelerate_js_wgsl_firefox_4bit.wgsl"]="$TARGET_DIR/src/worker/webgpu/shaders/firefox/matmul_4bit.wgsl"
    ["$BASE_DIR/ipfs_accelerate_js_package.json"]="$TARGET_DIR/package.json"
    ["$BASE_DIR/ipfs_accelerate_js_tsconfig.json"]="$TARGET_DIR/tsconfig.json"
    ["$BASE_DIR/ipfs_accelerate_js_rollup.config.js"]="$TARGET_DIR/rollup.config.js"
    ["$BASE_DIR/ipfs_accelerate_js_README.md"]="$TARGET_DIR/README.md"
    ["$BASE_DIR/WEBGPU_WEBNN_MIGRATION_PLAN.md"]="$TARGET_DIR/docs/MIGRATION_PLAN.md"
    ["$BASE_DIR/WEBGPU_WEBNN_MIGRATION_PROGRESS_UPDATED.md"]="$TARGET_DIR/docs/MIGRATION_PROGRESS.md"
)

copy_count=0
error_count=0

for source in "${!file_mappings[@]}"; do
    if copy_and_fix_file "$source" "${file_mappings[$source]}"; then
        copy_count=$((copy_count + 1))
    else
        error_count=$((error_count + 1))
    fi
done

# Scan additional files from the file list and copy where appropriate
log_message "${GREEN}Scanning for additional WebGPU/WebNN files to copy...${NC}"

additional_count=0
while IFS= read -r file; do
    # Skip already copied files
    already_copied=false
    for source in "${!file_mappings[@]}"; do
        if [[ "$file" == "$source" ]]; then
            already_copied=true
            break
        fi
    done
    
    if $already_copied; then
        continue
    fi
    
    # Determine destination for this file
    destination=$(map_file_to_destination "$file")
    
    # Only copy TypeScript/JavaScript/WGSL files
    if [[ "$file" =~ \.(ts|js|tsx|jsx|wgsl)$ ]]; then
        if copy_and_fix_file "$file" "$destination"; then
            additional_count=$((additional_count + 1))
        fi
    fi
done < "$file_list"

# Create a minimal .gitignore file
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

# Create the HTML example file if it wasn't copied from an existing file
if [ ! -f "$TARGET_DIR/examples/browser/basic/index.html" ]; then
    cat > "$TARGET_DIR/examples/browser/basic/index.html" << EOF
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
    // Import from local path for example purposes
    import { createAccelerator, detectCapabilities } from '../../dist/ipfs-accelerate.esm.js';
    
    // Detect capabilities button
    document.getElementById('check-capabilities').addEventListener('click', async () => {
      const output = document.getElementById('capabilities-output');
      
      try {
        output.textContent = 'Detecting capabilities...';
        const capabilities = await detectCapabilities();
        output.textContent = JSON.stringify(capabilities, null, 2);
      } catch (error) {
        output.textContent = `Error: \${error.message}`;
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
        output.textContent = `Error: \${error.message}`;
      }
    });
  </script>
</body>
</html>
EOF
fi

# Generate additional placeholder files in empty directories
log_message "${GREEN}Creating placeholder files in empty directories...${NC}"

create_placeholder() {
    local dir="$1"
    local name=$(basename "$dir")
    local placeholder="${dir}/index.ts"
    
    # Skip if directory already has files
    if [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        return
    fi
    
    mkdir -p "$dir"
    
    cat > "$placeholder" << EOF
/**
 * ${name} Module
 * 
 * This is a placeholder file for the ${name} module.
 * Implementation pending as part of the WebGPU/WebNN migration.
 * 
 * TODO: Implement ${name} functionality
 */

// Export placeholder interface
export interface ${name^}Options {
  // Configuration options will go here
}

// Export placeholder class
export class ${name^}Manager {
  private initialized = false;
  
  constructor(options?: ${name^}Options) {
    // Implementation pending
  }
  
  async initialize(): Promise<boolean> {
    this.initialized = true;
    return true;
  }
  
  // Additional methods will be implemented
}

// Default export
export default ${name^}Manager;
EOF

    log_message "Created placeholder: $placeholder"
}

# Create placeholders in empty directories
find "$TARGET_DIR/src" -type d -not -path "*/\.*" | while read dir; do
    create_placeholder "$dir"
done

# Set file permissions
log_message "${GREEN}Setting file permissions...${NC}"
chmod +x "$TARGET_DIR/setup_ipfs_accelerate_js_enhanced.sh" 2>/dev/null || true

# Final verification
log_message "${GREEN}Generating migration verification report...${NC}"

# Count files in target directory by extension
file_counts=$(find "$TARGET_DIR" -type f | grep -v "node_modules" | awk -F. '{print $NF}' | sort | uniq -c | sort -rn)
empty_dirs=$(find "$TARGET_DIR" -type d -empty | wc -l)

# Create verification report
cat > "$TARGET_DIR/migration_verification.json" << EOF
{
  "timestamp": "$(date +%s)",
  "date": "$(date)",
  "statistics": {
    "copied_key_files": $copy_count,
    "copied_additional_files": $additional_count,
    "errors": $error_count,
    "empty_directories_before_placeholders": $empty_dirs
  },
  "file_counts_by_extension": {
    $(echo "$file_counts" | awk '{print "\"" $2 "\": " $1 ","}' | sed '$s/,$//')
  },
  "source_files_analyzed": $file_count,
  "next_steps": [
    "Install dependencies with 'npm install'",
    "Fix any remaining import path issues",
    "Implement missing functionality in placeholder files",
    "Set up testing infrastructure"
  ]
}
EOF

log_message "${GREEN}IPFS Accelerate JavaScript SDK setup completed successfully!${NC}"
log_message
log_message "Directory structure created at: ${BLUE}${TARGET_DIR}${NC}"
log_message "Migration report: ${BLUE}${TARGET_DIR}/migration_verification.json${NC}"
log_message
log_message "Next steps:"
log_message "1. ${BLUE}cd ${TARGET_DIR}${NC}"
log_message "2. ${BLUE}npm install${NC} to install dependencies"
log_message "3. Fix any remaining import path issues"
log_message "4. Implement functionality in placeholder files"
log_message
log_message "${BLUE}=================================================================${NC}"

# Clean up temporary file
rm -f "$file_list"