#!/bin/bash
# Comprehensive setup script for ipfs_accelerate_js
# This is an enhanced version of the previous script with improved file discovery,
# better import path handling, and support for various web-related file types

# Set script to exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}  Comprehensive IPFS Accelerate JavaScript SDK Setup Tool  ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo

# Define base directories
BASE_DIR="$(pwd)"
PARENT_DIR="$(dirname "$BASE_DIR")"
TARGET_DIR="${PARENT_DIR}/ipfs_accelerate_js"
LOG_FILE="${PARENT_DIR}/ipfs_accelerate_js_setup_comprehensive.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DRY_RUN=false

# Process command line arguments
FORCE=false
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --dry-run           Show what would be done without making changes"
      echo "  --force             Skip confirmation and update existing files"
      echo "  --target-dir DIR    Set custom target directory"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Initialize log file
echo "Setup started at $(date)" > "$LOG_FILE"
echo "Dry run mode: $DRY_RUN" >> "$LOG_FILE"

# Function to log messages
log_message() {
    echo "$1" >> "$LOG_FILE"
    echo -e "$1"
}

# Check if target directory already exists
if [ -d "$TARGET_DIR" ] && [ "$DRY_RUN" = false ] && [ "$FORCE" = false ]; then
    log_message "${YELLOW}Warning: Directory ${TARGET_DIR} already exists.${NC}"
    read -p "Do you want to continue and update existing files? (y/n): " answer
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        log_message "${RED}Setup aborted by user.${NC}"
        exit 1
    fi
elif [ -d "$TARGET_DIR" ] && [ "$FORCE" = true ]; then
    log_message "${YELLOW}Directory ${TARGET_DIR} exists. Continuing with --force flag...${NC}"
fi

log_message "${GREEN}Setting up IPFS Accelerate JavaScript SDK directory structure...${NC}"

# Function to create directory in dry-run or normal mode
create_directory() {
    local dir="$1"
    if [ "$DRY_RUN" = true ]; then
        log_message "${CYAN}Would create directory: $dir${NC}"
    else
        mkdir -p "$dir"
        log_message "Created directory: $dir"
    fi
}

# Create the main directory structure
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$TARGET_DIR"
else
    log_message "${CYAN}Would create main directory: $TARGET_DIR${NC}"
fi

# Define all directories to create
declare -a directories=(
    # Source code directories
    "$TARGET_DIR/src/worker/webnn"
    "$TARGET_DIR/src/worker/webgpu/shaders/chrome"
    "$TARGET_DIR/src/worker/webgpu/shaders/firefox"
    "$TARGET_DIR/src/worker/webgpu/shaders/edge"
    "$TARGET_DIR/src/worker/webgpu/shaders/safari"
    "$TARGET_DIR/src/worker/webgpu/shaders/model_specific"
    "$TARGET_DIR/src/worker/webgpu/compute"
    "$TARGET_DIR/src/worker/webgpu/pipeline"
    "$TARGET_DIR/src/worker/wasm"
    "$TARGET_DIR/src/api_backends"
    "$TARGET_DIR/src/hardware/backends"
    "$TARGET_DIR/src/hardware/detection"
    "$TARGET_DIR/src/utils"
    "$TARGET_DIR/src/utils/browser"
    "$TARGET_DIR/src/model"
    "$TARGET_DIR/src/model/transformers"
    "$TARGET_DIR/src/model/loaders"
    "$TARGET_DIR/src/optimization/techniques"
    "$TARGET_DIR/src/optimization/memory"
    "$TARGET_DIR/src/quantization"
    "$TARGET_DIR/src/quantization/techniques"
    "$TARGET_DIR/src/benchmark"
    "$TARGET_DIR/src/storage"
    "$TARGET_DIR/src/storage/indexeddb"
    "$TARGET_DIR/src/react"
    "$TARGET_DIR/src/browser/optimizations"
    "$TARGET_DIR/src/tensor"
    "$TARGET_DIR/src/p2p"
    
    # Distribution directory
    "$TARGET_DIR/dist"
    
    # Example directories
    "$TARGET_DIR/examples/browser/basic"
    "$TARGET_DIR/examples/browser/advanced"
    "$TARGET_DIR/examples/browser/react"
    "$TARGET_DIR/examples/browser/streaming"
    "$TARGET_DIR/examples/node"
    
    # Test directories
    "$TARGET_DIR/test/unit"
    "$TARGET_DIR/test/integration"
    "$TARGET_DIR/test/browser"
    "$TARGET_DIR/test/performance"
    
    # Documentation directories
    "$TARGET_DIR/docs/api"
    "$TARGET_DIR/docs/examples"
    "$TARGET_DIR/docs/guides"
    "$TARGET_DIR/docs/architecture"
)

# Create all directories
for dir in "${directories[@]}"; do
    create_directory "$dir"
done

log_message "${GREEN}Directory structure setup complete.${NC}"

# Find and collect WebGPU/WebNN related files with expanded patterns
log_message "${GREEN}Scanning for WebGPU/WebNN and web-related files...${NC}"

# List of key patterns to identify relevant files
declare -a patterns=(
    # WebGPU patterns
    "webgpu"
    "gpu.requestAdapter"
    "GPUDevice"
    "GPUBuffer"
    "GPUCommandEncoder"
    "GPUShaderModule"
    "GPUComputePipeline"
    
    # WebNN patterns
    "webnn"
    "navigator.ml"
    "MLContext"
    "MLGraph"
    "MLGraphBuilder"
    
    # Shader patterns
    "wgsl"
    "shader"
    "computeShader"
    
    # Web-specific patterns
    "navigator.gpu"
    "createTexture"
    "createBuffer"
    "tensor"
    "tensorflow"
    "onnx"
    
    # Worker-related patterns
    "WebWorker"
    "postMessage"
    "MessageEvent"
    "transferControlToOffscreen"
    
    # React integration
    "useEffect"
    "useState"
    "useCallback"
    "React.FC"
    
    # File naming patterns
    "ipfs_accelerate_js_"
    "StreamingWebGPU"
    "WebGPUStreaming"
    "webgpu-utils"
    "webnn-utils"
)

# Define file types to search
declare -a file_types=(
    "ts"
    "js"
    "tsx"
    "jsx"
    "wgsl"
    "html"
    "css"
    "md"
    "json"
)

# Find files matching the patterns recursively
file_list=$(mktemp)

# First search by file extension
log_message "${GREEN}Searching for files by extension...${NC}"
for ext in "${file_types[@]}"; do
    find "$BASE_DIR" "$PARENT_DIR/fixed_web_platform" -type f -name "*.${ext}" 2>/dev/null >> "$file_list" || true
done

# Then filter by content patterns
log_message "${GREEN}Filtering files by content patterns...${NC}"
pattern_list=$(mktemp)
for pattern in "${patterns[@]}"; do
    echo "$pattern" >> "$pattern_list"
done

filtered_list=$(mktemp)
while IFS= read -r file; do
    if grep -f "$pattern_list" "$file" &>/dev/null; then
        echo "$file" >> "$filtered_list"
    elif [[ "$file" == *ipfs_accelerate_js* || "$file" == *WebGPU* || "$file" == *webgpu* || "$file" == *WebNN* || "$file" == *webnn* ]]; then
        # Also include files that match patterns in their names
        echo "$file" >> "$filtered_list"
    fi
done < "$file_list"

# Sort and remove duplicates
sort "$filtered_list" | uniq > "${filtered_list}.uniq"
mv "${filtered_list}.uniq" "$filtered_list"

file_count=$(wc -l < "$filtered_list")
log_message "${GREEN}Found ${file_count} relevant files for potential migration.${NC}"

# Additional search for WebGPU/WebNN related files in fixed_web_platform directory
if [ -d "$PARENT_DIR/fixed_web_platform" ]; then
    log_message "${GREEN}Scanning fixed_web_platform directory for WebGPU/WebNN files...${NC}"
    
    fixed_web_files=$(mktemp)
    find "$PARENT_DIR/fixed_web_platform" -type f \( -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" -o -name "*.wgsl" \) 2>/dev/null >> "$fixed_web_files" || true
    
    fixed_web_count=$(wc -l < "$fixed_web_files")
    log_message "${GREEN}Found ${fixed_web_count} files in fixed_web_platform directory.${NC}"
    
    # Add to main filtered list
    cat "$fixed_web_files" >> "$filtered_list"
    sort "$filtered_list" | uniq > "${filtered_list}.uniq"
    mv "${filtered_list}.uniq" "$filtered_list"
    
    # Updated count
    file_count=$(wc -l < "$filtered_list")
    log_message "${GREEN}Total files for potential migration: ${file_count}${NC}"
fi

# Create a mapping of source files to destination directories
log_message "${GREEN}Creating intelligent file migration mapping...${NC}"

# Function to analyze file content and determine appropriate destination
analyze_file_content() {
    local file="$1"
    local content=""
    
    # Read file content
    if [ -f "$file" ]; then
        content=$(cat "$file")
    else
        return 1
    fi
    
    # Check for specific code patterns
    if grep -q "class.*WebGPUBackend" "$file"; then
        echo "hardware/backends/webgpu_backend"
    elif grep -q "class.*WebNNBackend" "$file"; then
        echo "hardware/backends/webnn_backend"
    elif grep -q "navigator.gpu.requestAdapter" "$file"; then
        echo "hardware/detection/gpu_detection"
    elif grep -q "navigator.ml" "$file"; then
        echo "hardware/detection/ml_detection"
    elif grep -q "class.*HardwareAbstraction" "$file"; then
        echo "hardware/hardware_abstraction"
    elif grep -q "class.*ModelLoader" "$file"; then
        echo "model/model_loader"
    elif grep -q "class.*QuantizationEngine" "$file"; then
        echo "quantization/quantization_engine"
    elif grep -q "React.FC" "$file" || grep -q "useEffect" "$file"; then
        echo "react/components"
    elif grep -q "function.*createShaderModule" "$file"; then
        echo "worker/webgpu/pipeline/shader_modules"
    elif grep -q "function.*setupComputePipeline" "$file"; then
        echo "worker/webgpu/pipeline/compute_pipeline"
    elif grep -q "IndexedDB" "$file" || grep -q "openDatabase" "$file"; then
        echo "storage/indexeddb/storage_manager"
    elif grep -q "TensorFlow" "$file" || grep -q "tf\\." "$file"; then
        echo "model/transformers/tensorflow_adapter"
    elif grep -q "ONNX" "$file" || grep -q "onnx" "$file"; then
        echo "model/transformers/onnx_adapter"
    elif grep -q "@compute" "$file" || grep -q "workgroupSize" "$file"; then
        if grep -q "firefox" "$file"; then
            echo "worker/webgpu/shaders/firefox/compute_shader"
        elif grep -q "chrome" "$file"; then
            echo "worker/webgpu/shaders/chrome/compute_shader"
        elif grep -q "safari" "$file"; then
            echo "worker/webgpu/shaders/safari/compute_shader"
        elif grep -q "edge" "$file"; then
            echo "worker/webgpu/shaders/edge/compute_shader"
        else
            echo "worker/webgpu/shaders/model_specific/compute_shader"
        fi
    else
        # Default case
        return 1
    fi
    
    return 0
}

# Function to determine destination based on file name, extension, and content
map_file_to_destination() {
    local filename="$1"
    local basename=$(basename "$filename")
    local ext="${basename##*.}"
    
    # Check if we can determine destination by content analysis
    local content_dest=$(analyze_file_content "$filename")
    if [ $? -eq 0 ]; then
        echo "$TARGET_DIR/src/$content_dest.${ext}"
        return 0
    fi
    
    # Process based on filename patterns
    if [[ "$basename" == *"webgpu_backend"* ]]; then
        echo "$TARGET_DIR/src/hardware/backends/webgpu_backend.${ext}"
    elif [[ "$basename" == *"webnn_backend"* ]]; then
        echo "$TARGET_DIR/src/hardware/backends/webnn_backend.${ext}"
    elif [[ "$basename" == *"hardware_abstraction"* ]]; then
        echo "$TARGET_DIR/src/hardware/hardware_abstraction.${ext}"
    elif [[ "$basename" == *"model_loader"* ]]; then
        echo "$TARGET_DIR/src/model/model_loader.${ext}"
    elif [[ "$basename" == *"quantization_engine"* ]]; then
        echo "$TARGET_DIR/src/quantization/quantization_engine.${ext}"
    elif [[ "$basename" == *"react_hooks"* ]]; then
        echo "$TARGET_DIR/src/react/hooks.${ext}"
    elif [[ "$basename" == *"StreamingWebGPU"* ]]; then
        echo "$TARGET_DIR/examples/browser/streaming/StreamingWebGPU.${ext}"
    elif [[ "$basename" == *"WebGPUStreaming"* ]]; then
        echo "$TARGET_DIR/examples/browser/streaming/WebGPUStreaming.${ext}"
    elif [[ "$basename" == *"webgpu-utils"* ]]; then
        echo "$TARGET_DIR/src/utils/browser/webgpu-utils.${ext}"
    elif [[ "$basename" == *"webnn-utils"* ]]; then
        echo "$TARGET_DIR/src/utils/browser/webnn-utils.${ext}"
    elif [[ "$basename" == "package.json" ]]; then
        echo "$TARGET_DIR/package.json"
    elif [[ "$basename" == "tsconfig.json" ]]; then
        echo "$TARGET_DIR/tsconfig.json"
    elif [[ "$basename" == *"rollup.config"* ]]; then
        echo "$TARGET_DIR/rollup.config.js"
    elif [[ "$basename" == "README.md" || "$basename" == *"MIGRATION"*".md" ]]; then
        echo "$TARGET_DIR/docs/${basename}"
    elif [[ "$ext" == "wgsl" ]]; then
        # Handle WGSL shaders based on content or name
        if grep -q "firefox" "$filename"; then
            echo "$TARGET_DIR/src/worker/webgpu/shaders/firefox/${basename}"
        elif grep -q "chrome" "$filename"; then
            echo "$TARGET_DIR/src/worker/webgpu/shaders/chrome/${basename}"
        elif grep -q "safari" "$filename"; then
            echo "$TARGET_DIR/src/worker/webgpu/shaders/safari/${basename}"
        elif grep -q "edge" "$filename"; then
            echo "$TARGET_DIR/src/worker/webgpu/shaders/edge/${basename}"
        else
            echo "$TARGET_DIR/src/worker/webgpu/shaders/model_specific/${basename}"
        fi
    else
        # Default case - place in utils directory with cleaned name
        # Remove ipfs_accelerate_js_ prefix if it exists
        clean_name="${basename/ipfs_accelerate_js_/}"
        echo "$TARGET_DIR/src/utils/${clean_name}"
    fi
}

# Function to fix import paths in TypeScript/JavaScript files
fix_import_paths() {
    local content="$1"
    local fixed_content="$content"
    
    # Fix import paths
    fixed_content=$(echo "$fixed_content" | sed -E 's/from ["'\'']\.\/(ipfs_accelerate_js_)?([^"'\'']+)["'\'']/from ".\/\2"/g')
    fixed_content=$(echo "$fixed_content" | sed -E 's/import ["'\'']\.\/(ipfs_accelerate_js_)?([^"'\'']+)["'\'']/import ".\/\2"/g')
    fixed_content=$(echo "$fixed_content" | sed -E 's/require\(["'\'']\.\/(ipfs_accelerate_js_)?([^"'\'']+)["'\'']\)/require(".\/\2")/g')
    
    # Fix relative paths in other contexts
    fixed_content=$(echo "$fixed_content" | sed -E 's/path: ["'\'']\.\/(ipfs_accelerate_js_)?([^"'\'']+)["'\'']/path: ".\/\2"/g')
    fixed_content=$(echo "$fixed_content" | sed -E 's/url: ["'\'']\.\/(ipfs_accelerate_js_)?([^"'\'']+)["'\'']/url: ".\/\2"/g')
    
    echo "$fixed_content"
}

# Function to copy and fix file
copy_and_fix_file() {
    local source="$1"
    local destination="$2"
    
    if [ ! -f "$source" ]; then
        log_message "${YELLOW}Source file not found: $source${NC}"
        return 1
    fi
    
    # Determine file extension
    local ext="${source##*.}"
    
    if [ "$DRY_RUN" = true ]; then
        log_message "${CYAN}Would copy: $source -> $destination${NC}"
        return 0
    fi
    
    # Create destination directory if it doesn't exist
    mkdir -p "$(dirname "$destination")"
    
    # Process file based on extension
    if [[ "$ext" == "ts" || "$ext" == "js" || "$ext" == "tsx" || "$ext" == "jsx" ]]; then
        # Fix import paths in TypeScript/JavaScript files
        local content=$(cat "$source")
        local fixed_content=$(fix_import_paths "$content")
        echo "$fixed_content" > "$destination"
    else
        # Just copy other file types
        cp "$source" "$destination"
    fi
    
    log_message "Copied: $source -> $destination"
    return 0
}

# Define key files to copy with explicit mappings
declare -A key_file_mappings=(
    ["$BASE_DIR/ipfs_accelerate_js_webgpu_backend.ts"]="$TARGET_DIR/src/hardware/backends/webgpu_backend.ts"
    ["$BASE_DIR/ipfs_accelerate_js_webnn_backend.ts"]="$TARGET_DIR/src/hardware/backends/webnn_backend.ts"
    ["$BASE_DIR/ipfs_accelerate_js_hardware_abstraction.ts"]="$TARGET_DIR/src/hardware/hardware_abstraction.ts"
    ["$BASE_DIR/ipfs_accelerate_js_model_loader.ts"]="$TARGET_DIR/src/model/model_loader.ts"
    ["$BASE_DIR/ipfs_accelerate_js_quantization_engine.ts"]="$TARGET_DIR/src/quantization/quantization_engine.ts"
    ["$BASE_DIR/ipfs_accelerate_js_index.ts"]="$TARGET_DIR/src/index.ts"
    ["$BASE_DIR/ipfs_accelerate_js_react_hooks.ts"]="$TARGET_DIR/src/react/hooks.ts"
    ["$BASE_DIR/ipfs_accelerate_js_react_example.jsx"]="$TARGET_DIR/examples/browser/react/text_embedding_example.jsx"
    ["$BASE_DIR/ipfs_accelerate_js_package.json"]="$TARGET_DIR/package.json"
    ["$BASE_DIR/ipfs_accelerate_js_tsconfig.json"]="$TARGET_DIR/tsconfig.json"
    ["$BASE_DIR/ipfs_accelerate_js_rollup.config.js"]="$TARGET_DIR/rollup.config.js"
    ["$BASE_DIR/ipfs_accelerate_js_README.md"]="$TARGET_DIR/README.md"
    ["$BASE_DIR/WEBGPU_WEBNN_MIGRATION_PLAN.md"]="$TARGET_DIR/docs/MIGRATION_PLAN.md"
    ["$BASE_DIR/WEBGPU_WEBNN_MIGRATION_PROGRESS.md"]="$TARGET_DIR/docs/MIGRATION_PROGRESS.md"
    ["$BASE_DIR/StreamingWebGPUDemo.jsx"]="$TARGET_DIR/examples/browser/streaming/StreamingWebGPUDemo.jsx"
    ["$BASE_DIR/WebGPUStreamingExample.jsx"]="$TARGET_DIR/examples/browser/streaming/WebGPUStreamingExample.jsx"
    ["$BASE_DIR/WebGPUStreamingDemo.html"]="$TARGET_DIR/examples/browser/streaming/WebGPUStreamingDemo.html"
    ["$BASE_DIR/web_audio_tests/common/webgpu-utils.js"]="$TARGET_DIR/src/utils/browser/webgpu-utils.js"
    ["$BASE_DIR/web_audio_tests/common/webnn-utils.js"]="$TARGET_DIR/src/utils/browser/webnn-utils.js"
)

# Also check fixed_web_platform directory
if [ -d "$PARENT_DIR/fixed_web_platform" ]; then
    key_file_mappings["$PARENT_DIR/fixed_web_platform/unified_framework/webgpu_interface.ts"]="$TARGET_DIR/src/hardware/backends/webgpu_interface.ts"
    key_file_mappings["$PARENT_DIR/fixed_web_platform/unified_framework/webnn_interface.ts"]="$TARGET_DIR/src/hardware/backends/webnn_interface.ts"
    key_file_mappings["$PARENT_DIR/fixed_web_platform/wgsl_shaders/matmul_shader.wgsl"]="$TARGET_DIR/src/worker/webgpu/shaders/model_specific/matmul_shader.wgsl"
    key_file_mappings["$PARENT_DIR/fixed_web_platform/wgsl_shaders/firefox/"]="$TARGET_DIR/src/worker/webgpu/shaders/firefox/"
    key_file_mappings["$PARENT_DIR/fixed_web_platform/wgsl_shaders/chrome/"]="$TARGET_DIR/src/worker/webgpu/shaders/chrome/"
    key_file_mappings["$PARENT_DIR/fixed_web_platform/wgsl_shaders/safari/"]="$TARGET_DIR/src/worker/webgpu/shaders/safari/"
fi

# Copy key implementation files with explicit mappings
log_message "${GREEN}Copying key implementation files...${NC}"

copy_count=0
error_count=0

for source in "${!key_file_mappings[@]}"; do
    if [ -f "$source" ]; then
        if copy_and_fix_file "$source" "${key_file_mappings[$source]}"; then
            copy_count=$((copy_count + 1))
        else
            error_count=$((error_count + 1))
        fi
    elif [ -d "$source" ]; then
        # Handle directory mappings
        dest_dir="${key_file_mappings[$source]}"
        log_message "${GREEN}Copying directory: $source -> $dest_dir${NC}"
        
        if [ "$DRY_RUN" = true ]; then
            log_message "${CYAN}Would copy directory: $source -> $dest_dir${NC}"
        else
            mkdir -p "$dest_dir"
            find "$source" -type f | while read -r file; do
                rel_path="${file#$source}"
                dest_file="$dest_dir/$rel_path"
                mkdir -p "$(dirname "$dest_file")"
                
                if copy_and_fix_file "$file" "$dest_file"; then
                    copy_count=$((copy_count + 1))
                else
                    error_count=$((error_count + 1))
                fi
            done
        fi
    else
        log_message "${YELLOW}Source does not exist: $source${NC}"
    fi
done

# Process files from the filtered list
log_message "${GREEN}Processing additional WebGPU/WebNN files...${NC}"

additional_count=0
while IFS= read -r file; do
    # Skip already copied files
    already_copied=false
    for source in "${!key_file_mappings[@]}"; do
        if [[ "$file" == "$source" || "$file" == "$source"/* ]]; then
            already_copied=true
            break
        fi
    done
    
    if $already_copied; then
        continue
    fi
    
    # Get file extension
    ext="${file##*.}"
    
    # Only process web-related file types
    if [[ "$ext" == "ts" || "$ext" == "js" || "$ext" == "tsx" || "$ext" == "jsx" || "$ext" == "wgsl" || "$ext" == "html" || "$ext" == "css" ]]; then
        # Determine destination for this file
        destination=$(map_file_to_destination "$file")
        
        if copy_and_fix_file "$file" "$destination"; then
            additional_count=$((additional_count + 1))
        fi
    fi
done < "$filtered_list"

# Create a minimal package.json if it doesn't exist
if [ ! -f "$TARGET_DIR/package.json" ] && [ "$DRY_RUN" = false ]; then
    log_message "${GREEN}Creating package.json...${NC}"
    
    cat > "$TARGET_DIR/package.json" << EOF
{
  "name": "ipfs-accelerate",
  "version": "0.1.0",
  "description": "IPFS Accelerate JavaScript SDK for web browsers and Node.js",
  "main": "dist/ipfs-accelerate.js",
  "module": "dist/ipfs-accelerate.esm.js",
  "types": "dist/types/index.d.ts",
  "scripts": {
    "build": "rollup -c",
    "dev": "rollup -c -w",
    "test": "jest",
    "lint": "eslint 'src/**/*.{js,ts,tsx}'",
    "docs": "typedoc --out docs/api src/"
  },
  "repository": {
    "type": "git",
    "url": "git+https://github.com/your-org/ipfs-accelerate-js.git"
  },
  "keywords": [
    "webgpu",
    "webnn",
    "machine-learning",
    "ai",
    "hardware-acceleration",
    "browser"
  ],
  "author": "",
  "license": "MIT",
  "bugs": {
    "url": "https://github.com/your-org/ipfs-accelerate-js/issues"
  },
  "homepage": "https://github.com/your-org/ipfs-accelerate-js#readme",
  "devDependencies": {
    "@rollup/plugin-commonjs": "^21.0.1",
    "@rollup/plugin-node-resolve": "^13.1.3",
    "@rollup/plugin-typescript": "^8.3.0",
    "@types/jest": "^27.4.0",
    "@types/node": "^17.0.10",
    "@types/react": "^17.0.38",
    "@typescript-eslint/eslint-plugin": "^5.10.0",
    "@typescript-eslint/parser": "^5.10.0",
    "eslint": "^8.7.0",
    "jest": "^27.4.7",
    "rollup": "^2.66.1",
    "rollup-plugin-terser": "^7.0.2",
    "ts-jest": "^27.1.3",
    "tslib": "^2.3.1",
    "typedoc": "^0.22.11",
    "typescript": "^4.5.5"
  },
  "dependencies": {
    "comlink": "^4.3.1"
  },
  "peerDependencies": {
    "react": "^16.8.0 || ^17.0.0 || ^18.0.0"
  },
  "peerDependenciesMeta": {
    "react": {
      "optional": true
    }
  }
}
EOF
elif [ "$DRY_RUN" = true ]; then
    log_message "${CYAN}Would create package.json if it doesn't exist${NC}"
fi

# Create a tsconfig.json if it doesn't exist
if [ ! -f "$TARGET_DIR/tsconfig.json" ] && [ "$DRY_RUN" = false ]; then
    log_message "${GREEN}Creating tsconfig.json...${NC}"
    
    cat > "$TARGET_DIR/tsconfig.json" << EOF
{
  "compilerOptions": {
    "target": "es2020",
    "module": "esnext",
    "moduleResolution": "node",
    "declaration": true,
    "declarationDir": "./dist/types",
    "sourceMap": true,
    "outDir": "./dist",
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
  "exclude": ["node_modules", "dist", "examples", "**/*.test.ts"]
}
EOF
elif [ "$DRY_RUN" = true ]; then
    log_message "${CYAN}Would create tsconfig.json if it doesn't exist${NC}"
fi

# Create a rollup.config.js if it doesn't exist
if [ ! -f "$TARGET_DIR/rollup.config.js" ] && [ "$DRY_RUN" = false ]; then
    log_message "${GREEN}Creating rollup.config.js...${NC}"
    
    cat > "$TARGET_DIR/rollup.config.js" << EOF
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
      sourcemap: true,
      globals: {
        'react': 'React'
      }
    },
    plugins: [
      resolve(),
      commonjs(),
      typescript({ tsconfig: './tsconfig.json' }),
      terser()
    ],
    external: ['react']
  },
  
  // ESM build for modern bundlers
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
    ],
    external: ['react']
  }
];
EOF
elif [ "$DRY_RUN" = true ]; then
    log_message "${CYAN}Would create rollup.config.js if it doesn't exist${NC}"
fi

# Create index.ts files for empty directories
log_message "${GREEN}Creating index files for empty directories...${NC}"

create_index_file() {
    local dir="$1"
    local name=$(basename "$dir")
    local placeholder="${dir}/index.ts"
    
    # Skip if directory already has files
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_message "${CYAN}Would create placeholder: $placeholder${NC}"
        return
    fi
    
    mkdir -p "$dir"
    
    cat > "$placeholder" << EOF
/**
 * ${name} Module
 * 
 * This module provides functionality for ${name}.
 * Implementation pending as part of the WebGPU/WebNN migration.
 * 
 * @module ${name}
 */

/**
 * Configuration options for the ${name} module
 */
export interface ${name^}Options {
  /**
   * Whether to enable debug mode
   */
  debug?: boolean;
  
  /**
   * Custom configuration settings
   */
  config?: Record<string, any>;
}

/**
 * Main implementation class for the ${name} module
 */
export class ${name^}Manager {
  private initialized = false;
  private options: ${name^}Options;
  
  /**
   * Creates a new ${name} manager
   * @param options Configuration options
   */
  constructor(options: ${name^}Options = {}) {
    this.options = {
      debug: false,
      ...options
    };
  }
  
  /**
   * Initializes the ${name} manager
   * @returns Promise that resolves when initialization is complete
   */
  async initialize(): Promise<boolean> {
    // Implementation pending
    this.initialized = true;
    return true;
  }
  
  /**
   * Checks if the manager is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }
}

// Default export
export default ${name^}Manager;
EOF
    
    log_message "Created placeholder: $placeholder"
}

# Create index files in empty source directories
if [ "$DRY_RUN" = false ]; then
    find "$TARGET_DIR/src" -type d -not -path "*/\.*" | while read dir; do
        create_index_file "$dir"
    done
else
    log_message "${CYAN}Would create placeholder files in empty directories${NC}"
fi

# Create a README.md if it doesn't exist
if [ ! -f "$TARGET_DIR/README.md" ] && [ "$DRY_RUN" = false ]; then
    log_message "${GREEN}Creating README.md...${NC}"
    
    cat > "$TARGET_DIR/README.md" << EOF
# IPFS Accelerate JavaScript SDK

> Hardware-accelerated machine learning for web browsers and Node.js

## Features

- **WebGPU Acceleration**: Utilize browser GPU capabilities for fast inference
- **WebNN Support**: Access neural network acceleration on supported browsers
- **Cross-Browser Compatibility**: Works on Chrome, Firefox, Safari, and Edge
- **React Integration**: Simple hooks for React applications
- **Ultra-Low Precision**: Support for 2-bit to 16-bit quantization
- **P2P Content Distribution**: IPFS-based model distribution
- **Cross-Environment**: Works in browsers and Node.js

## Installation

\`\`\`bash
npm install ipfs-accelerate
\`\`\`

## Quick Start

\`\`\`javascript
import { createAccelerator } from 'ipfs-accelerate';

async function runInference() {
  // Create accelerator with automatic hardware detection
  const accelerator = await createAccelerator({
    autoDetectHardware: true
  });
  
  // Run inference
  const result = await accelerator.accelerate({
    modelId: 'bert-base-uncased',
    modelType: 'text',
    input: 'This is a sample text for embedding.'
  });
  
  console.log(result);
}

runInference();
\`\`\`

## React Integration

\`\`\`jsx
import { useAccelerator } from 'ipfs-accelerate/react';

function TextEmbeddingComponent() {
  const { model, loading, error } = useAccelerator({
    modelId: 'bert-base-uncased',
    modelType: 'text'
  });
  
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (model && input) {
      const embedding = await model.embed(input);
      setResult(embedding);
    }
  };
  
  return (
    <div>
      {loading && <p>Loading model...</p>}
      {error && <p>Error: {error.message}</p>}
      {model && (
        <form onSubmit={handleSubmit}>
          <input 
            value={input} 
            onChange={(e) => setInput(e.target.value)} 
            placeholder="Enter text to embed"
          />
          <button type="submit">Generate Embedding</button>
        </form>
      )}
      {result && (
        <pre>{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}
\`\`\`

## Documentation

For complete documentation, see the [docs directory](./docs).

## License

MIT
EOF
elif [ "$DRY_RUN" = true ]; then
    log_message "${CYAN}Would create README.md if it doesn't exist${NC}"
fi

# Create a .gitignore file
if [ ! -f "$TARGET_DIR/.gitignore" ] && [ "$DRY_RUN" = false ]; then
    log_message "${GREEN}Creating .gitignore...${NC}"
    
    cat > "$TARGET_DIR/.gitignore" << EOF
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
elif [ "$DRY_RUN" = true ]; then
    log_message "${CYAN}Would create .gitignore if it doesn't exist${NC}"
fi

# Generate migration verification report
log_message "${GREEN}Generating comprehensive migration report...${NC}"

# Count files in target directory by extension
if [ "$DRY_RUN" = false ] && [ -d "$TARGET_DIR" ]; then
    file_counts=$(find "$TARGET_DIR" -type f | grep -v "node_modules" | awk -F. '{print $NF}' | sort | uniq -c | sort -rn)
    empty_dirs=$(find "$TARGET_DIR" -type d -empty | wc -l)
    
    # Create verification report
    verification_report="$TARGET_DIR/migration_verification_${TIMESTAMP}.json"
    
    if [ "$DRY_RUN" = false ]; then
        cat > "$verification_report" << EOF
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
    "Set up testing infrastructure",
    "Update documentation with SDK usage examples",
    "Create build and release pipeline"
  ]
}
EOF
        log_message "${GREEN}Migration report saved to: ${BLUE}${verification_report}${NC}"
    else
        log_message "${CYAN}Would generate migration verification report${NC}"
    fi
else
    log_message "${CYAN}Would generate migration verification report in actual run${NC}"
fi

# Create migration summary markdown
if [ "$DRY_RUN" = false ]; then
    summary_report="$TARGET_DIR/MIGRATION_SUMMARY_${TIMESTAMP}.md"
    
    cat > "$summary_report" << EOF
# WebGPU/WebNN JavaScript SDK Migration Summary

**Migration Date:** $(date)

## Overview

This document summarizes the results of the comprehensive migration of WebGPU and WebNN implementations from the Python framework to a dedicated JavaScript SDK.

## Migration Statistics

- **Key Files Copied:** $copy_count
- **Additional Files Copied:** $additional_count
- **Total Files Migrated:** $(($copy_count + $additional_count))
- **Errors Encountered:** $error_count
- **Source Files Analyzed:** $file_count

## File Distribution by Type

\`\`\`
$(find "$TARGET_DIR" -type f | grep -v "node_modules" | awk -F. '{print $NF}' | sort | uniq -c | sort -rn)
\`\`\`

## Directory Structure

\`\`\`
$(find "$TARGET_DIR" -type d | sort)
\`\`\`

## Import Path Fixes

The migration script automatically fixed import paths in TypeScript and JavaScript files, replacing patterns like:

- \`from './ipfs_accelerate_js_xxx'\` → \`from './xxx'\`
- \`import './ipfs_accelerate_js_xxx'\` → \`import './xxx'\`
- \`require('./ipfs_accelerate_js_xxx')\` → \`require('./xxx')\`

## Next Steps

1. **Install Dependencies:**
   \`\`\`bash
   cd $TARGET_DIR
   npm install
   \`\`\`

2. **Test Compilation:**
   \`\`\`bash
   npm run build
   \`\`\`

3. **Fix Any Remaining Import Path Issues**

4. **Implement Missing Functionality:**
   - Complete the implementation of placeholder files
   - Prioritize core functionality like hardware detection and model loading

5. **Set Up Testing:**
   \`\`\`bash
   npm test
   \`\`\`

6. **Document API:**
   \`\`\`bash
   npm run docs
   \`\`\`

## Migration Log

For detailed migration logs, see \`$LOG_FILE\`.
EOF
    
    log_message "${GREEN}Migration summary saved to: ${BLUE}${summary_report}${NC}"
else
    log_message "${CYAN}Would create migration summary markdown${NC}"
fi

log_message "${GREEN}IPFS Accelerate JavaScript SDK setup completed successfully!${NC}"
log_message
if [ "$DRY_RUN" = true ]; then
    log_message "${YELLOW}THIS WAS A DRY RUN. No actual changes were made.${NC}"
    log_message "${YELLOW}Run without --dry-run to perform the actual migration.${NC}"
else
    log_message "Directory structure created at: ${BLUE}${TARGET_DIR}${NC}"
    log_message "Migration report: ${BLUE}${verification_report}${NC}"
    log_message "Migration summary: ${BLUE}${summary_report}${NC}"
fi
log_message
log_message "Next steps:"
log_message "1. ${BLUE}cd ${TARGET_DIR}${NC}"
log_message "2. ${BLUE}npm install${NC} to install dependencies"
log_message "3. Fix any remaining import path issues"
log_message "4. Implement functionality in placeholder files"
log_message
log_message "${BLUE}=================================================================${NC}"

# Clean up temporary files
rm -f "$file_list" "$pattern_list" "$filtered_list" "$fixed_web_files" 2>/dev/null || true