# Enhanced Migration Script Plan for ipfs_accelerate_js

## Overview

This document outlines a comprehensive plan for enhancing the migration scripts to properly move all WebGPU/WebNN implementations from the Python framework to the dedicated `ipfs_accelerate_js` directory at the root level. The current migration script (`setup_ipfs_accelerate_js.sh`) has several limitations that need to be addressed to ensure a complete and robust migration.

## Current Limitations

1. **Limited File Scope**: The script only copies specific files listed explicitly, missing many related components.
2. **Empty Directories**: Many directories are created but not populated with necessary files.
3. **Path Management**: Import paths need to be systematically updated across all files.
4. **Dependency Management**: The script doesn't account for dependencies between components.
5. **Verification**: No verification of the migration's completeness.
6. **Error Handling**: Limited error checking and recovery capabilities.

## Enhanced Migration Approach

### 1. Comprehensive File Scanning

Create a Python script that will:

- Scan the entire codebase for JavaScript/TypeScript/WGSL files
- Identify WebGPU/WebNN related files through pattern matching
- Create a dependency graph between components
- Categorize files into appropriate component types

```python
# scan_webgpu_webnn_files.py
import os
import re
import json
from pathlib import Path

def is_webgpu_webnn_file(content):
    """Check if file is related to WebGPU/WebNN implementation."""
    patterns = [
        r'webgpu', r'webnn', r'wgsl', r'shaders?', 
        r'gpu\.requestAdapter', r'navigator\.ml', r'compute[sS]hader'
    ]
    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False

def scan_codebase(root_dir):
    """Scan codebase for WebGPU/WebNN related files."""
    webgpu_webnn_files = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.js', '.ts', '.jsx', '.tsx', '.wgsl')):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    try:
                        content = f.read()
                        if is_webgpu_webnn_file(content):
                            webgpu_webnn_files.append({
                                'path': file_path,
                                'filename': file,
                                'content': content
                            })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    return webgpu_webnn_files

# Main execution
if __name__ == "__main__":
    root_directory = "/home/barberb/ipfs_accelerate_py"
    files = scan_codebase(root_directory)
    
    # Output results to JSON for further processing
    with open('webgpu_webnn_files.json', 'w') as f:
        json.dump(files, f, indent=2)
    
    print(f"Found {len(files)} WebGPU/WebNN related files.")
```

### 2. Dependency Analysis

Create a script to analyze and build a dependency graph:

```python
# analyze_dependencies.py
import json
import re
import networkx as nx
import matplotlib.pyplot as plt

def extract_imports(content):
    """Extract import statements from file content."""
    import_patterns = [
        r'import\s+.+\s+from\s+[\'"](.+)[\'"]',
        r'require\s*\(\s*[\'"](.+)[\'"]\s*\)',
        r'@import\s+[\'"](.+)[\'"]'
    ]
    
    imports = []
    for pattern in import_patterns:
        imports.extend(re.findall(pattern, content))
    
    return imports

def build_dependency_graph(files):
    """Build a dependency graph between files."""
    graph = nx.DiGraph()
    
    file_map = {f['path']: f for f in files}
    
    # Add all files as nodes
    for file in files:
        graph.add_node(file['path'], filename=file['filename'])
    
    # Add edges for dependencies
    for file in files:
        imports = extract_imports(file['content'])
        for imp in imports:
            # Resolve import to actual file path
            for potential_dep in file_map:
                if potential_dep.endswith(imp) or imp in potential_dep:
                    graph.add_edge(file['path'], potential_dep)
    
    return graph

# Main execution
if __name__ == "__main__":
    with open('webgpu_webnn_files.json', 'r') as f:
        files = json.load(f)
    
    graph = build_dependency_graph(files)
    
    # Save graph as JSON
    graph_data = nx.node_link_data(graph)
    with open('dependency_graph.json', 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    # Visualize graph
    nx.draw(graph, with_labels=True)
    plt.savefig("dependency_graph.png")
    
    print(f"Built dependency graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
```

### 3. Categorization and Destination Mapping

Create a script to categorize files and map them to their destination in the new structure:

```python
# categorize_files.py
import json
import re
import os

def categorize_file(content, filename):
    """Categorize file based on content and filename."""
    categories = {
        'webgpu_backend': re.compile(r'webgpu.*backend|backend.*webgpu', re.IGNORECASE),
        'webnn_backend': re.compile(r'webnn.*backend|backend.*webnn', re.IGNORECASE),
        'hardware_abstraction': re.compile(r'hardware.*abstraction|abstraction.*hardware', re.IGNORECASE),
        'model_loader': re.compile(r'model.*loader|loader.*model', re.IGNORECASE),
        'quantization': re.compile(r'quantization|quantize', re.IGNORECASE),
        'react_hooks': re.compile(r'react.*hooks|hooks.*react', re.IGNORECASE),
        'shader': re.compile(r'shader|wgsl', re.IGNORECASE),
        'storage': re.compile(r'storage|cache|persist', re.IGNORECASE),
        'utils': re.compile(r'util', re.IGNORECASE),
        'test': re.compile(r'test|spec', re.IGNORECASE),
    }
    
    for category, pattern in categories.items():
        if pattern.search(content) or pattern.search(filename):
            return category
    
    return 'other'

def map_to_destination(category, filename):
    """Map file to destination in ipfs_accelerate_js structure."""
    base_dir = "/home/barberb/ipfs_accelerate_py/ipfs_accelerate_js"
    
    category_mapping = {
        'webgpu_backend': f"{base_dir}/src/hardware/backends",
        'webnn_backend': f"{base_dir}/src/hardware/backends",
        'hardware_abstraction': f"{base_dir}/src/hardware",
        'model_loader': f"{base_dir}/src/model",
        'quantization': f"{base_dir}/src/quantization",
        'react_hooks': f"{base_dir}/src/react",
        'shader': f"{base_dir}/src/worker/webgpu/shaders",
        'storage': f"{base_dir}/src/storage",
        'utils': f"{base_dir}/src/utils",
        'test': f"{base_dir}/test",
        'other': f"{base_dir}/src/utils"
    }
    
    # Special handling for shaders based on browser-specific patterns
    if category == 'shader':
        if 'firefox' in filename.lower():
            return f"{base_dir}/src/worker/webgpu/shaders/firefox"
        elif 'chrome' in filename.lower():
            return f"{base_dir}/src/worker/webgpu/shaders/chrome"
        elif 'edge' in filename.lower():
            return f"{base_dir}/src/worker/webgpu/shaders/edge"
        elif 'safari' in filename.lower():
            return f"{base_dir}/src/worker/webgpu/shaders/safari"
        else:
            return f"{base_dir}/src/worker/webgpu/shaders"
    
    return category_mapping.get(category, f"{base_dir}/src")

# Main execution
if __name__ == "__main__":
    with open('webgpu_webnn_files.json', 'r') as f:
        files = json.load(f)
    
    for file in files:
        category = categorize_file(file['content'], file['filename'])
        destination = map_to_destination(category, file['filename'])
        
        file['category'] = category
        file['destination'] = destination
    
    # Output categorized files
    with open('categorized_files.json', 'w') as f:
        json.dump(files, f, indent=2)
    
    # Generate summary
    categories = {}
    for file in files:
        cat = file['category']
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    print("File categorization summary:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} files")
```

### 4. Enhanced Migration Script

Create a comprehensive migration script that handles:
- File copying with appropriate transformations
- Import path updates
- Destination directory creation
- Error handling and reporting

```bash
#!/bin/bash
# enhanced_migration.sh

# Set script to exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}  Enhanced IPFS Accelerate JavaScript SDK Migration Tool  ${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo

# Define base directories
BASE_DIR="/home/barberb/ipfs_accelerate_py"
TARGET_DIR="${BASE_DIR}/ipfs_accelerate_js"
LOG_FILE="${BASE_DIR}/migration_log.txt"

# Clean log file
echo "Migration started at $(date)" > "$LOG_FILE"

# Function to log messages
log_message() {
    echo "$1" >> "$LOG_FILE"
    echo -e "$1"
}

# Check if Python is available for running dependency analysis
if ! command -v python3 &> /dev/null; then
    log_message "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    log_message "${YELLOW}Warning: Target directory does not exist. Creating it now.${NC}"
    mkdir -p "$TARGET_DIR"
fi

# Run Python scripts for file scanning and analysis
log_message "${GREEN}Scanning codebase for WebGPU/WebNN files...${NC}"
python3 scan_webgpu_webnn_files.py

log_message "${GREEN}Analyzing dependencies...${NC}"
python3 analyze_dependencies.py

log_message "${GREEN}Categorizing files...${NC}"
python3 categorize_files.py

# Process each file from the categorized JSON
log_message "${GREEN}Starting file migration...${NC}"
python3 - <<EOF
import json
import os
import re
import shutil

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def update_imports(content, old_base, new_base):
    # Update relative imports
    updated = re.sub(
        r'from\s+[\'"](.+?)[\'"]', 
        lambda m: f'from "{m.group(1).replace(old_base, new_base)}"', 
        content
    )
    
    # Update require statements
    updated = re.sub(
        r'require\s*\(\s*[\'"](.+?)[\'"]\s*\)', 
        lambda m: f'require("{m.group(1).replace(old_base, new_base)}")', 
        updated
    )
    
    return updated

# Load categorized files
with open('categorized_files.json', 'r') as f:
    files = json.load(f)

success_count = 0
error_count = 0

for file in files:
    try:
        source_path = file['path']
        filename = os.path.basename(source_path)
        destination_dir = file['destination']
        
        # Ensure destination directory exists
        ensure_dir(destination_dir)
        
        # Determine new file path
        destination_path = os.path.join(destination_dir, filename)
        
        # Update content with corrected imports
        updated_content = update_imports(
            file['content'],
            'ipfs_accelerate_js_',
            ''
        )
        
        # Write updated content to destination
        with open(destination_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"Migrated: {source_path} -> {destination_path}")
        success_count += 1
        
    except Exception as e:
        print(f"Error migrating {file['path']}: {e}")
        error_count += 1

print(f"\nMigration complete. Successfully migrated {success_count} files. Errors: {error_count}")
EOF

# Create a verification report
log_message "${GREEN}Generating migration verification report...${NC}"
python3 - <<EOF
import os
import json

base_dir = "/home/barberb/ipfs_accelerate_py/ipfs_accelerate_js"

def count_files_by_type(directory):
    file_types = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext not in file_types:
                file_types[ext] = 0
            file_types[ext] += 1
    
    return file_types

def scan_empty_dirs(directory):
    empty_dirs = []
    
    for root, dirs, files in os.walk(directory):
        if not dirs and not files:
            empty_dirs.append(root)
    
    return empty_dirs

# Count files by type
file_types = count_files_by_type(base_dir)
print("Files by type:")
for ext, count in file_types.items():
    print(f"  {ext}: {count}")

# Find empty directories
empty_dirs = scan_empty_dirs(base_dir)
print(f"\nEmpty directories ({len(empty_dirs)}):")
for d in empty_dirs:
    print(f"  {d}")

# Save verification report
report = {
    "file_types": file_types,
    "empty_dirs": empty_dirs
}

with open('migration_verification.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\nVerification report saved to migration_verification.json")
EOF

log_message "${GREEN}Migration completed successfully!${NC}"
log_message "${BLUE}See migration_verification.json for detailed report.${NC}"
echo
echo -e "${BLUE}=========================================================${NC}"
```

### 5. Verification and Testing

Create a script to test the migrated files for proper functionality:

```python
# verify_migration.py
import os
import json
import subprocess
import time

def run_typescript_compilation_test(dir_path):
    """Test TypeScript compilation."""
    try:
        result = subprocess.run(
            ['npx', 'tsc', '--noEmit'],
            cwd=dir_path,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            return True, "TypeScript compilation successful"
        else:
            return False, f"TypeScript compilation failed: {result.stderr}"
    except Exception as e:
        return False, f"Failed to run TypeScript compilation: {e}"

def verify_import_paths(dir_path):
    """Verify import paths are correctly updated."""
    errors = []
    
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(('.ts', '.tsx', '.js', '.jsx')):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Check for old style imports
                    if 'ipfs_accelerate_js_' in content:
                        errors.append(f"{file_path}: Contains old style imports")
    
    return len(errors) == 0, errors

def check_file_existence(base_dir):
    """Check that key files exist in the appropriate locations."""
    key_files = [
        "src/hardware/backends/webgpu_backend.ts",
        "src/hardware/backends/webnn_backend.ts",
        "src/hardware/hardware_abstraction.ts",
        "src/model/model_loader.ts",
        "src/index.ts",
        "src/react/hooks.ts",
        "src/worker/webgpu/shaders/firefox/matmul_4bit.wgsl"
    ]
    
    missing = []
    for file in key_files:
        full_path = os.path.join(base_dir, file)
        if not os.path.exists(full_path):
            missing.append(file)
    
    return len(missing) == 0, missing

# Main execution
if __name__ == "__main__":
    base_dir = "/home/barberb/ipfs_accelerate_py/ipfs_accelerate_js"
    
    print("Running migration verification...")
    
    # Check file existence
    files_exist, missing_files = check_file_existence(base_dir)
    if files_exist:
        print("✅ All key files exist in the appropriate locations")
    else:
        print("❌ Some key files are missing:")
        for file in missing_files:
            print(f"  - {file}")
    
    # Verify import paths
    paths_correct, path_errors = verify_import_paths(base_dir)
    if paths_correct:
        print("✅ All import paths are correctly updated")
    else:
        print("❌ Some files have incorrect import paths:")
        for error in path_errors:
            print(f"  - {error}")
    
    # Run TypeScript compilation test
    ts_compile_success, ts_compile_msg = run_typescript_compilation_test(base_dir)
    if ts_compile_success:
        print(f"✅ {ts_compile_msg}")
    else:
        print(f"❌ {ts_compile_msg}")
    
    # Generate overall report
    verification_result = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {
            "file_existence": {
                "success": files_exist,
                "missing_files": missing_files
            },
            "import_paths": {
                "success": paths_correct,
                "errors": path_errors
            },
            "typescript_compilation": {
                "success": ts_compile_success,
                "message": ts_compile_msg
            }
        },
        "overall_success": files_exist and paths_correct and ts_compile_success
    }
    
    with open('migration_test_results.json', 'w') as f:
        json.dump(verification_result, f, indent=2)
    
    print(f"\nVerification {'passed' if verification_result['overall_success'] else 'failed'}.")
    print("Detailed results saved to migration_test_results.json")
```

## Implementation Plan

1. **Phase 1: Analysis and Preparation (1-2 days)**
   - Implement and run the file scanning script
   - Perform dependency analysis
   - Categorize files and map destinations

2. **Phase 2: Enhanced Migration (2-3 days)**
   - Execute enhanced migration script
   - Fix any import path issues
   - Create placeholder implementations for missing components

3. **Phase 3: Verification and Testing (1-2 days)**
   - Run verification script
   - Test TypeScript compilation
   - Address any issues found in verification

4. **Phase 4: Documentation Update (1 day)**
   - Update migration progress documentation
   - Document the new directory structure
   - Create README files for key components

## Conclusion

This enhanced migration approach will address the current limitations of the migration process by:

1. **Comprehensive Scanning**: Finding all WebGPU/WebNN related files across the codebase
2. **Dependency Analysis**: Understanding relationships between components
3. **Intelligent Categorization**: Placing files in appropriate directories
4. **Path Correction**: Systematically updating import paths
5. **Thorough Verification**: Ensuring all components are properly migrated
6. **Documentation**: Keeping documentation updated with migration progress

The result will be a complete and well-structured JavaScript SDK for WebGPU and WebNN that follows the architecture principles outlined in the migration plan.