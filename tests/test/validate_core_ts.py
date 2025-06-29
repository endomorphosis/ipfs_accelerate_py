#!/usr/bin/env python3
# validate_core_ts.py
# Script to validate the core TypeScript files

import os
import sys
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'validate_core_ts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

TARGET_DIR = os.path.abspath("../ipfs_accelerate_js")

def main():
    """Create a tsconfig.json file in the target directory and run tsc"""
    logger.info(f"Validating TypeScript in: {TARGET_DIR}")
    
    # Check if TypeScript is installed
    try:
        subprocess.run(
            ["npx", "tsc", "--version"],
            cwd=TARGET_DIR,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError:
        logger.error("TypeScript is not installed. Please install it first.")
        return
    
    # Create a simple tsconfig.json
    tsconfig_path = os.path.join(TARGET_DIR, "tsconfig.json")
    tsconfig_content = """
{
  "compilerOptions": {
    "target": "es2020",
    "module": "esnext",
    "moduleResolution": "node",
    "declaration": true,
    "sourceMap": true,
    "outDir": "./dist",
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "strict": false,
    "skipLibCheck": true,
    "noImplicitAny": false,
    "noImplicitThis": false,
    "strictNullChecks": false,
    "strictFunctionTypes": false,
    "lib": ["dom", "dom.iterable", "esnext", "webworker"],
    "jsx": "react",
    "noEmit": true
  },
  "include": ["src/**/*.ts"],
  "exclude": ["node_modules", "dist", "**/*.test.ts", "**/*.spec.ts"]
}
"""
    
    # Backup existing tsconfig if it exists
    if os.path.exists(tsconfig_path):
        backup_path = tsconfig_path + ".bak"
        logger.info(f"Backing up existing tsconfig.json to {backup_path}")
        with open(tsconfig_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(existing_content)
    
    # Write the new tsconfig
    with open(tsconfig_path, 'w', encoding='utf-8') as f:
        f.write(tsconfig_content)
    
    logger.info("Created tsconfig.json for validation")
    
    # Run TypeScript compiler
    logger.info("Running TypeScript compiler")
    
    result = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=TARGET_DIR,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Save output to file
    error_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ts_validation_errors.log")
    with open(error_log_path, 'w', encoding='utf-8') as f:
        f.write(result.stdout)
    
    if result.returncode == 0:
        logger.info("TypeScript validation successful!")
        print("\nSuccess! The TypeScript files compile without errors.")
    else:
        error_count = result.stdout.count("error TS")
        logger.warning(f"TypeScript validation found {error_count} errors.")
        print(f"\nFound {error_count} TypeScript errors. Check {error_log_path} for details.")
    
    # Restore original tsconfig if it existed
    if os.path.exists(backup_path):
        logger.info("Restoring original tsconfig.json")
        with open(backup_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        with open(tsconfig_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        os.unlink(backup_path)

if __name__ == "__main__":
    main()