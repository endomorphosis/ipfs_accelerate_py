#!/usr/bin/env python3
# Simple script to run TypeScript compiler and count errors

import os
import subprocess
import re

TARGET_DIR = "../ipfs_accelerate_js"

def main():
    """Run TypeScript compiler and count errors"""
    print(f"Running TypeScript compiler in {os.path.abspath(TARGET_DIR)}")
    
    try:
        result = subprocess.run(
            ["npx", "tsc", "--noEmit"],
            cwd=TARGET_DIR,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Count errors
        error_count = 0
        if result.stdout:
            error_matches = re.findall(r'error TS\d+:', result.stdout)
            error_count = len(error_matches)
            
        if result.returncode == 0:
            print("TypeScript compilation succeeded!")
        else:
            print(f"TypeScript compilation found {error_count} errors")
            
            # Save a summary of errors
            with open("ts_error_summary.txt", "w") as f:
                f.write(f"Total errors: {error_count}\n\n")
                
                # Count error types
                error_types = {}
                for line in result.stdout.splitlines():
                    if "error TS" in line:
                        match = re.search(r'error TS(\d+):', line)
                        if match:
                            error_code = match.group(1)
                            if error_code not in error_types:
                                error_types[error_code] = 0
                            error_types[error_code] += 1
                
                # Write error type summary
                f.write("Error types:\n")
                for code, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"TS{code}: {count} occurrences\n")
            
            print("Error summary saved to ts_error_summary.txt")
            
            # Save full error output
            with open("ts_errors.log", "w") as f:
                f.write(result.stdout)
            
            print("Full error log saved to ts_errors.log")
    except Exception as e:
        print(f"Error running TypeScript compiler: {e}")

if __name__ == "__main__":
    main()