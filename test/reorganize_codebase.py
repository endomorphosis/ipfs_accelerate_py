#!/usr/bin/env python3
"""
Master script for reorganizing the codebase according to the plan in CLAUDE.md.

This script guides the user through the entire process of reorganizing the codebase:
    1. Creating the new directory structure
    2. Fixing syntax errors in the files
    3. Moving files to their new locations
    4. Updating import statements
    5. Running tests to verify everything works

Usage:
    python reorganize_codebase.py [],--step N] [],--dry-run],
    """

    import os
    import sys
    import argparse
    import subprocess
    from pathlib import Path
    import time
    import importlib

# Steps in the reorganization process
    STEPS = [],
    {}}}}}}}
    "name": "Create Package Structure",
    "description": "Create the generators/ and duckdb_api/ directories with their subdirectories",
    "script": "create_package_structure.py",
    },
    {}}}}}}}
    "name": "Fix Syntax Errors",
    "description": "Fix syntax errors in generator files and other Python files",
    "script": "fix_syntax_errors.py",
    },
    {}}}}}}}
    "name": "Move Generator Files",
    "description": "Move generator files to their correct locations in the generators/ directory",
    "script": "move_files_to_packages.py --type generator",
    },
    {}}}}}}}
    "name": "Move Database Files",
    "description": "Move database files to their correct locations in the duckdb_api/ directory",
    "script": "move_files_to_packages.py --type database",
    },
    {}}}}}}}
    "name": "Update Import Statements",
    "description": "Update import statements in all files to use the new package structure",
    "script": "update_imports.py",
    },
    {}}}}}}}
    "name": "Update Documentation Paths",
    "description": "Update file paths in documentation files to reflect the new structure",
    "script": "update_doc_paths.py",
    },
    {}}}}}}}
    "name": "Run Tests",
    "description": "Run tests to verify everything works correctly",
    "script": "python -m pytest -xvs",  # Assumes pytest is available
    },
    ]

def run_script(script, dry_run=False):
    """Run a Python script with the appropriate command."""
    if script.startswith("python "):
        cmd = script
    else:
        cmd = f"python {}}}}}}}script}"
        
    if dry_run and not cmd.endswith("--dry-run"):
        cmd += " --dry-run"
    
        print(f"\n🚀 Running: {}}}}}}}cmd}")
        print("-" * 80)
    
        result = subprocess.run(cmd, shell=True)
    
        print("-" * 80)
    if result.returncode == 0:
        print(f"✅ Command completed successfully: {}}}}}}}cmd}")
    else:
        print(f"❌ Command failed with code {}}}}}}}result.returncode}: {}}}}}}}cmd}")
    
        return result.returncode

def check_script_exists(script):
    """Check if a script exists.""":
    if script.startswith("python "):
        # Extract the script name from the command
        parts = script.split()
        if len(parts) >= 2 and parts[],0] == "python":
            script = parts[],1]
            if script.startswith("-m"):
                # It's a module, not a file
            return True
    
        return os.path.exists(script)

def get_step_input(current_step, total_steps):
    """Get user input for the current step."""
    while True:
        print("\nOptions:")
        print(f"  [],Enter] - Run step {}}}}}}}current_step}")
        print("  s - Skip this step")
        if current_step > 1:
            print("  p - Go back to previous step")
            print("  q - Quit")
            print("  d - Toggle dry run mode")
            choice = input("\nChoice: ").strip().lower()
        
        if choice == "":
            return "run"
        elif choice == "s":
            return "skip"
        elif choice == "p" and current_step > 1:
            return "previous"
        elif choice == "q":
            return "quit"
        elif choice == "d":
            return "toggle_dry_run"
        else:
            print("Invalid choice, please try again.")

def main():
    parser = argparse.ArgumentParser(description='Reorganize codebase according to CLAUDE.md plan')
    parser.add_argument('--step', type=int, help='Start from specific step (1-based)')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry run mode (no actual changes)')
    parser.add_argument('--non-interactive', action='store_true', help='Run all steps without prompting')
    
    args = parser.parse_args()
    
    current_step = args.step if args.step else 1
    dry_run = args.dry_run
    interactive = not args.non_interactive
    
    # Check current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("\n===== IPFS ACCELERATE CODEBASE REORGANIZATION =====")
    print(f"Starting reorganization process from step {}}}}}}}current_step}"):
    print(f"Dry run mode: {}}}}}}}'ON' if dry_run else 'OFF'}"):
        print(f"Interactive mode: {}}}}}}}'ON' if interactive else 'OFF'}")
    print("\nThis script will guide you through the reorganization of the codebase"):
        print("according to the plan in CLAUDE.md. The following steps will be performed:")
    
    for i, step in enumerate(STEPS, 1):
        print(f"{}}}}}}}i}. {}}}}}}}step[],'name']}: {}}}}}}}step[],'description']}")
    
    if interactive:
        input("\nPress Enter to start...")
    
    # Main loop
    while current_step <= len(STEPS):
        step = STEPS[],current_step - 1]
        script = step[],'script']
        
        print(f"\n\n===== STEP {}}}}}}}current_step}/{}}}}}}}len(STEPS)}: {}}}}}}}step[],'name']} =====")
        print(f"Description: {}}}}}}}step[],'description']}")
        print(f"Script: {}}}}}}}script}")
        print(f"Dry run: {}}}}}}}'ON' if dry_run else 'OFF'}"):
        
        # Check if script exists
        if not check_script_exists(script):
            print(f"❌ Error: Script {}}}}}}}script} not found")
            if interactive:
                choice = get_step_input(current_step, len(STEPS))
                if choice == "skip":
                    current_step += 1
                continue
                elif choice == "previous" and current_step > 1:
                    current_step -= 1
                continue
                elif choice == "quit":
                    print("Quitting reorganization process.")
                return 0
                elif choice == "toggle_dry_run":
                    dry_run = not dry_run
                continue
                else:
                    print("Cannot run missing script. Skipping.")
                    current_step += 1
                continue
            else:
                print("Skipping step due to missing script.")
                current_step += 1
                continue
        
        if interactive:
            choice = get_step_input(current_step, len(STEPS))
            if choice == "skip":
                current_step += 1
            continue
            elif choice == "previous" and current_step > 1:
                current_step -= 1
            continue
            elif choice == "quit":
                print("Quitting reorganization process.")
            return 0
            elif choice == "toggle_dry_run":
                dry_run = not dry_run
            continue
        
        # Run the script
            result = run_script(script, dry_run)
        
        if result == 0:
            print(f"\n✅ Step {}}}}}}}current_step} completed successfully!")
        else:
            print(f"\n⚠️ Step {}}}}}}}current_step} completed with errors.")
            if interactive:
                choice = input("Continue anyway? (y/n): ").strip().lower()
                if choice != "y":
                    print("Stopping reorganization process.")
                return 1
        
        # Move to next step
                current_step += 1
    
                print("\n🎉 Reorganization process completed!")
                print("The codebase has been reorganized according to the plan in CLAUDE.md.")
                print("Please check the results and run tests to verify everything works correctly.")
    
            return 0

if __name__ == "__main__":
    sys.exit(main())