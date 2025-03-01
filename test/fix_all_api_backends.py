#!/usr/bin/env python
"""
Master script to fix all API backends with:
1. Environment variable handling for API keys
2. Request queueing
3. Exponential backoff retry
4. Test updates for the new functionality
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_script(script_path, args=None):
    """Run a Python script with optional arguments"""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
        
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(e.stdout)
        print(e.stderr)
        return False

def create_env_file():
    """Create a .env.example file if it doesn't exist"""
    env_example_path = Path(__file__).parent / ".env.example"
    
    if not env_example_path.exists():
        print("Creating .env.example file...")
        env_content = """# Example environment variables for API access
# Copy this file to .env and fill in your API keys
# WARNING: Never commit API keys to version control!

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Groq API
GROQ_API_KEY=your_groq_api_key_here

# Claude API
CLAUDE_API_KEY=your_claude_api_key_here

# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
"""
        with open(env_example_path, 'w') as f:
            f.write(env_content)
        print(f"Created {env_example_path}")
    else:
        print(f"{env_example_path} already exists")

def main():
    parser = argparse.ArgumentParser(description="Fix all API backends with environment variables, queue, and backoff")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Only print what would be done without making changes")
    parser.add_argument("--api", "-a", help="Only fix specific API backend", 
                        choices=["openai", "groq", "claude", "gemini", "all"])
    
    args = parser.parse_args()
    
    # Determine script directory
    script_dir = Path(__file__).parent
    
    # Create .env.example file
    create_env_file()
    
    # Fix specific API or all APIs
    success_count = 0
    failure_count = 0
    
    # 1. Fix OpenAI API implementation
    if args.api in ["openai", "all"]:
        print("\n=== Fixing OpenAI API implementation ===")
        script_path = script_dir / "fix_openai_api_implementation.py"
        script_args = ["--dry-run"] if args.dry_run else []
        
        if run_script(script_path, script_args):
            success_count += 1
        else:
            failure_count += 1
    
    # 2. Update OpenAI API tests
    if args.api in ["openai", "all"] and not args.dry_run:
        print("\n=== Updating OpenAI API tests ===")
        script_path = script_dir / "update_openai_api_tests.py"
        
        if run_script(script_path):
            success_count += 1
        else:
            failure_count += 1
    
    # 3. Fix all other API backends using general script
    if args.api in ["groq", "claude", "gemini", "all"]:
        print("\n=== Fixing other API backends ===")
        script_path = script_dir / "add_queue_backoff.py"
        
        # Determine which APIs to fix
        api_arg = args.api if args.api != "all" else "all"
        script_args = ["--api", api_arg]
        
        if args.dry_run:
            script_args.append("--dry-run")
        
        if run_script(script_path, script_args):
            success_count += 1
        else:
            failure_count += 1
    
    # 4. Update API tests for all backends
    if args.api != "openai" and not args.dry_run:
        print("\n=== Updating other API tests ===")
        script_path = script_dir / "update_api_tests.py"
        
        # Determine which APIs to update
        api_arg = args.api if args.api != "all" else "all"
        script_args = ["--api", api_arg]
        
        if run_script(script_path, script_args):
            success_count += 1
        else:
            failure_count += 1
    
    # Print summary
    print("\n=== Summary ===")
    if args.dry_run:
        print("Dry run completed - no files were modified")
    else:
        print(f"Successfully completed {success_count} operations")
        if failure_count > 0:
            print(f"Failed to complete {failure_count} operations")
    
    # Final instructions
    if not args.dry_run and success_count > 0:
        print("\n=== Next Steps ===")
        print("1. Install required dependencies:")
        print("   pip install -r requirements_api.txt")
        print("2. Create a .env file with your API keys based on .env.example")
        print("3. Run tests to verify the implementation:")
        print("   python -m test.apis.test_openai_api")
        print("4. Review README_API_ENHANCEMENTS.md for detailed documentation")

if __name__ == "__main__":
    main()