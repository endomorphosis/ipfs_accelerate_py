#!/usr/bin/env python3
"""
Test script with explicit file output
"""
import sys
import os
import datetime

# Create output files
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
stdout_file = os.path.join(log_dir, f"test_stdout_{timestamp}.log")
stderr_file = os.path.join(log_dir, f"test_stderr_{timestamp}.log")

with open(stdout_file, 'w') as stdout_f, open(stderr_file, 'w') as stderr_f:
    # Redirect output
    stdout_f.write("Starting test...\n")
    stdout_f.flush()
    
    try:
        stdout_f.write("Importing FastMCP...\n")
        stdout_f.flush()
        from fastmcp import FastMCP
        
        stdout_f.write("Creating FastMCP server...\n")
        stdout_f.flush()
        mcp = FastMCP("File Output Test")
        
        stdout_f.write(f"Created server: {mcp.name}\n")
        stdout_f.flush()
        
        @mcp.tool()
        def hello(name: str = "World") -> str:
            """Say hello to someone"""
            return f"Hello, {name}!"
        
        stdout_f.write("Added a tool\n")
        stdout_f.write(f"Tool count: {len(mcp.tools)}\n")
        stdout_f.flush()
        
        result = hello(name="File Test")
        stdout_f.write(f"Tool result: {result}\n")
        stdout_f.flush()
        
        stdout_f.write("Test completed successfully!\n")
        stdout_f.flush()
        
    except ImportError as e:
        stderr_f.write(f"Import error: {str(e)}\n")
        stderr_f.flush()
        
    except Exception as e:
        stderr_f.write(f"Error: {str(e)}\n")
        import traceback
        traceback.print_exc(file=stderr_f)
        stderr_f.flush()

# Print the file paths to the console
print(f"Test output written to: {stdout_file}")
print(f"Error log written to: {stderr_file}")

# Read and print the content of the files
print("\nTest output:")
with open(stdout_file, 'r') as f:
    print(f.read())

print("\nError log:")
with open(stderr_file, 'r') as f:
    errors = f.read()
    if errors:
        print(errors)
    else:
        print("No errors reported.")
