#!/usr/bin/env python3
"""
Docker Execution Examples

This script demonstrates various ways to use the Docker execution
features in IPFS Accelerate.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ipfs_accelerate_py.docker_executor import (
    DockerExecutor,
    DockerExecutionConfig,
    execute_docker_hub_container,
    build_and_execute_from_github
)

from ipfs_accelerate_py.mcp.tools.docker_tools import (
    execute_docker_container,
    build_and_execute_github_repo,
    execute_with_payload,
    list_running_containers
)


def example1_simple_python():
    """Example 1: Simple Python script execution"""
    print("\n" + "="*70)
    print("Example 1: Simple Python Script Execution")
    print("="*70)
    
    result = execute_docker_container(
        image="python:3.9-slim",
        command="python -c 'print(\"Hello from Docker!\"); print(2+2)'",
        memory_limit="256m",
        timeout=30
    )
    
    print(f"Success: {result['success']}")
    print(f"Exit code: {result['exit_code']}")
    print(f"Output:\n{result['stdout']}")
    print(f"Execution time: {result['execution_time']:.2f}s")


def example2_environment_variables():
    """Example 2: Using environment variables"""
    print("\n" + "="*70)
    print("Example 2: Environment Variables")
    print("="*70)
    
    result = execute_docker_container(
        image="python:3.9-slim",
        command="python -c 'import os; print(f\"VAR1={os.environ.get(\\\"VAR1\\\")}\"); print(f\"VAR2={os.environ.get(\\\"VAR2\\\")}\")'",
        environment={
            "VAR1": "value1",
            "VAR2": "value2"
        },
        memory_limit="256m"
    )
    
    print(f"Output:\n{result['stdout']}")


def example3_shell_commands():
    """Example 3: Running shell commands"""
    print("\n" + "="*70)
    print("Example 3: Shell Commands")
    print("="*70)
    
    commands = [
        "echo 'System Information:'",
        "uname -a",
        "echo 'Date:'",
        "date",
        "echo 'Done!'"
    ]
    
    result = execute_docker_container(
        image="ubuntu:20.04",
        command=" && ".join(commands),
        entrypoint="sh -c",
        memory_limit="256m"
    )
    
    print(f"Output:\n{result['stdout']}")


def example4_custom_payload():
    """Example 4: Execute with custom payload"""
    print("\n" + "="*70)
    print("Example 4: Custom Payload Execution")
    print("="*70)
    
    # Python script as payload
    python_script = """
import sys
import json

def process_data(data):
    return {
        'input': data,
        'length': len(data),
        'uppercase': data.upper(),
        'reversed': data[::-1]
    }

data = "Hello World"
result = process_data(data)
print(json.dumps(result, indent=2))
"""
    
    result = execute_with_payload(
        image="python:3.9-slim",
        payload=python_script,
        payload_path="/app/processor.py",
        entrypoint="python /app/processor.py",
        memory_limit="256m"
    )
    
    print(f"Output:\n{result['stdout']}")


def example5_data_processing():
    """Example 5: Data processing task"""
    print("\n" + "="*70)
    print("Example 5: Data Processing")
    print("="*70)
    
    # Process CSV data
    csv_processor = """
import csv
import io

data = '''name,age,city
Alice,30,NYC
Bob,25,LA
Charlie,35,Chicago'''

# Parse CSV
reader = csv.DictReader(io.StringIO(data))
people = list(reader)

# Process data
print("People older than 27:")
for person in people:
    if int(person['age']) > 27:
        print(f"  {person['name']}: {person['age']} years old in {person['city']}")

# Calculate average age
avg_age = sum(int(p['age']) for p in people) / len(people)
print(f"\\nAverage age: {avg_age:.1f}")
"""
    
    result = execute_with_payload(
        image="python:3.9-slim",
        payload=csv_processor,
        payload_path="/app/process.py",
        entrypoint="python /app/process.py",
        memory_limit="256m"
    )
    
    print(f"Output:\n{result['stdout']}")


def example6_github_repo():
    """Example 6: Build and execute from GitHub (mock)"""
    print("\n" + "="*70)
    print("Example 6: GitHub Repository Build (Demonstration)")
    print("="*70)
    
    print("This would build and execute a GitHub repository:")
    print("""
    result = build_and_execute_github_repo(
        repo_url="https://github.com/user/python-app",
        branch="main",
        dockerfile_path="Dockerfile",
        command="python app.py",
        environment={"ENV": "production"},
        build_args={"PYTHON_VERSION": "3.9"}
    )
    """)
    
    print("\nNote: Requires a valid GitHub repository with a Dockerfile")
    print("Skipping actual execution in this example")


def example7_resource_limits():
    """Example 7: Testing resource limits"""
    print("\n" + "="*70)
    print("Example 7: Resource Limits")
    print("="*70)
    
    # Test memory limit
    print("Testing with strict resource limits:")
    
    result = execute_docker_container(
        image="python:3.9-slim",
        command="python -c 'print(\"Running with resource limits\"); import sys; print(f\"Python version: {sys.version}\")'",
        memory_limit="128m",  # Very low memory
        cpu_limit=0.5,  # Half a CPU core
        timeout=10,
        memory_limit="256m"
    )
    
    print(f"Success: {result['success']}")
    print(f"Output:\n{result['stdout']}")


def example8_timeout_handling():
    """Example 8: Timeout handling"""
    print("\n" + "="*70)
    print("Example 8: Timeout Handling")
    print("="*70)
    
    print("Executing a command with short timeout:")
    
    result = execute_docker_container(
        image="ubuntu:20.04",
        command="echo 'Starting...'; echo 'Done!'",
        timeout=5,  # 5 second timeout
        memory_limit="128m"
    )
    
    if result['success']:
        print(f"Completed successfully in {result['execution_time']:.2f}s")
        print(f"Output: {result['stdout']}")
    else:
        print(f"Failed: {result['error_message']}")


def example9_container_management():
    """Example 9: Container management"""
    print("\n" + "="*70)
    print("Example 9: Container Management")
    print("="*70)
    
    print("Listing running containers:")
    
    result = list_running_containers()
    
    print(f"Found {result['count']} running containers")
    for container in result['containers']:
        print(f"  - {container.get('Names', 'N/A')}: {container.get('Status', 'N/A')}")


def example10_multi_language():
    """Example 10: Multiple language support"""
    print("\n" + "="*70)
    print("Example 10: Multiple Languages")
    print("="*70)
    
    examples = [
        {
            "name": "Python",
            "image": "python:3.9-slim",
            "command": "python -c 'print(\"Hello from Python\")'",
        },
        {
            "name": "Node.js",
            "image": "node:16-slim",
            "command": "node -e 'console.log(\"Hello from Node.js\")'",
        },
        {
            "name": "Ruby",
            "image": "ruby:3.0-slim",
            "command": "ruby -e 'puts \"Hello from Ruby\"'",
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        result = execute_docker_container(
            image=example['image'],
            command=example['command'],
            memory_limit="128m",
            timeout=30
        )
        if result['success']:
            print(f"  Output: {result['stdout'].strip()}")
        else:
            print(f"  Error: {result.get('error_message', 'Unknown error')}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("DOCKER EXECUTION EXAMPLES FOR IPFS ACCELERATE")
    print("="*70)
    print("\nThese examples demonstrate various Docker execution capabilities.")
    print("Note: Docker must be installed and running for these to work.")
    
    # Check if Docker is available
    try:
        from ipfs_accelerate_py.docker_executor import DockerExecutor
        executor = DockerExecutor()
        print("\n✅ Docker is available and ready!")
    except Exception as e:
        print(f"\n❌ Docker is not available: {e}")
        print("Please install Docker to run these examples.")
        return
    
    # Run examples
    try:
        example1_simple_python()
        example2_environment_variables()
        example3_shell_commands()
        example4_custom_payload()
        example5_data_processing()
        example6_github_repo()
        example7_resource_limits()
        example8_timeout_handling()
        example9_container_management()
        example10_multi_language()
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nFor more information, see docs/DOCKER_EXECUTION.md")


if __name__ == "__main__":
    main()
