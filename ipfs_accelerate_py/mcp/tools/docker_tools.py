"""
Docker Tools for MCP Server

This module exposes Docker execution capabilities as MCP tools.
Following the architecture:
  ipfs_accelerate_py.docker_executor (core module)
      ↓
  mcp/tools/docker_tools.py (MCP tool wrapper)
      ↓
  MCP server JavaScript SDK
"""

import logging
import json
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Import core docker_executor module
try:
    from ipfs_accelerate_py.docker_executor import (
        DockerExecutor,
        DockerExecutionConfig,
        GitHubDockerConfig,
        execute_docker_hub_container,
        build_and_execute_from_github
    )
    HAVE_DOCKER_EXECUTOR = True
except ImportError as e:
    logger.warning(f"Docker executor not available: {e}")
    HAVE_DOCKER_EXECUTOR = False


def execute_docker_container(
    image: str,
    command: Optional[str] = None,
    entrypoint: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    memory_limit: str = "2g",
    cpu_limit: Optional[float] = None,
    timeout: int = 300,
    network_mode: str = "none",
    working_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a Docker container from Docker Hub
    
    MCP Tool: Runs a pre-built Docker container with specified configuration.
    This allows executing arbitrary code in isolated containers.
    
    Args:
        image: Docker image name from Docker Hub (e.g., "python:3.9", "ubuntu:20.04")
        command: Command to run in the container (space-separated string)
        entrypoint: Custom entrypoint override (space-separated string)
        environment: Environment variables as key-value pairs
        memory_limit: Memory limit (e.g., "512m", "2g")
        cpu_limit: CPU limit as number of cores (e.g., 1.0, 2.5)
        timeout: Execution timeout in seconds
        network_mode: Network mode ("none", "bridge", "host")
        working_dir: Working directory in container
        
    Returns:
        Dictionary with execution results:
        - success: bool
        - exit_code: int
        - stdout: str
        - stderr: str
        - execution_time: float
        - error_message: str (if failed)
        
    Example:
        # Run Python script in container
        execute_docker_container(
            image="python:3.9",
            command='python -c "print(\'Hello from Docker!\')"',
            memory_limit="512m",
            timeout=60
        )
        
        # Run shell command
        execute_docker_container(
            image="ubuntu:20.04",
            command="echo 'Testing Docker execution'",
            network_mode="none"
        )
    """
    if not HAVE_DOCKER_EXECUTOR:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "Docker executor module not available",
            "error_message": "Docker executor module not available. Please ensure Docker is installed."
        }
    
    try:
        # Parse command and entrypoint strings into lists
        command_list = command.split() if command else None
        entrypoint_list = entrypoint.split() if entrypoint else None
        
        # Execute container
        result = execute_docker_hub_container(
            image=image,
            command=command_list,
            entrypoint=entrypoint_list,
            environment=environment or {},
            memory_limit=memory_limit,
            cpu_limit=cpu_limit,
            timeout=timeout,
            network_mode=network_mode,
            working_dir=working_dir
        )
        
        return {
            "success": result.success,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": result.execution_time,
            "error_message": result.error_message
        }
        
    except Exception as e:
        logger.error(f"Error executing Docker container: {e}")
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "error_message": f"Failed to execute container: {e}"
        }


def build_and_execute_github_repo(
    repo_url: str,
    branch: str = "main",
    dockerfile_path: str = "Dockerfile",
    command: Optional[str] = None,
    entrypoint: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    build_args: Optional[Dict[str, str]] = None,
    memory_limit: str = "2g",
    timeout: int = 600,
    context_path: str = "."
) -> Dict[str, Any]:
    """
    Clone a GitHub repository, build Docker image, and execute
    
    MCP Tool: Dockerizes a GitHub repository with a Dockerfile and executes it.
    This enables running any GitHub repository that contains a Dockerfile.
    
    Args:
        repo_url: GitHub repository URL (e.g., "https://github.com/user/repo")
        branch: Git branch to clone (default: "main")
        dockerfile_path: Path to Dockerfile in repo (default: "Dockerfile")
        command: Command to run in the built container
        entrypoint: Custom entrypoint for the container
        environment: Environment variables
        build_args: Docker build arguments
        memory_limit: Memory limit for execution
        timeout: Timeout for build + execution (seconds)
        context_path: Docker build context path in repo
        
    Returns:
        Dictionary with execution results including build logs
        
    Example:
        # Build and run a Python app from GitHub
        build_and_execute_github_repo(
            repo_url="https://github.com/user/python-app",
            branch="main",
            dockerfile_path="Dockerfile",
            command="python app.py",
            environment={"ENV": "production"},
            build_args={"PYTHON_VERSION": "3.9"}
        )
    """
    if not HAVE_DOCKER_EXECUTOR:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "Docker executor module not available",
            "error_message": "Docker executor module not available"
        }
    
    try:
        # Parse command and entrypoint
        command_list = command.split() if command else None
        entrypoint_list = entrypoint.split() if entrypoint else None
        
        # Build and execute
        result = build_and_execute_from_github(
            repo_url=repo_url,
            branch=branch,
            dockerfile_path=dockerfile_path,
            command=command_list,
            entrypoint=entrypoint_list,
            environment=environment or {},
            build_args=build_args or {},
            memory_limit=memory_limit,
            timeout=timeout
        )
        
        return {
            "success": result.success,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": result.execution_time,
            "error_message": result.error_message
        }
        
    except Exception as e:
        logger.error(f"Error building and executing GitHub repo: {e}")
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "error_message": f"Failed to build and execute: {e}"
        }


def execute_with_payload(
    image: str,
    payload: str,
    payload_path: str = "/tmp/payload",
    entrypoint: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    memory_limit: str = "2g",
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Execute a Docker container with a payload file
    
    MCP Tool: Executes a container with a custom payload (code, data, config).
    The payload is written to a temporary file and mounted into the container.
    
    Args:
        image: Docker image name
        payload: Content to write to payload file
        payload_path: Path where payload will be available in container
        entrypoint: Command to execute (can reference payload_path)
        environment: Environment variables
        memory_limit: Memory limit
        timeout: Execution timeout
        
    Returns:
        Dictionary with execution results
        
    Example:
        # Execute Python code from payload
        execute_with_payload(
            image="python:3.9",
            payload="print('Hello from payload!')\nprint(2+2)",
            payload_path="/app/script.py",
            entrypoint="python /app/script.py"
        )
        
        # Execute shell script
        execute_with_payload(
            image="ubuntu:20.04",
            payload="#!/bin/bash\necho 'Running script'\ndate",
            payload_path="/tmp/script.sh",
            entrypoint="bash /tmp/script.sh"
        )
    """
    if not HAVE_DOCKER_EXECUTOR:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "Docker executor module not available",
            "error_message": "Docker executor module not available"
        }
    
    import tempfile
    import os
    
    temp_file = None
    try:
        # Write payload to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.payload') as f:
            f.write(payload)
            temp_file = f.name
        
        # Make executable if it's a script
        if payload.startswith("#!"):
            os.chmod(temp_file, 0o755)
        
        # Mount the payload file into the container
        volumes = {temp_file: payload_path}
        
        # Parse entrypoint
        entrypoint_list = entrypoint.split() if entrypoint else None
        
        # Execute container with mounted payload
        result = execute_docker_hub_container(
            image=image,
            entrypoint=entrypoint_list,
            environment=environment or {},
            volumes=volumes,
            memory_limit=memory_limit,
            timeout=timeout
        )
        
        return {
            "success": result.success,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": result.execution_time,
            "error_message": result.error_message
        }
        
    except Exception as e:
        logger.error(f"Error executing with payload: {e}")
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "error_message": f"Failed to execute with payload: {e}"
        }
    finally:
        # Cleanup temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")


def list_running_containers() -> Dict[str, Any]:
    """
    List currently running Docker containers
    
    MCP Tool: Returns information about all running containers.
    
    Returns:
        Dictionary with:
        - success: bool
        - containers: List of container information
        - count: Number of running containers
        
    Example:
        result = list_running_containers()
        for container in result["containers"]:
            print(f"Container: {container['Names']} - {container['Status']}")
    """
    if not HAVE_DOCKER_EXECUTOR:
        return {
            "success": False,
            "containers": [],
            "count": 0,
            "error_message": "Docker executor module not available"
        }
    
    try:
        executor = DockerExecutor()
        containers = executor.list_running_containers()
        
        return {
            "success": True,
            "containers": containers,
            "count": len(containers)
        }
        
    except Exception as e:
        logger.error(f"Error listing containers: {e}")
        return {
            "success": False,
            "containers": [],
            "count": 0,
            "error_message": str(e)
        }


def stop_container(
    container_id: str,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Stop a running Docker container
    
    MCP Tool: Stops a running container by ID or name.
    
    Args:
        container_id: Container ID or name
        timeout: Timeout before force kill
        
    Returns:
        Dictionary with:
        - success: bool
        - container_id: str
        - message: str
        
    Example:
        stop_container(container_id="my_container")
    """
    if not HAVE_DOCKER_EXECUTOR:
        return {
            "success": False,
            "container_id": container_id,
            "message": "Docker executor module not available"
        }
    
    try:
        executor = DockerExecutor()
        success = executor.stop_container(container_id, timeout)
        
        return {
            "success": success,
            "container_id": container_id,
            "message": "Container stopped successfully" if success else "Failed to stop container"
        }
        
    except Exception as e:
        logger.error(f"Error stopping container: {e}")
        return {
            "success": False,
            "container_id": container_id,
            "message": str(e)
        }


def pull_docker_image(image: str) -> Dict[str, Any]:
    """
    Pull a Docker image from Docker Hub
    
    MCP Tool: Downloads a Docker image to prepare for execution.
    
    Args:
        image: Docker image name (e.g., "python:3.9", "ubuntu:20.04")
        
    Returns:
        Dictionary with:
        - success: bool
        - image: str
        - message: str
        
    Example:
        pull_docker_image(image="python:3.9-slim")
    """
    if not HAVE_DOCKER_EXECUTOR:
        return {
            "success": False,
            "image": image,
            "message": "Docker executor module not available"
        }
    
    try:
        executor = DockerExecutor()
        success = executor.pull_image(image)
        
        return {
            "success": success,
            "image": image,
            "message": f"Image {image} pulled successfully" if success else f"Failed to pull image {image}"
        }
        
    except Exception as e:
        logger.error(f"Error pulling image: {e}")
        return {
            "success": False,
            "image": image,
            "message": str(e)
        }


# Tool registration for MCP server
MCP_DOCKER_TOOLS = {
    "execute_docker_container": execute_docker_container,
    "build_and_execute_github_repo": build_and_execute_github_repo,
    "execute_with_payload": execute_with_payload,
    "list_running_containers": list_running_containers,
    "stop_container": stop_container,
    "pull_docker_image": pull_docker_image,
}


def register_docker_tools(mcp_server):
    """
    Register Docker tools with MCP server
    
    This function should be called during MCP server initialization
    to expose Docker execution capabilities.
    
    Args:
        mcp_server: MCP server instance
    """
    if not HAVE_DOCKER_EXECUTOR:
        logger.warning("Docker executor not available - Docker tools not registered")
        return
    
    logger.info("Registering Docker tools with MCP server")
    
    for tool_name, tool_func in MCP_DOCKER_TOOLS.items():
        try:
            # Use register_tool so we can attach execution context metadata.
            # FastMCP compatibility: register_tool shim will delegate to tool()
            # and ignore extra kwargs.
            mcp_server.register_tool(
                name=str(tool_name),
                function=tool_func,
                description=str(getattr(tool_func, "__doc__", "") or "Docker tool"),
                input_schema={"type": "object", "properties": {}, "required": []},
                execution_context="worker",
            )
            logger.debug(f"Registered Docker tool: {tool_name}")
        except Exception as e:
            logger.error(f"Failed to register Docker tool {tool_name}: {e}")
