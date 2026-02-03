"""
Docker Executor Module for IPFS Accelerate

This module provides core functionality for executing code in Docker containers,
including:
- Executing pre-built containers from Docker Hub
- Building containers from GitHub repositories
- Managing container lifecycle
- Handling entrypoint + payload execution

This is the core module that gets exposed through MCP tools.
"""

import os
import sys
import json
import logging
import tempfile
import subprocess
import shutil
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class DockerExecutionConfig:
    """Configuration for Docker execution"""
    # Container settings
    image: str
    command: Optional[List[str]] = None
    entrypoint: Optional[List[str]] = None
    working_dir: Optional[str] = None
    
    # Resource limits
    memory_limit: Optional[str] = "2g"  # e.g., "512m", "2g"
    cpu_limit: Optional[float] = None  # e.g., 1.0, 2.5
    timeout: Optional[int] = 300  # seconds
    
    # Environment and volumes
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)  # host_path: container_path
    
    # Network settings
    network_mode: str = "none"  # "none", "bridge", "host"
    
    # Security
    read_only: bool = False
    no_new_privileges: bool = True
    user: Optional[str] = None
    
    # Output
    capture_output: bool = True
    stream_output: bool = False


@dataclass
class GitHubDockerConfig:
    """Configuration for building and running from GitHub"""
    repo_url: str
    branch: Optional[str] = "main"
    dockerfile_path: str = "Dockerfile"
    build_args: Dict[str, str] = field(default_factory=dict)
    context_path: str = "."


@dataclass
class DockerExecutionResult:
    """Result of Docker execution"""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    container_id: Optional[str] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None


class DockerExecutor:
    """
    Core Docker executor for running containers
    
    This class provides the foundational functionality for executing
    code in Docker containers, which is then exposed through MCP tools.
    """
    
    def __init__(self, docker_command: str = "docker"):
        """
        Initialize Docker executor
        
        Args:
            docker_command: Path to docker executable (default: "docker")
        """
        self.docker_command = docker_command
        self._verify_docker_available()
        logger.info("DockerExecutor initialized")
    
    def _verify_docker_available(self):
        """Verify Docker is available and accessible"""
        try:
            result = subprocess.run(
                [self.docker_command, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"Docker not available: {result.stderr}")
            logger.info(f"Docker version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"Docker is not available: {e}")
    
    def execute_container(
        self,
        config: DockerExecutionConfig
    ) -> DockerExecutionResult:
        """
        Execute a Docker container with the given configuration
        
        Args:
            config: Docker execution configuration
            
        Returns:
            DockerExecutionResult with execution details
        """
        start_time = time.time()
        container_id = None
        
        try:
            # Build docker run command
            cmd = self._build_docker_command(config)
            
            logger.info(f"Executing Docker container: {config.image}")
            logger.debug(f"Docker command: {' '.join(cmd)}")
            
            # Execute container
            if config.stream_output:
                result = self._execute_streaming(cmd, config.timeout)
            else:
                result = self._execute_capture(cmd, config.timeout)
            
            execution_time = time.time() - start_time
            
            return DockerExecutionResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                container_id=container_id,
                execution_time=execution_time
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return DockerExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Execution timed out",
                execution_time=execution_time,
                error_message=f"Container execution exceeded timeout of {config.timeout}s"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Docker execution failed: {e}")
            return DockerExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _build_docker_command(self, config: DockerExecutionConfig) -> List[str]:
        """Build docker run command from configuration"""
        cmd = [self.docker_command, "run", "--rm"]
        
        # Resource limits
        if config.memory_limit:
            cmd.extend(["--memory", config.memory_limit])
        if config.cpu_limit:
            cmd.extend(["--cpus", str(config.cpu_limit)])
        
        # Security settings
        if config.read_only:
            cmd.append("--read-only")
        if config.no_new_privileges:
            cmd.append("--security-opt=no-new-privileges")
        if config.user:
            cmd.extend(["--user", config.user])
        
        # Network
        cmd.extend(["--network", config.network_mode])
        
        # Environment variables
        for key, value in config.environment.items():
            cmd.extend(["-e", f"{key}={value}"])
        
        # Volumes
        for host_path, container_path in config.volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])
        
        # Working directory
        if config.working_dir:
            cmd.extend(["-w", config.working_dir])
        
        # Entrypoint
        if config.entrypoint:
            cmd.extend(["--entrypoint", config.entrypoint[0]])
        
        # Image
        cmd.append(config.image)
        
        # Command/args
        if config.entrypoint and len(config.entrypoint) > 1:
            cmd.extend(config.entrypoint[1:])
        elif config.command:
            cmd.extend(config.command)
        
        return cmd
    
    def _execute_capture(self, cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
        """Execute command and capture output"""
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    
    def _execute_streaming(self, cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
        """Execute command with streaming output"""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout_lines = []
        stderr_lines = []
        
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            stdout_lines.append(stdout)
            stderr_lines.append(stderr)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            stdout_lines.append(stdout)
            stderr_lines.append(stderr)
            raise
        
        # Create a result object similar to subprocess.run
        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=''.join(stdout_lines),
            stderr=''.join(stderr_lines)
        )
        return result
    
    def build_and_execute_github_repo(
        self,
        github_config: GitHubDockerConfig,
        execution_config: DockerExecutionConfig
    ) -> DockerExecutionResult:
        """
        Clone a GitHub repository, build a Docker image, and execute it
        
        Args:
            github_config: GitHub repository configuration
            execution_config: Docker execution configuration
            
        Returns:
            DockerExecutionResult with execution details
        """
        temp_dir = None
        image_tag = None
        
        try:
            # Create temporary directory for cloning
            temp_dir = tempfile.mkdtemp(prefix="ipfs_docker_")
            logger.info(f"Cloning GitHub repo: {github_config.repo_url}")
            
            # Clone repository
            clone_result = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", github_config.branch,
                 github_config.repo_url, temp_dir],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if clone_result.returncode != 0:
                return DockerExecutionResult(
                    success=False,
                    exit_code=clone_result.returncode,
                    stdout=clone_result.stdout,
                    stderr=clone_result.stderr,
                    error_message="Failed to clone GitHub repository"
                )
            
            # Build Docker image
            image_tag = f"ipfs-github-{int(time.time())}"
            build_context = os.path.join(temp_dir, github_config.context_path)
            dockerfile_path = os.path.join(temp_dir, github_config.dockerfile_path)
            
            logger.info(f"Building Docker image: {image_tag}")
            
            build_cmd = [
                self.docker_command, "build",
                "-t", image_tag,
                "-f", dockerfile_path
            ]
            
            # Add build args
            for key, value in github_config.build_args.items():
                build_cmd.extend(["--build-arg", f"{key}={value}"])
            
            build_cmd.append(build_context)
            
            build_result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if build_result.returncode != 0:
                return DockerExecutionResult(
                    success=False,
                    exit_code=build_result.returncode,
                    stdout=build_result.stdout,
                    stderr=build_result.stderr,
                    error_message="Failed to build Docker image"
                )
            
            # Execute the built image
            execution_config.image = image_tag
            result = self.execute_container(execution_config)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to build and execute GitHub repo: {e}")
            return DockerExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                error_message=str(e)
            )
        finally:
            # Cleanup
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Remove built image
            if image_tag:
                try:
                    subprocess.run(
                        [self.docker_command, "rmi", "-f", image_tag],
                        capture_output=True,
                        timeout=30
                    )
                except Exception as e:
                    logger.warning(f"Failed to cleanup image {image_tag}: {e}")
    
    def list_running_containers(self) -> List[Dict[str, str]]:
        """
        List currently running containers
        
        Returns:
            List of container information dictionaries
        """
        try:
            result = subprocess.run(
                [self.docker_command, "ps", "--format", "{{json .}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to list containers: {result.stderr}")
                return []
            
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        containers.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            
            return containers
            
        except Exception as e:
            logger.error(f"Failed to list containers: {e}")
            return []
    
    def stop_container(self, container_id: str, timeout: int = 10) -> bool:
        """
        Stop a running container
        
        Args:
            container_id: Container ID or name
            timeout: Timeout in seconds before forcing kill
            
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            result = subprocess.run(
                [self.docker_command, "stop", "-t", str(timeout), container_id],
                capture_output=True,
                text=True,
                timeout=timeout + 5
            )
            
            if result.returncode == 0:
                logger.info(f"Stopped container: {container_id}")
                return True
            else:
                logger.error(f"Failed to stop container {container_id}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to stop container {container_id}: {e}")
            return False
    
    def pull_image(self, image: str) -> bool:
        """
        Pull a Docker image from Docker Hub
        
        Args:
            image: Image name (e.g., "ubuntu:20.04", "python:3.9")
            
        Returns:
            True if pulled successfully, False otherwise
        """
        try:
            logger.info(f"Pulling Docker image: {image}")
            result = subprocess.run(
                [self.docker_command, "pull", image],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully pulled image: {image}")
                return True
            else:
                logger.error(f"Failed to pull image {image}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to pull image {image}: {e}")
            return False


# Convenience functions for common operations

def execute_docker_hub_container(
    image: str,
    command: Optional[List[str]] = None,
    entrypoint: Optional[List[str]] = None,
    environment: Optional[Dict[str, str]] = None,
    volumes: Optional[Dict[str, str]] = None,
    timeout: int = 300,
    **kwargs
) -> DockerExecutionResult:
    """
    Execute a container from Docker Hub
    
    This is a convenience function that wraps DockerExecutor.execute_container
    for easy use from MCP tools.
    
    Args:
        image: Docker image name from Docker Hub
        command: Command to run in container
        entrypoint: Custom entrypoint
        environment: Environment variables
        volumes: Volume mounts (host_path: container_path)
        timeout: Execution timeout in seconds
        **kwargs: Additional DockerExecutionConfig parameters
        
    Returns:
        DockerExecutionResult
    """
    config = DockerExecutionConfig(
        image=image,
        command=command,
        entrypoint=entrypoint,
        environment=environment or {},
        volumes=volumes or {},
        timeout=timeout,
        **kwargs
    )
    
    executor = DockerExecutor()
    return executor.execute_container(config)


def build_and_execute_from_github(
    repo_url: str,
    branch: str = "main",
    dockerfile_path: str = "Dockerfile",
    command: Optional[List[str]] = None,
    entrypoint: Optional[List[str]] = None,
    environment: Optional[Dict[str, str]] = None,
    build_args: Optional[Dict[str, str]] = None,
    timeout: int = 300,
    **kwargs
) -> DockerExecutionResult:
    """
    Clone a GitHub repository, build Docker image, and execute
    
    This is a convenience function for building and running containers
    from GitHub repositories.
    
    Args:
        repo_url: GitHub repository URL
        branch: Git branch to clone
        dockerfile_path: Path to Dockerfile in repo
        command: Command to run in container
        entrypoint: Custom entrypoint
        environment: Environment variables
        build_args: Docker build arguments
        timeout: Execution timeout in seconds
        **kwargs: Additional configuration parameters
        
    Returns:
        DockerExecutionResult
    """
    github_config = GitHubDockerConfig(
        repo_url=repo_url,
        branch=branch,
        dockerfile_path=dockerfile_path,
        build_args=build_args or {}
    )
    
    execution_config = DockerExecutionConfig(
        image="placeholder",  # Will be replaced with built image
        command=command,
        entrypoint=entrypoint,
        environment=environment or {},
        timeout=timeout,
        **kwargs
    )
    
    executor = DockerExecutor()
    return executor.build_and_execute_github_repo(github_config, execution_config)
