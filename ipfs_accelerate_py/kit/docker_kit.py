"""
Docker Kit - Core Docker Operations

This module provides core Docker operations without CLI dependencies.
It can be used by both the unified CLI and MCP server.
"""

import json
import logging
import subprocess
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DockerResult:
    """Result from a Docker operation."""
    success: bool
    data: Any = None
    exit_code: int = 0
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class DockerKit:
    """
    Core Docker operations module.
    
    Provides Docker functionality that can be used by CLI, MCP tools,
    or directly in Python code.
    """
    
    def __init__(
        self,
        docker_path: str = "docker",
        timeout: int = 300
    ):
        """
        Initialize Docker Kit.
        
        Args:
            docker_path: Path to docker executable
            timeout: Default timeout for operations
        """
        self.docker_path = docker_path
        self.timeout = timeout
        self._verify_installation()
    
    def _verify_installation(self) -> bool:
        """
        Verify that Docker is installed and running.
        
        Returns:
            True if available, False otherwise
        """
        try:
            result = subprocess.run(
                [self.docker_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Docker version: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"Docker verification failed: {result.returncode}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Docker not found: {e}")
            return False
    
    def _run_command(
        self,
        args: List[str],
        timeout: Optional[int] = None,
        input_data: Optional[str] = None
    ) -> DockerResult:
        """
        Run a Docker command.
        
        Args:
            args: Command arguments
            timeout: Timeout in seconds
            input_data: Input data to pass to stdin
            
        Returns:
            DockerResult with command output
        """
        import time
        timeout = timeout or self.timeout
        full_command = [self.docker_path] + args
        
        start_time = time.time()
        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=input_data
            )
            
            execution_time = time.time() - start_time
            
            return DockerResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time
            )
        
        except subprocess.TimeoutExpired:
            return DockerResult(
                success=False,
                error=f"Command timed out after {timeout} seconds",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return DockerResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    # Container Operations
    
    def run_container(
        self,
        image: str,
        command: Optional[Union[str, List[str]]] = None,
        detach: bool = False,
        remove: bool = True,
        name: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, str]] = None,
        ports: Optional[Dict[str, str]] = None,
        network: str = "none",
        memory: Optional[str] = None,
        cpus: Optional[float] = None,
        timeout: Optional[int] = None
    ) -> DockerResult:
        """
        Run a Docker container.
        
        Args:
            image: Docker image name
            command: Command to run in container
            detach: Run in detached mode
            remove: Remove container after execution
            name: Container name
            environment: Environment variables
            volumes: Volume mounts (host:container)
            ports: Port mappings (host:container)
            network: Network mode
            memory: Memory limit (e.g., "512m")
            cpus: CPU limit (e.g., 1.5)
            timeout: Execution timeout
            
        Returns:
            DockerResult with execution output
        """
        args = ["run"]
        
        if detach:
            args.append("-d")
        if remove:
            args.append("--rm")
        if name:
            args.extend(["--name", name])
        if network:
            args.extend(["--network", network])
        if memory:
            args.extend(["--memory", memory])
        if cpus:
            args.extend(["--cpus", str(cpus)])
        
        # Add environment variables
        if environment:
            for key, value in environment.items():
                args.extend(["-e", f"{key}={value}"])
        
        # Add volumes
        if volumes:
            for host_path, container_path in volumes.items():
                args.extend(["-v", f"{host_path}:{container_path}"])
        
        # Add ports
        if ports:
            for host_port, container_port in ports.items():
                args.extend(["-p", f"{host_port}:{container_port}"])
        
        # Add image
        args.append(image)
        
        # Add command
        if command:
            if isinstance(command, str):
                args.extend(command.split())
            else:
                args.extend(command)
        
        return self._run_command(args, timeout=timeout)
    
    def exec_container(
        self,
        container: str,
        command: Union[str, List[str]],
        interactive: bool = False,
        tty: bool = False
    ) -> DockerResult:
        """
        Execute command in running container.
        
        Args:
            container: Container ID or name
            command: Command to execute
            interactive: Keep STDIN open
            tty: Allocate pseudo-TTY
            
        Returns:
            DockerResult with execution output
        """
        args = ["exec"]
        
        if interactive:
            args.append("-i")
        if tty:
            args.append("-t")
        
        args.append(container)
        
        if isinstance(command, str):
            args.extend(command.split())
        else:
            args.extend(command)
        
        return self._run_command(args)
    
    def list_containers(
        self,
        all_containers: bool = False,
        filters: Optional[Dict[str, str]] = None
    ) -> DockerResult:
        """
        List Docker containers.
        
        Args:
            all_containers: Include stopped containers
            filters: Filters to apply
            
        Returns:
            DockerResult with container list
        """
        args = ["ps", "--format", "json"]
        
        if all_containers:
            args.append("-a")
        
        if filters:
            for key, value in filters.items():
                args.extend(["--filter", f"{key}={value}"])
        
        result = self._run_command(args)
        
        # Parse JSON lines output
        if result.success and result.stdout:
            try:
                containers = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
                result.data = containers
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse container list: {e}")
        
        return result
    
    def stop_container(
        self,
        container: str,
        timeout: int = 10
    ) -> DockerResult:
        """
        Stop a running container.
        
        Args:
            container: Container ID or name
            timeout: Timeout before force kill
            
        Returns:
            DockerResult with stop status
        """
        args = ["stop", "-t", str(timeout), container]
        return self._run_command(args)
    
    def remove_container(
        self,
        container: str,
        force: bool = False
    ) -> DockerResult:
        """
        Remove a container.
        
        Args:
            container: Container ID or name
            force: Force removal
            
        Returns:
            DockerResult with removal status
        """
        args = ["rm"]
        if force:
            args.append("-f")
        args.append(container)
        
        return self._run_command(args)
    
    # Image Operations
    
    def pull_image(
        self,
        image: str,
        timeout: Optional[int] = None
    ) -> DockerResult:
        """
        Pull a Docker image.
        
        Args:
            image: Image name with optional tag
            timeout: Pull timeout
            
        Returns:
            DockerResult with pull status
        """
        args = ["pull", image]
        return self._run_command(args, timeout=timeout or 600)
    
    def list_images(
        self,
        filters: Optional[Dict[str, str]] = None
    ) -> DockerResult:
        """
        List Docker images.
        
        Args:
            filters: Filters to apply
            
        Returns:
            DockerResult with image list
        """
        args = ["images", "--format", "json"]
        
        if filters:
            for key, value in filters.items():
                args.extend(["--filter", f"{key}={value}"])
        
        result = self._run_command(args)
        
        # Parse JSON lines output
        if result.success and result.stdout:
            try:
                images = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
                result.data = images
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse image list: {e}")
        
        return result
    
    def build_image(
        self,
        context_path: str,
        tag: Optional[str] = None,
        dockerfile: str = "Dockerfile",
        build_args: Optional[Dict[str, str]] = None,
        no_cache: bool = False,
        timeout: Optional[int] = None
    ) -> DockerResult:
        """
        Build a Docker image.
        
        Args:
            context_path: Build context path
            tag: Image tag
            dockerfile: Dockerfile name
            build_args: Build arguments
            no_cache: Don't use cache
            timeout: Build timeout
            
        Returns:
            DockerResult with build status
        """
        args = ["build"]
        
        if tag:
            args.extend(["-t", tag])
        if dockerfile:
            args.extend(["-f", dockerfile])
        if no_cache:
            args.append("--no-cache")
        
        if build_args:
            for key, value in build_args.items():
                args.extend(["--build-arg", f"{key}={value}"])
        
        args.append(context_path)
        
        return self._run_command(args, timeout=timeout or 600)
    
    def remove_image(
        self,
        image: str,
        force: bool = False
    ) -> DockerResult:
        """
        Remove a Docker image.
        
        Args:
            image: Image name or ID
            force: Force removal
            
        Returns:
            DockerResult with removal status
        """
        args = ["rmi"]
        if force:
            args.append("-f")
        args.append(image)
        
        return self._run_command(args)
    
    # High-level Operations
    
    def execute_code_in_container(
        self,
        image: str,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None
    ) -> DockerResult:
        """
        Execute code in a container.
        
        Args:
            image: Docker image to use
            code: Code to execute
            language: Programming language
            timeout: Execution timeout
            
        Returns:
            DockerResult with execution output
        """
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'.{language}') as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Map language to execution command
            lang_commands = {
                'python': f'python {os.path.basename(temp_path)}',
                'javascript': f'node {os.path.basename(temp_path)}',
                'ruby': f'ruby {os.path.basename(temp_path)}',
                'bash': f'bash {os.path.basename(temp_path)}',
            }
            
            command = lang_commands.get(language, f'{language} {os.path.basename(temp_path)}')
            
            # Run container with code mounted
            return self.run_container(
                image=image,
                command=command,
                volumes={temp_path: f'/tmp/{os.path.basename(temp_path)}'},
                remove=True,
                timeout=timeout
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.debug(f"Failed to clean up temp file: {e}")


# Convenience functions

def get_docker_kit(docker_path: str = "docker", timeout: int = 300) -> DockerKit:
    """
    Get a DockerKit instance.
    
    Args:
        docker_path: Path to docker executable
        timeout: Default timeout for operations
        
    Returns:
        DockerKit instance
    """
    return DockerKit(docker_path=docker_path, timeout=timeout)


__all__ = [
    'DockerKit',
    'DockerResult',
    'get_docker_kit',
]
