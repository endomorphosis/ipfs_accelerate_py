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


_DOCKER_DAEMON_CACHE: Dict[str, Any] = {"ts": 0.0, "ok": False}


def docker_daemon_available(docker_command: str = "docker", timeout_s: float = 3.0, cache_s: float = 5.0) -> bool:
    """Return True if the Docker daemon appears reachable.

    This is intentionally lightweight and safe to call frequently.
    """

    now = time.time()
    ts = float(_DOCKER_DAEMON_CACHE.get("ts") or 0.0)
    if cache_s and now - ts < float(cache_s):
        return bool(_DOCKER_DAEMON_CACHE.get("ok"))

    ok = False
    try:
        proc = subprocess.run(
            [docker_command, "info"],
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
        ok = proc.returncode == 0
    except Exception:
        ok = False

    _DOCKER_DAEMON_CACHE["ts"] = now
    _DOCKER_DAEMON_CACHE["ok"] = ok
    return bool(ok)


@dataclass
class DockerExecutionConfig:
    """Configuration for Docker execution"""
    # Container settings
    image: str
    command: Optional[List[str]] = None
    entrypoint: Optional[List[str]] = None
    working_dir: Optional[str] = None

    # GPU settings
    # Common value: "all" (equivalent to docker run --gpus all)
    gpus: Optional[str] = None
    
    # Resource limits
    memory_limit: Optional[str] = "2g"  # e.g., "512m", "2g"
    cpu_limit: Optional[float] = None  # e.g., 1.0, 2.5
    timeout: Optional[int] = 300  # seconds
    
    # Environment and volumes
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)  # host_path: container_path
    model_artifact_cid: Optional[str] = None
    model_artifact_mount_path: Optional[str] = "/workspace/model_artifact"
    
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
    output_cid: Optional[str] = None
    provenance_cid: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DockerExecutor:
    """
    Core Docker executor for running containers
    
    This class provides the foundational functionality for executing
    code in Docker containers, which is then exposed through MCP tools.
    """
    
    def __init__(
        self,
        docker_command: str = "docker",
        *,
        storage: Any | None = None,
        datasets_manager: Any | None = None,
        provenance_logger: Any | None = None,
        persist_results: bool = True,
    ):
        """
        Initialize Docker executor
        
        Args:
            docker_command: Path to docker executable (default: "docker")
        """
        self.docker_command = docker_command
        self._artifact_storage = storage
        self._datasets_manager = datasets_manager
        self._provenance_logger = provenance_logger
        self._persist_results = bool(persist_results)
        self._verify_docker_available()
        logger.info("DockerExecutor initialized")

    def _record_execution_artifacts(
        self,
        *,
        config: DockerExecutionConfig,
        result: DockerExecutionResult,
        model_artifact_metadata: Optional[Dict[str, Any]] = None,
    ) -> DockerExecutionResult:
        if not self._persist_results:
            return result

        artifact_payload = {
            "image": config.image,
            "command": list(config.command or []),
            "entrypoint": list(config.entrypoint or []),
            "container_id": result.container_id,
            "execution_time": result.execution_time,
            "exit_code": result.exit_code,
            "success": result.success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error_message": result.error_message,
            "model_artifact": dict(model_artifact_metadata or {}),
        }

        output_cid = result.output_cid
        provenance_cid = result.provenance_cid

        if self._artifact_storage is not None:
            try:
                output_payload = json.dumps(artifact_payload, sort_keys=True, indent=2)
                output_cid = self._artifact_storage.store(
                    output_payload,
                    filename=f"docker-execution-{int(time.time() * 1000)}.json",
                    pin=False,
                )
            except Exception as exc:
                logger.debug("Failed to persist Docker execution output: %s", exc)

        if self._datasets_manager is not None:
            try:
                self._datasets_manager.log_event(
                    "container_execution_completed" if result.success else "container_execution_failed",
                    {
                        **artifact_payload,
                        "output_cid": output_cid,
                    },
                    level="INFO" if result.success else "ERROR",
                    category="PERFORMANCE",
                )
            except Exception as exc:
                logger.debug("Failed to log Docker execution event: %s", exc)

        if self._provenance_logger is not None:
            try:
                provenance_cid = self._provenance_logger.log_transformation(
                    "docker_execution",
                    artifact_payload,
                    output_cid=output_cid,
                )
            except Exception as exc:
                logger.debug("Failed to record Docker execution provenance: %s", exc)

        result.output_cid = output_cid
        result.provenance_cid = provenance_cid
        result.metadata = {**artifact_payload, "output_cid": output_cid, "provenance_cid": provenance_cid}
        return result

    def _materialize_model_artifact(self, config: DockerExecutionConfig) -> tuple[DockerExecutionConfig, Dict[str, Any]]:
        metadata: Dict[str, Any] = {
            "requested": bool(config.model_artifact_cid),
            "cid": config.model_artifact_cid,
            "retrieved": False,
            "mount_path": config.model_artifact_mount_path,
            "host_path": None,
            "error": None,
        }

        artifact_cid = str(config.model_artifact_cid or "").strip()
        if not artifact_cid:
            return config, metadata

        if self._artifact_storage is None:
            metadata["error"] = "artifact storage unavailable"
            return config, metadata

        try:
            artifact = self._artifact_storage.retrieve(artifact_cid)
            if artifact is None:
                metadata["error"] = "artifact not found"
                return config, metadata

            if isinstance(artifact, (bytes, bytearray)):
                suffix = Path(config.model_artifact_mount_path or "/workspace/model_artifact").name or "model_artifact"
                fd, temp_path = tempfile.mkstemp(prefix="ipfs_model_artifact_", suffix=f"_{suffix}")
                os.close(fd)
                with open(temp_path, "wb") as handle:
                    handle.write(bytes(artifact))
                host_path = temp_path
            else:
                host_path = str(artifact)

            mount_path = str(config.model_artifact_mount_path or "/workspace/model_artifact")
            updated_volumes = dict(config.volumes)
            updated_volumes[host_path] = mount_path
            updated_config = DockerExecutionConfig(
                image=config.image,
                command=list(config.command or []) or None,
                entrypoint=list(config.entrypoint or []) or None,
                working_dir=config.working_dir,
                gpus=config.gpus,
                memory_limit=config.memory_limit,
                cpu_limit=config.cpu_limit,
                timeout=config.timeout,
                environment=dict(config.environment),
                volumes=updated_volumes,
                model_artifact_cid=config.model_artifact_cid,
                model_artifact_mount_path=config.model_artifact_mount_path,
                network_mode=config.network_mode,
                read_only=config.read_only,
                no_new_privileges=config.no_new_privileges,
                user=config.user,
                capture_output=config.capture_output,
                stream_output=config.stream_output,
            )
            metadata["retrieved"] = True
            metadata["host_path"] = host_path
            return updated_config, metadata
        except Exception as exc:
            metadata["error"] = str(exc)
            return config, metadata
    
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
        model_artifact_metadata: Dict[str, Any] = {}
        
        try:
            config, model_artifact_metadata = self._materialize_model_artifact(config)
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
            
            result = DockerExecutionResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                container_id=container_id,
                execution_time=execution_time
            )
            return self._record_execution_artifacts(
                config=config,
                result=result,
                model_artifact_metadata=model_artifact_metadata,
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            result = DockerExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Execution timed out",
                execution_time=execution_time,
                error_message=f"Container execution exceeded timeout of {config.timeout}s"
            )
            return self._record_execution_artifacts(
                config=config,
                result=result,
                model_artifact_metadata=model_artifact_metadata,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Docker execution failed: {e}")
            result = DockerExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                error_message=str(e)
            )
            return self._record_execution_artifacts(
                config=config,
                result=result,
                model_artifact_metadata=model_artifact_metadata,
            )
    
    def _build_docker_command(self, config: DockerExecutionConfig) -> List[str]:
        """Build docker run command from configuration"""
        cmd = [self.docker_command, "run", "--rm"]

        # GPU settings
        if config.gpus:
            cmd.extend(["--gpus", str(config.gpus)])
        
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
                result = DockerExecutionResult(
                    success=False,
                    exit_code=clone_result.returncode,
                    stdout=clone_result.stdout,
                    stderr=clone_result.stderr,
                    error_message="Failed to clone GitHub repository"
                )
                return self._record_execution_artifacts(config=execution_config, result=result)
            
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
                result = DockerExecutionResult(
                    success=False,
                    exit_code=build_result.returncode,
                    stdout=build_result.stdout,
                    stderr=build_result.stderr,
                    error_message="Failed to build Docker image"
                )
                return self._record_execution_artifacts(config=execution_config, result=result)
            
            # Execute the built image
            execution_config.image = image_tag
            result = self.execute_container(execution_config)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to build and execute GitHub repo: {e}")
            result = DockerExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                error_message=str(e)
            )
            return self._record_execution_artifacts(config=execution_config, result=result)
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
    storage: Any | None = None,
    datasets_manager: Any | None = None,
    provenance_logger: Any | None = None,
    persist_results: bool = True,
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
    
    executor = DockerExecutor(
        storage=storage,
        datasets_manager=datasets_manager,
        provenance_logger=provenance_logger,
        persist_results=persist_results,
    )
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
    storage: Any | None = None,
    datasets_manager: Any | None = None,
    provenance_logger: Any | None = None,
    persist_results: bool = True,
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
    
    executor = DockerExecutor(
        storage=storage,
        datasets_manager=datasets_manager,
        provenance_logger=provenance_logger,
        persist_results=persist_results,
    )
    return executor.build_and_execute_github_repo(github_config, execution_config)
