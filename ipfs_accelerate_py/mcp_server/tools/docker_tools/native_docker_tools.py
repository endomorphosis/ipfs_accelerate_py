"""Native docker-tools category implementations for unified mcp_server.

Exposes Docker execution capabilities from ``ipfs_accelerate_py.docker_executor``
as unified MCP and MCP++ tools following the established native-tools pattern.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_docker_tools_api() -> Dict[str, Any]:
    """Resolve source docker-tools APIs with compatibility fallback."""
    try:
        from ipfs_accelerate_py.docker_executor import (  # type: ignore
            DockerExecutor,
            execute_docker_hub_container,
            build_and_execute_from_github,
        )

        return {
            "DockerExecutor": DockerExecutor,
            "execute_docker_hub_container": execute_docker_hub_container,
            "build_and_execute_from_github": build_and_execute_from_github,
        }
    except Exception:
        logger.warning(
            "Source docker_executor import unavailable, using fallback docker-tools functions"
        )
        return {}


_API = _load_docker_tools_api()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    result.update(extra)
    return result


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalise delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
        elif "status" not in envelope:
            envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _execution_result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert a DockerExecutionResult (or similar) to a plain dict."""
    if isinstance(result, dict):
        return _normalize_payload(result)
    envelope: Dict[str, Any] = {
        "success": bool(getattr(result, "success", False)),
        "exit_code": int(getattr(result, "exit_code", -1)),
        "stdout": str(getattr(result, "stdout", "")),
        "stderr": str(getattr(result, "stderr", "")),
        "execution_time": float(getattr(result, "execution_time", 0.0)),
    }
    error_message = getattr(result, "error_message", None)
    if error_message:
        envelope["error_message"] = str(error_message)
    if envelope["success"]:
        envelope["status"] = "success"
    else:
        envelope["status"] = "error"
    return envelope


def _require_string(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return _error_result(f"{field} must be a non-empty string", **{field: value})
    return None


def _optional_string(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        return _error_result(
            f"{field} must be a non-empty string when provided", **{field: value}
        )
    return None


def _validate_bool(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, bool):
        return _error_result(f"{field} must be a boolean", **{field: value})
    return None


def _validate_positive_int(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, int) or value < 1:
        return _error_result(
            f"{field} must be an integer >= 1", **{field: value}
        )
    return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def execute_docker_container(
    image: str,
    command: Optional[str] = None,
    entrypoint: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    memory_limit: str = "2g",
    cpu_limit: Optional[float] = None,
    timeout: int = 300,
    network_mode: str = "none",
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a Docker container from Docker Hub.

    Runs a pre-built Docker container with the specified configuration.
    This enables isolated, resource-bounded code execution.
    """
    validation = _require_string(image, "image")
    if validation is not None:
        return validation
    if command is not None:
        validation = _require_string(command, "command")
        if validation is not None:
            return validation
    if entrypoint is not None:
        validation = _require_string(entrypoint, "entrypoint")
        if validation is not None:
            return validation
    if environment is not None and not isinstance(environment, dict):
        return _error_result("environment must be an object when provided")
    if cpu_limit is not None and not isinstance(cpu_limit, (int, float)):
        return _error_result("cpu_limit must be a number when provided")
    validation = _validate_positive_int(timeout, "timeout")
    if validation is not None:
        return validation

    if not _API:
        return _error_result(
            "Docker executor module not available. Please ensure Docker is installed.",
            image=image.strip(),
        )

    clean_image = image.strip()
    command_list = command.split() if command else None
    entrypoint_list = entrypoint.split() if entrypoint else None

    try:
        result = _API["execute_docker_hub_container"](
            image=clean_image,
            command=command_list,
            entrypoint=entrypoint_list,
            environment=dict(environment) if environment else {},
            memory_limit=str(memory_limit or "2g"),
            cpu_limit=float(cpu_limit) if cpu_limit is not None else None,
            timeout=int(timeout),
            network_mode=str(network_mode or "none"),
            working_dir=working_dir,
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return _error_result(str(exc), image=clean_image)

    envelope = _execution_result_to_dict(result)
    envelope.setdefault("image", clean_image)
    return envelope


async def build_and_execute_github_repo(
    repo_url: str,
    branch: str = "main",
    dockerfile_path: str = "Dockerfile",
    command: Optional[str] = None,
    entrypoint: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    build_args: Optional[Dict[str, str]] = None,
    memory_limit: str = "2g",
    timeout: int = 600,
) -> Dict[str, Any]:
    """Clone a GitHub repository, build its Docker image, and execute it.

    Supports any GitHub repository that contains a Dockerfile.
    Build logs are included in the response.
    """
    validation = _require_string(repo_url, "repo_url")
    if validation is not None:
        return validation
    if not (
        repo_url.strip().startswith("https://github.com/")
        or repo_url.strip().startswith("http://github.com/")
    ):
        return _error_result(
            "repo_url must start with https://github.com/ or http://github.com/",
            repo_url=repo_url.strip(),
        )
    validation = _validate_positive_int(timeout, "timeout")
    if validation is not None:
        return validation

    if not _API:
        return _error_result(
            "Docker executor module not available.",
            repo_url=repo_url.strip(),
        )

    clean_repo_url = repo_url.strip()
    command_list = command.split() if command else None
    entrypoint_list = entrypoint.split() if entrypoint else None

    try:
        result = _API["build_and_execute_from_github"](
            repo_url=clean_repo_url,
            branch=str(branch or "main"),
            dockerfile_path=str(dockerfile_path or "Dockerfile"),
            command=command_list,
            entrypoint=entrypoint_list,
            environment=dict(environment) if environment else {},
            build_args=dict(build_args) if build_args else {},
            memory_limit=str(memory_limit or "2g"),
            timeout=int(timeout),
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return _error_result(str(exc), repo_url=clean_repo_url)

    envelope = _execution_result_to_dict(result)
    envelope.setdefault("repo_url", clean_repo_url)
    return envelope


async def execute_with_payload(
    image: str,
    payload: str,
    payload_path: str = "/tmp/payload",
    entrypoint: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    memory_limit: str = "2g",
    timeout: int = 300,
) -> Dict[str, Any]:
    """Execute a Docker container with an inline payload file.

    The payload string is written to a temporary file on the host and
    bind-mounted into the container at ``payload_path``.  This allows running
    arbitrary code (Python, shell scripts, etc.) in an isolated container
    without pre-baking it into the image.
    """
    validation = _require_string(image, "image")
    if validation is not None:
        return validation
    validation = _require_string(payload, "payload")
    if validation is not None:
        return validation
    validation = _validate_positive_int(timeout, "timeout")
    if validation is not None:
        return validation

    if not _API:
        return _error_result(
            "Docker executor module not available.",
            image=image.strip(),
        )

    import tempfile
    import os

    clean_image = image.strip()
    entrypoint_list = entrypoint.split() if entrypoint else None
    temp_file: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".payload") as fh:
            fh.write(payload)
            temp_file = fh.name

        if payload.startswith("#!"):
            os.chmod(temp_file, 0o755)

        volumes = {temp_file: str(payload_path or "/tmp/payload")}

        result = _API["execute_docker_hub_container"](
            image=clean_image,
            entrypoint=entrypoint_list,
            environment=dict(environment) if environment else {},
            volumes=volumes,
            memory_limit=str(memory_limit or "2g"),
            timeout=int(timeout),
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return _error_result(str(exc), image=clean_image)
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as cleanup_exc:
                logger.warning("Failed to clean up temp payload file %s: %s", temp_file, cleanup_exc)

    envelope = _execution_result_to_dict(result)
    envelope.setdefault("image", clean_image)
    return envelope


async def list_running_containers() -> Dict[str, Any]:
    """List currently running Docker containers managed by this host."""
    if not _API:
        return {
            "status": "success",
            "success": True,
            "containers": [],
            "count": 0,
            "available": False,
            "fallback": True,
        }

    try:
        executor = _API["DockerExecutor"]()
        containers = executor.list_running_containers()
        if hasattr(containers, "__await__"):
            containers = await containers
        containers = list(containers) if containers is not None else []
        return {
            "status": "success",
            "success": True,
            "containers": containers,
            "count": len(containers),
            "available": True,
        }
    except Exception as exc:
        return _error_result(str(exc))


async def stop_container(
    container_id: str,
    timeout: int = 10,
) -> Dict[str, Any]:
    """Stop a running Docker container by ID or name."""
    validation = _require_string(container_id, "container_id")
    if validation is not None:
        return validation
    validation = _validate_positive_int(timeout, "timeout")
    if validation is not None:
        return validation

    if not _API:
        return _error_result(
            "Docker executor module not available.",
            container_id=container_id.strip(),
        )

    clean_id = container_id.strip()
    try:
        executor = _API["DockerExecutor"]()
        success = executor.stop_container(clean_id, int(timeout))
        if hasattr(success, "__await__"):
            success = await success
        return {
            "status": "success" if success else "error",
            "success": bool(success),
            "container_id": clean_id,
            "message": (
                "Container stopped successfully" if success else "Failed to stop container"
            ),
        }
    except Exception as exc:
        return _error_result(str(exc), container_id=clean_id)


async def pull_docker_image(image: str) -> Dict[str, Any]:
    """Pull a Docker image from Docker Hub to prepare for execution."""
    validation = _require_string(image, "image")
    if validation is not None:
        return validation

    if not _API:
        return _error_result(
            "Docker executor module not available.",
            image=image.strip(),
        )

    clean_image = image.strip()
    try:
        executor = _API["DockerExecutor"]()
        success = executor.pull_image(clean_image)
        if hasattr(success, "__await__"):
            success = await success
        return {
            "status": "success" if success else "error",
            "success": bool(success),
            "image": clean_image,
            "message": (
                f"Image {clean_image} pulled successfully"
                if success
                else f"Failed to pull image {clean_image}"
            ),
        }
    except Exception as exc:
        return _error_result(str(exc), image=clean_image)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_native_docker_tools(manager: Any) -> None:
    """Register native docker-tools category tools in unified manager."""
    registrations = [
        {
            "name": "execute_docker_container",
            "func": execute_docker_container,
            "description": (
                "Execute a pre-built Docker container from Docker Hub with configurable "
                "resource limits, environment variables, and network isolation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Docker image name (e.g. 'python:3.9', 'ubuntu:22.04')",
                    },
                    "command": {
                        "type": ["string", "null"],
                        "description": "Command to run (space-separated string)",
                    },
                    "entrypoint": {
                        "type": ["string", "null"],
                        "description": "Custom entrypoint override (space-separated string)",
                    },
                    "environment": {
                        "type": ["object", "null"],
                        "additionalProperties": {"type": "string"},
                        "description": "Environment variables as key-value pairs",
                    },
                    "memory_limit": {
                        "type": "string",
                        "default": "2g",
                        "description": "Memory limit (e.g. '512m', '2g')",
                    },
                    "cpu_limit": {
                        "type": ["number", "null"],
                        "description": "CPU limit as number of cores",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 300,
                        "minimum": 1,
                        "description": "Execution timeout in seconds",
                    },
                    "network_mode": {
                        "type": "string",
                        "default": "none",
                        "enum": ["none", "bridge", "host"],
                        "description": "Container network mode",
                    },
                    "working_dir": {
                        "type": ["string", "null"],
                        "description": "Working directory inside the container",
                    },
                },
                "required": ["image"],
            },
        },
        {
            "name": "build_and_execute_github_repo",
            "func": build_and_execute_github_repo,
            "description": (
                "Clone a GitHub repository, build its Dockerfile, and execute the resulting "
                "image.  Build logs are captured in the response."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "minLength": 1,
                        "description": "GitHub repository URL (https://github.com/owner/repo)",
                    },
                    "branch": {
                        "type": "string",
                        "default": "main",
                        "description": "Git branch to clone",
                    },
                    "dockerfile_path": {
                        "type": "string",
                        "default": "Dockerfile",
                        "description": "Path to Dockerfile relative to repository root",
                    },
                    "command": {
                        "type": ["string", "null"],
                        "description": "Command to run in the built container",
                    },
                    "entrypoint": {
                        "type": ["string", "null"],
                        "description": "Custom entrypoint for the container",
                    },
                    "environment": {
                        "type": ["object", "null"],
                        "additionalProperties": {"type": "string"},
                    },
                    "build_args": {
                        "type": ["object", "null"],
                        "additionalProperties": {"type": "string"},
                        "description": "Docker build arguments (ARG values)",
                    },
                    "memory_limit": {
                        "type": "string",
                        "default": "2g",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 600,
                        "minimum": 1,
                        "description": "Total timeout for build + execution in seconds",
                    },
                },
                "required": ["repo_url"],
            },
        },
        {
            "name": "execute_with_payload",
            "func": execute_with_payload,
            "description": (
                "Execute a Docker container with an inline payload (code, script, config). "
                "The payload is written to a temporary file and bind-mounted into the container."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Docker image name",
                    },
                    "payload": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Content to write to the payload file inside the container",
                    },
                    "payload_path": {
                        "type": "string",
                        "default": "/tmp/payload",
                        "description": "Path where the payload will be available inside the container",
                    },
                    "entrypoint": {
                        "type": ["string", "null"],
                        "description": "Command to execute (may reference payload_path)",
                    },
                    "environment": {
                        "type": ["object", "null"],
                        "additionalProperties": {"type": "string"},
                    },
                    "memory_limit": {
                        "type": "string",
                        "default": "2g",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 300,
                        "minimum": 1,
                    },
                },
                "required": ["image", "payload"],
            },
        },
        {
            "name": "list_running_containers",
            "func": list_running_containers,
            "description": "List currently running Docker containers on this host.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "stop_container",
            "func": stop_container,
            "description": "Stop a running Docker container by ID or name.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Container ID or name to stop",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "description": "Seconds to wait before force-killing the container",
                    },
                },
                "required": ["container_id"],
            },
        },
        {
            "name": "pull_docker_image",
            "func": pull_docker_image,
            "description": "Pull a Docker image from Docker Hub to prepare it for execution.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Docker image name (e.g. 'python:3.9-slim')",
                    },
                },
                "required": ["image"],
            },
        },
    ]

    for registration in registrations:
        manager.register_tool(
            category="docker_tools",
            name=registration["name"],
            func=registration["func"],
            description=registration["description"],
            input_schema=registration["input_schema"],
            runtime="fastapi",
            tags=["native", "mcpp", "docker-tools"],
        )
