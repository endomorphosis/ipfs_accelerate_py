"""Provider CLI isolation backends and supervisor-visible log collection.

Implementation already uses git worktrees.  The default CLI boundary is that
worktree plus Grok's native sandbox when available.  Docker and Kubernetes
are opt-in container runtimes; when they wrap grok/codex/claude/gemini they
must still publish CLI stdout/stderr onto the supervisor host so census and
error recovery are not blind.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROVIDER_ISOLATION_WORKTREE: str = "worktree"
PROVIDER_ISOLATION_GROK_SANDBOX: str = "grok-sandbox"
PROVIDER_ISOLATION_DOCKER: str = "docker"
PROVIDER_ISOLATION_KUBERNETES: str = "kubernetes"
DEFAULT_PROVIDER_ISOLATION_BACKEND: str = PROVIDER_ISOLATION_WORKTREE
PROVIDER_ISOLATION_BACKEND_ENV: str = (
    "IPFS_ACCELERATE_AGENT_PROVIDER_ISOLATION_BACKEND"
)
PROVIDER_CLI_LOG_DIR_ENV: str = "IPFS_ACCELERATE_AGENT_PROVIDER_CLI_LOG_DIR"
KUBERNETES_SERVICE_HOST_ENV: str = "KUBERNETES_SERVICE_HOST"
KUBERNETES_POD_NAME_ENV: str = "HOSTNAME"
KUBERNETES_NAMESPACE_ENV: str = "KUBERNETES_NAMESPACE"
CLI_LOG_COLLECTION_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/provider-cli-log-collection@1"
)
PROVIDER_CLI_NAMES: frozenset[str] = frozenset(
    {
        "grok",
        "codex",
        "claude",
        "claude-code",
        "gemini",
        "cursor",
        "amp",
    }
)
_ERROR_LINE_RE = re.compile(
    r"(error|exception|traceback|fatal|quota|denied|unauthorized|failed)",
    re.IGNORECASE,
)
_MAX_LOG_BYTES = 1_048_576
_MAX_ERROR_SNIPPETS = 32


def requested_provider_isolation_backend() -> str:
    """Return the operator-requested isolation backend, defaulting to worktree."""

    raw = str(os.environ.get(PROVIDER_ISOLATION_BACKEND_ENV, "") or "").strip()
    if not raw:
        return DEFAULT_PROVIDER_ISOLATION_BACKEND
    lowered = raw.casefold()
    if lowered in {
        PROVIDER_ISOLATION_WORKTREE,
        PROVIDER_ISOLATION_GROK_SANDBOX,
        PROVIDER_ISOLATION_DOCKER,
        PROVIDER_ISOLATION_KUBERNETES,
    }:
        return lowered
    return DEFAULT_PROVIDER_ISOLATION_BACKEND


def kubernetes_runtime_available() -> bool:
    """True when this process is in-cluster or kubectl can talk to a cluster."""

    if str(os.environ.get(KUBERNETES_SERVICE_HOST_ENV, "") or "").strip():
        return True
    kubectl = shutil.which("kubectl")
    return bool(kubectl)


def kubernetes_in_cluster() -> bool:
    return bool(str(os.environ.get(KUBERNETES_SERVICE_HOST_ENV, "") or "").strip())


def select_provider_isolation_backend(
    *,
    docker_available: bool,
    sandbox_available: bool,
    kubernetes_available: bool = False,
    require_container_boundary: bool = False,
) -> str:
    """Select CLI isolation. Worktree/sandbox is default; containers are opt-in.

    Inside a Kubernetes pod the pod is already the container boundary, so the
    supervisor stays on worktree/sandbox and does not open a nested Docker
    socket.  Quota fallback may still require an explicit docker/kubernetes
    backend.
    """

    requested = requested_provider_isolation_backend()
    if kubernetes_in_cluster() and requested in {
        PROVIDER_ISOLATION_WORKTREE,
        PROVIDER_ISOLATION_GROK_SANDBOX,
        PROVIDER_ISOLATION_KUBERNETES,
        DEFAULT_PROVIDER_ISOLATION_BACKEND,
    }:
        if sandbox_available:
            return PROVIDER_ISOLATION_GROK_SANDBOX
        return PROVIDER_ISOLATION_WORKTREE
    if requested == PROVIDER_ISOLATION_KUBERNETES:
        if kubernetes_available or kubernetes_in_cluster():
            if kubernetes_in_cluster():
                return (
                    PROVIDER_ISOLATION_GROK_SANDBOX
                    if sandbox_available
                    else PROVIDER_ISOLATION_WORKTREE
                )
            return PROVIDER_ISOLATION_KUBERNETES
        if require_container_boundary:
            raise ValueError(
                "Kubernetes provider isolation was requested but kubectl/"
                "in-cluster config is unavailable"
            )
    if requested == PROVIDER_ISOLATION_DOCKER:
        if docker_available:
            return PROVIDER_ISOLATION_DOCKER
        if require_container_boundary:
            raise ValueError(
                "Docker provider isolation was requested but the pinned "
                "local Docker runtime is unavailable"
            )
    if requested == PROVIDER_ISOLATION_GROK_SANDBOX and sandbox_available:
        return PROVIDER_ISOLATION_GROK_SANDBOX
    if sandbox_available:
        return PROVIDER_ISOLATION_GROK_SANDBOX
    if require_container_boundary:
        if docker_available:
            return PROVIDER_ISOLATION_DOCKER
        raise ValueError(
            "Provider isolation requires a worktree sandbox or an explicit "
            "container backend"
        )
    return PROVIDER_ISOLATION_WORKTREE


def provider_cli_log_dir(explicit: Path | str | None = None) -> Path:
    """Supervisor-visible directory for collected CLI logs."""

    if explicit is not None:
        path = Path(explicit)
        path.mkdir(parents=True, exist_ok=True)
        return path
    raw = str(os.environ.get(PROVIDER_CLI_LOG_DIR_ENV, "") or "").strip()
    if raw:
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = Path.cwd() / ".ipfs-accelerate" / "provider-cli-logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_cli_error_snippets(text: str, *, limit: int = _MAX_ERROR_SNIPPETS) -> list[str]:
    """Return bounded error-like lines from CLI output."""

    snippets: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not _ERROR_LINE_RE.search(line):
            continue
        snippets.append(line[:1024])
        if len(snippets) >= limit:
            break
    return snippets


def _bounded_text(value: str) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= _MAX_LOG_BYTES:
        return value
    return encoded[-_MAX_LOG_BYTES:].decode("utf-8", errors="replace")


def collect_container_cli_logs(
    *,
    backend: str,
    provider: str,
    identity: Mapping[str, Any],
    log_dir: Path | str | None = None,
    returncode: int | None = None,
    captured_output: str = "",
    log_command: Sequence[str] | None = None,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Copy container or host CLI output into a supervisor-managed receipt.

    ``log_command`` is the exact docker/kubectl logs argv.  Tests inject
    ``runner``.  Host/worktree collection uses ``captured_output`` only.
    """

    provider_id = str(provider or "grok").strip().casefold() or "grok"
    if provider_id not in PROVIDER_CLI_NAMES:
        provider_id = "grok"
    backend_id = str(backend or PROVIDER_ISOLATION_WORKTREE).strip().casefold()
    directory = provider_cli_log_dir(log_dir)
    stem = str(identity.get("attempt_id") or identity.get("container_id") or provider_id)
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:128] or provider_id
    log_path = directory / f"{provider_id}-{stem}.cli.log"
    receipt_path = directory / f"{provider_id}-{stem}.cli-receipt.json"
    output = captured_output
    log_error = ""
    if log_command:
        try:
            completed = runner(
                list(log_command),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log_error = f"{type(exc).__name__}: log collection failed"
            completed = None
        if completed is not None:
            output = str(completed.stdout or "") + str(completed.stderr or "")
            if int(getattr(completed, "returncode", 0) or 0) != 0 and not output:
                log_error = (
                    f"log command exited {completed.returncode}"
                )
    output = _bounded_text(output)
    log_path.write_text(output, encoding="utf-8")
    snippets = extract_cli_error_snippets(output)
    receipt = {
        "schema": CLI_LOG_COLLECTION_SCHEMA,
        "backend": backend_id,
        "provider": provider_id,
        "returncode": returncode,
        "log_path": str(log_path),
        "error_snippets": snippets,
        "error_count": len(snippets),
        "log_bytes": len(output.encode("utf-8")),
        "collection_error": log_error,
        "identity": {
            str(key): value
            for key, value in dict(identity).items()
            if isinstance(value, (str, int, bool))
        },
        "kubernetes": kubernetes_log_identity()
        if backend_id == PROVIDER_ISOLATION_KUBERNETES or kubernetes_in_cluster()
        else {},
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def kubernetes_log_identity() -> dict[str, str]:
    """Pod identity used when collecting kubectl/in-cluster logs."""

    namespace = str(os.environ.get(KUBERNETES_NAMESPACE_ENV, "") or "").strip()
    if not namespace:
        namespace_file = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
        try:
            namespace = namespace_file.read_text(encoding="utf-8").strip()
        except OSError:
            namespace = "default"
    return {
        "service_host": str(os.environ.get(KUBERNETES_SERVICE_HOST_ENV, "") or ""),
        "pod_name": str(os.environ.get(KUBERNETES_POD_NAME_ENV, "") or ""),
        "namespace": namespace or "default",
    }


def docker_logs_command(
    *,
    docker_bin: str,
    docker_host: str,
    docker_config: str,
    container_id: str,
) -> list[str]:
    container = str(container_id or "").strip()
    if container.startswith("sha256:"):
        container = container.removeprefix("sha256:")
    if not container:
        raise ValueError("docker log collection requires a container id")
    command = [str(docker_bin), f"--host={docker_host}"]
    if docker_config:
        command.extend(["--config", str(docker_config)])
    command.extend(["logs", "--timestamps", container])
    return command


def kubectl_logs_command(
    *,
    kubectl_bin: str = "kubectl",
    namespace: str = "",
    pod_name: str = "",
    container: str = "",
) -> list[str]:
    identity = kubernetes_log_identity()
    pod = str(pod_name or identity.get("pod_name") or "").strip()
    if not pod:
        raise ValueError("kubernetes log collection requires a pod name")
    ns = str(namespace or identity.get("namespace") or "default").strip()
    command = [str(kubectl_bin or "kubectl"), "--namespace", ns, "logs", pod]
    if container:
        command.extend(["-c", str(container)])
    return command


__all__ = (
    "CLI_LOG_COLLECTION_SCHEMA",
    "DEFAULT_PROVIDER_ISOLATION_BACKEND",
    "KUBERNETES_NAMESPACE_ENV",
    "KUBERNETES_POD_NAME_ENV",
    "KUBERNETES_SERVICE_HOST_ENV",
    "PROVIDER_CLI_LOG_DIR_ENV",
    "PROVIDER_CLI_NAMES",
    "PROVIDER_ISOLATION_BACKEND_ENV",
    "PROVIDER_ISOLATION_DOCKER",
    "PROVIDER_ISOLATION_GROK_SANDBOX",
    "PROVIDER_ISOLATION_KUBERNETES",
    "PROVIDER_ISOLATION_WORKTREE",
    "collect_container_cli_logs",
    "docker_logs_command",
    "extract_cli_error_snippets",
    "kubernetes_in_cluster",
    "kubernetes_log_identity",
    "kubernetes_runtime_available",
    "kubectl_logs_command",
    "provider_cli_log_dir",
    "requested_provider_isolation_backend",
    "select_provider_isolation_backend",
)
