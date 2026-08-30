"""Provider CLI isolation backends and supervisor-visible log collection.

Implementation already uses git worktrees.  The default CLI boundary is that
worktree plus Grok's native sandbox when available.  Docker and Kubernetes
are opt-in container runtimes; when they wrap grok/codex/claude/gemini they
must still publish CLI stdout/stderr onto the supervisor host so census and
error recovery are not blind.

Kubernetes clustering compatibility means: never open a nested Docker socket
inside a pod, write CLI logs onto a shared emptyDir/hostPath volume, and
collect the same logs with ``kubectl logs`` (pod or label selector) so a
clustered supervisor is not blind to in-container errors.
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
IMPLEMENTATION_TASK_ID_ENV: str = "IPFS_ACCELERATE_AGENT_TASK_ID"
IMPLEMENTATION_ATTEMPT_ENV: str = "IPFS_ACCELERATE_AGENT_TASK_ATTEMPT"
KUBERNETES_SERVICE_HOST_ENV: str = "KUBERNETES_SERVICE_HOST"
KUBERNETES_POD_NAME_ENV: str = "HOSTNAME"
KUBERNETES_NAMESPACE_ENV: str = "KUBERNETES_NAMESPACE"
KUBERNETES_LOG_VOLUME_NAME: str = "agent-supervisor-provider-cli-logs"
KUBERNETES_LOG_VOLUME_MOUNT: str = "/var/log/agent-supervisor/provider-cli"
KUBERNETES_APP_LABEL: str = "agent-supervisor-provider"
KUBERNETES_LABEL_PREFIX: str = "agent-supervisor.ipfs-accelerate"
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
_SAFE_IDENTITY_RE = re.compile(r"[^A-Za-z0-9._-]+")


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
    if require_container_boundary and requested == PROVIDER_ISOLATION_DOCKER:
        if docker_available:
            return PROVIDER_ISOLATION_DOCKER
        raise ValueError(
            "Docker provider isolation was requested but the pinned "
            "local Docker runtime is unavailable"
        )
    return PROVIDER_ISOLATION_WORKTREE


def _safe_identity_token(value: str, *, limit: int = 63) -> str:
    token = _SAFE_IDENTITY_RE.sub("_", str(value or "").strip()).strip("._-")
    return token[:limit] or "unknown"


def kubernetes_cluster_log_spec(
    *,
    provider: str = "",
    task_id: str = "",
    attempt: str = "",
) -> dict[str, Any]:
    """Return the clustered log-volume and label convention.

    Supervisors running as Kubernetes Jobs/Deployments mount an emptyDir (or
    hostPath) at ``KUBERNETES_LOG_VOLUME_MOUNT``.  The same labels let
    ``kubectl logs -l`` collect grok/codex/claude/gemini output without a
    nested Docker socket.
    """

    identity = kubernetes_log_identity()
    provider_id = _normalize_provider_name(provider)
    labels = {
        "app.kubernetes.io/name": KUBERNETES_APP_LABEL,
        "app.kubernetes.io/part-of": "agent-supervisor",
        f"{KUBERNETES_LABEL_PREFIX}/provider": _safe_identity_token(provider_id),
    }
    if task_id:
        labels[f"{KUBERNETES_LABEL_PREFIX}/task"] = _safe_identity_token(task_id)
    if attempt:
        labels[f"{KUBERNETES_LABEL_PREFIX}/attempt"] = _safe_identity_token(
            str(attempt)
        )
    mount = Path(KUBERNETES_LOG_VOLUME_MOUNT)
    log_dir = ""
    if kubernetes_in_cluster() and mount.is_dir():
        log_dir = str(mount)
    selector = ",".join(f"{key}={value}" for key, value in sorted(labels.items()))
    return {
        "in_cluster": kubernetes_in_cluster(),
        "identity": identity,
        "log_volume": {
            "name": KUBERNETES_LOG_VOLUME_NAME,
            "mount_path": KUBERNETES_LOG_VOLUME_MOUNT,
            "empty_dir": {"medium": ""},
        },
        "labels": labels,
        "label_selector": selector,
        "log_dir": log_dir,
    }


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
    cluster = kubernetes_cluster_log_spec()
    cluster_dir = str(cluster.get("log_dir") or "").strip()
    if cluster_dir:
        path = Path(cluster_dir)
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


def _normalize_provider_name(provider: str) -> str:
    provider_id = str(provider or "grok").strip().casefold() or "grok"
    if provider_id in {"claude_code", "claude-cli", "anthropic"}:
        return "claude-code"
    if provider_id in {"gemini_cli", "gemini-cli"}:
        return "gemini"
    if provider_id in {"codex_cli", "codex-cli"}:
        return "codex"
    if provider_id in {"grok_cli", "grok-cli", "xai_cli"}:
        return "grok"
    if provider_id not in PROVIDER_CLI_NAMES:
        return "grok"
    return provider_id


def _attempt_identity(identity: Mapping[str, Any]) -> dict[str, str | int | bool]:
    merged: dict[str, str | int | bool] = {}
    task_id = str(
        identity.get("task_id")
        or os.environ.get(IMPLEMENTATION_TASK_ID_ENV, "")
        or ""
    ).strip()
    attempt = str(
        identity.get("attempt")
        or identity.get("attempt_id")
        or os.environ.get(IMPLEMENTATION_ATTEMPT_ENV, "")
        or ""
    ).strip()
    if task_id:
        merged["task_id"] = task_id
    if attempt:
        merged["attempt"] = attempt
        merged["attempt_id"] = attempt
    for key, value in dict(identity).items():
        if isinstance(value, (str, int, bool)):
            merged[str(key)] = value
    return merged


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

    provider_id = _normalize_provider_name(provider)
    backend_id = str(backend or PROVIDER_ISOLATION_WORKTREE).strip().casefold()
    directory = provider_cli_log_dir(log_dir)
    bound_identity = _attempt_identity(identity)
    stem = str(
        bound_identity.get("attempt_id")
        or bound_identity.get("attempt")
        or bound_identity.get("container_id")
        or bound_identity.get("pod_name")
        or bound_identity.get("task_id")
        or provider_id
    )
    stem = _safe_identity_token(stem, limit=128)
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
            collected = str(completed.stdout or "") + str(completed.stderr or "")
            if collected.strip():
                output = collected
            if int(getattr(completed, "returncode", 0) or 0) != 0 and not collected:
                log_error = f"log command exited {completed.returncode}"
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
        "identity": bound_identity,
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


def kubectl_logs_selector_command(
    *,
    kubectl_bin: str = "kubectl",
    namespace: str = "",
    selector: str = "",
    container: str = "",
) -> list[str]:
    identity = kubernetes_log_identity()
    label = str(selector or "").strip()
    if not label:
        raise ValueError("kubernetes selector log collection requires a label")
    ns = str(namespace or identity.get("namespace") or "default").strip()
    command = [
        str(kubectl_bin or "kubectl"),
        "--namespace",
        ns,
        "logs",
        "-l",
        label,
        "--prefix",
        "--timestamps",
    ]
    if container:
        command.extend(["-c", str(container)])
    return command


def publish_provider_cli_logs(
    *,
    backend: str,
    provider: str,
    identity: Mapping[str, Any] | None = None,
    returncode: int | None = None,
    captured_output: str = "",
    docker_bin: str = "",
    docker_host: str = "",
    docker_config: str = "",
    container_id: str = "",
    kubectl_bin: str = "kubectl",
    pod_name: str = "",
    namespace: str = "",
    container: str = "",
    log_dir: Path | str | None = None,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Publish grok/codex/claude/gemini CLI output for any isolation backend."""

    bound_identity = _attempt_identity(identity or {})
    backend_id = str(backend or PROVIDER_ISOLATION_WORKTREE).strip().casefold()
    provider_id = _normalize_provider_name(provider)
    log_command: list[str] | None = None
    if backend_id == PROVIDER_ISOLATION_DOCKER and docker_bin and container_id:
        log_command = docker_logs_command(
            docker_bin=docker_bin,
            docker_host=docker_host,
            docker_config=docker_config,
            container_id=container_id,
        )
    elif backend_id == PROVIDER_ISOLATION_KUBERNETES or kubernetes_in_cluster():
        spec = kubernetes_cluster_log_spec(
            provider=provider_id,
            task_id=str(bound_identity.get("task_id") or ""),
            attempt=str(
                bound_identity.get("attempt")
                or bound_identity.get("attempt_id")
                or ""
            ),
        )
        cluster_identity = spec.get("identity")
        if isinstance(cluster_identity, Mapping):
            for key, value in cluster_identity.items():
                if key not in bound_identity and isinstance(value, (str, int, bool)):
                    bound_identity[str(key)] = value
        pod = str(pod_name or bound_identity.get("pod_name") or "").strip()
        ns = str(namespace or bound_identity.get("namespace") or "").strip()
        try:
            if pod:
                log_command = kubectl_logs_command(
                    kubectl_bin=kubectl_bin,
                    namespace=ns,
                    pod_name=pod,
                    container=container or provider_id,
                )
            else:
                log_command = kubectl_logs_selector_command(
                    kubectl_bin=kubectl_bin,
                    namespace=ns,
                    selector=str(spec.get("label_selector") or ""),
                    container=container or provider_id,
                )
        except ValueError:
            log_command = None
    return collect_container_cli_logs(
        backend=backend_id or PROVIDER_ISOLATION_WORKTREE,
        provider=provider_id,
        identity=bound_identity,
        log_dir=log_dir,
        returncode=returncode,
        captured_output=captured_output,
        log_command=log_command,
        runner=runner,
    )


def load_provider_cli_receipts(
    log_dir: Path | str | None = None,
    *,
    task_id: str = "",
    attempt: str = "",
    limit: int = 16,
) -> list[dict[str, Any]]:
    """Load supervisor-visible CLI receipts from the log directory."""

    directory = Path(log_dir) if log_dir is not None else provider_cli_log_dir()
    if not directory.is_dir():
        return []
    wanted_task = str(task_id or "").strip().casefold()
    wanted_attempt = str(attempt or "").strip()
    receipts: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.cli-receipt.json"))[-64:]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema") != CLI_LOG_COLLECTION_SCHEMA:
            continue
        identity = payload.get("identity")
        identity = identity if isinstance(identity, Mapping) else {}
        receipt_task = str(identity.get("task_id") or "").strip().casefold()
        receipt_attempt = str(
            identity.get("attempt") or identity.get("attempt_id") or ""
        ).strip()
        if wanted_task and receipt_task and receipt_task != wanted_task:
            if wanted_task not in path.name.casefold():
                continue
        if wanted_attempt and receipt_attempt and receipt_attempt != wanted_attempt:
            if wanted_attempt not in path.name:
                continue
        payload["receipt_path"] = str(path)
        receipts.append(payload)
    return receipts[-max(1, int(limit)) :]


def supervisor_cli_failure_projection(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project CLI receipts into a bounded failure-visible payload."""

    snippets: list[str] = []
    projected: list[dict[str, Any]] = []
    for receipt in receipts:
        errors = [
            str(item)[:512]
            for item in (receipt.get("error_snippets") or ())
            if str(item).strip()
        ]
        snippets.extend(errors)
        projected.append(
            {
                "provider": str(receipt.get("provider") or ""),
                "backend": str(receipt.get("backend") or ""),
                "returncode": receipt.get("returncode"),
                "log_path": str(receipt.get("log_path") or ""),
                "receipt_path": str(receipt.get("receipt_path") or ""),
                "error_count": int(receipt.get("error_count") or 0),
                "error_snippets": errors[:8],
                "collection_error": str(receipt.get("collection_error") or ""),
            }
        )
    return {
        "provider_cli_receipts": projected[-8:],
        "cli_error_snippets": snippets[:16],
        "cli_error_count": len(snippets),
    }


__all__ = (
    "CLI_LOG_COLLECTION_SCHEMA",
    "DEFAULT_PROVIDER_ISOLATION_BACKEND",
    "IMPLEMENTATION_ATTEMPT_ENV",
    "IMPLEMENTATION_TASK_ID_ENV",
    "KUBERNETES_APP_LABEL",
    "KUBERNETES_LABEL_PREFIX",
    "KUBERNETES_LOG_VOLUME_MOUNT",
    "KUBERNETES_LOG_VOLUME_NAME",
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
    "kubernetes_cluster_log_spec",
    "kubernetes_in_cluster",
    "kubernetes_log_identity",
    "kubernetes_runtime_available",
    "kubectl_logs_command",
    "kubectl_logs_selector_command",
    "load_provider_cli_receipts",
    "provider_cli_log_dir",
    "publish_provider_cli_logs",
    "requested_provider_isolation_backend",
    "select_provider_isolation_backend",
    "supervisor_cli_failure_projection",
)
