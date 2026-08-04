"""Hugging Face Space inference and bucket compatibility helpers.

The voice pipeline keeps endpoint-specific payload construction in its caller.
This module owns the reusable transport concerns:

* Gradio config and endpoint discovery
* queued Space calls and Server-Sent Event result handling
* Gradio file upload and download
* read/write adapters for Hugging Face buckets

The implementation intentionally does not depend on ``gradio_client`` so it can
be used by lightweight supervisor and batch-worker environments.
"""

from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence, TypeVar
from urllib import parse as urllib_parse

if TYPE_CHECKING:
    from requests import Response as _RequestsResponse
    from requests import Session as _RequestsSession
else:
    _RequestsResponse = Any
    _RequestsSession = Any


HeadersFactory = Callable[[], Mapping[str, str]]
ResultT = TypeVar("ResultT")


def _load_requests() -> Any:
    """Load the HTTP client only when a transport operation needs it."""

    import requests

    return requests


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON document without exposing a partial file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        directory_descriptor: int | None = None
        try:
            directory_descriptor = os.open(
                destination.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            os.fsync(directory_descriptor)
        except OSError:
            # Some supported platforms do not permit directory fsync.
            pass
        finally:
            if directory_descriptor is not None:
                os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def normalize_api_name(value: str) -> str:
    """Return a Gradio API name with one leading slash."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw if raw.startswith("/") else f"/{raw}"


def _exception_chain(value: object) -> Iterator[BaseException]:
    """Yield an exception and its explicit/implicit causes without looping."""

    current = value if isinstance(value, BaseException) else None
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _exception_text(value: object) -> str:
    chain = list(_exception_chain(value))
    if not chain:
        return str(value or "").casefold()
    return " | ".join(
        f"{type(error).__name__}: {error}" for error in chain
    ).casefold()


def is_stale_gradio_file_error(value: object) -> bool:
    """Return whether a Space rejected an expired server-local FileData path."""

    text = _exception_text(value)
    has_gradio_path = any(
        marker in text
        for marker in (
            "/tmp/gradio/",
            "\\tmp\\gradio\\",
            "gradio/file",
            "gradio.filedata",
        )
    )
    has_missing_file = any(
        marker in text
        for marker in (
            "filenotfounderror",
            "file not found",
            "no such file or directory",
            "does not exist",
        )
    )
    return has_gradio_path and has_missing_file


def is_hf_space_transport_error(value: object) -> bool:
    """Return whether a failed Space call can be retried as transport I/O."""

    requests = _load_requests()
    transient_request_errors = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ContentDecodingError,
    )
    for error in _exception_chain(value):
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and (
            status_code in {408, 425, 429} or 500 <= status_code <= 599
        ):
            return True
        if isinstance(
            error,
            (TimeoutError, ConnectionError, *transient_request_errors),
        ):
            return True
    text = _exception_text(value)
    return any(
        marker in text
        for marker in (
            "response ended prematurely",
            "remote end closed connection",
            "remote disconnected",
            "connection reset",
            "connection aborted",
            "broken pipe",
            "read timed out",
        )
    )


def is_retryable_hf_space_error(value: object) -> bool:
    """Classify transient Space failures, including expired Gradio uploads."""

    if is_hf_space_transport_error(value) or is_stale_gradio_file_error(value):
        return True
    text = _exception_text(value)
    return any(
        marker in text
        for marker in (
            "queue full",
            "queue_full",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "space queue failed",
            "queue failed",
            "zerogpu worker error",
            "acceleratorerror",
        )
    )


class RefreshableGradioFile:
    """Cache a Gradio FileData upload and refresh it across transient failures.

    Gradio upload paths are server-local leases rather than durable object
    identifiers. A worker or Space restart can therefore invalidate a
    previously successful upload while a long-running batch process is still
    using it.
    """

    def __init__(
        self,
        uploader: Callable[[], Mapping[str, Any]],
        *,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self._uploader = uploader
        self._sleeper = sleeper
        self._value: dict[str, Any] | None = None
        self._lock = threading.RLock()

    def get(self) -> dict[str, Any]:
        """Return the current FileData mapping, uploading it when necessary."""

        with self._lock:
            if self._value is None:
                uploaded = self._uploader()
                if not isinstance(uploaded, Mapping):
                    raise TypeError(
                        "Gradio file uploader must return a mapping"
                    )
                self._value = dict(uploaded)
            return dict(self._value)

    def invalidate(self) -> None:
        """Discard the cached server-local upload reference."""

        with self._lock:
            self._value = None

    def run(
        self,
        operation: Callable[[Mapping[str, Any]], ResultT],
        *,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.0,
        retry_backoff_multiplier: float = 2.0,
        on_retry: Callable[[BaseException, int], None] | None = None,
    ) -> ResultT:
        """Run an operation and re-upload before retrying transient failures."""

        retries = max(0, int(max_retries))
        backoff = max(0.0, float(retry_backoff_seconds))
        multiplier = max(1.0, float(retry_backoff_multiplier))
        for attempt in range(retries + 1):
            try:
                return operation(self.get())
            except Exception as error:
                retryable = is_retryable_hf_space_error(error)
                if retryable:
                    self.invalidate()
                if not retryable or attempt >= retries:
                    raise
                retry_number = attempt + 1
                if on_retry is not None:
                    on_retry(error, retry_number)
                delay = backoff * (multiplier ** attempt)
                if delay > 0:
                    self._sleeper(delay)
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class SpaceRuntimeInfo:
    """Snapshot of a Hugging Face Space runtime."""

    stage: str
    hardware_current: str | None = None
    hardware_requested: str | None = None
    replicas: int = 0
    dev_mode: bool = False
    sleep_timeout: int | None = None
    domains: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EndpointContract:
    """One callable dependency discovered in a Gradio config."""

    fn_index: int
    dependency_id: int
    label: str
    api_name: str
    component_name: str
    input_count: int | None = None


class OutputBackend(ABC):
    """Storage interface used by resumable inference workers."""

    @abstractmethod
    def put_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload or copy a local file."""

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Return whether a remote path exists."""

    @abstractmethod
    def list_files(self, prefix: str) -> list[str]:
        """List files below a remote prefix."""

    def sync_directory(self, local_dir: Path, remote_prefix: str) -> int:
        """Copy every file below ``local_dir`` while preserving its layout."""

        source = Path(local_dir)
        if not source.is_dir():
            return 0
        copied = 0
        for local_path in source.rglob("*"):
            if not local_path.is_file():
                continue
            relative_path = local_path.relative_to(source).as_posix()
            remote_path = "/".join(
                part.strip("/")
                for part in (str(remote_prefix or ""), relative_path)
                if part.strip("/")
            )
            if self.put_file(local_path, remote_path):
                copied += 1
        return copied


class LocalFileSystemBackend(OutputBackend):
    """Filesystem implementation of :class:`OutputBackend`."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)

    def put_file(self, local_path: Path, remote_path: str) -> bool:
        destination = self.base_dir / str(remote_path).lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.write_bytes(Path(local_path).read_bytes())
            return True
        except OSError:
            return False

    def exists(self, remote_path: str) -> bool:
        return (self.base_dir / str(remote_path).lstrip("/")).exists()

    def list_files(self, prefix: str) -> list[str]:
        target = self.base_dir / str(prefix).lstrip("/")
        if target.is_file():
            return [target.relative_to(self.base_dir).as_posix()]
        if not target.is_dir():
            return []
        return sorted(
            path.relative_to(self.base_dir).as_posix()
            for path in target.rglob("*")
            if path.is_file()
        )


class HFBucketBackendError(RuntimeError):
    """Raised when bucket availability cannot be distinguished from absence."""


class HFBucketBackend(OutputBackend):
    """Hugging Face bucket adapter backed by the current ``hf`` CLI.

    The CLI's bucket commands are still evolving.  Centralizing them here keeps
    the voice wrapper independent from command-name changes and gives tests one
    place to verify that cache-existence checks remain read-only.
    """

    def __init__(
        self,
        bucket_uri: str,
        hf_token: str | None = None,
        *,
        cli_executable: str | None = None,
        timeout_seconds: float = 60.0,
    ):
        resolved_uri = str(bucket_uri or "").strip().rstrip("/")
        if not resolved_uri:
            raise ValueError("bucket_uri is required")
        self.bucket_uri = resolved_uri
        self.hf_token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        self.cli_executable = str(
            cli_executable or os.getenv("HF_CLI_BIN") or "hf"
        ).strip() or "hf"
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def _env(self) -> dict[str, str]:
        environment = os.environ.copy()
        if self.hf_token:
            environment["HF_TOKEN"] = self.hf_token
        return environment

    def _target_uri(self, remote_path: str) -> str:
        candidate = str(remote_path or "").strip()
        if candidate.startswith("hf://"):
            return candidate.rstrip("/")
        if not candidate:
            return self.bucket_uri
        return f"{self.bucket_uri}/{candidate.lstrip('/')}".rstrip("/")

    @staticmethod
    def _object_path(value: str) -> str:
        """Normalize a CLI path or bucket URI to its bucket-relative path."""

        candidate = urllib_parse.unquote(
            str(value or "").strip()
        ).replace("\\", "/")
        if candidate.startswith("hf://"):
            parsed = urllib_parse.urlsplit(candidate)
            if parsed.scheme != "hf" or parsed.netloc != "buckets":
                return ""
            # Bucket URIs are hf://buckets/{namespace}/{bucket}/{object}.
            parts = parsed.path.strip("/").split("/")
            if len(parts) < 3:
                return ""
            candidate = "/".join(parts[2:])
        parts = [
            part
            for part in candidate.strip("/").split("/")
            if part and part != "."
        ]
        if ".." in parts:
            return ""
        return "/".join(parts)

    def _run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.cli_executable, *[str(argument) for argument in arguments]],
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout_seconds or self.timeout_seconds),
        )

    @staticmethod
    def _parse_listing(stdout: str) -> list[Mapping[str, Any]]:
        payload = str(stdout or "").strip()
        if not payload:
            return []
        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError) as error:
            raise HFBucketBackendError(
                "Hugging Face bucket listing returned invalid JSON"
            ) from error
        if isinstance(parsed, list):
            if any(not isinstance(entry, Mapping) for entry in parsed):
                raise HFBucketBackendError(
                    "Hugging Face bucket listing entries are not objects"
                )
            return list(parsed)
        if isinstance(parsed, Mapping):
            for key in ("items", "files", "entries"):
                entries = parsed.get(key)
                if isinstance(entries, list):
                    if any(
                        not isinstance(entry, Mapping)
                        for entry in entries
                    ):
                        raise HFBucketBackendError(
                            "Hugging Face bucket listing entries are not objects"
                        )
                    return list(entries)
            if parsed.get("path"):
                return [parsed]
        raise HFBucketBackendError(
            "Hugging Face bucket listing returned an unsupported payload"
        )

    def _list_entries(self, remote_path: str, *, recursive: bool) -> list[Mapping[str, Any]]:
        arguments = [
            "buckets",
            "list",
            self._target_uri(remote_path),
            "--json",
        ]
        if recursive:
            arguments.append("--recursive")
        try:
            completed = self._run(arguments)
        except (OSError, subprocess.SubprocessError) as error:
            raise HFBucketBackendError(
                "Hugging Face bucket listing was unavailable"
            ) from error
        if completed.returncode != 0:
            detail = str(
                getattr(completed, "stderr", "")
                or getattr(completed, "stdout", "")
                or ""
            ).strip()
            suffix = f": {detail[:500]}" if detail else ""
            raise HFBucketBackendError(
                "Hugging Face bucket listing failed with exit code "
                f"{completed.returncode}{suffix}"
            )
        if not isinstance(completed.stdout, str):
            # Some legacy callers inject a minimal CompletedProcess-like test
            # double containing only ``returncode``.  A real text-mode
            # subprocess always supplies str output.
            return [{"type": "file", "path": self._target_uri(remote_path)}]
        return self._parse_listing(completed.stdout)

    def put_file(self, local_path: Path, remote_path: str) -> bool:
        source = Path(local_path)
        if not source.is_file():
            return False
        try:
            completed = self._run(
                ["buckets", "cp", str(source), self._target_uri(remote_path)],
                timeout_seconds=max(180.0, self.timeout_seconds),
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0

    def get_file(
        self,
        remote_path: str,
        *,
        max_bytes: int = 32 * 1024 * 1024,
    ) -> bytes:
        """Download one bucket object with an explicit post-copy size bound."""

        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        with tempfile.TemporaryDirectory(prefix="hf-bucket-download-") as directory:
            destination = Path(directory) / "artifact"
            try:
                completed = self._run(
                    [
                        "buckets",
                        "cp",
                        self._target_uri(remote_path),
                        str(destination),
                    ],
                    timeout_seconds=max(180.0, self.timeout_seconds),
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise HFBucketBackendError(
                    "Hugging Face bucket download was unavailable"
                ) from error
            if completed.returncode != 0:
                detail = str(
                    getattr(completed, "stderr", "")
                    or getattr(completed, "stdout", "")
                    or ""
                ).strip()
                suffix = f": {detail[:500]}" if detail else ""
                raise HFBucketBackendError(
                    "Hugging Face bucket download failed with exit code "
                    f"{completed.returncode}{suffix}"
                )
            try:
                with destination.open("rb") as handle:
                    content = handle.read(max_bytes + 1)
            except OSError as error:
                raise HFBucketBackendError(
                    "Hugging Face bucket download produced no readable file"
                ) from error
            if len(content) > max_bytes:
                raise HFBucketBackendError(
                    "Hugging Face bucket object exceeds the download limit"
                )
            return content

    def exists(self, remote_path: str) -> bool:
        expected_path = self._object_path(self._target_uri(remote_path))
        if not expected_path:
            return False
        for entry in self._list_entries(remote_path, recursive=False):
            if str(entry.get("type") or "file").lower() not in {
                "file",
                "blob",
            }:
                continue
            entry_path = self._object_path(
                str(entry.get("path") or entry.get("name") or "")
            )
            if entry_path == expected_path:
                return True
        return False

    def list_files(self, prefix: str) -> list[str]:
        files: list[str] = []
        for entry in self._list_entries(prefix, recursive=True):
            if str(entry.get("type") or "file").lower() not in {"file", "blob"}:
                continue
            path = str(entry.get("path") or entry.get("name") or "").strip()
            if path:
                files.append(path)
        return sorted(dict.fromkeys(files))


class HFSpaceClient:
    """Small, API-shape-agnostic client for Gradio-backed HF Spaces."""

    def __init__(
        self,
        space_url: str,
        timeout_seconds: float = 120.0,
        headers_factory: HeadersFactory | None = None,
        *,
        session: _RequestsSession | None = None,
    ):
        resolved_url = str(space_url or "").strip().rstrip("/")
        if not resolved_url:
            raise ValueError("space_url is required")
        self.space_url = resolved_url
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.headers_factory = headers_factory
        self._config_cache: dict[str, Any] | None = None
        self._session = session or _load_requests().Session()

    def close(self) -> None:
        """Release pooled HTTP connections."""

        self._session.close()

    def __enter__(self) -> "HFSpaceClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        headers = {"Accept": accept}
        if self.headers_factory is not None:
            supplied = self.headers_factory()
            if supplied:
                headers.update(
                    {
                        str(key): str(value)
                        for key, value in supplied.items()
                        if value is not None
                    }
                )
        return headers

    def _url(self, path: str) -> str:
        candidate = str(path or "").strip()
        if candidate.startswith(("http://", "https://")):
            return candidate
        return f"{self.space_url}/{candidate.lstrip('/')}"

    def request_json(self, method: str, path: str, payload: Any | None = None) -> Any:
        """Issue an HTTP request and parse its JSON response."""

        resolved_method = str(method or "GET").strip().upper()
        keyword_arguments: dict[str, Any] = {
            "headers": self._headers(),
            "timeout": self.timeout_seconds,
        }
        if resolved_method == "GET":
            if payload is not None:
                keyword_arguments["params"] = payload
        elif payload is not None:
            keyword_arguments["json"] = payload
        if resolved_method == "GET":
            response = self._session.get(
                self._url(path),
                **keyword_arguments,
            )
        elif resolved_method == "POST":
            response = self._session.post(
                self._url(path),
                **keyword_arguments,
            )
        else:
            response = self._session.request(
                resolved_method,
                self._url(path),
                **keyword_arguments,
            )
        response.raise_for_status()
        return response.json()

    def upload_file(
        self,
        file_name: str,
        data: bytes,
        mime_type: str = "application/octet-stream",
    ) -> Any:
        """Upload bytes to Gradio and return its FileData JSON."""

        headers = self._headers()
        for key in list(headers):
            if key.lower() == "content-type":
                headers.pop(key)
        response = self._session.post(
            self._url("gradio_api/upload"),
            files={
                "files": (
                    str(file_name or "upload.bin"),
                    bytes(data),
                    str(mime_type or "application/octet-stream"),
                )
            },
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def get_config(self, *, use_cache: bool = True) -> dict[str, Any]:
        """Fetch the live Gradio config, caching a successful response."""

        if use_cache and self._config_cache is not None:
            return self._config_cache
        config = self.request_json("GET", "config")
        if not isinstance(config, Mapping):
            raise ValueError("Space config response is not a JSON object")
        self._config_cache = dict(config)
        return self._config_cache

    # Compatibility alias retained for earlier callers.
    def _get_config(self) -> dict[str, Any]:
        return self.get_config()

    def get_endpoints(
        self,
        config: Mapping[str, Any] | None = None,
    ) -> list[EndpointContract]:
        """Convert Gradio dependency records into stable endpoint contracts."""

        resolved_config = config if config is not None else self.get_config()
        dependencies = resolved_config.get("dependencies")
        if not isinstance(dependencies, list):
            return []
        endpoints: list[EndpointContract] = []
        for position, dependency in enumerate(dependencies):
            if not isinstance(dependency, Mapping):
                continue
            raw_id = dependency.get("id")
            if isinstance(raw_id, int) and not isinstance(raw_id, bool):
                dependency_id = raw_id
            elif isinstance(raw_id, str) and raw_id.strip().isdigit():
                dependency_id = int(raw_id.strip())
            else:
                dependency_id = position
            inputs = dependency.get("inputs")
            raw_api_name = dependency.get("api_name")
            api_name = (
                normalize_api_name(raw_api_name)
                if isinstance(raw_api_name, str)
                else ""
            )
            endpoints.append(
                EndpointContract(
                    fn_index=dependency_id,
                    dependency_id=dependency_id,
                    label=str(
                        dependency.get("label") or f"fn_{dependency_id}"
                    ),
                    api_name=api_name,
                    component_name=str(
                        dependency.get("component_name") or "unknown"
                    ),
                    input_count=len(inputs) if isinstance(inputs, list) else None,
                )
            )
        return endpoints

    def dependency_api_names(
        self,
        config: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Return normalized, unique registered API names."""

        return sorted(
            {
                endpoint.api_name
                for endpoint in self.get_endpoints(config)
                if endpoint.api_name
            }
        )

    def resolve_fn_index(
        self,
        api_name: str,
        config: Mapping[str, Any] | None = None,
        *,
        fallback_markers: Sequence[str] | None = None,
    ) -> int:
        """Resolve an API name, optionally falling back to semantic markers."""

        target = normalize_api_name(api_name)
        endpoints = self.get_endpoints(config)
        if target:
            for endpoint in endpoints:
                if endpoint.api_name == target:
                    return endpoint.fn_index
        markers = [
            str(marker).strip().lower()
            for marker in (fallback_markers or ())
            if str(marker).strip()
        ]
        for endpoint in endpoints:
            searchable = (
                f"{endpoint.api_name} {endpoint.label} "
                f"{endpoint.component_name}"
            ).lower()
            if any(marker in searchable for marker in markers):
                return endpoint.fn_index
        raise ValueError(f"Space api_name {api_name!r} was not found")

    def lookup_dependency_input_count(
        self,
        fn_index: int,
        config: Mapping[str, Any] | None = None,
    ) -> int | None:
        """Return the input arity for a dependency, when advertised."""

        for endpoint in self.get_endpoints(config):
            if endpoint.fn_index == int(fn_index):
                return endpoint.input_count
        return None

    def call_endpoint(self, fn_index: int, data: Sequence[Any]) -> list[Any]:
        """Call the legacy synchronous Gradio prediction endpoint."""

        response = self.request_json(
            "POST",
            "api/predict",
            {"data": list(data), "fn_index": int(fn_index)},
        )
        output = response.get("data") if isinstance(response, Mapping) else None
        return list(output) if isinstance(output, list) else []

    def queue_join(
        self,
        fn_index: int,
        data: Sequence[Any],
        *,
        session_hash: str | None = None,
    ) -> str:
        """Submit one dependency call to the Gradio queue."""

        resolved_session_hash = str(session_hash or uuid.uuid4().hex)
        response = self.request_json(
            "POST",
            "gradio_api/queue/join",
            {
                "data": list(data),
                "fn_index": int(fn_index),
                "session_hash": resolved_session_hash,
            },
        )
        if isinstance(response, Mapping):
            error = response.get("error")
            if error:
                raise RuntimeError(f"Space queue rejected request: {error}")
            if response.get("success") is False:
                raise RuntimeError(f"Space queue rejected request: {response}")
        return resolved_session_hash

    @staticmethod
    def _iter_sse_events(
        response: _RequestsResponse,
    ) -> Iterator[Mapping[str, Any]]:
        for raw_line in response.iter_lines(decode_unicode=True):
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace").strip()
            else:
                line = str(raw_line or "").strip()
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if not payload:
                continue
            try:
                event = json.loads(payload)
            except (TypeError, ValueError):
                continue
            if isinstance(event, Mapping):
                yield event

    @staticmethod
    def _resolve_terminal_event(
        event: Mapping[str, Any],
        *,
        operation: str,
    ) -> dict[str, Any] | None:
        message = str(event.get("msg") or "")
        if message == "process_completed":
            if event.get("success") is False:
                raise RuntimeError(
                    f"Space {operation} failed: {event.get('output') or event}"
                )
            output = event.get("output")
            return dict(output) if isinstance(output, Mapping) else dict(event)
        if message in {"process_failed", "queue_full"}:
            raise RuntimeError(f"Space {operation} failed: {event}")
        return None

    def _wait_for_sse_result(
        self,
        stream_url: str,
        *,
        operation: str,
        timeout_seconds: float,
        poll_interval_seconds: float,
        timeout_message: str,
    ) -> dict[str, Any]:
        """Consume an SSE result, reconnecting the same admitted operation."""

        deadline = time.monotonic() + timeout_seconds
        last_transport_error: BaseException | None = None
        while time.monotonic() < deadline:
            remaining = max(1.0, deadline - time.monotonic())
            response: _RequestsResponse | None = None
            try:
                response = self._session.get(
                    stream_url,
                    headers=self._headers(),
                    timeout=min(30.0, max(5.0, remaining)),
                    stream=True,
                )
                response.raise_for_status()
                for event in self._iter_sse_events(response):
                    terminal = self._resolve_terminal_event(
                        event,
                        operation=operation,
                    )
                    if terminal is not None:
                        return terminal
            except Exception as error:
                if not is_hf_space_transport_error(error):
                    raise
                last_transport_error = error
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(
                    min(
                        remaining,
                        max(0.05, float(poll_interval_seconds)),
                    )
                )
        if last_transport_error is not None:
            raise TimeoutError(timeout_message) from last_transport_error
        raise TimeoutError(timeout_message)

    def wait_for_queue_result(
        self,
        session_hash: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.5,
    ) -> dict[str, Any]:
        """Wait for the terminal event on a submitted queue session."""

        timeout = (
            self.timeout_seconds
            if timeout_seconds is None
            else max(1.0, float(timeout_seconds))
        )
        stream_url = self._url(
            "gradio_api/queue/data?session_hash="
            f"{urllib_parse.quote(str(session_hash), safe='')}"
        )
        return self._wait_for_sse_result(
            stream_url,
            operation="queue",
            timeout_seconds=timeout,
            poll_interval_seconds=poll_interval_seconds,
            timeout_message="Space queue timed out",
        )

    def call_api_name(
        self,
        api_name: str,
        data: Sequence[Any],
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.5,
    ) -> dict[str, Any]:
        """Call Gradio's named endpoint API and consume its SSE result."""

        normalized_name = normalize_api_name(api_name).lstrip("/")
        if not normalized_name:
            raise ValueError("api_name is required")
        timeout = (
            self.timeout_seconds
            if timeout_seconds is None
            else max(1.0, float(timeout_seconds))
        )
        encoded_name = urllib_parse.quote(normalized_name, safe="")
        response = self._session.post(
            self._url(f"gradio_api/call/{encoded_name}"),
            json={"data": list(data)},
            headers=self._headers(),
            timeout=timeout,
        )
        response.raise_for_status()
        submitted = response.json()
        if not isinstance(submitted, Mapping):
            raise ValueError("Gradio call response is not a JSON object")
        event_id = str(
            submitted.get("event_id") or submitted.get("eventId") or ""
        ).strip()
        if not event_id:
            if isinstance(submitted.get("data"), list):
                return dict(submitted)
            raise ValueError("Gradio call response did not include event_id")

        stream_url = self._url(
            f"gradio_api/call/{encoded_name}/"
            f"{urllib_parse.quote(event_id, safe='')}"
        )
        return self._wait_for_sse_result(
            stream_url,
            operation="call",
            timeout_seconds=timeout,
            poll_interval_seconds=poll_interval_seconds,
            timeout_message="Space Gradio call timed out",
        )

    def file_url(self, reference: Any) -> str:
        """Resolve a Gradio FileData record into a downloadable URL."""

        if isinstance(reference, Mapping):
            direct_url = str(reference.get("url") or "").strip()
            path = str(
                reference.get("path") or reference.get("name") or ""
            ).strip()
        else:
            direct_url = str(reference or "").strip()
            path = direct_url
        if direct_url.startswith(("http://", "https://")):
            return direct_url
        if direct_url.startswith("/"):
            return urllib_parse.urljoin(f"{self.space_url}/", direct_url)
        encoded_path = urllib_parse.quote(path, safe="/:")
        return self._url(f"gradio_api/file={encoded_path}")

    def fetch_file(
        self,
        reference: Any,
        *,
        accept: str = "audio/*, application/octet-stream",
    ) -> tuple[bytes, str]:
        """Download a Gradio output file, or return embedded test bytes."""

        if isinstance(reference, Mapping):
            inline_bytes = reference.get("_inline_bytes")
            if isinstance(inline_bytes, (bytes, bytearray)):
                name = str(
                    reference.get("name") or reference.get("path") or ""
                )
                return bytes(inline_bytes), (
                    mimetypes.guess_type(name)[0]
                    or "application/octet-stream"
                )
        response = self._session.get(
            self.file_url(reference),
            headers=self._headers(accept=accept),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return (
            response.content,
            str(
                response.headers.get("Content-Type")
                or "application/octet-stream"
            ),
        )

    def probe_contract(
        self,
        expected_endpoints: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return a serializable, non-mutating endpoint compatibility report."""

        summary: dict[str, Any] = {
            "available": False,
            "endpoints": [],
            "errors": [],
        }
        try:
            endpoints = self.get_endpoints()
            summary["endpoints"] = [
                {
                    "fn_index": endpoint.fn_index,
                    "label": endpoint.label,
                    "api_name": endpoint.api_name,
                    "component_name": endpoint.component_name,
                    "input_count": endpoint.input_count,
                }
                for endpoint in endpoints
            ]
            available_names = {
                alias
                for endpoint in endpoints
                for alias in (
                    endpoint.api_name,
                    endpoint.api_name.lstrip("/"),
                    endpoint.label,
                )
                if alias
            }
            for expected in expected_endpoints or ():
                normalized = normalize_api_name(expected)
                if (
                    str(expected) not in available_names
                    and normalized not in available_names
                    and normalized.lstrip("/") not in available_names
                ):
                    summary["errors"].append(
                        f"Expected endpoint {expected!r} not found"
                    )
            summary["available"] = not summary["errors"]
        except Exception as exc:
            summary["errors"].append(f"{type(exc).__name__}: {exc}")
        return summary


@dataclass(frozen=True)
class BatchState:
    """Persistent checkpoint for resumable Space batch processing."""

    schema_version: int = 1
    updated_at: str = ""
    total_items: int = 0
    next_offset: int = 0
    batch_size: int = 32
    batches_completed: int = 0
    failures: int = 0
    last_batch_id: str = ""
    stop_reason: str = ""

    def validate(self) -> None:
        """Reject state that cannot be safely round-tripped."""

        integer_fields = (
            "schema_version",
            "total_items",
            "next_offset",
            "batch_size",
            "batches_completed",
            "failures",
        )
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise ValueError("checkpoint integer fields must be integers")
        if self.schema_version != 1:
            raise ValueError("unsupported checkpoint schemaVersion")
        if (
            self.total_items < 0
            or self.next_offset < 0
            or self.next_offset > self.total_items
            or self.batch_size <= 0
            or self.batches_completed < 0
            or self.failures < 0
        ):
            raise ValueError("checkpoint fields are outside valid bounds")
        for name in ("updated_at", "last_batch_id", "stop_reason"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"checkpoint {name} must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "updatedAt": self.updated_at,
            "totalItems": self.total_items,
            "nextOffset": self.next_offset,
            "batchSize": self.batch_size,
            "batchesCompleted": self.batches_completed,
            "failures": self.failures,
            "lastBatchId": self.last_batch_id,
            **(
                {"stopReason": self.stop_reason}
                if self.stop_reason
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BatchState":
        return cls(
            schema_version=int(data.get("schemaVersion", 1)),
            updated_at=str(data.get("updatedAt", "")),
            total_items=int(data.get("totalItems", 0)),
            next_offset=int(data.get("nextOffset", 0)),
            batch_size=int(data.get("batchSize", 32)),
            batches_completed=int(data.get("batchesCompleted", 0)),
            failures=int(data.get("failures", 0)),
            last_batch_id=str(data.get("lastBatchId", "")),
            stop_reason=str(data.get("stopReason", "")),
        )


class BatchProcessor:
    """Retrying generic Space processor with a package-owned checkpoint."""

    def __init__(
        self,
        client: HFSpaceClient,
        output_backend: OutputBackend,
        state_file: Path,
        batch_size: int = 32,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 10.0,
        retry_backoff_multiplier: float = 2.0,
        retry_backoff_max_seconds: float = 120.0,
    ):
        self.client = client
        self.output_backend = output_backend
        self.state_file = Path(state_file)
        self.batch_size = max(1, int(batch_size))
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_backoff_seconds = max(
            0.0,
            float(retry_backoff_seconds),
        )
        self.retry_backoff_multiplier = max(
            1.0,
            float(retry_backoff_multiplier),
        )
        self.retry_backoff_max_seconds = max(
            self.retry_backoff_seconds,
            float(retry_backoff_max_seconds),
        )

    def load_state(self) -> BatchState:
        if not self.state_file.exists():
            return BatchState(batch_size=self.batch_size)
        try:
            payload = json.loads(
                self.state_file.read_text(encoding="utf-8")
            )
            if not isinstance(payload, Mapping):
                raise ValueError("top level is not an object")
            required_fields = frozenset(
                {
                    "schemaVersion",
                    "updatedAt",
                    "totalItems",
                    "nextOffset",
                    "batchSize",
                    "batchesCompleted",
                    "failures",
                    "lastBatchId",
                }
            )
            missing_fields = sorted(required_fields - payload.keys())
            if missing_fields:
                raise ValueError(
                    "checkpoint is missing required fields: "
                    + ", ".join(missing_fields)
                )
            integer_fields = (
                "schemaVersion",
                "totalItems",
                "nextOffset",
                "batchSize",
                "batchesCompleted",
                "failures",
            )
            if any(type(payload.get(name)) is not int for name in integer_fields):
                raise ValueError("checkpoint integer fields must be integers")
            if payload.get("schemaVersion") != 1:
                raise ValueError("unsupported checkpoint schemaVersion")
            for name in ("updatedAt", "lastBatchId"):
                if not isinstance(payload.get(name), str):
                    raise ValueError(f"checkpoint {name} must be a string")
            if "stopReason" in payload and not isinstance(
                payload.get("stopReason"),
                str,
            ):
                raise ValueError("checkpoint stopReason must be a string")
            state = BatchState.from_dict(payload)
            state.validate()
            return state
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid batch checkpoint: {self.state_file}"
            ) from error

    def save_state(self, state: BatchState) -> None:
        state.validate()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = BatchState(
            schema_version=state.schema_version,
            updated_at=_utc_now(),
            total_items=state.total_items,
            next_offset=state.next_offset,
            batch_size=state.batch_size,
            batches_completed=state.batches_completed,
            failures=state.failures,
            last_batch_id=state.last_batch_id,
            stop_reason=state.stop_reason,
        )
        _atomic_write_json(self.state_file, checkpoint.to_dict())

    def calculate_retry_backoff(self, attempt: int) -> float:
        return min(
            self.retry_backoff_max_seconds,
            self.retry_backoff_seconds
            * (
                self.retry_backoff_multiplier
                ** max(0, int(attempt))
            ),
        )

    def process_batch(
        self,
        items: Sequence[Any],
        endpoint_fn_index: int,
        output_batch_id: str,
        output_dir: Path | None = None,
        *,
        payload_builder: Callable[
            [Sequence[Any]],
            Sequence[Any],
        ]
        | None = None,
        use_queue: bool = False,
        queue_timeout_seconds: float | None = None,
    ) -> tuple[bool, list[Any]]:
        """Process one batch, retrying only transport-level failures."""

        del output_batch_id  # Reserved for backend-specific subclasses.
        for attempt in range(self.retry_attempts):
            try:
                payload = list(
                    payload_builder(items)
                    if payload_builder is not None
                    else items
                )
                if use_queue:
                    session_hash = self.client.queue_join(
                        endpoint_fn_index,
                        payload,
                    )
                    response = self.client.wait_for_queue_result(
                        session_hash,
                        timeout_seconds=queue_timeout_seconds,
                    )
                    result = (
                        response.get("data")
                        if isinstance(response, Mapping)
                        else None
                    )
                    outputs = list(result) if isinstance(result, list) else []
                else:
                    outputs = self.client.call_endpoint(
                        endpoint_fn_index,
                        payload,
                    )
                if output_dir is not None:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                return True, outputs
            except Exception:
                if attempt + 1 >= self.retry_attempts:
                    return False, []
                time.sleep(self.calculate_retry_backoff(attempt))
        return False, []


__all__ = [
    "SpaceRuntimeInfo",
    "EndpointContract",
    "OutputBackend",
    "LocalFileSystemBackend",
    "HFBucketBackendError",
    "HFBucketBackend",
    "HFSpaceClient",
    "RefreshableGradioFile",
    "BatchState",
    "BatchProcessor",
    "is_hf_space_transport_error",
    "is_retryable_hf_space_error",
    "is_stale_gradio_file_error",
    "normalize_api_name",
]
