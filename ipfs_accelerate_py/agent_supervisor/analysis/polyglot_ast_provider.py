"""Bounded, deterministic producers for canonical polyglot AST facts.

The provider adapts Python, JSON-schema-like documents, and the local
TypeScript compiler API to the existing :class:`ASTBlobRecord` interchange.
It is deliberately lazy: importing this module, constructing a provider, and
inspecting its limits never starts Node or imports an optional parser.

Source text is accepted only as an input to extraction.  It is neither
retained on the provider nor serialized into an ``ASTBlobRecord``.
"""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..core.conflict_graph import ASTBlobRecord, build_python_ast_blob_record


POLYGLOT_AST_PROVIDER_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/polyglot-ast-provider@1"
)
TYPESCRIPT_EXTRACTOR_PROTOCOL_VERSION = 1
TYPESCRIPT_EXTRACTOR_VERSION = "typescript-ast-extractor@2"

DEFAULT_MAX_FILES = 256
DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
DEFAULT_PROCESS_TIMEOUT_SECONDS = 15.0
DEFAULT_NODE_MEMORY_MIB = 256

HARD_MAX_FILES = 10_000
# DCR-012 requires legitimate source/data blobs up to the 32 MiB snapshot bound
# to be inspected directly rather than rejected by the older 16 MiB provider cap.
HARD_MAX_FILE_BYTES = 32 * 1024 * 1024
HARD_MAX_TOTAL_BYTES = 256 * 1024 * 1024
HARD_MAX_OUTPUT_BYTES = 32 * 1024 * 1024
HARD_PROCESS_TIMEOUT_SECONDS = 120.0
HARD_NODE_MEMORY_MIB = 2_048

_JS_LANGUAGES = frozenset({"javascript", "jsx", "typescript", "tsx"})
_STRUCTURED_LANGUAGES = frozenset({"json", "json-schema", "openapi-json"})
_LANGUAGE_ALIASES = {
    "cjs": "javascript",
    "js": "javascript",
    "javascript": "javascript",
    "mjs": "javascript",
    "jsx": "jsx",
    "py": "python",
    "python": "python",
    "ts": "typescript",
    "typescript": "typescript",
    "tsx": "tsx",
    "json": "json",
    "json_schema": "json-schema",
    "json-schema": "json-schema",
    "schema": "json-schema",
    "openapi": "openapi-json",
    "openapi-json": "openapi-json",
}
_PATH_LANGUAGES = {
    ".cjs": "javascript",
    ".js": "javascript",
    ".mjs": "javascript",
    ".jsx": "jsx",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".json": "json",
}


class PolyglotASTReason(str, Enum):
    """Stable machine-readable failures at the provider boundary."""

    INVALID_SOURCE = "invalid_source"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    FILE_LIMIT_EXCEEDED = "file_limit_exceeded"
    FILE_BYTES_EXCEEDED = "file_bytes_exceeded"
    TOTAL_BYTES_EXCEEDED = "total_bytes_exceeded"
    NODE_UNAVAILABLE = "node_unavailable"
    EXTRACTOR_UNAVAILABLE = "extractor_unavailable"
    PROCESS_TIMEOUT = "process_timeout"
    OUTPUT_BYTES_EXCEEDED = "output_bytes_exceeded"
    PROCESS_FAILED = "process_failed"
    COMPILER_UNAVAILABLE = "compiler_unavailable"
    COMPILER_VERSION_MISMATCH = "compiler_version_mismatch"
    PROTOCOL_ERROR = "protocol_error"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"


class PolyglotASTProviderError(RuntimeError):
    """Typed, non-model failure raised by :class:`PolyglotASTProvider`."""

    def __init__(
        self,
        reason_code: PolyglotASTReason | str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason_code = str(
            reason_code.value if isinstance(reason_code, PolyglotASTReason) else reason_code
        )
        self.details = dict(details or {})
        super().__init__(f"{self.reason_code}: {message}")


def _bounded_positive_int(name: str, value: Any, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1 or result > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return result


@dataclass(frozen=True)
class PolyglotASTLimits:
    """Hard resource envelope for one provider and its batch operations."""

    max_files: int = DEFAULT_MAX_FILES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    process_timeout_seconds: float = DEFAULT_PROCESS_TIMEOUT_SECONDS
    node_memory_mib: int = DEFAULT_NODE_MEMORY_MIB

    def __post_init__(self) -> None:
        for name, maximum in (
            ("max_files", HARD_MAX_FILES),
            ("max_file_bytes", HARD_MAX_FILE_BYTES),
            ("max_total_bytes", HARD_MAX_TOTAL_BYTES),
            ("max_output_bytes", HARD_MAX_OUTPUT_BYTES),
            ("node_memory_mib", HARD_NODE_MEMORY_MIB),
        ):
            object.__setattr__(
                self,
                name,
                _bounded_positive_int(name, getattr(self, name), maximum),
            )
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("max_file_bytes cannot exceed max_total_bytes")
        timeout = self.process_timeout_seconds
        if isinstance(timeout, bool):
            raise ValueError("process_timeout_seconds must be positive")
        try:
            normalized_timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("process_timeout_seconds must be positive") from exc
        if not 0 < normalized_timeout <= HARD_PROCESS_TIMEOUT_SECONDS:
            raise ValueError(
                "process_timeout_seconds must be between 0 and "
                f"{HARD_PROCESS_TIMEOUT_SECONDS}"
            )
        object.__setattr__(
            self, "process_timeout_seconds", normalized_timeout
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "max_files": self.max_files,
            "max_file_bytes": self.max_file_bytes,
            "max_total_bytes": self.max_total_bytes,
            "max_output_bytes": self.max_output_bytes,
            "process_timeout_seconds": self.process_timeout_seconds,
            "node_memory_mib": self.node_memory_mib,
        }


@dataclass(frozen=True)
class PolyglotASTInput:
    """One source body and its exact, path-independent identity."""

    source: str | bytes
    language: str
    blob_identity: str = ""
    source_sha256: str = ""


@dataclass(frozen=True)
class PolyglotASTExtraction:
    """An AST record together with the exact producer identity that made it."""

    record: ASTBlobRecord
    language: str
    producer: str
    producer_version: str
    compiler_name: str = ""
    compiler_version: str = ""

    @property
    def tool_identity(self) -> str:
        compiler = (
            f"/{self.compiler_name}@{self.compiler_version}"
            if self.compiler_name and self.compiler_version
            else ""
        )
        return f"{self.producer_version}{compiler}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_schema": POLYGLOT_AST_PROVIDER_SCHEMA,
            "language": self.language,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "compiler_name": self.compiler_name,
            "compiler_version": self.compiler_version,
            "tool_identity": self.tool_identity,
            "record": self.record.to_dict(),
        }


_ProcessRunner = Callable[
    [Sequence[str], bytes, float, int, Mapping[str, str]],
    tuple[int, bytes, bytes],
]


def _source_text(source: str | bytes) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, bytes):
        try:
            return source.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PolyglotASTProviderError(
                PolyglotASTReason.INVALID_SOURCE,
                "source bytes must be valid UTF-8",
            ) from exc
    raise PolyglotASTProviderError(
        PolyglotASTReason.INVALID_SOURCE,
        "source must be text or UTF-8 bytes",
    )


def _source_bytes(source: str) -> bytes:
    return source.encode("utf-8", errors="surrogatepass")


def _source_hash(source: str) -> str:
    return "sha256:" + hashlib.sha256(_source_bytes(source)).hexdigest()


def _normalize_source_hash(value: str) -> str:
    result = str(value or "").strip()
    if result and ":" not in result:
        result = "sha256:" + result
    return result


def _normalize_language(value: str) -> str:
    raw = str(value or "").strip().casefold()
    if raw.startswith("."):
        raw = raw[1:]
    language = _LANGUAGE_ALIASES.get(raw)
    if language is None:
        raise PolyglotASTProviderError(
            PolyglotASTReason.UNSUPPORTED_LANGUAGE,
            f"unsupported source language {value!r}",
        )
    return language


def language_for_path(path: str | os.PathLike[str]) -> str:
    """Return the supported language selected solely from a file suffix."""

    suffix = Path(path).suffix.casefold()
    language = _PATH_LANGUAGES.get(suffix)
    if language is None:
        raise PolyglotASTProviderError(
            PolyglotASTReason.UNSUPPORTED_LANGUAGE,
            f"unsupported source suffix {suffix or '<none>'!r}",
        )
    return language


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


def _bounded_process_runner(
    command: Sequence[str],
    request: bytes,
    timeout_seconds: float,
    max_output_bytes: int,
    environment: Mapping[str, str],
) -> tuple[int, bytes, bytes]:
    """Run one extractor while streaming both output pipes under hard caps."""

    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=dict(environment),
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    selector = selectors.DefaultSelector()
    os.set_blocking(process.stdin.fileno(), False)
    selector.register(process.stdin, selectors.EVENT_WRITE, data="stdin")
    streams = ((process.stdout, "stdout"), (process.stderr, "stderr"))
    for stream, name in streams:
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, data=name)
    buffers: dict[str, bytearray] = {
        "stdout": bytearray(),
        "stderr": bytearray(),
    }
    request_offset = 0
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_tree(process)
                process.wait()
                raise PolyglotASTProviderError(
                    PolyglotASTReason.PROCESS_TIMEOUT,
                    f"extractor exceeded {timeout_seconds:g} seconds",
                )
            events = selector.select(min(remaining, 0.05))
            if not events and process.poll() is not None:
                events = [
                    (key, selectors.EVENT_READ)
                    for key in tuple(selector.get_map().values())
                ]
            for key, _ in events:
                if key.data == "stdin":
                    try:
                        written = os.write(
                            key.fd,
                            request[request_offset : request_offset + 65_536],
                        )
                    except (BlockingIOError, BrokenPipeError, OSError):
                        if process.poll() is None:
                            continue
                        written = 0
                    request_offset += written
                    if request_offset >= len(request) or process.poll() is not None:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                try:
                    chunk = os.read(key.fd, 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = buffers[str(key.data)]
                target.extend(chunk)
                if sum(len(value) for value in buffers.values()) > max_output_bytes:
                    _kill_process_tree(process)
                    process.wait()
                    raise PolyglotASTProviderError(
                        PolyglotASTReason.OUTPUT_BYTES_EXCEEDED,
                        f"extractor output exceeded {max_output_bytes} bytes",
                    )
        return_code = process.wait(
            timeout=max(0.001, deadline - time.monotonic())
        )
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(process)
        process.wait()
        raise PolyglotASTProviderError(
            PolyglotASTReason.PROCESS_TIMEOUT,
            f"extractor exceeded {timeout_seconds:g} seconds",
        ) from exc
    finally:
        selector.close()
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
        process.stderr.close()
    return return_code, bytes(buffers["stdout"]), bytes(buffers["stderr"])


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PolyglotASTProviderError(
            PolyglotASTReason.PROTOCOL_ERROR,
            f"extractor field {field_name!r} must be a string array",
        )
    return tuple(value)


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise PolyglotASTProviderError(
            PolyglotASTReason.PROTOCOL_ERROR,
            f"extractor field {field_name!r} must be an object",
        )
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise PolyglotASTProviderError(
                PolyglotASTReason.PROTOCOL_ERROR,
                f"extractor field {field_name!r} must contain strings",
            )
        result[key] = item
    return result


def _line_mapping(value: Any) -> dict[str, tuple[int, int]]:
    if not isinstance(value, Mapping):
        raise PolyglotASTProviderError(
            PolyglotASTReason.PROTOCOL_ERROR,
            "extractor field 'symbol_lines' must be an object",
        )
    result: dict[str, tuple[int, int]] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, list)
            or len(item) != 2
            or any(isinstance(number, bool) or not isinstance(number, int) for number in item)
            or item[0] < 0
            or item[1] < item[0]
        ):
            raise PolyglotASTProviderError(
                PolyglotASTReason.PROTOCOL_ERROR,
                "extractor symbol lines must be [start, end] integer pairs",
            )
        result[key] = (item[0], item[1])
    return result


def _schema_record(
    source: str,
    *,
    language: str,
    blob_identity: str,
    source_sha256: str,
) -> ASTBlobRecord:
    try:
        value = json.loads(source)
    except (json.JSONDecodeError, RecursionError) as exc:
        if isinstance(exc, json.JSONDecodeError):
            parse_error = (
                f"JSONDecodeError at line {exc.lineno}, column {exc.colno}: "
                f"{exc.msg}"
            )
        else:
            parse_error = "RecursionError: structured document is too deeply nested"
        return ASTBlobRecord(
            blob_identity=blob_identity,
            source_sha256=source_sha256,
            parse_error=parse_error,
            language=language,
        )

    symbols: set[str] = set()
    imports: set[str] = set()
    interfaces: set[str] = set()
    hashes: dict[str, str] = {}
    lines: dict[str, tuple[int, int]] = {}

    def add_symbol(name: str, item: Any) -> None:
        normalized = ".".join(part for part in name.split(".") if part)
        if not normalized:
            return
        symbols.add(normalized)
        semantic = _json_bytes(item)
        hashes[normalized] = "sha256:" + hashlib.sha256(semantic).hexdigest()
        lines.setdefault(normalized, (0, 0))

    def visit(item: Any, scope: tuple[str, ...] = ()) -> None:
        if isinstance(item, Mapping):
            reference = item.get("$ref")
            if isinstance(reference, str) and reference:
                imports.add(f"$ref:{reference}")
            title = item.get("title")
            if isinstance(title, str) and title.strip():
                add_symbol(".".join((*scope, title.strip())), item)
            for container_name in ("$defs", "definitions"):
                definitions = item.get(container_name)
                if isinstance(definitions, Mapping):
                    for key in sorted(definitions, key=str):
                        child = definitions[key]
                        name = ".".join((*scope, str(key)))
                        add_symbol(name, child)
                        visit(child, (*scope, str(key)))
            properties = item.get("properties")
            required = {
                str(name)
                for name in item.get("required", ())
                if isinstance(name, str)
            } if isinstance(item.get("required"), list) else set()
            if isinstance(properties, Mapping):
                for key in sorted(properties, key=str):
                    child = properties[key]
                    name = ".".join((*scope, str(key)))
                    add_symbol(name, child)
                    type_name = (
                        str(child.get("type") or "any")
                        if isinstance(child, Mapping)
                        else "any"
                    )
                    interfaces.add(
                        f"{name}:type={type_name};required={str(key) in required}"
                    )
                    visit(child, (*scope, str(key)))
            for key in sorted(item, key=str):
                if key not in {"$defs", "definitions", "properties"}:
                    visit(item[key], scope)
        elif isinstance(item, list):
            for child in item:
                visit(child, scope)

    try:
        visit(value)
    except RecursionError:
        return ASTBlobRecord(
            blob_identity=blob_identity,
            source_sha256=source_sha256,
            parse_error=(
                "RecursionError: structured document is too deeply nested"
            ),
            language=language,
        )
    if isinstance(value, Mapping) and not symbols:
        add_symbol(str(value.get("$id") or value.get("title") or "<schema>"), value)
    return ASTBlobRecord(
        blob_identity=blob_identity,
        source_sha256=source_sha256,
        qualified_symbols=tuple(symbols),
        imports=tuple(imports),
        interfaces=tuple(interfaces),
        symbol_hashes=hashes,
        symbol_lines=lines,
        language=language,
    )


class TypeScriptPersistentWorker:
    """One long-lived Node process that parses many TypeScript-family files.

    DCR-012 requires a bounded persistent worker rather than one process per
    file.  Requests and responses are newline-delimited JSON documents.
    """

    def __init__(
        self,
        *,
        node_executable: str,
        extractor_path: Path,
        typescript_path: str,
        node_memory_mib: int,
        max_output_bytes: int,
        process_timeout_seconds: float,
    ) -> None:
        self._node_executable = node_executable
        self._extractor_path = extractor_path
        self._typescript_path = typescript_path
        self._node_memory_mib = node_memory_mib
        self._max_output_bytes = max_output_bytes
        self._process_timeout_seconds = process_timeout_seconds
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = __import__("threading").Lock()

    def __enter__(self) -> "TypeScriptPersistentWorker":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        node = shutil.which(self._node_executable)
        if node is None:
            raise PolyglotASTProviderError(
                PolyglotASTReason.NODE_UNAVAILABLE,
                f"Node executable {self._node_executable!r} was not found",
            )
        if not self._extractor_path.is_file():
            raise PolyglotASTProviderError(
                PolyglotASTReason.EXTRACTOR_UNAVAILABLE,
                f"TypeScript extractor is missing: {self._extractor_path}",
            )
        environment = dict(os.environ)
        environment["NO_COLOR"] = "1"
        environment["POLYGLOT_AST_PERSISTENT"] = "1"
        environment["POLYGLOT_AST_MAX_INPUT_BYTES"] = str(self._max_output_bytes)
        if self._typescript_path:
            environment["TYPESCRIPT_PATH"] = self._typescript_path
        self._process = subprocess.Popen(
            [
                node,
                f"--max-old-space-size={self._node_memory_mib}",
                str(self._extractor_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=environment,
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            process.wait()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            self.start()
            process = self._process
            if (
                process is None
                or process.stdin is None
                or process.stdout is None
                or process.poll() is not None
            ):
                raise PolyglotASTProviderError(
                    PolyglotASTReason.PROCESS_FAILED,
                    "persistent TypeScript worker is not running",
                )
            encoded = _json_bytes(dict(payload)) + b"\n"
            if len(encoded) > self._max_output_bytes:
                raise PolyglotASTProviderError(
                    PolyglotASTReason.FILE_BYTES_EXCEEDED,
                    f"request exceeds {self._max_output_bytes} bytes",
                )
            try:
                process.stdin.write(encoded)
                process.stdin.flush()
            except OSError as exc:
                raise PolyglotASTProviderError(
                    PolyglotASTReason.PROCESS_FAILED,
                    "failed to write to the persistent TypeScript worker",
                ) from exc
            deadline = time.monotonic() + self._process_timeout_seconds
            line = bytearray()
            # Make stdout non-blocking so we can bound wall-clock time without
            # reading one byte at a time for large fact payloads.
            assert process.stdout is not None
            os.set_blocking(process.stdout.fileno(), False)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _kill_process_tree(process)
                    process.wait()
                    self._process = None
                    raise PolyglotASTProviderError(
                        PolyglotASTReason.PROCESS_TIMEOUT,
                        f"extractor exceeded {self._process_timeout_seconds:g} seconds",
                    )
                if process.poll() is not None and not line:
                    stderr = b""
                    if process.stderr is not None:
                        try:
                            stderr = process.stderr.read() or b""
                        except OSError:
                            stderr = b""
                    self._process = None
                    message = stderr.decode("utf-8", errors="replace").strip()[:512]
                    raise PolyglotASTProviderError(
                        PolyglotASTReason.PROCESS_FAILED,
                        "persistent TypeScript worker exited"
                        + (f": {message}" if message else ""),
                    )
                try:
                    chunk = process.stdout.read(65_536)
                except BlockingIOError:
                    time.sleep(0.001)
                    continue
                except OSError as exc:
                    raise PolyglotASTProviderError(
                        PolyglotASTReason.PROCESS_FAILED,
                        "failed to read from the persistent TypeScript worker",
                    ) from exc
                if not chunk:
                    if process.poll() is not None:
                        self._process = None
                        raise PolyglotASTProviderError(
                            PolyglotASTReason.PROCESS_FAILED,
                            "persistent TypeScript worker closed stdout",
                        )
                    time.sleep(0.001)
                    continue
                newline_at = chunk.find(b"\n")
                if newline_at < 0:
                    line.extend(chunk)
                else:
                    line.extend(chunk[:newline_at])
                    # A well-behaved worker emits one JSON object per line; any
                    # trailing bytes would belong to the next response.
                    if newline_at + 1 < len(chunk):
                        raise PolyglotASTProviderError(
                            PolyglotASTReason.PROTOCOL_ERROR,
                            "persistent TypeScript worker returned multiple lines",
                        )
                    break
                if len(line) > self._max_output_bytes:
                    _kill_process_tree(process)
                    process.wait()
                    self._process = None
                    raise PolyglotASTProviderError(
                        PolyglotASTReason.OUTPUT_BYTES_EXCEEDED,
                        f"extractor output exceeded {self._max_output_bytes} bytes",
                    )
            try:
                response = json.loads(bytes(line).decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PolyglotASTProviderError(
                    PolyglotASTReason.PROTOCOL_ERROR,
                    "extractor did not return one valid UTF-8 JSON object",
                ) from exc
            if not isinstance(response, Mapping):
                raise PolyglotASTProviderError(
                    PolyglotASTReason.PROTOCOL_ERROR,
                    "extractor response must be an object",
                )
            return response


class PolyglotASTProvider:
    """Lazy ``PolyglotASTProvider@1`` implementation over canonical records."""

    schema = POLYGLOT_AST_PROVIDER_SCHEMA

    def __init__(
        self,
        limits: PolyglotASTLimits | None = None,
        *,
        node_executable: str = "node",
        extractor_path: str | os.PathLike[str] | None = None,
        typescript_path: str | os.PathLike[str] | None = None,
        expected_typescript_version: str = "",
        process_runner: _ProcessRunner | None = None,
        persistent_typescript_worker: bool = False,
    ) -> None:
        self.limits = limits or PolyglotASTLimits()
        self.node_executable = str(node_executable or "node")
        self.extractor_path = Path(extractor_path) if extractor_path else (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "extract_typescript_ast.mjs"
        )
        self.typescript_path = (
            str(Path(typescript_path).expanduser().resolve())
            if typescript_path is not None
            else ""
        )
        self.expected_typescript_version = str(
            expected_typescript_version or ""
        ).strip()
        self._process_runner = process_runner or _bounded_process_runner
        self._persistent_typescript_worker = bool(persistent_typescript_worker)
        self._typescript_worker: TypeScriptPersistentWorker | None = None

    def _bounded_source(self, source: str | bytes) -> tuple[str, int]:
        text = _source_text(source)
        size = len(_source_bytes(text))
        if size > self.limits.max_file_bytes:
            raise PolyglotASTProviderError(
                PolyglotASTReason.FILE_BYTES_EXCEEDED,
                f"source exceeds {self.limits.max_file_bytes} UTF-8 bytes",
                details={"actual_bytes": size},
            )
        return text, size

    @staticmethod
    def _identity(
        source: str,
        blob_identity: str,
        source_sha256: str,
    ) -> tuple[str, str]:
        actual = _source_hash(source)
        claimed = _normalize_source_hash(source_sha256)
        if claimed and claimed != actual:
            raise PolyglotASTProviderError(
                PolyglotASTReason.SOURCE_IDENTITY_MISMATCH,
                "claimed source_sha256 does not match source bytes",
            )
        return str(blob_identity or actual), actual

    def extract_with_metadata(
        self,
        source: str | bytes | Mapping[str, Any],
        language: str,
        *,
        blob_identity: str = "",
        source_sha256: str = "",
    ) -> PolyglotASTExtraction:
        """Extract one body without retaining or serializing the source text."""

        normalized_language = _normalize_language(language)
        if isinstance(source, Mapping):
            if normalized_language not in _STRUCTURED_LANGUAGES:
                raise PolyglotASTProviderError(
                    PolyglotASTReason.INVALID_SOURCE,
                    "mapping sources are supported only for structured schemas",
                )
            try:
                source = _json_bytes(source).decode("utf-8")
            except (TypeError, ValueError, RecursionError) as exc:
                raise PolyglotASTProviderError(
                    PolyglotASTReason.INVALID_SOURCE,
                    "structured source must be finite canonical JSON",
                ) from exc
        text, _ = self._bounded_source(source)
        blob, source_hash = self._identity(
            text, blob_identity, source_sha256
        )

        if normalized_language == "python":
            try:
                record = build_python_ast_blob_record(
                    text,
                    blob_identity=blob,
                    source_sha256=source_hash,
                )
            except RecursionError:
                record = ASTBlobRecord(
                    blob_identity=blob,
                    source_sha256=source_hash,
                    parse_error=(
                        "RecursionError: Python source is too deeply nested"
                    ),
                    language="python",
                )
            return PolyglotASTExtraction(
                record=record,
                language="python",
                producer="python-ast",
                producer_version=(
                    f"python-ast@{os.sys.version_info.major}."
                    f"{os.sys.version_info.minor}"
                ),
            )
        if normalized_language in _STRUCTURED_LANGUAGES:
            record = _schema_record(
                text,
                language=normalized_language,
                blob_identity=blob,
                source_sha256=source_hash,
            )
            return PolyglotASTExtraction(
                record=record,
                language=normalized_language,
                producer="stdlib-json",
                producer_version="stdlib-json@1",
            )
        return self._extract_typescript(
            text,
            normalized_language,
            blob_identity=blob,
            source_sha256=source_hash,
        )

    def extract(
        self,
        source: str | bytes | Mapping[str, Any],
        language: str,
        *,
        blob_identity: str = "",
        source_sha256: str = "",
    ) -> ASTBlobRecord:
        return self.extract_with_metadata(
            source,
            language,
            blob_identity=blob_identity,
            source_sha256=source_sha256,
        ).record

    extract_source = extract
    build_ast_blob_record = extract

    def open_typescript_worker(self) -> TypeScriptPersistentWorker:
        """Start (or return) the bounded persistent TypeScript worker."""

        if self._typescript_worker is None:
            self._typescript_worker = TypeScriptPersistentWorker(
                node_executable=self.node_executable,
                extractor_path=self.extractor_path,
                typescript_path=self.typescript_path,
                node_memory_mib=self.limits.node_memory_mib,
                max_output_bytes=self.limits.max_output_bytes,
                process_timeout_seconds=self.limits.process_timeout_seconds,
            )
            self._typescript_worker.start()
        return self._typescript_worker

    def close_typescript_worker(self) -> None:
        worker = self._typescript_worker
        self._typescript_worker = None
        if worker is not None:
            worker.close()

    def _extract_typescript(
        self,
        source: str,
        language: str,
        *,
        blob_identity: str,
        source_sha256: str,
    ) -> PolyglotASTExtraction:
        request_body = {
            "protocol_version": TYPESCRIPT_EXTRACTOR_PROTOCOL_VERSION,
            "language": language,
            "source": source,
            "source_sha256": source_sha256,
        }
        if self._persistent_typescript_worker or self._typescript_worker is not None:
            payload = self.open_typescript_worker().request(request_body)
            return_code = 0
        else:
            node = shutil.which(self.node_executable)
            if node is None:
                raise PolyglotASTProviderError(
                    PolyglotASTReason.NODE_UNAVAILABLE,
                    f"Node executable {self.node_executable!r} was not found",
                )
            if not self.extractor_path.is_file():
                raise PolyglotASTProviderError(
                    PolyglotASTReason.EXTRACTOR_UNAVAILABLE,
                    f"TypeScript extractor is missing: {self.extractor_path}",
                )
            request = _json_bytes(request_body)
            environment = dict(os.environ)
            environment["NO_COLOR"] = "1"
            environment["POLYGLOT_AST_MAX_INPUT_BYTES"] = str(len(request))
            if self.typescript_path:
                environment["TYPESCRIPT_PATH"] = self.typescript_path
            command = [
                node,
                f"--max-old-space-size={self.limits.node_memory_mib}",
                str(self.extractor_path),
            ]
            try:
                return_code, stdout, stderr = self._process_runner(
                    command,
                    request,
                    self.limits.process_timeout_seconds,
                    self.limits.max_output_bytes,
                    environment,
                )
            except PolyglotASTProviderError:
                raise
            except OSError as exc:
                raise PolyglotASTProviderError(
                    PolyglotASTReason.PROCESS_FAILED,
                    "failed to start the TypeScript extractor",
                ) from exc
            try:
                payload = json.loads(stdout.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                message = stderr.decode("utf-8", errors="replace").strip()[:512]
                raise PolyglotASTProviderError(
                    PolyglotASTReason.PROCESS_FAILED
                    if return_code
                    else PolyglotASTReason.PROTOCOL_ERROR,
                    "extractor did not return one valid UTF-8 JSON object"
                    + (f": {message}" if message else ""),
                    details={"return_code": return_code},
                ) from exc
            if not isinstance(payload, Mapping):
                raise PolyglotASTProviderError(
                    PolyglotASTReason.PROTOCOL_ERROR,
                    "extractor response must be an object",
                )
        if payload.get("protocol_version") != TYPESCRIPT_EXTRACTOR_PROTOCOL_VERSION:
            raise PolyglotASTProviderError(
                PolyglotASTReason.PROTOCOL_ERROR,
                "extractor protocol version mismatch",
            )
        if payload.get("ok") is not True:
            error = payload.get("error")
            code = (
                str(error.get("code"))
                if isinstance(error, Mapping) and error.get("code")
                else PolyglotASTReason.PROCESS_FAILED.value
            )
            allowed = {item.value for item in PolyglotASTReason}
            reason = code if code in allowed else PolyglotASTReason.PROCESS_FAILED
            message = (
                str(error.get("message"))
                if isinstance(error, Mapping)
                else "TypeScript extraction failed"
            )
            raise PolyglotASTProviderError(
                reason,
                message[:1024],
                details={"return_code": return_code},
            )
        if return_code:
            raise PolyglotASTProviderError(
                PolyglotASTReason.PROCESS_FAILED,
                f"extractor exited with status {return_code}",
            )
        if payload.get("producer") != "typescript-compiler-api":
            raise PolyglotASTProviderError(
                PolyglotASTReason.PROTOCOL_ERROR,
                "unexpected TypeScript producer identity",
            )
        if payload.get("producer_version") != TYPESCRIPT_EXTRACTOR_VERSION:
            raise PolyglotASTProviderError(
                PolyglotASTReason.PROTOCOL_ERROR,
                "unexpected TypeScript extractor version",
            )
        compiler = payload.get("compiler")
        if (
            not isinstance(compiler, Mapping)
            or compiler.get("name") != "typescript"
            or not isinstance(compiler.get("version"), str)
            or not compiler["version"].strip()
        ):
            raise PolyglotASTProviderError(
                PolyglotASTReason.PROTOCOL_ERROR,
                "extractor omitted its TypeScript compiler identity",
            )
        compiler_version = compiler["version"].strip()
        if (
            self.expected_typescript_version
            and compiler_version != self.expected_typescript_version
        ):
            raise PolyglotASTProviderError(
                PolyglotASTReason.COMPILER_VERSION_MISMATCH,
                "TypeScript compiler version does not match the configured version",
                details={
                    "expected": self.expected_typescript_version,
                    "actual": compiler_version,
                },
            )
        if payload.get("source_sha256") != source_sha256:
            raise PolyglotASTProviderError(
                PolyglotASTReason.SOURCE_IDENTITY_MISMATCH,
                "extractor response is not bound to the requested source",
            )
        if payload.get("language") != language:
            raise PolyglotASTProviderError(
                PolyglotASTReason.PROTOCOL_ERROR,
                "extractor response language mismatch",
            )
        facts = payload.get("facts")
        if not isinstance(facts, Mapping):
            raise PolyglotASTProviderError(
                PolyglotASTReason.PROTOCOL_ERROR,
                "extractor response omitted AST facts",
            )
        parse_error = payload.get("parse_error") or ""
        if not isinstance(parse_error, str):
            raise PolyglotASTProviderError(
                PolyglotASTReason.PROTOCOL_ERROR,
                "extractor parse_error must be text",
            )
        record = ASTBlobRecord(
            blob_identity=blob_identity,
            source_sha256=source_sha256,
            qualified_symbols=_string_tuple(
                facts.get("qualified_symbols"), "qualified_symbols"
            ),
            imports=_string_tuple(facts.get("imports"), "imports"),
            calls=_string_tuple(facts.get("calls"), "calls"),
            state_transitions=_string_tuple(
                facts.get("state_transitions"), "state_transitions"
            ),
            interfaces=_string_tuple(facts.get("interfaces"), "interfaces"),
            symbol_hashes=_string_mapping(
                facts.get("symbol_hashes"), "symbol_hashes"
            ),
            symbol_lines=_line_mapping(facts.get("symbol_lines")),
            parse_error=parse_error,
            # Binding the compiler here makes ASTBlobRecord.record_id change
            # even for an empty source when the compiler changes.
            language=f"{language}@typescript-{compiler_version}",
        )
        return PolyglotASTExtraction(
            record=record,
            language=language,
            producer="typescript-compiler-api",
            producer_version=TYPESCRIPT_EXTRACTOR_VERSION,
            compiler_name="typescript",
            compiler_version=compiler_version,
        )

    def extract_many(
        self, inputs: Iterable[PolyglotASTInput | Mapping[str, Any]]
    ) -> tuple[ASTBlobRecord, ...]:
        materialized: list[PolyglotASTInput | Mapping[str, Any]] = []
        for index, item in enumerate(inputs):
            if index >= self.limits.max_files:
                raise PolyglotASTProviderError(
                    PolyglotASTReason.FILE_LIMIT_EXCEEDED,
                    f"batch exceeds {self.limits.max_files} files",
                )
            materialized.append(item)
        normalized: list[PolyglotASTInput] = []
        total = 0
        for item in materialized:
            if isinstance(item, PolyglotASTInput):
                candidate = item
            elif isinstance(item, Mapping):
                try:
                    candidate = PolyglotASTInput(
                        source=item["source"],
                        language=str(item["language"]),
                        blob_identity=str(item.get("blob_identity") or ""),
                        source_sha256=str(item.get("source_sha256") or ""),
                    )
                except KeyError as exc:
                    raise PolyglotASTProviderError(
                        PolyglotASTReason.INVALID_SOURCE,
                        "batch inputs require source and language",
                    ) from exc
            else:
                raise PolyglotASTProviderError(
                    PolyglotASTReason.INVALID_SOURCE,
                    "batch inputs must be PolyglotASTInput values or mappings",
                )
            text, size = self._bounded_source(candidate.source)
            total += size
            if total > self.limits.max_total_bytes:
                raise PolyglotASTProviderError(
                    PolyglotASTReason.TOTAL_BYTES_EXCEEDED,
                    f"batch exceeds {self.limits.max_total_bytes} UTF-8 bytes",
                    details={"actual_bytes": total},
                )
            normalized.append(
                PolyglotASTInput(
                    source=text,
                    language=candidate.language,
                    blob_identity=candidate.blob_identity,
                    source_sha256=candidate.source_sha256,
                )
            )
        return tuple(
            self.extract(
                item.source,
                item.language,
                blob_identity=item.blob_identity,
                source_sha256=item.source_sha256,
            )
            for item in normalized
        )

    def extract_files(
        self,
        files: Mapping[str, str | bytes],
        *,
        blob_identities: Mapping[str, str] | None = None,
    ) -> dict[str, ASTBlobRecord]:
        """Extract a bounded path map while keeping paths outside AST records."""

        if len(files) > self.limits.max_files:
            raise PolyglotASTProviderError(
                PolyglotASTReason.FILE_LIMIT_EXCEEDED,
                f"batch exceeds {self.limits.max_files} files",
            )
        ordered_paths = sorted(str(path) for path in files)
        inputs: list[PolyglotASTInput] = []
        for path in ordered_paths:
            inputs.append(
                PolyglotASTInput(
                    source=files[path],
                    language=language_for_path(path),
                    blob_identity=str((blob_identities or {}).get(path) or ""),
                )
            )
        records = self.extract_many(inputs)
        return dict(zip(ordered_paths, records))


def build_polyglot_ast_blob_record(
    source: str | bytes | Mapping[str, Any],
    language: str,
    *,
    blob_identity: str = "",
    source_sha256: str = "",
    provider: PolyglotASTProvider | None = None,
) -> ASTBlobRecord:
    """Convenience adapter that preserves the canonical record interface."""

    return (provider or PolyglotASTProvider()).extract(
        source,
        language,
        blob_identity=blob_identity,
        source_sha256=source_sha256,
    )


def build_structured_schema_ast_blob_record(
    source: str | bytes | Mapping[str, Any],
    *,
    language: str = "json-schema",
    blob_identity: str = "",
    source_sha256: str = "",
) -> ASTBlobRecord:
    """Build deterministic facts for a JSON Schema or JSON OpenAPI document."""

    return PolyglotASTProvider().extract(
        source,
        language,
        blob_identity=blob_identity,
        source_sha256=source_sha256,
    )


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "DEFAULT_PROCESS_TIMEOUT_SECONDS",
    "HARD_MAX_FILE_BYTES",
    "POLYGLOT_AST_PROVIDER_SCHEMA",
    "TYPESCRIPT_EXTRACTOR_PROTOCOL_VERSION",
    "TYPESCRIPT_EXTRACTOR_VERSION",
    "PolyglotASTExtraction",
    "PolyglotASTInput",
    "PolyglotASTLimits",
    "PolyglotASTProvider",
    "PolyglotASTProviderError",
    "PolyglotASTReason",
    "TypeScriptPersistentWorker",
    "build_polyglot_ast_blob_record",
    "build_structured_schema_ast_blob_record",
    "language_for_path",
]
