"""Codex proposal adapter bound to the installed ``codex exec`` mechanism (PCCE-031).

The supported installed integration discovered at implementation time is
``ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration`` with
command contract ``codex exec``. This adapter consumes only an admitted
TaskSpecification, ContextPack, and ModelRouteDecision and returns a
schema-valid PatchProposal bound to a CodingAgentInvocation.

Importing this module performs no I/O, network, process, or filesystem
mutation. Live ``codex exec`` is invoked only through an explicit
task-scoped permit. Recorded/fake transports never claim live provenance.
The adapter has no approval or canonical-branch authority and does not
silently substitute another provider mechanism.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Protocol

from ipfs_accelerate_py.proof_context.adapters.base import (
    ADAPTER_CONTRACT_CID,
    APPROVAL_AUTHORITY,
    CANONICAL_BRANCH_AUTHORITY,
    AdapterResult,
    CancellationToken,
    INTERFACE,
    admit_adapter_result,
    bind_adapter_request,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    CODING_AGENT_INVOCATION_SCHEMA,
    CONTEXT_PACK_SCHEMA,
    FORBIDDEN_WIRE_FIELDS,
    MAX_LOG_BYTES,
    MAX_PATCH_BYTES,
    MODEL_ROUTE_DECISION_SCHEMA,
    PATCH_PROPOSAL_SCHEMA,
    TASK_SPECIFICATION_SCHEMA,
    CodingAgentInvocation,
    ContextPack,
    ModelRouteDecision,
    PatchProposal,
    TaskSpecification,
    admit_bounded_log,
    admit_bounded_patch,
    admit_cid,
    admit_non_negative_int,
    admit_path_list,
    admit_relative_path,
    assert_declared_scope,
    wire_canonical_utf8,
)
from ipfs_accelerate_py.proof_context.errors import (
    REDACTED,
    BoundaryViolationError,
    IdentityInconsistentError,
    MalformedError,
    SimulatedPromotedError,
    UnavailableCapabilityError,
    from_provider_error,
)

ADAPTER: Final[str] = "CodexAdapter@0.1"
ADAPTER_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/codex-adapter"
COMMAND_CONTRACT: Final[str] = "codex exec"
SUPPORTED_MECHANISM: Final[str] = COMMAND_CONTRACT
SUPPORTED_INTEGRATION_MODULE: Final[str] = (
    "ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration"
)
SUPPORTED_INTEGRATION_CLASS: Final[str] = "OpenAICodexCLIIntegration"
SUPPORTED_ARGV_BUILDER: Final[str] = "build_codex_exec_argv"
SUPPORTED_REGISTRY_NAME: Final[str] = "openai_codex"
CODEX_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"codex", "codex_cli", "openai_codex"}
)
LIVE_PERMIT_ENV: Final[str] = "IPFS_ACCELERATE_PCCE_CODEX_LIVE_PERMIT"
LIVE_PERMIT_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes"})
DEFAULT_CODEX_BINARY: Final[str] = "codex"
VERSION_PROBE_TIMEOUT_SECONDS: Final[int] = 5
LIVE_INVOKE_TIMEOUT_SECONDS: Final[int] = 120
MAX_RESPONSE_CHARS: Final[int] = MAX_PATCH_BYTES

_SECRET_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|"
    r"authorization|bearer|private[_-]?key|credential)s?\s*([:=]|%3[dD])\s*\S+"
)
_BEARER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(bearer|sk-|ghp_|xox[baprs]-)\s*[A-Za-z0-9_\-./+=]{8,}"
)
_DIFF_GIT: Final[re.Pattern[str]] = re.compile(
    r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE
)
_DIFF_PLUS: Final[re.Pattern[str]] = re.compile(
    r"^\+\+\+ (?:b/)?(.+)$", re.MULTILINE
)
_JSON_FENCE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL
)

_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "command_contract",
        "task_id",
        "objective_id",
        "repository_state_cid",
        "pack_cid",
        "route_cid",
        "owned_paths",
        "declared_files",
        "provider",
        "model",
        "revision",
        "tier",
        "task",
        "context_pack",
        "route",
        "instruction",
    }
)


def _mint_cid(value: Mapping[str, Any] | bytes) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        digest = hashlib.sha256(bytes(value)).digest()
    else:
        digest = hashlib.sha256(wire_canonical_utf8(value).encode("utf-8")).digest()
    raw = bytes([0x01, 0x55, 0x12, 0x20]) + digest
    return "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")


def _same_identity(left: str | None, right: str | None, *, field: str) -> None:
    if left is None or right is None:
        return
    if left != right:
        raise IdentityInconsistentError(
            f"Codex identity field {field} drifted",
            details={"field": field},
        )


def live_permit_granted(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    value = str(env.get(LIVE_PERMIT_ENV, "")).strip().lower()
    return value in LIVE_PERMIT_VALUES


def redact_log_text(value: str) -> str:
    """Redact secrets from adapter logs without applying error-message truncation."""

    text = value.replace("\x00", "")
    text = _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}={REDACTED}", text)
    text = _BEARER_PATTERN.sub(REDACTED, text)
    return text


def bound_and_redact_log(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, (bytes, bytearray, memoryview)):
        text = bytes(value).decode("utf-8", "replace")
    else:
        text = str(value)
    redacted = redact_log_text(text).encode("utf-8")
    if len(redacted) > MAX_LOG_BYTES:
        redacted = redacted[:MAX_LOG_BYTES]
    return admit_bounded_log(redacted)


def _reject_forbidden_fields(payload: Mapping[str, Any], *, what: str) -> None:
    extra = FORBIDDEN_WIRE_FIELDS.intersection(payload)
    if extra:
        name = sorted(extra)[0]
        raise BoundaryViolationError(
            f"{what} cannot carry approval, credentials, or hidden evaluation field {name!r}",
            details={"field": name, "reason": "forbidden_field"},
        )


def extract_patch_paths(patch: bytes | str) -> tuple[str, ...]:
    text = patch.decode("utf-8", "replace") if isinstance(patch, (bytes, bytearray)) else patch
    paths: list[str] = []
    seen: set[str] = set()
    for match in _DIFF_GIT.finditer(text):
        for raw in match.groups():
            if raw in {"/dev/null", "dev/null"}:
                continue
            path = admit_relative_path(raw, field="patch_path")
            if path not in seen:
                seen.add(path)
                paths.append(path)
    if paths:
        return tuple(paths)
    for match in _DIFF_PLUS.finditer(text):
        raw = match.group(1).strip()
        if raw in {"/dev/null", "dev/null"}:
            continue
        path = admit_relative_path(raw, field="patch_path")
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return tuple(paths)


def parse_structured_proposal(text: str) -> Mapping[str, Any]:
    """Parse a fail-closed JSON proposal object from Codex output."""

    if not isinstance(text, str) or not text.strip():
        raise MalformedError("Codex response is empty")
    if len(text) > MAX_RESPONSE_CHARS:
        raise BoundaryViolationError(
            "Codex response exceeds the frozen byte bound",
            details={"reason": "provider_output_bound"},
        )
    candidates = [text.strip()]
    fence = _JSON_FENCE.search(text)
    if fence is not None:
        candidates.append(fence.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    seen: set[str] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(value, dict):
            _reject_forbidden_fields(value, what="Codex response")
            return value
        last_error = MalformedError("Codex response JSON is not an object")
    raise MalformedError("Codex response is not a structured JSON proposal") from last_error


@dataclass(frozen=True)
class CodexMechanismProbe:
    """Truthful discovery record for the supported Codex integration."""

    command_contract: str
    module: str
    available: bool
    status: str
    reason: str
    module_available: bool = False
    builder_available: bool = False
    binary_available: bool = False
    binary_path: str | None = None
    version: str | None = None
    substituted: bool = False

    def to_mapping(self) -> Mapping[str, Any]:
        payload = {
            "command_contract": self.command_contract,
            "module": self.module,
            "available": self.available,
            "status": self.status,
            "reason": self.reason,
            "module_available": self.module_available,
            "builder_available": self.builder_available,
            "binary_available": self.binary_available,
            "binary_path": self.binary_path,
            "version": self.version,
            "substituted": self.substituted,
        }
        return MappingProxyType({key: value for key, value in payload.items() if value is not None})


def _load_supported_integration() -> Any | None:
    spec = importlib.util.find_spec(SUPPORTED_INTEGRATION_MODULE)
    if spec is None:
        return None
    try:
        return importlib.import_module(SUPPORTED_INTEGRATION_MODULE)
    except Exception:
        return None


def discover_supported_mechanism(
    *,
    which: Callable[[str], str | None] | None = None,
    binary_name: str = DEFAULT_CODEX_BINARY,
) -> CodexMechanismProbe:
    """Discover the supported ``codex exec`` integration without spawning a process."""

    lookup = shutil.which if which is None else which
    module = _load_supported_integration()
    module_available = module is not None
    builder_available = bool(
        module_available
        and hasattr(module, SUPPORTED_ARGV_BUILDER)
        and hasattr(module, SUPPORTED_INTEGRATION_CLASS)
    )
    binary_path = lookup(binary_name) if callable(lookup) else None
    binary_available = bool(binary_path)
    if not module_available:
        return CodexMechanismProbe(
            command_contract=COMMAND_CONTRACT,
            module=SUPPORTED_INTEGRATION_MODULE,
            available=False,
            status="unavailable",
            reason="unavailable",
            module_available=False,
            builder_available=False,
            binary_available=binary_available,
            binary_path=binary_path,
        )
    if not builder_available:
        return CodexMechanismProbe(
            command_contract=COMMAND_CONTRACT,
            module=SUPPORTED_INTEGRATION_MODULE,
            available=False,
            status="unavailable",
            reason="unavailable",
            module_available=True,
            builder_available=False,
            binary_available=binary_available,
            binary_path=binary_path,
        )
    available = builder_available and binary_available
    return CodexMechanismProbe(
        command_contract=COMMAND_CONTRACT,
        module=SUPPORTED_INTEGRATION_MODULE,
        available=available,
        status="available" if available else "unavailable",
        reason="available" if available else "unavailable",
        module_available=True,
        builder_available=True,
        binary_available=binary_available,
        binary_path=binary_path,
    )


def probe_supported_mechanism(
    *,
    version_probe: bool = False,
    which: Callable[[str], str | None] | None = None,
    runner: Callable[[Sequence[str]], Any] | None = None,
    binary_name: str = DEFAULT_CODEX_BINARY,
    binary_path: str | None = None,
) -> CodexMechanismProbe:
    """Optional bounded ``codex --version`` probe. Never substitutes another mechanism."""

    discovered = discover_supported_mechanism(which=which, binary_name=binary_name)
    path = binary_path or discovered.binary_path
    if not version_probe:
        return discovered
    if not path:
        return CodexMechanismProbe(
            command_contract=discovered.command_contract,
            module=discovered.module,
            available=False,
            status="unavailable",
            reason="unavailable",
            module_available=discovered.module_available,
            builder_available=discovered.builder_available,
            binary_available=False,
            binary_path=None,
            version=None,
        )
    argv = [path, "--version"]
    try:
        if runner is not None:
            completed = runner(argv)
        else:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=VERSION_PROBE_TIMEOUT_SECONDS,
                shell=False,
                check=False,
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return CodexMechanismProbe(
            command_contract=discovered.command_contract,
            module=discovered.module,
            available=False,
            status="unavailable",
            reason="unavailable",
            module_available=discovered.module_available,
            builder_available=discovered.builder_available,
            binary_available=False,
            binary_path=path,
        )
    stdout = str(getattr(completed, "stdout", "") or "").strip()
    returncode = int(getattr(completed, "returncode", 1) or 0)
    version = stdout.splitlines()[0].strip() if stdout else None
    available = returncode == 0 and bool(version) and discovered.builder_available
    return CodexMechanismProbe(
        command_contract=COMMAND_CONTRACT,
        module=SUPPORTED_INTEGRATION_MODULE,
        available=available,
        status="available" if available else "unavailable",
        reason="available" if available else "unavailable",
        module_available=discovered.module_available,
        builder_available=discovered.builder_available,
        binary_available=available,
        binary_path=path,
        version=version,
    )


class CodexTransport(Protocol):
    """Transport that speaks only the supported ``codex exec`` contract."""

    kind: str
    command_contract: str

    def invoke(
        self,
        request: Mapping[str, Any],
        cancellation: CancellationToken | None = None,
    ) -> Mapping[str, Any]: ...

    def cancel(self, cancellation: CancellationToken) -> None: ...


def _require_codex_contract(command_contract: str) -> None:
    if command_contract != COMMAND_CONTRACT:
        raise UnavailableCapabilityError(
            "Codex adapter refuses to substitute another mechanism",
            details={"capability": COMMAND_CONTRACT, "reason": "unavailable"},
        )


@dataclass
class RecordedCodexTransport:
    """Fake-client / recorded-fixture transport. Never live."""

    response: Mapping[str, Any] | Callable[[Mapping[str, Any]], Mapping[str, Any]]
    command_contract: str = COMMAND_CONTRACT
    kind: str = "recorded"
    log: str | bytes = ""
    latency_ms: int = 9
    requests: list[Mapping[str, Any]] = field(default_factory=list)

    def invoke(
        self,
        request: Mapping[str, Any],
        cancellation: CancellationToken | None = None,
    ) -> Mapping[str, Any]:
        if cancellation is not None:
            cancellation.check()
        _require_codex_contract(self.command_contract)
        self.requests.append(MappingProxyType(dict(request)))
        if callable(self.response):
            payload = self.response(request)
        else:
            payload = dict(self.response)
        if not isinstance(payload, Mapping):
            raise MalformedError("recorded Codex transport must return a mapping")
        result = dict(payload)
        result.setdefault("log", self.log)
        result.setdefault("latency_ms", self.latency_ms)
        result.setdefault("command_contract", self.command_contract)
        result.setdefault("kind", self.kind)
        if cancellation is not None:
            cancellation.check()
        return result

    def cancel(self, cancellation: CancellationToken) -> None:
        cancellation.cancel()


@dataclass
class UnavailableCodexTransport:
    """Transport that reports the supported integration as unavailable."""

    command_contract: str = COMMAND_CONTRACT
    kind: str = "unavailable"
    reason: str = "unavailable"

    def invoke(
        self,
        request: Mapping[str, Any],
        cancellation: CancellationToken | None = None,
    ) -> Mapping[str, Any]:
        _ = request
        if cancellation is not None:
            cancellation.check()
        raise UnavailableCapabilityError(
            "supported Codex integration is unavailable",
            details={"capability": COMMAND_CONTRACT, "reason": self.reason},
        )

    def cancel(self, cancellation: CancellationToken) -> None:
        cancellation.cancel()


def _isolated_live_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    env = {
        "PATH": source.get("PATH", "/usr/bin"),
        "LC_ALL": "C",
        "LANG": "C",
    }
    home = source.get("HOME")
    if home:
        env["HOME"] = home
    if live_permit_granted(source):
        for key in ("CODEX_HOME", "OPENAI_API_KEY"):
            value = source.get(key)
            if value:
                env[key] = value
    return env


@dataclass
class InstalledCodexTransport:
    """Live transport using the installed ``codex exec`` argv builder.

    Construction does not spawn a process. Invoke requires an explicit live
    permit and never falls back to another CLI or HTTP mechanism.
    """

    command_contract: str = COMMAND_CONTRACT
    kind: str = "live"
    permit_live: bool = False
    timeout_seconds: int = LIVE_INVOKE_TIMEOUT_SECONDS
    binary_path: str | None = None
    sandbox: str = "read-only"
    environ: Mapping[str, str] | None = None
    runner: Callable[..., Any] | None = None

    def invoke(
        self,
        request: Mapping[str, Any],
        cancellation: CancellationToken | None = None,
    ) -> Mapping[str, Any]:
        if cancellation is not None:
            cancellation.check()
        _require_codex_contract(self.command_contract)
        if not self.permit_live:
            raise UnavailableCapabilityError(
                "live Codex invocation requires an explicit task-scoped permit",
                details={"capability": COMMAND_CONTRACT, "reason": "unavailable"},
            )
        module = _load_supported_integration()
        builder = getattr(module, SUPPORTED_ARGV_BUILDER, None) if module is not None else None
        if builder is None:
            raise UnavailableCapabilityError(
                "supported Codex argv builder is unavailable",
                details={"capability": COMMAND_CONTRACT, "reason": "unavailable"},
            )
        binary = self.binary_path or shutil.which(DEFAULT_CODEX_BINARY)
        if not binary:
            raise UnavailableCapabilityError(
                "supported Codex binary is unavailable",
                details={"capability": COMMAND_CONTRACT, "reason": "unavailable"},
            )
        prompt = wire_canonical_utf8(dict(request))
        last_message_path = os.path.join(
            os.environ.get("TMPDIR", "/tmp"),
            f"pcce-codex-{os.getpid()}-{time.time_ns()}.txt",
        )
        argv = builder(
            base_argv=[binary],
            model=str(request.get("model") or ""),
            prompt=prompt,
            last_message_path=last_message_path,
            sandbox=self.sandbox,
            skip_git_repo_check=True,
            json_mode=False,
        )
        if "exec" not in [part.lower() for part in argv]:
            raise UnavailableCapabilityError(
                "Codex adapter refuses to substitute another mechanism",
                details={"capability": COMMAND_CONTRACT, "reason": "unavailable"},
            )
        if cancellation is not None:
            cancellation.check()
        started = time.monotonic()
        try:
            if self.runner is not None:
                completed = self.runner(
                    argv,
                    stdin=prompt,
                    env=_isolated_live_env(self.environ),
                    timeout=self.timeout_seconds,
                )
            else:
                completed = subprocess.run(
                    list(argv),
                    input=prompt,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=self.timeout_seconds,
                    shell=False,
                    env=_isolated_live_env(self.environ),
                )
        except subprocess.TimeoutExpired as exc:
            raise from_provider_error(exc, code="timeout") from exc
        except FileNotFoundError as exc:
            raise from_provider_error(exc, code="unavailable_capability") from exc
        except OSError as exc:
            raise from_provider_error(exc, code="infrastructure_failure") from exc
        finally:
            if cancellation is not None:
                cancellation.check()
        latency_ms = max(0, int(round((time.monotonic() - started) * 1000)))
        stdout = str(getattr(completed, "stdout", "") or "")
        stderr = str(getattr(completed, "stderr", "") or "")
        text_out = ""
        try:
            with open(last_message_path, "r", encoding="utf-8", errors="replace") as handle:
                text_out = handle.read()
        except OSError:
            text_out = stdout
        finally:
            try:
                os.unlink(last_message_path)
            except OSError:
                pass
        if cancellation is not None:
            cancellation.check()
        returncode = int(getattr(completed, "returncode", 1) or 0)
        body = text_out.strip() or stdout.strip()
        if returncode != 0 and not body:
            raise UnavailableCapabilityError(
                "supported Codex integration returned no proposal",
                details={"capability": COMMAND_CONTRACT, "reason": "unavailable"},
            )
        log = bound_and_redact_log(f"{stdout}\n{stderr}")
        return {
            "text": body,
            "log": log.decode("utf-8", "replace"),
            "latency_ms": latency_ms,
            "command_contract": COMMAND_CONTRACT,
            "kind": self.kind,
            "returncode": returncode,
        }

    def cancel(self, cancellation: CancellationToken) -> None:
        cancellation.cancel()


def build_codex_request(
    task: TaskSpecification,
    context_pack: ContextPack,
    route: ModelRouteDecision,
) -> Mapping[str, Any]:
    """Build a closed transport request from admitted records only."""

    bind_adapter_request(task, context_pack, route)
    if route.provider not in CODEX_PROVIDERS:
        raise BoundaryViolationError(
            "Codex adapter cannot serve a non-Codex route",
            details={"field": "provider", "reason": "mechanism_mismatch"},
        )
    declared = task.declared_files if task.declared_files is not None else task.owned_paths
    revision = route.revision or "unspecified"
    payload = {
        "schema": "ipfs-accelerate.proof-context.v0.1/codex-transport-request",
        "command_contract": COMMAND_CONTRACT,
        "task_id": task.task_id,
        "objective_id": task.objective_id,
        "repository_state_cid": task.repository_state_cid,
        "pack_cid": context_pack.pack_cid,
        "route_cid": route.decision_cid,
        "owned_paths": list(task.owned_paths),
        "declared_files": list(declared),
        "provider": route.provider,
        "model": route.model,
        "revision": revision,
        "tier": route.tier,
        "task": dict(task.to_mapping()),
        "context_pack": dict(context_pack.to_mapping()),
        "route": dict(route.to_mapping()),
        "instruction": (
            "Propose a bounded unified-diff patch covering only owned_paths. "
            "Return JSON with task_id, repository_state_cid, pack_cid, route_cid, "
            "declared_files, patch, model, revision, token_count, cached_token_count, "
            "latency_ms, and cost_micros. Do not approve the patch."
        ),
    }
    _reject_forbidden_fields(payload, what="Codex request")
    extra = set(payload) - _REQUEST_KEYS
    if extra:
        raise UnknownRequestField(sorted(extra)[0])
    return MappingProxyType(payload)


class UnknownRequestField(MalformedError):
    """Internal closed-set violation while building a Codex request."""


def _as_int(value: Any, *, field: str, default: int) -> int:
    if value is None:
        return default
    return admit_non_negative_int(value, field=field)


def _response_text(payload: Mapping[str, Any]) -> str:
    for key in ("text", "stdout", "patch", "proposal"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            if key == "patch" and payload.get("declared_files") is not None:
                return json.dumps(
                    {name: payload[name] for name in payload if name in {
                        "task_id",
                        "repository_state_cid",
                        "pack_cid",
                        "route_cid",
                        "declared_files",
                        "patch",
                        "model",
                        "revision",
                        "token_count",
                        "cached_token_count",
                        "latency_ms",
                        "cost_micros",
                        "provenance",
                    }},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            return value
    return ""


def _structured_from_transport(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("declared_files") is not None and payload.get("patch") is not None:
        _reject_forbidden_fields(payload, what="Codex response")
        return payload
    text = _response_text(payload)
    return parse_structured_proposal(text)


@dataclass
class CodexAdapter:
    """CodingAgentAdapter for the installed Codex ``codex exec`` mechanism."""

    transport: CodexTransport | None = None
    permit_live: bool = False
    probe: CodexMechanismProbe | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", False)
        object.__setattr__(self, "approved", False)
        object.__setattr__(self, "canonical_branch_authority", CANONICAL_BRANCH_AUTHORITY)
        object.__setattr__(self, "approval_authority", APPROVAL_AUTHORITY)

    def cancel(self, cancellation: CancellationToken) -> None:
        cancellation.cancel()
        if self.transport is not None:
            self.transport.cancel(cancellation)

    def _resolve_transport(self) -> CodexTransport:
        if self.transport is not None:
            _require_codex_contract(self.transport.command_contract)
            return self.transport
        if not self.permit_live:
            raise UnavailableCapabilityError(
                "supported Codex integration is unavailable without a live permit or recorded transport",
                details={"capability": COMMAND_CONTRACT, "reason": "unavailable"},
            )
        discovered = self.probe or discover_supported_mechanism()
        if not discovered.available:
            raise UnavailableCapabilityError(
                "supported Codex integration is unavailable",
                details={"capability": COMMAND_CONTRACT, "reason": "unavailable"},
            )
        return InstalledCodexTransport(
            permit_live=True,
            binary_path=discovered.binary_path,
        )

    def propose(
        self,
        task: TaskSpecification,
        context_pack: ContextPack,
        route: ModelRouteDecision,
        cancellation: CancellationToken | None = None,
    ) -> AdapterResult:
        if cancellation is not None:
            cancellation.check()
        bind_adapter_request(task, context_pack, route)
        request = build_codex_request(task, context_pack, route)
        transport = self._resolve_transport()
        if cancellation is not None:
            cancellation.check()
        raw = transport.invoke(request, cancellation)
        if cancellation is not None:
            cancellation.check()
        if not isinstance(raw, Mapping):
            raise MalformedError("Codex transport must return a mapping")
        _reject_forbidden_fields(raw, what="Codex transport response")
        structured = _structured_from_transport(raw)
        return self._admit_transport_result(
            task,
            context_pack,
            route,
            request,
            transport,
            raw,
            structured,
            cancellation=cancellation,
        )

    def _admit_transport_result(
        self,
        task: TaskSpecification,
        context_pack: ContextPack,
        route: ModelRouteDecision,
        request: Mapping[str, Any],
        transport: CodexTransport,
        raw: Mapping[str, Any],
        structured: Mapping[str, Any],
        *,
        cancellation: CancellationToken | None,
    ) -> AdapterResult:
        if cancellation is not None:
            cancellation.check()
        _same_identity(task.task_id, structured.get("task_id"), field="task_id")
        _same_identity(
            task.repository_state_cid,
            structured.get("repository_state_cid"),
            field="repository_state_cid",
        )
        _same_identity(context_pack.pack_cid, structured.get("pack_cid"), field="pack_cid")
        _same_identity(route.decision_cid, structured.get("route_cid"), field="route_cid")
        declared = structured.get("declared_files")
        if declared is None:
            raise MalformedError("Codex proposal is missing declared_files")
        declared_files = admit_path_list(
            declared, field="declared_files", min_items=1, max_items=1024
        )
        extra_allow = task.declared_files
        assert_declared_scope(declared_files, task.owned_paths, extra_allow)
        patch_value = structured.get("patch")
        if not isinstance(patch_value, (str, bytes, bytearray)):
            raise MalformedError("Codex proposal is missing patch bytes")
        patch_bytes = admit_bounded_patch(patch_value)
        for path in extract_patch_paths(patch_bytes):
            assert_declared_scope((path,), task.owned_paths, extra_allow)
            if path not in declared_files:
                raise BoundaryViolationError(
                    "proposal patch path is not in declared files",
                    details={"field": "declared_files", "reason": "scope"},
                )
        log_bytes = bound_and_redact_log(raw.get("log") or structured.get("log") or b"")
        kind = str(raw.get("kind") or transport.kind)
        claimed = str(structured.get("provenance") or "")
        if kind in {"recorded", "replayed", "simulated"}:
            if claimed == "live":
                raise SimulatedPromotedError(
                    "recorded Codex results cannot claim live provenance"
                )
            provenance = claimed if claimed in {"replayed", "simulated"} else "replayed"
        elif kind == "live":
            provenance = "live"
        else:
            raise UnavailableCapabilityError(
                "supported Codex integration is unavailable",
                details={"capability": COMMAND_CONTRACT, "reason": "unavailable"},
            )
        model = str(structured.get("model") or route.model)
        revision = str(structured.get("revision") or route.revision or "unspecified")
        _same_identity(route.model, model, field="model")
        if route.revision is not None:
            _same_identity(route.revision, revision, field="revision")
        token_count = _as_int(structured.get("token_count"), field="token_count", default=0)
        cached_token_count = _as_int(
            structured.get("cached_token_count"), field="cached_token_count", default=0
        )
        latency_ms = _as_int(
            structured.get("latency_ms") or raw.get("latency_ms"),
            field="latency_ms",
            default=0,
        )
        cost_micros = _as_int(structured.get("cost_micros"), field="cost_micros", default=0)
        response_artifact_cid = None
        if provenance == "live":
            response_artifact_cid = _mint_cid(
                {
                    "command_contract": COMMAND_CONTRACT,
                    "request": dict(request),
                    "text": _response_text(raw) or json.dumps(
                        dict(structured), ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
            admit_cid(response_artifact_cid, field="response_artifact_cid")
        invocation_body = {
            "schema": CODING_AGENT_INVOCATION_SCHEMA,
            "task_id": task.task_id,
            "repository_state_cid": task.repository_state_cid,
            "route_cid": route.decision_cid,
            "provider": route.provider,
            "model": model,
            "revision": revision,
            "tier": route.tier,
            "token_count": token_count,
            "cached_token_count": cached_token_count,
            "latency_ms": latency_ms,
            "cost_micros": cost_micros,
            "provenance": provenance,
        }
        if response_artifact_cid is not None:
            invocation_body["response_artifact_cid"] = response_artifact_cid
        invocation_cid = _mint_cid(invocation_body)
        invocation = CodingAgentInvocation.from_mapping(
            {**invocation_body, "invocation_cid": invocation_cid}
        )
        patch_cid = _mint_cid(patch_bytes)
        proposal_body = {
            "schema": PATCH_PROPOSAL_SCHEMA,
            "task_id": task.task_id,
            "repository_state_cid": task.repository_state_cid,
            "invocation_cid": invocation_cid,
            "patch_cid": patch_cid,
            "declared_files": list(declared_files),
            "provenance": provenance,
        }
        proposal_cid = _mint_cid(proposal_body)
        proposal = PatchProposal.from_mapping({**proposal_body, "proposal_cid": proposal_cid})
        result = AdapterResult(
            proposal=proposal,
            invocation=invocation,
            patch_bytes=patch_bytes,
            log_bytes=log_bytes,
        )
        return admit_adapter_result(
            task,
            context_pack,
            route,
            result,
            cancellation=cancellation,
        )


_DESCRIPTOR_BODY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "schema": ADAPTER_SCHEMA,
        "adapter": ADAPTER,
        "interface": INTERFACE,
        "command_contract": COMMAND_CONTRACT,
        "supported_mechanism": SUPPORTED_MECHANISM,
        "supported_integration_module": SUPPORTED_INTEGRATION_MODULE,
        "supported_integration_class": SUPPORTED_INTEGRATION_CLASS,
        "supported_argv_builder": SUPPORTED_ARGV_BUILDER,
        "supported_registry_name": SUPPORTED_REGISTRY_NAME,
        "adapter_contract_cid": ADAPTER_CONTRACT_CID,
        "approval_authority": APPROVAL_AUTHORITY,
        "canonical_branch_authority": CANONICAL_BRANCH_AUTHORITY,
        "self_approval": False,
        "silently_substitutes_mechanism": False,
        "live_permit_env": LIVE_PERMIT_ENV,
        "task_specification_schema": TASK_SPECIFICATION_SCHEMA,
        "context_pack_schema": CONTEXT_PACK_SCHEMA,
        "model_route_decision_schema": MODEL_ROUTE_DECISION_SCHEMA,
        "coding_agent_invocation_schema": CODING_AGENT_INVOCATION_SCHEMA,
        "patch_proposal_schema": PATCH_PROPOSAL_SCHEMA,
        "max_log_bytes": MAX_LOG_BYTES,
        "max_patch_bytes": MAX_PATCH_BYTES,
    }
)
CODEX_ADAPTER_CID: Final[str] = _mint_cid(_DESCRIPTOR_BODY)
CODEX_ADAPTER_DESCRIPTOR: Final[Mapping[str, Any]] = MappingProxyType(
    {**dict(_DESCRIPTOR_BODY), "cid": CODEX_ADAPTER_CID}
)


def codex_adapter_descriptor() -> Mapping[str, Any]:
    return CODEX_ADAPTER_DESCRIPTOR


def codex_adapter_cid() -> str:
    return CODEX_ADAPTER_CID


def frozen_codex_adapter() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "adapter": ADAPTER,
            "cid": CODEX_ADAPTER_CID,
            "command_contract": COMMAND_CONTRACT,
            "approval_authority": APPROVAL_AUTHORITY,
            "canonical_branch_authority": CANONICAL_BRANCH_AUTHORITY,
            "adapter_contract_cid": ADAPTER_CONTRACT_CID,
        }
    )


__all__ = [
    "ADAPTER",
    "ADAPTER_SCHEMA",
    "CODEX_ADAPTER_CID",
    "CODEX_ADAPTER_DESCRIPTOR",
    "CODEX_PROVIDERS",
    "COMMAND_CONTRACT",
    "CodexAdapter",
    "CodexMechanismProbe",
    "CodexTransport",
    "InstalledCodexTransport",
    "LIVE_PERMIT_ENV",
    "RecordedCodexTransport",
    "SUPPORTED_ARGV_BUILDER",
    "SUPPORTED_INTEGRATION_CLASS",
    "SUPPORTED_INTEGRATION_MODULE",
    "SUPPORTED_MECHANISM",
    "SUPPORTED_REGISTRY_NAME",
    "UnavailableCodexTransport",
    "bound_and_redact_log",
    "build_codex_request",
    "codex_adapter_cid",
    "codex_adapter_descriptor",
    "discover_supported_mechanism",
    "extract_patch_paths",
    "frozen_codex_adapter",
    "live_permit_granted",
    "parse_structured_proposal",
    "probe_supported_mechanism",
    "redact_log_text",
]
