"""Reusable LLM subprocess helpers for optimizer todo daemons.

ASI-166: child process communication uses a bounded versioned JSON request
envelope and result file/pipe. Receipt IDs propagate without embedding prompts
or provider payloads. Off mode preserves the pre-ASI-166 wire behavior.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .llm_defaults import DEFAULT_CODEX_MODEL

from .engine import compact_message


LLM_CHILD_ENVELOPE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/todo-daemon-llm-child-envelope@1"
)
LLM_CHILD_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/todo-daemon-llm-child-result@1"
)
LLM_CHILD_ENVELOPE_VERSION = 1
MAX_ENVELOPE_BYTES = 16_384
MAX_RESULT_BYTES = 32_768
MAX_TEXT_FIELD_BYTES = 512
LLM_CHILD_PROVIDER_CAPACITY_REASON = "provider_capacity_unavailable"
MAX_LLM_CHILD_PROVIDER_BACKOFF_SECONDS = 31 * 24 * 60 * 60
_LLM_CHILD_PROVIDER_CAPACITY_REASONS = frozenset(
    {
        LLM_CHILD_PROVIDER_CAPACITY_REASON,
        "usage_limit",
        "quota_exceeded",
        "capacity_unavailable",
    }
)

# Modes mirror the provider execution gateway. ``off`` is the default and must
# remain behaviorally identical to the pre-ASI-166 child launch path.
LLM_USAGE_MODE_OFF = "off"
LLM_USAGE_MODE_OBSERVE = "observe"
LLM_USAGE_MODE_SHADOW = "shadow"
LLM_USAGE_MODE_ASSIST = "assist"
LLM_USAGE_MODE_ENFORCE = "enforce"
_VALID_USAGE_MODES = frozenset(
    {
        LLM_USAGE_MODE_OFF,
        LLM_USAGE_MODE_OBSERVE,
        LLM_USAGE_MODE_SHADOW,
        LLM_USAGE_MODE_ASSIST,
        LLM_USAGE_MODE_ENFORCE,
    }
)

_FORBIDDEN_ENVELOPE_KEYS = frozenset(
    {
        "prompt",
        "messages",
        "message",
        "source",
        "media",
        "output",
        "output_text",
        "completion",
        "payload",
        "raw_body",
        "raw_headers",
        "response_body",
        "credential",
        "credentials",
        "password",
        "secret",
        "api_key",
        "authorization",
        "token",
        "endpoint",
        "url",
        "uri",
    }
)


class LlmChildProviderCapacityError(RuntimeError):
    """Secret-safe capacity failure returned by the isolated LLM child."""

    reason_code = "reviewer_provider_capacity_unavailable"

    def __init__(
        self,
        *,
        provider_id: str,
        reason_codes: Sequence[str] = (),
        next_eligible_at: str = "",
    ) -> None:
        safe_reasons = tuple(
            str(item).strip()
            for item in reason_codes
            if str(item).strip() in _LLM_CHILD_PROVIDER_CAPACITY_REASONS
        )
        self.provider_id = str(provider_id or "").strip()
        self.reason_codes = safe_reasons or (
            LLM_CHILD_PROVIDER_CAPACITY_REASON,
        )
        self.next_eligible_at = _normalize_next_eligible_at(
            next_eligible_at
        )
        suffix = (
            f" until {self.next_eligible_at}"
            if self.next_eligible_at
            else ""
        )
        super().__init__(
            "LLM provider capacity unavailable"
            + (f" for {self.provider_id}" if self.provider_id else "")
            + suffix
        )


@dataclass(frozen=True)
class LlmRouterInvocation:
    """Configuration for one isolated ``llm_router.generate_text`` call."""

    repo_root: Path
    model_name: str = DEFAULT_CODEX_MODEL
    provider: Optional[str] = None
    allow_local_fallback: bool = False
    timeout_seconds: int = 300
    max_new_tokens: int = 2048
    max_prompt_chars: int = 60000
    temperature: float = 0.1
    backend_env_name: str = "TODO_DAEMON_LLM_BACKEND"
    backend_default: str = "llm_router"
    backend_label: str = "todo daemon LLM backend"
    env_prefix: str = "TODO_DAEMON_LLM"
    prompt_file_prefix: str = "todo-daemon-llm-prompt-"
    result_file_prefix: str = "todo-daemon-llm-result-"
    envelope_file_prefix: str = "todo-daemon-llm-envelope-"
    python_executable: str = "python3"
    timeout_grace_seconds: int = 30
    prompt_overage_allowance: str = "\n\n[truncated]\n"
    trace: bool = False
    trace_dir: Optional[Path] = None
    reject_effective_provider_name: Optional[str] = "local_hf"
    required_effective_providers: Sequence[str] = ()
    # Force Codex CLI into its read-only sandbox for independent review calls.
    # This is intentionally an invocation-only control: it is applied to the
    # isolated child environment and never broadens the generic request
    # envelope or permits callers to select an arbitrary sandbox profile.
    codex_read_only: bool = False
    # ASI-166 usage-aware child envelope. Default ``off`` keeps legacy behavior.
    usage_mode: str = LLM_USAGE_MODE_OFF
    request_id: str = ""
    attempt: int = 1
    idempotency_key: str = ""
    supervisor_receipt_id: str = ""
    endpoint_receipt_id: str = ""
    catalog_revision: str = ""
    usage_revision: str = ""
    lease_id: str = ""
    fence_id: str = ""
    deadline_at: str = ""
    side_effect_boundary: str = "idempotent"
    write_result_envelope: bool = True
    # Appended to preserve positional constructors used by older integrations.
    allow_cross_provider_fallback: Optional[bool] = None
    child_file_prefix: str = "todo-daemon-llm-child-"


@dataclass(frozen=True)
class LlmChildRequestEnvelope:
    """Bounded versioned JSON request metadata for the LLM child process.

    The prompt body is intentionally omitted; it remains in a separate file so
    receipt/envelope surfaces never leak prompt content.
    """

    schema: str = LLM_CHILD_ENVELOPE_SCHEMA
    contract_version: int = LLM_CHILD_ENVELOPE_VERSION
    usage_mode: str = LLM_USAGE_MODE_OFF
    request_id: str = ""
    attempt: int = 1
    idempotency_key: str = ""
    model_name: str = ""
    provider: str = ""
    timeout_seconds: int = 300
    max_new_tokens: int = 2048
    temperature: float = 0.1
    allow_local_fallback: bool = False
    catalog_revision: str = ""
    usage_revision: str = ""
    lease_id: str = ""
    fence_id: str = ""
    deadline_at: str = ""
    side_effect_boundary: str = "idempotent"
    supervisor_receipt_id: str = ""
    endpoint_receipt_id: str = ""
    input_digest: str = ""
    result_file: str = ""
    # Appended for positional compatibility with the version-one envelope.
    allow_cross_provider_fallback: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "contract_version": int(self.contract_version),
            "usage_mode": str(self.usage_mode or LLM_USAGE_MODE_OFF),
            "request_id": str(self.request_id or ""),
            "attempt": int(self.attempt or 1),
            "idempotency_key": str(self.idempotency_key or ""),
            "model_name": str(self.model_name or ""),
            "provider": str(self.provider or ""),
            "timeout_seconds": int(self.timeout_seconds),
            "max_new_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "allow_local_fallback": bool(self.allow_local_fallback),
            "allow_cross_provider_fallback": bool(
                self.allow_cross_provider_fallback
            ),
            "catalog_revision": str(self.catalog_revision or ""),
            "usage_revision": str(self.usage_revision or ""),
            "lease_id": str(self.lease_id or ""),
            "fence_id": str(self.fence_id or ""),
            "deadline_at": str(self.deadline_at or ""),
            "side_effect_boundary": str(self.side_effect_boundary or "idempotent"),
            "supervisor_receipt_id": str(self.supervisor_receipt_id or ""),
            "endpoint_receipt_id": str(self.endpoint_receipt_id or ""),
            "input_digest": str(self.input_digest or ""),
            "result_file": str(self.result_file or ""),
        }
        _assert_envelope_safe(payload)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_ENVELOPE_BYTES:
            raise RuntimeError("LLM child request envelope exceeds size bound")
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LlmChildRequestEnvelope":
        if not isinstance(payload, Mapping):
            raise RuntimeError("LLM child request envelope must be an object")
        if payload.get("schema") != LLM_CHILD_ENVELOPE_SCHEMA:
            raise RuntimeError("unsupported LLM child request envelope schema")
        version = int(payload.get("contract_version") or 0)
        if version != LLM_CHILD_ENVELOPE_VERSION:
            raise RuntimeError("unsupported LLM child request envelope version")
        _assert_envelope_safe(payload)
        return cls(
            schema=LLM_CHILD_ENVELOPE_SCHEMA,
            contract_version=version,
            usage_mode=str(payload.get("usage_mode") or LLM_USAGE_MODE_OFF),
            request_id=str(payload.get("request_id") or ""),
            attempt=int(payload.get("attempt") or 1),
            idempotency_key=str(payload.get("idempotency_key") or ""),
            model_name=str(payload.get("model_name") or ""),
            provider=str(payload.get("provider") or ""),
            timeout_seconds=int(payload.get("timeout_seconds") or 300),
            max_new_tokens=int(payload.get("max_new_tokens") or 2048),
            temperature=float(payload.get("temperature") or 0.1),
            allow_local_fallback=bool(payload.get("allow_local_fallback", False)),
            allow_cross_provider_fallback=bool(
                payload.get("allow_cross_provider_fallback", True)
            ),
            catalog_revision=str(payload.get("catalog_revision") or ""),
            usage_revision=str(payload.get("usage_revision") or ""),
            lease_id=str(payload.get("lease_id") or ""),
            fence_id=str(payload.get("fence_id") or ""),
            deadline_at=str(payload.get("deadline_at") or ""),
            side_effect_boundary=str(
                payload.get("side_effect_boundary") or "idempotent"
            ),
            supervisor_receipt_id=str(payload.get("supervisor_receipt_id") or ""),
            endpoint_receipt_id=str(payload.get("endpoint_receipt_id") or ""),
            input_digest=str(
                payload.get("input_digest")
                or payload.get("prompt_file_digest")
                or ""
            ),
            result_file=str(payload.get("result_file") or ""),
        )


@dataclass(frozen=True)
class LlmChildResultEnvelope:
    """Bounded versioned JSON result written by the LLM child process.

    Carries receipt IDs and status only. The generated text is returned on
    stdout (and optionally ``text_chars`` length) so envelopes never embed
    model output bodies.
    """

    schema: str = LLM_CHILD_RESULT_SCHEMA
    contract_version: int = LLM_CHILD_ENVELOPE_VERSION
    usage_mode: str = LLM_USAGE_MODE_OFF
    request_id: str = ""
    attempt: int = 1
    idempotency_key: str = ""
    status: str = "ok"
    reason_codes: tuple[str, ...] = ()
    supervisor_receipt_id: str = ""
    endpoint_receipt_id: str = ""
    execution_result_id: str = ""
    effective_provider: str = ""
    text_chars: int = 0
    text_bytes: int = 0
    text_sha256: str = ""
    exit_code: int = 0
    # Appended for version-one positional compatibility.
    next_eligible_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "next_eligible_at",
            _normalize_next_eligible_at(self.next_eligible_at),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "contract_version": int(self.contract_version),
            "usage_mode": str(self.usage_mode or LLM_USAGE_MODE_OFF),
            "request_id": str(self.request_id or ""),
            "attempt": int(self.attempt or 1),
            "idempotency_key": str(self.idempotency_key or ""),
            "status": str(self.status or "ok"),
            "reason_codes": list(self.reason_codes or ()),
            "supervisor_receipt_id": str(self.supervisor_receipt_id or ""),
            "endpoint_receipt_id": str(self.endpoint_receipt_id or ""),
            "execution_result_id": str(self.execution_result_id or ""),
            "effective_provider": str(self.effective_provider or ""),
            "text_chars": int(self.text_chars or 0),
            "text_bytes": int(self.text_bytes or 0),
            "text_sha256": str(self.text_sha256 or ""),
            "exit_code": int(self.exit_code or 0),
            "next_eligible_at": str(self.next_eligible_at or ""),
        }
        _assert_envelope_safe(payload)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
            raise RuntimeError("LLM child result envelope exceeds size bound")
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LlmChildResultEnvelope":
        if not isinstance(payload, Mapping):
            raise RuntimeError("LLM child result envelope must be an object")
        if payload.get("schema") != LLM_CHILD_RESULT_SCHEMA:
            raise RuntimeError("unsupported LLM child result envelope schema")
        version = int(payload.get("contract_version") or 0)
        if version != LLM_CHILD_ENVELOPE_VERSION:
            raise RuntimeError("unsupported LLM child result envelope version")
        _assert_envelope_safe(payload)
        reasons = payload.get("reason_codes") or ()
        if isinstance(reasons, str):
            reason_tuple = (reasons,)
        elif isinstance(reasons, Sequence):
            reason_tuple = tuple(str(item) for item in reasons)
        else:
            reason_tuple = ()
        return cls(
            schema=LLM_CHILD_RESULT_SCHEMA,
            contract_version=version,
            usage_mode=str(payload.get("usage_mode") or LLM_USAGE_MODE_OFF),
            request_id=str(payload.get("request_id") or ""),
            attempt=int(payload.get("attempt") or 1),
            idempotency_key=str(payload.get("idempotency_key") or ""),
            status=str(payload.get("status") or "ok"),
            reason_codes=reason_tuple,
            supervisor_receipt_id=str(payload.get("supervisor_receipt_id") or ""),
            endpoint_receipt_id=str(payload.get("endpoint_receipt_id") or ""),
            execution_result_id=str(payload.get("execution_result_id") or ""),
            effective_provider=str(payload.get("effective_provider") or ""),
            text_chars=int(payload.get("text_chars") or 0),
            text_bytes=int(payload.get("text_bytes") or 0),
            text_sha256=str(payload.get("text_sha256") or ""),
            exit_code=int(payload.get("exit_code") or 0),
            next_eligible_at=str(payload.get("next_eligible_at") or ""),
        )


_ACTIVE_LLM_PROCESS: Optional[subprocess.Popen[Any]] = None
_LAST_LLM_RESULT: Optional[LlmChildResultEnvelope] = None
_LAST_LLM_RESULT_LOCK = threading.Lock()
DeadlineTimeoutCallback = Callable[[float, float, str], None]
DeadlineMessageBuilder = Callable[[float, float, str], str]


def _assert_envelope_safe(payload: Mapping[str, Any]) -> None:
    for key in payload:
        lowered = str(key).casefold()
        if lowered in _FORBIDDEN_ENVELOPE_KEYS:
            raise RuntimeError(
                f"LLM envelope must not embed forbidden field {key!r}"
            )
        if any(
            marker in lowered
            for marker in ("prompt", "password", "secret", "api_key", "credential")
        ):
            raise RuntimeError(
                f"LLM envelope must not embed forbidden field {key!r}"
            )


def _normalize_next_eligible_at(value: object) -> str:
    """Validate and canonicalize bounded, timezone-aware retry metadata."""

    text = str(value or "").strip()
    if not text or len(text.encode("utf-8")) > 64:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        return ""
    normalized = parsed.astimezone(timezone.utc)
    if (
        normalized - datetime.now(timezone.utc)
    ).total_seconds() > MAX_LLM_CHILD_PROVIDER_BACKOFF_SECONDS:
        return ""
    return normalized.isoformat().replace(
        "+00:00", "Z"
    )


def _normalize_usage_mode(value: str) -> str:
    mode = str(value or LLM_USAGE_MODE_OFF).strip().casefold()
    if mode not in _VALID_USAGE_MODES:
        raise RuntimeError(
            f"unsupported LLM usage_mode {value!r}; "
            f"expected one of {sorted(_VALID_USAGE_MODES)}"
        )
    return mode


def build_child_request_envelope(
    config: LlmRouterInvocation,
    *,
    input_digest: str = "",
    prompt_file_digest: str = "",
    result_file: str = "",
) -> LlmChildRequestEnvelope:
    """Construct a bounded request envelope from an invocation config."""

    return LlmChildRequestEnvelope(
        usage_mode=_normalize_usage_mode(config.usage_mode),
        request_id=str(config.request_id or ""),
        attempt=int(config.attempt or 1),
        idempotency_key=str(config.idempotency_key or ""),
        model_name=str(config.model_name or ""),
        provider=str(config.provider or ""),
        timeout_seconds=int(config.timeout_seconds),
        max_new_tokens=int(config.max_new_tokens),
        temperature=float(config.temperature),
        allow_local_fallback=bool(config.allow_local_fallback),
        allow_cross_provider_fallback=_allow_cross_provider_fallback(config),
        catalog_revision=str(config.catalog_revision or ""),
        usage_revision=str(config.usage_revision or ""),
        lease_id=str(config.lease_id or ""),
        fence_id=str(config.fence_id or ""),
        deadline_at=str(config.deadline_at or ""),
        side_effect_boundary=str(config.side_effect_boundary or "idempotent"),
        supervisor_receipt_id=str(config.supervisor_receipt_id or ""),
        endpoint_receipt_id=str(config.endpoint_receipt_id or ""),
        input_digest=str(input_digest or prompt_file_digest or ""),
        result_file=str(result_file or ""),
    )


def parse_child_result_envelope(
    value: str | bytes | bytearray | Mapping[str, Any],
) -> LlmChildResultEnvelope:
    """Parse a result envelope from JSON text or a mapping."""

    if isinstance(value, Mapping):
        return LlmChildResultEnvelope.from_dict(value)
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8")
    if not isinstance(value, str):
        raise RuntimeError("LLM child result envelope must be text or object")
    if len(value.encode("utf-8")) > MAX_RESULT_BYTES:
        raise RuntimeError("LLM child result envelope exceeds size bound")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM child result envelope JSON is malformed") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("LLM child result envelope JSON must contain an object")
    return LlmChildResultEnvelope.from_dict(payload)


def last_llm_child_result() -> Optional[LlmChildResultEnvelope]:
    """Return the most recent child result envelope, if any."""

    with _LAST_LLM_RESULT_LOCK:
        return _LAST_LLM_RESULT


def _set_last_llm_result(result: Optional[LlmChildResultEnvelope]) -> None:
    global _LAST_LLM_RESULT
    with _LAST_LLM_RESULT_LOCK:
        _LAST_LLM_RESULT = result


def collect_descendant_pids(pid: int) -> list[int]:
    """Return all descendant pids for ``pid`` using ``pgrep -P`` recursion."""

    try:
        completed = subprocess.run(
            ["pgrep", "-P", str(pid)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return []
    descendants: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            child = int(line.strip())
        except ValueError:
            continue
        descendants.append(child)
        descendants.extend(collect_descendant_pids(child))
    return descendants


def process_groups_for_family(root_pid: int) -> set[int]:
    """Return process groups owned by a process family rooted at ``root_pid``."""

    groups: set[int] = set()
    for pid in [root_pid, *collect_descendant_pids(root_pid)]:
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            continue
    return groups


def terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = 5.0,
) -> None:
    """Terminate a subprocess and descendant process groups when it owns a session."""

    if process.poll() is not None:
        return
    process_groups = process_groups_for_family(process.pid) or {process.pid}
    try:
        for pgid in process_groups:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        for pgid in process_groups:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def _env_name(config: LlmRouterInvocation, suffix: str) -> str:
    return f"{config.env_prefix}_{suffix}"


def _allow_cross_provider_fallback(config: LlmRouterInvocation) -> bool:
    """Resolve remote failover while preserving ordinary route compatibility."""

    if config.allow_cross_provider_fallback is None:
        return True
    return bool(config.allow_cross_provider_fallback)


def _canonical_accelerator_source_root() -> Path:
    """Return the source root containing this exact accelerator package."""

    source_root = Path(__file__).resolve().parents[3]
    package_root = source_root / "ipfs_accelerate_py"
    if not (package_root / "__init__.py").is_file():
        raise RuntimeError("canonical ipfs_accelerate_py source root is unavailable")
    return source_root


def _prompt_digest(prompt: str) -> str:
    import hashlib

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _llm_router_child_code(config: LlmRouterInvocation) -> str:
    canonical_source_root = str(_canonical_accelerator_source_root())
    prompt_file_env = _env_name(config, "PROMPT_FILE")
    model_env = _env_name(config, "MODEL_NAME")
    provider_env = _env_name(config, "PROVIDER")
    fallback_env = _env_name(config, "ALLOW_LOCAL_FALLBACK")
    cross_provider_fallback_env = _env_name(
        config, "ALLOW_CROSS_PROVIDER_FALLBACK"
    )
    timeout_env = _env_name(config, "TIMEOUT")
    max_tokens_env = _env_name(config, "MAX_NEW_TOKENS")
    temperature_env = _env_name(config, "TEMPERATURE")
    trace_env = _env_name(config, "TRACE")
    trace_dir_env = _env_name(config, "TRACE_DIR")
    reject_provider_env = _env_name(config, "REJECT_EFFECTIVE_PROVIDER")
    required_providers_env = _env_name(config, "REQUIRED_EFFECTIVE_PROVIDERS")
    result_file_env = _env_name(config, "RESULT_FILE")
    envelope_file_env = _env_name(config, "ENVELOPE_FILE")
    usage_mode_env = _env_name(config, "USAGE_MODE")
    request_id_env = _env_name(config, "REQUEST_ID")
    attempt_env = _env_name(config, "ATTEMPT")
    idempotency_env = _env_name(config, "IDEMPOTENCY_KEY")
    supervisor_receipt_env = _env_name(config, "SUPERVISOR_RECEIPT_ID")
    endpoint_receipt_env = _env_name(config, "ENDPOINT_RECEIPT_ID")
    return f"""
import sys

# Bind this child to the exact accelerator package that authored it. Remove
# the implicit script directory before importing any non-builtin module.
_canonical_source_root = {canonical_source_root!r}
_implicit_script_root = sys.path[0] if sys.path else ""
sys.path[:] = [_canonical_source_root] + [
    entry for entry in sys.path
    if entry
    and entry != _canonical_source_root
    and entry != _implicit_script_root
]

import inspect
import hashlib
import datetime
import json
import os
import pathlib

from ipfs_accelerate_py import llm_router

prompt = pathlib.Path(os.environ[{prompt_file_env!r}]).read_text(encoding="utf-8")
provider = os.environ.get({provider_env!r}) or None
kwargs = dict(
    model_name=os.environ[{model_env!r}],
    provider=provider,
    allow_local_fallback=os.environ.get({fallback_env!r}) == "1",
    allow_cross_provider_fallback=(
        os.environ.get({cross_provider_fallback_env!r}) == "1"
    ),
    timeout=int(os.environ[{timeout_env!r}]),
    max_new_tokens=int(os.environ[{max_tokens_env!r}]),
    temperature=float(os.environ[{temperature_env!r}]),
)
try:
    parameters = inspect.signature(llm_router.generate_text).parameters
except (TypeError, ValueError):
    parameters = {{}}
if "trace" in parameters:
    kwargs["trace"] = os.environ.get({trace_env!r}) == "1"
if "trace_dir" in parameters:
    trace_dir = os.environ.get({trace_dir_env!r}) or None
    kwargs["trace_dir"] = trace_dir
usage_mode = (os.environ.get({usage_mode_env!r}) or "off").strip().lower() or "off"
result_path = os.environ.get({result_file_env!r}) or ""
envelope_path = os.environ.get({envelope_file_env!r}) or ""

def _write_result_payload(payload):
    if result_path:
        pathlib.Path(result_path).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

try:
    text = llm_router.generate_text(prompt, **kwargs)
except Exception as exc:
    capacity_type = getattr(llm_router, "UsageCapacityError", None)
    if not isinstance(capacity_type, type) or not isinstance(exc, capacity_type):
        raise
    raw_reasons = {{
        str(item).strip()
        for item in (getattr(exc, "reason_codes", ()) or ())
        if str(item).strip()
    }}
    reason_codes = [
        item
        for item in ("usage_limit", "quota_exceeded", "capacity_unavailable")
        if item in raw_reasons
    ] or [{LLM_CHILD_PROVIDER_CAPACITY_REASON!r}]
    next_eligible_at = str(
        getattr(exc, "next_eligible_at", "") or ""
    ).strip()
    if next_eligible_at:
        try:
            datetime.datetime.fromisoformat(
                next_eligible_at.replace("Z", "+00:00")
            )
        except (TypeError, ValueError):
            next_eligible_at = ""
    error_material = {{
        "request_id": os.environ.get({request_id_env!r}) or "",
        "attempt": int(os.environ.get({attempt_env!r}) or "1"),
        "idempotency_key": os.environ.get({idempotency_env!r}) or "",
        "reason_codes": reason_codes,
        "next_eligible_at": next_eligible_at,
        "exit_code": 75,
    }}
    _write_result_payload({{
        "schema": {LLM_CHILD_RESULT_SCHEMA!r},
        "contract_version": {LLM_CHILD_ENVELOPE_VERSION},
        "usage_mode": usage_mode,
        **error_material,
        "status": "error",
        "supervisor_receipt_id": os.environ.get({supervisor_receipt_env!r}) or "",
        "endpoint_receipt_id": os.environ.get({endpoint_receipt_env!r}) or "",
        "execution_result_id": "sha256:" + hashlib.sha256(
            json.dumps(
                error_material,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "effective_provider": "",
        "text_chars": 0,
        "text_bytes": 0,
        "text_sha256": hashlib.sha256(b"").hexdigest(),
    }})
    sys.stderr.write("llm_router provider capacity unavailable.\\n")
    raise SystemExit(75)
reject_provider = os.environ.get({reject_provider_env!r}) or ""
required_providers = {{
    value.strip()
    for value in (os.environ.get({required_providers_env!r}) or "").split(",")
    if value.strip()
}}
trace_getter = getattr(llm_router, "get_last_generation_trace", None)
effective_provider = ""
if callable(trace_getter):
    trace = trace_getter()
    if isinstance(trace, dict):
        effective_provider = str(trace.get("effective_provider_name") or "")
if (reject_provider or required_providers):
    if required_providers and effective_provider not in required_providers:
        sys.stderr.write(
            f"llm_router resolved to {{effective_provider or 'unknown'}}; "
            f"expected one of {{sorted(required_providers)}}.\\n"
        )
        raise SystemExit(2)
    if reject_provider and effective_provider == reject_provider:
        sys.stderr.write(
            f"llm_router resolved to {{reject_provider}} fallback; configure a real provider.\\n"
        )
        raise SystemExit(2)
text_out = "" if text is None else str(text)
text_sha256 = hashlib.sha256(text_out.encode("utf-8")).hexdigest()
# Propagate receipt IDs only; never write prompt/provider payload bodies.
result_payload = {{
    "schema": {LLM_CHILD_RESULT_SCHEMA!r},
    "contract_version": {LLM_CHILD_ENVELOPE_VERSION},
    "usage_mode": usage_mode,
    "request_id": os.environ.get({request_id_env!r}) or "",
    "attempt": int(os.environ.get({attempt_env!r}) or "1"),
    "idempotency_key": os.environ.get({idempotency_env!r}) or "",
    "status": "ok",
    "reason_codes": [],
    "supervisor_receipt_id": os.environ.get({supervisor_receipt_env!r}) or "",
    "endpoint_receipt_id": os.environ.get({endpoint_receipt_env!r}) or "",
    "execution_result_id": "sha256:" + hashlib.sha256(
        json.dumps(
            {{
                "request_id": os.environ.get({request_id_env!r}) or "",
                "attempt": int(os.environ.get({attempt_env!r}) or "1"),
                "idempotency_key": os.environ.get({idempotency_env!r}) or "",
                "effective_provider": effective_provider,
                "text_chars": len(text_out),
                "text_bytes": len(text_out.encode("utf-8")),
                "text_sha256": text_sha256,
            }},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest(),
    "effective_provider": effective_provider,
    "text_chars": len(text_out),
    "text_bytes": len(text_out.encode("utf-8")),
    "text_sha256": text_sha256,
    "exit_code": 0,
    "next_eligible_at": "",
}}
_write_result_payload(result_payload)
sys.stdout.write(text_out)
"""


def call_llm_router(prompt: str, config: LlmRouterInvocation) -> str:
    """Call canonical ``ipfs_accelerate_py.llm_router`` in an isolated child.

    Returns generated text for backward compatibility. When ``usage_mode`` is
    not ``off``, a bounded result envelope with receipt IDs is written beside
    the child and available via :func:`last_llm_child_result`.
    """

    text, _result = call_llm_router_with_receipt(prompt, config)
    return text


def call_llm_router_with_receipt(
    prompt: str, config: LlmRouterInvocation
) -> tuple[str, Optional[LlmChildResultEnvelope]]:
    """Call the LLM child and return ``(text, result_envelope)``.

    Off mode remains behaviorally compatible: prompt file + stdout text, same
    provider verification and timeouts. Non-off modes additionally emit a
    bounded versioned JSON result file carrying receipt IDs only.
    """

    backend = os.environ.get(config.backend_env_name, config.backend_default)
    if backend != "llm_router":
        raise RuntimeError(
            f"Unsupported {config.backend_label} {backend!r}; expected 'llm_router'."
        )
    if len(prompt) > config.max_prompt_chars + len(config.prompt_overage_allowance):
        raise RuntimeError(
            f"LLM prompt exceeds configured budget before llm_router child launch: "
            f"{len(prompt)} > {config.max_prompt_chars}"
        )

    usage_mode = _normalize_usage_mode(config.usage_mode)
    prompt_file: Optional[Path] = None
    result_file: Optional[Path] = None
    envelope_file: Optional[Path] = None
    child_file: Optional[Path] = None
    completed: Optional[subprocess.CompletedProcess[str]] = None
    timeout_seconds = int(config.timeout_seconds) + int(config.timeout_grace_seconds)
    result_envelope: Optional[LlmChildResultEnvelope] = None
    _set_last_llm_result(None)
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            prefix=config.prompt_file_prefix,
            suffix=".txt",
        ) as handle:
            handle.write(prompt)
            prompt_file = Path(handle.name)

        write_result = bool(config.write_result_envelope)
        if write_result:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                prefix=config.result_file_prefix,
                suffix=".json",
            ) as handle:
                result_file = Path(handle.name)

        # Request envelope is metadata only (prompt stays in prompt_file).
        if usage_mode != LLM_USAGE_MODE_OFF or write_result:
            envelope = build_child_request_envelope(
                config,
                input_digest=_prompt_digest(prompt),
                result_file=str(result_file or ""),
            )
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                prefix=config.envelope_file_prefix,
                suffix=".json",
            ) as handle:
                handle.write(envelope.to_json())
                envelope_file = Path(handle.name)

        # A real system-temporary script keeps the repository working tree out
        # of Python's implicit import path. Its source then pins the exact
        # accelerator tree that authored the child program.
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            prefix=config.child_file_prefix,
            suffix=".py",
        ) as handle:
            handle.write(_llm_router_child_code(config))
            child_file = Path(handle.name)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(_canonical_accelerator_source_root())
        for unsafe_python_setting in (
            "PYTHONHOME",
            "PYTHONINSPECT",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
        ):
            env.pop(unsafe_python_setting, None)
        env.update(
            {
                _env_name(config, "PROMPT_FILE"): str(prompt_file),
                _env_name(config, "MODEL_NAME"): config.model_name,
                _env_name(config, "PROVIDER"): config.provider or "",
                _env_name(config, "ALLOW_LOCAL_FALLBACK"): (
                    "1" if config.allow_local_fallback else "0"
                ),
                _env_name(config, "ALLOW_CROSS_PROVIDER_FALLBACK"): (
                    "1" if _allow_cross_provider_fallback(config) else "0"
                ),
                _env_name(config, "TIMEOUT"): str(config.timeout_seconds),
                _env_name(config, "MAX_NEW_TOKENS"): str(config.max_new_tokens),
                _env_name(config, "TEMPERATURE"): str(config.temperature),
                _env_name(config, "TRACE"): "1" if config.trace else "0",
                _env_name(config, "TRACE_DIR"): str(config.trace_dir or ""),
                _env_name(config, "REJECT_EFFECTIVE_PROVIDER"): (
                    config.reject_effective_provider_name or ""
                ),
                _env_name(config, "REQUIRED_EFFECTIVE_PROVIDERS"): ",".join(
                    config.required_effective_providers
                ),
                _env_name(config, "USAGE_MODE"): usage_mode,
                _env_name(config, "REQUEST_ID"): str(config.request_id or ""),
                _env_name(config, "ATTEMPT"): str(int(config.attempt or 1)),
                _env_name(config, "IDEMPOTENCY_KEY"): str(config.idempotency_key or ""),
                _env_name(config, "SUPERVISOR_RECEIPT_ID"): str(
                    config.supervisor_receipt_id or ""
                ),
                _env_name(config, "ENDPOINT_RECEIPT_ID"): str(
                    config.endpoint_receipt_id or ""
                ),
                _env_name(config, "RESULT_FILE"): str(result_file or ""),
                _env_name(config, "ENVELOPE_FILE"): str(envelope_file or ""),
            }
        )
        if config.codex_read_only:
            env["ipfs_accelerate_py_CODEX_SANDBOX"] = "read-only"
        command = [config.python_executable, str(child_file)]
        process = subprocess.Popen(
            command,
            cwd=str(config.repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        global _ACTIVE_LLM_PROCESS
        _ACTIVE_LLM_PROCESS = process
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            terminate_process_group(process)
            raise RuntimeError(
                f"llm_router child timed out after {timeout_seconds} seconds"
            ) from exc
        finally:
            if _ACTIVE_LLM_PROCESS is process:
                _ACTIVE_LLM_PROCESS = None
        completed = subprocess.CompletedProcess(
            command,
            returncode=int(process.returncode or 0),
            stdout=stdout,
            stderr=stderr,
        )
        if result_file is not None and result_file.exists():
            try:
                raw = result_file.read_text(encoding="utf-8")
                if raw.strip():
                    result_envelope = parse_child_result_envelope(raw)
            except (OSError, RuntimeError, UnicodeDecodeError):
                result_envelope = None
        if result_envelope is None and usage_mode != LLM_USAGE_MODE_OFF:
            # Synthesize a minimal envelope from parent-known receipt IDs so
            # callers can still propagate IDs when the child did not write one.
            result_envelope = LlmChildResultEnvelope(
                usage_mode=usage_mode,
                request_id=str(config.request_id or ""),
                attempt=int(config.attempt or 1),
                idempotency_key=str(config.idempotency_key or ""),
                status="ok" if completed.returncode == 0 else "error",
                reason_codes=()
                if completed.returncode == 0
                else ("child_exit_error",),
                supervisor_receipt_id=str(config.supervisor_receipt_id or ""),
                endpoint_receipt_id=str(config.endpoint_receipt_id or ""),
                text_chars=len(completed.stdout or ""),
                exit_code=int(completed.returncode or 0),
            )
        _set_last_llm_result(result_envelope)
    finally:
        for path in (prompt_file, result_file, envelope_file, child_file):
            if path is not None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
    if completed is None:
        raise RuntimeError("llm_router child did not produce a completed process result")
    if completed.returncode != 0:
        capacity_reasons = {
            LLM_CHILD_PROVIDER_CAPACITY_REASON,
            "usage_limit",
            "quota_exceeded",
            "capacity_unavailable",
        }
        if (
            result_envelope is not None
            and result_envelope.status == "error"
            and capacity_reasons.intersection(
                result_envelope.reason_codes
            )
        ):
            raise LlmChildProviderCapacityError(
                provider_id=str(config.provider or ""),
                reason_codes=result_envelope.reason_codes,
                next_eligible_at=result_envelope.next_eligible_at,
            )
        details = compact_message(
            (completed.stdout or "") + " " + (completed.stderr or ""), limit=1200
        )
        raise RuntimeError(
            f"llm_router child exited with code {completed.returncode}: {details}"
        )
    return completed.stdout, result_envelope


def call_with_thread_deadline(
    generator: Callable[..., Any],
    *args: Any,
    timeout_seconds: float,
    thread_name: str = "todo-daemon-call",
    on_timeout: DeadlineTimeoutCallback | None = None,
    timeout_message: DeadlineMessageBuilder | None = None,
    empty_result_message: str = "daemon call thread ended without returning a result",
    **kwargs: Any,
) -> str:
    """Run a blocking generator in a daemon thread and enforce a caller deadline."""

    timeout = float(timeout_seconds)
    if timeout <= 0:
        return str(generator(*args, **kwargs))

    result_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result_queue.put(("ok", str(generator(*args, **kwargs))))
        except BaseException as exc:  # pragma: no cover - defensive thread boundary.
            result_queue.put(("error", exc))

    thread = threading.Thread(target=invoke, name=thread_name, daemon=True)
    started = time.time()
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        elapsed = time.time() - started
        if on_timeout is not None:
            on_timeout(elapsed, timeout, thread.name)
        if timeout_message is None:
            message = (
                f"daemon call exceeded deadline after {elapsed:.1f}s "
                f"(timeout={timeout:.1f}s)"
            )
        else:
            message = timeout_message(elapsed, timeout, thread.name)
        raise TimeoutError(message)
    try:
        kind, payload = result_queue.get_nowait()
    except queue.Empty as exc:  # pragma: no cover - thread finished without publishing.
        raise RuntimeError(empty_result_message) from exc
    if kind == "error":
        raise payload
    return str(payload)


def active_llm_process() -> Optional[subprocess.Popen[Any]]:
    """Return the currently active LLM child process, if any."""

    return _ACTIVE_LLM_PROCESS


def terminate_active_llm_process(*, grace_seconds: float = 1.0) -> bool:
    """Terminate the active LLM child process, if one is running."""

    process = _ACTIVE_LLM_PROCESS
    if process is None or process.poll() is not None:
        return False
    terminate_process_group(process, grace_seconds=grace_seconds)
    return True


def handle_active_llm_signal(signum: int, _frame: object) -> None:
    """Signal handler that stops the active LLM child before exiting."""

    terminate_active_llm_process(grace_seconds=1.0)
    raise SystemExit(128 + signum)


def install_active_llm_signal_handlers(
    signals: Sequence[int] = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP),
) -> None:
    """Install signal handlers that clean up the active LLM child process."""

    for signum in signals:
        signal.signal(signum, handle_active_llm_signal)


__all__ = [
    "LLM_CHILD_ENVELOPE_SCHEMA",
    "LLM_CHILD_ENVELOPE_VERSION",
    "LLM_CHILD_PROVIDER_CAPACITY_REASON",
    "MAX_LLM_CHILD_PROVIDER_BACKOFF_SECONDS",
    "LLM_CHILD_RESULT_SCHEMA",
    "LLM_USAGE_MODE_ASSIST",
    "LLM_USAGE_MODE_ENFORCE",
    "LLM_USAGE_MODE_OBSERVE",
    "LLM_USAGE_MODE_OFF",
    "LLM_USAGE_MODE_SHADOW",
    "LlmChildRequestEnvelope",
    "LlmChildResultEnvelope",
    "LlmChildProviderCapacityError",
    "LlmRouterInvocation",
    "active_llm_process",
    "build_child_request_envelope",
    "call_llm_router",
    "call_llm_router_with_receipt",
    "call_with_thread_deadline",
    "collect_descendant_pids",
    "handle_active_llm_signal",
    "install_active_llm_signal_handlers",
    "last_llm_child_result",
    "parse_child_result_envelope",
    "process_groups_for_family",
    "terminate_active_llm_process",
    "terminate_process_group",
]
