"""Executable, evidence-bound capability registry for formal provers.

The older :mod:`formal_verification_capabilities` module answers the useful,
but deliberately weak, question "is a package or executable discoverable?".
This module answers a stricter question: "which exact toolchain has passed
which exact bounded fixture?".  In particular:

* finding source, a package, or an executable never counts as a self-test;
* every state above ``versioned`` is derived from a successful receipt;
* a receipt binds all executable, package, model, translator, semantic-profile,
  and fixture identities, including explicit ``unavailable`` identities; and
* the checked-in Markdown matrix is retained as documentation provenance only.

The module imports no optional prover package.  Calling :meth:`probe` may run
already-installed executables, but every invocation has a deadline and bounded
captured output.  Merely importing the supervisor remains side-effect free.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.machinery
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROVER_MATRIX_SCHEMA_VERSION = (
    "ipfs_accelerate_py/agent-supervisor/prover-matrix@1"
)
PROVER_SELF_TEST_SCHEMA_VERSION = (
    "ipfs_accelerate_py/agent-supervisor/prover-self-test@1"
)
PROVER_MATRIX_DUCKDB_SCHEMA_VERSION = (
    "ipfs_accelerate_py/agent-supervisor/prover-matrix-duckdb@1"
)
PROVER_MATRIX_REPORT_VERSION = 1
DEFAULT_SELF_TEST_TIMEOUT_SECONDS = 5.0
DEFAULT_MATRIX_TIMEOUT_SECONDS = 90.0
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
DEFAULT_MAX_IDENTITY_FILE_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_SELF_TESTS = 64
DEFAULT_DOCUMENTATION_MATRIX = (
    Path("ipfs_datasets_py") / "docs" / "security_verification" / "prover_matrix.md"
)


class ProverState(str, Enum):
    """Distinct state names exposed by every matrix entry."""

    ABSENT = "absent"
    DISCOVERED = "discovered"
    VERSIONED = "versioned"
    SMOKE_TESTED = "smoke_tested"
    TRANSLATION_CONFORMANT = "translation_conformant"
    RECONSTRUCTION_CAPABLE = "reconstruction_capable"
    AUTHORITATIVE_FOR = "authoritative_for"


class SelfTestStatus(str, Enum):
    """Terminal status of one bounded fixture execution."""

    PASSED = "passed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    ERROR = "error"
    NOT_RUN = "not_run"


class IdentityKind(str, Enum):
    EXECUTABLE = "executable"
    PACKAGE = "package"
    MODEL = "model"
    TRANSLATOR = "translator"
    SEMANTIC_PROFILE = "semantic_profile"
    FIXTURE = "fixture"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_timestamp(clock: Callable[[], float]) -> str:
    return (
        datetime.fromtimestamp(clock(), tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _strict_json_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    try:
        result = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain strict JSON values") from exc
    if not isinstance(result, dict):  # pragma: no cover - guarded by dict above
        raise ValueError(f"{field_name} must be an object")
    return result


def _nonempty_tuple(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values if str(value).strip())
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


@dataclass(frozen=True)
class BoundIdentity:
    """Canonical identity for one input dimension of a self-test."""

    kind: IdentityKind | str
    name: str
    content_identity: str
    available: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", IdentityKind(str(getattr(self.kind, "value", self.kind)))
        )
        name = str(self.name).strip()
        content_identity = str(self.content_identity).strip()
        if not name or not content_identity:
            raise ValueError("bound identity name and content_identity must not be empty")
        if not isinstance(self.available, bool):
            raise ValueError("bound identity available must be boolean")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "content_identity", content_identity)
        object.__setattr__(
            self,
            "metadata",
            _strict_json_mapping(self.metadata, field_name="identity metadata"),
        )

    @classmethod
    def unavailable(cls, kind: IdentityKind, name: str) -> "BoundIdentity":
        payload = {"kind": kind.value, "name": str(name), "available": False}
        return cls(
            kind=kind,
            name=str(name),
            content_identity=_identity(payload),
            available=False,
            metadata={"reason": "component was not resolved for this self-test"},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "content_identity": self.content_identity,
            "available": self.available,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SelfTestBinding:
    """All semantic and toolchain identities consumed by one fixture."""

    executable: BoundIdentity
    package: BoundIdentity
    model: BoundIdentity
    translator: BoundIdentity
    semantic_profile: BoundIdentity
    fixture: BoundIdentity

    def __post_init__(self) -> None:
        expected = (
            ("executable", self.executable, IdentityKind.EXECUTABLE),
            ("package", self.package, IdentityKind.PACKAGE),
            ("model", self.model, IdentityKind.MODEL),
            ("translator", self.translator, IdentityKind.TRANSLATOR),
            ("semantic_profile", self.semantic_profile, IdentityKind.SEMANTIC_PROFILE),
            ("fixture", self.fixture, IdentityKind.FIXTURE),
        )
        for field_name, value, kind in expected:
            if not isinstance(value, BoundIdentity) or value.kind is not kind:
                raise ValueError(f"{field_name} must be a {kind.value} identity")

    @property
    def binding_id(self) -> str:
        return _identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "executable": self.executable.to_dict(),
            "package": self.package.to_dict(),
            "model": self.model.to_dict(),
            "translator": self.translator.to_dict(),
            "semantic_profile": self.semantic_profile.to_dict(),
            "fixture": self.fixture.to_dict(),
        }
        if include_id:
            payload["binding_id"] = self.binding_id
        return payload


@dataclass(frozen=True)
class ProverFixture:
    """A native-language smoke fixture and the promotions it can establish.

    Promotions are part of the reviewed fixture definition, not claims parsed
    from command output.  ``authoritative_for`` is additionally intersected
    with the enclosing prover definition's allowlist.
    """

    fixture_id: str
    model_text: str
    file_name: str
    translator_id: str
    translator_version: str
    semantic_profile_id: str
    semantic_profile_version: str
    args: tuple[str, ...]
    stdin: bool = False
    expected_exit_codes: tuple[int, ...] = (0,)
    expected_output_all: tuple[str, ...] = ()
    expected_output_any: tuple[str, ...] = ()
    expected_output_lines: tuple[str, ...] = ()
    establishes_translation_conformance: bool = False
    establishes_reconstruction: bool = False
    authoritative_for: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "fixture_id",
            "file_name",
            "translator_id",
            "translator_version",
            "semantic_profile_id",
            "semantic_profile_version",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")
        if not self.model_text:
            raise ValueError("model_text must not be empty")
        if Path(self.file_name).name != self.file_name:
            raise ValueError("fixture file_name must be a basename")
        if not self.expected_exit_codes:
            raise ValueError("expected_exit_codes must not be empty")
        if any(isinstance(code, bool) or not isinstance(code, int) for code in self.expected_exit_codes):
            raise ValueError("expected_exit_codes must contain integers")
        object.__setattr__(self, "args", tuple(str(arg) for arg in self.args))
        object.__setattr__(
            self,
            "expected_output_all",
            tuple(str(value) for value in self.expected_output_all),
        )
        object.__setattr__(
            self,
            "expected_output_any",
            tuple(str(value) for value in self.expected_output_any),
        )
        object.__setattr__(
            self,
            "expected_output_lines",
            tuple(str(value) for value in self.expected_output_lines),
        )
        object.__setattr__(
            self,
            "authoritative_for",
            tuple(sorted({str(value).strip() for value in self.authoritative_for if str(value).strip()})),
        )
        object.__setattr__(
            self,
            "metadata",
            _strict_json_mapping(self.metadata, field_name="fixture metadata"),
        )

    @property
    def model_identity(self) -> BoundIdentity:
        payload = {
            "fixture_id": self.fixture_id,
            "file_name": self.file_name,
            "model_text": self.model_text,
        }
        return BoundIdentity(
            IdentityKind.MODEL,
            self.file_name,
            _identity(payload),
            True,
            {
                "bytes": len(self.model_text.encode("utf-8")),
                "encoding": "utf-8",
            },
        )

    @property
    def translator_identity(self) -> BoundIdentity:
        payload = {
            "translator_id": self.translator_id,
            "translator_version": self.translator_version,
        }
        return BoundIdentity(
            IdentityKind.TRANSLATOR,
            self.translator_id,
            _identity(payload),
            True,
            {"version": self.translator_version},
        )

    @property
    def semantic_profile_identity(self) -> BoundIdentity:
        payload = {
            "semantic_profile_id": self.semantic_profile_id,
            "semantic_profile_version": self.semantic_profile_version,
        }
        return BoundIdentity(
            IdentityKind.SEMANTIC_PROFILE,
            self.semantic_profile_id,
            _identity(payload),
            True,
            {"version": self.semantic_profile_version},
        )

    @property
    def fixture_identity(self) -> BoundIdentity:
        payload = {
            "fixture_id": self.fixture_id,
            "args": list(self.args),
            "stdin": self.stdin,
            "expected_exit_codes": list(self.expected_exit_codes),
            "expected_output_all": list(self.expected_output_all),
            "expected_output_any": list(self.expected_output_any),
            "expected_output_lines": list(self.expected_output_lines),
            "establishes_translation_conformance": self.establishes_translation_conformance,
            "establishes_reconstruction": self.establishes_reconstruction,
            "authoritative_for": list(self.authoritative_for),
            "metadata": dict(self.metadata),
        }
        return BoundIdentity(
            IdentityKind.FIXTURE,
            self.fixture_id,
            _identity(payload),
            True,
            {"fixture_definition": payload},
        )


@dataclass(frozen=True)
class ProverDefinition:
    """Static registry definition.  It is policy/configuration, not evidence."""

    prover_id: str
    display_name: str
    family: str
    executable_candidates: tuple[str, ...] = ()
    package_modules: tuple[str, ...] = ()
    package_distributions: tuple[str, ...] = ()
    version_args: tuple[str, ...] = ("--version",)
    fixture: ProverFixture | None = None
    maximum_authoritative_for: tuple[str, ...] = ()
    documentation_labels: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        prover_id = str(self.prover_id).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", prover_id):
            raise ValueError("prover_id must be lower snake case")
        if not str(self.display_name).strip() or not str(self.family).strip():
            raise ValueError("display_name and family must not be empty")
        if not self.executable_candidates and not self.package_modules:
            raise ValueError("a prover needs executable candidates or package modules")
        if self.package_distributions and (
            len(self.package_distributions) != len(self.package_modules)
        ):
            raise ValueError(
                "package_distributions must be empty or align with package_modules"
            )
        object.__setattr__(self, "prover_id", prover_id)
        object.__setattr__(
            self, "executable_candidates", tuple(self.executable_candidates)
        )
        object.__setattr__(self, "package_modules", tuple(self.package_modules))
        object.__setattr__(
            self, "package_distributions", tuple(self.package_distributions)
        )
        object.__setattr__(self, "version_args", tuple(self.version_args))
        object.__setattr__(
            self,
            "maximum_authoritative_for",
            tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in self.maximum_authoritative_for
                        if str(value).strip()
                    }
                )
            ),
        )
        labels = self.documentation_labels or (self.display_name,)
        object.__setattr__(self, "documentation_labels", tuple(labels))


@dataclass(frozen=True)
class CommandRequest:
    command: tuple[str, ...]
    stdin_text: str | None
    cwd: str | None
    timeout_seconds: float
    max_output_bytes: int


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str | None = None
    output_truncated: bool = False


CommandRunner = Callable[[CommandRequest], CommandResult | Mapping[str, Any]]
PackageFinder = Callable[[str], Any]
VersionFinder = Callable[[str], str]
ExecutableFinder = Callable[[str], str | None]


@dataclass(frozen=True)
class ProverSelfTestReceipt:
    """Immutable bounded execution evidence for one registry entry."""

    prover_id: str
    status: SelfTestStatus | str
    binding: SelfTestBinding
    command: tuple[str, ...]
    command_identity: str
    started_at: str
    duration_ms: int
    timeout_seconds: float
    max_output_bytes: int
    returncode: int | None
    stdout_sha256: str
    stderr_sha256: str
    output_truncated: bool
    reason: str
    translation_conformant: bool = False
    reconstruction_capable: bool = False
    authoritative_for: tuple[str, ...] = ()
    schema_version: str = PROVER_SELF_TEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            SelfTestStatus(str(getattr(self.status, "value", self.status))),
        )
        if self.schema_version != PROVER_SELF_TEST_SCHEMA_VERSION:
            raise ValueError("unsupported prover self-test receipt schema")
        if not str(self.prover_id).strip() or not str(self.reason).strip():
            raise ValueError("receipt prover_id and reason must not be empty")
        if self.duration_ms < 0 or self.timeout_seconds <= 0 or self.max_output_bytes < 1:
            raise ValueError("receipt resource bounds are invalid")
        authorities = tuple(sorted({str(value) for value in self.authoritative_for}))
        if self.status is not SelfTestStatus.PASSED and (
            self.translation_conformant
            or self.reconstruction_capable
            or authorities
        ):
            raise ValueError("failed self-tests cannot promote prover capabilities")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "authoritative_for", authorities)

    @property
    def receipt_id(self) -> str:
        return _identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "prover_id": self.prover_id,
            "status": self.status.value,
            "binding": self.binding.to_dict(),
            "command": list(self.command),
            "command_identity": self.command_identity,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "returncode": self.returncode,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "output_truncated": self.output_truncated,
            "reason": self.reason,
            "translation_conformant": self.translation_conformant,
            "reconstruction_capable": self.reconstruction_capable,
            "authoritative_for": list(self.authoritative_for),
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class DocumentationClaim:
    """One checked-in matrix row, explicitly excluded from runtime evidence."""

    source_path: str
    source_identity: str
    row_number: int
    prover_text: str
    access_path: str
    primary_fit: str

    @property
    def claim_id(self) -> str:
        return _identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "source_path": self.source_path,
            "source_identity": self.source_identity,
            "row_number": self.row_number,
            "prover_text": self.prover_text,
            "access_path": self.access_path,
            "primary_fit": self.primary_fit,
            "evidence_class": "documentation_only",
            "runtime_evidence": False,
        }
        if include_id:
            payload["claim_id"] = self.claim_id
        return payload


@dataclass(frozen=True)
class ProverMatrixEntry:
    """Observed runtime state for one prover definition."""

    prover_id: str
    display_name: str
    family: str
    absent: bool
    discovered: bool
    versioned: bool
    smoke_tested: bool
    translation_conformant: bool
    reconstruction_capable: bool
    authoritative_for: tuple[str, ...]
    executable_path: str | None
    executable_version: str | None
    package_module: str | None
    package_version: str | None
    reason: str
    receipt: ProverSelfTestReceipt | None = None
    documentation_claims: tuple[DocumentationClaim, ...] = ()

    def __post_init__(self) -> None:
        if self.absent == self.discovered:
            raise ValueError("absent must be the inverse of discovered")
        if self.versioned and not self.discovered:
            raise ValueError("versioned requires discovery")
        if self.smoke_tested and not self.versioned:
            raise ValueError("smoke-tested requires a versioned component")
        if (
            self.translation_conformant
            or self.reconstruction_capable
            or self.authoritative_for
        ) and not self.smoke_tested:
            raise ValueError("advanced states require a passing smoke test")
        if self.smoke_tested and self.receipt is None:
            raise ValueError("smoke-tested state requires a passing self-test receipt")
        if self.receipt is not None:
            if self.receipt.prover_id != self.prover_id:
                raise ValueError("self-test receipt belongs to a different prover")
            if self.smoke_tested != (self.receipt.status is SelfTestStatus.PASSED):
                raise ValueError("smoke-tested state must agree with its receipt")
            if self.translation_conformant != self.receipt.translation_conformant:
                raise ValueError("translation state must agree with its receipt")
            if self.reconstruction_capable != self.receipt.reconstruction_capable:
                raise ValueError("reconstruction state must agree with its receipt")
            if tuple(sorted(self.authoritative_for)) != tuple(
                sorted(self.receipt.authoritative_for)
            ):
                raise ValueError("authority state must agree with its receipt")
        object.__setattr__(
            self, "authoritative_for", tuple(sorted(set(self.authoritative_for)))
        )
        object.__setattr__(
            self, "documentation_claims", tuple(self.documentation_claims)
        )

    @property
    def states(self) -> Mapping[str, Any]:
        return {
            ProverState.ABSENT.value: self.absent,
            ProverState.DISCOVERED.value: self.discovered,
            ProverState.VERSIONED.value: self.versioned,
            ProverState.SMOKE_TESTED.value: self.smoke_tested,
            ProverState.TRANSLATION_CONFORMANT.value: self.translation_conformant,
            ProverState.RECONSTRUCTION_CAPABLE.value: self.reconstruction_capable,
            ProverState.AUTHORITATIVE_FOR.value: list(self.authoritative_for),
        }

    @property
    def highest_state(self) -> ProverState:
        if self.authoritative_for:
            return ProverState.AUTHORITATIVE_FOR
        if self.reconstruction_capable:
            return ProverState.RECONSTRUCTION_CAPABLE
        if self.translation_conformant:
            return ProverState.TRANSLATION_CONFORMANT
        if self.smoke_tested:
            return ProverState.SMOKE_TESTED
        if self.versioned:
            return ProverState.VERSIONED
        if self.discovered:
            return ProverState.DISCOVERED
        return ProverState.ABSENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "prover_id": self.prover_id,
            "display_name": self.display_name,
            "family": self.family,
            "highest_state": self.highest_state.value,
            "states": dict(self.states),
            "executable": {
                "path": self.executable_path,
                "version": self.executable_version,
            },
            "package": {
                "module": self.package_module,
                "version": self.package_version,
            },
            "reason": self.reason,
            "self_test_receipt": self.receipt.to_dict() if self.receipt else None,
            "documentation_claims": [
                claim.to_dict() for claim in self.documentation_claims
            ],
            "documentation_is_runtime_evidence": False,
        }


@dataclass(frozen=True)
class ProverMatrixSnapshot:
    entries: tuple[ProverMatrixEntry, ...]
    generated_at: str
    duration_ms: int
    self_tests_requested: bool
    bounded: bool
    max_self_tests: int
    matrix_timeout_seconds: float
    documentation_source: str | None
    schema_version: str = PROVER_MATRIX_SCHEMA_VERSION
    report_version: int = PROVER_MATRIX_REPORT_VERSION

    def __post_init__(self) -> None:
        ids = [entry.prover_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("prover matrix entry ids must be unique")
        if self.schema_version != PROVER_MATRIX_SCHEMA_VERSION:
            raise ValueError("unsupported prover matrix schema")
        if self.duration_ms < 0 or self.max_self_tests < 1:
            raise ValueError("invalid matrix bounds")
        object.__setattr__(self, "entries", tuple(self.entries))

    @property
    def capabilities(self) -> Mapping[str, ProverMatrixEntry]:
        return {entry.prover_id: entry for entry in self.entries}

    def entry(self, prover_id: str) -> ProverMatrixEntry:
        try:
            return self.capabilities[str(prover_id)]
        except KeyError as exc:
            raise KeyError(f"unknown prover matrix entry: {prover_id}") from exc

    @property
    def snapshot_id(self) -> str:
        stable = {
            "schema_version": self.schema_version,
            "report_version": self.report_version,
            "entries": [entry.to_dict() for entry in self.entries],
            "self_tests_requested": self.self_tests_requested,
            "bounded": self.bounded,
            "max_self_tests": self.max_self_tests,
            "matrix_timeout_seconds": self.matrix_timeout_seconds,
            "documentation_source": self.documentation_source,
        }
        return _identity(stable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_version": self.report_version,
            "snapshot_id": self.snapshot_id,
            "generated_at": self.generated_at,
            "duration_ms": self.duration_ms,
            "self_tests_requested": self.self_tests_requested,
            "bounded": self.bounded,
            "max_self_tests": self.max_self_tests,
            "matrix_timeout_seconds": self.matrix_timeout_seconds,
            "documentation_source": self.documentation_source,
            "documentation_is_runtime_evidence": False,
            "entries": {
                entry.prover_id: entry.to_dict() for entry in self.entries
            },
            "counts": {
                state.value: (
                    sum(
                        bool(entry.states[state.value])
                        for entry in self.entries
                    )
                    if state is not ProverState.AUTHORITATIVE_FOR
                    else sum(bool(entry.authoritative_for) for entry in self.entries)
                )
                for state in ProverState
            },
        }


@dataclass(frozen=True)
class ProverMatrixProbeConfig:
    run_self_tests: bool = True
    version_timeout_seconds: float = 2.0
    self_test_timeout_seconds: float = DEFAULT_SELF_TEST_TIMEOUT_SECONDS
    matrix_timeout_seconds: float = DEFAULT_MATRIX_TIMEOUT_SECONDS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    max_identity_file_bytes: int = DEFAULT_MAX_IDENTITY_FILE_BYTES
    max_self_tests: int = DEFAULT_MAX_SELF_TESTS
    documentation_path: Path | str | None = None

    def __post_init__(self) -> None:
        if (
            self.version_timeout_seconds <= 0
            or self.self_test_timeout_seconds <= 0
            or self.matrix_timeout_seconds <= 0
        ):
            raise ValueError("probe timeouts must be positive")
        if (
            self.max_output_bytes < 1
            or self.max_identity_file_bytes < 1
            or self.max_self_tests < 1
        ):
            raise ValueError("probe limits must be positive")


def _find_spec_without_import(module: str) -> Any:
    path: Sequence[str] | None = None
    parts = str(module).split(".")
    if not parts or any(not part for part in parts):
        return None
    spec: Any = None
    for index in range(len(parts)):
        name = ".".join(parts[: index + 1])
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        if index < len(parts) - 1:
            locations = spec.submodule_search_locations
            if locations is None:
                return None
            path = tuple(str(location) for location in locations)
    return spec


def _limit_text(text: str, maximum: int) -> tuple[str, bool]:
    encoded = str(text).encode("utf-8", errors="replace")
    if len(encoded) <= maximum:
        return str(text), False
    return encoded[:maximum].decode("utf-8", errors="replace"), True


def _default_command_runner(request: CommandRequest) -> CommandResult:
    """Execute a command with deadline and bounded in-memory output."""

    try:
        process = subprocess.Popen(
            list(request.command),
            stdin=subprocess.PIPE if request.stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=request.cwd,
            env={
                **os.environ,
                "NO_PROXY": "*",
                "no_proxy": "*",
            },
            start_new_session=True,
        )
    except OSError as exc:
        return CommandResult(returncode=None, error=f"{type(exc).__name__}: {exc}")

    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}

    def drain(name: str, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(16 * 1024)
                if not chunk:
                    break
                remaining = request.max_output_bytes - len(buffers[name])
                if remaining > 0:
                    buffers[name].extend(chunk[:remaining])
                if len(chunk) > max(0, remaining):
                    truncated[name] = True
        except (OSError, ValueError):
            return

    stdout_thread = threading.Thread(
        target=drain,
        args=("stdout", process.stdout),
        name="prover-matrix-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain,
        args=("stderr", process.stderr),
        name="prover-matrix-stderr",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    def write_stdin() -> None:
        if process.stdin is None or request.stdin_text is None:
            return
        try:
            process.stdin.write(request.stdin_text.encode("utf-8"))
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            return

    stdin_thread = threading.Thread(
        target=write_stdin,
        name="prover-matrix-stdin",
        daemon=True,
    )
    stdin_thread.start()
    try:
        returncode = process.wait(timeout=request.timeout_seconds)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows CI is not used by this project
                process.kill()
        except (OSError, ProcessLookupError):
            process.kill()
        process.wait()
    stdin_thread.join(timeout=1)
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    stdout = bytes(buffers["stdout"]).decode("utf-8", errors="replace")
    stderr = bytes(buffers["stderr"]).decode("utf-8", errors="replace")
    return CommandResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        output_truncated=truncated["stdout"] or truncated["stderr"],
    )


def _normalize_command_result(
    result: CommandResult | Mapping[str, Any],
) -> CommandResult:
    if isinstance(result, CommandResult):
        return result
    if not isinstance(result, Mapping):
        raise TypeError("command runner must return CommandResult or a mapping")
    return CommandResult(
        returncode=result.get("returncode"),
        stdout=str(result.get("stdout") or ""),
        stderr=str(result.get("stderr") or ""),
        timed_out=bool(result.get("timed_out", False)),
        error=str(result["error"]) if result.get("error") else None,
        output_truncated=bool(result.get("output_truncated", False)),
    )


def _file_identity(
    kind: IdentityKind,
    name: str,
    path: str | Path | None,
    *,
    maximum_bytes: int,
) -> BoundIdentity:
    if not path:
        return BoundIdentity.unavailable(kind, name)
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
    except OSError as exc:
        payload = {
            "path": str(resolved),
            "error": f"{type(exc).__name__}: {exc}",
        }
        return BoundIdentity(kind, name, _identity(payload), False, payload)
    metadata: dict[str, Any] = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if resolved.is_file() and stat.st_size <= maximum_bytes:
        digest = hashlib.sha256()
        try:
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            metadata["file_sha256"] = "sha256:" + digest.hexdigest()
            metadata["identity_strength"] = "full_file"
        except OSError as exc:
            metadata["read_error"] = f"{type(exc).__name__}: {exc}"
            metadata["identity_strength"] = "filesystem_metadata"
    else:
        metadata["identity_strength"] = "filesystem_metadata"
        metadata["hash_limit_bytes"] = maximum_bytes
    return BoundIdentity(kind, name, _identity(metadata), True, metadata)


def _package_identity(
    module: str | None,
    version: str | None,
    origin: str | None,
) -> BoundIdentity:
    if not module:
        return BoundIdentity.unavailable(IdentityKind.PACKAGE, "package")
    payload = {
        "module": module,
        "version": version,
        "origin": origin,
    }
    return BoundIdentity(
        IdentityKind.PACKAGE,
        module,
        _identity(payload),
        True,
        payload,
    )


def load_documentation_claims(path: Path | str) -> tuple[DocumentationClaim, ...]:
    """Parse the repository Markdown table as non-runtime claims.

    Only the first three-column ``Prover | Access path | Primary fit`` table is
    accepted.  Free-form prose, including soundness claims, is not promoted.
    """

    source = Path(path).resolve()
    try:
        raw = source.read_bytes()
    except OSError:
        return ()
    source_identity = "sha256:" + hashlib.sha256(raw).hexdigest()
    claims: list[DocumentationClaim] = []
    for line_number, line in enumerate(
        raw.decode("utf-8", errors="replace").splitlines(), start=1
    ):
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 3:
            continue
        if cells[0].lower() == "prover" or all(
            re.fullmatch(r":?-{3,}:?", cell) for cell in cells
        ):
            continue
        claims.append(
            DocumentationClaim(
                source_path=str(source),
                source_identity=source_identity,
                row_number=line_number,
                prover_text=cells[0],
                access_path=cells[1],
                primary_fit=cells[2],
            )
        )
    return tuple(claims)


def _claims_for(
    definition: ProverDefinition,
    claims: Sequence[DocumentationClaim],
) -> tuple[DocumentationClaim, ...]:
    labels = tuple(label.casefold() for label in definition.documentation_labels)
    return tuple(
        claim
        for claim in claims
        if any(label in claim.prover_text.casefold() for label in labels)
    )


class ProverMatrixRegistry:
    """Ordered definitions plus bounded runtime probing dependencies."""

    def __init__(
        self,
        definitions: Sequence[ProverDefinition] | None = None,
        *,
        config: ProverMatrixProbeConfig | None = None,
        which: ExecutableFinder | None = None,
        find_spec: PackageFinder | None = None,
        distribution_version: VersionFinder | None = None,
        command_runner: CommandRunner | None = None,
        monotonic: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or ProverMatrixProbeConfig()
        self._definitions = tuple(definitions or DEFAULT_PROVER_DEFINITIONS)
        ids = [definition.prover_id for definition in self._definitions]
        if len(ids) != len(set(ids)):
            raise ValueError("prover definitions must have unique ids")
        self._which = which or shutil.which
        self._find_spec = find_spec or _find_spec_without_import
        self._distribution_version = distribution_version or importlib.metadata.version
        self._command_runner = command_runner or _default_command_runner
        self._monotonic = monotonic or time.monotonic
        self._wall_clock = wall_clock or time.time

    @property
    def definitions(self) -> tuple[ProverDefinition, ...]:
        return self._definitions

    @classmethod
    def default(cls, **kwargs: Any) -> "ProverMatrixRegistry":
        return cls(DEFAULT_PROVER_DEFINITIONS, **kwargs)

    def probe(self, *, run_self_tests: bool | None = None) -> ProverMatrixSnapshot:
        started = self._monotonic()
        deadline = started + self.config.matrix_timeout_seconds
        execute = (
            self.config.run_self_tests
            if run_self_tests is None
            else bool(run_self_tests)
        )
        documentation_path = self._documentation_path()
        claims = (
            load_documentation_claims(documentation_path)
            if documentation_path is not None
            else ()
        )
        entries: list[ProverMatrixEntry] = []
        self_tests = 0
        for definition in self._definitions:
            allow_test = (
                execute
                and self_tests < self.config.max_self_tests
                and self._monotonic() < deadline
            )
            entry, attempted = self._probe_definition(
                definition,
                claims=_claims_for(definition, claims),
                allow_self_test=allow_test,
                deadline=deadline,
            )
            entries.append(entry)
            self_tests += int(attempted)
        finished = self._monotonic()
        return ProverMatrixSnapshot(
            entries=tuple(entries),
            generated_at=_utc_timestamp(self._wall_clock),
            duration_ms=max(0, round((finished - started) * 1000)),
            self_tests_requested=execute,
            bounded=True,
            max_self_tests=self.config.max_self_tests,
            matrix_timeout_seconds=self.config.matrix_timeout_seconds,
            documentation_source=(
                str(documentation_path.resolve())
                if documentation_path is not None and documentation_path.is_file()
                else None
            ),
        )

    def _documentation_path(self) -> Path | None:
        configured = self.config.documentation_path
        if configured is not None:
            return Path(configured)
        package_root = Path(__file__).resolve().parents[2]
        candidate = package_root / DEFAULT_DOCUMENTATION_MATRIX
        return candidate if candidate.is_file() else None

    def _discover_executable(
        self, definition: ProverDefinition
    ) -> tuple[str | None, str | None]:
        for candidate in definition.executable_candidates:
            try:
                found = self._which(candidate)
            except BaseException:
                continue
            if found:
                return candidate, str(found)
        return None, None

    def _discover_package(
        self, definition: ProverDefinition
    ) -> tuple[str | None, str | None, str | None]:
        distributions = definition.package_distributions or tuple(
            "" for _ in definition.package_modules
        )
        for module, distribution in zip(definition.package_modules, distributions):
            try:
                spec = self._find_spec(module)
            except BaseException:
                continue
            if spec is None:
                continue
            version: str | None = None
            if distribution:
                try:
                    version = str(self._distribution_version(distribution)).strip() or None
                except BaseException:
                    version = None
            origin = getattr(spec, "origin", None)
            return module, version, str(origin) if origin else None
        return None, None, None

    def _run_command(self, request: CommandRequest) -> CommandResult:
        try:
            raw = _normalize_command_result(self._command_runner(request))
        except BaseException as exc:
            return CommandResult(
                returncode=None,
                error=f"runner {type(exc).__name__}: {exc}",
            )
        stdout, stdout_cut = _limit_text(
            raw.stdout, self.config.max_output_bytes
        )
        stderr, stderr_cut = _limit_text(
            raw.stderr, self.config.max_output_bytes
        )
        return CommandResult(
            returncode=raw.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=raw.timed_out,
            error=raw.error,
            output_truncated=(
                raw.output_truncated or stdout_cut or stderr_cut
            ),
        )

    def _probe_version(
        self,
        definition: ProverDefinition,
        executable_path: str,
        *,
        timeout_seconds: float,
    ) -> tuple[str | None, str | None]:
        command = (executable_path, *definition.version_args)
        result = self._run_command(
            CommandRequest(
                command=command,
                stdin_text=None,
                cwd=None,
                timeout_seconds=timeout_seconds,
                max_output_bytes=self.config.max_output_bytes,
            )
        )
        if result.timed_out:
            return None, "version probe timed out"
        if result.error:
            return None, f"version probe failed: {result.error}"
        lines = (result.stdout or result.stderr).strip().splitlines()
        if result.returncode == 0 and lines and lines[0].strip():
            return lines[0].strip(), None
        return None, "version probe returned no successful version text"

    def _probe_definition(
        self,
        definition: ProverDefinition,
        *,
        claims: tuple[DocumentationClaim, ...],
        allow_self_test: bool,
        deadline: float,
    ) -> tuple[ProverMatrixEntry, bool]:
        executable_name, executable_path = self._discover_executable(definition)
        package_module, package_version, package_origin = self._discover_package(definition)
        discovered = bool(executable_path or package_module)
        executable_version: str | None = None
        version_reason: str | None = None
        if executable_path:
            remaining = deadline - self._monotonic()
            if remaining > 0:
                executable_version, version_reason = self._probe_version(
                    definition,
                    executable_path,
                    timeout_seconds=min(
                        self.config.version_timeout_seconds, remaining
                    ),
                )
            else:
                version_reason = "matrix time budget exhausted before version probe"
        versioned = bool(executable_version or package_version)
        receipt: ProverSelfTestReceipt | None = None
        attempted = False
        if (
            allow_self_test
            and versioned
            and executable_path
            and definition.fixture is not None
        ):
            remaining = deadline - self._monotonic()
            if remaining > 0:
                attempted = True
                receipt = self._execute_fixture(
                    definition,
                    executable_name=executable_name or Path(executable_path).name,
                    executable_path=executable_path,
                    package_module=package_module,
                    package_version=package_version,
                    package_origin=package_origin,
                    timeout_seconds=min(
                        self.config.self_test_timeout_seconds, remaining
                    ),
                )
        passed = bool(receipt and receipt.status is SelfTestStatus.PASSED)
        translation = bool(passed and receipt and receipt.translation_conformant)
        reconstruction = bool(passed and receipt and receipt.reconstruction_capable)
        authorities = receipt.authoritative_for if passed and receipt else ()
        if not discovered:
            reason = "no configured executable or package was discovered"
        elif not versioned:
            reason = version_reason or (
                "a component was discovered but no version identity was available"
            )
        elif receipt is not None:
            reason = receipt.reason
        elif not allow_self_test:
            reason = "versioned; bounded self-test was not requested or budgeted"
        elif definition.fixture is None:
            reason = "versioned; no reviewed executable fixture is registered"
        elif not executable_path:
            reason = "versioned package discovered; no isolated executable fixture runner is registered"
        else:
            reason = "versioned; self-test was not run"
        return (
            ProverMatrixEntry(
                prover_id=definition.prover_id,
                display_name=definition.display_name,
                family=definition.family,
                absent=not discovered,
                discovered=discovered,
                versioned=versioned,
                smoke_tested=passed,
                translation_conformant=translation,
                reconstruction_capable=reconstruction,
                authoritative_for=authorities,
                executable_path=executable_path,
                executable_version=executable_version,
                package_module=package_module,
                package_version=package_version,
                reason=reason,
                receipt=receipt,
                documentation_claims=claims,
            ),
            attempted,
        )

    def _execute_fixture(
        self,
        definition: ProverDefinition,
        *,
        executable_name: str,
        executable_path: str,
        package_module: str | None,
        package_version: str | None,
        package_origin: str | None,
        timeout_seconds: float,
    ) -> ProverSelfTestReceipt:
        fixture = definition.fixture
        assert fixture is not None
        binding = SelfTestBinding(
            executable=_file_identity(
                IdentityKind.EXECUTABLE,
                executable_name,
                executable_path,
                maximum_bytes=self.config.max_identity_file_bytes,
            ),
            package=_package_identity(
                package_module, package_version, package_origin
            ),
            model=fixture.model_identity,
            translator=fixture.translator_identity,
            semantic_profile=fixture.semantic_profile_identity,
            fixture=fixture.fixture_identity,
        )
        started_mono = self._monotonic()
        started_at = _utc_timestamp(self._wall_clock)
        with tempfile.TemporaryDirectory(prefix="prover-matrix-") as temp_dir:
            fixture_path = Path(temp_dir) / fixture.file_name
            fixture_path.write_text(fixture.model_text, encoding="utf-8")
            replacements = {
                "{fixture}": str(fixture_path),
                "{fixture_dir}": temp_dir,
                "{executable}": executable_path,
            }
            args = tuple(replacements.get(arg, arg) for arg in fixture.args)
            command = (executable_path, *args)
            result = self._run_command(
                CommandRequest(
                    command=command,
                    stdin_text=fixture.model_text if fixture.stdin else None,
                    cwd=temp_dir,
                    timeout_seconds=timeout_seconds,
                    max_output_bytes=self.config.max_output_bytes,
                )
            )
        duration_ms = max(0, round((self._monotonic() - started_mono) * 1000))
        combined = (result.stdout + "\n" + result.stderr).casefold()
        output_lines = {
            line.strip().casefold()
            for line in (result.stdout + "\n" + result.stderr).splitlines()
            if line.strip()
        }
        all_match = all(
            expected.casefold() in combined
            for expected in fixture.expected_output_all
        )
        any_match = (
            not fixture.expected_output_any
            or any(
                expected.casefold() in combined
                for expected in fixture.expected_output_any
            )
        )
        lines_match = all(
            expected.strip().casefold() in output_lines
            for expected in fixture.expected_output_lines
        )
        passed = (
            not result.timed_out
            and result.error is None
            and not result.output_truncated
            and binding.executable.available
            and result.returncode in fixture.expected_exit_codes
            and all_match
            and any_match
            and lines_match
        )
        if result.timed_out:
            status = SelfTestStatus.TIMED_OUT
            reason = f"bounded fixture timed out after {timeout_seconds:g}s"
        elif result.error:
            status = SelfTestStatus.ERROR
            reason = f"bounded fixture runner failed: {result.error}"
        elif passed:
            status = SelfTestStatus.PASSED
            reason = "bounded identity-bound fixture passed"
        else:
            status = SelfTestStatus.FAILED
            reason = (
                "bounded fixture did not meet its reviewed exit/output expectation"
            )
        allowed_authorities = set(definition.maximum_authoritative_for)
        authorities = (
            tuple(
                authority
                for authority in fixture.authoritative_for
                if authority in allowed_authorities
            )
            if passed
            else ()
        )
        canonical_command = (executable_path, *fixture.args)
        command_payload = {
            "command": list(canonical_command),
            "binding_id": binding.binding_id,
            "timeout_seconds": timeout_seconds,
            "max_output_bytes": self.config.max_output_bytes,
        }
        return ProverSelfTestReceipt(
            prover_id=definition.prover_id,
            status=status,
            binding=binding,
            command=canonical_command,
            command_identity=_identity(command_payload),
            started_at=started_at,
            duration_ms=duration_ms,
            timeout_seconds=timeout_seconds,
            max_output_bytes=self.config.max_output_bytes,
            returncode=result.returncode,
            stdout_sha256="sha256:"
            + hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
            stderr_sha256="sha256:"
            + hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
            output_truncated=result.output_truncated,
            reason=reason,
            translation_conformant=(
                passed and fixture.establishes_translation_conformance
            ),
            reconstruction_capable=(
                passed and fixture.establishes_reconstruction
            ),
            authoritative_for=authorities,
        )


def _fixture(
    fixture_id: str,
    model_text: str,
    file_name: str,
    args: Sequence[str],
    *,
    stdin: bool = False,
    output_any: Sequence[str] = (),
    output_lines: Sequence[str] = (),
    translator: str,
    semantics: str,
    conformant: bool = False,
    reconstruction: bool = False,
    authority: Sequence[str] = (),
) -> ProverFixture:
    return ProverFixture(
        fixture_id=fixture_id,
        model_text=model_text,
        file_name=file_name,
        translator_id=translator,
        translator_version="1",
        semantic_profile_id=semantics,
        semantic_profile_version="1",
        args=tuple(args),
        stdin=stdin,
        expected_output_any=tuple(output_any),
        expected_output_lines=tuple(output_lines),
        establishes_translation_conformance=conformant,
        establishes_reconstruction=reconstruction,
        authoritative_for=tuple(authority),
    )


_SMT_MODEL = "(set-logic QF_UF)\n(declare-const p Bool)\n(assert p)\n(check-sat)\n"
_TPTP_MODEL = "fof(matrix_smoke, conjecture, $true).\n"


DEFAULT_PROVER_DEFINITIONS: tuple[ProverDefinition, ...] = (
    ProverDefinition(
        "z3", "Z3", "smt", ("z3",), ("z3",), ("z3-solver",),
        fixture=_fixture(
            "z3-smtlib-smoke@1", _SMT_MODEL, "matrix.smt2", ("-in",),
            stdin=True, output_lines=("sat",), translator="supervisor-smtlib",
            semantics="smtlib-qf-uf", conformant=True,
            authority=("finite_constraint_satisfiability",),
        ),
        maximum_authoritative_for=("finite_constraint_satisfiability",),
        documentation_labels=("Z3",),
    ),
    ProverDefinition(
        "cvc5", "CVC5", "smt", ("cvc5",), ("cvc5",), ("cvc5",),
        fixture=_fixture(
            "cvc5-smtlib-smoke@1", _SMT_MODEL, "matrix.smt2",
            ("--lang=smt2",), stdin=True, output_lines=("sat",),
            translator="supervisor-smtlib", semantics="smtlib-qf-uf",
            conformant=True, authority=("finite_constraint_satisfiability",),
        ),
        maximum_authoritative_for=("finite_constraint_satisfiability",),
        documentation_labels=("CVC5",),
    ),
    ProverDefinition(
        "tla_tlc", "TLA+/TLC", "state_machine", ("tlc", "tlc2"),
        fixture=_fixture(
            "tlc-state-smoke@1",
            "---- MODULE MatrixSmoke ----\nVARIABLE x\nInit == x = 0\nNext == x' = 1 - x\nSpec == Init /\\ [][Next]_x\n====\n",
            "MatrixSmoke.tla", ("{fixture}",), output_any=("model checking completed", "no error"),
            translator="supervisor-tla", semantics="tla-finite-state",
            conformant=True, authority=("bounded_state_machine",),
        ),
        maximum_authoritative_for=("bounded_state_machine",),
        documentation_labels=("TLA+", "TLC"),
    ),
    ProverDefinition(
        "apalache", "Apalache", "state_machine", ("apalache-mc", "apalache"),
        version_args=("version",),
        fixture=_fixture(
            "apalache-state-smoke@1",
            "---- MODULE MatrixSmoke ----\nVARIABLE x\nInit == x = 0\nNext == x' = 1\nInv == x \\in {0, 1}\n====\n",
            "MatrixSmoke.tla", ("check", "--inv=Inv", "{fixture}"),
            output_any=("checking", "pass", "completed"), translator="supervisor-tla",
            semantics="apalache-bounded-symbolic", conformant=True,
            authority=("bounded_state_machine",),
        ),
        maximum_authoritative_for=("bounded_state_machine",),
        documentation_labels=("Apalache",),
    ),
    ProverDefinition(
        "datalog_secpal", "Datalog/SecPAL", "authorization",
        ("souffle", "runergo"), ("pyDatalog",), ("pyDatalog",),
        fixture=_fixture(
            "datalog-authorization-smoke@1",
            '.decl allowed(actor:symbol)\n.output allowed\nallowed("agent").\n',
            "matrix.dl", ("{fixture}",), translator="supervisor-datalog",
            semantics="secpal-finite-delegation", conformant=True,
            authority=("authorization_policy",),
        ),
        maximum_authoritative_for=("authorization_policy",),
        documentation_labels=("Datalog", "SecPAL"),
    ),
    ProverDefinition(
        "tamarin", "Tamarin", "protocol", ("tamarin-prover",),
        fixture=_fixture(
            "tamarin-protocol-smoke@1",
            "theory MatrixSmoke begin\nrule Emit: [ Fr(~x) ] --[ Seen(~x) ]-> [ ]\nlemma exists_trace: exists-trace \"Ex x #i. Seen(x) @ i\"\nend\n",
            "matrix.spthy", ("--prove", "{fixture}"), output_any=("verified",),
            translator="supervisor-tamarin", semantics="tamarin-trace",
            conformant=True, authority=("protocol_trace_property",),
        ),
        maximum_authoritative_for=("protocol_trace_property",),
        documentation_labels=("Tamarin",),
    ),
    ProverDefinition(
        "proverif", "ProVerif", "protocol", ("proverif",),
        version_args=("-version",),
        fixture=_fixture(
            "proverif-protocol-smoke@1",
            "free c: channel.\nfree secret: bitstring [private].\nquery attacker(secret).\nprocess 0\n",
            "matrix.pv", ("{fixture}",), output_any=("result",),
            translator="supervisor-proverif", semantics="proverif-process",
            conformant=True, authority=("protocol_reachability",),
        ),
        maximum_authoritative_for=("protocol_reachability",),
        documentation_labels=("ProVerif",),
    ),
    ProverDefinition(
        "hyperltl_autohyper_mchyper", "HyperLTL/AutoHyper/MCHyper",
        "hyperproperty", ("autohyper", "mchyper", "hyperltl"),
        package_modules=("autohyper",), package_distributions=("autohyper",),
        fixture=None, documentation_labels=("HyperLTL", "AutoHyper", "MCHyper"),
    ),
    ProverDefinition(
        "lean", "Lean", "kernel", ("lean",),
        package_modules=("ipfs_datasets_py.logic.external_provers.interactive.lean_prover_bridge",),
        package_distributions=("",),
        fixture=_fixture(
            "lean-kernel-smoke@1", "example : True := True.intro\n",
            "MatrixSmoke.lean", ("{fixture}",), translator="supervisor-lean",
            semantics="lean-kernel", conformant=True, reconstruction=True,
            authority=("lean_kernel_check",),
        ),
        maximum_authoritative_for=("lean_kernel_check",),
        documentation_labels=("Lean",),
    ),
    ProverDefinition(
        "coq", "Coq", "kernel", ("coqc", "rocq"),
        package_modules=("ipfs_datasets_py.logic.external_provers.interactive.coq_prover_bridge",),
        package_distributions=("",),
        fixture=_fixture(
            "coq-kernel-smoke@1", "Example matrix_smoke : True. exact I. Qed.\n",
            "MatrixSmoke.v", ("{fixture}",), translator="supervisor-coq",
            semantics="coq-kernel", conformant=True, reconstruction=True,
            authority=("coq_kernel_check",),
        ),
        maximum_authoritative_for=("coq_kernel_check",),
        documentation_labels=("Coq",),
    ),
    ProverDefinition(
        "runtime_mtl", "Runtime MTL", "runtime_monitor",
        ("rtamt",), ("rtamt",), ("rtamt",), fixture=None,
        documentation_labels=("Runtime MTL",),
    ),
    ProverDefinition(
        "dcec", "DCEC", "temporal_deontic",
        ("shadow-prover",), ("ipfs_datasets_py.logic.CEC",), ("",),
        fixture=None, documentation_labels=("DCEC",),
    ),
    ProverDefinition(
        "tdfol", "TDFOL", "temporal_first_order",
        ("tdfol",), ("ipfs_datasets_py.logic.TDFOL",), ("",),
        fixture=None, documentation_labels=("TDFOL",),
    ),
    ProverDefinition(
        "hammer", "Hammer", "proof_orchestration",
        ("ipfs-hammer",), ("ipfs_datasets_py.logic.hammers",), ("",),
        fixture=None, documentation_labels=("Hammer",),
    ),
    ProverDefinition(
        "vampire", "Vampire", "atp", ("vampire",),
        fixture=_fixture(
            "vampire-tptp-smoke@1", _TPTP_MODEL, "matrix.p",
            ("--mode", "casc", "{fixture}"), output_any=("theorem", "refutation"),
            translator="supervisor-tptp", semantics="tptp-fol", conformant=True,
            authority=("first_order_theorem",),
        ),
        maximum_authoritative_for=("first_order_theorem",),
        documentation_labels=("Vampire",),
    ),
    ProverDefinition(
        "e", "E", "atp", ("eprover",),
        fixture=_fixture(
            "e-tptp-smoke@1", _TPTP_MODEL, "matrix.p",
            ("--auto", "{fixture}"), output_any=("theorem", "proof found"),
            translator="supervisor-tptp", semantics="tptp-fol", conformant=True,
            authority=("first_order_theorem",),
        ),
        maximum_authoritative_for=("first_order_theorem",),
        documentation_labels=("E prover",),
    ),
    ProverDefinition(
        "isabelle", "Isabelle", "kernel", ("isabelle",),
        package_modules=("ipfs_datasets_py.logic.hammers.isabelle",),
        package_distributions=("",), fixture=None,
        documentation_labels=("Isabelle",),
    ),
    ProverDefinition(
        "shadowprover", "ShadowProver", "modal",
        ("shadow-prover",),
        ("ipfs_datasets_py.logic.CEC.native.shadow_prover",), ("",),
        fixture=None, documentation_labels=("ShadowProver",),
    ),
    ProverDefinition(
        "leanstral", "Leanstral", "model_assistant",
        ("leanstral",), ("ipfs_datasets_py.logic.modal.leanstral",), ("",),
        fixture=None, documentation_labels=("Leanstral",),
    ),
    ProverDefinition(
        "zkp_backends", "ZKP backends", "attestation",
        ("groth16", "provekit-cli"),
        ("ipfs_datasets_py.logic.zkp.backends",), ("",),
        fixture=None, documentation_labels=("ZKP",),
    ),
)

EXPECTED_PROVER_IDS = frozenset(
    definition.prover_id for definition in DEFAULT_PROVER_DEFINITIONS
)


@dataclass(frozen=True)
class ProverMatrixPaths:
    json_path: Path
    duckdb_path: Path


def prover_matrix_paths(path: Path | str) -> ProverMatrixPaths:
    resolved = Path(path).resolve()
    if resolved.suffix.lower() == ".json":
        return ProverMatrixPaths(resolved, resolved.with_suffix(".duckdb"))
    if resolved.suffix.lower() == ".duckdb":
        return ProverMatrixPaths(resolved.with_suffix(".json"), resolved)
    raise ValueError("prover matrix output path must end in .json or .duckdb")


def _duckdb_module() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise RuntimeError("DuckDB is required for prover matrix projection") from exc
    return duckdb


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_prover_matrix_projection(
    path: Path | str, snapshot: ProverMatrixSnapshot
) -> dict[str, Any]:
    """Atomically materialize equivalent portable JSON and normalized DuckDB."""

    if not isinstance(snapshot, ProverMatrixSnapshot):
        raise TypeError("snapshot must be a ProverMatrixSnapshot")
    paths = prover_matrix_paths(path)
    payload = snapshot.to_dict()
    payload["query_store"] = {
        "schema_version": PROVER_MATRIX_DUCKDB_SCHEMA_VERSION,
        "duckdb_path": paths.duckdb_path.name,
        "tables": [
            "prover_capabilities",
            "prover_authorities",
            "prover_components",
            "prover_self_tests",
            "documentation_claims",
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_write(paths.json_path, rendered)
    _write_prover_matrix_duckdb(
        paths.duckdb_path,
        payload,
        source_path=paths.json_path,
        source_sha256="sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    )
    return payload


def _write_prover_matrix_duckdb(
    path: Path,
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
) -> None:
    duckdb = _duckdb_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    try:
        connection = duckdb.connect(str(temporary))
        try:
            connection.execute(
                """CREATE TABLE prover_matrix_catalog (
                    schema_version VARCHAR NOT NULL,
                    report_version INTEGER NOT NULL,
                    snapshot_id VARCHAR NOT NULL,
                    generated_at VARCHAR NOT NULL,
                    source_path VARCHAR NOT NULL,
                    source_sha256 VARCHAR NOT NULL,
                    documentation_is_runtime_evidence BOOLEAN NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE prover_capabilities (
                    prover_id VARCHAR PRIMARY KEY,
                    display_name VARCHAR NOT NULL,
                    family VARCHAR NOT NULL,
                    highest_state VARCHAR NOT NULL,
                    absent BOOLEAN NOT NULL,
                    discovered BOOLEAN NOT NULL,
                    versioned BOOLEAN NOT NULL,
                    smoke_tested BOOLEAN NOT NULL,
                    translation_conformant BOOLEAN NOT NULL,
                    reconstruction_capable BOOLEAN NOT NULL,
                    executable_path VARCHAR,
                    executable_version VARCHAR,
                    package_module VARCHAR,
                    package_version VARCHAR,
                    reason VARCHAR NOT NULL,
                    receipt_id VARCHAR,
                    documentation_claim_count BIGINT NOT NULL,
                    payload_json VARCHAR NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE prover_authorities (
                    prover_id VARCHAR NOT NULL,
                    property_class VARCHAR NOT NULL,
                    receipt_id VARCHAR NOT NULL,
                    PRIMARY KEY (prover_id, property_class)
                )"""
            )
            connection.execute(
                """CREATE TABLE prover_components (
                    prover_id VARCHAR NOT NULL,
                    component_kind VARCHAR NOT NULL,
                    component_name VARCHAR NOT NULL,
                    content_identity VARCHAR NOT NULL,
                    available BOOLEAN NOT NULL,
                    metadata_json VARCHAR NOT NULL,
                    PRIMARY KEY (prover_id, component_kind)
                )"""
            )
            connection.execute(
                """CREATE TABLE prover_self_tests (
                    receipt_id VARCHAR PRIMARY KEY,
                    prover_id VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    binding_id VARCHAR NOT NULL,
                    command_identity VARCHAR NOT NULL,
                    started_at VARCHAR NOT NULL,
                    duration_ms BIGINT NOT NULL,
                    timeout_seconds DOUBLE NOT NULL,
                    max_output_bytes BIGINT NOT NULL,
                    returncode INTEGER,
                    output_truncated BOOLEAN NOT NULL,
                    translation_conformant BOOLEAN NOT NULL,
                    reconstruction_capable BOOLEAN NOT NULL,
                    reason VARCHAR NOT NULL,
                    payload_json VARCHAR NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE documentation_claims (
                    claim_id VARCHAR NOT NULL,
                    prover_id VARCHAR NOT NULL,
                    source_path VARCHAR NOT NULL,
                    source_identity VARCHAR NOT NULL,
                    row_number BIGINT NOT NULL,
                    prover_text VARCHAR NOT NULL,
                    access_path VARCHAR NOT NULL,
                    primary_fit VARCHAR NOT NULL,
                    evidence_class VARCHAR NOT NULL,
                    runtime_evidence BOOLEAN NOT NULL,
                    PRIMARY KEY (claim_id, prover_id)
                )"""
            )
            connection.execute(
                "INSERT INTO prover_matrix_catalog VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    payload["schema_version"],
                    payload["report_version"],
                    payload["snapshot_id"],
                    payload["generated_at"],
                    str(source_path),
                    source_sha256,
                    False,
                ),
            )
            entries = payload.get("entries", {})
            for prover_id in sorted(entries):
                entry = entries[prover_id]
                states = entry["states"]
                receipt = entry.get("self_test_receipt")
                receipt_id = receipt.get("receipt_id") if receipt else None
                connection.execute(
                    "INSERT INTO prover_capabilities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        prover_id,
                        entry["display_name"],
                        entry["family"],
                        entry["highest_state"],
                        states["absent"],
                        states["discovered"],
                        states["versioned"],
                        states["smoke_tested"],
                        states["translation_conformant"],
                        states["reconstruction_capable"],
                        entry["executable"]["path"],
                        entry["executable"]["version"],
                        entry["package"]["module"],
                        entry["package"]["version"],
                        entry["reason"],
                        receipt_id,
                        len(entry["documentation_claims"]),
                        _canonical_json(entry),
                    ),
                )
                for authority in states["authoritative_for"]:
                    connection.execute(
                        "INSERT INTO prover_authorities VALUES (?, ?, ?)",
                        (prover_id, authority, receipt_id),
                    )
                if receipt:
                    connection.execute(
                        "INSERT INTO prover_self_tests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            receipt_id,
                            prover_id,
                            receipt["status"],
                            receipt["binding"]["binding_id"],
                            receipt["command_identity"],
                            receipt["started_at"],
                            receipt["duration_ms"],
                            receipt["timeout_seconds"],
                            receipt["max_output_bytes"],
                            receipt["returncode"],
                            receipt["output_truncated"],
                            receipt["translation_conformant"],
                            receipt["reconstruction_capable"],
                            receipt["reason"],
                            _canonical_json(receipt),
                        ),
                    )
                    for kind in IdentityKind:
                        component = receipt["binding"][kind.value]
                        connection.execute(
                            "INSERT INTO prover_components VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                prover_id,
                                kind.value,
                                component["name"],
                                component["content_identity"],
                                component["available"],
                                _canonical_json(component["metadata"]),
                            ),
                        )
                for claim in entry["documentation_claims"]:
                    connection.execute(
                        "INSERT INTO documentation_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            claim["claim_id"],
                            prover_id,
                            claim["source_path"],
                            claim["source_identity"],
                            claim["row_number"],
                            claim["prover_text"],
                            claim["access_path"],
                            claim["primary_fit"],
                            claim["evidence_class"],
                            claim["runtime_evidence"],
                        ),
                    )
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}.wal").unlink(missing_ok=True)


_QUERY_TABLES = frozenset(
    {
        "prover_matrix_catalog",
        "prover_capabilities",
        "prover_authorities",
        "prover_components",
        "prover_self_tests",
        "documentation_claims",
    }
)
_QUERY_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def query_prover_matrix(
    path: Path | str,
    *,
    table: str = "prover_capabilities",
    columns: Sequence[str] = ("*",),
    where: str | None = None,
    parameters: Sequence[Any] = (),
    limit: int = 100,
) -> dict[str, Any]:
    """Run a bounded SELECT over an allowlisted matrix projection table."""

    if table not in _QUERY_TABLES:
        raise ValueError(f"unsupported prover matrix table: {table}")
    if not 1 <= limit <= 1_000:
        raise ValueError("query limit must be between 1 and 1000")
    if not columns:
        raise ValueError("at least one query column is required")
    if any(column != "*" and not _QUERY_IDENTIFIER.fullmatch(column) for column in columns):
        raise ValueError("query columns must be simple identifiers")
    if where and (";" in where or re.search(r"\b(insert|update|delete|drop|attach|copy)\b", where, re.I)):
        raise ValueError("matrix query predicate must be read-only")
    paths = prover_matrix_paths(path)
    if not paths.duckdb_path.is_file():
        raise FileNotFoundError(paths.duckdb_path)
    sql = f"SELECT {', '.join(columns)} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    sql += " LIMIT ?"
    connection = _duckdb_module().connect(str(paths.duckdb_path), read_only=True)
    try:
        cursor = connection.execute(sql, [*parameters, limit])
        names = [description[0] for description in cursor.description]
        rows = [dict(zip(names, row)) for row in cursor.fetchall()]
    finally:
        connection.close()
    return {"table": table, "columns": names, "rows": rows, "limit": limit}


def probe_prover_matrix(
    *,
    config: ProverMatrixProbeConfig | None = None,
    run_self_tests: bool | None = None,
) -> ProverMatrixSnapshot:
    """Convenience entry point for the default repository registry."""

    return ProverMatrixRegistry.default(config=config).probe(
        run_self_tests=run_self_tests
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-self-tests",
        action="store_true",
        help="project discovery/version state without executing fixtures",
    )
    parser.add_argument("--documentation-path", type=Path)
    parser.add_argument(
        "--self-test-timeout",
        type=float,
        default=DEFAULT_SELF_TEST_TIMEOUT_SECONDS,
    )
    args = parser.parse_args(argv)
    config = ProverMatrixProbeConfig(
        run_self_tests=not args.no_self_tests,
        self_test_timeout_seconds=args.self_test_timeout,
        documentation_path=args.documentation_path,
    )
    snapshot = ProverMatrixRegistry.default(config=config).probe()
    write_prover_matrix_projection(args.output, snapshot)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot.snapshot_id,
                "json_path": str(prover_matrix_paths(args.output).json_path),
                "duckdb_path": str(prover_matrix_paths(args.output).duckdb_path),
                "counts": snapshot.to_dict()["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI
    raise SystemExit(_main())
