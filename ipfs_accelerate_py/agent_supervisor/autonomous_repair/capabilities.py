"""DCR-004 deterministic capability and local-toolchain inventory.

The inventory is deliberately a *read-only receipt builder*.  It never imports
``ipfs_datasets_py.logic`` modules, executes a solver, installs a dependency,
or opens a network connection.  A package merely being importable is therefore
not evidence of readiness: exact origin, distribution version, source digest,
symbols, initialization/reconstruction declarations, and pre-recorded local
self-test evidence are all required.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final

DETERMINISTIC_REPAIR_CAPABILITIES_INTERFACE: Final[str] = "DeterministicRepairCapabilities@1"
SOLVER_READINESS_INTERFACE: Final[str] = "SolverReadiness@1"
DETERMINISTIC_REPAIR_CAPABILITIES_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-capabilities@1"
)
DETERMINISTIC_REPAIR_CAPABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-capability@1"
)
SOLVER_READINESS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-solver-readiness@1"
)
CAPABILITY_EVIDENCE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-capability-evidence@1"
)

_CAPABILITY_EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {"initialization", "reconstruction", "self_test"}
)

_SOURCE_RISK = re.compile(
    rb"\b(?:todo|stub|simulated|simulation|notimplementederror)\b", re.IGNORECASE
)


class CapabilityStatus(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Only ``AVAILABLE`` may be selected by a deterministic repair runtime."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class NetworkMode(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """This inventory has no network authority; authoritative mode is offline."""

    OFFLINE = "offline"
    UNSPECIFIED = "unspecified"
    NETWORKED = "networked"


@dataclass(frozen=True)
class CapabilityEvidenceReceipt:
    """Immutable evidence bound to the exact bytes and version it exercised.

    The read-only inventory never executes a self-test itself.  It accepts
    evidence only in this canonical shape; caller-supplied booleans cannot
    become readiness authority and receipts for stale bytes cannot be reused.
    """

    evidence_id: str
    evidence_kind: str
    subject_id: str
    subject_digest: str
    subject_version: str
    transcript_digest: str
    passed: bool
    network_mode: NetworkMode = NetworkMode.OFFLINE
    model_call_count: int = 0
    producer_id: str = "deterministic-capability-self-test-runner@1"

    def __post_init__(self) -> None:
        values = (
            self.evidence_id,
            self.subject_id,
            self.subject_digest,
            self.subject_version,
            self.transcript_digest,
            self.producer_id,
        )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("capability evidence fields must be non-empty text")
        if self.evidence_kind not in _CAPABILITY_EVIDENCE_KINDS:
            raise ValueError("unsupported capability evidence kind")
        if not self.subject_digest.startswith(("module:sha256:", "executable:sha256:")):
            raise ValueError("capability evidence subject digest is not canonical")
        if not self.transcript_digest.startswith("transcript:sha256:"):
            raise ValueError("capability evidence transcript digest is not canonical")
        if type(self.passed) is not bool:
            raise ValueError("capability evidence passed must be boolean")
        if isinstance(self.network_mode, str):
            object.__setattr__(self, "network_mode", NetworkMode(self.network_mode))
        if isinstance(self.model_call_count, bool) or self.model_call_count != 0:
            raise ValueError("capability evidence model_call_count must be exactly zero")

    @property
    def receipt_id(self) -> str:
        return _digest_object(self.to_dict(include_receipt=False), label="capability-evidence")

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CAPABILITY_EVIDENCE_RECEIPT_SCHEMA,
            "evidence_id": self.evidence_id,
            "evidence_kind": self.evidence_kind,
            "subject_id": self.subject_id,
            "subject_digest": self.subject_digest,
            "subject_version": self.subject_version,
            "transcript_digest": self.transcript_digest,
            "passed": self.passed,
            "network_mode": self.network_mode.value,
            "model_call_count": self.model_call_count,
            "producer_id": self.producer_id,
        }
        if include_receipt:
            payload["receipt_id"] = self.receipt_id
        return payload

    def verifies(
        self,
        *,
        evidence_id: str,
        evidence_kind: str,
        subject_id: str,
        subject_digest: str,
        subject_version: str,
    ) -> bool:
        return (
            self.evidence_id == evidence_id
            and self.evidence_kind == evidence_kind
            and self.subject_id == subject_id
            and self.subject_digest == subject_digest
            and self.subject_version == subject_version
            and self.passed
            and self.network_mode is NetworkMode.OFFLINE
            and self.model_call_count == 0
        )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest_bytes(value: bytes, *, label: str) -> str:
    return f"{label}:sha256:{hashlib.sha256(value).hexdigest()}"


def _digest_object(value: Any, *, label: str) -> str:
    return _digest_bytes(_canonical_json(value), label=label)


def _safe_spec(module: str) -> Any | None:
    """Resolve without running any package initializer."""

    parts = tuple(part for part in module.split(".") if part)
    if not parts:
        return None
    spec: Any | None = None
    parent = ""
    for index, part in enumerate(parts):
        name = part if not parent else f"{parent}.{part}"
        search_path = None if spec is None else spec.submodule_search_locations
        spec = (
            importlib.machinery.PathFinder.find_spec(name)
            if index == 0
            else importlib.machinery.PathFinder.find_spec(name, search_path)
        )
        if spec is None:
            return None
        parent = name
    return spec


def _source_symbols(source: bytes) -> tuple[set[str], dict[str, Any]]:
    """Statically collect public definitions and literal initialization flags."""

    tree = ast.parse(source.decode("utf-8"), mode="exec")
    symbols: set[str] = set()
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
                    values[target.id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            symbols.add(node.target.id)
            values[node.target.id] = value
    return symbols, values


@dataclass(frozen=True)
class LogicModuleRequirement:
    """An exact source-level contract for one datasets logic module."""

    module: str
    distribution: str
    expected_version: str
    required_symbols: tuple[str, ...]
    initialization_symbol: str = "INITIALIZED"
    reconstruction_symbol: str = "RECONSTRUCTION_READY"

    def __post_init__(self) -> None:
        if not self.module.startswith("ipfs_datasets_py.logic."):
            raise ValueError("logic module must be under ipfs_datasets_py.logic")
        if not self.distribution.strip() or not self.expected_version.strip():
            raise ValueError("module distribution and expected_version are required")
        if not self.required_symbols or any(not item.strip() for item in self.required_symbols):
            raise ValueError("module requirements need exact required symbols")
        if not self.initialization_symbol or not self.reconstruction_symbol:
            raise ValueError("initialization and reconstruction symbols are required")


@dataclass(frozen=True)
class ToolchainRequirement:
    """An exact local executable contract; no executable is run by the probe."""

    tool_id: str
    executable: str
    expected_version: str
    self_test_id: str
    reconstruction_id: str

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.tool_id,
                self.executable,
                self.expected_version,
                self.self_test_id,
                self.reconstruction_id,
            )
        ):
            raise ValueError("toolchain requirements require non-empty exact fields")


@dataclass(frozen=True)
class CapabilityReceipt:
    """Content-addressed read-only receipt for a logic source capability."""

    capability_id: str
    status: CapabilityStatus
    origin: str = ""
    distribution: str = ""
    expected_version: str = ""
    distribution_version: str = ""
    content_digest: str = ""
    symbols: tuple[str, ...] = ()
    missing_symbols: tuple[str, ...] = ()
    initialized: bool = False
    reconstructed: bool = False
    self_test_passed: bool = False
    network_mode: NetworkMode = NetworkMode.UNSPECIFIED
    reason_codes: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.status is CapabilityStatus.AVAILABLE

    @property
    def receipt_id(self) -> str:
        return _digest_object(self.to_dict(include_receipt=False), label="capability-receipt")

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DETERMINISTIC_REPAIR_CAPABILITY_SCHEMA,
            "capability_id": self.capability_id,
            "status": self.status.value,
            "available": self.available,
            "origin": self.origin,
            "distribution": self.distribution,
            "expected_version": self.expected_version,
            "distribution_version": self.distribution_version,
            "content_digest": self.content_digest,
            "symbols": list(self.symbols),
            "missing_symbols": list(self.missing_symbols),
            "initialized": self.initialized,
            "reconstructed": self.reconstructed,
            "self_test_passed": self.self_test_passed,
            "network_mode": self.network_mode.value,
            "reason_codes": list(self.reason_codes),
        }
        if include_receipt:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class SolverReadiness:
    """Content-addressed readiness for a local solver/prover executable."""

    tool_id: str
    status: CapabilityStatus
    executable: str
    path: str = ""
    expected_version: str = ""
    version: str = ""
    executable_digest: str = ""
    self_test_id: str = ""
    self_test_passed: bool = False
    reconstruction_id: str = ""
    reconstructed: bool = False
    network_mode: NetworkMode = NetworkMode.UNSPECIFIED
    reason_codes: tuple[str, ...] = ()

    INTERFACE: ClassVar[Final[str]] = SOLVER_READINESS_INTERFACE

    @property
    def available(self) -> bool:
        return self.status is CapabilityStatus.AVAILABLE

    @property
    def receipt_id(self) -> str:
        return _digest_object(self.to_dict(include_receipt=False), label="solver-readiness")

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SOLVER_READINESS_SCHEMA,
            "interface": self.INTERFACE,
            "tool_id": self.tool_id,
            "status": self.status.value,
            "available": self.available,
            "executable": self.executable,
            "path": self.path,
            "expected_version": self.expected_version,
            "version": self.version,
            "executable_digest": self.executable_digest,
            "self_test_id": self.self_test_id,
            "self_test_passed": self.self_test_passed,
            "reconstruction_id": self.reconstruction_id,
            "reconstructed": self.reconstructed,
            "network_mode": self.network_mode.value,
            "reason_codes": list(self.reason_codes),
        }
        if include_receipt:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class DeterministicRepairCapabilities:
    """The complete immutable inventory; unavailable entries cannot select."""

    modules: tuple[CapabilityReceipt, ...]
    toolchains: tuple[SolverReadiness, ...]
    network_mode: NetworkMode

    INTERFACE: ClassVar[Final[str]] = DETERMINISTIC_REPAIR_CAPABILITIES_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "modules", tuple(sorted(self.modules, key=lambda item: item.capability_id))
        )
        object.__setattr__(
            self, "toolchains", tuple(sorted(self.toolchains, key=lambda item: item.tool_id))
        )

    @property
    def available(self) -> bool:
        return self.network_mode is NetworkMode.OFFLINE and all(
            item.available for item in (*self.modules, *self.toolchains)
        )

    @property
    def receipt_id(self) -> str:
        return _digest_object(self.to_dict(include_receipt=False), label="capability-inventory")

    def module(self, capability_id: str) -> CapabilityReceipt:
        return next(item for item in self.modules if item.capability_id == capability_id)

    def toolchain(self, tool_id: str) -> SolverReadiness:
        return next(item for item in self.toolchains if item.tool_id == tool_id)

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DETERMINISTIC_REPAIR_CAPABILITIES_SCHEMA,
            "interface": self.INTERFACE,
            "network_mode": self.network_mode.value,
            "available": self.available,
            "modules": [item.to_dict() for item in self.modules],
            "toolchains": [item.to_dict() for item in self.toolchains],
            "selection_authority": "available_only",
            "probe_side_effects": "none",
        }
        if include_receipt:
            payload["receipt_id"] = self.receipt_id
        return payload


class DeterministicRepairCapabilityProbe:
    """Injectable, read-only inventory builder suitable for hermetic tests."""

    def __init__(
        self,
        *,
        module_requirements: Sequence[LogicModuleRequirement],
        toolchain_requirements: Sequence[ToolchainRequirement] = (),
        network_mode: NetworkMode | str = NetworkMode.OFFLINE,
        find_spec: Callable[[str], Any | None] = _safe_spec,
        distribution_version: Callable[[str], str] = importlib.metadata.version,
        executable_finder: Callable[[str], str | None] = shutil.which,
        read_bytes: Callable[[str], bytes] | None = None,
        initialization_evidence: Mapping[str, CapabilityEvidenceReceipt] | None = None,
        self_test_evidence: Mapping[str, CapabilityEvidenceReceipt] | None = None,
        reconstruction_evidence: Mapping[str, CapabilityEvidenceReceipt] | None = None,
        executable_versions: Mapping[str, str] | None = None,
    ) -> None:
        self.module_requirements = tuple(module_requirements)
        self.toolchain_requirements = tuple(toolchain_requirements)
        self.network_mode = NetworkMode(str(getattr(network_mode, "value", network_mode)))
        self.find_spec = find_spec
        self.distribution_version = distribution_version
        self.executable_finder = executable_finder
        self.read_bytes = read_bytes or (lambda path: Path(path).read_bytes())
        self.initialization_evidence = dict(initialization_evidence or {})
        self.self_test_evidence = dict(self_test_evidence or {})
        self.reconstruction_evidence = dict(reconstruction_evidence or {})
        self.executable_versions = dict(executable_versions or {})

    def probe(self) -> DeterministicRepairCapabilities:
        return DeterministicRepairCapabilities(
            modules=tuple(self.probe_module(item) for item in self.module_requirements),
            toolchains=tuple(self.probe_toolchain(item) for item in self.toolchain_requirements),
            network_mode=self.network_mode,
        )

    def probe_module(self, requirement: LogicModuleRequirement) -> CapabilityReceipt:
        reasons: list[str] = []
        spec = self.find_spec(requirement.module)
        origin = str(getattr(spec, "origin", "") or "") if spec else ""
        if not origin or origin in {"built-in", "frozen"}:
            reasons.append("module_missing_or_non_file_origin")
            return self._module_receipt(requirement, origin=origin, reasons=reasons)
        try:
            source = self.read_bytes(origin)
            symbols, values = _source_symbols(source)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            reasons.append("module_source_unreadable")
            return self._module_receipt(requirement, origin=origin, reasons=reasons)
        try:
            actual_version = str(self.distribution_version(requirement.distribution) or "")
        except Exception:  # metadata absence is unavailable, never an install request
            actual_version = ""
        if actual_version != requirement.expected_version:
            reasons.append("distribution_version_mismatch")
        if _SOURCE_RISK.search(source):
            reasons.append("stub_todo_or_simulated_source")
        content_digest = _digest_bytes(source, label="module")
        missing = tuple(sorted(set(requirement.required_symbols).difference(symbols)))
        if missing:
            reasons.append("required_symbols_missing")
        initialized = values.get(
            requirement.initialization_symbol
        ) is True and self._evidence_passes(
            self.initialization_evidence.get(requirement.module),
            evidence_id=requirement.module,
            evidence_kind="initialization",
            subject_id=requirement.module,
            subject_digest=content_digest,
            subject_version=actual_version,
        )
        reconstructed = values.get(
            requirement.reconstruction_symbol
        ) is True and self._evidence_passes(
            self.reconstruction_evidence.get(requirement.module),
            evidence_id=requirement.module,
            evidence_kind="reconstruction",
            subject_id=requirement.module,
            subject_digest=content_digest,
            subject_version=actual_version,
        )
        if not initialized:
            reasons.append("capability_uninitialized_or_unattested")
        if not reconstructed:
            reasons.append("reconstruction_unavailable_or_unattested")
        self_test = self._evidence_passes(
            self.self_test_evidence.get(requirement.module),
            evidence_id=requirement.module,
            evidence_kind="self_test",
            subject_id=requirement.module,
            subject_digest=content_digest,
            subject_version=actual_version,
        )
        if not self_test:
            reasons.append("self_test_missing_failed_or_unattested")
        if self.network_mode is not NetworkMode.OFFLINE:
            reasons.append("network_mode_not_offline")
        return self._module_receipt(
            requirement,
            origin=origin,
            actual_version=actual_version,
            source=source,
            symbols=symbols,
            missing=missing,
            initialized=initialized,
            reconstructed=reconstructed,
            self_test=self_test,
            reasons=reasons,
        )

    def _module_receipt(
        self,
        requirement: LogicModuleRequirement,
        *,
        origin: str,
        actual_version: str = "",
        source: bytes = b"",
        symbols: set[str] | None = None,
        missing: tuple[str, ...] = (),
        initialized: bool = False,
        reconstructed: bool = False,
        self_test: bool = False,
        reasons: Sequence[str],
    ) -> CapabilityReceipt:
        return CapabilityReceipt(
            capability_id=requirement.module,
            status=CapabilityStatus.AVAILABLE if not reasons else CapabilityStatus.UNAVAILABLE,
            origin=os.path.realpath(origin) if origin else "",
            distribution=requirement.distribution,
            expected_version=requirement.expected_version,
            distribution_version=actual_version,
            content_digest=_digest_bytes(source, label="module") if source else "",
            symbols=tuple(sorted(symbols or ())),
            missing_symbols=missing,
            initialized=initialized,
            reconstructed=reconstructed,
            self_test_passed=self_test,
            network_mode=self.network_mode,
            reason_codes=tuple(sorted(set(reasons))),
        )

    def probe_toolchain(self, requirement: ToolchainRequirement) -> SolverReadiness:
        reasons: list[str] = []
        path = self.executable_finder(requirement.executable) or ""
        blob = b""
        if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
            reasons.append("executable_missing")
        else:
            try:
                blob = self.read_bytes(path)
            except OSError:
                reasons.append("executable_unreadable")
        version = str(self.executable_versions.get(requirement.tool_id) or "")
        if version != requirement.expected_version:
            reasons.append("executable_version_mismatch")
        if blob and _SOURCE_RISK.search(blob):
            reasons.append("stub_todo_or_simulated_executable")
        executable_digest = _digest_bytes(blob, label="executable") if blob else ""
        self_test = self._evidence_passes(
            self.self_test_evidence.get(requirement.self_test_id),
            evidence_id=requirement.self_test_id,
            evidence_kind="self_test",
            subject_id=requirement.tool_id,
            subject_digest=executable_digest,
            subject_version=version,
        )
        reconstructed = self._evidence_passes(
            self.reconstruction_evidence.get(requirement.reconstruction_id),
            evidence_id=requirement.reconstruction_id,
            evidence_kind="reconstruction",
            subject_id=requirement.tool_id,
            subject_digest=executable_digest,
            subject_version=version,
        )
        if not self_test:
            reasons.append("self_test_missing_failed_or_unattested")
        if not reconstructed:
            reasons.append("reconstruction_unavailable_or_unattested")
        if self.network_mode is not NetworkMode.OFFLINE:
            reasons.append("network_mode_not_offline")
        return SolverReadiness(
            tool_id=requirement.tool_id,
            status=CapabilityStatus.AVAILABLE if not reasons else CapabilityStatus.UNAVAILABLE,
            executable=requirement.executable,
            path=os.path.realpath(path) if path else "",
            expected_version=requirement.expected_version,
            version=version,
            executable_digest=executable_digest,
            self_test_id=requirement.self_test_id,
            self_test_passed=self_test,
            reconstruction_id=requirement.reconstruction_id,
            reconstructed=reconstructed,
            network_mode=self.network_mode,
            reason_codes=tuple(sorted(set(reasons))),
        )

    @staticmethod
    def _evidence_passes(
        receipt: object,
        *,
        evidence_id: str,
        evidence_kind: str,
        subject_id: str,
        subject_digest: str,
        subject_version: str,
    ) -> bool:
        return isinstance(receipt, CapabilityEvidenceReceipt) and receipt.verifies(
            evidence_id=evidence_id,
            evidence_kind=evidence_kind,
            subject_id=subject_id,
            subject_digest=subject_digest,
            subject_version=subject_version,
        )


def probe_deterministic_repair_capabilities(
    module_requirements: Sequence[LogicModuleRequirement],
    *,
    toolchain_requirements: Sequence[ToolchainRequirement] = (),
    **kwargs: Any,
) -> DeterministicRepairCapabilities:
    """Functional entry point for a side-effect-free capability receipt."""

    return DeterministicRepairCapabilityProbe(
        module_requirements=module_requirements,
        toolchain_requirements=toolchain_requirements,
        **kwargs,
    ).probe()


__all__ = [
    "CAPABILITY_EVIDENCE_RECEIPT_SCHEMA",
    "CapabilityEvidenceReceipt",
    "CapabilityReceipt",
    "CapabilityStatus",
    "DETERMINISTIC_REPAIR_CAPABILITIES_INTERFACE",
    "DETERMINISTIC_REPAIR_CAPABILITIES_SCHEMA",
    "DeterministicRepairCapabilities",
    "DeterministicRepairCapabilityProbe",
    "LogicModuleRequirement",
    "NetworkMode",
    "SOLVER_READINESS_INTERFACE",
    "SOLVER_READINESS_SCHEMA",
    "SolverReadiness",
    "ToolchainRequirement",
    "probe_deterministic_repair_capabilities",
]
