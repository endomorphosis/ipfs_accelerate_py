"""DCR-090 read-only cross-root MCP conformance fixture validation.

The validator accepts observations captured by a reviewed in-process fixture;
it neither imports a named target, starts a server, nor opens a transport.  A
module must already be imported and its on-disk source is re-read to bind its
actual origin and digest.  Missing SwissKnife/MCP++ roots are therefore a
typed pending result, never a green skip.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity

DCR090_HERMETIC_CONFORMANCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-090-hermetic-conformance@1"
)
_REQUIRED_ROOTS: Final = frozenset({"accelerate", "swissknife", "mcpplusplus"})
_PROTOCOL_STEPS: Final = ("initialize", "tools/list", "tools/call")


class HermeticConformanceDisposition(StrEnum):
    PASS = "pass"
    INTEGRATION_PENDING = "integration_pending"
    FAILED = "failed"


class HermeticConformanceError(ValueError):
    """A fixture used a synthetic, non-canonical, or coupled observation."""


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_json(value: bytes, *, label: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise HermeticConformanceError(f"{label}_bytes_required")
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HermeticConformanceError(f"{label}_must_be_json") from exc
    encoded = json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if encoded != value:
        raise HermeticConformanceError(f"{label}_must_be_canonical_json")
    return value


def _payload(value: bytes) -> Mapping[str, Any]:
    decoded = json.loads(value.decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise HermeticConformanceError("protocol_payload_must_be_object")
    return decoded


@dataclass(frozen=True)
class ImportedModuleOrigin:
    """Recomputed source identity for an already imported in-root module."""

    root_id: str
    module_name: str
    source_path: str
    source_digest: str

    @classmethod
    def from_module(cls, *, root_id: str, root: Path, module: ModuleType) -> ImportedModuleOrigin:
        if root_id not in _REQUIRED_ROOTS:
            raise HermeticConformanceError("unsupported_root_id")
        if not isinstance(module, ModuleType) or not isinstance(module.__name__, str):
            raise HermeticConformanceError("actual_imported_module_required")
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path:
            raise HermeticConformanceError("module_origin_missing")
        resolved_root = root.resolve(strict=True)
        source = Path(raw_path).resolve(strict=True)
        try:
            source.relative_to(resolved_root)
        except ValueError as exc:
            raise HermeticConformanceError("module_origin_outside_declared_root") from exc
        if source.suffix not in {".py", ".ts", ".tsx", ".js", ".mjs"}:
            raise HermeticConformanceError("module_origin_is_not_source")
        return cls(
            root_id=root_id,
            module_name=module.__name__,
            source_path=source.relative_to(resolved_root).as_posix(),
            source_digest=_sha256(source.read_bytes()),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "root_id": self.root_id,
            "module_name": self.module_name,
            "source_path": self.source_path,
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class IndependentExpectedFact:
    """Expected fact loaded from separate fixture bytes, never from a request."""

    fact_id: str
    source_digest: str
    value_digest: str
    value_canonical: bytes

    @classmethod
    def from_bytes(cls, *, fact_id: str, source: bytes, value: bytes) -> IndependentExpectedFact:
        if not isinstance(fact_id, str) or not fact_id.strip():
            raise HermeticConformanceError("expected_fact_id_required")
        canonical_value = _canonical_json(value, label="expected_fact_value")
        return cls(
            fact_id=fact_id.strip(),
            source_digest=_sha256(_canonical_json(source, label="expected_fact_source")),
            value_digest=_sha256(canonical_value),
            value_canonical=canonical_value,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "fact_id": self.fact_id,
            "source_digest": self.source_digest,
            "value_digest": self.value_digest,
        }


@dataclass(frozen=True)
class McpProtocolObservation:
    """Raw canonical initialize/list/call exchanges from an injected fixture."""

    root_id: str
    schema_id: str
    profile_id: str
    requests: Mapping[str, bytes]
    results: Mapping[str, bytes]
    errors: Mapping[str, bytes]
    expected_fact_id: str
    observed_fact: bytes

    def __post_init__(self) -> None:
        if self.root_id not in _REQUIRED_ROOTS:
            raise HermeticConformanceError("unsupported_observation_root")
        if not self.schema_id or not self.profile_id or not self.expected_fact_id:
            raise HermeticConformanceError("schema_profile_expected_fact_required")
        for name, values in (
            ("request", self.requests),
            ("result", self.results),
            ("error", self.errors),
        ):
            if not isinstance(values, Mapping) or set(values) != set(_PROTOCOL_STEPS):
                raise HermeticConformanceError(f"{name}_step_set_not_exact")
            for step, raw in values.items():
                _canonical_json(raw, label=f"{name}_{step}")
        for step, expected_method in zip(_PROTOCOL_STEPS, _PROTOCOL_STEPS, strict=True):
            if _payload(self.requests[step]).get("method") != expected_method:
                raise HermeticConformanceError(f"request_method_mismatch_{step.replace('/', '_')}")
        _canonical_json(self.observed_fact, label="observed_fact")
        # A request cannot contain a detector identity or expected value digest;
        # otherwise a mock can merely echo the requested expected answer.
        request_blob = b"".join(self.requests.values())
        if self.expected_fact_id.encode("utf-8") in request_blob:
            raise HermeticConformanceError("request_echoes_expected_fact_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "root_id": self.root_id,
            "schema_id": self.schema_id,
            "profile_id": self.profile_id,
            "request_digests": {
                key: _sha256(value) for key, value in sorted(self.requests.items())
            },
            "result_digests": {key: _sha256(value) for key, value in sorted(self.results.items())},
            "error_digests": {key: _sha256(value) for key, value in sorted(self.errors.items())},
            "expected_fact_id": self.expected_fact_id,
            "observed_fact_digest": _sha256(self.observed_fact),
        }


@dataclass(frozen=True)
class HermeticConformanceReport:
    disposition: HermeticConformanceDisposition
    reason_codes: tuple[str, ...]
    origins: tuple[ImportedModuleOrigin, ...]
    observations: tuple[McpProtocolObservation, ...]

    @property
    def report_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DCR090_HERMETIC_CONFORMANCE_SCHEMA,
            "authoritative": False,
            "structural_fixture": True,
            "live_conformance": False,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "origins": [item.to_dict() for item in self.origins],
            "observations": [item.to_dict() for item in self.observations],
            "model_call_count": 0,
            "provider_call_count": 0,
            "network_call_count": 0,
            "execution_authorized": False,
            "completion_authorized": False,
        }


def validate_hermetic_conformance(
    *,
    origins: Sequence[ImportedModuleOrigin],
    observations: Sequence[McpProtocolObservation],
    expected_facts: Sequence[IndependentExpectedFact],
) -> HermeticConformanceReport:
    """Validate deterministic cross-root fixture observations without I/O routes."""

    checked_origins = tuple(origins)
    checked_observations = tuple(observations)
    reasons: list[str] = []
    if any(not isinstance(item, ImportedModuleOrigin) for item in checked_origins):
        reasons.append("typed_imported_module_origins_required")
    root_ids = [item.root_id for item in checked_origins if isinstance(item, ImportedModuleOrigin)]
    if len(root_ids) != len(set(root_ids)):
        reasons.append("duplicate_root_origin")
    missing = sorted(_REQUIRED_ROOTS - set(root_ids))
    if missing:
        reasons.extend(f"missing_real_root_{item}" for item in missing)
    facts = {
        item.fact_id: item for item in expected_facts if isinstance(item, IndependentExpectedFact)
    }
    if len(facts) != len(expected_facts):
        reasons.append("typed_independent_expected_facts_required")
    for observation in checked_observations:
        if not isinstance(observation, McpProtocolObservation):
            reasons.append("typed_protocol_observation_required")
            continue
        fact = facts.get(observation.expected_fact_id)
        if fact is None:
            reasons.append("missing_independent_expected_fact")
        else:
            request_blob = b"".join(observation.requests.values())
            if fact.value_canonical in request_blob:
                reasons.append("request_echoes_expected_detector_value")
            if _sha256(observation.observed_fact) != fact.value_digest:
                reasons.append("observed_fact_does_not_match_independent_expectation")
        if observation.root_id not in root_ids:
            reasons.append("observation_root_has_no_real_import_origin")
    if not checked_observations:
        reasons.append("protocol_observations_required")

    pending = any(reason.startswith("missing_real_root_") for reason in reasons)
    if not reasons:
        # Temporary imported modules and injected observations exercise only
        # structural boundaries.  They must never be projected as live
        # cross-root conformance, even when every structural assertion holds.
        reasons.append("structural_fixture_non_live")
    disposition = (
        HermeticConformanceDisposition.INTEGRATION_PENDING
        if pending or reasons == ["structural_fixture_non_live"]
        else HermeticConformanceDisposition.FAILED
        if reasons
        else HermeticConformanceDisposition.PASS
    )
    return HermeticConformanceReport(
        disposition=disposition,
        reason_codes=tuple(sorted(set(reasons))),
        origins=tuple(sorted(checked_origins, key=lambda item: item.root_id)),
        observations=tuple(sorted(checked_observations, key=lambda item: item.root_id)),
    )


__all__ = [
    "DCR090_HERMETIC_CONFORMANCE_SCHEMA",
    "HermeticConformanceDisposition",
    "HermeticConformanceError",
    "HermeticConformanceReport",
    "ImportedModuleOrigin",
    "IndependentExpectedFact",
    "McpProtocolObservation",
    "validate_hermetic_conformance",
]
