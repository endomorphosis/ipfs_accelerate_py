"""Normalize observed contracts into real datasets logic IR (DCR-030).

Interfaces
----------
* ``DatasetsLogicFacade@1`` — discover families, normalize evidence, inject
  rows into the existing datasets provider registry.
* ``IRInputEnvelope@1`` — one production IR input bound to original bytes and
  forest CID.

Effects (DCR-030):

* Inject observed AST, contract graph, knowledge-graph, UI IR, SecurityIR, and
  deterministic vector evidence into the datasets provider registry.
* Emit ``dcr/ir-normalization@1`` evidence with input roots, adapter versions,
  module origins, family availability, and normalization diagnostics.

Normative rules (fail-closed):

* Synthetic fixtures may exercise adapters but **cannot** establish production
  capability or proof authority.
* Bridge-only projections without retained original bytes **cannot** substitute
  for a required production input.
* Every normalized row binds the exact original bytes (digest + CID) and the
  multi-root forest CID for the observation epoch.
* Module import / package presence alone is never capability or proof authority.

Evidence term: ``dcr/ir-normalization@1``.
Generated artifact: ``data/agent_supervisor/deterministic_contract_repair/ir-input.json``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any, Final

from .formal_verification_contracts import (
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)

# ---------------------------------------------------------------------------
# Schemas / interfaces / constants
# ---------------------------------------------------------------------------

DATASETS_LOGIC_FACADE_INTERFACE: Final = "DatasetsLogicFacade@1"
IR_INPUT_ENVELOPE_INTERFACE: Final = "IRInputEnvelope@1"
NORMALIZED_IR_ROW_INTERFACE: Final = "NormalizedIRRow@1"
IR_NORMALIZATION_RESULT_INTERFACE: Final = "IRNormalizationResult@1"

IR_INPUT_ENVELOPE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-input-envelope@1"
)
NORMALIZED_IR_ROW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/normalized-ir-row@1"
)
IR_NORMALIZATION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-normalization-result@1"
)
IR_INTEGRATION_ARTIFACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-integration-artifact@1"
)

IR_NORMALIZATION_EVIDENCE_TERM: Final = "dcr/ir-normalization@1"
CONTRACT_VERSION: Final[int] = 1
IR_INTEGRATION_VERSION: Final = "1"
DCR_TASK_ID: Final = "DCR-030"
DCR_ARTIFACT_PATH: Final = (
    "data/agent_supervisor/deterministic_contract_repair/ir-input.json"
)

FACADE_PROVIDER_ID: Final = "datasets-logic-facade"
FACADE_ADAPTER_VERSION: Final = "datasets-logic-facade@1"
IR_ADAPTER_VERSION: Final = "1"

DEFAULT_MAX_BYTES: Final[int] = 1_048_576
_MAX_FIELD_BYTES: Final[int] = 4_096
_MAX_ROWS: Final[int] = 50_000
_MAX_DIAGNOSTICS: Final[int] = 4_096
_MAX_PAYLOAD_BYTES: Final[int] = 64 * 1024

# Relative paths for required DCR production artifacts under the workspace.
_FOREST_REL: Final = "data/agent_supervisor/deterministic_contract_repair/forest.json"
_GRAPH_REL: Final = (
    "data/agent_supervisor/deterministic_contract_repair/mcp_contract_graph.json"
)
_FINDINGS_REL: Final = (
    "data/agent_supervisor/deterministic_contract_repair/"
    "mcp_contract_mismatch_findings.json"
)
_TRANSCRIPT_REL: Final = (
    "data/agent_supervisor/deterministic_contract_repair/mcp-live-transcript.json"
)
_DESKTOP_REL: Final = (
    "data/agent_supervisor/deterministic_contract_repair/desktop-expectations.json"
)
_CAPABILITIES_REL: Final = (
    "data/agent_supervisor/deterministic_contract_repair/capabilities.json"
)
_CURRENT_STATE_REL: Final = (
    "data/agent_supervisor/deterministic_contract_repair/current-state.json"
)

# Datasets logic module origins probed for family availability (lazy, no I/O
# beyond importlib.util.find_spec / optional import for origin digests).
_DATASETS_FAMILY_MODULES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ir_core": "ipfs_datasets_py.logic.ir_core",
        "software_contracts": "ipfs_datasets_py.logic.software_contracts",
        "security_ir": "ipfs_datasets_py.logic.security_ir",
        "intent_ir": "ipfs_datasets_py.logic.intent_ir",
        "formalization": "ipfs_datasets_py.logic.formalization",
        "knowledge_graphs": "ipfs_datasets_py.logic.knowledge_graphs",
        "backends": "ipfs_datasets_py.logic.backends.registry",
        "hammers": "ipfs_datasets_py.logic.hammers",
    }
)

# Evidence families injected into the provider registry (effects of DCR-030).
INJECTED_EVIDENCE_FAMILIES: Final[tuple[str, ...]] = (
    "observed_ast",
    "contract_graph",
    "knowledge_graph",
    "ui_ir",
    "security_ir",
    "deterministic_vector",
)

# Required production inputs that fixture/bridge-only artifacts cannot replace.
REQUIRED_PRODUCTION_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "forest",
        "contract_graph",
        "mismatch_findings",
        "live_transcript",
        *INJECTED_EVIDENCE_FAMILIES,
    }
)

_DIGEST_RE_PREFIX: Final = "sha256:"

_MODULE_CACHE: dict[str, ModuleType | None] = {}
_MODULE_ERRORS: dict[str, BaseException] = {}
_IMPORT_LOCK: Final = threading.RLock()


# ---------------------------------------------------------------------------
# Errors / enums
# ---------------------------------------------------------------------------


class IRIntegrationError(ValueError):
    """IR integration input is malformed or violates a closed invariant."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "ir_integration_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class ProductionInputSubstitutionError(IRIntegrationError):
    """Fixture-derived or bridge-only input cannot replace production input."""

    def __init__(
        self,
        family: str,
        authority: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {"family": family, "authority": authority}
        if details:
            payload.update(dict(details))
        super().__init__(
            (
                f"required production input {family!r} cannot be satisfied by "
                f"{authority!r} authority"
            ),
            reason_code="production_input_substitution_forbidden",
            details=payload,
        )


class InputAuthority(str, Enum):
    """Closed authority vocabulary for IR input envelopes.

    Only ``production`` may satisfy required production families.  Fixtures
    exercise adapters; bridge-only rows are projections without retained
    original production bytes.
    """

    PRODUCTION = "production"
    FIXTURE = "fixture"
    BRIDGE_ONLY = "bridge_only"

    @property
    def may_satisfy_required_production(self) -> bool:
        return self is InputAuthority.PRODUCTION


class EvidenceFamily(str, Enum):
    """Closed set of IR evidence families handled by the facade."""

    FOREST = "forest"
    CONTRACT_GRAPH = "contract_graph"
    MISMATCH_FINDINGS = "mismatch_findings"
    LIVE_TRANSCRIPT = "live_transcript"
    OBSERVED_AST = "observed_ast"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    UI_IR = "ui_ir"
    SECURITY_IR = "security_ir"
    DETERMINISTIC_VECTOR = "deterministic_vector"
    CAPABILITY_MANIFEST = "capability_manifest"
    CURRENT_STATE = "current_state"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _norm_text(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    allow_empty: bool = False,
    maximum: int = _MAX_FIELD_BYTES,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise IRIntegrationError(
            f"{field_name} must be a string",
            reason_code="invalid_field_type",
            details={"field": field_name, "type": type(value).__name__},
        )
    if required and not text and not allow_empty:
        raise IRIntegrationError(
            f"{field_name} is required",
            reason_code="missing_required_field",
            details={"field": field_name},
        )
    if len(text.encode("utf-8")) > maximum:
        raise IRIntegrationError(
            f"{field_name} exceeds the {maximum}-byte limit",
            reason_code="field_too_large",
            details={"field": field_name},
        )
    if "\x00" in text:
        raise IRIntegrationError(
            f"{field_name} must not contain NUL",
            reason_code="invalid_field_value",
            details={"field": field_name},
        )
    return text


def _norm_enum(value: Any, kind: type[Enum], *, field_name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(getattr(value, "value", value)).strip())
    except (TypeError, ValueError) as exc:
        raise IRIntegrationError(
            f"{field_name} must be one of: "
            + ", ".join(item.value for item in kind),
            reason_code="invalid_enum",
            details={"field": field_name, "value": repr(value)},
        ) from exc


def _norm_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise IRIntegrationError(
            f"{field_name} must be a boolean",
            reason_code="invalid_field_type",
            details={"field": field_name},
        )
    return value


def _norm_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    converter = getattr(value, "to_dict", None)
    if not isinstance(value, Mapping) and callable(converter):
        value = converter()
    if not isinstance(value, Mapping):
        raise IRIntegrationError(
            f"{field_name} must be an object",
            reason_code="invalid_field_type",
            details={"field": field_name},
        )
    if any(not isinstance(key, str) for key in value):
        raise IRIntegrationError(
            f"{field_name} keys must be strings",
            reason_code="invalid_field_type",
            details={"field": field_name},
        )
    try:
        encoded = canonical_json_bytes(dict(value))
        decoded = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, ContractValidationError, json.JSONDecodeError) as exc:
        raise IRIntegrationError(
            f"{field_name} must contain canonical JSON values",
            reason_code="non_canonical_payload",
            details={"field": field_name},
        ) from exc
    if not isinstance(decoded, dict):
        raise IRIntegrationError(
            f"{field_name} must be an object",
            reason_code="invalid_field_type",
        )
    return decoded


def _bytes_digest(raw: bytes) -> str:
    return _DIGEST_RE_PREFIX + hashlib.sha256(raw).hexdigest()


def _bytes_cid(raw: bytes) -> str:
    """CIDv1 dag-json/sha2-256 over a digest-bound envelope of the raw bytes.

    Original production bytes may be arbitrary (not always JSON).  Identity is
    therefore taken over a canonical envelope that binds the exact digest of
    those bytes rather than re-interpreting the payload.
    """

    return content_identity(
        {
            "profile": "dcr-original-bytes-v1",
            "digest": _bytes_digest(raw),
            "byte_length": len(raw),
        }
    )


def canonical_ir_cid(value: Any) -> str:
    """Return the supervisor content identity for a canonical IR payload."""

    return content_identity(value)


def _default_workspace() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[5], here.parents[4], Path.cwd()):
        marker = candidate / "config" / "deterministic_contract_repair_services.json"
        if marker.is_file():
            return candidate
        # Fall back to DCR forest as a secondary workspace marker.
        forest = candidate / _FOREST_REL
        if forest.is_file():
            return candidate
    return Path.cwd()


def _resolve_relative(root: Path, relative: str) -> Path:
    return root.joinpath(*PurePosixPath(relative).parts)


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise IRIntegrationError(
            f"unable to read production input: {path}",
            reason_code="input_read_failed",
            details={"path": str(path), "error": type(exc).__name__},
        ) from exc


def _load_json_object(path: Path) -> dict[str, Any]:
    raw = _read_bytes(path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IRIntegrationError(
            f"production input is not valid JSON: {path}",
            reason_code="invalid_json",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise IRIntegrationError(
            f"production input must be a JSON object: {path}",
            reason_code="invalid_json_root",
            details={"path": str(path)},
        )
    return payload


def _compact_projection(
    payload: Mapping[str, Any],
    *,
    keys: Sequence[str],
    max_bytes: int = _MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    """Retain a compact subset of keys for the normalized row projection."""

    projection: dict[str, Any] = {}
    for key in keys:
        if key in payload:
            projection[key] = payload[key]
    try:
        encoded = canonical_json_bytes(projection)
    except (TypeError, ValueError, ContractValidationError) as exc:
        raise IRIntegrationError(
            "projection is not canonical-JSON serializable",
            reason_code="non_canonical_payload",
        ) from exc
    if len(encoded) > max_bytes:
        # Drop nested bulk; keep only scalar/top-level identifiers.
        slim: dict[str, Any] = {}
        for key, value in projection.items():
            if isinstance(value, (str, int, bool)) or value is None:
                slim[key] = value
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                slim[key] = f"count:{len(value)}"
            elif isinstance(value, Mapping):
                slim[key] = f"keys:{len(value)}"
        projection = slim
        encoded = canonical_json_bytes(projection)
        if len(encoded) > max_bytes:
            raise IRIntegrationError(
                "projection exceeds max_bytes even after compaction",
                reason_code="bounds_exceeded",
                details={"byte_length": len(encoded)},
            )
    return projection


# ---------------------------------------------------------------------------
# Module origin / family availability probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModuleOrigin:
    """Exact origin binding for one datasets logic module."""

    module: str
    available: bool
    module_file: str = ""
    source_digest: str = ""
    error_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "module": self.module,
            "available": self.available,
        }
        if self.module_file:
            payload["module_file"] = self.module_file
        if self.source_digest:
            payload["source_digest"] = self.source_digest
        if self.error_type:
            payload["error_type"] = self.error_type
        return payload


@dataclass(frozen=True, slots=True)
class FamilyAvailability:
    """Availability of one datasets logic family surface."""

    family: str
    module: str
    available: bool
    origin: ModuleOrigin

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "module": self.module,
            "available": self.available,
            "origin": self.origin.to_dict(),
        }


def probe_module_origin(module_name: str) -> ModuleOrigin:
    """Probe a module origin without establishing capability authority."""

    name = _norm_text(module_name, field_name="module", required=True)
    with _IMPORT_LOCK:
        if name in _MODULE_ERRORS and name not in _MODULE_CACHE:
            error = _MODULE_ERRORS[name]
            return ModuleOrigin(
                module=name,
                available=False,
                error_type=type(error).__name__,
            )
        if name in _MODULE_CACHE:
            module = _MODULE_CACHE[name]
            if module is None:
                error = _MODULE_ERRORS.get(name)
                return ModuleOrigin(
                    module=name,
                    available=False,
                    error_type=type(error).__name__ if error else "ImportError",
                )
            module_file = str(getattr(module, "__file__", "") or "")
            digest = ""
            if module_file and Path(module_file).is_file():
                try:
                    digest = _bytes_digest(Path(module_file).read_bytes())
                except OSError:
                    digest = ""
            return ModuleOrigin(
                module=name,
                available=True,
                module_file=module_file,
                source_digest=digest,
            )

        # Prefer find_spec first so missing packages stay cheap.
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            _MODULE_ERRORS[name] = exc
            _MODULE_CACHE[name] = None
            return ModuleOrigin(
                module=name,
                available=False,
                error_type=type(exc).__name__,
            )
        if spec is None:
            _MODULE_CACHE[name] = None
            _MODULE_ERRORS[name] = ModuleNotFoundError(name)
            return ModuleOrigin(
                module=name,
                available=False,
                error_type="ModuleNotFoundError",
            )
        module_file = str(getattr(spec, "origin", "") or "")
        digest = ""
        if module_file and module_file not in {"built-in", "frozen"}:
            path = Path(module_file)
            if path.is_file():
                try:
                    digest = _bytes_digest(path.read_bytes())
                except OSError:
                    digest = ""
        # Do not import the package body here — origin presence is enough for
        # family availability diagnostics and keeps cold import cheap.
        return ModuleOrigin(
            module=name,
            available=True,
            module_file=module_file if module_file not in {"built-in", "frozen"} else "",
            source_digest=digest,
        )


def probe_family_availability(
    families: Mapping[str, str] | None = None,
) -> tuple[FamilyAvailability, ...]:
    """Return sorted family availability diagnostics for datasets logic."""

    mapping = families or _DATASETS_FAMILY_MODULES
    results: list[FamilyAvailability] = []
    for family, module in sorted(mapping.items()):
        origin = probe_module_origin(module)
        results.append(
            FamilyAvailability(
                family=family,
                module=module,
                available=origin.available,
                origin=origin,
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# IRInputEnvelope@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IRInputEnvelope:
    """One IR input bound to original bytes and forest identity.

    Interface: ``IRInputEnvelope@1``
    """

    family: str
    input_root: str
    original_bytes: bytes
    forest_cid: str
    authority: InputAuthority = InputAuthority.PRODUCTION
    adapter_version: str = FACADE_ADAPTER_VERSION
    module_origin: str = ""
    source_path: str = ""
    projection: Mapping[str, Any] = field(default_factory=dict)
    schema: str = IR_INPUT_ENVELOPE_SCHEMA
    interface: str = IR_INPUT_ENVELOPE_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            _norm_text(self.family, field_name="family", required=True),
        )
        object.__setattr__(
            self,
            "input_root",
            _norm_text(self.input_root, field_name="input_root", required=True),
        )
        if not isinstance(self.original_bytes, (bytes, bytearray)):
            raise IRIntegrationError(
                "original_bytes must be bytes",
                reason_code="invalid_field_type",
                details={"field": "original_bytes"},
            )
        raw = bytes(self.original_bytes)
        if not raw:
            raise IRIntegrationError(
                "original_bytes must not be empty",
                reason_code="missing_original_bytes",
                details={"family": self.family},
            )
        object.__setattr__(self, "original_bytes", raw)
        object.__setattr__(
            self,
            "forest_cid",
            _norm_text(self.forest_cid, field_name="forest_cid", required=True),
        )
        object.__setattr__(
            self,
            "authority",
            _norm_enum(self.authority, InputAuthority, field_name="authority"),
        )
        object.__setattr__(
            self,
            "adapter_version",
            _norm_text(
                self.adapter_version, field_name="adapter_version", required=True
            ),
        )
        object.__setattr__(
            self,
            "module_origin",
            _norm_text(
                self.module_origin,
                field_name="module_origin",
                required=False,
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "source_path",
            _norm_text(
                self.source_path,
                field_name="source_path",
                required=False,
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "projection",
            MappingProxyType(_norm_mapping(self.projection, field_name="projection")),
        )
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        object.__setattr__(
            self,
            "interface",
            _norm_text(self.interface, field_name="interface", required=True),
        )
        if self.schema != IR_INPUT_ENVELOPE_SCHEMA:
            raise IRIntegrationError(
                "unsupported IR input envelope schema",
                reason_code="unsupported_schema",
            )
        if self.interface != IR_INPUT_ENVELOPE_INTERFACE:
            raise IRIntegrationError(
                "unsupported IR input envelope interface",
                reason_code="unsupported_interface",
            )

    @property
    def original_bytes_digest(self) -> str:
        return _bytes_digest(self.original_bytes)

    @property
    def original_bytes_cid(self) -> str:
        return _bytes_cid(self.original_bytes)

    @property
    def envelope_id(self) -> str:
        return canonical_ir_cid(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "family": self.family,
            "input_root": self.input_root,
            "original_bytes_digest": self.original_bytes_digest,
            "original_bytes_cid": self.original_bytes_cid,
            "forest_cid": self.forest_cid,
            "authority": self.authority.value,
            "adapter_version": self.adapter_version,
            "module_origin": self.module_origin,
            "source_path": self.source_path,
            "projection": dict(self.projection),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["envelope_id"] = self.envelope_id
        # original_bytes intentionally omitted from the durable projection —
        # only digest/CID bind them.  Callers that need the bytes retain the
        # envelope object in process.
        return payload

    def require_production(self) -> None:
        if self.family in REQUIRED_PRODUCTION_FAMILIES and (
            not self.authority.may_satisfy_required_production
        ):
            raise ProductionInputSubstitutionError(
                self.family, self.authority.value
            )


# ---------------------------------------------------------------------------
# NormalizedIRRow@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormalizedIRRow:
    """One normalized IR row bound to original bytes and forest CID.

    Interface: ``NormalizedIRRow@1``
    """

    family: str
    input_root: str
    original_bytes_digest: str
    original_bytes_cid: str
    forest_cid: str
    adapter_version: str
    module_origin: str
    family_available: bool
    authority: InputAuthority
    source_path: str = ""
    projection: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()
    schema: str = NORMALIZED_IR_ROW_SCHEMA
    interface: str = NORMALIZED_IR_ROW_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "family", _norm_text(self.family, field_name="family", required=True)
        )
        object.__setattr__(
            self,
            "input_root",
            _norm_text(self.input_root, field_name="input_root", required=True),
        )
        object.__setattr__(
            self,
            "original_bytes_digest",
            _norm_text(
                self.original_bytes_digest,
                field_name="original_bytes_digest",
                required=True,
            ),
        )
        if not self.original_bytes_digest.startswith(_DIGEST_RE_PREFIX):
            raise IRIntegrationError(
                "original_bytes_digest must be sha256-labeled",
                reason_code="invalid_digest",
            )
        object.__setattr__(
            self,
            "original_bytes_cid",
            _norm_text(
                self.original_bytes_cid,
                field_name="original_bytes_cid",
                required=True,
            ),
        )
        if self.original_bytes_cid.startswith(_DIGEST_RE_PREFIX):
            raise IRIntegrationError(
                "original_bytes_cid must not be a digest-shaped pseudo-CID",
                reason_code="pseudo_cid",
            )
        object.__setattr__(
            self,
            "forest_cid",
            _norm_text(self.forest_cid, field_name="forest_cid", required=True),
        )
        object.__setattr__(
            self,
            "adapter_version",
            _norm_text(
                self.adapter_version, field_name="adapter_version", required=True
            ),
        )
        object.__setattr__(
            self,
            "module_origin",
            _norm_text(
                self.module_origin,
                field_name="module_origin",
                required=False,
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "family_available",
            _norm_bool(self.family_available, field_name="family_available"),
        )
        object.__setattr__(
            self,
            "authority",
            _norm_enum(self.authority, InputAuthority, field_name="authority"),
        )
        object.__setattr__(
            self,
            "source_path",
            _norm_text(
                self.source_path,
                field_name="source_path",
                required=False,
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "projection",
            MappingProxyType(_norm_mapping(self.projection, field_name="projection")),
        )
        if isinstance(self.diagnostics, str) or not isinstance(
            self.diagnostics, Sequence
        ):
            raise IRIntegrationError(
                "diagnostics must be a sequence of strings",
                reason_code="invalid_field_type",
            )
        diags = tuple(
            _norm_text(item, field_name="diagnostic", required=True)
            for item in self.diagnostics
        )
        object.__setattr__(self, "diagnostics", diags)
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        object.__setattr__(
            self,
            "interface",
            _norm_text(self.interface, field_name="interface", required=True),
        )
        if self.schema != NORMALIZED_IR_ROW_SCHEMA:
            raise IRIntegrationError(
                "unsupported normalized IR row schema",
                reason_code="unsupported_schema",
            )
        if self.interface != NORMALIZED_IR_ROW_INTERFACE:
            raise IRIntegrationError(
                "unsupported normalized IR row interface",
                reason_code="unsupported_interface",
            )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "family": self.family,
            "input_root": self.input_root,
            "original_bytes_digest": self.original_bytes_digest,
            "original_bytes_cid": self.original_bytes_cid,
            "forest_cid": self.forest_cid,
            "adapter_version": self.adapter_version,
            "module_origin": self.module_origin,
            "family_available": self.family_available,
            "authority": self.authority.value,
            "source_path": self.source_path,
            "projection": dict(self.projection),
            "diagnostics": list(self.diagnostics),
        }

    @property
    def row_id(self) -> str:
        return canonical_ir_cid(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["row_id"] = self.row_id
        return payload

    @classmethod
    def from_envelope(
        cls,
        envelope: IRInputEnvelope,
        *,
        family_available: bool,
        diagnostics: Sequence[str] = (),
    ) -> "NormalizedIRRow":
        if not isinstance(envelope, IRInputEnvelope):
            raise IRIntegrationError(
                "envelope must be an IRInputEnvelope",
                reason_code="invalid_field_type",
            )
        return cls(
            family=envelope.family,
            input_root=envelope.input_root,
            original_bytes_digest=envelope.original_bytes_digest,
            original_bytes_cid=envelope.original_bytes_cid,
            forest_cid=envelope.forest_cid,
            adapter_version=envelope.adapter_version,
            module_origin=envelope.module_origin,
            family_available=family_available,
            authority=envelope.authority,
            source_path=envelope.source_path,
            projection=dict(envelope.projection),
            diagnostics=tuple(diagnostics),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NormalizedIRRow":
        if not isinstance(payload, Mapping):
            raise IRIntegrationError(
                "normalized row must be an object",
                reason_code="invalid_field_type",
            )
        return cls(
            family=str(payload.get("family") or ""),
            input_root=str(payload.get("input_root") or ""),
            original_bytes_digest=str(payload.get("original_bytes_digest") or ""),
            original_bytes_cid=str(payload.get("original_bytes_cid") or ""),
            forest_cid=str(payload.get("forest_cid") or ""),
            adapter_version=str(payload.get("adapter_version") or ""),
            module_origin=str(payload.get("module_origin") or ""),
            family_available=bool(payload.get("family_available", False)),
            authority=str(payload.get("authority") or InputAuthority.PRODUCTION.value),
            source_path=str(payload.get("source_path") or ""),
            projection=payload.get("projection") or {},
            diagnostics=tuple(payload.get("diagnostics") or ()),
            schema=str(payload.get("schema") or NORMALIZED_IR_ROW_SCHEMA),
            interface=str(payload.get("interface") or NORMALIZED_IR_ROW_INTERFACE),
        )


# ---------------------------------------------------------------------------
# Provider registry injection surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderRegistryEntry:
    """One injected evidence binding in the datasets provider registry."""

    family: str
    row_id: str
    original_bytes_cid: str
    forest_cid: str
    module_origin: str
    adapter_version: str
    producer_id: str = FACADE_PROVIDER_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "row_id": self.row_id,
            "original_bytes_cid": self.original_bytes_cid,
            "forest_cid": self.forest_cid,
            "module_origin": self.module_origin,
            "adapter_version": self.adapter_version,
            "producer_id": self.producer_id,
        }


class DatasetsProviderRegistry:
    """In-process registry for normalized IR evidence rows.

    This is the supervisor-owned injection surface for DCR-030.  It does not
    replace ``ipfs_datasets_py.logic.backends.registry`` and never grants
    execution or proof authority.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ProviderRegistryEntry] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._order.clear()

    def inject(self, row: NormalizedIRRow) -> ProviderRegistryEntry:
        if not isinstance(row, NormalizedIRRow):
            raise IRIntegrationError(
                "row must be a NormalizedIRRow",
                reason_code="invalid_field_type",
            )
        if not row.original_bytes_digest or not row.original_bytes_cid:
            raise IRIntegrationError(
                "injected row must bind original bytes",
                reason_code="missing_original_bytes",
            )
        if not row.forest_cid:
            raise IRIntegrationError(
                "injected row must bind forest_cid",
                reason_code="missing_forest_cid",
            )
        entry = ProviderRegistryEntry(
            family=row.family,
            row_id=row.row_id,
            original_bytes_cid=row.original_bytes_cid,
            forest_cid=row.forest_cid,
            module_origin=row.module_origin,
            adapter_version=row.adapter_version,
        )
        with self._lock:
            if row.row_id not in self._entries:
                self._order.append(row.row_id)
            self._entries[row.row_id] = entry
        return entry

    def inject_many(
        self, rows: Sequence[NormalizedIRRow]
    ) -> tuple[ProviderRegistryEntry, ...]:
        return tuple(self.inject(row) for row in rows)

    def get(self, row_id: str) -> ProviderRegistryEntry | None:
        return self._entries.get(row_id)

    def entries(self) -> tuple[ProviderRegistryEntry, ...]:
        with self._lock:
            return tuple(self._entries[key] for key in self._order)

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({entry.family for entry in self.entries()}))

    def to_dict(self) -> dict[str, Any]:
        items = [entry.to_dict() for entry in self.entries()]
        return {
            "entry_count": len(items),
            "families": list(self.families()),
            "entries": items,
            "grants_execution_authority": False,
            "grants_proof_authority": False,
        }


# ---------------------------------------------------------------------------
# IRNormalizationResult@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IRNormalizationResult:
    """Complete normalization result for one forest epoch.

    Interface: ``IRNormalizationResult@1``
    """

    forest_cid: str
    rows: tuple[NormalizedIRRow, ...]
    input_roots: tuple[str, ...]
    adapter_versions: tuple[str, ...]
    module_origins: tuple[str, ...]
    family_availability: tuple[FamilyAvailability, ...]
    diagnostics: tuple[str, ...]
    registry_entries: tuple[ProviderRegistryEntry, ...] = ()
    model_calls: int = 0
    authoritative: bool = False
    completion_authorized: bool = False
    schema: str = IR_NORMALIZATION_RESULT_SCHEMA
    interface: str = IR_NORMALIZATION_RESULT_INTERFACE
    evidence_term: str = IR_NORMALIZATION_EVIDENCE_TERM

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "forest_cid",
            _norm_text(self.forest_cid, field_name="forest_cid", required=True),
        )
        if isinstance(self.rows, str) or not isinstance(self.rows, Sequence):
            raise IRIntegrationError(
                "rows must be a sequence",
                reason_code="invalid_field_type",
            )
        if len(self.rows) > _MAX_ROWS:
            raise IRIntegrationError(
                "row count exceeds bound",
                reason_code="bounds_exceeded",
            )
        normalized_rows: list[NormalizedIRRow] = []
        for item in self.rows:
            if isinstance(item, NormalizedIRRow):
                normalized_rows.append(item)
            elif isinstance(item, Mapping):
                normalized_rows.append(NormalizedIRRow.from_dict(item))
            else:
                raise IRIntegrationError(
                    "rows must contain NormalizedIRRow values",
                    reason_code="invalid_field_type",
                )
        # Stable order by family then row_id.
        normalized_rows.sort(key=lambda row: (row.family, row.row_id))
        object.__setattr__(self, "rows", tuple(normalized_rows))

        for row in self.rows:
            if not row.original_bytes_digest or not row.original_bytes_cid:
                raise IRIntegrationError(
                    "every normalized row must bind original bytes",
                    reason_code="missing_original_bytes",
                    details={"family": row.family},
                )
            if not row.forest_cid:
                raise IRIntegrationError(
                    "every normalized row must bind forest_cid",
                    reason_code="missing_forest_cid",
                    details={"family": row.family},
                )
            if row.forest_cid != self.forest_cid:
                raise IRIntegrationError(
                    "row forest_cid does not match result forest_cid",
                    reason_code="forest_cid_mismatch",
                    details={
                        "family": row.family,
                        "row_forest_cid": row.forest_cid,
                        "result_forest_cid": self.forest_cid,
                    },
                )

        object.__setattr__(
            self,
            "input_roots",
            tuple(
                sorted(
                    {
                        _norm_text(item, field_name="input_root", required=True)
                        for item in self.input_roots
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "adapter_versions",
            tuple(
                sorted(
                    {
                        _norm_text(item, field_name="adapter_version", required=True)
                        for item in self.adapter_versions
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "module_origins",
            tuple(
                sorted(
                    {
                        _norm_text(
                            item,
                            field_name="module_origin",
                            required=False,
                            allow_empty=True,
                        )
                        for item in self.module_origins
                        if str(item or "").strip()
                    }
                )
            ),
        )
        if isinstance(self.family_availability, str) or not isinstance(
            self.family_availability, Sequence
        ):
            raise IRIntegrationError(
                "family_availability must be a sequence",
                reason_code="invalid_field_type",
            )
        avail: list[FamilyAvailability] = []
        for item in self.family_availability:
            if isinstance(item, FamilyAvailability):
                avail.append(item)
            elif isinstance(item, Mapping):
                origin_payload = item.get("origin") or {}
                origin = ModuleOrigin(
                    module=str(origin_payload.get("module") or item.get("module") or ""),
                    available=bool(origin_payload.get("available", item.get("available", False))),
                    module_file=str(origin_payload.get("module_file") or ""),
                    source_digest=str(origin_payload.get("source_digest") or ""),
                    error_type=str(origin_payload.get("error_type") or ""),
                )
                avail.append(
                    FamilyAvailability(
                        family=str(item.get("family") or ""),
                        module=str(item.get("module") or ""),
                        available=bool(item.get("available", False)),
                        origin=origin,
                    )
                )
            else:
                raise IRIntegrationError(
                    "family_availability entries are malformed",
                    reason_code="invalid_field_type",
                )
        object.__setattr__(
            self,
            "family_availability",
            tuple(sorted(avail, key=lambda item: item.family)),
        )
        if isinstance(self.diagnostics, str) or not isinstance(
            self.diagnostics, Sequence
        ):
            raise IRIntegrationError(
                "diagnostics must be a sequence",
                reason_code="invalid_field_type",
            )
        if len(self.diagnostics) > _MAX_DIAGNOSTICS:
            raise IRIntegrationError(
                "diagnostics exceed bound",
                reason_code="bounds_exceeded",
            )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(
                _norm_text(item, field_name="diagnostic", required=True)
                for item in self.diagnostics
            ),
        )
        entries: list[ProviderRegistryEntry] = []
        for item in self.registry_entries:
            if isinstance(item, ProviderRegistryEntry):
                entries.append(item)
            elif isinstance(item, Mapping):
                entries.append(
                    ProviderRegistryEntry(
                        family=str(item.get("family") or ""),
                        row_id=str(item.get("row_id") or ""),
                        original_bytes_cid=str(item.get("original_bytes_cid") or ""),
                        forest_cid=str(item.get("forest_cid") or ""),
                        module_origin=str(item.get("module_origin") or ""),
                        adapter_version=str(item.get("adapter_version") or ""),
                        producer_id=str(item.get("producer_id") or FACADE_PROVIDER_ID),
                    )
                )
            else:
                raise IRIntegrationError(
                    "registry_entries are malformed",
                    reason_code="invalid_field_type",
                )
        object.__setattr__(self, "registry_entries", tuple(entries))
        if (
            isinstance(self.model_calls, bool)
            or not isinstance(self.model_calls, int)
            or self.model_calls < 0
        ):
            raise IRIntegrationError(
                "model_calls must be a non-negative integer",
                reason_code="invalid_field_type",
            )
        object.__setattr__(
            self,
            "authoritative",
            _norm_bool(self.authoritative, field_name="authoritative"),
        )
        object.__setattr__(
            self,
            "completion_authorized",
            _norm_bool(
                self.completion_authorized, field_name="completion_authorized"
            ),
        )
        # Normalization never grants completion or proof authority.
        object.__setattr__(self, "authoritative", False)
        object.__setattr__(self, "completion_authorized", False)
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        object.__setattr__(
            self,
            "interface",
            _norm_text(self.interface, field_name="interface", required=True),
        )
        object.__setattr__(
            self,
            "evidence_term",
            _norm_text(self.evidence_term, field_name="evidence_term", required=True),
        )
        if self.schema != IR_NORMALIZATION_RESULT_SCHEMA:
            raise IRIntegrationError(
                "unsupported IR normalization result schema",
                reason_code="unsupported_schema",
            )
        if self.interface != IR_NORMALIZATION_RESULT_INTERFACE:
            raise IRIntegrationError(
                "unsupported IR normalization result interface",
                reason_code="unsupported_interface",
            )
        if self.evidence_term != IR_NORMALIZATION_EVIDENCE_TERM:
            raise IRIntegrationError(
                "unsupported IR normalization evidence term",
                reason_code="unsupported_evidence_term",
            )

    def _root_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "evidence_term": self.evidence_term,
            "version": IR_INTEGRATION_VERSION,
            "task_id": DCR_TASK_ID,
            "forest_cid": self.forest_cid,
            "rows": [row.to_dict() for row in self.rows],
            "input_roots": list(self.input_roots),
            "adapter_versions": list(self.adapter_versions),
            "module_origins": list(self.module_origins),
            "family_availability": [
                item.to_dict() for item in self.family_availability
            ],
            "diagnostics": list(self.diagnostics),
            "registry_entries": [item.to_dict() for item in self.registry_entries],
            "row_count": len(self.rows),
            "model_calls": self.model_calls,
            "authoritative": False,
            "completion_authorized": False,
        }

    @property
    def result_cid(self) -> str:
        return canonical_ir_cid(self._root_payload())

    @property
    def canonical_digest(self) -> str:
        return _DIGEST_RE_PREFIX + hashlib.sha256(
            canonical_json_bytes(self._root_payload())
        ).hexdigest()

    def verifies_cid(self) -> bool:
        return self.result_cid == canonical_ir_cid(self._root_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._root_payload()
        payload["result_cid"] = self.result_cid
        payload["canonical_digest"] = self.canonical_digest
        return payload

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "schema": IR_INTEGRATION_ARTIFACT_SCHEMA,
            "interface": DATASETS_LOGIC_FACADE_INTERFACE,
            "evidence_term": self.evidence_term,
            "task_id": DCR_TASK_ID,
            "normalization": self.to_dict(),
            "authoritative": False,
            "completion_authorized": False,
        }

    def to_artifact_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_artifact_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IRNormalizationResult":
        if not isinstance(payload, Mapping):
            raise IRIntegrationError(
                "normalization result must be an object",
                reason_code="invalid_field_type",
            )
        # Accept either the bare result or the artifact envelope.
        body: Mapping[str, Any] = payload
        if "normalization" in payload and isinstance(payload["normalization"], Mapping):
            body = payload["normalization"]
        return cls(
            forest_cid=str(body.get("forest_cid") or ""),
            rows=tuple(body.get("rows") or ()),
            input_roots=tuple(body.get("input_roots") or ()),
            adapter_versions=tuple(body.get("adapter_versions") or ()),
            module_origins=tuple(body.get("module_origins") or ()),
            family_availability=tuple(body.get("family_availability") or ()),
            diagnostics=tuple(body.get("diagnostics") or ()),
            registry_entries=tuple(body.get("registry_entries") or ()),
            model_calls=int(body.get("model_calls") or 0),
            authoritative=bool(body.get("authoritative", False)),
            completion_authorized=bool(body.get("completion_authorized", False)),
            schema=str(body.get("schema") or IR_NORMALIZATION_RESULT_SCHEMA),
            interface=str(body.get("interface") or IR_NORMALIZATION_RESULT_INTERFACE),
            evidence_term=str(
                body.get("evidence_term") or IR_NORMALIZATION_EVIDENCE_TERM
            ),
        )


# ---------------------------------------------------------------------------
# Production envelope builders
# ---------------------------------------------------------------------------


def _forest_cid_from_payload(payload: Mapping[str, Any], raw: bytes) -> str:
    for key in ("forest_id", "forest_cid", "portable_forest_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    local = payload.get("local")
    if isinstance(local, Mapping):
        for key in ("forest_id", "portable_forest_id"):
            value = local.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    # Fall back to content identity of the forest bytes themselves.
    return _bytes_cid(raw)


def build_envelope_from_bytes(
    *,
    family: str,
    input_root: str,
    original_bytes: bytes,
    forest_cid: str,
    authority: InputAuthority | str = InputAuthority.PRODUCTION,
    adapter_version: str = FACADE_ADAPTER_VERSION,
    module_origin: str = "",
    source_path: str = "",
    projection: Mapping[str, Any] | None = None,
) -> IRInputEnvelope:
    """Construct a validated IR input envelope from retained original bytes."""

    return IRInputEnvelope(
        family=family,
        input_root=input_root,
        original_bytes=original_bytes,
        forest_cid=forest_cid,
        authority=authority,
        adapter_version=adapter_version,
        module_origin=module_origin,
        source_path=source_path,
        projection=dict(projection or {}),
    )


def build_envelope_from_path(
    *,
    family: str,
    path: Path,
    forest_cid: str,
    input_root: str | None = None,
    authority: InputAuthority | str = InputAuthority.PRODUCTION,
    adapter_version: str = FACADE_ADAPTER_VERSION,
    module_origin: str = "",
    projection_keys: Sequence[str] = (),
) -> IRInputEnvelope:
    """Load original bytes from a production path and bind forest identity."""

    raw = _read_bytes(path)
    projection: dict[str, Any] = {}
    if projection_keys:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, Mapping):
            projection = _compact_projection(payload, keys=projection_keys)
    relative = input_root or path.name
    return build_envelope_from_bytes(
        family=family,
        input_root=relative,
        original_bytes=raw,
        forest_cid=forest_cid,
        authority=authority,
        adapter_version=adapter_version,
        module_origin=module_origin,
        source_path=str(path),
        projection=projection,
    )


def _derive_projection_envelope(
    *,
    family: str,
    source_family: str,
    source_bytes: bytes,
    forest_cid: str,
    input_root: str,
    module_origin: str,
    projection: Mapping[str, Any],
    authority: InputAuthority = InputAuthority.PRODUCTION,
) -> IRInputEnvelope:
    """Derive a family projection that still binds the source original bytes.

    Derived views (AST / KG / UI / Security / vector) retain the production
    authority of their source when the source is production, and always bind
    the exact original source bytes rather than synthetic fixture bytes.
    """

    # Domain-separate the derived projection while keeping original_bytes as
    # the production source preimage.
    bound_projection = {
        "derived_family": family,
        "source_family": source_family,
        "source_digest": _bytes_digest(source_bytes),
        "projection": dict(projection),
    }
    return IRInputEnvelope(
        family=family,
        input_root=input_root,
        original_bytes=source_bytes,
        forest_cid=forest_cid,
        authority=authority,
        adapter_version=FACADE_ADAPTER_VERSION,
        module_origin=module_origin,
        source_path=f"derived:{source_family}->{family}",
        projection=bound_projection,
    )


def collect_production_envelopes(
    *,
    repo_root: Path | None = None,
    forest_cid: str | None = None,
) -> tuple[IRInputEnvelope, ...]:
    """Collect production IR input envelopes from committed DCR artifacts."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    forest_path = _resolve_relative(root, _FOREST_REL)
    if not forest_path.is_file():
        raise IRIntegrationError(
            f"required production forest missing: {forest_path}",
            reason_code="missing_production_input",
            details={"family": "forest", "path": str(forest_path)},
        )
    forest_raw = _read_bytes(forest_path)
    forest_payload = _load_json_object(forest_path)
    resolved_forest_cid = forest_cid or _forest_cid_from_payload(
        forest_payload, forest_raw
    )

    envelopes: list[IRInputEnvelope] = []

    def _add_file(
        family: str,
        relative: str,
        *,
        module_origin: str,
        projection_keys: Sequence[str],
        required: bool = True,
    ) -> bytes | None:
        path = _resolve_relative(root, relative)
        if not path.is_file():
            if required:
                raise IRIntegrationError(
                    f"required production input missing: {relative}",
                    reason_code="missing_production_input",
                    details={"family": family, "path": relative},
                )
            return None
        envelope = build_envelope_from_path(
            family=family,
            path=path,
            forest_cid=resolved_forest_cid,
            input_root=relative,
            authority=InputAuthority.PRODUCTION,
            module_origin=module_origin,
            projection_keys=projection_keys,
        )
        envelopes.append(envelope)
        return envelope.original_bytes

    envelopes.append(
        build_envelope_from_bytes(
            family=EvidenceFamily.FOREST.value,
            input_root=_FOREST_REL,
            original_bytes=forest_raw,
            forest_cid=resolved_forest_cid,
            authority=InputAuthority.PRODUCTION,
            module_origin="ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_forest",
            source_path=str(forest_path),
            projection=_compact_projection(
                forest_payload,
                keys=(
                    "forest_id",
                    "schema",
                    "authoritative",
                    "completion_authorized",
                ),
            ),
        )
    )

    graph_bytes = _add_file(
        EvidenceFamily.CONTRACT_GRAPH.value,
        _GRAPH_REL,
        module_origin="ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_graph",
        projection_keys=(
            "schema",
            "interface",
            "graph_cid",
            "snapshot_id",
            "node_count",
            "edge_count",
            "blocker_count",
            "complete",
        ),
    )
    findings_bytes = _add_file(
        EvidenceFamily.MISMATCH_FINDINGS.value,
        _FINDINGS_REL,
        module_origin="ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_mismatch",
        projection_keys=(
            "schema",
            "interface",
            "evidence_term",
            "findings_cid",
            "finding_count",
            "complete",
            "snapshot_id",
            "graph_cid",
        ),
    )
    transcript_bytes = _add_file(
        EvidenceFamily.LIVE_TRANSCRIPT.value,
        _TRANSCRIPT_REL,
        module_origin="ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_observer",
        projection_keys=(
            "schema",
            "interface",
            "evidence_term",
            "service_id",
            "roles_observed",
            "passed",
            "model_calls",
        ),
    )
    _add_file(
        EvidenceFamily.CAPABILITY_MANIFEST.value,
        _CAPABILITIES_REL,
        module_origin="ipfs_accelerate_py.agent_supervisor.autonomous_repair.capabilities",
        projection_keys=(
            "capabilities_schema",
            "capabilities_interface",
            "non_executable",
        ),
        required=False,
    )
    _add_file(
        EvidenceFamily.CURRENT_STATE.value,
        _CURRENT_STATE_REL,
        module_origin="ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_current_state",
        projection_keys=(
            "schema",
            "report_id",
            "repository_commit",
            "authoritative",
            "completion_authorized",
        ),
        required=False,
    )

    # Derived production projections bind the contract-graph original bytes.
    if graph_bytes is not None:
        graph_payload = json.loads(graph_bytes.decode("utf-8"))
        if not isinstance(graph_payload, dict):
            graph_payload = {}
        # Observed AST projection: source span / node identity view.
        nodes = graph_payload.get("nodes") or []
        node_count = len(nodes) if isinstance(nodes, list) else 0
        envelopes.append(
            _derive_projection_envelope(
                family=EvidenceFamily.OBSERVED_AST.value,
                source_family=EvidenceFamily.CONTRACT_GRAPH.value,
                source_bytes=graph_bytes,
                forest_cid=resolved_forest_cid,
                input_root=_GRAPH_REL,
                module_origin="ipfs_datasets_py.logic.software_contracts",
                projection={
                    "view": "observed_ast",
                    "node_count": node_count,
                    "graph_cid": graph_payload.get("graph_cid", ""),
                    "snapshot_id": graph_payload.get("snapshot_id", ""),
                },
            )
        )
        # Knowledge-graph projection of the contract graph.
        edges = graph_payload.get("edges") or []
        edge_count = len(edges) if isinstance(edges, list) else 0
        envelopes.append(
            _derive_projection_envelope(
                family=EvidenceFamily.KNOWLEDGE_GRAPH.value,
                source_family=EvidenceFamily.CONTRACT_GRAPH.value,
                source_bytes=graph_bytes,
                forest_cid=resolved_forest_cid,
                input_root=_GRAPH_REL,
                module_origin="ipfs_datasets_py.logic.knowledge_graphs",
                projection={
                    "view": "knowledge_graph",
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "graph_cid": graph_payload.get("graph_cid", ""),
                },
            )
        )
        # UI IR projection from consumer/ui_action stages on the contract graph.
        # The desktop-expectations artifact can be multi-megabyte; the graph's
        # ui_action/descriptor/orb_idl stages are the compact production source
        # that still binds original graph bytes and forest CID.
        ui_nodes = 0
        if isinstance(nodes, list):
            for node in nodes:
                if isinstance(node, Mapping) and str(
                    node.get("kind") or node.get("stage") or ""
                ) in {"ui_action", "descriptor", "orb_idl"}:
                    ui_nodes += 1
        envelopes.append(
            _derive_projection_envelope(
                family=EvidenceFamily.UI_IR.value,
                source_family=EvidenceFamily.CONTRACT_GRAPH.value,
                source_bytes=graph_bytes,
                forest_cid=resolved_forest_cid,
                input_root=_GRAPH_REL,
                module_origin=(
                    "ipfs_accelerate_py.agent_supervisor.analysis."
                    "deterministic_desktop_expectations"
                ),
                projection={
                    "view": "ui_ir",
                    "ui_stage_nodes": ui_nodes,
                    "source": _GRAPH_REL,
                    "desktop_expectations_path": _DESKTOP_REL,
                },
                authority=InputAuthority.PRODUCTION,
            )
        )
        # SecurityIR projection from graph authority/policy edges + capabilities.
        cap_path = _resolve_relative(root, _CAPABILITIES_REL)
        security_source = graph_bytes
        security_root = _GRAPH_REL
        security_family = EvidenceFamily.CONTRACT_GRAPH.value
        if cap_path.is_file():
            security_source = _read_bytes(cap_path)
            security_root = _CAPABILITIES_REL
            security_family = EvidenceFamily.CAPABILITY_MANIFEST.value
        envelopes.append(
            _derive_projection_envelope(
                family=EvidenceFamily.SECURITY_IR.value,
                source_family=security_family,
                source_bytes=security_source,
                forest_cid=resolved_forest_cid,
                input_root=security_root,
                module_origin="ipfs_datasets_py.logic.security_ir",
                projection={
                    "view": "security_ir",
                    "source": security_root,
                    "graph_cid": graph_payload.get("graph_cid", ""),
                },
            )
        )
        # Deterministic vector evidence: ordered digest vector of production
        # inputs (never model embeddings).
        vector_components = [
            ("forest", forest_raw),
            ("contract_graph", graph_bytes),
        ]
        if findings_bytes is not None:
            vector_components.append(("mismatch_findings", findings_bytes))
        if transcript_bytes is not None:
            vector_components.append(("live_transcript", transcript_bytes))
        vector_payload = {
            "view": "deterministic_vector",
            "components": [
                {
                    "family": name,
                    "digest": _bytes_digest(raw),
                    "byte_length": len(raw),
                }
                for name, raw in vector_components
            ],
        }
        vector_bytes = canonical_json_bytes(vector_payload)
        envelopes.append(
            build_envelope_from_bytes(
                family=EvidenceFamily.DETERMINISTIC_VECTOR.value,
                input_root="deterministic_vector",
                original_bytes=vector_bytes,
                forest_cid=resolved_forest_cid,
                authority=InputAuthority.PRODUCTION,
                module_origin="ipfs_accelerate_py.agent_supervisor.proof.ir_integration",
                source_path="derived:production_digests",
                projection=vector_payload,
            )
        )

    return tuple(envelopes)


# ---------------------------------------------------------------------------
# normalize_contract_evidence
# ---------------------------------------------------------------------------


def normalize_contract_evidence(
    envelopes: Sequence[IRInputEnvelope] | None = None,
    *,
    repo_root: Path | None = None,
    forest_cid: str | None = None,
    registry: DatasetsProviderRegistry | None = None,
    require_production: bool = True,
    inject: bool = True,
    family_modules: Mapping[str, str] | None = None,
) -> IRNormalizationResult:
    """Normalize observed contract evidence into datasets logic IR rows.

    Parameters
    ----------
    envelopes:
        Explicit IR input envelopes.  When omitted, production artifacts are
        collected from ``repo_root``.
    require_production:
        When true (default), fixture-derived and bridge-only envelopes cannot
        satisfy any family in :data:`REQUIRED_PRODUCTION_FAMILIES`.
    inject:
        When true (default), inject normalized rows into ``registry``.
    """

    if envelopes is None:
        collected = collect_production_envelopes(
            repo_root=repo_root, forest_cid=forest_cid
        )
    else:
        if isinstance(envelopes, (str, bytes)) or not isinstance(envelopes, Sequence):
            raise IRIntegrationError(
                "envelopes must be a sequence of IRInputEnvelope",
                reason_code="invalid_field_type",
            )
        collected = tuple(envelopes)
        if not collected:
            raise IRIntegrationError(
                "envelopes must not be empty",
                reason_code="missing_required_field",
            )

    validated: list[IRInputEnvelope] = []
    for item in collected:
        if not isinstance(item, IRInputEnvelope):
            raise IRIntegrationError(
                "envelopes must contain IRInputEnvelope values",
                reason_code="invalid_field_type",
            )
        if require_production:
            item.require_production()
        validated.append(item)

    # Single forest epoch.
    forest_ids = {item.forest_cid for item in validated}
    if len(forest_ids) != 1:
        raise IRIntegrationError(
            "all envelopes must share one forest_cid",
            reason_code="forest_cid_mismatch",
            details={"forest_cids": sorted(forest_ids)},
        )
    epoch_forest = next(iter(forest_ids))
    if forest_cid is not None and forest_cid != epoch_forest:
        raise IRIntegrationError(
            "provided forest_cid does not match envelope forest_cid",
            reason_code="forest_cid_mismatch",
            details={"provided": forest_cid, "observed": epoch_forest},
        )

    if require_production:
        present = {item.family for item in validated}
        missing = sorted(REQUIRED_PRODUCTION_FAMILIES - present)
        if missing:
            raise IRIntegrationError(
                "required production families are missing",
                reason_code="missing_production_input",
                details={"missing_families": missing},
            )

    availability = probe_family_availability(family_modules)
    available_by_module = {
        item.module: item.available for item in availability
    }
    available_by_family = {item.family: item.available for item in availability}

    diagnostics: list[str] = []
    rows: list[NormalizedIRRow] = []
    for envelope in validated:
        family_ok = True
        row_diags: list[str] = []
        # Map envelope family to datasets module availability when known.
        origin_module = envelope.module_origin
        if origin_module:
            if origin_module in available_by_module:
                family_ok = available_by_module[origin_module]
                if not family_ok:
                    row_diags.append(
                        f"module_unavailable:{origin_module}"
                    )
                    diagnostics.append(
                        f"{envelope.family}:module_unavailable:{origin_module}"
                    )
            else:
                # Probe one-off origins not in the default family map.
                origin = probe_module_origin(origin_module)
                family_ok = origin.available
                if not family_ok:
                    row_diags.append(f"module_unavailable:{origin_module}")
                    diagnostics.append(
                        f"{envelope.family}:module_unavailable:{origin_module}"
                    )
        # Family-name availability for well-known injected families.
        mapped = {
            "observed_ast": "software_contracts",
            "knowledge_graph": "knowledge_graphs",
            "security_ir": "security_ir",
            "contract_graph": "ir_core",
            "deterministic_vector": "ir_core",
        }.get(envelope.family)
        if mapped is not None and mapped in available_by_family:
            if not available_by_family[mapped]:
                family_ok = False
                row_diags.append(f"family_unavailable:{mapped}")
                diagnostics.append(f"{envelope.family}:family_unavailable:{mapped}")

        row = NormalizedIRRow.from_envelope(
            envelope,
            family_available=family_ok,
            diagnostics=row_diags,
        )
        rows.append(row)

    registry = registry if registry is not None else DatasetsProviderRegistry()
    registry_entries: tuple[ProviderRegistryEntry, ...] = ()
    if inject:
        # Inject only the declared evidence families (effects of DCR-030).
        injectable = [
            row for row in rows if row.family in INJECTED_EVIDENCE_FAMILIES
        ]
        registry_entries = registry.inject_many(injectable)
        diagnostics.append(f"injected_families:{','.join(registry.families())}")

    input_roots = tuple(sorted({row.input_root for row in rows}))
    adapter_versions = tuple(sorted({row.adapter_version for row in rows}))
    module_origins = tuple(
        sorted({row.module_origin for row in rows if row.module_origin})
    )

    return IRNormalizationResult(
        forest_cid=epoch_forest,
        rows=tuple(rows),
        input_roots=input_roots,
        adapter_versions=adapter_versions,
        module_origins=module_origins,
        family_availability=availability,
        diagnostics=tuple(diagnostics),
        registry_entries=registry_entries,
        model_calls=0,
        authoritative=False,
        completion_authorized=False,
    )


# ---------------------------------------------------------------------------
# DatasetsLogicFacade@1
# ---------------------------------------------------------------------------


class DatasetsLogicFacade:
    """Facade for datasets logic IR normalization and registry injection.

    Interface: ``DatasetsLogicFacade@1``

    Construction and metadata inspection import nothing from
    ``ipfs_datasets_py``.  Family probes use ``importlib.util.find_spec`` only.
    """

    interface: Final = DATASETS_LOGIC_FACADE_INTERFACE
    provider_id: Final = FACADE_PROVIDER_ID
    adapter_version: Final = FACADE_ADAPTER_VERSION
    protocol_version: Final = CONTRACT_VERSION

    def __init__(
        self,
        *,
        registry: DatasetsProviderRegistry | None = None,
        family_modules: Mapping[str, str] | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self._registry = registry if registry is not None else DatasetsProviderRegistry()
        self._family_modules = dict(family_modules or _DATASETS_FAMILY_MODULES)
        self._repo_root = Path(repo_root) if repo_root is not None else None
        self._last_result: IRNormalizationResult | None = None

    @property
    def registry(self) -> DatasetsProviderRegistry:
        return self._registry

    @property
    def last_result(self) -> IRNormalizationResult | None:
        return self._last_result

    def discover_families(self) -> tuple[FamilyAvailability, ...]:
        """Discover datasets logic family availability (no authority grant)."""

        return probe_family_availability(self._family_modules)

    def normalize(
        self,
        envelopes: Sequence[IRInputEnvelope] | None = None,
        *,
        forest_cid: str | None = None,
        require_production: bool = True,
        inject: bool = True,
    ) -> IRNormalizationResult:
        """Normalize envelopes (or production artifacts) into IR rows."""

        result = normalize_contract_evidence(
            envelopes,
            repo_root=self._repo_root,
            forest_cid=forest_cid,
            registry=self._registry,
            require_production=require_production,
            inject=inject,
            family_modules=self._family_modules,
        )
        self._last_result = result
        return result

    def inject_rows(
        self, rows: Sequence[NormalizedIRRow]
    ) -> tuple[ProviderRegistryEntry, ...]:
        """Inject already-normalized rows into the provider registry."""

        return self._registry.inject_many(rows)

    def capability_receipt(self) -> dict[str, Any]:
        """Return a non-authoritative capability receipt for the facade."""

        families = self.discover_families()
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/datasets-logic-facade-capability@1",
            "interface": self.interface,
            "provider_id": self.provider_id,
            "adapter_version": self.adapter_version,
            "protocol_version": self.protocol_version,
            "family_availability": [item.to_dict() for item in families],
            "injected_families": list(INJECTED_EVIDENCE_FAMILIES),
            "required_production_families": sorted(REQUIRED_PRODUCTION_FAMILIES),
            "grants_execution_authority": False,
            "grants_proof_authority": False,
            "completion_authorized": False,
            "authoritative": False,
            "model_calls": 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "provider_id": self.provider_id,
            "adapter_version": self.adapter_version,
            "registry": self._registry.to_dict(),
            "last_result_cid": (
                self._last_result.result_cid if self._last_result is not None else ""
            ),
        }


# ---------------------------------------------------------------------------
# Artifact materialization
# ---------------------------------------------------------------------------


def materialize_ir_input(
    *,
    repo_root: Path | None = None,
    forest_cid: str | None = None,
    facade: DatasetsLogicFacade | None = None,
) -> IRNormalizationResult:
    """Materialize IR normalization from committed production DCR artifacts."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    owner = facade or DatasetsLogicFacade(repo_root=root)
    return owner.normalize(forest_cid=forest_cid, require_production=True, inject=True)


def write_ir_input(
    destination: str | Path | None = None,
    *,
    result: IRNormalizationResult | None = None,
    repo_root: Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """Atomically write the IR input artifact."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    artifact = result or materialize_ir_input(repo_root=root)
    if not artifact.verifies_cid():
        raise IRIntegrationError(
            "result CID does not reconstruct from canonical bytes",
            reason_code="cid_reconstruction_failed",
        )
    data = artifact.to_artifact_bytes()
    if len(data) > max_bytes:
        raise IRIntegrationError(
            f"artifact exceeds {max_bytes} bytes",
            reason_code="bounds_exceeded",
            details={"byte_length": len(data)},
        )
    if destination is None:
        path = _resolve_relative(root, DCR_ARTIFACT_PATH)
    else:
        path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def ensure_ir_input_artifact(
    *,
    repo_root: Path | None = None,
    force: bool = False,
) -> Path:
    """Ensure the declared DCR-030 artifact exists without unnecessary rewrites."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    out = _resolve_relative(root, DCR_ARTIFACT_PATH)
    if out.is_file() and not force:
        try:
            loaded = load_ir_input(out)
        except IRIntegrationError:
            loaded = None
        if (
            loaded is not None
            and loaded.schema == IR_NORMALIZATION_RESULT_SCHEMA
            and loaded.interface == IR_NORMALIZATION_RESULT_INTERFACE
            and loaded.evidence_term == IR_NORMALIZATION_EVIDENCE_TERM
            and loaded.verifies_cid()
        ):
            return out
    return write_ir_input(out, repo_root=root)


def load_ir_input(
    source: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> IRNormalizationResult:
    """Load and revalidate an IR input artifact."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    if source is None:
        path = _resolve_relative(root, DCR_ARTIFACT_PATH)
    else:
        path = Path(source)
    if not path.is_file():
        raise IRIntegrationError(
            f"IR input artifact missing: {path}",
            reason_code="artifact_missing",
            details={"path": str(path)},
        )
    raw = _read_bytes(path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IRIntegrationError(
            "IR input artifact is not valid JSON",
            reason_code="invalid_json",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, Mapping):
        raise IRIntegrationError(
            "IR input artifact must be a JSON object",
            reason_code="invalid_json_root",
        )
    result = IRNormalizationResult.from_dict(payload)
    if not result.verifies_cid():
        raise IRIntegrationError(
            "stored IR input CID does not reconstruct from canonical bytes",
            reason_code="cid_reconstruction_failed",
        )
    return result


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "CONTRACT_VERSION",
    "DATASETS_LOGIC_FACADE_INTERFACE",
    "DCR_ARTIFACT_PATH",
    "DCR_TASK_ID",
    "DatasetsLogicFacade",
    "DatasetsProviderRegistry",
    "EvidenceFamily",
    "FACADE_ADAPTER_VERSION",
    "FACADE_PROVIDER_ID",
    "FamilyAvailability",
    "INJECTED_EVIDENCE_FAMILIES",
    "IRInputEnvelope",
    "IRIntegrationError",
    "IRNormalizationResult",
    "IR_INPUT_ENVELOPE_INTERFACE",
    "IR_INPUT_ENVELOPE_SCHEMA",
    "IR_INTEGRATION_ARTIFACT_SCHEMA",
    "IR_NORMALIZATION_EVIDENCE_TERM",
    "IR_NORMALIZATION_RESULT_INTERFACE",
    "IR_NORMALIZATION_RESULT_SCHEMA",
    "InputAuthority",
    "ModuleOrigin",
    "NORMALIZED_IR_ROW_INTERFACE",
    "NORMALIZED_IR_ROW_SCHEMA",
    "NormalizedIRRow",
    "ProductionInputSubstitutionError",
    "ProviderRegistryEntry",
    "REQUIRED_PRODUCTION_FAMILIES",
    "build_envelope_from_bytes",
    "build_envelope_from_path",
    "canonical_ir_cid",
    "collect_production_envelopes",
    "ensure_ir_input_artifact",
    "load_ir_input",
    "materialize_ir_input",
    "normalize_contract_evidence",
    "probe_family_availability",
    "probe_module_origin",
    "write_ir_input",
]
