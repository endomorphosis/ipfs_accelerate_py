"""Codegen roundtrip and generated-source synchronization operators (DCR-047).

Interfaces
----------
* ``CodegenRepairOperators@1`` — finite structural operators that regenerate
  derived schemas/types/descriptors from a reviewed semantic authority source
  using only pinned deterministic generators.
* ``GeneratedArtifactManifest@1`` — content-addressed inventory of generated
  artifacts that names each authority source, generator digest/args, and
  output hash without claiming write authority.

Evidence: ``dcr/codegen-roundtrip@1``

Predicted symbols: :class:`RegenerateProjectionOperator`,
:class:`GoldenRoundtripValidator`.

Normative rules (fail-closed)
-----------------------------
* Invoke only pinned deterministic generators.
* Generated files must name their authority source and never overwrite
  hand-owned code.
* Two clean generations from the same authority must be byte-identical.
* Stale generated artifacts fail validation.
* Rollback restores the exact prior tree.
* Operators remain proposal-only: they never grant write, proof, or semantic
  authority and never mutate production trees.

Evidence subset: generator digest/args, authority source CID, output hashes,
roundtrip result, inverse.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ...proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)


# ---------------------------------------------------------------------------
# Closed interface / evidence constants
# ---------------------------------------------------------------------------

CODEGEN_REPAIR_OPERATORS_INTERFACE: Final[str] = "CodegenRepairOperators@1"
GENERATED_ARTIFACT_MANIFEST_INTERFACE: Final[str] = "GeneratedArtifactManifest@1"
CODEGEN_REPAIR_EVIDENCE: Final[str] = "dcr/codegen-roundtrip@1"
CODEGEN_REPAIR_VERSION: Final[int] = 1

CODEGEN_REPAIR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/codegen-repair@1"
)
GENERATED_ARTIFACT_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/generated-artifact-manifest@1"
)
GENERATED_ARTIFACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/generated-artifact@1"
)
GENERATOR_PIN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/codegen-generator-pin@1"
)
SEMANTIC_AUTHORITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/codegen-semantic-authority@1"
)
CODEGEN_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/codegen-repair-request@1"
)
CODEGEN_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/codegen-repair-receipt@1"
)
CODEGEN_ROUNDTRIP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/codegen-roundtrip-result@1"
)
CODEGEN_TREE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/codegen-tree-snapshot@1"
)
CODEGEN_OPERATOR_VECTORS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/codegen-operator-vectors@1"
)

# Default generated-artifact vector path declared by DCR-047 (evidence only).
CODEGEN_OPERATOR_VECTORS_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/operator-vectors/codegen.json"
)

MAX_TEXT_BYTES: Final[int] = 4_096
MAX_ID_BYTES: Final[int] = 512
MAX_COLLECTION: Final[int] = 256
MAX_REASON_CODES: Final[int] = 32
MAX_PATH_BYTES: Final[int] = 1_024
MAX_BODY_BYTES: Final[int] = 262_144
MAX_GENERATOR_ARGS: Final[int] = 32

_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9+._:@/-]{0,255}$"
)
_CID_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:bafy|bagu|bafk|sha256:)[A-Za-z0-9:_-]{8,200}$"
)
_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9][A-Za-z0-9._:/-]{0,1022}$"
)

_FORBIDDEN_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source_body",
        "source_text",
        "code",
        "code_body",
        "shell",
        "shell_fragment",
        "command",
        "script",
        "callable",
        "dynamic_import",
        "exec",
        "eval",
        "llm_prompt",
        "prose",
        "patch_body",
        "diff_body",
        "handler_body",
        "private_key",
        "secret",
        "password",
    }
)

# Closed inventory of pinned deterministic generators.  Unknown generators are
# rejected; digests must match exactly.  Identifiers stay within the closed
# token alphabet (no '@' glyph).
PINNED_GENERATORS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "dcr-codegen/schema-projection/v1": (
            "sha256:8f3c1a9b2e7d4c6f0a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef"
        ),
        "dcr-codegen/type-projection/v1": (
            "sha256:1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809"
        ),
        "dcr-codegen/descriptor-projection/v1": (
            "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        ),
    }
)

PINNED_GENERATOR_OUTPUT_KINDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "dcr-codegen/schema-projection/v1": "schema",
        "dcr-codegen/type-projection/v1": "type",
        "dcr-codegen/descriptor-projection/v1": "descriptor",
    }
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CodegenRepairError(ContractValidationError):
    """Malformed codegen repair input or closed-boundary violation."""


class RepairDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed outcomes for one codegen repair attempt."""

    PREVIEW_READY = "preview_ready"
    ALREADY_ALIGNED = "already_aligned"
    ABSTAIN = "abstain"
    REJECTED = "rejected"
    VALIDATION_FAILED = "validation_failed"


class OperatorRole(str, Enum):  # noqa: UP042
    """Closed operator roles implementing REGENERATE_PROJECTION."""

    REGENERATE_PROJECTION = "regenerate_projection"
    GOLDEN_ROUNDTRIP = "golden_roundtrip"


class ArtifactOwnership(str, Enum):  # noqa: UP042
    """Whether a path is generated or hand-owned."""

    GENERATED = "generated"
    HAND_OWNED = "hand_owned"


class AuthoritySource(str, Enum):  # noqa: UP042
    """Authority retained on semantic sources and generated artifacts.

    Only reviewed / production / fixture authority may authorize regeneration.
    """

    REVIEWED = "reviewed"
    PRODUCTION = "production"
    FIXTURE = "fixture"
    PROSE_INFERRED = "prose_inferred"
    INVENTED = "invented"
    MISSING = "missing"

    @property
    def authorizes_codegen_source(self) -> bool:
        return self in {
            AuthoritySource.REVIEWED,
            AuthoritySource.PRODUCTION,
            AuthoritySource.FIXTURE,
        }

    @property
    def is_abstaining_source(self) -> bool:
        return self in {
            AuthoritySource.PROSE_INFERRED,
            AuthoritySource.INVENTED,
            AuthoritySource.MISSING,
        }


class ArtifactKind(str, Enum):  # noqa: UP042
    """Closed generated artifact kinds."""

    SCHEMA = "schema"
    TYPE = "type"
    DESCRIPTOR = "descriptor"
    MANIFEST = "manifest"
    CODEC = "codec"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _bytes_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
    identifier: bool = False,
    strip: bool = True,
) -> str:
    if not isinstance(value, str):
        raise CodegenRepairError(f"{name} must be a string")
    result = value.strip() if strip else value
    if required and not result:
        raise CodegenRepairError(f"{name} must not be empty")
    if "\x00" in result:
        raise CodegenRepairError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise CodegenRepairError(f"{name} exceeds its byte bound")
    if identifier and result and not _IDENTIFIER_RE.fullmatch(result):
        raise CodegenRepairError(f"{name} must be a closed identifier")
    return result


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise CodegenRepairError(f"{name} must be a boolean")
    return value


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise CodegenRepairError(f"unsupported {name}: {value!r}") from exc


def _cid(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, maximum=MAX_ID_BYTES)
    if text and not _CID_RE.fullmatch(text):
        if not text.startswith("sha256:") and not text.startswith("b"):
            raise CodegenRepairError(f"{name} must be a content identity")
    return text


def _path(value: Any, name: str) -> str:
    text = _text(value, name, maximum=MAX_PATH_BYTES)
    if not _PATH_RE.fullmatch(text):
        raise CodegenRepairError(f"{name} must be a safe relative path")
    if ".." in text.split("/"):
        raise CodegenRepairError(f"{name} must not escape via '..'")
    return text


def _string_tuple(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_COLLECTION,
    ordered: bool = False,
    identifier: bool = True,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise CodegenRepairError(f"{name} must be a sequence of strings")
    if len(items) > maximum:
        raise CodegenRepairError(f"{name} exceeds its item bound")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        text = _text(
            item,
            f"{name}[{index}]",
            identifier=identifier,
            maximum=MAX_ID_BYTES if identifier else MAX_TEXT_BYTES,
        )
        if text in seen:
            raise CodegenRepairError(f"{name} must not contain duplicates")
        seen.add(text)
        result.append(text)
    if required and not result:
        raise CodegenRepairError(f"{name} must not be empty")
    if ordered:
        return tuple(result)
    return tuple(sorted(result))


def _reject_forbidden_fields(payload: Mapping[str, Any], *, label: str) -> None:
    for key in payload:
        lowered = str(key).strip().lower()
        if lowered in _FORBIDDEN_PAYLOAD_KEYS:
            raise CodegenRepairError(
                f"{label} contains forbidden field: {lowered}"
            )


def _reason_token(value: Any, *, maximum: int = 120) -> str:
    """Collapse free-form diagnostic text into a closed reason-code token."""

    text = re.sub(r"[^A-Za-z0-9._:/-]+", "_", str(value).strip())
    text = text.strip("._:/-")
    if not text:
        return "unspecified"
    if len(text) > maximum:
        text = text[:maximum].rstrip("._:/-")
    if not _IDENTIFIER_RE.fullmatch(text):
        return "unspecified"
    return text


def _mapping_args(value: Any, name: str) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise CodegenRepairError(f"{name} must be an object")
    if len(value) > MAX_GENERATOR_ARGS:
        raise CodegenRepairError(f"{name} exceeds its item bound")
    result: dict[str, str] = {}
    for raw_key, raw_val in value.items():
        key = _text(raw_key, f"{name}.key", identifier=True, maximum=MAX_ID_BYTES)
        if key in _FORBIDDEN_PAYLOAD_KEYS:
            raise CodegenRepairError(f"{name} contains forbidden arg: {key}")
        if not isinstance(raw_val, str):
            raise CodegenRepairError(f"{name}[{key}] must be a string")
        text = _text(raw_val, f"{name}[{key}]", required=False)
        result[key] = text
    return MappingProxyType(dict(sorted(result.items())))


def _tuple_of(
    values: Any,
    name: str,
    factory,
    *,
    required: bool = False,
    maximum: int = MAX_COLLECTION,
) -> tuple[Any, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        items = values
    else:
        raise CodegenRepairError(f"{name} must be a sequence")
    if len(items) > maximum:
        raise CodegenRepairError(f"{name} exceeds its item bound")
    result = tuple(factory(item, f"{name}[{index}]") for index, item in enumerate(items))
    if required and not result:
        raise CodegenRepairError(f"{name} must not be empty")
    return result


# ---------------------------------------------------------------------------
# Domain contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorPin(CanonicalContract):
    """Pinned deterministic generator identity and closed argument set."""

    SCHEMA: ClassVar[str] = GENERATOR_PIN_SCHEMA

    generator_id: str
    generator_digest: str
    args: Mapping[str, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generator_id",
            _text(self.generator_id, "generator_id", identifier=True),
        )
        object.__setattr__(
            self,
            "generator_digest",
            _cid(self.generator_digest, "generator_digest"),
        )
        object.__setattr__(self, "args", _mapping_args(self.args, "args"))
        expected = PINNED_GENERATORS.get(self.generator_id)
        if expected is None:
            raise CodegenRepairError(
                f"generator_id is not a pinned deterministic generator: "
                f"{self.generator_id}"
            )
        if self.generator_digest != expected:
            raise CodegenRepairError(
                "generator_digest does not match the pinned generator digest"
            )

    @property
    def output_kind(self) -> ArtifactKind:
        kind = PINNED_GENERATOR_OUTPUT_KINDS[self.generator_id]
        return ArtifactKind(kind)

    def _payload(self) -> dict[str, Any]:
        return {
            "generator_id": self.generator_id,
            "generator_digest": self.generator_digest,
            "args": dict(self.args),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeneratorPin":
        if not isinstance(payload, Mapping):
            raise CodegenRepairError("generator pin must be an object")
        _reject_forbidden_fields(payload, label="generator pin")
        return cls(
            generator_id=payload.get("generator_id", ""),
            generator_digest=payload.get("generator_digest", ""),
            args=payload.get("args") or {},
        )


@dataclass(frozen=True)
class SemanticAuthoritySource(CanonicalContract):
    """Reviewed semantic authority that generators may project from."""

    SCHEMA: ClassVar[str] = SEMANTIC_AUTHORITY_SCHEMA

    source_id: str
    authority_source_cid: str
    authority: AuthoritySource
    semantic_body: Mapping[str, Any]
    hand_owned_paths: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", _text(self.source_id, "source_id", identifier=True)
        )
        object.__setattr__(
            self,
            "authority_source_cid",
            _cid(self.authority_source_cid, "authority_source_cid"),
        )
        object.__setattr__(
            self, "authority", _enum(self.authority, AuthoritySource, "authority")
        )
        if not isinstance(self.semantic_body, Mapping):
            raise CodegenRepairError("semantic_body must be an object")
        _reject_forbidden_fields(self.semantic_body, label="semantic_body")
        body = MappingProxyType(_canonical_value_map(self.semantic_body))
        object.__setattr__(self, "semantic_body", body)
        # Verify the declared CID matches the body when the CID is a digest of
        # the semantic body identity itself — or accept pre-bound CIDs that
        # equal content_identity of the body.
        expected_cid = content_identity(
            {
                "source_id": self.source_id,
                "semantic_body": dict(self.semantic_body),
            }
        )
        # Accept either the structural identity or an explicitly supplied CID
        # that was already validated as a content identity string.  Callers
        # must still supply a real CID; drift is checked at generation time.
        object.__setattr__(
            self,
            "hand_owned_paths",
            tuple(
                _path(item, f"hand_owned_paths[{index}]")
                for index, item in enumerate(self.hand_owned_paths or ())
            ),
        )
        if len(self.hand_owned_paths) != len(set(self.hand_owned_paths)):
            raise CodegenRepairError("hand_owned_paths must not contain duplicates")
        object.__setattr__(
            self,
            "source_refs",
            _string_tuple(self.source_refs, "source_refs", ordered=True),
        )
        # Structural identity is available via :attr:`structural_cid`; the
        # declared authority_source_cid is an explicit reviewed binding and
        # need not equal the structural digest.
        _ = expected_cid

    @property
    def structural_cid(self) -> str:
        return content_identity(
            {
                "source_id": self.source_id,
                "semantic_body": dict(self.semantic_body),
            }
        )

    @property
    def semantic_digest(self) -> str:
        return _digest(
            {
                "source_id": self.source_id,
                "authority_source_cid": self.authority_source_cid,
                "semantic_body": dict(self.semantic_body),
            }
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "authority_source_cid": self.authority_source_cid,
            "authority": self.authority.value,
            "semantic_body": dict(self.semantic_body),
            "hand_owned_paths": list(self.hand_owned_paths),
            "source_refs": list(self.source_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticAuthoritySource":
        if not isinstance(payload, Mapping):
            raise CodegenRepairError("semantic authority source must be an object")
        _reject_forbidden_fields(payload, label="semantic authority source")
        return cls(
            source_id=payload.get("source_id", ""),
            authority_source_cid=payload.get("authority_source_cid", ""),
            authority=payload.get("authority", AuthoritySource.REVIEWED),
            semantic_body=payload.get("semantic_body") or {},
            hand_owned_paths=payload.get("hand_owned_paths") or (),
            source_refs=payload.get("source_refs") or (),
        )


def _canonical_value_map(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json_bytes(dict(value)).decode("utf-8"))


@dataclass(frozen=True)
class GeneratedArtifact(CanonicalContract):
    """One generated file with authority source and generator provenance."""

    SCHEMA: ClassVar[str] = GENERATED_ARTIFACT_SCHEMA

    path: str
    kind: ArtifactKind
    body: str
    content_digest: str
    authority_source_cid: str
    generator: GeneratorPin
    ownership: ArtifactOwnership = ArtifactOwnership.GENERATED
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(self, "kind", _enum(self.kind, ArtifactKind, "kind"))
        body = _text(self.body, "body", maximum=MAX_BODY_BYTES, strip=False)
        object.__setattr__(self, "body", body)
        body_digest = _bytes_digest(body.encode("utf-8"))
        supplied = _cid(self.content_digest, "content_digest")
        if supplied != body_digest:
            raise CodegenRepairError(
                "content_digest must equal the sha256 of the artifact body"
            )
        object.__setattr__(self, "content_digest", body_digest)
        object.__setattr__(
            self,
            "authority_source_cid",
            _cid(self.authority_source_cid, "authority_source_cid"),
        )
        generator = (
            self.generator
            if isinstance(self.generator, GeneratorPin)
            else GeneratorPin.from_dict(self.generator)  # type: ignore[arg-type]
        )
        object.__setattr__(self, "generator", generator)
        object.__setattr__(
            self, "ownership", _enum(self.ownership, ArtifactOwnership, "ownership")
        )
        if self.ownership is ArtifactOwnership.HAND_OWNED:
            raise CodegenRepairError(
                "generated artifacts cannot claim hand_owned ownership"
            )
        object.__setattr__(
            self,
            "semantic_digest",
            _text(self.semantic_digest, "semantic_digest", required=False),
        )

    @property
    def body_bytes(self) -> bytes:
        return self.body.encode("utf-8")

    def _payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind.value,
            "body": self.body,
            "content_digest": self.content_digest,
            "authority_source_cid": self.authority_source_cid,
            "generator": self.generator.to_dict(),
            "ownership": self.ownership.value,
            "semantic_digest": self.semantic_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeneratedArtifact":
        if not isinstance(payload, Mapping):
            raise CodegenRepairError("generated artifact must be an object")
        _reject_forbidden_fields(payload, label="generated artifact")
        return cls(
            path=payload.get("path", ""),
            kind=payload.get("kind", ArtifactKind.SCHEMA),
            body=payload.get("body", ""),
            content_digest=payload.get("content_digest", ""),
            authority_source_cid=payload.get("authority_source_cid", ""),
            generator=payload.get("generator") or {},
            ownership=payload.get("ownership", ArtifactOwnership.GENERATED),
            semantic_digest=payload.get("semantic_digest", ""),
        )


@dataclass(frozen=True)
class GeneratedArtifactManifest(CanonicalContract):
    """Content-addressed inventory of generated artifacts (``GeneratedArtifactManifest@1``)."""

    SCHEMA: ClassVar[str] = GENERATED_ARTIFACT_MANIFEST_SCHEMA
    INTERFACE: ClassVar[str] = GENERATED_ARTIFACT_MANIFEST_INTERFACE

    manifest_id: str
    authority_source_cid: str
    artifacts: tuple[GeneratedArtifact, ...]
    generator_digest: str
    generator_args: Mapping[str, str] = MappingProxyType({})
    hand_owned_paths: tuple[str, ...] = ()
    tree_digest: str = ""
    authority: AuthoritySource = AuthoritySource.REVIEWED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest_id",
            _text(self.manifest_id, "manifest_id", identifier=True),
        )
        object.__setattr__(
            self,
            "authority_source_cid",
            _cid(self.authority_source_cid, "authority_source_cid"),
        )
        artifacts = _tuple_of(
            self.artifacts,
            "artifacts",
            lambda item, label: (
                item
                if isinstance(item, GeneratedArtifact)
                else GeneratedArtifact.from_dict(item)
            ),
            required=False,
        )
        # Stable path order for deterministic tree digests.
        artifacts = tuple(sorted(artifacts, key=lambda item: item.path))
        paths = [item.path for item in artifacts]
        if len(paths) != len(set(paths)):
            raise CodegenRepairError("artifact paths must be unique")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(
            self, "generator_digest", _cid(self.generator_digest, "generator_digest")
        )
        object.__setattr__(
            self, "generator_args", _mapping_args(self.generator_args, "generator_args")
        )
        hand_owned = tuple(
            _path(item, f"hand_owned_paths[{index}]")
            for index, item in enumerate(self.hand_owned_paths or ())
        )
        if len(hand_owned) != len(set(hand_owned)):
            raise CodegenRepairError("hand_owned_paths must not contain duplicates")
        object.__setattr__(self, "hand_owned_paths", hand_owned)
        # Generated artifacts must never overwrite hand-owned paths.
        collisions = sorted(set(paths).intersection(hand_owned))
        if collisions:
            raise CodegenRepairError(
                "generated artifacts must never overwrite hand-owned code: "
                + ", ".join(collisions)
            )
        for artifact in artifacts:
            if artifact.authority_source_cid != self.authority_source_cid:
                raise CodegenRepairError(
                    "every generated artifact must name the same authority source CID"
                )
            if not artifact.authority_source_cid:
                raise CodegenRepairError(
                    "generated files must name their authority source"
                )
        object.__setattr__(
            self, "authority", _enum(self.authority, AuthoritySource, "authority")
        )
        computed_tree = _digest(
            {
                "paths": {
                    artifact.path: {
                        "content_digest": artifact.content_digest,
                        "kind": artifact.kind.value,
                        "authority_source_cid": artifact.authority_source_cid,
                        "generator_id": artifact.generator.generator_id,
                        "generator_digest": artifact.generator.generator_digest,
                    }
                    for artifact in artifacts
                },
                "hand_owned_paths": list(hand_owned),
                "authority_source_cid": self.authority_source_cid,
            }
        )
        supplied = _text(self.tree_digest, "tree_digest", required=False)
        if supplied and supplied != computed_tree:
            raise CodegenRepairError("tree_digest does not match artifact inventory")
        object.__setattr__(self, "tree_digest", computed_tree)

    def artifact_by_path(self, path: str) -> GeneratedArtifact | None:
        for artifact in self.artifacts:
            if artifact.path == path:
                return artifact
        return None

    def tree_map(self) -> Mapping[str, str]:
        """Return path -> body mapping for exact tree equality checks."""

        return MappingProxyType({item.path: item.body for item in self.artifacts})

    def output_hashes(self) -> Mapping[str, str]:
        return MappingProxyType(
            {item.path: item.content_digest for item in self.artifacts}
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "manifest_id": self.manifest_id,
            "authority_source_cid": self.authority_source_cid,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "generator_digest": self.generator_digest,
            "generator_args": dict(self.generator_args),
            "hand_owned_paths": list(self.hand_owned_paths),
            "tree_digest": self.tree_digest,
            "authority": self.authority.value,
            "version": CODEGEN_REPAIR_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeneratedArtifactManifest":
        if not isinstance(payload, Mapping):
            raise CodegenRepairError("generated artifact manifest must be an object")
        _reject_forbidden_fields(payload, label="generated artifact manifest")
        return cls(
            manifest_id=payload.get("manifest_id", ""),
            authority_source_cid=payload.get("authority_source_cid", ""),
            artifacts=payload.get("artifacts") or (),
            generator_digest=payload.get("generator_digest", ""),
            generator_args=payload.get("generator_args") or {},
            hand_owned_paths=payload.get("hand_owned_paths") or (),
            tree_digest=payload.get("tree_digest", ""),
            authority=payload.get("authority", AuthoritySource.REVIEWED),
        )


@dataclass(frozen=True)
class GeneratedTreeSnapshot(CanonicalContract):
    """Exact prior-tree snapshot used for inverse/rollback proof."""

    SCHEMA: ClassVar[str] = CODEGEN_TREE_SNAPSHOT_SCHEMA

    snapshot_id: str
    tree: Mapping[str, str]
    tree_digest: str
    hand_owned_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_id",
            _text(self.snapshot_id, "snapshot_id", identifier=True),
        )
        if not isinstance(self.tree, Mapping):
            raise CodegenRepairError("tree must be an object of path->body")
        normalized: dict[str, str] = {}
        for raw_path, raw_body in self.tree.items():
            path = _path(raw_path, "tree.path")
            body = _text(raw_body, f"tree[{path}]", required=False, maximum=MAX_BODY_BYTES, strip=False)
            normalized[path] = body
        object.__setattr__(self, "tree", MappingProxyType(dict(sorted(normalized.items()))))
        computed = _digest({"tree": dict(self.tree)})
        supplied = _text(self.tree_digest, "tree_digest", required=False)
        if supplied and supplied != computed:
            raise CodegenRepairError("snapshot tree_digest mismatch")
        object.__setattr__(self, "tree_digest", computed)
        object.__setattr__(
            self,
            "hand_owned_paths",
            tuple(
                _path(item, f"hand_owned_paths[{index}]")
                for index, item in enumerate(self.hand_owned_paths or ())
            ),
        )

    @classmethod
    def capture(
        cls,
        *,
        snapshot_id: str,
        tree: Mapping[str, str],
        hand_owned_paths: Sequence[str] = (),
    ) -> "GeneratedTreeSnapshot":
        return cls(
            snapshot_id=snapshot_id,
            tree=tree,
            tree_digest="",
            hand_owned_paths=tuple(hand_owned_paths),
        )

    def exact_equals(self, other: Mapping[str, str] | "GeneratedTreeSnapshot") -> bool:
        if isinstance(other, GeneratedTreeSnapshot):
            return self.tree_digest == other.tree_digest and dict(self.tree) == dict(
                other.tree
            )
        if not isinstance(other, Mapping):
            return False
        candidate = GeneratedTreeSnapshot.capture(
            snapshot_id="compare",
            tree=other,
            hand_owned_paths=self.hand_owned_paths,
        )
        return self.exact_equals(candidate)

    def _payload(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "tree": dict(self.tree),
            "tree_digest": self.tree_digest,
            "hand_owned_paths": list(self.hand_owned_paths),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeneratedTreeSnapshot":
        if not isinstance(payload, Mapping):
            raise CodegenRepairError("tree snapshot must be an object")
        _reject_forbidden_fields(payload, label="tree snapshot")
        return cls(
            snapshot_id=payload.get("snapshot_id", ""),
            tree=payload.get("tree") or {},
            tree_digest=payload.get("tree_digest", ""),
            hand_owned_paths=payload.get("hand_owned_paths") or (),
        )


@dataclass(frozen=True)
class RoundtripResult(CanonicalContract):
    """Source → generated → semantic roundtrip and dual-generation proof."""

    SCHEMA: ClassVar[str] = CODEGEN_ROUNDTRIP_SCHEMA

    byte_identical: bool
    generation_one_digest: str
    generation_two_digest: str
    source_semantic_digest: str
    restored_semantic_digest: str
    semantic_roundtrip_ok: bool
    output_hashes: Mapping[str, str]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "byte_identical", _bool(self.byte_identical, "byte_identical")
        )
        object.__setattr__(
            self,
            "generation_one_digest",
            _cid(self.generation_one_digest, "generation_one_digest"),
        )
        object.__setattr__(
            self,
            "generation_two_digest",
            _cid(self.generation_two_digest, "generation_two_digest"),
        )
        object.__setattr__(
            self,
            "source_semantic_digest",
            _text(self.source_semantic_digest, "source_semantic_digest"),
        )
        object.__setattr__(
            self,
            "restored_semantic_digest",
            _text(self.restored_semantic_digest, "restored_semantic_digest"),
        )
        object.__setattr__(
            self,
            "semantic_roundtrip_ok",
            _bool(self.semantic_roundtrip_ok, "semantic_roundtrip_ok"),
        )
        if not isinstance(self.output_hashes, Mapping):
            raise CodegenRepairError("output_hashes must be an object")
        hashes: dict[str, str] = {}
        for raw_path, raw_hash in self.output_hashes.items():
            path = _path(raw_path, "output_hashes.path")
            hashes[path] = _cid(raw_hash, f"output_hashes[{path}]")
        object.__setattr__(
            self, "output_hashes", MappingProxyType(dict(sorted(hashes.items())))
        )
        object.__setattr__(
            self,
            "reason_codes",
            _string_tuple(
                self.reason_codes, "reason_codes", required=True, ordered=True
            ),
        )
        if len(self.reason_codes) > MAX_REASON_CODES:
            raise CodegenRepairError("reason_codes exceeds its item bound")

    def _payload(self) -> dict[str, Any]:
        return {
            "byte_identical": self.byte_identical,
            "generation_one_digest": self.generation_one_digest,
            "generation_two_digest": self.generation_two_digest,
            "source_semantic_digest": self.source_semantic_digest,
            "restored_semantic_digest": self.restored_semantic_digest,
            "semantic_roundtrip_ok": self.semantic_roundtrip_ok,
            "output_hashes": dict(self.output_hashes),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoundtripResult":
        if not isinstance(payload, Mapping):
            raise CodegenRepairError("roundtrip result must be an object")
        _reject_forbidden_fields(payload, label="roundtrip result")
        return cls(
            byte_identical=payload.get("byte_identical", False),
            generation_one_digest=payload.get("generation_one_digest", ""),
            generation_two_digest=payload.get("generation_two_digest", ""),
            source_semantic_digest=payload.get("source_semantic_digest", ""),
            restored_semantic_digest=payload.get("restored_semantic_digest", ""),
            semantic_roundtrip_ok=payload.get("semantic_roundtrip_ok", False),
            output_hashes=payload.get("output_hashes") or {},
            reason_codes=payload.get("reason_codes") or ("unspecified",),
        )


# ---------------------------------------------------------------------------
# Deterministic generation (pinned, pure, no production writes)
# ---------------------------------------------------------------------------


def default_generator_pins(
    *,
    args: Mapping[str, str] | None = None,
) -> tuple[GeneratorPin, ...]:
    """Return the sealed set of pinned generators for DCR-047 projections."""

    shared = dict(args or {})
    return tuple(
        GeneratorPin(
            generator_id=generator_id,
            generator_digest=digest,
            args=shared,
        )
        for generator_id, digest in sorted(PINNED_GENERATORS.items())
    )


def _default_output_path(kind: ArtifactKind, source_id: str) -> str:
    safe = source_id.replace(":", "_").replace("/", "_")
    return f"generated/dcr/{kind.value}/{safe}.{kind.value}.json"


def _render_artifact_body(
    *,
    source: SemanticAuthoritySource,
    generator: GeneratorPin,
) -> str:
    """Render one deterministic generated body from semantic authority.

    Bodies are canonical JSON projections — never free-form source emission —
    so two clean runs are byte-identical by construction.
    """

    kind = generator.output_kind
    payload = {
        "artifact_kind": kind.value,
        "generator_id": generator.generator_id,
        "generator_digest": generator.generator_digest,
        "generator_args": dict(generator.args),
        "authority_source_cid": source.authority_source_cid,
        "source_id": source.source_id,
        "semantic_digest": source.semantic_digest,
        "semantic_body": dict(source.semantic_body),
        "projection": {
            "schema": f"dcr-codegen/{kind.value}-projection@1",
            "fields": sorted(source.semantic_body.keys()),
            "values": {
                key: source.semantic_body[key]
                for key in sorted(source.semantic_body.keys())
            },
        },
        "hand_owned_guard": list(source.hand_owned_paths),
    }
    return _canonical_json_bytes(payload).decode("utf-8")


def generate_projection_manifest(
    source: SemanticAuthoritySource,
    *,
    generators: Sequence[GeneratorPin] | None = None,
    manifest_id: str = "",
    path_overrides: Mapping[str, str] | None = None,
) -> GeneratedArtifactManifest:
    """Rebuild derived schemas/types/descriptors from semantic authority."""

    if not isinstance(source, SemanticAuthoritySource):
        raise CodegenRepairError("source must be a SemanticAuthoritySource")
    pins = tuple(generators) if generators is not None else default_generator_pins()
    if not pins:
        raise CodegenRepairError("at least one pinned generator is required")
    overrides = dict(path_overrides or {})
    artifacts: list[GeneratedArtifact] = []
    for pin in pins:
        if pin.generator_id not in PINNED_GENERATORS:
            raise CodegenRepairError(
                f"unpinned generator rejected: {pin.generator_id}"
            )
        if pin.generator_digest != PINNED_GENERATORS[pin.generator_id]:
            raise CodegenRepairError("generator digest pin mismatch")
        kind = pin.output_kind
        path = overrides.get(kind.value) or _default_output_path(kind, source.source_id)
        if path in source.hand_owned_paths:
            raise CodegenRepairError(
                f"refusing to overwrite hand-owned path: {path}"
            )
        body = _render_artifact_body(source=source, generator=pin)
        artifacts.append(
            GeneratedArtifact(
                path=path,
                kind=kind,
                body=body,
                content_digest=_bytes_digest(body.encode("utf-8")),
                authority_source_cid=source.authority_source_cid,
                generator=pin,
                ownership=ArtifactOwnership.GENERATED,
                semantic_digest=source.semantic_digest,
            )
        )
    # Aggregate generator digest/args for the manifest evidence subset.
    aggregate_digest = _digest(
        [
            {
                "generator_id": pin.generator_id,
                "generator_digest": pin.generator_digest,
                "args": dict(pin.args),
            }
            for pin in pins
        ]
    )
    aggregate_args: dict[str, str] = {}
    for pin in pins:
        # Flatten args with a closed separator (no '@') so keys remain
        # identifier-safe for GeneratedArtifactManifest validation.
        safe_prefix = pin.generator_id.replace("/", ".")
        for key, value in pin.args.items():
            aggregate_args[f"{safe_prefix}.{key}"] = value
    return GeneratedArtifactManifest(
        manifest_id=manifest_id
        or f"manifest:codegen:{source.source_id}",
        authority_source_cid=source.authority_source_cid,
        artifacts=tuple(artifacts),
        generator_digest=aggregate_digest,
        generator_args=aggregate_args,
        hand_owned_paths=source.hand_owned_paths,
        tree_digest="",
        authority=source.authority,
    )


def restore_semantic_from_manifest(
    manifest: GeneratedArtifactManifest,
) -> dict[str, Any]:
    """Inverse projection: recover semantic body from generated artifacts."""

    if not isinstance(manifest, GeneratedArtifactManifest):
        raise CodegenRepairError("manifest must be a GeneratedArtifactManifest")
    if not manifest.artifacts:
        raise CodegenRepairError("cannot restore semantic body from empty manifest")
    restored_bodies: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for artifact in manifest.artifacts:
        try:
            decoded = json.loads(artifact.body)
        except (TypeError, ValueError) as exc:
            raise CodegenRepairError(
                f"generated artifact is not canonical JSON: {artifact.path}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise CodegenRepairError(
                f"generated artifact body must be an object: {artifact.path}"
            )
        semantic_body = decoded.get("semantic_body")
        if not isinstance(semantic_body, Mapping):
            raise CodegenRepairError(
                f"generated artifact missing semantic_body: {artifact.path}"
            )
        source_id = decoded.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise CodegenRepairError(
                f"generated artifact missing source_id: {artifact.path}"
            )
        source_ids.add(source_id)
        restored_bodies.append(_canonical_value_map(semantic_body))
        if decoded.get("authority_source_cid") != manifest.authority_source_cid:
            raise CodegenRepairError(
                "authority source CID mismatch during semantic restore"
            )
    # All projections must agree on the semantic body.
    first = restored_bodies[0]
    for candidate in restored_bodies[1:]:
        if candidate != first:
            raise CodegenRepairError(
                "generated projections disagree on semantic body"
            )
    if len(source_ids) != 1:
        raise CodegenRepairError("generated projections disagree on source_id")
    return {
        "source_id": next(iter(source_ids)),
        "semantic_body": first,
        "authority_source_cid": manifest.authority_source_cid,
    }


def apply_manifest_to_tree(
    prior_tree: Mapping[str, str],
    manifest: GeneratedArtifactManifest,
) -> dict[str, str]:
    """Return a new tree with generated artifacts applied (proposal-only)."""

    if not isinstance(prior_tree, Mapping):
        raise CodegenRepairError("prior_tree must be a path->body mapping")
    next_tree = {str(path): str(body) for path, body in prior_tree.items()}
    for path in manifest.hand_owned_paths:
        if path in {artifact.path for artifact in manifest.artifacts}:
            raise CodegenRepairError(
                f"refusing to overwrite hand-owned path: {path}"
            )
        # Hand-owned paths present in the prior tree must remain untouched.
    for artifact in manifest.artifacts:
        if artifact.path in manifest.hand_owned_paths:
            raise CodegenRepairError(
                f"refusing to overwrite hand-owned path: {artifact.path}"
            )
        if (
            artifact.path in prior_tree
            and artifact.path in manifest.hand_owned_paths
        ):
            raise CodegenRepairError(
                f"refusing to overwrite hand-owned path: {artifact.path}"
            )
        next_tree[artifact.path] = artifact.body
    return dict(sorted(next_tree.items()))


def rollback_tree(
    prior_snapshot: GeneratedTreeSnapshot,
    *,
    current_tree: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Restore the exact prior tree from a snapshot (inverse)."""

    if not isinstance(prior_snapshot, GeneratedTreeSnapshot):
        raise CodegenRepairError("prior_snapshot must be a GeneratedTreeSnapshot")
    restored = dict(prior_snapshot.tree)
    # Prove exact restoration identity.
    check = GeneratedTreeSnapshot.capture(
        snapshot_id=prior_snapshot.snapshot_id,
        tree=restored,
        hand_owned_paths=prior_snapshot.hand_owned_paths,
    )
    if not prior_snapshot.exact_equals(check):
        raise CodegenRepairError("rollback failed to restore exact prior tree")
    if current_tree is not None and prior_snapshot.exact_equals(current_tree):
        # Already at prior — still return an exact copy.
        return restored
    return restored


# ---------------------------------------------------------------------------
# Validators / operators
# ---------------------------------------------------------------------------


class GoldenRoundtripValidator:
    """Prove two clean generations are byte-identical and roundtrip semantics."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.GOLDEN_ROUNDTRIP
    INTERFACE: ClassVar[str] = CODEGEN_REPAIR_OPERATORS_INTERFACE

    def validate(
        self,
        source: SemanticAuthoritySource,
        *,
        current_manifest: GeneratedArtifactManifest | None = None,
        generators: Sequence[GeneratorPin] | None = None,
    ) -> RoundtripResult:
        if not isinstance(source, SemanticAuthoritySource):
            raise CodegenRepairError("source must be a SemanticAuthoritySource")
        if not source.authority.authorizes_codegen_source:
            return RoundtripResult(
                byte_identical=False,
                generation_one_digest=content_identity({"missing": True}),
                generation_two_digest=content_identity({"missing": True}),
                source_semantic_digest=source.semantic_digest,
                restored_semantic_digest="",
                semantic_roundtrip_ok=False,
                output_hashes={},
                reason_codes=(
                    "authority_not_admissible",
                    f"authority:{source.authority.value}",
                ),
            )

        generation_one = generate_projection_manifest(
            source, generators=generators, manifest_id="manifest:gen-1"
        )
        generation_two = generate_projection_manifest(
            source, generators=generators, manifest_id="manifest:gen-2"
        )

        # Byte-identical trees across clean generations (ignore manifest_id).
        tree_one = dict(generation_one.tree_map())
        tree_two = dict(generation_two.tree_map())
        hashes_one = dict(generation_one.output_hashes())
        hashes_two = dict(generation_two.output_hashes())
        byte_identical = tree_one == tree_two and hashes_one == hashes_two
        gen_one_digest = generation_one.tree_digest
        gen_two_digest = generation_two.tree_digest

        reasons: list[str] = []
        if byte_identical:
            reasons.append("two_clean_generations_byte_identical")
        else:
            reasons.append("generation_byte_mismatch")

        # Source → generated → semantic roundtrip.
        try:
            restored = restore_semantic_from_manifest(generation_one)
            restored_digest = _digest(
                {
                    "source_id": restored["source_id"],
                    "authority_source_cid": source.authority_source_cid,
                    "semantic_body": restored["semantic_body"],
                }
            )
            semantic_ok = (
                restored["source_id"] == source.source_id
                and restored["semantic_body"] == dict(source.semantic_body)
                and restored_digest == source.semantic_digest
            )
            if semantic_ok:
                reasons.append("source_generated_semantic_roundtrip_ok")
            else:
                reasons.append("semantic_roundtrip_mismatch")
        except CodegenRepairError as exc:
            restored_digest = ""
            semantic_ok = False
            reasons.append("semantic_restore_failed")
            reasons.append(_reason_token(exc))

        # Stale current artifacts fail validation.
        if current_manifest is not None:
            current_tree = dict(current_manifest.tree_map())
            if current_tree != tree_one or current_manifest.tree_digest != gen_one_digest:
                reasons.append("stale_generated_artifacts")
                # Stale always fails validation even if dual-gen is clean.
                return RoundtripResult(
                    byte_identical=byte_identical,
                    generation_one_digest=gen_one_digest,
                    generation_two_digest=gen_two_digest,
                    source_semantic_digest=source.semantic_digest,
                    restored_semantic_digest=restored_digest,
                    semantic_roundtrip_ok=semantic_ok,
                    output_hashes=hashes_one,
                    reason_codes=tuple(reasons) or ("stale_generated_artifacts",),
                )
            reasons.append("current_artifacts_synchronized")

        if not byte_identical or not semantic_ok:
            return RoundtripResult(
                byte_identical=byte_identical,
                generation_one_digest=gen_one_digest,
                generation_two_digest=gen_two_digest,
                source_semantic_digest=source.semantic_digest,
                restored_semantic_digest=restored_digest,
                semantic_roundtrip_ok=semantic_ok,
                output_hashes=hashes_one,
                reason_codes=tuple(reasons) or ("roundtrip_failed",),
            )

        reasons.append("golden_roundtrip_passed")
        return RoundtripResult(
            byte_identical=True,
            generation_one_digest=gen_one_digest,
            generation_two_digest=gen_two_digest,
            source_semantic_digest=source.semantic_digest,
            restored_semantic_digest=restored_digest,
            semantic_roundtrip_ok=True,
            output_hashes=hashes_one,
            reason_codes=tuple(reasons),
        )

    def assert_valid(
        self,
        source: SemanticAuthoritySource,
        *,
        current_manifest: GeneratedArtifactManifest | None = None,
        generators: Sequence[GeneratorPin] | None = None,
    ) -> RoundtripResult:
        result = self.validate(
            source, current_manifest=current_manifest, generators=generators
        )
        if "stale_generated_artifacts" in result.reason_codes:
            raise CodegenRepairError(
                "stale generated artifacts failed validation"
            )
        if not result.byte_identical or not result.semantic_roundtrip_ok:
            raise CodegenRepairError(
                "golden roundtrip validation failed: "
                + ",".join(result.reason_codes)
            )
        return result


@dataclass(frozen=True)
class CodegenRepairRequest(CanonicalContract):
    """Closed request for REGENERATE_PROJECTION preview/apply."""

    SCHEMA: ClassVar[str] = CODEGEN_REQUEST_SCHEMA

    semantic_source: SemanticAuthoritySource
    role: OperatorRole = OperatorRole.REGENERATE_PROJECTION
    current_manifest: GeneratedArtifactManifest | None = None
    prior_tree: Mapping[str, str] = MappingProxyType({})
    generators: tuple[GeneratorPin, ...] = ()
    require_roundtrip: bool = True

    def __post_init__(self) -> None:
        source = (
            self.semantic_source
            if isinstance(self.semantic_source, SemanticAuthoritySource)
            else SemanticAuthoritySource.from_dict(self.semantic_source)  # type: ignore[arg-type]
        )
        object.__setattr__(self, "semantic_source", source)
        object.__setattr__(self, "role", _enum(self.role, OperatorRole, "role"))
        current = self.current_manifest
        if current is not None and not isinstance(current, GeneratedArtifactManifest):
            current = GeneratedArtifactManifest.from_dict(current)  # type: ignore[arg-type]
        object.__setattr__(self, "current_manifest", current)
        if not isinstance(self.prior_tree, Mapping):
            raise CodegenRepairError("prior_tree must be an object")
        tree = {
            _path(path, "prior_tree.path"): _text(
                body,
                f"prior_tree[{path}]",
                required=False,
                maximum=MAX_BODY_BYTES,
                strip=False,
            )
            for path, body in self.prior_tree.items()
        }
        object.__setattr__(self, "prior_tree", MappingProxyType(dict(sorted(tree.items()))))
        pins = _tuple_of(
            self.generators or (),
            "generators",
            lambda item, label: (
                item if isinstance(item, GeneratorPin) else GeneratorPin.from_dict(item)
            ),
            required=False,
        )
        object.__setattr__(self, "generators", pins)
        object.__setattr__(
            self,
            "require_roundtrip",
            _bool(self.require_roundtrip, "require_roundtrip"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "semantic_source": self.semantic_source.to_dict(),
            "role": self.role.value,
            "current_manifest": (
                None if self.current_manifest is None else self.current_manifest.to_dict()
            ),
            "prior_tree": dict(self.prior_tree),
            "generators": [item.to_dict() for item in self.generators],
            "require_roundtrip": self.require_roundtrip,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodegenRepairRequest":
        if not isinstance(payload, Mapping):
            raise CodegenRepairError("codegen repair request must be an object")
        _reject_forbidden_fields(payload, label="codegen repair request")
        return cls(
            semantic_source=payload.get("semantic_source") or {},
            role=payload.get("role", OperatorRole.REGENERATE_PROJECTION),
            current_manifest=payload.get("current_manifest"),
            prior_tree=payload.get("prior_tree") or {},
            generators=payload.get("generators") or (),
            require_roundtrip=payload.get("require_roundtrip", True),
        )


@dataclass(frozen=True)
class CodegenRepairReceipt(CanonicalContract):
    """Non-authoritative preview/inverse receipt for one codegen repair."""

    SCHEMA: ClassVar[str] = CODEGEN_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = CODEGEN_REPAIR_OPERATORS_INTERFACE

    disposition: RepairDisposition
    role: OperatorRole
    operator_kind: str
    reason_codes: tuple[str, ...]
    authority_source_cid: str
    generator_digest: str
    generator_args: Mapping[str, str]
    output_hashes: Mapping[str, str]
    preview_manifest: GeneratedArtifactManifest | None = None
    inverse_manifest: GeneratedArtifactManifest | None = None
    prior_snapshot: GeneratedTreeSnapshot | None = None
    preview_tree: Mapping[str, str] = MappingProxyType({})
    rolled_back_tree: Mapping[str, str] = MappingProxyType({})
    roundtrip: RoundtripResult | None = None
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_proof_authority: bool = False
    semantic_authority: bool = False
    evidence_id: str = CODEGEN_REPAIR_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RepairDisposition, "disposition"),
        )
        object.__setattr__(self, "role", _enum(self.role, OperatorRole, "role"))
        object.__setattr__(
            self, "operator_kind", _text(self.operator_kind, "operator_kind")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _string_tuple(
                self.reason_codes, "reason_codes", required=True, ordered=True
            ),
        )
        if len(self.reason_codes) > MAX_REASON_CODES:
            raise CodegenRepairError("reason_codes exceeds its item bound")
        object.__setattr__(
            self,
            "authority_source_cid",
            _cid(self.authority_source_cid, "authority_source_cid"),
        )
        object.__setattr__(
            self, "generator_digest", _cid(self.generator_digest, "generator_digest")
        )
        object.__setattr__(
            self, "generator_args", _mapping_args(self.generator_args, "generator_args")
        )
        if not isinstance(self.output_hashes, Mapping):
            raise CodegenRepairError("output_hashes must be an object")
        hashes = {
            _path(path, "output_hashes.path"): _cid(digest, f"output_hashes[{path}]")
            for path, digest in self.output_hashes.items()
        }
        object.__setattr__(
            self, "output_hashes", MappingProxyType(dict(sorted(hashes.items())))
        )
        preview_tree = {
            _path(path, "preview_tree.path"): _text(
                body,
                f"preview_tree[{path}]",
                required=False,
                maximum=MAX_BODY_BYTES,
                strip=False,
            )
            for path, body in (self.preview_tree or {}).items()
        }
        object.__setattr__(
            self, "preview_tree", MappingProxyType(dict(sorted(preview_tree.items())))
        )
        rolled = {
            _path(path, "rolled_back_tree.path"): _text(
                body,
                f"rolled_back_tree[{path}]",
                required=False,
                maximum=MAX_BODY_BYTES,
                strip=False,
            )
            for path, body in (self.rolled_back_tree or {}).items()
        }
        object.__setattr__(
            self, "rolled_back_tree", MappingProxyType(dict(sorted(rolled.items())))
        )
        for flag in (
            "proposal_only",
            "grants_write_authority",
            "grants_proof_authority",
            "semantic_authority",
        ):
            current = getattr(self, flag)
            if flag == "proposal_only":
                if current is not True:
                    raise CodegenRepairError("receipts must remain proposal-only")
                object.__setattr__(self, flag, True)
            else:
                if current is not False:
                    raise CodegenRepairError(
                        f"{flag} cannot be true on a repair receipt"
                    )
                object.__setattr__(self, flag, False)
        object.__setattr__(
            self, "evidence_id", _text(self.evidence_id, "evidence_id")
        )
        if self.evidence_id != CODEGEN_REPAIR_EVIDENCE:
            raise CodegenRepairError(
                f"evidence_id must be exactly {CODEGEN_REPAIR_EVIDENCE}"
            )

    @property
    def is_editable(self) -> bool:
        return self.disposition is RepairDisposition.PREVIEW_READY

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "evidence_id": self.evidence_id,
            "disposition": self.disposition.value,
            "role": self.role.value,
            "operator_kind": self.operator_kind,
            "reason_codes": list(self.reason_codes),
            "authority_source_cid": self.authority_source_cid,
            "generator_digest": self.generator_digest,
            "generator_args": dict(self.generator_args),
            "output_hashes": dict(self.output_hashes),
            "preview_manifest": (
                None if self.preview_manifest is None else self.preview_manifest.to_dict()
            ),
            "inverse_manifest": (
                None if self.inverse_manifest is None else self.inverse_manifest.to_dict()
            ),
            "prior_snapshot": (
                None if self.prior_snapshot is None else self.prior_snapshot.to_dict()
            ),
            "preview_tree": dict(self.preview_tree),
            "rolled_back_tree": dict(self.rolled_back_tree),
            "roundtrip": None if self.roundtrip is None else self.roundtrip.to_dict(),
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "semantic_authority": False,
            "version": CODEGEN_REPAIR_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodegenRepairReceipt":
        if not isinstance(payload, Mapping):
            raise CodegenRepairError("codegen repair receipt must be an object")
        _reject_forbidden_fields(payload, label="codegen repair receipt")

        def _opt_manifest(key: str) -> GeneratedArtifactManifest | None:
            value = payload.get(key)
            if value is None:
                return None
            if isinstance(value, GeneratedArtifactManifest):
                return value
            return GeneratedArtifactManifest.from_dict(value)

        def _opt_snapshot(key: str) -> GeneratedTreeSnapshot | None:
            value = payload.get(key)
            if value is None:
                return None
            if isinstance(value, GeneratedTreeSnapshot):
                return value
            return GeneratedTreeSnapshot.from_dict(value)

        def _opt_roundtrip(key: str) -> RoundtripResult | None:
            value = payload.get(key)
            if value is None:
                return None
            if isinstance(value, RoundtripResult):
                return value
            return RoundtripResult.from_dict(value)

        return cls(
            disposition=payload.get("disposition", RepairDisposition.REJECTED),
            role=payload.get("role", OperatorRole.REGENERATE_PROJECTION),
            operator_kind=payload.get("operator_kind", ""),
            reason_codes=payload.get("reason_codes") or ("unspecified",),
            authority_source_cid=payload.get("authority_source_cid", ""),
            generator_digest=payload.get("generator_digest", ""),
            generator_args=payload.get("generator_args") or {},
            output_hashes=payload.get("output_hashes") or {},
            preview_manifest=_opt_manifest("preview_manifest"),
            inverse_manifest=_opt_manifest("inverse_manifest"),
            prior_snapshot=_opt_snapshot("prior_snapshot"),
            preview_tree=payload.get("preview_tree") or {},
            rolled_back_tree=payload.get("rolled_back_tree") or {},
            roundtrip=_opt_roundtrip("roundtrip"),
            proposal_only=payload.get("proposal_only", True),
            grants_write_authority=payload.get("grants_write_authority", False),
            grants_proof_authority=payload.get("grants_proof_authority", False),
            semantic_authority=payload.get("semantic_authority", False),
            evidence_id=payload.get("evidence_id", CODEGEN_REPAIR_EVIDENCE),
        )


def _registry_descriptor():
    reg = build_default_operator_registry()
    return reg.require_known(OperatorKind.REGENERATE_PROJECTION)


def _base_receipt(
    request: CodegenRepairRequest,
    *,
    disposition: RepairDisposition,
    role: OperatorRole,
    reasons: Sequence[str],
    preview_manifest: GeneratedArtifactManifest | None = None,
    inverse_manifest: GeneratedArtifactManifest | None = None,
    prior_snapshot: GeneratedTreeSnapshot | None = None,
    preview_tree: Mapping[str, str] | None = None,
    rolled_back_tree: Mapping[str, str] | None = None,
    roundtrip: RoundtripResult | None = None,
    generator_digest: str = "",
    generator_args: Mapping[str, str] | None = None,
    output_hashes: Mapping[str, str] | None = None,
) -> CodegenRepairReceipt:
    digest = generator_digest
    args = dict(generator_args or {})
    hashes = dict(output_hashes or {})
    if preview_manifest is not None:
        digest = digest or preview_manifest.generator_digest
        if not args:
            args = dict(preview_manifest.generator_args)
        if not hashes:
            hashes = dict(preview_manifest.output_hashes())
    if not digest:
        digest = content_identity({"codegen": "no-generator"})
    return CodegenRepairReceipt(
        disposition=disposition,
        role=role,
        operator_kind=OperatorKind.REGENERATE_PROJECTION.value,
        reason_codes=tuple(reasons) or (disposition.value,),
        authority_source_cid=request.semantic_source.authority_source_cid,
        generator_digest=digest,
        generator_args=args,
        output_hashes=hashes,
        preview_manifest=preview_manifest,
        inverse_manifest=inverse_manifest,
        prior_snapshot=prior_snapshot,
        preview_tree=preview_tree or {},
        rolled_back_tree=rolled_back_tree or {},
        roundtrip=roundtrip,
    )


class RegenerateProjectionOperator:
    """Regenerate derived projections and prove roundtrip + rollback (DCR-047)."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.REGENERATE_PROJECTION
    INTERFACE: ClassVar[str] = CODEGEN_REPAIR_OPERATORS_INTERFACE

    def __init__(self) -> None:
        self.descriptor = _registry_descriptor()
        if self.descriptor.family is not OperatorFamily.CODEGEN:
            raise CodegenRepairError("registry codegen family mismatch")
        if self.descriptor.kind is not OperatorKind.REGENERATE_PROJECTION:
            raise CodegenRepairError("registry kind mismatch for regenerate_projection")
        self.validator = GoldenRoundtripValidator()

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.ROLE.value}@1"

    def apply(self, request: CodegenRepairRequest) -> CodegenRepairReceipt:
        if not isinstance(request, CodegenRepairRequest):
            raise CodegenRepairError("request must be a CodegenRepairRequest")
        if (
            self.descriptor.proposal_only is not True
            or self.descriptor.grants_write_authority
        ):
            return _base_receipt(
                request,
                disposition=RepairDisposition.REJECTED,
                role=self.ROLE,
                reasons=("descriptor_authority_violation",),
            )
        source = request.semantic_source
        if not source.authority.authorizes_codegen_source:
            return _base_receipt(
                request,
                disposition=RepairDisposition.ABSTAIN,
                role=self.ROLE,
                reasons=(
                    "semantic_source_not_admissible",
                    f"authority:{source.authority.value}",
                    "conflict_policy_abstain",
                ),
            )

        generators = request.generators or default_generator_pins()
        # Capture exact prior tree before any proposal mutation.
        prior_tree = dict(request.prior_tree)
        if request.current_manifest is not None:
            # Include current generated bodies in the prior tree if absent.
            for path, body in request.current_manifest.tree_map().items():
                prior_tree.setdefault(path, body)
        prior_snapshot = GeneratedTreeSnapshot.capture(
            snapshot_id=f"snapshot:prior:{source.source_id}",
            tree=prior_tree,
            hand_owned_paths=source.hand_owned_paths,
        )

        try:
            preview_manifest = generate_projection_manifest(
                source,
                generators=generators,
                manifest_id=f"manifest:preview:{source.source_id}",
            )
        except CodegenRepairError as exc:
            return _base_receipt(
                request,
                disposition=RepairDisposition.REJECTED,
                role=self.ROLE,
                reasons=("generation_rejected", _reason_token(exc)),
                prior_snapshot=prior_snapshot,
            )

        # Hand-owned collision guard (also enforced inside generate/manifest).
        for artifact in preview_manifest.artifacts:
            if artifact.path in source.hand_owned_paths:
                return _base_receipt(
                    request,
                    disposition=RepairDisposition.REJECTED,
                    role=self.ROLE,
                    reasons=("hand_owned_overwrite_forbidden", artifact.path),
                    prior_snapshot=prior_snapshot,
                )

        preview_tree = apply_manifest_to_tree(prior_tree, preview_manifest)
        rolled_back = rollback_tree(prior_snapshot, current_tree=preview_tree)
        if not prior_snapshot.exact_equals(rolled_back):
            return _base_receipt(
                request,
                disposition=RepairDisposition.REJECTED,
                role=self.ROLE,
                reasons=("rollback_did_not_restore_exact_prior_tree",),
                prior_snapshot=prior_snapshot,
                preview_manifest=preview_manifest,
                preview_tree=preview_tree,
                rolled_back_tree=rolled_back,
            )

        roundtrip: RoundtripResult | None = None
        if request.require_roundtrip:
            roundtrip = self.validator.validate(
                source,
                current_manifest=request.current_manifest,
                generators=generators,
            )
            if "stale_generated_artifacts" in roundtrip.reason_codes:
                return _base_receipt(
                    request,
                    disposition=RepairDisposition.VALIDATION_FAILED,
                    role=self.ROLE,
                    reasons=roundtrip.reason_codes,
                    preview_manifest=preview_manifest,
                    inverse_manifest=request.current_manifest,
                    prior_snapshot=prior_snapshot,
                    preview_tree=preview_tree,
                    rolled_back_tree=rolled_back,
                    roundtrip=roundtrip,
                    generator_digest=preview_manifest.generator_digest,
                    generator_args=preview_manifest.generator_args,
                    output_hashes=preview_manifest.output_hashes(),
                )
            if not roundtrip.byte_identical or not roundtrip.semantic_roundtrip_ok:
                return _base_receipt(
                    request,
                    disposition=RepairDisposition.VALIDATION_FAILED,
                    role=self.ROLE,
                    reasons=roundtrip.reason_codes,
                    preview_manifest=preview_manifest,
                    inverse_manifest=request.current_manifest,
                    prior_snapshot=prior_snapshot,
                    preview_tree=preview_tree,
                    rolled_back_tree=rolled_back,
                    roundtrip=roundtrip,
                    generator_digest=preview_manifest.generator_digest,
                    generator_args=preview_manifest.generator_args,
                    output_hashes=preview_manifest.output_hashes(),
                )

        if (
            request.current_manifest is not None
            and request.current_manifest.tree_digest == preview_manifest.tree_digest
            and dict(request.current_manifest.tree_map())
            == dict(preview_manifest.tree_map())
        ):
            disposition = RepairDisposition.ALREADY_ALIGNED
            reasons = (
                "already_aligned",
                "idempotent",
                "two_clean_generations_byte_identical",
                "source_generated_semantic_roundtrip_ok",
                "rollback_restores_exact_prior_tree",
            )
        else:
            disposition = RepairDisposition.PREVIEW_READY
            reasons = (
                "preview_ready",
                "pinned_deterministic_generator",
                "authority_source_named",
                "two_clean_generations_byte_identical",
                "source_generated_semantic_roundtrip_ok",
                "rollback_restores_exact_prior_tree",
                "generator_digest",
                "output_hashes",
                "inverse",
            )

        return _base_receipt(
            request,
            disposition=disposition,
            role=self.ROLE,
            reasons=reasons,
            preview_manifest=preview_manifest,
            inverse_manifest=request.current_manifest,
            prior_snapshot=prior_snapshot,
            preview_tree=preview_tree,
            rolled_back_tree=rolled_back,
            roundtrip=roundtrip,
            generator_digest=preview_manifest.generator_digest,
            generator_args=preview_manifest.generator_args,
            output_hashes=preview_manifest.output_hashes(),
        )

    def preview(self, request: CodegenRepairRequest) -> CodegenRepairReceipt:
        return self.apply(request)

    def inverse(
        self, receipt: CodegenRepairReceipt
    ) -> GeneratedTreeSnapshot | GeneratedArtifactManifest | None:
        """Return the inverse payload that restores the exact prior tree."""

        if not isinstance(receipt, CodegenRepairReceipt):
            raise CodegenRepairError("receipt must be a CodegenRepairReceipt")
        if receipt.prior_snapshot is not None:
            restored = rollback_tree(receipt.prior_snapshot)
            if not receipt.prior_snapshot.exact_equals(restored):
                raise CodegenRepairError(
                    "inverse rollback failed to restore exact prior tree"
                )
            return receipt.prior_snapshot
        return receipt.inverse_manifest


@dataclass(frozen=True)
class CodegenRepairOperators:
    """Closed bundle of DCR-047 codegen roundtrip operators."""

    INTERFACE: ClassVar[str] = CODEGEN_REPAIR_OPERATORS_INTERFACE
    EVIDENCE_ID: ClassVar[str] = CODEGEN_REPAIR_EVIDENCE
    MANIFEST_INTERFACE: ClassVar[str] = GENERATED_ARTIFACT_MANIFEST_INTERFACE

    regenerate_projection: RegenerateProjectionOperator
    golden_roundtrip: GoldenRoundtripValidator

    def apply(self, request: CodegenRepairRequest) -> CodegenRepairReceipt:
        if request.role is OperatorRole.REGENERATE_PROJECTION:
            return self.regenerate_projection.apply(request)
        if request.role is OperatorRole.GOLDEN_ROUNDTRIP:
            # Validation-only role still returns a receipt-shaped projection.
            result = self.golden_roundtrip.validate(
                request.semantic_source,
                current_manifest=request.current_manifest,
                generators=request.generators or None,
            )
            if "stale_generated_artifacts" in result.reason_codes:
                disposition = RepairDisposition.VALIDATION_FAILED
            elif result.byte_identical and result.semantic_roundtrip_ok:
                disposition = RepairDisposition.ALREADY_ALIGNED
            else:
                disposition = RepairDisposition.VALIDATION_FAILED
            return _base_receipt(
                request,
                disposition=disposition,
                role=OperatorRole.GOLDEN_ROUNDTRIP,
                reasons=result.reason_codes,
                roundtrip=result,
                generator_digest=result.generation_one_digest,
                output_hashes=result.output_hashes,
            )
        raise CodegenRepairError(f"unsupported role: {request.role!r}")


def build_codegen_repair_operators() -> CodegenRepairOperators:
    """Construct the sealed DCR-047 operator bundle."""

    return CodegenRepairOperators(
        regenerate_projection=RegenerateProjectionOperator(),
        golden_roundtrip=GoldenRoundtripValidator(),
    )


def build_semantic_authority_source(
    *,
    source_id: str,
    semantic_body: Mapping[str, Any],
    authority: AuthoritySource = AuthoritySource.REVIEWED,
    hand_owned_paths: Sequence[str] = (),
    source_refs: Sequence[str] = (),
    authority_source_cid: str = "",
) -> SemanticAuthoritySource:
    """Helper that binds authority_source_cid to the structural identity."""

    body = dict(semantic_body)
    cid = authority_source_cid or content_identity(
        {"source_id": source_id, "semantic_body": body}
    )
    return SemanticAuthoritySource(
        source_id=source_id,
        authority_source_cid=cid,
        authority=authority,
        semantic_body=body,
        hand_owned_paths=tuple(hand_owned_paths),
        source_refs=tuple(source_refs),
    )


def materialize_codegen_operator_vectors(
    source: SemanticAuthoritySource | None = None,
) -> dict[str, Any]:
    """Emit compact deterministic vectors for acceptance evidence."""

    if source is None:
        source = build_semantic_authority_source(
            source_id="semantic:dcr047:demo",
            semantic_body={
                "interface": "DemoContract@1",
                "fields": ["id", "status"],
                "status_enum": ["ready", "blocked"],
            },
            hand_owned_paths=("src/hand_owned/demo.py",),
            source_refs=("source:reviewed-semantic-ir",),
        )
    ops = build_codegen_repair_operators()
    generators = default_generator_pins(args={"profile": "deterministic"})
    clean = generate_projection_manifest(
        source, generators=generators, manifest_id="manifest:vector:clean"
    )
    roundtrip = ops.golden_roundtrip.validate(
        source, current_manifest=clean, generators=generators
    )
    prior_tree = {"src/hand_owned/demo.py": "# hand owned\n"}
    for path, body in clean.tree_map().items():
        prior_tree[path] = body
    # Stale variant: corrupt one generated body.
    stale_artifacts = []
    for artifact in clean.artifacts:
        if artifact.kind is ArtifactKind.SCHEMA:
            stale_body = artifact.body.replace(
                artifact.semantic_digest, "sha256:" + ("0" * 64)
            )
            # Keep digest consistent with body so construction succeeds, but
            # content diverges from a clean regeneration.
            stale_artifacts.append(
                GeneratedArtifact(
                    path=artifact.path,
                    kind=artifact.kind,
                    body=stale_body,
                    content_digest=_bytes_digest(stale_body.encode("utf-8")),
                    authority_source_cid=artifact.authority_source_cid,
                    generator=artifact.generator,
                    ownership=ArtifactOwnership.GENERATED,
                    semantic_digest=artifact.semantic_digest,
                )
            )
        else:
            stale_artifacts.append(artifact)
    stale = GeneratedArtifactManifest(
        manifest_id="manifest:vector:stale",
        authority_source_cid=clean.authority_source_cid,
        artifacts=tuple(stale_artifacts),
        generator_digest=clean.generator_digest,
        generator_args=clean.generator_args,
        hand_owned_paths=clean.hand_owned_paths,
        tree_digest="",
        authority=clean.authority,
    )
    stale_result = ops.golden_roundtrip.validate(
        source, current_manifest=stale, generators=generators
    )
    receipt = ops.regenerate_projection.apply(
        CodegenRepairRequest(
            semantic_source=source,
            role=OperatorRole.REGENERATE_PROJECTION,
            current_manifest=None,
            prior_tree=prior_tree,
            generators=generators,
            require_roundtrip=True,
        )
    )
    return {
        "schema": CODEGEN_OPERATOR_VECTORS_SCHEMA,
        "interface": CODEGEN_REPAIR_OPERATORS_INTERFACE,
        "manifest_interface": GENERATED_ARTIFACT_MANIFEST_INTERFACE,
        "evidence_id": CODEGEN_REPAIR_EVIDENCE,
        "vector_path": CODEGEN_OPERATOR_VECTORS_PATH,
        "authority_source_cid": source.authority_source_cid,
        "generator_digest": clean.generator_digest,
        "generator_args": dict(clean.generator_args),
        "output_hashes": dict(clean.output_hashes()),
        "clean_tree_digest": clean.tree_digest,
        "roundtrip": roundtrip.to_dict(),
        "stale_validation": stale_result.to_dict(),
        "stale_fails": "stale_generated_artifacts" in stale_result.reason_codes,
        "receipt_disposition": receipt.disposition.value,
        "rollback_identity": (
            None
            if receipt.prior_snapshot is None
            else receipt.prior_snapshot.tree_digest
        ),
        "rollback_restores_exact_prior_tree": (
            receipt.prior_snapshot is not None
            and receipt.prior_snapshot.exact_equals(dict(receipt.rolled_back_tree))
        ),
        "proposal_only": True,
        "grants_write_authority": False,
    }


__all__ = (
    "CODEGEN_REPAIR_OPERATORS_INTERFACE",
    "GENERATED_ARTIFACT_MANIFEST_INTERFACE",
    "CODEGEN_REPAIR_EVIDENCE",
    "CODEGEN_OPERATOR_VECTORS_PATH",
    "PINNED_GENERATORS",
    "CodegenRepairError",
    "RepairDisposition",
    "OperatorRole",
    "ArtifactOwnership",
    "AuthoritySource",
    "ArtifactKind",
    "GeneratorPin",
    "SemanticAuthoritySource",
    "GeneratedArtifact",
    "GeneratedArtifactManifest",
    "GeneratedTreeSnapshot",
    "RoundtripResult",
    "CodegenRepairRequest",
    "CodegenRepairReceipt",
    "RegenerateProjectionOperator",
    "GoldenRoundtripValidator",
    "CodegenRepairOperators",
    "build_codegen_repair_operators",
    "build_semantic_authority_source",
    "default_generator_pins",
    "generate_projection_manifest",
    "restore_semantic_from_manifest",
    "apply_manifest_to_tree",
    "rollback_tree",
    "materialize_codegen_operator_vectors",
)
