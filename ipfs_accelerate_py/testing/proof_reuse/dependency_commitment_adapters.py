"""Reviewed dependency-injection commitment adapters (PCTDD-024).

Accelerate-owned adapter execution for privacy-safe fixture-instance and
injected-dependency commitments.  Datasets owns the ``@1`` schemas; this
module only classifies values and binds those schemas.

Rules:

* Only the closed review-candidate families may commit supported values.
* Opaque, unreviewed, incomplete, unknown, or unsupported values force full
  execution.  They never authorize pytest skip and never admit production.
* Public identity is provider identity, version, epoch, and policy.  Secret
  bytes, credentials, and witness material are rejected from every payload.
* ``TestExecutionKeyV2`` is not assembled here.  Adapter output is an
  integrity commitment, not execution or semantics.

Import is cold-safe: no pytest, network, package installer, or prover.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

REVIEWED_DEPENDENCY_COMMITMENT_ADAPTER_INTERFACE: Final = (
    "ReviewedDependencyCommitmentAdapter@1"
)
DEPENDENCY_COMMITMENT_ADAPTER_REGISTRY_INTERFACE: Final = (
    "DependencyCommitmentAdapterRegistry@1"
)
OPAQUE_DEPENDENCY_ADAPTER_INTERFACE: Final = "OpaqueDependencyAdapter@1"
ADAPTER_COMMIT_RESULT_INTERFACE: Final = "DependencyAdapterCommitResult@1"
ADAPTER_COMMIT_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/dependency-adapter-commit-result@1"
)
ADAPTER_POLICY_INTERFACE: Final = "ReviewedDependencyAdapterPolicy@1"
CLAIM_CLASS: Final = "IntegrityCommitment"
DEFAULT_REUSE_CLASS: Final = "opaque"
DEFAULT_COMPLETENESS: Final = "unknown"
AUTHORITATIVE_AFTER: Final = "current_setup"
PRIVACY_SECRET_RULE: Final = (
    "commit provider identity/version/epoch/policy, never secret bytes"
)
DEFAULT_EPOCH: Final = "epoch:pctdd-024-adapter-v1"
DEFAULT_PROVIDER_IDENTITY: Final = (
    "ipfs_accelerate_py.testing.proof_reuse.dependency_commitment_adapters"
)
MAX_TEXT_CHARS: Final = 4_096
MAX_NAME_CHARS: Final = 256
MAX_SEQUENCE_ITEMS: Final = 64
MAX_EXTRA_KEYS: Final = 32
_DIGEST_PREFIX: Final = "sha256:"

CLOSED_REUSE_CLASSES: Final[tuple[str, ...]] = (
    "pure",
    "deterministic_snapshot",
    "transactional",
    "idempotent_external",
    "effectful_replayable",
    "effectful_nonreplayable",
    "opaque",
)
COMMIT_ELIGIBLE_REUSE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "pure",
        "deterministic_snapshot",
        "transactional",
        "idempotent_external",
        "effectful_replayable",
    }
)
FULL_EXECUTION_REUSE_CLASSES: Final[frozenset[str]] = frozenset(
    {"opaque", "effectful_nonreplayable"}
)
REVIEW_CANDIDATE_FAMILIES: Final[tuple[str, ...]] = (
    "immutable_scalar",
    "frozen_canonical_record",
    "path_independent_temp_directory_root",
    "transaction_baseline",
    "clock",
    "rng",
    "service_image_config_corpus",
    "explicit_environment_projection",
)
REVIEW_CANDIDATE_INVENTORY_LABELS: Final[tuple[str, ...]] = (
    "immutable scalar",
    "frozen canonical record",
    "path-independent temp directory root",
    "transaction baseline",
    "clock",
    "RNG",
    "service image/config/corpus",
    "explicit environment projection",
)
REVIEW_CANDIDATE_FAMILY_TO_INVENTORY: Final[dict[str, str]] = dict(
    zip(REVIEW_CANDIDATE_FAMILIES, REVIEW_CANDIDATE_INVENTORY_LABELS, strict=True)
)
FAMILY_DEFAULT_KIND: Final[dict[str, str]] = {
    "immutable_scalar": "immutable_scalar",
    "frozen_canonical_record": "frozen_canonical_record",
    "path_independent_temp_directory_root": "temp_directory_root",
    "transaction_baseline": "transaction_baseline",
    "clock": "clock",
    "rng": "rng",
    "service_image_config_corpus": "service_image",
    "explicit_environment_projection": "environment_projection",
}
FAMILY_DEFAULT_REUSE_CLASS: Final[dict[str, str]] = {
    "immutable_scalar": "pure",
    "frozen_canonical_record": "deterministic_snapshot",
    "path_independent_temp_directory_root": "transactional",
    "transaction_baseline": "transactional",
    "clock": "pure",
    "rng": "pure",
    "service_image_config_corpus": "idempotent_external",
    "explicit_environment_projection": "deterministic_snapshot",
}
DI_HANDLE_KINDS: Final[tuple[str, ...]] = ("lookup", "store", "provider", "issuer")
COMMITTED_PROVIDER_FIELDS: Final[tuple[str, ...]] = (
    "provider_identity",
    "provider_version",
    "epoch",
    "policy_cid",
)
FORBIDDEN_COMMIT_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "raw_value",
        "refresh_token",
        "secret",
        "secret_bytes",
        "session_token",
        "value_bytes",
        "witness",
    }
)
_PRIVATE_SUBSTRINGS: Final[tuple[str, ...]] = (
    "private",
    "secret",
    "password",
    "api_key",
    "witness",
    "credential",
    "token",
)
_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "baseline_cid",
        "commitment_cid",
        "config_cid",
        "corpus_cid",
        "image_cid",
        "policy_cid",
        "schema_cid",
        "seed_digest",
        "snapshot_cid",
    }
)
_TYPED_UNAVAILABLE: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "test_execution_key_v2",
        "test_execution_key_v2_not_implemented",
        "reviewed adapters commit privacy-safe identities only; "
        "TestExecutionKeyV2 remains a later datasets contract",
    ),
    (
        "pre_call_fixture_instance_key",
        "fixture_instance_authoritative_only_after_current_setup",
        "adapter commitments become safely authoritative only after current "
        "setup; they do not mint a pre-call fixture-instance key",
    ),
    (
        "composite_phase_receipt",
        "composite_phase_receipt_not_implemented",
        "composite phase receipts are outside adapter execution and remain "
        "typed unavailable without changing TestPassStatementV1",
    ),
    (
        "production_zk",
        "production_zk_key_ceremony_unavailable",
        "production ZK proving remains typed unavailable; adapter integrity "
        "commitments cannot admit simulated or self-verified proofs",
    ),
    (
        "key_ceremony",
        "production_zk_key_ceremony_unavailable",
        "no production-eligible key ceremony is admitted by dependency "
        "commitment adapters",
    ),
    (
        "direct_execution_profile",
        "direct_execution_profile_optional",
        "direct CPython execution profiles remain optional and unadmitted; "
        "they cannot upgrade adapter integrity commitments",
    ),
)


class AdapterCommitError(ValueError):
    """Raised when adapter classification or commitment is unsafe."""

    __test__ = False


class AdapterExecutionDisposition(str, Enum):
    FULL_EXECUTION = "full_execution"
    COMMITTED = "committed"


def _is_private_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in FORBIDDEN_COMMIT_FIELD_MARKERS:
        return True
    return any(marker in lowered for marker in _PRIVATE_SUBSTRINGS)


def _name_looks_private(name: str) -> bool:
    return _is_private_key(str(name or ""))


def _reject_private_and_nonfinite(value: Any, field_name: str) -> None:
    if isinstance(value, float):
        raise AdapterCommitError(
            f"{field_name} must not contain floating-point or nonfinite values"
        )
    if isinstance(value, Mapping):
        if len(value) > MAX_EXTRA_KEYS:
            raise AdapterCommitError(
                f"{field_name} exceeds bounded key count of {MAX_EXTRA_KEYS}"
            )
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise AdapterCommitError(f"{field_name} keys must be non-empty strings")
            if _is_private_key(key):
                raise AdapterCommitError(
                    f"{field_name} rejects private material key {key!r}"
                )
            _reject_private_and_nonfinite(item, field_name=f"{field_name}.{key}")
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise AdapterCommitError(f"{field_name} rejects secret or raw bytes")
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise AdapterCommitError(
                f"{field_name} exceeds bounded item count of {MAX_SEQUENCE_ITEMS}"
            )
        for index, item in enumerate(value):
            _reject_private_and_nonfinite(item, field_name=f"{field_name}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > MAX_TEXT_CHARS:
            raise AdapterCommitError(
                f"{field_name} exceeds bounded length of {MAX_TEXT_CHARS}"
            )
        return
    raise AdapterCommitError(
        f"{field_name} has unsupported value type {type(value).__name__}"
    )


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise AdapterCommitError("nonfinite values are not JSON-safe")
        raise AdapterCommitError("floating-point values are not JSON-safe")
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_ready(to_dict())
    raise AdapterCommitError(
        f"value of type {type(value).__name__} is not JSON-serializable"
    )


def canonical_public_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes for a privacy-safe public payload."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def public_digest(value: Any) -> str:
    """Return ``sha256:<hex>`` of canonical public bytes."""

    return _DIGEST_PREFIX + hashlib.sha256(canonical_public_bytes(value)).hexdigest()


ADAPTER_POLICY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "interface": ADAPTER_POLICY_INTERFACE,
        "adapter_interface": REVIEWED_DEPENDENCY_COMMITMENT_ADAPTER_INTERFACE,
        "secret_rule": PRIVACY_SECRET_RULE,
        "default_reuse_class": DEFAULT_REUSE_CLASS,
        "may_authorize_skip": False,
        "production_admitted": False,
        "authoritative_after": AUTHORITATIVE_AFTER,
        "review_candidate_families": list(REVIEW_CANDIDATE_FAMILIES),
        "closed_reuse_classes": list(CLOSED_REUSE_CLASSES),
    }
)
DEFAULT_POLICY_CID: Final[str] = public_digest(dict(ADAPTER_POLICY))


def classify_reuse_class(value: Any) -> str:
    """Map missing or unknown reuse labels onto the closed default ``opaque``."""

    if value is None:
        return DEFAULT_REUSE_CLASS
    if not isinstance(value, str):
        raise AdapterCommitError("reuse_class must be a string or omitted")
    text = value.strip()
    if not text or text not in CLOSED_REUSE_CLASSES:
        return DEFAULT_REUSE_CLASS
    return text


def _require_text(
    value: Any,
    field_name: str,
    *,
    max_chars: int = MAX_TEXT_CHARS,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise AdapterCommitError(f"{field_name} must be a string")
    if value != value.strip():
        raise AdapterCommitError(f"{field_name} must be a trimmed string")
    if not allow_empty and not value:
        raise AdapterCommitError(f"{field_name} must be a non-empty string")
    if len(value) > max_chars:
        raise AdapterCommitError(
            f"{field_name} exceeds bounded length of {max_chars} characters"
        )
    return value


def _require_kind(value: Any) -> str:
    text = _require_text(value, "kind", max_chars=64)
    return text


def _require_adapter_id(adapter_id: Any, reviewed: bool) -> str:
    adapter = _require_text(
        adapter_id if adapter_id is not None else "",
        "adapter_id",
        max_chars=MAX_NAME_CHARS,
        allow_empty=True,
    )
    if reviewed and not adapter:
        raise AdapterCommitError(
            "reviewed adapters must name a closed review-candidate family"
        )
    if adapter and adapter not in REVIEW_CANDIDATE_FAMILIES:
        raise AdapterCommitError(
            "adapter_id must be empty or a closed review-candidate family: "
            + ", ".join(REVIEW_CANDIDATE_FAMILIES)
        )
    if reviewed and adapter not in REVIEW_CANDIDATE_FAMILIES:
        raise AdapterCommitError(
            "reviewed adapter_id must be a closed review-candidate family"
        )
    return adapter


def _is_json_scalar(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    if isinstance(value, str) and len(value) <= MAX_TEXT_CHARS:
        return True
    return False


def _is_identity_string(value: Any) -> bool:
    if not isinstance(value, str) or not value or len(value) > MAX_TEXT_CHARS:
        return False
    if value.startswith(_DIGEST_PREFIX) and len(value) == len(_DIGEST_PREFIX) + 64:
        hex_part = value[len(_DIGEST_PREFIX) :]
        try:
            int(hex_part, 16)
        except ValueError:
            return False
        return hex_part == hex_part.lower()
    if value.startswith(("cid:", "bafy", "epoch:")):
        return True
    return False


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    return None


def _public_mapping(value: Any) -> dict[str, Any] | None:
    mapping = _mapping_or_none(value)
    if mapping is None:
        return None
    try:
        _reject_private_and_nonfinite(mapping, "value")
        ready = _json_ready(mapping)
    except AdapterCommitError:
        return None
    if not isinstance(ready, dict):
        return None
    return ready


def _value_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, str):
        return "str"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "sequence"
    return type(value).__name__


def _supported_immutable_scalar(value: Any) -> dict[str, Any] | None:
    if callable(value) or isinstance(value, (bytes, bytearray, memoryview, float)):
        return None
    if not _is_json_scalar(value):
        return None
    return {
        "family": "immutable_scalar",
        "value_type": _value_type_name(value),
        "canonical": _json_ready(value),
    }


def _supported_frozen_record(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            return None
        if not all(_is_json_scalar(item) for item in value):
            return None
        try:
            canonical = _json_ready(list(value))
        except AdapterCommitError:
            return None
        return {
            "family": "frozen_canonical_record",
            "value_type": "sequence",
            "length": len(value),
            "canonical": canonical,
        }
    mapping = _public_mapping(value)
    if mapping is None:
        return None
    return {
        "family": "frozen_canonical_record",
        "value_type": "mapping",
        "public_keys": sorted(mapping),
        "canonical": mapping,
    }


def _supported_temp_root(value: Any) -> dict[str, Any] | None:
    mapping = _public_mapping(value)
    if mapping is None:
        return None
    for banned in ("path", "abspath", "absolute_path", "tmp_path", "dir"):
        raw = mapping.get(banned)
        if not isinstance(raw, str) or not raw:
            continue
        if raw.startswith(("/", "\\")) or (len(raw) >= 2 and raw[1] == ":"):
            return None
    provider = mapping.get("provider_identity")
    version = mapping.get("provider_version")
    if not isinstance(provider, str) or not provider:
        return None
    if not isinstance(version, str) or not version:
        return None
    layout = mapping.get("layout", "relative")
    if layout != "relative":
        return None
    return {
        "family": "path_independent_temp_directory_root",
        "provider_identity": provider,
        "provider_version": version,
        "layout": "relative",
        "canonical": {
            key: mapping[key]
            for key in (
                "provider_identity",
                "provider_version",
                "layout",
                "policy_cid",
                "epoch",
            )
            if key in mapping
        },
    }


def _supported_transaction_baseline(value: Any) -> dict[str, Any] | None:
    mapping = _public_mapping(value)
    if mapping is None:
        return None
    identity_keys = [
        key
        for key in ("baseline_cid", "snapshot_cid", "schema_cid")
        if _is_identity_string(mapping.get(key))
    ]
    if not identity_keys:
        return None
    return {
        "family": "transaction_baseline",
        "identity_keys": identity_keys,
        "canonical": {key: mapping[key] for key in identity_keys},
    }


def _supported_clock(value: Any) -> dict[str, Any] | None:
    if callable(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return {
            "family": "clock",
            "value_type": "epoch_int",
            "canonical": {"epoch_int": value},
        }
    if isinstance(value, str) and (
        value.startswith("epoch:") or _is_identity_string(value)
    ):
        return {
            "family": "clock",
            "value_type": "epoch_id",
            "canonical": {"epoch_id": value},
        }
    mapping = _public_mapping(value)
    if mapping is None:
        return None
    provider = mapping.get("provider_identity")
    if not isinstance(provider, str) or not provider:
        return None
    canonical = {
        key: mapping[key]
        for key in (
            "provider_identity",
            "provider_version",
            "epoch",
            "policy_cid",
            "clock_epoch",
        )
        if key in mapping
    }
    if "epoch" not in canonical and "clock_epoch" not in canonical:
        if not _is_identity_string(mapping.get("policy_cid")):
            return None
    return {
        "family": "clock",
        "value_type": "mapping",
        "canonical": canonical,
    }


def _supported_rng(value: Any) -> dict[str, Any] | None:
    if callable(value) or isinstance(value, (bytes, bytearray, memoryview)):
        return None
    mapping = _public_mapping(value)
    if mapping is None:
        return None
    if "seed" in mapping:
        seed = mapping["seed"]
        if isinstance(seed, (bytes, bytearray, memoryview)):
            return None
        if not _is_json_scalar(seed):
            return None
        mapping = dict(mapping)
        mapping.pop("seed")
        mapping["seed_digest"] = public_digest({"seed": _json_ready(seed)})
    digest = mapping.get("seed_digest")
    algorithm = mapping.get("algorithm")
    provider = mapping.get("provider_identity")
    if not _is_identity_string(digest) and not (
        isinstance(provider, str) and provider and isinstance(algorithm, str) and algorithm
    ):
        return None
    canonical = {
        key: mapping[key]
        for key in (
            "provider_identity",
            "provider_version",
            "algorithm",
            "seed_digest",
            "epoch",
            "policy_cid",
        )
        if key in mapping
    }
    return {"family": "rng", "canonical": canonical}


def _supported_service_image(value: Any) -> dict[str, Any] | None:
    mapping = _public_mapping(value)
    if mapping is None:
        return None
    identity_keys = [
        key
        for key in ("image_cid", "config_cid", "corpus_cid")
        if _is_identity_string(mapping.get(key))
    ]
    if not identity_keys:
        return None
    return {
        "family": "service_image_config_corpus",
        "identity_keys": identity_keys,
        "canonical": {key: mapping[key] for key in identity_keys},
    }


def _supported_environment_projection(value: Any) -> dict[str, Any] | None:
    mapping = _public_mapping(value)
    if mapping is None:
        return None
    projected: dict[str, str] = {}
    for key, item in mapping.items():
        if _is_private_key(key):
            return None
        if _is_identity_string(item):
            projected[key] = item
            continue
        if _is_json_scalar(item):
            projected[key] = public_digest({"env": key, "value": _json_ready(item)})
            continue
        return None
    return {
        "family": "explicit_environment_projection",
        "public_keys": sorted(projected),
        "canonical": projected,
    }


_FAMILY_EXTRACTORS: Final[dict[str, Callable[[Any], dict[str, Any] | None]]] = {
    "immutable_scalar": _supported_immutable_scalar,
    "frozen_canonical_record": _supported_frozen_record,
    "path_independent_temp_directory_root": _supported_temp_root,
    "transaction_baseline": _supported_transaction_baseline,
    "clock": _supported_clock,
    "rng": _supported_rng,
    "service_image_config_corpus": _supported_service_image,
    "explicit_environment_projection": _supported_environment_projection,
}


def adapter_supports_value(adapter_id: str, value: Any) -> bool:
    """Return whether *adapter_id* can commit *value* without secret bytes."""

    extractor = _FAMILY_EXTRACTORS.get(adapter_id)
    if extractor is None:
        return False
    try:
        return extractor(value) is not None
    except AdapterCommitError:
        return False


def extract_public_identity(adapter_id: str, value: Any) -> dict[str, Any] | None:
    """Return the privacy-safe public identity payload, or ``None``."""

    extractor = _FAMILY_EXTRACTORS.get(adapter_id)
    if extractor is None:
        return None
    try:
        return extractor(value)
    except AdapterCommitError:
        return None


def _load_datasets_contracts() -> Any | None:
    try:
        from ipfs_datasets_py.logic.zkp.pctdd import fixture_instance_contracts as contracts
    except Exception:
        return None
    return contracts


def datasets_contracts_available() -> bool:
    """Return whether datasets-owned ``@1`` commitment codecs import."""

    return _load_datasets_contracts() is not None


def record_typed_unavailable(
    *,
    capability: str,
    reason_code: str,
    message: str,
) -> dict[str, Any]:
    """Record a typed unavailable case without changing claim meaning."""

    record = {
        "capability": capability,
        "reason_code": reason_code,
        "message": message,
        "status": "typed_unavailable",
        "production_admitted": False,
        "claim_unchanged": True,
        "self_approved": False,
    }
    if record["production_admitted"] or record["self_approved"] or not record["claim_unchanged"]:
        raise AdapterCommitError(
            "typed unavailable cases cannot admit, self-approve, or change claims"
        )
    return record


def typed_unavailable_records() -> tuple[dict[str, Any], ...]:
    """Closed set of PCTDD-024 typed unavailable capabilities."""

    records = [
        record_typed_unavailable(
            capability=capability,
            reason_code=reason_code,
            message=message,
        )
        for capability, reason_code, message in _TYPED_UNAVAILABLE
    ]
    if not datasets_contracts_available():
        records.append(
            record_typed_unavailable(
                capability="datasets_fixture_instance_contracts",
                reason_code="datasets_fixture_instance_contracts_unresolved",
                message=(
                    "datasets InjectedDependencyCommitment@1 codecs did not import "
                    "in this sealed environment; adapters force full execution "
                    "and do not commit values"
                ),
            )
        )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class AdapterCommitResult:
    """Outcome of one reviewed or opaque adapter attempt.

    ``committed`` is true only when a reviewed adapter bound a supported
    value.  Opaque and unsupported cases execute fully.  Not a pytest test
    class.
    """

    __test__ = False

    adapter_id: str
    reviewed: bool
    committed: bool
    execution_disposition: str
    full_execution_reasons: tuple[str, ...]
    kind: str
    name: str
    commitment: Any
    payload: Mapping[str, Any]
    may_authorize_skip: bool = False
    production_admitted: bool = False
    claim_unchanged: bool = True
    self_approved: bool = False
    claim_class: str = CLAIM_CLASS

    def __post_init__(self) -> None:
        if self.may_authorize_skip:
            raise AdapterCommitError("dependency adapters must not authorize skip")
        if self.production_admitted or self.self_approved or not self.claim_unchanged:
            raise AdapterCommitError(
                "adapters cannot admit production, self-approve, or change claims"
            )
        if self.committed and self.execution_disposition != (
            AdapterExecutionDisposition.COMMITTED.value
        ):
            raise AdapterCommitError("committed results must use committed disposition")
        if (not self.committed) and self.execution_disposition != (
            AdapterExecutionDisposition.FULL_EXECUTION.value
        ):
            raise AdapterCommitError(
                "non-committed results must execute fully"
            )
        if self.committed and not self.reviewed:
            raise AdapterCommitError("only reviewed adapters may commit values")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        object.__setattr__(
            self, "full_execution_reasons", tuple(self.full_execution_reasons)
        )

    @property
    def interface(self) -> str:
        return ADAPTER_COMMIT_RESULT_INTERFACE

    @property
    def schema(self) -> str:
        return ADAPTER_COMMIT_RESULT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        commitment_payload: dict[str, Any] | None
        if self.commitment is None:
            commitment_payload = None
        elif hasattr(self.commitment, "to_dict"):
            commitment_payload = self.commitment.to_dict()
        else:
            commitment_payload = dict(self.payload)
        return {
            "schema": self.schema,
            "interface": self.interface,
            "adapter_id": self.adapter_id,
            "reviewed": self.reviewed,
            "committed": self.committed,
            "execution_disposition": self.execution_disposition,
            "full_execution_reasons": list(self.full_execution_reasons),
            "kind": self.kind,
            "name": self.name,
            "may_authorize_skip": False,
            "production_admitted": False,
            "claim_unchanged": True,
            "self_approved": False,
            "claim_class": self.claim_class,
            "commitment": commitment_payload,
        }


def _opaque_reasons(
    *,
    reuse_class: str,
    completeness: str,
    reviewed: bool,
    supported: bool,
    identity_complete: bool,
    commitment_cid: str,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if reuse_class in FULL_EXECUTION_REUSE_CLASSES:
        reasons.append(f"reuse_class_{reuse_class}")
    if completeness != "exact":
        reasons.append(f"completeness_{completeness}")
    if not reviewed:
        reasons.append("adapter_unreviewed")
    if not supported:
        reasons.append("adapter_value_unsupported")
    if not identity_complete:
        reasons.append("injected_identity_incomplete")
    if reviewed and completeness == "exact" and reuse_class in COMMIT_ELIGIBLE_REUSE_CLASSES:
        if not commitment_cid:
            reasons.append("commitment_cid_missing")
    return tuple(reasons)


def _build_injected_commitment(
    *,
    kind: str,
    name: str,
    provider_identity: str,
    provider_version: str,
    epoch: str,
    policy_cid: str,
    commitment_cid: str,
    reuse_class: str,
    completeness: str,
    adapter_id: str,
    reviewed: bool,
    extra: Mapping[str, Any],
) -> Any:
    contracts = _load_datasets_contracts()
    if contracts is None:
        return None
    return contracts.build_injected_dependency_commitment(
        kind=kind,
        name=name,
        provider_identity=provider_identity,
        provider_version=provider_version,
        epoch=epoch,
        policy_cid=policy_cid,
        commitment_cid=commitment_cid,
        reuse_class=reuse_class,
        completeness=completeness,
        adapter_id=adapter_id,
        reviewed=reviewed,
        extra=extra,
    )


def _build_fixture_commitment(
    *,
    fixture_name: str,
    fixture_scope: str,
    definition_cid: str,
    instance_commitment_cid: str,
    reuse_class: str,
    completeness: str,
    adapter_id: str,
    reviewed: bool,
    extra: Mapping[str, Any],
) -> Any:
    contracts = _load_datasets_contracts()
    if contracts is None:
        return None
    return contracts.build_fixture_instance_commitment(
        fixture_name=fixture_name,
        fixture_scope=fixture_scope,
        definition_cid=definition_cid,
        instance_commitment_cid=instance_commitment_cid,
        reuse_class=reuse_class,
        completeness=completeness,
        adapter_id=adapter_id,
        reviewed=reviewed,
        extra=extra,
    )


def commit_opaque_dependency(
    *,
    kind: str,
    name: str,
    reason_code: str = "reuse_class_opaque",
    extra: Mapping[str, Any] | None = None,
) -> AdapterCommitResult:
    """Force full execution for an opaque or unsupported dependency."""

    kind_text = _require_kind(kind)
    name_text = _require_text(name, "name", max_chars=MAX_NAME_CHARS)
    if extra:
        _reject_private_and_nonfinite(extra, "extra")
    reasons = (reason_code, "adapter_unreviewed", "completeness_unknown")
    extra_payload = {"reason_code": reason_code}
    if extra:
        extra_payload.update(_json_ready(dict(extra)))
    commitment = _build_injected_commitment(
        kind=kind_text,
        name=name_text,
        provider_identity="",
        provider_version="",
        epoch="",
        policy_cid="",
        commitment_cid="",
        reuse_class=DEFAULT_REUSE_CLASS,
        completeness=DEFAULT_COMPLETENESS,
        adapter_id="",
        reviewed=False,
        extra=extra_payload,
    )
    payload = (
        commitment.to_dict()
        if commitment is not None
        else {
            "schema": "pctdd/injected-dependency-commitment@1",
            "interface": "InjectedDependencyCommitment@1",
            "kind": kind_text,
            "name": name_text,
            "adapter_id": "",
            "reviewed": False,
            "reuse_class": DEFAULT_REUSE_CLASS,
            "completeness": DEFAULT_COMPLETENESS,
            "execution_disposition": AdapterExecutionDisposition.FULL_EXECUTION.value,
            "full_execution_reasons": list(reasons),
            "may_authorize_skip": False,
            "production_admitted": False,
            "claim_class": CLAIM_CLASS,
        }
    )
    return AdapterCommitResult(
        adapter_id="",
        reviewed=False,
        committed=False,
        execution_disposition=AdapterExecutionDisposition.FULL_EXECUTION.value,
        full_execution_reasons=reasons,
        kind=kind_text,
        name=name_text,
        commitment=commitment,
        payload=payload,
    )


def _provider_fields(
    *,
    public_identity: Mapping[str, Any] | None,
    provider_identity: str,
    provider_version: str,
    epoch: str,
    policy_cid: str,
    adapter_id: str,
) -> tuple[str, str, str, str]:
    identity_map = public_identity or {}
    canonical = identity_map.get("canonical")
    nested = canonical if isinstance(canonical, Mapping) else identity_map
    resolved_identity = provider_identity or str(
        nested.get("provider_identity") or DEFAULT_PROVIDER_IDENTITY
    )
    resolved_version = provider_version or str(
        nested.get("provider_version")
        or REVIEWED_DEPENDENCY_COMMITMENT_ADAPTER_INTERFACE
    )
    resolved_epoch = epoch or str(nested.get("epoch") or DEFAULT_EPOCH)
    resolved_policy = policy_cid or str(nested.get("policy_cid") or DEFAULT_POLICY_CID)
    if adapter_id and not provider_identity:
        resolved_identity = f"{DEFAULT_PROVIDER_IDENTITY}:{adapter_id}"
    return resolved_identity, resolved_version, resolved_epoch, resolved_policy


def commit_injected_dependency(
    *,
    kind: str,
    name: str,
    value: Any = None,
    adapter_id: str = "",
    reviewed: bool | None = None,
    provider_identity: str = "",
    provider_version: str = "",
    epoch: str = "",
    policy_cid: str = "",
    reuse_class: Any = None,
    completeness: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> AdapterCommitResult:
    """Commit a supported injected dependency or force full execution.

    Unreviewed, opaque, incomplete, unknown, or unsupported values never
    commit.  Reviewed families commit only privacy-safe public identity.
    """

    kind_text = _require_kind(kind)
    name_text = _require_text(name, "name", max_chars=MAX_NAME_CHARS)
    if _name_looks_private(name_text):
        return commit_opaque_dependency(
            kind=kind_text,
            name=name_text,
            reason_code="private_dependency_name",
        )
    requested_reviewed = bool(reviewed) if reviewed is not None else False
    adapter = _require_adapter_id(adapter_id, requested_reviewed)
    if extra:
        _reject_private_and_nonfinite(extra, "extra")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise AdapterCommitError("value rejects secret or raw bytes")
    mapping = _mapping_or_none(value)
    if mapping is not None:
        _reject_private_and_nonfinite(mapping, "value")

    public_identity = extract_public_identity(adapter, value) if adapter else None
    supported = public_identity is not None
    effective_reviewed = bool(adapter) and supported and (
        requested_reviewed or reviewed is None
    )
    if reviewed is False:
        effective_reviewed = False
    if not adapter or not effective_reviewed or not supported:
        if adapter and not supported:
            reason_code = "adapter_value_unsupported"
        else:
            reason_code = "adapter_unreviewed"
        return commit_opaque_dependency(
            kind=kind_text,
            name=name_text,
            reason_code=reason_code,
        )

    default_reuse = FAMILY_DEFAULT_REUSE_CLASS[adapter]
    classified_reuse = classify_reuse_class(
        reuse_class if reuse_class is not None else default_reuse
    )
    if classified_reuse in FULL_EXECUTION_REUSE_CLASSES:
        return commit_opaque_dependency(
            kind=kind_text,
            name=name_text,
            reason_code=f"reuse_class_{classified_reuse}",
        )
    resolved_reuse = (
        classified_reuse if classified_reuse in COMMIT_ELIGIBLE_REUSE_CLASSES else default_reuse
    )
    resolved_completeness = "exact" if completeness is None else str(completeness).strip()
    if resolved_completeness != "exact":
        return commit_opaque_dependency(
            kind=kind_text,
            name=name_text,
            reason_code=f"completeness_{resolved_completeness or DEFAULT_COMPLETENESS}",
        )

    resolved_kind = kind_text or FAMILY_DEFAULT_KIND[adapter]
    identity, version, resolved_epoch, resolved_policy = _provider_fields(
        public_identity=public_identity,
        provider_identity=provider_identity,
        provider_version=provider_version,
        epoch=epoch,
        policy_cid=policy_cid,
        adapter_id=adapter,
    )
    identity_complete = bool(identity and version and resolved_epoch and resolved_policy)
    if not identity_complete:
        return commit_opaque_dependency(
            kind=resolved_kind,
            name=name_text,
            reason_code="injected_identity_incomplete",
        )

    extra_payload = {
        "value_type": public_identity.get("value_type", _value_type_name(value)),
        "family": adapter,
        "public_keys": public_identity.get("public_keys", []),
        "identity_keys": public_identity.get("identity_keys", []),
    }
    if extra:
        extra_payload.update(_json_ready(dict(extra)))
    _reject_private_and_nonfinite(extra_payload, "extra")
    digest_payload = {
        "adapter_id": adapter,
        "kind": resolved_kind,
        "name": name_text,
        "provider_identity": identity,
        "provider_version": version,
        "epoch": resolved_epoch,
        "policy_cid": resolved_policy,
        "canonical": public_identity.get("canonical"),
    }
    commitment_cid = public_digest(digest_payload)
    reasons = _opaque_reasons(
        reuse_class=resolved_reuse,
        completeness=resolved_completeness,
        reviewed=True,
        supported=True,
        identity_complete=True,
        commitment_cid=commitment_cid,
    )
    if reasons:
        return commit_opaque_dependency(
            kind=resolved_kind,
            name=name_text,
            reason_code=reasons[0],
        )
    commitment = _build_injected_commitment(
        kind=resolved_kind,
        name=name_text,
        provider_identity=identity,
        provider_version=version,
        epoch=resolved_epoch,
        policy_cid=resolved_policy,
        commitment_cid=commitment_cid,
        reuse_class=resolved_reuse,
        completeness=resolved_completeness,
        adapter_id=adapter,
        reviewed=True,
        extra=extra_payload,
    )
    if commitment is None:
        return commit_opaque_dependency(
            kind=resolved_kind,
            name=name_text,
            reason_code="datasets_fixture_instance_contracts_unresolved",
        )
    payload = commitment.to_dict()
    if payload.get("execution_disposition") != AdapterExecutionDisposition.COMMITTED.value:
        return AdapterCommitResult(
            adapter_id=adapter,
            reviewed=True,
            committed=False,
            execution_disposition=AdapterExecutionDisposition.FULL_EXECUTION.value,
            full_execution_reasons=tuple(payload.get("full_execution_reasons") or ("commitment_rejected",)),
            kind=resolved_kind,
            name=name_text,
            commitment=commitment,
            payload=payload,
        )
    return AdapterCommitResult(
        adapter_id=adapter,
        reviewed=True,
        committed=True,
        execution_disposition=AdapterExecutionDisposition.COMMITTED.value,
        full_execution_reasons=(),
        kind=resolved_kind,
        name=name_text,
        commitment=commitment,
        payload=payload,
    )


def commit_fixture_instance(
    *,
    fixture_name: str,
    value: Any = None,
    fixture_scope: str = "function",
    adapter_id: str = "",
    reviewed: bool | None = None,
    definition_cid: str = "",
    reuse_class: Any = None,
    completeness: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> AdapterCommitResult:
    """Commit a supported fixture-instance value or force full execution."""

    name_text = _require_text(fixture_name, "fixture_name", max_chars=MAX_NAME_CHARS)
    injected = commit_injected_dependency(
        kind=FAMILY_DEFAULT_KIND.get(adapter_id, "pytest_fixture"),
        name=name_text,
        value=value,
        adapter_id=adapter_id,
        reviewed=reviewed,
        reuse_class=reuse_class,
        completeness=completeness,
        extra=extra,
    )
    if not injected.committed:
        contracts = _load_datasets_contracts()
        fixture = None
        if contracts is not None:
            fixture = contracts.build_fixture_instance_commitment(
                fixture_name=name_text,
                fixture_scope=fixture_scope,
                definition_cid=definition_cid,
                instance_commitment_cid="",
                reuse_class=DEFAULT_REUSE_CLASS,
                completeness=DEFAULT_COMPLETENESS,
                adapter_id="",
                reviewed=False,
                extra={"reason_code": injected.full_execution_reasons[0] if injected.full_execution_reasons else "opaque"},
            )
        payload = fixture.to_dict() if fixture is not None else dict(injected.payload)
        return AdapterCommitResult(
            adapter_id="",
            reviewed=False,
            committed=False,
            execution_disposition=AdapterExecutionDisposition.FULL_EXECUTION.value,
            full_execution_reasons=injected.full_execution_reasons,
            kind="pytest_fixture",
            name=name_text,
            commitment=fixture if fixture is not None else injected.commitment,
            payload=payload,
        )

    contracts = _load_datasets_contracts()
    if contracts is None:
        return injected
    extra_payload = {
        "value_type": injected.payload.get("extra", {}).get("value_type", ""),
        "family": injected.adapter_id,
    }
    fixture = contracts.build_fixture_instance_commitment(
        fixture_name=name_text,
        fixture_scope=fixture_scope,
        definition_cid=definition_cid,
        instance_commitment_cid=injected.payload.get("commitment_cid", ""),
        reuse_class=injected.payload.get("reuse_class", FAMILY_DEFAULT_REUSE_CLASS[injected.adapter_id]),
        completeness="exact",
        adapter_id=injected.adapter_id,
        reviewed=True,
        extra=extra_payload,
    )
    payload = fixture.to_dict()
    committed = payload.get("execution_disposition") == AdapterExecutionDisposition.COMMITTED.value
    return AdapterCommitResult(
        adapter_id=injected.adapter_id,
        reviewed=True,
        committed=committed,
        execution_disposition=(
            AdapterExecutionDisposition.COMMITTED.value
            if committed
            else AdapterExecutionDisposition.FULL_EXECUTION.value
        ),
        full_execution_reasons=tuple(payload.get("full_execution_reasons") or ()),
        kind="pytest_fixture",
        name=name_text,
        commitment=fixture,
        payload=payload,
    )


def commit_di_handle(
    *,
    kind: str,
    handle: Any,
    name: str = "",
) -> AdapterCommitResult:
    """Commit lookup/store/provider/issuer identity when publicly complete."""

    kind_text = _require_kind(kind)
    if kind_text not in DI_HANDLE_KINDS:
        raise AdapterCommitError(
            "DI handle kind must be one of " + ", ".join(DI_HANDLE_KINDS)
        )
    handle_name = name or kind_text
    if handle is None:
        return commit_opaque_dependency(
            kind=kind_text,
            name=handle_name,
            reason_code="di_handle_missing",
        )
    interface = getattr(handle, "interface", None)
    if not isinstance(interface, str) or not interface.strip():
        return commit_opaque_dependency(
            kind=kind_text,
            name=handle_name,
            reason_code="di_handle_identity_incomplete",
        )
    record = {
        "provider_identity": f"{DEFAULT_PROVIDER_IDENTITY}.{kind_text}",
        "provider_version": interface.strip(),
        "kind": kind_text,
        "handle_class": type(handle).__name__,
    }
    return commit_injected_dependency(
        kind=kind_text,
        name=handle_name,
        value=record,
        adapter_id="frozen_canonical_record",
        reviewed=True,
        provider_identity=record["provider_identity"],
        provider_version=record["provider_version"],
    )


def commit_di_services(
    *,
    lookup: Any = None,
    store: Any = None,
    provider: Any = None,
    issuer: Any = None,
) -> tuple[AdapterCommitResult, ...]:
    """Commit the closed proof-reuse DI handle set.  Missing handles run fully."""

    return (
        commit_di_handle(kind="lookup", handle=lookup),
        commit_di_handle(kind="store", handle=store),
        commit_di_handle(kind="provider", handle=provider),
        commit_di_handle(kind="issuer", handle=issuer),
    )


def requires_full_execution(results: Iterable[AdapterCommitResult]) -> bool:
    """Any opaque, unreviewed, or unsupported member forces full execution."""

    items = tuple(results)
    if not items:
        return True
    return any(not item.committed for item in items)


def build_committed_closure(
    results: Iterable[AdapterCommitResult],
    *,
    locator_cid: str = "",
) -> Any:
    """Assemble a datasets fixture-instance closure from adapter results.

    Does not mint ``TestExecutionKeyV2``.  Opaque members keep the closure on
    full execution.
    """

    contracts = _load_datasets_contracts()
    if contracts is None:
        return None
    injected = []
    instances = []
    for item in results:
        commitment = item.commitment
        if commitment is None:
            continue
        type_name = type(commitment).__name__
        if type_name == "FixtureInstanceCommitment":
            instances.append(commitment)
        else:
            injected.append(commitment)
    return contracts.build_fixture_instance_closure(
        instances=instances,
        injected_dependencies=injected,
        locator_cid=locator_cid,
    )


@dataclass(frozen=True, slots=True)
class ReviewedDependencyCommitmentAdapter:
    """One closed review-candidate family.

    Not a pytest test class.
    """

    __test__ = False

    adapter_id: str
    inventory_label: str
    default_kind: str
    default_reuse_class: str

    def __post_init__(self) -> None:
        if self.adapter_id not in REVIEW_CANDIDATE_FAMILIES:
            raise AdapterCommitError(
                "adapter_id must be a closed review-candidate family"
            )
        if self.inventory_label != REVIEW_CANDIDATE_FAMILY_TO_INVENTORY[self.adapter_id]:
            raise AdapterCommitError("inventory label drift for reviewed adapter")
        if self.default_kind != FAMILY_DEFAULT_KIND[self.adapter_id]:
            raise AdapterCommitError("default kind drift for reviewed adapter")
        if self.default_reuse_class != FAMILY_DEFAULT_REUSE_CLASS[self.adapter_id]:
            raise AdapterCommitError("default reuse-class drift for reviewed adapter")

    @property
    def interface(self) -> str:
        return REVIEWED_DEPENDENCY_COMMITMENT_ADAPTER_INTERFACE

    @property
    def reviewed(self) -> bool:
        return True

    def supports(self, value: Any) -> bool:
        return adapter_supports_value(self.adapter_id, value)

    def commit(
        self,
        *,
        kind: str | None = None,
        name: str,
        value: Any,
        **kwargs: Any,
    ) -> AdapterCommitResult:
        return commit_injected_dependency(
            kind=kind or self.default_kind,
            name=name,
            value=value,
            adapter_id=self.adapter_id,
            reviewed=True,
            reuse_class=kwargs.get("reuse_class", self.default_reuse_class),
            completeness=kwargs.get("completeness"),
            provider_identity=kwargs.get("provider_identity", ""),
            provider_version=kwargs.get("provider_version", ""),
            epoch=kwargs.get("epoch", ""),
            policy_cid=kwargs.get("policy_cid", ""),
            extra=kwargs.get("extra"),
        )


class DependencyCommitmentAdapterRegistry:
    """Closed registry: only reviewed families commit supported values."""

    __test__ = False
    interface: str = DEPENDENCY_COMMITMENT_ADAPTER_REGISTRY_INTERFACE

    def __init__(self) -> None:
        self._adapters = {
            family: ReviewedDependencyCommitmentAdapter(
                adapter_id=family,
                inventory_label=REVIEW_CANDIDATE_FAMILY_TO_INVENTORY[family],
                default_kind=FAMILY_DEFAULT_KIND[family],
                default_reuse_class=FAMILY_DEFAULT_REUSE_CLASS[family],
            )
            for family in REVIEW_CANDIDATE_FAMILIES
        }

    def families(self) -> tuple[str, ...]:
        return REVIEW_CANDIDATE_FAMILIES

    def get(self, adapter_id: str) -> ReviewedDependencyCommitmentAdapter | None:
        return self._adapters.get(adapter_id)

    def require(self, adapter_id: str) -> ReviewedDependencyCommitmentAdapter:
        adapter = self.get(adapter_id)
        if adapter is None:
            raise AdapterCommitError(
                "adapter_id must be a closed review-candidate family"
            )
        return adapter

    def commit(
        self,
        *,
        adapter_id: str = "",
        kind: str,
        name: str,
        value: Any = None,
        reviewed: bool | None = None,
        **kwargs: Any,
    ) -> AdapterCommitResult:
        if not adapter_id:
            return commit_opaque_dependency(kind=kind, name=name)
        return commit_injected_dependency(
            kind=kind,
            name=name,
            value=value,
            adapter_id=adapter_id,
            reviewed=reviewed,
            **kwargs,
        )


def default_registry() -> DependencyCommitmentAdapterRegistry:
    """Return a new closed reviewed-adapter registry."""

    return DependencyCommitmentAdapterRegistry()


def authority_descriptor() -> dict[str, Any]:
    """Return the adapter execution authority without widening others."""

    return {
        "canonical_semantic_and_statement_authority": "ipfs_datasets_py",
        "execution_scheduling_admission_authority": "ipfs_accelerate_py",
        "verified_storage_wal_cas_authority": "ipfs_kit_py",
        "adapter_execution": "ipfs_accelerate_py.testing.proof_reuse.dependency_commitment_adapters",
        "schema_authority": "ipfs_datasets_py.logic.zkp.pctdd.fixture_instance_contracts",
        "does_not": (
            "execution or semantics; skip; TestExecutionKeyV2; pre-call "
            "fixture-instance authority; composite phase receipts; "
            "current-root publication; production ZK"
        ),
        "establishes": (
            "only reviewed adapters commit supported privacy-safe values; "
            "opaque dependencies execute fully"
        ),
        "claim_class": CLAIM_CLASS,
        "may_authorize_skip": False,
        "production_admitted": False,
        "self_approved": False,
        "worker_authored_test_is_sufficient_alone": False,
        "authoritative_after": AUTHORITATIVE_AFTER,
        "privacy_secret_rule": PRIVACY_SECRET_RULE,
        "default_reuse_class": DEFAULT_REUSE_CLASS,
        "closed_reuse_classes": list(CLOSED_REUSE_CLASSES),
        "review_candidate_families": list(REVIEW_CANDIDATE_FAMILIES),
        "committed_provider_fields": list(COMMITTED_PROVIDER_FIELDS),
        "opaque_adapter_interface": OPAQUE_DEPENDENCY_ADAPTER_INTERFACE,
        "registry_interface": DEPENDENCY_COMMITMENT_ADAPTER_REGISTRY_INTERFACE,
        "adapter_interface": REVIEWED_DEPENDENCY_COMMITMENT_ADAPTER_INTERFACE,
    }
