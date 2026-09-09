"""Fail-closed PCPR-043 cross-repository compatibility checks.

One explicit supported combination binds Accelerate, Datasets, and Kit
to the PCPR-040 catalog CID, PCPR-041 vector CIDs, and PCPR-042 negative
document CID. Compact recipes generate missing-repository, reminted,
sibling-import, cross-version, mixed-catalog, partial-publication,
unsupported-Python, authority-boundary, and claimed-early combinations
that must fail identically.

This module is not a freeze (PCPR-002), not a remint of PCPR-040/041/042
identities, and not the signed portfolio compatibility lock (PCPR-056).
It is not release authority: it does not write DuckDB or Quack state and
never emits a closed PCPR release outcome. Live claims require live
evidence. Simulated results are not live.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.canonical_byte_cid_vectors import (
    PINNED_CATALOG_CID as PCPR_041_CATALOG_CID,
    PINNED_CONTRACT_VECTORS,
    PINNED_VECTOR_DOCUMENT_CID as PCPR_041_VECTOR_DOCUMENT_CID,
)
from ipfs_accelerate_py.assurance.negative_cross_language_vectors import (
    PINNED_CATALOG_CID as PCPR_042_CATALOG_CID,
    PINNED_VECTOR_DOCUMENT_CID as PCPR_042_NEGATIVE_DOCUMENT_CID,
)
from ipfs_accelerate_py.assurance.python_compatibility import (
    FORBIDDEN_PYTHON_VERSIONS,
    PYTHON_FLOOR,
    REQUIRES_PYTHON,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    CONTRACT_AUTHORITIES,
    CONTRACT_SCHEMA_IDS,
    PCPR_002_TASK_ID,
    PCPR_040_TASK_ID,
    PCPR_041_TASK_ID,
    PCPR_042_TASK_ID,
    PCPR_043_TASK_ID,
    REQUIRED_CONTRACT_NAMES,
    catalog_cid,
    content_identity,
)

INTERFACE: Final = "CrossRepositoryCompatibility@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/cross-repository-compatibility@1"
DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/cross-repository-compatibility-document@1"
)
PCPR_043_GOAL_ID: Final = "PCPR-G520"
PCPR_056_TASK_ID: Final = "PCPR-056"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
EVIDENCE_ID: Final = "pcpr/cross-repository-compatibility@1"

PINNED_CATALOG_CID: Final = (
    "baguqeeraqpptps2gf55wmthv3sm5nzlxqqdjfyg2ggkbl5ovfzuvvvsxap4a"
)
if PINNED_CATALOG_CID != PCPR_041_CATALOG_CID:
    raise RuntimeError("PCPR-043 catalog CID remints PCPR-041")
if PINNED_CATALOG_CID != PCPR_042_CATALOG_CID:
    raise RuntimeError("PCPR-043 catalog CID remints PCPR-042")

PINNED_041_VECTOR_DOCUMENT_CID: Final = PCPR_041_VECTOR_DOCUMENT_CID
PINNED_042_NEGATIVE_DOCUMENT_CID: Final = PCPR_042_NEGATIVE_DOCUMENT_CID

REMINT_CID: Final = (
    "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)
SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = ("Python", "JavaScript")
REQUIRED_REPOSITORIES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py",
    "ipfs_datasets_py",
    "ipfs_kit_py",
)
DATASETS_OWNED_CONTRACTS: Final[tuple[str, ...]] = (
    "SupervisorContextPack",
    "SemanticArtifactIdentity",
)
KIT_OWNED_CONTRACTS: Final[tuple[str, ...]] = ("DurableArtifactReceipt",)

INCOMPATIBLE_CATEGORIES: Final[tuple[str, ...]] = (
    "missing_repository",
    "reminted_identity",
    "sibling_import",
    "cross_version",
    "mixed_catalog_vectors",
    "partial_publication",
    "unsupported_python",
    "authority_boundary",
    "claimed_early",
)

SUPPORTED_COMBINATION_ID: Final = (
    "pcpr.v1.python312.accelerate-datasets-kit.shared-contracts-v1"
)

PINNED_VECTOR_CIDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        name: str(PINNED_CONTRACT_VECTORS[name]["cid"])
        for name in REQUIRED_CONTRACT_NAMES
    }
)


def _supported_combination_mapping() -> dict[str, Any]:
    return {
        "id": SUPPORTED_COMBINATION_ID,
        "repositories": list(REQUIRED_REPOSITORIES),
        "python": PYTHON_FLOOR,
        "requires_python": REQUIRES_PYTHON,
        "languages": list(SUPPORTED_LANGUAGES),
        "typescript_compiler": "unavailable",
        "catalog_cid": PINNED_CATALOG_CID,
        "vector_document_cid": PINNED_041_VECTOR_DOCUMENT_CID,
        "negative_document_cid": PINNED_042_NEGATIVE_DOCUMENT_CID,
        "schema_suffix": "@1",
        "contract_count": len(REQUIRED_CONTRACT_NAMES),
        "lock": False,
        "frozen": False,
        "sibling_import": False,
        "duckdb_or_quack_state_written": False,
    }


SUPPORTED_COMBINATION: Final[Mapping[str, Any]] = MappingProxyType(
    _supported_combination_mapping()
)

INCOMPATIBLE_RECIPES: Final[tuple[Mapping[str, Any], ...]] = (
    MappingProxyType(
        {
            "id": "missing_repository.datasets",
            "category": "missing_repository",
            "reject_kind": "missing_repository",
            "mutation": MappingProxyType(
                {"drop_repository": "ipfs_datasets_py"}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "missing_repository.kit",
            "category": "missing_repository",
            "reject_kind": "missing_repository",
            "mutation": MappingProxyType({"drop_repository": "ipfs_kit_py"}),
        }
    ),
    MappingProxyType(
        {
            "id": "missing_repository.accelerate",
            "category": "missing_repository",
            "reject_kind": "missing_repository",
            "mutation": MappingProxyType(
                {"drop_repository": "ipfs_accelerate_py"}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "reminted_identity.catalog",
            "category": "reminted_identity",
            "reject_kind": "reminted_identity",
            "identical_without_reverse": True,
            "mutation": MappingProxyType(
                {
                    "set": MappingProxyType({"catalog_cid": REMINT_CID}),
                    "reverse_keys": True,
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "reminted_identity.vector_document",
            "category": "reminted_identity",
            "reject_kind": "reminted_identity",
            "mutation": MappingProxyType(
                {"set": MappingProxyType({"vector_document_cid": REMINT_CID})}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "reminted_identity.negative_document",
            "category": "reminted_identity",
            "reject_kind": "reminted_identity",
            "mutation": MappingProxyType(
                {
                    "set": MappingProxyType(
                        {"negative_document_cid": REMINT_CID}
                    )
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "reminted_identity.owned_vector",
            "category": "reminted_identity",
            "reject_kind": "reminted_identity",
            "mutation": MappingProxyType(
                {
                    "set_vector": MappingProxyType(
                        {"DurableArtifactReceipt": REMINT_CID}
                    )
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "sibling_import.datasets_imports_accelerate",
            "category": "sibling_import",
            "reject_kind": "sibling_import",
            "mutation": MappingProxyType(
                {"set": MappingProxyType({"sibling_import": True})}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "sibling_import.kit_imports_datasets",
            "category": "sibling_import",
            "reject_kind": "sibling_import",
            "mutation": MappingProxyType(
                {"set": MappingProxyType({"sibling_import": True})}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "cross_version.schema_v2",
            "category": "cross_version",
            "reject_kind": "cross_version",
            "mutation": MappingProxyType(
                {"set": MappingProxyType({"schema_suffix": "@2"})}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "mixed_catalog_vectors.catalog_is_vector_cid",
            "category": "mixed_catalog_vectors",
            "reject_kind": "mixed_catalog_vectors",
            "mutation": MappingProxyType(
                {
                    "set": MappingProxyType(
                        {
                            "catalog_cid": PINNED_041_VECTOR_DOCUMENT_CID,
                        }
                    )
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "partial_publication.thirteen_contracts",
            "category": "partial_publication",
            "reject_kind": "partial_publication",
            "mutation": MappingProxyType(
                {"drop_contract": "PortfolioCompatibilityManifest"}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "unsupported_python.3_11",
            "category": "unsupported_python",
            "reject_kind": "unsupported_python",
            "mutation": MappingProxyType(
                {"set": MappingProxyType({"python": "3.11"})}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "unsupported_python.3_8",
            "category": "unsupported_python",
            "reject_kind": "unsupported_python",
            "mutation": MappingProxyType(
                {"set": MappingProxyType({"python": "3.8"})}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "authority_boundary.kit_owns_context_pack",
            "category": "authority_boundary",
            "reject_kind": "authority_boundary",
            "mutation": MappingProxyType(
                {
                    "set_authority": MappingProxyType(
                        {"SupervisorContextPack": "ipfs_kit_py"}
                    )
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "authority_boundary.datasets_owns_durable",
            "category": "authority_boundary",
            "reject_kind": "authority_boundary",
            "mutation": MappingProxyType(
                {
                    "set_authority": MappingProxyType(
                        {"DurableArtifactReceipt": "ipfs_datasets_py"}
                    )
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "claimed_early.lock_pcpr056",
            "category": "claimed_early",
            "reject_kind": "claimed_early",
            "mutation": MappingProxyType(
                {"set": MappingProxyType({"lock": True})}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "claimed_early.freeze_pcpr002",
            "category": "claimed_early",
            "reject_kind": "claimed_early",
            "mutation": MappingProxyType(
                {"set": MappingProxyType({"frozen": True})}
            ),
        }
    ),
)


class CrossRepositoryCompatibilityError(ValueError):
    """Malformed compatibility evidence or a forbidden remint."""

    def __init__(self, message: str, *, reject_kind: str | None = None) -> None:
        super().__init__(message)
        self.reject_kind = reject_kind


def pinned_vector_cids() -> dict[str, str]:
    return dict(PINNED_VECTOR_CIDS)


def supported_snapshot() -> dict[str, Any]:
    snapshot = _supported_combination_mapping()
    snapshot["contract_schema_ids"] = dict(CONTRACT_SCHEMA_IDS)
    snapshot["vector_cids"] = pinned_vector_cids()
    snapshot["authorities"] = dict(CONTRACT_AUTHORITIES)
    return snapshot


def apply_mutation(
    base: Mapping[str, Any], mutation: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        key: (list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict) else value)
        for key, value in dict(base).items()
    }
    drop_repository = mutation.get("drop_repository")
    if isinstance(drop_repository, str) and drop_repository:
        repositories = list(payload.get("repositories") or [])
        payload["repositories"] = [
            item for item in repositories if item != drop_repository
        ]
    for key, value in dict(mutation.get("set") or {}).items():
        payload[key] = value
    for name, cid in dict(mutation.get("set_vector") or {}).items():
        vectors = dict(payload.get("vector_cids") or {})
        vectors[name] = cid
        payload["vector_cids"] = vectors
    for name, authority in dict(mutation.get("set_authority") or {}).items():
        authorities = dict(payload.get("authorities") or {})
        authorities[name] = authority
        payload["authorities"] = authorities
    drop_contract = mutation.get("drop_contract")
    if isinstance(drop_contract, str) and drop_contract:
        schemas = dict(payload.get("contract_schema_ids") or {})
        schemas.pop(drop_contract, None)
        payload["contract_schema_ids"] = schemas
        payload["contract_count"] = len(schemas)
        vectors = dict(payload.get("vector_cids") or {})
        vectors.pop(drop_contract, None)
        payload["vector_cids"] = vectors
    if mutation.get("reverse_keys"):
        payload = dict(reversed(list(payload.items())))
    return payload


def classify_admission_error(message: str) -> str:
    text = message.lower()
    if "sibling" in text:
        return "sibling_import"
    if "python" in text:
        return "unsupported_python"
    if "lock" in text or "frozen" in text or "freeze" in text:
        return "claimed_early"
    if "authorit" in text:
        return "authority_boundary"
    if "partial" in text or "contract_count" in text or "fourteen" in text:
        return "partial_publication"
    if "mixed" in text:
        return "mixed_catalog_vectors"
    if "@2" in text or "cross-version" in text or "cross_version" in text:
        return "cross_version"
    if "repository" in text or "repositories" in text:
        return "missing_repository"
    if "remint" in text or "catalog" in text or "vector" in text:
        return "reminted_identity"
    return "invalid"


def admit_combination(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Admit only the explicit supported combination. Everything else fails."""

    if not isinstance(payload, Mapping):
        raise CrossRepositoryCompatibilityError(
            "combination must be a mapping", reject_kind="invalid"
        )
    repositories = payload.get("repositories")
    if list(repositories or ()) != list(REQUIRED_REPOSITORIES):
        raise CrossRepositoryCompatibilityError(
            "supported combination requires Accelerate, Datasets, and Kit",
            reject_kind="missing_repository",
        )
    python = payload.get("python")
    if python != PYTHON_FLOOR or python in FORBIDDEN_PYTHON_VERSIONS:
        raise CrossRepositoryCompatibilityError(
            f"python {python!r} is not the declared 3.12 floor",
            reject_kind="unsupported_python",
        )
    if payload.get("requires_python") != REQUIRES_PYTHON:
        raise CrossRepositoryCompatibilityError(
            "requires_python remints the declared floor",
            reject_kind="unsupported_python",
        )
    if payload.get("schema_suffix") != "@1":
        raise CrossRepositoryCompatibilityError(
            "schema_suffix @2 is a cross-version combination",
            reject_kind="cross_version",
        )
    if payload.get("sibling_import") is True:
        raise CrossRepositoryCompatibilityError(
            "sibling import is not a supported combination",
            reject_kind="sibling_import",
        )
    if payload.get("lock") is True:
        raise CrossRepositoryCompatibilityError(
            "signed portfolio lock remains PCPR-056 and is not claimed here",
            reject_kind="claimed_early",
        )
    if payload.get("frozen") is True:
        raise CrossRepositoryCompatibilityError(
            "freeze remains PCPR-002 and is not claimed here",
            reject_kind="claimed_early",
        )
    catalog = payload.get("catalog_cid")
    if catalog != PINNED_CATALOG_CID:
        if catalog == PINNED_041_VECTOR_DOCUMENT_CID:
            raise CrossRepositoryCompatibilityError(
                "mixed catalog and vector identities are not supported",
                reject_kind="mixed_catalog_vectors",
            )
        raise CrossRepositoryCompatibilityError(
            f"catalog CID {catalog} remints {PINNED_CATALOG_CID}",
            reject_kind="reminted_identity",
        )
    if payload.get("vector_document_cid") != PINNED_041_VECTOR_DOCUMENT_CID:
        raise CrossRepositoryCompatibilityError(
            "vector document CID remints PCPR-041",
            reject_kind="reminted_identity",
        )
    if payload.get("negative_document_cid") != PINNED_042_NEGATIVE_DOCUMENT_CID:
        raise CrossRepositoryCompatibilityError(
            "negative document CID remints PCPR-042",
            reject_kind="reminted_identity",
        )
    schemas = payload.get("contract_schema_ids")
    if not isinstance(schemas, Mapping) or dict(schemas) != dict(CONTRACT_SCHEMA_IDS):
        raise CrossRepositoryCompatibilityError(
            "partial publication or reminted schema set; fourteen contracts required",
            reject_kind="partial_publication",
        )
    if payload.get("contract_count") != len(REQUIRED_CONTRACT_NAMES):
        raise CrossRepositoryCompatibilityError(
            "contract_count must remain fourteen",
            reject_kind="partial_publication",
        )
    vectors = payload.get("vector_cids")
    if not isinstance(vectors, Mapping) or dict(vectors) != dict(PINNED_VECTOR_CIDS):
        raise CrossRepositoryCompatibilityError(
            "a vector CID remints the PCPR-041 pin",
            reject_kind="reminted_identity",
        )
    authorities = payload.get("authorities")
    if not isinstance(authorities, Mapping) or dict(authorities) != dict(
        CONTRACT_AUTHORITIES
    ):
        raise CrossRepositoryCompatibilityError(
            "authority boundary remint is not a supported combination",
            reject_kind="authority_boundary",
        )
    if payload.get("id") != SUPPORTED_COMBINATION_ID:
        raise CrossRepositoryCompatibilityError(
            "supported combination id reminted",
            reject_kind="reminted_identity",
        )
    if payload.get("duckdb_or_quack_state_written") is True:
        raise CrossRepositoryCompatibilityError(
            "compatibility checks must not write DuckDB or Quack state",
            reject_kind="invalid",
        )
    live_catalog = catalog_cid()
    if live_catalog != PINNED_CATALOG_CID:
        raise CrossRepositoryCompatibilityError(
            f"live catalog CID {live_catalog} remints {PINNED_CATALOG_CID}",
            reject_kind="reminted_identity",
        )
    return MappingProxyType(dict(payload))


def evaluate_incompatible_combinations() -> list[dict[str, Any]]:
    """Every incompatible recipe must reject with its declared kind."""

    rows: list[dict[str, Any]] = []
    base = supported_snapshot()
    admit_combination(base)
    for recipe in INCOMPATIBLE_RECIPES:
        mutation = dict(recipe["mutation"])
        forward = apply_mutation(base, {k: v for k, v in mutation.items() if k != "reverse_keys"})
        reversed_payload = apply_mutation(base, mutation) if mutation.get("reverse_keys") else forward
        rejected = False
        reject_kind = None
        message = ""
        try:
            admit_combination(forward)
        except CrossRepositoryCompatibilityError as exc:
            rejected = True
            reject_kind = exc.reject_kind or classify_admission_error(str(exc))
            message = str(exc)
        if not rejected:
            raise CrossRepositoryCompatibilityError(
                f"{recipe['id']} was admitted",
                reject_kind="invalid",
            )
        expected = str(recipe["reject_kind"])
        if reject_kind != expected:
            raise CrossRepositoryCompatibilityError(
                f"{recipe['id']} reject_kind {reject_kind} drifted from {expected}",
                reject_kind=reject_kind,
            )
        identical_reversed = True
        if mutation.get("reverse_keys"):
            try:
                admit_combination(reversed_payload)
                identical_reversed = False
            except CrossRepositoryCompatibilityError as exc:
                reversed_kind = exc.reject_kind or classify_admission_error(str(exc))
                identical_reversed = reversed_kind == reject_kind
            if not identical_reversed:
                raise CrossRepositoryCompatibilityError(
                    f"{recipe['id']} did not fail identically under key reorder",
                    reject_kind=reject_kind,
                )
        rows.append(
            {
                "id": recipe["id"],
                "category": recipe["category"],
                "reject_kind": reject_kind,
                "rejected": True,
                "identical_reversed": identical_reversed,
                "reason": message,
            }
        )
    return rows


def refuse_compatibility_remint(cid: str) -> str:
    if cid != PINNED_COMPATIBILITY_DOCUMENT_CID:
        raise CrossRepositoryCompatibilityError(
            f"compatibility document CID {cid} remints {PINNED_COMPATIBILITY_DOCUMENT_CID}"
        )
    return cid


def compatibility_document() -> dict[str, Any]:
    incompatibles = evaluate_incompatible_combinations()
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_043_TASK_ID,
        "goal_id": PCPR_043_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "board_namespace": PCPR_BOARD_NAMESPACE,
        "evidence_id": EVIDENCE_ID,
        "frozen": False,
        "freeze_task": PCPR_002_TASK_ID,
        "normative_task": PCPR_040_TASK_ID,
        "positive_vector_task": PCPR_041_TASK_ID,
        "negative_vector_task": PCPR_042_TASK_ID,
        "lock_task": PCPR_056_TASK_ID,
        "catalog_cid": PINNED_CATALOG_CID,
        "vector_document_cid": PINNED_041_VECTOR_DOCUMENT_CID,
        "negative_document_cid": PINNED_042_NEGATIVE_DOCUMENT_CID,
        "languages": list(SUPPORTED_LANGUAGES),
        "typescript_compiler": "unavailable",
        "python": PYTHON_FLOOR,
        "requires_python": REQUIRES_PYTHON,
        "supported_combination_id": SUPPORTED_COMBINATION_ID,
        "supported_combination": _supported_combination_mapping(),
        "supported_combination_count": 1,
        "incompatible_categories": list(INCOMPATIBLE_CATEGORIES),
        "incompatible_count": len(incompatibles),
        "incompatibles": incompatibles,
        "contract_schema_ids": dict(CONTRACT_SCHEMA_IDS),
        "vector_cids": pinned_vector_cids(),
        "authorities": dict(CONTRACT_AUTHORITIES),
        "datasets_owned_contracts": list(DATASETS_OWNED_CONTRACTS),
        "kit_owned_contracts": list(KIT_OWNED_CONTRACTS),
        "lock": False,
        "remint": False,
        "duckdb_or_quack_state_written": False,
    }
    _ = content_identity(document)
    return document


def compatibility_document_cid() -> str:
    cid = content_identity(compatibility_document())
    if cid != PINNED_COMPATIBILITY_DOCUMENT_CID:
        raise CrossRepositoryCompatibilityError(
            f"compatibility document CID {cid} remints {PINNED_COMPATIBILITY_DOCUMENT_CID}"
        )
    return cid


# Pinned after the document encoder is measured. Drift is a remint.
PINNED_COMPATIBILITY_DOCUMENT_CID: Final = (
    "baguqeerafnibnbrsfaqvbxx444kqk3gi4sh244r7bl7usdxk2krg6njlquza"
)


__all__ = (
    "CONTRACT_AUTHORITIES",
    "CONTRACT_SCHEMA_IDS",
    "CrossRepositoryCompatibilityError",
    "DATASETS_OWNED_CONTRACTS",
    "INCOMPATIBLE_CATEGORIES",
    "INCOMPATIBLE_RECIPES",
    "INTERFACE",
    "KIT_OWNED_CONTRACTS",
    "PINNED_041_VECTOR_DOCUMENT_CID",
    "PINNED_042_NEGATIVE_DOCUMENT_CID",
    "PINNED_CATALOG_CID",
    "PINNED_COMPATIBILITY_DOCUMENT_CID",
    "PINNED_VECTOR_CIDS",
    "REQUIRED_REPOSITORIES",
    "SCHEMA",
    "SUPPORTED_COMBINATION",
    "SUPPORTED_COMBINATION_ID",
    "SUPPORTED_LANGUAGES",
    "admit_combination",
    "apply_mutation",
    "compatibility_document",
    "compatibility_document_cid",
    "evaluate_incompatible_combinations",
    "refuse_compatibility_remint",
    "supported_snapshot",
)
