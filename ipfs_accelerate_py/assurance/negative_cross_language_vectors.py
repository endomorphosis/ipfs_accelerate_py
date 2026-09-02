"""Fail-closed PCPR-042 negative and cross-language vectors.

Compact recipes generate invalid, stale, unknown, out-of-bound, reordered,
reminted, and cross-version inputs that must fail identically. A second
language encoder (JavaScript) must mint the same CIDs for positive
PCPR-041 fixtures and the same reject kinds for those negatives.

This module is not a freeze (PCPR-002), not a remint of PCPR-040/041
identities, and not cross-repository compatibility (PCPR-043). It is not
release authority: it does not write DuckDB or Quack state and never
emits a closed PCPR release outcome. Live claims require live evidence.
Simulated results are not live. TypeScript compilation stays typed
unavailable when tsc is absent.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.canonical_byte_cid_vectors import (
    PINNED_CATALOG_CID as PCPR_041_CATALOG_CID,
    PINNED_CONTRACT_VECTORS,
    PINNED_NFC_CID,
    PINNED_NFC_SHA256,
    PINNED_VECTOR_SEED_CID,
    CanonicalByteCidVectorError,
    compact_recipes,
    contract_vector,
    key_order_vector,
    refuse_vector_remint,
    unicode_nfc_vector,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    CONTRACT_BY_NAME,
    INT64_MAX,
    MAX_ARRAY_ITEMS,
    MAX_STRING_BYTES,
    PCPR_002_TASK_ID,
    PCPR_040_TASK_ID,
    PCPR_041_TASK_ID,
    PCPR_042_TASK_ID,
    PCPR_043_TASK_ID,
    SharedContractAdmissionError,
    admit_shared_contract,
    canonical_json_bytes,
    catalog_cid,
    content_identity,
)

INTERFACE: Final = "NegativeCrossLanguageVectors@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/negative-cross-language-vectors@1"
VECTOR_DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/negative-cross-language-vector-document@1"
)
PCPR_042_GOAL_ID: Final = "PCPR-G520"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
EVIDENCE_ID: Final = "pcpr/negative-cross-language-vectors@1"

NEGATIVE_CATEGORIES: Final[tuple[str, ...]] = (
    "invalid",
    "stale",
    "unknown",
    "out_of_bound",
    "reordered",
    "reminted",
    "cross_version",
)
SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = ("Python", "JavaScript")
STALE_ZERO_TREE: Final = "0" * 40
PINNED_CATALOG_CID: Final = (
    "baguqeeraqpptps2gf55wmthv3sm5nzlxqqdjfyg2ggkbl5ovfzuvvvsxap4a"
)
if PINNED_CATALOG_CID != PCPR_041_CATALOG_CID:
    raise RuntimeError("PCPR-042 catalog CID remints PCPR-041")
NFC_COLLISION_OBJECT: Final[Mapping[str, int]] = MappingProxyType(
    {"caf\u00e9": 1, "cafe\u0301": 2}
)
JAVASCRIPT_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/assurance/negative_cross_language_vectors.mjs"
)
TYPESCRIPT_SOURCE_RELPATH: Final = (
    "ipfs_accelerate_js/src/assurance/negative_cross_language_vectors.ts"
)

# Compact mutation recipes. Positive envelopes come from PCPR-041 recipes.
NEGATIVE_RECIPES: Final[tuple[Mapping[str, Any], ...]] = (
    MappingProxyType(
        {
            "id": "invalid.non_mapping",
            "contract": "SupervisorObjectiveIntent",
            "category": "invalid",
            "reject_kind": "invalid",
            "mutation": MappingProxyType({"replace": None}),
        }
    ),
    MappingProxyType(
        {
            "id": "invalid.empty_language",
            "contract": "SupervisorObjectiveIntent",
            "category": "invalid",
            "reject_kind": "invalid",
            "mutation": MappingProxyType({"set": MappingProxyType({"language": ""})}),
        }
    ),
    MappingProxyType(
        {
            "id": "invalid.bool_as_int64",
            "contract": "SupervisorEvent",
            "category": "invalid",
            "reject_kind": "invalid",
            "mutation": MappingProxyType({"set": MappingProxyType({"sequence": True})}),
        }
    ),
    MappingProxyType(
        {
            "id": "stale.zero_tree",
            "contract": "ObjectiveMaterializationReceipt",
            "category": "stale",
            "reject_kind": "stale",
            "mutation": MappingProxyType(
                {"set": MappingProxyType({"current_tree": STALE_ZERO_TREE})}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "stale.catalog_cid",
            "contract": "PortfolioCompatibilityManifest",
            "category": "stale",
            "reject_kind": "stale",
            "mutation": MappingProxyType(
                {
                    "set": MappingProxyType(
                        {"contract_catalog_cid": PINNED_VECTOR_SEED_CID}
                    )
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "unknown.extra_field",
            "contract": "SupervisorObjectiveIntent",
            "category": "unknown",
            "reject_kind": "unknown",
            "mutation": MappingProxyType(
                {"set": MappingProxyType({"unexpected": "field"})}
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "unknown.extra_field_reordered",
            "contract": "ProofObligation",
            "category": "reordered",
            "reject_kind": "unknown",
            "identical_without_reverse": True,
            "mutation": MappingProxyType(
                {
                    "set": MappingProxyType({"unexpected": "field"}),
                    "reverse_keys": True,
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "out_of_bound.int64",
            "contract": "SupervisorEvent",
            "category": "out_of_bound",
            "reject_kind": "out_of_bound",
            "mutation": MappingProxyType({"overflow_int64": "sequence"}),
        }
    ),
    MappingProxyType(
        {
            "id": "out_of_bound.float",
            "contract": "DurableArtifactReceipt",
            "category": "out_of_bound",
            "reject_kind": "out_of_bound",
            "mutation": MappingProxyType({"set": MappingProxyType({"byte_length": 1.5})}),
        }
    ),
    MappingProxyType(
        {
            "id": "out_of_bound.string",
            "contract": "ProofObligation",
            "category": "out_of_bound",
            "reject_kind": "out_of_bound",
            "mutation": MappingProxyType({"oversize_string": "statement"}),
        }
    ),
    MappingProxyType(
        {
            "id": "out_of_bound.array",
            "contract": "PortfolioCompatibilityManifest",
            "category": "out_of_bound",
            "reject_kind": "out_of_bound",
            "mutation": MappingProxyType({"oversize_array": "component_ids"}),
        }
    ),
    MappingProxyType(
        {
            "id": "out_of_bound.cidv0",
            "contract": "DurableArtifactReceipt",
            "category": "invalid",
            "reject_kind": "invalid",
            "mutation": MappingProxyType(
                {
                    "set": MappingProxyType(
                        {"cid": "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"}
                    )
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "reordered.nfc_key_collision",
            "contract": "SupervisorObjectiveIntent",
            "category": "reordered",
            "reject_kind": "reordered",
            "mutation": MappingProxyType({"canonical_only": True}),
        }
    ),
    MappingProxyType(
        {
            "id": "reminted.schema",
            "contract": "SupervisorObjectiveIntent",
            "category": "reminted",
            "reject_kind": "reminted",
            "mutation": MappingProxyType(
                {
                    "set": MappingProxyType(
                        {
                            "schema": (
                                "pcpr/shared-contracts/supervisor-objective-intent@1-remint"
                            )
                        }
                    )
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "reminted.interface",
            "contract": "SupervisorObjectiveIntent",
            "category": "reminted",
            "reject_kind": "reminted",
            "mutation": MappingProxyType(
                {
                    "set": MappingProxyType(
                        {"interface": "SupervisorObjectiveIntent@1-other"}
                    )
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "reminted.vector_cid",
            "contract": "SupervisorObjectiveIntent",
            "category": "reminted",
            "reject_kind": "reminted",
            "mutation": MappingProxyType(
                {
                    "remint_cid": (
                        "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    )
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "cross_version.schema_v2",
            "contract": "SupervisorObjectiveIntent",
            "category": "cross_version",
            "reject_kind": "cross_version",
            "mutation": MappingProxyType(
                {
                    "set": MappingProxyType(
                        {"schema": "pcpr/shared-contracts/supervisor-objective-intent@2"}
                    )
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "id": "cross_version.interface_v2",
            "contract": "SupervisorObjectiveIntent",
            "category": "cross_version",
            "reject_kind": "cross_version",
            "mutation": MappingProxyType(
                {"set": MappingProxyType({"interface": "SupervisorObjectiveIntent@2"})}
            ),
        }
    ),
)

CROSS_LANGUAGE_CID_IDS: Final[tuple[str, ...]] = (
    "python_js.intent",
    "python_js.nfc",
    "python_js.key_order",
)
CROSS_LANGUAGE_REJECT_IDS: Final[tuple[str, ...]] = (
    "python_js.unknown_identical",
    "python_js.int64_identical",
    "python_js.cross_version_identical",
    "python_js.stale_identical",
)
CROSS_LANGUAGE_REJECT_SOURCE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "python_js.unknown_identical": "unknown.extra_field",
        "python_js.int64_identical": "out_of_bound.int64",
        "python_js.cross_version_identical": "cross_version.schema_v2",
        "python_js.stale_identical": "stale.zero_tree",
    }
)

PINNED_VECTOR_DOCUMENT_CID: Final = (
    "baguqeeraior5lq3fvefhgwsm3cjjaxozffmdmqmszwxy4ozfb37mnqx6lfca"
)


class NegativeCrossLanguageVectorError(ValueError):
    """Malformed negative/cross-language evidence or a forbidden remint."""

    def __init__(self, message: str, *, reject_kind: str | None = None) -> None:
        super().__init__(message)
        self.reject_kind = reject_kind


def _base_payload(name: str) -> dict[str, Any]:
    spec = CONTRACT_BY_NAME[name]
    payload = {"schema": spec.schema, "interface": spec.interface}
    payload.update(compact_recipes()[name])
    return payload


def apply_mutation(base: Mapping[str, Any], mutation: Mapping[str, Any]) -> Any:
    if "replace" in mutation:
        return mutation["replace"]
    payload = dict(base)
    for key, value in dict(mutation.get("set") or {}).items():
        payload[key] = value
    for key in mutation.get("delete") or ():
        payload.pop(key, None)
    overflow = mutation.get("overflow_int64")
    if isinstance(overflow, str) and overflow:
        payload[overflow] = INT64_MAX + 1
    oversize = mutation.get("oversize_string")
    if isinstance(oversize, str) and oversize:
        payload[oversize] = "x" * (MAX_STRING_BYTES + 1)
    oversize_array = mutation.get("oversize_array")
    if isinstance(oversize_array, str) and oversize_array:
        payload[oversize_array] = [f"item:{index}" for index in range(MAX_ARRAY_ITEMS + 1)]
    if mutation.get("reverse_keys"):
        payload = dict(reversed(list(payload.items())))
    return payload


def classify_admission_error(message: str, payload: Any) -> str:
    text = message.lower()
    if isinstance(payload, Mapping):
        schema = payload.get("schema")
        interface = payload.get("interface")
        if isinstance(schema, str) and schema.endswith("@2"):
            return "cross_version"
        if isinstance(interface, str) and interface.endswith("@2"):
            return "cross_version"
    if "unknown fields" in text:
        return "unknown"
    if "remint" in text:
        return "reminted"
    if isinstance(payload, Mapping) and any(type(item) is float for item in payload.values()):
        return "out_of_bound"
    if "must be an int64 integer" in text:
        return "invalid"
    if (
        "outside the int64" in text
        or "float" in text
        or "max_string" in text
        or "max_array" in text
        or "max_object" in text
        or "max_record" in text
    ):
        return "out_of_bound"
    if "stale" in text:
        return "stale"
    if "collide" in text:
        return "reordered"
    return "invalid"


def admit_negative_candidate(name: str, payload: Any) -> dict[str, Any]:
    """Fail-closed overlay used by negative vectors. Same order as JavaScript."""

    if not isinstance(payload, Mapping):
        raise NegativeCrossLanguageVectorError(
            "invalid: payload must be a mapping",
            reject_kind="invalid",
        )
    schema = payload.get("schema")
    interface = payload.get("interface")
    if isinstance(schema, str) and schema.endswith("@2"):
        raise NegativeCrossLanguageVectorError(
            "cross_version: schema @2 is not admitted",
            reject_kind="cross_version",
        )
    if isinstance(interface, str) and interface.endswith("@2"):
        raise NegativeCrossLanguageVectorError(
            "cross_version: interface @2 is not admitted",
            reject_kind="cross_version",
        )
    for field in ("current_tree", "tree_id", "scanned_tree_oid"):
        if payload.get(field) == STALE_ZERO_TREE:
            raise NegativeCrossLanguageVectorError(
                "stale: all-zero tree oid is the PCPR-040 fixture, not current",
                reject_kind="stale",
            )
    try:
        admitted = admit_shared_contract(name, payload)
    except SharedContractAdmissionError as exc:
        kind = classify_admission_error(str(exc), payload)
        raise NegativeCrossLanguageVectorError(
            f"{kind}: {exc}",
            reject_kind=kind,
        ) from exc
    catalog = admitted.get("contract_catalog_cid")
    if isinstance(catalog, str) and catalog != PINNED_CATALOG_CID:
        raise NegativeCrossLanguageVectorError(
            "stale: contract_catalog_cid is not the current PCPR-040 catalog",
            reject_kind="stale",
        )
    return admitted


def _reject_kind_for_recipe(recipe: Mapping[str, Any], *, reverse: bool | None = None) -> str:
    mutation = dict(recipe["mutation"])
    if reverse is True:
        mutation["reverse_keys"] = True
    elif reverse is False:
        mutation.pop("reverse_keys", None)
    if mutation.get("canonical_only"):
        try:
            canonical_json_bytes(dict(NFC_COLLISION_OBJECT))
        except SharedContractAdmissionError as exc:
            return classify_admission_error(str(exc), dict(NFC_COLLISION_OBJECT))
        raise NegativeCrossLanguageVectorError(
            f"{recipe['id']} NFC key collision was admitted"
        )
    remint_cid = mutation.get("remint_cid")
    if isinstance(remint_cid, str) and remint_cid:
        try:
            refuse_vector_remint(str(recipe["contract"]), remint_cid)
        except CanonicalByteCidVectorError:
            return "reminted"
        raise NegativeCrossLanguageVectorError(f"{recipe['id']} remint CID was admitted")
    payload = apply_mutation(_base_payload(str(recipe["contract"])), mutation)
    try:
        admit_negative_candidate(str(recipe["contract"]), payload)
    except NegativeCrossLanguageVectorError as exc:
        if exc.reject_kind:
            return exc.reject_kind
        raise
    raise NegativeCrossLanguageVectorError(f"{recipe['id']} was admitted")


def evaluate_negative_vector(recipe: Mapping[str, Any]) -> dict[str, Any]:
    expected = str(recipe["reject_kind"])
    kind = _reject_kind_for_recipe(recipe)
    if kind != expected:
        raise NegativeCrossLanguageVectorError(
            f"{recipe['id']} reject_kind {kind} != {expected}"
        )
    identical = False
    if recipe.get("identical_without_reverse"):
        other = _reject_kind_for_recipe(recipe, reverse=False)
        if other != kind:
            raise NegativeCrossLanguageVectorError(
                f"{recipe['id']} reversed and original reject kinds differ"
            )
        identical = True
    return {
        "id": recipe["id"],
        "contract": recipe["contract"],
        "category": recipe["category"],
        "reject_kind": kind,
        "rejected": True,
        "identical_reversed": identical,
    }


def evaluate_negative_vectors() -> tuple[dict[str, Any], ...]:
    results = tuple(evaluate_negative_vector(item) for item in NEGATIVE_RECIPES)
    observed = {item["category"] for item in results}
    if observed != set(NEGATIVE_CATEGORIES):
        raise NegativeCrossLanguageVectorError(
            f"negative categories {sorted(observed)} != {list(NEGATIVE_CATEGORIES)}"
        )
    if len(results) != len(NEGATIVE_RECIPES):
        raise NegativeCrossLanguageVectorError("negative recipe count drifted")
    return results


def cross_language_cid_vectors() -> tuple[dict[str, Any], ...]:
    intent = contract_vector("SupervisorObjectiveIntent")
    nfc = unicode_nfc_vector()
    ordered = key_order_vector()
    if intent["cid"] != PINNED_CONTRACT_VECTORS["SupervisorObjectiveIntent"]["cid"]:
        raise NegativeCrossLanguageVectorError("PCPR-041 intent CID reminted")
    if nfc["cid"] != PINNED_NFC_CID or nfc["canonical_sha256"] != PINNED_NFC_SHA256:
        raise NegativeCrossLanguageVectorError("PCPR-041 NFC CID reminted")
    if ordered["cid"] != intent["cid"]:
        raise NegativeCrossLanguageVectorError("key-order CID reminted")
    catalog = catalog_cid()
    if catalog != PINNED_CATALOG_CID:
        raise NegativeCrossLanguageVectorError(
            f"catalog CID {catalog} remints {PINNED_CATALOG_CID}"
        )
    return (
        {
            "id": "python_js.intent",
            "contract": "SupervisorObjectiveIntent",
            "cid": intent["cid"],
            "canonical_sha256": intent["canonical_sha256"],
            "byte_length": intent["byte_length"],
            "invariant": "Python and JavaScript mint the same SupervisorObjectiveIntent CID",
        },
        {
            "id": "python_js.nfc",
            "contract": "ProofObligation",
            "cid": nfc["cid"],
            "canonical_sha256": nfc["canonical_sha256"],
            "byte_length": nfc["byte_length"],
            "invariant": "Python and JavaScript NFC café share one CID",
        },
        {
            "id": "python_js.key_order",
            "contract": "SupervisorObjectiveIntent",
            "cid": ordered["cid"],
            "canonical_sha256": ordered["canonical_sha256"],
            "byte_length": ordered["byte_length"],
            "invariant": "Python and JavaScript key order share one CID",
        },
    )


def cross_language_reject_vectors(
    negatives: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    by_id = {item["id"]: item for item in negatives}
    rows: list[dict[str, Any]] = []
    for vector_id, source_id in CROSS_LANGUAGE_REJECT_SOURCE.items():
        source = by_id.get(source_id)
        if source is None:
            raise NegativeCrossLanguageVectorError(f"missing source vector {source_id}")
        rows.append(
            {
                "id": vector_id,
                "source_id": source_id,
                "reject_kind": source["reject_kind"],
                "rejected": True,
                "invariant": "Python and JavaScript reject this input with the same kind",
            }
        )
    return tuple(rows)


def negative_cross_language_vector_document() -> dict[str, Any]:
    negatives = evaluate_negative_vectors()
    cid_vectors = cross_language_cid_vectors()
    reject_vectors = cross_language_reject_vectors(negatives)
    document = {
        "schema": VECTOR_DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_042_TASK_ID,
        "goal_id": PCPR_042_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "board_namespace": PCPR_BOARD_NAMESPACE,
        "evidence_id": EVIDENCE_ID,
        "frozen": False,
        "freeze_task": PCPR_002_TASK_ID,
        "normative_task": PCPR_040_TASK_ID,
        "positive_vector_task": PCPR_041_TASK_ID,
        "compatibility_task": PCPR_043_TASK_ID,
        "catalog_cid": PINNED_CATALOG_CID,
        "vector_seed_cid": PINNED_VECTOR_SEED_CID,
        "languages": list(SUPPORTED_LANGUAGES),
        "typescript_compiler": "unavailable",
        "javascript_module": JAVASCRIPT_MODULE_RELPATH,
        "typescript_source": TYPESCRIPT_SOURCE_RELPATH,
        "negative_categories": list(NEGATIVE_CATEGORIES),
        "negative_count": len(negatives),
        "cross_language_count": len(cid_vectors) + len(reject_vectors),
        "negatives": [dict(item) for item in negatives],
        "cross_language": [dict(item) for item in (*cid_vectors, *reject_vectors)],
        "duckdb_or_quack_state_written": False,
    }
    _ = content_identity(document)
    return document


def vector_document_cid() -> str:
    cid = content_identity(negative_cross_language_vector_document())
    if cid != PINNED_VECTOR_DOCUMENT_CID:
        raise NegativeCrossLanguageVectorError(
            f"vector document CID {cid} remints {PINNED_VECTOR_DOCUMENT_CID}"
        )
    return cid


def refuse_negative_remint(cid: str) -> str:
    if cid != PINNED_VECTOR_DOCUMENT_CID:
        raise NegativeCrossLanguageVectorError(
            f"negative/cross-language document CID {cid} remints {PINNED_VECTOR_DOCUMENT_CID}"
        )
    return cid


def javascript_module_path() -> Path:
    return Path(__file__).with_name("negative_cross_language_vectors.mjs")


def run_javascript_vectors(*, node: str = "/usr/bin/node") -> dict[str, Any]:
    """Execute the independent JavaScript encoder. Missing node stays unavailable."""

    module = javascript_module_path()
    if not module.is_file():
        raise NegativeCrossLanguageVectorError(
            f"javascript module missing: {module}"
        )
    node_path = Path(node)
    if not node_path.is_file():
        raise FileNotFoundError(str(node_path))
    proc = subprocess.run(
        [str(node_path), str(module)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise NegativeCrossLanguageVectorError(
            f"javascript encoder exited {proc.returncode}: {proc.stderr.strip()}"
        )
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise NegativeCrossLanguageVectorError("javascript encoder must emit a mapping")
    if payload.get("simulated") is True:
        raise NegativeCrossLanguageVectorError(
            "javascript encoder must not represent simulated results as live"
        )
    return payload


def javascript_agrees_with_python(report: Mapping[str, Any]) -> None:
    negatives = evaluate_negative_vectors()
    by_id = {item["id"]: item for item in negatives}
    js_negatives = report.get("negatives")
    if not isinstance(js_negatives, list) or len(js_negatives) != len(negatives):
        raise NegativeCrossLanguageVectorError(
            "javascript negatives do not match the Python recipe set"
        )
    for item in js_negatives:
        if not isinstance(item, Mapping):
            raise NegativeCrossLanguageVectorError("javascript negative must be a mapping")
        expected = by_id.get(item.get("id"))
        if expected is None:
            raise NegativeCrossLanguageVectorError(
                f"javascript emitted unknown negative {item.get('id')}"
            )
        if item.get("reject_kind") != expected["reject_kind"] or item.get("rejected") is not True:
            raise NegativeCrossLanguageVectorError(
                f"javascript reject_kind for {expected['id']} does not match Python"
            )
    cid_rows = {item["id"]: item for item in cross_language_cid_vectors()}
    js_cross = report.get("cross_language")
    if not isinstance(js_cross, list):
        raise NegativeCrossLanguageVectorError("javascript cross_language must be a list")
    for item in js_cross:
        if not isinstance(item, Mapping):
            continue
        pinned = cid_rows.get(item.get("id"))
        if pinned is None:
            continue
        if (
            item.get("cid") != pinned["cid"]
            or item.get("canonical_sha256") != pinned["canonical_sha256"]
            or item.get("byte_length") != pinned["byte_length"]
        ):
            raise NegativeCrossLanguageVectorError(
                f"javascript CID for {pinned['id']} remints the Python pin"
            )


def negative_recipe_ids() -> tuple[str, ...]:
    return tuple(str(item["id"]) for item in NEGATIVE_RECIPES)


__all__ = (
    "CROSS_LANGUAGE_REJECT_SOURCE",
    "INTERFACE",
    "JAVASCRIPT_MODULE_RELPATH",
    "NEGATIVE_CATEGORIES",
    "NEGATIVE_RECIPES",
    "NegativeCrossLanguageVectorError",
    "PINNED_CATALOG_CID",
    "PINNED_VECTOR_DOCUMENT_CID",
    "SCHEMA",
    "SUPPORTED_LANGUAGES",
    "TYPESCRIPT_SOURCE_RELPATH",
    "admit_negative_candidate",
    "apply_mutation",
    "classify_admission_error",
    "cross_language_cid_vectors",
    "evaluate_negative_vector",
    "evaluate_negative_vectors",
    "javascript_agrees_with_python",
    "javascript_module_path",
    "negative_cross_language_vector_document",
    "negative_recipe_ids",
    "refuse_negative_remint",
    "run_javascript_vectors",
    "vector_document_cid",
)
