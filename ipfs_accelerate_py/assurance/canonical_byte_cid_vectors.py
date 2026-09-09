"""Fail-closed PCPR-041 canonical-byte and CID vectors.

Compact recipes generate exact DAG-JSON bytes and CIDv1 identities for
the fourteen PCPR shared contracts from the PCPR-040 catalog encoder.
Datasets and Kit bind those identities; they do not remint them.

This module is not a freeze (PCPR-002), not negative or cross-language
vectors (PCPR-042), and not cross-repository compatibility (PCPR-043).
It is not release authority: it does not write DuckDB or Quack state and
never emits a closed PCPR release outcome. Live claims require live
evidence. Simulated results are not live.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.shared_contracts import (
    CONTRACT_BY_NAME,
    PCPR_002_TASK_ID,
    PCPR_040_TASK_ID,
    PCPR_041_TASK_ID,
    PCPR_042_TASK_ID,
    PCPR_043_TASK_ID,
    REQUIRED_CONTRACT_NAMES,
    admit_shared_contract,
    canonical_json_bytes,
    catalog_cid,
    catalog_mapping,
    content_identity,
)

INTERFACE: Final = "CanonicalByteCidVectors@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/canonical-byte-cid-vectors@1"
VECTOR_DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/canonical-byte-cid-vector-document@1"
)
PCPR_041_GOAL_ID: Final = "PCPR-G510"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
EVIDENCE_ID: Final = "pcpr/canonical-byte-cid-vectors@1"

VECTOR_SEED: Final[Mapping[str, str]] = MappingProxyType(
    {
        "schema": "pcpr/shared-contracts/canonical-byte-cid-vector-seed@1",
        "interface": "CanonicalByteCidVectorSeed@1",
        "kind": "vector-seed",
        "task_id": PCPR_041_TASK_ID,
    }
)
VECTOR_ARTIFACT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "kind": "vector-artifact",
        "name": "DurableArtifactReceipt",
        "task_id": PCPR_041_TASK_ID,
    }
)
VECTOR_TREE_OID: Final = "41" * 20
NFC_COMPOSED: Final = "caf\u00e9"
NFC_DECOMPOSED: Final = "cafe\u0301"

PINNED_VECTOR_SEED_CID: Final = (
    "baguqeeragko4w64wovfw2643pvwkugj4x7vqxdvzqzpzzqh3liuzrlibzt2a"
)
PINNED_VECTOR_ARTIFACT_CID: Final = (
    "baguqeeraulqxswkwlfiexdb45mbekkbdzuz7xqjgmm3keoqmkmatlsz4zd5q"
)
PINNED_VECTOR_ARTIFACT_SHA256: Final = (
    "a2e179595659504b8c3ceb02452823cd33fbc1266336a23a0c530135cb3cc8fb"
)
PINNED_VECTOR_ARTIFACT_BYTES: Final = 79
PINNED_CATALOG_CID: Final = (
    "baguqeeraqpptps2gf55wmthv3sm5nzlxqqdjfyg2ggkbl5ovfzuvvvsxap4a"
)
PINNED_CATALOG_SHA256: Final = (
    "83df37cb462f7b664cf5dc99d6e577840692e0da319415f5d52e695ad65703f8"
)
PINNED_CATALOG_BYTES: Final = 8111
PINNED_NFC_CID: Final = (
    "baguqeeraaaockar3cte5dpud6wuxfbl4k5nvcmg2hz3ex3coajioewdiaonq"
)
PINNED_NFC_SHA256: Final = (
    "001c25023b14c9d1be83f5a972857c575b5130da3e764bec4e0250e25868039b"
)
PINNED_NFC_BYTES: Final = 178
PINNED_VECTOR_DOCUMENT_CID: Final = (
    "baguqeera6gwcyi3f7fkw5eufnhkovmqw7eevkeas4qxds5ws7b7rloic63da"
)

# Compact exact-byte pins: sha256 of canonical DAG-JSON plus CIDv1.
# Recipes generate the bytes; these pins detect remint or encoder drift.
PINNED_CONTRACT_VECTORS: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        "SupervisorObjectiveIntent": MappingProxyType(
            {
                "byte_length": 231,
                "canonical_sha256": (
                    "5746f304b00ef1b5e5038e4e29be70e0b283ad883ae2ec8b12846d3577eba670"
                ),
                "cid": "baguqeerak5dpgbfqb3y3lzidrzhctptq4czihlmihlrozcysqrwtk57luzya",
            }
        ),
        "ObjectiveMaterializationReceipt": MappingProxyType(
            {
                "byte_length": 245,
                "canonical_sha256": (
                    "29bfb465b63ec68e508673d626fad51ac8e15616d181ff257529e07b374f271c"
                ),
                "cid": "baguqeerafg73iznwh3di4uegoplcn6wvdleocvqw2ga76jlvfhqhwn2pe4oa",
            }
        ),
        "SupervisorContextPack": MappingProxyType(
            {
                "byte_length": 523,
                "canonical_sha256": (
                    "4db47515a46dabce4e26f61d5073f953609fed20eb6bc2bb14681039b23b4e5d"
                ),
                "cid": "baguqeerajw2hkfnenwv44trg6yova47zknqj73ja5nv4foyunaidtmr3jzoq",
            }
        ),
        "SemanticArtifactIdentity": MappingProxyType(
            {
                "byte_length": 257,
                "canonical_sha256": (
                    "35f4e21a37bc0847157f5e0c645b082ab9610fec44d75820608173e2002d7d23"
                ),
                "cid": "baguqeeragx2oegrxxqeeofl7lyggiwyifk4wcd7mitlvqidaqfz6eabnpurq",
            }
        ),
        "DurableArtifactReceipt": MappingProxyType(
            {
                "byte_length": 282,
                "canonical_sha256": (
                    "41b0c07eec3ca34bf9c5a78f48298101d225b6cd6c25f81aa5455082b7281e7a"
                ),
                "cid": "baguqeeraigyma7xmhsrux6ofu6huqkmbahjclnwnnqs7qgvfiviifnzidz5a",
            }
        ),
        "ProofObligation": MappingProxyType(
            {
                "byte_length": 198,
                "canonical_sha256": (
                    "b20cf38692f04b9dfa37dcb0cd46fc32ca4d9fce10548004ef6311d516f82eb3"
                ),
                "cid": "baguqeerawigphbus6bfz36rx3sym2rx4glfe3h6ocbkiabhpmmi5kfxyf2zq",
            }
        ),
        "ProofResult": MappingProxyType(
            {
                "byte_length": 169,
                "canonical_sha256": (
                    "777b2e223902094425f093656f9c6f46e59708e950574231308add2b253d0860"
                ),
                "cid": "baguqeerao55s4irzaieuijpqsnsw7hdpi3szochjkbluemjqrloswjj5bbqa",
            }
        ),
        "ProofAdmissionDecision": MappingProxyType(
            {
                "byte_length": 189,
                "canonical_sha256": (
                    "79480a60bcf0e4e0962018429d8dff676ba68e3f2583a5b28848f866ce7aaa88"
                ),
                "cid": "baguqeerapfeauyf46dsobfradbbj3dp7m5v2ndr7ewb2lmuijd4gntt2vkea",
            }
        ),
        "ExecutionInvocation": MappingProxyType(
            {
                "byte_length": 222,
                "canonical_sha256": (
                    "5f05495b92f55681d8d52d5a3289a85470968814f983f0f36ed52d7db4ab4bd5"
                ),
                "cid": "baguqeeral4cusw4s6vlidwgvfvndfcnikryjncau7gb7b43o2uwx3nfljpkq",
            }
        ),
        "ExecutionReceipt": MappingProxyType(
            {
                "byte_length": 173,
                "canonical_sha256": (
                    "f0d8098747a12008d63ce73dd071631b93e5fdf4c404a8f39c79e3cdb6e4f5bc"
                ),
                "cid": "baguqeera6dmatb2hueqarvr44465a4lddoj6l7puyqckr444phr43nxe6w6a",
            }
        ),
        "SupervisorEvent": MappingProxyType(
            {
                "byte_length": 167,
                "canonical_sha256": (
                    "eb1bdfa51a703be2a726a25c69ae3883f213bb984107278cd29f826c516b42dd"
                ),
                "cid": "baguqeera5mn57ji2oa56fjzgujogtlryqpzbho4yiedspdgst6bgyulliloq",
            }
        ),
        "TaskStateTransition": MappingProxyType(
            {
                "byte_length": 171,
                "canonical_sha256": (
                    "a3d05bf736524f22de8a99be36a865cd41b46c321f18c2c51c0a4d6c22fbfcde"
                ),
                "cid": "baguqeeraupifx5zwkjhsfxuktg7dnkdfzva3i3bsd4mmfri4bjgwyix37tpa",
            }
        ),
        "ReleaseComponentManifest": MappingProxyType(
            {
                "byte_length": 206,
                "canonical_sha256": (
                    "0533aeea98b844f53cc6485c71bf151e3b57b25ffd56fe72b6e915df44cef116"
                ),
                "cid": "baguqeeraauz252uyxbcpkpggjbohdpyvdy5vpms77vlp44vw5ek56rgo6ela",
            }
        ),
        "PortfolioCompatibilityManifest": MappingProxyType(
            {
                "byte_length": 340,
                "canonical_sha256": (
                    "cc1864ed619fa57253237d3ad7c95fdbb83a2d812d40b14f90e0269704f11e98"
                ),
                "cid": "baguqeerazqmgj3lbt6sxeuzdpu5npsk73o4dulmbfvalct4q4atjobhrd2ma",
            }
        ),
    }
)


class CanonicalByteCidVectorError(ValueError):
    """Malformed vector evidence or a forbidden remint."""


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def vector_seed_cid() -> str:
    cid = content_identity(dict(VECTOR_SEED))
    if cid != PINNED_VECTOR_SEED_CID:
        raise CanonicalByteCidVectorError(
            f"vector seed CID {cid} remints {PINNED_VECTOR_SEED_CID}"
        )
    return cid


def vector_artifact_bytes() -> bytes:
    encoded = canonical_json_bytes(dict(VECTOR_ARTIFACT))
    digest = _sha256_hex(encoded)
    if (
        len(encoded) != PINNED_VECTOR_ARTIFACT_BYTES
        or digest != PINNED_VECTOR_ARTIFACT_SHA256
        or content_identity(dict(VECTOR_ARTIFACT)) != PINNED_VECTOR_ARTIFACT_CID
    ):
        raise CanonicalByteCidVectorError("vector artifact identity reminted")
    return encoded


def compact_recipes() -> dict[str, dict[str, Any]]:
    """Compact field recipes. Schema and interface come from the catalog."""

    seed = vector_seed_cid()
    artifact = vector_artifact_bytes()
    catalog = catalog_cid()
    if catalog != PINNED_CATALOG_CID:
        raise CanonicalByteCidVectorError(
            f"catalog CID {catalog} remints {PINNED_CATALOG_CID}"
        )
    return {
        "SupervisorObjectiveIntent": {
            "objective_id": PCPR_041_GOAL_ID,
            "language": "Python",
            "idea_digest": seed,
        },
        "ObjectiveMaterializationReceipt": {
            "objective_id": PCPR_041_GOAL_ID,
            "plan_id": "plan:pcpr-041",
            "admitted": True,
            "current_tree": VECTOR_TREE_OID,
        },
        "SupervisorContextPack": {
            "task_id": PCPR_041_TASK_ID,
            "repository_state_cid": seed,
            "scanned_tree_oid": VECTOR_TREE_OID,
            "target_source_cid": seed,
            "surrounding_source_cid": seed,
            "test_source_cid": seed,
        },
        "SemanticArtifactIdentity": {
            "artifact_id": "semantic:pcpr-041",
            "ir_identity": "ir-canonical-identity-v1",
            "lineage_cid": seed,
        },
        "DurableArtifactReceipt": {
            "cid": PINNED_VECTOR_ARTIFACT_CID,
            "byte_length": len(artifact),
            "digest_hex": PINNED_VECTOR_ARTIFACT_SHA256,
            "durable": True,
        },
        "ProofObligation": {
            "obligation_id": "obligation:pcpr-041",
            "statement": "canonical-byte-and-cid-vector",
            "logic_family": "propositional",
        },
        "ProofResult": {
            "obligation_id": "obligation:pcpr-041",
            "outcome": "unavailable",
            "evidence_kind": "unavailable",
        },
        "ProofAdmissionDecision": {
            "obligation_id": "obligation:pcpr-041",
            "admitted": False,
            "reason": "live-proof-unavailable",
        },
        "ExecutionInvocation": {
            "invocation_id": "invocation:pcpr-041",
            "target": "canonical-byte-cid-vectors",
            "tree_id": VECTOR_TREE_OID,
        },
        "ExecutionReceipt": {
            "invocation_id": "invocation:pcpr-041",
            "outcome": "observed",
            "evidence_kind": "measured",
        },
        "SupervisorEvent": {
            "event_id": "event:pcpr-041",
            "event_type": "canonical-byte-cid-vector",
            "sequence": 1,
        },
        "TaskStateTransition": {
            "task_id": PCPR_041_TASK_ID,
            "from_state": "ready",
            "to_state": "implemented",
            "fence": 1,
        },
        "ReleaseComponentManifest": {
            "component_id": "component:ipfs_accelerate_py",
            "repository": "ipfs_accelerate_py",
            "version": "0.0.0-rnd",
        },
        "PortfolioCompatibilityManifest": {
            "portfolio_id": "portfolio:pcpr-v1",
            "contract_catalog_cid": catalog,
            "component_ids": [
                "component:ipfs_accelerate_py",
                "component:ipfs_datasets_py",
                "component:ipfs_kit_py",
            ],
        },
    }


def _envelope(name: str, extras: Mapping[str, Any]) -> dict[str, Any]:
    spec = CONTRACT_BY_NAME[name]
    payload = {"schema": spec.schema, "interface": spec.interface}
    payload.update(extras)
    return admit_shared_contract(name, payload)


def _vector_record(
    *,
    vector_id: str,
    name: str,
    admitted: Mapping[str, Any],
    invariant: str,
) -> dict[str, Any]:
    encoded = canonical_json_bytes(admitted)
    digest = _sha256_hex(encoded)
    cid = content_identity(admitted)
    pinned = PINNED_CONTRACT_VECTORS.get(name)
    if pinned is not None and vector_id.startswith("contract."):
        if (
            cid != pinned["cid"]
            or digest != pinned["canonical_sha256"]
            or len(encoded) != pinned["byte_length"]
        ):
            raise CanonicalByteCidVectorError(
                f"{name} canonical bytes or CID reminted"
            )
    reordered = canonical_json_bytes(dict(reversed(list(admitted.items()))))
    if reordered != encoded:
        raise CanonicalByteCidVectorError(
            f"{name} canonical bytes changed under key reordering"
        )
    if content_identity(dict(reversed(list(admitted.items())))) != cid:
        raise CanonicalByteCidVectorError(
            f"{name} CID changed under key reordering"
        )
    spec = CONTRACT_BY_NAME[name]
    return {
        "id": vector_id,
        "contract": name,
        "schema": spec.schema,
        "interface": spec.interface,
        "domain": "structured",
        "codec": "dag-json",
        "multihash": "sha2-256",
        "base": "base32",
        "version": 1,
        "byte_length": len(encoded),
        "canonical_hex": encoded.hex(),
        "canonical_sha256": digest,
        "cid": cid,
        "invariant": invariant,
        "recipe": dict(compact_recipes()[name]),
    }


def contract_vector(name: str) -> dict[str, Any]:
    if name not in REQUIRED_CONTRACT_NAMES:
        raise CanonicalByteCidVectorError(f"{name} is not a PCPR shared contract")
    admitted = _envelope(name, compact_recipes()[name])
    return _vector_record(
        vector_id=f"contract.{name}",
        name=name,
        admitted=admitted,
        invariant="canonical DAG-JSON bytes mint one CIDv1 identity",
    )


def catalog_identity_vector() -> dict[str, Any]:
    mapping = catalog_mapping()
    encoded = canonical_json_bytes(mapping)
    digest = _sha256_hex(encoded)
    cid = catalog_cid()
    if (
        cid != PINNED_CATALOG_CID
        or digest != PINNED_CATALOG_SHA256
        or len(encoded) != PINNED_CATALOG_BYTES
        or content_identity(mapping) != cid
    ):
        raise CanonicalByteCidVectorError("shared-contract catalog CID reminted")
    reordered = canonical_json_bytes(dict(reversed(list(mapping.items()))))
    if reordered != encoded or content_identity(
        dict(reversed(list(mapping.items())))
    ) != cid:
        raise CanonicalByteCidVectorError(
            "catalog canonical bytes changed under key reordering"
        )
    return {
        "id": "catalog.identity",
        "contract": "SharedContractCatalog",
        "schema": mapping["schema"],
        "interface": mapping["interface"],
        "domain": "structured",
        "codec": "dag-json",
        "multihash": "sha2-256",
        "base": "base32",
        "version": 1,
        "byte_length": len(encoded),
        "canonical_sha256": digest,
        "cid": cid,
        "invariant": "PCPR-040 catalog identity is not reminted",
        "normative_task": PCPR_040_TASK_ID,
    }


def unicode_nfc_vector() -> dict[str, Any]:
    extras = dict(compact_recipes()["ProofObligation"])
    composed = _envelope("ProofObligation", {**extras, "statement": NFC_COMPOSED})
    decomposed = _envelope(
        "ProofObligation", {**extras, "statement": NFC_DECOMPOSED}
    )
    composed_bytes = canonical_json_bytes(composed)
    decomposed_bytes = canonical_json_bytes(decomposed)
    cid = content_identity(composed)
    if composed_bytes != decomposed_bytes or cid != content_identity(decomposed):
        raise CanonicalByteCidVectorError(
            "NFC composed and decomposed statements mint different identities"
        )
    if composed["statement"] != NFC_COMPOSED:
        raise CanonicalByteCidVectorError("NFC statement is not composed")
    digest = _sha256_hex(composed_bytes)
    if (
        cid != PINNED_NFC_CID
        or digest != PINNED_NFC_SHA256
        or len(composed_bytes) != PINNED_NFC_BYTES
    ):
        raise CanonicalByteCidVectorError(
            f"NFC vector CID {cid} remints {PINNED_NFC_CID}"
        )
    return {
        "id": "invariant.unicode_nfc",
        "contract": "ProofObligation",
        "schema": CONTRACT_BY_NAME["ProofObligation"].schema,
        "interface": CONTRACT_BY_NAME["ProofObligation"].interface,
        "domain": "structured",
        "codec": "dag-json",
        "multihash": "sha2-256",
        "base": "base32",
        "version": 1,
        "byte_length": len(composed_bytes),
        "canonical_hex": composed_bytes.hex(),
        "canonical_sha256": digest,
        "cid": cid,
        "invariant": "NFC composed and decomposed café share canonical bytes and CID",
        "statement_nfc": NFC_COMPOSED,
    }


def key_order_vector() -> dict[str, Any]:
    name = "SupervisorObjectiveIntent"
    admitted = _envelope(name, compact_recipes()[name])
    reversed_payload = dict(reversed(list(admitted.items())))
    record = _vector_record(
        vector_id="invariant.key_order",
        name=name,
        admitted=reversed_payload,
        invariant=(
            "opposite key insertion order yields the same canonical bytes and CID"
        ),
    )
    primary = contract_vector(name)
    if record["cid"] != primary["cid"] or record["canonical_hex"] != primary[
        "canonical_hex"
    ]:
        raise CanonicalByteCidVectorError("key-order invariant broken")
    return record


def refuse_vector_remint(name: str, cid: str) -> str:
    pinned = PINNED_CONTRACT_VECTORS.get(name)
    if pinned is None:
        raise CanonicalByteCidVectorError(f"{name} is not a PCPR vector contract")
    if cid != pinned["cid"]:
        raise CanonicalByteCidVectorError(
            f"{name} vector CID {cid} remints {pinned['cid']}"
        )
    return pinned["cid"]


def vector_cid_map() -> dict[str, str]:
    return {name: item["cid"] for name, item in PINNED_CONTRACT_VECTORS.items()}


def canonical_byte_cid_vector_document() -> dict[str, Any]:
    """Live vector document. Bytes and CIDs must match the compact pins."""

    contract_vectors = [contract_vector(name) for name in REQUIRED_CONTRACT_NAMES]
    catalog = catalog_identity_vector()
    key_order = key_order_vector()
    nfc = unicode_nfc_vector()
    if len(contract_vectors) != 14:
        raise CanonicalByteCidVectorError("expected fourteen contract vectors")
    cids = [item["cid"] for item in contract_vectors]
    if len(set(cids)) != 14:
        raise CanonicalByteCidVectorError("contract vector CIDs are not unique")
    document = {
        "schema": VECTOR_DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_041_TASK_ID,
        "goal_id": PCPR_041_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "board_namespace": PCPR_BOARD_NAMESPACE,
        "evidence_id": EVIDENCE_ID,
        "frozen": False,
        "freeze_task": PCPR_002_TASK_ID,
        "normative_task": PCPR_040_TASK_ID,
        "negative_vector_task": PCPR_042_TASK_ID,
        "compatibility_task": PCPR_043_TASK_ID,
        "catalog_cid": PINNED_CATALOG_CID,
        "vector_seed_cid": PINNED_VECTOR_SEED_CID,
        "contract_count": 14,
        "vector_count": 17,
        "vectors": [*contract_vectors, catalog, key_order, nfc],
        "duckdb_or_quack_state_written": False,
    }
    _ = content_identity(document)
    return document


def vector_document_cid() -> str:
    cid = content_identity(canonical_byte_cid_vector_document())
    if cid != PINNED_VECTOR_DOCUMENT_CID:
        raise CanonicalByteCidVectorError(
            f"vector document CID {cid} remints {PINNED_VECTOR_DOCUMENT_CID}"
        )
    return cid


def decode_and_recompute(vector: Mapping[str, Any]) -> str:
    """Require retained canonical bytes to rehash to the claimed CID."""

    if not isinstance(vector, Mapping):
        raise CanonicalByteCidVectorError("vector must be a mapping")
    claimed = vector.get("cid")
    if not isinstance(claimed, str) or not claimed.startswith("b"):
        raise CanonicalByteCidVectorError("vector cid is missing")
    hex_payload = vector.get("canonical_hex")
    if isinstance(hex_payload, str) and hex_payload:
        retained = bytes.fromhex(hex_payload)
        recomputed = content_identity(json.loads(retained.decode("utf-8")))
        if recomputed != claimed:
            raise CanonicalByteCidVectorError(
                f"retained bytes rehash to {recomputed} not {claimed}"
            )
        if _sha256_hex(retained) != vector.get("canonical_sha256"):
            raise CanonicalByteCidVectorError("canonical_sha256 does not match bytes")
        if len(retained) != vector.get("byte_length"):
            raise CanonicalByteCidVectorError("byte_length does not match bytes")
        return recomputed
    if vector.get("id") == "catalog.identity":
        if catalog_cid() != claimed:
            raise CanonicalByteCidVectorError("catalog identity reminted")
        return claimed
    raise CanonicalByteCidVectorError("vector has no retained canonical bytes")


__all__ = (
    "INTERFACE",
    "PINNED_CATALOG_CID",
    "PINNED_CONTRACT_VECTORS",
    "PINNED_NFC_CID",
    "PINNED_VECTOR_SEED_CID",
    "SCHEMA",
    "CanonicalByteCidVectorError",
    "canonical_byte_cid_vector_document",
    "catalog_identity_vector",
    "compact_recipes",
    "contract_vector",
    "decode_and_recompute",
    "key_order_vector",
    "refuse_vector_remint",
    "unicode_nfc_vector",
    "vector_cid_map",
    "vector_document_cid",
    "vector_seed_cid",
)
