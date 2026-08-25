#!/usr/bin/env python3
"""Create the write-once PGIR-203 deterministic-only tokenizer freeze.

PGIR-203 is permitted to close with a deterministic-only restriction.  This
builder deliberately does that: the canonical LegalIR vocabulary is audited,
but is not misrepresented as an admitted learned tokenizer.  Existing evidence
is never overwritten with different bytes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


FREEZE_DIR = Path(__file__).resolve().parent
# ``FREEZE_DIR`` is the tokenizer directory itself, so its fifth parent is
# the checkout root (rather than the sixth parent of this file path).
REPOSITORY_ROOT = FREEZE_DIR.parents[5]
DATASETS_ROOT = REPOSITORY_ROOT / "ipfs_datasets_py"

POLICY_SCHEMA = "IRTokenizerFreezePolicy@1"
RECEIPT_SCHEMA = "PGIRTokenizerGoldenTokenClassReceipt@1"
RESULT_SCHEMA = "pgir-task-result@1"
MANIFEST_SCHEMA = "PGIRTokenizerFreezeManifest@1"


class TokenizerFreezeError(RuntimeError):
    """Raised when immutable tokenizer evidence cannot be constructed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def rendered_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def dag_json_cid(value: Any) -> str:
    digest = hashlib.sha256(canonical_bytes(value)).digest()
    multihash = b"\x01\xa9\x02\x12\x20" + digest
    return "b" + base64.b32encode(multihash).decode("ascii").rstrip("=").lower()


def raw_cid(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    multihash = b"\x01\x55\x12\x20" + digest
    return "b" + base64.b32encode(multihash).decode("ascii").rstrip("=").lower()


def projection_identity(value: dict[str, Any], *, cid_field: str, sha_field: str) -> dict[str, Any]:
    projection = dict(value)
    result = dict(value)
    result[sha_field] = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
    result[cid_field] = dag_json_cid(projection)
    return result


def file_binding(relative_path: str) -> dict[str, Any]:
    path = REPOSITORY_ROOT / relative_path
    if not path.is_file() or path.is_symlink():
        raise TokenizerFreezeError(f"required evidence is absent or unsafe: {relative_path}")
    data = path.read_bytes()
    return {
        "path": relative_path,
        "raw_cid": raw_cid(data),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _tokenizer() -> Any:
    import sys

    if str(DATASETS_ROOT) not in sys.path:
        sys.path.insert(0, str(DATASETS_ROOT))
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.legal_ir_grammar_decoder import (
        LegalIRFrozenTokenizer,
    )

    return LegalIRFrozenTokenizer.canonical()


def _golden_receipt() -> dict[str, Any]:
    tokenizer = _tokenizer()
    candidate = {
        "family": "deontic",
        "rules": [
            {
                "action": "provide_notice",
                "modality": "obligation",
                "subject": "agency",
            }
        ],
    }
    encoding = tokenizer.encode_canonical(candidate, family="deontic")
    surface = tokenizer.encode_source_surface("The agency shall provide notice.")
    expected_classes = {
        "<pad>": "padding",
        "<bos>": "special",
        "forall": "binder",
        "obligation": "operator",
        "DeonticRule": "type",
        "deontic": "family",
        "proved": "proof",
        "hammer": "tactic",
        "<source_ref>": "source",
        "<id_bucket_00>": "identifier",
        "emit_deontic_rule": "production",
    }
    for piece, token_class in expected_classes.items():
        if tokenizer.require(piece).token_class != token_class:
            raise TokenizerFreezeError(f"golden class drift for {piece!r}")
    if surface.token_class_histogram()["source_surface"] < 1:
        raise TokenizerFreezeError("source-surface token class is not represented")

    rejection_cases: list[dict[str, str]] = []
    checks = (
        (
            "unknown_closed_class_value",
            lambda: tokenizer.encode_canonical(
                {
                    "family": "tdfol",
                    "formulas": [
                        {"arguments": ["x"], "predicate": "Holds", "quantifier": "most"}
                    ],
                },
                family="tdfol",
            ),
        ),
        ("unknown_closed_class_piece", lambda: tokenizer.require("brand_new_family")),
        ("unknown_token_id", lambda: tokenizer.entry_for_id(tokenizer.vocabulary_size)),
        ("frozen_vocabulary_mutation", lambda: tokenizer.add_token("new_piece", "family")),
    )
    for name, action in checks:
        try:
            action()
        except Exception as exc:  # The evidence records the public rejection type and message.
            rejection_cases.append(
                {"case": name, "error_message": str(exc), "error_type": type(exc).__name__}
            )
        else:
            raise TokenizerFreezeError(f"negative golden case unexpectedly passed: {name}")

    payload = {
        "canonical_case": {
            "candidate_ir": candidate,
            "encoding": encoding.to_dict(),
            "family": "deontic",
        },
        "contract_source": file_binding(
            "ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_grammar_decoder.py"
        ),
        "coverage": {
            "declared_token_classes": tokenizer.vocabulary_manifest()["token_classes"],
            "direct_class_assertions": expected_classes,
            "source_surface_assertion": {
                "class": "source_surface",
                "count": surface.token_class_histogram()["source_surface"],
                "separated": surface.source_surface_separated,
            },
        },
        "interface": "proof-grounded-ir-learning/tokenizer-golden-token-class-receipt/v1",
        "negative_cases": rejection_cases,
        "schema": RECEIPT_SCHEMA,
        "status": "passed_for_deterministic_canonical_tokenizer_only",
        "surface_separation_case": surface.to_dict(),
        "unknown_token_behavior": "fail_closed",
        "vocabulary_identity": {
            "frozen": tokenizer.frozen,
            "vocabulary_cid": tokenizer.vocabulary_cid,
            "vocabulary_sha256": tokenizer.vocabulary_sha256,
            "vocabulary_size": tokenizer.vocabulary_size,
        },
    }
    return projection_identity(payload, cid_field="receipt_cid", sha_field="receipt_sha256")


def expected_artifacts() -> dict[str, dict[str, Any]]:
    receipt = _golden_receipt()
    evidence_inputs = {
        "historical_tokenizer_policy": file_binding(
            "data/agent_supervisor/proof_grounded_ir_learning/freeze/tokenizer_policy.json"
        ),
        "successor_corpus_root": file_binding(
            "ipfs_datasets_py/data/ir_learning/corpora/successor-v1/corpus_root.json"
        ),
        "successor_split_root": file_binding(
            "ipfs_datasets_py/data/ir_learning/splits/successor-v1/split_root.json"
        ),
    }
    policy = {
        "canonical_tokenization": {
            "allowed_uses": [
                "contract validation",
                "deterministic compiler/decompiler replay",
                "schema migration testing",
            ],
            "identity": "deterministic-canonical-structure-only",
            "learned_vocabulary_identity": "none",
            "model_checkpoint_identity": "none",
            "vocabulary_identity": receipt["vocabulary_identity"],
        },
        "contract_version": 1,
        "evidence_inputs": evidence_inputs,
        "golden_token_class_receipt": {
            "receipt_cid": receipt["receipt_cid"],
            "status": receipt["status"],
            "vocabulary_cid": receipt["vocabulary_identity"]["vocabulary_cid"],
        },
        "interface": "proof-grounded-ir-learning/tokenizer-freeze-policy/v1",
        "learned_tokenizer_admission": {
            "admission_status": "not_admitted",
            "exact_tokenizer_identity_evidence_passed": False,
            "golden_token_class_evidence_passed": True,
            "reason_codes": [
                "no_learned_tokenizer_candidate_identity",
                "no_rights_admitted_materialized_training_rows",
                "successor_split_is_permanent_no_go",
            ],
        },
        "mutation_policy": "supersede_never_overwrite",
        "schema": POLICY_SCHEMA,
        "status": "permanently_deterministic_only",
        "supersedes": {
            "mode": "new_successor_location_without_historical_freeze_mutation",
            "policy_path": "data/agent_supervisor/proof_grounded_ir_learning/freeze/tokenizer_policy.json",
        },
        "training_policy": {
            "authorization_rule": "all_required_admission_evidence_must_pass",
            "authorized": False,
            "blocked_campaign_stages": ["R2", "R3", "R4", "R5", "R6"],
            "failed_requirements": [
                "learned_tokenizer_admission.admission_status == admitted",
                "learned_tokenizer_admission.exact_tokenizer_identity_evidence_passed == true",
            ],
            "reason": "PGIR-203 has no admitted learned tokenizer, and RESULT(PGIR-201) and RESULT(PGIR-202) seal an empty permanent-no-go training population.",
            "superseding_root_required": True,
        },
        "unknown_token_behavior": "fail_closed",
    }
    policy = projection_identity(policy, cid_field="policy_cid", sha_field="policy_sha256")
    result = {
        "completion_authoritative": False,
        "decision": "permanent_deterministic_only",
        "disposition": "frozen_no_learned_tokenizer_admission",
        "golden_token_class_receipt_cid": receipt["receipt_cid"],
        "policy_cid": policy["policy_cid"],
        "reason_codes": policy["learned_tokenizer_admission"]["reason_codes"],
        "result_identity": "RESULT(PGIR-203)",
        "schema": RESULT_SCHEMA,
        "task_id": "PGIR-203",
        "training_authorized": False,
        "unknown_token_behavior": "fail_closed",
    }
    result = projection_identity(result, cid_field="result_cid", sha_field="result_sha256")
    core = {
        "tokenizer_policy.json": policy,
        "golden_token_class_receipt.json": receipt,
        "result.json": result,
    }
    manifest = {
        "artifacts": {
            name: {
                "content_cid": raw_cid(rendered_bytes(value)),
                "path": name,
                "sha256": "sha256:" + hashlib.sha256(rendered_bytes(value)).hexdigest(),
                "size_bytes": len(rendered_bytes(value)),
            }
            for name, value in core.items()
        },
        "policy_cid": policy["policy_cid"],
        "result_cid": result["result_cid"],
        "schema": MANIFEST_SCHEMA,
        "task_id": "PGIR-203",
        "tokenizer_status": policy["status"],
    }
    core["manifest.json"] = projection_identity(
        manifest, cid_field="manifest_cid", sha_field="manifest_sha256"
    )
    return core


def write_once(path: Path, data: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise TokenizerFreezeError(f"existing artifact is unsafe: {path}")
        if path.read_bytes() != data:
            raise TokenizerFreezeError(f"write-once artifact differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true", help="create absent write-once JSON artifacts")
    args = parser.parse_args()
    artifacts = expected_artifacts()
    if not args.initialize:
        from verify_tokenizer_freeze import verify

        verify(artifacts)
        print("PGIR-203 tokenizer freeze verifies")
        return 0
    for name, value in artifacts.items():
        write_once(FREEZE_DIR / name, rendered_bytes(value))
    print("PGIR-203 tokenizer freeze initialized or already exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
