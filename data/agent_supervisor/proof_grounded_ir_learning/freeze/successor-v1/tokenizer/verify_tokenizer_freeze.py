#!/usr/bin/env python3
"""Independently verify the immutable PGIR-203 tokenizer restriction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from build_tokenizer_freeze import (
    FREEZE_DIR,
    MANIFEST_SCHEMA,
    POLICY_SCHEMA,
    RECEIPT_SCHEMA,
    RESULT_SCHEMA,
    TokenizerFreezeError,
    canonical_bytes,
    dag_json_cid,
    expected_artifacts,
    raw_cid,
    rendered_bytes,
)


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise TokenizerFreezeError(f"duplicate key {key!r} in {path.name}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise TokenizerFreezeError(f"{path.name} must contain an object")
    return value


def verify_identity(value: Mapping[str, Any], cid_field: str, sha_field: str) -> None:
    projection = dict(value)
    claimed_cid = projection.pop(cid_field, None)
    claimed_sha = projection.pop(sha_field, None)
    if claimed_cid != dag_json_cid(projection):
        raise TokenizerFreezeError(f"{cid_field} does not match canonical projection")
    expected_sha = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
    if claimed_sha != expected_sha:
        raise TokenizerFreezeError(f"{sha_field} does not match canonical projection")


def verify(expected: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    expected = expected or expected_artifacts()
    actual = {name: strict_json(FREEZE_DIR / name) for name in expected}
    for name, value in expected.items():
        if actual[name] != value:
            raise TokenizerFreezeError(f"sealed artifact differs from deterministic replay: {name}")

    policy = actual["tokenizer_policy.json"]
    receipt = actual["golden_token_class_receipt.json"]
    result = actual["result.json"]
    manifest = actual["manifest.json"]
    if policy.get("schema") != POLICY_SCHEMA or receipt.get("schema") != RECEIPT_SCHEMA:
        raise TokenizerFreezeError("unexpected tokenizer policy or receipt schema")
    if result.get("schema") != RESULT_SCHEMA or manifest.get("schema") != MANIFEST_SCHEMA:
        raise TokenizerFreezeError("unexpected result or manifest schema")
    verify_identity(policy, "policy_cid", "policy_sha256")
    verify_identity(receipt, "receipt_cid", "receipt_sha256")
    verify_identity(result, "result_cid", "result_sha256")
    verify_identity(manifest, "manifest_cid", "manifest_sha256")
    if policy["unknown_token_behavior"] != "fail_closed":
        raise TokenizerFreezeError("unknown token behavior is not fail closed")
    if policy["status"] != "permanently_deterministic_only":
        raise TokenizerFreezeError("learned tokenizer restriction was weakened")
    if policy["training_policy"]["authorized"] or result["training_authorized"]:
        raise TokenizerFreezeError("learned training is unauthorized by this freeze")
    admission = policy["learned_tokenizer_admission"]
    if admission["admission_status"] != "not_admitted" or admission["exact_tokenizer_identity_evidence_passed"]:
        raise TokenizerFreezeError("a learned tokenizer was incorrectly admitted")
    if not admission["golden_token_class_evidence_passed"]:
        raise TokenizerFreezeError("golden token-class evidence is missing")
    if policy["golden_token_class_receipt"]["receipt_cid"] != receipt["receipt_cid"]:
        raise TokenizerFreezeError("policy does not bind the golden receipt")
    if result["policy_cid"] != policy["policy_cid"] or result["golden_token_class_receipt_cid"] != receipt["receipt_cid"]:
        raise TokenizerFreezeError("result does not bind tokenizer evidence")
    for name in ("tokenizer_policy.json", "golden_token_class_receipt.json", "result.json"):
        binding = manifest["artifacts"].get(name)
        if not isinstance(binding, Mapping):
            raise TokenizerFreezeError(f"manifest is missing {name}")
        data = (FREEZE_DIR / name).read_bytes()
        if binding.get("content_cid") != raw_cid(data):
            raise TokenizerFreezeError(f"manifest CID drift for {name}")
        if binding.get("sha256") != "sha256:" + hashlib.sha256(data).hexdigest():
            raise TokenizerFreezeError(f"manifest SHA-256 drift for {name}")
        if binding.get("size_bytes") != len(data):
            raise TokenizerFreezeError(f"manifest size drift for {name}")


def main() -> int:
    verify()
    print("PGIR-203 tokenizer freeze verifies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
