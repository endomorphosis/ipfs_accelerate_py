"""Pure fail-closed tests for the additive PCTDD-005 r26 authority.

The singleton ``@3`` manifest is intentionally separate from the historical
``@1`` and retained-occurrence ``@2`` manifests.  These tests pin all three
content identities and cover only the acyclic manifest-to-credit boundary;
database admission and consumption are exercised by their owning daemon tests.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID,
    DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_OCCURRENCE_FIELDS,
    DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
    DATABASE_PCTDD005_SUCCESSOR_CREDIT_SCHEMA,
    DATABASE_PCTDD005_SUCCESSOR_MANIFEST_ID,
    DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
    DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PREDECESSOR_ID,
    DATABASE_PCTDD005_SUCCESSOR_MANIFEST_SCHEMA,
    DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_ID,
    DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_SCHEMA,
    DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_PINS,
    database_fenced_provider_retained_manifest,
    database_fenced_provider_retained_manifest_valid,
    database_pctdd005_successor_credit,
    database_pctdd005_successor_credit_valid,
    database_pctdd005_successor_manifest,
    database_pctdd005_successor_manifest_valid,
)


def _sha256(value: Any) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _legacy_manifest() -> dict[str, Any]:
    return {
        "schema": (
            DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_SCHEMA
        ),
        "revision": "pctdd-provider-recovery-2026-09-02",
        "operator_owned": True,
        "one_shot": True,
        "occurrences": [
            dict(item)
            for item in DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_PINS
        ],
    }


def test_predecessor_manifest_bytes_and_ids_remain_unchanged() -> None:
    legacy = _legacy_manifest()
    retained = database_fenced_provider_retained_manifest()

    assert _sha256(legacy) == (
        "sha256:3b4e8c471c67839e4ce5e45596065d02a0da180bb8a617c0f7cc1b5ae48bbbe0"
    )
    assert _sha256(legacy) == (
        DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_ID
    )
    assert _sha256(legacy["occurrences"]) == (
        "sha256:f938bc2122bbf56cf101c4bad58a8e37d43ed55a2a0a5d52b51155ee6b6bb762"
    )
    assert _sha256(retained) == (
        "sha256:23b7fb59fcf73c901f2b93e95b3433d2c07704beeb64121ceef911d9b4f775b1"
    )
    assert _sha256(retained) == DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID
    assert _sha256(retained["occurrences"]) == (
        "sha256:e438aa2ad5c703e3e3c3f857f6e378015970260d06372206b7824884aa3482fc"
    )
    assert database_fenced_provider_retained_manifest_valid(retained)
    assert [item["task_alias"] for item in retained["occurrences"]] == [
        "PCTDD-006",
        "PCTDD-007",
        "PCTDD-034",
    ]


def test_pctdd005_successor_manifest_is_exact_singleton_and_content_addressed() -> None:
    manifest = database_pctdd005_successor_manifest()
    pin = dict(DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN)

    assert DATABASE_PCTDD005_SUCCESSOR_MANIFEST_SCHEMA.endswith("manifest@3")
    assert DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PREDECESSOR_ID == (
        DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID
    )
    assert manifest["predecessor_manifest_id"] == (
        DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID
    )
    assert manifest["occurrence_count"] == 1
    assert manifest["occurrences"] == [pin]
    assert set(pin) == DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_OCCURRENCE_FIELDS
    assert pin["task_cid"] == (
        "baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza"
    )
    assert pin["task_alias"] == "PCTDD-005"
    assert pin["blocked_task_revision"] == 26
    assert pin["blocked_task_status"] == "blocked"
    assert pin["predecessor_attempt_id"] == (
        "attempt:b542756073104e90b20eb8bbeda7096a"
    )
    assert pin["predecessor_claim_id"] == (
        "claim:67fd00b2d4df471594d371bd36460b14"
    )
    assert pin["predecessor_lease_id"] == (
        "lease:aa8dce7a8e0c40dfaf1f385bb48f0887"
    )
    assert pin["predecessor_attempt_number"] == 6
    assert pin["predecessor_fencing_token"] == 6
    assert pin["predecessor_fence_epoch"] == 6
    assert pin["recovery_mode"] == (
        "runner_fenced_no_task_edits_candidate_unavailable"
    )
    assert pin["candidate_disposition"] == (
        "no_task_edits_candidate_unavailable"
    )
    assert pin["source_relative_path"] == "external/ipfs_datasets"
    assert pin["clean_baseline_ref"] == (
        "fd38aa56b03bfd19a21ffd465b498d11657428fb"
    )
    assert pin["retained_ref"] == ""
    assert pin["retained_commit"] == ""
    assert pin["retained_worktree_path"] == ""
    assert pin["receipt_nonce"] == "retained-recovery:PCTDD-005:r26"
    assert pin["owner_generation_floor"] == 59
    assert pin["allow_pool"] is False
    assert pin["seed_prior_attempt"] is False
    assert pin["one_shot"] is True
    assert pin["credit_ordinal"] == 1
    assert _sha256(pin) == (
        "sha256:43ca6518ead86b667968c3e9bac84466cc671dddea8a626259f37d2124c22fc1"
    )
    assert _sha256(manifest) == (
        "sha256:75ef62f466ee2608da0bbc3ba5312be70af62c0713ab66ac2472fc843142b63f"
    )
    assert _sha256(manifest) == DATABASE_PCTDD005_SUCCESSOR_MANIFEST_ID
    assert DATABASE_PCTDD005_SUCCESSOR_MANIFEST_ID not in {
        DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_ID,
        DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID,
    }
    assert database_pctdd005_successor_manifest_valid(manifest)

    # Callers receive fresh mutable projections, never the sealed pin itself.
    manifest["occurrences"][0]["task_alias"] = "PCTDD-OTHER"
    assert database_pctdd005_successor_manifest()["occurrences"][0][
        "task_alias"
    ] == "PCTDD-005"


@pytest.mark.parametrize(
    ("target", "field", "replacement"),
    [
        ("manifest", "schema", "unreviewed@3"),
        ("manifest", "predecessor_manifest_id", "sha256:" + "0" * 64),
        ("manifest", "occurrence_count", True),
        ("manifest", "one_shot", False),
        ("occurrence", "task_cid", "baguqeera" + "a" * 52),
        ("occurrence", "blocked_task_revision", True),
        ("occurrence", "owner_generation_floor", 58),
        ("occurrence", "allow_pool", True),
        ("occurrence", "seed_prior_attempt", True),
    ],
)
def test_pctdd005_successor_manifest_tampering_fails_closed(
    target: str,
    field: str,
    replacement: Any,
) -> None:
    manifest = copy.deepcopy(database_pctdd005_successor_manifest())
    if target == "manifest":
        manifest[field] = replacement
    else:
        manifest["occurrences"][0][field] = replacement

    assert not database_pctdd005_successor_manifest_valid(manifest)

    extra = copy.deepcopy(database_pctdd005_successor_manifest())
    extra["unreviewed"] = True
    assert not database_pctdd005_successor_manifest_valid(extra)


def test_pctdd005_successor_credit_is_closed_acyclic_and_exact() -> None:
    pin = dict(DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN)
    credit = database_pctdd005_successor_credit(pin)

    assert credit == {
        "schema": DATABASE_PCTDD005_SUCCESSOR_CREDIT_SCHEMA,
        "manifest_id": DATABASE_PCTDD005_SUCCESSOR_MANIFEST_ID,
        "occurrence": pin,
    }
    assert database_pctdd005_successor_credit_valid(credit)
    assert _sha256(credit) == (
        "sha256:07e2a0839f28f6b8d73451f0b3079350556e0c3a3a0bc68ecb3a374da85199b3"
    )
    encoded = canonical_json(credit)
    for forbidden in (
        "admission_id",
        "consumption_id",
        "inner_receipt_cid",
        "outer_receipt_cid",
    ):
        assert forbidden not in encoded

    with pytest.raises(ValueError, match="exact manifest member"):
        database_pctdd005_successor_credit(
            dict(DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS[0])
        )
    crossed = dict(pin)
    crossed["task_cid"] = DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS[0][
        "task_cid"
    ]
    with pytest.raises(ValueError, match="exact manifest member"):
        database_pctdd005_successor_credit(crossed)

    for field, replacement in (
        ("schema", "unreviewed-credit@3"),
        ("manifest_id", DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID),
    ):
        malformed = copy.deepcopy(credit)
        malformed[field] = replacement
        assert not database_pctdd005_successor_credit_valid(malformed)
    malformed = copy.deepcopy(credit)
    malformed["occurrence"]["blocked_task_revision"] = 27
    assert not database_pctdd005_successor_credit_valid(malformed)
    malformed = copy.deepcopy(credit)
    malformed["unreviewed"] = True
    assert not database_pctdd005_successor_credit_valid(malformed)
