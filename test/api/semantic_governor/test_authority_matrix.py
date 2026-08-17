"""SCG-005: executable authority consumption matrix tests.

Rejects alternate CID / envelope / store / index / compiler / cache /
provider / profile ownership claims and stale authority pins.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = (
    REPO_ROOT
    / "docs/architecture/semantic_compression_governor_inventory/authority_matrix.json"
)
INVENTORY_DIR = (
    REPO_ROOT / "docs/architecture/semantic_compression_governor_inventory"
)

SCHEMA = "scg/authority-matrix@1"
EVIDENCE_ID = "scg/authority-matrix@1"
INTERFACE_ID = "SemanticGovernorAuthorityMatrix@1"
TASK_ID = "SCG-005"

REQUIRED_CATEGORIES = (
    "cid",
    "envelope",
    "store",
    "index",
    "compiler",
    "cache",
    "provider",
    "profile",
)

REQUIRED_RESPONSIBILITIES = (
    "identity",
    "receipt",
    "state",
    "proof",
    "execution",
    "storage",
)

CANONICAL_OWNERS = {
    "cid": "ipfs_datasets_py",
    "envelope": "Mcp-Plus-Plus",
    "store": "ipfs_kit_py",
    "index": "ipfs_datasets_py",
    "compiler": "ipfs_datasets_py",
    "cache": "ipfs_accelerate_py",
    "provider": "ipfs_accelerate_py",
    "profile": "Mcp-Plus-Plus",
}

PLANNING_PINS = {
    "accelerate_planning": "dfd92b554e662d4312411f2e8e63a52368806f2a",
    "datasets": "1330038f626ef92993f03d46f21e1a57719e9c25",
    "kit": "df2f9cc092456329de9724c45a50c54b410875d1",
    "mcplusplus": "dc3164653a48d059ae9812078359daeafb451c07",
    "incremental_verification_freeze": "8c7800cedc5e1b848367db9952f912428466f8cc",
    "incremental_proof_sealer_program": "7dc8f1422cb7e80757077948dc0785c1aaa4fd25",
}


class AuthorityMatrixError(ValueError):
    """Raised when an ownership claim or authority pin is rejected."""


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AuthorityMatrixError(f"{path} must contain a JSON object")
    return data


def load_matrix(path: Path = MATRIX_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise AuthorityMatrixError(f"authority matrix missing: {path}")
    return _load_json(path)


def load_inventory(name: str) -> dict[str, Any]:
    path = INVENTORY_DIR / f"{name}.json"
    if not path.is_file():
        raise AuthorityMatrixError(f"source inventory missing: {path}")
    return _load_json(path)


def _git_head(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _git_is_ancestor(path: Path, ancestor: str, head: str) -> bool:
    """Return whether ``ancestor`` is an ancestor of ``head`` in ``path``."""

    if not path.exists() or not ancestor or not head:
        return False
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "merge-base",
                "--is-ancestor",
                ancestor,
                head,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return completed.returncode == 0


def validate_matrix_document(matrix: Mapping[str, Any]) -> None:
    """Fail closed on incomplete or contradictory matrix structure."""
    if matrix.get("schema") != SCHEMA:
        raise AuthorityMatrixError(
            f"schema must be {SCHEMA!r}, got {matrix.get('schema')!r}"
        )
    if matrix.get("evidence_id") != EVIDENCE_ID:
        raise AuthorityMatrixError(
            f"evidence_id must be {EVIDENCE_ID!r}, got {matrix.get('evidence_id')!r}"
        )
    if matrix.get("interface_id") != INTERFACE_ID:
        raise AuthorityMatrixError(
            f"interface_id must be {INTERFACE_ID!r}, got {matrix.get('interface_id')!r}"
        )
    if matrix.get("task_id") != TASK_ID:
        raise AuthorityMatrixError(
            f"task_id must be {TASK_ID!r}, got {matrix.get('task_id')!r}"
        )

    categories = matrix.get("ownership_categories")
    if not isinstance(categories, dict):
        raise AuthorityMatrixError("ownership_categories must be an object")
    missing = [c for c in REQUIRED_CATEGORIES if c not in categories]
    if missing:
        raise AuthorityMatrixError(f"missing ownership categories: {missing}")

    for category, expected_owner in CANONICAL_OWNERS.items():
        entry = categories[category]
        if not isinstance(entry, dict):
            raise AuthorityMatrixError(
                f"ownership_categories.{category} must be an object"
            )
        sole = entry.get("sole_owner")
        if sole != expected_owner:
            raise AuthorityMatrixError(
                f"category {category!r} sole_owner must be {expected_owner!r}, "
                f"got {sole!r}"
            )
        forbidden = entry.get("forbidden_owners")
        if not isinstance(forbidden, list) or not forbidden:
            raise AuthorityMatrixError(
                f"category {category!r} must declare forbidden_owners"
            )
        if expected_owner in forbidden:
            raise AuthorityMatrixError(
                f"category {category!r} lists sole_owner among forbidden_owners"
            )
        if entry.get("category") != category:
            raise AuthorityMatrixError(
                f"category field mismatch for {category!r}: {entry.get('category')!r}"
            )

    # Exactly one sole owner per required category key.
    owners_by_category = {
        category: categories[category]["sole_owner"] for category in REQUIRED_CATEGORIES
    }
    if len(owners_by_category) != len(REQUIRED_CATEGORIES):
        raise AuthorityMatrixError("duplicate ownership category keys")

    pins = matrix.get("authority_pins")
    if not isinstance(pins, dict):
        raise AuthorityMatrixError("authority_pins must be an object")
    for pin_id, expected_commit in PLANNING_PINS.items():
        pin = pins.get(pin_id)
        if not isinstance(pin, dict):
            raise AuthorityMatrixError(f"authority pin missing: {pin_id}")
        commit = pin.get("commit")
        if commit != expected_commit:
            raise AuthorityMatrixError(
                f"authority pin {pin_id!r} commit must be {expected_commit!r}, "
                f"got {commit!r}"
            )

    responsibilities = matrix.get("responsibilities")
    if not isinstance(responsibilities, list) or not responsibilities:
        raise AuthorityMatrixError("responsibilities must be a non-empty list")
    seen_resp: set[str] = set()
    for row in responsibilities:
        if not isinstance(row, dict):
            raise AuthorityMatrixError("responsibility rows must be objects")
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            raise AuthorityMatrixError("responsibility id must be a non-empty string")
        if rid in seen_resp:
            raise AuthorityMatrixError(f"duplicate responsibility id: {rid}")
        seen_resp.add(rid)
        resp = row.get("responsibility")
        if resp not in REQUIRED_RESPONSIBILITIES and resp is not None:
            # allow only the closed vocabulary
            raise AuthorityMatrixError(
                f"responsibility {rid!r} uses unknown responsibility {resp!r}"
            )
        if resp is None:
            raise AuthorityMatrixError(f"responsibility {rid!r} missing responsibility")
        owner = row.get("owner")
        if not isinstance(owner, str) or not owner:
            raise AuthorityMatrixError(f"responsibility {rid!r} missing owner")
        category = row.get("category")
        if category not in REQUIRED_CATEGORIES:
            raise AuthorityMatrixError(
                f"responsibility {rid!r} category {category!r} not in matrix"
            )

    covered_resp = {row["responsibility"] for row in responsibilities}
    missing_resp = [r for r in REQUIRED_RESPONSIBILITIES if r not in covered_resp]
    if missing_resp:
        raise AuthorityMatrixError(
            f"responsibilities missing coverage for: {missing_resp}"
        )

    conflict = matrix.get("conflict_policy")
    if not isinstance(conflict, dict):
        raise AuthorityMatrixError("conflict_policy must be an object")
    if conflict.get("new_mcplusplus_profile_allowed") is not False:
        raise AuthorityMatrixError("new_mcplusplus_profile_allowed must be false")
    if conflict.get("local_generic_envelope_allowed") is not False:
        raise AuthorityMatrixError("local_generic_envelope_allowed must be false")
    if conflict.get("new_content_identity_allowed") is not False:
        raise AuthorityMatrixError("new_content_identity_allowed must be false")
    if conflict.get("ivp_merkle_commitment_may_substitute_for_sealer") is not False:
        raise AuthorityMatrixError(
            "ivp_merkle_commitment_may_substitute_for_sealer must be false"
        )

    sources = matrix.get("source_inventories")
    if not isinstance(sources, dict):
        raise AuthorityMatrixError("source_inventories must be an object")
    for name in ("accelerate", "datasets", "kit", "interoperability"):
        if name not in sources:
            raise AuthorityMatrixError(f"source_inventories missing {name}")

    rejection_table = matrix.get("alternate_ownership_rejection_table")
    if not isinstance(rejection_table, list) or not rejection_table:
        raise AuthorityMatrixError(
            "alternate_ownership_rejection_table must be a non-empty list"
        )
    covered_categories = {row.get("category") for row in rejection_table if isinstance(row, dict)}
    missing_cats = [c for c in REQUIRED_CATEGORIES if c not in covered_categories]
    if missing_cats:
        raise AuthorityMatrixError(
            f"rejection table missing categories: {missing_cats}"
        )

    stale_examples = matrix.get("stale_authority_rejection_examples")
    if not isinstance(stale_examples, list) or not stale_examples:
        raise AuthorityMatrixError(
            "stale_authority_rejection_examples must be a non-empty list"
        )


def owner_for(matrix: Mapping[str, Any], category: str) -> str:
    categories = matrix["ownership_categories"]
    if category not in categories:
        raise AuthorityMatrixError(f"unknown ownership category: {category}")
    owner = categories[category]["sole_owner"]
    if not isinstance(owner, str) or not owner:
        raise AuthorityMatrixError(f"category {category!r} has no sole_owner")
    return owner


def assert_ownership_claim(
    matrix: Mapping[str, Any],
    category: str,
    claimed_owner: str,
) -> str:
    """Accept only the sole owner for a category; reject all alternates."""
    if category not in REQUIRED_CATEGORIES:
        raise AuthorityMatrixError(f"unknown ownership category: {category}")
    sole = owner_for(matrix, category)
    entry = matrix["ownership_categories"][category]
    forbidden = set(entry.get("forbidden_owners") or [])
    if claimed_owner != sole:
        raise AuthorityMatrixError(
            f"reject alternate {category} ownership: claimed {claimed_owner!r}, "
            f"sole_owner is {sole!r}"
        )
    if claimed_owner in forbidden:
        raise AuthorityMatrixError(
            f"reject {category} ownership: {claimed_owner!r} is forbidden"
        )
    return sole


def assert_authority_pin(
    matrix: Mapping[str, Any],
    authority_id: str,
    claimed_commit: str | None,
    *,
    claimed_as_released_api: bool | None = None,
) -> str:
    """Accept only the matrix pin commit; reject stale or empty pins."""
    pins = matrix.get("authority_pins")
    if not isinstance(pins, dict) or authority_id not in pins:
        raise AuthorityMatrixError(f"unknown authority pin: {authority_id}")
    pin = pins[authority_id]
    expected = pin.get("commit")
    if not isinstance(expected, str) or len(expected) != 40:
        raise AuthorityMatrixError(
            f"authority pin {authority_id!r} has invalid expected commit"
        )
    if not claimed_commit or not isinstance(claimed_commit, str):
        raise AuthorityMatrixError(
            f"reject stale authority pin {authority_id!r}: commit missing"
        )
    claimed = claimed_commit.strip().lower()
    if len(claimed) != 40 or any(c not in "0123456789abcdef" for c in claimed):
        raise AuthorityMatrixError(
            f"reject stale authority pin {authority_id!r}: "
            f"malformed commit {claimed_commit!r}"
        )
    if claimed != expected.lower():
        raise AuthorityMatrixError(
            f"reject stale authority pin {authority_id!r}: "
            f"claimed {claimed_commit!r}, expected {expected!r}"
        )
    if authority_id == "incremental_proof_sealer_program":
        if pin.get("is_released_public_api") is True:
            raise AuthorityMatrixError(
                "reject sealer observation pin marked as released public API"
            )
        if claimed_as_released_api is True:
            raise AuthorityMatrixError(
                "reject stale sealer authority: observation pin is not released API"
            )
    return expected


def cross_check_inventories(matrix: Mapping[str, Any]) -> None:
    """Join matrix pins and owners against SCG-001..004 inventory artifacts."""
    accelerate = load_inventory("accelerate")
    datasets = load_inventory("datasets")
    kit = load_inventory("kit")
    interop = load_inventory("interoperability")

    if accelerate.get("schema") != "scg/accelerate-inventory@1":
        raise AuthorityMatrixError("accelerate inventory schema mismatch")
    if datasets.get("schema") != "scg/datasets-inventory@1":
        raise AuthorityMatrixError("datasets inventory schema mismatch")
    if kit.get("schema") != "scg/kit-inventory@1":
        raise AuthorityMatrixError("kit inventory schema mismatch")
    if interop.get("schema") != "scg/mcplusplus-boundary@1":
        raise AuthorityMatrixError("interoperability inventory schema mismatch")

    pins = matrix["authority_pins"]

    acc_planning = accelerate["repository"]["planning_authority_revision"]
    assert_authority_pin(matrix, "accelerate_planning", acc_planning)

    ds_commit = datasets["authority"]["commit"]
    assert_authority_pin(matrix, "datasets", ds_commit)

    nested = accelerate.get("nested_gitlinks_observed") or {}
    assert_authority_pin(matrix, "datasets", nested.get("ipfs_datasets_py"))
    assert_authority_pin(matrix, "kit", nested.get("ipfs_kit_py"))
    assert_authority_pin(
        matrix, "mcplusplus", nested.get("ipfs_accelerate_py/mcplusplus")
    )

    kit_commit = kit["repository"]["planning_bound_revision"]
    assert_authority_pin(matrix, "kit", kit_commit)
    if kit["repository"].get("observed_revision"):
        assert_authority_pin(matrix, "kit", kit["repository"]["observed_revision"])

    mcpp_commit = interop["mcplusplus_authorities"]["shared_wire_and_conformance"][
        "commit"
    ]
    assert_authority_pin(matrix, "mcplusplus", mcpp_commit)

    freeze = accelerate["repository"]["incremental_verification_freeze_revision"]
    assert_authority_pin(matrix, "incremental_verification_freeze", freeze)

    # Ownership join: datasets content identity and index/compiler claims.
    identity_module = datasets["canonical_identity"]["authority_module"]
    if identity_module != matrix["ownership_categories"]["cid"]["owner_module"]:
        raise AuthorityMatrixError(
            "cid owner_module diverges from datasets canonical_identity"
        )
    if "content" not in identity_module:
        raise AuthorityMatrixError("datasets identity module unexpected")

    compiler_note = datasets["conflict_policy"]["semantic_capsule_compiler"]
    if "must not be re-created" not in compiler_note:
        raise AuthorityMatrixError("datasets compiler conflict policy missing")

    # Kit store ownership join.
    backing = kit["acceptance"]["backing_primitive"]["primary"]
    if backing != "DurableCoordinationStore":
        raise AuthorityMatrixError("kit backing primitive mismatch")

    # Interop profile and sealer joins.
    conflict = interop["conflict_policy"]
    if conflict.get("new_mcplusplus_profile_allowed") is not False:
        raise AuthorityMatrixError("interop allows new MCP++ profile")
    if conflict.get("local_generic_envelope_allowed") is not False:
        raise AuthorityMatrixError("interop allows local generic envelope")
    if conflict.get("ivp_merkle_commitment_may_substitute_for_sealer") is not False:
        raise AuthorityMatrixError("interop allows IVP sealer substitution")

    sealer = matrix.get("sealer_capability_gating")
    if not isinstance(sealer, dict):
        raise AuthorityMatrixError("sealer_capability_gating missing")
    if sealer.get("missing_disposition") != "typed_unavailable":
        raise AuthorityMatrixError("sealer missing_disposition must be typed_unavailable")
    if sealer["ivp_merkle_commitment"].get("may_substitute_for_sealer") is not False:
        raise AuthorityMatrixError("matrix allows IVP to substitute for sealer")
    if sealer["full_checkpoint_seal"].get("status") != "typed_unavailable":
        raise AuthorityMatrixError("full checkpoint seal must be typed_unavailable")
    if sealer["delta_or_incremental_seal"].get("status") != "typed_unavailable":
        raise AuthorityMatrixError("delta seal must be typed_unavailable")

    # Gitlink live heads must match the planning pin or be a descendant of it.
    # SCG implementation advances nested gitlinks after the inventory snapshot;
    # unrelated tips remain stale.
    for authority_id, gitlink in (
        ("datasets", REPO_ROOT / "ipfs_datasets_py"),
        ("kit", REPO_ROOT / "ipfs_kit_py"),
        ("mcplusplus", REPO_ROOT / "ipfs_accelerate_py" / "mcplusplus"),
    ):
        head = _git_head(gitlink)
        if head is None:
            continue
        expected = str(pins[authority_id]["commit"])
        if head.lower() == expected.lower():
            continue
        if _git_is_ancestor(gitlink, expected, head):
            continue
        raise AuthorityMatrixError(
            f"reject stale authority pin {authority_id!r}: "
            f"claimed {head!r}, expected {expected!r} or a descendant"
        )

    # Source paths declared for critical categories must exist.
    for category in REQUIRED_CATEGORIES:
        paths = matrix["ownership_categories"][category].get("owner_source_paths") or []
        if not paths:
            raise AuthorityMatrixError(
                f"category {category!r} missing owner_source_paths"
            )
        for rel in paths:
            absolute = REPO_ROOT / rel
            if not absolute.exists():
                raise AuthorityMatrixError(
                    f"category {category!r} owner_source_path missing: {rel}"
                )


def reject_table_claims(matrix: Mapping[str, Any]) -> None:
    table = matrix["alternate_ownership_rejection_table"]
    for row in table:
        category = row["category"]
        claimed = row["claimed_owner"]
        with pytest.raises(AuthorityMatrixError):
            assert_ownership_claim(matrix, category, claimed)
        if row.get("disposition") != "reject":
            raise AuthorityMatrixError(
                f"rejection table row for {category}/{claimed} must disposition=reject"
            )


def reject_stale_examples(matrix: Mapping[str, Any]) -> None:
    for row in matrix["stale_authority_rejection_examples"]:
        authority_id = row["authority_id"]
        if row.get("claimed_as_released_api") is True:
            with pytest.raises(AuthorityMatrixError):
                assert_authority_pin(
                    matrix,
                    authority_id,
                    matrix["authority_pins"][authority_id]["commit"],
                    claimed_as_released_api=True,
                )
            continue
        claimed = row.get("claimed_commit")
        with pytest.raises(AuthorityMatrixError):
            assert_authority_pin(matrix, authority_id, claimed)
        if row.get("disposition") != "reject":
            raise AuthorityMatrixError(
                f"stale pin example for {authority_id} must disposition=reject"
            )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matrix() -> dict[str, Any]:
    data = load_matrix()
    validate_matrix_document(data)
    return data


# ---------------------------------------------------------------------------
# Document / join tests
# ---------------------------------------------------------------------------


def test_matrix_file_exists_and_is_object() -> None:
    assert MATRIX_PATH.is_file()
    data = load_matrix()
    assert isinstance(data, dict)
    assert data["schema"] == SCHEMA
    assert data["evidence_id"] == EVIDENCE_ID
    assert data["interface_id"] == INTERFACE_ID
    assert data["task_id"] == TASK_ID


def test_validate_matrix_document(matrix: dict[str, Any]) -> None:
    validate_matrix_document(matrix)


def test_source_inventories_exist_and_match_matrix(matrix: dict[str, Any]) -> None:
    sources = matrix["source_inventories"]
    for name, meta in sources.items():
        path = REPO_ROOT / meta["path"]
        assert path.is_file(), path
        inv = _load_json(path)
        assert inv["schema"] == meta["schema"]
        assert inv.get("evidence_id", meta["evidence_id"]) == meta["evidence_id"]
        assert inv["task_id"] == meta["task_id"]


def test_cross_check_inventories_and_live_gitlinks(matrix: dict[str, Any]) -> None:
    cross_check_inventories(matrix)


def test_live_gitlink_descendant_of_planning_pin_is_accepted(
    matrix: dict[str, Any],
) -> None:
    datasets_head = _git_head(REPO_ROOT / "ipfs_datasets_py")
    kit_head = _git_head(REPO_ROOT / "ipfs_kit_py")
    if datasets_head is None or kit_head is None:
        pytest.skip("nested gitlinks are not checked out")
    assert _git_is_ancestor(
        REPO_ROOT / "ipfs_datasets_py",
        PLANNING_PINS["datasets"],
        datasets_head,
    )
    assert _git_is_ancestor(
        REPO_ROOT / "ipfs_kit_py",
        PLANNING_PINS["kit"],
        kit_head,
    )
    cross_check_inventories(matrix)


def test_one_owner_per_required_category(matrix: dict[str, Any]) -> None:
    for category, expected in CANONICAL_OWNERS.items():
        assert owner_for(matrix, category) == expected
        assert_ownership_claim(matrix, category, expected)


def test_responsibilities_cover_identity_receipt_state_proof_execution_storage(
    matrix: dict[str, Any],
) -> None:
    covered = {row["responsibility"] for row in matrix["responsibilities"]}
    assert set(REQUIRED_RESPONSIBILITIES).issubset(covered)
    # Each responsibility row has exactly one owner string.
    for row in matrix["responsibilities"]:
        assert isinstance(row["owner"], str) and row["owner"]


def test_conflict_policy_fail_closed(matrix: dict[str, Any]) -> None:
    policy = matrix["conflict_policy"]
    assert policy["admission"] == "fail_closed"
    assert policy["alternate_owner_disposition"] == "reject"
    assert policy["stale_pin_disposition"] == "reject"
    assert policy["new_mcplusplus_profile_allowed"] is False
    assert policy["local_generic_envelope_allowed"] is False
    assert policy["new_content_identity_allowed"] is False
    assert policy["new_receipt_format_allowed"] is False
    assert policy["ivp_merkle_commitment_may_substitute_for_sealer"] is False


# ---------------------------------------------------------------------------
# Alternate ownership rejection (acceptance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,claimed_owner",
    [
        ("cid", "ipfs_accelerate_py"),
        ("cid", "ipfs_kit_py"),
        ("cid", "Mcp-Plus-Plus"),
        ("cid", "scg_local"),
        ("envelope", "scg_local"),
        ("envelope", "semantic_compression_governor"),
        ("envelope", "local_generic_envelope"),
        ("store", "ipfs_datasets_py"),
        ("store", "ipfs_accelerate_py"),
        ("store", "Mcp-Plus-Plus"),
        ("store", "scg_local"),
        ("index", "ipfs_accelerate_py"),
        ("index", "ipfs_kit_py"),
        ("index", "Mcp-Plus-Plus"),
        ("index", "scg_local"),
        ("compiler", "ipfs_accelerate_py"),
        ("compiler", "ipfs_kit_py"),
        ("compiler", "scg_local"),
        ("compiler", "semantic_compression_governor"),
        ("cache", "ipfs_datasets_py"),
        ("cache", "ipfs_kit_py"),
        ("cache", "Mcp-Plus-Plus"),
        ("provider", "ipfs_datasets_py"),
        ("provider", "ipfs_kit_py"),
        ("provider", "Mcp-Plus-Plus"),
        ("profile", "semantic_compression_governor"),
        ("profile", "scg_local"),
        ("profile", "ipfs_datasets_py_as_profile_definer"),
    ],
)
def test_reject_alternate_ownership_claims(
    matrix: dict[str, Any],
    category: str,
    claimed_owner: str,
) -> None:
    with pytest.raises(AuthorityMatrixError, match="reject alternate|forbidden"):
        assert_ownership_claim(matrix, category, claimed_owner)


def test_reject_table_covers_all_categories(matrix: dict[str, Any]) -> None:
    reject_table_claims(matrix)


def test_accepts_only_canonical_owners(matrix: dict[str, Any]) -> None:
    for category, owner in CANONICAL_OWNERS.items():
        assert assert_ownership_claim(matrix, category, owner) == owner


def test_mutated_sole_owner_fails_document_validation(matrix: dict[str, Any]) -> None:
    bad = copy.deepcopy(matrix)
    bad["ownership_categories"]["cid"]["sole_owner"] = "ipfs_kit_py"
    with pytest.raises(AuthorityMatrixError, match="sole_owner"):
        validate_matrix_document(bad)


def test_duplicate_responsibility_id_fails(matrix: dict[str, Any]) -> None:
    bad = copy.deepcopy(matrix)
    bad["responsibilities"].append(copy.deepcopy(bad["responsibilities"][0]))
    with pytest.raises(AuthorityMatrixError, match="duplicate responsibility"):
        validate_matrix_document(bad)


# ---------------------------------------------------------------------------
# Stale authority pin rejection (acceptance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "authority_id,claimed_commit",
    [
        ("datasets", "0000000000000000000000000000000000000000"),
        ("datasets", "ffffffffffffffffffffffffffffffffffffffff"),
        ("kit", "05ba937500000000000000000000000000000000"),
        ("kit", "df2f9cc092456329de9724c45a50c54b410875d0"),  # off-by-one
        ("mcplusplus", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("accelerate_planning", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
        (
            "incremental_verification_freeze",
            "1111111111111111111111111111111111111111",
        ),
        ("incremental_proof_sealer_program", "2222222222222222222222222222222222222222"),
    ],
)
def test_reject_stale_authority_pins(
    matrix: dict[str, Any],
    authority_id: str,
    claimed_commit: str,
) -> None:
    with pytest.raises(AuthorityMatrixError, match="reject stale authority pin"):
        assert_authority_pin(matrix, authority_id, claimed_commit)


@pytest.mark.parametrize(
    "authority_id,claimed_commit",
    [
        ("datasets", None),
        ("kit", ""),
        ("mcplusplus", "not-a-commit"),
        ("accelerate_planning", "dfd92b55"),  # truncated
    ],
)
def test_reject_missing_or_malformed_authority_pins(
    matrix: dict[str, Any],
    authority_id: str,
    claimed_commit: str | None,
) -> None:
    with pytest.raises(AuthorityMatrixError, match="reject stale authority pin"):
        assert_authority_pin(matrix, authority_id, claimed_commit)


def test_reject_sealer_observation_claimed_as_released_api(
    matrix: dict[str, Any],
) -> None:
    pin = matrix["authority_pins"]["incremental_proof_sealer_program"]
    with pytest.raises(AuthorityMatrixError, match="not released API"):
        assert_authority_pin(
            matrix,
            "incremental_proof_sealer_program",
            pin["commit"],
            claimed_as_released_api=True,
        )


def test_stale_examples_in_matrix_are_enforced(matrix: dict[str, Any]) -> None:
    reject_stale_examples(matrix)


def test_accepts_canonical_authority_pins(matrix: dict[str, Any]) -> None:
    for authority_id, commit in PLANNING_PINS.items():
        assert assert_authority_pin(matrix, authority_id, commit) == commit


def test_mutated_matrix_pin_fails_validation(matrix: dict[str, Any]) -> None:
    bad = copy.deepcopy(matrix)
    bad["authority_pins"]["datasets"][
        "commit"
    ] = "0000000000000000000000000000000000000000"
    with pytest.raises(AuthorityMatrixError, match="authority pin"):
        validate_matrix_document(bad)


def test_cross_check_rejects_inventory_with_stale_datasets_pin(
    matrix: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = copy.deepcopy(load_inventory("datasets"))
    stale["authority"]["commit"] = "0000000000000000000000000000000000000000"

    original = load_inventory

    def _fake_load(name: str) -> dict[str, Any]:
        if name == "datasets":
            return stale
        return original(name)

    # Patch the name used by cross_check_inventories in this module.
    monkeypatch.setattr(
        sys.modules[__name__],
        "load_inventory",
        _fake_load,
    )
    with pytest.raises(AuthorityMatrixError, match="reject stale authority pin"):
        cross_check_inventories(matrix)


# ---------------------------------------------------------------------------
# Sealer / forbidden reimplementation gates
# ---------------------------------------------------------------------------


def test_sealer_capability_gating(matrix: dict[str, Any]) -> None:
    sealer = matrix["sealer_capability_gating"]
    assert sealer["full_checkpoint_seal"]["status"] == "typed_unavailable"
    assert sealer["delta_or_incremental_seal"]["status"] == "typed_unavailable"
    assert sealer["missing_disposition"] == "typed_unavailable"
    assert sealer["ivp_merkle_commitment"]["is_zero_knowledge_proof"] is False
    assert sealer["ivp_merkle_commitment"]["is_proof_sealer"] is False
    assert sealer["ivp_merkle_commitment"]["may_substitute_for_sealer"] is False


def test_forbidden_reimplementations_cover_acceptance_categories(
    matrix: dict[str, Any],
) -> None:
    rows = matrix["forbidden_reimplementations"]
    covered = {row["category"] for row in rows}
    for category in REQUIRED_CATEGORIES:
        assert category in covered, f"missing forbidden reimplementation for {category}"


def test_profile_g_is_existing_not_new_scg_profile(matrix: dict[str, Any]) -> None:
    profile = matrix["ownership_categories"]["profile"]
    assert profile["profile_g_is_existing_authority"] is True
    assert profile["profile_g_is_new_scg_profile"] is False
    assert profile["new_scg_profile_allowed"] is False
    admitted = set(profile["admitted_profiles"])
    assert admitted == {"profile_a", "profile_b", "profile_f", "profile_g"}


def test_compiler_is_interface_identifier_not_public_class(
    matrix: dict[str, Any],
) -> None:
    compiler = matrix["ownership_categories"]["compiler"]
    assert compiler["interface_identifier_not_public_class"] is True
    assert "SemanticCapsuleCompiler@1" in compiler["interfaces"]
    assert "recreate_SemanticCapsuleCompiler_public_class" in compiler["forbidden_claims"]


def test_acceptance_block_matches_task_contract(matrix: dict[str, Any]) -> None:
    acceptance = matrix["acceptance"]
    assert set(acceptance["required_categories"]) == set(REQUIRED_CATEGORIES)
    assert acceptance["one_owner_per_category"] is True
    assert "alternate" in acceptance["criterion"].lower()
    assert "stale" in acceptance["criterion"].lower()
    for task in ("SCG-001", "SCG-002", "SCG-003", "SCG-004"):
        assert task in acceptance["source_inventories_required"]
