"""ASEH-060: one writable owner per mutable fact, complete migration dispositions."""

from __future__ import annotations

import copy
import importlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control.task_transition_service import (
    PRODUCTION_CUTOVER_DEFERRED_TO,
    TaskTransitionService,
    TransitionAuthorityWarning,
    TransitionCompatibilityBypassError,
)
from ipfs_accelerate_py.agent_supervisor.merge.patch_admission import PatchAdmission, ValidationReceipt
from ipfs_accelerate_py.agent_supervisor.merge.patch_plan import PatchPlan, PatchScope
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    ControlPlaneContractError,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


ROOT = Path(__file__).resolve().parents[4]
MATRIX_PATH = (
    ROOT
    / "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory"
    / "migration_matrix.json"
)

SCHEMA = "ipfs_accelerate_py/agent-supervisor/aseh-authority-disposition@1"
PROGRAM_ID = "agent-supervisor-efficiency-and-state-hardening-v1"
TASK_ID = "ASEH-060"
OBJECTIVE_ID = "ASEH-G070"
PLAN_REVISION = "ASEH-PLAN-R1"

REQUIRED_FACT_IDS = (
    "task_objective_state",
    "routing",
    "context",
    "reuse",
    "repair",
    "merge",
    "promotion",
    "recovery",
    "receipts",
)
VALID_DISPOSITIONS = frozenset({"canonical", "adapter", "deprecated", "fixture", "removed"})
BYPASS_OUTCOMES = frozenset({"fail_closed", "warn_then_fail"})
REQUIRED_OWNER_FIELDS = {
    "id",
    "path",
    "interface",
    "disposition",
    "writable",
    "store",
    "bypass",
    "deletion_supported",
    "evidence",
}
REQUIRED_DUPLICATE_FIELDS = {
    "id",
    "path",
    "interface",
    "disposition",
    "writable",
    "bypass",
    "replacement",
    "deletion_supported",
    "evidence",
}
HARD_FAILURES = frozenset(
    {
        "two_writers",
        "silent_legacy_fallback",
        "pseudo_cid_or_mock_capability",
        "unclassified_duplicate",
        "unsupported_deletion",
    }
)
CID_RE = re.compile(r"^b[a-z2-7]{20,}$")
PSEUDO_CID_RE = re.compile(
    r"(?i)\b(?:Qm[1-9A-HJ-NP-Za-km-z]{44}|cid:[A-Za-z0-9_-]+|mock[_-]?cid|fake[_-]?cid|"
    r"pseudo[_-]?cid|test[_-]?cid|uuid:[0-9a-f-]{36})\b"
)
MOCK_CAPABILITY_RE = re.compile(
    r"(?i)(?:mock|fake|simulated|stub|pseudo)[_-]?(?:capability|provider|cid|prover)"
)
SILENT_BYPASS_VALUES = frozenset(
    {"", "silent", "warn_only", "ignore", "allow", "passthrough", "compat"}
)


def _load() -> dict[str, Any]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _rebind_identity(payload: dict[str, Any]) -> dict[str, Any]:
    payload.pop("identity", None)
    payload["identity"] = content_identity(payload["authority_fingerprint"])
    return payload


def _fact_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    owner = payload["canonical_owner"]
    records = [owner]
    records.extend(payload.get("duplicates") or [])
    return records


def _validate_matrix(payload: dict[str, Any]) -> None:
    assert payload["schema"] == SCHEMA
    assert payload["program_id"] == PROGRAM_ID
    assert payload["task_id"] == TASK_ID
    assert payload["objective_id"] == OBJECTIVE_ID
    assert payload["plan_revision"] == PLAN_REVISION
    assert payload["authority"] is False
    assert payload["public_deletion"] is False
    assert payload["policy_promotion"] is False
    assert "no public deletion" in payload["authority_requirement"]
    assert "ASEH-061" in payload["production_cutover"]["deferred_to"]

    cid_profile = payload["cid_profile"]
    assert cid_profile["algorithm"] == "cidv1_base32_sha2_256"
    assert cid_profile["producer"].endswith("content_identity")
    assert "placeholder_identity" in cid_profile["forbidden_forms"]

    vocabulary = payload["vocabulary"]
    assert set(vocabulary["dispositions"]) == VALID_DISPOSITIONS
    assert set(vocabulary["bypass_outcomes"]) == BYPASS_OUTCOMES
    assert tuple(vocabulary["required_fact_ids"]) == REQUIRED_FACT_IDS
    assert set(vocabulary["required_owner_fields"]) == REQUIRED_OWNER_FIELDS
    assert set(vocabulary["required_duplicate_fields"]) == REQUIRED_DUPLICATE_FIELDS

    assert set(payload["hard_failures"]) == HARD_FAILURES
    assert payload["nonclaims"], "migration matrix must record nonclaims"
    for note in payload["nonclaims"]:
        assert isinstance(note, str) and note.strip()

    fingerprint = payload["authority_fingerprint"]
    assert fingerprint["schema"] == SCHEMA
    assert fingerprint["task_id"] == TASK_ID
    identity = content_identity(fingerprint)
    assert isinstance(identity, str) and CID_RE.match(identity), identity
    assert not identity.startswith("Qm"), "CIDv0/multihash identities are pseudo-CIDs here"
    if "identity" in payload:
        assert payload["identity"] == identity
    payload = {**payload, "identity": identity}

    bindings = payload["capability_bindings"]
    assert set(bindings) == set(REQUIRED_FACT_IDS)
    for fact_id, binding in bindings.items():
        assert set(binding) == {"module", "symbol", "expected"}, fact_id
        module = importlib.import_module(binding["module"])
        symbol = getattr(module, binding["symbol"])
        assert symbol == binding["expected"], fact_id
        assert MOCK_CAPABILITY_RE.search(str(symbol)) is None, fact_id

    facts = payload["mutable_facts"]
    assert isinstance(facts, list) and facts
    by_id = {fact["fact_id"]: fact for fact in facts}
    assert tuple(fact["fact_id"] for fact in facts) == REQUIRED_FACT_IDS
    assert set(by_id) == set(REQUIRED_FACT_IDS)
    assert len(by_id) == len(facts), "duplicate fact_id hides an unclassified concern"
    assert fingerprint["owners"] == {
        fact["fact_id"]: fact["writable_owner_id"] for fact in facts
    }

    for fact_id in REQUIRED_FACT_IDS:
        fact = by_id[fact_id]
        assert fact["title"], fact_id
        owner = fact["canonical_owner"]
        duplicates = fact["duplicates"]
        assert isinstance(duplicates, list) and duplicates, fact_id
        assert REQUIRED_OWNER_FIELDS <= set(owner), fact_id
        assert owner["id"] == fact["writable_owner_id"], fact_id
        assert owner["disposition"] == "canonical", fact_id
        assert owner["writable"] is True, fact_id
        assert owner["bypass"] in BYPASS_OUTCOMES, fact_id
        assert owner["deletion_supported"] is False, fact_id
        assert isinstance(owner["store"], str) and owner["store"], fact_id
        assert isinstance(owner["evidence"], str) and owner["evidence"], fact_id
        _assert_live_path(owner["path"])
        if "host_path" in owner:
            _assert_live_path(owner["host_path"])

        seen_ids: set[str] = {owner["id"]}
        seen_paths: set[str] = {owner["path"]}
        writable_ids = [owner["id"]]
        for duplicate in duplicates:
            assert set(duplicate) == REQUIRED_DUPLICATE_FIELDS, f"{fact_id}:{duplicate.get('id')}"
            dup_id = duplicate["id"]
            assert dup_id and dup_id not in seen_ids, f"{fact_id}:{dup_id}"
            seen_ids.add(dup_id)
            path = duplicate["path"]
            assert path, f"{fact_id}:{dup_id}"
            if duplicate["disposition"] != "removed":
                assert path not in seen_paths, f"{fact_id}:{path}"
                seen_paths.add(path)
                _assert_live_path(path)
            assert duplicate["interface"], f"{fact_id}:{dup_id}"
            assert duplicate["disposition"] in VALID_DISPOSITIONS, f"{fact_id}:{dup_id}"
            assert duplicate["disposition"] != "canonical", f"{fact_id}:{dup_id}"
            assert duplicate["writable"] is False, f"{fact_id}:{dup_id} is a second writer"
            assert duplicate["bypass"] in BYPASS_OUTCOMES, f"{fact_id}:{dup_id}"
            assert duplicate["bypass"] not in SILENT_BYPASS_VALUES, f"{fact_id}:{dup_id}"
            assert duplicate["replacement"], f"{fact_id}:{dup_id}"
            assert duplicate["evidence"], f"{fact_id}:{dup_id}"
            if duplicate["disposition"] == "removed":
                assert duplicate["deletion_supported"] is True, (
                    f"{fact_id}:{dup_id} unsupported deletion"
                )
            else:
                assert duplicate["deletion_supported"] is False, (
                    f"{fact_id}:{dup_id} unsupported deletion"
                )

        assert writable_ids == [owner["id"]], f"{fact_id} has two writers: {writable_ids}"
        classified = {duplicate["disposition"] for duplicate in duplicates}
        assert classified <= VALID_DISPOSITIONS, fact_id
        assert "removed" not in classified, (
            f"{fact_id} classifies a public path as removed without ASEH-061 replacement proof"
        )

        for related in fact.get("related_authorities") or []:
            assert related["id"]
            _assert_live_path(related["path"])
            assert related["owns"]
            assert related["must_not_own"]

    scanned = {
        payload["identity"],
        payload["cid_profile"]["producer"],
    }
    scanned.update(
        value
        for fact in payload["mutable_facts"]
        for record in _fact_records(fact)
        for value in (record["id"], record["path"], record["interface"])
    )
    scanned.update(binding["expected"] for binding in payload["capability_bindings"].values())
    scanned.update(
        value
        for fact in payload["mutable_facts"]
        for related in fact.get("related_authorities") or []
        for value in (related["id"], related["path"], related["interface"])
    )
    scanned.discard("")
    for text in scanned:
        assert PSEUDO_CID_RE.search(text) is None, text
        assert MOCK_CAPABILITY_RE.search(text) is None, text


def _assert_live_path(relative: str) -> None:
    path = ROOT / relative
    assert path.is_file(), relative


def test_matrix_schema_single_writable_owner_and_complete_dispositions() -> None:
    _validate_matrix(_load())


def test_capability_bindings_import_real_current_tree_interfaces() -> None:
    payload = _load()
    bindings = payload["capability_bindings"]
    from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_synthesis import (
        DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack_selector import (
        CONTEXT_PACK_SELECTOR_INTERFACE,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
        CANONICAL_ROUTING_AUTHORITY,
        MODEL_ROUTING_INTERFACE,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_INTERFACE,
    )

    assert bindings["task_objective_state"]["expected"] == TYPED_STATE_OWNER_INTERFACE
    assert bindings["routing"]["expected"] == MODEL_ROUTING_INTERFACE == CANONICAL_ROUTING_AUTHORITY
    assert bindings["context"]["expected"] == CONTEXT_PACK_SELECTOR_INTERFACE
    assert bindings["repair"]["expected"] == DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE
    assert "mock" not in TYPED_STATE_OWNER_INTERFACE.lower()
    assert CID_RE.match(content_identity({"capability": TYPED_STATE_OWNER_INTERFACE}))


def test_two_writers_unclassified_duplicate_and_silent_bypass_fail_closed() -> None:
    payload = _load()

    two_writers = copy.deepcopy(payload)
    two_writers["mutable_facts"][0]["duplicates"][0]["writable"] = True
    with pytest.raises(AssertionError, match="second writer"):
        _validate_matrix(_rebind_identity(two_writers))

    unclassified = copy.deepcopy(payload)
    extra = copy.deepcopy(payload["mutable_facts"][0]["duplicates"][0])
    extra["id"] = "unclassified-shadow-writer"
    extra["path"] = "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py"
    extra.pop("disposition")
    unclassified["mutable_facts"][0]["duplicates"].append(extra)
    with pytest.raises(AssertionError):
        _validate_matrix(_rebind_identity(unclassified))

    silent = copy.deepcopy(payload)
    silent["mutable_facts"][0]["duplicates"][0]["bypass"] = "silent"
    with pytest.raises(AssertionError):
        _validate_matrix(_rebind_identity(silent))


def test_pseudo_cid_mock_capability_and_unsupported_deletion_fail_closed() -> None:
    payload = _load()

    fake_cid = copy.deepcopy(payload)
    fake_cid["identity"] = "QmFakeCidThatIsNotARealCadv1Identity000000000000"
    with pytest.raises(AssertionError):
        _validate_matrix(fake_cid)

    mock_capability = copy.deepcopy(payload)
    mock_capability["capability_bindings"]["routing"]["expected"] = "mock_capability"
    with pytest.raises(AssertionError):
        _validate_matrix(_rebind_identity(mock_capability))

    deleted = copy.deepcopy(payload)
    deleted["mutable_facts"][0]["duplicates"][0]["disposition"] = "removed"
    deleted["mutable_facts"][0]["duplicates"][0]["deletion_supported"] = False
    with pytest.raises(AssertionError, match="unsupported deletion|removed"):
        _validate_matrix(_rebind_identity(deleted))

    public_delete = copy.deepcopy(payload)
    public_delete["public_deletion"] = True
    with pytest.raises(AssertionError):
        _validate_matrix(_rebind_identity(public_delete))


def test_legacy_task_state_bypass_warns_then_fails_without_writing(tmp_path: Path) -> None:
    repository = IntentRepository(tmp_path / "intent.duckdb")
    repository.upsert_goal(
        goal_cid="goal:aseh-060",
        goal_alias="GOAL-ASEH-060",
        objective_id="objective:aseh-060",
        title="single-authority",
    )
    repository.upsert_task(
        task_cid="task:aseh-060",
        task_alias="TASK-ASEH-060",
        goal_cid="goal:aseh-060",
        status="ready",
    )
    service = TaskTransitionService(repository)

    with pytest.warns(TransitionAuthorityWarning, match="compatibility bypass rejected"):
        with pytest.raises(TransitionCompatibilityBypassError, match="not an admitted"):
            service.transition_legacy(caller="legacy-adapter")

    task = repository.get_task("task:aseh-060")
    assert task is not None
    assert task["status"] == "ready"
    assert PRODUCTION_CUTOVER_DEFERRED_TO == ("ASEH-060", "ASEH-061")


def test_merge_self_validation_and_empty_patch_fail_closed() -> None:
    patch = (
        "diff --git a/pkg/example.py b/pkg/example.py\n"
        "index 0000000..1111111 100644\n"
        "--- a/pkg/example.py\n"
        "+++ b/pkg/example.py\n"
        "@@ -1 +1 @@\n"
        "-old = 1\n"
        "+new = 2\n"
    )
    plan = PatchPlan.create(
        patch_text=patch,
        plan_id="plan:aseh-060",
        base_tree="tree:aseh-060",
        semantic_intent="bounded example",
        files=("pkg/example.py",),
        symbols=("pkg.example.new",),
        preconditions=("visible",),
        postconditions=("active",),
        invariants=("bounded",),
        tests=("pytest",),
        proofs=("proof:example",),
        scope=PatchScope(("pkg/**",)),
    )
    receipt = ValidationReceipt(
        "receipt:aseh-060",
        "tree:aseh-060",
        plan.digest,
        plan.patch_digest,
        "pytest -q",
        True,
        1_700_000_000,
        False,
    )
    denied = PatchAdmission().admit(
        plan, patch, receipt, current_tree="tree:aseh-060", now_epoch_seconds=1_700_000_000
    )
    assert denied.accepted is False
    assert "self_validation" in denied.reason_codes

    empty = PatchAdmission().admit(
        plan,
        "",
        ValidationReceipt(
            "receipt:aseh-060-empty",
            "tree:aseh-060",
            plan.digest,
            plan.patch_digest,
            "pytest -q",
            True,
            1_700_000_000,
            True,
        ),
        current_tree="tree:aseh-060",
        now_epoch_seconds=1_700_000_000,
    )
    assert empty.accepted is False
    assert "empty_patch" in empty.reason_codes


def test_real_cid_producer_rejects_non_canonical_payloads() -> None:
    cid = content_identity({"task_id": TASK_ID, "fact": "routing"})
    assert CID_RE.match(cid)
    assert cid == content_identity({"fact": "routing", "task_id": TASK_ID})
    assert cid != content_identity({"task_id": TASK_ID, "fact": "routing", "mock": True})
    with pytest.raises(ControlPlaneContractError):
        content_identity({"float_is_forbidden": 1.5})
