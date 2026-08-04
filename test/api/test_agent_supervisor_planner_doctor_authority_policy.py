"""PDR-002 conformance tests for the Planner/Doctor authority boundary."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorMode,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    EffectKind,
    OperationAuthority,
)
from ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_rollout import (
    SAFETY_FLOOR_KEYS,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_constraint_compiler import (
    AdmissionRejectionCode,
)
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy import (
    DeterministicDoctorPolicy,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POLICY_PATH = (
    _REPO_ROOT / "config" / "agent_supervisor_planner_doctor_authority_policy.json"
)
_SEAL_PATH = (
    _REPO_ROOT
    / "config"
    / "agent_supervisor_planner_doctor_authority_policy.seal.json"
)
_THREAT_MODEL_PATH = (
    _REPO_ROOT
    / "docs"
    / "architecture"
    / "agent_supervisor_planner_doctor_threat_model.md"
)
_SCHEDULER_PATH = (
    _REPO_ROOT
    / "config"
    / "agent_supervisor_proof_directed_planner_doctor_scheduler.json"
)

_SEAL_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.planner_doctor.authority_policy_seal@1"
)
_SEAL_INTERFACE = "PlannerDoctorAuthorityPolicySeal@1"
_BOARD_NAMESPACE = "agent-supervisor-proof-directed-planner-doctor-v1"
_ARTIFACT_PATHS = {
    "authority_policy": "config/agent_supervisor_planner_doctor_authority_policy.json",
    "authority_test": "test/api/test_agent_supervisor_planner_doctor_authority_policy.py",
    "threat_model": "docs/architecture/agent_supervisor_planner_doctor_threat_model.md",
}
_SEAL_TOP_LEVEL_FIELDS = {
    "artifacts",
    "board_namespace",
    "decision",
    "grant",
    "interface",
    "operator",
    "policy_revision",
    "receipt_id",
    "receipt_version",
    "reviewed_base",
    "schema",
    "task_id",
}
_ACTIVATION_ONLY_GRANT = {
    "type": "policy_activation",
    "allowed_actions": ["activate_policy_revision"],
    "board_namespace": _BOARD_NAMESPACE,
    "policy_revision": "1",
    "delegable": False,
    "mutation_authority": False,
    "completion_authority": False,
    "promotion_authority": False,
    "task_status_authority": False,
    "protected_anchor_write_authority": False,
}


class _DuplicateKey(ValueError):
    """Raised when a JSON object is not a unique-key mapping."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    def reject_number(value: str) -> Any:
        raise ValueError(f"non-canonical JSON number: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_float=reject_number,
        parse_constant=reject_number,
    )
    assert isinstance(payload, dict)
    return payload


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _walk(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _seal_body(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in receipt.items() if key != "receipt_id"}


def _seal_identity(receipt: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(_seal_body(receipt)))


def _reseal(receipt: dict[str, Any]) -> dict[str, Any]:
    receipt["receipt_id"] = _seal_identity(receipt)
    return receipt


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_external_seal(
    policy: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    """Independent fail-closed reconstruction expected from a fresh loader."""

    if set(receipt) != _SEAL_TOP_LEVEL_FIELDS:
        raise ValueError("seal receipt fields do not match the closed schema")
    if any(isinstance(value, float) for value in _walk(receipt)):
        raise ValueError("floating-point receipt values are forbidden")
    if receipt["schema"] != _SEAL_SCHEMA:
        raise ValueError("seal schema mismatch")
    if receipt["interface"] != _SEAL_INTERFACE:
        raise ValueError("seal interface mismatch")
    if receipt["receipt_version"] != "1":
        raise ValueError("seal receipt version mismatch")
    if receipt["task_id"] != "PDR-002":
        raise ValueError("seal task mismatch")
    if receipt["board_namespace"] != _BOARD_NAMESPACE:
        raise ValueError("seal board namespace mismatch")
    if receipt["decision"] != "sealed":
        raise ValueError("seal decision mismatch")
    if receipt["policy_revision"] != policy["policy_revision"]:
        raise ValueError("seal policy revision mismatch")
    if not isinstance(receipt["receipt_id"], str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", receipt["receipt_id"]
    ):
        raise ValueError("seal receipt identity is malformed")
    if receipt["receipt_id"] != _seal_identity(receipt):
        raise ValueError("seal receipt identity mismatch")
    scheduler = _load_json(_SCHEDULER_PATH)
    pinned_identity = scheduler["manual_completion_seals"]["PDR-002"][
        "expected_receipt_id"
    ]
    if not isinstance(pinned_identity, str) or not hmac.compare_digest(
        receipt["receipt_id"],
        pinned_identity,
    ):
        raise ValueError("seal receipt does not match protected scheduler pin")

    reviewed_base = receipt["reviewed_base"]
    if not isinstance(reviewed_base, Mapping) or set(reviewed_base) != {
        "commit",
        "git_object_format",
        "relation_to_activation_head",
        "tree",
    }:
        raise ValueError("reviewed base fields do not match the closed schema")
    if reviewed_base["git_object_format"] != "sha1":
        raise ValueError("reviewed Git object format mismatch")
    if reviewed_base["relation_to_activation_head"] != "equal_or_ancestor":
        raise ValueError("reviewed base relation mismatch")
    commit = reviewed_base["commit"]
    tree = reviewed_base["tree"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("reviewed base commit is malformed")
    if not isinstance(tree, str) or not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise ValueError("reviewed base tree is malformed")
    try:
        if _git("cat-file", "-t", commit) != "commit":
            raise ValueError("reviewed base commit object is unavailable")
        if _git("cat-file", "-t", tree) != "tree":
            raise ValueError("reviewed base tree object is unavailable")
        if _git("rev-parse", f"{commit}^{{tree}}") != tree:
            raise ValueError("reviewed base commit/tree binding mismatch")
    except subprocess.CalledProcessError as exc:
        raise ValueError("reviewed base Git object is unavailable") from exc
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("reviewed base is not equal to or an ancestor of HEAD")

    artifacts = receipt["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(_ARTIFACT_PATHS):
        raise ValueError("seal must bind exactly three artifacts")
    by_role: dict[str, Mapping[str, Any]] = {}
    seen_paths: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "role",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("artifact binding fields do not match the closed schema")
        role = item["role"]
        path = item["path"]
        if not isinstance(role, str) or role in by_role:
            raise ValueError("artifact roles must be unique")
        if not isinstance(path, str) or path in seen_paths:
            raise ValueError("artifact paths must be unique")
        by_role[role] = item
        seen_paths.add(path)
    if set(by_role) != set(_ARTIFACT_PATHS):
        raise ValueError("artifact roles do not match the closed set")
    if seen_paths != set(_ARTIFACT_PATHS.values()):
        raise ValueError("artifact paths do not match the closed set")
    repository_root = _REPO_ROOT.resolve()
    for role, relative_path in _ARTIFACT_PATHS.items():
        item = by_role[role]
        if item["path"] != relative_path:
            raise ValueError("artifact role/path binding mismatch")
        unresolved = _REPO_ROOT / relative_path
        target = unresolved.resolve()
        if (
            unresolved.is_symlink()
            or not target.is_relative_to(repository_root)
            or not target.is_file()
        ):
            raise ValueError("sealed artifact path escapes its repository root")
        payload = target.read_bytes()
        if item["sha256"] != _sha256_bytes(payload):
            raise ValueError("sealed artifact SHA-256 mismatch")
        if type(item["size_bytes"]) is not int or item["size_bytes"] != len(payload):
            raise ValueError("sealed artifact byte count mismatch")

    operator = receipt["operator"]
    if operator != {
        "identity": "interactive_user",
        "authority_basis": "interactive_user_delegation",
        "candidate": False,
        "model": False,
        "automatic_controller": False,
    }:
        raise ValueError("seal operator identity or authority basis mismatch")
    if receipt["grant"] != _ACTIVATION_ONLY_GRANT:
        raise ValueError("seal grant is not activation-only")

    lifecycle = policy["lifecycle"]
    manual_seal = policy["manual_seal"]
    if lifecycle["self_sealing_forbidden"] is not True:
        raise ValueError("policy must remain self-sealing forbidden")
    if lifecycle["seal_state"] != "external_receipt_required":
        raise ValueError("policy must require its external receipt")
    if lifecycle["activation_requires_verified_seal_receipt"] is not True:
        raise ValueError("policy activation must require the verified receipt")
    if manual_seal["receipt_path"] != _SEAL_PATH.relative_to(_REPO_ROOT).as_posix():
        raise ValueError("policy names a foreign seal path")
    if manual_seal["receipt_schema"] != _SEAL_SCHEMA:
        raise ValueError("policy names a foreign seal schema")
    if manual_seal["receipt_interface"] != _SEAL_INTERFACE:
        raise ValueError("policy names a foreign seal interface")
    if (
        manual_seal["receipt_authentication"]
        != "protected_scheduler_expected_receipt_id_pin"
        or manual_seal["receipt_self_hash_is_authorization"] is not False
        or manual_seal["protected_scheduler_identity_pin_required"] is not True
    ):
        raise ValueError("policy receipt authentication is not independently pinned")
    if set(manual_seal["required_artifact_paths"]) != set(_ARTIFACT_PATHS.values()):
        raise ValueError("policy seal artifact set mismatch")
    if manual_seal["allowed_grant_actions"] != ["activate_policy_revision"]:
        raise ValueError("policy seal action set is not activation-only")
    for field in (
        "mutation_authority_granted",
        "completion_authority_granted",
        "promotion_authority_granted",
        "protected_anchor_write_authority_granted",
    ):
        if manual_seal[field] is not False:
            raise ValueError(f"policy seal must not grant {field}")


def _transition_ids(policy: Mapping[str, Any]) -> set[str]:
    return {str(item["id"]) for item in policy["forbidden_transitions"]}


def test_declared_outputs_and_interfaces_exist() -> None:
    assert _POLICY_PATH.is_file()
    assert _SEAL_PATH.is_file()
    assert _THREAT_MODEL_PATH.is_file()
    assert Path(__file__).is_file()

    policy = _load_json(_POLICY_PATH)
    receipt = _load_json(_SEAL_PATH)
    text = _THREAT_MODEL_PATH.read_text(encoding="utf-8")
    assert (
        policy["schema"]
        == "ipfs_accelerate_py.agent_supervisor.planner_doctor.authority_policy@1"
    )
    assert policy["interface"] == "PlannerDoctorAuthorityPolicy@1"
    assert receipt["schema"] == _SEAL_SCHEMA
    assert receipt["interface"] == _SEAL_INTERFACE
    assert "Interface: `PlannerDoctorThreatModel@1`" in text
    assert "Policy interface: `PlannerDoctorAuthorityPolicy@1`" in text


def test_policy_has_deterministic_unique_key_canonical_preimage() -> None:
    policy = _load_json(_POLICY_PATH)
    values = tuple(_walk(policy))
    assert not any(isinstance(value, float) for value in values)
    assert policy["canonicalization"] == {
        "encoding": "utf-8",
        "profile": "sorted-key compact JSON with no insignificant whitespace",
        "duplicate_keys_forbidden": True,
        "floating_point_forbidden": True,
        "non_finite_numbers_forbidden": True,
        "content_identity": "sha2-256 over canonical UTF-8 policy bytes",
        "stored_self_identity": False,
    }

    first = _canonical_bytes(policy)
    second = _canonical_bytes(_load_json(_POLICY_PATH))
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    assert b"NaN" not in first and b"Infinity" not in first


def test_policy_self_sealing_is_forbidden_and_external_receipt_is_required() -> None:
    policy = _load_json(_POLICY_PATH)
    lifecycle = policy["lifecycle"]
    seal = policy["manual_seal"]

    assert lifecycle["seal_state"] == "external_receipt_required"
    assert (
        lifecycle["seal_mechanism"]
        == "external_content_addressed_operator_receipt"
    )
    assert lifecycle["completion_mode"] == "manual"
    assert lifecycle["self_sealing_forbidden"] is True
    assert lifecycle["immutable_after_manual_completion"] is True
    assert lifecycle["activation_requires_verified_seal_receipt"] is True
    assert seal["required"] is True
    assert seal["operator_controlled_receipt_required"] is True
    assert (
        seal["receipt_path"]
        == "config/agent_supervisor_planner_doctor_authority_policy.seal.json"
    )
    assert seal["receipt_schema"] == _SEAL_SCHEMA
    assert seal["receipt_interface"] == _SEAL_INTERFACE
    assert (
        seal["receipt_identity_algorithm"]
        == "sha2-256 over canonical receipt body excluding receipt_id"
    )
    assert (
        seal["receipt_authentication"]
        == "protected_scheduler_expected_receipt_id_pin"
    )
    assert seal["receipt_self_hash_is_authorization"] is False
    assert seal["protected_scheduler_identity_pin_required"] is True
    assert seal["task_status_is_seal_authority"] is False
    assert seal["execution_permit_is_seal_authority"] is False
    assert seal["candidate_may_issue_receipt"] is False
    assert seal["model_may_issue_receipt"] is False
    assert seal["automatic_controller_may_issue_receipt"] is False
    assert seal["absence_or_invalid_disposition"] == "shadow_report_only"
    assert seal["activation_requires_fresh_policy_load"] is True
    assert set(seal["receipt_bindings"]) == {
        "authority_policy_canonical_root",
        "authority_test_content_root",
        "operator_identity",
        "policy_revision",
        "repository_tree",
        "threat_model_content_root",
    }
    assert set(seal["required_artifact_paths"]) == set(_ARTIFACT_PATHS.values())
    assert seal["allowed_grant_actions"] == ["activate_policy_revision"]
    assert seal["mutation_authority_granted"] is False
    assert seal["completion_authority_granted"] is False
    assert seal["promotion_authority_granted"] is False
    assert seal["protected_anchor_write_authority_granted"] is False


def test_external_operator_seal_reconstructs_exact_artifacts_git_base_and_grant() -> None:
    policy = _load_json(_POLICY_PATH)
    receipt = _load_json(_SEAL_PATH)

    _validate_external_seal(policy, receipt)
    assert receipt["receipt_id"] == _seal_identity(receipt)
    assert receipt["receipt_id"] == _load_json(_SCHEDULER_PATH)[
        "manual_completion_seals"
    ]["PDR-002"]["expected_receipt_id"]
    assert receipt["operator"] == {
        "identity": "interactive_user",
        "authority_basis": "interactive_user_delegation",
        "candidate": False,
        "model": False,
        "automatic_controller": False,
    }
    assert receipt["grant"] == _ACTIVATION_ONLY_GRANT
    assert receipt["decision"] == "sealed"
    assert receipt["policy_revision"] == policy["policy_revision"] == "1"

    expected = {
        role: {
            "path": relative_path,
            "role": role,
            "sha256": _sha256_bytes((_REPO_ROOT / relative_path).read_bytes()),
            "size_bytes": (_REPO_ROOT / relative_path).stat().st_size,
        }
        for role, relative_path in _ARTIFACT_PATHS.items()
    }
    assert {item["role"]: item for item in receipt["artifacts"]} == expected
    assert (
        _git("rev-parse", f"{receipt['reviewed_base']['commit']}^{{tree}}")
        == receipt["reviewed_base"]["tree"]
    )


def test_external_operator_seal_rejects_tamper_even_after_attacker_rehashes() -> None:
    policy = _load_json(_POLICY_PATH)
    original = _load_json(_SEAL_PATH)

    identity_tamper = copy.deepcopy(original)
    identity_tamper["receipt_id"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_external_seal(policy, identity_tamper)

    mutations: list[tuple[str, Any]] = [
        (
            "artifact root",
            lambda value: value["artifacts"][0].__setitem__(
                "sha256", "sha256:" + ("0" * 64)
            ),
        ),
        (
            "artifact byte count",
            lambda value: value["artifacts"][0].__setitem__(
                "size_bytes", value["artifacts"][0]["size_bytes"] + 1
            ),
        ),
        (
            "policy revision",
            lambda value: value.__setitem__("policy_revision", "2"),
        ),
        (
            "reviewed base tree",
            lambda value: value["reviewed_base"].__setitem__("tree", "0" * 40),
        ),
        (
            "operator identity",
            lambda value: value["operator"].__setitem__("identity", "candidate"),
        ),
        (
            "operator authority basis",
            lambda value: value["operator"].__setitem__(
                "authority_basis", "task_status"
            ),
        ),
        (
            "mutation authority",
            lambda value: value["grant"].__setitem__("mutation_authority", True),
        ),
        (
            "completion authority",
            lambda value: value["grant"].__setitem__(
                "completion_authority", True
            ),
        ),
        (
            "promotion authority",
            lambda value: value["grant"].__setitem__("promotion_authority", True),
        ),
        (
            "extra field",
            lambda value: value.__setitem__("candidate_note", "self-approved"),
        ),
    ]
    for _label, mutate in mutations:
        tampered = copy.deepcopy(original)
        mutate(tampered)
        _reseal(tampered)
        with pytest.raises(ValueError, match=".+"):
            _validate_external_seal(policy, tampered)

    policy_tamper = copy.deepcopy(policy)
    policy_tamper["manual_seal"]["receipt_path"] = "candidate/seal.json"
    with pytest.raises(ValueError, match="foreign seal path"):
        _validate_external_seal(policy_tamper, original)


def test_policy_reuses_the_closed_control_and_assurance_vocabularies() -> None:
    policy = _load_json(_POLICY_PATH)
    bindings = policy["contract_bindings"]

    assert bindings["control_authorities"] == [
        item.value for item in OperationAuthority
    ]
    assert bindings["proof_assurance_levels"] == [
        item.value for item in AssuranceLevel
    ]
    assert bindings["proof_assurance_order"] == [
        "unverified",
        "candidate",
        "solver_checked",
        "kernel_verified",
        "attested",
    ]
    assert AssuranceLevel.SOLVER_CHECKED.rank < AssuranceLevel.KERNEL_VERIFIED.rank
    assert AssuranceLevel.KERNEL_VERIFIED.rank < AssuranceLevel.ATTESTED.rank
    assert bindings["reuse_existing_control_permits"] is True
    assert bindings["reuse_existing_proof_cache"] is True
    assert bindings["reuse_existing_completion_authority"] is True
    assert bindings["new_authority_lattice_forbidden"] is True


def test_authority_ladder_distinguishes_all_five_stages() -> None:
    policy = _load_json(_POLICY_PATH)
    ladder = policy["authority_ladder"]

    assert [item["name"] for item in ladder] == [
        "nomination",
        "observation",
        "bounded_check",
        "kernel_proof",
        "attestation",
    ]
    assert [item["order"] for item in ladder] == list(range(5))
    assert [item["maximum_assurance"] for item in ladder] == [
        "candidate",
        "candidate",
        "solver_checked",
        "kernel_verified",
        "attested",
    ]
    for stage in ladder:
        assert stage["mutation_authority"] is False
        assert stage["completion_authority"] is False

    nomination = ladder[0]
    assert nomination["semantic_authority"] is False
    assert {
        "bm25",
        "embedding",
        "graphrag",
        "knowledge_graph",
        "llm_output",
        "synthetic_fixture",
        "vector_index",
    } <= set(nomination["sources"])
    assert {
        "admit",
        "authorize_write",
        "certify_self",
        "claim_completion",
        "promote",
        "satisfy_proof",
    } <= set(nomination["may_not"])

    assert ladder[2]["semantic_authority"] == "bounded_only"
    assert ladder[3]["semantic_authority"] == "encoded_theorem_only"
    assert (
        ladder[4]["semantic_authority"]
        == "inherited_from_verified_statement"
    )
    assert "kernel_verified_receipt" in ladder[4]["required_bindings"]
    assert "upgrade_underlying_semantic_scope" in ladder[4]["may_not"]


def test_defaults_and_deterministic_doctor_remain_fail_closed() -> None:
    policy = _load_json(_POLICY_PATH)
    defaults = policy["defaults"]
    boundary = policy["deterministic_doctor_boundary"]
    shipped = DeterministicDoctorPolicy()

    assert defaults == {
        "planner_mode": "shadow",
        "doctor_mode": "report_only",
        "rollout_mode": "shadow",
        "mutation_authorized": False,
        "automatic_promotion_enabled": False,
        "derived_refill_enabled": False,
        "completion_authoritative": False,
        "unknown_or_unsupported_disposition": "abstain",
        "ambiguous_disposition": "abstain",
    }
    assert shipped.default_mode is DoctorMode.REPORT_ONLY
    assert shipped.enabled is False
    assert shipped.narrow_autonomous_mutation_enabled is False

    hard_false = {
        "llm_router_enabled",
        "llm_invocations_allowed",
        "remote_model_provider_calls_allowed",
        "remote_embeddings_allowed",
        "network_access_allowed",
        "target_code_import_allowed",
        "knowledge_graph_semantic_authority",
        "vector_semantic_authority",
        "embedding_semantic_authority",
        "proof_cache_metadata_semantic_authority",
    }
    for name in hard_false:
        assert boundary[name] is False
        assert getattr(shipped, name) is False
    assert boundary["automatic_fallback_to_model_forbidden"] is True
    assert boundary["optional_provider_failure_disposition"] == "typed_abstention"

    hybrid = policy["hybrid_model_boundary"]
    assert hybrid["distinct_from_deterministic_doctor"] is True
    assert hybrid["enabled_by_default"] is False
    assert hybrid["model_output_authority"] == "nomination"
    assert hybrid["model_write_authority"] is False
    assert hybrid["model_completion_authority"] is False
    assert hybrid["inherits_deterministic_doctor_authority"] is False


def test_preview_never_inherits_apply_authority() -> None:
    policy = _load_json(_POLICY_PATH)
    separation = policy["preview_apply_separation"]
    preview = separation["preview"]
    apply = separation["apply"]

    assert separation["required"] is True
    assert preview["control_authorities"] == [
        OperationAuthority.READ.value,
        OperationAuthority.PROPOSAL.value,
    ]
    assert set(preview["allowed_effect_kinds"]) <= {
        EffectKind.OBSERVE.value,
        EffectKind.PROPOSE.value,
    }
    assert preview["repository_write_allowed"] is False
    assert preview["authoritative_state_write_allowed"] is False
    assert preview["permit_grants_apply_authority"] is False
    assert apply["control_authority"] == OperationAuthority.MUTATION.value
    assert apply["enabled_by_default"] is False
    for key, value in apply.items():
        if key.endswith("_required"):
            assert value is True, key
    assert separation["preview_artifact_may_be_applied_directly"] is False
    assert separation["apply_requires_new_request_and_fresh_bindings"] is True


def test_capabilities_transactions_and_roots_are_exact_and_fail_closed() -> None:
    policy = _load_json(_POLICY_PATH)
    scope = policy["capability_scope"]
    roots = policy["current_tree_revalidation"]
    transaction = policy["mutation_transaction"]

    assert {
        "expected_effect_ids",
        "fencing_epoch",
        "idempotency_key",
        "lease_id",
        "operation",
        "path_allowlist",
        "policy_revision",
        "repository_root",
        "resource_bounds",
        "tree_id",
    } <= set(scope["permit_fields_required"])
    for key in (
        "exact_operation_required",
        "exact_effect_set_required",
        "exact_path_allowlist_required",
        "exact_resource_bounds_required",
        "expiry_required",
        "scope_widening_forbidden",
        "delegation_forbidden",
        "path_escape_forbidden",
    ):
        assert scope[key] is True, key

    assert roots["required"] is True
    assert roots["unknown_is_current"] is False
    assert {
        "dirty_overlay_root",
        "intent_ir_root",
        "policy_root",
        "recursive_gitlink_roots",
        "security_ir_root",
        "superproject_commit",
        "superproject_tree",
        "task_source_revision",
        "toolchain_roots",
    } <= set(roots["bindings"])
    assert {
        "cache_replay",
        "immediately_before_mutation_dispatch",
        "merged_tree_revalidation",
        "completion_evaluation",
        "promotion",
    } <= set(roots["required_at"])
    assert roots["stale_replay_disposition"] == "reject"

    assert transaction["partial_commit_allowed"] is False
    assert transaction["prebuilt_fixed_point_mapping_is_authoritative"] is False
    assert transaction["fixed_point_boolean_is_authoritative"] is False
    for key in (
        "real_byte_change_required_for_commit",
        "complete_impact_closure_required",
        "one_disposition_per_resolved_consumer_required",
        "atomic_scc_transaction_required",
        "prepared_before_dispatch_required",
        "compare_and_swap_ref_update_required",
        "expected_effect_match_required",
        "independent_rollback_validation_required",
        "post_edit_reindex_required",
        "dependency_local_cache_invalidation_required",
        "live_logic_and_program_fixed_point_required",
    ):
        assert transaction[key] is True, key


def test_proof_security_cache_and_attestation_cannot_launder_authority() -> None:
    policy = _load_json(_POLICY_PATH)
    gates = policy["proof_and_security_gates"]

    for key in (
        "candidate_proof_is_authoritative",
        "solver_result_is_kernel_proof",
        "ordinary_test_is_proof",
        "generated_test_may_certify_its_generator",
        "serialized_boolean_may_claim_proof",
        "proof_cache_hit_may_upgrade_assurance",
        "simulated_zk_may_emit_attested",
        "zkp_semantic_scope_may_expand_underlying_claim",
    ):
        assert gates[key] is False, key
    for key in (
        "retrieval_is_nomination_only",
        "native_kernel_reconstruction_required_for_authoritative_proof",
        "independent_countermodel_validation_required_for_refutation",
        "proof_cache_hit_rederives_assurance",
        "security_ir_required",
        "intent_ir_required",
        "intent_and_generated_code_checked",
        "attestation_requires_kernel_verified_receipt",
    ):
        assert gates[key] is True, key
    assert gates["authoritative_security_proof_minimum"] == "kernel_verified"
    assert gates["security_unknown_disposition"] == "reject"
    assert gates["security_conflict_disposition"] == "reject"
    assert AdmissionRejectionCode.SECURITY_UNKNOWN.value == "security_unknown"
    assert AdmissionRejectionCode.SECURITY_CONFLICT.value == "security_conflict"


def test_protected_anchors_cover_task_source_policy_and_holdout() -> None:
    policy = _load_json(_POLICY_PATH)
    anchors = policy["protected_anchors"]
    paths = set(anchors["paths"])

    required_paths = {
        "config/agent_supervisor_planner_doctor_authority_policy.json",
        "config/agent_supervisor_planner_doctor_authority_policy.seal.json",
        "config/agent_supervisor_planner_doctor_benchmark.json",
        "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json",
        "docs/architecture/AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md",
        "docs/architecture/agent_supervisor_planner_doctor_threat_model.md",
        "docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md",
        "docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md",
        "test/api/test_agent_supervisor_planner_doctor_authority_policy.py",
        "test/fixtures/agent_supervisor/planner_doctor_holdout",
    }
    assert required_paths <= paths
    assert {
        "benchmark.denominators",
        "benchmark.hidden_oracle",
        "benchmark.holdout_manifest",
        "benchmark.safety_floors",
        "promotion_policy",
        "task_source.seed",
    } <= set(anchors["protected_fields"])
    assert anchors["candidate_read_hidden_oracle_body_allowed"] is False
    assert anchors["candidate_write_allowed"] is False
    assert anchors["model_write_allowed"] is False
    assert anchors["automatic_controller_write_allowed"] is False
    assert anchors["derived_runtime_task_source_may_mutate_seed_board"] is False

    scheduler = _load_json(_SCHEDULER_PATH)
    staged = scheduler["protected_after_manual_completion"]["PDR-002"]
    assert {
        "config/agent_supervisor_planner_doctor_authority_policy.json",
        "docs/architecture/agent_supervisor_planner_doctor_threat_model.md",
        "test/api/test_agent_supervisor_planner_doctor_authority_policy.py",
    } <= set(staged)


def test_task_source_completion_and_promotion_cannot_change_their_judge() -> None:
    policy = _load_json(_POLICY_PATH)
    task_source = policy["task_source_and_holdout_protection"]
    completion = policy["completion_policy"]
    promotion = policy["promotion_policy"]

    assert task_source["seed_task_source_is_operator_owned"] is True
    assert task_source["candidate_may_set_seed_task_status"] is False
    assert task_source["candidate_may_add_seed_tasks"] is False
    assert task_source["candidate_may_edit_seed_goals"] is False
    assert task_source["derived_refill_uses_separate_runtime_source"] is True
    assert task_source["derived_refill_enabled_at_bootstrap"] is False
    assert task_source["hidden_oracle_body_available_to_candidate"] is False
    assert task_source["hidden_oracle_result_available_before_submission"] is False
    assert task_source["synthetic_fixture_is_live_evidence"] is False
    assert task_source["skipped_check_is_live_evidence"] is False

    assert completion["task_count_is_completion_authority"] is False
    assert completion["task_status_is_completion_authority"] is False
    assert completion["candidate_claim_is_completion_authority"] is False
    assert completion["all_tasks_complete_is_sufficient"] is False
    assert completion["fresh_acceptance_evidence_required"] is True
    assert completion["independent_validation_required"] is True
    assert completion["replayed_terminal_receipt_required"] is True
    assert completion["zero_safety_floors_required"] is True
    assert completion["completion_authority_owner"] == "existing_goal_completion_contract"

    assert promotion["default_decision"] == "shadow"
    assert promotion["manual"] is True
    assert promotion["paired_live_evidence_required"] is True
    assert promotion["fresh_current_tree_evaluation_required"] is True
    assert promotion["independent_holdout_required"] is True
    assert promotion["synthetic_evidence_may_promote"] is False
    assert promotion["skipped_checks_may_promote"] is False
    assert promotion["candidate_may_change_judge"] is False


def test_holdout_secrecy_requires_an_external_two_phase_evaluator_mount() -> None:
    policy = _load_json(_POLICY_PATH)
    boundary = policy["holdout_storage_boundary"]

    assert boundary["same_repository_path_protection_establishes_secrecy"] is False
    assert boundary["in_repository_artifacts_are_metadata_only"] is True
    assert set(boundary["in_repository_allowed_fields"]) == {
        "denominators",
        "opaque_handles",
        "population_metadata",
        "root_cids",
    }
    assert (
        boundary["hidden_bodies_storage"]
        == "independent_evaluator_owned_read_only_repository_or_cas"
    )
    for key in (
        "hidden_bodies_in_candidate_worktree_allowed",
        "hidden_bodies_in_provider_context_allowed",
        "hidden_bodies_in_transcript_allowed",
        "candidate_os_read_access_allowed",
    ):
        assert boundary[key] is False, key
    for key in (
        "candidate_finalization_required_before_evaluator_mount",
        "two_phase_unmounted_then_evaluator_only_mount_required",
        "mount_read_only_required",
        "symlink_escape_rejected",
        "hardlink_escape_rejected",
    ):
        assert boundary[key] is True, key
    assert set(boundary["development_holdout_disjointness_checks"]) == {
        "entry_cid",
        "normalized_ast_id",
        "normalized_contract_id",
        "provenance_family_id",
    }
    assert boundary["overlap_disposition"] == "reject_benchmark_run"


def test_all_required_forbidden_transitions_are_closed() -> None:
    policy = _load_json(_POLICY_PATH)
    required = {
        "benchmark_or_oracle_mutation_to_valid_evaluation",
        "cache_hit_to_assurance_upgrade",
        "candidate_or_self_certification_to_authority",
        "candidate_work_to_protected_anchor_mutation",
        "model_output_to_write",
        "partial_transaction_to_commit",
        "prebuilt_or_boolean_fixed_point_to_completion",
        "provider_or_retrieval_result_to_semantic_authority",
        "simulated_zk_to_attested",
        "stale_replay_to_current_evidence",
        "synthetic_or_skipped_evidence_to_live_promotion",
        "task_count_or_task_status_to_completion",
        "unproved_unknown_or_conflicting_security_to_admission",
        "verified_attestation_to_unencoded_semantic_claim",
    }
    assert _transition_ids(policy) == required
    for transition in policy["forbidden_transitions"]:
        assert transition["disposition"] in {
            "reject",
            "reject_quarantine_and_escalate",
            "rollback_or_quarantine",
        }


def test_safety_floors_are_exact_integer_zero_and_cover_existing_doctor_floors() -> None:
    policy = _load_json(_POLICY_PATH)
    floors = policy["safety_floors"]

    assert floors
    assert set(SAFETY_FLOOR_KEYS) <= set(floors)
    for name, value in floors.items():
        assert type(value) is int, name
        assert value == 0, name
    for required in (
        "authority_violation_count",
        "benchmark_oracle_mutation_count",
        "false_completion_count",
        "false_fixed_point_count",
        "partial_transaction_count",
        "policy_scope_escape_count",
        "protected_anchor_mutation_count",
        "rollback_failure_count",
        "secret_exposure_count",
        "stale_replay_admission_count",
        "unknown_security_admission_count",
        "unproved_security_admission_count",
    ):
        assert floors[required] == 0


def test_kill_switch_and_manual_escalation_fail_closed() -> None:
    policy = _load_json(_POLICY_PATH)
    kill = policy["kill_switch"]
    escalation = policy["manual_escalation"]

    assert kill["available"] is True
    assert kill["engaged_by_default"] is False
    assert kill["automatic_engage_on_noncompensable_gate_failure"] is True
    assert kill["operator_may_engage"] is True
    assert kill["candidate_may_clear"] is False
    assert kill["model_may_clear"] is False
    assert kill["automatic_controller_may_clear"] is False
    assert kill["operator_only_clear"] is True
    assert kill["clear_requires_audit_and_current_tree_revalidation"] is True
    assert kill["engaged_mode"] == "report_only"
    assert kill["blocks_new_mutation_dispatch"] is True
    assert kill["blocks_promotion"] is True
    assert kill["blocks_derived_refill"] is True

    assert escalation["required"] is True
    assert escalation["default_disposition"] == "abstain"
    assert escalation["mode_while_pending"] == "report_only"
    assert escalation["operator_decision_binds_fresh_roots"] is True
    assert escalation["candidate_approval_allowed"] is False
    assert escalation["model_approval_allowed"] is False
    assert escalation["automatic_self_approval_allowed"] is False
    assert {
        "hidden_oracle_or_benchmark_change",
        "kill_switch_clear",
        "new_network_or_provider_scope",
        "protected_anchor_change",
        "proof_or_security_requirement_not_met",
        "rollback_or_recovery_failure",
        "scope_or_effect_expansion",
    } <= set(escalation["triggers"])


def test_provider_secret_boundary_is_body_free_and_network_deny_by_default() -> None:
    policy = _load_json(_POLICY_PATH)
    boundary = policy["provider_and_secret_boundaries"]

    assert boundary["credentials_source"] == "backend_or_environment_only"
    for key in (
        "credentials_in_prompt_forbidden",
        "credentials_in_task_source_forbidden",
        "credentials_in_argv_forbidden",
        "credentials_in_log_forbidden",
        "credentials_in_receipt_forbidden",
        "credentials_in_cache_key_forbidden",
        "private_witness_in_public_artifact_forbidden",
        "provider_payload_body_in_control_receipt_forbidden",
        "network_requires_separate_scoped_operator_policy",
        "provider_health_version_and_policy_binding_required",
    ):
        assert boundary[key] is True, key
    assert boundary["network_default"] == "deny"
    assert boundary["provider_failure_is_pass"] is False
    assert boundary["provider_result_semantic_authority"] is False


def test_threat_model_covers_policy_transitions_and_operator_seal() -> None:
    policy = _load_json(_POLICY_PATH)
    text = _THREAT_MODEL_PATH.read_text(encoding="utf-8")

    for heading in (
        "## 3. Protected assets and anchors",
        "## 4. Actors, assumptions, and trust boundaries",
        "## 5. Closed authority ladder",
        "## 6. Threats and required mitigations",
        "## 7. Preview and mutation protocol",
        "## 8. Proof, security, cache, and ZKP boundary",
        "## 9. Deterministic Doctor and hybrid model boundary",
        "## 10. Completion, promotion, and refill",
        "## 11. Kill switch, rollback, and manual escalation",
        "## 13. Seal-receipt validation rules",
        "## 14. Manual seal checklist",
    ):
        assert heading in text
    for name in (
        "Nomination",
        "Observation",
        "Bounded check",
        "Kernel proof",
        "Attestation",
    ):
        assert name in text
    for transition_id in sorted(_transition_ids(policy)):
        assert f"`{transition_id}`" in text
    assert "Taskboard status is not a seal" in text
    assert "execution permit cannot issue it" in text
    assert "long-running supervisor" in text
    assert "Completed task counts are neither proof nor authority" in text
