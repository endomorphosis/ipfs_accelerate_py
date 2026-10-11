"""Real native owner observations with closed, non-authoritative exports."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import identify_strict_artifact
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import (
    DoctorIdentityBinding, DoctorProofCacheGate,
)
from ipfs_accelerate_py.agent_supervisor.runtime import formal_status_census as census
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata
from test.api.test_agent_supervisor_doctor_proof_cache import _key, _receipt, _identity


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def native(tmp_path):
    with public_replay_without_metadata():
        gate = DoctorProofCacheGate(tmp_path / "PRIVATE_OWNER_PATH")
        key = _key()
        assert gate.put(key, _receipt(metadata={"private_note": "PRIVATE_PROOF_BODY"})).stored
    binding = census.FormalStatusCensusBinding(task_sha256=sha("PRIVATE_TASK"),
        revision_sha256=sha("PRIVATE_REVISION"), source_sha256=sha("PRIVATE_SOURCE"),
        context_sha256=sha("PRIVATE_CONTEXT"), candidate_tree_id=key.tree.logical_id,
        owner_path_sha256=sha(str(gate.db_path)), expected_key_ids=(key.key_id,))
    return gate, key, binding


def observe(native, **changes):
    gate, key, binding = native
    values = dict(gate=gate, keys=(key,), binding=binding, current_binding_probe=lambda: binding)
    values.update(changes)
    return census.census_formal_status(**values)


def assert_closed(report):
    rendered = json.dumps(report, sort_keys=True)
    assert len(rendered.encode()) <= census.MAX_OUTPUT_BYTES
    for text in ("PRIVATE_TASK", "PRIVATE_REVISION", "PRIVATE_SOURCE", "PRIVATE_CONTEXT",
                 "PRIVATE_OWNER_PATH", "PRIVATE_PROOF_BODY", "goal-1", "tree-1", "password"):
        assert text not in rendered
    assert report["repeated_status_population"] is None
    assert report["atomic_owner_snapshot"] is False
    assert report["eligibility_at_return_verified"] is False
    assert report["receipt_hash_authenticates_source_or_owner"] is False
    for field in census._AUTHORITIES:
        assert report[field] is False


def assert_no_calls(report):
    assert report["current_binding_probe_calls"] == 0
    assert report["gate_lookup_calls"] == 0
    assert report["gate_render_revalidation_calls"] == 0
    assert report["gate_denial_state_checks"] == 0
    assert report["owner_identity_checks"] == 0


def test_explicit_hit_is_observed_without_exported_bodies_or_authority(native):
    report = observe(native)
    assert report["status"] == "measured"
    assert report["accepted_receipts"] == 1
    assert report["lookup_dispositions"]["hit"] == 1
    assert report["render_dispositions"]["hit"] == 1
    assert report["observation_current"] is True
    assert report["current_binding_probe_calls"] == 3
    assert report["gate_lookup_calls"] == report["gate_render_revalidation_calls"] == 1
    assert report["native_gate_bookkeeping_may_write_existing_owner"] is True
    assert report["storage_read_only"] is False
    assert_closed(report)


@pytest.mark.parametrize("missing", ["binding", "gate", "keys", "current_binding_probe"])
def test_missing_admission_does_not_probe_or_touch_owner(native, monkeypatch, missing):
    def forbidden(*args, **kwargs):
        pytest.fail("missing binding must not touch storage or call the host")
    monkeypatch.setattr(native[0], "lookup", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    report = observe(native, **{missing: None}, **({"current_binding_probe": forbidden}
                    if missing != "current_binding_probe" else {}))
    assert report["status"] == "unmeasured"
    assert report["accepted_receipts"] is None
    assert report["requested_key_count"] is None
    assert report["observation_current"] is False
    assert_no_calls(report)


def test_unavailable_all_none_is_content_free():
    report = census.census_formal_status()
    assert report["status"] == "unmeasured"
    assert report["reason_code"] == "native_owner_binding_unavailable"
    assert_no_calls(report)
    assert_closed(report)


@pytest.mark.parametrize("missing", ["owner_path_sha256", "expected_key_ids"])
def test_partial_host_binding_remains_unknown(native, missing):
    report = observe(native, binding=replace(native[2], **{missing: None}))
    assert report["status"] == "unmeasured"
    assert_no_calls(report)


def test_explicit_empty_population_is_measured_zero(native):
    empty = replace(native[2], expected_key_ids=())
    report = observe(native, keys=(), binding=empty, current_binding_probe=lambda: empty)
    assert report["status"] == "measured"
    assert report["requested_key_count"] == report["accepted_receipts"] == 0
    assert report["current_binding_probe_calls"] == 1
    assert report["gate_lookup_calls"] == report["gate_render_revalidation_calls"] == 0
    assert_closed(report)


def test_exact_missing_key_is_not_a_counterexample(native):
    missing = _key(goal=_identity("other-goal", "other-goal"))
    bound = replace(native[2], expected_key_ids=(missing.key_id,))
    report = observe(native, keys=(missing,), binding=bound, current_binding_probe=lambda: bound)
    assert report["status"] == "measured"
    assert report["accepted_receipts"] == 0
    assert report["lookup_dispositions"]["miss"] == 1
    assert report["render_dispositions"]["miss"] == 1
    assert_closed(report)


@pytest.mark.parametrize("kind", ["foreign", "dictionary", "list", "duplicate", "many",
                                  "wrong_context", "wrong_expected", "bad_digest", "private", "oversized"])
def test_invalid_native_request_refuses_before_any_observation(native, monkeypatch, kind):
    gate, key, binding = native
    keys = (key,)
    if kind == "foreign":
        gate = object()
    elif kind == "dictionary":
        keys = (key.to_dict(),)
    elif kind == "list":
        keys = [key]
    elif kind == "duplicate":
        keys = (key, key)
    elif kind == "many":
        keys = (key,) * (census.MAX_KEYS + 1)
    elif kind == "wrong_context":
        binding = replace(binding, candidate_tree_id="other-tree")
    elif kind == "wrong_expected":
        binding = replace(binding, expected_key_ids=())
    elif kind == "bad_digest":
        binding = replace(binding, source_sha256=True)
    elif kind == "private":
        private = DoctorIdentityBinding.from_identity(identify_strict_artifact({"password": "placeholder"}),
                                                      logical_id="private-goal")
        keys = (_key(goal=private),)
        binding = replace(binding, expected_key_ids=(keys[0].key_id,))
    else:
        large = _identity("oversized", "oversized", "x" * census.MAX_KEY_BYTES)
        keys = (_key(goal=large),)
        binding = replace(binding, expected_key_ids=(keys[0].key_id,))
    def forbidden(*args, **kwargs):
        pytest.fail("invalid request must refuse before any observation")
    monkeypatch.setattr(Path, "stat", forbidden)
    report = census.census_formal_status(gate=gate, keys=keys, binding=binding,
                                        current_binding_probe=forbidden)
    assert report["status"] == "refused"
    assert report["reason_code"] == "invalid_native_request"
    assert_no_calls(report)
    assert_closed(report)


@pytest.mark.parametrize("when", [1, 2, 3])
def test_host_context_drift_at_each_probe_fails_closed(native, when):
    calls = 0
    def probe():
        nonlocal calls
        calls += 1
        return replace(native[2], source_sha256=sha("changed")) if calls == when else native[2]
    report = observe(native, current_binding_probe=probe)
    assert report["status"] == "refused"
    assert report["accepted_receipts"] is None
    assert report["observation_current"] is False
    assert report["current_binding_probe_calls"] == when
    assert_closed(report)


def test_probe_failure_does_not_export_error_text(native):
    def probe():
        raise RuntimeError("PRIVATE_PROOF_BODY")
    report = observe(native, current_binding_probe=probe)
    assert report["status"] == "refused"
    assert_closed(report)


def test_native_revocation_race_is_preserved(native, monkeypatch):
    gate, key, _ = native
    original = gate.authoritative_cache.lookup
    def raced(*args, **kwargs):
        result = original(*args, **kwargs)
        gate.invalidate_semantic_root(root_field="forest", root_cid=key.forest.cid)
        return result
    monkeypatch.setattr(gate.authoritative_cache, "lookup", raced)
    report = observe(native)
    assert report["status"] == "refused"
    assert report["accepted_receipts"] is None
    assert report["lookup_dispositions"]["tombstoned"] == 1
    assert_closed(report)


def test_existing_quarantine_remains_a_refusal_observation(native):
    gate, key, _ = native
    gate.quarantine(key)
    report = observe(native)
    assert report["status"] == "refused"
    assert report["accepted_receipts"] is None
    assert report["render_dispositions"]["quarantined"] == 1
    assert_closed(report)


@pytest.mark.parametrize("damage", ["dict", "key", "receipt_type", "claimed_hit", "audit"])
def test_forged_native_result_is_refused(native, monkeypatch, damage):
    gate, key, _ = native
    with public_replay_without_metadata():
        result = gate.lookup(key)
    if damage == "dict":
        forged = {"accepted_receipts": 1000, "body": "PRIVATE_PROOF_BODY"}
    elif damage == "key":
        forged = replace(result, key=_key(goal=_identity("other", "other")))
    elif damage == "receipt_type":
        forged = replace(result, receipt=result.receipt.to_dict())
    elif damage == "claimed_hit":
        forged = replace(result, receipt=replace(result.receipt, evidence=()))
    else:
        forged = replace(result, audit=replace(result.audit, authoritative=1))
    monkeypatch.setattr(gate, "lookup", lambda key: forged)
    report = observe(native)
    assert report["status"] == "refused"
    assert report["accepted_receipts"] is None
    assert_closed(report)


def test_census_never_activates_configured_metadata_owner(native, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index
    def forbidden(*args, **kwargs):
        pytest.fail("public census must suppress metadata owner activation")
    monkeypatch.setattr(supervisor_meta_index.SupervisorMetaIndex, "from_env", forbidden)
    assert observe(native)["accepted_receipts"] == 1


def test_owner_replacement_during_probe_fails_closed(native):
    gate, _, binding = native
    original = gate.db_path
    backup = original.with_suffix(".retained")
    def probe():
        original.rename(backup)
        original.write_bytes(backup.read_bytes())
        return binding
    report = observe(native, current_binding_probe=probe)
    assert report["status"] == "refused"
    assert report["gate_lookup_calls"] == 0
    assert_closed(report)


def test_foreign_owner_path_digest_refuses_without_probe(native):
    report = observe(native, binding=replace(native[2], owner_path_sha256=sha("other-owner")))
    assert report["status"] == "refused"
    assert report["current_binding_probe_calls"] == report["owner_identity_checks"] == 0
    assert_closed(report)


def test_post_render_host_probe_revocation_refuses(native):
    gate, key, binding = native
    calls = 0
    def probe():
        nonlocal calls
        calls += 1
        if calls == 3:
            gate.invalidate_semantic_root(root_field="forest", root_cid=key.forest.cid)
        return binding
    report = observe(native, current_binding_probe=probe)
    assert report["status"] == "refused"
    assert report["accepted_receipts"] is None
    assert report["render_dispositions"]["hit"] == 1
    assert report["gate_denial_state_checks"] == 2
    assert_closed(report)


def test_later_key_probe_revoking_earlier_key_refuses_whole_census(native):
    gate, first, binding = native
    second = _key(goal=_identity("second", "second"), forest=_identity("second-forest", "second-forest"))
    receipt = _receipt(obligation_id="second")
    with public_replay_without_metadata():
        assert gate.put(second, receipt).stored
    bound = replace(binding, expected_key_ids=(first.key_id, second.key_id))
    calls = 0
    def probe():
        nonlocal calls
        calls += 1
        if calls == 5:
            gate.invalidate_semantic_root(root_field="forest", root_cid=first.forest.cid)
        return bound
    report = observe(native, keys=(first, second), binding=bound, current_binding_probe=probe)
    assert report["status"] == "refused"
    assert report["accepted_receipts"] is None
    assert report["render_dispositions"]["hit"] == 2
    assert_closed(report)


def test_post_validation_expiry_does_not_claim_current_eligibility(native, monkeypatch):
    gate, _, binding = native
    current_time = gate._clock()
    monkeypatch.setattr(gate, "_clock", lambda: current_time)
    monkeypatch.setattr(gate.authoritative_cache, "_clock", lambda: current_time)
    calls = 0
    def probe():
        nonlocal calls, current_time
        calls += 1
        if calls == 3:
            current_time += 2 * gate.positive_ttl_seconds
        return binding
    report = observe(native, current_binding_probe=probe)
    assert report["status"] == "measured"
    assert report["accepted_receipts"] == 1  # Earlier per-key native validation points.
    assert report["eligibility_at_return_verified"] is False
    assert report["accepted_receipt_count_scope"] == "per-key-native-lookup-and-render-observations@1"
    assert_closed(report)
