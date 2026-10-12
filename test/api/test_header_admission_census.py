"""Typed census bindings reuse native owners with authored prover transcripts.

No model or prover process is run. Signed-source preparation, complete keys,
sealed SQLite lineage, scoped receipt production and Doctor DuckDB are native.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_doctor_header_cache_bridge import authored, _import, _forbid_owners, PROGRAM
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ScopedProofReceipt, canonical_json, content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_header_cache_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.runtime import formal_status_census as census


@pytest.fixture(autouse=True)
def isolated_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def admitted(authored, monkeypatch):
    handle = _import(authored)
    key, receipt = handle.key, handle.receipt
    binding = census.FormalStatusCensusBinding(task_sha256=_sha(authored["scoped"].task_cid),
        revision_sha256=_sha("authored-revision"), source_sha256=_sha("authored-source"),
        context_sha256=_sha("authored-context"), candidate_tree_id=key.tree.logical_id,
        owner_path_sha256=authored["owner_pin"], expected_key_ids=(key.key_id,),
        expected_receipt_ids=((key.key_id, receipt.receipt_id),))
    _forbid_owners(monkeypatch)
    return {**authored, "handle": handle, "key": key, "receipt": receipt, "binding": binding}


def _observe(values, **changes):
    options = dict(binding=values["binding"], gate=values["gate"], keys=(values["key"],),
        current_binding_probe=lambda: values["binding"], header_proof_admissions=(values["handle"],))
    options.update(changes)
    return census.census_formal_status(**options)


def _bound(values, **changes):
    options = dict(gate=values["gate"], scoped_analysis=values["scoped"],
        expected_key_id=values["key"].key_id, expected_receipt_id=values["receipt"].receipt_id,
        expected_owner_path_sha256=values["owner_pin"])
    options.update(changes)
    return values["handle"].assert_bound(**options)


def _closed(values, report):
    assert report["accepted_receipts"] is None or report["accepted_receipts"] >= 0
    assert report["repeated_status_population"] is None
    assert report["eligibility_at_return_verified"] is False and report["atomic_owner_snapshot"] is False
    assert report["new_storage_owners_created"] == 0
    assert all(report[name] is False for name in census._AUTHORITIES)
    body = json.dumps(report, sort_keys=True)
    assert len(body.encode()) <= census.MAX_OUTPUT_BYTES
    for raw in (str(values["repository"]), str(values["store"].path), str(values["gate"].db_path),
                values["scoped"].task_cid, values["key"].key_id, values["receipt"].receipt_id,
                values["prepared"].theorem.theorem_body):
        assert raw not in body


def test_public_exact_host_binding_is_pure_and_current_check_runs_no_kernel(admitted, monkeypatch):
    before = list(admitted["calls"])
    with monkeypatch.context() as blocked:
        blocked.setattr(Path, "stat", lambda *args, **kwargs: pytest.fail("pure host binding touched storage"))
        assert _bound(admitted) is None
    assert admitted["handle"].assert_current() is None
    assert admitted["calls"] == before


@pytest.mark.parametrize("wrong", ["gate", "scope-object", "key", "receipt", "owner-path"])
def test_public_exact_host_binding_refuses_aliases_before_storage(admitted, monkeypatch, wrong):
    changes = {}
    if wrong == "gate":
        changes["gate"] = object()
    elif wrong == "scope-object":
        changes["scoped_analysis"] = replace(admitted["scoped"])
    elif wrong == "key":
        changes["expected_key_id"] = "doctor-proof-cache-key:sha256:" + "0" * 64
    elif wrong == "receipt":
        changes["expected_receipt_id"] = content_identity({"wrong": "receipt"})
    else:
        changes["expected_owner_path_sha256"] = "0" * 64
    with monkeypatch.context() as blocked:
        blocked.setattr(Path, "stat", lambda *args, **kwargs: pytest.fail("bad pure binding touched storage"))
        with pytest.raises(bridge.HeaderProofCacheBridgeError):
            _bound(admitted, **changes)


def test_bound_census_retains_expected_receipt_identity_and_closed_local_counts(admitted, monkeypatch):
    before = list(admitted["calls"])
    original_render, render_calls = admitted["handle"].revalidate_for_render, []
    def bound_render(self):
        assert self is admitted["handle"]
        render_calls.append(True)
        return original_render()
    monkeypatch.setattr(bridge.HeaderProofCacheAdmission, "revalidate_for_render", bound_render)
    report = _observe(admitted)
    assert report["schema"].endswith("@2") and report["status"] == "measured"
    assert report["observation_current"] is True and report["accepted_receipts"] == 1
    assert report["expected_receipt_set_sha256"] == _sha(canonical_json(
        sorted(admitted["binding"].expected_receipt_ids)))
    assert report["header_admission_current_checks"] >= 4
    assert report["gate_lookup_calls"] == report["gate_render_revalidation_calls"] == 1
    assert render_calls == [True] and admitted["calls"] == before
    assert type(admitted["handle"].receipt) is ScopedProofReceipt
    assert not admitted["handle"].receipt.satisfies_completion()
    _closed(admitted, report)


def test_consistent_wrong_receipt_return_cannot_satisfy_independent_expected_identity(admitted, monkeypatch):
    original_lookup = admitted["gate"].lookup
    def substituted(key, **kwargs):
        result = original_lookup(key, **kwargs)
        receipt = replace(result.receipt, metadata={**result.receipt.metadata, "authored_extra": "different-canonical-body"})
        assert receipt.receipt_id != admitted["receipt"].receipt_id
        audit = replace(result.audit, receipt_id=receipt.receipt_id)
        return replace(result, receipt=receipt, audit=audit)
    monkeypatch.setattr(admitted["gate"], "lookup", substituted)
    report = _observe(admitted)
    assert report["status"] == "refused" and report["accepted_receipts"] is None
    assert report["gate_lookup_calls"] == 1 and report["gate_render_revalidation_calls"] == 0
    _closed(admitted, report)


@pytest.mark.parametrize("drift", ["source", "profile", "pins", "bounds", "native-owner", "cache-owner", "state-owner"])
def test_bound_census_rechecks_native_inputs_beyond_unchanged_context_hashes(admitted, drift):
    if drift == "source":
        (admitted["repository"] / "answer.py").write_text(PROGRAM + "\n# authored source drift\n")
    elif drift == "profile":
        object.__setattr__(admitted["prepared"].proof_key_profiles, "policy", admitted["profiles"].environment)
    elif drift == "pins":
        admitted["hammer"]._trusted_executable_pins = {}
    elif drift == "bounds":
        admitted["hammer"]._bounds = replace(admitted["prepared"].bounds, wall_time_ms=60_001)
    elif drift == "native-owner":
        admitted["hammer"]._authoritative_store = admitted["replacement_store"]
    elif drift == "cache-owner":
        admitted["gate"]._cache = admitted["replacement_cache"]
    else:
        admitted["gate"]._state._cache = admitted["replacement_cache"]
    report = _observe(admitted)
    assert report["status"] == "refused" and report["accepted_receipts"] is None
    assert report["gate_lookup_calls"] == report["gate_render_revalidation_calls"] == 0
    _closed(admitted, report)


@pytest.mark.parametrize("malformed", ["receipt-none", "receipt-list", "receipt-duplicate", "receipt-missing", "receipt-wrong",
                                      "handles-none", "handles-list", "handles-foreign", "handles-duplicate", "keys-missing"])
def test_malformed_bound_population_refuses_before_storage_or_callbacks(admitted, monkeypatch, malformed):
    binding = admitted["binding"]
    changes = {}
    if malformed == "receipt-none":
        binding = replace(binding, expected_receipt_ids=None)
    elif malformed == "receipt-list":
        binding = replace(binding, expected_receipt_ids=list(binding.expected_receipt_ids))
    elif malformed == "receipt-duplicate":
        binding = replace(binding, expected_receipt_ids=binding.expected_receipt_ids * 2)
    elif malformed == "receipt-missing":
        binding = replace(binding, expected_receipt_ids=())
    elif malformed == "receipt-wrong":
        binding = replace(binding, expected_receipt_ids=((admitted["key"].key_id, content_identity({"wrong": "expected"})),))
    elif malformed == "handles-none":
        changes["header_proof_admissions"] = None
    elif malformed == "handles-list":
        changes["header_proof_admissions"] = [admitted["handle"]]
    elif malformed == "handles-foreign":
        changes["header_proof_admissions"] = (object(),)
    elif malformed == "handles-duplicate":
        changes["header_proof_admissions"] = (admitted["handle"], admitted["handle"])
    else:
        changes["keys"] = ()
    def forbidden(*args, **kwargs):
        pytest.fail("malformed complete bound census touched storage or called host")
    with monkeypatch.context() as blocked:
        blocked.setattr(Path, "stat", forbidden)
        for name in ("lookup", "revalidate_for_render", "is_quarantined", "is_tombstoned"):
            blocked.setattr(admitted["gate"], name, forbidden)
        report = _observe(admitted, binding=binding, current_binding_probe=forbidden, **changes)
    assert report["status"] == "refused" and report["accepted_receipts"] is None
    assert report["current_binding_probe_calls"] == report["owner_identity_checks"] == 0
    assert report["gate_lookup_calls"] == report["gate_render_revalidation_calls"] == 0
    _closed(admitted, report)


def test_explicit_empty_bound_population_is_distinct_from_missing(admitted):
    binding = replace(admitted["binding"], expected_key_ids=(), expected_receipt_ids=())
    report = _observe(admitted, binding=binding, keys=(), header_proof_admissions=(),
                      current_binding_probe=lambda: binding)
    assert report["schema"].endswith("@2") and report["status"] == "measured"
    assert report["requested_key_count"] == report["accepted_receipts"] == 0
    assert report["gate_lookup_calls"] == report["gate_render_revalidation_calls"] == 0
    assert report["header_admission_current_checks"] == 0
    _closed(admitted, report)


def test_recursive_public_current_or_render_callback_refuses_and_guard_recovers(admitted, monkeypatch):
    original = admitted["gate"].is_tombstoned
    def recursive(key):
        admitted["handle"].assert_current()
        return original(key)
    with monkeypatch.context() as active:
        active.setattr(admitted["gate"], "is_tombstoned", recursive)
        with pytest.raises(bridge.HeaderProofCacheBridgeError, match="recursive"):
            admitted["handle"].assert_current()
        report = _observe(admitted)
        assert report["status"] == "refused" and report["accepted_receipts"] is None
    assert admitted["handle"].assert_current() is None
    assert _observe(admitted)["accepted_receipts"] == 1


def test_new_binding_none_field_preserves_legacy_canonical_payload_and_digest():
    binding = census.FormalStatusCensusBinding(task_sha256="a" * 64, revision_sha256="b" * 64,
        source_sha256="c" * 64, context_sha256="d" * 64, candidate_tree_id="tree:legacy-bound",
        owner_path_sha256="e" * 64, expected_key_ids=("doctor-proof-cache-key:sha256:" + "f" * 64,))
    payload = census._binding_payload(binding)
    assert "expected_receipt_ids" not in payload
    assert _sha(canonical_json(payload)) == "642d0c7f11710c5916159c5ffb4ee945eb5b6d0ece3d2148038f18845e4954a1"


def test_required_binding_refuses_raw_scoped_hit_before_render_and_default_remains_observable(admitted):
    binding = replace(admitted["binding"], expected_receipt_ids=None)
    options = dict(binding=binding, header_proof_admissions=None, current_binding_probe=lambda: binding)
    default = _observe(admitted, **options)
    assert default["schema"] == census.SCHEMA
    assert default["status"] == "measured" and default["accepted_receipts"] == 1
    required = _observe(admitted, **options, require_bound_scoped_receipts=True)
    assert required["status"] == "refused" and required["accepted_receipts"] is None
    assert required["observation_current"] is False
    assert required["gate_lookup_calls"] == 1 and required["gate_render_revalidation_calls"] == 0
    _closed(admitted, default)
    _closed(admitted, required)


def test_require_bound_scoped_selection_rejects_nonboolean_before_storage_or_probe(admitted, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid scoped binding selection touched owner storage or host callback")
    with monkeypatch.context() as blocked:
        blocked.setattr(Path, "stat", forbidden)
        blocked.setattr(admitted["gate"], "lookup", forbidden)
        report = _observe(admitted, current_binding_probe=forbidden, require_bound_scoped_receipts=1)
    assert report["status"] == "refused" and report["accepted_receipts"] is None
    assert report["current_binding_probe_calls"] == report["owner_identity_checks"] == 0
    assert report["gate_lookup_calls"] == report["gate_render_revalidation_calls"] == 0
    _closed(admitted, report)
