"""Task-bound worker transport; authored Source384 advice and real numerical replay.

The authored Source384 owner double measures transport only. Numerical cases
exercise the production grammar, source replay, artifacts and DuckLake owners.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundle
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source384
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical
from benchmarks.agent_supervisor.container_coding.test_terminal_numerical_context import context


def _prepared(**receipts):
    return dict(schema="supervisor-task-context-preparation@1", task_cid="task:one", task_id="one",
                metadata={}, **receipts)


def _write(root, prepared):
    ref = bundle.write_task_context_bundle(repository=root, prepared=prepared,
                                           output=root / "bundle.json")
    return dict(repository=root, artifact=ref["artifact"], expected_sha256=ref["sha256"],
                task_cid=prepared[0]["task_cid"], task_id=prepared[0]["task_id"])


@pytest.fixture
def source_context(tmp_path, monkeypatch):
    root, output = tmp_path / "repository", tmp_path / "source-advice"
    root.mkdir(); output.mkdir()
    raw = b"def predicate(left: int, right: int) -> bool:\n    return left < right\n"
    (root / "sample.py").write_bytes(raw)
    summary = dict(schema="terminal-source384-planning-summary@1",
        candidate_samples=[dict(unit_id="authored-unit", candidate_ir_json='{"kind":"program_expression"}',
            candidate_sha256="c" * 64, candidate_origin="learned_checkpoint_inference")],
        coverage={}, source_semantics_verified=False, whole_program_semantics_verified=False,
        **source384.AUTHORITY)
    receipt = dict(schema=source384.SCHEMA, repository=str(root), output=str(output),
        source_hashes={"sample.py": hashlib.sha256(raw).hexdigest()}, summary=summary,
        **source384.AUTHORITY)
    (output / "receipt.json").write_bytes(bundle._raw(receipt))
    calls = []

    def validate(*, repository, expected_receipt, **options):
        # A deliberate owner double: no checkpoint/model was loaded.
        assert repository == root and expected_receipt == receipt
        calls.append(options)

    monkeypatch.setattr(source384, "validate_source384_context", validate)
    return root, receipt, calls


def test_legacy_metadata_only_bundle_has_no_formalization_evidence(tmp_path):
    root = tmp_path / "repository"; root.mkdir()
    options = _write(root, [_prepared()])
    payload = json.loads((root / "bundle.json").read_bytes())
    assert payload["schema"] == bundle.SCHEMA
    assert bundle.load_task_context_selection(**options) == {"metadata": {}}
    assert bundle.load_task_context_nomination(**options) == {}


@pytest.mark.parametrize("schema", [bundle.SOURCE384_SCHEMA, bundle.SOURCE384_REFERENCE_SCHEMA])
def test_legacy_source_receipts_deliver_exact_task_bound_summary(source_context, schema):
    root, receipt, calls = source_context
    options = _write(root, [_prepared(source384_context=receipt)])
    if schema == bundle.SOURCE384_SCHEMA:
        payload = json.loads((root / "bundle.json").read_bytes())
        payload["schema"] = schema
        payload["tasks"][0]["source384_context"] = receipt
        raw = bundle._raw(payload); (root / "bundle.json").write_bytes(raw)
        options["expected_sha256"] = hashlib.sha256(raw).hexdigest()
    historical = bundle.read_task_context_historical_selection(**options)
    assert "worker_evidence" not in historical
    selected = bundle.load_task_context_selection(**options)
    evidence = selected["worker_evidence"]
    assert calls == [{}]
    assert evidence["schema"] == bundle.WORKER_EVIDENCE_SCHEMA
    assert evidence["repository"] == str(root)
    assert evidence["task_id"] == "one" and evidence["task_cid"] == "task:one"
    assert evidence["context_bundle"] == dict(artifact=options["artifact"], sha256=options["expected_sha256"])
    context = evidence["contexts"][0]
    assert context["kind"] == "source384-candidate-context"
    assert context["summary"] == receipt["summary"]
    assert context["summary_sha256"] == hashlib.sha256(bundle._raw(receipt["summary"])).hexdigest()
    assert context["source_hashes_sha256"] == hashlib.sha256(bundle._raw(receipt["source_hashes"])).hexdigest()
    assert context["receipt_reference"]["receipt_sha256"] == hashlib.sha256(bundle._raw(receipt)).hexdigest()
    assert all(evidence[key] is False for key in (
        "proof_authority", "formalization_authority", "execution_authority", "completion_authority",
        "planning_authority", "dispatch_authority", "source_semantics_verified", "whole_program_semantics_verified"))
    assert evidence["source_obligations_discharged"] == 0
    assert bundle.load_task_context_nomination(**options) == {}
    assert len(bundle._raw(evidence)) <= bundle.MAX_WORKER_EVIDENCE_BYTES


@pytest.mark.parametrize("identity", [{"task_id": "two"}, {"task_cid": "task:two"},
                                    {"task_cid": "task:two", "task_id": "two"}])
def test_foreign_task_is_refused_before_owner_validation(source_context, identity):
    root, receipt, calls = source_context
    options = _write(root, [_prepared(source384_context=receipt)])
    with pytest.raises(ValueError, match="task"):
        bundle.load_task_context_selection(**{**options, **identity})
    assert calls == []


@pytest.mark.parametrize("mutation", ["receipt", "source", "bundle"])
def test_mutation_during_source_validation_cannot_reach_worker(source_context, monkeypatch, mutation):
    root, receipt, _ = source_context
    options = _write(root, [_prepared(source384_context=receipt)])
    def validate(**kwargs):
        path = {"receipt": Path(receipt["output"]) / "receipt.json", "source": root / "sample.py",
                "bundle": root / "bundle.json"}[mutation]
        path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(source384, "validate_source384_context", validate)
    with pytest.raises(ValueError):
        bundle.load_task_context_selection(**options)


@pytest.mark.parametrize("mutation", ["missing", "oversize"])
def test_missing_or_oversize_selected_summary_cannot_be_silently_dropped(source_context, mutation):
    root, receipt, _ = source_context
    if mutation == "missing":
        del receipt["summary"]
    else:
        receipt["summary"]["extra"] = "x" * 8192
    (Path(receipt["output"]) / "receipt.json").write_bytes(bundle._raw(receipt))
    options = _write(root, [_prepared(source384_context=receipt)])
    with pytest.raises((ValueError, KeyError)):
        bundle.load_task_context_selection(**options)


def test_numerical_bundle_replays_real_owners_and_exposes_unit_logic_links(context):
    root, receipt = context
    options = _write(root, [_prepared(numerical_formalization=receipt)])
    payload = json.loads((root / "bundle.json").read_bytes())
    assert payload["schema"] == bundle.FORMALIZATION_REFERENCE_SCHEMA
    ref = payload["tasks"][0]["numerical_formalization"]
    assert ref["schema"] == bundle.NUMERICAL_REFERENCE_SCHEMA
    assert set(ref) == {"schema", "output", "receipt_sha256", "receipt_bytes"}
    assert ref["receipt_sha256"] == receipt["receipt_sha256"]
    historical = bundle.read_task_context_historical_selection(**options)
    assert historical == dict(metadata={}, numerical_formalization=receipt)
    selected = bundle.load_task_context_selection(**options)
    ctx = selected["worker_evidence"]["contexts"][0]
    assert ctx["kind"] == "numerical-formalization-context"
    assert ctx["summary"] == receipt["summary"]
    assert ctx["summary"]["source_unit_links"][0]["function_id"]
    assert ctx["summary"]["source_unit_links"][0]["link_report_sha256"]
    assert ctx["summary"]["source_unit_links_establish_learned_coverage"] is False
    assert ctx["summary"]["source_obligations_discharged"] == 0


@pytest.mark.parametrize("mutation", ["source", "report", "unit_link", "lake", "receipt", "producer"])
def test_numerical_bundle_rejects_currentness_drift_before_worker_delivery(context, monkeypatch, mutation):
    root, receipt = context
    options = _write(root, [_prepared(numerical_formalization=receipt)])
    output = Path(receipt["output"])
    if mutation == "producer":
        monkeypatch.setattr(numerical, "_producers", lambda: {"different": "producer"})
    else:
        path = {"source": root / "sample.py", "report": output / "report-0000.json",
                "unit_link": output / "unit-links-0000.json", "lake": output / "metadata-lake/metadata.ducklake",
                "receipt": output / "receipt.json"}[mutation]
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        bundle.load_task_context_selection(**options)


def test_numerical_nomination_rejects_forged_summary_even_with_valid_outer_hash(context):
    root, receipt = context
    forged = deepcopy(receipt)
    forged["summary"]["source_obligations_discharged"] = 100
    unsigned = {k: v for k, v in forged.items() if k != "receipt_sha256"}
    raw = bundle._raw(unsigned)
    (Path(receipt["output"]) / "receipt.json").write_bytes(raw)
    forged["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    options = _write(root, [_prepared(numerical_formalization=forged)])
    with pytest.raises(ValueError, match="planning/world evidence differs"):
        bundle.load_task_context_selection(**options)


def test_numerical_bundle_task_selection_does_not_require_other_task_receipts(context):
    root, receipt = context
    options = _write(root, [_prepared(numerical_formalization=receipt),
        {**_prepared(), "task_id": "two", "task_cid": "task:two"}])
    assert bundle.load_task_context_selection(**{**options, "task_id": "two", "task_cid": "task:two"}) == {"metadata": {}}


def test_combined_evidence_preserves_distinct_origins_and_cross_owner_currentness(context, monkeypatch):
    root, receipt = context
    output = root.parent / "source384"; output.mkdir()
    learned = dict(schema=source384.SCHEMA, repository=str(root), output=str(output),
        source_hashes=receipt["source_hashes"], summary=dict(schema="terminal-source384-planning-summary@1",
            candidate_samples=[], **source384.AUTHORITY), **source384.AUTHORITY)
    (output / "receipt.json").write_bytes(bundle._raw(learned))
    monkeypatch.setattr(source384, "validate_source384_context", lambda **kw: None)
    options = _write(root, [_prepared(source384_context=learned, numerical_formalization=receipt)])
    selected = bundle.load_task_context_selection(**options)
    assert [row["kind"] for row in selected["worker_evidence"]["contexts"]] == [
        "source384-candidate-context", "numerical-formalization-context"]
    real = numerical.validate_numerical_context
    def changed(**kwargs):
        result = real(**kwargs)
        (root / "sample.py").write_text("# changed after owner replay\n")
        return result
    monkeypatch.setattr(numerical, "validate_numerical_context", changed)
    with pytest.raises(ValueError, match="signed population"):
        bundle.load_task_context_selection(**options)
