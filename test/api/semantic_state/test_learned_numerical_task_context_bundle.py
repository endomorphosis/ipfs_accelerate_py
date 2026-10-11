"""Authored transport controls; these tests do not qualify learned inference."""
from copy import deepcopy
import hashlib
import json
import sys
import types

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundle


@pytest.fixture
def selected(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    source = b"import numpy as np\ndef select(values):\n    return np.argmax(values)\n"
    (root / "select.py").write_bytes(source)
    hashes = {"select.py": hashlib.sha256(source).hexdigest()}
    owners, calls, receipts = {}, [], {}
    definitions = (
        ("source384_context", "source384_repository_context", "terminal-source384-repository-context@1", "validate_source384_context"),
        ("numerical_formalization", "numerical_formalization_context", "supervisor-numerical-formalization@1", "validate_numerical_context"),
        ("numerical_selection_384", "numerical_selection_context", "supervisor-learned-numerical-selection@1", "validate_numerical_selection_context"),
    )
    def sources(repository, expected):
        if any(hashlib.sha256((repository / path).read_bytes()).hexdigest() != digest
               for path, digest in expected.items()):
            raise ValueError("selected source changed")
    for key, module_name, schema, validator in definitions:
        owner = types.ModuleType("ipfs_accelerate_py.agent_supervisor.runtime." + module_name)
        owner.SCHEMA, owner._sources = schema, sources
        def validate(*, repository, expected_receipt, _key=key, **kwargs):
            calls.append(_key)
            sources(repository, expected_receipt["source_hashes"])
        setattr(owner, validator, validate)
        owners[key] = owner
        monkeypatch.setitem(sys.modules, owner.__name__, owner)
        output = tmp_path / key
        output.mkdir()
        receipt = dict(schema=schema, repository=str(root), output=str(output), source_hashes=hashes,
            summary=dict(schema="authored-transport-summary@1", candidates=[dict(candidate_ir_json='{"op":"argmax"}')],
                proof_authority=False), proof_authority=False)
        raw = bundle._raw(receipt)
        (output / "receipt.json").write_bytes(raw)
        if key != "source384_context":
            receipt["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
        receipts[key] = receipt
    item = dict(schema="supervisor-task-context-preparation@1", task_id="TASK-1", task_cid="cid:task", metadata={}, **receipts)
    binding = bundle.write_task_context_bundle(repository=root, prepared=[item], output=root / "bundle.json")
    args = dict(repository=root, artifact=binding["artifact"], expected_sha256=binding["sha256"],
                task_cid=item["task_cid"], task_id=item["task_id"])
    return dict(root=root, receipts=receipts, owners=owners, calls=calls, item=item, args=args)


def test_three_families_keep_exact_references_and_ordered_worker_advice(selected):
    c = selected
    assert json.loads((c["root"] / "bundle.json").read_bytes())["schema"] == bundle.LEARNED_FORMALIZATION_REFERENCE_SCHEMA
    result = bundle.load_task_context_selection(**c["args"])
    assert all(result[key] == receipt for key, receipt in c["receipts"].items())
    evidence = result["worker_evidence"]
    assert [row["kind"] for row in evidence["contexts"]] == [
        "source384-candidate-context", "numerical-formalization-context", "learned-numerical-selection-context"]
    assert c["calls"] == list(bundle.FORMALIZATION_KEYS)
    assert evidence["source_obligations_discharged"] == 0
    assert evidence["proof_authority"] is evidence["execution_authority"] is False
    row = evidence["contexts"][-1]
    assert row["summary"] == c["receipts"]["numerical_selection_384"]["summary"]
    assert row["receipt_reference"]["receipt_sha256"] == c["receipts"]["numerical_selection_384"]["receipt_sha256"]


def test_learned_only_task_can_share_bundle_with_unselected_task(selected):
    c = selected
    learned = {key: value for key, value in c["item"].items() if key not in bundle.FORMALIZATION_KEYS[:2]}
    plain = dict(schema=learned["schema"], task_id="TASK-2", task_cid="cid:plain", metadata={})
    binding = bundle.write_task_context_bundle(repository=c["root"], prepared=[learned, plain], output=c["root"] / "mixed.json")
    args = {**c["args"], "artifact": binding["artifact"], "expected_sha256": binding["sha256"]}
    assert [row["kind"] for row in bundle.load_task_context_selection(**args)["worker_evidence"]["contexts"]] == ["learned-numerical-selection-context"]
    assert bundle.load_task_context_selection(**{**args, "task_id": "TASK-2", "task_cid": "cid:plain"}) == {"metadata": {}}


@pytest.mark.parametrize("change", ["receipt", "source", "validator", "during_validation", "downgrade"])
def test_changed_learned_selection_cannot_reach_live_worker(selected, change):
    c = selected
    owner = c["owners"]["numerical_selection_384"]
    if change == "receipt":
        path = __import__("pathlib").Path(c["receipts"]["numerical_selection_384"]["output"]) / "receipt.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "source":
        (c["root"] / "select.py").write_text("changed source")
        # Historical reading checks retained identity, not present source bytes.
        assert bundle.read_task_context_historical_selection(**c["args"])["numerical_selection_384"] == c["receipts"]["numerical_selection_384"]
    elif change == "validator":
        def stale(**kwargs):
            raise ValueError("selected checkpoint changed")
        owner.validate_numerical_selection_context = stale
    elif change == "during_validation":
        owner.validate_numerical_selection_context = lambda **kwargs: (c["root"] / "select.py").write_text("changed during last owner validation")
    else:
        path = c["root"] / "bundle.json"
        value = json.loads(path.read_bytes())
        value["schema"] = bundle.FORMALIZATION_REFERENCE_SCHEMA
        raw = bundle._raw(value)
        path.write_bytes(raw)
        c["args"]["expected_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        bundle.load_task_context_selection(**c["args"])


def test_receipt_overlap_conflict_is_rejected_even_if_owners_accept(selected):
    c = selected
    receipt = deepcopy(c["receipts"]["numerical_selection_384"])
    receipt["source_hashes"]["select.py"] = "f" * 64
    receipt.pop("receipt_sha256")
    raw = bundle._raw(receipt)
    from pathlib import Path
    (Path(receipt["output"]) / "receipt.json").write_bytes(raw)
    receipt["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    c["owners"]["numerical_selection_384"].validate_numerical_selection_context = lambda **kwargs: None
    binding = bundle.write_task_context_bundle(repository=c["root"],
        prepared=[{**c["item"], "numerical_selection_384": receipt}], output=c["root"] / "conflict.json")
    with pytest.raises(ValueError, match="disagree"):
        bundle.load_task_context_selection(**{**c["args"], "artifact": binding["artifact"], "expected_sha256": binding["sha256"]})


@pytest.mark.parametrize("omit", [True, False])
def test_rehashed_published_successor_cannot_drop_or_reuse_learned_family(selected, omit):
    from ipfs_accelerate_py.agent_supervisor.runtime import published_task_context as published
    c = selected
    value = dict(task_cid=c["args"]["task_cid"], task_id=c["args"]["task_id"],
        predecessor_bundle={"artifact": c["args"]["artifact"], "sha256": c["args"]["expected_sha256"]}, **c["receipts"])
    if omit:
        value.pop("numerical_selection_384")
    with pytest.raises(ValueError, match="cannot omit or add" if omit else "regeneration policy is unavailable"):
        published._verify_predecessor(root=c["root"], result=value)
