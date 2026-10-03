"""Full receipt selection remains bound while task nominations stay small."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture
def case(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "private"
    output.mkdir()
    # An authored complete signed population exceeds the obsolete inline bound.
    sources = {f"modules/module_{i:03d}.py": hashlib.sha256(str(i).encode()).hexdigest() for i in range(256)}
    receipt = dict(schema="terminal-source384-repository-context@1", repository=str(repository),
        output=str(output), source_hashes=sources,
        source_inventory=[dict(path=p, sha256=h, source_cid="cid:" + h, disposition="captured")
                          for p, h in sources.items()], producer={"consumer": "a" * 64},
        summary={"candidate_samples": [], "completion_authority": False},
        completion_authority=False, proof_authority=False)
    (output / "receipt.json").write_bytes(raw(receipt))
    calls = []
    module = types.ModuleType("ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context")
    module.validate_source384_context = lambda **kw: calls.append(deepcopy(kw))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return types.SimpleNamespace(repository=repository, output=output, receipt=receipt, calls=calls,
                                 validator=module)


def item(case, i=0):
    return dict(schema="supervisor-task-context-preparation@1", task_cid=f"cid:{i}", task_id=f"TASK-{i}",
        metadata={"semantic context artifact": ".runtime/semantic.json", "semantic context sha256": "a" * 64},
        source384_context=deepcopy(case.receipt))


def write(case, count=1):
    return bundles.write_task_context_bundle(repository=case.repository,
        prepared=[item(case, i) for i in range(count)], output=case.repository / "bundle.json")


def load(case, bundle, i=0, historical=False):
    function = bundles.read_task_context_historical_selection if historical else bundles.load_task_context_selection
    return function(repository=case.repository, artifact=bundle["artifact"], expected_sha256=bundle["sha256"],
        task_cid=f"cid:{i}", task_id=f"TASK-{i}")


def rewrite(case, bundle, payload):
    encoded = raw(payload)
    (case.repository / bundle["artifact"]).write_bytes(encoded)
    return dict(bundle, sha256=hashlib.sha256(encoded).hexdigest())


def test_sixteen_full_receipts_become_small_exact_references(case):
    assert 32768 < len(raw(case.receipt)) <= bundles.MAX_SOURCE384_RECEIPT_BYTES
    bundle = write(case, 16)
    encoded = (case.repository / bundle["artifact"]).read_bytes()
    payload = json.loads(encoded)
    assert payload["schema"] == bundles.SOURCE384_REFERENCE_SCHEMA
    assert len(encoded) < 16384 < bundles.MAX_BYTES
    for row in payload["tasks"]:
        reference = row["source384_context"]
        assert set(reference) == {"schema", "output", "receipt_sha256", "receipt_bytes"}
        assert reference["receipt_sha256"] == hashlib.sha256(raw(case.receipt)).hexdigest()
        assert reference["receipt_bytes"] == len(raw(case.receipt))
    for _ in range(2):
        assert load(case, bundle, 15)["source384_context"] == case.receipt
    assert case.calls == [{"repository": case.repository, "expected_receipt": case.receipt}] * 2


def test_historical_read_binds_bytes_without_granting_currentness(case):
    bundle = write(case)
    def stale(**kw):
        raise ValueError("live source changed")
    case.validator.validate_source384_context = stale
    assert load(case, bundle, historical=True)["source384_context"] == case.receipt
    with pytest.raises(ValueError, match="live source changed"):
        load(case, bundle)
    changed = dict(case.receipt, producer={"consumer": "b" * 64})
    (case.output / "receipt.json").write_bytes(raw(changed))
    with pytest.raises(ValueError, match="receipt bytes differ"):
        load(case, bundle, historical=True)


def test_unselected_references_are_not_read(case):
    bundle = write(case, 2)
    payload = json.loads((case.repository / bundle["artifact"]).read_bytes())
    payload["tasks"][0]["source384_context"]["output"] = str(case.output / "absent")
    bundle = rewrite(case, bundle, payload)
    assert load(case, bundle, 1)["source384_context"] == case.receipt
    with pytest.raises(FileNotFoundError):
        load(case, bundle, 0)


@pytest.mark.parametrize("mutation", ["extra", "missing", "schema", "size-bool", "size-zero", "size-large",
    "hash", "relative", "parent", "worker", "ancestor", "oversize", "downgrade"])
def test_reference_shape_refuses_before_native_validation(case, mutation):
    bundle = write(case)
    payload = json.loads((case.repository / bundle["artifact"]).read_bytes())
    ref = payload["tasks"][0]["source384_context"]
    if mutation == "extra": ref["completion_authority"] = True
    elif mutation == "missing": del ref["receipt_sha256"]
    elif mutation == "schema": ref["schema"] = "unknown@1"
    elif mutation == "size-bool": ref["receipt_bytes"] = True
    elif mutation == "size-zero": ref["receipt_bytes"] = 0
    elif mutation == "size-large": ref["receipt_bytes"] = bundles.MAX_SOURCE384_RECEIPT_BYTES + 1
    elif mutation == "hash": ref["receipt_sha256"] = "G" * 64
    elif mutation == "relative": ref["output"] = "private"
    elif mutation == "parent": ref["output"] = str(case.output) + "/../private"
    elif mutation == "worker": ref["output"] = str(case.repository / "private")
    elif mutation == "ancestor": ref["output"] = str(case.repository.parent)
    elif mutation == "oversize": ref["output"] = "/" + "x" * bundles.MAX_SOURCE384_REFERENCE_BYTES
    else: payload["schema"] = bundles.SOURCE384_SCHEMA
    with pytest.raises(ValueError):
        load(case, rewrite(case, bundle, payload))
    assert case.calls == []


@pytest.mark.parametrize("mutation", ["hash", "size", "schema", "output", "repository", "duplicate", "noncanonical"])
def test_full_receipt_envelope_drift_rejects_even_if_reference_is_resigned(case, mutation):
    bundle = write(case)
    payload = json.loads((case.repository / bundle["artifact"]).read_bytes())
    ref = payload["tasks"][0]["source384_context"]
    receipt = deepcopy(case.receipt)
    if mutation == "schema": receipt["schema"] = "unknown@1"
    elif mutation == "output": receipt["output"] = str(case.output / "foreign")
    elif mutation == "repository": receipt["repository"] = str(case.repository / "foreign")
    encoded = raw(receipt)
    if mutation == "duplicate": encoded = encoded[:-1] + b',"schema":"terminal-source384-repository-context@1"}'
    elif mutation == "noncanonical": encoded += b" "
    (case.output / "receipt.json").write_bytes(encoded)
    ref["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    ref["receipt_bytes"] = len(encoded)
    if mutation == "hash": ref["receipt_sha256"] = "0" * 64
    elif mutation == "size": ref["receipt_bytes"] -= 1
    with pytest.raises(ValueError):
        load(case, rewrite(case, bundle, payload))
    assert case.calls == []


@pytest.mark.parametrize("mutation", ["symlink", "parent-link", "hardlink", "fifo", "oversize", "missing"])
def test_selected_artifact_requires_bounded_canonical_regular_bytes(case, mutation):
    bundle = write(case)
    leaf = case.output / "receipt.json"
    if mutation == "symlink":
        other = case.output / "other.json"
        leaf.rename(other)
        leaf.symlink_to(other)
    elif mutation == "parent-link":
        other = case.output.with_name("moved")
        case.output.rename(other)
        case.output.symlink_to(other, target_is_directory=True)
    elif mutation == "hardlink": os.link(leaf, case.output / "other.json")
    elif mutation == "fifo":
        leaf.unlink()
        os.mkfifo(leaf)
    elif mutation == "oversize": leaf.write_bytes(b"x" * (bundles.MAX_SOURCE384_RECEIPT_BYTES + 1))
    else: leaf.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        load(case, bundle)
    assert case.calls == []


def test_writer_refuses_nonretained_selection_and_missing_task_selection(case):
    prepared = item(case)
    prepared["source384_context"]["producer"]["consumer"] = "b" * 64
    with pytest.raises(ValueError, match="receipt bytes differ"):
        bundles.write_task_context_bundle(repository=case.repository, prepared=[prepared],
            output=case.repository / "bundle.json")
    missing = item(case, 1)
    del missing["source384_context"]
    with pytest.raises(ValueError, match="selected Source384"):
        bundles.write_task_context_bundle(repository=case.repository, prepared=[item(case), missing],
            output=case.repository / "bundle.json")
    assert not (case.repository / "bundle.json").exists()


def test_receipt_limit_matches_canonical_consumer():
    # The isolated fake validator is not active in this compatibility check.
    from ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context import MAX_RECEIPT_BYTES
    assert bundles.MAX_SOURCE384_RECEIPT_BYTES == MAX_RECEIPT_BYTES == 131072
