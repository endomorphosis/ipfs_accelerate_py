"""Transport/gating controls; authored receipts do not qualify model inference."""
from copy import deepcopy
import hashlib
import json
import sys
import types

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles


@pytest.fixture
def validator(monkeypatch):
    # Exercise the production import/API while this control isolates the bundle
    # contract from model loading. Native receipt verification has separate tests.
    module = types.ModuleType("ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context")
    calls = []
    module.validate_source384_context = lambda **kwargs: calls.append(deepcopy(kwargs))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module, calls


def _prepared(alias="TASK-1", *, selected=True):
    item = {"schema": "supervisor-task-context-preparation@1", "task_cid": "cid:" + alias,
            "task_id": alias, "metadata": {"semantic context artifact": ".runtime/semantic.json",
                                            "semantic context sha256": "a" * 64}}
    if selected:
        item["source384_context"] = {"schema": "terminal-source384-repository-context@1",
            "artifact": ".runtime/source384/" + alias + ".json", "sha256": "b" * 64,
            "completion_authority": False}
    return item


def _load(root, bundle, alias="TASK-1"):
    return bundles.load_task_context_nomination(repository=root, artifact=bundle["artifact"],
        expected_sha256=bundle["sha256"], task_cid="cid:" + alias, task_id=alias)


def _rewrite(root, bundle, payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    (root / bundle["artifact"]).write_bytes(raw)
    return {**bundle, "sha256": hashlib.sha256(raw).hexdigest()}


def _write_legacy(*, repository, prepared, output):
    # Retained @2 envelopes remain readable after writers move to references.
    payload = {"schema": bundles.SOURCE384_SCHEMA, "completion_authority": False,
        "tasks": [{key: item[key] for key in ("task_cid", "task_id", "metadata", "source384_context")}
                  for item in prepared]}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    output.write_bytes(raw)
    return {"artifact": output.relative_to(repository).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def test_unselected_bundle_preserves_v1_bytes_and_skips_source384(tmp_path, validator):
    item = _prepared(selected=False)
    bundle = bundles.write_task_context_bundle(repository=tmp_path, prepared=[item], output=tmp_path / "v1.json")
    expected = {"schema": bundles.SCHEMA, "tasks": [{key: item[key] for key in ("task_cid", "task_id", "metadata")}],
                "completion_authority": False}
    assert (tmp_path / "v1.json").read_bytes() == json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    assert _load(tmp_path, bundle) == item["metadata"]
    assert validator[1] == []


def test_v2_revalidates_only_exact_selected_task_receipt_on_every_read(tmp_path, validator):
    first, second = _prepared(), _prepared("TASK-2")
    bundle = _write_legacy(repository=tmp_path, prepared=[first, second], output=tmp_path / "v2.json")
    assert json.loads((tmp_path / "v2.json").read_bytes())["schema"] == bundles.SOURCE384_SCHEMA
    for _ in range(2):
        assert _load(tmp_path, bundle, "TASK-2") == second["metadata"]
    assert validator[1] == [{"repository": tmp_path, "expected_receipt": second["source384_context"]}] * 2
    with pytest.raises(ValueError, match="foreign task"):
        bundles.load_task_context_nomination(repository=tmp_path, artifact=bundle["artifact"],
            expected_sha256=bundle["sha256"], task_cid=first["task_cid"], task_id=second["task_id"])
    assert len(validator[1]) == 2


@pytest.mark.parametrize("mutation", ["missing", "null", "unknown-schema", "oversize", "nan", "extra-task-field", "downgrade"])
def test_v2_malformed_reference_rejects_before_validation(tmp_path, validator, mutation):
    bundle = _write_legacy(repository=tmp_path, prepared=[_prepared()], output=tmp_path / "v2.json")
    payload = json.loads((tmp_path / "v2.json").read_bytes())
    task = payload["tasks"][0]
    if mutation == "missing":
        del task["source384_context"]
    elif mutation == "null":
        task["source384_context"] = None
    elif mutation == "unknown-schema":
        task["source384_context"]["schema"] = "foreign@1"
    elif mutation == "oversize":
        task["source384_context"]["summary"] = "x" * 32_768
    elif mutation == "nan":
        task["source384_context"]["score"] = float("nan")
    elif mutation == "extra-task-field":
        task["execution_authority"] = True
    else:
        payload["schema"] = bundles.SCHEMA
    rewritten = _rewrite(tmp_path, bundle, payload)
    with pytest.raises(ValueError):
        _load(tmp_path, rewritten)
    assert validator[1] == []


def test_v2_writer_cannot_silently_drop_missing_selection(tmp_path):
    with pytest.raises(ValueError):
        bundles.write_task_context_bundle(repository=tmp_path,
            prepared=[_prepared(), _prepared("TASK-2", selected=False)], output=tmp_path / "v2.json")
    assert not (tmp_path / "v2.json").exists()


def test_v2_digest_drift_rejects_before_native_validator(tmp_path, validator):
    bundle = _write_legacy(repository=tmp_path, prepared=[_prepared()], output=tmp_path / "v2.json")
    with (tmp_path / bundle["artifact"]).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="digest differs"):
        _load(tmp_path, bundle)
    assert validator[1] == []


def test_validator_drift_refusal_propagates_without_metadata(tmp_path, validator):
    bundle = _write_legacy(repository=tmp_path, prepared=[_prepared()], output=tmp_path / "v2.json")
    def stale(**kwargs):
        raise ValueError("selected Source384 producer changed")
    validator[0].validate_source384_context = stale
    with pytest.raises(ValueError, match="producer changed"):
        _load(tmp_path, bundle)


def test_worker_context_read_rechecks_v2_receipt_before_prompt(tmp_path, validator):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import TodoImplementationDaemon, PortalTask

    item = _prepared()
    bundle = _write_legacy(repository=tmp_path, prepared=[item], output=tmp_path / "v2.json")
    daemon = object.__new__(TodoImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon._task_context_nomination_bundle = bundle
    task = PortalTask(task_id=item["task_id"], title="Read context", status="ready", completion="manual",
        priority="P1", track="context", outputs=[], validation=[], acceptance="context remains current",
        metadata={}, canonical_task_cid=item["task_cid"])
    assert daemon._task_metadata_value(task, "semantic context artifact") == item["metadata"]["semantic context artifact"]
    assert len(validator[1]) == 1
    def stale(**kwargs):
        raise ValueError("selected Source384 source changed")
    validator[0].validate_source384_context = stale
    with pytest.raises(ValueError, match="source changed"):
        daemon._task_metadata_value(task, "world context artifact")
    assert task.metadata == {}
