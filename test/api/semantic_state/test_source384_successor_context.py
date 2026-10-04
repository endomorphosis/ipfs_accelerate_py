"""Advisory successor contracts with explicitly authored numerical seams.

These tests exercise real receipt/config/source files and canonical consumers;
the index and numerical worker are isolated seams, not inference qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as context
from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles


@pytest.fixture
def successor_case(tmp_path, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import codebase_source_384 as models
    from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as units
    from ipfs_datasets_py.logic.software_contracts import codebase_resources as resources
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "module.py"
    source.write_text("def answer(): return 1\n")
    config_path = tmp_path / "config.json"
    config = {"schema": "terminal-source384-config@2", "checkpoint_path": str(tmp_path / "checkpoint"),
              "checkpoint_sha256": "c" * 64, "embedding_snapshot": str(tmp_path / "embedding")}
    config_path.write_bytes(context._raw(config))
    cid = cid_for_structured({"authored": True})
    old_head = CodebaseHead("fixture", 1, cid, cid, f"rev:fixture:snapshot:{cid}", cid)
    new_head = CodebaseHead("fixture-next", 1, cid, cid, f"rev:fixture-next:snapshot:{cid}", cid)
    producer = {"consumer": "authored-test-consumer", "source_owners": {"fixture": "authored"}}
    previous = tmp_path / "previous"
    previous.mkdir()
    inference = {"authored_predecessor": True}
    (previous / "inference.json").write_bytes(context._raw(inference))
    predecessor = dict(schema=context.HEADER_SCHEMA, output=str(previous), repository=str(repository),
        source_hashes={"module.py": context._sha(source.read_bytes())}, producer=producer,
        training_steps=0, config_path=str(config_path), config_sha256=context._sha(config_path.read_bytes()),
        checkpoint_sha256=config["checkpoint_sha256"], version_id="authored-parent",
        inference_sha256=context._sha(context._raw(inference)), source_head=old_head.to_dict(),
        source_applicability_nomination={"authored": True}, header_consumer_sha256=context._header_pin(),
        **context.AUTHORITY)
    (previous / "receipt.json").write_bytes(context._raw(predecessor))
    source.write_text("def answer(): return 2\n")
    observations = []
    hooks = SimpleNamespace(after_inference=None, after_validation=None, load_config=None)

    def load_config(path):
        assert path == config_path
        if hooks.load_config:
            hooks.load_config()
        return deepcopy(config)

    @contextmanager
    def lease(**kwargs):
        observations.append(("lease", kwargs))
        yield object()

    class Index:
        def prepare_current(self, root, **kwargs):
            assert root == repository
            observations.append(("index", kwargs))
            return SimpleNamespace(head=new_head)

    @contextmanager
    def owners(path):
        yield Index(), object()

    def infer(index, root, **kwargs):
        observations.append(("infer", kwargs))
        value = {"native_worker_executed": True, "inference_executed": True,
            "report": {"key": {"source_head": new_head.to_dict(), "version_id": "authored-parent",
                "original_checkpoint_sha256": config["checkpoint_sha256"]},
                "preparation": {"paths": ["module.py"], "max_functions": 1024, "max_selected_units": 128},
                "output": {"model_loads": 1}}}
        if hooks.after_inference:
            hooks.after_inference()
        return value

    def validate(*args, **kwargs):
        observations.append(("validate", kwargs))
        if hooks.after_validation:
            hooks.after_validation()

    monkeypatch.setattr(context, "load_source384_config", load_config)
    monkeypatch.setattr(context, "_pins", lambda: deepcopy(producer))
    monkeypatch.setattr(context, "_owners", owners)
    monkeypatch.setattr(context, "_inventory", lambda index, head, hashes:
        [{"path": name, "sha256": sha, "disposition": "captured"} for name, sha in hashes.items()])
    monkeypatch.setattr(context, "_summary", lambda inference, **kwargs: {"authored_summary": kwargs})
    monkeypatch.setattr(resources, "acquire_codebase_resources", lease)
    monkeypatch.setattr(models, "register_shared_parent", lambda *args, **kwargs: "authored-parent")
    monkeypatch.setattr(units, "infer_shared_parent_units", infer)
    monkeypatch.setattr(units, "validate_shared_parent_units", validate)
    case = SimpleNamespace(repository=repository, source=source, predecessor=predecessor,
        output=tmp_path / "successor", previous=previous, config_path=config_path,
        observations=observations, hooks=hooks)
    case.prepare = lambda **kwargs: context.prepare_source384_successor_context(
        repository=repository, source_hashes={"module.py": context._sha(source.read_bytes())},
        output=case.output, predecessor_receipt=predecessor, **kwargs)
    return case


def test_advisory_successor_preserves_assets_scope_and_historical_lineage(successor_case):
    c = successor_case
    old_bytes = (c.previous / "receipt.json").read_bytes()
    receipt = c.prepare()
    assert receipt["schema"] == context.SUCCESSOR_SCHEMA
    assert receipt["config_path"] == c.predecessor["config_path"]
    for name in ("config_sha256", "checkpoint_sha256", "version_id", "producer"):
        assert receipt[name] == c.predecessor[name]
    assert "source_applicability_nomination" not in receipt
    assert "header_consumer_sha256" not in receipt
    assert receipt["source_hashes"].keys() == c.predecessor["source_hashes"].keys()
    assert receipt["source_hashes"] != c.predecessor["source_hashes"]
    assert receipt["successor_lineage"]["predecessor_receipt_sha256"] == context._sha(old_bytes)
    assert receipt["requires_independent_manifest"] is True
    assert receipt["planning_authority"] is receipt["dispatch_authority"] is False
    assert all(receipt[name] is False for name in context.AUTHORITY)
    assert context.validate_source384_context(repository=c.repository, expected_receipt=receipt) == receipt
    assert len([item for item in c.observations if item[0] == "infer"]) == 1
    assert (c.previous / "receipt.json").read_bytes() == old_bytes
    bundle = bundles.write_task_context_bundle(repository=c.repository, prepared=[{
        "schema": "supervisor-task-context-preparation@1", "task_id": "TASK", "task_cid": "cid:task",
        "metadata": {}, "source384_context": receipt}], output=c.repository / "bundle.json")
    assert bundles.load_task_context_selection(repository=c.repository, artifact=bundle["artifact"],
        expected_sha256=bundle["sha256"], task_id="TASK", task_cid="cid:task")["source384_context"] == receipt
    assert len([item for item in c.observations if item[0] == "infer"]) == 1


@pytest.mark.parametrize("damage", ["predecessor_bytes", "config_bytes", "source_during_inference", "source_during_validation"])
def test_successor_refuses_changed_original_inputs_or_current_source(successor_case, damage):
    c = successor_case
    if damage == "predecessor_bytes":
        (c.previous / "receipt.json").write_bytes(b"{}")
    elif damage == "config_bytes":
        c.config_path.write_bytes(c.config_path.read_bytes() + b" ")
    elif damage == "source_during_inference":
        c.hooks.after_inference = lambda: c.source.write_text("def answer(): return 3\n")
    else:
        receipt = c.prepare()
        c.hooks.after_validation = lambda: c.source.write_text("def answer(): return 3\n")
        with pytest.raises(ValueError):
            context.validate_source384_context(repository=c.repository, expected_receipt=receipt)
        return
    with pytest.raises(ValueError):
        c.prepare()


@pytest.mark.parametrize("damage", ["authority", "nomination", "lineage_digest", "extra_lineage", "model_pin",
    "worker_observation", "model_count_bool", "model_count_negative", "extra_receipt", "direct_cycle"])
def test_successor_receipt_cannot_gain_authority_or_change_identity(successor_case, damage):
    c = successor_case
    receipt = c.prepare()
    if damage == "authority":
        receipt["planning_authority"] = True
    elif damage == "nomination":
        receipt["source_applicability_nomination"] = c.predecessor["source_applicability_nomination"]
    elif damage == "lineage_digest":
        receipt["successor_lineage"]["predecessor_receipt_sha256"] = "0" * 64
    elif damage == "extra_lineage":
        receipt["successor_lineage"]["unknown"] = False
    elif damage == "model_pin":
        receipt["checkpoint_sha256"] = "0" * 64
    elif damage == "worker_observation":
        receipt["inference_execution"]["inference_executed"] = False
    elif damage == "model_count_bool":
        receipt["inference_execution"]["model_loads"] = True
    elif damage == "model_count_negative":
        receipt["inference_execution"]["model_loads"] = -1
    elif damage == "extra_receipt":
        receipt["unknown"] = False
    else:
        receipt["successor_lineage"]["predecessor_output"] = receipt["output"]
    (c.output / "receipt.json").write_bytes(context._raw(receipt))
    with pytest.raises(ValueError):
        context.validate_source384_context(repository=c.repository, expected_receipt=receipt)


def test_successor_excludes_declared_new_output_without_expanding_original_population(successor_case):
    c = successor_case
    (c.repository / "new_output.py").write_text("value = 3\n")
    receipt = c.prepare(excluded_new_output_paths=["new_output.py"])
    assert receipt["successor_lineage"]["excluded_new_output_paths"] == ["new_output.py"]
    assert receipt["source_hashes"].keys() == {"module.py"}
    assert [value for key, value in c.observations if key == "index"][0]["exclusions"] == [".runtime", "new_output.py"]
    with pytest.raises(ValueError, match="hide predecessor"):
        context._successor_exclusions(["module.py"], {"nested/module.py": "a" * 64})


def test_successor_preflight_time_consumes_original_budget(successor_case, monkeypatch):
    c = successor_case
    clock = [100.]
    monkeypatch.setattr(context, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    c.hooks.load_config = lambda: clock.__setitem__(0, 104.)
    seen = []
    monkeypatch.setattr(context, "_prepare_source384_context", lambda **kwargs: seen.append(kwargs))
    c.prepare(timeout_seconds=10.)
    assert seen[0]["timeout_seconds"] == 6.
    c.hooks.load_config = lambda: clock.__setitem__(0, 120.)
    with pytest.raises(ValueError, match="deadline expired"):
        c.prepare(timeout_seconds=10.)
    assert len(seen) == 1


@pytest.mark.parametrize("damage", ["authority", "extra_field", "direct_cycle"])
def test_historical_successor_predecessor_must_remain_closed_advice(successor_case, damage):
    c = successor_case
    receipt = c.prepare()
    if damage == "authority":
        receipt["planning_authority"] = True
    elif damage == "extra_field":
        receipt["source_applicability_nomination"] = {}
    else:
        receipt["successor_lineage"]["predecessor_output"] = receipt["output"]
    (c.output / "receipt.json").write_bytes(context._raw(receipt))
    with pytest.raises(ValueError):
        context.validate_historical_source384_selection(repository=c.repository, expected_receipt=receipt)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True, "100"])
def test_refresh_rejects_invalid_enclosing_deadline_before_owner(deadline, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import published_task_context as published
    monkeypatch.setattr(published, "_owner_state", lambda **kwargs: pytest.fail("owner invoked"))
    with pytest.raises(ValueError, match="finite"):
        published.refresh_published_task_context(server=None, admission={}, predecessor_bundle={},
            task_cid="fixture", output=None, deadline_monotonic=deadline)


def test_historical_prefix_consumes_whole_refresh_deadline(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import published_task_context as published
    clock = [100.]
    monkeypatch.setattr(published, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    def owner(**kwargs):
        clock[0] = 104.
        return tmp_path, {"task_id": "TASK"}, {}
    def historical(**kwargs):
        clock[0] = 111.
        return {"source384_context": {}}
    monkeypatch.setattr(published, "_owner_state", owner)
    monkeypatch.setattr(published, "read_task_context_historical_selection", historical)
    with pytest.raises(TimeoutError, match="refresh deadline"):
        published.refresh_published_task_context(server=None, admission={},
            predecessor_bundle={"artifact": "fixture", "sha256": "a" * 64}, task_cid="fixture",
            output=tmp_path / "output", deadline_monotonic=110.)
    assert not (tmp_path / "output").exists()
