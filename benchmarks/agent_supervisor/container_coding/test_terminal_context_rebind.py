"""Actual native-owner context recapture preserves historical world semantics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.terminal_native_preflight import _authored_graph
from benchmarks.agent_supervisor.container_coding.terminal_context_rebind import rebind_full_context, verify_worker_context_prompt
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original as original
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon


@pytest.mark.parametrize("source384", ["absent", "selected", "drift-during-capture",
    "numerical", "combined", "numerical-drift-during-capture"])
def test_rebind_actual_owner_and_native_prompt_checks_fresh_sources_without_world_authority(original, monkeypatch, source384):
    root, instruction, state = original
    numeric = source384 in {"numerical", "combined", "numerical-drift-during-capture"}
    learned = source384 in {"selected", "combined", "drift-during-capture"}
    if numeric:
        from benchmarks.agent_supervisor.container_coding.test_terminal_numerical_context import SOURCE
        (root / "bottle.py").write_text(SOURCE)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    graph = _authored_graph(prepared)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    prep._write(state / "admission.json", admission)
    with IntentRepository(state / "intent.duckdb") as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    prep.context(state=state)
    source384_calls = []
    source384_stale = []
    selected_receipts = {}
    if learned:
        from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
        from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as native_source384
        # Only the transport/recapture is under test; native model validation is
        # separately qualified. The receipt is explicitly authored here.
        module = types.ModuleType("ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context")
        def validate(**kwargs):
            source384_calls.append(kwargs)
            if source384_stale:
                raise ValueError("selected Source384 changed during capture")
        module.validate_source384_context = validate
        module._sources = native_source384._sources
        monkeypatch.setitem(sys.modules, module.__name__, module)
        receipt_output = state / "authored-source384"
        receipt_output.mkdir()
        receipt = {"schema": "terminal-source384-repository-context@1",
            "repository": str(root), "output": str(receipt_output), "completion_authority": False,
            "source_hashes": {"bottle.py": hashlib.sha256((root / "bottle.py").read_bytes()).hexdigest()},
            "summary": {"schema": "authored-source384-transport-summary@1", "candidate_samples": [],
                "proof_authority": False, "source_semantics_verified": False}}
        (receipt_output / "receipt.json").write_bytes(
            json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        selected_receipts["source384_context"] = receipt
    if numeric:
        from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical
        selected_receipts["numerical_formalization"] = numerical.prepare_numerical_context(
            repository=root, source_hashes={"bottle.py": hashlib.sha256((root / "bottle.py").read_bytes()).hexdigest()},
            output=state / "numerical-formalization", profile=numerical.PROFILE)
    if selected_receipts:
        from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
        context_result = json.loads((state / "context-result.json").read_bytes())
        previous = json.loads((root / context_result["context_bundle"]["artifact"]).read_bytes())["tasks"][0]
        context_result["context_bundle"] = write_task_context_bundle(repository=root,
            prepared=[{"schema": "supervisor-task-context-preparation@1", **previous,
                       **selected_receipts}], output=root / ".runtime/selected-formalization-bundle.json")
        # Rebinding must preserve the selected bundle receipt, never this
        # unbound optional field from the preparation result.
        for key in selected_receipts:
            context_result[key] = {"unbound": "must not be consumed"}
        prep._write(state / "context-result.json", context_result)
        if source384 in {"drift-during-capture", "numerical-drift-during-capture"}:
            from benchmarks.agent_supervisor.container_coding import terminal_context_rebind as rebind
            persist = rebind.persist_intent_world_snapshot
            def change_after_capture(*args, **kwargs):
                result = persist(*args, **kwargs)
                if numeric:
                    (root / "bottle.py").write_text(SOURCE + "\n# changed during owner capture\n")
                else:
                    source384_stale.append(True)
                return result
            monkeypatch.setattr(rebind, "persist_intent_world_snapshot", change_after_capture)
    clone = state / "clone.duckdb"
    with IntentRepository(clone) as intent:
        result = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    task_spec = graph.tasks[0]
    with open_existing_native_owner(database=clone, checkout=root, state_dir=state / "owner",
            repository_id=prepared["manifest"]["payload"]["repository_cid"],
            execution_routes={task_spec.task_key: GROK_CODEX_EXECUTION_MODE}) as owner:
        before = owner.source.get_task(task_spec.task_cid)
        if source384 in {"drift-during-capture", "numerical-drift-during-capture"}:
            with pytest.raises(ValueError):
                rebind_full_context(prepared_state=state, admission=admission,
                    server=owner.server, output=root / ".runtime/rebound")
            assert len(source384_calls) == (2 if learned else 0)
            assert not (root / ".runtime/rebound/context-bundle.json").exists()
            assert owner.source.get_task(task_spec.task_cid) == before
            return
        rebound = rebind_full_context(prepared_state=state, admission=admission,
            server=owner.server, output=root / ".runtime/rebound")
        if selected_receipts:
            assert all(rebound[key] == receipt for key, receipt in selected_receipts.items())
            assert len(source384_calls) == (2 if learned else 0)
            rebound_bundle = json.loads((root / rebound["context_bundle"]["artifact"]).read_bytes())
            assert rebound_bundle["schema"] == ("supervisor-task-context-nominations@4" if numeric
                                                 else "supervisor-task-context-nominations@3")
            from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import read_task_context_historical_selection
            selected = read_task_context_historical_selection(repository=root,
                artifact=rebound["context_bundle"]["artifact"], expected_sha256=rebound["context_bundle"]["sha256"],
                task_cid=task_spec.task_cid, task_id=task_spec.task_key)
            assert all(selected[key] == receipt for key, receipt in selected_receipts.items())
        assert owner.source.get_task(task_spec.task_cid) == before
        assert rebound["new_embedding_calls"] == rebound["text_generation_calls"] == 0
        assert rebound["intent_fresh_at_capture"] is True
        assert rebound["intent_fresh_after_claim"] is False
        portal = PortalTask(task_id=task_spec.task_key, canonical_task_cid=task_spec.task_cid,
            title=before.body["title"], status="ready", completion="manual", priority="P2",
            track="implementation", outputs=["bottle.py", "report.jsonl"],
            validation=[], acceptance="Public shape and syntax", metadata={"database task cid": task_spec.task_cid})
        daemon = TodoImplementationDaemon(todo_path=root / ".runtime/test.todo.md",
            state_path=state / "prompt/tasks.json", strategy_path=state / "prompt/strategy.json",
            events_path=state / "prompt/events.jsonl", repo_root=root, task_header_prefix="## TB-")
        daemon._task_context_nomination_bundle = rebound["context_bundle"]
        try:
            # Same production behavior: sealed world evidence without claiming
            # to read a live owner from this worker-facing Portal instance.
            prompt = daemon._build_implementation_prompt(portal, attempt=1)
            observed = verify_worker_context_prompt(prompt=prompt, rebound=rebound)
            assert observed["intent_freshness_checked_by_worker"] is False
            assert observed["completion_authority"] is False
            assert len(observed["required_context_kinds"]) == (4 if selected_receipts else 3)
            if selected_receipts:
                wire, end = json.JSONDecoder().raw_decode(prompt)
                refs = sorted((row for row in wire["evidence"] if row["kind"] == "source-formalization-context"),
                              key=lambda row: row["reference_id"])
                formal = json.loads("".join(row["summary"] for row in refs))
                assert formal["task_cid"] == task_spec.task_cid
                assert formal["context_bundle"] == rebound["context_bundle"]
                assert formal["source_obligations_discharged"] == 0
                if numeric:
                    summary = next(row["summary"] for row in formal["contexts"]
                        if row["kind"] == "numerical-formalization-context")
                    assert summary == selected_receipts["numerical_formalization"]["summary"]
                    assert summary["source_unit_links"][0]["unit_id"] in prompt
                    assert "selected_eigenpair" in prompt
                for mutation in ("omitted", "task", "receipt", "authority", "unit_link", "duplicate_chunk"):
                    changed = deepcopy(wire)
                    selected_rows = [row for row in changed["evidence"] if row["kind"] == "source-formalization-context"]
                    if mutation == "omitted":
                        changed["evidence"] = [row for row in changed["evidence"] if row["kind"] != "source-formalization-context"]
                    elif mutation == "duplicate_chunk":
                        changed["evidence"].append(deepcopy(selected_rows[0]))
                    else:
                        altered = deepcopy(formal)
                        if mutation == "task": altered["task_cid"] = "foreign-task"
                        elif mutation == "receipt": altered["contexts"][0]["receipt_reference"]["receipt_sha256"] = "0" * 64
                        elif mutation == "authority": altered["proof_authority"] = True
                        else: altered["contexts"][0]["summary"]["injected_link"] = "foreign-unit"
                        # Deliberately rehash every chunk to show that intact
                        # transport digests do not authorize changed advice.
                        text = json.dumps(altered, sort_keys=True, separators=(",", ":"))
                        from ipfs_accelerate_py.agent_supervisor.context.context_compiler import _canonical_digest
                        for index, row in enumerate(sorted(selected_rows, key=lambda row: row["reference_id"])):
                            row["summary"] = text if index == 0 else ""
                            row["byte_count"] = len(row["summary"].encode())
                            row["referenced_content_id"] = "sha256:" + hashlib.sha256(row["summary"].encode()).hexdigest()
                            row["metadata"]["artifact_content_id"] = _canonical_digest({"text": text})
                    with pytest.raises(ValueError, match="formalization"):
                        verify_worker_context_prompt(prompt=json.dumps(changed) + prompt[end:], rebound=rebound)
            assert observed["source_sha256"] == {"bottle.py": prepared["manifest"]["payload"]["sources"]["bottle.py"]["sha256"]}
            foreign_query = {**rebound, "public_query_sha256": "0" * 64}
            with pytest.raises(ValueError, match="identity/status"):
                verify_worker_context_prompt(prompt=prompt, rebound=foreign_query)
            # Genuine task-state advancement makes the earlier world capture
            # historical; it cannot turn into fresh owner state by reuse.
            owner.source.compare_and_set_status(before.task_cid, before.revision, "retrying")
            with pytest.raises(ValueError, match="unclaimed native task"):
                rebind_full_context(prepared_state=state, admission=admission,
                    server=owner.server, output=root / ".runtime/rebound-rejected")
            foreign = {**rebound, "world_snapshot_cid": "foreign"}
            with pytest.raises(ValueError, match="identity/status"):
                verify_worker_context_prompt(prompt=prompt, rebound=foreign)
            # Byte changes invalidate reused semantic nominations before any
            # new context bundle is published.
            (root / "bottle.py").write_text("def application():\n    return 'changed'\n")
            with pytest.raises(ValueError):
                rebind_full_context(prepared_state=state, admission=admission,
                    server=owner.server, output=root / ".runtime/rebound-stale")
            assert not (root / ".runtime/rebound-stale/context-bundle.json").exists()
        finally:
            daemon.close_event_runtime()
