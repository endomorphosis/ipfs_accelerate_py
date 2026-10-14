"""Real source/context/worker cycle with authored task and world declarations.

No live provider or native world authority is claimed. Native source indexing,
context compilation, runner dispatch, child MCP edits and task checks are real.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextReference, ContextTier
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import ArtifactBindings
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.world_model import (
    RepositoryWorldState, extract_world_state_delta,
)
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_policy import POLICIES
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    prepare_repository_world_packet, repository_world_task_binding,
)
from test.api.test_repository_context_compiler import compiler, compile_context, obligation
from test.api.test_repository_source_mcp import real_client
from test.api.test_repository_source_router import route


TASK = "Change square(value) to return value * value + 1; preserve its signature and all other files."
CHECK = "from module import square; assert square(3) == 10"
EFFECTS = ("repository_write", "validation")
WRITE_PATHS = ("module.py",)


def digest(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def authored_world(capsule, *, test_status_id=""):
    """Explicit fixture owner inputs, not a manufactured accepted world root."""
    return RepositoryWorldState(
        bindings=ArtifactBindings(repository_id=capsule.repository_id,
            repository_commit="fixture-commit", tree_id=capsule.tree_id,
            objective_id=capsule.objective_id, task_id="WORLD-CYCLE",
            contract_revision="fixture-contract", policy_revision=capsule.policy_revision,
            environment_id="fixture-environment"),
        world_snapshot_cid=digest(b"authored world declaration" + capsule.capsule_id.encode()),
        repository_reference="fixture-repository", repository_snapshot_id="fixture-analysis",
        analysis_head_tree_id="fixture-head", analysis_index_tree_id="fixture-index",
        active_task_ids=("WORLD-CYCLE",), test_status_id=test_status_id,
        source_evidence_ids=(capsule.capsule_id,),
        unavailable_dimensions=("dependency_graph", "effect_graph", "proof_status"))


def context(adapter, *, generation, extra=()):
    required = (obligation("signature", summary="Preserve square(value) and do not modify other files."), *extra)
    return compile_context(adapter, required, generation=generation, query="square",
        goal={"intent": TASK}, authority={"mode": "proposal", "allowed_paths": WRITE_PATHS,
            "world_task_packet": repository_world_task_binding(task_id="WORLD-CYCLE", task_text=TASK,
                allowed_effects=EFFECTS, write_paths=WRITE_PATHS)},
        scope={"paths": ("module.py",)}, acceptance={"criteria": ("square(3) == 10",)})


def packet(adapter, compiled, world, *, generation, residual):
    return prepare_repository_world_packet(adapter=adapter, compilation=compiled,
        generation=generation, task_id="WORLD-CYCLE", task_text=TASK,
        unresolved_obligations=(residual,), validation_commands=("python -B -c '" + CHECK + "'",),
        acceptance_ids=("square-result",), allowed_effects=EFFECTS,
        write_paths=WRITE_PATHS, current_world=world)


def test_native_world_packet_edit_observe_and_successor_dispatch(route, tmp_path, record_property):
    workspace, state = route
    before = (workspace / "module.py").read_bytes()
    with RepositorySourceSession(workspace, tmp_path / "owner-state") as session:
        adapter = session.context_adapter(compiler())
        initial = context(adapter, generation=1)
        first_world = authored_world(initial.capsule)
        first = packet(adapter, initial, first_world, generation=1, residual="Implement the requested arithmetic change.")

        def worker(binding):
            assert binding is not None
            with real_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
                # Source context and the worker child have distinct native roots;
                # query current child handles instead of reusing owner CIDs.
                found, failed = client.call("repository_context_query", generation=1, query="square", limit=1)
                assert not failed, found
                candidate = found["candidates"][0]
                read, failed = client.call("repository_read", generation=1, path="module.py",
                    symbol_id=candidate["symbol_id"])
                assert not failed, read
                edited, failed = client.call("repository_replace_symbol", generation=1, path="module.py",
                    symbol=read["symbol"], expected_sha256=read["source_sha256"],
                    representation=read["representation"],
                    replacement=read["content"].replace("value * value", "value * value + 1"))
                assert not failed and edited["canonical_changed"] is True

        state["callback"] = worker
        _, first_receipt = runner.run(prompt=TASK, provider="codex_cli", model="pinned", timeout=30,
            max_output_tokens=128, world_task_packet=first, repository_source_policy=POLICIES[2])
        assert first.prompt in state["seen"][0][0]
        assert first_receipt["world_task_packet"]["provider_attempted"] is True
        assert (workspace / "module.py").read_bytes() != before
        assert (workspace / "module.py").read_bytes().startswith(b"# retained comment\n")

        # Reusing either the old context or its world must not dispatch again.
        state["callback"] = None
        with pytest.raises(ValueError):
            runner.run(prompt=TASK, provider="codex_cli", model="pinned", timeout=30,
                max_output_tokens=128, world_task_packet=first)
        assert len(state["seen"]) == 1
        refreshed = session.refresh(generation=1)
        assert refreshed["generation"] == 2

        # The owner observes a real command, rather than trusting model prose.
        checked = subprocess.run([sys.executable, "-B", "-c", CHECK], cwd=workspace,
                                 capture_output=True, timeout=10)
        assert checked.returncode == 0, checked.stderr
        observation = json.dumps(dict(command=[sys.executable, "-B", "-c", CHECK],
            exit_code=checked.returncode, stdout_sha256=hashlib.sha256(checked.stdout).hexdigest(),
            stderr_sha256=hashlib.sha256(checked.stderr).hexdigest(),
            source_tree_sha256=refreshed["source_tree_sha256"],
            evidence_kind="authored_command_observation", proof_authority=False), sort_keys=True)
        observed = ContextReference(reference_id="observed-square-check", kind="command-observation",
            tier=ContextTier.INVARIANT, referenced_content_id=digest(observation.encode()),
            repository_id=initial.capsule.repository_id, tree_id="sha256:" + refreshed["source_tree_sha256"],
            summary=observation, metadata={"required": True, "coverage_ids": ("observed-check",)})
        successor = context(adapter, generation=2, extra=(observed,))
        second_world = authored_world(successor.capsule, test_status_id=observed.referenced_content_id)
        delta = extract_world_state_delta(first_world, second_world)
        assert delta.has_changes and delta.before_state_id != delta.after_state_id
        with pytest.raises(ValueError):
            packet(adapter, successor, first_world, generation=2, residual="Review the updated result.")
        second = packet(adapter, successor, second_world, generation=2,
                        residual="Review the observed check and report remaining limitations.")
        _, second_receipt = runner.run(prompt=TASK, provider="codex_cli", model="pinned", timeout=30,
            max_output_tokens=128, world_task_packet=second)
        assert len(state["seen"]) == 2
        assert second.prompt in state["seen"][1][0]
        assert first.prompt not in state["seen"][1][0]
        assert CHECK in second.prompt and "observed-square-check" in second.prompt
        for receipt in (first_receipt, second_receipt):
            assert receipt["world_task_packet"]["dispatch_rechecked"] is True
            assert receipt["world_task_packet"]["provider_retained_parent"] is False
            assert receipt["completion_authority"] is False
        record_property("world_packet_cycle", json.dumps(dict(
            source_generations=2, provider_calls_authored=2, live_provider_calls=0,
            stale_dispatches_refused=1, command_exit_code=checked.returncode,
            first_body_bytes=len(first.prompt.encode()), second_body_bytes=len(second.prompt.encode()),
            full_session_token_savings=None, world_owner_inputs="authored declarations",
            native_source_index=True, real_child_mcp=True), sort_keys=True))


def test_unqualified_world_mode_cannot_be_silently_audited_as_terminal_legacy(tmp_path):
    from benchmarks.agent_supervisor.container_coding.terminal_context_audit import (
        _receipt, _model_projection, collect_terminal_context_audit,
    )
    for marker in ({}, None, {"provider_attempted": True}):
        value = {"schema": "router-implementation-invocation@1", "world_task_packet": marker}
        with pytest.raises(ValueError, match="qualified Terminal Bench replay"):
            _receipt(value, tmp_path)
        with pytest.raises(ValueError, match="qualified Terminal Bench replay"):
            _model_projection(rendered="legacy", receipt=value, repository=None)
        # The public collector must refuse before dropping unknown fields for
        # transport to its replay child, including a null marker.
        result = collect_terminal_context_audit(state=tmp_path / "missing-state",
            receipts=[dict(value, phase="coding")], workspace_root=tmp_path)
        assert result["reason"] == "world_task_packet_replay_unqualified"
