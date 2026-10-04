"""Native daemon retry reconstruction retains required producer capsules."""

import json
import subprocess

import pytest

pytest.importorskip("ipfs_datasets_py.logic.software_contracts.semantic_state")

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompileResult,
    RetryContextResult,
    reconstruct_context,
    render_context_capsule,
    render_retry_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    prepare_semantic_context,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)


@pytest.fixture
def semantic_daemon(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("from dependency import add\ndef target(x): return add(x, 1)\n")
    (repo / "dependency.py").write_text("def add(a,b): return a+b\n")
    (repo / "tasks.todo.md").write_text("# Semantic context retries\n")
    for args in [
        ("init", "-q"),
        ("add", "."),
        (
            "-c",
            "user.name=Integration",
            "-c",
            "user.email=local@example.invalid",
            "commit",
            "-qm",
            "seed",
        ),
    ]:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    result = prepare_semantic_context(
        repository=repo,
        paths=["target.py", "dependency.py"],
        required_raw_paths=["target.py"],
        objective="Preserve addition",
        task_id="RETRY-001",
        output=repo / ".runtime/semantic/initial",
    )
    task = PortalTask(
        task_id="RETRY-001",
        title="Preserve addition",
        status="ready",
        completion="manual",
        priority="P1",
        track="semantic-context",
        outputs=["target.py"],
        validation=["python3 -m pytest test_target.py"],
        acceptance="Addition remains correct",
        metadata={
            "Semantic context artifact": ".runtime/semantic/initial/worker-context.json",
            "Semantic context sha256": result["worker_payload_sha256"],
            "Semantic context refresh": "true",
        },
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "tasks.todo.md",
        state_path=tmp_path / "state/tasks.json",
        strategy_path=tmp_path / "state/strategy.json",
        events_path=tmp_path / "state/events.jsonl",
        implementation_log_dir=tmp_path / "state/logs",
        repo_root=repo,
        task_header_prefix="## RETRY-",
    )
    return repo, task, daemon, result


def _diagnose(daemon, task):
    return daemon.record_implementation_failure_context(
        task,
        {
            "kind": "validation_failure",
            "returncode": 1,
            "validation_result": {
                "attempted": True,
                "passed": False,
                "returncode": 1,
                "failed_command": "python3 -m pytest test_target.py",
                "failure_head": "AssertionError: addition differs",
            },
        },
        changed_files=("target.py",),
        unresolved_requirements=("requirement:addition",),
    )


def test_daemon_preserves_semantic_capsules_through_verified_retry_delta(
    semantic_daemon, record_property
):
    _, task, daemon, prepared = semantic_daemon
    first_prompt = daemon._build_implementation_prompt(task, attempt=1)
    base = daemon._last_implementation_context
    semantic = {
        ref.reference_id: ref for ref in base.capsule.evidence if ref.kind == "semantic-context"
    }
    assert semantic and all(ref.required for ref in semantic.values())
    _diagnose(daemon, task)
    second_prompt = daemon._build_implementation_prompt(task, attempt=2)
    retry = daemon._last_implementation_retry
    assert isinstance(retry, RetryContextResult)
    delta = retry.delta_result
    assert delta.verifier.verify_delta_result(delta) is delta
    assert reconstruct_context(base.capsule, delta.delta_capsule) == retry.reconstructed_capsule
    reconstructed = {ref.reference_id: ref for ref in retry.reconstructed_capsule.evidence}
    assert all(reconstructed[key] == ref for key, ref in semantic.items())
    assert not set(semantic) & {ref.reference_id for ref in delta.delta_capsule.evidence}
    assert second_prompt == render_context_capsule(retry.reconstructed_capsule)
    assert prepared["semantic_root_cid"] in first_prompt
    assert prepared["semantic_root_cid"] in second_prompt
    assert "Addition remains correct" in second_prompt
    # Measure the actual emitted full prompt separately from the local delta.
    delta_text = render_retry_context(retry.capsule)
    measurements = {
        "full_rendered_bytes": len(second_prompt.encode()),
        "local_delta_rendered_bytes": len(delta_text.encode()),
        "full_estimated_tokens": delta.verifier.estimator.estimate(second_prompt),
        "local_delta_estimated_tokens": delta.verifier.estimator.estimate(delta_text),
        "token_estimator": delta.verifier.estimator.name,
        "provider_usage_reported": False,
        "provider_retained_parent": False,
    }
    assert measurements["local_delta_rendered_bytes"] < measurements["full_rendered_bytes"]
    assert measurements["local_delta_estimated_tokens"] < measurements["full_estimated_tokens"]
    for key, value in measurements.items():
        record_property(key, value)
    print("SEMANTIC_RETRY_MEASUREMENTS=" + json.dumps(measurements, sort_keys=True))


def test_dirty_source_retry_refreshes_capsules_and_preserves_failure(semantic_daemon):
    repo, task, daemon, prepared = semantic_daemon
    daemon._build_implementation_prompt(task, attempt=1)
    diagnostic = _diagnose(daemon, task)
    (repo / "dependency.py").write_text("def add(a,b): return a-b\n")
    prompt = daemon._build_implementation_prompt(task, attempt=2)
    assert isinstance(daemon._last_implementation_context, ContextCompileResult)
    assert daemon._last_implementation_retry is None
    fresh_paths = tuple((repo / ".runtime/semantic-refresh").glob("*/worker-context.json"))
    assert fresh_paths
    fresh = json.loads(fresh_paths[-1].read_bytes())
    assert fresh["semantic_root_cid"] != prepared["semantic_root_cid"]
    assert fresh["semantic_root_cid"] in prompt
    assert diagnostic.failure_id in prompt
    assert "validation_failure" in prompt
    rebound = daemon._implementation_diagnostics[daemon._canonical_ref(task)]
    assert rebound.failure_id == diagnostic.failure_id
    assert rebound.failure == diagnostic.failure
    assert rebound.prior_decision_id == daemon._last_implementation_context.receipt.receipt_id


def test_dirty_source_retry_requires_explicit_refresh(semantic_daemon):
    repo, task, daemon, _ = semantic_daemon
    task.metadata["Semantic context refresh"] = "false"
    daemon._build_implementation_prompt(task, attempt=1)
    _diagnose(daemon, task)
    (repo / "dependency.py").write_text("def add(a,b): return a-b\n")
    with pytest.raises(ValueError, match="stale"):
        daemon._build_implementation_prompt(task, attempt=2)
    assert not (repo / ".runtime/semantic-refresh").exists()
