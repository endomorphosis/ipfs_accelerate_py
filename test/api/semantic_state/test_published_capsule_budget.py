"""Actual publication refresh retains the selected bounded lexical policy."""
import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_local_completion_bridge import typed_claim, native_published_transition, complete
from test.api.semantic_state import test_published_task_context as fixtures
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import published_task_context as published
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations


@pytest.mark.parametrize("limit,drift", [(1, False), (2, False), (1, True)])
def test_publication_preserves_or_refuses_changed_lexical_capsule_budget(scenario, tmp_path, monkeypatch, limit, drift):
    native = fixtures.prepare_semantic_context
    monkeypatch.setattr(fixtures, "prepare_semantic_context",
        lambda **kw: native(**{**kw, "worker_query": "answer", "worker_capsule_limit": limit}))
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        bundle = fixtures.initial_bundle(scenario, owner)
        before = (scenario["repository"] / bundle["artifact"]).read_bytes()
        task = owner.source.get_task(attempt.task_cid)
        transition = native_published_transition(scenario, tmp_path, owner, attempt)
        passed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
            attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
        assert passed["passed"] is True
        complete(owner, attempt, passed["results"][0]["evidence_digest"])
        canonical = owner.source.get_task(task.task_cid)
        args = dict(server=owner.server, admission=admission, predecessor_bundle=bundle,
            task_cid=attempt.task_cid, output=scenario["repository"] / ".runtime/bounded-successor",
            retrieval_rebuilder=fixtures.vectors)
        if drift:
            # Produce internally coherent bytes with an unauthorized wider
            # projection; predecessor continuity must reject the new cap.
            monkeypatch.setattr(published, "prepare_semantic_context",
                lambda **kw: native(**{**kw, "worker_capsule_limit": 8}))
            with pytest.raises(ValueError, match="changed capsule selection policy"):
                published.refresh_published_task_context(**args)
        else:
            result = published.refresh_published_task_context(**args)
            loaded = published.load_published_task_context(server=owner.server, admission=admission,
                artifact=result["refresh_artifact"], expected_sha256=result["refresh_sha256"])
            assert loaded["semantic"]["worker_projection"]["max_capsules"] == limit
            assert loaded["semantic"]["worker_projection"]["selector"] == "lexical-query-symbol-name@1"
            assert loaded["semantic"]["worker_capsules"] <= limit
            assert loaded["semantic"]["capsules"] > 0
            assert loaded["completion_authority"] is False
        assert owner.source.get_task(task.task_cid) == canonical
        assert (scenario["repository"] / bundle["artifact"]).read_bytes() == before
