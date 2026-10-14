"""Actual indexed preparation and router input sizes with an authored provider."""
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding.indexed_doctor_lifecycle import _context
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_symbolic_efficiency import _prepare_case, _git
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import load_public_instruction
from test.api.test_semantic_router_integration import provider  # noqa: F401


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("function_count", [4, 8])
def test_capsule_budget_reduces_real_indexed_router_input_without_dropping_required_core(
        tmp_path, monkeypatch, provider, record_property, function_count):
    observations = []
    baseline_raw = None
    for ordinal, (budget, limit) in enumerate((("legacy", 8), ("one-capsule@1", 1), ("two-capsules@1", 2))):
        # Equal-length roots avoid a path-length advantage. Native identities
        # are independently generated; this is not replay of a live prompt.
        case = _prepare_case(tmp_path / f"budget{ordinal}", function_count,
            "declared-contract@1", monkeypatch, capsule_budget=budget)
        root, workspace, prompt = case["root"], case["workspace"], case["prompt"]
        semantic = _context(prompt, "semantic-context")
        projection = semantic["worker_projection"]
        assert projection["max_capsules"] == limit
        assert projection["full_capsule_count"] == function_count + 1  # Includes the module.
        assert len(semantic["capsules"]) == projection["selected_capsule_count"] <= limit
        assert projection["omitted_capsule_count"] == projection["full_capsule_count"] - len(semantic["capsules"])
        assert set(projection["raw_source_fetch_required"]) == {"labels.py"}
        assert projection["semantic_equivalence_claimed"] is projection["completion_authority"] is False
        assert set(semantic["raw_sources"]) == {prep.INSTRUCTION, prep.SMOKE, prep.PROFILE}
        if baseline_raw is None:
            baseline_raw = semantic["raw_sources"]
        assert semantic["raw_sources"] == baseline_raw
        assert case["planned"]["provider_calls"] == 0
        assert case["indexed"]["initial_indexes_reused"] is True
        assert case["indexed"]["new_embedding_calls"] == 0
        original, _ = json.JSONDecoder().raw_decode(prompt)
        encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
        transport = json.loads(encoded.provider_prompt)
        assert transport["native_context"] == {
            **original, "evidence": [row for row in original["evidence"] if row["kind"] != "semantic-context"]}
        assert codec.restore_semantic_router_prompt(provider_prompt=encoded.provider_prompt,
            table=encoded.table, repository=root) == prompt
        instruction = case["instruction_context"]
        block, _ = load_public_instruction(artifact=Path(instruction["artifact"]),
            expected_sha256=instruction["sha256"], task_cid=case["task"]["task_cid"],
            prompt=prompt, workspace=workspace, repository=root)
        with monkeypatch.context() as worker:
            worker.chdir(workspace)
            worker.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
            before = len(provider[1])
            output, receipt = runner.run(prompt=prompt, provider="codex_cli", model="pinned",
                timeout=1, max_output_tokens=128, semantic_repository=root,
                public_instruction_artifact=Path(instruction["artifact"]),
                public_instruction_sha256=instruction["sha256"],
                public_instruction_task_cid=case["task"]["task_cid"])
            assert output == "literal summary" and len(provider[1]) == before + 1
            actual = provider[1][-1][0]
            expected, _ = runner.render_model_prompt(prompt=encoded.provider_prompt + block,
                purpose="coding", workspace=workspace, semantic_transport=True)
            assert actual == expected and actual.count(block) == 1
            assert receipt["model_prompt_bytes"] == len(actual.encode())
            assert receipt["model_prompt_sha256"] == _sha(actual)
        assert (root / "labels.py").read_text() == (workspace / "labels.py").read_text() == case["source"]
        observations.append(dict(capsule_budget=budget, maximum_capsules=limit,
            full_capsules=projection["full_capsule_count"], selected_capsules=len(semantic["capsules"]),
            omitted_capsules=projection["omitted_capsule_count"],
            semantic_bytes=len(json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()),
            complete_prompt_bytes=receipt["model_prompt_bytes"], complete_prompt_sha256=receipt["model_prompt_sha256"],
            required_raw_bytes=sum(len(text.encode()) for text in semantic["raw_sources"].values()),
            required_core_unchanged_by_transport=True, initial_indexes_reused=True,
            new_embedding_calls=0, authored_coding_dispatches=1, planning_provider_calls=0))
        _git(root, "worktree", "remove", str(workspace))
    legacy, one, two = observations
    assert one["selected_capsules"] == 1 and two["selected_capsules"] == 2
    assert one["complete_prompt_bytes"] < two["complete_prompt_bytes"] < legacy["complete_prompt_bytes"]
    record_property("capsule_budget_census", json.dumps(dict(
        schema="terminal-capsule-budget-census@1", functions=function_count, observations=observations,
        semantic_transport_schema=codec.TRANSPORT_SCHEMA, worker_max_bytes=32768,
        required_raw_equal_across_profiles=True, external_model_calls=0,
        provider_usage_measurements=False, benchmark_reward_measured=False,
        end_to_end_token_savings_claimed=False, source384_inference_included=False,
        caveat="Authored inputs and provider; native indexed pipeline and dispatch. Independent identities per profile."), sort_keys=True))
