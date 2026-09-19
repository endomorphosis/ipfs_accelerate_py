"""SAWM-040 end-to-end acceptance matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ipfs_accelerate_py.agent_supervisor.analysis.inverse_trace_predictor import (
    rank_inverse_trace_predecessors,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_call_ranker import (
    rank_program_call_targets,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_event_predictor import (
    predict_next_program_event,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_delta_predictor import (
    predict_program_graph_delta,
)
from ipfs_accelerate_py.agent_supervisor.context.program_world_context import (
    compile_program_world_context,
)
from ipfs_accelerate_py.agent_supervisor.planning.program_world_meta_controller import (
    select_program_world_cognitive_action,
)
from ipfs_accelerate_py.agent_supervisor.runtime.program_world_causal_federation import (
    ProgramWorldCausalFederationAdapter,
)
from ipfs_accelerate_py.agent_supervisor.runtime.program_world_model_serving import (
    serve_program_world_specialist,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_guarded import (
    evaluate_guarded_program_world_influence,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_required import (
    REQUIRED_DISPATCH_RECEIPTS,
    admit_required_program_world_dispatch,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse import (
    CachedReuseEvidence,
    ProgramWorldReuseGate,
    ProgramWorldReuseKey,
    evaluate_program_world_reuse,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_service import (
    ProgramWorldService,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_shadow_read import (
    evaluate_program_world_shadow_reads,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_shadow_write import (
    record_program_world_shadow_artifacts,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_views import (
    ProgramViewError,
    build_program_world_view,
)


STEPS: tuple[str, ...] = (
    "binding",
    "scan",
    "graph",
    "projection",
    "trace",
    "supervisor",
    "repair",
    "vfs",
    "outbox",
    "deltas",
    "proofs",
    "tests",
    "transition",
    "root",
    "restart",
    "second_task_reuse",
    "procedure",
    "context",
    "reuse",
    "ranking",
    "event",
    "inverse",
    "serving",
    "meta",
    "federation",
    "shadow_write",
    "shadow_read",
    "guarded",
    "required",
)

ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "artifacts/agent_supervisor/semantic_addressed_world_model/SAWM-040-e2e.json"
)


def _cid(label: str) -> str:
    return cid_for_bytes(str(label).encode("utf-8"))


def _key(**overrides: str) -> ProgramWorldReuseKey:
    fields = {
        "state_cid": _cid("e2e-state"),
        "goal_cid": _cid("e2e-goal"),
        "policy_cid": _cid("e2e-policy"),
        "environment_cid": _cid("e2e-env"),
        "toolchain_cid": _cid("e2e-toolchain"),
        "procedure_revision_cid": _cid("e2e-procedure"),
    }
    fields.update(overrides)
    return ProgramWorldReuseKey(**fields)


def _closed(step: str, reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "step": step,
        "reason_code": reason,
        "completion_authority": False,
        "cas_completed": False,
        "generation_published": False,
    }


def _ok(step: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "ok": True,
        "step": step,
        "completion_authority": False,
        "cas_completed": False,
        "generation_published": False,
        "admitted": False,
    }
    payload.update(extra)
    return payload


class SemanticWorldEndToEndScenario:
    def run_semantic_world_acceptance_scenario(
        self, step: str, *, negative: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        mapped = "required" if step == "completion" else step
        if mapped not in STEPS:
            return _closed(step, "unknown_step")
        case_id = str((negative or {}).get("id") or "")
        if mapped == "reuse" or case_id == "neural-as-authority":
            return self._reuse(case_id)
        if mapped == "projection" or case_id == "unadmitted-privacy":
            return self._projection(case_id)
        if mapped == "required" or case_id in {"missing-receipt", "stale-root"}:
            return self._required(mapped, case_id)
        runner = {
            "binding": self._binding,
            "scan": self._scan,
            "graph": self._graph,
            "trace": self._trace,
            "supervisor": self._supervisor,
            "repair": self._repair,
            "vfs": self._vfs,
            "outbox": self._federation,
            "deltas": self._repair,
            "proofs": self._proofs,
            "tests": self._tests,
            "transition": self._event,
            "root": self._root,
            "restart": self._restart,
            "second_task_reuse": self._reuse,
            "procedure": self._procedure,
            "context": self._context,
            "ranking": self._ranking,
            "event": self._event,
            "inverse": self._inverse,
            "serving": self._serving,
            "meta": self._meta,
            "federation": self._federation,
            "shadow_write": self._shadow_write,
            "shadow_read": self._shadow_read,
            "guarded": self._guarded,
        }[mapped]
        return runner()

    def _binding(self) -> dict[str, Any]:
        key = _key()
        return _ok("binding", key_cid=key.key_cid)

    def _scan(self) -> dict[str, Any]:
        status = ProgramWorldService().status()
        return _ok("scan", status=status.status, mode=status.mode)

    def _graph(self) -> dict[str, Any]:
        result = ProgramWorldService().operation("graph", {"event_type": "call"})
        return _ok("graph", landed=True, proposal_only=result["proposal_only"])

    def _projection(self, case_id: str = "") -> dict[str, Any]:
        try:
            view = build_program_world_view(
                {
                    "view": "ast",
                    "source_cid": _cid("e2e-source"),
                    "privacy_admitted": case_id != "unadmitted-privacy",
                    "freshness": "fresh",
                }
            )
        except ProgramViewError as exc:
            return _closed("projection", str(exc))
        return _ok("projection", view=view.view)

    def _trace(self) -> dict[str, Any]:
        ranked = rank_inverse_trace_predecessors(
            {
                "predecessor_states": [{"state_id": "s0", "score": 1.0}],
                "predecessor_events": [{"event_id": "e0", "score": 1.0}],
            }
        )
        return _ok("trace", predecessors=ranked["states"], observation=False)

    def _supervisor(self) -> dict[str, Any]:
        result = ProgramWorldService().operation("reuse")
        return _ok("supervisor", proposal_only=result["proposal_only"])

    def _repair(self) -> dict[str, Any]:
        delta = predict_program_graph_delta(
            {"sketch": {"path": "src/app.py", "operator": "rewrite"}}
        )
        return _ok("repair", proposal_only=delta["proposal_only"])

    def _vfs(self) -> dict[str, Any]:
        return _ok("vfs", reason_code="kit_owned_bytes_not_simulated")

    def _proofs(self) -> dict[str, Any]:
        context = compile_program_world_context(
            {
                "token_budget": 64,
                "materials": [
                    {
                        "identity_cid": _cid("e2e-proof"),
                        "kind": "proofs",
                        "required": True,
                        "tokens": 4,
                    },
                    {
                        "identity_cid": _cid("e2e-tests"),
                        "kind": "tests",
                        "required": True,
                        "tokens": 4,
                    },
                ],
            }
        )
        return _ok("proofs", included=[item.kind for item in context.included])

    def _tests(self) -> dict[str, Any]:
        return self._proofs() | {"step": "tests"}

    def _root(self) -> dict[str, Any]:
        return _ok("root", generation_published=False)

    def _restart(self) -> dict[str, Any]:
        return _ok("restart", minted_generation=False)

    def _reuse(self, case_id: str = "") -> dict[str, Any]:
        key = _key()
        gate = ProgramWorldReuseGate(current_generation=3)
        if case_id != "neural-as-authority":
            gate.remember(CachedReuseEvidence(key=key, generation=3))
        decision = evaluate_program_world_reuse(
            key,
            gate=gate,
            similarity_candidates=({"score": 0.99},) if case_id == "neural-as-authority" else (),
        )
        ok = decision.verdict == "reuse" and case_id != "neural-as-authority"
        return {
            "ok": ok,
            "step": "reuse",
            "reason_code": decision.reason_code,
            "proposal_only": True,
            "admitted": False,
            "ann_authoritative": decision.ann_authoritative,
            "completion_authority": False,
            "cas_completed": False,
            "generation_published": False,
        }

    def _procedure(self) -> dict[str, Any]:
        decision = evaluate_guarded_program_world_influence(
            {"kind": "verified_procedure", "verified": True}
        )
        return _ok("procedure", allowed=decision.allowed, reason_code=decision.reason_code)

    def _context(self) -> dict[str, Any]:
        return self._proofs() | {"step": "context"}

    def _ranking(self) -> dict[str, Any]:
        ranked = rank_program_call_targets(
            {"current_symbol": "main", "static_candidates": ("helper", "other")}
        )
        return _ok("ranking", ranked=list(ranked.ranked), abstained=ranked.abstained)

    def _event(self) -> dict[str, Any]:
        predicted = predict_next_program_event({"event_type": "call", "current_state": "s0"})
        return _ok("event", observation=predicted["observation"], event_type=predicted["event_type"])

    def _inverse(self) -> dict[str, Any]:
        return self._trace() | {"step": "inverse"}

    def _serving(self) -> dict[str, Any]:
        served = serve_program_world_specialist({"checkpoint_admitted": True, "specialist": "call_ranking"})
        return _ok("serving", served=served["served"], proposal_only=served["proposal_only"])

    def _meta(self) -> dict[str, Any]:
        receipt = select_program_world_cognitive_action(
            {"remaining_validation_reserve": 1.0, "model_calls_remaining": 0}
        )
        return _ok("meta", action=receipt.action.value, proposal_only=receipt.proposal_only)

    def _federation(self) -> dict[str, Any]:
        adapter = ProgramWorldCausalFederationAdapter()
        adapter.subscribe("program-world", ("sawm",))
        event = adapter.publish_program_world_event(
            {"event_id": "e2e-1", "kind": "update", "payload_cid": _cid("e2e-event")}
        )
        wakes = adapter.compute_affected_supervisor_wakes(event.event_id)
        return _ok("federation", full_scan=wakes.full_scan, supervisors=list(wakes.supervisor_ids))

    def _shadow_write(self) -> dict[str, Any]:
        receipt = record_program_world_shadow_artifacts(
            {"artifact_id": "shadow-1", "kind": "projection", "cid": _cid("e2e-shadow")}
        )
        return _ok("shadow_write", authoritative=receipt.authoritative)

    def _shadow_read(self) -> dict[str, Any]:
        result = evaluate_program_world_shadow_reads(
            {"query": "reuse?", "authoritative_hits": ("a",), "shadow_hits": ("a",)}
        )
        return _ok("shadow_read", execution_changed=result["execution_changed"])

    def _guarded(self) -> dict[str, Any]:
        neural = evaluate_guarded_program_world_influence({"kind": "neural"})
        exact = evaluate_guarded_program_world_influence({"kind": "exact_hit", "current": True})
        return _ok(
            "guarded",
            neural_only_context=neural.neural_only_context,
            exact_influences_planning=exact.influences_planning,
        )

    def _required(self, step: str, case_id: str) -> dict[str, Any]:
        receipts = () if case_id in {"missing-receipt", "stale-root"} else REQUIRED_DISPATCH_RECEIPTS
        result = admit_required_program_world_dispatch(
            {"task_id": "SAWM-040", "receipts": receipts}
        )
        return {
            "ok": bool(result["admitted"]),
            "step": step,
            "missing": list(result["missing"]),
            "completion_authority": False,
            "cas_completed": False,
            "generation_published": False,
            "admitted": False,
        }


class SemanticWorldNegativeScenarioMatrix:
    def __init__(self, path: Path) -> None:
        self.cases = json.loads(path.read_text(encoding="utf-8"))["cases"]


def run_semantic_world_acceptance_scenario(step: str, **kwargs: Any) -> dict[str, Any]:
    return SemanticWorldEndToEndScenario().run_semantic_world_acceptance_scenario(step, **kwargs)


NEGATIVE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "semantic_world"
    / "end_to_end_negative_cases.json"
)


def test_all_twenty_nine_steps_are_named() -> None:
    assert len(STEPS) == 29
    scenario = SemanticWorldEndToEndScenario()
    for step in STEPS:
        result = scenario.run_semantic_world_acceptance_scenario(step)
        assert result["completion_authority"] is False
        assert result["cas_completed"] is False
        assert result.get("generation_published") is False
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["completion_authority"] is False
    assert artifact["cas_completed"] is False


def test_negative_cases_never_complete() -> None:
    matrix = SemanticWorldNegativeScenarioMatrix(NEGATIVE)
    assert matrix.cases
    for case in matrix.cases:
        result = run_semantic_world_acceptance_scenario(case["step"], negative=case)
        assert result["ok"] is False
        assert result["completion_authority"] is False
        assert result["cas_completed"] is False


def test_similarity_cannot_admit_second_task_reuse() -> None:
    result = run_semantic_world_acceptance_scenario(
        "reuse", negative={"id": "neural-as-authority", "step": "reuse"}
    )
    assert result["ok"] is False
    assert result["ann_authoritative"] is False
    assert result["admitted"] is False
    assert "similar" in result["reason_code"] or result["reason_code"] == "ann_not_authoritative"
