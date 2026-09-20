"""SAWM-039 typed program-world service.

Deterministic JSON operations over landed remaining-task modules.
Never writes DuckDB or completes tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ProgramWorldServiceError(ValueError):
    """Closed program-world service contract violation."""


@dataclass(frozen=True, slots=True)
class SemanticWorldStatusResult:
    status: str
    mode: str
    completion_authority: bool = False


@dataclass(frozen=True, slots=True)
class SemanticWorldResolveResult:
    query: str
    resolved: bool
    reason_code: str
    completion_authority: bool = False


def _cid(label: str) -> str:
    from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    return cid_for_bytes(str(label).encode("utf-8"))


def _reuse_key(payload: Mapping[str, Any]):
    from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse import (
        ProgramWorldReuseKey,
    )

    return ProgramWorldReuseKey(
        state_cid=str(payload.get("state_cid") or _cid("sawm-svc-state")),
        goal_cid=str(payload.get("goal_cid") or _cid("sawm-svc-goal")),
        policy_cid=str(payload.get("policy_cid") or _cid("sawm-svc-policy")),
        environment_cid=str(payload.get("environment_cid") or _cid("sawm-svc-env")),
        toolchain_cid=str(payload.get("toolchain_cid") or _cid("sawm-svc-toolchain")),
        procedure_revision_cid=str(
            payload.get("procedure_revision_cid") or _cid("sawm-svc-procedure")
        ),
    )


class ProgramWorldService:
    INTERFACE = "SemanticWorldService@1"

    def status(self) -> SemanticWorldStatusResult:
        return SemanticWorldStatusResult(status="ready", mode="required")

    def resolve(self, query: Mapping[str, Any]) -> SemanticWorldResolveResult:
        q = str(query.get("query") or "")
        if not q:
            return SemanticWorldResolveResult(
                query=q, resolved=False, reason_code="empty_query"
            )
        cid = str(query.get("object_cid") or query.get("cid") or "")
        if not cid:
            return SemanticWorldResolveResult(
                query=q,
                resolved=False,
                reason_code="identity_evidence_required",
            )
        try:
            from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
                validate_opaque_cid,
            )

            validate_opaque_cid(cid, "object_cid")
        except Exception:
            return SemanticWorldResolveResult(
                query=q,
                resolved=False,
                reason_code="identity_cid_rejected",
            )
        return SemanticWorldResolveResult(
            query=q,
            resolved=False,
            reason_code="proposal_only_citation",
        )

    def operation(self, name: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        allowed = {
            "world",
            "graph",
            "state",
            "trace",
            "transition",
            "call-target",
            "repair",
            "relation",
            "projection",
            "reuse",
            "procedure",
            "dogfood",
            "index",
            "benchmark",
        }
        if name not in allowed:
            raise ProgramWorldServiceError(f"unknown operation {name}")
        body = dict(payload or {})
        handlers = {
            "reuse": self._reuse,
            "call-target": self._rank,
            "projection": self._project,
            "repair": self._repair,
            "graph": self._graph,
            "trace": self._trace,
            "procedure": self._procedure,
            "benchmark": self._benchmark,
            "world": self._world,
            "state": self._state,
            "transition": self._transition,
            "relation": self._relation,
            "dogfood": self._dogfood,
            "index": self._index,
        }
        result = handlers[name](body)
        result.setdefault("operation", name)
        result.setdefault("proposal_only", True)
        result.setdefault("completion_authority", False)
        result.setdefault("admitted", False)
        result.setdefault("cas_completed", False)
        result.setdefault("task_id", body.get("task_id") or "SAWM-039")
        result.setdefault("board", "sawm")
        try:
            from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import (
                persist_program_world_record,
            )

            persist_program_world_record(result)
        except Exception:
            pass
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
                compose_semantic_work,
            )

            result["semantic_work"] = compose_semantic_work(
                subject_kind="task_id",
                subject_ref=str(result.get("task_id") or "SAWM-039"),
            )
        except Exception:
            pass
        return result

    def _reuse(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse import (
            evaluate_program_world_reuse,
        )

        decision = evaluate_program_world_reuse(
            _reuse_key(payload),
            similarity_candidates=tuple(payload.get("similarity_candidates") or ()),
            typed_available=payload.get("typed_available", True),
        )
        return {
            "verdict": str(decision.verdict),
            "reason_code": decision.reason_code,
            "proposal_only": True,
            "admitted": False,
            "ann_authoritative": bool(decision.ann_authoritative),
        }

    def _rank(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.analysis.program_call_ranker import (
            rank_program_call_targets,
        )

        ranked = rank_program_call_targets(
            {
                "current_symbol": payload.get("current_symbol") or "main",
                "static_candidates": payload.get("static_candidates") or ("helper",),
                "ood": bool(payload.get("ood")),
                "stale": bool(payload.get("stale")),
            }
        )
        return {
            "ranked": list(ranked.ranked),
            "abstained": ranked.abstained,
            "reason_code": ranked.reason_code,
            "proposal_only": True,
        }

    def _project(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from ipfs_datasets_py.logic.software_contracts.semantic_state.program_views import (
            ProgramViewError,
            build_program_world_view,
        )

        try:
            view = build_program_world_view(
                {
                    "view": payload.get("view") or "ast",
                    "source_cid": payload.get("source_cid") or _cid("sawm-svc-source"),
                    "privacy_admitted": payload.get("privacy_admitted", True),
                    "freshness": payload.get("freshness") or "fresh",
                }
            )
        except ProgramViewError as exc:
            return {"ok": False, "reason_code": str(exc), "proposal_only": True}
        return {
            "ok": True,
            "view": view.view,
            "privacy_admitted": view.privacy_admitted,
            "proposal_only": True,
        }

    def _repair(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_delta_predictor import (
            RepairPredictionError,
            predict_program_graph_delta,
        )

        try:
            delta = predict_program_graph_delta(
                {
                    "operators": payload.get("operators") or ("rewrite",),
                    "sketch": payload.get("sketch") or {"path": "src/app.py", "operator": "rewrite"},
                }
            )
        except RepairPredictionError as exc:
            return {"ok": False, "reason_code": str(exc), "proposal_only": True}
        return {"ok": True, "delta": dict(delta), "proposal_only": True}

    def _graph(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.analysis.program_event_predictor import (
            predict_next_program_event,
        )

        predicted = predict_next_program_event(
            {
                "event_type": payload.get("event_type") or "call",
                "current_state": payload.get("current_state") or "state",
            }
        )
        return {"ok": True, "prediction": dict(predicted), "proposal_only": True}

    def _trace(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.analysis.inverse_trace_predictor import (
            rank_inverse_trace_predecessors,
        )

        ranked = rank_inverse_trace_predecessors(
            {
                "predecessor_states": payload.get("predecessor_states")
                or ({"state_id": "s0", "score": 1.0},),
                "predecessor_events": payload.get("predecessor_events")
                or ({"event_id": "e0", "score": 1.0},),
                "stale": bool(payload.get("stale")),
                "ood": bool(payload.get("ood")),
            }
        )
        return {"ok": True, "predecessors": dict(ranked), "proposal_only": True}

    def _procedure(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_guarded import (
            evaluate_guarded_program_world_influence,
        )

        decision = evaluate_guarded_program_world_influence(
            {
                "kind": payload.get("kind") or "verified_procedure",
                "verified": payload.get("verified", True),
            }
        )
        return {
            "allowed": decision.allowed,
            "reason_code": decision.reason_code,
            "influences_planning": decision.influences_planning,
            "proposal_only": True,
        }

    def _benchmark(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from benchmarks.agent_supervisor.semantic_addressed_world_model.ablation import (
            run_semantic_world_ablation,
        )

        result = run_semantic_world_ablation(
            payload.get("rungs") or ({"rung": "A", "model_calls": 0, "tokens": 0},)
        )
        return {
            "rungs": list(result["rungs"]),
            "proposal_only": True,
            "completion_authority": False,
        }

    def _world(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, Mapping) or "snapshot_cid" not in snapshot:
            return {
                "bound": False,
                "reason_code": "snapshot_unavailable",
                "proposal_only": True,
            }
        from ipfs_accelerate_py.agent_supervisor.semantic_state.world_view import (
            SupervisorWorldView,
            WorldViewError,
        )

        try:
            SupervisorWorldView(snapshot)
        except (WorldViewError, ValueError) as exc:
            return {
                "bound": False,
                "reason_code": str(exc)[:200],
                "proposal_only": True,
            }
        return {"bound": True, "proposal_only": True}

    def _state(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_required import (
            admit_required_program_world_dispatch,
        )

        result = admit_required_program_world_dispatch(
            {
                "task_id": payload.get("task_id") or "SAWM-039",
                "receipts": tuple(payload.get("receipts") or ()),
            }
        )
        return {
            "admitted": False,
            "dispatch_admitted": bool(result["admitted"]),
            "missing": list(result["missing"]),
            "proposal_only": True,
        }

    def _transition(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.analysis.program_event_predictor import (
            EventPredictionError,
            predict_next_program_event,
        )

        try:
            predicted = predict_next_program_event(
                {
                    "event_type": payload.get("event_type") or "call",
                    "current_state": payload.get("current_state") or "state",
                    "observed_event": payload.get("observed_event"),
                }
            )
        except EventPredictionError as exc:
            return {"ok": False, "reason_code": str(exc), "proposal_only": True}
        return {
            "ok": True,
            "prediction": dict(predicted),
            "observation": False,
            "proposal_only": True,
        }

    def _relation(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(payload.get("kind") or payload.get("relation_kind") or "exact")
        if kind.lower() in {"similar", "similarity", "ann", "knn", "nearest"}:
            return {
                "ok": False,
                "reason_code": "neural_similarity_is_never_a_semantic_relation",
                "ann_authoritative": False,
                "proposal_only": True,
            }
        return {
            "ok": True,
            "kind": kind,
            "source_authority": "ipfs_datasets_py.logic.software_contracts.semantic_state.program_relations",
            "proposal_only": True,
            "ann_authoritative": False,
        }

    def _dogfood(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        import importlib.util
        import sys
        from pathlib import Path

        module_path = (
            Path(__file__).resolve().parents[1] / "evaluation" / "program_graph_sequence.py"
        )
        spec = importlib.util.spec_from_file_location(
            "program_graph_sequence_dogfood", module_path
        )
        if spec is None or spec.loader is None:
            return {
                "ok": False,
                "reason_code": "graph_sequence_unavailable",
                "proposal_only": True,
            }
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result = module.run_program_graph_sequence_ablation(
            nodes=tuple(payload.get("nodes") or ("a", "b")),
            types=dict(payload.get("types") or {"a": "fn", "b": "unknown"}),
            gold=str(payload.get("gold") or "a"),
            backends=dict(
                payload.get("backends")
                or {"gnn": False, "graph_transformer": False, "tagseq": False, "linear": True}
            ),
        )
        return {
            "ok": True,
            "static_baseline_hit": result["static_baseline_hit"],
            "encoders": result["encoders"],
            "runtime_authority": False,
            "proposal_only": True,
        }

    def _index(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        del payload
        return {
            "available": False,
            "surface": "ipfs_kit_py.projection_index",
            "reason_code": "ann_index_unavailable",
            "ann_authoritative": False,
            "proposal_only": True,
        }


def describe_controls() -> dict[str, Any]:
    return {
        "interface": ProgramWorldService.INTERFACE,
        "commands": [
            "status",
            "resolve",
            "world",
            "graph",
            "state",
            "trace",
            "transition",
            "call-target",
            "repair",
            "relation",
            "projection",
            "reuse",
            "procedure",
            "dogfood",
            "index",
            "benchmark",
        ],
        "completion_authority": False,
        "authoritative": False,
        "subprocess": False,
    }
