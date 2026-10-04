"""Bounded native SMT tasks beneath the datasets/supervisor shared lease.

Success means every requested solver returned an explicitly expected raw
satisfiability verdict. It grants neither source correspondence nor a checked
proof. A failed/unknown/missing attempt keeps dependent tasks blocked and its
verdict ledger is retained in the supervisor receipt. These tasks deliberately
do not use the supervisor's execution-bypassing deterministic result cache.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .multi_prover_resources import (
    MultiProverResourceClass, ProverResourceRequest, ProverTask, ProverTaskFailure,
)

SCHEMA = "datasets-supervisor-smt-observation@1"
MAX_TARGET_BYTES = 1024 * 1024
_MIB = 1024 * 1024


class _Cancellation:
    def __init__(self, context):
        self.context = context

    def is_set(self):
        return self.context.cancelled


def make_datasets_smt_task(
    *, target: Mapping[str, Any], expected_verdict: str,
    dependencies: Sequence[str] = (), timeout_seconds: float = 30.0,
    memory_mb_per_solver: int = 256, max_parallel_processes: int = 1,
) -> ProverTask:
    """Prepare one exact propositional Z3/CVC5 collection task without launch.

    ``target`` contains the arguments of datasets ``prepare_family_portfolio``.
    ``expected_verdict`` is explicitly ``sat`` or ``unsat``; neither means a
    kernel-checked theorem. At execution, an actual bridged child is required.
    The default one-process portfolio lets the outer supervisor distribute
    independent tasks; requesting two permits a bounded inner solver race.
    Every requested solver must finish, so early winner cancellation is off.
    """
    from ipfs_datasets_py.logic.hammers import semantic_routing as routing

    if type(target) is not dict:
        raise TypeError("target must be a concrete native target dictionary")
    if expected_verdict not in ("sat", "unsat"):
        raise ValueError("expected_verdict must be sat or unsat")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300):
        raise ValueError("timeout_seconds must be finite in (0, 300]")
    if type(memory_mb_per_solver) is not int or not 128 <= memory_mb_per_solver <= 4096:
        raise ValueError("memory_mb_per_solver must be an integer in [128, 4096]")
    if type(max_parallel_processes) is not int or not 1 <= max_parallel_processes <= 2:
        raise ValueError("max_parallel_processes must be one or two")
    if (type(dependencies) not in (tuple, list) or len(dependencies) > 64
            or any(type(item) is not str or not item or len(item.encode()) > 1024 for item in dependencies)):
        raise ValueError("bounded explicit dependency identifiers required")
    prepared, routing_receipt = routing.prepare_family_portfolio(**target)
    if any(row["solver_name"] not in ("z3", "cvc5") or row["operation"] != "check_satisfiability"
           for row in routing_receipt["routes"]):
        raise ValueError("this task adapter supports only native Z3/CVC5 satisfiability")
    # Replay validation bounds the AST before this deep copy and serialization.
    encoded = json.dumps(target, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_TARGET_BYTES:
        raise ValueError("serialized SMT target exceeds 1 MiB")
    sealed = json.loads(encoded)
    names = tuple(sealed["solver_names"])
    width = min(max_parallel_processes, len(names))
    task_id = sealed["request_id"]
    target_sha256 = hashlib.sha256(encoded).hexdigest()
    bindings = {item.solver_name: (item.translation.translation_id,
        routing.portfolio.compute_content_digest({"solver_name": item.solver_name,
            "target": item.translation.target.value,
            "text": routing.portfolio.build_solver_input_text(item.translation)})) for item in prepared}

    def run(context):
        from .datasets_prover_resources import DatasetsChildResourceLease
        from ipfs_datasets_py.logic.hammers.models import HammerPolicy
        from ipfs_datasets_py.logic.hammers.policy import PortfolioPolicy

        if not isinstance(context.lease, DatasetsChildResourceLease):
            raise TypeError("native SMT task requires a datasets-backed supervisor child")
        remaining = context.remaining_seconds
        if context.cancelled or (remaining is not None and remaining <= 0):
            raise ProverTaskFailure("SMT task cancelled before launch", reasons=("cancelled",))
        wall = min(timeout_seconds, remaining) if remaining is not None else timeout_seconds
        policy = PortfolioPolicy(
            hammer_policy=HammerPolicy(allowed_solvers=list(names), timeout_seconds=wall,
                cpu_seconds=max(1, math.ceil(wall)), memory_mb=memory_mb_per_solver,
                network_allowed=False),
            max_parallel_processes=width, cancel_on_first_conclusive=False,
        )
        result = routing.run_family_portfolio(
            expected_routing=routing_receipt, run_policy=policy,
            parent_lease=context.lease.datasets_parent_lease,
            resource_scheduler=context.lease.datasets_scheduler,
            resource_wait_timeout_seconds=min(30.0, wall), cancel_event=_Cancellation(context),
            **sealed,
        )
        attempts = [row.to_dict() for row in result.attempts]
        by_solver = {row["solver_name"]: row for row in attempts}
        evidence_inputs = {key: value.input_digest for key, value in result.evidence.items()}
        matched = (not result.denied and not result.cancelled_attempt_ids
            and len(attempts) == len(names) and set(by_solver) == set(names)
            and all(row["verdict"] == expected_verdict and type(row["exit_code"]) is int
                and row["exit_code"] == 0 and bool(row["solver_version"])
                and row["request_id"] == task_id and row["target"] == "smtlib"
                and row["translation_id"] == bindings[row["solver_name"]][0]
                and evidence_inputs.get(row["attempt_id"]) == bindings[row["solver_name"]][1]
                for row in attempts))
        payload = {
            "schema": SCHEMA, "request_id": task_id, "target_sha256": target_sha256,
            "native_ast_sha256": routing_receipt["native_ast_sha256"],
            "operation": "check_satisfiability", "input_semantics": "assert_formula",
            "profile": routing_receipt["profile"], "expected_verdict": expected_verdict,
            "all_expected_verdicts_observed": matched, "attempts": attempts,
            "denied": result.denied, "cancelled_attempt_ids": result.cancelled_attempt_ids,
            "evidence_input_digests": evidence_inputs,
            "datasets_parent_lease_id": context.lease.datasets_parent_lease.lease_id,
            "portfolio_cpu_slots": result.resource_telemetry.get("portfolio_cpu_slots"),
            "portfolio_memory_mb": result.resource_telemetry.get("portfolio_memory_mb"),
            "proof_authority": False, "source_semantics_verified": False,
            "behavior_authority": False, "execution_authority": False, "completion_authority": False,
        }
        if not matched:
            raise ProverTaskFailure("SMT portfolio did not return every expected verdict",
                result=payload, reasons=("smt_expected_verdict_not_observed",))
        return payload

    return ProverTask(task_id=task_id,
        resources=ProverResourceRequest.for_family(task_id, MultiProverResourceClass.SMT,
            cpu_slots=width, process_slots=width, thread_slots=width,
            memory_bytes=(128 + width * memory_mb_per_solver) * _MIB),
        runner=run, dependencies=tuple(dependencies), timeout_ms=math.ceil(timeout_seconds * 1000))


__all__ = ["SCHEMA", "MAX_TARGET_BYTES", "make_datasets_smt_task"]
