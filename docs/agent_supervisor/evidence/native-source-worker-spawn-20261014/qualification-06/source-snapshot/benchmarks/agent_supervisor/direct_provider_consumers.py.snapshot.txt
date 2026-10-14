"""Bounded direct-consumer qualification with authored transports and real quota.

Exercises prompt planning's default router boundary and the public Leanstral
proof provider against an isolated durable SystemClock ledger. No model, solver,
child process, target program, or network provider executes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    PromptGoalPlannerConfig, _default_router,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import ResourceBudget
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import ProviderRequest
from ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider import (
    LEANSTRAL_MODEL_RESOURCE_CLASS, LeanstralProofProvider, LeanstralProofProviderConfig,
)
from ipfs_accelerate_py.agent_supervisor.runtime.direct_provider_usage import DirectProviderUsageBinding
from ipfs_accelerate_py.agent_supervisor.runtime.provider_usage_migration import ProviderUsageMigrationError
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    LaneResourceRequirements, ResourcePolicy, ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import stable_id
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope, LimitWindow, ProtocolKind, Quantity, UsageDimension, UsageLimit, WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import DurableUsageLedgerStore, SystemClock


SCHEMA = "direct-provider-consumers-authored-benchmark@1"


def run_benchmark(request_count: int = 12) -> dict[str, object]:
    if type(request_count) is not int or not 2 <= request_count <= 32:
        raise ValueError("request_count must be an integer between 2 and 32")
    started = time.monotonic()
    calls = {"prompt": 0, "proof": 0}
    scope = EndpointUsageScope(
        provider_id=stable_id("provider", "direct-consumer-benchmark"),
        protocol=ProtocolKind.HTTPS, operation="text.generate",
        deployment_id=stable_id("deployment", "authored"),
        credential_pseudonym=stable_id("credential", "authored"),
    )

    def prompt_transport(prompt, invocation):
        assert invocation.provider == "authored-provider"
        assert invocation.model_name == "authored-model"
        assert invocation.allow_cross_provider_fallback is False
        calls["prompt"] += 1
        return "authored plan"

    def proof_transport(prompt, **kwargs):
        assert kwargs["provider"] == "authored-provider"
        assert kwargs["model_name"] == "authored-model"
        assert kwargs["disable_model_retry"] is True
        assert "usage_binding" not in kwargs and "usage_coordinator" not in kwargs
        calls["proof"] += 1
        return "by exact premise_1"

    with tempfile.TemporaryDirectory(prefix="direct-provider-consumers-") as directory:
        store = DurableUsageLedgerStore(Path(directory) / "ledger.json", clock=SystemClock(), fence=1)
        owner = UsageCoordinator(store, fence=1)
        owner.configure_limits(scope.scope_id, [UsageLimit(
            scope_id=scope.scope_id, dimension=UsageDimension(name), ceiling=Quantity.finite(amount),
            window=LimitWindow(kind=WindowKind.LIFETIME),
        ) for name, amount in {"requests": request_count, "input_tokens": 100_000,
                              "output_tokens": 100_000}.items()])
        binding = DirectProviderUsageBinding(
            coordinator=owner, scope=scope, provider_id="authored-provider", model="authored-model",
            reservation_ttl_ms=60_000, fence=1,
        )
        prompt_config = PromptGoalPlannerConfig(
            repo_root=Path(directory), provider="authored-provider", model="authored-model",
            timeout_seconds=5, max_new_tokens=32, usage_binding=binding,
        )
        # Authored request attributes at the production transport boundary;
        # this benchmark does not claim to parse a complete planning graph.
        prompt_request = SimpleNamespace(
            request_id="request:benchmark-prompt", repository_root=directory,
            planning_policy=SimpleNamespace(provider_preferences=(), model_preferences=()),
            budget=SimpleNamespace(max_latency_ms=5_000, max_provider_tokens=32, max_prompt_tokens=256),
        )
        proof = LeanstralProofProvider(LeanstralProofProviderConfig(
            llm_provider="authored-provider", model="authored-model", timeout_seconds=5,
            max_new_tokens=32,
        ), usage_binding=binding)
        with patch.dict("os.environ", {"IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE": "off"}), \
                patch("ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.call_llm_router", prompt_transport), \
                patch("ipfs_accelerate_py.llm_router.generate_text", proof_transport):
            for index in range(request_count):
                assert time.monotonic() - started < 30, "bounded campaign deadline exceeded"
                if index % 2 == 0:
                    assert _default_router("Plan the authored bounded task.", prompt_request,
                                           prompt_config) == "authored plan"
                else:
                    result = proof.prove(ProviderRequest(
                        request_id=f"request:benchmark-proof-{index}", operation="prove",
                        payload={"prompt": "Prove using premise_1.", "obligation_ids": ["obligation-1"],
                                 "canonical_source_digest": "sha256:authored",
                                 "resource_class": LEANSTRAL_MODEL_RESOURCE_CLASS},
                        resource_budget=ResourceBudget(wall_time_ms=5_000, model_token_limit=32,
                                                       max_output_bytes=4096),
                    ))
                    assert result["draft_text"] == "by exact premise_1" and not result["verified"]
            try:
                _default_router("Plan after quota exhaustion.", prompt_request, prompt_config)
            except ProviderUsageMigrationError as exc:
                assert "limit_exhausted" in exc.reason_codes
            else:
                raise AssertionError("exhausted quota must refuse consumer dispatch")
        assert sum(calls.values()) == request_count
        records = tuple(store.read()["reservations"].values())
        assert len(records) == request_count and all(r["state"] == "committed" for r in records)
        charges = {name: sum(r.get("charged_amounts", {}).get(name, 0) for r in records)
                   for name in ("requests", "input_tokens", "output_tokens")}
        assert charges["requests"] == request_count
        assert charges["input_tokens"] == (
            calls["prompt"] * len("Plan the authored bounded task.".encode("utf-8"))
            + calls["proof"] * len("Prove using premise_1.".encode("utf-8"))
        )
        assert charges["output_tokens"] == request_count * 32
        assert all(r.get("admission_revision_at_decision", "").startswith("uadm_") for r in records)
    return {
        "schema": SCHEMA, "status": "completed", "request_count": request_count,
        "authored_callbacks": calls, "capacity_refusals": 1, "remaining_usage_holds": 0,
        "bound_enforcement_despite_legacy_off_mode": True, "settled_estimated_units": charges,
        "clock": "SystemClock", "store": "isolated durable CAS ledger",
        "elapsed_ms": round((time.monotonic() - started) * 1000),
        "live_provider_calls": 0, "solver_executions": 0, "target_executions": 0,
        "claims_provider_token_savings": False, "cross_process_store_tested": False,
        "hard_process_containment_tested": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    host = ResourceScheduler(ResourcePolicy(max_lanes=1))
    with public_replay_without_metadata():
        decision, lease = host.acquire(LaneResourceRequirements(
            lane_id="direct-provider-consumer-benchmark", stage="analysis",
            memory_bytes=256 * 1024**2, process_slots=1,
        ))
    if lease is None:
        report = {"schema": SCHEMA, "status": "deferred", "reason": decision.reason,
                  "live_host_preflight": "refused", "live_provider_calls": 0}
        code = 2
    else:
        try:
            report = run_benchmark(args.requests)
            report.update(live_host_preflight="admitted", live_host_reserved_process_slots=1,
                          live_host_reserved_memory_bytes=256 * 1024**2)
            code = 0
        finally:
            host.release(lease)
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
