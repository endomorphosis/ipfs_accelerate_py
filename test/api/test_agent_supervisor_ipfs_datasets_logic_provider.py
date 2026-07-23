"""Contract tests for the supervisor-to-Hammer provider adapter."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    CodeProofObligation,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_provider import (
    ProviderFailureCode,
    ProviderRequest,
    dispatch_provider_request,
)
from ipfs_accelerate_py.agent_supervisor.ipfs_datasets_logic_provider import (
    HAMMER_TRANSLATOR_ID,
    HammerSupervisorPolicy,
    IpfsDatasetsLogicProvider,
)


def _lock(*, solvers=("z3",)):
    return {
        "itp": "lean",
        "itp_version": "4.19.0",
        "kernel_command_template": "lean {source}",
        "solver_versions": {solver: f"{solver}-pinned" for solver in solvers},
        "executable_paths": {
            "lean": "/opt/pinned/bin/lean",
            **{solver: f"/opt/pinned/bin/{solver}" for solver in solvers},
        },
        "os_info": "linux-x86_64-pinned",
        "container_digest": "sha256:environment",
    }


def _policy(**overrides):
    values = {
        "allowed_solvers": ("cvc5", "z3"),
        "timeout_ms": 20_000,
        "cpu_time_ms": 12_000,
        "memory_bytes": 256 * 1024 * 1024,
        "max_premises": 4,
        "max_parallel_processes": 2,
        "network_allowed": False,
        "environment_lock": _lock(solvers=("cvc5", "z3")),
        "fallback_checks": ("pytest:provider-fallback",),
    }
    values.update(overrides)
    return HammerSupervisorPolicy(**values)


def _obligation(**overrides):
    values = {
        "repository_id": "repo",
        "repository_tree_id": "tree:candidate",
        "ast_scope_ids": ("src/state.py::advance",),
        "statement": "(assert (not bad_transition))",
        "premise_ids": ("premise:relation", "premise:state"),
        "template_id": "legal-state-transitions",
        "template_version": "1.0.0",
        "template_semantic_hash": "sha256:template",
        "invariant_class": "state_transition",
        "task_id": "REF-253",
        "fallback_checks": ("pytest:state-transitions",),
        "metadata": {
            "translation_family": "smtlib2",
            "statement_format": "smtlib2",
            "corpus_revision": "corpus:reviewed",
            "upstream_receipt_ids": ["receipt:obligation"],
        },
    }
    values.update(overrides)
    return CodeProofObligation(**values)


def _premises():
    # Deliberately reverse the obligation order. The adapter must emit the
    # explicit records in canonical obligation order, not caller order.
    return [
        {
            "premise_id": "premise:state",
            "statement": "The current state is ready.",
            "receipt_id": "receipt:state",
            "content_digest": "sha256:state",
        },
        {
            "premise_id": "premise:relation",
            "statement": "Ready may transition only to running.",
            "upstream_receipt_ids": ["receipt:relation"],
            "content_digest": "sha256:relation",
        },
    ]


def _request(operation="translate", **payload):
    body = {"obligation": _obligation().to_dict(), "premises": _premises()}
    body.update(payload)
    return ProviderRequest(
        request_id=f"provider-{operation}",
        operation=operation,
        payload=body,
        resource_budget=ResourceBudget(
            wall_time_ms=8_000,
            cpu_time_ms=5_000,
            memory_bytes=96 * 1024 * 1024,
            max_processes=1,
            max_premises=2,
            network_allowed=True,
        ),
        # Network remains denied because the supervisor policy denies it.
        network_allowed=True,
    )


def test_supported_obligation_is_a_deterministic_explicit_hammer_request():
    provider = IpfsDatasetsLogicProvider(_policy())

    first = dispatch_provider_request(provider, _request()).require_result()
    second_request = _request()
    second_request = ProviderRequest(
        request_id="different-provider-envelope",
        operation=second_request.operation,
        payload=second_request.payload,
        resource_budget=second_request.resource_budget,
        network_allowed=second_request.network_allowed,
    )
    second = dispatch_provider_request(provider, second_request).require_result()

    assert first["hammer_request"] == second["hammer_request"]
    assert first["hammer_request"]["request_id"].startswith(
        "hammer-request:sha256:"
    )
    assert first["hammer_request"]["theorem_id"] == _obligation().obligation_id
    assert first["hammer_request"]["created_at"] == "1970-01-01T00:00:00+00:00"
    assert [item["premise_id"] for item in first["premises"]] == [
        "premise:relation",
        "premise:state",
    ]
    assert first["hammer_request"]["metadata"]["premise_ids"] == [
        "premise:relation",
        "premise:state",
    ]
    assert first["environment_lock"]["lock_id"].startswith(
        "hammer-environment:sha256:"
    )
    assert first["environment_lock"]["policy_digest"] == first[
        "portfolio_policy"
    ]["supervisor_policy_id"]
    assert first["provenance"]["translator_id"] == HAMMER_TRANSLATOR_ID


def test_all_resource_and_capability_limits_flow_from_supervisor_policy():
    result = dispatch_provider_request(
        IpfsDatasetsLogicProvider(_policy()),
        _request(
            supervisor_policy={
                "allowed_solvers": ["z3"],
                "timeout_ms": 7_000,
                "cpu_time_ms": 4_000,
                "memory_bytes": 80 * 1024 * 1024,
                "max_premises": 2,
                "max_parallel_processes": 1,
                # This cannot enable network when the provider policy denies it.
                "network_allowed": True,
            }
        ),
    ).require_result()

    hammer_policy = result["hammer_request"]["policy"]
    portfolio_policy = result["portfolio_policy"]
    assert hammer_policy["allowed_solvers"] == ["z3"]
    assert hammer_policy["timeout_seconds"] == 7
    assert hammer_policy["cpu_seconds"] == 4
    assert hammer_policy["memory_mb"] == 80
    assert hammer_policy["max_premises"] == 2
    assert hammer_policy["network_allowed"] is False
    assert portfolio_policy["max_parallel_processes"] == 1
    assert result["environment_lock"]["solver_versions"]["z3"] == "z3-pinned"


def test_request_cannot_expand_solver_or_resource_policy():
    response = dispatch_provider_request(
        IpfsDatasetsLogicProvider(_policy(allowed_solvers=("z3",), environment_lock=_lock())),
        _request(supervisor_policy={"allowed_solvers": ["vampire"]}),
    )

    assert response.ok is False
    assert response.error.code is ProviderFailureCode.MALFORMED_REQUEST
    assert response.error.details["configured_allowed_solvers"] == ["z3"]
    assert response.error.details["requested_allowed_solvers"] == ["vampire"]


def test_portfolio_attempt_and_candidate_keep_upstream_receipt_provenance():
    captured = []

    def fake_portfolio(invocation):
        captured.append(invocation)
        attempt_id = f"{invocation.bundle.request_id}:translation:z3:0"
        return {
            "request_id": invocation.bundle.request_id,
            "status": "candidate",
            "attempts": [
                {
                    "attempt_id": attempt_id,
                    "request_id": invocation.bundle.request_id,
                    "translation_id": invocation.translations[0].translation_id,
                    "solver_name": "z3",
                }
            ],
            "proof_candidate": {
                "candidate_id": "candidate:1",
                "request_id": invocation.bundle.request_id,
                "solver_attempt_id": attempt_id,
                "premise_ids": ["premise:relation"],
            },
        }

    provider = IpfsDatasetsLogicProvider(
        _policy(allowed_solvers=("z3",), environment_lock=_lock()),
        portfolio_runner=fake_portfolio,
    )
    result = dispatch_provider_request(
        provider,
        _request(
            operation="prove",
            supervisor_policy={"allowed_solvers": ["z3"]},
        ),
    ).require_result()

    assert len(captured) == 1
    assert len(captured[0].attempt_specs) == 1
    assert result["status"] == "candidate"
    assert result["authoritative_assurance"] == "unverified"
    assert result["kernel_checked"] is False
    assert result["proof_success"] is False
    assert result["upstream_receipt_ids"] == [
        "receipt:obligation",
        "receipt:relation",
        "receipt:state",
    ]
    attempt = next(iter(result["provenance"]["solver_attempts"].values()))
    candidate = result["provenance"]["proof_candidates"]["candidate:1"]
    assert attempt["upstream_receipt_ids"] == result["upstream_receipt_ids"]
    assert candidate["upstream_receipt_ids"] == result["upstream_receipt_ids"]
    assert candidate["trusted"] is False
    assert candidate["solver_attempt_id"] in result["provenance"]["solver_attempts"]


def test_unknown_translation_is_typed_unsupported_with_exact_fallbacks():
    obligation = _obligation(
        metadata={"translation_family": "higher_order_dependent"}
    )
    request = _request()
    request = ProviderRequest(
        request_id=request.request_id,
        operation=request.operation,
        payload={"obligation": obligation.to_dict(), "premises": _premises()},
        resource_budget=request.resource_budget,
        network_allowed=request.network_allowed,
    )

    response = dispatch_provider_request(
        IpfsDatasetsLogicProvider(_policy()), request
    )

    assert response.ok is False
    assert response.error.code is ProviderFailureCode.UNSUPPORTED
    assert response.error.details["status"] == "unsupported"
    assert response.error.details["reason_code"] == "translation_family_unsupported"
    assert response.error.details["fallback_checks"] == [
        "pytest:provider-fallback",
        "pytest:state-transitions",
    ]
    assert response.error.details["proof_success"] is False


def test_missing_reviewed_lowering_and_premise_overflow_fail_closed():
    no_lowering = _obligation(
        statement="English statement requiring a reviewed lowering",
        metadata={"translation_family": "lean4"},
    )
    request = _request(operation="prove")
    request = ProviderRequest(
        request_id=request.request_id,
        operation=request.operation,
        payload={"obligation": no_lowering.to_dict(), "premises": _premises()},
        resource_budget=request.resource_budget,
        network_allowed=request.network_allowed,
    )
    unsupported = dispatch_provider_request(
        IpfsDatasetsLogicProvider(
            _policy(allowed_solvers=("z3",), environment_lock=_lock())
        ),
        request,
    )
    assert unsupported.error.code is ProviderFailureCode.UNSUPPORTED
    assert unsupported.error.details["reason_code"] == "lowering_artifact_missing"

    overflow_request = _request()
    overflow_request = ProviderRequest(
        request_id=overflow_request.request_id,
        operation=overflow_request.operation,
        payload=overflow_request.payload,
        resource_budget=ResourceBudget(max_premises=1),
    )
    overflow = dispatch_provider_request(
        IpfsDatasetsLogicProvider(_policy()), overflow_request
    )
    assert overflow.error.code is ProviderFailureCode.RESOURCE_EXHAUSTED
    assert overflow.error.details == {"max_premises": 1, "premise_count": 2}
