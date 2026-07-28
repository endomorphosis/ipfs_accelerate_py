"""Contract tests for the supervisor-to-Hammer provider adapter."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    CodeProofObligation,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    ProviderFailureCode,
    ProviderRequest,
    dispatch_provider_request,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider import (
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
            "goal_id": "goal:reviewed",
            "accepted_plan_id": "plan:reviewed",
            "assumptions_digest": "assumptions:reviewed",
            "scope_set_id": "scope-set:reviewed",
            "effect_scope_map": {
                "effect:advance": ["src/state.py::advance"],
            },
            "code_proof_toolchain_id": "toolchain:reviewed",
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
    assert first["provenance"]["semantic_bindings"] == {
        "accepted_plan_id": "plan:reviewed",
        "assumptions_digest": "assumptions:reviewed",
        "changed_scope_set_id": "scope-set:reviewed",
        "effect_scope_map": {
            "effect:advance": ["src/state.py::advance"],
        },
        "goal_id": "goal:reviewed",
        "policy_id": first["portfolio_policy"]["supervisor_policy_id"],
        "toolchain_id": "toolchain:reviewed",
    }


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


def test_timeout_and_missing_reconstructor_remain_explicit():
    def time_out(_invocation):
        raise TimeoutError("fixture deadline")

    timed_out = dispatch_provider_request(
        IpfsDatasetsLogicProvider(
            _policy(allowed_solvers=("z3",), environment_lock=_lock()),
            portfolio_runner=time_out,
        ),
        _request(
            operation="prove",
            supervisor_policy={"allowed_solvers": ["z3"]},
        ),
    )
    assert timed_out.ok is False
    assert timed_out.error.code is ProviderFailureCode.TIMED_OUT
    assert timed_out.error.details["status"] == "timed_out"
    assert timed_out.error.details["proof_success"] is False
    assert timed_out.error.details["provenance"]["semantic_bindings"][
        "goal_id"
    ] == "goal:reviewed"

    unsupported = dispatch_provider_request(
        IpfsDatasetsLogicProvider(_policy()),
        _request(operation="reconstruct"),
    )
    assert unsupported.ok is False
    assert unsupported.error.code is ProviderFailureCode.UNSUPPORTED
    assert unsupported.error.details["status"] == "unsupported"
    assert (
        unsupported.error.details["reason_code"]
        == "independent_kernel_provider_required"
    )

    def candidate_without_kernel(invocation):
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
                "candidate_id": "candidate:policy-required",
                "request_id": invocation.bundle.request_id,
                "solver_attempt_id": attempt_id,
                "premise_ids": ["premise:relation"],
            },
        }

    required = dispatch_provider_request(
        IpfsDatasetsLogicProvider(
            _policy(
                allowed_solvers=("z3",),
                environment_lock=_lock(),
                require_authoritative_reconstruction=True,
            ),
            portfolio_runner=candidate_without_kernel,
        ),
        _request(
            operation="prove",
            supervisor_policy={"allowed_solvers": ["z3"]},
        ),
    )
    assert required.ok is False
    assert required.error.code is ProviderFailureCode.UNSUPPORTED
    assert (
        required.error.details["reason_code"]
        == "independent_kernel_provider_required"
    )
    assert required.error.details["candidate"]["candidate_id"] == (
        "candidate:policy-required"
    )


def test_reviewed_premise_selection_runs_through_hammer_boundary():
    from ipfs_datasets_py.logic import hammers

    manifest = hammers.CorpusManifest(manifest_id="manifest:legal-reviewed")
    manifest.register_source(
        hammers.CorpusSource(
            corpus_id="corpus:legal",
            name="Reviewed legal transition premises",
            version_ref="commit:reviewed",
            license_id="CC0-1.0",
        )
    )
    manifest.add_theorem(
        theorem_id="premise:relation",
        corpus_id="corpus:legal",
        statement="Ready may transition only to running.",
    )
    manifest.add_theorem(
        theorem_id="premise:state",
        corpus_id="corpus:legal",
        statement="The current state is ready.",
    )
    obligation = _obligation(
        metadata={
            **dict(_obligation().metadata),
            "corpus_revision": manifest.revision,
        }
    )
    request = _request(
        obligation=obligation.to_dict(),
        premise_selection={"top_k": 2},
        corpus_manifest=manifest.to_dict(),
    )

    result = dispatch_provider_request(
        IpfsDatasetsLogicProvider(_policy()), request
    ).require_result()
    repeated = dispatch_provider_request(
        IpfsDatasetsLogicProvider(_policy()), request
    ).require_result()

    assert repeated["hammer_request"] == result["hammer_request"]
    assert set(
        result["provenance"]["premise_selection"][
            "selected_premise_ids"
        ]
    ) == {"premise:relation", "premise:state"}
    assert {
        item["selection_method"] for item in result["premises"]
    } == {"deterministic-baseline"}
