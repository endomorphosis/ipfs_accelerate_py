"""Compact recipe catalogue for the exactly-40-task benchmark corpus.

Each task pins a controlled-fixture mutation and declares an independent oracle.
Oracles are reviewed fixture authority — never derived by executing the harness
or the benchmark implementation under test.
"""

from __future__ import annotations

from .task_record import (
    BaselineRetrievalPolicy,
    BenchmarkTask,
    CandidatePatch,
    TaskOracle,
)

# Stable fixture identities (mirror controlled_repo recipes; not scan output).
SYM_CORE_ADD = "sch_fixture.core:add"
SYM_CORE_MULTIPLY = "sch_fixture.core:multiply"
SYM_CORE_PROCESS = "sch_fixture.core:process"
SYM_API_FETCH = "sch_fixture.api:fetch_value"
SYM_API_CALL_CORE = "sch_fixture.api:call_core"
SYM_SCHEMA_USER = "sch_fixture.schema:UserRecord"
SYM_SCHEMA_DUMP = "sch_fixture.schema:dump_user"
SYM_SEC_AUTHORIZE = "sch_fixture.security:authorize"
SYM_ADAPTER_CLIENT = "sch_fixture.adapters:McpClientAdapter"
SYM_DYNAMIC_LOAD = "sch_fixture.dynamic_loader:load_plugin"
SYM_NATIVE_BRIDGE = "sch_fixture.native_bridge:native_hash"
SYM_GENERATED_BIND = "sch_fixture.generated.bindings:generated_constant"
SYM_POLICY = "policy.admission:AdmissionPolicy"
SYM_LOCK = "deps.lockfile:LockedDependencySet"
SYM_IFACE = "interfaces.mcp_client:McpClientDescriptor"
SYM_PYTEST_CFG = "pytest.ini:pytest_config"
SYM_FIXTURE_SAMPLE = "tests.conftest:sample_record"
SYM_RENAMED = "sch_fixture.core:renamed_process"
SYM_DELETED = "sch_fixture.core:legacy_helper"

TEST_CORE_ADD = "tests/test_core.py::test_add"
TEST_CORE_PROCESS = "tests/test_core.py::test_process"
TEST_API_FETCH = "tests/test_api.py::test_fetch_value"
TEST_API_CALL = "tests/test_api.py::test_call_core"
TEST_SCHEMA = "tests/test_schema.py::test_user_roundtrip"
TEST_SECURITY = "tests/test_security.py::test_authorize_allows"
TEST_ADAPTER = "tests/test_adapters.py::test_adapter_ping"
TEST_DYNAMIC = "tests/test_dynamic.py::test_load_plugin_name"
TEST_NATIVE = "tests/test_native.py::test_native_marker"
TEST_GENERATED = "tests/test_generated.py::test_generated_constant"

PROOF_CORE_ADD = "proof:sch_fixture.core:add"
PROOF_API_SIG = "proof:sch_fixture.api:fetch_value.signature"
PROOF_SCHEMA = "proof:sch_fixture.schema:UserRecord"
PROOF_SECURITY = "proof:sch_fixture.security:authorize.effects"
PROOF_OPAQUE = "proof:sch_fixture.native_bridge:native_hash.raw"

FULL_SUITE = tuple(
    sorted(
        {
            TEST_CORE_ADD,
            TEST_CORE_PROCESS,
            TEST_API_FETCH,
            TEST_API_CALL,
            TEST_SCHEMA,
            TEST_SECURITY,
            TEST_ADAPTER,
            TEST_DYNAMIC,
            TEST_NATIVE,
            TEST_GENERATED,
        }
    )
)

FIXTURE_CORPUS_ID = "semantic-state-controlled-repo-v1"
FIXTURE_PACKAGE_PATH = "test/fixtures/semantic_state_harness/controlled_repo"
TOKENIZER_ID = "sch-fixture/token-estimator@1"
ESTIMATOR_VERSION = "semantic-state-token-estimator-v1"

DEFAULT_BASELINE_RETRIEVAL = BaselineRetrievalPolicy(
    tokenizer_id=TOKENIZER_ID,
    estimator_version=ESTIMATOR_VERSION,
    coverage_policy="hard_coverage_no_omit_required",
    fixture_corpus_id=FIXTURE_CORPUS_ID,
    fixture_package_path=FIXTURE_PACKAGE_PATH,
    require_exact_target_source=True,
    allow_omit_required_raw=False,
    allow_network=False,
    allow_model_derived_expected_outcome=False,
)


def _oracle(
    *,
    invalidation: list[str],
    selected: list[str],
    proofs: list[str],
    assumptions: list[str],
    uncertainty: list[str],
    candidate_verification_outcome: str,
    production_acceptance: str = "not_applicable",
    fallback: str = "none",
    false_negatives: int = 0,
    full: list[str] | None = None,
    authority: str = "controlled_fixture_oracle",
) -> TaskOracle:
    return TaskOracle(
        invalidation_symbol_ids=tuple(sorted(invalidation)),
        selected_test_node_ids=tuple(sorted(selected)),
        full_suite_test_node_ids=tuple(sorted(full or list(FULL_SUITE))),
        proof_obligation_ids=tuple(sorted(proofs)),
        assumptions=tuple(sorted(assumptions)),
        uncertainty=tuple(sorted(uncertainty)),
        expected_false_negatives=false_negatives,
        fallback=fallback,
        candidate_verification_outcome=candidate_verification_outcome,
        production_acceptance=production_acceptance,
        oracle_authority=authority,
    )


def _candidate(
    task_id: str,
    mutation: str,
    paths: list[str],
    *,
    notes: str,
) -> CandidatePatch:
    return CandidatePatch(
        candidate_id=f"{task_id}:oracle-candidate",
        source="controlled_fixture_mutation",
        production_eligible=False,
        base_mutation_case_id=mutation,
        declared_paths=tuple(sorted(paths)),
        notes=notes,
    )


def _task(
    *,
    task_id: str,
    category: str,
    objective: str,
    targets: list[str],
    mutation: str,
    risk: str,
    route: str,
    multi_file: bool,
    frontier_or_human: bool,
    oracle: TaskOracle,
    candidate_notes: str,
) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        category=category,
        objective=objective,
        target_paths=tuple(sorted(targets)),
        base_mutation_case_id=mutation,
        risk=risk,
        expected_route=route,
        multi_file=multi_file,
        frontier_or_human=frontier_or_human,
        baseline_retrieval=DEFAULT_BASELINE_RETRIEVAL,
        oracle=oracle,
        candidate=_candidate(
            task_id,
            mutation,
            targets,
            notes=candidate_notes,
        ),
    )


def build_tasks() -> tuple[BenchmarkTask, ...]:
    """Return the closed, sorted catalogue of exactly 40 benchmark tasks."""

    tasks: list[BenchmarkTask] = []

    # ------------------------------------------------------------------
    # small_bug_fix (10)
    # ------------------------------------------------------------------
    tasks.append(
        _task(
            task_id="sch-bench-01-core-add-body-fix",
            category="small_bug_fix",
            objective=(
                "Repair off-by-noop body drift in sch_fixture.core.add while "
                "keeping the public signature stable."
            ),
            targets=["src/sch_fixture/core.py"],
            mutation="local_function_body",
            risk="low",
            route="small_local_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_CORE_ADD],
                selected=[TEST_CORE_ADD],
                proofs=[PROOF_CORE_ADD],
                assumptions=["signature_stable", "callers_reusable"],
                uncertainty=["none_declared"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes=(
                "Oracle replay of local_function_body; production_eligible=false."
            ),
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-02-core-process-pipeline",
            category="small_bug_fix",
            objective=(
                "Correct process pipeline composition after a local core body "
                "mutation without widening the edit surface."
            ),
            targets=["src/sch_fixture/core.py"],
            mutation="local_function_body",
            risk="low",
            route="small_local_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_CORE_ADD],
                selected=[TEST_CORE_ADD],
                proofs=[PROOF_CORE_ADD],
                assumptions=["process_calls_add"],
                uncertainty=["indirect_callers_not_rewritten"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Replay fixture for process-adjacent body repair.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-03-exception-behavior-fix",
            category="small_bug_fix",
            objective=(
                "Restore authorize exception/deny behavior after an exception "
                "contract mutation."
            ),
            targets=["src/sch_fixture/security.py"],
            mutation="exception_behavior",
            risk="medium",
            route="small_local_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_SEC_AUTHORIZE],
                selected=[TEST_SECURITY],
                proofs=[PROOF_SECURITY],
                assumptions=["security_boundary_local"],
                uncertainty=["caller_recovery_not_auto_rewritten"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Oracle exception-behavior repair candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-04-security-side-effect",
            category="small_bug_fix",
            objective=(
                "Fix authorize side-effect policy regression while preserving "
                "admin allow semantics."
            ),
            targets=["src/sch_fixture/security.py"],
            mutation="side_effect_security",
            risk="high",
            route="medium_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_SEC_AUTHORIZE],
                selected=[TEST_SECURITY],
                proofs=[PROOF_SECURITY],
                assumptions=["effect_set_must_match_policy"],
                uncertainty=["external_role_matrix_unknown"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Security side-effect oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-05-deleted-legacy-helper",
            category="small_bug_fix",
            objective=(
                "Complete deletion of legacy_helper and keep residual call sites "
                "from regressing selection recall."
            ),
            targets=["src/sch_fixture/core.py"],
            mutation="deleted_symbol",
            risk="medium",
            route="small_local_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_DELETED],
                selected=[TEST_CORE_PROCESS],
                proofs=[],
                assumptions=["delete_invalidates_dependents"],
                uncertainty=["dynamic_refs_may_exist"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Deleted-symbol oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-06-monkey-patch-surface",
            category="small_bug_fix",
            objective=(
                "Repair a monkey-patched helper surface without treating the "
                "patch as production model output."
            ),
            targets=["src/sch_fixture/core.py", "tests/test_core.py"],
            mutation="monkey_patch",
            risk="medium",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_CORE_PROCESS],
                selected=[TEST_CORE_PROCESS],
                proofs=[],
                assumptions=["test_patch_is_local"],
                uncertainty=["runtime_patch_order"],
                candidate_verification_outcome="pass",
                fallback="full_pytest",
            ),
            candidate_notes="Monkey-patch oracle/replay candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-07-dynamic-import-name",
            category="small_bug_fix",
            objective=(
                "Correct dynamic plugin name resolution after a syntactic "
                "dynamic-import mutation."
            ),
            targets=["src/sch_fixture/dynamic_loader.py"],
            mutation="dynamic_import",
            risk="high",
            route="medium_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_DYNAMIC_LOAD],
                selected=[TEST_DYNAMIC],
                proofs=[],
                assumptions=["import_string_is_literal"],
                uncertainty=["dynamic_dispatch_incomplete"],
                candidate_verification_outcome="pass",
                fallback="full_pytest",
            ),
            candidate_notes="Dynamic import oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-08-generated-bindings",
            category="small_bug_fix",
            objective=(
                "Refresh generated bindings constant after generated-file drift."
            ),
            targets=["src/sch_fixture/generated/bindings.py"],
            mutation="generated_file",
            risk="low",
            route="deterministic_only",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_GENERATED_BIND],
                selected=[TEST_GENERATED],
                proofs=[],
                assumptions=["generated_tree_is_checked_in"],
                uncertainty=["generator_not_reexecuted"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Generated-file oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-09-core-multiply-guard",
            category="small_bug_fix",
            objective=(
                "Apply a bounded body guard near multiply without expanding "
                "beyond the local_function_body mutation cone."
            ),
            targets=["src/sch_fixture/core.py"],
            mutation="local_function_body",
            risk="low",
            route="small_local_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_CORE_ADD],
                selected=[TEST_CORE_ADD],
                proofs=[PROOF_CORE_ADD],
                assumptions=["multiply_untouched_by_add_body"],
                uncertainty=["adjacent_symbol_noise"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Bounded body-guard oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-10-formatting-noise-adjacent",
            category="small_bug_fix",
            objective=(
                "Distinguish unrelated formatting noise from functional drift "
                "and keep invalidation empty for pure formatting."
            ),
            targets=["src/sch_fixture/core.py"],
            mutation="unrelated_formatting",
            risk="low",
            route="deterministic_only",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[],
                selected=[],
                proofs=[],
                assumptions=["formatting_is_non_semantic"],
                uncertainty=["formatter_version_drift"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Formatting-only oracle candidate; no symbol invalidation.",
        )
    )

    # ------------------------------------------------------------------
    # test_repair (6)
    # ------------------------------------------------------------------
    tasks.append(
        _task(
            task_id="sch-bench-11-fixture-sample-record",
            category="test_repair",
            objective=(
                "Repair tests.conftest sample_record fixture after fixture "
                "dependency mutation."
            ),
            targets=["tests/conftest.py"],
            mutation="fixture_dependency",
            risk="medium",
            route="small_local_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_FIXTURE_SAMPLE],
                selected=[TEST_SCHEMA],
                proofs=[],
                assumptions=["fixture_binds_schema_tests"],
                uncertainty=["indirect_fixture_consumers"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Fixture dependency oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-12-pytest-ini-markers",
            category="test_repair",
            objective=(
                "Repair pytest.ini marker/config drift after configuration "
                "mutation."
            ),
            targets=["pytest.ini"],
            mutation="pytest_configuration",
            risk="medium",
            route="small_local_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_PYTEST_CFG],
                selected=[TEST_CORE_ADD],
                proofs=[],
                assumptions=["config_binds_all_receipts"],
                uncertainty=["plugin_defaults"],
                candidate_verification_outcome="pass",
                fallback="full_pytest",
            ),
            candidate_notes="Pytest configuration oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-13-test-core-expectation",
            category="test_repair",
            objective=(
                "Update core test expectations to match a body-only mutation "
                "without rewriting production callers."
            ),
            targets=["tests/test_core.py"],
            mutation="local_function_body",
            risk="low",
            route="small_local_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_CORE_ADD],
                selected=[TEST_CORE_ADD],
                proofs=[PROOF_CORE_ADD],
                assumptions=["test_is_source_of_expectation"],
                uncertainty=["golden_vs_property_tests"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Test-expectation repair oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-14-test-api-signature-assert",
            category="test_repair",
            objective=(
                "Repair API tests after public signature change on fetch_value."
            ),
            targets=["tests/test_api.py", "src/sch_fixture/api.py"],
            mutation="public_signature",
            risk="medium",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_API_FETCH, SYM_ADAPTER_CLIENT],
                selected=[TEST_API_FETCH, TEST_ADAPTER],
                proofs=[PROOF_API_SIG],
                assumptions=["tests_cover_new_kwargs"],
                uncertainty=["downstream_clients_unknown"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="API test repair oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-15-test-schema-roundtrip",
            category="test_repair",
            objective=(
                "Repair schema roundtrip tests after UserRecord dataclass "
                "migration."
            ),
            targets=["tests/test_schema.py", "src/sch_fixture/schema.py"],
            mutation="dataclass_schema",
            risk="medium",
            route="small_local_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_SCHEMA_USER, SYM_SCHEMA_DUMP],
                selected=[TEST_SCHEMA],
                proofs=[PROOF_SCHEMA],
                assumptions=["asdict_shape_is_authoritative"],
                uncertainty=["storage_adapters_out_of_tree"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Schema test repair oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-16-test-security-assert",
            category="test_repair",
            objective=(
                "Repair security authorization tests after exception-behavior "
                "mutation."
            ),
            targets=["tests/test_security.py"],
            mutation="exception_behavior",
            risk="medium",
            route="small_local_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_SEC_AUTHORIZE],
                selected=[TEST_SECURITY],
                proofs=[PROOF_SECURITY],
                assumptions=["deny_paths_tested"],
                uncertainty=["role_matrix_coverage"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Security test repair oracle candidate.",
        )
    )

    # ------------------------------------------------------------------
    # api_adapter (6)
    # ------------------------------------------------------------------
    tasks.append(
        _task(
            task_id="sch-bench-17-api-fetch-signature",
            category="api_adapter",
            objective=(
                "Adapt public fetch_value signature and bound adapters to the "
                "new contract."
            ),
            targets=["src/sch_fixture/api.py"],
            mutation="public_signature",
            risk="medium",
            route="medium_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_API_FETCH, SYM_ADAPTER_CLIENT],
                selected=[TEST_API_FETCH, TEST_ADAPTER],
                proofs=[PROOF_API_SIG],
                assumptions=["strict_kw_only_addition"],
                uncertainty=["external_sdk_clients"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Public signature adapter oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-18-cross-module-call-core",
            category="api_adapter",
            objective=(
                "Update cross-module call_core adapter after core call-edge "
                "mutation."
            ),
            targets=["src/sch_fixture/api.py", "src/sch_fixture/core.py"],
            mutation="cross_module_call",
            risk="medium",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_API_CALL_CORE, SYM_CORE_ADD],
                selected=[TEST_API_CALL, TEST_CORE_ADD],
                proofs=[PROOF_CORE_ADD],
                assumptions=["call_edge_is_static"],
                uncertainty=["indirect_imports"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Cross-module call adapter oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-19-mcp-client-adapter",
            category="api_adapter",
            objective=(
                "Adapt McpClientAdapter after MCP interface descriptor change."
            ),
            targets=[
                "src/sch_fixture/adapters.py",
                "interfaces/mcp_client.json",
            ],
            mutation="mcp_interface_client_adapter",
            risk="high",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_ADAPTER_CLIENT, SYM_IFACE],
                selected=[TEST_ADAPTER],
                proofs=[],
                assumptions=["interface_cid_binds_adapter"],
                uncertainty=["remote_server_compat"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="MCP client adapter oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-20-mcp-ping-contract",
            category="api_adapter",
            objective=(
                "Keep adapter ping contract aligned with interface descriptor "
                "operations list."
            ),
            targets=[
                "src/sch_fixture/adapters.py",
                "interfaces/mcp_client.json",
                "tests/test_adapters.py",
            ],
            mutation="mcp_interface_client_adapter",
            risk="medium",
            route="small_local_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_ADAPTER_CLIENT, SYM_IFACE],
                selected=[TEST_ADAPTER],
                proofs=[],
                assumptions=["ping_is_required_operation"],
                uncertainty=["additional_ops_optional"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="MCP ping contract oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-21-api-default-param",
            category="api_adapter",
            objective=(
                "Preserve default-parameter compatibility while admitting a "
                "signature extension on fetch_value."
            ),
            targets=["src/sch_fixture/api.py", "tests/test_api.py"],
            mutation="public_signature",
            risk="medium",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_API_FETCH, SYM_ADAPTER_CLIENT],
                selected=[TEST_API_FETCH, TEST_ADAPTER],
                proofs=[PROOF_API_SIG],
                assumptions=["defaults_preserve_old_callers"],
                uncertainty=["keyword-only_migration_risk"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Default-parameter adapter oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-22-adapter-endpoint-binding",
            category="api_adapter",
            objective=(
                "Rebind adapter endpoint construction after interface/client "
                "adapter mutation."
            ),
            targets=["src/sch_fixture/adapters.py"],
            mutation="mcp_interface_client_adapter",
            risk="medium",
            route="small_local_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_ADAPTER_CLIENT, SYM_IFACE],
                selected=[TEST_ADAPTER],
                proofs=[],
                assumptions=["endpoint_is_constructor_arg"],
                uncertainty=["env_specific_endpoints"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Adapter endpoint binding oracle candidate.",
        )
    )

    # ------------------------------------------------------------------
    # schema_migration (6)
    # ------------------------------------------------------------------
    tasks.append(
        _task(
            task_id="sch-bench-23-user-record-field-add",
            category="schema_migration",
            objective=(
                "Migrate UserRecord dataclass field set and dump_user shape."
            ),
            targets=["src/sch_fixture/schema.py"],
            mutation="dataclass_schema",
            risk="high",
            route="medium_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_SCHEMA_USER, SYM_SCHEMA_DUMP],
                selected=[TEST_SCHEMA],
                proofs=[PROOF_SCHEMA],
                assumptions=["frozen_dataclass_migration"],
                uncertainty=["persisted_records_out_of_tree"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Dataclass field migration oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-24-user-record-dump-shape",
            category="schema_migration",
            objective=(
                "Align serialization dump_user mapping with migrated UserRecord."
            ),
            targets=["src/sch_fixture/schema.py", "tests/test_schema.py"],
            mutation="dataclass_schema",
            risk="medium",
            route="small_local_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_SCHEMA_USER, SYM_SCHEMA_DUMP],
                selected=[TEST_SCHEMA],
                proofs=[PROOF_SCHEMA],
                assumptions=["asdict_keys_sorted_stable"],
                uncertainty=["consumers_of_extra_keys"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Dump-shape migration oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-25-lockfile-pytest-pin",
            category="schema_migration",
            objective=(
                "Migrate dependency lockfile pin and invalidate dependent "
                "verification receipts."
            ),
            targets=["requirements.lock", "requirements.txt"],
            mutation="dependency_lockfile",
            risk="high",
            route="deterministic_only",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_LOCK],
                selected=[TEST_CORE_ADD],
                proofs=[],
                assumptions=["lock_cid_binds_receipts"],
                uncertainty=["transitive_hashes"],
                candidate_verification_outcome="pass",
                fallback="full_pytest",
            ),
            candidate_notes="Lockfile migration oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-26-policy-admission-mode",
            category="schema_migration",
            objective=(
                "Migrate admission policy document while keeping simulation "
                "disabled."
            ),
            targets=["policy/admission.json"],
            mutation="policy_change",
            risk="critical",
            route="medium_model",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_POLICY],
                selected=[TEST_SECURITY],
                proofs=[PROOF_SECURITY],
                assumptions=["policy_cid_change_invalidates_decisions"],
                uncertainty=["operator_override_paths"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Policy admission migration oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-27-requirements-txt-pin",
            category="schema_migration",
            objective=(
                "Keep requirements.txt synchronized with lockfile migration."
            ),
            targets=["requirements.txt", "requirements.lock"],
            mutation="dependency_lockfile",
            risk="medium",
            route="deterministic_only",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_LOCK],
                selected=[TEST_CORE_ADD],
                proofs=[],
                assumptions=["txt_and_lock_coherent"],
                uncertainty=["hash_lines_optional_in_txt"],
                candidate_verification_outcome="pass",
                fallback="full_pytest",
            ),
            candidate_notes="Requirements pin migration oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-28-schema-fixture-co-migration",
            category="schema_migration",
            objective=(
                "Co-migrate schema fields with sample_record fixture consumers."
            ),
            targets=[
                "src/sch_fixture/schema.py",
                "tests/conftest.py",
                "tests/test_schema.py",
            ],
            mutation="dataclass_schema",
            risk="high",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_SCHEMA_USER, SYM_SCHEMA_DUMP, SYM_FIXTURE_SAMPLE],
                selected=[TEST_SCHEMA],
                proofs=[PROOF_SCHEMA],
                assumptions=["fixture_matches_schema"],
                uncertainty=["serialized_historical_rows"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Schema+fixture co-migration oracle candidate.",
        )
    )

    # ------------------------------------------------------------------
    # multi_file_refactor (6)
    # ------------------------------------------------------------------
    tasks.append(
        _task(
            task_id="sch-bench-29-rename-process-symbol",
            category="multi_file_refactor",
            objective=(
                "Rename process helper across definition and call sites without "
                "losing delete/rename evidence."
            ),
            targets=[
                "src/sch_fixture/core.py",
                "src/sch_fixture/api.py",
                "tests/test_core.py",
            ],
            mutation="renamed_symbol",
            risk="high",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_CORE_PROCESS, SYM_RENAMED, SYM_API_FETCH],
                selected=[TEST_CORE_PROCESS, TEST_API_FETCH],
                proofs=[],
                assumptions=["rename_preserves_semantics"],
                uncertainty=["stringized_imports"],
                candidate_verification_outcome="pass",
                fallback="full_pytest",
            ),
            candidate_notes="Rename refactor oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-30-split-core-api-edge",
            category="multi_file_refactor",
            objective=(
                "Refactor cross-module call edge between core and api as a "
                "coordinated multi-file change."
            ),
            targets=[
                "src/sch_fixture/core.py",
                "src/sch_fixture/api.py",
                "tests/test_api.py",
            ],
            mutation="cross_module_call",
            risk="medium",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_API_CALL_CORE, SYM_CORE_ADD],
                selected=[TEST_API_CALL, TEST_CORE_ADD],
                proofs=[PROOF_CORE_ADD],
                assumptions=["edge_refactor_keeps_tests"],
                uncertainty=["transitive_callers"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Cross-module multi-file refactor oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-31-adapter-interface-split",
            category="multi_file_refactor",
            objective=(
                "Split interface descriptor updates from client adapter "
                "implementation across files."
            ),
            targets=[
                "interfaces/mcp_client.json",
                "src/sch_fixture/adapters.py",
                "tests/test_adapters.py",
            ],
            mutation="mcp_interface_client_adapter",
            risk="high",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_IFACE, SYM_ADAPTER_CLIENT],
                selected=[TEST_ADAPTER],
                proofs=[],
                assumptions=["descriptor_before_adapter"],
                uncertainty=["wire_compat_window"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Interface/adapter multi-file refactor oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-32-schema-test-co-refactor",
            category="multi_file_refactor",
            objective=(
                "Refactor schema module and bound tests/fixtures together."
            ),
            targets=[
                "src/sch_fixture/schema.py",
                "tests/test_schema.py",
                "tests/conftest.py",
            ],
            mutation="dataclass_schema",
            risk="high",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_SCHEMA_USER, SYM_SCHEMA_DUMP, SYM_FIXTURE_SAMPLE],
                selected=[TEST_SCHEMA],
                proofs=[PROOF_SCHEMA],
                assumptions=["tests_and_schema_move_together"],
                uncertainty=["external_serializers"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Schema multi-file refactor oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-33-security-api-co-refactor",
            category="multi_file_refactor",
            objective=(
                "Coordinate security policy surface with API consumers across "
                "modules."
            ),
            targets=[
                "src/sch_fixture/security.py",
                "src/sch_fixture/api.py",
                "tests/test_security.py",
            ],
            mutation="side_effect_security",
            risk="critical",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_SEC_AUTHORIZE],
                selected=[TEST_SECURITY],
                proofs=[PROOF_SECURITY],
                assumptions=["security_first_then_api"],
                uncertainty=["effect_lattice_incomplete"],
                candidate_verification_outcome="pass",
            ),
            candidate_notes="Security/API multi-file refactor oracle candidate.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-34-delete-and-rename-wave",
            category="multi_file_refactor",
            objective=(
                "Apply a delete/rename wave across core and tests while "
                "preserving selection recall."
            ),
            targets=[
                "src/sch_fixture/core.py",
                "tests/test_core.py",
                "src/sch_fixture/api.py",
            ],
            mutation="deleted_symbol",
            risk="high",
            route="medium_model",
            multi_file=True,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_DELETED],
                selected=[TEST_CORE_PROCESS],
                proofs=[],
                assumptions=["delete_before_rename_consumers"],
                uncertainty=["string_refs_to_legacy_helper"],
                candidate_verification_outcome="pass",
                fallback="full_pytest",
            ),
            candidate_notes="Delete/rename multi-file refactor oracle candidate.",
        )
    )

    # ------------------------------------------------------------------
    # rejection_or_escalation (6) — includes frontier/human cases
    # ------------------------------------------------------------------
    tasks.append(
        _task(
            task_id="sch-bench-35-stale-receipt-reject",
            category="rejection_or_escalation",
            objective=(
                "Reject work that would admit a stale verification receipt "
                "against a newer tree binding."
            ),
            targets=["src/sch_fixture/core.py"],
            mutation="stale_receipt",
            risk="critical",
            route="deterministic_only",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_CORE_ADD],
                selected=[TEST_CORE_ADD],
                proofs=[PROOF_CORE_ADD],
                assumptions=["receipts_bind_tree_and_config"],
                uncertainty=["clock_skew_not_trusted"],
                candidate_verification_outcome="reject",
                production_acceptance="rejected",
            ),
            candidate_notes=(
                "Stale-receipt scenario must reject; never production-accept."
            ),
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-36-out-of-scope-patch-reject",
            category="rejection_or_escalation",
            objective=(
                "Reject an out-of-scope model patch that touches undeclared "
                "paths."
            ),
            targets=["src/sch_fixture/core.py"],
            mutation="out_of_scope_patch",
            risk="critical",
            route="deterministic_only",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_CORE_ADD],
                selected=[TEST_CORE_ADD],
                proofs=[],
                assumptions=["immutable_scope_enforced"],
                uncertainty=["path_normalization_edge_cases"],
                candidate_verification_outcome="reject",
                production_acceptance="rejected",
            ),
            candidate_notes="Out-of-scope patch must reject without production accept.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-37-opaque-native-frontier",
            category="rejection_or_escalation",
            objective=(
                "Escalate opaque native_bridge behavior to a frontier model "
                "with mandatory raw-source inclusion."
            ),
            targets=["src/sch_fixture/native_bridge.py", "tests/test_native.py"],
            mutation="opaque_native",
            risk="high",
            route="frontier_model",
            multi_file=True,
            frontier_or_human=True,
            oracle=_oracle(
                invalidation=[SYM_NATIVE_BRIDGE],
                selected=[TEST_NATIVE],
                proofs=[PROOF_OPAQUE],
                assumptions=["raw_source_required_for_opaque"],
                uncertainty=["native_semantics_unknown"],
                candidate_verification_outcome="escalate",
                production_acceptance="blocked",
                fallback="both",
            ),
            candidate_notes=(
                "Opaque native requires frontier escalation; not production-eligible."
            ),
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-38-post-scan-source-race",
            category="rejection_or_escalation",
            objective=(
                "Reject packs that would admit post-scan source-race marker "
                "bytes into target context."
            ),
            targets=["src/sch_fixture/core.py"],
            mutation="post_scan_source_race",
            risk="critical",
            route="deterministic_only",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_CORE_ADD],
                selected=[TEST_CORE_ADD],
                proofs=[PROOF_CORE_ADD],
                assumptions=["scan_tree_not_live_fs"],
                uncertainty=["watcher_race_window"],
                candidate_verification_outcome="reject",
                production_acceptance="rejected",
            ),
            candidate_notes=(
                "Source-race marker must never enter packs; reject candidate."
            ),
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-39-failed-cas-reject",
            category="rejection_or_escalation",
            objective=(
                "Reject failed ABA/CAS root transitions without advancing "
                "production state."
            ),
            targets=["src/sch_fixture/core.py"],
            mutation="failed_aba_cas",
            risk="critical",
            route="deterministic_only",
            multi_file=False,
            frontier_or_human=False,
            oracle=_oracle(
                invalidation=[SYM_CORE_ADD],
                selected=[TEST_CORE_ADD],
                proofs=[],
                assumptions=["expected_old_root_required"],
                uncertainty=["concurrent_writers"],
                candidate_verification_outcome="reject",
                production_acceptance="rejected",
            ),
            candidate_notes="Failed CAS must reject; production root unchanged.",
        )
    )
    tasks.append(
        _task(
            task_id="sch-bench-40-concurrent-human-review",
            category="rejection_or_escalation",
            objective=(
                "Require human review when concurrent watchers/writers fence "
                "the harness transition."
            ),
            targets=["src/sch_fixture/core.py", "src/sch_fixture/api.py"],
            mutation="concurrent_watchers_writers",
            risk="critical",
            route="human_review_required",
            multi_file=True,
            frontier_or_human=True,
            oracle=_oracle(
                invalidation=[SYM_CORE_ADD, SYM_API_FETCH],
                selected=[TEST_CORE_ADD, TEST_API_FETCH],
                proofs=[],
                assumptions=["single_winner_cas"],
                uncertainty=["interleaving_order"],
                candidate_verification_outcome="escalate",
                production_acceptance="blocked",
                fallback="full_pytest",
            ),
            candidate_notes=(
                "Concurrent fence requires human review; not production-eligible."
            ),
        )
    )

    tasks.sort(key=lambda item: item.task_id)
    return tuple(tasks)
