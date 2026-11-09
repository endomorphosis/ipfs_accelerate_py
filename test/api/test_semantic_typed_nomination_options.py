"""Typed nomination mode reaches signed owner argv and retained source pins.

These tests exercise actual pure option/argument/source-inventory APIs. They do
not stage a nomination, build an archive, import Harbor fixtures, start a
provider, or assert benchmark success or token savings.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import (
    container_worker_deployment as deployment,
    full_supervisor_benchmark as benchmark,
    terminal_doctor_dispatch as doctor,
)
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_owner as owner

NOMINATION = {
    "semantic_decision_nomination_artifact": Path("/owner/explicit nomination.json"),
    "semantic_decision_nomination_sha256": "a" * 64,
}
PREBOUND = {
    "semantic_decision_policy_artifact": Path("/owner/prebound policy.json"),
    "semantic_decision_policy_sha256": "b" * 64,
}
BASE_OPTIONS = {
    "prompt_tokenizer_artifact": None,
    "prompt_tokenizer_sha256": None,
    "max_input_tokens": None,
    "semantic_slot_policy_revision": "legacy@1",
    "semantic_context_projection": False,
}
COMMAND = {
    "router": Path("/router"), "model": "pinned-model", "reasoning": "high",
    "timeout": 30, "semantic_repository": Path("/app"), "doctor": None,
}
MODE_FLAG = "--semantic-decision-typed-nomination"
OPT_OUT_FLAG = "--disable-semantic-typed-nomination"


@pytest.mark.parametrize("disabled,mode", [(False, "enabled"), (True, "disabled")])
def test_shared_options_separate_driver_optout_from_literal_owner_mode(disabled, mode):
    before = deepcopy(NOMINATION)
    driver = deployment.semantic_decision_arguments(
        **NOMINATION, disable_semantic_typed_nomination=disabled,
        semantic_decision_max_calls=2, semantic_decision_max_total_input_tokens=12345,
    )
    signed = deployment.semantic_decision_arguments(
        **NOMINATION, disable_semantic_typed_nomination=disabled, owner_binding=True,
        semantic_decision_max_calls=2, semantic_decision_max_total_input_tokens=12345,
    )
    pair = ["--semantic-decision-nomination-artifact", str(NOMINATION["semantic_decision_nomination_artifact"]),
            "--semantic-decision-nomination-sha256", NOMINATION["semantic_decision_nomination_sha256"]]
    budgets = ["--semantic-decision-max-calls", "2", "--semantic-decision-max-total-input-tokens", "12345"]
    assert driver == pair + ([OPT_OUT_FLAG] if disabled else []) + budgets
    assert signed == pair + [MODE_FLAG, mode] + budgets
    assert MODE_FLAG not in driver and OPT_OUT_FLAG not in signed
    assert NOMINATION == before


def test_ordinary_and_prebound_routes_keep_their_existing_argument_shape():
    assert deployment.semantic_decision_arguments() == []
    assert deployment.semantic_decision_arguments(**PREBOUND) == [
        "--semantic-decision-policy-artifact", str(PREBOUND["semantic_decision_policy_artifact"]),
        "--semantic-decision-policy-sha256", PREBOUND["semantic_decision_policy_sha256"],
        "--semantic-decision-max-calls", "3",
    ]
    assert benchmark._prompt_options("full") == BASE_OPTIONS
    assert benchmark._prompt_options("full", **PREBOUND) == {
        **BASE_OPTIONS, **{key: str(value) for key, value in PREBOUND.items()},
        "semantic_decision_max_calls": 3, "semantic_decision_max_total_input_tokens": None,
    }


@pytest.mark.parametrize("options", [
    {"disable_semantic_typed_nomination": True},
    {**PREBOUND, "disable_semantic_typed_nomination": True},
    {**NOMINATION, "disable_semantic_typed_nomination": 1},
])
def test_optout_requires_nomination_and_a_literal_boolean(options):
    with pytest.raises(ValueError):
        deployment.semantic_decision_arguments(**options)
    with pytest.raises(ValueError):
        benchmark._prompt_options("full", **options)


@pytest.mark.parametrize("disabled,mode", [(False, "enabled"), (True, "disabled")])
def test_signed_command_has_one_mode_and_one_independently_admitted_task(disabled, mode):
    default = doctor.implementation_argv(**COMMAND)
    signed = doctor.implementation_argv(
        **COMMAND, **NOMINATION, semantic_decision_task_id="admitted-task",
        disable_semantic_typed_nomination=disabled,
    )
    assert signed == default + [
        "--semantic-decision-nomination-artifact", str(NOMINATION["semantic_decision_nomination_artifact"]),
        "--semantic-decision-nomination-sha256", NOMINATION["semantic_decision_nomination_sha256"],
        MODE_FLAG, mode, "--semantic-decision-max-calls", "3",
        "--semantic-decision-task-id", "admitted-task",
    ]
    assert signed.count(MODE_FLAG) == signed.count("--semantic-decision-task-id") == 1
    assert OPT_OUT_FLAG not in signed
    declarations, child = owner._literal_arguments(signed[1:])
    assert declarations[MODE_FLAG] == mode
    assert declarations["--semantic-decision-task-id"] == "admitted-task"
    assert MODE_FLAG not in child and "--semantic-decision-task-id" not in child
    for flag, value in (("--model", "pinned-model"), ("--reasoning-effort", "high"),
                        ("--timeout", "30"), ("--max-output-tokens", "4096")):
        assert child[child.index(flag) + 1] == value


def test_signer_refuses_a_nomination_without_its_independent_task_key():
    with pytest.raises(ValueError, match="independently admitted task"):
        doctor.implementation_argv(**COMMAND, **NOMINATION)
    with pytest.raises(ValueError, match="requires an owner nomination"):
        doctor.implementation_argv(**COMMAND, semantic_decision_task_id="unbound-task")


def test_owner_literal_parser_rejects_duplicate_equals_and_unknown_mode():
    signed = doctor.implementation_argv(**COMMAND, **NOMINATION, semantic_decision_task_id="admitted-task")[1:]
    mode_at = signed.index(MODE_FLAG)
    invalid = [
        signed + [MODE_FLAG, "enabled"],
        signed[:mode_at] + [MODE_FLAG + "=enabled"] + signed[mode_at + 2:],
        signed[:mode_at + 1] + ["true"] + signed[mode_at + 2:],
    ]
    for arguments in invalid:
        with pytest.raises(ValueError):
            owner._literal_arguments(arguments)
    assert owner._literal_arguments(signed)[0][MODE_FLAG] == "enabled"


@pytest.mark.parametrize("disabled", [False, True])
def test_host_options_retain_explicit_mode_and_closed_nomination_shape(disabled):
    options = benchmark._prompt_options(
        "full", **NOMINATION, disable_semantic_typed_nomination=disabled,
    )
    expected = {
        **BASE_OPTIONS,
        "semantic_decision_nomination_artifact": str(NOMINATION["semantic_decision_nomination_artifact"]),
        "semantic_decision_nomination_sha256": NOMINATION["semantic_decision_nomination_sha256"],
        "semantic_decision_max_calls": 3, "semantic_decision_max_total_input_tokens": None,
        "disable_semantic_typed_nomination": disabled,
    }
    assert options == expected and len(options) == 10
    observation = benchmark._prompt_option_observation(
        {"arm": "full", "prompt_comparison_options": options}, expected,
    )
    assert observation["prompt_comparison_options_valid"] is True
    assert observation["prompt_comparison_config_unchanged"] is True
    assert observation["prompt_comparison_options"] == expected


def test_historical_closed_nomination_keys_are_validated_with_default_mode():
    options = benchmark._prompt_options("full", **NOMINATION)
    legacy = {key: value for key, value in options.items() if key != "disable_semantic_typed_nomination"}
    assert len(legacy) == 9
    observation = benchmark._prompt_option_observation(
        {"arm": "full", "prompt_comparison_options": legacy}, dict(legacy),
    )
    assert observation["prompt_comparison_options_valid"] is True
    assert observation["prompt_comparison_config_unchanged"] is True
    assert observation["prompt_comparison_options"] == options
    malformed = {**legacy, "unrecognized_policy_toggle": False}
    refused = benchmark._prompt_option_observation(
        {"arm": "full", "prompt_comparison_options": malformed}, dict(legacy),
    )
    assert refused["prompt_comparison_options_valid"] is True
    assert refused["prompt_comparison_config_unchanged"] is False


def test_retained_optout_change_is_detected_without_reclassifying_valid_configuration():
    enabled = benchmark._prompt_options("full", **NOMINATION)
    disabled = {**enabled, "disable_semantic_typed_nomination": True}
    observation = benchmark._prompt_option_observation(
        {"arm": "full", "prompt_comparison_options": enabled}, disabled,
    )
    assert observation["prompt_comparison_options_valid"] is True
    assert observation["prompt_comparison_config_unchanged"] is False
    assert observation["prompt_comparison_options"]["disable_semantic_typed_nomination"] is True


def test_declared_runtime_inventory_retains_typed_graph_and_lineage_dependencies():
    # This API returns declared paths; no directory scan, archive construction,
    # source reads or existence tests are needed to check the capability set.
    sources = benchmark._decision_runtime_sources()
    assert len(sources) == 36 and len(sources) == len(set(sources))
    assert benchmark.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + "terminal_context_evidence.py" in sources
    assert {benchmark.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + name for name in (
        "terminal_decision_policy.py", "terminal_decision_usage_projection.py",
        "full_supervisor_benchmark.py", "benchmark_controls.py",
    )} <= set(sources)
    required = {
        benchmark.DECISION_RUNTIME_ARCHIVE_PREFIX + name for name in (
            "semantic_typed_owner_policy.py", "semantic_typed_source_context.py",
            "semantic_typed_dependency_proposal.py", "semantic_typed_nomination_lineage.py",
            "semantic_selection_relation_audit.py", "semantic_selection_dependency_audit.py",
            "semantic_decision_owner.py", "semantic_decision_usage.py",
        )
    } | {"source/ipfs_accelerate_py/agent_supervisor/analysis/semantic_dependency_graph.py"}
    assert required <= set(sources)
    assert all(path.is_absolute() and path.name == Path(name).name for name, path in sources.items())
