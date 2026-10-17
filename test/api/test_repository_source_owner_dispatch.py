"""Authored dispatch controls; no admission, source service, provider or worker.

The real bounded argument helper is used. Protected configuration loading and
the lifecycle constructor are explicit seams; neither is deployment evidence.
"""
import copy
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_dispatch as dispatch


@pytest.fixture
def case(monkeypatch):
    configuration = {"schema": "repository-source-owner-resources@1",
        "scheduler_state_path": "/opt/owner/state/resource-scheduler.json",
        "scheduler_policy_sha256": "a" * 64, "admission_timeout_seconds": 7.5,
        "resource_scope": "repository_source_service_only"}
    loaded, created = [], []
    lifecycle = SimpleNamespace(start=lambda: pytest.fail("configure invoked native startup"))
    def load(boundary):
        loaded.append(boundary)
        return copy.deepcopy(configuration)
    def construct(**kwargs):
        created.append(kwargs)
        return lifecycle
    monkeypatch.setattr(dispatch, "load_source_owner_configuration", load)
    monkeypatch.setattr(dispatch, "_new_lifecycle", construct)
    deadline = time.monotonic() + 90
    def configure(argv, **kwargs):
        return dispatch.configure_repository_source_owner_dispatch(argv=argv,
            boundary=Path("/opt/owner/container-boundary.json"),
            workspace=Path("/opt/owner/worktrees/task"),
            deadline_monotonic=kwargs.pop("deadline_monotonic", deadline), **kwargs)
    return SimpleNamespace(configuration=configuration, loaded=loaded, created=created,
        lifecycle=lifecycle, deadline=deadline, configure=configure)


def selected(policy="repository-source-compact@1"):
    return ["--repository-source-policy", policy, "--semantic-repository", "/app"]


@pytest.mark.parametrize("argv", [[], ["--repository-source-policy", "legacy"],
    ["--repository-source-policy=legacy"], ["--provider", "grok_cli", "--preflight"],
    ["--purpose", "planning", "--semantic-decision-policy-artifact", "/owned/policy.json"],
    ["--provider", "a", "--provider", "b", "--source-view-profile", "identity@1"],
])
def test_legacy_and_unrelated_routes_do_not_access_boundary_or_lifecycle(case, argv):
    assert case.configure(argv, deadline_monotonic=None) is None
    assert case.loaded == case.created == []


@pytest.mark.parametrize("policy,representation", [
    ("repository-source-raw@1", "original"), ("repository-source-compact@1", "compact")])
@pytest.mark.parametrize("equals", [False, True])
def test_selected_route_only_constructs_exact_unstarted_owner(case, policy, representation, equals):
    argv = selected(policy)
    if equals:
        argv = ["--repository-source-policy=" + policy, "--semantic-repository=/app"]
    before = list(argv)
    assert case.configure(argv) is case.lifecycle
    assert argv == before
    assert case.loaded == [Path("/opt/owner/container-boundary.json")]
    assert case.created == [{"repository": Path("/app"),
        "workspace": Path("/opt/owner/worktrees/task"),
        "boundary": Path("/opt/owner/container-boundary.json"),
        "scheduler_state_path": case.configuration["scheduler_state_path"],
        "expected_policy_sha256": case.configuration["scheduler_policy_sha256"],
        "admission_timeout_seconds": 7.5, "deadline_monotonic": case.deadline,
        "representation": representation}]
    assert "parent_lease" not in case.created[0] and "cancel_event" not in case.created[0]


def test_explicit_compatible_controls_preserve_unrelated_worker_arguments(case):
    argv = selected() + ["--provider=codex_cli", "--purpose=coding",
        "--source-view-advisory-policy=legacy", "--source-view-profile=disabled",
        "--source-edit-mode=disabled", "--model", "authored-model", "--timeout", "90",
        "--source-input-policy", "source-input-compact@1", "--query-reference-policy", "legacy",
        "--semantic-decision-max-calls", "3", "--preflight-sleep", "0",
        "--prompt-tokenizer-artifact", "/public/tokenizer.json", "--prompt-tokenizer-sha256", "b" * 64,
        "--source-formalization-handoff", "/public/formalization.json",
        "--source-formalization-handoff-sha256", "c" * 64,
        "--numerical-obligation-descriptor", "/public/numerical.json",
        "--numerical-obligation-descriptor-sha256", "d" * 64]
    original = list(argv)
    assert case.configure(argv) is case.lifecycle and argv == original


@pytest.mark.parametrize("argv", [None, (), "--repository-source-policy=legacy", [1],
    [Path("/app")], ["bad\0arg"], ["bad\ud800arg"], ["x" * 8193],
    ["x"] * 257, ["x" * 8192] * 9])
def test_invalid_or_oversized_argv_is_rejected_without_loading(case, argv):
    with pytest.raises(ValueError):
        case.configure(argv)
    assert case.loaded == case.created == []


@pytest.mark.parametrize("argv", [
    ["--repository-source-policy"], ["--repository-source-policy="],
    ["--repository-source-policy", "--model", "example"],
    ["--repository-source-policy", "unknown"],
    ["--re=legacy"], ["--repository-source-p", "repository-source-raw@1"],
    ["--repository-source-policy=repository-source-raw@1", "--repository-source-policy=repository-source-raw@1"],
    ["--repository-source-policy=repository-source-raw@1", "--repository-source-policy=legacy"],
    ["--", "--repository-source-policy=legacy"],
])
def test_policy_ambiguity_and_abbreviation_refuse_even_without_active_source_route(case, argv):
    with pytest.raises(ValueError):
        case.configure(argv)
    assert case.loaded == case.created == []


@pytest.mark.parametrize("flag", ["--repository-source-policy"[:end]
                                  for end in range(len("--rep"), len("--repository-source-policy") + 1)])
@pytest.mark.parametrize("equals", [False, True])
def test_every_previously_unique_legacy_policy_prefix_is_a_noop(case, flag, equals):
    argv = [flag + "=legacy"] if equals else [flag, "legacy"]
    original = list(argv)
    assert case.configure(argv, deadline_monotonic=None) is None
    assert argv == original and case.loaded == case.created == []


@pytest.mark.parametrize("argv", [
    ["--repository-source-policy=legacy", "--repository-source-policy=legacy"],
    ["--rep", "legacy", "--repository-source", "legacy", "--repo=legacy"],
    ["--repository-source-p=legacy", "--repository-source-policy", "legacy", "--rep=legacy"],
])
def test_repeated_legacy_policy_occurrences_do_not_enter_owner_configuration(case, argv):
    assert case.configure(argv, deadline_monotonic=None) is None
    assert case.loaded == case.created == []


@pytest.mark.parametrize("flag", ["--rep", "--repository-source", "--repository-source-p",
                                 "--repository-source-policy"])
@pytest.mark.parametrize("legacy_first", [False, True])
def test_legacy_aliases_cannot_hide_any_active_policy_occurrence(case, flag, legacy_first):
    active = [flag + "=repository-source-raw@1"]
    legacy = ["--repo", "legacy"]
    with pytest.raises(ValueError):
        case.configure((legacy + active if legacy_first else active + legacy)
                       + ["--semantic-repository", "/app"])
    assert case.loaded == case.created == []


@pytest.mark.parametrize("suffix", [[], [""], ["--model", "example"], ["foreign"]])
def test_malformed_legacy_prefix_is_not_an_omitted_selection(case, suffix):
    with pytest.raises(ValueError):
        case.configure(["--repo", *suffix])
    assert case.loaded == case.created == []


@pytest.mark.parametrize("private", ["--repository-source-owner-descriptor", "--resource-grant",
    "--repository-source-owner-d", "--repository-source-parent-lease", "--parent-t"])
def test_repeated_legacy_aliases_never_exempt_actual_private_options(case, private):
    with pytest.raises(ValueError):
        case.configure(["--repo=legacy", "--rep", "legacy", private + "=legacy"])
    assert case.loaded == case.created == []


@pytest.mark.parametrize("flag", sorted(dispatch._PRIVATE))
@pytest.mark.parametrize("enabled", [False, True])
def test_caller_cannot_supply_owner_descriptors_or_private_capacity(case, flag, enabled):
    argv = (selected() if enabled else []) + [flag + "=/caller/capability"]
    with pytest.raises(ValueError):
        case.configure(argv)
    assert case.loaded == case.created == []


@pytest.mark.parametrize("flag", ["--repository-source-owner-d", "--resource-g",
    "--repository-source-parent", "--scheduler-state", "--parent-t", "--lease-k"])
def test_private_capacity_abbreviations_are_refused(case, flag):
    with pytest.raises(ValueError):
        case.configure([flag, "candidate"])
    assert case.loaded == case.created == []


@pytest.mark.parametrize("flag", ["--prov", "--pur", "--semantic-rep", "--source-view-a",
    "--source-view-p", "--source-edit-m", "--prefl", "--semantic-metadata-census",
    "--doctor-candidate-a", "--finite-proof-query-c", "--semantic-decision-policy-a"])
def test_selected_relevant_abbreviations_refuse_before_loading(case, flag):
    with pytest.raises(ValueError):
        case.configure(selected() + [flag, "authored"])
    assert case.loaded == case.created == []


@pytest.mark.parametrize("flag,value", [("--provider", "codex_cli"), ("--purpose", "coding"),
    ("--semantic-repository", "/app"), ("--source-view-advisory-policy", "legacy"),
    ("--source-view-profile", "disabled"), ("--source-edit-mode", "disabled")])
def test_selected_relevant_duplicates_refuse_before_loading(case, flag, value):
    argv = selected() + [flag + "=" + value]
    if flag != "--semantic-repository":
        argv += [flag, value]
    with pytest.raises(ValueError):
        case.configure(argv)
    assert case.loaded == case.created == []


@pytest.mark.parametrize("flag", sorted(dispatch._OTHER_ROUTES))
def test_other_selected_routes_refuse_before_loading(case, flag):
    # Presence is sufficient; this adapter need not parse a competing route.
    with pytest.raises(ValueError):
        case.configure(selected() + [flag])
    assert case.loaded == case.created == []


@pytest.mark.parametrize("flag,value", [("--provider", "grok_cli"), ("--provider", ""),
    ("--purpose", "planning"), ("--purpose", ""),
    ("--source-view-advisory-policy", "source-view-compact@1"),
    ("--source-view-profile", "identity@1"), ("--source-edit-mode", "mapped-span@1")])
def test_incompatible_coding_controls_refuse(case, flag, value):
    with pytest.raises(ValueError):
        case.configure(selected() + [flag + "=" + value])
    assert case.loaded == case.created == []


@pytest.mark.parametrize("repository", [None, "", "relative", "/", "//app", "/app/", "/app/./code",
                                      "/app/../other", "/app/bad\\name", "/app/bad\nname"])
def test_repository_must_be_explicit_canonical_absolute(case, repository):
    argv = selected()[:2]
    if repository is not None:
        argv += ["--semantic-repository", repository]
    with pytest.raises(ValueError):
        case.configure(argv)
    assert case.loaded == case.created == []


@pytest.mark.parametrize("deadline", [None, True, "90", 0, -1, float("nan"),
                                      float("inf"), float("-inf"), 10 ** 500])
def test_invalid_inherited_deadline_refuses_before_configuration_load(case, deadline):
    with pytest.raises(ValueError):
        case.configure(selected(), deadline_monotonic=deadline)
    assert case.loaded == case.created == []


def test_far_future_deadline_cannot_expand_native_lifetime(case):
    with pytest.raises(ValueError):
        case.configure(selected(), deadline_monotonic=time.monotonic() + 3600)
    assert case.loaded == case.created == []


def test_missing_configuration_is_not_an_environment_or_default_fallback(case, monkeypatch):
    monkeypatch.setenv("IPFS_DATASETS_RESOURCE_SCHEDULER_PATH", "/untrusted/default.json")
    monkeypatch.setenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", "local-benchmark@1")
    monkeypatch.setattr(dispatch, "load_source_owner_configuration", lambda _: None)
    with pytest.raises(ValueError, match="explicit deployed"):
        case.configure(selected())
    assert case.created == []


def test_loader_refusal_is_preserved_without_construction_or_retry(case, monkeypatch):
    attempts = []
    def unavailable(boundary):
        attempts.append(boundary)
        raise ValueError("authored protected boundary refusal")
    monkeypatch.setattr(dispatch, "load_source_owner_configuration", unavailable)
    with pytest.raises(ValueError, match="protected boundary refusal"):
        case.configure(selected())
    assert len(attempts) == 1 and case.created == []


def test_constructor_refusal_is_preserved_without_start_or_retry(case, monkeypatch):
    attempts = []
    def refuse(**kwargs):
        attempts.append(kwargs)
        raise ValueError("authored constructor refusal")
    monkeypatch.setattr(dispatch, "_new_lifecycle", refuse)
    with pytest.raises(ValueError, match="constructor refusal"):
        case.configure(selected())
    assert len(attempts) == 1 and len(case.loaded) == 1


def test_argument_adapter_delegates_to_existing_exact_helper(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_broker as numerical
    calls = []
    def observe(argv, name, default=None):
        calls.append((argv, name, default))
        return "authored-value"
    monkeypatch.setattr(numerical, "_argument", observe)
    argv = ["--purpose", "coding"]
    assert dispatch._argument(argv, "--purpose", "coding") == "authored-value"
    assert calls == [(argv, "--purpose", "coding")]
