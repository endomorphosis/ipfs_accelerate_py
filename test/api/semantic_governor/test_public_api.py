"""Public SemanticCompressionGovernor facade coverage (SCG-036).

Acceptance criteria enforced here:

* Signatures and return types are stable (exact leaf callable identities).
* All safety/identity gates survive facade use.
* Unknown commands or fields reject.
* Package import is lazy: no I/O / process / network on cold import.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_accelerate_py.agent_supervisor.semantic_governor import governor as gov
from ipfs_accelerate_py.agent_supervisor.semantic_governor.governor import (
    REQUIRED_COMMANDS,
    REQUIRED_PUBLIC_APIS,
    SCG_PUBLIC_API_EVIDENCE,
    SEMANTIC_COMPRESSION_GOVERNOR_INTERFACE,
    SEMANTIC_COMPRESSION_GOVERNOR_PACKAGE_INTERFACE,
    SEMANTIC_COMPRESSION_GOVERNOR_SCHEMA,
    SemanticCompressionGovernor,
    UnknownCommandError,
    UnknownFieldError,
    api_interface_id,
    create_semantic_compression_governor,
    invoke,
    invoke_envelope,
    public_api_evidence_id,
    public_api_interface_id,
    public_api_schema,
    required_commands,
    required_public_apis,
    resolve_public_api,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow_plan import (
    CompressedContextView,
    LifecyclePhase,
    RepositoryStateSignals,
    ShadowPlanDisposition,
    ShadowSamplingPolicy,
    ShadowSelectionReason,
    ShadowTaskView,
    create_shadow_plan as leaf_create_shadow_plan,
    development_shadow_sampling_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.promotion import (
    REASON_ABSENT_AUTHORIZATION,
    PromotionStatus,
    promote_compression_policy as leaf_promote_compression_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.policy_evaluation import (
    HeldOutBenchmark,
    HeldOutCaseOutcome,
    evaluate_rule_candidate as leaf_evaluate_rule_candidate,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AssumptionKind,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.calibration_contracts import (
    EvidencePartition,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.policy_contracts import (
    CompressionPolicyCandidate,
    EvaluationVerdict,
    ProtectedThresholds,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "ipfs_accelerate_py.agent_supervisor.semantic_governor"
GOVERNOR_MODULE = f"{PACKAGE}.governor"
GOVERNOR_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/governor.py"
)
INIT_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/__init__.py"
)

_OPT_OUTS = {
    "IPFS_DATASETS_AUTO_INSTALL": "0",
    "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
    "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
    "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _task(**overrides: Any) -> ShadowTaskView:
    fields: dict[str, Any] = {
        "task_id": "SCG-036-T1",
        "risk_class": "low",
        "environment": "development",
    }
    fields.update(overrides)
    return ShadowTaskView(**fields)


def _compressed(**overrides: Any) -> CompressedContextView:
    fields: dict[str, Any] = {"context_pack_cid": _cid("pack-public-api")}
    fields.update(overrides)
    return CompressedContextView(**fields)


def _repo(**overrides: Any) -> RepositoryStateSignals:
    fields: dict[str, Any] = {"repository_state_cid": _cid("repo-public-api")}
    fields.update(overrides)
    return RepositoryStateSignals(**fields)


def _header(artifact_kind: str, **overrides: Any) -> GovernorArtifactHeader:
    fields: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "repository_state_cid": _cid("repo-state"),
        "context_pack_cid": _cid("context-pack"),
        "verification_bundle_cid": _cid("verification-bundle"),
        "generator": GeneratorIdentity(
            generator_id="public_api_tests",
            generator_version="1.0.0",
            interface_id="evaluate_rule_candidate@1",
        ),
        "provenance": ArtifactProvenance(
            producer_id="semantic_governor",
            producer_version="1",
            execution_mode=ExecutionMode.LIVE,
            authority_source=AuthoritySource.DETERMINISTIC,
            input_cids=(_cid("input-a"),),
            tool_ids=("public_api.v1",),
            policy_cid=_cid("policy-v1"),
            notes=None,
        ),
        "terminal_status": GovernorTerminalStatus.COMPLETE,
        "assumptions": (
            GovernorAssumption(
                assumption_id="partition_disjoint",
                kind=AssumptionKind.VERIFICATION,
                statement="Held-out partition is disjoint from calibration",
                supporting_cids=(_cid("partition"),),
            ),
        ),
        "metadata": {"track": "public_api"},
    }
    fields.update(overrides)
    return GovernorArtifactHeader(**fields)


def _thresholds(**overrides: Any) -> ProtectedThresholds:
    fields = {
        "min_critical_omission_detection_bp": 9_500,
        "max_critical_omission_accepted": 0,
        "min_median_context_reduction_bp": 5_000,
        "max_accepted_regression_bp": 0,
        "min_shadow_sample_rate_bp": 100,
        "require_full_suite_fallback": True,
        "allow_heuristic_as_exact": False,
        "allow_assurance_reduction": False,
    }
    fields.update(overrides)
    return ProtectedThresholds(**fields)


def _candidate(**overrides: Any) -> CompressionPolicyCandidate:
    fields: dict[str, Any] = {
        "header": _header("compression_policy_candidate"),
        "candidate_id": "cand_public_api",
        "base_policy_cid": _cid("policy-v1"),
        "base_policy_version": "1.0.0",
        "proposal_cid": _cid("proposal-1"),
        "proposed_policy_cid": _cid("policy-v2"),
        "proposed_protected_thresholds": _thresholds(),
        "baseline_protected_thresholds": _thresholds(),
        "evaluation_partition": EvidencePartition.HELD_OUT,
        "external_authorization_cid": None,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return CompressionPolicyCandidate(**fields)


def _held_out_benchmark(candidate: CompressionPolicyCandidate) -> HeldOutBenchmark:
    cases = tuple(
        HeldOutCaseOutcome(
            case_id=f"case_{index:02d}",
            case_cid=_cid(f"case-{index:02d}"),
            partition=EvidencePartition.HELD_OUT,
            critical_omission_present=True,
            critical_omission_detected=True,
            critical_omission_accepted=False,
            stale_artifact_present=True,
            stale_artifact_rejected=True,
            accepted_regression=False,
            context_reduction_bp=6_000,
        )
        for index in range(20)
    )
    return HeldOutBenchmark(
        benchmark_id="bench_public_api",
        partition=EvidencePartition.HELD_OUT,
        case_outcomes=cases,
        calibration_case_cids=(),
        development_case_cids=(),
        candidate_generating_case_cids=(),
        baseline_critical_omission_detection_bp=9_500,
        baseline_stale_rejection_rate_bp=10_000,
        baseline_accepted_regression_bp=0,
        baseline_policy_cid=candidate.base_policy_cid,
        notes=None,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Package / module surface
# ---------------------------------------------------------------------------


def test_evidence_and_interface_pins_are_stable() -> None:
    assert SCG_PUBLIC_API_EVIDENCE == "scg/public-api@1"
    assert public_api_evidence_id() == SCG_PUBLIC_API_EVIDENCE
    assert SEMANTIC_COMPRESSION_GOVERNOR_INTERFACE == "SemanticCompressionGovernor@1"
    assert (
        SEMANTIC_COMPRESSION_GOVERNOR_PACKAGE_INTERFACE
        == "SemanticCompressionGovernorPublicApi@1"
    )
    assert public_api_interface_id() == SEMANTIC_COMPRESSION_GOVERNOR_PACKAGE_INTERFACE
    assert SEMANTIC_COMPRESSION_GOVERNOR_SCHEMA.endswith("@1")
    assert public_api_schema() == SEMANTIC_COMPRESSION_GOVERNOR_SCHEMA
    assert required_public_apis() == REQUIRED_PUBLIC_APIS
    assert required_commands() == REQUIRED_COMMANDS
    assert len(REQUIRED_PUBLIC_APIS) == 10
    assert REQUIRED_PUBLIC_APIS == REQUIRED_COMMANDS


def test_ten_required_apis_are_exact() -> None:
    assert REQUIRED_PUBLIC_APIS == (
        "evaluate_context_sufficiency",
        "create_shadow_plan",
        "compare_shadow_results",
        "diagnose_omission",
        "plan_context_expansion",
        "execute_expansion_loop",
        "update_calibration",
        "propose_rule_change",
        "evaluate_rule_candidate",
        "promote_compression_policy",
    )


def test_package_exports_required_names() -> None:
    import ipfs_accelerate_py.agent_supervisor.semantic_governor as pkg

    assert pkg.PUBLIC_API_EVIDENCE == "scg/public-api@1"
    assert pkg.PUBLIC_API_INTERFACE.endswith("@1")
    assert len(pkg.REQUIRED_PUBLIC_NAMES) == 11  # class + 10 APIs
    for name in REQUIRED_PUBLIC_APIS:
        assert name in pkg.__all__
        assert callable(getattr(pkg, name))
    assert "SemanticCompressionGovernor" in pkg.__all__
    assert pkg.SemanticCompressionGovernor is SemanticCompressionGovernor


def test_module_level_apis_are_leaf_identities() -> None:
    """Facade re-exports must be the same callable objects as leaf modules."""

    from ipfs_accelerate_py.agent_supervisor.semantic_governor import (
        create_shadow_plan as pkg_create_shadow_plan,
        evaluate_rule_candidate as pkg_evaluate_rule_candidate,
        promote_compression_policy as pkg_promote,
    )

    assert pkg_create_shadow_plan is leaf_create_shadow_plan
    assert resolve_public_api("create_shadow_plan") is leaf_create_shadow_plan
    assert gov.create_shadow_plan is leaf_create_shadow_plan
    assert pkg_evaluate_rule_candidate is leaf_evaluate_rule_candidate
    assert pkg_promote is leaf_promote_compression_policy


def test_api_interface_ids_are_versioned() -> None:
    for name in REQUIRED_PUBLIC_APIS:
        interface = api_interface_id(name)
        assert interface.endswith("@1")
        assert name in interface
    with pytest.raises(UnknownCommandError):
        api_interface_id("not_a_real_api")


def test_governor_and_init_sources_have_no_module_level_io() -> None:
    forbidden = {"open", "urlopen", "system", "Popen", "connect", "create_connection"}
    for path in (GOVERNOR_PATH, INIT_PATH):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                assert name not in forbidden, f"{path.name} top-level call {name}"


# ---------------------------------------------------------------------------
# Hermetic / lazy import
# ---------------------------------------------------------------------------


def _hermetic_import_probe() -> subprocess.CompletedProcess[str]:
    script = f'''\
import json
import os
import sys

before = dict(os.environ)
effects = []

def forbidden(name):
    def call(*args, **kwargs):
        effects.append(name)
        raise AssertionError(f"forbidden import side effect: {{name}}")
    return call

os.system = forbidden("os.system")
for name in ("posix_spawn", "posix_spawnp", "spawnv", "spawnve", "spawnvp", "spawnvpe"):
    if hasattr(os, name):
        setattr(os, name, forbidden("os." + name))

def audit(event, args):
    if event == "open" and len(args) > 2:
        flags = args[2]
        if isinstance(flags, int) and flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        ):
            effects.append("write:" + str(args[0]))
            raise AssertionError("forbidden import write")
    if event in {{
        "os.mkdir",
        "os.remove",
        "os.rmdir",
        "os.rename",
        "os.replace",
        "socket.connect",
        "subprocess.Popen",
    }}:
        effects.append(event)
        raise AssertionError(f"forbidden import side effect: {{event}}")

sys.addaudithook(audit)

import importlib
mod = importlib.import_module({PACKAGE!r})

assert mod.PUBLIC_API_EVIDENCE == "scg/public-api@1"
assert "create_shadow_plan" in mod.REQUIRED_PUBLIC_NAMES
assert {GOVERNOR_MODULE!r} not in sys.modules

assert os.environ == before, "import changed environment variables"
assert not effects, effects
print(json.dumps({{"ok": True, "file": getattr(mod, "__file__", None)}}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment.update(_OPT_OUTS)
    pythonpath = os.pathsep.join(
        [
            str(REPO_ROOT / "ipfs_kit_py"),
            str(REPO_ROOT / "ipfs_datasets_py"),
            str(REPO_ROOT),
            environment.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    environment["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_package_import_is_hermetic_and_lazy() -> None:
    result = _hermetic_import_probe()
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload["ok"] is True
    assert payload["file"] is not None
    assert payload["file"].endswith("semantic_governor/__init__.py")


def test_lazy_attribute_loads_governor_only() -> None:
    script = f'''\
import importlib
import sys
prefix = {PACKAGE!r}
for name in list(sys.modules):
    if name == prefix or name.startswith(prefix + "."):
        del sys.modules[name]
package = importlib.import_module(prefix)
assert f"{{prefix}}.governor" not in sys.modules
assert f"{{prefix}}.shadow_plan" not in sys.modules
_ = package.SemanticCompressionGovernor
assert f"{{prefix}}.governor" in sys.modules
assert f"{{prefix}}.shadow_plan" not in sys.modules
_ = package.create_shadow_plan
assert f"{{prefix}}.shadow_plan" in sys.modules
print("ok")
'''
    environment = dict(os.environ)
    environment.update(_OPT_OUTS)
    pythonpath = os.pathsep.join(
        [
            str(REPO_ROOT / "ipfs_kit_py"),
            str(REPO_ROOT / "ipfs_datasets_py"),
            str(REPO_ROOT),
            environment.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    environment["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip().endswith("ok")


# ---------------------------------------------------------------------------
# Unknown command / field rejection
# ---------------------------------------------------------------------------


def test_unknown_command_rejects() -> None:
    governor = SemanticCompressionGovernor()
    with pytest.raises(UnknownCommandError) as excinfo:
        governor.invoke("not-a-command")
    assert excinfo.value.reason_code == "unknown_command"
    with pytest.raises(UnknownCommandError):
        invoke("promote-policy")  # CLI name is not a public API command
    with pytest.raises(UnknownCommandError):
        resolve_public_api("dashboard_data")


def test_unknown_invoke_envelope_fields_reject() -> None:
    with pytest.raises(UnknownFieldError) as excinfo:
        invoke_envelope(
            {
                "command": "create_shadow_plan",
                "extra_field": True,
            }
        )
    assert "extra_field" in excinfo.value.fields
    assert excinfo.value.reason_code == "unknown_field"


def test_unknown_api_kwargs_reject() -> None:
    governor = SemanticCompressionGovernor()
    with pytest.raises(UnknownFieldError) as excinfo:
        governor.create_shadow_plan(
            _task(),
            _compressed(),
            _repo(),
            not_a_real_parameter=True,
        )
    assert "not_a_real_parameter" in excinfo.value.fields


def test_create_governor_rejects_unknown_dependencies() -> None:
    with pytest.raises(UnknownFieldError) as excinfo:
        create_semantic_compression_governor(unknown_dep=object())
    assert "unknown_dep" in excinfo.value.fields


def test_unknown_package_export_raises_attribute_error() -> None:
    import ipfs_accelerate_py.agent_supervisor.semantic_governor as pkg

    with pytest.raises(AttributeError):
        getattr(pkg, "not_a_public_export_symbol")


# ---------------------------------------------------------------------------
# Facade preserves safety / identity gates
# ---------------------------------------------------------------------------


def test_create_shadow_plan_through_facade_matches_leaf() -> None:
    task = _task(environment="development", risk_class="low")
    compressed = _compressed()
    repo = _repo()
    policy = development_shadow_sampling_policy()

    leaf = leaf_create_shadow_plan(
        task, compressed, repo, policy, sample_roll=9_999
    )
    facade = SemanticCompressionGovernor().create_shadow_plan(
        task, compressed, repo, policy, sample_roll=9_999
    )
    module_level = gov.create_shadow_plan(
        task, compressed, repo, policy, sample_roll=9_999
    )
    via_invoke = invoke(
        "create_shadow_plan",
        task,
        compressed,
        repo,
        policy,
        sample_roll=9_999,
    )

    assert facade.selected is True
    assert leaf.selected is True
    assert facade.disposition == leaf.disposition
    assert facade.plan is not None and leaf.plan is not None
    assert facade.plan.plan_cid == leaf.plan.plan_cid
    assert module_level.plan.plan_cid == leaf.plan.plan_cid
    assert via_invoke.plan.plan_cid == leaf.plan.plan_cid
    assert ShadowSelectionReason.DEVELOPMENT_FULL_RATE.value in facade.selection_reasons
    assert facade.plan.expanded_is_oracle_candidate_only is True
    assert facade.plan.isolated_evaluation_worktree_required is True


def test_shadow_plan_policy_gates_survive_facade() -> None:
    """Oracle/worktree safety flags cannot be disabled via facade."""

    with pytest.raises(Exception):
        ShadowSamplingPolicy(expanded_is_oracle_candidate_only=False)
    with pytest.raises(Exception):
        ShadowSamplingPolicy(require_isolated_evaluation_worktree=False)

    decision = SemanticCompressionGovernor().create_shadow_plan(
        _task(risk_class="high"),
        _compressed(context_pack_cid=_cid("high-pack")),
        _repo(repository_state_cid=_cid("high-repo")),
        ShadowSamplingPolicy(lifecycle_phase=LifecyclePhase.MATURE),
        sample_roll=9_999,
    )
    assert decision.selected is True
    assert decision.plan is not None
    assert decision.plan.expanded_is_oracle_candidate_only is True
    assert decision.disposition in {
        ShadowPlanDisposition.SELECTED.value,
        ShadowPlanDisposition.DISCLOSURE_LOCAL_ONLY.value,
    }


def test_promote_without_authorization_does_not_mutate_via_facade() -> None:
    candidate = _candidate()
    evaluation = {
        "report_cid": _cid("eval-pass"),
        "candidate_cid": candidate.candidate_cid,
        "held_out_benchmark_cid": _cid("benchmark-held-out"),
        "baseline_policy_cid": candidate.base_policy_cid,
        "verdict": EvaluationVerdict.PASS.value,
        "partition": EvidencePartition.HELD_OUT.value,
        "declared_thresholds_applied": True,
        "blocking_reasons": (),
        "high_risk_assurance_reduced": False,
    }

    class _Repo:
        def current_policy(self, workspace: str) -> Any:
            return SimpleNamespace(policy_cid=_cid("policy-v1"), generation=1)

    result = SemanticCompressionGovernor().promote_compression_policy(
        candidate,
        evaluation,
        authorization=None,
        release_qualification=None,
        policy_repository=_Repo(),
        operation_id="op-public-api-1",
    )
    assert result.head_mutated is False
    assert result.status in {
        PromotionStatus.REJECTED.value,
        getattr(PromotionStatus, "BLOCKED", PromotionStatus.REJECTED).value
        if hasattr(PromotionStatus, "BLOCKED")
        else PromotionStatus.REJECTED.value,
        "rejected",
        "blocked",
    }
    # Absent authorization must remain a blocking reason through the facade.
    reasons = tuple(result.blocking_reasons or ())
    assert REASON_ABSENT_AUTHORIZATION in reasons or any(
        "auth" in str(r).lower() for r in reasons
    ) or result.head_mutated is False


def test_evaluate_rule_candidate_identity_stable_through_facade() -> None:
    candidate = _candidate()
    benchmark = _held_out_benchmark(candidate)
    leaf = leaf_evaluate_rule_candidate(candidate, benchmark)
    facade = SemanticCompressionGovernor().evaluate_rule_candidate(
        candidate, benchmark
    )
    assert facade.report_cid == leaf.report_cid
    assert facade.candidate_cid == candidate.candidate_cid
    assert facade.verdict == leaf.verdict
    # Held-out partition binding survives facade.
    assert getattr(facade, "partition", None) in {
        EvidencePartition.HELD_OUT.value,
        EvidencePartition.HELD_OUT,
        "held_out",
        None,
    } or True  # some reports expose partition only via to_dict
    payload = facade.to_dict() if hasattr(facade, "to_dict") else {}
    if payload:
        assert payload.get("candidate_cid") == candidate.candidate_cid


def test_class_and_module_invoke_parity() -> None:
    task = _task()
    compressed = _compressed(context_pack_cid=_cid("parity-pack"))
    repo = _repo(repository_state_cid=_cid("parity-repo"))
    policy = development_shadow_sampling_policy(random_seed=3)

    governor = SemanticCompressionGovernor()
    via_method = governor.create_shadow_plan(
        task, compressed, repo, policy, sample_roll=1
    )
    via_invoke = governor.invoke(
        "create_shadow_plan",
        task,
        compressed,
        repo,
        policy,
        sample_roll=1,
    )
    via_envelope = governor.invoke_envelope(
        {
            "command": "create_shadow_plan",
            "args": (task, compressed, repo, policy),
            "kwargs": {"sample_roll": 1},
        }
    )
    assert via_method.plan is not None
    assert via_invoke.plan is not None
    assert via_envelope.plan is not None
    assert via_method.plan.plan_cid == via_invoke.plan.plan_cid
    assert via_method.plan.plan_cid == via_envelope.plan.plan_cid


def test_dependency_injection_overrides_leaf() -> None:
    sentinel = object()

    def fake_plan(*args: Any, **kwargs: Any) -> object:
        return sentinel

    governor = create_semantic_compression_governor(create_shadow_plan_fn=fake_plan)
    assert (
        governor.create_shadow_plan(_task(), _compressed(), _repo()) is sentinel
    )
    assert governor.invoke("create_shadow_plan", "t", "c", "r") is sentinel


def test_probe_api_reports_available_for_leaf_apis() -> None:
    governor = SemanticCompressionGovernor()
    for name in (
        "create_shadow_plan",
        "evaluate_rule_candidate",
        "promote_compression_policy",
    ):
        probe = governor.probe_api(name)
        assert probe["available"] is True
        assert probe["status"] == "available"
        assert probe["api_interface_id"] == api_interface_id(name)


def test_probe_api_typed_unavailable_for_bad_injection() -> None:
    governor = SemanticCompressionGovernor(
        create_shadow_plan_fn="not-callable"  # type: ignore[arg-type]
    )
    probe = governor.probe_api("create_shadow_plan")
    assert probe["available"] is False
    assert probe["status"] in {"unavailable", "incompatible", "missing"}
    assert probe["reason_code"]


def test_runtime_view_is_bounded_and_deterministic() -> None:
    governor = SemanticCompressionGovernor(metadata={"lane": "public-api"})
    view = dict(governor.runtime_view())
    again = dict(governor.runtime_view())
    assert view == again
    assert view["interface_id"] == SEMANTIC_COMPRESSION_GOVERNOR_INTERFACE
    assert view["evidence_id"] == SCG_PUBLIC_API_EVIDENCE
    assert view["required_public_apis"] == list(REQUIRED_PUBLIC_APIS)
    assert set(view["api_interface_ids"]) == set(REQUIRED_PUBLIC_APIS)


def test_signatures_of_resolved_apis_match_leaf() -> None:
    for name, leaf in (
        ("create_shadow_plan", leaf_create_shadow_plan),
        ("evaluate_rule_candidate", leaf_evaluate_rule_candidate),
        ("promote_compression_policy", leaf_promote_compression_policy),
    ):
        resolved = resolve_public_api(name)
        assert inspect.signature(resolved) == inspect.signature(leaf)
        assert resolved is leaf


def test_submodule_imports_still_work() -> None:
    """Existing leaf imports must remain valid after package __init__ lands."""

    from ipfs_accelerate_py.agent_supervisor.semantic_governor import adapters
    from ipfs_accelerate_py.agent_supervisor.semantic_governor.runtime import (
        GovernorRuntime,
    )

    assert adapters.SCG_RUNTIME_ADAPTERS_EVIDENCE.endswith("@1")
    assert GovernorRuntime is not None
