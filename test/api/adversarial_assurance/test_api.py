"""Public adversarial-assurance campaign API coverage (AAE-048).

Acceptance criteria enforced here:

* Every required API has stable typed inputs/outputs.
* Exact canonical leaf bindings for re-exported APIs.
* Safe errors (unknown command/field, missing surfaces).
* No arbitrary host path exposure through public inputs.
* End-to-end contract coverage for analyze_vacuity and
  execute_mutation_campaign, plus facade dispatch.
* Package import is lazy: no I/O / process / network on cold import.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    ArtifactProvenance,
    AssuranceArtifactHeader,
    AssuranceTerminalStatus,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    VersionBinding,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.analysis_contracts import (
    MinimizedEvidenceBinding,
    SourceSpan,
    VacuityFamily,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.mutation_contracts import (
    CampaignBudget,
    MutationCampaignPlan,
    MutationRiskClass,
    SeedConfigBinding,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.vacuity_formal_policy import (
    FormalProofVacuitySubject,
    analyze_formal_vacuity,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance import api as aae_api
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.api import (
    AAE_PUBLIC_API_EVIDENCE,
    ADVERSARIAL_ASSURANCE_PUBLIC_API_INTERFACE,
    ADVERSARIAL_ASSURANCE_PUBLIC_API_SCHEMA,
    ANALYZE_VACUITY_INTERFACE,
    ASSURANCE_CAMPAIGN_API_INTERFACE,
    EXECUTE_MUTATION_CAMPAIGN_INTERFACE,
    REQUIRED_COMMANDS,
    REQUIRED_PUBLIC_APIS,
    AssuranceCampaignApi,
    AssurancePublicApiError,
    MutationCampaignExecutionResult,
    PathExposureError,
    UnknownCommandError,
    UnknownFieldError,
    VacuityCampaignAnalysisResult,
    analyze_vacuity,
    api_interface_id,
    create_assurance_campaign_api,
    execute_mutation_campaign,
    invoke,
    invoke_envelope,
    public_api_evidence_id,
    public_api_interface_id,
    public_api_schema,
    required_commands,
    required_public_apis,
    resolve_public_api,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.execution import (
    classify_mutation_outcome as leaf_classify_mutation_outcome,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.manifest import (
    create_assurance_manifest as leaf_create_assurance_manifest,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.planning import (
    plan_mutation_campaign as leaf_plan_mutation_campaign,
    predict_detection_set as leaf_predict_detection_set,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.promotion import (
    promote_assurance_policy as leaf_promote_assurance_policy,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.remediation import (
    evaluate_remediation as leaf_evaluate_remediation,
    propose_gap_remediation as leaf_propose_gap_remediation,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.diagnosis import (
    diagnose_surviving_mutant as leaf_diagnose_surviving_mutant,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.execution import (
    execute_mutation as leaf_execute_mutation,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.planning import (
    generate_mutation_candidates as leaf_generate_mutation_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "ipfs_accelerate_py.agent_supervisor.adversarial_assurance"
API_MODULE = f"{PACKAGE}.api"
API_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/api.py"
)
INIT_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/__init__.py"
)

_OPT_OUTS = {
    "IPFS_DATASETS_AUTO_INSTALL": "0",
    "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
    "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
    "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}

REPO_ID = "repository:sha256:aae048-public-api-test"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _generator(**overrides: object) -> GeneratorIdentity:
    fields = {
        "generator_id": "public_api_tests",
        "generator_version": "1.0.0",
        "interface_id": "public_api_tests@1",
    }
    fields.update(overrides)
    return GeneratorIdentity(**fields)  # type: ignore[arg-type]


def _versions(**overrides: object) -> VersionBinding:
    fields = {
        "operator_id": "control_flow_invert",
        "operator_version": "1",
        "campaign_policy_id": "default_campaign",
        "campaign_policy_version": "1.0.0",
        "generator": _generator(),
    }
    fields.update(overrides)
    return VersionBinding(**fields)  # type: ignore[arg-type]


def _provenance(**overrides: object) -> ArtifactProvenance:
    fields = {
        "producer_id": "adversarial_assurance",
        "producer_version": "1",
        "execution_mode": ExecutionMode.LIVE,
        "authority_source": AuthoritySource.DETERMINISTIC,
        "input_cids": (_cid("input-a"),),
        "tool_ids": ("public_api.v1",),
        "policy_cid": _cid("policy"),
        "notes": None,
    }
    fields.update(overrides)
    return ArtifactProvenance(**fields)  # type: ignore[arg-type]


def _header(artifact_kind: str = "vacuity_finding", **overrides: object) -> AssuranceArtifactHeader:
    fields = {
        "artifact_kind": artifact_kind,
        "repository_id": REPO_ID,
        "repository_state_cid": _cid("repo-state"),
        "target_symbol_ids": ("mod.fn",),
        "target_artifact_cids": (_cid("artifact-a"),),
        "capsule_cids": (_cid("capsule-a"),),
        "proof_unit_cids": (_cid("proof-unit-a"),),
        "environment_cid": _cid("environment"),
        "dependency_lock_cid": _cid("dependency-lock"),
        "versions": _versions(),
        "provenance": _provenance(),
        "terminal_status": AssuranceTerminalStatus.COMPLETE,
        "receipt_cids": (_cid("receipt-a"),),
        "proof_cids": (_cid("proof-a"),),
        "metadata": {"track": "public_api"},
    }
    fields.update(overrides)
    return AssuranceArtifactHeader(**fields)  # type: ignore[arg-type]


def _span(**overrides: object) -> SourceSpan:
    fields = {
        "path": "proofs/authz.lean",
        "start_line": 10,
        "end_line": 40,
        "start_col": 0,
        "end_col": 80,
    }
    fields.update(overrides)
    return SourceSpan(**fields)  # type: ignore[arg-type]


def _evidence(**overrides: object) -> MinimizedEvidenceBinding:
    fields = {
        "evidence_cids": (_cid("min-evidence-1"),),
        "minimized": True,
        "minimization_failed": False,
        "reproduction_input_cid": None,
        "notes": None,
    }
    fields.update(overrides)
    return MinimizedEvidenceBinding(**fields)  # type: ignore[arg-type]


def _formal_subject(**overrides: object) -> FormalProofVacuitySubject:
    fields = {
        "subject_id": "proof.authz_guard",
        "claimed_property": "authorization guard rejects unauthorized callers",
        "symbol_ids": ("mod.fn", "proof.authz_guard"),
        "source_spans": (_span(),),
        "dependency_path": ("mod.fn", "proof.authz_guard"),
        "minimized_evidence": _evidence(),
        "proposition": "forall caller, authorized(caller) -> admit(caller)",
        "antecedent": "caller has valid capability token",
        "antecedent_satisfiable": True,
        "modeled_state_ids": ("state.authorized", "state.denied"),
        "reachable_state_ids": ("state.authorized", "state.denied"),
        "discharge_possible": True,
        "result_constrained": True,
        "unconstrained_result_ids": (),
        "required_behavior_ids": ("behavior.reject_unauth", "behavior.admit_auth"),
        "modeled_behavior_ids": ("behavior.reject_unauth", "behavior.admit_auth"),
        "assumed_ids": ("asm.token_wellformed",),
        "proven_ids": ("asm.token_wellformed", "lemma.capability_sound"),
        "assumptions_used_as_proven": (),
        "declared_nonclaims": (
            "does not prove hardware root of trust",
            "does not prove side-channel absence",
        ),
        "subject_cid": _cid("formal-subject"),
        "observation_complete": True,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return FormalProofVacuitySubject(**fields)  # type: ignore[arg-type]


def _seed_config(**overrides: object) -> SeedConfigBinding:
    fields = {
        "seed": 42,
        "config": {"max_depth": 2, "operator_budget": 4, "mode": "bounded"},
    }
    fields.update(overrides)
    return SeedConfigBinding(**fields)  # type: ignore[arg-type]


def _budget(**overrides: object) -> CampaignBudget:
    fields = {
        "max_total_candidates": 64,
        "max_candidates_per_target": 8,
        "max_candidates_per_operator": 16,
        "max_targets": 32,
        "max_operators": 16,
        "max_execution_seconds": 3_600,
        "max_worktrees": 8,
    }
    fields.update(overrides)
    return CampaignBudget(**fields)  # type: ignore[arg-type]


def _campaign_plan(**overrides: object) -> MutationCampaignPlan:
    fields = {
        "header": _header("mutation_campaign_plan"),
        "plan_id": "plan_public_api",
        "policy_id": "default_campaign",
        "policy_version": "1.0.0",
        "policy_cid": _cid("campaign-policy"),
        "repository_id": REPO_ID,
        "repository_state_cid": _cid("repo-state"),
        "baseline_receipt_cid": _cid("baseline-receipt"),
        "seed_config": _seed_config(),
        "budget": _budget(),
        "target_cids": (_cid("target-a"),),
        "operator_cids": (_cid("operator-a"),),
        "candidate_cids": (_cid("candidate-a"), _cid("candidate-b")),
        "admitted_risk_classes": (
            MutationRiskClass.LOCAL_BUG,
            MutationRiskClass.AUTHORIZATION,
        ),
        "require_sandbox": True,
        "require_rollback": True,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return MutationCampaignPlan(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Package / module surface
# ---------------------------------------------------------------------------


def test_evidence_and_interface_pins_are_stable() -> None:
    assert AAE_PUBLIC_API_EVIDENCE == "aae/public-api@1"
    assert public_api_evidence_id() == AAE_PUBLIC_API_EVIDENCE
    assert ASSURANCE_CAMPAIGN_API_INTERFACE == "AssuranceCampaignApi@1"
    assert (
        ADVERSARIAL_ASSURANCE_PUBLIC_API_INTERFACE
        == "AdversarialAssurancePublicApi@1"
    )
    assert public_api_interface_id() == ADVERSARIAL_ASSURANCE_PUBLIC_API_INTERFACE
    assert ADVERSARIAL_ASSURANCE_PUBLIC_API_SCHEMA.endswith("@1")
    assert public_api_schema() == ADVERSARIAL_ASSURANCE_PUBLIC_API_SCHEMA
    assert required_public_apis() == REQUIRED_PUBLIC_APIS
    assert required_commands() == REQUIRED_COMMANDS
    assert len(REQUIRED_PUBLIC_APIS) == 12
    assert REQUIRED_PUBLIC_APIS == REQUIRED_COMMANDS


def test_twelve_required_apis_are_exact() -> None:
    assert REQUIRED_PUBLIC_APIS == (
        "create_assurance_manifest",
        "generate_mutation_candidates",
        "predict_detection_set",
        "execute_mutation",
        "classify_mutation_outcome",
        "diagnose_surviving_mutant",
        "analyze_vacuity",
        "propose_gap_remediation",
        "evaluate_remediation",
        "promote_assurance_policy",
        "plan_mutation_campaign",
        "execute_mutation_campaign",
    )


def test_package_exports_required_names() -> None:
    import ipfs_accelerate_py.agent_supervisor.adversarial_assurance as pkg

    assert pkg.PUBLIC_API_EVIDENCE == "aae/public-api@1"
    assert pkg.PUBLIC_API_INTERFACE.endswith("@1")
    assert len(pkg.REQUIRED_PUBLIC_NAMES) == 13  # class + 12 APIs
    for name in REQUIRED_PUBLIC_APIS:
        assert name in pkg.__all__
        assert callable(getattr(pkg, name))
    assert "AssuranceCampaignApi" in pkg.__all__
    assert pkg.AssuranceCampaignApi is AssuranceCampaignApi


def test_module_level_apis_are_leaf_identities() -> None:
    """Facade re-exports must be the same callable objects as leaf modules."""

    import ipfs_accelerate_py.agent_supervisor.adversarial_assurance as pkg

    assert pkg.create_assurance_manifest is leaf_create_assurance_manifest
    assert resolve_public_api("create_assurance_manifest") is leaf_create_assurance_manifest
    assert aae_api.create_assurance_manifest is leaf_create_assurance_manifest
    assert resolve_public_api("plan_mutation_campaign") is leaf_plan_mutation_campaign
    assert resolve_public_api("predict_detection_set") is leaf_predict_detection_set
    assert resolve_public_api("generate_mutation_candidates") is leaf_generate_mutation_candidates
    assert resolve_public_api("execute_mutation") is leaf_execute_mutation
    assert resolve_public_api("classify_mutation_outcome") is leaf_classify_mutation_outcome
    assert resolve_public_api("diagnose_surviving_mutant") is leaf_diagnose_surviving_mutant
    assert resolve_public_api("propose_gap_remediation") is leaf_propose_gap_remediation
    assert resolve_public_api("evaluate_remediation") is leaf_evaluate_remediation
    assert resolve_public_api("promote_assurance_policy") is leaf_promote_assurance_policy
    # Local composers are owned by api.py
    assert resolve_public_api("analyze_vacuity") is analyze_vacuity
    assert resolve_public_api("execute_mutation_campaign") is execute_mutation_campaign
    assert pkg.analyze_vacuity is analyze_vacuity
    assert pkg.execute_mutation_campaign is execute_mutation_campaign


def test_api_interface_ids_are_versioned() -> None:
    for name in REQUIRED_PUBLIC_APIS:
        interface = api_interface_id(name)
        assert interface.endswith("@1")
        assert name in interface
    with pytest.raises(UnknownCommandError):
        api_interface_id("not_a_real_api")


def test_api_and_init_sources_have_no_module_level_io() -> None:
    forbidden = {"open", "urlopen", "system", "Popen", "connect", "create_connection"}
    for path in (API_PATH, INIT_PATH):
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

assert mod.PUBLIC_API_EVIDENCE == "aae/public-api@1"
assert "create_assurance_manifest" in mod.REQUIRED_PUBLIC_NAMES
assert {API_MODULE!r} not in sys.modules

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
    assert payload["file"].endswith("adversarial_assurance/__init__.py")


def test_lazy_attribute_loads_api_only() -> None:
    script = f'''\
import importlib
import sys
prefix = {PACKAGE!r}
for name in list(sys.modules):
    if name == prefix or name.startswith(prefix + "."):
        del sys.modules[name]
package = importlib.import_module(prefix)
assert f"{{prefix}}.api" not in sys.modules
assert f"{{prefix}}.planning" not in sys.modules
_ = package.AssuranceCampaignApi
assert f"{{prefix}}.api" in sys.modules
assert f"{{prefix}}.planning" not in sys.modules
_ = package.plan_mutation_campaign
assert f"{{prefix}}.planning" in sys.modules
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
    api = AssuranceCampaignApi()
    with pytest.raises(UnknownCommandError) as excinfo:
        api.invoke("not-a-command")
    assert excinfo.value.reason_code == "unknown_command"
    with pytest.raises(UnknownCommandError):
        invoke("assurance-report")
    with pytest.raises(UnknownCommandError):
        resolve_public_api("dashboard_data")


def test_unknown_invoke_envelope_fields_reject() -> None:
    with pytest.raises(UnknownFieldError) as excinfo:
        invoke_envelope(
            {
                "command": "analyze_vacuity",
                "extra_field": True,
            }
        )
    assert "extra_field" in excinfo.value.fields
    assert excinfo.value.reason_code == "unknown_field"


def test_create_api_rejects_unknown_dependencies() -> None:
    with pytest.raises(UnknownFieldError) as excinfo:
        create_assurance_campaign_api(unknown_dep=object())
    assert "unknown_dep" in excinfo.value.fields


def test_unknown_package_export_raises_attribute_error() -> None:
    import ipfs_accelerate_py.agent_supervisor.adversarial_assurance as pkg

    with pytest.raises(AttributeError):
        _ = pkg.not_a_real_export  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Path exposure safety
# ---------------------------------------------------------------------------


def test_execute_mutation_campaign_rejects_absolute_host_paths() -> None:
    plan = _campaign_plan()
    with pytest.raises(PathExposureError) as excinfo:
        execute_mutation_campaign(
            plan,
            {"policy_cid": _cid("policy"), "worktree_path": "/home/secret/worktree"},
            precomputed_reports=[
                {
                    "candidate_id": "c1",
                    "candidate_cid": _cid("candidate-a"),
                    "terminal_status": "killed_by_test",
                    "outcome_cid": _cid("outcome-a"),
                }
            ],
        )
    assert excinfo.value.reason_code == "path_exposure"

    with pytest.raises(PathExposureError):
        execute_mutation_campaign(
            plan,
            {"policy_cid": _cid("policy")},
            precomputed_reports=[
                {
                    "candidate_id": "c1",
                    "candidate_cid": _cid("candidate-a"),
                    "terminal_status": "killed_by_test",
                    "host_path": "/tmp/leak",
                }
            ],
        )


def test_analyze_vacuity_rejects_absolute_repository_path() -> None:
    with pytest.raises(PathExposureError):
        analyze_vacuity(
            {"manifest_cid": _cid("manifest")},
            {
                "repository_state_cid": _cid("repo-state"),
                "repo_root": "/home/user/project",
            },
            formal_subject=_formal_subject(),
            header=_header(),
        )


def test_facade_rejects_absolute_path_kwargs() -> None:
    api = AssuranceCampaignApi()
    with pytest.raises(PathExposureError):
        api.analyze_vacuity(
            {"manifest_cid": _cid("manifest")},
            {"repository_state_cid": _cid("repo-state")},
            formal_subject=_formal_subject(),
            header=_header(),
            metadata={"local_path": "C:\\Users\\secret\\file.py"},
        )


# ---------------------------------------------------------------------------
# analyze_vacuity end-to-end
# ---------------------------------------------------------------------------


def test_analyze_vacuity_composes_formal_family() -> None:
    subject = _formal_subject(antecedent_satisfiable=False)
    result = analyze_vacuity(
        {"manifest_cid": _cid("manifest"), "repository_id": REPO_ID},
        {
            "repository_id": REPO_ID,
            "repository_state_cid": _cid("repo-state"),
        },
        formal_subject=subject,
        header=_header(),
    )
    assert isinstance(result, VacuityCampaignAnalysisResult)
    assert result.interface_id == ANALYZE_VACUITY_INTERFACE
    assert VacuityFamily.FORMAL_PROOF.value in result.families_analyzed
    assert result.production_policy_changed is False
    assert "no_production_policy_change" in result.reason_codes
    assert "no_arbitrary_path_exposure" in result.reason_codes
    assert result.findings  # unsatisfiable antecedent yields a finding
    assert result.result_cid
    # Deterministic identity
    again = analyze_vacuity(
        {"manifest_cid": _cid("manifest"), "repository_id": REPO_ID},
        {
            "repository_id": REPO_ID,
            "repository_state_cid": _cid("repo-state"),
        },
        formal_subject=subject,
        header=_header(),
    )
    assert again.result_cid == result.result_cid


def test_analyze_vacuity_requires_subjects() -> None:
    with pytest.raises(AssurancePublicApiError) as excinfo:
        analyze_vacuity(
            {"manifest_cid": _cid("manifest")},
            {"repository_state_cid": _cid("repo-state")},
        )
    assert excinfo.value.reason_code == "missing_vacuity_subjects"


def test_analyze_vacuity_via_subjects_sequence() -> None:
    subject = _formal_subject()
    result = analyze_vacuity(
        {"manifest_cid": _cid("manifest")},
        {"repository_state_cid": _cid("repo-state"), "repository_id": REPO_ID},
        subjects=[
            {
                "vacuity_family": VacuityFamily.FORMAL_PROOF.value,
                "subject": subject.to_dict(),
            }
        ],
        header=_header(),
    )
    assert result.families_analyzed == (VacuityFamily.FORMAL_PROOF.value,)
    # Same leaf analyzer produces findings subset
    leaf = analyze_formal_vacuity(subject, _header())
    assert len(result.findings) == len(leaf.findings)


def test_analyze_vacuity_via_facade_and_invoke() -> None:
    api = AssuranceCampaignApi()
    subject = _formal_subject()
    result = api.analyze_vacuity(
        {"manifest_cid": _cid("manifest")},
        {"repository_state_cid": _cid("repo-state"), "repository_id": REPO_ID},
        formal_subject=subject,
        header=_header(),
    )
    assert isinstance(result, VacuityCampaignAnalysisResult)

    envelope_result = invoke_envelope(
        {
            "command": "analyze_vacuity",
            "args": [
                {"manifest_cid": _cid("manifest")},
                {"repository_state_cid": _cid("repo-state"), "repository_id": REPO_ID},
            ],
            "kwargs": {
                "formal_subject": subject,
                "header": _header(),
            },
        }
    )
    assert envelope_result.result_cid == result.result_cid


# ---------------------------------------------------------------------------
# execute_mutation_campaign end-to-end
# ---------------------------------------------------------------------------


def test_execute_mutation_campaign_with_precomputed_reports() -> None:
    plan = _campaign_plan()
    reports = [
        {
            "candidate_id": "cand_a",
            "candidate_cid": _cid("candidate-a"),
            "terminal_status": "killed_by_test",
            "outcome_cid": _cid("outcome-a"),
            "report_cid": _cid("report-a"),
        },
        {
            "candidate_id": "cand_b",
            "candidate_cid": _cid("candidate-b"),
            "terminal_status": "survived_selected_verification",
            "outcome_cid": _cid("outcome-b"),
            "report_cid": _cid("report-b"),
        },
        {
            "candidate_id": "cand_c",
            "candidate_cid": _cid("candidate-c"),
            "terminal_status": "invalid_mutant",
            "outcome_cid": _cid("outcome-c"),
        },
    ]
    result = execute_mutation_campaign(
        plan,
        {"policy_cid": _cid("verification-policy")},
        precomputed_reports=reports,
    )
    assert isinstance(result, MutationCampaignExecutionResult)
    assert result.interface_id == EXECUTE_MUTATION_CAMPAIGN_INTERFACE
    assert result.plan_id == plan.plan_id
    assert result.plan_cid == plan.plan_cid
    assert result.killed_count == 1
    assert result.survivor_count == 1
    assert result.invalid_count == 1
    assert result.production_policy_changed is False
    assert result.require_sandbox is True
    assert result.network_disabled is True
    assert "no_production_policy_change" in result.reason_codes
    assert "disposable_worktree_required" in result.reason_codes
    assert "network_disabled" in result.reason_codes
    assert result.result_cid

    # Deterministic
    again = execute_mutation_campaign(
        plan,
        {"policy_cid": _cid("verification-policy")},
        precomputed_reports=reports,
    )
    assert again.result_cid == result.result_cid


def test_execute_mutation_campaign_with_injected_executor() -> None:
    plan = _campaign_plan()

    def _executor(
        *,
        candidate: Any,
        expected_detection: Any,
        plan: Any,
        verification_policy: Any,
        index: int,
    ) -> dict[str, Any]:
        cid = (
            candidate.get("candidate_cid")
            if isinstance(candidate, dict)
            else getattr(candidate, "candidate_cid", _cid(f"cand-{index}"))
        )
        return {
            "candidate_id": f"cand_{index}",
            "candidate_cid": cid,
            "terminal_status": "killed_by_static_analysis" if index == 0 else "inconclusive",
            "outcome_cid": _cid(f"outcome-{index}"),
        }

    result = execute_mutation_campaign(
        plan,
        {"policy_cid": _cid("verification-policy")},
        candidate_executor=_executor,
    )
    assert result.killed_count == 1
    assert result.inconclusive_count == 1
    assert len(result.candidate_reports) == 2
    assert "injected_candidate_executor" in result.reason_codes


def test_execute_mutation_campaign_fails_closed_without_execution_surface() -> None:
    plan = _campaign_plan()
    with pytest.raises(AssurancePublicApiError) as excinfo:
        execute_mutation_campaign(
            plan,
            {"policy_cid": _cid("verification-policy")},
        )
    assert excinfo.value.reason_code == "missing_execution_surface"


def test_execute_mutation_campaign_via_facade() -> None:
    api = AssuranceCampaignApi()
    plan = _campaign_plan()
    result = api.execute_mutation_campaign(
        plan,
        {"policy_cid": _cid("verification-policy")},
        precomputed_reports=[
            {
                "candidate_id": "c1",
                "candidate_cid": _cid("candidate-a"),
                "terminal_status": "killed_by_test",
                "outcome_cid": _cid("outcome-a"),
            }
        ],
    )
    assert result.killed_count == 1
    assert result.interface_id == EXECUTE_MUTATION_CAMPAIGN_INTERFACE


def test_execute_mutation_campaign_accepts_plan_mapping() -> None:
    plan = _campaign_plan()
    result = execute_mutation_campaign(
        plan.to_dict(),
        {"policy_cid": _cid("verification-policy")},
        precomputed_reports=[
            {
                "candidate_id": "c1",
                "candidate_cid": _cid("candidate-a"),
                "terminal_status": "killed_by_test",
                "outcome_cid": _cid("outcome-a"),
            }
        ],
    )
    assert result.plan_cid == plan.plan_cid


# ---------------------------------------------------------------------------
# Facade probes / descriptor / classify binding
# ---------------------------------------------------------------------------


def test_probe_api_reports_available_for_required_apis() -> None:
    api = AssuranceCampaignApi()
    for name in REQUIRED_PUBLIC_APIS:
        probe = api.probe_api(name)
        assert probe["available"] is True
        assert probe["status"] == "available"
        assert probe["api_interface_id"] == api_interface_id(name)


def test_descriptor_exposes_closed_surface() -> None:
    api = AssuranceCampaignApi()
    descriptor = api.descriptor()
    assert descriptor["evidence_id"] == AAE_PUBLIC_API_EVIDENCE
    assert descriptor["production_policy_change"] is False
    assert descriptor["path_exposure"] is False
    assert descriptor["required_public_apis"] == list(REQUIRED_PUBLIC_APIS)


def test_classify_mutation_outcome_through_facade_matches_leaf() -> None:
    from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.execution import (
        DetectorRunObservation,
        DetectorRunStatus,
    )
    from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
        DetectorKind,
    )

    obs = DetectorRunObservation(
        detector_id="test.unit.auth",
        detector_kind=DetectorKind.UNIT_TEST,
        status=DetectorRunStatus.DETECTED,
    )
    leaf = leaf_classify_mutation_outcome(
        predicted_detector_ids=("test.unit.auth",),
        selected_detector_ids=("test.unit.auth",),
        observations=(obs,),
    )
    facade = AssuranceCampaignApi().classify_mutation_outcome(
        predicted_detector_ids=("test.unit.auth",),
        selected_detector_ids=("test.unit.auth",),
        observations=(obs,),
    )
    assert facade.to_dict() == leaf.to_dict()


def test_injected_override_is_used() -> None:
    calls: list[str] = []

    def fake_analyze(manifest: Any, repository_state: Any, **kwargs: Any) -> str:
        calls.append("hit")
        return "overridden"

    api = create_assurance_campaign_api(analyze_vacuity_fn=fake_analyze)
    assert api.analyze_vacuity({}, {}) == "overridden"
    assert calls == ["hit"]
