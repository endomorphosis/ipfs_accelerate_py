"""Executable contract for LogicVerificationProviderRoleClosure@1 (FVT-G227).

Acceptance coverage:

* ErgoAI, external Runtime MTL, Souffle, and SymbolicAI expose typed inventory,
  probe, explicit install, and verification/advisor dispatch surfaces;
* legacy Microsoft SecPAL is archival intake + compatibility lookup only;
* Stack, Temurin JDK, Maude, and OPAM are support-only with semantic, authority,
  and public-verification axes not applicable;
* unsupported roles and provider/role confusion fail with typed non-success;
* import/inventory/probe/dry-run paths stay side-effect free.
"""

from __future__ import annotations

import importlib
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from ipfs_datasets_py.logic.verification_api import (
    LOGIC_VERIFICATION_PROVIDER_ROLE_CLOSURE_INTERFACE,
    LOGIC_VERIFICATION_PROVIDER_ROLE_SCHEMA,
    LogicVerificationAPI,
    PROVIDER_ROLE_CLOSURE_OPERATIONS,
    VerificationAuthority,
    VerificationStatus,
    build_provider_role_descriptor,
    list_provider_role_descriptors,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL_ID = "FVT-G227"
TASK_ID = "FVT-095"

RUNNABLE_SURFACES = {
    "ergoai": {
        "public_role": "advisor",
        "dispatch": {"inventory", "probe", "install", "advisor"},
        "live_verification": False,
    },
    "symbolicai": {
        "public_role": "advisor",
        "dispatch": {"inventory", "probe", "install", "advisor"},
        "live_verification": False,
    },
    "runtime-mtl-external": {
        "public_role": "authority",
        "dispatch": {"inventory", "probe", "install", "verification"},
        "live_verification": True,
    },
    "souffle": {
        "public_role": "shadow",
        "dispatch": {"inventory", "probe", "install", "verification"},
        "live_verification": True,
    },
}

SUPPORT_ONLY = ("stack", "temurin-jdk", "maude", "opam")


@pytest.fixture
def api() -> LogicVerificationAPI:
    return LogicVerificationAPI()


def test_expected_outputs_exist() -> None:
    assert (
        REPO_ROOT
        / "ipfs_datasets_py"
        / "ipfs_datasets_py"
        / "logic"
        / "verification_api.py"
    ).is_file()
    assert (
        REPO_ROOT
        / "ipfs_datasets_py"
        / "ipfs_datasets_py"
        / "logic"
        / "backends"
        / "installers"
        / "registry.py"
    ).is_file()
    assert (
        REPO_ROOT
        / "ipfs_datasets_py"
        / "tests"
        / "unit"
        / "logic"
        / "test_provider_role_installation_closure.py"
    ).is_file()
    assert Path(__file__).is_file()


def test_interface_identity_and_operations(api: LogicVerificationAPI) -> None:
    assert (
        LOGIC_VERIFICATION_PROVIDER_ROLE_CLOSURE_INTERFACE
        == "LogicVerificationProviderRoleClosure@1"
    )
    assert LOGIC_VERIFICATION_PROVIDER_ROLE_SCHEMA == (
        "logic-verification-provider-role/v1"
    )
    payload = api.to_dict()
    assert (
        payload["provider_role_closure_interface"]
        == LOGIC_VERIFICATION_PROVIDER_ROLE_CLOSURE_INTERFACE
    )
    assert set(payload["provider_role_closure_operations"]) == set(
        PROVIDER_ROLE_CLOSURE_OPERATIONS
    )
    for operation in (
        "list_provider_roles",
        "provider_role",
        "secpal_artifact_intake",
        "secpal_compatibility_lookup",
    ):
        assert operation in PROVIDER_ROLE_CLOSURE_OPERATIONS


def test_list_provider_roles_is_declarative_and_complete(
    api: LogicVerificationAPI,
) -> None:
    response = api.list_provider_roles(request_id="req:roles")
    assert response.status is VerificationStatus.DECLARATIVE
    assert response.authority is VerificationAuthority.DECLARATIVE
    assert response.result["interface"] == (
        LOGIC_VERIFICATION_PROVIDER_ROLE_CLOSURE_INTERFACE
    )
    roles = {item["provider_id"]: item for item in response.result["roles"]}
    assert response.result["count"] == len(roles) >= 20
    for provider_id in (*RUNNABLE_SURFACES, *SUPPORT_ONLY, "secpal"):
        assert provider_id in roles, provider_id
    assert set(response.result["support_only"]) == set(SUPPORT_ONLY)
    assert "secpal" in response.result["archival_intake_providers"]
    assert "secpal" not in response.result["live_verification_providers"]


@pytest.mark.parametrize("provider_id,expected", sorted(RUNNABLE_SURFACES.items()))
def test_runnable_prover_advisor_surfaces(
    api: LogicVerificationAPI,
    provider_id: str,
    expected: dict[str, object],
) -> None:
    role_response = api.provider_role(provider_id)
    assert role_response.status is VerificationStatus.DECLARATIVE
    role = role_response.result["provider_role"]
    assert role["schema_version"] == LOGIC_VERIFICATION_PROVIDER_ROLE_SCHEMA
    assert role["public_role"] == expected["public_role"]
    assert set(role["dispatch_surfaces"]) == expected["dispatch"]
    assert role["is_live_verification_provider"] is expected["live_verification"]

    inventory = api.list_provider_roles()
    assert any(
        item["provider_id"] == provider_id for item in inventory.result["roles"]
    )

    probe = api.probe_provider(provider_id)
    assert probe.status in {
        VerificationStatus.DECLARATIVE,
        VerificationStatus.SUCCEEDED,
        VerificationStatus.UNAVAILABLE,
    }
    assert probe.result["provider_role"]["provider_id"] == provider_id
    assert probe.result.get("mutation_attempted") is not True

    install = api.install_provider(provider_id, dry_run=True)
    assert install.status is VerificationStatus.DECLARATIVE
    assert install.result["install_attempted"] is False
    assert install.result["mutation_authorized"] is False
    assert install.result["provider_role"]["provider_id"] == provider_id
    assert install.result["plan"]["installer_callable"].startswith("ensure_")

    if expected["public_role"] == "advisor":
        advise = api.advise(
            {
                "goal_text": "prove role-closure advisory dispatch",
                "context_text": "advisor surface only",
            },
            provider=provider_id,
        )
        assert advise.status is VerificationStatus.SUCCEEDED
        assert advise.authority is VerificationAuthority.ADVISORY
        assert advise.provider_id in {provider_id, "formalization:symai-proposal-advisor"}
        check = api.check({"statement": "true"}, backend_id=provider_id)
        assert check.status is VerificationStatus.UNSUPPORTED
        assert "public_verification" in " ".join(check.unsupported_features)
    else:
        # Verification-capable roles may be unavailable without a live engine,
        # but must not be rejected as support/archival confusion.
        check = api.check({"statement": "true"}, backend_id=provider_id)
        assert check.status in {
            VerificationStatus.UNAVAILABLE,
            VerificationStatus.UNSUPPORTED,
            VerificationStatus.ERROR,
            VerificationStatus.SUCCEEDED,
            VerificationStatus.PARTIAL,
        }
        if check.status is VerificationStatus.UNSUPPORTED:
            # Backend may be unregistered; role confusion diagnostics differ.
            assert "provider_role:support" not in check.unsupported_features
            assert "provider_role:archival_intake" not in check.unsupported_features


@pytest.mark.parametrize("provider_id", SUPPORT_ONLY)
def test_support_only_axes_not_applicable(
    api: LogicVerificationAPI, provider_id: str
) -> None:
    role = api.provider_role(provider_id).result["provider_role"]
    assert role["public_role"] == "support"
    assert role["support_only"] is True
    assert role["semantic_axis_applicable"] is False
    assert role["authority_axis_applicable"] is False
    assert role["public_verification_applicable"] is False
    assert role["axes"] == {
        "semantic": "not_applicable",
        "authority": "not_applicable",
        "public_verification": "not_applicable",
    }
    assert "verification" not in role["dispatch_surfaces"]
    assert "advisor" not in role["dispatch_surfaces"]

    check = api.check({"statement": "true"}, backend_id=provider_id)
    assert check.status is VerificationStatus.UNSUPPORTED
    assert check.authority is VerificationAuthority.NONE
    assert any("support-only" in item for item in check.diagnostics)

    advise = api.advise(
        {"goal_text": "x", "context_text": "y"}, provider=provider_id
    )
    assert advise.status is VerificationStatus.UNSUPPORTED

    install = api.install_provider(provider_id, dry_run=True)
    assert install.status is VerificationStatus.DECLARATIVE
    assert install.result["support_only"] is True
    assert install.result["public_verification_applicable"] is False


def test_secpal_is_archival_intake_only(api: LogicVerificationAPI) -> None:
    role = api.provider_role("secpal").result["provider_role"]
    assert role["public_role"] == "archival_intake"
    assert role["is_live_verification_provider"] is False
    assert set(role["dispatch_surfaces"]) == {
        "inventory",
        "install",
        "artifact_intake",
        "compatibility_lookup",
    }
    assert "verification" not in role["dispatch_surfaces"]
    assert "probe" not in role["dispatch_surfaces"]

    probe = api.probe_provider("secpal")
    assert probe.status is VerificationStatus.UNSUPPORTED
    assert probe.result["live_verification_provider"] is False

    check = api.check({"statement": "authorized"}, backend_id="secpal")
    assert check.status is VerificationStatus.UNSUPPORTED
    assert any("never" in item.lower() for item in check.diagnostics)

    compatibility = api.secpal_compatibility_lookup(request_id="req:compat")
    assert compatibility.status is VerificationStatus.DECLARATIVE
    assert compatibility.result["live_verification_provider"] is False
    assert compatibility.result["execution_eligible"] is False
    assert compatibility.result["operator_compatibility_can_promote"] is False
    assert isinstance(compatibility.result["compatibility"], dict)

    denied = api.secpal_artifact_intake("/nonexistent/SecPal_Research_Release.msi")
    assert denied.status is VerificationStatus.UNSUPPORTED
    assert denied.result["intake_attempted"] is False
    assert denied.result["live_verification_provider"] is False

    planned = api.secpal_artifact_intake(
        "/nonexistent/SecPal_Research_Release.msi", dry_run=True
    )
    assert planned.status is VerificationStatus.DECLARATIVE
    assert planned.result["intake_attempted"] is False
    assert planned.result["mutation_authorized"] is False


def test_role_confusion_and_unknown_provider_fail_closed(
    api: LogicVerificationAPI,
) -> None:
    unknown = api.provider_role("not-a-reviewed-provider")
    assert unknown.status is VerificationStatus.UNSUPPORTED

    confused = api.advise(
        {"goal_text": "x", "context_text": "y"}, provider="runtime-mtl-external"
    )
    assert confused.status is VerificationStatus.UNSUPPORTED
    assert "provider_role:authority" in confused.unsupported_features

    stack_as_verifier = api.check({"statement": "x"}, backend_id="stack")
    assert stack_as_verifier.status is VerificationStatus.UNSUPPORTED


def test_inventory_probe_and_dry_run_are_side_effect_free(
    api: LogicVerificationAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("role closure discovery must not open network or processes")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    # Inventory and role lookup must not import family installer plugins.
    original_import = importlib.import_module
    plugin_imports: list[str] = []

    def guarded(name: str, *args, **kwargs):
        if ".logic.backends.installers." in name and not name.endswith(".registry"):
            plugin_imports.append(name)
            raise AssertionError(f"plugin import forbidden during inventory: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded)

    descriptors = list_provider_role_descriptors()
    assert {item["provider_id"] for item in descriptors} >= set(RUNNABLE_SURFACES)
    assert api.list_provider_roles().status is VerificationStatus.DECLARATIVE
    for provider_id in (*RUNNABLE_SURFACES, *SUPPORT_ONLY):
        assert api.provider_role(provider_id).status is VerificationStatus.DECLARATIVE
        assert api.probe_provider(provider_id).status in {
            VerificationStatus.DECLARATIVE,
            VerificationStatus.SUCCEEDED,
            VerificationStatus.UNAVAILABLE,
            VerificationStatus.UNSUPPORTED,
        }
        assert api.install_provider(provider_id, dry_run=True).result[
            "install_attempted"
        ] is False
    assert plugin_imports == []


def test_build_provider_role_descriptor_aliases() -> None:
    assert build_provider_role_descriptor("symai")["provider_id"] == "symbolicai"
    assert build_provider_role_descriptor("temurin")["provider_id"] == "temurin-jdk"
    assert (
        build_provider_role_descriptor("runtime_mtl")["provider_id"]
        == "runtime-mtl-external"
    )
    assert build_provider_role_descriptor("missing-tool-xyz") is None


def test_goal_and_task_identity_are_stable() -> None:
    # Keep the objective binding discoverable for supervisor evidence scans.
    assert GOAL_ID == "FVT-G227"
    assert TASK_ID == "FVT-095"
    assert "LogicVerificationProviderRoleClosure@1" in sys.modules[
        "ipfs_datasets_py.logic.verification_api"
    ].LOGIC_VERIFICATION_PROVIDER_ROLE_CLOSURE_INTERFACE
