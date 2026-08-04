"""Public install_provider surface tests (FVT-G216 / FVT-G227 implied validation).

Covers role-aware planning for advisors, support tools, SecPAL archival intake,
and the stable LogicVerificationLazyInstaller@1 mutation boundary.
"""

from __future__ import annotations

import pytest

from ipfs_datasets_py.logic.verification_api import (
    LogicVerificationAPI,
    VerificationAuthority,
    VerificationStatus,
)


@pytest.fixture
def api() -> LogicVerificationAPI:
    return LogicVerificationAPI()


@pytest.mark.parametrize(
    "provider_id,family,public_role",
    [
        ("ergoai", "advisors", "advisor"),
        ("symbolicai", "advisors", "advisor"),
        ("runtime-mtl-external", "runtime_mtl", "authority"),
        ("souffle", "authorization", "shadow"),
        ("secpal", "authorization", "archival_intake"),
        ("stack", "tamarin", "support"),
        ("temurin-jdk", "advisors", "support"),
        ("maude", "tamarin", "support"),
        ("opam", "rocq", "support"),
        ("z3", "solver", "authority"),
    ],
)
def test_install_provider_dry_run_is_role_aware(
    api: LogicVerificationAPI,
    provider_id: str,
    family: str,
    public_role: str,
) -> None:
    response = api.install_provider(
        provider_id, dry_run=True, request_id=f"req:{provider_id}"
    )
    assert response.status is VerificationStatus.DECLARATIVE
    assert response.authority is VerificationAuthority.NONE
    assert response.result["status"] == "planned"
    assert response.result["install_attempted"] is False
    assert response.result["mutation_authorized"] is False
    plan = response.result["plan"]
    assert plan["provider_id"] == provider_id
    assert plan["family"] == family
    assert plan["installer_callable"].startswith("ensure_")
    assert plan["discovery_imports_plugin"] is False
    role = response.result["provider_role"]
    assert role["public_role"] == public_role
    assert role["provider_id"] == provider_id
    if public_role == "support":
        assert response.result["support_only"] is True
        assert response.result["public_verification_applicable"] is False
    if public_role == "archival_intake":
        assert response.result["live_verification_provider"] is False
        assert response.result["artifact_intake_only"] is True


def test_install_provider_requires_boolean_allow_install(api: LogicVerificationAPI) -> None:
    response = api.install_provider("z3", allow_install="true")  # type: ignore[arg-type]
    assert response.status is VerificationStatus.INVALID
    assert response.result["install_attempted"] is False


def test_install_provider_denied_and_offline_never_invoke_executor() -> None:
    calls: list[object] = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("executor must not run")

    api = LogicVerificationAPI(installer_executor=forbidden)
    assert api.install_provider("z3").status is VerificationStatus.UNSUPPORTED
    assert api.install_provider("z3", dry_run=True).status is VerificationStatus.DECLARATIVE
    assert (
        api.install_provider("z3", allow_install=True, offline=True).status
        is VerificationStatus.UNAVAILABLE
    )
    assert calls == []


def test_install_provider_explicit_mutation_attaches_role_metadata() -> None:
    def executor(provider_id: str, **_kwargs):
        assert provider_id == "z3"
        return {
            "schema_version": "logic-verification-install-receipt/v1",
            "interface": "LogicVerificationLazyInstaller@1",
            "provider_id": "z3",
            "status": "installed",
            "installed": True,
            "certified": True,
            "authority": "none",
            "install_attempted": True,
            "mutation_authorized": True,
            "evidence": {
                "identity_bound": True,
                "checksum": {"required": True, "verified": True},
                "rollback": {"required": True, "verified": True},
                "semantic_probe": {"version": "4.12.2"},
                "dependency": {"callable": "ensure_z3"},
            },
        }

    response = LogicVerificationAPI(installer_executor=executor).install_provider(
        "z3", allow_install=True
    )
    assert response.status is VerificationStatus.SUCCEEDED
    assert response.result["provider_role"]["public_role"] == "authority"
    assert response.result["provider_role"]["is_live_verification_provider"] is True
    assert response.result["certified"] is True


def test_unknown_provider_install_is_unsupported(api: LogicVerificationAPI) -> None:
    response = api.install_provider("definitely-not-a-tool", dry_run=True)
    assert response.status is VerificationStatus.UNSUPPORTED
    assert response.result["installed"] is False


def test_secpal_install_plan_never_claims_live_verification(
    api: LogicVerificationAPI,
) -> None:
    response = api.install_provider("secpal", dry_run=True)
    assert response.status is VerificationStatus.DECLARATIVE
    assert response.result["provider_role"]["public_role"] == "archival_intake"
    assert response.result["live_verification_provider"] is False
    assert "verification" not in response.result["provider_role"]["dispatch_surfaces"]
