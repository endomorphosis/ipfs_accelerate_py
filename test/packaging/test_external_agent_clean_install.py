"""EAAEF-164: clean external installation admission.

Does not pip-install the world.  Tests the admission function: refuse editable
paths, sibling repo paths, and mutable branch checkouts as released artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest


RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "packaging.json"
)

DIGEST_A = "sha256:" + "1" * 64
DIGEST_B = "sha256:" + "2" * 64


class CleanInstallError(ValueError):
    """Install descriptor is not a released artifact identity."""


def admit_clean_install(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Admit only released sdist/wheel names plus digests."""

    if not isinstance(spec, Mapping):
        raise CleanInstallError("clean install spec must be an object")
    source = str(spec.get("source") or spec.get("path") or "").strip()
    lowered = source.lower()
    if ".egg-link" in lowered or spec.get("egg_link"):
        raise CleanInstallError("editable egg-link path is not a released artifact")
    if spec.get("editable") or spec.get("pip_e") or "-e" in lowered.split():
        raise CleanInstallError("pip -e editable install is not a released artifact")
    if spec.get("sibling_repo") or spec.get("sibling_checkout"):
        raise CleanInstallError("sibling repo path is not a released install source")
    if "/../" in source or source.endswith("/..") or spec.get("sibling_path"):
        raise CleanInstallError("sibling repo path is not a released install source")
    if spec.get("mutable_branch") or spec.get("branch_checkout"):
        raise CleanInstallError("mutable branch checkout is not a released artifact")
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, (list, tuple)) or not artifacts:
        raise CleanInstallError("released artifact identities are required")
    admitted: list[dict[str, str]] = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise CleanInstallError("artifact identity must be an object")
        name = str(item.get("name") or "").strip()
        digest = str(item.get("digest") or item.get("sha256") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if not name or not digest:
            raise CleanInstallError("artifact name and digest are required")
        if not (name.endswith(".whl") or name.endswith(".tar.gz")):
            raise CleanInstallError("only sdist/wheel names are released artifact identities")
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise CleanInstallError("artifact digest must be sha256:<64 hex>")
        if kind and kind not in {"sdist", "wheel"}:
            raise CleanInstallError("artifact kind must be sdist or wheel")
        admitted.append({"name": name, "digest": digest, "kind": kind or ("wheel" if name.endswith(".whl") else "sdist")})
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/clean-install-admission@1",
        "admitted": True,
        "editable": False,
        "sibling_checkout": False,
        "mutable_branch": False,
        "artifacts": admitted,
    }


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_admit_clean_install_refuses_editable_sibling_and_branch() -> None:
    admitted = admit_clean_install(
        {
            "artifacts": [
                {
                    "name": "ipfs_accelerate_py-0.0.0-py3-none-any.whl",
                    "digest": DIGEST_A,
                    "kind": "wheel",
                },
                {
                    "name": "ipfs_datasets_py-0.0.0.tar.gz",
                    "digest": DIGEST_B,
                    "kind": "sdist",
                },
            ]
        }
    )
    assert admitted["admitted"] is True
    assert admitted["editable"] is False
    assert {item["name"] for item in admitted["artifacts"]} == {
        "ipfs_accelerate_py-0.0.0-py3-none-any.whl",
        "ipfs_datasets_py-0.0.0.tar.gz",
    }

    with pytest.raises(CleanInstallError, match="egg-link"):
        admit_clean_install({"source": "src/ipfs_accelerate_py.egg-link", "artifacts": []})
    with pytest.raises(CleanInstallError, match="pip -e"):
        admit_clean_install({"source": "pip -e ./ipfs_accelerate_py", "editable": True, "artifacts": []})
    with pytest.raises(CleanInstallError, match="sibling"):
        admit_clean_install({"sibling_repo": True, "artifacts": [{"name": "x.whl", "digest": DIGEST_A}]})
    with pytest.raises(CleanInstallError, match="branch"):
        admit_clean_install(
            {
                "mutable_branch": True,
                "artifacts": [{"name": "x.whl", "digest": DIGEST_A}],
            }
        )
    with pytest.raises(CleanInstallError, match="sdist/wheel"):
        admit_clean_install({"artifacts": [{"name": "repo.git", "digest": DIGEST_A}]})

    payload = _write_receipt(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-overlay-receipt@1",
            "task_id": "EAAEF-164",
            "evidence_mode": "contract_fail_closed",
            "live_runtime_invoked": False,
            "live_eight_container_qualification": False,
            "pip_install_invoked": False,
            "admitted": True,
            "refused": ["editable_egg_link", "pip_-e", "sibling_repo", "mutable_branch"],
            "artifacts": list(admitted["artifacts"]),
        }
    )
    saved = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert saved["evidence_mode"] == "contract_fail_closed"
    assert saved["pip_install_invoked"] is False
    assert saved["live_runtime_invoked"] is False
    assert payload["admitted"] is True
