from __future__ import annotations

import argparse
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts/qualify_external_agent_bootstrap_container.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "config/external_agent_autonomous_execution_fabric_bootstrap.json"
)
CONTAINERFILE_PATH = (
    REPO_ROOT
    / "containers/external-agent/bootstrap-reconciliation.Containerfile"
)

SPEC = importlib.util.spec_from_file_location(
    "eaaef_container_qualification",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def _policy():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_bootstrap_container_policy_is_closed_and_nonlaunching():
    policy = _policy()

    qualification._validate_policy(policy, repo_root=REPO_ROOT)

    assert policy["image_build"]["base_image_id"] == (
        qualification.EXPECTED_BASE_IMAGE_ID
    )
    assert policy["runtime_policy"]["workload_class"] == (
        "bootstrap_diagnostic_only"
    )
    assert policy["runtime_policy"]["task_dispatch_admitted"] is False
    assert policy["runtime_policy"]["maximum_parallel_workers"] == 0
    assert policy["runtime_policy"]["maximum_parallel_containers"] == 1
    assert (
        policy["runtime_policy"]["execution_modes"]
        ["rootful_daemon_nonroot_worker"]["admitted"]
        is False
    )
    assert policy["launch_policy"]["this_policy_grants_launch"] is False
    assert (
        policy["launch_policy"]["this_qualification_grants_launch"]
        is False
    )
    assert (
        policy["launch_policy"]["final_materializer_admission_required"]
        is True
    )


def test_policy_cannot_resign_in_rootful_fallback_without_reviewed_source():
    policy = deepcopy(_policy())
    policy["runtime_policy"]["execution_modes"][
        "rootful_daemon_nonroot_worker"
    ]["admitted"] = True
    body = dict(policy)
    body.pop("policy_cid")
    policy["policy_cid"] = qualification._cid(body)

    with pytest.raises(
        qualification.QualificationError,
        match="policy is invalid",
    ):
        qualification._validate_policy(policy, repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("top", "unexpected"),
        ("provider_route", "authorization_path"),
        ("runtime_policy", "ambient_environment"),
        ("qualification_ceremony", "artifact_path"),
    ],
)
def test_policy_rejects_resigned_unknown_fields(location, field):
    policy = deepcopy(_policy())
    target = policy if location == "top" else policy[location]
    target[field] = "not-authority"
    body = dict(policy)
    body.pop("policy_cid")
    policy["policy_cid"] = qualification._cid(body)

    with pytest.raises(
        qualification.QualificationError,
        match="policy is invalid",
    ):
        qualification._validate_policy(policy, repo_root=REPO_ROOT)


def test_spdx_statement_is_deterministic_bounded_and_discloses_scope():
    values = {
        "image_id": "sha256:" + "1" * 64,
        "image_tag": "eaaef-bootstrap-reconciliation:test",
        "base_image_id": qualification.EXPECTED_BASE_IMAGE_ID,
        "toolchains": {
            "git": "git version 2.43.0",
            "python": "Python 3.12.3",
        },
        "source_date_epoch": 1_800_000_000,
    }

    first = qualification._canonical(qualification._spdx_document(**values))
    second = qualification._canonical(qualification._spdx_document(**values))

    assert first == second
    assert len(first) < qualification.MAXIMUM_SBOM_BYTES
    document = json.loads(first)
    assert document["spdxVersion"] == "SPDX-2.3"
    assert all(package["filesAnalyzed"] is False for package in document["packages"])
    assert "were not analyzed" in document["documentComment"]


def test_missing_runtime_returns_typed_no_go_without_minting(monkeypatch):
    monkeypatch.setattr(qualification.shutil, "which", lambda _name: None)
    args = argparse.Namespace(
        config=CONFIG_PATH,
        runtime="docker",
        image_tag="eaaef-bootstrap-reconciliation:test",
        source_date_epoch=1_800_000_000,
        diagnostic_build=False,
        prior_failed_attempt=[],
    )

    report, sbom = qualification.qualify(args)

    assert report["status"] == "host_capability_no_go"
    assert report["blockers"] == [
        "agent_worker_image_unavailable",
        "provider_task_dispatch_not_admitted",
        "container_runtime_unavailable",
    ]
    assert report["workload_class"] == "bootstrap_diagnostic_only"
    assert report["task_dispatch_admitted"] is False
    assert report["maximum_parallel_workers"] == 0
    assert report["maximum_parallel_containers"] == 1
    assert report["image_qualification_minted"] is False
    assert report["provider_container_qualification_minted"] is False
    assert report["supervisor_process_started"] is False
    assert report["provider_process_started"] is False
    assert report["report_cid"] == qualification._cid(
        {key: value for key, value in report.items() if key != "report_cid"}
    )
    assert sbom == b""


def test_containerfile_uses_offline_local_base_and_clears_environment():
    text = CONTAINERFILE_PATH.read_text(encoding="utf-8")

    assert not text.startswith("# syntax=")
    assert (
        "ARG BASE_IMAGE=ipfs-accelerate-authority-validation:20260803-v2"
        in text
    )
    assert qualification.EXPECTED_BASE_IMAGE_ID in text
    assert 'USER 65532:65532' in text
    assert 'ENTRYPOINT ["/usr/bin/env", "-i"' in text
    assert "COPY " not in text
    assert "ADD " not in text
