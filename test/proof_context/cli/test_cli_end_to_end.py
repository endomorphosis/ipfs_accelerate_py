"""End-to-end qualification for the public ``python -m`` proof-context CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.proof_context.adapters.external_patch import cid_for_bytes
from ipfs_accelerate_py.proof_context.adapters.models import (
    CONTEXT_PACK_SCHEMA, MODEL_ROUTE_DECISION_SCHEMA, TASK_SPECIFICATION_SCHEMA,
)
from ipfs_accelerate_py.proof_context.bootstrap import create_ordinary_python_repository
from ipfs_accelerate_py.proof_context.cli.execution_commands import ExecutionContext
from ipfs_accelerate_py.proof_context.cli.state_commands import _identities

MODULE = "ipfs_accelerate_py.proof_context.cli"
PATCH = """diff --git a/src/demo/__init__.py b/src/demo/__init__.py
--- a/src/demo/__init__.py
+++ b/src/demo/__init__.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""


def _invoke(*args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    return subprocess.run([sys.executable, "-m", MODULE, *args], cwd=root, env=environment, text=True, capture_output=True, check=False)


def _request(tmp_path: Path, repository: Path, *, adapter: str = "external-patch") -> Path:
    context = ExecutionContext("run", repository, "production", "PCCE-044", "trace-044", "json", "run-044")
    identities = _identities(context)
    route_cid = cid_for_bytes(b"route")
    patch = tmp_path / "change.diff"
    patch.write_text(PATCH, encoding="utf-8")
    document = {"schema": "ipfs-accelerate.proof-context.v0.1/cli-run-request@1", "task": {"schema": TASK_SPECIFICATION_SCHEMA, "task_id": "PCCE-044", "objective_id": "PCCE-G400", "repository_state_cid": identities.repository_state_cid, "owned_paths": ["src/demo/__init__.py"], "declared_files": ["src/demo/__init__.py"], "route_cid": route_cid, "provenance": "live"}, "context_pack": {"schema": CONTEXT_PACK_SCHEMA, "pack_cid": cid_for_bytes(b"pack"), "task_id": "PCCE-044", "repository_state_cid": identities.repository_state_cid, "sufficiency": "sufficient", "provenance": "live"}, "route": {"schema": MODEL_ROUTE_DECISION_SCHEMA, "decision_cid": route_cid, "task_id": "PCCE-044", "repository_state_cid": identities.repository_state_cid, "provider": "external", "model": adapter, "revision": "r1", "tier": "medium", "provenance": "live"}, "adapter": {"name": adapter, "options": {"patch_file": str(patch), "declared_files": ["src/demo/__init__.py"]}}}
    result = tmp_path / "request.json"
    result.write_text(json.dumps(document), encoding="utf-8")
    return result


def _common(repository: Path) -> list[str]:
    return ["--repository", str(repository), "--task", "PCCE-044", "--correlation", "trace-044", "--run-id", "run-044"]


def test_module_help_discovers_every_governed_command_and_human_mode(tmp_path: Path) -> None:
    help_result = _invoke("--help")
    assert help_result.returncode == 0
    for command in ("init", "scan", "status", "plan", "run", "verify", "resume", "expand-context", "explain-impact", "assurance", "seal", "report"):
        assert command in help_result.stdout
    repository = tmp_path / "ordinary-repository"
    initialized = _invoke("init", "--repository", str(repository), "--task", "PCCE-044", "--correlation", "trace-044", "--output-mode", "human", "--human-report")
    assert initialized.returncode == 0
    assert "command: init" in initialized.stdout
    assert "status: succeeded" in initialized.stdout
    assert "Proof-carrying context patch report" in initialized.stdout


def test_external_patch_good_and_bad_flows_are_governed_and_resume_is_idempotent(tmp_path: Path) -> None:
    repository = create_ordinary_python_repository(tmp_path / "repository")
    good = _invoke("run", *_common(repository), "--request", str(_request(tmp_path, repository)))
    payload = json.loads(good.stdout)
    assert good.returncode == 0
    assert payload["status"] == "succeeded"
    assert payload["payload"]["adapter_registry_admitted"] is True
    assert payload["payload"]["canonical_mutated"] is False
    bad_request = _request(tmp_path, repository)
    bad = json.loads(bad_request.read_text(encoding="utf-8"))
    bad["adapter"]["options"]["declared_files"] = ["outside.py"]
    bad_request.write_text(json.dumps(bad), encoding="utf-8")
    rejected = _invoke("run", *_common(repository), "--request", str(bad_request))
    assert rejected.returncode == 3
    assert json.loads(rejected.stdout)["status"] == "rejected"
    patch_id = payload["identities"]["patch_id"]
    # Resume by the durable run/patch identity after the preceding run. A
    # second process must safely observe the same settled recovery outcome.
    first = _invoke("resume", *_common(repository), "--patch-id", patch_id)
    second = _invoke("resume", *_common(repository), "--patch-id", patch_id)
    assert first.returncode == second.returncode
    assert json.loads(first.stdout)["payload"]["resumed_by_run_id"] == "run-044"


def test_replay_and_evidence_inputs_fail_closed_at_the_module_boundary(tmp_path: Path) -> None:
    repository = create_ordinary_python_repository(tmp_path / "repository")
    request = _request(tmp_path, repository, adapter="replay")
    replay = json.loads(request.read_text(encoding="utf-8"))
    replay["adapter"]["options"] = {"fixtures": [], "selected_fixture_cid": "not-a-cid", "selected_response_artifact_cid": "not-a-cid"}
    request.write_text(json.dumps(replay), encoding="utf-8")
    result = _invoke("run", *_common(repository), "--request", str(request))
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "invalid"
    parent = tmp_path / "wrong-parent.json"
    parent.write_text("{}", encoding="utf-8")
    evidence = _invoke("seal", *_common(repository), "--repository-id", "ordinary-python:repository", "--patch-id", cid_for_bytes(PATCH.encode()), "--parent", str(parent))
    assert evidence.returncode == 2
    assert json.loads(evidence.stdout)["status"] == "invalid"
