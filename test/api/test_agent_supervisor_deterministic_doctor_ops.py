"""Tests for the thin deterministic-doctor ops facade (LPR-039)."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DeterministicDoctorPlan,
    DoctorAuthorityRoots,
    DoctorConsumerDisposition,
    DoctorEditSite,
    DoctorEvidenceSnapshot,
    DoctorMode,
    DoctorPlanDisposition,
    DoctorPlanStep,
    DoctorRepairDisposition,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = (
    REPO_ROOT
    / "scripts"
    / "ops"
    / "agent_supervisor"
    / "deterministic_doctor.py"
)
SERVICE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "control"
    / "deterministic_doctor_service.py"
)

_FORBIDDEN_WRAPPER_LOGIC = re.compile(
    r"\b(?:DeterministicDoctorImpactAnalyzer|DeterministicDoctorTransaction|"
    r"DeterministicDoctorFixedPointValidator|DeterministicDoctorHammer|"
    r"DeterministicDoctorSynthesizer|DeterministicDoctorTactician|"
    r"compile_deterministic_doctor_plan|DoctorRepairOperatorRegistry|"
    r"duckdb|sqlite3|socket\.|urllib|requests\.|openai|anthropic|neo4j|"
    r"llm_router|render_|materialize_|apply_span|ProgramGraph)\b"
)


def _roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:fixture",
        "forest_id": "forest:fixture",
        "tree_id": "tree:fixture",
        "overlay_id": "overlay:fixture",
        "file_root_id": "file-root:fixture",
        "ast_root_id": "ast:fixture",
        "graph_id": "graph:fixture",
        "corpus_id": "corpus:fixture",
        "index_id": "index:fixture",
        "model_id": "model:fixture",
        "cache_id": "cache:fixture",
        "operator_registry_id": "operators:fixture",
        "translator_id": "translator:fixture",
        "solver_id": "solver:fixture",
        "kernel_id": "kernel:fixture",
        "toolchain_id": "toolchain:fixture",
        "policy_id": "policy:fixture",
        "sandbox_id": "sandbox:fixture",
        "environment_id": "environment:fixture",
        "lease_id": "lease:fixture",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def _snapshot() -> DoctorEvidenceSnapshot:
    roots = _roots()
    return DoctorEvidenceSnapshot(
        roots=roots,
        snapshot_id="snapshot:fixture",
        file_blob_cids=("blob:a", "blob:b"),
        completeness="complete",
        invalidation_refs=("tree:fixture",),
        clean_rebuild_equivalence_receipt_id="rebuild:eq:1",
    )


def _admitted_plan() -> DeterministicDoctorPlan:
    roots = _roots()
    site = DoctorEditSite(
        path="pkg/module.py",
        before_hash="sha256:before",
        span_start=0,
        span_end=10,
        artifact_id="blob:module",
    )
    step = DoctorPlanStep(
        step_id="step:1",
        kind="analytical",
        operator_id="operator:rename",
        consumer_ids=("consumer:one",),
        edit_site_refs=(site.content_id,),
        write_paths=("pkg/module.py",),
    )
    consumer = DoctorConsumerDisposition(
        roots=roots,
        consumer_id="consumer:one",
        disposition=DoctorRepairDisposition.SUPPORTED,
        reason_codes=("ok",),
    )
    return DeterministicDoctorPlan(
        roots=roots,
        plan_id="plan:fixture",
        snapshot_id="snapshot:fixture",
        finding_ids=("finding:one",),
        disposition=DoctorPlanDisposition.ADMITTED,
        consumer_dispositions=(consumer,),
        impact_closure_id="impact:fixture",
        steps=(step,),
        edit_sites=(site,),
        operator_ids=("operator:rename",),
        target_ref="symbol:target",
        value_source_ref="value:source",
        placement_ref="placement:site",
        selected_operator_id="operator:rename",
        permitted_read_paths=("pkg/module.py",),
        permitted_write_paths=("pkg/module.py",),
        lease_id="lease:fixture",
        checkpoint_ref="checkpoint:fixture",
        rollback_ref="rollback:fixture",
        proof_refs=("proof:fixture",),
        invalidation_refs=("tree:fixture",),
    )


def _run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    command_env = {
        **dict(os.environ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "IPFS_ACCEL_SKIP_CORE": "1",
        "PYTHONPATH": str(REPO_ROOT)
        + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
    }
    if env:
        command_env.update(env)
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=command_env,
        check=False,
    )


def test_wrapper_contains_only_argument_config_bootstrap_delegation_code() -> None:
    source = CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # No class definitions that implement domain engines.
    assert not [n for n in tree.body if isinstance(n, ast.ClassDef)]
    forbidden = _FORBIDDEN_WRAPPER_LOGIC.findall(source)
    assert forbidden == [], f"wrapper embeds engine/provider logic: {forbidden}"
    assert "argparse" in source
    assert "build_parser" in source
    assert "main" in source
    assert "DeterministicDoctorService" in source or "create_deterministic_doctor_service" in source
    # Must not implement analysis/proof/transaction logic inline.
    assert "def compile_" not in source
    assert "def apply_" not in source
    assert "def render_" not in source


def test_help_starts_no_process_opens_no_db_accesses_no_network_or_storage() -> None:
    script = f"""
import json, sys, os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['IPFS_ACCEL_SKIP_CORE'] = '1'
forbidden_roots = (
    'torch', 'transformers', 'openai', 'anthropic', 'neo4j', 'duckdb',
    'psycopg2', 'sqlalchemy', 'requests', 'httpx', 'aiohttp', 'llm_router',
)
before = {{name for name in sys.modules if name.split('.')[0] in forbidden_roots}}
import importlib.util
spec = importlib.util.spec_from_file_location(
    'deterministic_doctor_cli',
    {str(CLI)!r},
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
parser = mod.build_parser()
help_text = parser.format_help()
after = {{name for name in sys.modules if name.split('.')[0] in forbidden_roots}}
print(json.dumps({{
    'added': sorted(after - before),
    'has_inspect': 'inspect' in help_text,
    'has_repair': 'repair' in help_text,
    'has_replay': 'replay' in help_text,
    'has_status': 'status' in help_text,
    'has_verify': 'verify' in help_text,
    'has_main': hasattr(mod, 'main'),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={
            **dict(os.environ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "IPFS_ACCEL_SKIP_CORE": "1",
            "PYTHONPATH": str(REPO_ROOT),
        },
    )
    payload = json.loads(completed.stdout)
    assert payload["added"] == []
    assert payload["has_inspect"]
    assert payload["has_repair"]
    assert payload["has_replay"]
    assert payload["has_status"]
    assert payload["has_verify"]
    assert payload["has_main"]


def test_cli_help_exit_code_is_success() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "inspect" in result.stdout
    assert "repair" in result.stdout
    assert "replay" in result.stdout


def test_cli_missing_command_is_usage_error() -> None:
    result = _run_cli()
    assert result.returncode == 2


def test_cli_forbids_secret_body_argv() -> None:
    result = _run_cli("--token", "sekrit", "inspect")
    assert result.returncode == 2
    assert "forbidden argument" in result.stderr.lower() or "error" in result.stderr.lower()


def test_cli_discovery_is_cold_and_bounded() -> None:
    result = _run_cli("discovery")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["llm_router_enabled"] is False
    assert payload["automatic_fallback"] is False
    assert payload["processes_started"] is False
    assert payload["database_opened"] is False
    assert "inspect" in payload["operations"]
    assert "repair" in payload["operations"]


def test_cli_status_read_only() -> None:
    result = _run_cli("status")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["operation"] == "status"
    assert payload["read_only"] is True
    assert payload["changed"] is False
    assert payload["status"]["llm_router_enabled"] is False
    assert payload["status"]["automatic_fallback"] is False


def test_cli_inspect_with_snapshot_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = Path(tmp) / "snapshot.json"
        snap_path.write_text(
            json.dumps(_snapshot().to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        result = _run_cli(
            "--snapshot-json",
            str(snap_path),
            "--incident-id",
            "incident:cli-inspect",
            "inspect",
        )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["operation"] == "inspect"
    assert payload["read_only"] is True
    assert payload["disposition"] == "supported"
    assert payload["run_receipt"]["operation"] == "inspect"
    assert payload["run_receipt"]["committed_tree_cid"] == ""
    assert payload["run_receipt"]["llm_router_invoked"] is False
    assert payload["run_receipt"]["model_invocation_count"] == 0


def test_cli_plan_is_read_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plan_path = Path(tmp) / "plan.json"
        plan_path.write_text(
            json.dumps(_admitted_plan().to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        result = _run_cli(
            "--plan-json",
            str(plan_path),
            "--mode",
            "plan",
            "--incident-id",
            "incident:cli-plan",
            "plan",
        )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["operation"] == "plan"
    assert payload["read_only"] is True
    assert payload["changed"] is False
    assert payload["run_receipt"]["committed_tree_cid"] == ""


def test_cli_repair_without_enabled_policy_does_not_write() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plan_path = Path(tmp) / "plan.json"
        snap_path = Path(tmp) / "snapshot.json"
        plan_path.write_text(
            json.dumps(_admitted_plan().to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        snap_path.write_text(
            json.dumps(_snapshot().to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        result = _run_cli(
            "--plan-json",
            str(plan_path),
            "--snapshot-json",
            str(snap_path),
            "--mode",
            "narrow_auto",
            "--lease-id",
            "lease:fixture",
            "--checkpoint-ref",
            "checkpoint:fixture",
            "--rollback-ref",
            "rollback:fixture",
            "--exact-clean-target",
            "--incident-id",
            "incident:cli-repair-denied",
            "repair",
        )
    # Non-zero: policy default is disabled / mode forbids write.
    assert result.returncode != 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["operation"] == "repair"
    assert payload["changed"] is False
    assert payload["disposition"] != "supported" or payload.get("run_receipt") is None


def test_cli_rejects_secret_keys_in_json_payload() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.json"
        bad.write_text(json.dumps({"token": "sekrit", "operation": "inspect"}), encoding="utf-8")
        result = _run_cli("--snapshot-json", str(bad), "inspect")
    assert result.returncode in (1, 2)
    assert "secret" in result.stderr.lower() or "error" in result.stderr.lower()


def test_cli_main_symbol_exists_and_returns_int() -> None:
    script = f"""
import importlib.util, json, io, contextlib
spec = importlib.util.spec_from_file_location('dd_cli', {str(CLI)!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    code = mod.main(['discovery'])
# Emit only the assertion payload on stdout.
print(json.dumps({{'code': int(code), 'has_main': True, 'stdout_len': len(buf.getvalue())}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={
            **dict(os.environ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "IPFS_ACCEL_SKIP_CORE": "1",
            "PYTHONPATH": str(REPO_ROOT),
        },
    )
    payload = json.loads(completed.stdout)
    assert payload["code"] == 0
    assert payload["has_main"] is True
    assert payload["stdout_len"] > 0


def test_service_and_ops_files_exist_as_declared_outputs() -> None:
    assert SERVICE.is_file()
    assert CLI.is_file()
    assert "DeterministicDoctorService" in SERVICE.read_text(encoding="utf-8")
    assert "def main" in CLI.read_text(encoding="utf-8")
