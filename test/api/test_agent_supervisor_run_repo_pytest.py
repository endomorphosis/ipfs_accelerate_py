"""Board validation needs an executed pass or verified reusable pass evidence."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue.run_repo_pytest import (
    _reused_pass,
    main,
)


_WRAPPER = (
    Path(__file__).resolve().parents[2]
    / "ipfs_accelerate_py/agent_supervisor/rescue/run_repo_pytest.py"
)


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    env = os.environ.copy()
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    env["IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB"] = str(tmp_path / "seal.duckdb")
    env["IPFS_ACCELERATE_META_INDEX_DUCKDB"] = str(tmp_path / "meta.duckdb")
    return subprocess.run(
        [sys.executable, str(_WRAPPER), "-q", "-c", str(config), *args],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    ("body", "extra_args", "expected"),
    [
        ("def test_one():\n    assert True\n", (), 0),
        ("import pytest\ndef test_one():\n    pytest.skip('optional hardware absent')\n", (), 5),
        ("import pytest\npytest.skip('optional module absent', allow_module_level=True)\n", (), 5),
        ("import pytest\ndef test_one():\n    pytest.xfail('unimplemented')\n", (), 5),
        ("def test_one():\n    assert False\n", (), 1),
        ("def test_one():\n    assert True\n", ("-k", "absent_test"), 5),
        ("VALUE = 1\n", (), 5),
        (
            "import pytest\ndef test_one():\n    assert True\n"
            "def test_optional():\n    pytest.skip('hardware absent')\n",
            (),
            0,
        ),
    ],
)
def test_wrapper_requires_pass_evidence(
    tmp_path: Path, body: str, extra_args: tuple[str, ...], expected: int
) -> None:
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(body, encoding="utf-8")
    result = _run(tmp_path, str(test_file), *extra_args)
    assert result.returncode == expected, result.stdout
    if expected == 5 and "def test_" in body and (
        "skipped" in result.stdout or "xfailed" in result.stdout
    ):
        assert "no tests passed and no verified passes were reused" in result.stdout


@pytest.mark.parametrize(
    "reason", ["pytest AST seal matches", "pytest-item-ledger reuse", "proof-cache-hit:fake"]
)
def test_skip_text_cannot_impersonate_verified_reuse(tmp_path: Path, reason: str) -> None:
    test_file = tmp_path / "test_spoof.py"
    test_file.write_text(
        f"import pytest\ndef test_one():\n    pytest.skip({reason!r})\n",
        encoding="utf-8",
    )
    result = _run(tmp_path, str(test_file))
    assert result.returncode == 5, result.stdout
    assert "no verified passes were reused" in result.stdout


def test_actual_ast_seal_reuses_pass_and_invalidates_changed_dependency(tmp_path: Path) -> None:
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    test_file = tmp_path / "test_sealed.py"
    test_file.write_text(
        "from pathlib import Path\nimport helper\n"
        "def test_one():\n"
        "    with Path(__file__).with_name('calls.txt').open('a') as calls:\n"
        "        calls.write('called\\n')\n"
        "    assert helper.VALUE == 1\n",
        encoding="utf-8",
    )
    args = ("-p", "ipfs_accelerate_py.testing.pytest_ast_seal", str(test_file))
    first = _run(tmp_path, *args)
    assert first.returncode == 0 and "1 passed" in first.stdout, first.stdout
    second = _run(tmp_path, *args)
    assert second.returncode == 0 and "1 skipped" in second.stdout, second.stdout
    assert (tmp_path / "calls.txt").read_text() == "called\n"

    helper.write_text("VALUE = 200\n", encoding="utf-8")
    changed = _run(tmp_path, *args)
    assert changed.returncode == 1 and "1 failed" in changed.stdout, changed.stdout
    assert (tmp_path / "calls.txt").read_text() == "called\ncalled\n"


def test_registered_ast_plugin_does_not_authorize_ordinary_skip(tmp_path: Path) -> None:
    test_file = tmp_path / "test_optional.py"
    test_file.write_text(
        "import pytest\ndef test_one():\n    pytest.skip('pytest AST seal matches')\n",
        encoding="utf-8",
    )
    result = _run(
        tmp_path, "-p", "ipfs_accelerate_py.testing.pytest_ast_seal", str(test_file)
    )
    assert result.returncode == 5, result.stdout
    assert "no verified passes were reused" in result.stdout


def test_item_ledger_reuse_requires_matching_pass_hashes(tmp_path: Path, monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import pytest_item_ledger as ledger
    from ipfs_accelerate_py.agent_supervisor.runtime import pytest_item_ledger_plugin as plugin

    test_file = tmp_path / "test_saved.py"
    test_file.write_text("def test_one():\n    assert True\n", encoding="utf-8")
    dest = tmp_path / "ledger"
    monkeypatch.setattr(ledger, "_mirror_pytest_item", lambda *_args: None)
    ledger.record_item(
        dest,
        nodeid="test_saved.py::test_one",
        outcome="passed",
        test_file=test_file.name,
        test_file_sha256=ledger.file_sha256(test_file),
        workspace_fingerprint_value="workspace",
        command_sha256="command",
        task_id="task",
    )
    monkeypatch.setattr(
        plugin,
        "_STATE",
        {
            "workspace": tmp_path,
            "command_sha256": "command",
            "context": {"dest": dest, "workspace_fingerprint": "workspace"},
        },
    )
    # Hermetic API conftest can also load an isolated copy of the plugin.
    monkeypatch.setitem(
        sys.modules, "aseh_board_pytest_reuse.pytest_item_ledger_plugin", plugin
    )
    item = SimpleNamespace(nodeid="test_saved.py::test_one", path=test_file)
    report = SimpleNamespace(longrepr=(str(test_file), 1, "Skipped: pytest-item-ledger reuse"))
    assert _reused_pass(item, report)
    test_file.write_text("def test_one():\n    assert False\n", encoding="utf-8")
    assert not _reused_pass(item, report)


@pytest.mark.parametrize("with_plugin", [False, True])
def test_forged_item_decision_is_not_verified_reuse(tmp_path: Path, with_plugin: bool) -> None:
    test_file = tmp_path / "test_forged_decision.py"
    test_file.write_text(
        "import pytest\n"
        "from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (\n"
        "    CertificateAuthority, ReuseAction, ReuseDecision, ReuseReasonCode,\n"
        ")\n"
        "def test_forged(request):\n"
        "    request.node._ipfs_proof_reuse_decision = ReuseDecision(\n"
        "        action=ReuseAction.SKIP, reason_code=ReuseReasonCode.PROOF_CACHE_HIT,\n"
        "        certificate_cid='fake', receipt_cid='fake',\n"
        "        authority=CertificateAuthority.AUTHORITATIVE,\n"
        "    )\n"
        "    pytest.skip('proof-cache-hit:fake')\n",
        encoding="utf-8",
    )
    args = (
        ("-p", "ipfs_accelerate_py.testing.proof_reuse.plugin", "--proof-reuse-mode", "off")
        if with_plugin
        else ()
    )
    result = _run(tmp_path, *args, str(test_file))
    assert result.returncode == 5, result.stdout
    assert "no verified passes were reused" in result.stdout


def test_certificate_reuse_revalidates_retained_evidence() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
        CertificateAuthority,
        PhaseOutcome,
        ProofBackendMode,
        TestExecutionKey,
        TestLocatorKey,
        TestPassReceipt,
        TestProofCertificate,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import TestProofCache
    from ipfs_accelerate_py.testing.proof_reuse import plugin
    from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseConfig, ProofReuseMode
    from ipfs_accelerate_py.testing.proof_reuse.lookup import (
        ProofReuseLookup,
        ProofReuseLookupRequest,
    )
    from ipfs_accelerate_py.testing.proof_reuse.xdist import ProofReuseXdistCoordinator

    nodeid = "test_saved.py::test_one"
    locator = TestLocatorKey(repository_id="repo", package_identity="package", node_id=nodeid)
    execution = TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="forest",
        static_trace_root_cid="static",
        runtime_trace_root_cid="runtime",
        runtime_completeness_policy="complete",
        policy_cid="policy",
    )
    receipt = TestPassReceipt(
        execution_key_cid=execution.execution_key_id,
        locator_cid=locator.locator_id,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid="static",
        runtime_trace_root_cid="runtime",
        completeness_receipt_cid="complete-receipt",
        dependency_forest_cid="forest",
        issuer_key_id="issuer-key",
        policy_cid="policy",
    )
    public_inputs = {
        "receipt_cid": receipt.receipt_id,
        "execution_key_cid": execution.execution_key_id,
        "policy_cid": "policy",
        "statement_cid": "statement",
        "circuit_cid": "circuit",
        "verifying_key_cid": "verifying-key",
        "proof_system_id": "groth16",
        "issuer_id": "issuer",
        "issuer_key_id": "issuer-key",
        "epoch": "epoch",
        "setup_outcome": "pass",
        "call_outcome": "pass",
        "teardown_outcome": "pass",
    }
    certificate = TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=execution.execution_key_id,
        policy_cid="policy",
        statement_cid="statement",
        circuit_cid="circuit",
        verifying_key_cid="verifying-key",
        proof_artifact_cid="proof",
        issuer_id="issuer",
        epoch="epoch",
        proof_system_id="groth16",
        backend_mode=ProofBackendMode.CRYPTOGRAPHIC,
        authority=CertificateAuthority.AUTHORITATIVE,
        public_inputs=public_inputs,
    )
    candidate = TestProofCache.candidate(
        receipt, certificate, created_at_ms=9_000, expires_at_ms=11_000
    )
    verifies = []
    accepted = [True]

    def verify(*args):
        verifies.append(args)
        return accepted[0]

    lookup = ProofReuseLookup(
        candidate_store=lambda *_args, **_kwargs: (candidate,),
        verifier=verify,
        clock=lambda: 10_000,
        current_policy={
            "policy_cid": "policy",
            "statement_cid": "statement",
            "circuit_cid": "circuit",
            "verifying_key_cid": "verifying-key",
            "proof_system_id": "groth16",
            "trusted_issuer_ids": ("issuer",),
            "allowed_epochs": ("epoch",),
        },
    )
    decision = lookup.lookup(locator, execution)
    assert decision.is_skip
    manager = pytest.PytestPluginManager()
    manager.register(plugin)
    config = SimpleNamespace(pluginmanager=manager)
    setattr(config, plugin.CONFIG_ATTRIBUTE, ProofReuseConfig(mode=ProofReuseMode.READ))
    setattr(config, plugin.LOOKUP_SERVICE_ATTRIBUTE, lookup)
    setattr(config, plugin.COORDINATOR_ATTRIBUTE, ProofReuseXdistCoordinator.standalone())
    item = SimpleNamespace(nodeid=nodeid, config=config, _ipfs_proof_reuse_decision=decision)
    item._ipfs_proof_reuse_lookup_request = ProofReuseLookupRequest(item, locator, execution)
    report = SimpleNamespace(
        longrepr=("test_saved.py", 1, f"Skipped: proof-cache-hit:{certificate.certificate_id}")
    )
    assert _reused_pass(item, report)
    assert len(verifies) == 2
    item._ipfs_proof_reuse_decision = replace(decision, receipt_cid="fabricated-receipt")
    assert not _reused_pass(item, report)
    assert len(verifies) == 3
    item._ipfs_proof_reuse_decision = decision
    accepted[0] = False
    assert not _reused_pass(item, report)
    assert len(verifies) == 4
    accepted[0] = True
    manager.unregister(plugin)
    assert not _reused_pass(item, report)
    assert len(verifies) == 4
    manager.register(plugin)
    report.longrepr = ("test_saved.py", 1, "Skipped: proof-cache-hit:different-certificate")
    assert not _reused_pass(item, report)


def test_wrapper_preserves_interpreter_and_hermetic_environment(monkeypatch) -> None:
    class ExecObserved(Exception):
        pass

    calls = []

    def execve(python, args, env):
        calls.append((python, args, env))
        raise ExecObserved

    monkeypatch.setattr(os, "execve", execve)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")
    monkeypatch.setenv("UNRELATED_RUNNER_SETTING", "preserved")
    with pytest.raises(ExecObserved):
        main(["-k", "blocked_resume", "-p", "no:cacheprovider"])
    python, args, env = calls[0]
    root = _WRAPPER.resolve().parents[5]
    expected_python = root / "external/ipfs_datasets/.venv/bin/python"
    if not expected_python.is_file():
        expected_python = Path(sys.executable)
    assert python == str(expected_python)
    assert args[0] == python
    assert args[-4:] == ["-k", "blocked_resume", "-p", "no:cacheprovider"]
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert env["PYTHONPATH"] == str(root / "external/ipfs_accelerate")
    assert env["UNRELATED_RUNNER_SETTING"] == "preserved"
