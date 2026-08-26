"""Compatibility checks for the supervised Grok runner's router surface."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
    GROK_QUOTA_PROBE_CONTRACT,
    build_grok_failure_receipt,
    valid_grok_failure_receipt,
)

from ipfs_accelerate_py import llm_router


def test_runtime_grok_runner_router_import_surface_is_complete() -> None:
    """Every statically declared router dependency remains importable."""

    source_path = Path(grok_cli_runner.__file__)
    syntax = ast.parse(source_path.read_text(encoding="utf-8"))
    required = {
        alias.name
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom) and node.module == "ipfs_accelerate_py.llm_router"
        for alias in node.names
    }

    assert required
    assert sorted(name for name in required if not hasattr(llm_router, name)) == []


def test_transient_preflight_retry_export_has_reviewed_signature() -> None:
    """Keep the reviewed positional and keyword-only API boundary exact."""

    assert str(
        inspect.signature(
            llm_router.retryable_agent_implementation_preflight_failure
        )
    ) == (
        "(stderr_text: 'str', receipt: 'Mapping[str, object]', *, nonce: 'str', "
        "model: 'str', probe_returncode: 'int') -> 'bool'"
    )



def test_transient_preflight_retry_uses_current_validated_receipt() -> None:
    """The legacy export stays 4.5 while current 4.6 receipts stay exact."""

    nonce = "a" * 64
    evidence = "Error: max turns reached\n"
    receipt = build_grok_failure_receipt(
        probe_stderr_text=evidence,
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )

    assert llm_router.AGENT_IMPLEMENTATION_PRIMARY_MODEL_ID == "grok-4.6"
    assert GROK_QUOTA_PROBE_CONTRACT["model"] == "grok-4.5"
    assert (
        llm_router._LEGACY_AGENT_IMPLEMENTATION_ROUTE.primary_model_id
        == "grok-4.5"
    )
    assert "grok45" in llm_router._LEGACY_AGENT_IMPLEMENTATION_ROUTE.route_id
    assert (
        llm_router._AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE.primary_model_id
        == "grok-4.5"
    )
    assert (
        llm_router._GROK46_QUOTA_HIGH_AGENT_IMPLEMENTATION_ROUTE.primary_model_id
        == "grok-4.6"
    )
    assert receipt["probe_contract_id"] == (
        llm_router._AGENT_IMPLEMENTATION_PROBE_CONTRACT_IDS["grok-4.6"]
    )
    assert valid_grok_failure_receipt(
        receipt,
        nonce=nonce,
        model="grok-4.6",
        returncode=41,
    )
    assert llm_router.retryable_agent_implementation_preflight_failure(
        evidence,
        receipt,
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
    )
    assert not llm_router.retryable_agent_implementation_preflight_failure(
        evidence.rstrip("\n"),
        receipt,
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
    )
def test_current_runner_typed_preflight_reaches_retry_without_a_model_call(
    tmp_path: Path,
) -> None:
    """The exact runner path retries a local deterministic probe, not a model."""

    source_home = tmp_path / "source-home"
    source_home.mkdir()
    invocation_log = tmp_path / "invocations.jsonl"
    fake_grok = tmp_path / "grok"
    fake_grok.write_text(
        "#!/usr/bin/python3\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        f"log_path = Path({str(invocation_log)!r})\n"
        "record = {\n"
        "    'argv': sys.argv[1:],\n"
        "    'openai_key_present': bool(os.environ.get('OPENAI_API_KEY')),\n"
        "    'xai_key': os.environ.get('XAI_API_KEY'),\n"
        "    'codex_compat': os.environ.get('GROK_CODEX_AGENTS_ENABLED'),\n"
        "}\n"
        "with log_path.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps(record, sort_keys=True) + '\\n')\n"
        "sys.stderr.write('Error: max turns reached\\n')\n"
        "raise SystemExit(41)\n",
        encoding="utf-8",
    )
    fake_grok.chmod(0o700)

    returncode, receipt, overflow = grok_cli_runner._run_typed_grok_preflight(
        grok_bin=str(fake_grok),
        base_env={
            "HOME": str(source_home),
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "OPENAI_API_KEY": "must-not-cross-the-isolation-boundary",
            "XAI_API_KEY": "fake-local-grok-authority",
        },
        nonce="a" * 64,
    )

    assert returncode == 41
    assert receipt["failure_class"] == "unknown"
    assert receipt["evidence_size"] == 25
    assert overflow is False
    invocations = [
        json.loads(line)
        for line in invocation_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(invocations) == 2
    for invocation in invocations:
        assert invocation["openai_key_present"] is False
        assert invocation["xai_key"] == "fake-local-grok-authority"
        assert invocation["codex_compat"] == "0"
        assert invocation["argv"][0:2] == ["--model", "grok-4.6"]
        assert invocation["argv"][2:4] == ["--max-turns", "1"]
        assert invocation["argv"][-2:] == [
            "--disallowed-tools",
            grok_cli_runner._SEALED_GROK_DISALLOWED_TOOLS,
        ]
