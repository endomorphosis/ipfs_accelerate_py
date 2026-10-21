"""Run pytest against this checkout without plugin autoload.

The lane invokes this file with python3. It switches to the datasets
virtualenv when that interpreter is present, and it never launches an agent.
Successful validation requires a passing test or verified reuse of a pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _reused_pass(item: Any, report: Any) -> bool:
    """Recognize pass reuse from its verifier state, never its skip text alone."""

    longrepr = getattr(report, "longrepr", None)
    if not isinstance(longrepr, tuple) or len(longrepr) != 3:
        return False
    reason = str(longrepr[2]).removeprefix("Skipped: ")
    nodeid = str(item.nodeid)
    try:
        if reason == "pytest AST seal matches":
            state = getattr(item.config, "_pytest_ast_seal_state", None)
            if not isinstance(state, dict) or not state.get("enabled"):
                return False
            sealed = state.get("seals", {}).get(nodeid)
            return bool(
                sealed is not None
                and sealed.complete
                and state["oracle"].reuse(nodeid, sealed.digest)
            )
        if reason == "pytest-item-ledger reuse":
            # API conftest loads this plugin under an isolated package name.
            plugin = sys.modules.get("aseh_board_pytest_reuse.pytest_item_ledger_plugin")
            if plugin is None:
                plugin = sys.modules.get(
                    "ipfs_accelerate_py.agent_supervisor.runtime.pytest_item_ledger_plugin"
                )
            state = getattr(plugin, "_STATE", {})
            if not state:
                return False
            workspace = state["workspace"]
            context = state["context"]
            relative = plugin._relative_test_file(workspace, item)
            digest = plugin.file_sha256(workspace / relative)
            reusable = plugin.reusable_nodeids(
                plugin.load_records(context["dest"], workspace=workspace),
                command_sha256=state["command_sha256"],
                workspace_fingerprint_value=context["workspace_fingerprint"],
                test_file_hashes={relative: digest},
            )
            return nodeid in reusable
        if reason.startswith("proof-cache-hit:"):
            plugin = sys.modules.get("ipfs_accelerate_py.testing.proof_reuse.plugin")
            lookup_module = sys.modules.get("ipfs_accelerate_py.testing.proof_reuse.lookup")
            contracts = sys.modules.get(
                "ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts"
            )
            decision = getattr(item, "_ipfs_proof_reuse_decision", None)
            decision_type = getattr(contracts, "ReuseDecision", None)
            if decision_type is None or not isinstance(decision, decision_type):
                return False
            config = item.config
            if (
                plugin is None
                or lookup_module is None
                or not config.pluginmanager.is_registered(plugin)
                or not plugin.get_proof_reuse_config(config).may_skip
            ):
                return False
            coordinator = getattr(config, plugin.COORDINATOR_ATTRIBUTE, None)
            lookup = getattr(config, plugin.LOOKUP_SERVICE_ATTRIBUTE, None)
            if (
                not getattr(coordinator, "can_skip", False)
                or not isinstance(lookup, lookup_module.ProofReuseLookup)
            ):
                return False
            request = lookup_module._request_from_value(item)
            if request is None or request.item is not item:
                return False
            claimed = decision_type.from_dict(decision.to_dict())
            if not claimed.is_skip or reason != f"proof-cache-hit:{claimed.certificate_cid}":
                return False
            # A typed item attribute is only a claim. Re-run the configured,
            # bounded verifier against retained bytes and the current request.
            kwargs = {
                "eligibility": request.eligibility,
                "current_policy": request.current_policy,
                "now_ms": request.now_ms,
            }
            if isinstance(lookup, lookup_module.ProofReuseTwoStageLookup):
                kwargs["item"] = item
            verified = lookup.lookup(request.locator, request.execution_key, **kwargs)
            return bool(
                isinstance(verified, decision_type)
                and verified.is_skip
                and verified.receipt_cid == claimed.receipt_cid
                and verified.certificate_cid == claimed.certificate_cid
            )
    except Exception:
        # Missing/corrupt proof never turns an ordinary skip into validation.
        return False
    return False


class _ValidationEvidence:
    def __init__(self) -> None:
        self.passed: set[str] = set()
        self.reused: set[str] = set()

    def pytest_runtest_makereport(self, item: Any, call: Any):
        del call
        outcome = yield
        report = outcome.get_result()
        if report.when == "call" and report.passed:
            self.passed.add(str(item.nodeid))
        elif report.skipped and _reused_pass(item, report):
            self.reused.add(str(item.nodeid))

    pytest_runtest_makereport.pytest_impl = {"hookwrapper": True}


def _run_pytest(args: list[str]) -> int:
    # Import pytest only after switching to the selected interpreter.
    import pytest

    evidence = _ValidationEvidence()
    result = int(pytest.main(args, plugins=[evidence]))
    if result == 0 and not (evidence.passed or evidence.reused):
        print(
            "Validation failed: no tests passed and no verified passes were reused.",
            file=sys.stderr,
        )
        return int(pytest.ExitCode.NO_TESTS_COLLECTED)
    return result


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[5]
    python = root / "external/ipfs_datasets/.venv/bin/python"
    if not python.is_file():
        python = Path(sys.executable)
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = str(root / "external/ipfs_accelerate")
    bootstrap = (
        "import runpy, sys; "
        "runner = runpy.run_path(sys.argv.pop(1)); "
        "raise SystemExit(runner['_run_pytest'](sys.argv[1:]))"
    )
    os.execve(
        str(python),
        [str(python), "-c", bootstrap, str(Path(__file__).resolve()), *args],
        env,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
