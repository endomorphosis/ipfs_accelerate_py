"""Bounded static checks for the supervisor's intent adapter contracts.

Reads only named Python files. Never imports the target checkout, loads a
provider, opens a database, or grants completion/repair authority.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

BASE = "ipfs_accelerate_py/agent_supervisor/task_sources/"
PROVIDER = BASE + "intent_repository.py"
CONSUMERS = (
    BASE + "database_task_source.py",
    BASE + "typed_state_owner.py",
    "test/api/test_agent_supervisor_intent_repository.py",
    "test/api/causal_federation/test_admitted_executor.py",
)


def inspect_runtime_contracts(checkout_root: str | Path) -> dict:
    root = Path(checkout_root).resolve(strict=True)
    findings, inputs, trees = [], [], {}
    for relative in (PROVIDER, *CONSUMERS):
        path = root / relative
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError("path escapes checkout")
            if path.stat().st_size > 4_000_000:
                raise ValueError("source exceeds static-check bound")
            raw = path.read_bytes()
            trees[relative] = ast.parse(raw, filename=relative)
            inputs.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
        except (OSError, ValueError, SyntaxError) as exc:
            findings.append(
                {
                    "path": relative,
                    "reason_code": "contract_source_unavailable",
                    "error_type": type(exc).__name__,
                }
            )
    provider = trees.get(PROVIDER)
    exports, methods = set(), set()
    if provider is not None:
        for node in provider.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exports.add(node.name)
                if isinstance(node, ast.ClassDef) and node.name == "IntentRepository":
                    methods.update(
                        item.name
                        for item in node.body
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    )
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                exports.add(node.target.id)
            elif isinstance(node, ast.Assign):
                exports.update(target.id for target in node.targets if isinstance(target, ast.Name))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                exports.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        for relative in CONSUMERS:
            tree = trees.get(relative)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and (node.module or "").split(".")[-1] == "intent_repository"
                ):
                    for alias in node.names:
                        if alias.name not in exports:
                            findings.append(
                                {
                                    "path": relative,
                                    "line": node.lineno,
                                    "reason_code": "missing_intent_export",
                                    "symbol": alias.name,
                                }
                            )
                if relative == BASE + "database_task_source.py" and isinstance(node, ast.Call):
                    target = node.func
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Attribute)
                        and isinstance(target.value.value, ast.Name)
                        and target.value.value.id == "self"
                        and target.value.attr == "_intent"
                        and target.attr not in methods
                    ):
                        findings.append(
                            {
                                "path": relative,
                                "line": node.lineno,
                                "reason_code": "missing_intent_method",
                                "symbol": target.attr,
                            }
                        )
    # Integration-critical call signatures are checked without executing the
    # checkout. Keep this list explicit so this doctor remains bounded.
    bridge_path = "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py"
    try:
        path = root / bridge_path
        if not path.resolve().is_relative_to(root) or path.stat().st_size > 4_000_000:
            raise ValueError("bridge source outside bounds")
        raw = path.read_bytes()
        bridge_tree = ast.parse(raw, filename=bridge_path)
        inputs.append({"path": bridge_path, "sha256": hashlib.sha256(raw).hexdigest()})
        bridge = next(
            (
                n
                for n in bridge_tree.body
                if isinstance(n, ast.ClassDef) and n.name == "DatabasePortalExecutionBridge"
            ),
            None,
        )
        init = (
            next(
                (n for n in bridge.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
                None,
            )
            if bridge
            else None
        )
        arguments = {n.arg for n in (*init.args.args, *init.args.kwonlyargs)} if init else set()
        if "max_task_attempts" not in arguments:
            findings.append(
                {
                    "path": bridge_path,
                    "reason_code": "missing_bridge_attempt_budget_contract",
                    "symbol": "DatabasePortalExecutionBridge.__init__.max_task_attempts",
                }
            )
    except (OSError, ValueError, SyntaxError) as exc:
        findings.append(
            {
                "path": bridge_path,
                "reason_code": "contract_source_unavailable",
                "error_type": type(exc).__name__,
            }
        )
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/doctor-runtime-contracts@1",
        "status": "failed" if findings else "passed",
        "scope": "static_intent_exports_adapter_methods_and_bridge_attempt_budget",
        "findings": sorted(
            findings, key=lambda item: (item["path"], item.get("line", 0), item.get("symbol", ""))
        ),
        "inputs": inputs,
        "automatic_repair_attempted": False,
        "behavioral_tests_run": False,
        "full_system_qualified": False,
    }
