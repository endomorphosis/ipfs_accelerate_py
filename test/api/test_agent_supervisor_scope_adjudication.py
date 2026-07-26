from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
)
from ipfs_accelerate_py.agent_supervisor.scope_adjudication import (
    ScopeAdjudicationReceipt,
    ScopeExpansionReason,
    adjudicate_scope_expansion,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
)


def _entry(
    path: str,
    before: str | None,
    after: str,
    *,
    change_kind: DiffChangeKind = DiffChangeKind.MODIFY,
) -> CandidateDiffEntry:
    return CandidateDiffEntry(
        old_path="" if change_kind is DiffChangeKind.ADD else path,
        new_path=path,
        change_kind=change_kind,
        before_source=before,
        after_source=after,
    )


def _adjudicate(
    entries: tuple[CandidateDiffEntry, ...],
    *,
    scope: tuple[str, ...] = ("pkg/compiler.py",),
    finding_codes: tuple[str, ...] = ("path_outside_scope",),
    validation_commands: tuple[str, ...] = (),
    max_expansion_paths: int = 8,
    workspace_path: Path | None = None,
):
    return adjudicate_scope_expansion(
        task_id="ASI-TEST",
        proposal_id="proposal:test",
        initial_policy_id="policy:test",
        repository_id="repository:test",
        repository_tree_id="tree:test",
        baseline_id="tree:test",
        original_scope_paths=scope,
        candidate_diff=entries,
        initial_finding_codes=finding_codes,
        validation_commands=validation_commands,
        max_expansion_paths=max_expansion_paths,
        workspace_path=workspace_path,
    )


def test_declared_import_dependency_justifies_contract_companion() -> None:
    receipt = _adjudicate(
        (
            _entry(
                "pkg/compiler.py",
                "def compile_text():\n    return None\n",
                "from .contracts import Contract\n"
                "def compile_text():\n    return Contract()\n",
            ),
            _entry(
                "pkg/contracts.py",
                "class Contract:\n    pass\n",
                "class Contract:\n    kind = 'legal-ir'\n",
            ),
        )
    )

    assert receipt.justified is True
    assert receipt.accepted is False
    assert receipt.justified_paths == ("pkg/contracts.py",)
    assert receipt.authorized_paths == ()
    assert receipt.decisions[0].reason_codes == (
        ScopeExpansionReason.DECLARED_PATH_IMPORTS_CANDIDATE,
    )
    restored = ScopeAdjudicationReceipt.from_dict(receipt.to_record())
    assert restored == receipt
    assert restored.receipt_id == receipt.receipt_id
    forged = receipt.to_record()
    forged["decisions"] = []
    with pytest.raises(ValueError, match="every undeclared candidate path"):
        ScopeAdjudicationReceipt.from_dict(forged)


def test_non_weakened_regression_test_justifies_companion_test() -> None:
    before_test = (
        "from pkg.compiler import compile_text\n\n"
        "def test_compile_text():\n"
        "    assert compile_text('shall') == 'old'\n"
    )
    after_test = before_test.replace("'old'", "'new'")
    receipt = _adjudicate(
        (
            _entry(
                "pkg/compiler.py",
                "def compile_text(text):\n    return 'old'\n",
                "def compile_text(text):\n    return 'new'\n",
            ),
            _entry(
                "test/test_compiler.py",
                before_test,
                after_test,
            ),
        )
    )

    assert receipt.justified is True
    assert receipt.decisions[0].reason_codes == (
        ScopeExpansionReason.REGRESSION_TEST_IMPORTS_DECLARED_PATH,
    )
    assert receipt.decisions[0].evidence_paths == ("pkg/compiler.py",)


def test_explicit_validation_target_justifies_guarded_integration_test() -> None:
    receipt = _adjudicate(
        (
            _entry(
                "pkg/compiler.py",
                "FORMAT = 'v1'\n",
                "FORMAT = 'v2'\n",
            ),
            _entry(
                "test/test_cli.py",
                "import subprocess\n\n"
                "def test_cli():\n"
                "    assert subprocess.run(['legal-ir']).returncode == 1\n",
                "import subprocess\n\n"
                "def test_cli():\n"
                "    assert subprocess.run(['legal-ir']).returncode == 0\n",
            ),
        ),
        validation_commands=(
            "python -m pytest test/test_cli.py",
        ),
    )

    assert receipt.justified is True
    assert receipt.decisions[0].reason_codes == (
        ScopeExpansionReason.EXPLICIT_VALIDATION_TARGET,
    )


def test_changed_golden_fixture_cannot_claim_validation_target() -> None:
    receipt = _adjudicate(
        (
            _entry(
                "pkg/compiler.py",
                "FORMAT = 'v1'\n",
                "FORMAT = 'v2'\n",
            ),
            _entry(
                "test/fixtures/legal-ir.case",
                "v1\n",
                "v2\n",
            ),
        ),
        validation_commands=(
            "python -m pytest test/fixtures/legal-ir.case",
        ),
    )

    assert receipt.accepted is False
    assert receipt.decisions[0].reason_codes == (
        ScopeExpansionReason.TEST_CHANGE_UNVERIFIABLE,
    )


def test_unrelated_same_package_path_remains_denied() -> None:
    receipt = _adjudicate(
        (
            _entry(
                "pkg/compiler.py",
                "MODE = 'old'\n",
                "MODE = 'new'\n",
            ),
            _entry(
                "pkg/unrelated.py",
                "SETTING = False\n",
                "SETTING = True\n",
            ),
        )
    )

    assert receipt.accepted is False
    assert receipt.denied_paths == ("pkg/unrelated.py",)
    assert receipt.decisions[0].reason_codes == (
        ScopeExpansionReason.NO_DEPENDENCY_EVIDENCE,
    )


def test_package_import_closure_justifies_lazy_provider_fix(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text(
        "from .runtime import runtime\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "runtime.py").write_text(
        "from .provider_bridge import provider\nruntime = provider\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "provider_bridge.py").write_text(
        "import importlib\nprovider = None\n"
        "def load_provider():\n"
        "    return importlib.import_module('optional_provider')\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "control.py").write_text(
        "def status():\n    return 'ready'\n",
        encoding="utf-8",
    )
    test_source = (
        "from pkg.control import status\n\n"
        "def test_status():\n"
        "    assert status() == 'ready'\n"
    )
    (tmp_path / "test" / "test_control.py").write_text(
        test_source,
        encoding="utf-8",
    )

    receipt = _adjudicate(
        (
            _entry(
                "pkg/control.py",
                "def status():\n    return 'old'\n",
                "def status():\n    return 'ready'\n",
            ),
            _entry(
                "test/test_control.py",
                test_source.replace("'ready'", "'old'"),
                test_source,
            ),
            _entry(
                "pkg/provider_bridge.py",
                "import optional_provider\nprovider = optional_provider\n",
                (tmp_path / "pkg" / "provider_bridge.py").read_text(
                    encoding="utf-8"
                ),
            ),
        ),
        scope=("pkg/control.py", "test/test_control.py"),
        workspace_path=tmp_path,
    )

    assert receipt.justified is True
    assert receipt.justified_paths == ("pkg/provider_bridge.py",)
    assert receipt.decisions[0].reason_codes == (
        (
            ScopeExpansionReason
            .DECLARED_PATH_TRANSITIVELY_IMPORTS_CANDIDATE
        ),
    )
    assert receipt.decisions[0].evidence_paths == (
        "pkg/__init__.py",
        "pkg/provider_bridge.py",
        "pkg/runtime.py",
        "test/test_control.py",
    )


def test_test_weakening_is_denied_even_with_direct_import() -> None:
    receipt = _adjudicate(
        (
            _entry(
                "pkg/compiler.py",
                "def compile_text(text):\n    return text\n",
                "def compile_text(text):\n    return text.strip()\n",
            ),
            _entry(
                "test/test_compiler.py",
                "from pkg.compiler import compile_text\n\n"
                "def test_compile_text():\n"
                "    assert compile_text(' x ') == ' x '\n",
                "from pkg.compiler import compile_text\n\n"
                "def test_compile_text():\n"
                "    compile_text(' x ')\n",
            ),
        )
    )

    assert receipt.accepted is False
    assert receipt.decisions[0].reason_codes == (
        ScopeExpansionReason.TEST_WEAKENING,
    )


def test_non_scope_gate_and_expansion_limit_fail_closed() -> None:
    entries = (
        _entry(
            "pkg/compiler.py",
            "MODE = 'old'\n",
            "MODE = 'new'\n",
        ),
        _entry("pkg/contracts.py", "X = 1\n", "X = 2\n"),
    )

    mixed_failure = _adjudicate(
        entries,
        finding_codes=("path_outside_scope", "binary_change_forbidden"),
    )
    two_extras = _adjudicate(
        (
            *entries,
            _entry("pkg/second.py", "Y = 1\n", "Y = 2\n"),
        ),
        max_expansion_paths=1,
    )

    assert mixed_failure.accepted is False
    assert mixed_failure.decisions[0].reason_codes == (
        ScopeExpansionReason.INITIAL_GATE_NOT_SCOPE_ONLY,
    )
    assert two_extras.accepted is False
    assert {
        decision.reason_codes
        for decision in two_extras.decisions
    } == {(ScopeExpansionReason.EXPANSION_LIMIT_EXCEEDED,)}


class _PassingValidationScheduler:
    def run_validated(self, *_args, **_kwargs):
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
        }


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


def test_daemon_revalidates_justified_expansion_and_exposes_receipt(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "test").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    compiler_path = repo / "pkg" / "compiler.py"
    test_path = repo / "test" / "test_compiler.py"
    compiler_path.write_text(
        "def compile_text(text):\n    return text\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "from pkg.compiler import compile_text\n\n"
        "def test_compile_text():\n"
        "    assert compile_text('shall') == 'shall'\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "supervisor@example.invalid")
    _git(repo, "config", "user.name", "Supervisor Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")

    compiler_path.write_text(
        "def compile_text(text):\n    return text.strip()\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "from pkg.compiler import compile_text\n\n"
        "def test_compile_text():\n"
        "    assert compile_text(' shall ') == 'shall'\n",
        encoding="utf-8",
    )
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        validation_scheduler=_PassingValidationScheduler(),
        worktree_pool_enabled=False,
    )
    task = PortalTask(
        task_id="ASI-TEST",
        title="Compile normalized legal text",
        status="todo",
        completion="manual",
        priority="P1",
        track="quality",
        outputs=["pkg/compiler.py"],
        validation=[
            "python -m pytest test/test_compiler.py",
        ],
    )

    proposal_validation = daemon._validate_implementation_patch(
        repo,
        task,
        baseline_ref=baseline,
    )

    assert proposal_validation.accepted is True
    assert proposal_validation.policy.policy_version.endswith(
        "+scope-adjudication-v1"
    )
    validation_result = daemon._run_validation_commands(
        repo,
        task,
        repo / "validation.log",
        proposal_validation=proposal_validation,
    )
    assert validation_result["passed"] is True
    assert validation_result["scope_adjudication"]["accepted"] is True
    assert validation_result["scope_adjudication"][
        "authorized_policy_id"
    ] == proposal_validation.policy.policy_id
    assert validation_result["scope_adjudication"]["authorized_paths"] == [
        "test/test_compiler.py"
    ]
    events = [
        json.loads(line)
        for line in (repo / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [
        event["type"]
        for event in events
        if event["type"].startswith("implementation_")
    ] == [
        "implementation_scope_adjudicated",
        "implementation_proposal_validated",
    ]


def test_daemon_keeps_unrelated_expansion_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    compiler_path = repo / "pkg" / "compiler.py"
    unrelated_path = repo / "pkg" / "unrelated.py"
    compiler_path.write_text("MODE = 'old'\n", encoding="utf-8")
    unrelated_path.write_text("SETTING = False\n", encoding="utf-8")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "supervisor@example.invalid")
    _git(repo, "config", "user.name", "Supervisor Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")
    compiler_path.write_text("MODE = 'new'\n", encoding="utf-8")
    unrelated_path.write_text("SETTING = True\n", encoding="utf-8")
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        worktree_pool_enabled=False,
    )
    task = PortalTask(
        task_id="ASI-TEST",
        title="Change compiler mode",
        status="todo",
        completion="manual",
        priority="P1",
        track="quality",
        outputs=["pkg/compiler.py"],
        validation=["python -m pytest test/test_compiler.py"],
    )

    result = daemon._validate_implementation_patch(
        repo,
        task,
        baseline_ref=baseline,
    )

    assert result.accepted is False
    assert {
        finding.code.value for finding in result.findings
    } == {"path_outside_scope"}
    receipt = next(iter(daemon._implementation_scope_adjudications.values()))
    assert receipt.denied_paths == ("pkg/unrelated.py",)
    assert receipt.decisions[0].reason_codes == (
        ScopeExpansionReason.NO_DEPENDENCY_EVIDENCE,
    )


def test_merge_binding_rejects_detached_or_forged_scope_receipt() -> None:
    valid = {
        "passed": True,
        "selection": {"scope": "pre_merge"},
        "proposal_gate": {
            "proposal_id": "proposal:one",
            "policy_id": "policy:expanded",
            "repository_tree_id": "tree:baseline",
            "changed_paths": ["pkg/compiler.py", "pkg/contracts.py"],
        },
        "scope_adjudication": {
            "accepted": True,
            "receipt_id": "receipt:one",
            "proposal_id": "proposal:one",
            "authorized_policy_id": "policy:expanded",
            "repository_tree_id": "tree:baseline",
            "authorized_paths": ["pkg/contracts.py"],
            "denied_paths": [],
            "proof_authoritative": False,
            "completion_authoritative": False,
        },
    }
    check = (
        PortalImplementationDaemon
        ._scope_adjudication_merge_binding_error
    )

    assert check(valid) == ""
    mutations = {
        "scope_adjudication_proposal_mismatch": (
            "proposal_id",
            "proposal:other",
        ),
        "scope_adjudication_policy_mismatch": (
            "authorized_policy_id",
            "policy:other",
        ),
        "scope_adjudication_authority_forged": (
            "proof_authoritative",
            True,
        ),
        "scope_adjudication_paths_mismatch": (
            "authorized_paths",
            ["pkg/unrelated.py"],
        ),
    }
    for expected, (field_name, value) in mutations.items():
        candidate = json.loads(json.dumps(valid))
        candidate["scope_adjudication"][field_name] = value
        assert check(candidate) == expected
