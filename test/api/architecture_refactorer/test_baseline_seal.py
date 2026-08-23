"""Hermetic PCAR-000 current-tree baseline, gitlink, prerequisite, and ledger seal."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    CONTROL_MUTATION_AUDIT_RECEIPT_SCHEMA,
    CONTROL_OPERATION_CATALOG_SCHEMA,
    CONTROL_PROPOSAL_AUDIT_RECEIPT_SCHEMA,
    CONTROL_QUERY_AUDIT_RECEIPT_SCHEMA,
    OPERATION_CATALOG_V2,
    OPERATION_CATALOG_V2_REQUIREMENT_ID,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ASSURANCE_ASSESSMENT_SCHEMA,
    CODE_PROOF_OBLIGATION_SCHEMA,
    PROOF_ATTEMPT_SCHEMA,
    PROOF_EVIDENCE_SCHEMA,
    PROOF_PLAN_SCHEMA,
    PROOF_PLAN_STEP_SCHEMA,
    PROOF_RECEIPT_SCHEMA,
    RESOURCE_BUDGET_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    SIGNED_TEST_PASS_RECEIPT_V2_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.receipts import (
    RECEIPT_INDEX_SCHEMA,
    RECEIPT_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    probe_quack_capabilities,
)

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "docs/architecture/architecture_refactorer_inventory"
BASELINE_PATH = INVENTORY / "sealed_current_tree_baseline.json"
MATRIX_PATH = INVENTORY / "sealed_prerequisite_matrix.json"
BOOTSTRAP_LEDGER_PATH = INVENTORY / "qualified_tests.json"
GITMODULES_PATH = ROOT / ".gitmodules"
PYPROJECT_PATH = ROOT / "pyproject.toml"
WORKFLOW_PATH = ROOT / ".github/workflows/documentation-gates.yml"

SEALED_COMMIT = "a2d1529934197dc64fe18cfbaec9dc7daf438703"
STARTING_COMMIT = "bbf7f68799072c2b81f7d96eac91f2df3c4b3952"
STARTING_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
MERGE_TARGET_BRANCH = "codex/proof-carrying-architecture-refactorer-v1"
REQUIRED_GITLINKS = {
    "ipfs_datasets_py": "66a02063496fd200f2372b3083e376f1978c6be1",
    "ipfs_kit_py": "2564aea1ae35061f2165872aff91e8a40801ab7e",
    "ipfs_accelerate_py/mcplusplus": "5ac0ab162f420264fd224073a5df3f2d7c054ae3",
}
ALLOWED_STATUSES = (
    "available",
    "available_with_caveats",
    "stale",
    "incompatible",
    "missing",
)
EXECUTION_STATUSES = ("pass", "fail", "skip", "not-run")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed.stdout.strip()


def _git_ok(*args: str) -> bool:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _peel_tree(spec: str) -> str:
    peeled = _git("rev-parse", spec)
    kind = _git("cat-file", "-t", peeled)
    if kind == "tree":
        return peeled
    if kind == "commit":
        return _git("rev-parse", f"{peeled}^{{tree}}")
    raise AssertionError(f"{spec!r} peels to {kind}, not a commit or tree")


def _gitlink_sha(path: str) -> str:
    raw = _git("ls-tree", "HEAD", path)
    mode, kind, rest = raw.split(None, 2)
    sha, name = rest.split("\t", 1)
    assert mode == "160000", path
    assert kind == "commit", path
    assert name == path
    assert SHA1_RE.fullmatch(sha)
    return sha


def _checkout_head(path: str) -> str | None:
    gitfile = ROOT / path / ".git"
    if not gitfile.exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(ROOT / path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _gitmodules_paths() -> list[str]:
    paths: list[str] = []
    for line in GITMODULES_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("path = "):
            paths.append(stripped.split(" = ", 1)[1])
    return paths


def _class_line(path: Path, class_name: str) -> int | None:
    pattern = re.compile(rf"^class {re.escape(class_name)}\b")
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern.match(line):
            return index
    return None


def _class_present(class_name: str) -> bool:
    pattern = re.compile(rf"^class {re.escape(class_name)}\b", re.M)
    root = ROOT / "ipfs_accelerate_py" / "agent_supervisor"
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if pattern.search(text):
            return True
    return False


def _pyproject_version() -> str:
    match = re.search(r'^version = "([^"]+)"', PYPROJECT_PATH.read_text(encoding="utf-8"), re.M)
    assert match is not None
    return match.group(1)


def _blob(path: str) -> str:
    return _git("rev-parse", f"HEAD:{path}")


def _split_source(source: str) -> tuple[str, int]:
    path, line = source.rsplit(":", 1)
    return path, int(line)


def test_sealed_manifests_are_canonical_json() -> None:
    for path in (BASELINE_PATH, MATRIX_PATH):
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        assert raw == _canonical(payload)
        assert payload["authority"] is False
        assert payload["task_id"] == "PCAR-000"


def test_source_tree_seal() -> None:
    baseline = _load(BASELINE_PATH)
    repository = baseline["repository"]
    head = _git("rev-parse", "HEAD")
    # PCAR-000 seals the inspected pre-implementation tree.  Later accepted
    # tasks must remain descendants of that commit; requiring every later
    # HEAD to have the same tree would make the baseline self-invalidating.
    sealed_tree = _peel_tree(repository["tree"])
    sealed_commit_tree = _peel_tree(repository["commit"])

    assert baseline["schema"] == (
        "ipfs_accelerate_py.agent_supervisor.architecture-refactorer"
        ".sealed-current-tree-baseline@1"
    )
    assert repository["commit"] == SEALED_COMMIT
    assert repository["starting_commit"] == STARTING_COMMIT
    assert repository["starting_tree"] == STARTING_TREE
    assert repository["starting_commit_is_ancestor"] is True
    assert repository["merge_target_branch"] == MERGE_TARGET_BRANCH
    assert repository["origin"] == "https://github.com/endomorphosis/ipfs_accelerate_py"
    assert SHA1_RE.fullmatch(head)
    assert SHA1_RE.fullmatch(sealed_tree)
    assert sealed_commit_tree == sealed_tree
    assert _git_ok("merge-base", "--is-ancestor", STARTING_COMMIT, "HEAD")
    assert _git_ok("merge-base", "--is-ancestor", SEALED_COMMIT, "HEAD")
    assert _git("rev-parse", f"{STARTING_COMMIT}^{{tree}}") == STARTING_TREE
    assert head == SEALED_COMMIT or _git_ok(
        "merge-base", "--is-ancestor", SEALED_COMMIT, head
    )

    protection = baseline["branch_protection"]
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert protection["live_github_query_status"] == "not-run"
    assert protection["live_github_query_reason"] == (
        "pcar_000_external_effect_scope_network_deny"
    )
    assert protection["planning_record_is_not_authority"] is True
    assert protection["local_required_check_name"] == "documentation-gates"
    assert protection["local_workflow_path"] == (
        ".github/workflows/documentation-gates.yml"
    )
    assert "name: Documentation Gates" in workflow
    assert "name: documentation-gates" in workflow
    assert protection["required_status_checks_from_local_workflow"] == [
        "documentation-gates"
    ]

    environment = baseline["environment"]
    assert environment["canonical_python_interpreter"] == "/usr/bin/python3.12"
    assert environment["canonical_python_major_minor"] == [3, 12]
    assert sys.version_info[:2] == (3, 12)
    assert environment["package_version"] == _pyproject_version()
    assert environment["package_version"] == "0.0.45"
    assert environment["ducklake"]["gates_readiness_acceptance_completion_or_release"] is False
    quack = environment["quack"]
    assert quack["allow_network_install"] is False
    assert quack["claimed_without_probe"] is False
    assert quack["planning_compatible_claim_is_not_authority"] is True
    report = probe_quack_capabilities(allow_network_install=False, use_cache=False)
    assert report.network_install_attempted is False
    assert report.network_install_allowed is False

    catalog = baseline["operation_catalog"]
    assert catalog["schema"] == CONTROL_OPERATION_CATALOG_SCHEMA
    assert catalog["requirement_id"] == OPERATION_CATALOG_V2_REQUIREMENT_ID
    assert catalog["operation_count"] == len(OPERATION_CATALOG_V2)
    assert catalog["operation_count"] == 35
    assert catalog["content_id"] == OPERATION_CATALOG_V2.content_id
    assert catalog["content_id"] == OPERATION_CATALOG_V2.catalog_id
    assert catalog["source"] == (
        "ipfs_accelerate_py/agent_supervisor/control/control_contracts.py"
    )

    assert baseline["proof_schemas"] == [
        CODE_PROOF_OBLIGATION_SCHEMA,
        PROOF_PLAN_SCHEMA,
        PROOF_PLAN_STEP_SCHEMA,
        PROOF_ATTEMPT_SCHEMA,
        PROOF_RECEIPT_SCHEMA,
        PROOF_EVIDENCE_SCHEMA,
        RESOURCE_BUDGET_SCHEMA,
        ASSURANCE_ASSESSMENT_SCHEMA,
    ]
    assert baseline["receipt_schemas"] == [
        RECEIPT_SCHEMA,
        RECEIPT_INDEX_SCHEMA,
        CONTROL_QUERY_AUDIT_RECEIPT_SCHEMA,
        CONTROL_PROPOSAL_AUDIT_RECEIPT_SCHEMA,
        CONTROL_MUTATION_AUDIT_RECEIPT_SCHEMA,
        SIGNED_TEST_PASS_RECEIPT_V2_SCHEMA,
    ]


def test_exact_gitlinks() -> None:
    baseline = _load(BASELINE_PATH)
    sealed = baseline["gitlinks"]
    declared = _gitmodules_paths()
    assert set(sealed) == set(declared)
    for path in declared:
        record = sealed[path]
        gitlink = _gitlink_sha(path)
        checkout = _checkout_head(path)
        assert record["gitlink"] == gitlink
        assert SHA1_RE.fullmatch(record["gitlink"])
        assert record["initialized"] is bool(checkout)
        if checkout is None:
            assert record["checkout_matches_gitlink"] is False
            assert "checkout_head" not in record
        else:
            assert record["checkout_head"] == checkout
            assert record["checkout_matches_gitlink"] is (checkout == gitlink)

    for path, pin in REQUIRED_GITLINKS.items():
        record = sealed[path]
        assert record["required_pin"] is True
        assert record["gitlink"] == pin
        assert record["initialized"] is True
        assert record["checkout_head"] == pin
        assert record["checkout_matches_gitlink"] is True

    assert _peel_tree(baseline["repository"]["tree"]) == _git(
        "rev-parse", f"{SEALED_COMMIT}^{{tree}}"
    )


def test_current_prerequisite_matrix() -> None:
    matrix = _load(MATRIX_PATH)
    baseline = _load(BASELINE_PATH)
    assert matrix["schema"] == (
        "ipfs_accelerate_py.agent_supervisor.architecture-refactorer"
        ".sealed-prerequisite-matrix@1"
    )
    assert matrix["allowed_statuses"] == list(ALLOWED_STATUSES)
    assert matrix["repository_commit"] == baseline["repository"]["commit"]
    assert _peel_tree(matrix["repository_tree"]) == _peel_tree(
        baseline["repository"]["tree"]
    )
    assert _peel_tree(matrix["repository_tree"]) == _git(
        "rev-parse", f"{SEALED_COMMIT}^{{tree}}"
    )

    names = [item["name"] for item in matrix["prerequisites"]]
    assert names == [
        "SemanticCompressionHarness",
        "SemanticCompressionGovernor",
        "AdversarialAssuranceEngine",
        "IncrementalVerificationPlanner",
        "IncrementalProofSealer",
        "AdaptivePlanner",
        "ContextCompiler",
        "SupervisorControlService",
        "AutonomousMetaController",
        "ProofCarryingProcedureCompiler",
    ]
    assert len(set(names)) == len(names)

    classified = 0
    for item in matrix["prerequisites"]:
        classified += 1
        assert item["status"] in ALLOWED_STATUSES
        assert item["qualified"] is False
        if item["status"] == "missing":
            assert item["source"] is None
            assert item["test"] is None
            assert item["source_blob"] is None
            assert item["test_blob"] is None
            assert item["blocker"]
            assert _class_present(item["class_name"]) is False
            continue

        path, line = _split_source(item["source"])
        source_path = ROOT / path
        test_path = ROOT / item["test"]
        assert source_path.is_file(), path
        assert test_path.is_file(), item["test"]
        assert _class_line(source_path, item["class_name"]) == line
        assert item["source_blob"] == _blob(path)
        assert item["test_blob"] == _blob(item["test"])
        assert SHA1_RE.fullmatch(item["source_blob"])
        assert SHA1_RE.fullmatch(item["test_blob"])
        if item["status"] == "available_with_caveats":
            assert item["caveat"]

    assert classified == 10

    adversarial = next(
        item
        for item in matrix["prerequisites"]
        if item["name"] == "AdversarialAssuranceEngine"
    )
    assert adversarial["class_name"] == "AssuranceCampaignApi"
    assert adversarial["requested_class_name"] == "AdversarialAssuranceEngine"
    assert adversarial["requested_class_present"] is False
    assert adversarial["blocker"] == (
        "prerequisite.adversarial_assurance.exact_engine_symbol_absent"
    )
    assert _class_present("AdversarialAssuranceEngine") is False
    assert _class_present("AssuranceCampaignApi") is True

    procedure_compiler = next(
        item
        for item in matrix["prerequisites"]
        if item["name"] == "ProofCarryingProcedureCompiler"
    )
    assert procedure_compiler["status"] == "available"
    assert procedure_compiler["class_name"] == "ProofCarryingProcedureCompiler"
    assert procedure_compiler.get("blocker") is None
    assert _class_present("ProofCarryingProcedureCompiler") is True

    missing = {
        item["name"]: item["blocker"]
        for item in matrix["prerequisites"]
        if item["status"] == "missing"
    }
    assert missing == {}


def test_qualified_test_ledger() -> None:
    baseline = _load(BASELINE_PATH)
    bootstrap = _load(BOOTSTRAP_LEDGER_PATH)
    ledger = baseline["qualified_test_ledger"]
    sealed_tree = _git("rev-parse", f"{SEALED_COMMIT}^{{tree}}")

    assert ledger["schema"] == (
        "ipfs_accelerate_py.agent_supervisor.architecture-refactorer"
        ".qualified-test-ledger@1"
    )
    assert ledger["network_required"] is False
    assert ledger["status"] == "commands_bound_execution_not_run"
    assert bootstrap["network_required"] is False

    expected_argv = [command.split() for command in bootstrap["commands"]]
    observed_argv = [item["argv"] for item in ledger["commands"]]
    assert observed_argv == expected_argv
    assert len(ledger["commands"]) == 9

    for item in ledger["commands"]:
        assert item["execution_status"] in EXECUTION_STATUSES
        assert item["environment"]["network"] == "deny"
        assert item["environment"]["python"] == "/usr/bin/python3.12"
        assert _peel_tree(item["tree"]) == sealed_tree
        assert item["argv"][:4] == ["python3", "-m", "pytest", "-q"]
        for target in item["argv"][4:]:
            path = ROOT / target
            assert path.exists(), target
        if item["execution_status"] == "not-run":
            assert item["collected"] is None
            assert item["passed"] is None
            assert item["failed"] is None
            assert item["skipped"] is None
            assert item["duration_seconds"] is None
            assert item["exit_code"] is None
            assert item["receipt_identity"] is None
            assert item["reason"]
        elif item["execution_status"] == "pass":
            assert item["exit_code"] == 0
            assert item["failed"] == 0
            assert item["receipt_identity"]
        else:
            assert item["exit_code"] is not None
            assert item["receipt_identity"] is not None or item["reason"]
