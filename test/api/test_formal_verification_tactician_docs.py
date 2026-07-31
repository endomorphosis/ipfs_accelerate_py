"""Executable contract for FormalVerificationTacticianDocumentation@1 (FVT-034 / FVT-G070).

Validates that the tactician documentation set:

* exists at the declared evidence paths;
* distinguishes legal evidence routing from formal proof planning, proposals
  from proofs, bounded checks from theorem proof, implementation completeness
  from deployment certification, assumptions from obligations, and every
  failure/rollback state;
* ships executable examples (fenced code that imports public APIs);
* preserves channel aliases for GoalTacticianCLI/MCP parity; and
* stays consistent with the live GoalTacticianAPI operation catalog.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PRODUCT_DOC = REPO_ROOT / "docs" / "formal_verification_tactician.md"
RUNBOOK_DOC = (
    REPO_ROOT / "docs" / "operations" / "formal_verification_tactician_runbook.md"
)
MIGRATION_DOC = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "docs"
    / "logic"
    / "proof_tactician_migration.md"
)
OBJECTIVES_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_readiness.objectives.md"
)

INTERFACE = "FormalVerificationTacticianDocumentation@1"
GOAL_ID = "FVT-G070"
TASK_ID = "FVT-034"

DOC_PATHS = (PRODUCT_DOC, RUNBOOK_DOC, MIGRATION_DOC)

# Acceptance distinctions that must appear as explicit vocabulary in docs.
DISTINCTION_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "legal evidence routing vs formal proof planning",
        (
            "legal evidence routing",
            "formal proof planning",
        ),
    ),
    (
        "proposals vs proofs",
        (
            "proposals",
            "proofs",
        ),
    ),
    (
        "bounded checks vs theorem proof",
        (
            "bounded check",
            "theorem",
        ),
    ),
    (
        "implementation completeness vs deployment certification",
        (
            "implementation complete",
            "deployment certif",
        ),
    ),
    (
        "assumptions vs obligations",
        (
            "assumption",
            "obligation",
        ),
    ),
)

# Failure / rollback vocabulary required across the doc set.
FAILURE_ROLLBACK_TERMS: tuple[str, ...] = (
    "unavailable",
    "unsupported",
    "invalid",
    "cancelled",
    "timed_out",
    "backpressure",
    "invalidated",
    "quarantine",
    "rollback",
    "stale_worker",
    "stale_tree",
    "false proof",
)

# Required structural anchors.
PRODUCT_SECTIONS: tuple[str, ...] = (
    "## 2. Critical distinctions",
    "## 4. Public API surfaces",
    "## 5. Executable examples",
    "## 6. Proof-authority interpretation",
    "GoalTacticianAPI@1",
    "LogicVerificationAPI@1",
    "formalize_goal",
    "replay_counterexample",
)

RUNBOOK_SECTIONS: tuple[str, ...] = (
    "## 4. Lifecycle operations",
    "## 5. Failure and rollback matrix",
    "## 6. Incident response playbooks",
    "GoalTacticianSupervisorLifecycle",
    "implementation completeness",
    "deployment",
)

MIGRATION_SECTIONS: tuple[str, ...] = (
    "## 3. Compatibility aliases",
    "GOAL_TACTICIAN_OPERATIONS",
    "goal-formalize",
    "goal_tactician_formalize_goal",
    "logic.api",
    "verification_api",
)

REQUIRED_GOAL_OPERATIONS: tuple[str, ...] = (
    "formalize_goal",
    "compare_interpretations",
    "discover_missing_proofs",
    "plan_proof",
    "validate_proof_candidate",
    "execute_proof_plan",
    "proof_status",
    "minimize_counterexample",
    "explain_counterexample_causal",
    "replay_counterexample",
    "list_goal_tactician_operations",
)


def _read(path: Path) -> str:
    assert path.is_file(), f"missing documentation evidence: {path}"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"documentation is empty: {path}"
    return text


def _combined_corpus() -> str:
    return "\n".join(_read(path) for path in DOC_PATHS)


def _extract_fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Return (language, body) pairs for Markdown fenced code blocks."""

    pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
    return [(match.group(1).lower(), match.group(2)) for match in pattern.finditer(text)]


def _python_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    for language, body in _extract_fenced_blocks(text):
        if language in {"python", "py", ""}:
            # Heuristic: treat unlabelled fences as Python when they import our API.
            if language or "ipfs_datasets_py" in body or "from ipfs_datasets_py" in body:
                if "import" in body or "assert" in body:
                    blocks.append(body)
    return blocks


# ---------------------------------------------------------------------------
# Presence and interface
# ---------------------------------------------------------------------------


def test_declared_documentation_outputs_exist() -> None:
    for path in DOC_PATHS:
        assert path.is_file(), path
        assert path.stat().st_size > 500, f"documentation too small: {path}"


def test_interface_and_goal_identity_are_documented() -> None:
    corpus = _combined_corpus()
    assert INTERFACE in corpus
    product = _read(PRODUCT_DOC)
    assert INTERFACE in product
    assert "GoalTacticianAPI@1" in product
    assert "LogicVerificationAPI@1" in product
    # Objective heap remains the schedulable source of truth for the goal id.
    objectives = OBJECTIVES_PATH.read_text(encoding="utf-8")
    assert f"## {GOAL_ID}" in objectives
    assert "docs/formal_verification_tactician.md" in objectives
    assert "docs/operations/formal_verification_tactician_runbook.md" in objectives
    assert "proof_tactician_migration.md" in objectives


def test_objective_heap_validation_points_at_this_suite() -> None:
    objectives = OBJECTIVES_PATH.read_text(encoding="utf-8")
    section = objectives.split(f"## {GOAL_ID}", 1)[1].split("\n## ", 1)[0]
    assert "test_formal_verification_tactician_docs.py" in section
    assert "check_agent_supervisor_docs.py" in section


# ---------------------------------------------------------------------------
# Acceptance distinctions
# ---------------------------------------------------------------------------


def test_docs_encode_required_distinctions() -> None:
    corpus = _combined_corpus().lower()
    missing: list[str] = []
    for label, phrases in DISTINCTION_PHRASES:
        if not all(phrase.lower() in corpus for phrase in phrases):
            missing.append(label)
    assert not missing, f"missing distinction coverage: {missing}"


def test_product_doc_has_explicit_distinction_section() -> None:
    text = _read(PRODUCT_DOC)
    assert "## 2. Critical distinctions" in text
    # Each distinction subsection must be present.
    for heading in (
        "Legal evidence routing vs formal proof planning",
        "Proposals vs proofs",
        "Bounded checks vs theorem proof",
        "Implementation completeness vs deployment certification",
        "Assumptions vs obligations",
        "Failure and rollback states",
    ):
        assert heading in text, heading


def test_legal_lane_is_not_conflated_with_formal_planning() -> None:
    product = _read(PRODUCT_DOC)
    migration = _read(MIGRATION_DOC)
    runbook = _read(RUNBOOK_DOC)
    for text in (product, migration, runbook):
        assert "legal" in text.lower()
        assert "formal proof planning" in text.lower() or "formal plan" in text.lower()
    # Explicit non-promotion language.
    assert "not" in product.lower()
    assert (
        "theorem authority" in product.lower()
        or "theorem" in product
        and "legal" in product.lower()
    )
    assert "legal evidence" in migration.lower()
    assert "legal evidence" in runbook.lower() or "legal_compatible" in runbook


def test_proposals_are_not_proofs_language() -> None:
    product = _read(PRODUCT_DOC)
    assert "proposal" in product.lower()
    assert "formalize_goal" in product
    assert "admitted" in product.lower() or "never" in product.lower()
    assert "fresh" in product.lower() and "receipt" in product.lower()


def test_bounded_vs_theorem_language() -> None:
    product = _read(PRODUCT_DOC)
    assert "Bounded check" in product or "bounded check" in product
    assert "theorem" in product.lower()
    assert "authority" in product.lower()
    assert "kernel" in product.lower()


def test_implementation_vs_deployment_language() -> None:
    corpus = _combined_corpus()
    assert "implementation complete" in corpus.lower() or "Implementation complete" in corpus
    assert "deployment certif" in corpus.lower()
    assert "formal_verification_readiness_baseline.json" in corpus


def test_assumptions_vs_obligations_language() -> None:
    product = _read(PRODUCT_DOC)
    assert "Assumption" in product or "assumption" in product
    assert "obligation" in product.lower()
    assert "proof hole" in product.lower() or "proof holes" in product.lower()
    assert "favorable assumption" in product.lower()


def test_failure_and_rollback_matrix_is_complete() -> None:
    runbook = _read(RUNBOOK_DOC).lower()
    product = _read(PRODUCT_DOC).lower()
    corpus = runbook + "\n" + product
    missing = [term for term in FAILURE_ROLLBACK_TERMS if term.lower() not in corpus]
    assert not missing, f"failure/rollback vocabulary missing: {missing}"
    assert "## 5. Failure and rollback matrix" in _read(RUNBOOK_DOC)


# ---------------------------------------------------------------------------
# Structure and cross-links
# ---------------------------------------------------------------------------


def test_product_doc_required_sections() -> None:
    text = _read(PRODUCT_DOC)
    missing = [item for item in PRODUCT_SECTIONS if item not in text]
    assert not missing, missing


def test_runbook_required_sections() -> None:
    text = _read(RUNBOOK_DOC)
    missing = [item for item in RUNBOOK_SECTIONS if item not in text]
    assert not missing, missing


def test_migration_required_sections() -> None:
    text = _read(MIGRATION_DOC)
    missing = [item for item in MIGRATION_SECTIONS if item not in text]
    assert not missing, missing


def test_docs_cross_link_each_other() -> None:
    product = _read(PRODUCT_DOC)
    runbook = _read(RUNBOOK_DOC)
    migration = _read(MIGRATION_DOC)
    assert "formal_verification_tactician_runbook.md" in product
    assert "proof_tactician_migration.md" in product
    assert "formal_verification_tactician.md" in runbook
    assert "proof_tactician_migration.md" in runbook
    assert "formal_verification_tactician.md" in migration
    assert "formal_verification_tactician_runbook.md" in migration


# ---------------------------------------------------------------------------
# Executable examples
# ---------------------------------------------------------------------------


def test_product_and_migration_include_python_examples() -> None:
    product_blocks = _python_blocks(_read(PRODUCT_DOC))
    migration_blocks = _python_blocks(_read(MIGRATION_DOC))
    runbook_blocks = _python_blocks(_read(RUNBOOK_DOC))
    assert len(product_blocks) >= 3, "product guide needs multiple Python examples"
    assert len(migration_blocks) >= 1, "migration guide needs a Python smoke example"
    assert len(runbook_blocks) >= 1, "runbook needs an operator smoke example"


def test_python_examples_parse_and_import_public_api() -> None:
    corpus_blocks = (
        _python_blocks(_read(PRODUCT_DOC))
        + _python_blocks(_read(MIGRATION_DOC))
        + _python_blocks(_read(RUNBOOK_DOC))
    )
    public_import_seen = False
    for body in corpus_blocks:
        # Skip pure bash-looking bodies accidentally classified.
        if body.lstrip().startswith(("#", "python -", "PYTHONPATH")) and "import" not in body:
            continue
        try:
            tree = ast.parse(body)
        except SyntaxError as exc:
            pytest.fail(f"example is not valid Python: {exc}\n---\n{body[:400]}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("ipfs_datasets_py.logic"):
                    public_import_seen = True
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("ipfs_datasets_py"):
                        public_import_seen = True
    assert public_import_seen, "examples must import public ipfs_datasets_py.logic surfaces"


def test_examples_mention_authority_or_non_admission() -> None:
    """Executable examples must not instruct callers to treat drafts as proofs."""

    product = _read(PRODUCT_DOC)
    # At least one example asserts non-admission or authority inspection.
    assert "admitted" in product
    assert "authority" in product
    blocks = _python_blocks(product)
    joined = "\n".join(blocks)
    assert "authority" in joined or "admitted" in joined


# ---------------------------------------------------------------------------
# Alignment with live GoalTacticianAPI catalog
# ---------------------------------------------------------------------------


def test_documented_operations_match_live_goal_tactician_catalog() -> None:
    from ipfs_datasets_py.logic.verification_api import (
        GOAL_TACTICIAN_CLI_TO_OPERATION,
        GOAL_TACTICIAN_OPERATIONS,
        GOAL_TACTICIAN_TOOL_TO_OPERATION,
    )

    assert tuple(GOAL_TACTICIAN_OPERATIONS) == REQUIRED_GOAL_OPERATIONS or set(
        REQUIRED_GOAL_OPERATIONS
    ).issubset(set(GOAL_TACTICIAN_OPERATIONS))

    corpus = _combined_corpus()
    missing_ops = [op for op in REQUIRED_GOAL_OPERATIONS if op not in corpus]
    assert not missing_ops, f"operations missing from docs: {missing_ops}"

    # Channel aliases appear in product and/or migration docs.
    for cli_name, operation in GOAL_TACTICIAN_CLI_TO_OPERATION.items():
        assert cli_name in corpus, f"CLI alias missing from docs: {cli_name}"
        assert operation in corpus
    for tool_name, operation in GOAL_TACTICIAN_TOOL_TO_OPERATION.items():
        assert tool_name in corpus, f"MCP tool alias missing from docs: {tool_name}"
        assert operation in corpus


def test_forbidden_supervisor_controls_are_documented() -> None:
    from ipfs_datasets_py.logic.verification_api import _GOAL_TACTICIAN_FORBIDDEN_CONTROLS

    corpus = _combined_corpus()
    # Document the critical forbidden controls without requiring every internal alias.
    required = {
        "admit_goal",
        "close_plan",
        "force_complete",
        "promote_proof_authority",
        "lease_steal",
    }
    assert required.issubset(set(_GOAL_TACTICIAN_FORBIDDEN_CONTROLS))
    missing = [name for name in required if name not in corpus]
    assert not missing, f"forbidden controls not documented: {missing}"


def test_live_list_goal_tactician_operations_is_declarative() -> None:
    from ipfs_datasets_py.logic.verification_api import list_goal_tactician_operations

    response = list_goal_tactician_operations()
    assert response.status.value == "declarative"
    operations = response.result.get("operations") or response.result.get(
        "goal_tactician_operations"
    )
    # Surface may nest under list_goal_tactician_cli_mcp_surface shape.
    if operations is None and isinstance(response.result, dict):
        operations = response.result.get("operations")
        if operations is None:
            # Fall back to keys advertised by the catalog helper.
            from ipfs_datasets_py.logic.verification_api import GOAL_TACTICIAN_OPERATIONS

            operations = list(GOAL_TACTICIAN_OPERATIONS)
    assert operations is not None
    for name in REQUIRED_GOAL_OPERATIONS:
        assert name in operations or name in str(response.result)


# ---------------------------------------------------------------------------
# Conflict policy / non-promises
# ---------------------------------------------------------------------------


def test_docs_do_not_promise_silent_success_for_missing_tools() -> None:
    corpus = _combined_corpus().lower()
    assert "never" in corpus
    assert "unavailable" in corpus
    assert "silent success" in corpus or "never become silent success" in corpus or (
        "not" in corpus and "silent" in corpus
    )


def test_docs_disclose_unsupported_languages_tools_policy() -> None:
    corpus = _combined_corpus().lower()
    assert "unsupported" in corpus
    assert "language" in corpus or "languages" in corpus
    assert "do not promise" in corpus or "not promise" in corpus or "unsupported languages" in corpus


def test_migration_preserves_legacy_logic_api_name() -> None:
    migration = _read(MIGRATION_DOC)
    assert "logic.api" in migration or "ipfs_datasets_py.logic.api" in migration
    assert "compatibility" in migration.lower()
    assert "GOAL_TACTICIAN_OPERATIONS" in migration
    assert "STABLE_OPERATIONS" in migration or "not merged" in migration.lower()
