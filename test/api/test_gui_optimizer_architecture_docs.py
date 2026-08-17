"""VGO-096: cross-repository architecture documentation tests.

A documentation test verifies required sections, current module/interface
references, exclusions, evidence-level language, the application-extension
checklist, and the narrow final claim. It fail-closes on stale module paths
and overclaims.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.cli import (
    COMMAND_INTERFACES,
    GUI_OPT_COMMANDS,
    TARGET_REGISTRY,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.improvement_loop import (
    VERIFIED_GUI_OPTIMIZER_INTERFACE,
)
from ipfs_datasets_py.logic.gui_optimizer.formal_adapter import (
    FORBIDDEN_CLAIM_KINDS,
    GUI_FORMAL_ADAPTER_INTERFACE,
    SUPPORTED_PROPERTY_KINDS,
)
from ipfs_datasets_py.logic.gui_optimizer.invariants import (
    UI_INVARIANT_ENGINE_INTERFACE,
)
from ipfs_datasets_py.logic.gui_optimizer.schema import (
    REQUIRED_MODEL_INTERFACES,
    AnalysisClassification,
    EvidenceLevel,
    VerificationStatus,
)


ACCELERATE_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_SUPERPROJECT_CANDIDATES: Final[tuple[Path, ...]] = (
    ACCELERATE_ROOT.parent.parent,
    ACCELERATE_ROOT.parent,
)
ARCHITECTURE_INTERFACE: Final[str] = "VerifiedGuiOptimizerArchitecture@1"
ADAPTER_INTERFACE: Final[str] = "GuiApplicationAdapter@1"
EVIDENCE_MATRIX_INTERFACE: Final[str] = "GuiEvidenceAuthorityMatrix@1"
SELECTED_SOURCE: Final[str] = "swissknife/web/js/apps/agent-supervisor.js"
SELECTED_APPLICATION_ID: Final[str] = "app:agent-supervisor"
SELECTED_SCREEN_ID: Final[str] = "screen:agent-supervisor"
NARROW_CLAIM: Final[str] = (
    "The selected GUI workflow was incrementally analyzed and improved against "
    "declared interaction, accessibility, policy, and visual-regression criteria, "
    "with content-addressed evidence for the evaluated scenarios."
)

DATASETS_MODULES: Final[tuple[str, ...]] = (
    "schema.py",
    "models.py",
    "identity.py",
    "formal_adapter.py",
    "invariants.py",
    "receipts.py",
)
ACCELERATE_MODULES: Final[tuple[str, ...]] = (
    "authority.py",
    "patch_scope.py",
    "proposal.py",
    "worktree_executor.py",
    "check_plan.py",
    "run_journal.py",
    "artifact_store.py",
    "improvement_loop.py",
    "cli.py",
    "benchmark.py",
)
SWISSKNIFE_MODULES: Final[tuple[str, ...]] = (
    "models.ts",
    "scanner.ts",
    "identity.ts",
    "component-graph.ts",
    "ui-capsule.ts",
    "state-machine.ts",
    "scenario-catalog.ts",
    "policy-validator.ts",
    "invalidation.ts",
    "context-pack.ts",
    "accessibility.ts",
    "visual-regression.ts",
    "interaction-runner.ts",
    "baseline.ts",
    "evaluator.ts",
    "cli.ts",
    "targets/agent-supervisor.ts",
)

SWISS_REQUIRED_SECTIONS: Final[tuple[str, ...]] = (
    "## 1. Purpose and selected screen",
    "## 2. Implementation boundaries",
    "## 3. Current modules and interfaces",
    "## 4. Static analysis, graph, and state",
    "## 5. Evaluation, invalidation, and context",
    "## 6. Evidence authority matrix",
    "## 7. Commands",
    "## 8. Application-extension checklist",
    "## 9. Exclusions and non-goals",
    "## 10. Narrow final claim",
    "## Diagrams tied to tests",
)
CONTRACT_REQUIRED_SECTIONS: Final[tuple[str, ...]] = (
    "## 1. Purpose",
    "## 2. Package and wire identity",
    "## 3. Required closed models",
    "## 4. Evidence authority matrix",
    "## 5. Formal adapter and invariants",
    "## 6. Receipts and content identity",
    "## 7. Current modules and interfaces",
    "## 8. Application-extension contract additions",
    "## 9. Exclusions and non-goals",
    "## 10. Narrow final claim",
)
ACCEL_REQUIRED_SECTIONS: Final[tuple[str, ...]] = (
    "## 1. Purpose and selected screen",
    "## 2. Package boundaries",
    "## 3. Current modules and interfaces",
    "## 4. Security model",
    "## 5. Improvement loop",
    "## 6. Evidence authority matrix",
    "## 7. Commands",
    "## 8. Application-extension checklist",
    "## 9. Exclusions and non-goals",
    "## 10. Narrow final claim",
    "## Diagrams tied to tests",
)

EXTENSION_PATHS: Final[tuple[str, ...]] = (
    "swissknife/src/services/apps/virtual-desktop-app-manifest.ts",
    "swissknife/web/js/main-simple.js",
    "swissknife/src/services/gui-optimizer/targets/",
    "swissknife/src/services/gui-optimizer/cli.ts",
    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/gui_optimizer/cli.py",
    "swissknife/src/services/gui-optimizer/scenario-catalog.ts",
    "swissknife/test/fixtures/gui-optimizer/scenarios/",
    "swissknife/src/services/apps/all-app-executable-backend-contract.ts",
    "swissknife/src/services/apps/all-app-live-tool-bindings.ts",
    "swissknife/src/services/apps/app-capability-policy.ts",
    "swissknife/src/services/apps/mcp-deontic-ui-manifest.ts",
    "swissknife/src/services/mcp/mcp-control-surface-mediator.ts",
    "swissknife/src/services/mcp/all-app-tool-gateway.ts",
    "swissknife/test/unit/services/gui-optimizer/",
    "swissknife/test/browser/verified-gui-optimizer-",
    "swissknife/test/e2e/verified-gui-optimizer-",
    "implementation_plan/evidence/verified_gui_optimizer/",
)
EXTENSION_KEYWORDS: Final[tuple[str, ...]] = (
    "manifest",
    "target",
    "scenario",
    "action",
    "policy",
    "test",
    "screenshot",
    "acceptance",
)

FORBIDDEN_DEPENDENCIES: Final[tuple[str, ...]] = (
    "semantic-index",
    "semantic-capsule",
    "proof-cache",
    "formal-verification-cache",
    "model-routing",
    "ipfs_datasets_py/logic/ui_ux_ir",
)
EXCLUDED_MISSING_PATH_MARKERS: Final[tuple[str, ...]] = (
    "ipfs_datasets_py/logic/ui_ux_ir",
    "legacy-archive",
    "emergency-archive",
    "cleanup-archive",
    "config/archive",
    "test/archived",
    "virtual-desktop-live-gateway.ts",
    "<app-id>",
    "path-or-component",
)
OVERCLAIM_PHRASES: Final[tuple[str, ...]] = (
    "proved optimal",
    "proved optimality",
    "wcag certification",
    "complete security",
    "complete accessibility",
    "beauty is proved",
    "unbounded correctness",
)
NEGATION_MARKERS: Final[tuple[str, ...]] = (
    "not ",
    "never",
    "must not",
    "do not",
    "does not",
    "cannot",
    "can not",
    "forbidden",
    "non-goal",
    "non-goals",
    "without",
    "excluded",
    "exclusion",
    "distinct from",
    "is not",
    "are not",
    "no ",
    "not a ",
    "not an ",
)
_BACKTICK_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"`((?:swissknife|external|implementation_plan|scripts)/[^`\s]+)`"
)


def _first_existing(candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise AssertionError(
        "could not resolve repository root from candidates: "
        + ", ".join(str(item) for item in candidates)
    )


def _superproject_root() -> Path:
    for candidate in _SUPERPROJECT_CANDIDATES:
        if (candidate / "swissknife").is_dir() and (
            candidate / "external" / "ipfs_datasets"
        ).is_dir():
            return candidate
        if (candidate / "swissknife").is_dir() and (candidate / "ipfs_datasets").is_dir():
            return candidate
    raise AssertionError("could not locate superproject containing swissknife and datasets")


def _datasets_root(superproject: Path) -> Path:
    return _first_existing(
        (
            superproject / "external" / "ipfs_datasets",
            superproject / "ipfs_datasets",
            ACCELERATE_ROOT.parent / "ipfs_datasets",
        )
    )


def _swissknife_root(superproject: Path) -> Path:
    return _first_existing(
        (
            superproject / "swissknife",
            ACCELERATE_ROOT.parent.parent / "swissknife",
        )
    )


@pytest.fixture(scope="module")
def roots() -> dict[str, Path]:
    superproject = _superproject_root()
    return {
        "superproject": superproject,
        "accelerate": ACCELERATE_ROOT,
        "datasets": _datasets_root(superproject),
        "swissknife": _swissknife_root(superproject),
    }


@pytest.fixture(scope="module")
def docs(roots: Mapping[str, Path]) -> dict[str, str]:
    paths = {
        "swiss": roots["swissknife"] / "docs" / "gui-optimizer" / "ARCHITECTURE.md",
        "contracts": roots["datasets"] / "docs" / "gui_optimizer_contracts.md",
        "accel": roots["accelerate"] / "docs" / "architecture" / "VERIFIED_GUI_OPTIMIZER.md",
    }
    loaded: dict[str, str] = {}
    for key, path in paths.items():
        assert path.is_file(), f"missing architecture document: {path}"
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"empty architecture document: {path}"
        assert path.stat().st_size > 2000, f"architecture document unexpectedly small: {path}"
        loaded[key] = text
        loaded[f"{key}_path"] = str(path)
    loaded["corpus"] = "\n\n".join(loaded[name] for name in ("swiss", "contracts", "accel"))
    return loaded


def _require_phrases(text: str, phrases: Iterable[str], *, label: str) -> None:
    missing = [phrase for phrase in phrases if phrase.lower() not in text.lower()]
    assert not missing, f"{label} missing required phrase(s): {missing}"


def _require_sections(text: str, sections: Iterable[str], *, label: str) -> None:
    missing = [section for section in sections if section not in text]
    assert not missing, f"{label} missing required section(s): {missing}"


def _windows(text: str, start: int, width: int = 360) -> str:
    lo = max(0, start - width)
    hi = min(len(text), start + width)
    return text[lo:hi].lower()


def _negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    needle = phrase.lower()
    index = 0
    found = False
    while True:
        hit = lowered.find(needle, index)
        if hit < 0:
            return found
        found = True
        window = _windows(lowered, hit)
        if not any(marker in window for marker in NEGATION_MARKERS):
            return False
        index = hit + len(needle)
    return found


def _current_python_modules(package_dir: Path) -> set[str]:
    return {path.name for path in package_dir.glob("*.py") if path.name != "__init__.py"}


def _current_swissknife_modules(package_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in package_dir.rglob("*.ts"):
        if path.name.endswith(".d.ts"):
            continue
        names.add(str(path.relative_to(package_dir)).replace("\\", "/"))
    return names


def _quoted_repo_paths(text: str) -> set[str]:
    return {match.group(1) for match in _BACKTICK_PATH_RE.finditer(text)}


def _is_excluded_missing_path(path: str) -> bool:
    return any(marker in path for marker in EXCLUDED_MISSING_PATH_MARKERS)


def test_documents_exist_and_declare_architecture_interfaces(docs: Mapping[str, str]) -> None:
    corpus = docs["corpus"]
    _require_phrases(
        corpus,
        (
            ARCHITECTURE_INTERFACE,
            ADAPTER_INTERFACE,
            EVIDENCE_MATRIX_INTERFACE,
            VERIFIED_GUI_OPTIMIZER_INTERFACE,
        ),
        label="architecture corpus",
    )
    for name in ("swiss", "contracts", "accel"):
        assert "VerifiedGuiOptimizer" in docs[name]


def test_required_sections_are_present(docs: Mapping[str, str]) -> None:
    _require_sections(docs["swiss"], SWISS_REQUIRED_SECTIONS, label="SwissKnife ARCHITECTURE.md")
    _require_sections(
        docs["contracts"], CONTRACT_REQUIRED_SECTIONS, label="gui_optimizer_contracts.md"
    )
    _require_sections(
        docs["accel"], ACCEL_REQUIRED_SECTIONS, label="VERIFIED_GUI_OPTIMIZER.md"
    )


def test_selected_screen_and_implementation_boundaries(docs: Mapping[str, str]) -> None:
    corpus = docs["corpus"]
    _require_phrases(
        corpus,
        (
            "Agent Supervisor",
            SELECTED_SOURCE,
            SELECTED_APPLICATION_ID,
            SELECTED_SCREEN_ID,
            "route:agent-supervisor",
            "ipfs_datasets_py.logic.gui_optimizer",
            "ipfs_accelerate_py.agent_supervisor.gui_optimizer",
            "swissknife/src/services/gui-optimizer",
        ),
        label="selected-screen corpus",
    )
    assert "agent-supervisor" in TARGET_REGISTRY
    assert TARGET_REGISTRY["agent-supervisor"].source_paths == (SELECTED_SOURCE,)
    assert TARGET_REGISTRY["agent-supervisor"].application_id == SELECTED_APPLICATION_ID
    assert TARGET_REGISTRY["agent-supervisor"].screen_id == SELECTED_SCREEN_ID


def test_current_modules_are_documented_and_present(docs: Mapping[str, str], roots: Mapping[str, Path]) -> None:
    corpus = docs["corpus"]
    datasets_dir = (
        roots["datasets"] / "ipfs_datasets_py" / "logic" / "gui_optimizer"
    )
    accelerate_dir = (
        roots["accelerate"]
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "gui_optimizer"
    )
    swiss_dir = roots["swissknife"] / "src" / "services" / "gui-optimizer"

    live_datasets = _current_python_modules(datasets_dir)
    live_accelerate = _current_python_modules(accelerate_dir)
    live_swiss = _current_swissknife_modules(swiss_dir)

    assert live_datasets == set(DATASETS_MODULES), (
        "docs inventory drifted from datasets package: "
        f"live={sorted(live_datasets)} expected={list(DATASETS_MODULES)}"
    )
    assert set(ACCELERATE_MODULES) <= live_accelerate, (
        "docs missing live accelerator modules: "
        f"{sorted(live_accelerate - set(ACCELERATE_MODULES))}"
    )
    assert set(SWISSKNIFE_MODULES) <= live_swiss, (
        "docs missing live SwissKnife modules: "
        f"{sorted(set(SWISSKNIFE_MODULES) - live_swiss)}"
    )

    for module in (*DATASETS_MODULES, *ACCELERATE_MODULES):
        assert module in corpus, f"documentation omits current module {module}"
    for module in SWISSKNIFE_MODULES:
        assert module in corpus, f"documentation omits current module {module}"


def test_current_interfaces_are_documented(docs: Mapping[str, str]) -> None:
    corpus = docs["corpus"]
    required = (
        *REQUIRED_MODEL_INTERFACES,
        GUI_FORMAL_ADAPTER_INTERFACE,
        UI_INVARIANT_ENGINE_INTERFACE,
        VERIFIED_GUI_OPTIMIZER_INTERFACE,
        "GuiStaticScanner@1",
        "DeterministicScenarioCatalog@1",
        "GuiPatchAuthority@1",
        "GuiPatchProposer@1",
        "GuiIsolatedWorktreeExecutor@1",
        "GuiRunJournal@1",
        "GuiEvidenceArtifactStore@1",
        "GuiOptimizerCli@1",
        "AgentSupervisorTarget@1",
        *COMMAND_INTERFACES.values(),
    )
    missing = [item for item in required if item not in corpus]
    assert not missing, f"documentation omits current interface(s): {missing}"
    assert tuple(GUI_OPT_COMMANDS) == (
        "scan",
        "baseline",
        "impact",
        "evaluate",
        "pack-context",
        "verify",
        "improve",
        "report",
    )
    for command in GUI_OPT_COMMANDS:
        assert f"gui-opt {command}" in corpus


def test_evidence_taxonomy_and_independence(docs: Mapping[str, str]) -> None:
    corpus = docs["corpus"]
    _require_phrases(
        corpus,
        (
            "formally verified",
            "structurally validated",
            "heuristic",
            "human-reviewed",
            VerificationStatus.VERIFIED.value,
            VerificationStatus.STRUCTURALLY_VALID.value,
            VerificationStatus.INTEGRITY_VALID.value,
            VerificationStatus.SIMULATED.value,
            AnalysisClassification.EXACT.value,
            AnalysisClassification.CONSERVATIVE.value,
            AnalysisClassification.HEURISTIC.value,
            AnalysisClassification.OPAQUE.value,
            EvidenceLevel.HUMAN_REVIEWED.value,
            "content identities and receipts do not prove truth",
            "independent",
        ),
        label="evidence taxonomy",
    )
    missing_kinds = [
        kind
        for kind in sorted(SUPPORTED_PROPERTY_KINDS)
        if kind not in docs["contracts"] or kind not in docs["accel"]
    ]
    assert not missing_kinds, f"docs omit formal property kind(s): {missing_kinds}"


def test_exclusions_and_forbidden_dependencies(docs: Mapping[str, str]) -> None:
    corpus = docs["corpus"]
    for name in ("swiss", "contracts", "accel"):
        text = docs[name]
        assert re.search(r"## \d+\. Exclusions and non-goals", text), (
            f"{name} lacks an exclusions/non-goals section"
        )
    for dependency in FORBIDDEN_DEPENDENCIES:
        assert dependency in corpus, f"exclusion {dependency!r} is not recorded"
        assert _negated(corpus, dependency), (
            f"{dependency!r} must be recorded as prohibited, not as an implementation dependency"
        )
    _require_phrases(
        corpus,
        (
            "must not import",
            "UiSemanticCapsule@1",
            "not a proof cache",
            "does not choose or route models",
            "virtual-desktop-live-gateway.ts",
        ),
        label="standalone boundary",
    )


def test_extension_checklist_lists_exact_additions(docs: Mapping[str, str]) -> None:
    checklist = docs["swiss"] + "\n" + docs["accel"]
    _require_phrases(checklist, EXTENSION_KEYWORDS, label="extension checklist")
    _require_phrases(checklist, EXTENSION_PATHS, label="extension checklist paths")
    _require_phrases(
        checklist,
        (
            "TARGET_REGISTRY",
            "COMPONENT_REGISTRY",
            "UiActionBinding@1",
            "UiConfirmationBinding@1",
            "GuiImprovementReceipt@1",
            "does not implement a second application",
        ),
        label="extension checklist contracts",
    )
    assert "<app-id>" in checklist
    assert "fixture-host.html" in checklist
    assert "verified-gui-optimizer-<app-id>-baseline.spec.ts" in checklist


def test_narrow_final_claim_and_no_overclaims(docs: Mapping[str, str]) -> None:
    for name in ("swiss", "contracts", "accel"):
        assert NARROW_CLAIM in docs[name], f"{name} is missing the narrow final claim"
        assert "proved optimal" in docs[name].lower()
        assert _negated(docs[name], "proved optimal"), (
            f"{name} must not claim the GUI is proved optimal"
        )
    corpus = docs["corpus"]
    for phrase in OVERCLAIM_PHRASES:
        if phrase not in corpus.lower():
            continue
        assert _negated(corpus, phrase), f"overclaim without negation: {phrase!r}"
    for kind in FORBIDDEN_CLAIM_KINDS:
        assert _negated(corpus, kind.replace("_", " ")) or kind in docs["contracts"]


def test_documented_implementation_paths_exist(
    docs: Mapping[str, str], roots: Mapping[str, Path]
) -> None:
    superproject = roots["superproject"]
    missing: list[str] = []
    for raw in sorted(_quoted_repo_paths(docs["corpus"])):
        path = raw.split()[0].rstrip(".,);")
        if _is_excluded_missing_path(path):
            continue
        if path.endswith("/"):
            candidate = superproject / path
            if not candidate.is_dir():
                missing.append(path)
            continue
        candidate = superproject / path
        if candidate.is_file() or candidate.is_dir():
            continue
        # Allow accelerate-relative paths written without the external/ prefix.
        alt = roots["accelerate"] / path
        if alt.is_file() or alt.is_dir():
            continue
        missing.append(path)
    assert not missing, f"documentation references missing path(s): {missing}"


_KNOWN_EXTERNAL_MODULES: Final[frozenset[str]] = frozenset(
    {
        "virtual-desktop-app-manifest.ts",
        "all-app-executable-backend-contract.ts",
        "all-app-live-tool-bindings.ts",
        "app-capability-policy.ts",
        "mcp-deontic-ui-manifest.ts",
        "mcp-control-surface-mediator.ts",
        "all-app-tool-gateway.ts",
        "mcp-deontic-interface-broker.ts",
        "ui-ux-ir-codec.ts",
        "virtual-desktop-live-gateway.ts",
        "canonical.py",
        "agent-supervisor.ts",
        "agent-supervisor.js",
        "main-simple.js",
        "fixture-services.js",
    }
)


def test_no_stale_gui_optimizer_module_references(
    docs: Mapping[str, str], roots: Mapping[str, Path]
) -> None:
    """Documented package module basenames must exist in the current tree."""
    datasets_dir = roots["datasets"] / "ipfs_datasets_py" / "logic" / "gui_optimizer"
    accelerate_dir = (
        roots["accelerate"]
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "gui_optimizer"
    )
    swiss_dir = roots["swissknife"] / "src" / "services" / "gui-optimizer"
    live = (
        _current_python_modules(datasets_dir)
        | _current_python_modules(accelerate_dir)
        | {Path(name).name for name in _current_swissknife_modules(swiss_dir)}
    )
    mentioned = set(re.findall(r"`([A-Za-z][A-Za-z0-9_-]+\.(?:py|ts|js))`", docs["corpus"]))
    stale = sorted(
        name
        for name in mentioned
        if not name.startswith("test_")
        and name not in live
        and name not in _KNOWN_EXTERNAL_MODULES
        and not _is_excluded_missing_path(name)
    )
    assert not stale, f"documentation names stale gui-optimizer module(s): {stale}"


def test_cli_commands_match_live_help(docs: Mapping[str, str]) -> None:
    corpus = docs["corpus"]
    for command, interface in COMMAND_INTERFACES.items():
        assert interface in corpus
        assert f"gui-opt {command}" in corpus
    assert "scripts/gui-opt" in docs["accel"]


def test_documents_do_not_expand_to_every_application(docs: Mapping[str, str]) -> None:
    corpus = docs["corpus"]
    _require_phrases(
        corpus,
        (
            "one bounded screen",
            "one additional application",
            "optimizing every",
        ),
        label="scope bound",
    )
    assert _negated(corpus, "optimizing every")


def test_required_model_interfaces_are_live() -> None:
    assert "GuiApplicationIdentity@1" in REQUIRED_MODEL_INTERFACES
    assert "GuiImprovementReceipt@1" in REQUIRED_MODEL_INTERFACES
    assert len(REQUIRED_MODEL_INTERFACES) == 23


def test_architecture_docs_parse_as_utf8_markdown(docs: Mapping[str, str]) -> None:
    for name in ("swiss", "contracts", "accel"):
        text = docs[name]
        assert "\x00" not in text
        assert text.startswith("# ")
        # Fence integrity: even number of triple-backtick openers.
        fence_count = text.count("```")
        assert fence_count % 2 == 0, f"{name} has unbalanced markdown fences"


def test_live_packages_do_not_import_forbidden_layers(roots: Mapping[str, Path]) -> None:
    forbidden = (
        "semantic_index",
        "semanticindex",
        "proof_cache",
        "formal_verification_cache",
        "model_routing",
        "provider_routing",
        "logic.ui_ux_ir",
        "logic/ui_ux_ir",
    )
    package_dirs = (
        roots["datasets"] / "ipfs_datasets_py" / "logic" / "gui_optimizer",
        roots["accelerate"]
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "gui_optimizer",
    )
    offenders: list[str] = []
    for package_dir in package_dirs:
        for path in package_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    lowered = name.replace(".", "_").lower()
                    if any(token in lowered or token in name for token in forbidden):
                        offenders.append(f"{path.name}:{name}")
    assert not offenders, f"gui_optimizer packages import forbidden layers: {offenders}"
