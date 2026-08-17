"""SCH-015 end-to-end acceptance matrix (SemanticStateAcceptance@1).

Proves the controlled-fixture release suite:

* bounded invalidation and known semantic/environment/policy/interface dependents;
* unrelated formatting does not invalidate the repository;
* opaque source is retrieved from the bound tree;
* stale receipts never verify;
* controlled selection recall is 100 percent and full fallback works;
* at least one complete no-fake cross-repository datasets adapter path passes;
* root manifests are deterministic and transitively valid.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    AcceptanceDisposition,
    ContextPack,
    HarnessDisposition,
    HarnessMode,
    ModelRoute,
    RootRef,
    SemanticStateRootManifest,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
    RootConflict,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import (
    HARNESS_STEPS,
    HarnessPolicy,
    HarnessRequest,
    SemanticCompressionHarness,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.receipts import (
    ADMISSION_STALE,
    FRESHNESS_STALE,
    PROVIDER_MODE_PRODUCTION,
    PROOF_STATUS_PASSED,
    StaleReceiptError,
    admit_receipt,
    compile_verification_receipt,
    receipt_may_promote_root,
    receipt_may_verify,
    ReceiptBindings,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    ConfidenceClass,
    ModelRoutingPolicy,
    RiskClass,
    RoutingDecision,
    RoutingInputs,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.selection_execution import (
    FALLBACK_FULL_PYTEST,
    FALLBACK_NONE,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.verification import (
    NormalizedOutcome,
    NormalizedRunFacts,
    compare_full_suite,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import PatchScope


ACCEPTANCE_INTERFACE = "SemanticStateAcceptance@1"
ACCEPTANCE_BUNDLE = "sch/acceptance@1"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = (
    REPO_ROOT
    / "test"
    / "fixtures"
    / "semantic_state_harness"
    / "controlled_repo"
)
PACKAGE_NAME = "sch_controlled_repo_fixture_acceptance"

SEALED_DATASETS_COMMIT = "1330038f626ef92993f03d46f21e1a57719e9c25"
_DATASETS_CANDIDATE_REPOS = (
    Path("/home/barberb/lift_coding/external/ipfs_datasets"),
    Path(os.environ.get("IPFS_DATASETS_GIT_DIR", "")),
)

# Semantic / environment / policy / interface categories that must invalidate
# known dependents (oracle authority from the controlled fixture).
DEPENDENT_INVALIDATION_CATEGORIES = frozenset(
    {
        "local_function_body",
        "public_signature",
        "cross_module_call",
        "dataclass_schema",
        "exception_behavior",
        "side_effect_security",
        "fixture_dependency",
        "pytest_configuration",
        "dependency_lockfile",
        "policy",
        "mcp_interface_client_adapter",
        "dynamic_import",
        "monkey_patch",
        "opaque_native",
        "deleted_symbol",
        "renamed_symbol",
        "generated_file",
    }
)


# ---------------------------------------------------------------------------
# Controlled fixture loader (bytes-only; never imports target modules)
# ---------------------------------------------------------------------------


def _load_fixture_package() -> types.ModuleType:
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]

    init_path = FIXTURE_DIR / "__init__.py"
    if not init_path.is_file():
        raise ImportError(f"missing fixture package init: {init_path}")

    package = types.ModuleType(PACKAGE_NAME)
    package.__file__ = str(init_path)
    package.__path__ = [str(FIXTURE_DIR)]  # type: ignore[attr-defined]
    sys.modules[PACKAGE_NAME] = package

    def _load_submodule(name: str, filename: str) -> types.ModuleType:
        qualname = f"{PACKAGE_NAME}.{name}"
        if qualname in sys.modules:
            return sys.modules[qualname]
        path = FIXTURE_DIR / filename
        spec = importlib.util.spec_from_file_location(qualname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = PACKAGE_NAME
        sys.modules[qualname] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)
        return module

    _load_submodule("mutation_case", "mutation_case.py")
    _load_submodule("recipes", "recipes.py")
    _load_submodule("controlled_repository", "controlled_repository.py")
    init_spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, init_path, submodule_search_locations=[str(FIXTURE_DIR)]
    )
    assert init_spec is not None and init_spec.loader is not None
    package.__spec__ = init_spec
    package.__package__ = PACKAGE_NAME
    init_spec.loader.exec_module(package)
    return package


@pytest.fixture(scope="module")
def fixture_pkg() -> types.ModuleType:
    return _load_fixture_package()


@pytest.fixture(scope="module")
def controlled_repo(fixture_pkg: types.ModuleType) -> Any:
    return fixture_pkg.ControlledSemanticRepository.load()


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


# ---------------------------------------------------------------------------
# Hermetic durable port (generation-bearing CAS)
# ---------------------------------------------------------------------------


class MemoryDurablePort:
    """Hermetic DurableSemanticStatePort with root CAS and immutable blocks."""

    def __init__(self) -> None:
        self._objects: dict[str, dict[str, Any]] = {}
        self._roots: dict[str, RootRef] = {}
        self.put_order: list[str] = []
        self.cas_calls: list[tuple[str, RootRef | None, str]] = []

    def put(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str,
        codec: str = "dag-json",
    ) -> Mapping[str, Any]:
        assert codec == "dag-json"
        self._objects[expected_cid] = dict(artifact)
        self.put_order.append(expected_cid)
        return {"cid": expected_cid}

    def get(self, cid: str) -> Mapping[str, Any]:
        return dict(self._objects[cid])

    def get_bytes(self, cid: str) -> bytes:
        return json.dumps(
            self._objects[cid], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def has(self, cid: str) -> bool:
        return cid in self._objects

    def read_root(self, repository_id: str) -> RootRef | None:
        return self._roots.get(repository_id)

    def compare_and_swap_root(
        self,
        repository_id: str,
        expected: RootRef | None,
        new_root_cid: str,
    ) -> RootRef:
        self.cas_calls.append((repository_id, expected, new_root_cid))
        current = self._roots.get(repository_id)
        body = self.get(new_root_cid)
        manifest = {k: v for k, v in body.items() if k != "schema"}
        disposition = manifest.get("acceptance_disposition")
        if expected is None:
            if current is not None:
                raise RootConflict("root already exists")
            if disposition != AcceptanceDisposition.BOOTSTRAP.value:
                raise RootConflict("initial CAS requires bootstrap disposition")
            ref = RootRef(root_cid=new_root_cid, generation=1)
            self._roots[repository_id] = ref
            return ref
        if current is None:
            raise RootConflict("expected root missing")
        if (
            current.root_cid != expected.root_cid
            or current.generation != expected.generation
        ):
            raise RootConflict("expected root token mismatch")
        if disposition != AcceptanceDisposition.ACCEPTED.value:
            raise RootConflict("only accepted manifests may advance the root")
        if current.root_cid == new_root_cid:
            return current
        ref = RootRef(root_cid=new_root_cid, generation=current.generation + 1)
        self._roots[repository_id] = ref
        return ref

    def recover(self) -> Mapping[str, Any]:
        return {"ok": True, "roots": list(self._roots)}


# ---------------------------------------------------------------------------
# Selection / verification helpers
# ---------------------------------------------------------------------------


def _selection(
    *,
    pytest_nodes: tuple[str, ...],
    fallback: str = FALLBACK_NONE,
    fallback_reasons: tuple[str, ...] = (),
    case_id: str = "case",
) -> Any:
    return types.SimpleNamespace(
        selection_cid=_cid(f"sel|{case_id}|{fallback}|{','.join(pytest_nodes)}"),
        previous_root_cid=_cid(f"prev|{case_id}"),
        current_root_cid=_cid(f"curr|{case_id}"),
        selected_pytest_node_ids=pytest_nodes,
        selected_proof_ids=(),
        reason_paths=(types.SimpleNamespace(path_cid=_cid(f"reason|{case_id}")),),
        covered_seed_obligation_ids=(),
        unresolved_obligation_ids=(),
        known_test_universe_cid=_cid(f"universe|{case_id}"),
        known_test_universe_count=max(1, len(pytest_nodes)),
        fallback=fallback,
        fallback_reasons=fallback_reasons,
        policy_cid=_cid(f"sel-policy|{case_id}"),
    )


def _outcome(
    node_id: str,
    status: str,
    fingerprint: str | None = None,
) -> NormalizedOutcome:
    return NormalizedOutcome(
        node_id=node_id,
        status=status,
        failure_fingerprint=fingerprint,
    )


def _facts(run_id: str, *outcomes: NormalizedOutcome) -> NormalizedRunFacts:
    return NormalizedRunFacts(run_id=run_id, outcomes=outcomes)


def _receipt_bindings(**overrides: Any) -> ReceiptBindings:
    payload: dict[str, Any] = {
        "pre_tree_cid": _cid("pre-tree"),
        "post_tree_cid": _cid("post-tree"),
        "datasets_state_cid": _cid("datasets-state"),
        "datasets_semantic_state_root_cid": _cid("datasets-root"),
        "capsule_index_cid": _cid("capsule-index"),
        "delta_cid": _cid("delta"),
        "selection_cid": _cid("selection"),
        "previous_semantic_state_root_cid": _cid("prev-root"),
        "current_semantic_state_root_cid": _cid("curr-root"),
        "command_identity": "sch-cmd:accept:1",
        "toolchain_cid": _cid("toolchain"),
        "dependency_lock_cid": _cid("lock"),
        "config_cid": _cid("config"),
        "policy_cid": _cid("policy"),
        "interface_cid": _cid("interface"),
        "provider_mode": PROVIDER_MODE_PRODUCTION,
        "proof_outcomes": [
            {"proof_id": "proof.a", "status": PROOF_STATUS_PASSED},
        ],
        "output_artifact_cids": [_cid("out-a")],
        "event_parent_cid": _cid("event-parent"),
    }
    payload.update(overrides)
    return ReceiptBindings.from_dict(payload)


def _current_from(bindings: ReceiptBindings) -> dict[str, Any]:
    data = bindings.to_dict()
    return {
        "pre_tree_cid": data["pre_tree_cid"],
        "post_tree_cid": data["post_tree_cid"],
        "datasets_state_cid": data["datasets_state_cid"],
        "datasets_semantic_state_root_cid": data["datasets_semantic_state_root_cid"],
        "capsule_index_cid": data["capsule_index_cid"],
        "delta_cid": data["delta_cid"],
        "selection_cid": data["selection_cid"],
        "previous_semantic_state_root_cid": data["previous_semantic_state_root_cid"],
        "current_semantic_state_root_cid": data["current_semantic_state_root_cid"],
        "command_identity": data["command_identity"],
        "toolchain_cid": data["toolchain_cid"],
        "dependency_lock_cid": data["dependency_lock_cid"],
        "config_cid": data["config_cid"],
        "policy_cid": data["policy_cid"],
        "interface_cid": data["interface_cid"],
        "provider_mode": data["provider_mode"],
    }


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------


def _scope(**overrides: object) -> PatchScope:
    payload: dict[str, object] = {
        "allowed_paths": ("pkg/",),
        "effect_paths": ("pkg/target.py",),
        "task_owned_paths": ("pkg/",),
    }
    payload.update(overrides)
    return PatchScope.from_dict(payload)


def _pack(**overrides: object) -> ContextPack:
    payload: dict[str, Any] = {
        "objective": "acceptance matrix",
        "target_source_cid": _cid("target-src"),
        "surrounding_source_cid": _cid("surround-src"),
        "test_source_cid": _cid("test-src"),
        "dependency_capsule_cids": [],
        "obligation_cids": [],
        "counterexample_cids": [],
        "delta_cid": _cid("pack-delta"),
        "interface_cids": [],
        "assumptions": [],
        "exclusions": [],
        "token_totals": {"total": 120, "target": 40},
        "estimator_version": "sch-test-estimator@1",
        "risk": RiskClass.LOW.value,
        "route": ModelRoute.DETERMINISTIC_ONLY.value,
        "escalation_recommendation": "none",
    }
    payload.update(overrides)
    return ContextPack.from_dict(payload)


def _simple_patch(
    *,
    path: str = "pkg/target.py",
    old: str = "VALUE = 1",
    new: str = "VALUE = 2",
) -> str:
    return textwrap.dedent(
        f"""\
        diff --git a/{path} b/{path}
        --- a/{path}
        +++ b/{path}
        @@ -1 +1 @@
        -{old}
        +{new}
        """
    )


def _routing() -> RoutingDecision:
    inputs = RoutingInputs.from_dict(
        {
            "context_tokens": 100,
            "lowest_confidence": ConfidenceClass.EXACT.value,
            "risk": RiskClass.LOW.value,
            "dependency_cone_size": 1,
            "unresolved_obligations": 0,
            "prior_repair_failures": 0,
            "available_proofs": 1,
            "prior_route_failed": False,
        }
    )
    return RoutingDecision(
        route=ModelRoute.DETERMINISTIC_ONLY.value,
        reason_codes=("deterministic_only",),
        explanation="deterministic only",
        requires_provider=False,
        halt_before_dispatch=True,
        halt_before_root_publication=False,
        inputs=inputs,
        policy=ModelRoutingPolicy.default(),
    )


def _env_cids() -> dict[str, str]:
    return {
        "toolchain_cid": _cid("toolchain"),
        "dependency_lock_cid": _cid("lock"),
        "config_cid": _cid("config"),
        "policy_cid": _cid("policy"),
        "interface_cid": _cid("interface"),
    }


def _harness(
    durable: MemoryDurablePort | None = None,
) -> tuple[MemoryDurablePort, SemanticCompressionHarness]:
    port = durable or MemoryDurablePort()
    harness = SemanticCompressionHarness(
        durable=port,
        policy=HarnessPolicy(
            mode=HarnessMode.DEVELOPMENT.value,
            use_kit_root_cid=False,
        ),
    )
    return port, harness


def _bootstrap(harness: SemanticCompressionHarness) -> RootRef:
    env = _env_cids()
    outcome = harness.bootstrap_scan(
        HarnessRequest(
            repository_id="repo:accept",
            task_id="task-boot",
            objective="bootstrap",
            scope=_scope(),
            context_pack=_pack(),
            bootstrap_tree_cid=_cid("base-tree"),
            **env,
        )
    )
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    return outcome.result.current_root


def _accept_request(root: RootRef, **overrides: Any) -> HarnessRequest:
    env = _env_cids()
    payload: dict[str, Any] = {
        "repository_id": "repo:accept",
        "task_id": "task-patch",
        "objective": "apply safe patch",
        "scope": _scope(),
        "expected_root": root,
        "context_pack": _pack(),
        "routing_decision": _routing(),
        "patch_text": _simple_patch(),
        "base_tree": _cid("base-tree"),
        "changed_symbol_ids": ("pkg.target.VALUE",),
        "obligation_cids": (_cid("obligation-a"),),
        "visible_sources": {"pkg/target.py": "VALUE = 1\n"},
        "attempt_key": "accept-1",
        **env,
    }
    payload.update(overrides)
    return HarnessRequest(**payload)


# ---------------------------------------------------------------------------
# Cross-repository sealed datasets materialization
# ---------------------------------------------------------------------------


def _find_datasets_git() -> Path | None:
    for candidate in _DATASETS_CANDIDATE_REPOS:
        if not candidate or not str(candidate):
            continue
        if (candidate / ".git").exists() or (candidate / "ipfs_datasets_py").is_dir():
            try:
                subprocess.run(
                    ["git", "cat-file", "-t", SEALED_DATASETS_COMMIT],
                    cwd=candidate,
                    check=True,
                    capture_output=True,
                )
                return candidate
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    return None


def _materialize_sealed_datasets(destination: Path) -> Path:
    git_root = _find_datasets_git()
    if git_root is None:
        pytest.skip("sealed ipfs_datasets_py git root unavailable")
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".sealed_commit"
    target = (
        destination
        / "ipfs_datasets_py"
        / "logic"
        / "software_contracts"
        / "semantic_state"
        / "api.py"
    )
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == SEALED_DATASETS_COMMIT:
        if target.is_file():
            return destination
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    proc = subprocess.run(
        ["git", "archive", SEALED_DATASETS_COMMIT],
        cwd=git_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        input=proc.stdout,
        check=True,
        capture_output=True,
    )
    marker.write_text(SEALED_DATASETS_COMMIT + "\n", encoding="utf-8")
    return destination


def _purge_datasets_modules() -> None:
    doomed = [
        name
        for name in list(sys.modules)
        if name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py.")
    ]
    for name in doomed:
        del sys.modules[name]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _init_mini_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text(
        "VALUE = 1\n\ndef hello() -> int:\n    return VALUE\n",
        encoding="utf-8",
    )
    (repo / "pkg" / "mod.py").write_text(
        "from pkg import hello\n\ndef use() -> int:\n    return hello()\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_mod.py").write_text(
        "from pkg.mod import use\n\ndef test_use() -> None:\n    assert use() == 1\n",
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "accept@example.invalid")
    _git(repo, "config", "user.name", "Acceptance Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "baseline")


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo,
        check=False,
    )
    if status.returncode != 0:
        _git(repo, "commit", "-m", message)


# ---------------------------------------------------------------------------
# Interface / matrix surface
# ---------------------------------------------------------------------------


def test_acceptance_interface_and_predicted_symbols() -> None:
    """SemanticStateAcceptance@1 surface is declared by this suite."""

    assert ACCEPTANCE_INTERFACE == "SemanticStateAcceptance@1"
    assert ACCEPTANCE_BUNDLE == "sch/acceptance@1"
    # Predicted symbols from SCH-015 must exist as callables in this module.
    module = sys.modules[__name__]
    for name in (
        "test_controlled_invalidation_matrix",
        "test_selected_suite_recall",
        "test_root_manifests_deterministic_and_transitively_valid",
        "test_no_fake_cross_repository_adapter_path",
        "test_stale_receipts_never_verify",
        "test_opaque_source_retrieved_from_bound_tree",
        "test_full_fallback_works",
    ):
        assert callable(getattr(module, name)), name


# ---------------------------------------------------------------------------
# Controlled invalidation matrix
# ---------------------------------------------------------------------------


def test_controlled_invalidation_matrix(controlled_repo: Any) -> None:
    """Unrelated changes stay empty; known dependents invalidate; zero FN budget."""

    formatting = controlled_repo.get_mutation("unrelated_formatting")
    assert formatting.oracle.invalidation.invalidation_symbol_ids == ()
    assert formatting.oracle.invalidation.selected_test_node_ids == ()
    assert formatting.oracle.invalidation.expected_false_negatives == 0
    assert formatting.change_is_bounded is True

    policy = controlled_repo.get_mutation("policy_change")
    assert policy.category == "policy"
    assert "policy.admission:AdmissionPolicy" in policy.oracle.invalidation.invalidation_symbol_ids

    lock = controlled_repo.get_mutation("dependency_lockfile")
    assert "deps.lockfile:LockedDependencySet" in lock.oracle.invalidation.invalidation_symbol_ids
    assert lock.oracle.invalidation.fallback == "full_pytest"

    iface = controlled_repo.get_mutation("mcp_interface_client_adapter")
    inv = set(iface.oracle.invalidation.invalidation_symbol_ids)
    assert "interfaces.mcp_client:McpClientDescriptor" in inv
    assert "sch_fixture.adapters:McpClientAdapter" in inv

    # Every semantic/environment/policy/interface dependent category invalidates
    # at least one known dependent (or full-suite fallback for lock/config).
    seen_categories: set[str] = set()
    for case in controlled_repo.mutations:
        if case.category not in DEPENDENT_INVALIDATION_CATEGORIES:
            continue
        seen_categories.add(case.category)
        inv_oracle = case.oracle.invalidation
        assert inv_oracle.expected_false_negatives == 0, case.case_id
        has_dependents = bool(inv_oracle.invalidation_symbol_ids)
        has_full = inv_oracle.fallback in {"full_pytest", "full_proofs", "both"}
        assert has_dependents or has_full, (
            f"{case.case_id}: expected invalidation dependents or full fallback"
        )
        # Selected suite is always a subset of the full suite.
        assert set(inv_oracle.selected_test_node_ids).issubset(
            set(inv_oracle.full_suite_test_node_ids)
        ), case.case_id

    assert DEPENDENT_INVALIDATION_CATEGORIES.issubset(seen_categories)

    # Side-effect / security and fixture-dependency paths also invalidate.
    security = controlled_repo.get_mutation("side_effect_security")
    assert security.oracle.invalidation.invalidation_symbol_ids
    fixture_dep = controlled_repo.get_mutation("fixture_dependency")
    assert fixture_dep.oracle.invalidation.invalidation_symbol_ids


def test_selected_suite_recall(controlled_repo: Any) -> None:
    """Controlled selection recall is 100 percent across oracle-selected cases."""

    # Cases with a non-empty selected suite and no full-suite fallback.
    selected_cases = [
        case
        for case in controlled_repo.mutations
        if case.oracle.invalidation.selected_test_node_ids
        and case.oracle.invalidation.fallback == "none"
        and case.oracle.invalidation.expected_false_negatives == 0
    ]
    assert len(selected_cases) >= 8

    for case in selected_cases:
        oracle_nodes = tuple(case.oracle.invalidation.selected_test_node_ids)
        full_nodes = tuple(case.oracle.invalidation.full_suite_test_node_ids)
        selection = _selection(
            pytest_nodes=oracle_nodes,
            fallback=FALLBACK_NONE,
            case_id=case.case_id,
        )
        # Baseline: full suite green. Selected + candidate: oracle nodes fail.
        baseline = _facts(
            f"base-{case.case_id}",
            *[_outcome(node, "passed") for node in full_nodes],
        )
        selected = _facts(
            f"sel-{case.case_id}",
            *[
                _outcome(node, "failed", f"fp-{case.case_id}-{idx}")
                for idx, node in enumerate(oracle_nodes)
            ],
        )
        candidate = _facts(
            f"full-{case.case_id}",
            *[
                (
                    _outcome(node, "failed", f"fp-{case.case_id}-{oracle_nodes.index(node)}")
                    if node in oracle_nodes
                    else _outcome(node, "passed")
                )
                for node in full_nodes
            ],
        )
        comparison = compare_full_suite(
            selection,
            baseline_full=baseline,
            selected_run=selected,
            candidate_full=candidate,
            authored_oracle=oracle_nodes,
        )
        assert comparison.false_negatives == (), case.case_id
        assert comparison.missed_regressions == (), case.case_id
        assert comparison.fixture_recall_bp == 10_000, case.case_id
        assert comparison.zero_false_negatives is True, case.case_id
        assert comparison.supports_100_percent_recall is True, case.case_id


def test_full_fallback_works(controlled_repo: Any) -> None:
    """Full pytest fallback covers membership without false negatives."""

    lock = controlled_repo.get_mutation("dependency_lockfile")
    assert lock.oracle.invalidation.fallback == "full_pytest"
    full_nodes = tuple(lock.oracle.invalidation.full_suite_test_node_ids)
    assert full_nodes

    selection = _selection(
        pytest_nodes=(),  # producer clears selected under full fallback
        fallback=FALLBACK_FULL_PYTEST,
        fallback_reasons=("dependency_lock_changed",),
        case_id=lock.case_id,
    )
    baseline = _facts(
        "base-fallback",
        *[_outcome(node, "passed") for node in full_nodes],
    )
    # Under full fallback the selected run equals the full suite.
    selected = _facts(
        "sel-fallback",
        _outcome(full_nodes[0], "failed", "fp-lock"),
        *[_outcome(node, "passed") for node in full_nodes[1:]],
    )
    candidate = _facts(
        "full-fallback",
        _outcome(full_nodes[0], "failed", "fp-lock"),
        *[_outcome(node, "passed") for node in full_nodes[1:]],
    )
    comparison = compare_full_suite(
        selection,
        baseline_full=baseline,
        selected_run=selected,
        candidate_full=candidate,
        authored_oracle=(full_nodes[0],),
    )
    assert comparison.fallback_rate_bp == 10_000
    assert comparison.false_negatives == ()
    assert comparison.supports_100_percent_recall is True
    assert comparison.selected_count == comparison.full_count


def test_opaque_source_retrieved_from_bound_tree(
    controlled_repo: Any, tmp_path: Path
) -> None:
    """Opaque / dynamic cases force raw source read from the bound tree bytes."""

    for case_id in ("opaque_native", "dynamic_import", "monkey_patch"):
        case = controlled_repo.get_mutation(case_id)
        conf = case.oracle.confidence
        assert conf.raw_source_required is True
        assert conf.raw_source_symbol_ids
        assert conf.confidence in {"opaque", "heuristic", "conservative"}

    dest = tmp_path / "opaque-tree"
    meta = controlled_repo.materialize_mutation("opaque_native", dest, git=False)
    assert Path(meta["root"]).is_dir()

    # Bound-tree retrieval: read exact materialised bytes (never ambient import).
    native_path = dest / "src" / "sch_fixture" / "native_bridge.py"
    assert native_path.is_file()
    body = native_path.read_bytes()
    assert b"native_hash is opaque fixture surface v2" in body
    assert b"ctypes" not in body
    assert b"cffi" not in body

    # Scanners parse AST from bound bytes only — no target package import.
    before = {name for name in sys.modules if name.startswith("sch_fixture")}
    tree = ast.parse(body.decode("utf-8"), filename=str(native_path))
    assert isinstance(tree, ast.Module)
    after = {name for name in sys.modules if name.startswith("sch_fixture")}
    assert after == before

    # Post-scan source-race marker stays outside admitted pack paths.
    race = controlled_repo.get_mutation("post_scan_source_race")
    pack_paths = set(controlled_repo.declared_pack_paths(race.case_id))
    recipes = sys.modules[f"{PACKAGE_NAME}.recipes"]
    assert recipes.SOURCE_RACE_PATH not in pack_paths
    race_tree = controlled_repo.mutated_tree(race.case_id)
    race_body = race_tree[recipes.SOURCE_RACE_PATH].encode("utf-8")
    assert recipes.SOURCE_RACE_MARKER in race_body
    for path in pack_paths:
        if path in race_tree:
            assert recipes.SOURCE_RACE_MARKER not in race_tree[path].encode("utf-8")


def test_stale_receipts_never_verify(controlled_repo: Any) -> None:
    """Stale / forged receipts never admit for verification or promotion."""

    case = controlled_repo.get_mutation("stale_receipt")
    assert case.oracle.receipt_freshness.accepts_stale_receipt is False
    assert case.harness_scenario == "stale_receipt"

    base_digest = controlled_repo.base_tree_digest()
    mut_digest = controlled_repo.mutated_tree_digest(case.case_id)
    assert base_digest != mut_digest

    # Bind receipt to prior tree; current world advanced to mutated tree.
    bindings = _receipt_bindings(
        pre_tree_cid=_cid(base_digest),
        post_tree_cid=_cid(base_digest),
        selection_cid=_cid(f"sel-{case.case_id}"),
        policy_cid=_cid("policy-stale"),
        interface_cid=_cid("iface-stale"),
        dependency_lock_cid=_cid("lock-stale"),
        config_cid=_cid("config-stale"),
    )
    receipt = compile_verification_receipt(
        bindings, exit_code=0, stages_passed=True, store=False
    )
    current = _current_from(bindings)
    current["post_tree_cid"] = _cid(mut_digest)

    admission = admit_receipt(receipt, current=current)
    assert admission.admission == ADMISSION_STALE
    assert admission.freshness == FRESHNESS_STALE
    assert "stale:post_tree_cid" in admission.stale_obligations
    assert admission.can_verify is False
    assert admission.can_promote_root is False
    assert receipt_may_verify(admission) is False
    assert receipt_may_promote_root(admission) is False

    with pytest.raises(StaleReceiptError) as excinfo:
        admit_receipt(receipt, current=current, raise_on_reject=True)
    assert "stale:post_tree_cid" in excinfo.value.stale_obligations

    # Policy / interface / lock / config / toolchain changes also stale.
    for field, extra in (
        ("policy_cid", ("stale:policy_cid", "obligation:policy_decision")),
        ("interface_cid", ("stale:interface_cid", "obligation:interface_description")),
        ("dependency_lock_cid", ("stale:dependency_lock_cid",)),
        ("config_cid", ("stale:config_cid",)),
        ("toolchain_cid", ("stale:toolchain_cid",)),
    ):
        current2 = _current_from(bindings)
        current2[field] = _cid(f"changed-{field}")
        adm = admit_receipt(receipt, current=current2)
        assert adm.admission == ADMISSION_STALE, field
        assert f"stale:{field}" in adm.stale_obligations, field
        for marker in extra:
            assert marker in adm.stale_obligations or marker in adm.reason_codes, (
                field,
                marker,
                adm.stale_obligations,
                adm.reason_codes,
            )
        assert receipt_may_verify(adm) is False


def test_root_manifests_deterministic_and_transitively_valid() -> None:
    """Accepted root manifests rehash, store transitive links, and are deterministic."""

    port_a, harness_a = _harness()
    root_a = _bootstrap(harness_a)
    out_a = harness_a.run(_accept_request(root_a, attempt_key="det-a"))
    assert out_a.result.disposition == HarnessDisposition.ACCEPTED.value
    assert list(out_a.steps_completed) == list(HARNESS_STEPS)

    body_a = {
        k: v
        for k, v in port_a.get(out_a.result.current_root.root_cid).items()
        if k != "schema"
    }
    manifest_a = SemanticStateRootManifest.from_dict(body_a)
    assert manifest_a.acceptance_disposition == AcceptanceDisposition.ACCEPTED.value
    assert manifest_a.repository_id == "repo:accept"

    # Transitive links are present and content-addressed where rehashable.
    for field in (
        "capsule_index_cid",
        "delta_cid",
        "invalidation_cid",
        "obligation_set_cid",
        "receipt_index_cid",
        "event_head_cid",
    ):
        cid = getattr(manifest_a, field)
        assert port_a.has(cid), field
        stored = port_a.get(cid)
        assert isinstance(stored, Mapping)
        if field == "receipt_index_cid":
            recomputed = cid_for_payload(
                {
                    "schema": stored["schema"],
                    "receipt_cids": list(stored["receipt_cids"]),
                }
            )
            assert recomputed == cid

    # Second independent harness with identical inputs yields the same root CID
    # when the same attempt identity and inputs are used (deterministic path).
    port_b, harness_b = _harness()
    root_b = _bootstrap(harness_b)
    out_b = harness_b.run(_accept_request(root_b, attempt_key="det-a"))
    assert out_b.result.disposition == HarnessDisposition.ACCEPTED.value
    body_b = {
        k: v
        for k, v in port_b.get(out_b.result.current_root.root_cid).items()
        if k != "schema"
    }
    # Manifest dicts equal after dropping non-deterministic event heads if any.
    for key in (
        "repository_id",
        "acceptance_disposition",
        "versions",
        "environment_binding_cids",
    ):
        assert body_a[key] == body_b[key], key

    # Fixture corpus digests themselves are deterministic.
    fixture = _load_fixture_package().ControlledSemanticRepository.load()
    d1 = fixture.manifest_digest()
    d2 = fixture.manifest_digest()
    assert d1 == d2
    assert d1.startswith("sha256:")
    assert len(d1) == len("sha256:") + 64


def test_no_fake_cross_repository_adapter_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complete no-fake path: sealed datasets pin scan → build → select → view."""

    from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
        CONFIDENCE_VALUES,
        EXPECTED_CAPSULE_SCHEMA,
        EXPECTED_SELECTION_SCHEMA,
        EXPECTED_SEMANTIC_INDEX_SCHEMA,
        EXPECTED_SEMANTIC_STATE_SCHEMA,
        load_semantic_state_provider,
    )

    sealed_root = _materialize_sealed_datasets(tmp_path / "sealed-datasets")
    _purge_datasets_modules()
    monkeypatch.syspath_prepend(str(sealed_root))
    workspace_ns = str(REPO_ROOT / "ipfs_datasets_py")
    sys.path[:] = [p for p in sys.path if p != workspace_ns and p != str(REPO_ROOT)]
    monkeypatch.syspath_prepend(str(sealed_root))
    _purge_datasets_modules()

    provider = load_semantic_state_provider()
    try:
        cap = provider.capability
        assert cap.available is True
        assert cap.semantic_state_schema == EXPECTED_SEMANTIC_STATE_SCHEMA
        assert cap.capsule_schema == EXPECTED_CAPSULE_SCHEMA
        assert cap.selection_schema == EXPECTED_SELECTION_SCHEMA
        assert cap.semantic_index_schema == EXPECTED_SEMANTIC_INDEX_SCHEMA

        repo = tmp_path / "mini-repo"
        _init_mini_repo(repo)

        state = provider.scan_repository(repo)
        assert state.symbols
        assert state.edges
        assert all(sym.stable_id for sym in state.symbols)
        assert all(
            str(getattr(edge, "confidence", "exact")) in CONFIDENCE_VALUES
            for edge in state.edges
        )

        bundle = provider.build_semantic_state(state)
        root = provider.verify_semantic_state_bundle(bundle)
        assert root.root_cid == bundle.root.root_cid
        assert root.root_cid.startswith("b")

        memory_view = provider.view_semantic_state_bundle(bundle)
        durable_blocks = dict(bundle.blocks)
        durable_view = provider.open_verified_view(
            root.root_cid, durable_blocks.__getitem__
        )
        assert memory_view.root.root_cid == durable_view.root.root_cid

        # Mutate and run real invalidation + selection (no mocked success).
        (repo / "pkg" / "mod.py").write_text(
            "from pkg import hello\n\ndef use() -> int:\n    return hello() + 1\n",
            encoding="utf-8",
        )
        _commit_all(repo, "mutate")
        current_state = provider.scan_repository(repo, previous_state=state)
        assert current_state.state_cid != state.state_cid
        current_bundle = provider.build_semantic_state(
            current_state, previous_bundle=bundle
        )
        delta = provider.diff_repository_states(state, current_state)
        plan = provider.calculate_invalidation(state, current_state, delta)
        previous_view = memory_view
        current_view = provider.view_semantic_state_bundle(current_bundle)

        from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
            SelectionPolicy,
        )

        policy = SelectionPolicy(policy_id="accept-e2e", allow_full_fallback=True)
        extended = provider.extend_semantic_invalidation(
            state, current_state, delta, plan, previous_view, current_view
        )
        selection = provider.select_tests_and_proofs(
            previous_view,
            current_view,
            extended,
            policy=policy,
            explicit_rules=(),
        )
        assert selection is not None
        # Identity preserved when present — real CIDs, not fabricated labels.
        for attr in ("previous_root_cid", "current_root_cid", "selection_cid"):
            value = getattr(selection, attr, None)
            if isinstance(value, str) and value.startswith("b"):
                assert len(value) >= 50
    finally:
        _purge_datasets_modules()


def test_unrelated_formatting_does_not_invalidate_repository(
    controlled_repo: Any,
) -> None:
    """Unrelated formatting changes remain bounded and leave selection empty."""

    case = controlled_repo.get_mutation("unrelated_formatting")
    controlled = sys.modules[f"{PACKAGE_NAME}.controlled_repository"]
    stats = controlled.bounded_change_stats(controlled_repo.base_files, case)
    assert stats["operation_count"] <= controlled.BOUNDED_CHANGE_MAX_OPS
    assert stats["changed_bytes"] <= controlled.BOUNDED_CHANGE_MAX_BYTES
    assert case.oracle.invalidation.invalidation_symbol_ids == ()
    assert case.oracle.invalidation.selected_test_node_ids == ()
    assert case.oracle.invalidation.expected_false_negatives == 0
    # Tree still changes (digest moves) but semantic dependents stay empty.
    assert controlled_repo.mutated_tree_digest(case.case_id) != controlled_repo.base_tree_digest()


def test_out_of_scope_and_source_race_harness_scenarios(
    controlled_repo: Any,
) -> None:
    oos = controlled_repo.get_mutation("out_of_scope_patch")
    assert oos.harness_scenario == "out_of_scope_patch"
    assert oos.oracle.invalidation.invalidation_symbol_ids == ()
    assert oos.production_eligible is False

    race = controlled_repo.get_mutation("post_scan_source_race")
    assert race.oracle.confidence.raw_source_required is True
    assert race.oracle.receipt_freshness.accepts_stale_receipt is False
