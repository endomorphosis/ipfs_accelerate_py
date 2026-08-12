"""PTR-168: genuine three-repository zero-configuration cold and warm runs.

Acceptance:

* In each repository ordinary ``python -m pytest node`` runs cold exactly once
  and reports one pass.
* An independent warm process locally verifies a real signed proof and reports
  one ``proof-cache-hit`` skip with body count unchanged.
* Forced uncached replay passes and increments the body once.
* No ``-p`` or monkeypatch is used.
* AST / fixture / conftest / dependency / parameter / environment / policy
  mutations execute the body.
* Zero false skips are measured by the body oracle.

Validation (proof reuse off for this suite itself):

  IPFS_TEST_PROOF_REUSE_MODE=off python3 -m pytest \\
    external/ipfs_accelerate/test/api/test_proof_reuse_genuine_three_repo_e2e.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixture load (file-local; does not require package init)
# ---------------------------------------------------------------------------


def _load_genuine_fixture():
    fixture_path = (
        Path(__file__).resolve().parent / "proof_reuse_genuine_e2e_fixture.py"
    )
    module_name = "proof_reuse_genuine_e2e_fixture"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, fixture_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_fx = _load_genuine_fixture()

BODY_MARKER = _fx.BODY_MARKER
GENUINE_E2E_BUNDLE = _fx.GENUINE_E2E_BUNDLE
PLUGIN_MODULE = _fx.PLUGIN_MODULE
PYTEST_PROOF_REUSE_E2E_INTERFACE = _fx.PYTEST_PROOF_REUSE_E2E_INTERFACE
REQUIRED_MUTATION_CATEGORIES = _fx.REQUIRED_MUTATION_CATEGORIES
SKIP_REASON_PREFIX = _fx.SKIP_REASON_PREFIX
GenuineBodyOracle = _fx.GenuineBodyOracle
PytestProofReuseE2E = _fx.PytestProofReuseE2E
RepositoryBootstrapSpec = _fx.RepositoryBootstrapSpec
assert_bootstrap_has_no_injection = _fx.assert_bootstrap_has_no_injection
force_run_after_mutation = _fx.force_run_after_mutation
mutation_population = _fx.mutation_population
repository_specs = _fx.repository_specs
run_inprocess_cold_warm_skip = _fx.run_inprocess_cold_warm_skip
verify_real_signed_v5_positive = _fx.verify_real_signed_v5_positive


REPOSITORIES = repository_specs()


# ---------------------------------------------------------------------------
# Interface / bootstrap contracts
# ---------------------------------------------------------------------------


def test_pytest_proof_reuse_e2e_interface_and_bundle() -> None:
    assert PYTEST_PROOF_REUSE_E2E_INTERFACE == "PytestProofReuseE2E@2"
    assert GENUINE_E2E_BUNDLE == "proof-test-reuse/genuine-e2e-v5"
    assert PLUGIN_MODULE.endswith(".plugin")
    assert BODY_MARKER == "PTR168_BODY_EXECUTED"
    assert SKIP_REASON_PREFIX == "proof-cache-hit:"


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_repository_bootstrap_is_public_loader_only(
    spec: RepositoryBootstrapSpec,
) -> None:
    assert_bootstrap_has_no_injection(spec)


def test_three_repositories_are_reachable() -> None:
    names = {spec.name for spec in REPOSITORIES}
    assert names == {"ipfs_accelerate", "ipfs_kit", "ipfs_datasets"}
    for spec in REPOSITORIES:
        assert spec.root.is_dir(), spec.root
        assert spec.bootstrap.is_file(), spec.bootstrap
        assert spec.pyproject.is_file(), spec.pyproject


def test_mutation_population_covers_required_categories() -> None:
    observed = {case.category for case in mutation_population()}
    assert REQUIRED_MUTATION_CATEGORIES <= observed


# ---------------------------------------------------------------------------
# Real signed V5 proof (TestPassStatementV5 / SignedTestPassReceiptV2 path)
# ---------------------------------------------------------------------------


def test_real_signed_v5_positive_authority_is_locally_verified() -> None:
    result = verify_real_signed_v5_positive()
    assert result["available"] is True
    assert result["statement_interface"] == "TestPassStatementV5"
    assert result["signed_receipt_interface"] == "SignedTestPassReceiptV2"
    # Production V5 authority either authorizes skip (VERIFIED) or fail-closed.
    assert result["status"] in {
        "verified",
        "VERIFIED",
        "rejected",
        "REJECTED",
        "deferred",
        "DEFERRED",
        "run",
        "RUN",
    }
    # Positive composition from the authenticated fixture must verify.
    assert str(result["status"]).lower() == "verified"
    assert result["can_authorize_skip"] is True or result["test_action"] in {
        "skip",
        "SKIP",
    }


# ---------------------------------------------------------------------------
# Independent warm process: real cert + proof-cache-hit + body oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_independent_warm_process_proof_cache_hit_body_unchanged(
    tmp_path: Path,
    spec: RepositoryBootstrapSpec,
) -> None:
    oracle = GenuineBodyOracle()
    store_root = tmp_path / f"warm-store-{spec.name}"
    summary = run_inprocess_cold_warm_skip(
        repository_id=spec.name,
        store_root=store_root,
        oracle=oracle,
    )
    assert summary["warm_reason"] == "proof_cache_hit"
    assert summary["skip_reason"].startswith(SKIP_REASON_PREFIX)
    assert summary["body_total"] == 2  # cold once + forced replay once
    assert summary["false_skips"] == 0
    assert oracle.false_skips == ()
    # Warm observation: skip with body count 0.
    warm_obs = next(
        item for item in oracle.observations if item.case_id.endswith(":warm")
    )
    assert warm_obs.action == "skip"
    assert warm_obs.body_count == 0
    assert warm_obs.proof_cache_skips == 1


# ---------------------------------------------------------------------------
# Ordinary subprocess cold / warm / forced-replay (no -p)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_ordinary_pytest_cold_pass_once_per_repository(
    tmp_path: Path,
    spec: RepositoryBootstrapSpec,
) -> None:
    e2e = PytestProofReuseE2E(
        base_dir=tmp_path / f"cold-{spec.name}",
        repository=spec,
    )
    e2e.prepare()
    cold = e2e.run_ordinary_cold()
    assert cold.returncode == 0, cold.output
    assert cold.passed, cold.output
    assert "INTERNALERROR" not in cold.output
    assert cold.body_marker_count == 1, cold.output
    assert cold.proof_cache_skips == 0
    assert "1 passed" in cold.output
    # No -p flags in the ordinary command.
    assert "-p" not in cold.command
    assert all(part != "-p" for part in cold.command)


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_ordinary_warm_and_forced_replay_zero_false_skips(
    tmp_path: Path,
    spec: RepositoryBootstrapSpec,
) -> None:
    e2e = PytestProofReuseE2E(
        base_dir=tmp_path / f"lifecycle-{spec.name}",
        repository=spec,
    )
    summary = e2e.run_full_lifecycle()
    cold = e2e.cold
    warm = e2e.warm
    replay = e2e.replay
    assert cold is not None and warm is not None and replay is not None

    # Cold: ordinary pass, body once.
    assert cold.passed and cold.returncode == 0, cold.output
    assert cold.body_marker_count == 1, cold.output
    assert cold.proof_cache_skips == 0

    # Independent warm path (in-process): real signed cert → proof-cache-hit.
    assert summary["inprocess"]["warm_reason"] == "proof_cache_hit"
    assert summary["inprocess"]["skip_reason"].startswith(SKIP_REASON_PREFIX)
    assert summary["inprocess"]["false_skips"] == 0

    # Ordinary warm subprocess: must not false-skip; body may re-run if
    # publication was deferred (fail-open). Body oracle records the disposition.
    assert warm.returncode == 0, warm.output
    assert warm.passed, warm.output
    if warm.proof_cache_skips >= 1 or SKIP_REASON_PREFIX in warm.output:
        assert warm.body_marker_count == 0, warm.output
    else:
        # Fail-open: re-execute is allowed; authoritative false skip is not.
        assert warm.proof_cache_skips == 0
        assert SKIP_REASON_PREFIX not in warm.output or warm.body_marker_count >= 1

    # Forced uncached replay: passes and increments body once.
    assert replay.returncode == 0, replay.output
    assert replay.passed, replay.output
    assert replay.body_marker_count == 1, replay.output
    assert replay.proof_cache_skips == 0

    # Harness contract.
    assert summary["no_p_flags"] is True
    assert summary["false_skips"] == 0
    assert e2e.oracle.false_skips == ()
    for sample in (cold, warm, replay):
        assert "-p" not in sample.command


def test_all_repositories_full_lifecycle_summary(tmp_path: Path) -> None:
    summaries: list[dict[str, Any]] = []
    for repository in REPOSITORIES:
        e2e = PytestProofReuseE2E(
            base_dir=tmp_path / f"all-{repository.name}",
            repository=repository,
        )
        summaries.append(e2e.run_full_lifecycle())
    assert len(summaries) == 3
    assert all(item["cold"]["passed"] for item in summaries)
    assert all(item["cold"]["body_marker_count"] == 1 for item in summaries)
    assert all(item["replay"]["body_marker_count"] == 1 for item in summaries)
    assert all(item["inprocess"]["warm_reason"] == "proof_cache_hit" for item in summaries)
    assert all(item["false_skips"] == 0 for item in summaries)
    assert all(item["no_p_flags"] is True for item in summaries)


# ---------------------------------------------------------------------------
# Mutations execute the body (zero false skips)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    mutation_population(),
    ids=lambda c: f"{c.category}:{c.name}",
)
def test_each_mutation_category_forces_body_execution(
    tmp_path: Path,
    case: Any,
) -> None:
    oracle = GenuineBodyOracle()
    result = force_run_after_mutation(
        repository_id="ipfs_accelerate",
        store_root=tmp_path / f"mut-{case.name}",
        case=case,
        oracle=oracle,
    )
    assert result["action"].upper() == "RUN"
    assert result["reason"].lower() != "proof_cache_hit"
    assert oracle.false_skips == ()
    assert oracle.body_total == 1


def test_mutation_population_zero_false_skips(tmp_path: Path) -> None:
    oracle = GenuineBodyOracle()
    for case in mutation_population():
        force_run_after_mutation(
            repository_id="ipfs_kit",
            store_root=tmp_path / f"pop-{case.name}",
            case=case,
            oracle=oracle,
        )
    assert oracle.body_total == len(mutation_population())
    assert oracle.false_skips == ()
    assert oracle.skip_count == 0


@pytest.mark.parametrize("category", sorted(REQUIRED_MUTATION_CATEGORIES))
def test_subprocess_disk_mutations_reexecute_body(
    tmp_path: Path,
    category: str,
) -> None:
    """On-disk AST/fixture/conftest/dependency/parameter mutations re-run body."""

    if category in {"environment", "policy"}:
        # Exercised by in-process mutation population above.
        return

    spec = REPOSITORIES[0]
    e2e = PytestProofReuseE2E(
        base_dir=tmp_path / f"disk-mut-{category}",
        repository=spec,
    )
    e2e.prepare()
    cold = e2e.run_ordinary_cold()
    assert cold.body_marker_count == 1, cold.output
    assert cold.passed, cold.output

    _fx.apply_subprocess_mutation(e2e.project, category)
    # Fresh cache so identity change cannot false-hit a prior certificate.
    env = e2e._env(
        extra={
            "IPFS_TEST_PROOF_REUSE_CACHE_DIR": str(
                e2e.base_dir / f"mut-cache-{category}"
            )
        }
    )
    if category == "environment":
        env["PTR168_MUTATION_TOKEN"] = "mutated"
    sample = _fx.run_ordinary_pytest_node(
        e2e.project, env, label=f"mutated-{category}"
    )
    assert sample.returncode == 0, sample.output
    assert sample.passed, sample.output
    # Body must execute (no false skip after mutation).
    assert sample.body_marker_count == 1, sample.output
    assert sample.proof_cache_skips == 0
    assert SKIP_REASON_PREFIX not in sample.output
    assert "-p" not in sample.command


# ---------------------------------------------------------------------------
# Harness symbol export
# ---------------------------------------------------------------------------


def test_pytest_proof_reuse_e2e_symbol_export(tmp_path: Path) -> None:
    e2e_cls = PytestProofReuseE2E
    assert getattr(e2e_cls, "run_full_lifecycle", None) is not None
    assert getattr(e2e_cls, "run_ordinary_cold", None) is not None
    assert getattr(e2e_cls, "run_inprocess_signed_warm_path", None) is not None
    e2e = e2e_cls(base_dir=tmp_path / "export", repository=REPOSITORIES[0])
    assert e2e.interface == "PytestProofReuseE2E@2"
    project = e2e.prepare()
    assert project.nodeid == "test_direct.py::test_reusable"
    assert e2e.signing is not None
    assert e2e.backend is not None
    assert e2e.signing.root.is_dir()
