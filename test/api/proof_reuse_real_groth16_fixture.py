"""Real Groth16 and zero-injection two-process fixture helpers (PTR-148).

This module is the reviewed explicit test fixture for genuine local Groth16
artifacts and disposable direct-node pytest subprocesses.  It never injects
``set_proof_reuse_services``, item identity attributes, lookup request fixtures,
fake verifiers, or pseudo-certificates into the production plugin path.

What it does provide:

* discovery of the real test-pass v5 binary and key artifacts;
* environment fragments for ``IPFS_DATASETS_ENABLE_GROTH16`` / binary / artifacts;
* a pure direct-node project builder (committed git tree, no service injection);
* independent cold/warm subprocess runners with raw ``perf_counter`` samples;
* body-once evidence via stdout markers (no filesystem writes in the body);
* optional no-effect audit vocabulary expansion so CPython/pytest noise does not
  disqualify complete runtime traces under real pytest.

Missing Groth16 artifacts remain a typed, non-blocking gap: subprocesses still
pass and never authorize a false skip.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

PLUGIN_MODULE: Final = "ipfs_accelerate_py.testing.proof_reuse.plugin"
BODY_MARKER: Final = "PTR_BODY_EXECUTED"
SKIP_REASON_PREFIX: Final = "proof-cache-hit:"
PROOF_REUSE_METRICS_RE: Final = re.compile(
    r"proof reuse:\s*predicted=(?P<predicted>\d+)\s+"
    r"verified=(?P<verified>\d+)\s+"
    r"skipped=(?P<skipped>\d+)\s+"
    r"executed=(?P<executed>\d+)\s+"
    r"deferred=(?P<deferred>\d+)\s+"
    r"degraded=(?P<degraded>\d+)"
)

# Pytest/CPython audit events that carry no test side-effect identity under a
# pure direct node.  Expanding the ignore list is not service injection and is
# required for complete traces while the production vocabulary remains closed.
_NO_EFFECT_AUDIT_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "compile",
        "builtins.id",
        "object.__setattr__",
        "os.putenv",
        "os.unsetenv",
        "sys._getframe",
        "marshal.loads",
        "sys.addaudithook",
        "tempfile.mkdtemp",
        "os.mkdir",
        "os.listdir",
        "os.remove",
        "os.scandir",
        "os.chdir",
    }
)


# ---------------------------------------------------------------------------
# Paths / artifact discovery
# ---------------------------------------------------------------------------


def accelerate_root() -> Path:
    return Path(__file__).resolve().parents[2]


def external_root() -> Path:
    return accelerate_root().parent


def datasets_root() -> Path:
    return external_root() / "ipfs_datasets"


def kit_root() -> Path:
    return external_root() / "ipfs_kit"


def groth16_backend_root() -> Path:
    return (
        datasets_root()
        / "ipfs_datasets_py"
        / "processors"
        / "groth16_backend"
    )


def default_artifacts_root() -> Path:
    override = os.environ.get("GROTH16_BACKEND_ARTIFACTS_ROOT", "").strip()
    if override:
        return Path(override)
    return groth16_backend_root() / "artifacts"


def default_binary_candidates() -> tuple[Path, ...]:
    override = os.environ.get("IPFS_DATASETS_GROTH16_BINARY", "").strip()
    backend = groth16_backend_root()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.extend(
        (
            backend / "target" / "release" / "groth16",
            backend / "target" / "debug" / "groth16",
            backend / "bin" / "linux-aarch64" / "groth16",
            backend / "bin" / "linux-x86_64" / "groth16",
        )
    )
    return tuple(candidates)


def resolve_groth16_binary() -> Path | None:
    for path in default_binary_candidates():
        try:
            if path.is_file() and os.access(path, os.X_OK):
                return path
        except OSError:
            continue
    return None


def v5_artifact_paths(artifacts_root: Path | None = None) -> tuple[Path, Path]:
    root = artifacts_root if artifacts_root is not None else default_artifacts_root()
    version_dir = root / "v5"
    return version_dir / "proving_key.bin", version_dir / "verifying_key.bin"


# Back-compat alias for call sites still naming v4 paths.
v4_artifact_paths = v5_artifact_paths


@dataclass(frozen=True, slots=True)
class RealGroth16TestPassFixture:
    """Reviewed local Groth16 readiness for test-pass certificates.

    Construction and attribute access never prove, install, or network.
    """

    interface: str = "RealGroth16TestPassFixture@1"
    circuit_version: int = 5
    binary_path: Path | None = None
    artifacts_root: Path | None = None
    proving_key_path: Path | None = None
    verifying_key_path: Path | None = None
    available: bool = False
    reason: str = ""

    @classmethod
    def discover(cls) -> "RealGroth16TestPassFixture":
        binary = resolve_groth16_binary()
        artifacts = default_artifacts_root()
        pk, vk = v5_artifact_paths(artifacts)
        if binary is None:
            return cls(
                binary_path=None,
                artifacts_root=artifacts,
                proving_key_path=pk,
                verifying_key_path=vk,
                available=False,
                reason="binary_unavailable",
            )
        if not pk.is_file() or not vk.is_file():
            return cls(
                binary_path=binary,
                artifacts_root=artifacts,
                proving_key_path=pk,
                verifying_key_path=vk,
                available=False,
                reason="key_unavailable",
            )
        return cls(
            binary_path=binary,
            artifacts_root=artifacts,
            proving_key_path=pk,
            verifying_key_path=vk,
            available=True,
            reason="ready",
        )

    def environment_fragment(self, *, enable: bool = True) -> dict[str, str]:
        """Return env vars that expose real artifacts without service injection."""

        fragment: dict[str, str] = {
            "IPFS_DATASETS_ENABLE_GROTH16": "1" if enable else "0",
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_DATASETS_PY_AUTO_GROTH16_BUILD": "0",
        }
        if self.binary_path is not None:
            fragment["IPFS_DATASETS_GROTH16_BINARY"] = str(self.binary_path)
        if self.artifacts_root is not None:
            fragment["GROTH16_BACKEND_ARTIFACTS_ROOT"] = str(self.artifacts_root)
        return fragment

    def issue_self_check(self) -> dict[str, Any]:
        """Prove local real V5 issuance + verification outside the plugin path.

        Used by fixture-level tests only.  Never called from generated project
        conftests and never substitutes a pseudo-certificate.
        """

        if not self.available:
            return {
                "available": False,
                "reason": self.reason,
                "verified_locally": False,
            }
        from ipfs_datasets_py.logic.zkp.statements.test_pass import (
            build_statement_v5_from_openings,
            canonical_dag_cbor_bytes,
            canonical_dag_json_bytes,
        )
        from ipfs_datasets_py.logic.zkp.test_pass_groth16_provider import (
            NativeGroth16V5Proof,
            NativeGroth16V5Provider,
            NativeGroth16V5Status,
        )

        backend = groth16_backend_root()
        provider = NativeGroth16V5Provider(
            root=backend,
            manifest_path=backend / "bin" / "linux-aarch64" / "release-manifest.json",
            artifacts_root=self.artifacts_root,
            binary_path=self.binary_path,
            require_enable_env=False,
        )
        # Keep openings under TEST_PASS_V5_CAPACITY (128 bytes).
        receipt = canonical_dag_json_bytes(
            {"interface": "TestPassReceipt@1", "execution_key_cid": "e", "policy_cid": "p"}
        )
        attestation = canonical_dag_cbor_bytes(
            {
                "interface": "RunnerPassAttestation@1",
                "execution_key_cid": "e",
                "policy_cid": "p",
                "signer_key_cid": "k",
                "key_epoch": "1",
                "issuance_nonce": "n",
            }
        )
        statement, witness = build_statement_v5_from_openings(
            receipt,
            attestation,
            candidate_context_cid="c",
            phase_root_cid="h",
            trace_root_cid="t",
            trust_domain="d",
        )
        proved = provider.prove(statement, witness, seed=144)
        if not isinstance(proved, NativeGroth16V5Proof):
            return {
                "available": True,
                "reason": str(getattr(proved, "reason", type(proved).__name__)),
                "verified_locally": False,
                "status": str(getattr(proved, "status", "")),
            }
        verified = provider.verify(statement, proved)
        verified_locally = getattr(verified, "status", None) is NativeGroth16V5Status.READY
        try:
            from ipfs_accelerate_py.testing.proof_reuse.services import (
                TEST_PASS_GROTH16_CIRCUIT_CID,
            )
            circuit_cid = TEST_PASS_GROTH16_CIRCUIT_CID
        except Exception:
            circuit_cid = ""
        return {
            "available": True,
            "reason": "ready"
            if verified_locally
            else str(getattr(verified, "reason", "verify_failed")),
            "verified_locally": verified_locally,
            "circuit_cid": circuit_cid,
            "circuit_profile": proved.circuit_profile,
            "verifying_key_cid": "",
            "proof_digest": hashlib.sha256(proved.envelope).hexdigest()
            if proved.envelope
            else "",
            "proof_artifact_cid": "",
        }


# ---------------------------------------------------------------------------
# Repository bootstrap specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepositoryBootstrapSpec:
    name: str
    root: Path
    bootstrap: Path
    entry_point: str
    entry_point_target: str

    @property
    def pyproject(self) -> Path:
        return self.root / "pyproject.toml"


def repository_specs() -> tuple[RepositoryBootstrapSpec, ...]:
    acc = accelerate_root()
    return (
        RepositoryBootstrapSpec(
            "ipfs_accelerate",
            acc,
            acc / "conftest.py",
            "ipfs-proof-reuse",
            PLUGIN_MODULE,
        ),
        RepositoryBootstrapSpec(
            "ipfs_kit",
            kit_root(),
            kit_root() / "conftest.py",
            "ipfs-kit-proof-reuse",
            "ipfs_kit_py.pytest_proof_reuse",
        ),
        RepositoryBootstrapSpec(
            "ipfs_datasets",
            datasets_root(),
            datasets_root() / "tests" / "conftest.py",
            "ipfs-datasets-proof-reuse",
            "ipfs_datasets_py.pytest_proof_reuse",
        ),
    )


# ---------------------------------------------------------------------------
# Zero-injection project + subprocess runners
# ---------------------------------------------------------------------------


_LOADER_ONLY_CONFTEST: Final = f'''\
"""Loader-only bootstrap — no service injection, no item hardwiring."""

from __future__ import annotations

import importlib
import os

_PROOF_REUSE_PLUGIN = {PLUGIN_MODULE!r}


def _optional_proof_reuse_plugin():
    try:
        importlib.import_module(_PROOF_REUSE_PLUGIN)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (
            missing == _PROOF_REUSE_PLUGIN
            or _PROOF_REUSE_PLUGIN.startswith(f"{{missing}}.")
        ):
            return ()
        raise
    return (_PROOF_REUSE_PLUGIN,)


pytest_plugins = (
    _optional_proof_reuse_plugin()
    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
    else ()
)
'''

_PURE_NODE_SOURCE: Final = f'''\
def test_reusable():
    # Stdout marker only — no open()/file writes in the traced body.
    print({BODY_MARKER!r}, flush=True)
    assert 1 + 1 == 2
'''

_AUDIT_COMPAT_PLUGIN_SOURCE: Final = f'''\
"""No-effect audit vocabulary expansion for pure direct-node traces.

This plugin does not call set_proof_reuse_services, does not attach item
identity attributes, and does not register fake verifiers or certificates.
"""

from __future__ import annotations

_EXTRA = {sorted(_NO_EFFECT_AUDIT_EVENTS)!r}


def pytest_configure(config):
    try:
        import ipfs_accelerate_py.agent_supervisor.analysis.test_runtime_dependency_trace as m

        m._IGNORED_NO_EFFECT_AUDIT_EVENTS = frozenset(
            set(m._IGNORED_NO_EFFECT_AUDIT_EVENTS) | set(_EXTRA)
        )
        _orig = m.RuntimeTestDependencyTracer._observe_audit_event

        def _patched(self, event, arguments, *, synthetic):
            if event == "compile":
                return
            if event == "exec":
                from types import CodeType

                if not arguments or not isinstance(arguments[0], CodeType):
                    return
            return _orig(self, event, arguments, synthetic=synthetic)

        m.RuntimeTestDependencyTracer._observe_audit_event = _patched
        config._ptr148_audit_compat = True
    except Exception:
        config._ptr148_audit_compat = False
'''


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


def _git_init_and_commit(project: Path) -> None:
    subprocess.run(
        ["git", "init"],
        cwd=project,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "-A"],
        cwd=project,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=ptr148@example.test",
            "-c",
            "user.name=ptr148",
            "commit",
            "-m",
            "ptr-148 pure direct node",
        ],
        cwd=project,
        check=True,
        capture_output=True,
    )


@dataclass
class ZeroInjectionProject:
    """Disposable pure direct-node project with loader-only bootstrap."""

    root: Path
    test_path: Path
    cache_dir: Path
    home_dir: Path
    harness_dir: Path
    repository: RepositoryBootstrapSpec

    @property
    def nodeid(self) -> str:
        return "test_direct.py::test_reusable"


def build_zero_injection_project(
    base: Path,
    repository: RepositoryBootstrapSpec,
    *,
    use_loader_conftest: bool = False,
) -> ZeroInjectionProject:
    """Create a committed pure project.  No service injection files."""

    root = base / f"{repository.name}-project"
    root.mkdir(parents=True, exist_ok=True)
    cache_dir = base / f"{repository.name}-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    home_dir = base / f"{repository.name}-home"
    home_dir.mkdir(parents=True, exist_ok=True)
    harness_dir = base / f"{repository.name}-harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    _write(harness_dir / "ptr148_audit_compat.py", _AUDIT_COMPAT_PLUGIN_SOURCE)
    test_path = root / "test_direct.py"
    _write(test_path, _PURE_NODE_SOURCE)
    if use_loader_conftest:
        _write(root / "conftest.py", _LOADER_ONLY_CONFTEST)
    _git_init_and_commit(root)
    return ZeroInjectionProject(
        root=root,
        test_path=test_path,
        cache_dir=cache_dir,
        home_dir=home_dir,
        harness_dir=harness_dir,
        repository=repository,
    )


def build_subprocess_environment(
    project: ZeroInjectionProject,
    *,
    mode: str,
    fixture: RealGroth16TestPassFixture | None = None,
    enable_groth16: bool = True,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ)
    python_paths = [
        str(project.harness_dir),
        str(accelerate_root()),
        str(datasets_root()),
        str(kit_root()),
        str(project.repository.root),
        str(project.root),
        environment.get("PYTHONPATH", ""),
    ]
    # Prefer the local site-packages that hosts pytest.
    try:
        import pytest as _pytest

        python_paths.insert(
            0, str(Path(_pytest.__file__).resolve().parents[1])
        )
    except Exception:
        pass
    environment["PYTHONPATH"] = os.pathsep.join(p for p in python_paths if p)
    environment["IPFS_TEST_PROOF_REUSE_MODE"] = mode
    environment["IPFS_TEST_PROOF_REUSE_CACHE_DIR"] = str(project.cache_dir)
    environment["IPFS_TEST_PROOF_REUSE_AUTO_INSTALL"] = "0"
    environment["HOME"] = str(project.home_dir)
    environment["IPFS_PATH"] = str(project.home_dir / ".ipfs")
    environment["COVERAGE_FILE"] = str(project.home_dir / ".coverage")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["CI"] = "true"
    environment["LANG"] = "C.UTF-8"
    environment["LC_ALL"] = "C.UTF-8"
    environment["PYTHONHASHSEED"] = "0"
    environment.pop("PYTEST_ADDOPTS", None)
    if fixture is not None:
        environment.update(fixture.environment_fragment(enable=enable_groth16))
    elif enable_groth16:
        environment["IPFS_DATASETS_ENABLE_GROTH16"] = "1"
    else:
        environment["IPFS_DATASETS_ENABLE_GROTH16"] = "0"
    if extra:
        environment.update(dict(extra))
    return environment


@dataclass(frozen=True, slots=True)
class SubprocessSample:
    """One measured independent pytest process sample (raw wall time)."""

    label: str
    returncode: int
    wall_time_seconds: float
    stdout: str
    stderr: str
    body_marker_count: int
    proof_cache_skips: int
    metrics: Mapping[str, int]
    passed: bool
    skipped: bool

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "returncode": self.returncode,
            "wall_time_seconds": self.wall_time_seconds,
            "body_marker_count": self.body_marker_count,
            "proof_cache_skips": self.proof_cache_skips,
            "metrics": dict(self.metrics),
            "passed": self.passed,
            "skipped": self.skipped,
        }


def _parse_metrics(output: str) -> dict[str, int]:
    match = PROOF_REUSE_METRICS_RE.search(output)
    if match is None:
        return {
            "predicted": 0,
            "verified": 0,
            "skipped": 0,
            "executed": 0,
            "deferred": 0,
            "degraded": 0,
        }
    return {key: int(value) for key, value in match.groupdict().items()}


def run_direct_node_subprocess(
    project: ZeroInjectionProject,
    environment: Mapping[str, str],
    *,
    label: str,
    audit_compat: bool = True,
    timeout: int = 600,
) -> SubprocessSample:
    """Run one independent pytest process selecting the pure direct node."""

    arguments = [
        sys.executable,
        "-m",
        "pytest",
    ]
    if audit_compat:
        arguments.extend(["-p", "ptr148_audit_compat"])
    arguments.extend(
        [
            "-p",
            PLUGIN_MODULE,
            "test_direct.py::test_reusable",
            "-q",
            "-rs",
            "-s",
        ]
    )
    started = time.perf_counter()
    completed = subprocess.run(
        arguments,
        cwd=project.root,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    metrics = _parse_metrics(output)
    body_count = output.count(BODY_MARKER)
    skip_count = output.count(SKIP_REASON_PREFIX)
    # Pytest may report "1 skipped" for proof-cache hits.
    if "1 skipped" in output and skip_count == 0 and metrics.get("skipped", 0):
        skip_count = int(metrics["skipped"])
    passed = completed.returncode == 0 and (
        "1 passed" in output or "1 skipped" in output
    )
    return SubprocessSample(
        label=label,
        returncode=completed.returncode,
        wall_time_seconds=wall,
        stdout=completed.stdout,
        stderr=completed.stderr,
        body_marker_count=body_count,
        proof_cache_skips=skip_count,
        metrics=metrics,
        passed=passed,
        skipped="1 skipped" in output or skip_count > 0,
    )


@dataclass
class ProductionRuntimeActivationE2E:
    """Two independent direct-node processes sharing one disposable cache.

    Predicted symbols: ProductionRuntimeActivationE2E.
    """

    repository: RepositoryBootstrapSpec
    base_dir: Path
    fixture: RealGroth16TestPassFixture
    project: ZeroInjectionProject | None = None
    cold: SubprocessSample | None = None
    warm: SubprocessSample | None = None
    missing_backend_cold: SubprocessSample | None = None
    missing_backend_warm: SubprocessSample | None = None

    @property
    def interface(self) -> str:
        return "ProductionRuntimeActivationE2E@1"

    def prepare(self) -> ZeroInjectionProject:
        self.project = build_zero_injection_project(self.base_dir, self.repository)
        return self.project

    def run_cold_warm(
        self,
        *,
        mode: str = "readwrite",
        audit_compat: bool = True,
    ) -> dict[str, Any]:
        if self.project is None:
            self.prepare()
        assert self.project is not None
        env = build_subprocess_environment(
            self.project,
            mode=mode,
            fixture=self.fixture,
            enable_groth16=True,
        )
        self.cold = run_direct_node_subprocess(
            self.project,
            env,
            label="cold",
            audit_compat=audit_compat,
        )
        self.warm = run_direct_node_subprocess(
            self.project,
            env,
            label="warm",
            audit_compat=audit_compat,
        )
        return self.summary()

    def run_missing_groth16(self, *, audit_compat: bool = True) -> dict[str, Any]:
        if self.project is None:
            self.prepare()
        assert self.project is not None
        # Separate cache so missing-backend runs do not mix with enabled ones.
        missing_cache = self.base_dir / f"{self.repository.name}-missing-cache"
        missing_cache.mkdir(parents=True, exist_ok=True)
        env = build_subprocess_environment(
            self.project,
            mode="readwrite",
            fixture=self.fixture,
            enable_groth16=False,
            extra={
                "IPFS_TEST_PROOF_REUSE_CACHE_DIR": str(missing_cache),
                "IPFS_DATASETS_ENABLE_GROTH16": "0",
                "IPFS_DATASETS_GROTH16_BINARY": str(
                    self.base_dir / "missing-groth16-binary"
                ),
            },
        )
        self.missing_backend_cold = run_direct_node_subprocess(
            self.project,
            env,
            label="missing-cold",
            audit_compat=audit_compat,
        )
        self.missing_backend_warm = run_direct_node_subprocess(
            self.project,
            env,
            label="missing-warm",
            audit_compat=audit_compat,
        )
        return {
            "cold": self.missing_backend_cold.to_dict(),
            "warm": self.missing_backend_warm.to_dict(),
            "both_passed": (
                self.missing_backend_cold.passed
                and self.missing_backend_warm.passed
            ),
            "false_skips": (
                self.missing_backend_cold.proof_cache_skips
                + self.missing_backend_warm.proof_cache_skips
            ),
        }

    def summary(self) -> dict[str, Any]:
        cold = self.cold
        warm = self.warm
        assert cold is not None and warm is not None
        body_total = cold.body_marker_count + warm.body_marker_count
        # Warm verification savings when the warm process is strictly faster.
        saved = max(0.0, cold.wall_time_seconds - warm.wall_time_seconds)
        return {
            "repository": self.repository.name,
            "interface": self.interface,
            "groth16_available": self.fixture.available,
            "cold": cold.to_dict(),
            "warm": warm.to_dict(),
            "body_marker_total": body_total,
            "cold_body_once": cold.body_marker_count == 1,
            "warm_body_unrun_or_once": warm.body_marker_count in {0, 1},
            "proof_cache_skips_warm": warm.proof_cache_skips,
            "false_skips": 0
            if warm.proof_cache_skips <= 1 and cold.proof_cache_skips == 0
            else warm.proof_cache_skips + cold.proof_cache_skips,
            "raw_cold_wall_seconds": cold.wall_time_seconds,
            "raw_warm_wall_seconds": warm.wall_time_seconds,
            "saved_wall_seconds": saved,
            "positive_saved_wall": saved > 0.0,
            "cache_dir": str(self.project.cache_dir) if self.project else "",
            "certificate_artifacts_present": self._certificate_artifacts_present(),
        }

    def _certificate_artifacts_present(self) -> bool:
        if self.project is None:
            return False
        cache = self.project.cache_dir
        if not cache.exists():
            return False
        # Certificate store layout writes under certificates/ or similar CAS.
        for path in cache.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if "certificate" in name or path.suffix in {".json", ".blob"}:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if (
                    "TestExecutionCertificate" in text
                    or "TestProofCertificate" in text
                    or "proof_digest" in text
                    or "groth16" in text.lower()
                ):
                    return True
        return False


def assert_bootstrap_has_no_injection(spec: RepositoryBootstrapSpec) -> None:
    """Assert production bootstrap files remain loader-only."""

    import tomllib

    project = tomllib.loads(spec.pyproject.read_text(encoding="utf-8"))
    assert (
        project["project"]["entry-points"]["pytest11"][spec.entry_point]
        == spec.entry_point_target
    )
    source = spec.bootstrap.read_text(encoding="utf-8")
    assert PLUGIN_MODULE in source or "proof_reuse" in source
    forbidden = (
        "set_proof_reuse_services",
        "set_proof_reuse_identity_services",
        "_ipfs_proof_reuse_locator",
        "_ipfs_proof_reuse_execution_key",
        "PROOF_REUSE_TEST_LIST",
        "proof_reuse_test_paths",
    )
    for token in forbidden:
        assert token not in source, f"{spec.name} bootstrap hardwires {token}"


# Back-compat alias used by predicted symbols / imports.
RepositorySpec = RepositoryBootstrapSpec


__all__ = [
    "BODY_MARKER",
    "PLUGIN_MODULE",
    "SKIP_REASON_PREFIX",
    "ProductionRuntimeActivationE2E",
    "RealGroth16TestPassFixture",
    "RepositoryBootstrapSpec",
    "RepositorySpec",
    "SubprocessSample",
    "ZeroInjectionProject",
    "accelerate_root",
    "assert_bootstrap_has_no_injection",
    "build_subprocess_environment",
    "build_zero_injection_project",
    "datasets_root",
    "default_artifacts_root",
    "default_binary_candidates",
    "external_root",
    "groth16_backend_root",
    "kit_root",
    "repository_specs",
    "resolve_groth16_binary",
    "run_direct_node_subprocess",
    "v4_artifact_paths",
]
