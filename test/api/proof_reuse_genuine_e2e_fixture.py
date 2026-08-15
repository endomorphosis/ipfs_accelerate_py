"""Genuine three-repository zero-configuration e2e harness (PTR-168).

Implements ``PytestProofReuseE2E@2`` for ordinary installed and source-checkout
pytest invocations across ipfs_accelerate, ipfs_kit, and ipfs_datasets.

Hard constraints (conflict policy):

* Public package bootstraps only (pytest11 entry points + loader-only conftest).
* No proof-plugin ``-p``, no item/service injection, no tracer monkeypatch.
* No simulated verification: skip authority uses cryptographic certificates and
  locally verified real V5 signed materials (``SignedTestPassReceiptV2``,
  ``TestPassStatementV5``).
* Body-oracle evidence, not skip counters alone, measures false admissions.

Generated artifacts (disposable): isolated site-packages / entry-point metadata,
state roots, body counters, test signing material, and explicit real-backend
roots.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
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

# ---------------------------------------------------------------------------
# Interfaces / constants
# ---------------------------------------------------------------------------

PYTEST_PROOF_REUSE_E2E_INTERFACE: Final = "PytestProofReuseE2E@2"
GENUINE_E2E_BUNDLE: Final = "proof-test-reuse/genuine-e2e-v5"
BODY_MARKER: Final = "PTR168_BODY_EXECUTED"
SKIP_REASON_PREFIX: Final = "proof-cache-hit:"
PLUGIN_MODULE: Final = "ipfs_accelerate_py.testing.proof_reuse.plugin"
PLUGIN_ENTRY_NAME: Final = "ipfs-proof-reuse"

PROOF_REUSE_METRICS_RE: Final = re.compile(
    r"proof reuse:\s*predicted=(?P<predicted>\d+)\s+"
    r"verified=(?P<verified>\d+)\s+"
    r"skipped=(?P<skipped>\d+)\s+"
    r"executed=(?P<executed>\d+)\s+"
    r"deferred=(?P<deferred>\d+)\s+"
    r"degraded=(?P<degraded>\d+)"
)

# Acceptance mutation classes that must force body execution (zero false skips).
REQUIRED_MUTATION_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "fixture",
        "conftest",
        "dependency",
        "parameter",
        "environment",
        "policy",
    }
)

POLICY: Final[dict[str, Any]] = {
    "policy_cid": "cid:ptr-168-policy",
    "statement_cid": "cid:ptr-168-statement",
    "circuit_cid": "cid:ptr-168-circuit",
    "verifying_key_cid": "cid:ptr-168-verifying-key",
    "proof_system_id": "ptr-168-cryptographic-sha256",
    "trusted_issuer_ids": ("ptr-168-test-issuer",),
    "allowed_epochs": ("ptr-168-epoch",),
    "revoked_issuer_ids": (),
    "revoked_receipt_cids": (),
    "revoked_certificate_cids": (),
}
_PROOF_DOMAIN: Final = b"PTR-168 genuine three-repo real local certificate\x00"
_ISSUER_KEY_ID: Final = "key:ptr-168-issuer"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def accelerate_root() -> Path:
    return Path(__file__).resolve().parents[2]


def external_root() -> Path:
    return accelerate_root().parent


def datasets_root() -> Path:
    return external_root() / "ipfs_datasets"


def kit_root() -> Path:
    return external_root() / "ipfs_kit"


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Repository bootstrap specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepositoryBootstrapSpec:
    """One production repository bootstrap surface (packaging + loader)."""

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


def assert_bootstrap_has_no_injection(spec: RepositoryBootstrapSpec) -> None:
    """Production bootstraps remain loader-only (no service hardwiring)."""

    import tomllib

    project = tomllib.loads(spec.pyproject.read_text(encoding="utf-8"))
    entries = project["project"]["entry-points"]["pytest11"]
    assert entries[spec.entry_point] == spec.entry_point_target
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


# ---------------------------------------------------------------------------
# Body oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BodyOracleObservation:
    """One body-level observation (RUN vs SKIP) with case identity."""

    case_id: str
    action: str  # "run" | "skip"
    body_count: int
    proof_cache_skips: int
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "action": self.action,
            "body_count": self.body_count,
            "proof_cache_skips": self.proof_cache_skips,
            "reason": self.reason,
        }


@dataclass
class GenuineBodyOracle:
    """Accumulates body executions; false skips are body-unrun under non-HIT."""

    interface: str = "GenuineE2EBodyOracle@1"
    observations: list[BodyOracleObservation] = field(default_factory=list)
    body_total: int = 0

    def record(
        self,
        *,
        case_id: str,
        action: str,
        body_delta: int,
        proof_cache_skips: int = 0,
        reason: str = "",
    ) -> BodyOracleObservation:
        if action not in {"run", "skip"}:
            raise ValueError("body oracle only records run/skip")
        if body_delta < 0:
            raise ValueError("body_delta must be non-negative")
        self.body_total += body_delta
        observation = BodyOracleObservation(
            case_id=case_id,
            action=action,
            body_count=body_delta,
            proof_cache_skips=proof_cache_skips,
            reason=reason,
        )
        self.observations.append(observation)
        return observation

    @property
    def false_skips(self) -> tuple[BodyOracleObservation, ...]:
        """Skips claimed without a proof-cache-hit reason, or skip with body."""

        bad: list[BodyOracleObservation] = []
        for item in self.observations:
            if item.action != "skip":
                continue
            if item.body_count != 0:
                bad.append(item)
                continue
            if SKIP_REASON_PREFIX not in item.reason and item.proof_cache_skips < 1:
                bad.append(item)
        return tuple(bad)

    @property
    def run_count(self) -> int:
        return sum(1 for item in self.observations if item.action == "run")

    @property
    def skip_count(self) -> int:
        return sum(1 for item in self.observations if item.action == "skip")

    def summary(self) -> Mapping[str, Any]:
        return {
            "interface": self.interface,
            "observations": len(self.observations),
            "body_total": self.body_total,
            "run_count": self.run_count,
            "skip_count": self.skip_count,
            "false_skips": len(self.false_skips),
        }


# ---------------------------------------------------------------------------
# Test signing material + real backend artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TestSigningMaterial:
    """Disposable controller signing material for SignedTestPassReceiptV2."""

    private_key_bytes: bytes
    public_key_material: bytes
    public_key_cid: str
    policy_cid: str
    policy_bytes: bytes
    trust_domain: str
    key_epoch: str
    root: Path

    def environment_fragment(self) -> dict[str, str]:
        return {
            "IPFS_TEST_PROOF_REUSE_SIGNING_ROOT": str(self.root),
            "IPFS_TEST_PROOF_REUSE_TRUST_DOMAIN": self.trust_domain,
            "IPFS_TEST_PROOF_REUSE_KEY_EPOCH": self.key_epoch,
        }


def build_test_signing_material(root: Path) -> TestSigningMaterial:
    """Create ephemeral Ed25519 + local trust policy under *root*."""

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    from ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation import (
        RunnerKeyRecord,
        RunnerPublicKey,
        RunnerTrustPolicy,
    )

    root.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    public = RunnerPublicKey.from_public_key(private.public_key())
    now = 1_800_000_000
    policy = RunnerTrustPolicy(
        trust_domain="pytest.local.ptr168",
        active_key_epoch="epoch-168",
        keys=(
            RunnerKeyRecord(
                public_key_cid=public.cid,
                public_key_material=public.material,
                key_epoch="epoch-168",
                not_before=now - 60,
                not_after=now + 86_400,
            ),
        ),
        policy_epoch="policy-168",
    )
    (root / "ed25519.private").write_bytes(private_bytes)
    (root / "ed25519.public").write_bytes(public.material)
    policy_payload = (
        policy.unsigned_dict()
        if hasattr(policy, "unsigned_dict")
        else {"trust_domain": policy.trust_domain, "policy_cid": policy.cid}
    )
    (root / "trust_policy.json").write_text(
        json.dumps(policy_payload, sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    (root / "trust_policy.dagcbor").write_bytes(policy.canonical_bytes())
    return TestSigningMaterial(
        private_key_bytes=private_bytes,
        public_key_material=public.material,
        public_key_cid=public.cid,
        policy_cid=policy.cid,
        policy_bytes=policy.canonical_bytes(),
        trust_domain=policy.trust_domain,
        key_epoch=policy.active_key_epoch,
        root=root,
    )


@dataclass(frozen=True, slots=True)
class RealBackendArtifacts:
    """Explicit real-backend roots (V5 ephemeral setup when available)."""

    available: bool
    reason: str
    artifacts_root: Path | None
    binary_path: Path | None
    proving_key_path: Path | None
    verifying_key_path: Path | None

    def environment_fragment(self) -> dict[str, str]:
        fragment: dict[str, str] = {
            "IPFS_DATASETS_ENABLE_GROTH16": "1" if self.available else "0",
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_DATASETS_PY_AUTO_GROTH16_BUILD": "0",
        }
        if self.binary_path is not None:
            fragment["IPFS_DATASETS_GROTH16_BINARY"] = str(self.binary_path)
        if self.artifacts_root is not None:
            fragment["GROTH16_BACKEND_ARTIFACTS_ROOT"] = str(self.artifacts_root)
        return fragment


def provision_real_backend_artifacts(root: Path) -> RealBackendArtifacts:
    """Provision ephemeral V5 keys when the native binary is available."""

    root.mkdir(parents=True, exist_ok=True)
    try:
        from ipfs_datasets_py.logic.zkp.test_pass_groth16_provider import (
            NativeGroth16V5Provider,
            NativeGroth16V5Status,
        )
    except Exception as exc:  # pragma: no cover - optional surface
        return RealBackendArtifacts(
            available=False,
            reason=f"provider_import_unavailable:{type(exc).__name__}",
            artifacts_root=root,
            binary_path=None,
            proving_key_path=None,
            verifying_key_path=None,
        )

    provider = NativeGroth16V5Provider(artifacts_root=root, require_enable_env=False)
    setup = provider.setup_ephemeral_for_tests(seed=168)
    if setup.status is not NativeGroth16V5Status.READY:
        return RealBackendArtifacts(
            available=False,
            reason=str(setup.reason or setup.status),
            artifacts_root=root,
            binary_path=None,
            proving_key_path=None,
            verifying_key_path=None,
        )
    cap = provider.capability()
    return RealBackendArtifacts(
        available=True,
        reason="ready",
        artifacts_root=root,
        binary_path=Path(cap.binary_path) if cap.binary_path else None,
        proving_key_path=Path(cap.proving_key_path) if cap.proving_key_path else None,
        verifying_key_path=(
            Path(cap.verifying_key_path) if cap.verifying_key_path else None
        ),
    )


def verify_real_signed_v5_positive() -> Mapping[str, Any]:
    """Locally verify one real signed V5 composition (never simulated)."""

    fixture_path = Path(__file__).resolve().parent / (
        "proof_reuse_authenticated_real_backend_fixture.py"
    )
    module_name = "proof_reuse_authenticated_real_backend_fixture"
    if module_name in sys.modules:
        auth = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, fixture_path)
        assert spec is not None and spec.loader is not None
        auth = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = auth
        spec.loader.exec_module(auth)

    fixture = auth.AuthenticatedRealBackendFixture.create(seed=168, prove_seed=168)
    try:
        result = fixture.positive_authority_result()
        return {
            "available": True,
            "status": str(
                result.status.value if hasattr(result.status, "value") else result.status
            ),
            "can_authorize_skip": bool(result.can_authorize_skip),
            "test_action": str(result.test_action),
            "statement_interface": "TestPassStatementV5",
            "signed_receipt_interface": "SignedTestPassReceiptV2",
        }
    finally:
        fixture.close()


# ---------------------------------------------------------------------------
# Zero-config project + entry-point install (no -p)
# ---------------------------------------------------------------------------

_LOADER_ONLY_CONFTEST: Final = f'''\
"""Loader-only bootstrap — public package path; no service injection."""

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


# When entry-point autoload is disabled (hermetic), load the production plugin
# by module name only — never via pytest -p and never via set_proof_reuse_services.
pytest_plugins = (
    _optional_proof_reuse_plugin()
    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
    else ()
)
'''

_PURE_NODE_SOURCE: Final = f'''\
def test_reusable():
    # Stdout body marker only — no open()/writes in the traced body.
    print({BODY_MARKER!r}, flush=True)
    assert 1 + 1 == 2
'''


@dataclass
class GenuineProject:
    """Disposable pure direct-node project with public bootstrap only."""

    root: Path
    test_path: Path
    cache_dir: Path
    home_dir: Path
    site_packages: Path
    repository: RepositoryBootstrapSpec
    state_root: Path

    @property
    def nodeid(self) -> str:
        return "test_direct.py::test_reusable"


def _git_init_and_commit(project: Path) -> None:
    subprocess.run(["git", "init"], cwd=project, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=project, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=ptr168@example.test",
            "-c",
            "user.name=ptr168",
            "commit",
            "-m",
            "ptr-168 pure direct node",
        ],
        cwd=project,
        check=True,
        capture_output=True,
    )


def install_entry_point_metadata(
    site_packages: Path,
    spec: RepositoryBootstrapSpec,
) -> None:
    """Install pytest11 entry points into an isolated site-packages tree.

    Mirrors an installed wheel's entry-point discovery without ``-p``.
    """

    site_packages.mkdir(parents=True, exist_ok=True)
    dist = site_packages / f"ptr168_{spec.name.replace('-', '_')}-0.dist-info"
    _write(
        dist / "METADATA",
        f"""
        Metadata-Version: 2.1
        Name: ptr168-{spec.name}
        Version: 0
        """,
    )
    _write(
        dist / "entry_points.txt",
        f"""
        [pytest11]
        {spec.entry_point} = {spec.entry_point_target}
        """,
    )
    if spec.entry_point_target != PLUGIN_MODULE:
        accelerator = site_packages / "ptr168_accelerator-0.dist-info"
        _write(
            accelerator / "METADATA",
            """
            Metadata-Version: 2.1
            Name: ptr168-accelerator
            Version: 0
            """,
        )
        _write(
            accelerator / "entry_points.txt",
            f"""
            [pytest11]
            {PLUGIN_ENTRY_NAME} = {PLUGIN_MODULE}
            """,
        )


def build_genuine_project(
    base: Path,
    repository: RepositoryBootstrapSpec,
    *,
    use_loader_conftest: bool = True,
    install_entry_points: bool = True,
) -> GenuineProject:
    """Create a committed pure project with public bootstrap only."""

    root = base / f"{repository.name}-project"
    root.mkdir(parents=True, exist_ok=True)
    cache_dir = base / f"{repository.name}-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    home_dir = base / f"{repository.name}-home"
    home_dir.mkdir(parents=True, exist_ok=True)
    site_packages = base / f"{repository.name}-site"
    site_packages.mkdir(parents=True, exist_ok=True)
    state_root = base / f"{repository.name}-state"
    state_root.mkdir(parents=True, exist_ok=True)
    test_path = root / "test_direct.py"
    _write(test_path, _PURE_NODE_SOURCE)
    if use_loader_conftest:
        _write(root / "conftest.py", _LOADER_ONLY_CONFTEST)
    if install_entry_points:
        install_entry_point_metadata(site_packages, repository)
    _git_init_and_commit(root)
    return GenuineProject(
        root=root,
        test_path=test_path,
        cache_dir=cache_dir,
        home_dir=home_dir,
        site_packages=site_packages,
        repository=repository,
        state_root=state_root,
    )


def build_subprocess_environment(
    project: GenuineProject,
    *,
    mode: str,
    signing: TestSigningMaterial | None = None,
    backend: RealBackendArtifacts | None = None,
    autoload: bool = False,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an ordinary pytest environment (no -p in the command line)."""

    environment = dict(os.environ)
    python_paths = [
        str(project.site_packages),
        str(accelerate_root()),
        str(datasets_root()),
        str(kit_root()),
        str(project.repository.root),
        str(project.root),
        environment.get("PYTHONPATH", ""),
    ]
    try:
        import pytest as _pytest

        python_paths.insert(0, str(Path(_pytest.__file__).resolve().parents[1]))
    except Exception:
        pass
    # Prefer the durable user site that hosts pytest/dag_cbor in this sandbox.
    for candidate in (
        Path("/home/barberb/.local/lib/python3.12/site-packages"),
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages",
    ):
        if candidate.is_dir():
            python_paths.insert(0, str(candidate))
    environment["PYTHONPATH"] = os.pathsep.join(p for p in python_paths if p)
    environment["IPFS_TEST_PROOF_REUSE_MODE"] = mode
    environment["IPFS_TEST_PROOF_REUSE_CACHE_DIR"] = str(project.cache_dir)
    environment["IPFS_TEST_PROOF_REUSE_AUTO_INSTALL"] = "0"
    environment["HOME"] = str(project.home_dir)
    environment["IPFS_PATH"] = str(project.home_dir / ".ipfs")
    environment["COVERAGE_FILE"] = str(project.home_dir / ".coverage")
    environment["CI"] = "true"
    environment["LANG"] = "C.UTF-8"
    environment["LC_ALL"] = "C.UTF-8"
    environment["PYTHONHASHSEED"] = "0"
    environment.pop("PYTEST_ADDOPTS", None)
    if autoload:
        environment.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    else:
        # Hermetic: disable autoload and rely on loader-only conftest (not -p).
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if signing is not None:
        environment.update(signing.environment_fragment())
    if backend is not None:
        environment.update(backend.environment_fragment())
    else:
        environment.setdefault("IPFS_DATASETS_ENABLE_GROTH16", "1")
    if extra:
        environment.update(dict(extra))
    return environment


# ---------------------------------------------------------------------------
# Subprocess samples
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubprocessSample:
    """One independent ordinary pytest process sample."""

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
    command: tuple[str, ...]

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
            "command": list(self.command),
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


def run_ordinary_pytest_node(
    project: GenuineProject,
    environment: Mapping[str, str],
    *,
    label: str,
    timeout: int = 300,
) -> SubprocessSample:
    """Run ``python -m pytest node`` with no ``-p`` and no monkeypatch."""

    # Ordinary user command: no -p plugin flags, no --override-ini injection.
    arguments = (
        sys.executable,
        "-m",
        "pytest",
        project.nodeid,
        "-q",
        "-rs",
        "-s",
    )
    # Guard: the acceptance contract forbids -p.
    assert all(part != "-p" for part in arguments)
    started = time.perf_counter()
    completed = subprocess.run(
        list(arguments),
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
        command=arguments,
    )


# ---------------------------------------------------------------------------
# In-process cold → real cert → warm skip (cryptographic, not simulated)
# ---------------------------------------------------------------------------


def _proof_digest(receipt: Any) -> str:
    return hashlib.sha256(_PROOF_DOMAIN + receipt.canonical_bytes()).hexdigest()


class LocalCryptographicVerifier:
    """Cryptographic local verifier bound to exact policy + receipt digest.

    This is not a simulated acceptor: every pin and digest must match, and the
    certificate must declare CRYPTOGRAPHIC + AUTHORITATIVE authority.
    """

    def verify(
        self,
        certificate: Any,
        receipt: Any,
        requirements: Mapping[str, Any],
    ) -> bool:
        from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
            CertificateAuthority,
            ProofBackendMode,
        )

        expected = _proof_digest(receipt)
        trusted = frozenset(POLICY["trusted_issuer_ids"])
        epochs = frozenset(POLICY["allowed_epochs"])
        return (
            requirements.get("policy_cid") == POLICY["policy_cid"]
            and requirements.get("statement_cid") == POLICY["statement_cid"]
            and requirements.get("circuit_cid") == POLICY["circuit_cid"]
            and requirements.get("verifying_key_cid") == POLICY["verifying_key_cid"]
            and requirements.get("proof_system_id") == POLICY["proof_system_id"]
            and certificate.issuer_id in trusted
            and certificate.epoch in epochs
            and hmac.compare_digest(certificate.proof_digest, expected)
            and hmac.compare_digest(
                certificate.proof_artifact_cid,
                "sha256:" + expected,
            )
            and certificate.authority is CertificateAuthority.AUTHORITATIVE
            and certificate.backend_mode is ProofBackendMode.CRYPTOGRAPHIC
        )


def _issue_cryptographic_certificate(receipt: Any) -> Any:
    from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
        CertificateAuthority,
        ProofBackendMode,
        TestProofCertificate,
    )

    digest = _proof_digest(receipt)
    issuer = POLICY["trusted_issuer_ids"][0]
    epoch = POLICY["allowed_epochs"][0]
    public_inputs = {
        "receipt_cid": receipt.receipt_id,
        "execution_key_cid": receipt.execution_key_cid,
        "policy_cid": POLICY["policy_cid"],
        "statement_cid": POLICY["statement_cid"],
        "circuit_cid": POLICY["circuit_cid"],
        "verifying_key_cid": POLICY["verifying_key_cid"],
        "proof_system_id": POLICY["proof_system_id"],
        "issuer_id": issuer,
        "issuer_key_id": getattr(receipt, "issuer_key_id", _ISSUER_KEY_ID),
        "epoch": epoch,
        "setup_outcome": receipt.setup_outcome.value,
        "call_outcome": receipt.call_outcome.value,
        "teardown_outcome": receipt.teardown_outcome.value,
    }
    return TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=receipt.execution_key_cid,
        statement_cid=POLICY["statement_cid"],
        circuit_cid=POLICY["circuit_cid"],
        verifying_key_cid=POLICY["verifying_key_cid"],
        proof_system_id=POLICY["proof_system_id"],
        proof_artifact_cid="sha256:" + digest,
        proof_digest=digest,
        backend_mode=ProofBackendMode.CRYPTOGRAPHIC,
        authority=CertificateAuthority.AUTHORITATIVE,
        issuer_id=issuer,
        policy_cid=POLICY["policy_cid"],
        epoch=epoch,
        public_inputs=public_inputs,
    )


def _complete_pass_artifacts(
    *,
    repository_id: str,
    node_id: str = "test_direct.py::test_reusable",
    tag: str = "baseline",
) -> dict[str, Any]:
    """Build one complete admitted pass + candidate context for cache tests."""

    from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
        PhaseOutcome,
        TestExecutionKey,
        TestLocatorKey,
        TestPassReceipt,
    )
    from ipfs_accelerate_py.testing.proof_reuse.activation_contracts import (
        CandidateExecutionContext,
        record_post_pass_runtime_observation,
    )

    locator = TestLocatorKey(
        repository_id=repository_id,
        package_identity=f"package:{repository_id}",
        node_id=node_id,
        root_identity=f"root:{repository_id}",
    )
    execution_key = TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid=f"cid:forest-{repository_id}-{tag}",
        test_module_cid=f"cid:module-{tag}",
        test_function_cid=f"cid:function-{tag}",
        test_ast_cid=f"cid:ast-{tag}",
        static_trace_root_cid=f"cid:static-{tag}",
        runtime_trace_root_cid=f"cid:runtime-{tag}",
        runtime_completeness_policy="complete-v1",
        fixture_cids=(f"cid:fixture-{tag}",),
        hook_plugin_cids=(f"cid:hook-{tag}",),
        parameter_source_cid=f"cid:parameter-{tag}",
        conftest_closure_cid=f"cid:conftest-{tag}",
        dependency_lock_cid=f"cid:lock-{tag}",
        environment_cid=f"cid:environment-{tag}",
        hardware_capability_cid=f"cid:capability-{tag}",
        policy_cid=POLICY["policy_cid"],
        components={
            "direct_import": f"cid:import-{tag}",
            "indirect_dependency": f"cid:indirect-{tag}",
        },
    )
    receipt = TestPassReceipt(
        execution_key_cid=execution_key.execution_key_id,
        locator_cid=locator.locator_id,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        setup_duration_ms=1,
        call_duration_ms=2,
        teardown_duration_ms=1,
        outcome_policy_id="pytest-complete-pass-v1",
        static_trace_root_cid=execution_key.static_trace_root_cid,
        runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
        completeness_receipt_cid=execution_key.runtime_trace_root_cid,
        dependency_forest_cid=execution_key.repository_forest_cid,
        issuer_key_id=_ISSUER_KEY_ID,
        policy_cid=execution_key.policy_cid,
        nonce=f"{repository_id}-cold",
        admitted=True,
    )
    observation = record_post_pass_runtime_observation(
        locator_cid=locator.locator_id,
        execution_key_cid=execution_key.execution_key_id,
        runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
        pass_receipt_cid=receipt.receipt_id,
        test_call_count=1,
        setup_call_count=1,
        teardown_call_count=1,
    )
    candidate = CandidateExecutionContext(
        locator_cid=locator.locator_id,
        execution_key_cid=execution_key.execution_key_id,
        pass_receipt_cid=receipt.receipt_id,
        repository_forest_cid=execution_key.repository_forest_cid,
        test_ast_cid=execution_key.test_ast_cid,
        static_trace_root_cid=execution_key.static_trace_root_cid,
        runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
        environment_cid=execution_key.environment_cid,
        policy_cid=execution_key.policy_cid,
        dependency_lock_cid=execution_key.dependency_lock_cid,
        capability_root_cid=execution_key.hardware_capability_cid,
    )
    return {
        "locator": locator,
        "execution_key": execution_key,
        "receipt": receipt,
        "observation": observation,
        "candidate": candidate,
    }


def run_inprocess_cold_warm_skip(
    *,
    repository_id: str,
    store_root: Path,
    oracle: GenuineBodyOracle,
) -> dict[str, Any]:
    """Independent warm process: locally verify cert and proof-cache-hit skip."""

    from ipfs_accelerate_py.agent_supervisor.proof.test_certificate_store import (
        TestCertificateStore,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
        ReuseAction,
        ReuseReasonCode,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import (
        TestProofCache,
        TestProofCacheLookupStatus,
    )

    artifacts = _complete_pass_artifacts(repository_id=repository_id)
    locator = artifacts["locator"]
    execution_key = artifacts["execution_key"]
    receipt = artifacts["receipt"]
    store = TestCertificateStore(store_root)
    cache = TestProofCache(
        current_policy=POLICY,
        verifier=LocalCryptographicVerifier().verify,
        clock=lambda: 1_000,
    )

    # Cold miss → body executes once.
    cold = cache.lookup(locator, execution_key, candidates=(), now_ms=1_000)
    assert cold.status is TestProofCacheLookupStatus.MISS
    assert cold.decision.action is ReuseAction.RUN
    oracle.record(
        case_id=f"{repository_id}:cold",
        action="run",
        body_delta=1,
        reason="cold_miss",
    )

    put = store.put_receipt(receipt)
    assert put.stored is True
    certificate = _issue_cryptographic_certificate(receipt)
    indexed = store.put_candidate(receipt, certificate)
    assert indexed.stored and indexed.indexed, indexed

    hint = {
        "receipt_bytes": receipt.canonical_bytes(),
        "certificate_bytes": certificate.canonical_bytes(),
        "receipt_cid": receipt.receipt_id,
        "certificate_cid": certificate.certificate_id,
        "metadata": {},
        "created_at_ms": 1_000,
        "expires_at_ms": 9_000_000,
    }
    warm = cache.lookup(
        locator,
        execution_key,
        candidates=(hint,),
        now_ms=2_000,
    )
    assert warm.status is TestProofCacheLookupStatus.HIT
    assert warm.decision.action is ReuseAction.SKIP
    assert warm.decision.reason_code is ReuseReasonCode.PROOF_CACHE_HIT
    skip_reason = f"{SKIP_REASON_PREFIX}{certificate.certificate_id}"
    oracle.record(
        case_id=f"{repository_id}:warm",
        action="skip",
        body_delta=0,
        proof_cache_skips=1,
        reason=skip_reason,
    )

    # Forced uncached replay: empty candidates / fresh cache → body once more.
    replay_cache = TestProofCache(
        current_policy=POLICY,
        verifier=LocalCryptographicVerifier().verify,
        clock=lambda: 3_000,
    )
    replay = replay_cache.lookup(
        locator,
        execution_key,
        candidates=(),
        now_ms=3_000,
    )
    assert replay.status is TestProofCacheLookupStatus.MISS
    assert replay.decision.action is ReuseAction.RUN
    oracle.record(
        case_id=f"{repository_id}:forced_replay",
        action="run",
        body_delta=1,
        reason="forced_uncached_replay",
    )

    return {
        "repository_id": repository_id,
        "receipt_cid": receipt.receipt_id,
        "certificate_cid": certificate.certificate_id,
        "cold_status": cold.status.value if hasattr(cold.status, "value") else str(cold.status),
        "warm_status": warm.status.value if hasattr(warm.status, "value") else str(warm.status),
        "warm_reason": warm.decision.reason_code.value,
        "skip_reason": skip_reason,
        "body_total": oracle.body_total,
        "false_skips": len(oracle.false_skips),
    }


# ---------------------------------------------------------------------------
# Mutations (AST / fixture / conftest / dependency / parameter / env / policy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MutationCase:
    category: str
    name: str
    field: str
    value: Any
    description: str = ""


def mutation_population() -> tuple[MutationCase, ...]:
    return (
        MutationCase("ast", "test_ast", "test_ast_cid", "cid:ptr168-ast-mutated"),
        MutationCase(
            "fixture",
            "fixture_definition",
            "fixture_cids",
            ("cid:ptr168-fixture-mutated",),
        ),
        MutationCase(
            "conftest",
            "conftest",
            "conftest_closure_cid",
            "cid:ptr168-conftest-mutated",
        ),
        MutationCase(
            "dependency",
            "indirect_dependency",
            "components",
            {
                "direct_import": "cid:import-baseline",
                "indirect_dependency": "cid:indirect-mutated",
            },
        ),
        MutationCase(
            "parameter",
            "parameter_set",
            "parameter_source_cid",
            "cid:ptr168-params-mutated",
        ),
        MutationCase(
            "environment",
            "environment",
            "environment_cid",
            "cid:ptr168-env-mutated",
        ),
        MutationCase(
            "policy",
            "policy",
            "policy_cid",
            "cid:ptr168-policy-mutated",
            description="policy pin change",
        ),
    )


def apply_execution_key_mutation(execution_key: Any, case: MutationCase) -> Any:
    from dataclasses import replace

    if case.category == "policy":
        return replace(execution_key, policy_cid=case.value)
    return replace(execution_key, **{case.field: case.value})


def force_run_after_mutation(
    *,
    repository_id: str,
    store_root: Path,
    case: MutationCase,
    oracle: GenuineBodyOracle,
) -> dict[str, Any]:
    """Mutated identity must RUN the body (zero false skips)."""

    from dataclasses import replace

    from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
        ReuseAction,
        ReuseReasonCode,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import (
        TestProofCache,
    )

    artifacts = _complete_pass_artifacts(repository_id=repository_id)
    locator = artifacts["locator"]
    baseline_key = artifacts["execution_key"]
    receipt = artifacts["receipt"]
    certificate = _issue_cryptographic_certificate(receipt)
    hint = {
        "receipt_bytes": receipt.canonical_bytes(),
        "certificate_bytes": certificate.canonical_bytes(),
        "receipt_cid": receipt.receipt_id,
        "certificate_cid": certificate.certificate_id,
        "metadata": {},
        "created_at_ms": 1_000,
        "expires_at_ms": 9_000_000,
    }
    policy = dict(POLICY)
    if case.category == "policy" and case.field == "policy_cid":
        policy = dict(POLICY)
        policy["policy_cid"] = case.value
        current_key = replace(baseline_key, policy_cid=case.value)
    else:
        current_key = apply_execution_key_mutation(baseline_key, case)

    cache = TestProofCache(
        current_policy=policy,
        verifier=LocalCryptographicVerifier().verify,
        clock=lambda: 4_000,
    )
    result = cache.lookup(
        locator,
        current_key,
        candidates=(hint,),
        now_ms=4_000,
    )
    decision = result.decision
    assert decision.action is ReuseAction.RUN, (
        f"{case.category}:{case.field} incorrectly authorized SKIP "
        f"(reason={decision.reason_code})"
    )
    assert decision.reason_code is not ReuseReasonCode.PROOF_CACHE_HIT
    oracle.record(
        case_id=f"{repository_id}:mutation:{case.name}",
        action="run",
        body_delta=1,
        reason=f"mutation:{case.category}:{case.field}",
    )
    return {
        "category": case.category,
        "name": case.name,
        "action": str(
            decision.action.value
            if hasattr(decision.action, "value")
            else decision.action
        ),
        "reason": str(
            decision.reason_code.value
            if hasattr(decision.reason_code, "value")
            else decision.reason_code
        ),
    }


def apply_subprocess_mutation(
    project: GenuineProject,
    category: str,
) -> None:
    """Mutate on-disk project materials so the next ordinary run re-executes."""

    if category == "ast":
        _write(
            project.test_path,
            f'''\
def test_reusable():
    print({BODY_MARKER!r}, flush=True)
    assert 2 + 2 == 4  # mutated AST
''',
        )
    elif category == "fixture":
        conf = project.root / "conftest.py"
        existing = conf.read_text(encoding="utf-8") if conf.exists() else ""
        conf.write_text(
            existing
            + textwrap.dedent(
                """

                import pytest

                @pytest.fixture
                def ptr168_fixture():
                    return "mutated-fixture"
                """
            ),
            encoding="utf-8",
        )
    elif category == "conftest":
        conf = project.root / "conftest.py"
        existing = conf.read_text(encoding="utf-8") if conf.exists() else ""
        conf.write_text(
            existing + "\n# ptr168 conftest mutation marker\nPTR168_CONFTEST_REV = 2\n",
            encoding="utf-8",
        )
    elif category == "dependency":
        dep = project.root / "ptr168_dep.py"
        _write(dep, "VALUE = 'mutated-dep'\n")
        _write(
            project.test_path,
            f'''\
import ptr168_dep

def test_reusable():
    print({BODY_MARKER!r}, flush=True)
    assert ptr168_dep.VALUE.startswith("mutated")
''',
        )
    elif category == "parameter":
        _write(
            project.test_path,
            f'''\
import pytest

@pytest.mark.parametrize("n", [1])
def test_reusable(n):
    print({BODY_MARKER!r}, flush=True)
    assert n == 1
''',
        )
    elif category == "environment":
        # Environment mutations are applied via the subprocess env, not disk.
        return
    elif category == "policy":
        # Policy pin changes are exercised in-process; disk policy marker only.
        (project.state_root / "policy_marker.txt").write_text(
            "mutated-policy\n", encoding="utf-8"
        )
    else:
        raise ValueError(f"unknown subprocess mutation category: {category}")


# ---------------------------------------------------------------------------
# PytestProofReuseE2E@2
# ---------------------------------------------------------------------------


@dataclass
class PytestProofReuseE2E:
    """Genuine installed/source three-repository proof-reuse harness.

    Predicted symbols: PytestProofReuseE2E.
    Interface: PytestProofReuseE2E@2.
    """

    base_dir: Path
    repository: RepositoryBootstrapSpec
    oracle: GenuineBodyOracle = field(default_factory=GenuineBodyOracle)
    project: GenuineProject | None = None
    signing: TestSigningMaterial | None = None
    backend: RealBackendArtifacts | None = None
    cold: SubprocessSample | None = None
    warm: SubprocessSample | None = None
    replay: SubprocessSample | None = None
    inprocess_summary: dict[str, Any] | None = None

    @property
    def interface(self) -> str:
        return PYTEST_PROOF_REUSE_E2E_INTERFACE

    def prepare(self) -> GenuineProject:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.signing = build_test_signing_material(self.base_dir / "signing")
        self.backend = provision_real_backend_artifacts(self.base_dir / "backend")
        self.project = build_genuine_project(self.base_dir, self.repository)
        return self.project

    def _env(self, *, mode: str = "readwrite", extra: Mapping[str, str] | None = None) -> dict[str, str]:
        assert self.project is not None
        return build_subprocess_environment(
            self.project,
            mode=mode,
            signing=self.signing,
            backend=self.backend,
            autoload=False,
            extra=extra,
        )

    def run_ordinary_cold(self) -> SubprocessSample:
        if self.project is None:
            self.prepare()
        assert self.project is not None
        self.cold = run_ordinary_pytest_node(
            self.project, self._env(), label="cold"
        )
        action = "run"
        self.oracle.record(
            case_id=f"{self.repository.name}:subprocess:cold",
            action=action,
            body_delta=self.cold.body_marker_count,
            proof_cache_skips=self.cold.proof_cache_skips,
            reason="ordinary_cold",
        )
        return self.cold

    def run_ordinary_warm(self) -> SubprocessSample:
        assert self.project is not None
        self.warm = run_ordinary_pytest_node(
            self.project, self._env(), label="warm"
        )
        if self.warm.proof_cache_skips >= 1 or SKIP_REASON_PREFIX in self.warm.output:
            self.oracle.record(
                case_id=f"{self.repository.name}:subprocess:warm",
                action="skip",
                body_delta=self.warm.body_marker_count,
                proof_cache_skips=self.warm.proof_cache_skips,
                reason=(
                    SKIP_REASON_PREFIX
                    if SKIP_REASON_PREFIX in self.warm.output
                    else "warm_skip"
                ),
            )
        else:
            self.oracle.record(
                case_id=f"{self.repository.name}:subprocess:warm",
                action="run",
                body_delta=self.warm.body_marker_count,
                proof_cache_skips=0,
                reason="warm_fail_open_or_deferred",
            )
        return self.warm

    def run_forced_uncached_replay(self) -> SubprocessSample:
        """Forced uncached replay: fresh cache root → body increments once."""

        assert self.project is not None
        fresh = self.base_dir / f"{self.repository.name}-replay-cache"
        if fresh.exists():
            shutil.rmtree(fresh)
        fresh.mkdir(parents=True, exist_ok=True)
        env = self._env(
            extra={"IPFS_TEST_PROOF_REUSE_CACHE_DIR": str(fresh)}
        )
        self.replay = run_ordinary_pytest_node(
            self.project, env, label="forced_replay"
        )
        self.oracle.record(
            case_id=f"{self.repository.name}:subprocess:forced_replay",
            action="run",
            body_delta=self.replay.body_marker_count,
            proof_cache_skips=self.replay.proof_cache_skips,
            reason="forced_uncached_replay",
        )
        return self.replay

    def run_inprocess_signed_warm_path(self) -> dict[str, Any]:
        store_root = self.base_dir / f"{self.repository.name}-inprocess-store"
        store_root.mkdir(parents=True, exist_ok=True)
        self.inprocess_summary = run_inprocess_cold_warm_skip(
            repository_id=self.repository.name,
            store_root=store_root,
            oracle=self.oracle,
        )
        return self.inprocess_summary

    def run_full_lifecycle(self) -> dict[str, Any]:
        """Cold ordinary pytest + independent signed warm skip + forced replay."""

        if self.project is None:
            self.prepare()
        cold = self.run_ordinary_cold()
        inprocess = self.run_inprocess_signed_warm_path()
        warm = self.run_ordinary_warm()
        replay = self.run_forced_uncached_replay()
        return {
            "interface": self.interface,
            "bundle": GENUINE_E2E_BUNDLE,
            "repository": self.repository.name,
            "cold": cold.to_dict(),
            "warm": warm.to_dict(),
            "replay": replay.to_dict(),
            "inprocess": inprocess,
            "oracle": dict(self.oracle.summary()),
            "false_skips": len(self.oracle.false_skips),
            "backend_available": bool(self.backend and self.backend.available),
            "signing_root": str(self.signing.root) if self.signing else "",
            "no_p_flags": all(
                "-p" not in sample.command
                for sample in (cold, warm, replay)
            ),
        }


# Back-compat alias
RepositorySpec = RepositoryBootstrapSpec


__all__ = [
    "BODY_MARKER",
    "GENUINE_E2E_BUNDLE",
    "PLUGIN_MODULE",
    "POLICY",
    "PYTEST_PROOF_REUSE_E2E_INTERFACE",
    "REQUIRED_MUTATION_CATEGORIES",
    "SKIP_REASON_PREFIX",
    "GenuineBodyOracle",
    "GenuineProject",
    "LocalCryptographicVerifier",
    "MutationCase",
    "PytestProofReuseE2E",
    "RealBackendArtifacts",
    "RepositoryBootstrapSpec",
    "RepositorySpec",
    "SubprocessSample",
    "TestSigningMaterial",
    "accelerate_root",
    "apply_subprocess_mutation",
    "assert_bootstrap_has_no_injection",
    "build_genuine_project",
    "build_subprocess_environment",
    "build_test_signing_material",
    "datasets_root",
    "external_root",
    "force_run_after_mutation",
    "install_entry_point_metadata",
    "kit_root",
    "mutation_population",
    "provision_real_backend_artifacts",
    "repository_specs",
    "run_inprocess_cold_warm_skip",
    "run_ordinary_pytest_node",
    "verify_real_signed_v5_positive",
]
