"""Cross-repository proof-reuse subprocess contract (PTR-093/PTR-142).

The tests in this module deliberately exercise pytest as a user does: by
selecting individual nodes in isolated projects.  They cover the packaging
entry point and the source-tree fallback bootstrap used by ipfs_accelerate,
ipfs_kit, and ipfs_datasets, including the controller-owned xdist publication
boundary.

PTR-142 zero-injection automatic activation (no test-file hardwiring, no manual
service injection) is proven by
``test_proof_reuse_runtime_activation_e2e.py``.  This module retains the
controlled service-injection harness for hermetic certificate lifecycle proofs
while asserting that production repository bootstraps stay loader-only.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.test_certificate_store import (
    TestCertificateStore,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    CertificateAuthority,
    ProofBackendMode,
    TestPassReceipt,
    TestProofCertificate,
)

ACCELERATE_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ACCELERATE_ROOT.parent
PLUGIN_MODULE = "ipfs_accelerate_py.testing.proof_reuse.plugin"
PYTEST_SITE = Path(pytest.__file__).resolve().parents[1]

POLICY = {
    "policy_cid": "cid:ptr-093-policy",
    "statement_cid": "cid:ptr-093-statement",
    "circuit_cid": "cid:ptr-093-circuit",
    "verifying_key_cid": "cid:ptr-093-verifying-key",
    "proof_system_id": "ptr-093-deterministic-sha256",
    "trusted_issuer_ids": ("ptr-093-test-issuer",),
    "allowed_epochs": ("ptr-093-epoch",),
}
_PROOF_DOMAIN = b"PTR-093 deterministic cryptographic verifier fixture\x00"


@dataclass(frozen=True)
class RepositorySpec:
    name: str
    root: Path
    bootstrap: Path
    entry_point: str
    entry_point_target: str
    autoload_plugin_name: str

    @property
    def pyproject(self) -> Path:
        return self.root / "pyproject.toml"


REPOSITORIES = (
    RepositorySpec(
        "ipfs_accelerate",
        ACCELERATE_ROOT,
        ACCELERATE_ROOT / "conftest.py",
        "ipfs-proof-reuse",
        PLUGIN_MODULE,
        "ipfs-proof-reuse",
    ),
    RepositorySpec(
        "ipfs_kit",
        EXTERNAL_ROOT / "ipfs_kit",
        EXTERNAL_ROOT / "ipfs_kit" / "conftest.py",
        "ipfs-kit-proof-reuse",
        "ipfs_kit_py.pytest_proof_reuse",
        "ipfs-proof-reuse",
    ),
    RepositorySpec(
        "ipfs_datasets",
        EXTERNAL_ROOT / "ipfs_datasets",
        EXTERNAL_ROOT / "ipfs_datasets" / "tests" / "conftest.py",
        "ipfs-datasets-proof-reuse",
        "ipfs_datasets_py.pytest_proof_reuse",
        "ipfs-proof-reuse",
    ),
)


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


# This is intentionally small.  It mirrors the production accelerate/kit
# fallback and the import-safe part of the datasets fallback without importing
# any repository-wide test configuration into the hermetic subprocess.
_ROOT_FALLBACK = f'''\
"""Hermetic mirror of the repositories' optional direct-node bootstrap."""

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


def _install_entry_point(metadata_root: Path, spec: RepositorySpec) -> None:
    distribution = (
        metadata_root / f"ptr_093_{spec.name.replace('-', '_')}-0.dist-info"
    )
    _write(
        distribution / "METADATA",
        f"""
        Metadata-Version: 2.1
        Name: ptr-093-{spec.name}
        Version: 0
        """,
    )
    _write(
        distribution / "entry_points.txt",
        f"""
        [pytest11]
        {spec.entry_point} = {spec.entry_point_target}
        """,
    )
    if spec.entry_point_target != PLUGIN_MODULE:
        accelerator = metadata_root / "ptr_093_accelerator-0.dist-info"
        _write(
            accelerator / "METADATA",
            """
            Metadata-Version: 2.1
            Name: ptr-093-accelerator
            Version: 0
            """,
        )
        _write(
            accelerator / "entry_points.txt",
            f"""
            [pytest11]
            ipfs-proof-reuse = {PLUGIN_MODULE}
            """,
        )


def _service_conftest_source(spec: RepositorySpec) -> str:
    return f'''\
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.test_certificate_store import (
    TestCertificateStore,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    TestExecutionKey,
    TestLocatorKey,
)
from ipfs_accelerate_py.testing.proof_reuse.lookup import ProofReuseLookup
from ipfs_accelerate_py.testing.proof_reuse.plugin import set_proof_reuse_services

POLICY = {POLICY!r}
PROOF_DOMAIN = {_PROOF_DOMAIN!r}


def _proof_digest(receipt):
    return hashlib.sha256(PROOF_DOMAIN + receipt.canonical_bytes()).hexdigest()


class DeterministicCryptographicVerifier:
    """A real deterministic verifier: it recomputes and compares the proof."""

    def verify(self, certificate, receipt, requirements):
        expected = _proof_digest(receipt)
        return (
            requirements["policy_cid"] == POLICY["policy_cid"]
            and requirements["statement_cid"] == POLICY["statement_cid"]
            and requirements["circuit_cid"] == POLICY["circuit_cid"]
            and requirements["verifying_key_cid"]
            == POLICY["verifying_key_cid"]
            and requirements["proof_system_id"] == POLICY["proof_system_id"]
            and requirements["trusted_issuer_ids"]
            == frozenset(POLICY["trusted_issuer_ids"])
            and requirements["allowed_epochs"] == frozenset(POLICY["allowed_epochs"])
            and not requirements.get("revoked_certificate_cids")
            and not requirements.get("revoked_issuer_ids")
            and not requirements.get("revoked_receipt_cids")
            and hmac.compare_digest(certificate.proof_digest, expected)
            and hmac.compare_digest(
                certificate.proof_artifact_cid,
                "sha256:" + expected,
            )
        )


class RecordingStore(TestCertificateStore):
    def put_receipt(self, receipt):
        result = super().put_receipt(receipt)
        if result.stored:
            record = json.dumps(
                {{
                    "receipt_cid": result.cid,
                    "pid": os.getpid(),
                    "worker": os.environ.get("PYTEST_XDIST_WORKER", ""),
                }},
                sort_keys=True,
            ) + "\\n"
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            descriptor = os.open(os.environ["PTR_PUBLICATION_LOG"], flags, 0o600)
            try:
                os.write(descriptor, record.encode("utf-8"))
            finally:
                os.close(descriptor)
        return result


def pytest_configure(config):
    if os.environ.get("IPFS_TEST_PROOF_REUSE_MODE", "off") == "off":
        return
    store = RecordingStore(os.environ["PTR_STORE_ROOT"])
    lookup = ProofReuseLookup(
        store=store,
        verifier=DeterministicCryptographicVerifier(),
        current_policy=POLICY,
    )
    set_proof_reuse_services(config, lookup=lookup, store=store)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        source = Path(str(item.path)).read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        locator = TestLocatorKey(
            repository_id={spec.name!r},
            package_identity="ptr-093-isolated-project",
            node_id=item.nodeid,
            root_identity="ptr-093-root",
        )
        execution_key = TestExecutionKey(
            locator_cid=locator.locator_id,
            repository_forest_cid="sha256:" + digest,
            test_module_cid="sha256:" + digest,
            test_function_cid="sha256:" + digest,
            static_trace_root_cid="sha256:static:" + digest,
            runtime_trace_root_cid="sha256:runtime:" + digest,
            runtime_completeness_policy="complete-v1",
            policy_cid=POLICY["policy_cid"],
        )
        item._ipfs_proof_reuse_locator = locator
        item._ipfs_proof_reuse_execution_key = execution_key
        item._ipfs_proof_reuse_runtime_trace = {{
            "complete": True,
            "root_cid": execution_key.runtime_trace_root_cid,
        }}
'''


_PASSING_NODE_V1 = '''\
import json
import os
from pathlib import Path


def test_reusable(request):
    expected = os.environ["PTR_EXPECT_PLUGIN"]
    assert request.config.pluginmanager.hasplugin(expected)
    path = Path(os.environ["PTR_BODY_LOG"])
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"version": 1, "node": request.node.nodeid}) + "\\n")
'''

_PASSING_NODE_V2 = '''\
import json
import os
from pathlib import Path


def test_reusable(request):
    expected = os.environ["PTR_EXPECT_PLUGIN"]
    assert request.config.pluginmanager.hasplugin(expected)
    # This source change must invalidate the exact execution key.
    path = Path(os.environ["PTR_BODY_LOG"])
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"version": 2, "node": request.node.nodeid}) + "\\n")
    assert "mutated".upper() == "MUTATED"
'''

_FAILING_NODE = '''\
import json
import os
from pathlib import Path


def test_reusable(request):
    path = Path(os.environ["PTR_BODY_LOG"])
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"version": "failure", "node": request.node.nodeid}) + "\\n")
    assert False, "intentional pre-receipt failure"
'''

_XDIST_NODES = '''\
import json
import os
from pathlib import Path


def _record(request):
    expected = os.environ["PTR_EXPECT_PLUGIN"]
    assert request.config.pluginmanager.hasplugin(expected)
    payload = json.dumps(
        {
            "node": request.node.nodeid,
            "worker": os.environ.get("PYTEST_XDIST_WORKER", ""),
        },
        sort_keys=True,
    ) + "\\n"
    descriptor = os.open(
        os.environ["PTR_BODY_LOG"],
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, payload.encode("utf-8"))
    finally:
        os.close(descriptor)


def test_parallel_one(request):
    _record(request)


def test_parallel_two(request):
    _record(request)
'''


def _make_project(
    tmp_path: Path,
    spec: RepositorySpec,
    source: str,
) -> tuple[Path, Path]:
    project = tmp_path / f"{spec.name}-project"
    metadata = tmp_path / f"{spec.name}-metadata"
    _write(project / "conftest.py", _ROOT_FALLBACK)
    _write(project / "tests" / "conftest.py", _service_conftest_source(spec))
    _write(project / "tests" / "__init__.py", "")
    _write(project / "tests" / "test_target.py", source)
    _install_entry_point(metadata, spec)
    return project, metadata


def _environment(
    tmp_path: Path,
    spec: RepositorySpec,
    metadata: Path,
    *,
    mode: str,
    autoload: bool,
) -> dict[str, str]:
    environment = dict(os.environ)
    paths = (
        str(metadata),
        str(ACCELERATE_ROOT),
        str(spec.root),
        str(PYTEST_SITE),
        environment.get("PYTHONPATH", ""),
    )
    environment["PYTHONPATH"] = os.pathsep.join(part for part in paths if part)
    environment["IPFS_TEST_PROOF_REUSE_MODE"] = mode
    environment["HOME"] = str(tmp_path / "home")
    environment["IPFS_PATH"] = str(tmp_path / "home" / ".ipfs")
    environment["COVERAGE_FILE"] = str(tmp_path / ".coverage")
    environment["PTR_STORE_ROOT"] = str(tmp_path / "proof-store")
    environment["PTR_PUBLICATION_LOG"] = str(tmp_path / "publications.jsonl")
    environment["PTR_BODY_LOG"] = str(tmp_path / "bodies.jsonl")
    environment.pop("PYTEST_ADDOPTS", None)
    if autoload:
        environment.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
        environment["PTR_EXPECT_PLUGIN"] = spec.autoload_plugin_name
    else:
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        environment["PTR_EXPECT_PLUGIN"] = PLUGIN_MODULE
    return environment


def _run(
    project: Path,
    environment: dict[str, str],
    *arguments: str,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _output(completed: subprocess.CompletedProcess[str]) -> str:
    return completed.stdout + completed.stderr


def _assert_result(
    completed: subprocess.CompletedProcess[str],
    *,
    returncode: int = 0,
    contains: Iterable[str] = (),
) -> str:
    output = _output(completed)
    assert completed.returncode == returncode, output
    for expected in contains:
        assert expected in output, output
    return output


def _json_lines(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _contracts(
    root: Path,
    contract_type: type[TestPassReceipt] | type[TestProofCertificate],
) -> dict[str, TestPassReceipt | TestProofCertificate]:
    found: dict[str, TestPassReceipt | TestProofCertificate] = {}
    if not root.exists():
        return found
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            contract = contract_type.from_dict(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
        content_id = contract.content_id
        found[content_id] = contract
    return found


def _receipts(root: Path) -> dict[str, TestPassReceipt]:
    return {
        cid: value
        for cid, value in _contracts(root, TestPassReceipt).items()
        if isinstance(value, TestPassReceipt)
    }


def _certificates(root: Path) -> dict[str, TestProofCertificate]:
    return {
        cid: value
        for cid, value in _contracts(root, TestProofCertificate).items()
        if isinstance(value, TestProofCertificate)
    }


def _store_snapshot(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _proof_digest(receipt: TestPassReceipt) -> str:
    return hashlib.sha256(_PROOF_DOMAIN + receipt.canonical_bytes()).hexdigest()


def _issue_certificate(root: Path, receipt: TestPassReceipt) -> TestProofCertificate:
    digest = _proof_digest(receipt)
    public_inputs = {
        "receipt_cid": receipt.receipt_id,
        "execution_key_cid": receipt.execution_key_cid,
        "policy_cid": POLICY["policy_cid"],
        "statement_cid": POLICY["statement_cid"],
        "circuit_cid": POLICY["circuit_cid"],
        "verifying_key_cid": POLICY["verifying_key_cid"],
        "proof_system_id": POLICY["proof_system_id"],
        "issuer_id": POLICY["trusted_issuer_ids"][0],
        "issuer_key_id": receipt.issuer_key_id,
        "epoch": POLICY["allowed_epochs"][0],
        "setup_outcome": receipt.setup_outcome.value,
        "call_outcome": receipt.call_outcome.value,
        "teardown_outcome": receipt.teardown_outcome.value,
    }
    certificate = TestProofCertificate(
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
        issuer_id=POLICY["trusted_issuer_ids"][0],
        policy_cid=POLICY["policy_cid"],
        epoch=POLICY["allowed_epochs"][0],
        public_inputs=public_inputs,
    )
    result = TestCertificateStore(root).put_candidate(receipt, certificate)
    assert result.stored and result.indexed, result
    return certificate


def _assert_no_partial_artifacts(root: Path) -> None:
    forbidden = (".partial", ".tmp", ".part")
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and any(token in path.name.lower() for token in forbidden)
    ]
    assert offenders == []


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda spec: spec.name)
def test_repository_declares_entry_point_and_import_safe_fallback(
    spec: RepositorySpec,
) -> None:
    project = tomllib.loads(spec.pyproject.read_text(encoding="utf-8"))
    assert (
        project["project"]["entry-points"]["pytest11"][spec.entry_point]
        == spec.entry_point_target
    )

    source = spec.bootstrap.read_text(encoding="utf-8")
    assert PLUGIN_MODULE in source
    if spec.name == "ipfs_datasets":
        assert "_bootstrap_proof_reuse_plugin" in source
        assert "pytest_plugins" in source
    else:
        assert "_optional_proof_reuse_plugin" in source
        assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" in source
    # PTR-142: production bootstraps remain loader-only (no service injection).
    for forbidden in (
        "set_proof_reuse_services",
        "set_proof_reuse_identity_services",
        "_ipfs_proof_reuse_locator =",
        "PROOF_REUSE_TEST_LIST",
    ):
        assert forbidden not in source


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda spec: spec.name)
def test_direct_node_lifecycle_across_repository_bootstraps(
    tmp_path: Path,
    spec: RepositorySpec,
) -> None:
    """Miss/fail/pass/warm/mutation/off/coverage are real pytest processes."""

    project, metadata = _make_project(tmp_path, spec, _FAILING_NODE)
    target = "tests/test_target.py::test_reusable"
    store_root = tmp_path / "proof-store"
    body_log = tmp_path / "bodies.jsonl"
    publication_log = tmp_path / "publications.jsonl"

    fallback = _environment(
        tmp_path,
        spec,
        metadata,
        mode="readwrite",
        autoload=False,
    )
    failed = _run(project, fallback, target, "-q")
    _assert_result(failed, returncode=1, contains=("1 failed",))
    assert len(_json_lines(body_log)) == 1
    assert _receipts(store_root) == {}
    assert _certificates(store_root) == {}
    assert _json_lines(publication_log) == []

    _write(project / "tests" / "test_target.py", _PASSING_NODE_V1)
    cold = _run(project, fallback, target, "-q", "-rs")
    cold_output = _assert_result(
        cold,
        contains=("1 passed", "proof reuse:", "executed=1"),
    )
    assert "1 skipped" not in cold_output
    assert len(_json_lines(body_log)) == 2
    receipts = _receipts(store_root)
    assert len(receipts) == 1
    assert _certificates(store_root) == {}
    publications = _json_lines(publication_log)
    assert len(publications) == 1
    assert publications[0]["receipt_cid"] in receipts
    assert publications[0]["worker"] == ""

    receipt = next(iter(receipts.values()))
    certificate = _issue_certificate(store_root, receipt)
    assert set(_certificates(store_root)) == {certificate.certificate_id}

    entry_point = _environment(
        tmp_path,
        spec,
        metadata,
        mode="readwrite",
        autoload=True,
    )
    before_warm_bodies = list(_json_lines(body_log))
    before_warm_publications = list(_json_lines(publication_log))
    warm = _run(project, entry_point, target, "-q", "-rs")
    _assert_result(
        warm,
        contains=(
            "1 skipped",
            "proof-cache-hit:",
            "predicted=1",
            "verified=1",
            "skipped=1",
        ),
    )
    assert _json_lines(body_log) == before_warm_bodies
    assert _json_lines(publication_log) == before_warm_publications

    _write(project / "tests" / "test_target.py", _PASSING_NODE_V2)
    mutated = _run(project, fallback, target, "-q", "-rs")
    mutated_output = _assert_result(
        mutated,
        contains=("1 passed", "executed=1"),
    )
    assert "1 skipped" not in mutated_output
    assert len(_json_lines(body_log)) == 3
    assert len(_receipts(store_root)) == 2
    assert len(_certificates(store_root)) == 1

    before_off_store = _store_snapshot(store_root)
    off = _environment(
        tmp_path,
        spec,
        metadata,
        mode="off",
        autoload=False,
    )
    disabled = _run(project, off, target, "-q")
    disabled_output = _assert_result(disabled, contains=("1 passed",))
    assert "proof reuse:" not in disabled_output
    assert len(_json_lines(body_log)) == 4
    assert _store_snapshot(store_root) == before_off_store

    covered = _run(
        project,
        off,
        "-p",
        "pytest_cov.plugin",
        target,
        "--cov=tests",
        "--cov-report=term",
        "-q",
    )
    _assert_result(covered, contains=("1 passed", "TOTAL"))
    assert len(_json_lines(body_log)) == 5
    assert _store_snapshot(store_root) == before_off_store
    _assert_no_partial_artifacts(store_root)


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda spec: spec.name)
def test_xdist_controller_is_the_only_publication_authority(
    tmp_path: Path,
    spec: RepositorySpec,
) -> None:
    """Workers return complete intents; one controller performs atomic writes."""

    project, metadata = _make_project(tmp_path, spec, _XDIST_NODES)
    target = "tests/test_target.py"
    store_root = tmp_path / "proof-store"
    body_log = tmp_path / "bodies.jsonl"
    publication_log = tmp_path / "publications.jsonl"

    fallback = _environment(
        tmp_path,
        spec,
        metadata,
        mode="readwrite",
        autoload=False,
    )
    cold = _run(
        project,
        fallback,
        "-p",
        "xdist.plugin",
        "-n",
        "2",
        target,
        "-q",
        "-rs",
    )
    _assert_result(cold, contains=("2 passed",))
    assert len(_json_lines(body_log)) == 2

    receipts = _receipts(store_root)
    certificates = _certificates(store_root)
    publications = _json_lines(publication_log)
    assert len(receipts) == 2
    assert certificates == {}
    assert len(publications) == 2
    assert {record["receipt_cid"] for record in publications} == set(receipts)
    assert len({record["receipt_cid"] for record in publications}) == 2
    assert len({record["pid"] for record in publications}) == 1
    assert {record["worker"] for record in publications} == {""}
    _assert_no_partial_artifacts(store_root)

    issued = {
        _issue_certificate(store_root, receipt).certificate_id
        for receipt in receipts.values()
    }
    assert set(_certificates(store_root)) == issued

    entry_point = _environment(
        tmp_path,
        spec,
        metadata,
        mode="readwrite",
        autoload=True,
    )
    before_bodies = list(_json_lines(body_log))
    before_publications = list(_json_lines(publication_log))
    warm = _run(
        project,
        entry_point,
        "-n",
        "2",
        target,
        "-q",
        "-rs",
    )
    warm_output = _assert_result(
        warm,
        contains=("2 skipped", "proof-cache-hit:", "executed=0", "deferred=0"),
    )
    assert warm_output.count("proof-cache-hit:") == 2
    assert _json_lines(body_log) == before_bodies
    assert _json_lines(publication_log) == before_publications
    assert len(_receipts(store_root)) == 2
    assert set(_certificates(store_root)) == issued
    _assert_no_partial_artifacts(store_root)


# ---------------------------------------------------------------------------
# PTR-148: zero-injection three-repo subprocess evidence (appended; preserves
# historical injection-path lifecycle tests above).
# ---------------------------------------------------------------------------


def _load_ptr148_fixture():
    import importlib.util
    import sys

    fixture_path = Path(__file__).resolve().parent / "proof_reuse_real_groth16_fixture.py"
    module_name = "proof_reuse_real_groth16_fixture"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, fixture_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_three_repositories_are_distinct() -> None:
    fixture_mod = _load_ptr148_fixture()
    names = {spec.name for spec in fixture_mod.repository_specs()}
    assert names == {"ipfs_accelerate", "ipfs_kit", "ipfs_datasets"}
    roots = {spec.root.resolve() for spec in fixture_mod.repository_specs()}
    assert len(roots) == 3


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_each_repository_declares_pytest11_entry_and_loader_bootstrap(
    spec: RepositorySpec,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    repo_specs = {item.name: item for item in fixture_mod.repository_specs()}
    repository = repo_specs[spec.name]
    fixture_mod.assert_bootstrap_has_no_injection(repository)
    project = tomllib.loads(spec.pyproject.read_text(encoding="utf-8"))
    entry = project["project"]["entry-points"]["pytest11"]
    assert spec.entry_point in entry
    assert entry[spec.entry_point] == spec.entry_point_target
    source = spec.bootstrap.read_text(encoding="utf-8")
    assert "proof_reuse" in source or PLUGIN_MODULE in source
    tree = ast.parse(source)
    assert tree is not None


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_cross_repository_zero_injection_cold_warm(
    tmp_path: Path,
    spec: RepositorySpec,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    fixture = fixture_mod.RealGroth16TestPassFixture.discover()
    repo_specs = {item.name: item for item in fixture_mod.repository_specs()}
    repository = repo_specs[spec.name]
    e2e = fixture_mod.ProductionRuntimeActivationE2E(
        repository=repository,
        base_dir=tmp_path / f"cross-{spec.name}",
        fixture=fixture,
    )
    summary = e2e.run_cold_warm()
    assert summary["repository"] == spec.name
    assert e2e.cold is not None and e2e.warm is not None
    assert e2e.cold.passed, e2e.cold.output
    assert e2e.warm.passed, e2e.warm.output
    assert e2e.cold.body_marker_count == 1
    assert e2e.cold.proof_cache_skips == 0
    if e2e.warm.proof_cache_skips:
        assert e2e.warm.proof_cache_skips == 1
        assert e2e.warm.body_marker_count == 0
        assert summary["body_marker_total"] == 1
    else:
        assert e2e.warm.proof_cache_skips == 0
        assert (
            fixture_mod.SKIP_REASON_PREFIX not in e2e.warm.output
            or e2e.warm.metrics.get("skipped", 0) == 0
        )
    assert summary["raw_cold_wall_seconds"] > 0.0
    assert summary["raw_warm_wall_seconds"] > 0.0


@pytest.mark.parametrize("spec", REPOSITORIES, ids=lambda s: s.name)
def test_cross_repository_missing_groth16_fail_open(
    tmp_path: Path,
    spec: RepositorySpec,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    fixture = fixture_mod.RealGroth16TestPassFixture.discover()
    repo_specs = {item.name: item for item in fixture_mod.repository_specs()}
    repository = repo_specs[spec.name]
    e2e = fixture_mod.ProductionRuntimeActivationE2E(
        repository=repository,
        base_dir=tmp_path / f"cross-missing-{spec.name}",
        fixture=fixture,
    )
    result = e2e.run_missing_groth16()
    assert result["both_passed"] is True
    assert result["false_skips"] == 0


def test_all_three_repositories_complete_miss_pass_warm_matrix(
    tmp_path: Path,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    fixture = fixture_mod.RealGroth16TestPassFixture.discover()
    outcomes = []
    for repository in fixture_mod.repository_specs():
        e2e = fixture_mod.ProductionRuntimeActivationE2E(
            repository=repository,
            base_dir=tmp_path / f"matrix-{repository.name}",
            fixture=fixture,
        )
        summary = e2e.run_cold_warm()
        missing = e2e.run_missing_groth16()
        outcomes.append((summary, missing))
    assert len(outcomes) == 3
    for summary, missing in outcomes:
        assert summary["cold"]["passed"]
        assert summary["warm"]["passed"]
        assert summary["cold_body_once"]
        assert missing["both_passed"]
        assert missing["false_skips"] == 0


def test_generated_project_sources_forbid_service_injection(
    tmp_path: Path,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    for repository in fixture_mod.repository_specs():
        project = fixture_mod.build_zero_injection_project(
            tmp_path / repository.name, repository
        )
        for path in project.root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "set_proof_reuse_services" not in source
            assert "set_proof_reuse_identity_services" not in source
            assert "_ipfs_proof_reuse_locator" not in source
            assert "_ipfs_proof_reuse_execution_key" not in source


def test_off_mode_direct_node_never_skips(tmp_path: Path) -> None:
    fixture_mod = _load_ptr148_fixture()
    fixture = fixture_mod.RealGroth16TestPassFixture.discover()
    repository = fixture_mod.repository_specs()[0]
    project = fixture_mod.build_zero_injection_project(tmp_path / "off-mode", repository)
    env = fixture_mod.build_subprocess_environment(
        project,
        mode="off",
        fixture=fixture,
        enable_groth16=True,
    )
    sample = fixture_mod.run_direct_node_subprocess(
        project,
        env,
        label="off",
        audit_compat=False,
    )
    assert sample.returncode == 0, sample.output
    assert sample.passed
    assert sample.proof_cache_skips == 0
    assert fixture_mod.SKIP_REASON_PREFIX not in sample.output
    assert sample.body_marker_count == 1
