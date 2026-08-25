from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py import agent_implementation_route as route

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(root: Path, message: str) -> None:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=LGCVF Test",
        "-c",
        "user.email=lgcvf@example.invalid",
        "commit",
        "-m",
        message,
    )


def _candidate_config() -> bytes:
    value = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "logic_governed_compositional_verification_fabric."
            "scheduler_config@1"
        ),
        "board_namespace": "logic-governed-compositional-verification-fabric-v1",
        "merge_target_branch": (
            "agent/logic-governed-compositional-verification-fabric-v1"
        ),
        "validator_path": route._LGCVF_LIVE_VALIDATOR_PATH,
        "max_lanes": 4,
        "strict_task_sharding": True,
        "lanes": [{"index": index} for index in range(4)],
        "database_program": {
            "authority_mode": "quack",
            "task_source_kind": "duckdb",
            "schema_revision": "datasets-authoritative-operational-v1",
            "failover_policy": "fail_closed",
            "authoritative_transactional_data_model": True,
        },
        "source_binding": {
            "ipfs_datasets_submodule_path": "ipfs_datasets_py",
            "require_initialized_gitlinks": True,
            "require_superproject_gitlink_equals_nested_head": True,
        },
        "provider": {"max_concurrency": 4},
        "authority_policy": {
            "quack_exclusive_transport_required": True,
            "direct_multi_process_duckdb_file_open_permitted": False,
            "ducklake_projection_authoritative": False,
        },
        "ducklake_projection_program": {
            "authority": False,
            "scheduling_prerequisite": False,
        },
        "protected_paths": [
            route._LGCVF_LIVE_CANDIDATE_CONFIG_PATH,
            route._LGCVF_LIVE_OPERATOR_PATH,
            route._LGCVF_LIVE_VALIDATOR_PATH,
        ],
    }
    return json.dumps(value, sort_keys=True).encode() + b"\n"


def _seed_repository(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q")
    datasets = root / "ipfs_datasets_py"
    datasets.mkdir()
    _git(datasets, "init", "-q")
    (datasets / "__init__.py").write_text("# nested root\n")
    package = datasets / "ipfs_datasets_py"
    package.mkdir()
    (package / "__init__.py").write_text("# nested package\n")
    _commit(datasets, "nested")

    for relative in route._LGCVF_LIVE_REQUIRED_SUPERPROJECT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == route._LGCVF_LIVE_CANDIDATE_CONFIG_PATH:
            path.write_bytes(_candidate_config())
        elif path.suffix == ".py":
            path.write_text("# admitted source\n")
        elif path.suffix == ".json":
            path.write_text("{}\n")
        else:
            path.write_text("admitted projection\n")
    unselected_gitlink = root / "ipfs_accelerate_py/mcplusplus"
    unselected_gitlink.mkdir()
    _git(unselected_gitlink, "init", "-q")
    (unselected_gitlink / "README.md").write_text("unselected gitlink\n")
    _commit(unselected_gitlink, "unselected nested repository")
    _commit(root, "superproject")
    return root, _git(root, "rev-parse", "HEAD"), _git(
        root, "rev-parse", "HEAD^{tree}"
    )


def _seed_real_daemon_repository(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "real-daemon-repository"
    root.mkdir()
    _git(root, "init", "-q")
    datasets = root / "ipfs_datasets_py"
    datasets.mkdir()
    _git(datasets, "init", "-q")
    (datasets / "__init__.py").write_text("# nested root\n")
    nested_package = datasets / "ipfs_datasets_py"
    nested_package.mkdir()
    (nested_package / "__init__.py").write_text("# nested package\n")
    _commit(datasets, "nested")

    supervisor_root = REPO_ROOT / "ipfs_accelerate_py/agent_supervisor"
    for source in supervisor_root.rglob("*.py"):
        relative = source.relative_to(REPO_ROOT)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for relative in route._LGCVF_LIVE_REQUIRED_SUPERPROJECT_FILES:
        destination = root / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    _commit(root, "real daemon closure")
    return root, _git(root, "rev-parse", "HEAD"), _git(
        root, "rev-parse", "HEAD^{tree}"
    )


def _record_digest(raw: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def _seed_duckdb_runtime(tmp_path: Path) -> dict[str, object]:
    site = tmp_path / "site-packages"
    package = site / "duckdb"
    metadata = site / "duckdb-1.5.5.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    facade = b"from _duckdb import *\n"
    metadata_bytes = b"Metadata-Version: 2.1\nName: duckdb\nVersion: 1.5.5\n"
    wheel = b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
    (package / "__init__.py").write_bytes(facade)
    (metadata / "METADATA").write_bytes(metadata_bytes)
    (metadata / "WHEEL").write_bytes(wheel)
    rows = [
        ("duckdb/__init__.py", _record_digest(facade), str(len(facade))),
        (
            "duckdb-1.5.5.dist-info/METADATA",
            _record_digest(metadata_bytes),
            str(len(metadata_bytes)),
        ),
        (
            "duckdb-1.5.5.dist-info/WHEEL",
            _record_digest(wheel),
            str(len(wheel)),
        ),
        ("duckdb-1.5.5.dist-info/RECORD", "", ""),
    ]
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    (metadata / "RECORD").write_text(stream.getvalue())

    extensions = tmp_path / "extensions" / "v1.5.5" / "linux_arm64"
    extensions.mkdir(parents=True)
    versions = {
        "quack": "c154811",
        "httpfs": "6c2d9f1",
        "ducklake": "d8a1881e",
    }
    paths: dict[str, Path] = {}
    for name, version in versions.items():
        path = extensions / f"{name}.duckdb_extension"
        path.write_bytes((name + "-extension-bytes").encode())
        Path(str(path) + ".info").write_bytes(
            b"reviewed-extension-version:" + version.encode()
        )
        paths[name] = path
    return {
        "package": package,
        "metadata": metadata,
        "versions": versions,
        "paths": paths,
    }


def test_lgcvf_live_capsule_materialize_seal_read_and_project(
    tmp_path: Path,
) -> None:
    root, head, tree = _seed_repository(tmp_path)
    assert _git(
        root, "ls-tree", head, "--", "ipfs_accelerate_py/mcplusplus"
    ).startswith("160000 commit ")
    runtime = _seed_duckdb_runtime(tmp_path)
    native_authorization = "sha256:" + "a" * 64
    native_dependency = "sha256:" + "b" * 64
    pin = route.materialize_lgcvf_configured_board_live_capsule(
        source_root=root,
        capsule_parent=tmp_path / "capsules",
        source_head=head,
        source_tree=tree,
        python_executable=sys.executable,
        duckdb_package_root=runtime["package"],
        duckdb_distribution_metadata_root=runtime["metadata"],
        duckdb_distribution_version="1.5.5",
        quack_extension_path=runtime["paths"]["quack"],
        quack_extension_version=runtime["versions"]["quack"],
        httpfs_extension_path=runtime["paths"]["httpfs"],
        httpfs_extension_version=runtime["versions"]["httpfs"],
        ducklake_extension_path=runtime["paths"]["ducklake"],
        ducklake_extension_version=runtime["versions"]["ducklake"],
        native_authorization_id=native_authorization,
        native_dependency_id=native_dependency,
    )
    assert pin.source_head == head
    assert pin.datasets_gitlink == pin.datasets_head
    assert pin.python_path_prefixes == (".", "ipfs_datasets_py")
    assert pin.native_authorization_id == native_authorization
    assert pin.native_dependency_id == native_dependency
    assert pin.quack_extension.load_policy == "load_only"
    assert pin.httpfs_extension.authority_role == "quack_transport_dependency"
    assert pin.httpfs_extension.load_policy == "load_only"
    assert pin.ducklake_extension.authority_role == (
        "non_authoritative_projection_only"
    )
    with pytest.raises(ValueError, match="native acceptance differs"):
        route.build_lgcvf_configured_board_live_capsule_pin(
            capsule_root=pin.capsule_root,
            python_executable=sys.executable,
            native_authorization_id="sha256:" + "c" * 64,
            native_dependency_id=native_dependency,
        )
    noncanonical = pin.as_dict()
    noncanonical["unreviewed"] = True
    with pytest.raises(ValueError, match="pin fields"):
        route.parse_lgcvf_configured_board_live_capsule_pin(noncanonical)
    wrong_httpfs_policy = pin.as_dict()
    wrong_httpfs_policy["httpfs_extension"]["load_policy"] = "install"
    with pytest.raises(ValueError, match="extension identity"):
        route.parse_lgcvf_configured_board_live_capsule_pin(
            wrong_httpfs_policy
        )

    sealed = route.seal_lgcvf_configured_board_live_capsule(pin)
    try:
        assert route.verify_lgcvf_configured_board_live_sealed_capsule(
            pin, sealed.descriptor
        ) == f"/proc/self/fd/{sealed.descriptor}"
        assert route.read_lgcvf_configured_board_live_capsule_member(
            pin,
            sealed.descriptor,
            route._LGCVF_LIVE_CANDIDATE_CONFIG_PATH,
        ) == _candidate_config()
        assert route.read_lgcvf_configured_board_live_capsule_member(
            pin,
            sealed.descriptor,
            pin.httpfs_extension.member_path,
        ) == runtime["paths"]["httpfs"].read_bytes()
        assert route.read_lgcvf_configured_board_live_capsule_member(
            pin,
            sealed.descriptor,
            pin.httpfs_extension.info_member_path,
        ) == Path(str(runtime["paths"]["httpfs"]) + ".info").read_bytes()
        with pytest.raises(ValueError, match="not admitted"):
            route.read_lgcvf_configured_board_live_capsule_member(
                pin, sealed.descriptor, "__main__.py"
            )
        parent = tmp_path / "qualification-homes"
        home = route.project_lgcvf_configured_board_live_extensions(
            pin, sealed.descriptor, parent
        )
        assert home == parent / pin.capsule_id.removeprefix("sha256:")
        assert stat.S_IMODE(os.lstat(home).st_mode) == 0o500
        assert stat.S_IMODE(os.lstat(home / ".cache" / "xdg").st_mode) == 0o700
        assert stat.S_IMODE(
            os.lstat(
                home
                / ".duckdb/extensions"
                / pin.quack_extension.relative_path
            ).st_mode
        ) == 0o400
        projected_httpfs = (
            home
            / ".duckdb/extensions"
            / pin.httpfs_extension.relative_path
        )
        assert projected_httpfs.read_bytes() == runtime["paths"][
            "httpfs"
        ].read_bytes()
        assert stat.S_IMODE(os.lstat(projected_httpfs).st_mode) == 0o400
        assert (
            home
            / ".duckdb/extensions"
            / pin.httpfs_extension.info_relative_path
        ).read_bytes() == Path(
            str(runtime["paths"]["httpfs"]) + ".info"
        ).read_bytes()
    finally:
        os.close(sealed.descriptor)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="sealed LGCVF capsules require Linux memfd sealing",
)
def test_lgcvf_sealed_capsule_imports_real_daemon_main_dependency_closure(
    tmp_path: Path,
) -> None:
    root, head, tree = _seed_real_daemon_repository(tmp_path)
    runtime = _seed_duckdb_runtime(tmp_path)
    pin = route.materialize_lgcvf_configured_board_live_capsule(
        source_root=root,
        capsule_parent=tmp_path / "real-daemon-capsules",
        source_head=head,
        source_tree=tree,
        python_executable=sys.executable,
        duckdb_package_root=runtime["package"],
        duckdb_distribution_metadata_root=runtime["metadata"],
        duckdb_distribution_version="1.5.5",
        quack_extension_path=runtime["paths"]["quack"],
        quack_extension_version=runtime["versions"]["quack"],
        httpfs_extension_path=runtime["paths"]["httpfs"],
        httpfs_extension_version=runtime["versions"]["httpfs"],
        ducklake_extension_path=runtime["paths"]["ducklake"],
        ducklake_extension_version=runtime["versions"]["ducklake"],
        native_authorization_id="sha256:" + "a" * 64,
        native_dependency_id="sha256:" + "b" * 64,
    )
    sealed = route.seal_lgcvf_configured_board_live_capsule(pin)
    archive = f"/proc/self/fd/{sealed.descriptor}"
    probe = r"""
import importlib
import sys

archive = sys.argv[1]
stdlib = [
    entry
    for entry in sys.path
    if isinstance(entry, str)
    and entry
    and "site-packages" not in entry.casefold()
    and "dist-packages" not in entry.casefold()
]
sys.path[:] = [archive, *stdlib]
daemon = importlib.import_module(
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
)
owner = importlib.import_module(
    "ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner"
)
intent = importlib.import_module(
    "ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository"
)
prefix = archive + "/"
for module in (daemon, owner, intent):
    origin = getattr(module, "__file__", "")
    if not isinstance(origin, str) or not origin.startswith(prefix):
        raise SystemExit(76)
if not callable(getattr(daemon, "main", None)):
    raise SystemExit(77)

# Portal execution reaches this classifier after a ready claim has been
# promoted through its context phase and immediately before provider
# dispatch.  A negative classification must not import the independently
# qualified EAAEF dispatcher: that closure requires native ``cryptography``,
# which is deliberately absent from this sealed LGCVF capsule.
forbidden = {
    (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "external_agent_container_dispatcher"
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.task_sources."
        "eaaef_borrowed_transaction"
    ),
}
if importlib.util.find_spec("cryptography") is not None:
    raise SystemExit(78)
if forbidden.intersection(sys.modules):
    raise SystemExit(79)

class PortalProvider:
    def run_provider(self):
        raise AssertionError("provider dispatch is outside this import probe")

portal_provider = PortalProvider()
if daemon._database_daemon_exact_container_callback(
    portal_provider.run_provider,
    method_name="run_provider",
):
    raise SystemExit(80)
if forbidden.intersection(sys.modules) or "cryptography" in sys.modules:
    raise SystemExit(81)
daemon.main(["--help"])
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", probe, archive],
            cwd=tmp_path,
            env={"PATH": os.environ.get("PATH", "")},
            pass_fds=(sealed.descriptor,),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.startswith("usage: -c ")
        assert completed.stderr == ""
    finally:
        os.close(sealed.descriptor)
