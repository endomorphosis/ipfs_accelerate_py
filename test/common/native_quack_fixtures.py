"""Disposable native Quack owner fixtures independent of retired inbox protocols."""
from __future__ import annotations

from pathlib import Path

import hashlib
import json
import os
import stat
import tarfile

import pytest


def _seed(database: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import open_intent_repository

    repo = open_intent_repository(database, owner_id="seed")
    try:
        repo.upsert_objective(
            objective_id="objective:test", objective_alias="O", title="Objective"
        )
        repo.upsert_goal(
            goal_cid="goal:test",
            goal_alias="G",
            title="Goal",
            objective_id="objective:test",
        )
        repo.upsert_plan(
            plan_cid="plan:test",
            goal_cid="goal:test",
            plan_alias="P",
        )
        repo.upsert_task(
            task_cid="task:test",
            task_alias="T",
            goal_cid="goal:test",
            plan_cid="plan:test",
            objective_id="objective:test",
            ordinal=1,
            status="ready",
        )
    finally:
        repo.close()


# Only these authored semantic fixture outputs may be compacted.
NATIVE_CATALOG_RETENTION_PROFILE = "archive-after-pass@1"
NATIVE_CATALOG_RETENTION_ENV = "IPFS_TEST_NATIVE_CATALOG_RETENTION"
_NATIVE_OUTPUT_NAMES = ("semantic", "semantic-projected", "full-inventory", "mixed-context", "program-context")


def _all_native_case_phases_passed(reports):
    return all(phase in reports and reports[phase].passed is True
               for phase in ("setup", "call", "teardown"))


def _catalog_identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid():
        raise ValueError("native catalog fixture must be one owned regular file")
    return {key: getattr(info, key) for key in (
        "st_dev", "st_ino", "st_size", "st_mode", "st_uid", "st_nlink", "st_mtime_ns", "st_ctime_ns")}


def _sync_native_fixture_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_native_catalog_archive(path, rows):
    expected = {row["path"]: row for row in rows}
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise ValueError("native catalog archive member population differs")
        for member in members:
            if not member.isfile() or member.size != expected[member.name]["identity"]["st_size"]:
                raise ValueError("native catalog archive member type or size differs")
            stream = archive.extractfile(member)
            if stream is None or hashlib.file_digest(stream, "sha256").hexdigest() != expected[member.name]["sha256"]:
                raise ValueError("native catalog archive bytes differ")


def _archive_native_catalogs(*, tmp_path, repository, nodeid):
    """Preserve verified catalog bytes before retiring only their raw copies."""
    if repository != tmp_path / "repository" or repository.resolve(strict=True) != repository:
        raise ValueError("exact authored native repository required")
    candidates = []
    for name in _NATIVE_OUTPUT_NAMES:
        output = repository / ".runtime" / name
        for path in (output / "world.duckdb", output / "metadata.duckdb"):
            if os.path.lexists(path):
                candidates.append(path)
        for lake in (output / "world-lake", output / "metadata-lake"):
            if not os.path.lexists(lake):
                continue
            if lake.is_symlink() or not lake.is_dir():
                raise ValueError("native catalog fixture lake must be an ordinary directory")
            for path in sorted(lake.rglob("*")):
                info = path.lstat()
                if stat.S_ISDIR(info.st_mode):
                    continue
                candidates.append(path)
    if not candidates:
        return None
    if len(candidates) > 256:
        raise ValueError("native catalog fixture file population exceeds bound")
    rows = []
    for path in sorted(candidates):
        for parent in path.parents:
            if parent == tmp_path:
                break
            if parent.is_symlink():
                raise ValueError("native catalog fixture ancestor is a symlink")
        identity = _catalog_identity(path)
        rows.append(dict(path=path.relative_to(tmp_path).as_posix(), identity=identity,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if sum(row["identity"]["st_size"] for row in rows) > 64 * 1024**2:
        raise ValueError("native catalog fixture byte population exceeds bound")
    target = tmp_path / "native-catalog-files.tar.gz"
    pending = tmp_path / "native-catalog-files.tar.gz.pending"
    receipt = tmp_path / "native-catalog-retirement.json"
    if any(os.path.lexists(path) for path in (target, pending, receipt)):
        raise ValueError("native catalog retirement outputs already exist")
    with pending.open("xb") as output:
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for row in rows:
                path = tmp_path / row["path"]
                if _catalog_identity(path) != row["identity"]:
                    raise ValueError("native catalog fixture identity changed before archival")
                archive.add(path, arcname=row["path"], recursive=False)
        output.flush()
        os.fsync(output.fileno())
    _verify_native_catalog_archive(pending, rows)
    os.replace(pending, target)
    _sync_native_fixture_directory(tmp_path)
    record = dict(schema="test-native-catalog-retirement@1", profile=NATIVE_CATALOG_RETENTION_PROFILE,
                  nodeid_sha256=hashlib.sha256(nodeid.encode()).hexdigest(), files=rows,
                  archive=target.name, archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  archived_bytes=target.stat().st_size, retired=False, assertions_completed=True,
                  pytest_phases={phase: "passed" for phase in ("setup", "call", "teardown")},
                  validation_authority=False)
    def write_record():
        staged = receipt.with_suffix(".json.pending")
        with staged.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, receipt)
        _sync_native_fixture_directory(tmp_path)
    write_record()
    for row in rows:
        path = tmp_path / row["path"]
        if (_catalog_identity(path) != row["identity"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError("native catalog fixture changed before raw-copy retirement")
        path.unlink()
    record["retired"] = True
    write_record()
    return record


class _NativeCatalogCaseRetention:
    def __init__(self, request):
        self.request = request
        self.reports = {}
        self.native_paths = None

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        if item is not self.request.node:
            return
        report = outcome.get_result()
        self.reports[report.when] = report
        if report.when == "setup" and report.passed:
            values = item.funcargs
            native = values.get("native")
            tmp_path = values.get("tmp_path")
            if type(native) is tuple and len(native) == 4 and isinstance(tmp_path, Path):
                self.native_paths = (tmp_path, native[0])
        if report.when == "teardown":
            try:
                if _all_native_case_phases_passed(self.reports) and self.native_paths is not None:
                    tmp_path, repository = self.native_paths
                    _archive_native_catalogs(tmp_path=tmp_path, repository=repository, nodeid=item.nodeid)
            finally:
                self.request.config.pluginmanager.unregister(self)


@pytest.fixture(autouse=True)
def archive_native_catalogs(request):
    """Explicit owned-run profile; ordinary pytest retains the original files."""
    profile = os.environ.get(NATIVE_CATALOG_RETENTION_ENV, "")
    if not profile:
        return
    if profile != NATIVE_CATALOG_RETENTION_PROFILE:
        raise ValueError("unknown native catalog fixture retention profile")
    plugin = _NativeCatalogCaseRetention(request)
    request.config.pluginmanager.register(plugin, "native-catalog-retention-" + hashlib.sha256(request.node.nodeid.encode()).hexdigest())
