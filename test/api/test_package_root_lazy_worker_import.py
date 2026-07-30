"""Focused compatibility checks for the package-root worker export."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


def _subprocess_env(*, skip_core: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    env["IPFS_ACCEL_SKIP_CORE"] = "1" if skip_core else "0"
    env["IPFS_ACCEL_IMPORT_EAGER"] = "0"
    return env


def test_agent_supervisor_import_does_not_eagerly_load_worker_skillsets() -> None:
    script = """
import json
import sys

forbidden = ("torch", "transformers", "openai", "neo4j", "duckdb", "anthropic")
before = {name for name in sys.modules if name.split(".")[0] in forbidden}

import ipfs_accelerate_py as package
import ipfs_accelerate_py.agent_supervisor

after = {name for name in sys.modules if name.split(".")[0] in forbidden}
worker_modules = sorted(
    name
    for name in sys.modules
    if name == "ipfs_accelerate_py.worker"
    or name.startswith("ipfs_accelerate_py.worker.")
)
print(json.dumps({
    "added_optional_modules": sorted(after - before),
    "raw_export_worker_is_lazy_module": (
        type(dict.__getitem__(package.export, "worker")).__name__
        == "_LazyWorkerSnapshot"
    ),
    "root_worker_is_unresolved": "worker" not in package.__dict__,
    "dir_has_worker": "worker" in dir(package),
    "all_has_worker": "worker" in package.__all__,
    "worker_modules": worker_modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "added_optional_modules": [],
        "raw_export_worker_is_lazy_module": True,
        "root_worker_is_unresolved": True,
        "dir_has_worker": True,
        "all_has_worker": True,
        "worker_modules": [],
    }


def test_explicit_root_worker_access_resolves_and_synchronizes_export() -> None:
    script = """
import json
import sys
import types

import ipfs_accelerate_py as package

worker_package = types.ModuleType("ipfs_accelerate_py.worker")
worker_package.__path__ = []
worker_module = types.ModuleType("ipfs_accelerate_py.worker.worker")
worker_module.compatibility_sentinel = "historical-worker-module"
worker_package.worker = worker_module
sys.modules[worker_package.__name__] = worker_package
sys.modules[worker_module.__name__] = worker_module
package.worker = worker_package

from ipfs_accelerate_py import worker as imported_worker

print(json.dumps({
    "resolved_historical_module": (
        imported_worker.compatibility_sentinel == "historical-worker-module"
    ),
    "root_is_imported": package.worker is imported_worker,
    "export_is_imported": package.export["worker"] is imported_worker,
    "export_get_is_imported": package.export.get("worker") is imported_worker,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "resolved_historical_module": True,
        "root_is_imported": True,
        "export_is_imported": True,
        "export_get_is_imported": True,
    }


def test_concurrent_first_worker_access_waits_for_canonical_result() -> None:
    script = """
import json
import threading
import time
import types

import ipfs_accelerate_py as package

worker_module = types.ModuleType("ipfs_accelerate_py.worker.worker")
started = threading.Event()
release = threading.Event()
original_import = package.importlib.import_module

def delayed_import(name, *args, **kwargs):
    if name == "ipfs_accelerate_py.worker.worker":
        started.set()
        release.wait(5)
        return worker_module
    return original_import(name, *args, **kwargs)

package.importlib.import_module = delayed_import
results = {}

def read_root():
    results["root"] = package.worker

def read_export():
    results["export"] = package.export["worker"]

first = threading.Thread(target=read_root)
second = threading.Thread(target=read_export)
first.start()
assert started.wait(5)
second.start()
time.sleep(0.05)
second_waited = second.is_alive()
release.set()
first.join(5)
second.join(5)

print(json.dumps({
    "second_waited": second_waited,
    "root_is_canonical": results["root"] is worker_module,
    "export_is_canonical": results["export"] is worker_module,
    "final_root_is_canonical": package.worker is worker_module,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "second_waited": True,
        "root_is_canonical": True,
        "export_is_canonical": True,
        "final_root_is_canonical": True,
    }


def test_worker_assignment_and_raw_export_snapshots_remain_compatible() -> None:
    script = """
import json
import types

import ipfs_accelerate_py as package

snapshot = dict(package.export)["worker"]
union_snapshot = (package.export | {})["worker"]
setdefault_snapshot = package.export.setdefault("worker", object())
no_eager_worker_modules = not any(
    name == "ipfs_accelerate_py.worker"
    or name.startswith("ipfs_accelerate_py.worker.")
    for name in __import__("sys").modules
)

sentinel = types.SimpleNamespace(name="assigned-worker")
package.worker = sentinel
assigned = (
    package.worker is sentinel
    and package.export["worker"] is sentinel
    and package.export.get("worker") is sentinel
)
del package.worker
raw_reset_is_lazy = (
    type(dict.__getitem__(package.export, "worker")).__name__
    == "_LazyWorkerSnapshot"
)

print(json.dumps({
    "snapshot_is_module_like": isinstance(snapshot, types.ModuleType),
    "union_snapshot_is_same": union_snapshot is snapshot,
    "setdefault_snapshot_is_same": setdefault_snapshot is snapshot,
    "no_eager_worker_modules": no_eager_worker_modules,
    "assigned": assigned,
    "raw_reset_is_lazy": raw_reset_is_lazy,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "snapshot_is_module_like": True,
        "union_snapshot_is_same": True,
        "setdefault_snapshot_is_same": True,
        "no_eager_worker_modules": True,
        "assigned": True,
        "raw_reset_is_lazy": True,
    }


@pytest.mark.parametrize("skip_core", [False, True])
def test_direct_export_mutations_synchronize_the_root_worker_contract(
    skip_core: bool,
) -> None:
    script = """
import json
import types

import ipfs_accelerate_py as package

first = types.SimpleNamespace(name="first")
second = types.SimpleNamespace(name="second")
third = types.SimpleNamespace(name="third")

package.export["worker"] = first
assigned = package.export["worker"] is first and package.worker is first
popped = package.export.pop("worker")
pop_removed = (
    popped is first
    and "worker" not in package.export
    and "worker" not in package.__dict__
)
try:
    package.export["worker"]
except KeyError:
    missing_read_raises = True
else:
    missing_read_raises = False

defaulted = package.export.setdefault("worker", second)
setdefault_synchronized = defaulted is second and package.worker is second
package.export.update({"worker": third})
update_synchronized = package.export["worker"] is third and package.worker is third
package.export |= {"worker": first}
ior_synchronized = package.export["worker"] is first and package.worker is first
del package.export["worker"]
delete_removed = "worker" not in package.export and "worker" not in package.__dict__
package.export["worker"] = second
package.export.clear()
clear_removed = "worker" not in package.export and "worker" not in package.__dict__

print(json.dumps({
    "assigned": assigned,
    "pop_removed": pop_removed,
    "missing_read_raises": missing_read_raises,
    "setdefault_synchronized": setdefault_synchronized,
    "update_synchronized": update_synchronized,
    "ior_synchronized": ior_synchronized,
    "delete_removed": delete_removed,
    "clear_removed": clear_removed,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(skip_core=skip_core),
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "assigned": True,
        "pop_removed": True,
        "missing_read_raises": True,
        "setdefault_synchronized": True,
        "update_synchronized": True,
        "ior_synchronized": True,
        "delete_removed": True,
        "clear_removed": True,
    }


def test_concurrent_lazy_load_then_export_assignment_is_last_writer_wins() -> None:
    script = """
import json
import threading
import time
import types

import ipfs_accelerate_py as package

loaded_worker = types.ModuleType("ipfs_accelerate_py.worker.worker")
assigned_worker = types.SimpleNamespace(name="assigned")
started = threading.Event()
release = threading.Event()
original_import = package.importlib.import_module

def delayed_import(name, *args, **kwargs):
    if name == "ipfs_accelerate_py.worker.worker":
        started.set()
        release.wait(5)
        return loaded_worker
    return original_import(name, *args, **kwargs)

package.importlib.import_module = delayed_import
results = {}

def read_export():
    results["read"] = package.export["worker"]

def assign_export():
    package.export["worker"] = assigned_worker
    results["assigned"] = True

reader = threading.Thread(target=read_export)
writer = threading.Thread(target=assign_export)
reader.start()
assert started.wait(5)
writer.start()
time.sleep(0.05)
writer_waited = writer.is_alive()
release.set()
reader.join(5)
writer.join(5)

print(json.dumps({
    "writer_waited": writer_waited,
    "read_received_loaded_worker": results["read"] is loaded_worker,
    "writer_completed": results.get("assigned") is True,
    "final_export_is_assignment": package.export["worker"] is assigned_worker,
    "final_root_is_assignment": package.worker is assigned_worker,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "writer_waited": True,
        "read_received_loaded_worker": True,
        "writer_completed": True,
        "final_export_is_assignment": True,
        "final_root_is_assignment": True,
    }


def test_skip_core_keeps_worker_disabled_without_provider_imports() -> None:
    script = """
import json
import sys

forbidden = ("torch", "transformers", "openai", "neo4j", "duckdb", "anthropic")
before = {name for name in sys.modules if name.split(".")[0] in forbidden}

import ipfs_accelerate_py as package
from ipfs_accelerate_py import worker as imported_worker

after = {name for name in sys.modules if name.split(".")[0] in forbidden}
print(json.dumps({
    "added_optional_modules": sorted(after - before),
    "root_worker_is_none": imported_worker is None and package.worker is None,
    "export_worker_is_none": package.export["worker"] is None,
    "worker_modules": sorted(
        name for name in sys.modules
        if name == "ipfs_accelerate_py.worker"
        or name.startswith("ipfs_accelerate_py.worker.")
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(skip_core=True),
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "added_optional_modules": [],
        "root_worker_is_none": True,
        "export_worker_is_none": True,
        "worker_modules": [],
    }
