"""Authored current archive fixtures with genuine capability bytes, no worker execution."""
import io
from pathlib import Path
import tarfile
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as supervisor
from benchmarks.agent_supervisor.container_coding import native_codex_baseline as native
from benchmarks.agent_supervisor.container_coding import terminal_worker_capability as capability

def _timeout_runtime_archive(tmp_path, *, stale_payload=None, omit_payload=None, extra_member=None):
    archive = tmp_path / "runtime"
    archive.mkdir()
    inventory = []
    with tarfile.open(archive / "runtime.tar.gz", "w:gz") as bundle:
        sources = {supervisor.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + name: Path(supervisor.__file__).with_name(name)
                   for name in supervisor.TIMEOUT_RUNTIME_SOURCE_NAMES}
        checkout = Path(supervisor.__file__).resolve().parents[3]
        sources.update({"source/" + name: checkout / name for name in capability.SOURCE_PATHS})
        for path, source in sources.items():
            name = source.name
            raw = source.read_bytes()
            inventory.append({"path": path, "bytes": len(raw),
                              "sha256": supervisor._hash(source),
                              "mode": 0o644})
            if name == omit_payload:
                continue
            payload = (b"#" if raw[0:1] != b"#" else b" ") + raw[1:] if name == stale_payload else raw
            entry = tarfile.TarInfo(path)
            entry.size, entry.mode = len(payload), 0o644
            bundle.addfile(entry, io.BytesIO(payload))
        if extra_member is not None:
            entry, raw = extra_member
            bundle.addfile(entry, io.BytesIO(raw) if raw is not None else None)
    manifest = {"schema": "terminal-supervisor-runtime-archive@1", "files": inventory,
                "archive_sha256": supervisor._hash(archive / "runtime.tar.gz"),
                "codex_version": native.CLI_VERSION, "learned_requirements": [],
                capability.KEY: capability.worker_capability_for_inventory(inventory)}
    supervisor._json(archive / "manifest.json", manifest)
    return archive, manifest

def _authored_prepare_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    task = dataset / supervisor.TASK
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Authored preparation transport fixture.\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "environment").mkdir()
    (task / "tests").mkdir()
    (task / "environment/Dockerfile").write_text("FROM scratch\n")
    (task / "tests/test.sh").write_text("# Authored inert verifier fixture; never executed.\n")
    return dataset
