"""Deployment capability bytes are checked before a selected trial is prepared."""
import io
import tarfile

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from test.api.test_terminal_decision_archive import _decision_runtime_archive
from test.api.test_terminal_timeout_configuration import _authored_prepare_dataset


@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_changed_deployment_archive_refuses_before_selected_trial_or_harbor(tmp_path, monkeypatch, fault):
    archive, manifest = _decision_runtime_archive(tmp_path)
    target = benchmark.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + "terminal_deployment.py"
    assert target in benchmark._decision_runtime_sources()
    members = []
    with tarfile.open(archive / "runtime.tar.gz", "r:gz") as bundle:
        for member in bundle:
            raw = bundle.extractfile(member).read()
            if member.name == target:
                if fault == "missing":
                    continue
                raw = b"#" + raw[1:]
            members.append((member.name, raw))
    with tarfile.open(archive / "runtime.tar.gz", "w:gz") as bundle:
        for name, raw in members:
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(raw), 0o644
            bundle.addfile(member, io.BytesIO(raw))
    manifest["archive_sha256"] = benchmark._hash(archive / "runtime.tar.gz")
    benchmark._json(archive / "manifest.json", manifest)
    output = tmp_path / "selected-output"
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("changed deployment reached Harbor preflight"))
    with pytest.raises(ValueError, match="selected decision"):
        benchmark.prepare(dataset=_authored_prepare_dataset(tmp_path), archive=archive,
                          output=output, arm="full", semantic_native_input=True)
    assert not output.exists()
