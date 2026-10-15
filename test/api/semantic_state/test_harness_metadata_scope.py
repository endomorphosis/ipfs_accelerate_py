"""Optional owner metadata scope surrounds execution without weakening gates."""

from contextlib import contextmanager

import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import HarnessDisposition
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import HarnessLoopError
from test.api.semantic_state.test_harness import (
    MemoryDurablePort,
    _accept_request,
    _bootstrap,
    _harness,
)


def test_optional_metadata_scope_covers_bootstrap_run_and_error_cleanup(monkeypatch):
    class ScopedPort(MemoryDurablePort):
        def __init__(self):
            super().__init__()
            self.active = False
            self.events = []

        @contextmanager
        def metadata_scope(self):
            assert not self.active
            self.active = True
            self.events.append("enter")
            try:
                yield
            finally:
                self.events.append("exit")
                self.active = False

        def read_root(self, repository_id):
            assert self.active
            return super().read_root(repository_id)

        def put(self, artifact, **kwargs):
            assert self.active
            return super().put(artifact, **kwargs)

    durable = ScopedPort()
    _, harness = _harness(durable)
    root = _bootstrap(harness)
    assert durable.events == ["enter", "exit"]
    assert not durable.active
    outcome = harness.run(_accept_request(root))
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    assert durable.events == ["enter", "exit"] * 2
    assert not durable.active

    with pytest.raises(HarnessLoopError, match="empty root"):
        _bootstrap(harness)
    assert durable.events == ["enter", "exit"] * 3
    assert not durable.active

    sentinel = RuntimeError("storage error remains visible")

    def failing_loop(*_args, **_kwargs):
        assert durable.active
        raise sentinel

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.harness._run_loop", failing_loop
    )
    with pytest.raises(RuntimeError) as failure:
        harness.run(_accept_request(root))
    assert failure.value is sentinel
    assert durable.events == ["enter", "exit"] * 4
    assert not durable.active


def test_port_without_metadata_scope_keeps_bootstrap_and_acceptance_behavior():
    durable, harness = _harness()
    assert not hasattr(durable, "metadata_scope")
    root = _bootstrap(harness)
    outcome = harness.run(_accept_request(root))
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    assert durable.read_root("repo:test") == outcome.result.current_root
    assert outcome.result.current_root.generation == 2
