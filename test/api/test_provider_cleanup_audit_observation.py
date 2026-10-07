"""Cleanup diagnostics must never create provider custody while observing it."""
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)


@pytest.mark.parametrize("store_kind", ["missing", "symlink"])
def test_terminal_audit_does_not_create_or_follow_a_store(tmp_path, store_kind):
    store = tmp_path / "uncreated-parent" / "attempts"
    if store_kind == "symlink":
        target = tmp_path / "other-private-state"
        target.mkdir(mode=0o700)
        store.parent.mkdir(mode=0o700)
        store.symlink_to(target, target_is_directory=True)
    invocation = {
        "provider_attempt_store": str(store),
        "provider_attempt_store_identity": "sha256:" + "a" * 64,
        "logical_attempt_id": "sha256:" + "b" * 64,
    }
    audit = PortalImplementationDaemon._protected_provider_effect_audit(
        repo_root=tmp_path,
        command_items=["runner", "--agent-implementation-route-json",
                       json.dumps({"invocation_binding": invocation})],
        receipt_text="", returncode=0,
    )
    assert audit == {"exhausted": False, "providers": [], "reason": ""}
    if store_kind == "missing":
        assert not store.parent.exists()
    else:
        assert store.is_symlink()
        assert list(target.iterdir()) == []
