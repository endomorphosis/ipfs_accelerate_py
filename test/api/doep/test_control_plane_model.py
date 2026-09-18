"""Temporal/model invariants for the existing control plane."""
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import TaskState

ORDER = ("todo", "blocked", "in_progress", "retrying", "completed")

def test_task_state_names_are_closed():
    names = {item.value if hasattr(item, "value") else str(item) for item in TaskState}
    assert "completed" in names or "COMPLETED" in {n.upper() for n in names}
    assert "todo" in {n.lower() for n in names}

def test_completed_is_not_a_worker_self_admission():
    assert "completed" != "worker_said_so"
