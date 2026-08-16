from ipfs_accelerate_py.agent_supervisor.autonomous_repair.drift_monitor import (
    DependencyGraph,
    DriftRoots,
    monitor_contract_drift,
)


def _roots(s="s", t="t", c="c", r="r", f="f"):
    return DriftRoots(f, s, c, t, r)


def test_closure_irrelevant_and_repeat_are_deterministic():
    prior = _roots()
    graph = DependencyGraph(
        prior, (("source", "proof"), ("proof", "plan"), ("toolchain", "cache"), ("runtime", "task"))
    )
    changed = _roots(s="s2")
    result = monitor_contract_drift(prior, changed, graph)
    assert result.affected == ("plan", "proof", "source") and result.execution_authorized is False
    assert monitor_contract_drift(prior, prior, graph) == monitor_contract_drift(
        prior, prior, graph
    )


def test_tool_runtime_and_forged_graph():
    prior = _roots()
    graph = DependencyGraph(prior, (("toolchain", "cache"), ("runtime", "task")))
    assert "cache" in monitor_contract_drift(prior, _roots(t="t2"), graph).affected
    assert monitor_contract_drift(_roots(s="other"), prior, graph).disposition == "rejected"
