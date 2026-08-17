from ipfs_accelerate_py.agent_supervisor.runtime.decision_receipts import emit_decision
def test_decision_receipt_is_typed():
    assert emit_decision({"decision": "select"})["schema"] == "lgswf/decision-receipt@1"
