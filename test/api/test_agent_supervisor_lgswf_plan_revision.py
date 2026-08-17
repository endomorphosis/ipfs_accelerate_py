from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import PlanRevisionStore
def test_cas_roundtrip(tmp_path):
    store = PlanRevisionStore(tmp_path)
    cid = store.put_cas({"revision": "R2", "r2_cid": "x"})
    assert store.get_cas(cid)["revision"] == "R2"
