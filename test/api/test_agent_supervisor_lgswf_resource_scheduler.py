from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    lgswf_release_vector,
    lgswf_reserve_vector,
)

def test_integer_vector_lease_and_release():
    lease = lgswf_reserve_vector({"cpu_ms": 10, "ram_mib": 8}, {"cpu_ms": 4, "ram_mib": 2}, owner="T1")
    assert lease["leased"] is True
    assert lease["reserved"]["cpu_ms"] == 4
    released = lgswf_release_vector(lease)
    assert released["released"] is True
