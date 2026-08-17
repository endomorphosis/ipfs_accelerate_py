from ipfs_accelerate_py.agent_supervisor.runtime.work_cache_coordinator import WorkCacheCoordinator

def test_single_flight_and_integer_estimates() -> None:
    cache = WorkCacheCoordinator()
    cache.remember_estimate("T1", 7)
    assert cache.estimate("T1") == 7
    first = cache.begin_single_flight("T1", "a")
    second = cache.begin_single_flight("T1", "b")
    assert first["joined"] is False
    assert second["joined"] is True
    cache.end_single_flight("T1", "a")
    third = cache.begin_single_flight("T1", "b")
    assert third["joined"] is False
