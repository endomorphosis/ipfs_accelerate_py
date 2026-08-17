import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state.post_merge_refresh import refresh_canonical, RefreshError
def test_only_post_merge_rescan_refreshes():
    assert refresh_canonical({"accepted_merge": True, "fresh_rescan": True})["canonical"] is True
    with pytest.raises(RefreshError):
        refresh_canonical({"provisional": True, "accepted_merge": True, "fresh_rescan": True})
