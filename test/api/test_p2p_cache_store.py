import time
from pathlib import Path


def test_disk_ttl_cache_roundtrip(tmp_path: Path) -> None:
	from ipfs_accelerate_py.p2p_tasks.cache_store import DiskTTLCache

	cache = DiskTTLCache(tmp_path)
	assert cache.get("missing") is None

	cache.set("k", {"a": 1}, ttl_s=10.0)
	assert cache.get("k") == {"a": 1}
	assert cache.has("k") is True


def test_disk_ttl_cache_expires(tmp_path: Path) -> None:
	from ipfs_accelerate_py.p2p_tasks.cache_store import DiskTTLCache

	cache = DiskTTLCache(tmp_path)
	cache.set("k", "v", ttl_s=0.05)
	assert cache.get("k") == "v"

	time.sleep(0.1)
	assert cache.get("k") is None
	assert cache.has("k") is False


def test_disk_ttl_cache_delete(tmp_path: Path) -> None:
	from ipfs_accelerate_py.p2p_tasks.cache_store import DiskTTLCache

	cache = DiskTTLCache(tmp_path)
	cache.set("k", 123)
	assert cache.get("k") == 123
	assert cache.delete("k") is True
	assert cache.get("k") is None
