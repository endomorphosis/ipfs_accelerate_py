import json
import sys
import types


def test_base_cache_ipfs_pointer_mode_resolves_payload(monkeypatch, tmp_path):
	from ipfs_accelerate_py.common.base_cache import BaseAPICache

	# The ipfs_accelerate_py cache code imports the IPFS router as
	# `from ipfs_datasets_py import ipfs_backend_router as ipfs_router`.
	# When running tests from within the vendored ipfs_accelerate_py tree, the
	# top-level workspace package `ipfs_datasets_py` may not be on sys.path.
	# Provide a stub module so the cache code can import and we can monkeypatch.
	fake_parent = types.ModuleType("ipfs_datasets_py")
	fake_router = types.ModuleType("ipfs_datasets_py.ipfs_backend_router")
	fake_parent.ipfs_backend_router = fake_router
	sys.modules.setdefault("ipfs_datasets_py", fake_parent)
	sys.modules.setdefault("ipfs_datasets_py.ipfs_backend_router", fake_router)

	class DummyCache(BaseAPICache):
		def get_cache_namespace(self) -> str:
			return "dummy_ipfs_pointer"

		def extract_validation_fields(self, operation: str, value):
			return None

		def get_default_ttl_for_operation(self, operation: str) -> int:
			return 60

	# Enable pointer mode
	monkeypatch.setenv("IPFS_ACCELERATE_CACHE_IPFS_POINTERS", "1")

	# Mock IPFS router used by BaseAPICache
	import ipfs_datasets_py.ipfs_backend_router as router

	stored = {}
	calls = {"put": 0, "get": 0}

	def fake_block_put(data: bytes, *, codec: str = "raw", **kwargs):
		assert codec == "raw"
		calls["put"] += 1
		cid = "bafybeigdyrstubcidpayload"
		stored[cid] = data
		return cid

	def fake_block_get(cid: str, **kwargs):
		calls["get"] += 1
		return stored[cid]

	monkeypatch.setattr(router, "block_put", fake_block_put, raising=False)
	monkeypatch.setattr(router, "block_get", fake_block_get, raising=False)

	# Stub p2p methods: capture what would be written remotely and echo it back.
	remote_written = {}

	cache = DummyCache(
		cache_dir=str(tmp_path),
		enable_persistence=False,
		enable_p2p=True,
		p2p_shared_secret="secret",
	)

	def fake_task_p2p_set(cache_key: str, payload: dict, ttl_s: float):
		remote_written[cache_key] = payload
		return True

	def fake_task_p2p_get(cache_key: str):
		return remote_written.get(cache_key)

	monkeypatch.setattr(cache, "_task_p2p_set", fake_task_p2p_set)
	monkeypatch.setattr(cache, "_task_p2p_get", fake_task_p2p_get)

	# Put should store payload in IPFS and only a pointer remotely
	cache.put("op", {"hello": "world"}, prompt="p")

	assert calls["put"] == 1
	# Remote payload should be a pointer, not the full payload
	assert len(remote_written) == 1
	pointer = next(iter(remote_written.values()))
	assert "ipfs_cid" in pointer
	assert "data" not in pointer

	# Force a local miss so get() must resolve pointer via IPFS
	with cache._lock:
		cache._cache.clear()

	# Get should resolve pointer via IPFS
	val = cache.get("op", prompt="p")
	assert val == {"hello": "world"}
	assert calls["get"] == 1
