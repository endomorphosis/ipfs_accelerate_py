import json
import sys
import types


def test_ipns_index_lookup_fetches_payload(monkeypatch, tmp_path):
	from ipfs_accelerate_py.common.base_cache import BaseAPICache
	from ipfs_accelerate_py.common.ipfs_mutable_index import reset_global_mutable_index

	# Provide a stub `ipfs_datasets_py.ipfs_backend_router` module (see comment in
	# test_base_cache_ipfs_pointer_mode.py) so the cache/index code can import it.
	fake_parent = types.ModuleType("ipfs_datasets_py")
	fake_router = types.ModuleType("ipfs_datasets_py.ipfs_backend_router")
	fake_parent.ipfs_backend_router = fake_router
	sys.modules.setdefault("ipfs_datasets_py", fake_parent)
	sys.modules.setdefault("ipfs_datasets_py.ipfs_backend_router", fake_router)

	class DummyCache(BaseAPICache):
		def get_cache_namespace(self) -> str:
			return "dummy_ipns_index"

		def extract_validation_fields(self, operation: str, value):
			return None

		def get_default_ttl_for_operation(self, operation: str) -> int:
			return 60

	# Enable mutable index via env
	monkeypatch.setenv("IPFS_ACCELERATE_CACHE_IPNS_INDEX", "1")
	monkeypatch.setenv("IPFS_ACCELERATE_CACHE_IPNS_NAME", "k51testname")
	monkeypatch.setenv("IPFS_ACCELERATE_CACHE_PUBSUB_REPLICATION", "0")
	monkeypatch.setenv("IPFS_FALLBACK_ENABLED", "false")

	import ipfs_datasets_py.ipfs_backend_router as router

	# Payload storage via raw blocks
	blocks = {}

	def fake_block_put(data: bytes, *, codec: str = "raw", **kwargs):
		cid = "bafybeipayloadcid"
		blocks[cid] = data
		return cid

	def fake_block_get(cid: str, **kwargs):
		return blocks[cid]

	monkeypatch.setattr(router, "block_put", fake_block_put, raising=False)
	monkeypatch.setattr(router, "block_get", fake_block_get, raising=False)

	# Snapshot storage via add_bytes/cat
	snapshots = {}

	def fake_add_bytes(data: bytes, *, pin: bool = True, **kwargs):
		cid = "bafybeiindexcid"
		snapshots[cid] = data
		return cid

	def fake_cat(cid: str, **kwargs):
		return snapshots[cid]

	monkeypatch.setattr(router, "add_bytes", fake_add_bytes, raising=False)
	monkeypatch.setattr(router, "cat", fake_cat, raising=False)

	published = {"cid": None}

	def fake_name_publish(cid: str, *, key=None, allow_offline: bool = True, **kwargs):
		published["cid"] = cid
		return f"Published to k51testname: /ipfs/{cid}"

	def fake_name_resolve(name: str, *, timeout_s: float = 10.0, **kwargs):
		assert name == "k51testname"
		return f"/ipfs/{published['cid']}"

	monkeypatch.setattr(router, "name_publish", fake_name_publish, raising=False)
	monkeypatch.setattr(router, "name_resolve", fake_name_resolve, raising=False)

	reset_global_mutable_index()
	try:
		# Create cache after env is set (so it creates the global index)
		cache = DummyCache(cache_dir=str(tmp_path), enable_persistence=False, enable_p2p=False)

		# Put stores payload in IPFS and publishes index snapshot
		cache.put("op", {"v": 1}, prompt="p")
		assert published["cid"] is not None

		# Force local miss
		with cache._lock:
			cache._cache.clear()

		# Should retrieve via IPNS index -> payload CID -> block_get
		val = cache.get("op", prompt="p")
		assert val == {"v": 1}

		# Snapshot should include mapping from cache_key to payload CID
		idx = json.loads(snapshots[published["cid"]].decode("utf-8"))
		assert "entries" in idx
	finally:
		reset_global_mutable_index()


def test_ipns_index_wrong_secret_cannot_decrypt_payload(monkeypatch, tmp_path):
	from ipfs_accelerate_py.common.base_cache import BaseAPICache
	from ipfs_accelerate_py.common.ipfs_mutable_index import reset_global_mutable_index

	# Provide a stub `ipfs_datasets_py.ipfs_backend_router` module so the cache/index
	# code can import it from within the vendored ipfs_accelerate_py test tree.
	fake_parent = types.ModuleType("ipfs_datasets_py")
	fake_router = types.ModuleType("ipfs_datasets_py.ipfs_backend_router")
	fake_parent.ipfs_backend_router = fake_router
	sys.modules.setdefault("ipfs_datasets_py", fake_parent)
	sys.modules.setdefault("ipfs_datasets_py.ipfs_backend_router", fake_router)

	class DummyCache(BaseAPICache):
		def get_cache_namespace(self) -> str:
			return "dummy_ipns_index_secret"

		def extract_validation_fields(self, operation: str, value):
			return None

		def get_default_ttl_for_operation(self, operation: str) -> int:
			return 60

	# Enable mutable index via env (but suppress snapshot publishing to avoid needing IPNS mocks)
	monkeypatch.setenv("IPFS_ACCELERATE_CACHE_IPNS_INDEX", "1")
	monkeypatch.setenv("IPFS_ACCELERATE_CACHE_IPNS_NAME", "k51testname")
	monkeypatch.setenv("IPFS_ACCELERATE_CACHE_PUBSUB_REPLICATION", "0")
	monkeypatch.setenv("IPFS_ACCELERATE_CACHE_IPNS_PUBLISH_MIN_INTERVAL_S", "999999")
	monkeypatch.setenv("IPFS_FALLBACK_ENABLED", "false")

	import ipfs_datasets_py.ipfs_backend_router as router

	# Payload storage via raw blocks
	blocks = {}
	put_calls = {"n": 0}
	get_calls = {"n": 0}

	def fake_block_put(data: bytes, *, codec: str = "raw", **kwargs):
		assert codec == "raw"
		put_calls["n"] += 1
		cid = "bafybeipayloadcid-secret"
		blocks[cid] = data
		return cid

	def fake_block_get(cid: str, **kwargs):
		get_calls["n"] += 1
		return blocks[cid]

	monkeypatch.setattr(router, "block_put", fake_block_put, raising=False)
	monkeypatch.setattr(router, "block_get", fake_block_get, raising=False)

	reset_global_mutable_index()
	try:
		good = DummyCache(cache_dir=str(tmp_path), enable_persistence=False, enable_p2p=True, p2p_shared_secret="good")
		# Ensure no network dialing even if bootstrap env is set outside pytest.
		monkeypatch.setattr(good, "_task_p2p_remote", lambda: None)

		good.put("op", {"v": 1}, prompt="p")
		assert put_calls["n"] == 1

		bad = DummyCache(cache_dir=str(tmp_path), enable_persistence=False, enable_p2p=True, p2p_shared_secret="bad")
		monkeypatch.setattr(bad, "_task_p2p_remote", lambda: None)

		# Force local misses so both go through index -> IPFS
		with good._lock:
			good._cache.clear()
		with bad._lock:
			bad._cache.clear()

		assert good.get("op", prompt="p") == {"v": 1}
		assert bad.get("op", prompt="p") is None
		assert get_calls["n"] >= 1
	finally:
		reset_global_mutable_index()
