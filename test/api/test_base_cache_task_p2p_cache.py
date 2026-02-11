import time

import ipfs_accelerate_py.p2p_tasks.client as p2p_client
from ipfs_accelerate_py.common.base_cache import BaseAPICache


class _DummyCache(BaseAPICache):
	def get_cache_namespace(self) -> str:
		return "dummy_api"

	def extract_validation_fields(self, operation: str, data):
		return None


def test_base_cache_task_p2p_remote_hit_is_encrypted_and_populates_local(tmp_path, monkeypatch):
	# Provide a shared secret so BaseAPICache enables encryption.
	monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

	remote_store = {}
	counts = {"gets": 0, "sets": 0}

	def fake_cache_get_sync(*, remote, key, timeout_s=10.0):
		counts["gets"] += 1
		return {"ok": True, "hit": key in remote_store, "value": remote_store.get(key)}

	def fake_cache_set_sync(*, remote, key, value, ttl_s=None, timeout_s=10.0):
		counts["sets"] += 1
		remote_store[key] = value
		return {"ok": True}

	monkeypatch.setattr(p2p_client, "cache_get_sync", fake_cache_get_sync)
	monkeypatch.setattr(p2p_client, "cache_set_sync", fake_cache_set_sync)

	cache = _DummyCache(
		cache_dir=str(tmp_path),
		enable_persistence=False,
		enable_p2p=True,
		p2p_shared_secret="test-openai-key",
		p2p_secret_salt=b"dummy-task-p2p-cache",
		enable_pubsub=False,
		default_ttl=60,
	)

	# Force remote availability regardless of bootstrap env.
	monkeypatch.setattr(_DummyCache, "_task_p2p_remote", lambda self: object())

	# Seed remote with encrypted payload under namespaced key.
	key = cache.make_cache_key("op", "a", x=1)
	payload = {
		"data": {"ok": True},
		"timestamp": time.time(),
		"ttl": 60,
		"content_hash": None,
		"validation_fields": None,
		"metadata": None,
		"operation": "op",
	}

	remote_key = cache._task_p2p_key(key)
	remote_store[remote_key] = cache._task_p2p_encrypt_value(payload)

	# First get should hit remote and populate local
	assert cache.get("op", "a", x=1) == {"ok": True}
	assert counts["gets"] == 1

	stats = cache.get_stats()
	assert stats["peer_hits"] == 1

	# Second get should be local hit (no extra remote get)
	assert cache.get("op", "a", x=1) == {"ok": True}
	assert counts["gets"] == 1


def test_base_cache_task_p2p_write_through_stores_ciphertext_only(tmp_path, monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

	remote_store = {}

	def fake_cache_get_sync(*, remote, key, timeout_s=10.0):
		return {"ok": True, "hit": key in remote_store, "value": remote_store.get(key)}

	def fake_cache_set_sync(*, remote, key, value, ttl_s=None, timeout_s=10.0):
		remote_store[key] = value
		return {"ok": True}

	monkeypatch.setattr(p2p_client, "cache_get_sync", fake_cache_get_sync)
	monkeypatch.setattr(p2p_client, "cache_set_sync", fake_cache_set_sync)

	cache = _DummyCache(
		cache_dir=str(tmp_path),
		enable_persistence=False,
		enable_p2p=True,
		p2p_shared_secret="test-openai-key",
		p2p_secret_salt=b"dummy-task-p2p-cache",
		enable_pubsub=False,
		default_ttl=60,
	)

	monkeypatch.setattr(_DummyCache, "_task_p2p_remote", lambda self: object())

	cache.put("op", {"value": 123}, "a", x=1, ttl=60)

	key = cache.make_cache_key("op", "a", x=1)
	remote_key = cache._task_p2p_key(key)

	assert remote_key in remote_store
	wrapped = remote_store[remote_key]

	# Ensure remote value is an encryption wrapper, not plaintext.
	assert isinstance(wrapped, dict)
	assert wrapped.get("enc") == "fernet-v1"
	assert "ct" in wrapped

	# Ensure it decrypts back to the payload dict.
	decrypted = cache._task_p2p_decrypt_value(wrapped)
	assert isinstance(decrypted, dict)
	assert decrypted.get("data") == {"value": 123}
