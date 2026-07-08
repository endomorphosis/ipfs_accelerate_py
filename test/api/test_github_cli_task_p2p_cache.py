import time

import ipfs_accelerate_py.p2p_tasks.client as p2p_client
from ipfs_accelerate_py.github_cli.cache import GitHubAPICache


def test_task_p2p_remote_hit_populates_local_and_counts_saved_calls(tmp_path, monkeypatch):
	# Ensure encryption can initialize deterministically.
	monkeypatch.setenv("GITHUB_TOKEN", "test-token")

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
	monkeypatch.setattr(GitHubAPICache, "_task_p2p_remote", lambda self: object())

	cache = GitHubAPICache(
		cache_dir=str(tmp_path),
		enable_persistence=True,
		enable_p2p=False,
		enable_task_p2p_cache=True,
		default_ttl=60,
	)

	# Seed remote with an encrypted cache entry payload
	key = cache._make_cache_key("op", "a", x=1)
	payload = {
		"cache_key": key,
		"data": {"ok": True},
		"timestamp": time.time(),
		"ttl": 60,
		"content_hash": None,
		"validation_fields": None,
	}
	remote_store[key] = cache._task_p2p_encrypt_value(payload)

	assert cache.get("op", "a", x=1) == {"ok": True}
	assert counts["gets"] == 1

	stats = cache.get_stats()
	assert stats["peer_hits"] == 1
	assert stats["api_calls_saved"] == 1
	assert stats["misses"] == 0

	# Second get should be local hit (no extra remote get)
	assert cache.get("op", "a", x=1) == {"ok": True}
	assert counts["gets"] == 1


def test_task_p2p_write_through_on_put(tmp_path, monkeypatch):
	monkeypatch.setenv("GITHUB_TOKEN", "test-token")

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
	monkeypatch.setattr(GitHubAPICache, "_task_p2p_remote", lambda self: object())

	cache = GitHubAPICache(
		cache_dir=str(tmp_path),
		enable_persistence=False,
		enable_p2p=False,
		enable_task_p2p_cache=True,
		default_ttl=60,
	)

	cache.put("op2", {"v": 1}, ttl=30, k="z")

	key = cache._make_cache_key("op2", k="z")
	assert counts["sets"] == 1
	assert key in remote_store

	# Remote store should contain only encrypted payloads.
	wrapped = remote_store[key]
	assert isinstance(wrapped, dict)
	assert wrapped.get("enc") == "fernet-v1"

	# Decrypt to validate payload correctness.
	decrypted = cache._task_p2p_decrypt_value(wrapped)
	assert decrypted["data"] == {"v": 1}
