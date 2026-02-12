"""IPFS-backed mutable cache index.

IPFS is content-addressed; it can fetch payloads by CID, but it cannot answer
"given query-key CID, find payload CID" without an index.

This module provides:
- An IPNS-published snapshot index (mutable pointer to an immutable CID)
- Pubsub replication of incremental key→payloadCID updates

All features are best-effort and opt-in via env vars.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _truthy(value: str | None) -> bool:
	return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


_PUBLISHED_RE = re.compile(r"Published to\s+(?P<name>[^:]+):\s+(?P<path>/ipfs/[^\s]+)")


@dataclass
class IndexEntry:
	payload_cid: str
	ts: float
	ttl_s: float | None
	operation: str | None = None
	cache_name: str | None = None

	def expired(self, now: float | None = None) -> bool:
		if self.ttl_s is None:
			return False
		n = float(time.time() if now is None else now)
		return (n - float(self.ts)) >= float(self.ttl_s)


class IPFSMutableCacheIndex:
	"""Maintains a best-effort key→payloadCID mapping.

	Storage:
	- Snapshot: JSON stored in IPFS (immutable CID)
	- Pointer: published via IPNS (mutable name)
	- Updates: broadcast via pubsub (incremental)
	"""

	def __init__(
		self,
		*,
		enable_ipns: bool,
		enable_pubsub: bool,
		ipns_name: str | None,
		ipns_key: str | None,
		pubsub_topic: str,
		publish_min_interval_s: float = 30.0,
		refresh_min_interval_s: float = 30.0,
	):
		self.enable_ipns = bool(enable_ipns)
		self.enable_pubsub = bool(enable_pubsub)
		self.ipns_name = (ipns_name or "").strip() or None
		self.ipns_key = (ipns_key or "").strip() or None
		self.pubsub_topic = pubsub_topic
		self.publish_min_interval_s = float(publish_min_interval_s)
		self.refresh_min_interval_s = float(refresh_min_interval_s)

		self._lock = threading.Lock()
		self._entries: Dict[str, IndexEntry] = {}
		self._last_publish_at: float = 0.0
		self._last_refresh_at: float = 0.0

		self._stop_event = threading.Event()
		self._sub_thread: threading.Thread | None = None
		self._sub_proc = None

		# Lazy import to keep import-time side effects low.
		ipfs_router = sys.modules.get("ipfs_datasets_py.ipfs_backend_router")
		if ipfs_router is None:
			try:
				# Use local ipfs_backend_router (preferred)
				from .. import ipfs_backend_router as ipfs_router
			except Exception:
				try:
					# Fallback to ipfs_datasets_py for backward compatibility
					from ipfs_datasets_py import ipfs_backend_router as ipfs_router
				except Exception:
					ipfs_router = None

		self._ipfs = ipfs_router

		if self.enable_pubsub:
			self._start_pubsub_subscriber()

	def shutdown(self) -> None:
		self._stop_event.set()
		proc = self._sub_proc
		self._sub_proc = None
		if proc is not None:
			try:
				proc.terminate()
			except Exception:
				pass
		thr = self._sub_thread
		self._sub_thread = None
		if thr is not None:
			try:
				thr.join(timeout=1.0)
			except Exception:
				pass

	def _start_pubsub_subscriber(self) -> None:
		if self._sub_thread is not None:
			return

		def _run() -> None:
			try:
				proc = self._ipfs.pubsub_sub_process(self.pubsub_topic)
				self._sub_proc = proc

				# ipfshttpclient may return an iterator-like subscription.
				if hasattr(proc, "__iter__") and not hasattr(proc, "stdout"):
					for msg in proc:
						if self._stop_event.is_set():
							break
						try:
							data = msg.get("data") if isinstance(msg, dict) else msg
							if isinstance(data, (bytes, bytearray)):
								text = data.decode("utf-8", errors="replace")
							else:
								text = str(data)
							self.apply_pubsub_message(text)
						except Exception:
							continue
					return

				stdout = getattr(proc, "stdout", None)
				if stdout is None:
					return
				for line in stdout:
					if self._stop_event.is_set():
						break
					line = str(line).strip()
					if not line:
						continue
					self.apply_pubsub_message(line)
			except Exception:
				return

		self._sub_thread = threading.Thread(target=_run, name="ipfs-cache-index-pubsub", daemon=True)
		self._sub_thread.start()

	def apply_pubsub_message(self, text: str) -> None:
		try:
			obj = json.loads(text)
		except Exception:
			return
		if not isinstance(obj, dict):
			return
		key = obj.get("cache_key")
		payload_cid = obj.get("payload_cid")
		if not isinstance(key, str) or not isinstance(payload_cid, str):
			return
		try:
			ts = float(obj.get("ts", time.time()))
		except Exception:
			ts = time.time()
		ttl_s = obj.get("ttl_s")
		try:
			ttl_f = float(ttl_s) if ttl_s is not None else None
		except Exception:
			ttl_f = None
		entry = IndexEntry(
			payload_cid=payload_cid,
			ts=ts,
			ttl_s=ttl_f,
			operation=obj.get("operation"),
			cache_name=obj.get("cache_name"),
		)
		with self._lock:
			self._entries[key] = entry

	def update(
		self,
		*,
		cache_key: str,
		payload_cid: str,
		ts: float,
		ttl_s: float | None,
		operation: str | None,
		cache_name: str | None,
	) -> None:
		with self._lock:
			self._entries[cache_key] = IndexEntry(
				payload_cid=payload_cid,
				ts=float(ts),
				ttl_s=float(ttl_s) if ttl_s is not None else None,
				operation=operation,
				cache_name=cache_name,
			)

		if self.enable_pubsub:
			self._publish_pubsub(cache_key, payload_cid, ts, ttl_s, operation, cache_name)

		if self.enable_ipns:
			self._maybe_publish_snapshot()

	def lookup(self, cache_key: str) -> Optional[str]:
		now = time.time()
		with self._lock:
			entry = self._entries.get(cache_key)
			if entry is not None and not entry.expired(now):
				return entry.payload_cid

		if self.enable_ipns and self.ipns_name:
			self._maybe_refresh_snapshot(now)
			with self._lock:
				entry = self._entries.get(cache_key)
				if entry is not None and not entry.expired(now):
					return entry.payload_cid

		return None

	def _publish_pubsub(
		self,
		cache_key: str,
		payload_cid: str,
		ts: float,
		ttl_s: float | None,
		operation: str | None,
		cache_name: str | None,
	) -> None:
		try:
			msg = {
				"cache_key": cache_key,
				"payload_cid": payload_cid,
				"ts": float(ts),
				"ttl_s": float(ttl_s) if ttl_s is not None else None,
				"operation": operation,
				"cache_name": cache_name,
			}
			self._ipfs.pubsub_pub(self.pubsub_topic, json.dumps(msg, sort_keys=True, separators=(",", ":")))
		except Exception:
			return

	def _snapshot_json(self) -> bytes:
		with self._lock:
			entries = {
				k: {
					"payload_cid": v.payload_cid,
					"ts": v.ts,
					"ttl_s": v.ttl_s,
					"operation": v.operation,
					"cache_name": v.cache_name,
				}
				for k, v in self._entries.items()
				if k and v.payload_cid
			}
		obj = {"version": 1, "updated_at": time.time(), "entries": entries}
		return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

	def _maybe_publish_snapshot(self) -> None:
		now = time.time()
		if (now - self._last_publish_at) < self.publish_min_interval_s:
			return
		self._last_publish_at = now

		try:
			cid = self._ipfs.add_bytes(self._snapshot_json(), pin=True)
		except Exception:
			return

		try:
			out = self._ipfs.name_publish(cid, key=self.ipns_key, allow_offline=True)
			# If user didn't configure ipns_name, try to learn it from publish output.
			m = _PUBLISHED_RE.search(str(out) or "")
			if m and not self.ipns_name:
				self.ipns_name = m.group("name").strip()
		except Exception:
			return

	def _maybe_refresh_snapshot(self, now: float) -> None:
		if (now - self._last_refresh_at) < self.refresh_min_interval_s:
			return
		self._last_refresh_at = now

		name = self.ipns_name
		if not name:
			return
		try:
			path = self._ipfs.name_resolve(name, timeout_s=10.0)
		except Exception:
			return
		cid = str(path).strip()
		if cid.startswith("/ipfs/"):
			cid = cid[len("/ipfs/") :]
		if not cid:
			return

		try:
			raw = self._ipfs.cat(cid)
			obj = json.loads(raw.decode("utf-8", errors="replace"))
		except Exception:
			return
		if not isinstance(obj, dict):
			return
		entries = obj.get("entries")
		if not isinstance(entries, dict):
			return

		loaded: Dict[str, IndexEntry] = {}
		for k, v in entries.items():
			if not isinstance(k, str) or not isinstance(v, dict):
				continue
			pc = v.get("payload_cid")
			if not isinstance(pc, str) or not pc:
				continue
			try:
				ts = float(v.get("ts", time.time()))
			except Exception:
				ts = time.time()
			ttl_s = v.get("ttl_s")
			try:
				ttl_f = float(ttl_s) if ttl_s is not None else None
			except Exception:
				ttl_f = None
			loaded[k] = IndexEntry(
				payload_cid=pc,
				ts=ts,
				ttl_s=ttl_f,
				operation=v.get("operation"),
				cache_name=v.get("cache_name"),
			)

		with self._lock:
			self._entries.update(loaded)


_global_index: IPFSMutableCacheIndex | None = None
_global_lock = threading.Lock()


def get_global_mutable_index() -> Optional[IPFSMutableCacheIndex]:
	"""Create a process-global mutable index if enabled by env vars."""
	global _global_index
	enable_ipns = _truthy(os.environ.get("IPFS_ACCELERATE_CACHE_IPNS_INDEX") or os.environ.get("IPFS_DATASETS_PY_CACHE_IPNS_INDEX"))
	enable_pubsub = _truthy(
		os.environ.get("IPFS_ACCELERATE_CACHE_PUBSUB_REPLICATION")
		or os.environ.get("IPFS_DATASETS_PY_CACHE_PUBSUB_REPLICATION")
	)
	with _global_lock:
		# If env no longer enables the index, shut down any prior instance.
		if not (enable_ipns or enable_pubsub):
			if _global_index is not None:
				try:
					_global_index.shutdown()
				except Exception:
					pass
				_global_index = None
			return None

		if _global_index is not None:
			return _global_index

		ipns_name = os.environ.get("IPFS_ACCELERATE_CACHE_IPNS_NAME") or os.environ.get("IPFS_DATASETS_PY_CACHE_IPNS_NAME")
		ipns_key = os.environ.get("IPFS_ACCELERATE_CACHE_IPNS_KEY") or os.environ.get("IPFS_DATASETS_PY_CACHE_IPNS_KEY")
		topic = os.environ.get("IPFS_ACCELERATE_CACHE_PUBSUB_TOPIC") or os.environ.get("IPFS_DATASETS_PY_CACHE_PUBSUB_TOPIC") or "ipfs-accelerate-cache-index"

		publish_min = float(os.environ.get("IPFS_ACCELERATE_CACHE_IPNS_PUBLISH_MIN_INTERVAL_S", "30") or "30")
		refresh_min = float(os.environ.get("IPFS_ACCELERATE_CACHE_IPNS_REFRESH_MIN_INTERVAL_S", "30") or "30")

		_global_index = IPFSMutableCacheIndex(
			enable_ipns=enable_ipns,
			enable_pubsub=enable_pubsub,
			ipns_name=ipns_name,
			ipns_key=ipns_key,
			pubsub_topic=topic,
			publish_min_interval_s=publish_min,
			refresh_min_interval_s=refresh_min,
		)
		logger.info("✓ IPFS mutable cache index enabled (ipns=%s, pubsub=%s)", enable_ipns, enable_pubsub)
		return _global_index


def reset_global_mutable_index() -> None:
	"""Force-stop and clear the process-global mutable index (primarily for tests)."""
	global _global_index
	with _global_lock:
		idx = _global_index
		_global_index = None
	if idx is not None:
		try:
			idx.shutdown()
		except Exception:
			pass
