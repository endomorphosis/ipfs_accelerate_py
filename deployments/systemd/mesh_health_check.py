#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from typing import Any


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return int(proc.returncode), str(proc.stdout or "").strip(), str(proc.stderr or "").strip()


def _service_state(unit: str) -> dict[str, Any]:
    rc, out, err = _run(["systemctl", "is-active", unit])
    state = out if out else ("inactive" if rc != 0 else "unknown")
    return {"unit": unit, "active": state == "active", "state": state, "error": err or None}


def _service_env(unit: str) -> dict[str, str]:
    rc, out, _err = _run(["systemctl", "show", "-p", "Environment", unit])
    if rc != 0 or not out.startswith("Environment="):
        return {}
    raw = out.split("=", 1)[1].strip()
    if not raw:
        return {}

    env: dict[str, str] = {}
    for item in raw.split(" "):
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"')
        if key:
            env[key] = val
    return env


def _expand_aliases(values: list[str]) -> list[str]:
    alias_groups: list[set[str]] = [
        {"embedding", "embeddings", "text-embedding", "text_embedding", "text_embeddings"},
        {"text2text-generation", "text2text_generation", "text2text"},
        {"text-classification", "text_classification"},
        {"hf.pipeline", "hf_pipeline"},
        {"llm.generate", "llm_generate"},
        {"tool.call", "tool"},
    ]

    base = [str(x).strip() for x in (values or []) if str(x).strip()]
    out = set(base)
    for group in alias_groups:
        if out.intersection(group):
            out.update(group)

    return [x for x in base if x in out] + [x for x in sorted(out) if x not in set(base)]


def _parse_task_types(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return _expand_aliases(parts)


def _collect_effective_task_types(units: list[str]) -> tuple[list[str], str]:
    for unit in units:
        env = _service_env(unit)
        if "IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES" in env:
            return _parse_task_types(env["IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES"]), f"{unit}:Environment"

    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES")
    if raw:
        return _parse_task_types(raw), "process-env"

    return [], "none"


def _truthy(text: str | None) -> bool:
    return str(text or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check task mesh health and remote queue drain readiness.")
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--probe-claim", action="store_true", help="Try claiming one remote task and release it immediately.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    units = [
        "ipfs-accelerate.service",
        "ipfs-accelerate-mcp.service",
        "ipfs-accelerate-task-worker.service",
    ]

    services = [_service_state(u) for u in units]
    task_types, task_types_source = _collect_effective_task_types(units)

    embedding_aliases = {"embedding", "embeddings", "text-embedding", "text_embedding", "text_embeddings"}
    supports_embedding = bool(set(task_types).intersection(embedding_aliases))

    summary: dict[str, Any] = {
        "timestamp": time.time(),
        "host": socket.gethostname(),
        "services": services,
        "task_types_source": task_types_source,
        "task_types": task_types,
        "supports_embedding_aliases": supports_embedding,
        "embedding_aliases": sorted(embedding_aliases),
        "peer_discovery": {"ok": False, "error": None, "count": 0, "peers": []},
        "remote_queues": {"total_queued": 0, "embedding_queued": 0, "by_peer": []},
        "probe": None,
    }

    try:
        from ipfs_accelerate_py.p2p_tasks.client import (
            RemoteQueue,
            claim_many_sync,
            discover_peers_via_mdns_sync,
            release_task_sync,
            request_status_sync,
        )

        discovered = discover_peers_via_mdns_sync(
            timeout_s=float(args.timeout_s),
            limit=max(1, int(args.limit)),
            exclude_self=True,
        )

        peers = []
        total_queued = 0
        total_embedding = 0

        for rq in list(discovered or []):
            pid = str(getattr(rq, "peer_id", "") or "").strip()
            ma = str(getattr(rq, "multiaddr", "") or "").strip()
            if not pid or not ma:
                continue

            row: dict[str, Any] = {"peer_id": pid, "multiaddr": ma}
            try:
                status = request_status_sync(remote=rq, timeout_s=float(args.timeout_s), detail=True)
                row["status_ok"] = bool(isinstance(status, dict) and status.get("ok"))
                row["session"] = str((status or {}).get("session") or "")
                queued = int((status or {}).get("queued") or 0)
                row["queued"] = queued
                qbt = (status or {}).get("queued_by_type")
                row["queued_by_type"] = qbt if isinstance(qbt, dict) else {}

                emb_n = 0
                if isinstance(qbt, dict):
                    for key, val in qbt.items():
                        if str(key).strip() in embedding_aliases:
                            try:
                                emb_n += int(val)
                            except Exception:
                                continue
                row["embedding_queued"] = emb_n

                total_queued += max(0, queued)
                total_embedding += max(0, emb_n)
            except Exception as exc:
                row["status_ok"] = False
                row["error"] = str(exc)
                row["queued"] = 0
                row["embedding_queued"] = 0
                row["queued_by_type"] = {}

            peers.append(row)

        summary["peer_discovery"] = {
            "ok": True,
            "error": None,
            "count": len(peers),
            "peers": [{"peer_id": p["peer_id"], "multiaddr": p["multiaddr"]} for p in peers],
        }
        summary["remote_queues"] = {
            "total_queued": int(total_queued),
            "embedding_queued": int(total_embedding),
            "by_peer": peers,
        }

        if args.probe_claim:
            probe_worker = f"mesh-health-{socket.gethostname()}"
            probe = {"attempted": True, "claims": [], "errors": []}

            for row in peers:
                try:
                    rq = RemoteQueue(peer_id=str(row["peer_id"]), multiaddr=str(row["multiaddr"]))
                    claimed = claim_many_sync(
                        remote=rq,
                        worker_id=probe_worker,
                        supported_task_types=sorted(list(embedding_aliases)),
                        max_tasks=1,
                        same_task_type=False,
                        session_id=None,
                        peer_id=probe_worker,
                        clock=None,
                    )
                    task_ids = [str(t.get("task_id") or "") for t in (claimed or []) if isinstance(t, dict)]
                    released: list[str] = []
                    for tid in task_ids:
                        if not tid:
                            continue
                        try:
                            release_task_sync(
                                remote=rq,
                                task_id=tid,
                                worker_id=probe_worker,
                                reason="mesh-health-probe",
                            )
                            released.append(tid)
                        except Exception as rel_exc:
                            probe["errors"].append(
                                f"release failed for {row['peer_id']}:{tid}: {rel_exc}"
                            )

                    probe["claims"].append(
                        {
                            "peer_id": row["peer_id"],
                            "claimed_task_ids": task_ids,
                            "released_task_ids": released,
                        }
                    )
                except Exception as exc:
                    probe["errors"].append(f"claim failed for {row['peer_id']}: {exc}")

            summary["probe"] = probe

    except Exception as exc:
        summary["peer_discovery"] = {
            "ok": False,
            "error": str(exc),
            "count": 0,
            "peers": [],
        }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print("Mesh Health Check")
    print(f"host: {summary['host']}")
    print("services:")
    for svc in summary["services"]:
        print(f"  - {svc['unit']}: {svc['state']}")

    print(f"task_types_source: {summary['task_types_source']}")
    print(f"supports_embedding_aliases: {summary['supports_embedding_aliases']}")
    print(f"task_types: {', '.join(summary['task_types']) if summary['task_types'] else '(none)'}")

    pd = summary["peer_discovery"]
    print(f"peer_discovery: {'ok' if pd['ok'] else 'failed'} (count={pd['count']})")
    if not pd["ok"]:
        print(f"  error: {pd['error']}")

    rq = summary["remote_queues"]
    print(f"remote_queues: total={rq['total_queued']} embedding={rq['embedding_queued']}")
    for row in rq["by_peer"]:
        print(
            f"  - {row['peer_id']}: queued={row.get('queued', 0)} embedding={row.get('embedding_queued', 0)}"
        )

    if summary.get("probe"):
        print("probe:")
        probe = summary["probe"]
        for c in probe.get("claims", []):
            print(
                f"  - {c['peer_id']}: claimed={len(c.get('claimed_task_ids', []))} released={len(c.get('released_task_ids', []))}"
            )
        for err in probe.get("errors", []):
            print(f"  ! {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
