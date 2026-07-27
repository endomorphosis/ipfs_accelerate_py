# agent_supervisor.merge

Merge queue, train, resolver, checkout lock, and lane lease surfaces
(`ASREF-G070` / bundle `asref/merge`).

## Purpose

`merge` owns git checkout locking, merge queue/train/checkpoint, conflict
repair, leased lanes, and the merge resolver console entry point.

This package scaffold was introduced by the ASREF-017 validation
retry-budget repair after ASREF-011 exhausted three attempts with **no
repository changes** (`declared validation plan requires changed paths`).

## Public modules (move_map ownership)

| Source (flat) | Target |
|---|---|
| `checkout_lock.py` | `merge/checkout_lock.py` |
| `git_gc.py` | `merge/git_gc.py` |
| `lease_coordination.py` | `merge/lease_coordination.py` |
| `leased_lane.py` | `merge/leased_lane.py` |
| `merge_checkpoint.py` | `merge/merge_checkpoint.py` |
| `merge_conflict_repair.py` | `merge/merge_conflict_repair.py` |
| `merge_queue.py` | `merge/merge_queue.py` |
| `merge_resolver.py` | `merge/merge_resolver.py` |
| `merge_train.py` | `merge/merge_train.py` |

Inventory source: `docs/architecture/asref/move_map.json`.

## Entry points (post-move)

| Console script | Target |
|---|---|
| `ipfs-accelerate-agent-merge-resolver` | `merge.merge_resolver:main` |

## Forbidden dependencies

`merge` **must not** import `todo_daemon`, `self_improvement`, or optional
integration surfaces.

## Import policy

After each module moves into `merge/`, update all callers in the same
change. Do not leave thin re-export stubs at former flat paths. Prefer::

    from ipfs_accelerate_py.agent_supervisor.merge.<module> import ...

## Status

- Package scaffold (`__init__.py`, this README): present
- Module `git mv` + caller rewrites + entry-point retargets: owned by ASREF-011
  once released from strategy `blocked_tasks` after this repair completes
