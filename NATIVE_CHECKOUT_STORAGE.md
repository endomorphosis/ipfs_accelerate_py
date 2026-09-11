# Native checkout storage admission

Cold `WorktreePool` creation, ordinary implementation worktrees and newly
created local submodule worktrees now require storage admission before their
checkout mutations. Configured dependency paths, including explicit nested
paths, are resolved through the bound parent gitlinks and locally available
Git objects. Missing objects or estimates produce a typed retry before provider
dispatch; they cannot fall through to an unmeasured fetch or alternate revision.

The estimator uses immutable commits and `git ls-tree -r -l -z` for exact stored
blob sizes and file counts. Git's own `check-attr --source=<commit>` includes the
committed, global and info attributes. Unbounded filters, custom encodings and
executable post-checkout hooks refuse rather than being silently disabled.
CRLF and `ident` expansion receive conservative allowances. Git versions without
the exact-tree attribute query refuse the estimate.

Available-to-the-user bytes and inodes are sampled through opened directory
descriptors on both destination and source Git-store filesystems. Required
capacity includes rounded checkout blocks, directories, administrative files,
a Git metadata allowance (8 MiB plus 512 bytes per file, with inode allowance)
and operational headroom. Requirements sharing a device are summed; headroom
is added once per device. Unknown measurements refuse.

The default operational headroom is 8 GiB and 50,000 inodes. It can be supplied
as `CheckoutStoragePolicy`, through `WorktreePool(storage_policy=...)` or
`PortalImplementationDaemon(worktree_storage_policy=...)`. The environment
settings `IPFS_ACCELERATE_WORKTREE_MIN_FREE_BYTES` and
`IPFS_ACCELERATE_WORKTREE_MIN_FREE_INODES` accept nonnegative decimal integers.
Zero explicitly disables its individual floor, not the estimate/measurement.

One fixed per-user kernel flock at
`/tmp/ipfs-agent-supervisor-allocation-<uid>/allocation.lock` serializes cooperating
native allocators. The directory and inode ownership, permissions, links and
identity are checked. The persistent inode is never unlinked. A reentrant thread
lock permits synchronous nested preparation; fork children close only their
inherited descriptor copies and acquire independently. Contention has a default
five-second deadline and returns a typed retry. This host-wide exclusion avoids
cross-filesystem lock-order deadlocks. Existing warm lease/renewal behavior and
all lifecycle, callback, cleanup and branch ownership gates remain in force.

The exclusion remains held through parent checkout and dependency preparation.
Additional recursively selected submodules are admitted at their own allocation
boundaries under that same exclusion. New branches consume the exact estimated
commit. Restoring an existing branch holds Git's native verify/prepare reference
lock, uses `worktree add --no-checkout`, and materializes the admitted commit with
`read-tree --reset -u`. This preserves Git's branch-in-use refusal. The verification
transaction changes no reference value and aborts on EOF in `finally`, including
checkout/read-tree errors. Partial worktrees remain subject to existing native
lifecycle cleanup; releasing a storage or Git reference lock does not certify
that a task or callback completed.

This is physical admission for cooperating processes using this helper and the
same host `/tmp` namespace. It is not a reservation ledger, task/coordination
authority, completion evidence, filesystem quota or guarantee against arbitrary
shell writers, future test output or unknown preparation callback allocations.
Custom preparation must separately budget extra output. Unavailable dependency
objects require a separately bounded fetch/repair before checkout can proceed.
No reclamation, pruning, deletion of unrelated work, automatic maintenance or
live deployment is introduced by this change.
