# LPC-171 / LPC-030 worktree setup repair

LPC-030 failed three times at `worktree_setup` with:

`dependency_missing:external/ipfs_datasets`

Cause: validation and predicted-file strings contained
`/home/barberb/lift_coding/external/ipfs_datasets/...`, so the daemon implied
a submodule pin `external/ipfs_datasets` that this accelerate worktree does
not have.

Repair:

- Initialize the existing `ipfs_datasets_py` submodule at authority revision
  `ac82107e246b30e35a2bbdcf75e01370d22350c6`.
- Retarget remaining task paths to `ipfs_datasets_py/`.
- Declare `Submodules: ipfs_datasets_py` on remaining implementation tasks.
- Release LPC-030 from strategy `blocked_tasks` and reset attempt counters.
