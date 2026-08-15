# LPC-172 / LPC-020 worktree setup repair

LPC-020 failed three times at `worktree_setup` with:

`dependency_missing:external/ipfs_datasets`

Same root cause and repair as LPC-171. LPC-020 is released to implement
against `ipfs_datasets_py` at `ac82107e246b30e35a2bbdcf75e01370d22350c6`.
