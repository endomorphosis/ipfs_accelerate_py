# Recovering retained verification waits

The `ipfs_accelerate_py.agent_supervisor.rescue.verification_deferral_recovery`
module handles a protected-path verification timeout whose retained snapshot
subsequently blocks provider-free Portal projection retries. It never completes
a task or changes historical execution attempts.

The native owner must be stopped with an independently verified dead process
birth. Under normal exclusive DuckDB connections, recovery checks the exact
blocked receipt, latest attempt, and every unconsumed retry-budget entry. It
rejects committed provider/effect phases and manual tasks. The historical Portal
event chain and semantic task binding must verify. The normal crash-fence
reconciler qualifies protected content under a maintenance lease, and both
workspaces must still match their protected baseline. Candidate files remain in
their original workspace.

A canonical CAS records the previous receipt, snapshot and event identities,
and original execution-route binding. Each historical snapshot can rearm only
one exhausted attempt. A fresh native claim then runs all ordinary acceptance
gates. Held boards, active owners, foreign outcomes and changed protected paths
remain untouched.

For a stopped board with a fleet inventory:

```sh
python3 -m ipfs_accelerate_py.agent_supervisor.rescue.verification_deferral_recovery \
  --inventory ~/.config/ipfs-taskboard-watchdog/inventory.json --board spar --apply
```

Inventory mode discovers `lane-*/*_database_execution.duckdb` sidecars. Other
filename conventions use the individual-path arguments shown by `--help`.
Omitting `--apply` is read-only. A still-active snapshot requires explicit
`--apply` before its native reconciliation or any recovery evidence is written.

A systemd `ExecStartPre` hook can run inventory mode before the native owner
starts. Prefix the command with `-` so inapplicable recovery does not prevent
ordinary startup. This command never starts a model server.
