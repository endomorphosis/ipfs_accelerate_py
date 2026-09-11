# Repair launch storage admission

The fleet repair worker checks available bytes and inodes before its native
probe, after that probe, and immediately before spawning the coding process.
It samples the actual repair checkout, watchdog state directory, job directory
and temporary directory through opened directory descriptors. Each unavailable
measurement defers the launch. Future state/job directories use their nearest
existing ancestor; an unavailable checkout or temporary directory refuses.

The optional `repair_worker` settings are:

```json
{
  "minimum_disk_available_bytes": 8589934592,
  "minimum_disk_available_inodes": 50000,
  "temporary_directory": "/tmp"
}
```

The byte and inode defaults are 8 GiB and 50,000. Overrides must be nonnegative
JSON integers; booleans, fractions, strings, null and negatives defer the launch
as invalid policy. Zero explicitly disables that individual floor, while the
measurement must still succeed. Equality with the configured floor is admitted.
These are available-to-the-current-user measurements (`f_bavail`, `f_favail`),
not capacity reserved for root and not a percentage of a large disk.

Without `temporary_directory`, the worker chooses its first nonempty `TMPDIR`,
`TEMP` or `TMP`, falling back to `/tmp`. The result must be an existing absolute
directory. The same path is passed to the systemd child as all three variables,
so the user manager's environment cannot select an unmeasured temporary root.
Tools that explicitly choose another destination still need their own check.

Storage refusals reuse the existing queued launch retry, preserving the selected
attempt count. A refusal after claiming the slot refunds only that exact claim
when no child was started; it cannot change a superseding job. The job's
`last_launch_failure.storage` records the affected role/path, measured directory
and device, available bytes/inodes, requested floors or observation error.
Existing running repair units remain owned and are not stopped for disk pressure.
No cleanup, deletion or pruning is performed.

This is admission, not a reservation. Another process can consume space after
the sample, and arbitrary later clones, archives or test output can exceed the
headroom. Repair instructions therefore require an allocation estimate and
fresh headroom check before substantial work. Strict protection of native
checkout allocation requires a separate bounded allocation mechanism; the
runtime ResourceScheduler's local reservations do not coordinate independent
supervisors or arbitrary shell commands. This change does not claim otherwise.

If a filesystem is already entirely exhausted, persisting even the small retry
record can fail. The default floor is intended to defer new work while space
remains; it is not authority to reclaim another task's files or state.
