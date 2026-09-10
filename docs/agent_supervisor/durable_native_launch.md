# Native launches from fleet repairs

A detached process retains its parent's systemd control group. Ending the
temporary fleet repair service therefore used to terminate supervisors that
the repair had just launched, even when they used a new session.

The SAWM detached launch command now delegates to a separate user scope when
called from an `ipfs-taskboard-repair-job` service. Delegation happens before
configuration admission, credential retirement, or worker creation. The inner
native command performs all its normal source, owner, and credential checks.
Its exit status is returned unchanged; failure to create the scope stops the
launch. The scope remains alive until its processes finish.

Other native detached launchers can call `delegate_repair_service_launch`
before their admission and descriptor setup. Foreground jobs and existing
dedicated board services retain their own lifetime. The fleet repair service
continues to use `KillMode=control-group` to clean up its temporary work.
