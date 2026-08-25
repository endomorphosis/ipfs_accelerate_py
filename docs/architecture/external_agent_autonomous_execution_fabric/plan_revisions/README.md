# EAAEF plan revisions

This directory holds admitted plan-revision artifacts for the External Agent
Autonomous Execution Fabric.  Plan R1 is the reviewed source board.  Plan R2 is
created only by the independently signed process-remote three-operation owner
seam after EAAEF-008 promotion: prepare, apply, observe.

Live mutation of the operational task catalog is not a document in this
directory.  The exclusive Quack owner applies closed CAS updates to its private
DuckDB.  DuckLake is downstream immutable history and is never claim, lease,
fence, or merge authority.

Do not edit completed R1 task specifications in place.  Bounded add/supersede
repairs belong to an admitted Plan R2 revision with an exact epoch and fence.
