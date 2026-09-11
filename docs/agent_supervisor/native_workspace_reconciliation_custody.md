# Native workspace reconciliation custody

Whole-root quarantine also covers the older native supervisor's direct dirty
rescue and repository-wide backlog reconciliation. The direct rescue holds a
shared custody lock through branch creation, staging and commit. A pending freeze
waits for that operation to finish; once installed, it prevents another rescue
from changing the retained branch, index or files. Independent workspaces in the
declared fresh root keep their existing rescue behavior.

Repository-wide backlog reconciliation defers while any shared or ancestor Git
store has retained workspace custody. The older native daemon's peer cleanup
remains disabled because it has no canonical peer-completion proof.

This follow-up is a filesystem-owner prerequisite for the first native freeze.
It does not supply task admission, callback settlement, deployment or permission
to retry a retained provider attempt. Every affected running process must load
the compatible guards before a freeze is installed.
