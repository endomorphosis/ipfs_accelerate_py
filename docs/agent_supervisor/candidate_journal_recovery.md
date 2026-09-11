# Candidate cleanup journal recovery

Future nonpooled protected Codex proposal rejections can recover an interrupted
native lifecycle deletion without invoking the provider again. This follows the
native deletion component described in [candidate_lifecycle_delete_journal.md](candidate_lifecycle_delete_journal.md).
Legacy [candidate rejection closure](candidate_rejection_closure.md) receipts and
ordinary validation, consumed-attempt and deferral schemas keep their existing
meaning.

The producer records a new terminal schema only after independently verified
protected provider cleanup, actual rejected-proposal validation, a retained Git
rescue commit, and successful nonpooled worktree and branch cleanup. The native
prepared deletion binds that terminal receipt to the exact original lifecycle
record, task index, lease, fence, repository and native store identities. Retained
original inodes and a committed native receipt establish removal; missing rows
alone never do.

The Bridge's ordinary verifier only reads. It verifies the original
started/proposal/terminal event prefix, current signed provider cleanup CAS,
unchanged attempt projection and binding, retained rescue ref, and the canonical
native journal. A distinct recovered closure records these actual observations.
It does not invent historical finish, preservation or post-delete events. Later
real producer events must agree and do not change this prefix-based receipt.

Only the actual bound outer daemon's admitted attempt path can call mutating
resume. That path verifies the original unknown callback identities, repeats
execution and typed-attempt admission, protects the exact accepted claim and
renews its lease around replay. Resume can finish an existing prepared native
operation; it cannot start a new operation from old terminal metadata. Even a
visible committed journal is re-synced on this mutating path before a new callback
CAS. Publication uncertainty retains custody and exact callback CAS replay avoids
a second provider invocation. A negative-only event classification selects this
retention path for future handoffs. If that classification itself is unavailable,
it retains custody instead of assuming a legacy failure policy. It grants no
mutation or completion authority.

This is not acceptance, queue publication, a new effect grant or expiry-based
claim takeover. A missing prepared receipt, replaced inode, changed binding,
unavailable cleanup proof, failed heartbeat or expired claim still denies
recovery. Old unknown callbacks, pooled checkout custody and old absent lifecycle
rows are not migrated. The existing journal retention and same-filesystem limits
still apply. A new daemon must possess the ordinary exact attempt admission; a
crash does not supply that admission.

Tests exercise actual owner-issued typed grants, task admission, callback CAS,
Git rescue, native lifecycle files and interruption/replay. The existing fixture
uses doubles for protected Docker absence, accepted source capsule admission and
TCP Quack transport; the typed gateway and durable stores are real. The public
supervisor tick is covered using the actual bound Bridge maintenance callback.
