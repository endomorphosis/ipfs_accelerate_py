# PCPR-057 declared branch and release gates

These files are the Accelerate *declared* default-branch protection
policy and release gate for proof-carrying-platform-0.1.0. They are not
live GitHub branch protection, not live tag protection, not a live
required-status configuration, not a freeze (PCPR-002), not a closed
PCPR release (PCPR-093/094), and not a published wheel or sdist.

- `cpython312/release.branch-protection.json` requires protecting
  `main`, pull-request reviews, current-head and release status checks,
  no force-push, no deletion, administrator enforcement, and signed
  commits. Exact commit and tree are bound by the PCPR-057 receipt
  `current_tree_binding` because nested admission rewrites HEAD.
  `origin/main` is not the release identity.
- `cpython312/release.gate.json` requires every named status check to
  pass and prohibits publishing a release after a partial required-build
  failure. Live GitHub enforcement stays typed unavailable.
- `platform-catalog.json` binds Datasets and Kit gate-binding documents
  when those sibling trees are present. Sibling source is never required.
- Missing repository-admin permission is an explicit operator-blocking
  task. The governance gate is not complete.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
Provider-local `~/.local` tools and provider `GITHUB_TOKEN` values are
not sealed-environment authority.
