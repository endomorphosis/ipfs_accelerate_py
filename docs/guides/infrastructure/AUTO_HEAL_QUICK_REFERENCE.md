# Auto-Healing Quick Reference

Auto-healing is an opt-in CLI error-reporting and proposal path. It can capture
errors, create GitHub issues, and optionally create draft PR/Copilot proposals.
It does not authorize a merge or bypass deterministic validation, repository
policy, or agent-supervisor assurance gates.

## Enable explicitly

```bash
export IPFS_AUTO_ISSUE=true
export IPFS_AUTO_PR=true
export IPFS_AUTO_HEAL=true
export IPFS_REPO=owner/repo
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

Keep these settings disabled for ordinary local commands unless the target
repository, credentials, permissions, rate limits, and human review process
are ready. Prefer a test repository while validating issue/PR behavior.

## Verify the implementation

```bash
python -m pytest test/test_error_handler.py -q
```

The current implementation and detailed configuration are documented in
[Auto-Healing](../../features/auto-healing/README.md) and
[Auto-Healing Configuration](../../features/auto-healing/AUTO_HEALING_CONFIGURATION.md).
