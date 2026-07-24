# GitHub Integration Guides

These guides cover the optional GitHub CLI, workflow inspection, self-hosted
runners, autoscaling, and GitHub-backed cache integrations. They are
operational runbooks; GitHub support is not required for the core Python
package.

## Current CLI

The unified CLI is exposed as `ipfs-accelerate`. Check authentication before
using GitHub-backed operations:

```bash
ipfs-accelerate github auth
ipfs-accelerate github repos --limit 10
ipfs-accelerate github workflows owner/repo --limit 20
ipfs-accelerate github queues --owner owner --since-days 1
ipfs-accelerate github runners list --org owner
ipfs-accelerate github autoscaler --owner owner --interval 60
```

The commands require the optional GitHub integration dependencies and suitable
credentials. Prefer environment variables or the `gh` credential store over
putting tokens in command history. Provisioning runners or starting the
autoscaler can create external resources, so review limits and permissions
before enabling them.

## Authentication

- [GitHub Auth Setup](GITHUB_AUTH_SETUP.md)
- [GitHub Token Setup](GITHUB_TOKEN_SETUP.md)
- [GH CLI Auth Fix](GH_CLI_AUTH_FIX.md)

## Workflows, Runners, And Autoscaling

- [GitHub Actions Infrastructure](GITHUB_ACTIONS_INFRASTRUCTURE.md)
- [GitHub Runner Installation](GITHUB_RUNNER_INSTALLATION.md)
- [GitHub Autoscaler](GITHUB_AUTOSCALER_README.md)
- [Self-Hosted Runner Setup](SELF_HOSTED_RUNNER_SETUP.md)
- [Containerized CI Security](CONTAINERIZED_CI_SECURITY.md)

## Caching And P2P

- [GitHub API Cache](GITHUB_API_CACHE.md)
- [GitHub Cache Comprehensive](GITHUB_CACHE_COMPREHENSIVE.md)
- [GitHub Cache Quick Reference](GITHUB_CACHE_QUICK_REF.md)
- [GitHub Actions P2P Setup](GITHUB_ACTIONS_P2P_SETUP.md)
- [GitHub P2P Cache Two-Laptop Runbook](GITHUB_P2P_CACHE_TWO_LAPTOP_RUNBOOK.md)
- [GitHub CLI MCP Integration](GITHUB_CLI_MCP_INTEGRATION.md)

## Related Runtime Guides

- [MCP Dashboard](../../MCP_DASHBOARD_GUIDE.md)
- [Infrastructure Guides](../infrastructure/README.md)
- [Main Documentation](../../README.md)

Workflow-specific pages may describe a historical incident or a deployment
variant. Confirm their commands against `ipfs-accelerate github --help` and the
current repository permissions before reusing them.
