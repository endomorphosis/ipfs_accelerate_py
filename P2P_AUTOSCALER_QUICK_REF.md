# P2P Autoscaler Quick Reference

## Quick Start

```bash
# 1. Authenticate with GitHub
gh auth login

# 2. Start autoscaler with P2P monitoring (default)
ipfs-accelerate github autoscaler

# 3. Monitor P2P workflows across repos
ipfs-accelerate github p2p-discover monitor --owner myorg
```

## CLI Commands

### Autoscaler Commands

```bash
# Basic autoscaler (monitors GitHub + P2P)
ipfs-accelerate github autoscaler

# Specific organization
ipfs-accelerate github autoscaler --owner myorg

# Custom settings
ipfs-accelerate github autoscaler \
  --owner myorg \
  --interval 60 \
  --max-runners 8 \
  --since-days 2

# Disable P2P monitoring
ipfs-accelerate github autoscaler --no-p2p
```

### P2P Discovery Commands

```bash
# Continuous monitoring
ipfs-accelerate github p2p-discover monitor
ipfs-accelerate github p2p-discover monitor --owner myorg
ipfs-accelerate github p2p-discover monitor --interval 300

# One-time discovery
ipfs-accelerate github p2p-discover once
ipfs-accelerate github p2p-discover once --owner myorg
ipfs-accelerate github p2p-discover once --output-json
```

## Workflow Tagging

### Simple Format
```yaml
env:
  WORKFLOW_TAGS: p2p-only,code-generation
```

### All Tag Options
```yaml
env:
  # Execution mode (choose one)
  WORKFLOW_TAGS: p2p-only          # Must use P2P
  WORKFLOW_TAGS: p2p-eligible      # Can use P2P or GitHub
  WORKFLOW_TAGS: github-api        # GitHub only (default)
  
  # Task types (add as needed)
  WORKFLOW_TAGS: p2p-only,code-generation
  WORKFLOW_TAGS: p2p-eligible,web-scraping
  WORKFLOW_TAGS: p2p-eligible,data-processing
  WORKFLOW_TAGS: github-api,unit-test
```

### Comment Format
```yaml
# P2P: p2p-eligible, code-generation
jobs:
  generate:
    runs-on: ubuntu-latest
```

## Tag Reference

| Tag | Meaning | When to Use |
|-----|---------|-------------|
| `p2p-only` | Must execute via P2P | Long-running, resource-intensive tasks |
| `p2p-eligible` | P2P or GitHub | Flexible execution preference |
| `github-api` | GitHub only | Needs GitHub infrastructure |
| `code-generation` | Code gen task | Type indicator |
| `web-scraping` | Web scrape task | Type indicator |
| `data-processing` | Data process task | Type indicator |
| `unit-test` | Unit test | Type indicator |

## Python API

### Autoscaler
```python
from github_autoscaler import GitHubRunnerAutoscaler

autoscaler = GitHubRunnerAutoscaler(
    owner="myorg",
    poll_interval=120,
    max_runners=8,
    enable_p2p=True  # Default
)
autoscaler.start()
```

### Discovery Service
```python
from ipfs_accelerate_py.p2p_workflow_discovery import (
    P2PWorkflowDiscoveryService
)

service = P2PWorkflowDiscoveryService(
    owner="myorg",
    poll_interval=300
)

# Run once
stats = service.run_discovery_cycle()

# Or continuously
service.start()
```

## Autoscaler Output

```
--- Check #1 at 2025-11-17 14:30:00 ---
Checking workflow queues...
P2P workflows: 3 pending, 2 assigned
Found 2 repos with 4 workflows
  Running: 2, Failed: 1
Allocating 3 runners for P2P tasks, 5 for GitHub workflows
Provisioning runners...
✓ Generated 5 runner token(s)
  myorg/repo1: 2 workflows
  myorg/repo2: 2 workflows
P2P Summary: 3 pending, 2 assigned, 3 runners allocated for P2P
Sleeping for 120s...
```

## Discovery Output

```bash
$ ipfs-accelerate github p2p-discover once --owner myorg

Discovery Results:
  Workflows discovered: 5
  Workflows submitted: 5

Scheduler Status:
  Pending tasks: 5
  Assigned tasks: 2
  Completed tasks: 10
  Known peers: 3
```

## Common Patterns

### Pattern 1: Code Generation
```yaml
name: Generate Module

on:
  workflow_dispatch:

env:
  WORKFLOW_TAGS: p2p-only,code-generation

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: ./generate.sh
```

### Pattern 2: Scheduled Scraping
```yaml
name: Scrape Data

on:
  schedule:
    - cron: '0 */6 * * *'

env:
  WORKFLOW_TAGS: p2p-eligible,web-scraping

jobs:
  scrape:
    runs-on: ubuntu-latest
    steps:
      - run: python scraper.py
```

### Pattern 3: Data Processing
```yaml
name: Process Data

on:
  push:
    paths: ['data/**']

env:
  WORKFLOW_TAGS: p2p-eligible,data-processing

jobs:
  process:
    runs-on: ubuntu-latest
    steps:
      - run: ./process.sh
```

## Monitoring

### Check Autoscaler Status
```bash
# View real-time logs
ipfs-accelerate github autoscaler --owner myorg 2>&1 | tee autoscaler.log

# Look for:
# - "P2P workflows: X pending, Y assigned"
# - "Allocating X runners for P2P tasks"
# - "P2P Summary: ..."
```

### Check Discovery Status
```bash
# Run once to see current state
ipfs-accelerate github p2p-discover once --owner myorg
```

### Check Scheduler Status
```bash
ipfs-accelerate p2p-workflow status
```

## Troubleshooting

### No Workflows Discovered
```bash
# Check auth
gh auth status

# Run discovery with specific owner
ipfs-accelerate github p2p-discover once --owner myorg

# Verify workflow has tags
cat .github/workflows/myworkflow.yml | grep WORKFLOW_TAGS
```

### Autoscaler Not Allocating for P2P
```bash
# Check if P2P is enabled
ipfs-accelerate github autoscaler  # Should show "P2P workflow monitoring: enabled"

# If disabled, don't use --no-p2p flag
ipfs-accelerate github autoscaler --owner myorg  # Remove --no-p2p
```

### Discovery Not Finding Workflows
```bash
# Verify tags are correct
ipfs-accelerate github p2p-discover once --owner myorg --output-json

# Valid tags: p2p-only, p2p-eligible, code-generation, web-scraping, data-processing
```

## Configuration Reference

### Autoscaler Parameters
```
--owner          Organization/user to monitor
--interval       Poll interval in seconds (default: 120)
--since-days     Monitor repos from last N days (default: 1)
--max-runners    Maximum runners to provision (default: system cores)
--no-p2p         Disable P2P monitoring
--no-arch-filter Disable architecture filtering
```

### Discovery Parameters
```
--owner      Organization/user to monitor
--interval   Poll interval in seconds (default: 300)
--output-json Output as JSON
```

## Performance Tips

1. **Reduce API Calls**: Use longer poll intervals (300-600s)
2. **Limit Scope**: Use `--owner` to monitor specific org
3. **Adjust Window**: Use `--since-days` to skip old repos
4. **Balance Load**: Don't tag everything as `p2p-only`

## Security Notes

- Uses existing GitHub authentication (`gh auth login`)
- Read-only access to workflow files
- No modification of repository content
- Respects repository permissions
- P2P execution in your infrastructure

## Best Practices

1. **Start small**: Tag 1-2 workflows as `p2p-eligible` first
2. **Monitor closely**: Watch autoscaler logs for allocation
3. **Use appropriate tags**: Reserve `p2p-only` for truly P2P-suitable tasks
4. **Balance workload**: Keep critical workflows on GitHub
5. **Test thoroughly**: Verify P2P execution works before scaling up

## Examples

### Full Setup
```bash
# 1. Start autoscaler with P2P
ipfs-accelerate github autoscaler --owner myorg --interval 60 &

# 2. Start dedicated discovery monitor (optional)
ipfs-accelerate github p2p-discover monitor --owner myorg --interval 300 &

# 3. Monitor logs
tail -f autoscaler.log
```

### Check System Status
```bash
# Autoscaler status
ps aux | grep autoscaler

# Discovery status
ipfs-accelerate github p2p-discover once --owner myorg

# Scheduler status
ipfs-accelerate p2p-workflow status

# Runner status
ipfs-accelerate github runners list --org myorg
```

## Related Documentation

- **Full Guide**: See `P2P_WORKFLOW_DISCOVERY.md`
- **P2P Scheduler**: See `P2P_WORKFLOW_SCHEDULER.md`
- **Autoscaler**: See `AUTOSCALER.md`
- **GitHub Cache**: See `GITHUB_CACHE_QUICK_REF.md`

## Support

- GitHub Issues: https://github.com/endomorphosis/ipfs_accelerate_py/issues
- Tests: `python test_p2p_workflow_discovery_simple.py`
