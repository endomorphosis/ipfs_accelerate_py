# P2P Workflow Discovery

## Overview

The P2P Workflow Discovery service enables automatic detection and execution of GitHub Actions workflows across multiple repositories using the peer-to-peer network. This allows workflows tagged for P2P execution to be discovered and executed without consuming GitHub Actions minutes.

## Key Features

✅ **Cross-Repository Discovery** - Automatically scans all accessible repositories for P2P workflows  
✅ **Tag-Based Detection** - Identifies workflows by parsing YAML for P2P tags  
✅ **Automatic Submission** - Submits discovered workflows to P2P scheduler  
✅ **Continuous Monitoring** - Continuously polls for new P2P workflows  
✅ **Autoscaler Integration** - Works with GitHub runner autoscaler for dynamic provisioning  

## How It Works

### 1. Workflow Tagging

Tag your workflows to indicate they can be executed via P2P:

```yaml
name: P2P Code Generation

on:
  workflow_dispatch:

# Tag this workflow for P2P execution
env:
  WORKFLOW_TAGS: p2p-only,code-generation

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - name: Generate code
        run: ./generate.sh
```

### 2. Tag Types

| Tag | Description | Execution |
|-----|-------------|-----------|
| `p2p-only` | Must execute via P2P network | P2P only, never on GitHub |
| `p2p-eligible` | Can execute on P2P or GitHub | Prefers P2P, falls back to GitHub |
| `github-api` | Standard GitHub workflow | GitHub only |
| `code-generation` | Code generation task | Task type indicator |
| `web-scraping` | Web scraping task | Task type indicator |
| `data-processing` | Data processing task | Task type indicator |
| `unit-test` | Unit test workflow | Task type indicator |

### 3. Tag Formats

The discovery service supports multiple tag formats:

#### Environment Variable Format
```yaml
env:
  WORKFLOW_TAGS: p2p-only,code-generation
```

#### With Quotes
```yaml
env:
  WORKFLOW_TAGS: "p2p-eligible, web-scraping"
```

#### Comment Format
```yaml
# P2P: p2p-only, data-processing
jobs:
  process:
    runs-on: ubuntu-latest
```

#### Alternative Variable Name
```yaml
env:
  P2P_TAGS: p2p-eligible,code-generation
```

## Usage

### CLI Commands

#### Continuous Monitoring

Monitor all accessible repositories continuously:
```bash
ipfs-accelerate github p2p-discover monitor
```

Monitor specific organization:
```bash
ipfs-accelerate github p2p-discover monitor --owner myorg
```

Custom poll interval (default 300 seconds):
```bash
ipfs-accelerate github p2p-discover monitor --owner myorg --interval 180
```

#### One-Time Discovery

Run discovery once and exit:
```bash
ipfs-accelerate github p2p-discover once --owner myorg
```

Get results as JSON:
```bash
ipfs-accelerate github p2p-discover once --owner myorg --output-json
```

### Python API

#### Basic Usage

```python
from ipfs_accelerate_py.p2p_workflow_discovery import P2PWorkflowDiscoveryService

# Create service
service = P2PWorkflowDiscoveryService(
    owner="myorg",
    poll_interval=300
)

# Run once
stats = service.run_discovery_cycle()
print(f"Discovered: {stats['discovered']}")
print(f"Submitted: {stats['submitted']}")

# Or run continuously
service.start()
```

#### With Custom Scheduler

```python
from ipfs_accelerate_py.p2p_workflow_discovery import P2PWorkflowDiscoveryService
from ipfs_accelerate_py.p2p_workflow_scheduler import P2PWorkflowScheduler

# Create custom scheduler
scheduler = P2PWorkflowScheduler(peer_id="my-custom-peer")

# Create service with custom scheduler
service = P2PWorkflowDiscoveryService(
    owner="myorg",
    scheduler=scheduler
)

# Run discovery
service.start()
```

#### Manual Discovery

```python
from ipfs_accelerate_py.p2p_workflow_discovery import P2PWorkflowDiscoveryService

service = P2PWorkflowDiscoveryService(owner="myorg")

# Discover workflows in specific repo
discoveries = service.discover_workflows_in_repo("myorg", "myrepo")

for discovery in discoveries:
    print(f"Found: {discovery.workflow_name}")
    print(f"  Tags: {', '.join(discovery.tags)}")
    
    # Submit to scheduler
    service.submit_workflow_to_scheduler(discovery)
```

## Integration with Autoscaler

The P2P workflow discovery service is automatically integrated with the GitHub runner autoscaler.

### How It Works

1. **Discovery Phase**: Service scans repositories for P2P workflows
2. **Submission Phase**: Discovered workflows are submitted to P2P scheduler
3. **Autoscaling Phase**: Autoscaler provisions runners based on:
   - GitHub workflow queue depth
   - P2P scheduler pending tasks
   - System capacity limits

### Autoscaler Usage

#### With P2P (Default)

```bash
ipfs-accelerate github autoscaler
```

The autoscaler will:
- Monitor GitHub workflow queues
- Run P2P discovery every poll interval
- Allocate runners for both GitHub and P2P workloads

#### Without P2P

```bash
ipfs-accelerate github autoscaler --no-p2p
```

Disables P2P discovery and only monitors GitHub workflows.

### Runner Allocation

The autoscaler intelligently allocates runners:

```
Total Max Runners: 8
P2P Pending Tasks: 3
Runners for P2P: min(3, 8/2) = 3
Runners for GitHub: 8 - 3 = 5
```

Example output:
```
Checking workflow queues...
P2P workflows: 3 pending, 2 assigned
Found 2 repos with 4 workflows
  Running: 2, Failed: 1
Allocating 3 runners for P2P tasks, 5 for GitHub workflows
✓ Generated 5 runner token(s)
P2P Summary: 3 pending, 2 assigned, 3 runners allocated for P2P
```

## Discovery Process

### Step 1: Repository Listing
```
GET /user/repos
→ Returns list of accessible repositories
```

### Step 2: Workflow File Discovery
```
GET /repos/{owner}/{repo}/contents/.github/workflows
→ Returns list of workflow files
```

### Step 3: Content Parsing
```
GET /repos/{owner}/{repo}/contents/.github/workflows/{file}
→ Downloads and parses YAML content
→ Extracts P2P tags
```

### Step 4: Scheduler Submission
```
P2PWorkflowScheduler.submit_task(task)
→ Adds to priority queue
→ Assigns to peer based on merkle clock + hamming distance
```

## Examples

### Example 1: Code Generation Workflow

`.github/workflows/code-gen.yml`:
```yaml
name: P2P Code Generation

on:
  workflow_dispatch:
    inputs:
      module:
        description: 'Module to generate'
        required: true

env:
  WORKFLOW_TAGS: p2p-only,code-generation

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Generate code
        run: |
          ./generate.py ${{ github.event.inputs.module }}
```

### Example 2: Web Scraping Workflow

`.github/workflows/scrape.yml`:
```yaml
name: P2P Web Scraping

on:
  schedule:
    - cron: '0 */6 * * *'

# Can run on P2P or GitHub
env:
  WORKFLOW_TAGS: p2p-eligible,web-scraping

jobs:
  scrape:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Scrape data
        run: python scraper.py
```

### Example 3: Data Processing Workflow

`.github/workflows/process.yml`:
```yaml
name: P2P Data Processing

on:
  push:
    paths:
      - 'data/**'

# P2P: p2p-eligible, data-processing
jobs:
  process:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Process data
        run: ./process.sh
```

## Monitoring and Debugging

### Check Discovery Status

```bash
# Run once to see what's discovered
ipfs-accelerate github p2p-discover once --owner myorg
```

Output:
```
Discovery Results:
  Workflows discovered: 5
  Workflows submitted: 5

Scheduler Status:
  Pending tasks: 5
  Assigned tasks: 2
  Completed tasks: 10
  Known peers: 3
```

### View Scheduler Status

```bash
# Get detailed scheduler status
ipfs-accelerate p2p-workflow status
```

### Check Autoscaler Logs

```bash
# Run autoscaler with debug logging
ipfs-accelerate github autoscaler --owner myorg 2>&1 | tee autoscaler.log
```

Look for lines like:
```
P2P workflows: 3 pending, 2 assigned
Allocating 3 runners for P2P tasks, 5 for GitHub workflows
P2P Summary: 3 pending, 2 assigned, 3 runners allocated for P2P
```

## Configuration

### Service Configuration

```python
service = P2PWorkflowDiscoveryService(
    owner="myorg",           # Organization or user to monitor
    poll_interval=300,       # Seconds between discovery cycles
    scheduler=None           # Optional custom scheduler
)
```

### Autoscaler Configuration

```python
autoscaler = GitHubRunnerAutoscaler(
    owner="myorg",           # Organization or user to monitor
    poll_interval=120,       # Seconds between checks
    since_days=1,            # Monitor repos updated in last N days
    max_runners=8,           # Maximum runners to provision
    filter_by_arch=True,     # Filter workflows by architecture
    enable_p2p=True          # Enable P2P workflow monitoring
)
```

## Performance Considerations

### Rate Limiting

The discovery service:
- Respects GitHub API rate limits
- Uses caching to reduce API calls
- Falls back to GraphQL when REST is rate-limited
- Only scans repositories updated recently (configurable)

### Polling Frequency

Recommended intervals:
- **Continuous monitoring**: 300 seconds (5 minutes)
- **Active development**: 180 seconds (3 minutes)
- **Production**: 600 seconds (10 minutes)

### Resource Usage

Discovery cycle typically:
- Makes 1 API call per repository
- Makes 1 API call per workflow file
- Parses YAML files (minimal CPU)
- Submits to in-memory scheduler (fast)

For 50 repositories with 3 workflows each:
- API calls: 50 + (50 × 3) = 200 calls
- With caching: ~50 calls (workflows cached)
- Time: ~10-30 seconds depending on rate limits

## Troubleshooting

### No Workflows Discovered

**Problem**: Discovery returns 0 workflows.

**Solutions**:
1. Verify workflows have P2P tags
2. Check authentication: `gh auth status`
3. Verify repository access
4. Check workflow file format (must be `.yml` or `.yaml`)

### Workflows Not Executing

**Problem**: Workflows discovered but not executing.

**Solutions**:
1. Check scheduler status: `ipfs-accelerate p2p-workflow status`
2. Verify P2P peers are running
3. Check task assignment: tasks assigned to correct peer?
4. Review autoscaler logs for runner provisioning

### High API Usage

**Problem**: Too many GitHub API calls.

**Solutions**:
1. Increase poll interval (e.g., 600 seconds)
2. Reduce `since_days` to scan fewer repos
3. Use specific owner instead of all accessible repos
4. Enable caching (automatic)

### Discovery Too Slow

**Problem**: Discovery cycle takes too long.

**Solutions**:
1. Reduce number of repositories scanned
2. Use GraphQL API (faster for bulk queries)
3. Increase `since_days` filter to skip old repos
4. Run discovery less frequently

## Best Practices

### 1. Tag Workflows Appropriately

Use `p2p-only` for:
- Long-running tasks that don't need GitHub infrastructure
- Tasks that can run anywhere
- Resource-intensive workflows

Use `p2p-eligible` for:
- Tasks that benefit from P2P but can fall back to GitHub
- Mixed workloads
- Testing P2P execution

Use `github-api` or no tags for:
- Workflows that need GitHub-specific features
- Security-sensitive workflows
- Workflows with dependencies on GitHub infrastructure

### 2. Monitor Regularly

- Run `p2p-discover once` periodically to verify discovery
- Check autoscaler logs for P2P allocation
- Monitor scheduler status for pending tasks

### 3. Start Small

1. Tag 1-2 workflows as `p2p-eligible` (not `p2p-only`)
2. Monitor execution and results
3. Gradually increase P2P usage as confidence grows

### 4. Balance Workload

- Don't tag all workflows as `p2p-only`
- Keep critical workflows on GitHub
- Use `p2p-eligible` for flexible execution

## Security Considerations

### Authentication

- Uses existing GitHub CLI authentication
- No additional credentials required
- Respects repository permissions
- Only accesses repositories user has access to

### Workflow Validation

- Only processes workflows with valid P2P tags
- Filters out unknown tags
- Does not modify workflow files
- Read-only access to repository content

### Data Privacy

- Workflow content is parsed locally
- No external services involved
- P2P scheduler runs in your infrastructure
- No workflow data leaves your control

## Future Enhancements

Planned features:
1. **Dynamic Priority**: Adjust task priority based on workflow age
2. **Workflow Results**: Store and retrieve workflow results
3. **Status Updates**: Report workflow status back to GitHub
4. **Artifact Storage**: Store workflow artifacts in IPFS
5. **Cross-Org Discovery**: Discover workflows across multiple organizations
6. **Smart Caching**: Cache workflow content for faster discovery

## Support

For issues or questions:
- GitHub Issues: https://github.com/endomorphosis/ipfs_accelerate_py/issues
- Documentation: See README.md and P2P_WORKFLOW_SCHEDULER.md
- Tests: Run `python test_p2p_workflow_discovery_simple.py`

## License

This component is part of ipfs_accelerate_py and follows the same license terms.
