# Pre-Job Cleanup Action
# Add this step to the beginning of your workflow jobs to prevent permission errors

This directory contains a reusable composite action for cleaning up runner workspace before checkout.

## Usage

Add this as the FIRST step in your workflow job, before the checkout action:

```yaml
jobs:
  your-job:
    runs-on: [self-hosted, linux, arm64]
    steps:
      - name: Pre-job cleanup
        uses: ./.github/actions/cleanup-workspace
        
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          clean: true
```

## What it does

1. Removes stale git lock files
2. Fixes permissions on .git directory
3. Fixes permissions on .github directory
4. Cleans Python cache files
5. Reports disk usage

## When to use

- Before checkout in self-hosted runner workflows
- When you encounter EACCES permission denied errors
- When workflows fail with "Unable to create index.lock"
- As a preventive measure in all self-hosted runner jobs
