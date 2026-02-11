# Workflow Analysis and Cleanup Plan

## Current Workflow Status

### Active Workflows

1. **amd64-ci.yml** - AMD64 CI/CD Pipeline
   - Runner: ubuntu-latest (GitHub-hosted)
   - Triggers: push (main, develop), pull_request (main), schedule, workflow_dispatch
   - Status: ✅ Working, core tests pass
   - Skipped jobs: Docker builds (only on schedule/manual to save resources)

2. **multiarch-ci.yml** - Multi-Architecture CI/CD Pipeline
   - Runner: ubuntu-latest (GitHub-hosted)
   - Triggers: push (main, develop), pull_request (main), schedule, workflow_dispatch
   - Status: ✅ Working, core tests pass
   - Skipped jobs: Docker builds (only on schedule/manual), GPU tests (manual only), benchmarks (manual only)

3. **arm64-ci.yml** - ARM64 CI/CD Pipeline
   - Runner: self-hosted with ARM64 label
   - Triggers: push (main, develop), pull_request (main), workflow_dispatch
   - Status: ⚠️ Requires self-hosted ARM64 runner (may not be available)
   - Skipped jobs: Full tests (only on main or manual), Docker tests (needs runner)

4. **ci-arm64.yml** - CI/CD Pipeline - ARM64 Test
   - Runner: self-hosted (no specific label)
   - Triggers: push (main, develop), pull_request (main), workflow_dispatch
   - Status: ⚠️ DUPLICATE of arm64-ci.yml, simpler implementation
   - Skipped jobs: May skip if self-hosted runner not available

5. **enhanced-ci-cd.yml** - Enhanced CI/CD Pipeline - Multi-Architecture Support
   - Runner: self-hosted
   - Triggers: push (main, develop, excluding docs), pull_request (main), workflow_dispatch
   - Status: ⚠️ OVERLAPS with multiarch-ci.yml, requires self-hosted runner
   - Skipped jobs: Docker tests (if docker not available), GPU tests (manual only), benchmarks (manual only)

6. **package-test.yml** - Package Installation Test
   - Runner: self-hosted
   - Triggers: push (main, only when requirements/pyproject.toml change), workflow_dispatch
   - Status: ⚠️ Limited scope, may skip if runner not available

7. **test-runner.yml** - Test ARM64 Runner
   - Runner: self-hosted with ARM64 label
   - Triggers: workflow_dispatch, push (main, only when workflow file changes)
   - Status: ⚠️ Basic test, mainly for runner validation

## Issues Identified

### Redundancy Issues

1. **Duplicate ARM64 Workflows**
   - `arm64-ci.yml` and `ci-arm64.yml` both test ARM64 on self-hosted runners
   - `ci-arm64.yml` is simpler and likely the older version
   - **Recommendation**: Delete `ci-arm64.yml`, keep `arm64-ci.yml`

2. **Overlapping Multi-Architecture Support**
   - `multiarch-ci.yml` (GitHub-hosted) and `enhanced-ci-cd.yml` (self-hosted) both test multiple architectures
   - `multiarch-ci.yml` is more comprehensive and doesn't require self-hosted runners
   - **Recommendation**: Delete `enhanced-ci-cd.yml`, keep `multiarch-ci.yml`

3. **Test Runner Validation**
   - `test-runner.yml` is a simple runner test
   - Could be merged into `arm64-ci.yml` or removed if not needed
   - **Recommendation**: Delete `test-runner.yml` if ARM64 self-hosted runner is not consistently available

### Self-Hosted Runner Dependencies

Several workflows require self-hosted runners which may not be available:
- `arm64-ci.yml` - requires ARM64 self-hosted runner
- `ci-arm64.yml` - requires self-hosted runner
- `enhanced-ci-cd.yml` - requires self-hosted runner
- `package-test.yml` - requires self-hosted runner
- `test-runner.yml` - requires ARM64 self-hosted runner

**Issue**: When self-hosted runners are not available, these workflows remain in "queued" state indefinitely.

**Recommendation**: 
- Keep only essential self-hosted workflows if runners are available
- Otherwise, rely on GitHub-hosted runners with QEMU emulation for ARM64 testing

## Recommended Actions

### Priority 1: Remove Redundant Workflows

1. **Delete `ci-arm64.yml`** - Duplicate of `arm64-ci.yml`
2. **Delete `enhanced-ci-cd.yml`** - Overlaps with `multiarch-ci.yml`, requires unavailable self-hosted runner
3. **Delete `test-runner.yml`** - Basic test that's not essential

### Priority 2: Conditional Self-Hosted Workflows

If self-hosted ARM64 runners are not consistently available:
4. **Consider disabling `arm64-ci.yml`** - Move to archived or make optional
5. **Consider disabling `package-test.yml`** - Move to archived or make optional

If self-hosted runners ARE available and needed:
- Keep `arm64-ci.yml` for native ARM64 testing
- Keep `package-test.yml` for package installation validation

### Priority 3: Workflow Optimization

For remaining workflows:
- Keep Docker/GPU/benchmark jobs skipped on PR (resource conservation is good)
- Ensure summary jobs handle skipped dependencies gracefully (already done)

## Final Recommended Workflow Structure

### GitHub-Hosted Runners (Always Run)
- ✅ `amd64-ci.yml` - Native AMD64 testing
- ✅ `multiarch-ci.yml` - Multi-architecture testing with QEMU emulation

### Self-Hosted Runners (Optional, if available)
- ⚠️ `arm64-ci.yml` - Native ARM64 testing (requires ARM64 self-hosted runner)
- ⚠️ `package-test.yml` - Package installation testing (requires self-hosted runner)

### To Delete (Redundant)
- ❌ `ci-arm64.yml` - Duplicate of arm64-ci.yml
- ❌ `enhanced-ci-cd.yml` - Overlaps with multiarch-ci.yml
- ❌ `test-runner.yml` - Not essential

## Implementation Plan

1. Move redundant workflows to archived folder
2. Update documentation to reflect workflow structure
3. Monitor remaining workflows for proper execution
4. Verify that core tests (amd64-ci, multiarch-ci) continue to pass on PRs
