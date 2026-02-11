# CI/CD Troubleshooting Guide

This document provides guidance on troubleshooting common CI/CD issues in the IPFS Accelerate project.

## Common GitHub Actions Issues

### Missing Download Info for Actions

**Error Message:**
```
Error: Missing download info for [action-name]@[version]
```

**Cause:**
This error typically occurs when:
1. The action version is outdated or deprecated
2. There's a connectivity issue to GitHub's action registry
3. The action might be temporarily unavailable

**Solution:**
- Update the action to the latest version, typically `v4` as of March 2025
- All core GitHub Actions should use v4:
  - `actions/checkout@v4`
  - `actions/setup-python@v4`
  - `actions/setup-node@v3` (or newer)
  - `actions/upload-artifact@v4`
  - `peaceiris/actions-gh-pages@v4`

**Example Fix:**
```yaml
# Before
- uses: actions/upload-artifact@v3

# After
- uses: actions/upload-artifact@v4
```

### Runner Environment Issues

**Error Message:**
```
Current runner version: '[version]'
Operating System
Runner Image
Runner Image Provisioner
```

**Cause:**
This error may indicate an issue with the GitHub-hosted runner environment.

**Solution:**
- Ensure your workflow is compatible with the runner image
- Use more specific runner versions if needed
- Consider using container-based workflows for consistent environments

### Action Permissions Issues

**Error Message:**
```
GITHUB_TOKEN Permissions
Secret source: Actions
```

**Cause:**
The workflow doesn't have the necessary permissions to perform the requested action.

**Solution:**
- Add explicit permissions to your workflow:
```yaml
permissions:
  contents: read
  pages: write
  id-token: write
```

## Hardware Compatibility Testing Issues

### Hardware Detection Failures

**Error Message:**
```
Error detecting hardware capabilities
```

**Cause:**
The workflow cannot properly detect the hardware capabilities of the runner.

**Solution:**
- Ensure the hardware detection scripts are working correctly
- Add more logging to identify the specific issue
- Consider adding fallback mechanisms for hardware detection

### Test Result Upload Failures

**Error Message:**
```
Error: Unable to upload artifact
```

**Cause:**
The workflow is unable to upload test results as artifacts.

**Solution:**
- Ensure the file paths for artifacts are correct
- Check that the artifacts are being generated correctly
- Verify that the upload-artifact action version is current

## Distributed Testing Issues

### Worker Communication Failures

**Error Message:**
```
Error: Worker communication timeout
```

**Cause:**
The distributed testing framework cannot communicate with worker nodes.

**Solution:**
- Check network connectivity between coordinator and workers
- Ensure all required ports are open
- Verify that worker nodes are running and properly configured

### Task Distribution Failures

**Error Message:**
```
Error: Failed to distribute tasks
```

**Cause:**
The coordinator cannot distribute tasks to workers.

**Solution:**
- Check task queue implementation
- Ensure workers are registering correctly
- Verify task serialization/deserialization logic

## Best Practices

1. **Use Template Files**: Reference the up-to-date template file at `docs/github-actions-template-2025.yml` for correct action versions.

2. **Regular Updates**: Periodically review and update GitHub Actions versions.

3. **Testing Workflows**: Test workflow changes in a branch before merging to main.

4. **Artifacts Management**: Be mindful of artifact sizes and retention periods.

5. **Environment Variables**: Use environment variables for configuration rather than hardcoding values.

6. **Conditional Execution**: Use conditional execution to skip unnecessary steps.

7. **Error Handling**: Implement proper error handling in workflow scripts.

8. **Documentation**: Keep CI/CD documentation up-to-date.

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Runner Documentation](https://docs.github.com/en/actions/using-github-hosted-runners)
- [Distributed Testing Design](../distributed_testing/DISTRIBUTED_TESTING_DESIGN.md)
- [Hardware Compatibility Testing](../skills/HARDWARE_COMPATIBILITY_README.md)