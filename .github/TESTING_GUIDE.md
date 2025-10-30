# Auto-Heal System Testing Guide

This guide helps you test the auto-healing system to ensure it works correctly.

## 🧪 Quick Test

### Method 1: Use the Test Workflow (Recommended)

1. **Navigate to Actions Tab**
   ```
   https://github.com/YOUR_USERNAME/ipfs_accelerate_py/actions
   ```

2. **Select "Test Auto-Heal System" workflow**

3. **Click "Run workflow"**

4. **Choose a failure type:**
   - `dependency_error` - Simulates missing Python module
   - `syntax_error` - Simulates Python syntax error
   - `timeout_error` - Simulates workflow timeout
   - `resource_error` - Simulates disk space issue
   - `docker_error` - Simulates Docker build failure
   - `test_failure` - Simulates test failure

5. **Watch the Auto-Heal System Activate!**

### Expected Results

After running the test workflow:

1. ✅ Test workflow fails as expected
2. ✅ Auto-Heal workflow triggers automatically
3. ✅ Issue is created with failure analysis
4. ✅ Auto-heal branch is created
5. ✅ GitHub Copilot is notified via issue comment
6. ✅ Copilot analyzes and creates a fix (may take a few minutes)
7. ✅ Pull request is created with the fix

### Verification Checklist

- [ ] Test workflow runs and fails
- [ ] Auto-heal workflow triggers within 1 minute
- [ ] Issue is created with label `auto-heal`
- [ ] Issue contains failure analysis
- [ ] Branch `auto-heal/workflow-XXX-TIMESTAMP` is created
- [ ] Copilot comment is posted to the issue
- [ ] Artifacts are uploaded to Actions tab

## 🔍 Detailed Testing

### Test Each Failure Type

#### Test 1: Dependency Error

**Expected Behavior:**
- Category: DEPENDENCY
- Confidence: 95%
- Suggested Fix: Add missing module to requirements.txt

**Copilot Should:**
- Identify the missing module name
- Add it to requirements.txt
- Create PR with the fix

**How to Verify:**
```bash
# Check the created issue
gh issue list --label auto-heal

# Check the auto-heal branch
git fetch origin
git branch -r | grep auto-heal

# View the failure analysis artifact
gh run download <run-id> --name workflow-failure-analysis-<id>
cat failure_analysis.json
```

#### Test 2: Syntax Error

**Expected Behavior:**
- Category: SYNTAX
- Confidence: 95%
- Suggested Fix: Fix the syntax error in the indicated file

**Copilot Should:**
- Identify the file with syntax error
- Fix the syntax
- Create PR with the fix

#### Test 3: Timeout Error

**Expected Behavior:**
- Category: TIMEOUT
- Confidence: 90%
- Suggested Fix: Increase timeout value or optimize operation

**Copilot Should:**
- Increase the timeout-minutes value
- Or suggest optimization strategies
- Create PR with the fix

#### Test 4: Resource Error

**Expected Behavior:**
- Category: RESOURCE
- Confidence: 95%
- Suggested Fix: Add disk cleanup or use more space

**Copilot Should:**
- Add disk cleanup steps
- Suggest removing unnecessary files
- Create PR with the fix

#### Test 5: Docker Error

**Expected Behavior:**
- Category: DOCKER
- Confidence: 85%
- Suggested Fix: Review Dockerfile syntax

**Copilot Should:**
- Identify the Dockerfile issue
- Fix the syntax or build step
- Create PR with the fix

#### Test 6: Test Failure

**Expected Behavior:**
- Category: TEST
- Confidence: 80%
- Suggested Fix: Fix the failing test

**Copilot Should:**
- Identify the failing test
- Analyze the assertion error
- Either fix the code or update the test
- Create PR with the fix

## 🛠️ Manual Testing

### Test Auto-Heal with Real Failure

1. **Create a branch with a bug:**
   ```bash
   git checkout -b test-autoheal
   echo "import nonexistent_module" >> test_bug.py
   git add test_bug.py
   git commit -m "Add intentional bug for testing"
   git push origin test-autoheal
   ```

2. **Create a workflow that uses the buggy code:**
   ```yaml
   # .github/workflows/test-real-bug.yml
   name: Test Real Bug
   on: [push]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - run: python test_bug.py
   ```

3. **Push and watch:**
   ```bash
   git add .github/workflows/test-real-bug.yml
   git commit -m "Add test workflow"
   git push origin test-autoheal
   ```

4. **Observe:**
   - Workflow fails
   - Auto-heal activates
   - Issue created
   - Copilot fixes the bug
   - PR created

## 📊 Monitoring Auto-Heal Activity

### View Auto-Heal Issues

```bash
# List all auto-heal issues
gh issue list --label auto-heal

# View a specific issue
gh issue view <issue-number>

# Check issue comments (Copilot instructions)
gh issue view <issue-number> --comments
```

### View Auto-Heal PRs

```bash
# List auto-heal PRs
gh pr list --label automated-fix

# View a specific PR
gh pr view <pr-number>

# Check PR diff
gh pr diff <pr-number>
```

### View Workflow Runs

```bash
# List auto-heal workflow runs
gh run list --workflow=auto-heal-failures.yml

# View details of a specific run
gh run view <run-id>

# Download artifacts
gh run download <run-id>
```

### View Artifacts

```bash
# Download and inspect analysis artifacts
gh run download <run-id> --name workflow-failure-analysis-<id>

# View the analysis
cat failure_analysis.json | jq '.'
cat failure_report.md
cat copilot_healing_prompt.md
```

## 🐛 Troubleshooting Tests

### Auto-Heal Doesn't Trigger

**Check:**
1. Is the workflow enabled?
   ```bash
   # View workflow status
   gh workflow view auto-heal-failures.yml
   ```

2. Did the workflow actually fail (not cancel or skip)?
   ```bash
   gh run list --workflow=test-auto-heal.yml
   ```

3. Are permissions correct?
   ```yaml
   permissions:
     contents: write
     pull-requests: write
     issues: write
     actions: read
   ```

### Issue Not Created

**Debug:**
```bash
# Check auto-heal workflow logs
gh run view <run-id> --log

# Look for errors in the "Create issue for tracking" step
gh run view <run-id> --log | grep -A 10 "Create issue"
```

### Copilot Doesn't Respond

**Try:**
1. Check Copilot subscription is active
2. Manually trigger Copilot:
   ```
   Comment on the issue: @github-copilot workspace
   
   Please analyze this workflow failure and create a fix.
   ```

3. Check Copilot has repo access

### Branch Not Created

**Debug:**
```bash
# Check if branch was created
git ls-remote origin | grep auto-heal

# Check auto-heal workflow logs for branch creation step
gh run view <run-id> --log | grep -A 10 "Create auto-heal branch"
```

## 📈 Performance Testing

### Test Auto-Heal Performance

1. **Trigger multiple failures simultaneously:**
   ```bash
   # Run multiple test workflows
   for type in dependency_error syntax_error timeout_error; do
     gh workflow run test-auto-heal.yml -f failure_type=$type
   done
   ```

2. **Monitor system behavior:**
   - Are all failures detected?
   - Are separate issues created?
   - Do workflows interfere with each other?

3. **Check rate limits:**
   - GitHub API rate limits
   - Issue creation limits
   - PR creation limits

### Test Configuration Changes

1. **Test exclusions:**
   ```yaml
   # In .github/auto-heal-config.yml
   excluded_workflows:
     - "Test Auto-Heal System"
   ```
   
   Run test - should NOT trigger auto-heal

2. **Test max attempts:**
   ```yaml
   max_heal_attempts_per_day: 1
   ```
   
   Run test twice - second should be skipped

3. **Test custom patterns:**
   ```yaml
   failure_patterns:
     - pattern: "my custom error"
       suggestion: "My custom fix"
   ```
   
   Create workflow with custom error - verify detection

## 🎓 Advanced Testing

### Test with Real Workflow Failures

1. **Temporarily break a real workflow:**
   - Comment out a required step
   - Remove a dependency
   - Add a typo

2. **Let auto-heal fix it**

3. **Review the PR carefully**

4. **Learn from Copilot's approach**

### Test Edge Cases

1. **Multiple failures in one workflow:**
   - Different jobs fail for different reasons
   - Should create one issue with all failures

2. **Cascading failures:**
   - Workflow A fails
   - Fix for A breaks workflow B
   - B fails and triggers auto-heal

3. **Repeated failures:**
   - Same workflow fails multiple times
   - Should respect max_heal_attempts_per_day

### Integration Testing

Test auto-heal with:
- Branch protection rules
- Required status checks
- Code owners
- Review requirements

## 📝 Test Report Template

Use this template to document your test results:

```markdown
# Auto-Heal Test Report

## Test Information
- Date: YYYY-MM-DD
- Tester: Your Name
- Test Type: [Simulated/Real]
- Failure Type: [dependency/syntax/timeout/etc]

## Test Results

### Auto-Heal Workflow
- [ ] Triggered successfully
- [ ] Completed without errors
- [ ] Duration: X minutes

### Issue Creation
- [ ] Issue created
- [ ] Correct labels applied
- [ ] Analysis included
- [ ] Issue #: XXX

### Failure Analysis
- [ ] Category identified correctly
- [ ] Confidence level: X%
- [ ] Affected files identified
- [ ] Suggested fixes appropriate

### Copilot Integration
- [ ] Copilot notified via comment
- [ ] Copilot responded
- [ ] Response time: X minutes

### Branch Creation
- [ ] Branch created
- [ ] Correct naming convention
- [ ] Based on correct commit

### PR Creation (if Copilot completed fix)
- [ ] PR created
- [ ] Linked to issue
- [ ] Contains appropriate changes
- [ ] Tests pass
- [ ] PR #: XXX

## Issues Found
List any problems or unexpected behavior

## Recommendations
Suggestions for improvement

## Notes
Additional observations
```

## 🎯 Success Criteria

A successful test should demonstrate:

✅ **Detection**: Workflow failure detected within 1 minute
✅ **Analysis**: Accurate failure categorization and analysis
✅ **Issue**: Issue created with all required information
✅ **Branch**: Auto-heal branch created successfully
✅ **Copilot**: Copilot receives clear instructions
✅ **Artifacts**: All analysis artifacts saved
✅ **Logs**: Clear, actionable logs throughout

## 🔄 Continuous Testing

### Weekly Testing Schedule

- **Monday**: Test dependency errors
- **Tuesday**: Test syntax errors
- **Wednesday**: Test timeout/resource errors
- **Thursday**: Test Docker/build errors
- **Friday**: Test with real workflow failures

### Monthly Review

1. Review all auto-heal issues from the month
2. Analyze success rate
3. Identify common failure patterns
4. Update configuration based on learnings
5. Improve failure detection patterns

---

## 🆘 Getting Help

If tests fail or you encounter issues:

1. Check the [troubleshooting section](#troubleshooting-tests)
2. Review workflow logs in Actions tab
3. Download and inspect artifacts
4. Create an issue with label `auto-heal-support`
5. Include test report and relevant logs

---

*Happy Testing! 🎉*
