# Runner Permission Fix - Quick Reference

## 🚨 Problem
```
Error: EACCES: permission denied, unlink '/home/actions-runner/_work/.../file'
```

## ⚡ Quick Fix (Run on runner host)
```bash
cd /home/barberb/ipfs_accelerate_py
./.github/scripts/fix_runner_permissions.sh
```

## 🛡️ Prevent in Workflows

Add this as the **FIRST** step in jobs using self-hosted runners:

```yaml
jobs:
  your-job:
    runs-on: [self-hosted, linux, arm64]
    steps:
      - name: Pre-job cleanup
        uses: ./.github/actions/cleanup-workspace
      
      - uses: actions/checkout@v4
        with:
          clean: true
          fetch-depth: 1
```

## 📋 Files Created

| File | Purpose |
|------|---------|
| `.github/scripts/fix_runner_permissions.sh` | Manual fix script |
| `.github/workflows/runner-cleanup.yml` | Auto cleanup (every 6h) |
| `.github/actions/cleanup-workspace/` | Reusable cleanup action |
| `RUNNER_PERMISSION_FIX_GUIDE.md` | Full documentation |
| `WORKFLOW_UPDATE_EXAMPLES.md` | Usage examples |

## 🔄 Automated Cleanup

Already configured! Runs automatically every 6 hours.

**Manual trigger:**
```bash
gh workflow run runner-cleanup.yml
```

**Check status:**
```bash
gh run list --workflow=runner-cleanup.yml
```

## ✅ Checklist

- [ ] Run fix script once: `./.github/scripts/fix_runner_permissions.sh`
- [ ] Add pre-job cleanup to self-hosted workflows
- [ ] Use `clean: true` in checkout actions
- [ ] Test workflows after changes
- [ ] Monitor cleanup workflow runs

## 📊 Monitoring

```bash
# View cleanup runs
gh run list --workflow=runner-cleanup.yml

# Check runner logs
tail -f /home/actions-runner/_diag/Runner*.log

# Check disk usage
df -h /home/actions-runner/_work
```

## 🆘 Troubleshooting

**Issue:** Permission denied on file  
**Fix:** Run `./.github/scripts/fix_runner_permissions.sh`

**Issue:** Git lock file exists  
**Fix:** Same as above

**Issue:** Disk full  
**Fix:** `df -h` and clean old workspaces

## 📚 Full Documentation

- **RUNNER_PERMISSION_FIX_GUIDE.md** - Complete guide
- **WORKFLOW_UPDATE_EXAMPLES.md** - Workflow examples
- **RUNNER_PERMISSION_FIX_IMPLEMENTATION.md** - Implementation details

## 💡 Best Practices

1. Always use `clean: true` in checkout
2. Add pre-job cleanup to all self-hosted jobs
3. Let automated cleanup run (every 6 hours)
4. Monitor runner health regularly
5. Use `fetch-depth: 1` for faster checkouts

---

**Quick Help:** Run `./.github/scripts/fix_runner_permissions.sh --help`
