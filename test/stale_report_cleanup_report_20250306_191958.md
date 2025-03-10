# Stale and Problematic Report Cleanup Report

Generated: 2025-03-06 19:19:58

## Summary

- Total problematic files found: 12
- File types:
  - markdown: 12

## Problematic Files

| File Path | Type | Issue | Last Modified |
|-----------|------|-------|---------------|
| benchmark_results/20250306_191614/benchmark_report_20250306_191614.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:16:14.765394 |
| benchmark_results/20250306_191548/benchmark_report_20250306_191548.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:15:48.788170 |
| benchmark_results/20250306_191631/benchmark_report_20250306_191631.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:16:31.358178 |
| benchmark_results/20250306_191448/benchmark_report_20250306_191448.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:14:48.853363 |
| benchmark_results/20250306_191618/benchmark_report_20250306_191618.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:16:18.737582 |
| benchmark_results/20250306_191502/benchmark_report_20250306_191502.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:15:03.034025 |
| benchmark_results/20250306_191601/benchmark_report_20250306_191601.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:16:01.672776 |
| benchmark_results/20250306_191627/benchmark_report_20250306_191627.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:16:27.261985 |
| benchmark_results/20250306_191553/benchmark_report_20250306_191553.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:15:53.336384 |
| benchmark_results/20250306_191605/benchmark_report_20250306_191605.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:16:06.004981 |
| benchmark_results/20250306_191444/benchmark_report_20250306_191444.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:14:44.861177 |
| benchmark_results/20250306_191457/benchmark_report_20250306_191457.md | markdown | May contain simulation results presented as real data | 2025-03-06T19:14:57.589771 |

## Recommendations

Based on the scan results, we recommend the following actions:

1. **Mark files with warnings**: Add clear warnings to files that may contain misleading data
   ```bash
   python cleanup_stale_reports.py --mark
   ```

2. **Archive problematic files**: Move problematic files to an archive directory
   ```bash
   python cleanup_stale_reports.py --archive
   ```

3. **Fix report generators**: Add validation to report generator scripts
   ```bash
   python cleanup_stale_reports.py --fix-report-py
   ```

4. **Update database schema**: Ensure the database schema includes simulation flags
   ```bash
   python update_db_schema_for_simulation.py
   ```


## Next Steps

After implementing these recommendations, re-run the scan to verify that all issues have been addressed:

```bash
python cleanup_stale_reports.py --scan
```

If you need to remove problematic files entirely (use with caution):

```bash
python cleanup_stale_reports.py --remove
```
