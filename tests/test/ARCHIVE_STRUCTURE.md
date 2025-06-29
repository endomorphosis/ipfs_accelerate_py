# Archive Structure and Management

**Last Updated: March 10, 2025**

This document provides a comprehensive overview of the archiving system used in the IPFS Accelerate Python repository.

## Archive Directory Structure

The main archive directories include:

- `/test/archive/`: Main archive for stale test directory files (created March 10, 2025)
- `/archive/`: Top-level repository for archived files from all directories (planned Q2 2025)
  - `/archive/backup_files/`: Contains backup (*.bak) files with original directory structure preserved
  - `/archive/old_reports/`: Contains old benchmark reports and results files
  - `/archive/stale_scripts/`: Contains deprecated Python scripts that are no longer in active use
  - `/archive/old_documentation/`: Contains older documentation files

## March 10, 2025 Archive Operation

On March 10, 2025, a major cleanup of the test directory was performed:

- Approximately 480 files were moved to `/test/archive/`
- Categories of archived files included:
  - Backup files (*.bak, *.bak_*)
  - Old database backups (benchmark_db*.duckdb.bak*)
  - Fixed/superseded implementation files
  - Completed status reports and documentation
  - Older benchmark reports
  - One-time utility scripts
  - Deprecated test runners
  - Duplicate files with newer versions available

This cleanup significantly reduced clutter while preserving all historical files for reference. 
See `/test/cleanup_summary.md` for complete details about the archiving process.

## Archiving Tools

The following scripts are used to manage the archiving process:

- `/test/archive_stale_files.sh`: Archives stale files from the test directory
- `/test/archive_markdown_files.sh`: Archives older markdown documentation files
- `/test/archive_json_files.sh`: Archives outdated JSON configuration files

## Archived File Categories

### Python Implementation Files
- Fixed versions of generators (e.g., `fixed_*_generator.py`)
- Improved/merged test generators (e.g., `improved_*_generator.py`)
- One-time fixes for API modules
- Simulation detection fixes

### Documentation Files
- Completed phase reports (e.g., `PHASE16_COMPLETION_REPORT.md`)
- Implementation summaries (e.g., `*_IMPLEMENTATION_SUMMARY.md`)
- Migration reports
- Archived documentation references

### Benchmark/Test Results
- Old benchmark reports (e.g., `benchmark_report_*.md`)
- Test coverage summaries
- Hardware compatibility reports

### Configuration Files
- Old JSON configuration files (e.g., `*_results.json`)
- Database backups (e.g., `benchmark_db*.duckdb.bak*`)

### Utility Scripts
- One-time migrations
- Path fixes
- Cleanup utilities

## Archiving Guidelines

When archiving files, follow these guidelines:

1. **Never delete files** - Always move to appropriate archive directory
2. **Maintain file paths** - Archive with relative paths preserved
3. **Document what was archived** - Update cleanup_summary.md with details
4. **Archive selectively** - Only archive files that are:
   - Superseded by newer versions
   - No longer needed for active development
   - Duplicates with slight variations
   - One-time utilities that have completed their purpose
   - Old backup files (.bak)
5. **Update documentation** - Ensure README.md and DOCUMENTATION_INDEX.md reflect archive status

## Accessing Archived Files

Archived files can be accessed directly in their archive directories. They remain in the repository for historical reference but should not be used for active development.

If you need to restore an archived file:

1. Locate it in the appropriate archive directory
2. Copy (don't move) it to the required location
3. Make sure to document in commits why you're using an archived file

## Future Archive Plans (Q2-Q3 2025)

As the project continues to evolve, a more comprehensive archiving strategy will be implemented:

1. **Unified Archive Structure** (Q2 2025)
   - Consolidate various archive directories into a single structure
   - Implement clearer categorization
   - Add archive metadata for easier searching

2. **Automated Archiving Tools** (Q3 2025)
   - Create tools to automatically identify archiving candidates
   - Implement version tracking for archived files
   - Add archive search and restore utilities

## Conclusion

This archiving system helps maintain a clean and focused repository while preserving the complete history of development. The archived files serve as valuable reference material and documentation of the project's evolution.

If you have questions about the archive structure or need help locating specific archived files, refer to the `/test/cleanup_summary.md` document or contact the repository maintainers.