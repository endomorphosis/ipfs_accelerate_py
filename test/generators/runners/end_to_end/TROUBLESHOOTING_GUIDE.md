# Troubleshooting Guide for Integrated Visualization and Reports System

This guide provides solutions for common issues encountered when using the Integrated Visualization and Reports System. It covers problems related to both the Visualization Dashboard and the Enhanced CI/CD Reports Generator components.

## Table of Contents

1. [Dashboard Process Management Issues](#dashboard-process-management-issues)
2. [Database Connectivity Problems](#database-connectivity-problems)
3. [Report Generation Failures](#report-generation-failures)
4. [Visualization Rendering Issues](#visualization-rendering-issues)
5. [Browser Integration Problems](#browser-integration-problems)
6. [Combined Workflow Challenges](#combined-workflow-challenges)
7. [CI/CD Integration Issues](#cicd-integration-issues)
8. [Performance and Resource Concerns](#performance-and-resource-concerns)
9. [Installation and Dependency Problems](#installation-and-dependency-problems)
10. [Advanced Troubleshooting](#advanced-troubleshooting)

## Dashboard Process Management Issues

### Dashboard Fails to Start

**Problem**: The dashboard process fails to start when using `--dashboard` flag.

**Solutions**:

1. **Check port availability**:
   ```bash
   # Check if port is already in use
   sudo lsof -i:8050
   # Use a different port
   python integrated_visualization_reports.py --dashboard --dashboard-port 8051
   ```

2. **Verbose logging**:
   ```bash
   # Enable verbose logging to see detailed startup messages
   python integrated_visualization_reports.py --dashboard --verbose
   ```

3. **Verify Dash installation**:
   ```bash
   pip install dash==2.14.0 dash-bootstrap-components==1.5.0
   ```

4. **Check dashboard script**:
   ```bash
   # Verify the dashboard script works standalone
   python visualization_dashboard.py
   ```

### Process Hangs or Doesn't Terminate

**Problem**: The dashboard process doesn't terminate properly after keyboard interrupt or when the program completes.

**Solutions**:

1. **Force terminate** (not recommended for regular use):
   ```bash
   # Find process ID
   ps aux | grep visualization_dashboard
   # Kill the process
   kill -9 [process_id]
   ```

2. **Use dashboard-only mode**:
   ```bash
   python integrated_visualization_reports.py --dashboard-only
   ```

3. **Check for blocking operations**:
   ```bash
   # Run in verbose mode to identify potential blocking operations
   python integrated_visualization_reports.py --dashboard --verbose
   ```

### Process Startup Timeout

**Problem**: The system reports a timeout waiting for the dashboard to start.

**Solutions**:

1. **Run the dashboard directly** to check for startup issues:
   ```bash
   python visualization_dashboard.py --debug
   ```

2. **Check system resources**:
   ```bash
   # Ensure you have sufficient CPU and memory available
   htop
   ```

3. **Reduce database complexity**:
   ```bash
   # Try with a smaller or empty database
   python integrated_visualization_reports.py --dashboard --db-path ./new_benchmark_db.duckdb
   ```

## Database Connectivity Problems

### Database Connection Errors

**Problem**: The system fails to connect to the DuckDB database.

**Solutions**:

1. **Verify DuckDB installation**:
   ```bash
   pip install duckdb==0.9.2
   ```

2. **Check database path**:
   ```bash
   # Ensure the database file exists
   ls -la ./benchmark_db.duckdb
   
   # Try with a new database file
   python integrated_visualization_reports.py --dashboard --db-path ./new_benchmark_db.duckdb
   ```

3. **Check file permissions**:
   ```bash
   # Ensure read/write permissions
   chmod 644 ./benchmark_db.duckdb
   ```

4. **Verify database is not corrupted**:
   ```bash
   # Check database integrity
   python duckdb_api/core/benchmark_db_maintenance.py --check-integrity --db-path ./benchmark_db.duckdb
   ```

### Missing or Empty Database Tables

**Problem**: The dashboard shows no data or empty visualizations.

**Solutions**:

1. **Check if database tables exist**:
   ```bash
   # Run query to verify tables
   python -c "import duckdb; conn = duckdb.connect('./benchmark_db.duckdb'); print(conn.execute('SELECT table_name FROM information_schema.tables').fetchall())"
   ```

2. **Generate sample data**:
   ```bash
   python duckdb_api/utils/generate_sample_benchmarks.py --db ./benchmark_db.duckdb
   ```

3. **Run some tests to populate the database**:
   ```bash
   python unified_component_tester.py --model bert-base-uncased --hardware cpu --db-path ./benchmark_db.duckdb
   ```

### Data Type Conversion Errors

**Problem**: The system reports errors related to database column types or conversion issues.

**Solutions**:

1. **Update database schema**:
   ```bash
   python duckdb_api/schema/update_db_schema.py --db-path ./benchmark_db.duckdb
   ```

2. **Regenerate the database**:
   ```bash
   # Backup existing database
   cp ./benchmark_db.duckdb ./benchmark_db.duckdb.bak
   
   # Create new database with correct schema
   python duckdb_api/schema/creation/create_benchmark_schema.py --sample-data --db-path ./benchmark_db.duckdb
   ```

3. **Migrate existing data**:
   ```bash
   python duckdb_api/migration/migrate_data.py --source ./benchmark_db.duckdb.bak --target ./benchmark_db.duckdb
   ```

## Report Generation Failures

### Report Generation Errors

**Problem**: The `--reports` flag results in errors or empty reports.

**Solutions**:

1. **Check required directories**:
   ```bash
   # Ensure the output directory exists
   mkdir -p ./reports
   python integrated_visualization_reports.py --reports --output-dir ./reports
   ```

2. **Verify report generator script**:
   ```bash
   # Run the report generator directly
   python enhanced_ci_cd_reports.py --output-dir ./reports
   ```

3. **Test with a specific report type**:
   ```bash
   # Try generating just one report type
   python integrated_visualization_reports.py --reports --simulation-validation
   ```

### Missing Visualization Images in Reports

**Problem**: Reports are generated, but visualization images are missing.

**Solutions**:

1. **Ensure matplotlib is installed**:
   ```bash
   pip install matplotlib
   ```

2. **Enable visualization generation explicitly**:
   ```bash
   python integrated_visualization_reports.py --reports --include-visualizations
   ```

3. **Specify visualization format**:
   ```bash
   python integrated_visualization_reports.py --reports --include-visualizations --visualization-format png
   ```

### Incorrect Data in Reports

**Problem**: Reports contain incorrect or outdated data.

**Solutions**:

1. **Verify database contains expected results**:
   ```bash
   python duckdb_api/core/benchmark_db_query.py --sql "SELECT model_name, hardware_type, success, test_date FROM test_results ORDER BY test_date DESC LIMIT 10" --db-path ./benchmark_db.duckdb
   ```

2. **Clean database cache**:
   ```bash
   python duckdb_api/core/benchmark_db_maintenance.py --clean-cache --db-path ./benchmark_db.duckdb
   ```

3. **Run with specified data range**:
   ```bash
   python integrated_visualization_reports.py --reports --historical --days 7
   ```

## Visualization Rendering Issues

### Dashboard Fails to Display Charts

**Problem**: The dashboard loads, but charts do not appear.

**Solutions**:

1. **Check browser console for errors**:
   Open browser developer tools (F12) and check the console for errors.

2. **Verify Plotly installation**:
   ```bash
   pip install plotly==5.14.0
   ```

3. **Try a different browser**:
   Chrome or Firefox are recommended for best compatibility.

4. **Clear browser cache**:
   Clear your browser's cache and reload the dashboard.

### Charts Show No Data

**Problem**: Charts load but show "No data available" or appear empty.

**Solutions**:

1. **Check data query**:
   ```bash
   # Run in verbose mode to see queries
   python integrated_visualization_reports.py --dashboard --verbose
   ```

2. **Verify model and hardware filters**:
   Make sure you've selected valid model and hardware types in the dashboard dropdowns.

3. **Check database content**:
   ```bash
   python duckdb_api/core/benchmark_db_query.py --db-path ./benchmark_db.duckdb --sql "SELECT COUNT(*) FROM test_results"
   ```

### Dashboard Layout Issues

**Problem**: Dashboard layout appears broken or elements overlap.

**Solutions**:

1. **Verify dash and dash-bootstrap-components versions**:
   ```bash
   pip install dash==2.14.0 dash-bootstrap-components==1.5.0
   ```

2. **Try a standard browser window size**:
   Resize your browser to 1920x1080 or another standard resolution.

3. **Check CSS loading**:
   Check browser developer tools (F12) to verify that CSS files are loading correctly.

## Browser Integration Problems

### Browser Doesn't Open Automatically

**Problem**: Using `--open-browser` flag doesn't automatically open the browser.

**Solutions**:

1. **Verify webbrowser module**:
   The system uses Python's built-in `webbrowser` module, which may have limitations on some systems.

2. **Specify a browser**:
   ```bash
   # Set default browser in environment
   export BROWSER=firefox
   python integrated_visualization_reports.py --dashboard --open-browser
   ```

3. **Open manually**:
   ```bash
   # Get the URL from the console output and open manually
   # Usually http://localhost:8050
   ```

### Dashboard Opens but Doesn't Connect

**Problem**: Browser opens but shows "Connecting..." or other loading issues.

**Solutions**:

1. **Check host settings**:
   ```bash
   # Use localhost explicitly
   python integrated_visualization_reports.py --dashboard --dashboard-host localhost
   ```

2. **Check firewall settings**:
   Ensure your firewall isn't blocking the connection.

3. **Try a different port**:
   ```bash
   python integrated_visualization_reports.py --dashboard --dashboard-port 8080
   ```

## Combined Workflow Challenges

### Dashboard and Reports Conflict

**Problem**: Using `--dashboard` and `--reports` together causes issues.

**Solutions**:

1. **Run sequentially**:
   ```bash
   # Generate reports first, then start dashboard
   python integrated_visualization_reports.py --reports
   python integrated_visualization_reports.py --dashboard
   ```

2. **Use the export option**:
   ```bash
   # Export dashboard visualizations without running the live dashboard
   python integrated_visualization_reports.py --dashboard-export --reports
   ```

3. **Check resource usage**:
   Generating reports and running the dashboard simultaneously may require significant resources.

### Process Flow Interruptions

**Problem**: The workflow gets interrupted or hangs when combining dashboard and reports.

**Solutions**:

1. **Increase verbosity**:
   ```bash
   python integrated_visualization_reports.py --dashboard --reports --verbose
   ```

2. **Run in debug mode**:
   ```bash
   python integrated_visualization_reports.py --dashboard --reports --debug
   ```

3. **Focus on one operation at a time**:
   ```bash
   python integrated_visualization_reports.py --dashboard-only
   python integrated_visualization_reports.py --reports
   ```

## CI/CD Integration Issues

### CI/CD Badge Generation Failures

**Problem**: Status badges are not generated correctly in CI environments.

**Solutions**:

1. **Explicitly specify output directory**:
   ```bash
   python integrated_visualization_reports.py --reports --badge-only --output-dir ./badges
   ```

2. **Check CI environment variables**:
   Ensure your CI system has the necessary permissions and environment variables set.

3. **Use simpler badge mode**:
   ```bash
   python integrated_visualization_reports.py --reports --badge-only --ci
   ```

### GitHub Pages Integration Problems

**Problem**: Reports generated with `--github-pages` don't work on GitHub Pages.

**Solutions**:

1. **Check file paths**:
   GitHub Pages requires relative paths. Ensure reports use relative paths.

2. **Verify output directory structure**:
   ```bash
   python integrated_visualization_reports.py --reports --github-pages --output-dir ./docs
   ```

3. **Check GitHub Pages settings**:
   Ensure your GitHub repository has GitHub Pages enabled and is pointed to the correct directory.

### CI Pipeline Exit Codes

**Problem**: CI pipeline doesn't detect report generation failures.

**Solutions**:

1. **Check exit code explicitly**:
   ```bash
   python integrated_visualization_reports.py --reports --ci
   echo $?  # Should be 0 for success, non-zero for failure
   ```

2. **Add error handling to CI script**:
   ```bash
   # Example for bash script
   if ! python integrated_visualization_reports.py --reports --ci; then
     echo "Report generation failed"
     exit 1
   fi
   ```

## Performance and Resource Concerns

### High Memory Usage

**Problem**: The system uses excessive memory, especially with large databases.

**Solutions**:

1. **Limit data scope**:
   ```bash
   # Limit to recent data
   python integrated_visualization_reports.py --reports --historical --days 7
   ```

2. **Use smaller database**:
   ```bash
   # Create a filtered copy of the database with just the needed data
   python duckdb_api/core/benchmark_db_query.py --sql "CREATE TABLE benchmark_db_small AS SELECT * FROM test_results WHERE test_date > '20250301'" --db-path ./benchmark_db.duckdb
   ```

3. **Optimize database**:
   ```bash
   python duckdb_api/core/benchmark_db_maintenance.py --optimize-db --vacuum
   ```

### Slow Dashboard Loading

**Problem**: Dashboard takes a long time to load or update.

**Solutions**:

1. **Optimize queries**:
   ```bash
   # Use a streamlined version of the dashboard
   python visualization_dashboard.py --lite-mode
   ```

2. **Reduce data volume**:
   Filter data more aggressively in dashboard dropdowns.

3. **Add database indexes**:
   ```bash
   python duckdb_api/schema/update_db_schema.py --add-indexes --db-path ./benchmark_db.duckdb
   ```

### High CPU Usage

**Problem**: The system uses excessive CPU, especially during report generation.

**Solutions**:

1. **Limit visualization complexity**:
   ```bash
   python integrated_visualization_reports.py --reports --simple-visualizations
   ```

2. **Generate reports in sequence**:
   ```bash
   # Generate one report at a time
   python integrated_visualization_reports.py --reports --simulation-validation
   python integrated_visualization_reports.py --reports --cross-hardware-comparison
   ```

3. **Reduce data processing**:
   ```bash
   # Use pre-aggregated data where possible
   python integrated_visualization_reports.py --reports --use-aggregates
   ```

## Installation and Dependency Problems

### Missing Dependencies

**Problem**: System reports missing Python packages or dependencies.

**Solutions**:

1. **Install all requirements**:
   ```bash
   pip install -r dashboard_requirements.txt
   ```

2. **Install specific dependencies**:
   ```bash
   pip install dash==2.14.0 dash-bootstrap-components==1.5.0 plotly==5.14.0 pandas numpy duckdb==0.9.2
   ```

3. **Check for version conflicts**:
   ```bash
   pip check
   ```

### Version Compatibility Issues

**Problem**: System reports version conflicts or incompatibilities.

**Solutions**:

1. **Use known working versions**:
   ```bash
   pip install dash==2.14.0 dash-bootstrap-components==1.5.0 plotly==5.14.0 duckdb==0.9.2
   ```

2. **Create a separate environment**:
   ```bash
   # Using venv
   python -m venv viz_env
   source viz_env/bin/activate
   pip install -r dashboard_requirements.txt
   ```

3. **Check Python version**:
   ```bash
   python --version  # Should be 3.8 or higher
   ```

## Advanced Troubleshooting

### Debug Mode

For more detailed debugging, you can use the debug mode:

```bash
# Run with debug mode
python integrated_visualization_reports.py --dashboard --debug

# Run with verbose logging and debug mode
python integrated_visualization_reports.py --dashboard --verbose --debug
```

### Diagnostic Logging

To increase log verbosity and save logs to a file:

```bash
# Redirect output to a log file
python integrated_visualization_reports.py --dashboard --verbose > dashboard.log 2>&1

# View logs in real-time
tail -f dashboard.log
```

### Standalone Component Testing

Test each component separately to isolate issues:

1. **Test dashboard alone**:
   ```bash
   python visualization_dashboard.py --port 8050 --db-path ./benchmark_db.duckdb
   ```

2. **Test report generator alone**:
   ```bash
   python enhanced_ci_cd_reports.py --output-dir ./reports --db-path ./benchmark_db.duckdb
   ```

3. **Test database queries directly**:
   ```bash
   python duckdb_api/core/benchmark_db_query.py --sql "SELECT * FROM test_results LIMIT 10" --db-path ./benchmark_db.duckdb
   ```

### Process Monitoring

Monitor running processes to identify issues:

```bash
# Monitor Python processes
watch -n 1 'ps aux | grep python'

# Check for zombie processes
ps aux | grep defunc

# Monitor network connections (for dashboard port)
watch -n 1 'netstat -tuln | grep 8050'
```

### Temporary File Inspection

Check temporary files that might be created during operation:

```bash
# List temporary files
find /tmp -name "viz_dashboard*" -o -name "report_gen*" | sort

# Monitor file creation
watch -n 1 'ls -la /tmp/viz_*'
```

### Database Schema Verification

Verify that the database schema is correct:

```bash
# Get schema information
python -c "import duckdb; conn = duckdb.connect('./benchmark_db.duckdb'); print(conn.execute('SELECT * FROM information_schema.columns').fetchdf())"

# Check for specific tables
python -c "import duckdb; conn = duckdb.connect('./benchmark_db.duckdb'); print(conn.execute('SELECT COUNT(*) FROM test_results').fetchone())"
```

## Getting Additional Help

If you've tried the solutions in this guide and are still experiencing issues:

1. **Check the source code** for the specific components:
   - `integrated_visualization_reports.py`: Core integration system
   - `visualization_dashboard.py`: Dashboard implementation
   - `enhanced_ci_cd_reports.py`: Report generation

2. **Review tests** for expected behavior:
   - `test_integrated_visualization_reports.py`: Tests for the integrated system
   - `test_visualization_dashboard.py`: Tests for the dashboard

3. **Consult related documentation**:
   - `VISUALIZATION_DASHBOARD_README.md`: Dashboard documentation
   - `INTEGRATION_SUMMARY.md`: Overview of the integrated system
   - `USER_GUIDE.md`: General usage guide with visualization section

4. **Contact the infrastructure team** with detailed information about the issue:
   - Include full error messages
   - Describe steps to reproduce
   - Share environment details (OS, Python version, package versions)
   - Provide relevant logs