# Mobile Edge CI/CD Integration Plan

**Date: April 29, 2025**  
**Status: Planning Phase**

## Overview

This document outlines the strategy for integrating the Mobile Edge testing framework (Android and iOS Test Harnesses) into the project's CI/CD pipeline. The integration aims to ensure consistent testing across mobile platforms as part of the automated testing infrastructure.

## Goals

1. Automate mobile device testing as part of the CI/CD pipeline
2. Enable regression detection for mobile performance
3. Provide real-time feedback on mobile compatibility for new code changes
4. Generate comprehensive mobile performance reports
5. Visualize cross-platform performance insights
6. Support both real device and emulator/simulator testing

## Integration Strategy

### 1. CI Runner Requirements

**Hardware Requirements:**
- Dedicated macOS runners for iOS testing
- Dedicated Linux runners for Android testing
- Physical devices connected via USB for real device testing
- Virtual devices for emulator/simulator testing

**Software Requirements:**
- Android SDK and build tools
- Xcode and iOS development tools
- Python 3.7+ with required dependencies
- DuckDB for benchmark storage
- Required mobile frameworks (ONNX Runtime, TFLite, Core ML)

### 2. Mobile Test Workflow

The mobile test workflow will be triggered:
- On pull requests to main branch
- On scheduled basis (nightly runs)
- Manually when requested

**Workflow Stages:**
1. **Preparation**
   - Setup build environment
   - Install dependencies
   - Prepare test artifacts

2. **Android Testing**
   - Deploy tests to Android devices/emulators
   - Run benchmark suite
   - Collect and store results

3. **iOS Testing**
   - Deploy tests to iOS devices/simulators
   - Run benchmark suite
   - Collect and store results

4. **Cross-Platform Analysis**
   - Compare results across platforms
   - Generate performance reports
   - Identify performance regressions

5. **Results Publication**
   - Store benchmark data in DuckDB
   - Generate visualizations
   - Publish results to dashboard

### 3. GitHub Actions Configuration

```yaml
name: Mobile Edge CI Testing

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:
  schedule:
    - cron: '0 0 * * *'  # Run daily at midnight

jobs:
  android-testing:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        android-device: ['pixel-4', 'samsung-s21', 'oneplus-9']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r test/android_test_harness/requirements.txt
        
    - name: Setup Android environment
      uses: android-actions/setup-android@v2
        
    - name: Start Android emulator
      uses: reactivecircus/android-emulator-runner@v2
      with:
        api-level: 30
        arch: x86_64
        profile: ${{ matrix.android-device }}
        
    - name: Run Android benchmarks
      run: |
        python test/android_test_harness/run_ci_benchmarks.py --output-db benchmark_results.duckdb
        
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: android-benchmark-results-${{ matrix.android-device }}
        path: benchmark_results.duckdb

  ios-testing:
    runs-on: macos-latest
    strategy:
      matrix:
        ios-device: ['iphone-12', 'iphone-13', 'ipad-pro']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r test/ios_test_harness/requirements.txt
        
    - name: Setup iOS environment
      uses: maxim-lobanov/setup-xcode@v1
      with:
        xcode-version: '13.2.1'
        
    - name: Start iOS simulator
      run: |
        xcrun simctl boot ${{ matrix.ios-device }}
        
    - name: Run iOS benchmarks
      run: |
        python test/ios_test_harness/run_ci_benchmarks.py --output-db benchmark_results.duckdb
        
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: ios-benchmark-results-${{ matrix.ios-device }}
        path: benchmark_results.duckdb

  cross-platform-analysis:
    needs: [android-testing, ios-testing]
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Download all benchmark results
      uses: actions/download-artifact@v3
      
    - name: Merge benchmark databases
      run: |
        python test/merge_benchmark_databases.py --output merged_results.duckdb
        
    - name: Run cross-platform analysis
      run: |
        python test/cross_platform_analysis.py analyze --db-path merged_results.duckdb --output analysis_results.json
        python test/cross_platform_analysis.py compare --db-path merged_results.duckdb --output cross_platform_report.md --format markdown
        python test/cross_platform_analysis.py visualize --db-path merged_results.duckdb --output performance_comparison.png
        
    - name: Generate performance dashboard
      run: |
        python test/generate_mobile_dashboard.py --data-file analysis_results.json --output mobile_dashboard.html
        
    - name: Upload analysis results
      uses: actions/upload-artifact@v3
      with:
        name: cross-platform-analysis
        path: |
          analysis_results.json
          cross_platform_report.md
          performance_comparison.png
          mobile_dashboard.html
          
    - name: Check for performance regressions
      run: |
        python test/check_mobile_regressions.py --data-file analysis_results.json
```

### 4. Real Device Testing

For testing on physical devices:

1. **Android Devices**:
   - Connect devices via USB to dedicated Linux runners
   - Use ADB to manage device connections
   - Implement device management service to handle multiple devices

2. **iOS Devices**:
   - Connect devices via USB to dedicated macOS runners
   - Use Xcode tools to manage device connections
   - Implement device management service with provisioning profiles

### 5. Result Storage and Visualization

1. **DuckDB Integration**:
   - Store all benchmark results in DuckDB
   - Implement schema for mobile-specific metrics
   - Enable time-series queries for trend analysis

2. **Dashboard Integration**:
   - Create mobile-specific dashboard panels
   - Implement cross-platform comparison visualizations
   - Add regression detection with alerting

## Implementation Phases

### Phase 1: Emulator/Simulator Integration (Weeks 1-2)
- Setup CI runners for Android emulators and iOS simulators
- Implement basic CI workflow for automated testing
- Create initial benchmark suite for emulated environments
- Establish result storage in DuckDB

### Phase 2: Basic Real Device Integration (Weeks 3-4)
- Configure CI runners for physical device testing
- Implement device management services
- Enable basic real device benchmarking
- Implement device selection strategy

### Phase 3: Advanced Analysis and Reporting (Weeks 5-6)
- Implement cross-platform analysis in CI pipeline
- Create comprehensive reporting system
- Add regression detection and alerting
- Develop visualization components

### Phase 4: Dashboard and Monitoring (Weeks 7-8)
- Integrate mobile results into main dashboard
- Implement real-time monitoring of mobile metrics
- Add historical trend analysis for mobile performance
- Create comprehensive documentation

## Risk Management

### Identified Risks

1. **Device Availability**:
   - Risk: Physical devices may not always be available or connected
   - Mitigation: Implement fallback to emulators/simulators when physical devices are unavailable

2. **Test Flakiness**:
   - Risk: Mobile tests may be more flaky than other environments
   - Mitigation: Implement retry mechanisms and stability analysis

3. **Resource Contention**:
   - Risk: Mobile testing requires dedicated hardware resources
   - Mitigation: Implement resource scheduling and prioritization

4. **Long Test Duration**:
   - Risk: Mobile tests may take longer to run than other tests
   - Mitigation: Implement parallel testing and selective test execution

5. **Cross-Platform Consistency**:
   - Risk: Maintaining consistent test behavior across platforms
   - Mitigation: Create platform-agnostic test definitions with platform-specific adaptations

## Success Metrics

1. **Integration Completeness**:
   - All mobile tests run automatically in CI pipeline
   - Results are stored in benchmark database
   - Cross-platform analysis is automated

2. **Test Coverage**:
   - 95%+ of models tested on both Android and iOS
   - Key hardware accelerators (NPU, GPU) utilized in tests

3. **Report Quality**:
   - Comprehensive performance reports generated
   - Cross-platform insights available to developers
   - Performance regressions detected automatically

4. **User Experience**:
   - Dashboard provides clear mobile performance insights
   - Developers can request mobile testing for specific changes
   - Test results are available within reasonable timeframe

## Next Steps

1. Create device inventory and setup dedicated CI runners
2. ✓ Implement run_ci_benchmarks.py scripts for both platforms
3. ✓ Develop merge_benchmark_databases.py utility
4. ✓ Create check_mobile_regressions.py detection tool
5. ✓ Implement generate_mobile_dashboard.py visualization system
6. Test the complete workflow on a small set of models
7. Scale to full model coverage

## Implementation Status

As of May 2025, **all planned components have been implemented** (100% completion):

### Core CI Components
- ✓ Android Test Harness CI Benchmark Runner (`test/android_test_harness/run_ci_benchmarks.py`)
- ✓ iOS Test Harness CI Benchmark Runner (`test/ios_test_harness/run_ci_benchmarks.py`)
- ✓ Benchmark Database Merger Utility (`test/merge_benchmark_databases.py`)
- ✓ Mobile Performance Regression Detection Tool (`test/check_mobile_regressions.py`)
- ✓ Mobile Performance Dashboard Generator (`test/generate_mobile_dashboard.py`)
- ✓ Unit Tests for CI Integration Components (`test/test_mobile_ci_integration.py`)

### GitHub Actions Workflows
- ✓ Android CI Workflow (`test/android_test_harness/android_ci_workflow.yml`)
- ✓ iOS CI Workflow (`test/ios_test_harness/ios_ci_workflow.yml`)
- ✓ Cross-Platform Analysis Workflow (`test/mobile_cross_platform_workflow.yml`)
- ✓ CI Runner Setup Workflow (`test/setup_mobile_ci_runners_workflow.yml`)

### CI Runner Setup Tools
- ✓ Test Model Downloaders
  - ✓ Android Test Models (`test/android_test_harness/download_test_models.py`)
  - ✓ iOS Test Models (`test/ios_test_harness/download_test_models.py`)
- ✓ CI Runner Setup Utility (`test/setup_mobile_ci_runners.py`)
- ✓ CI Workflow Installation Tool (`test/setup_ci_workflows.py`)

### Runner Deployment Tools
- ✓ Comprehensive Mobile CI Runner Setup Guide (`test/MOBILE_CI_RUNNER_SETUP_GUIDE.md`)
- ✓ Android CI Runner Setup Script (`test/setup_android_ci_runner.sh`)
- ✓ iOS CI Runner Setup Script (`test/setup_ios_ci_runner.sh`)
- ✓ CI Installation Script (`test/install_ci_integration.sh`)

### Usage Instructions

To complete the CI/CD integration in your environment:

1. **Install CI Workflows**:
   ```bash
   ./test/install_ci_integration.sh
   ```

2. **Setup Self-Hosted Runners**:
   - For Android:
     ```bash
     ./test/setup_android_ci_runner.sh --register --token YOUR_GITHUB_TOKEN
     ```
   - For iOS:
     ```bash
     ./test/setup_ios_ci_runner.sh --register --token YOUR_GITHUB_TOKEN
     ```

3. **Run Initial Benchmarks**:
   - Trigger the workflows manually through GitHub Actions
   - Review results in the generated reports and dashboard

For detailed instructions, refer to the [Mobile CI Runner Setup Guide](test/MOBILE_CI_RUNNER_SETUP_GUIDE.md).

## Conclusion

The Mobile Edge CI/CD Integration Plan provides a comprehensive strategy for incorporating mobile testing into the automated testing infrastructure. By following this plan, the project will gain valuable insights into mobile performance, ensure cross-platform compatibility, and maintain high-quality standards across all supported platforms.