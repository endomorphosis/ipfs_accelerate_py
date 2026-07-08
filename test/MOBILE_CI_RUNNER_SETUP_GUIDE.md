# Mobile CI Runner Setup Guide

**Date: May 2025**  
**Status: Complete**

## Overview

This guide provides comprehensive instructions for setting up self-hosted GitHub Actions runners for mobile testing in the IPFS Accelerate Python Framework. It covers both Android and iOS platforms, with detailed steps for configuring physical devices, setting up the environment, and validating connections.

## Prerequisites

### For All Platforms

- GitHub account with administrator access to the repository
- Python 3.9 or newer
- Git client

### For Android Testing

- Ubuntu 20.04 or newer (recommended) or macOS
- Android SDK installed
- Java 11 or newer
- ADB (Android Debug Bridge)
- Physical Android device or emulator
- USB cable for physical device connection

### For iOS Testing

- macOS 12.0 (Monterey) or newer
- Xcode 13.0 or newer
- Physical iOS device or simulator
- Apple Developer account for physical device testing
- USB cable for physical device connection

## Installation Process

The setup process involves several steps, which can be largely automated using the provided utilities.

### 1. Prepare the Host Machine

Start by cloning the repository and preparing the environment on your host machine.

```bash
# Clone the repository
git clone https://github.com/yourusername/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install base dependencies
pip install -r requirements.txt

# Install platform-specific dependencies
# For Android:
pip install -r test/android_test_harness/requirements.txt

# For iOS:
pip install -r test/ios_test_harness/requirements.txt
```

### 2. Verify the Environment

Use the setup utility to check if your environment is properly configured for mobile testing.

```bash
# Check Android environment
python test/setup_mobile_ci_runners.py --action check --platform android --verbose

# Check iOS environment
python test/setup_mobile_ci_runners.py --action check --platform ios --verbose
```

If any components are missing, you'll see warnings in the output. Follow the guided instructions to install missing components.

### 3. Configure the Environment

Once you've verified and installed all required components, configure the environment for mobile testing.

```bash
# Configure Android environment
python test/setup_mobile_ci_runners.py --action configure --platform android --verbose

# Configure iOS environment
python test/setup_mobile_ci_runners.py --action configure --platform ios --verbose
```

This will:
- Install test dependencies
- Create necessary directories
- Download test models
- Configure environment variables

### 4. Connect Physical Devices

#### Android Device Setup

1. Enable Developer Options on your Android device:
   - Go to Settings > About phone
   - Tap "Build number" 7 times to enable developer mode
   - Go back to Settings > System > Developer options
   - Enable "USB debugging"

2. Connect the device via USB and authorize the connection:
   - Connect the device to your computer
   - When prompted on the device, allow USB debugging
   - Verify the connection with ADB:
     ```bash
     adb devices
     ```

3. Validate device connectivity:
   ```bash
   python test/setup_mobile_ci_runners.py --action verify --platform android --device-id DEVICE_ID --verbose
   ```
   Replace `DEVICE_ID` with the ID shown in `adb devices` (e.g., `emulator-5554` or a serial number).

#### iOS Device Setup

1. Enable Developer Mode on your iOS device (iOS 16+):
   - Go to Settings > Privacy & Security
   - Scroll down and tap "Developer Mode"
   - Toggle Developer Mode on and restart the device

2. Connect the device via USB:
   - Connect the device to your macOS computer
   - When prompted on the device, trust the computer
   - Open Xcode to verify the device is recognized

3. Validate device connectivity:
   ```bash
   python test/setup_mobile_ci_runners.py --action verify --platform ios --device-id DEVICE_ID --verbose
   ```
   Replace `DEVICE_ID` with the UDID of your iOS device, which you can find in Xcode or by running:
   ```bash
   xcrun xctrace list devices
   ```

### 5. Register GitHub Actions Runner

#### Manual Registration (Recommended for First-Time Setup)

1. Go to your GitHub repository
2. Navigate to Settings > Actions > Runners
3. Click "New self-hosted runner"
4. Select the appropriate OS (Linux for Android-only, macOS for iOS or both)
5. Follow the instructions to download and configure the runner

When configuring labels, add the following:
- For Android: `android,mobile,physical`
- For iOS: `ios,mobile,physical`
- For both: `android,ios,mobile,physical`

You can also add device-specific labels like `pixel6` or `iphone13`.

#### Using the Setup Utility

Our utility can help register the runner with the appropriate labels:

```bash
# Set the GitHub token and repository URL as environment variables
export GITHUB_TOKEN="your_token_here"
export GITHUB_REPOSITORY="username/repo"

# Register Android runner
python test/setup_mobile_ci_runners.py --action register --platform android --verbose

# Register iOS runner
python test/setup_mobile_ci_runners.py --action register --platform ios --verbose
```

### 6. Install GitHub Actions Workflows

Install the workflow files to the .github/workflows directory:

```bash
python test/setup_ci_workflows.py --install --verbose
```

This copies the workflow files from their current locations to the GitHub Actions directory.

### 7. Verify Complete Setup

Run a comprehensive verification to ensure everything is working correctly:

```bash
# For Android
python test/android_test_harness/run_ci_benchmarks.py \
  --device-id YOUR_DEVICE_ID \
  --output-db benchmark_results.duckdb \
  --verbose

# For iOS
python test/ios_test_harness/run_ci_benchmarks.py \
  --device-id YOUR_DEVICE_ID \
  --output-db benchmark_results.duckdb \
  --verbose \
  --simulator  # Remove this flag for physical devices
```

## Advanced Configuration

### Multiple Devices

You can set up multiple devices by registering multiple runners or by connecting multiple devices to a single runner.

#### Multiple Runners (Recommended)

Register one runner per device, following the steps above for each device, with unique labels.

#### Multiple Devices on One Runner

If you want to connect multiple devices to a single runner:

1. Connect all devices to the host machine
2. Verify each device:
   ```bash
   # For Android
   adb devices
   
   # For iOS
   xcrun xctrace list devices
   ```
3. The CI scripts will detect all connected devices when run without a specific device ID

### Persistent Device Connections

For CI environments where devices might disconnect:

#### Android Devices

Create a udev rule to ensure consistent permissions:

```bash
# Create a udev rules file
sudo nano /etc/udev/rules.d/51-android.rules

# Add this line for each device (adjust vendor ID as needed)
SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0666", GROUP="plugdev"

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

#### iOS Devices

On macOS, use the following to prevent device sleep:

```bash
# Prevent Mac from sleeping
sudo pmset -a disablesleep 1

# Keep USB devices powered
sudo pmset -a disksleep 0
```

### Network Devices

For devices connected over network:

#### Android Network Debugging

```bash
# Connect device over network
adb connect IP_ADDRESS:5555

# Verify connection
adb devices
```

#### iOS Network Debugging

iOS wireless debugging requires Xcode 12+ and iOS 14+:

1. Connect device via USB
2. In Xcode, go to Window > Devices and Simulators
3. Select your device
4. Check "Connect via network"
5. Disconnect the USB cable

## Troubleshooting

### Common Android Issues

1. **Device not detected**
   - Ensure USB debugging is enabled
   - Try a different USB cable or port
   - Run `adb kill-server && adb start-server`

2. **Permission denied**
   - Setup udev rules as shown above
   - Run ADB as root: `sudo adb devices`

3. **Benchmark errors**
   - Check model compatibility with your device
   - Increase timeout for larger models

### Common iOS Issues

1. **Device not detected**
   - Ensure the device is unlocked
   - Trust the computer on the device
   - Check the Lightning/USB-C cable

2. **Unauthorized access**
   - Open Xcode and verify device connection
   - Check developer team settings in Xcode

3. **CoreML model loading errors**
   - Verify model format compatibility
   - Check device iOS version compatibility

## CI Workflow Monitoring

Once your self-hosted runners are set up and workflows are installed, you can monitor them:

1. On your GitHub repository, go to the Actions tab
2. You'll see all configured workflows
3. Click on a workflow to see its execution history
4. For each run, you can see logs and download artifacts

## Security Considerations

Self-hosted runners have access to your environment, so consider these security practices:

1. Use dedicated machines for runners
2. Keep the runner environment updated
3. Use the least privileged account necessary
4. Consider network isolation for the runner machines
5. Regularly review the list of registered runners

## Maintenance

### Updating the Environment

Periodically update the test environment:

```bash
# Pull latest changes
git pull

# Update dependencies
pip install -r requirements.txt
pip install -r test/android_test_harness/requirements.txt
pip install -r test/ios_test_harness/requirements.txt

# Update test models
python test/android_test_harness/download_test_models.py
python test/ios_test_harness/download_test_models.py
```

### Monitoring Runner Health

GitHub provides runner health metrics:

1. Go to Settings > Actions > Runners
2. Check the status of each runner
3. Look for errors or offline status

## Example: Complete Setup Script

Here's a comprehensive script that performs a complete setup for Android runners on Linux:

```bash
#!/bin/bash
# Android CI Runner Setup Script

# Install dependencies
sudo apt-get update
sudo apt-get install -y openjdk-11-jdk android-tools-adb python3 python3-pip git

# Clone repository
git clone https://github.com/yourusername/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install Python dependencies
pip3 install -r requirements.txt
pip3 install -r test/android_test_harness/requirements.txt

# Configure environment
python3 test/setup_mobile_ci_runners.py --action configure --platform android --verbose

# Connect and verify device
adb devices
python3 test/setup_mobile_ci_runners.py --action verify --platform android --verbose

# Install workflows
python3 test/setup_ci_workflows.py --install --verbose

# Run a test benchmark
python3 test/android_test_harness/run_ci_benchmarks.py --output-db benchmark_results.duckdb --verbose

echo "Setup complete! Register this machine as a GitHub Actions runner to complete the integration."
```

For macOS and iOS, a similar script would be:

```bash
#!/bin/bash
# iOS CI Runner Setup Script

# Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install dependencies
brew install python@3.9

# Clone repository
git clone https://github.com/yourusername/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install Python dependencies
pip3 install -r requirements.txt
pip3 install -r test/ios_test_harness/requirements.txt

# Configure environment
python3 test/setup_mobile_ci_runners.py --action configure --platform ios --verbose

# Verify device connectivity
xcrun xctrace list devices
python3 test/setup_mobile_ci_runners.py --action verify --platform ios --verbose

# Install workflows
python3 test/setup_ci_workflows.py --install --verbose

# Run a test benchmark
python3 test/ios_test_harness/run_ci_benchmarks.py --simulator --output-db benchmark_results.duckdb --verbose

echo "Setup complete! Register this machine as a GitHub Actions runner to complete the integration."
```

## Conclusion

Following this guide will give you a fully configured CI environment for mobile testing with physical devices. The self-hosted runners will automatically execute benchmarks, analyze performance, and identify regressions across both Android and iOS platforms.

For any issues or enhancements, please file an issue on the repository or contact the mobile testing team.