# Self-Hosted Runner Docker Configuration Update

**Date**: October 24, 2025  
**Status**: Completed

## Summary

Added comprehensive documentation and updated existing CI/CD guides to address the critical requirement that self-hosted GitHub Actions runners must have the runner user added to the docker group when running Docker-based tests.

## Changes Made

### 1. New Documentation Created

Created comprehensive guide at `docs/SELF_HOSTED_RUNNER_SETUP.md` covering:

- **Docker Group Configuration**: Step-by-step instructions for adding runner users to docker group
- **Runner Installation**: Complete setup process for GitHub Actions self-hosted runners
- **Hardware-Specific Setup**: Instructions for NVIDIA CUDA, AMD ROCm, and Intel OpenVINO
- **Security Considerations**: Best practices for secure runner deployment
- **Troubleshooting**: Common issues and solutions
- **Monitoring and Maintenance**: Guidelines for runner health and updates

Key command documented:
```bash
sudo usermod -aG docker <runner-user>
```

### 2. Updated CI/CD Integration Guides

Updated the following files to reference the docker group requirement:

1. **test/distributed_testing/CI_CD_INTEGRATION_GUIDE.md**
   - Added "Self-Hosted Runner Setup" section
   - Referenced comprehensive setup guide
   - Included quick-start docker group configuration

2. **data/benchmarks/BENCHMARK_CI_INTEGRATION.md**
   - Added "Self-Hosted Runner Requirements" section at the beginning
   - Included docker group configuration command
   - Referenced comprehensive setup guide

3. **test/docs/CICD_INTEGRATION_GUIDE.md**
   - Added "Self-Hosted Runner Setup" section
   - Included docker group and service restart commands
   - Referenced comprehensive setup guide with hardware configurations

4. **scripts/generators/test_generator/CI_CD_INTEGRATION.md**
   - Added "Self-Hosted Runner Configuration" section
   - Included docker group configuration
   - Referenced comprehensive setup guide

### 3. Updated Main README

Updated `README.md` to include the new self-hosted runner setup guide in the documentation section:

- Added link to `docs/SELF_HOSTED_RUNNER_SETUP.md` in the "Specialized Enterprise Guides" section
- Description: "Complete guide for setting up GitHub Actions self-hosted runners with Docker and hardware acceleration"

## Documentation Structure

```
docs/
  └── SELF_HOSTED_RUNNER_SETUP.md (NEW)
      ├── Docker Group Configuration
      ├── Runner Installation
      ├── Hardware-Specific Setup
      ├── Security Considerations
      ├── Workflow Configuration
      ├── Troubleshooting
      └── Monitoring and Maintenance

test/
  ├── distributed_testing/
  │   └── CI_CD_INTEGRATION_GUIDE.md (UPDATED)
  └── docs/
      └── CICD_INTEGRATION_GUIDE.md (UPDATED)

data/benchmarks/
  └── BENCHMARK_CI_INTEGRATION.md (UPDATED)

scripts/generators/
  └── test_generator/
      └── CI_CD_INTEGRATION.md (UPDATED)

README.md (UPDATED)
```

## Key Requirements Documented

### Essential Docker Configuration

```bash
# Add runner user to docker group
sudo usermod -aG docker <runner-user>

# Restart runner service to apply changes
sudo systemctl restart actions-runner

# Verify group membership
groups <runner-user>

# Test docker access (should work without sudo)
docker ps
```

### Hardware-Specific Additions

**NVIDIA CUDA:**
```bash
sudo usermod -aG docker <runner-user>
sudo usermod -aG video <runner-user>
```

**AMD ROCm:**
```bash
sudo usermod -aG docker <runner-user>
sudo usermod -aG render <runner-user>
sudo usermod -aG video <runner-user>
```

**Intel OpenVINO:**
```bash
sudo usermod -aG docker <runner-user>
sudo usermod -aG render <runner-user>
```

## Benefits

1. **Prevents Permission Errors**: Eliminates "permission denied" errors when tests try to access Docker
2. **Enables Container Testing**: Allows workflows to run tests in Docker containers
3. **Hardware Testing**: Supports GPU/NPU container testing with proper group memberships
4. **Documentation Clarity**: All CI/CD guides now reference this critical requirement
5. **Comprehensive Coverage**: Includes troubleshooting for common issues

## Workflow Integration

Example workflow configuration documented:

```yaml
name: Hardware Tests
on: [push, pull_request]

jobs:
  cpu-tests:
    runs-on: [self-hosted, linux, docker, cpu-only]
    steps:
      - uses: actions/checkout@v4
      - name: Run CPU tests
        run: |
          docker run --rm -v $PWD:/workspace \
            python:3.10 python /workspace/test/run_tests.py --hardware cpu

  gpu-tests:
    runs-on: [self-hosted, linux, docker, cuda]
    steps:
      - uses: actions/checkout@v4
      - name: Run GPU tests
        run: |
          docker run --rm --gpus all -v $PWD:/workspace \
            nvidia/cuda:11.8.0-runtime-ubuntu22.04 \
            python /workspace/test/run_tests.py --hardware cuda
```

## Security Considerations Documented

- Repository access restrictions for self-hosted runners
- Secrets management best practices
- Network isolation recommendations
- Docker security configurations
- Runner isolation techniques

## Troubleshooting Guide Included

Common issues covered:
- Docker permission denied errors
- Runner not picking up jobs
- GPU not accessible in containers
- Disk space issues
- Service management problems

Each issue includes:
- Error description
- Root cause
- Step-by-step solution
- Verification steps

## Next Steps

Users setting up self-hosted runners should:

1. Read `docs/SELF_HOSTED_RUNNER_SETUP.md` for complete setup instructions
2. Execute the docker group configuration commands
3. Configure hardware-specific settings if needed
4. Verify runner connectivity and Docker access
5. Monitor runner health and maintain regularly

## References

- Main setup guide: `docs/SELF_HOSTED_RUNNER_SETUP.md`
- GitHub Actions documentation: https://docs.github.com/en/actions/hosting-your-own-runners
- Docker post-installation: https://docs.docker.com/engine/install/linux-postinstall/
- NVIDIA Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- AMD ROCm: https://rocm.docs.amd.com/
- Intel OpenVINO: https://docs.openvino.ai/

## Conclusion

The documentation now comprehensively covers the critical requirement for self-hosted runners to have proper Docker group permissions. All CI/CD integration guides reference this requirement, and a detailed setup guide provides step-by-step instructions for various hardware configurations.
