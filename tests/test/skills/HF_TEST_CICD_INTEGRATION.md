# HuggingFace Test CI/CD Integration Guide

This document provides guidelines for integrating the mock detection system into CI/CD workflows for HuggingFace model tests in the IPFS Accelerate Python Framework.

## Overview

The mock detection system provides critical visibility into whether tests are running with real model inference or mock objects. In CI/CD environments, it's particularly important to ensure that tests can run correctly with mock objects when necessary, while still providing validation of real inference in appropriate testing stages.

## CI/CD Integration Strategy

The CI/CD integration for HuggingFace tests follows a multi-stage approach:

1. **Syntax Validation**: Verify that all test files have syntactically correct implementation of mock detection
2. **Configuration Testing**: Test each file with multiple mock configurations to ensure they work correctly
3. **Real Inference Testing**: Run selected tests with real inference in environments with all dependencies available
4. **Performance Benchmarking**: Run benchmarks with real models on dedicated hardware (optional stage)

## GitHub Actions Integration

The repository includes a GitHub Actions workflow file at `test/skills/ci_templates/mock_detection_ci.yml` that implements the first two stages of the integration strategy.

### Workflow Stages

#### 1. Verify Mock Detection

This stage checks that all test files have the proper mock detection implementation:

```yaml
- name: Verify mock detection
  working-directory: test/skills
  run: |
    # Check mock detection in all files without making changes
    python verify_all_mock_detection.py --check-only
```

#### 2. Test Mock Configurations

This stage tests each file with multiple mock configurations to ensure they work correctly:

```yaml
- name: Run test with configuration
  working-directory: test/skills
  run: |
    # Set environment variables based on the configuration
    export MOCK_TORCH="False"
    export MOCK_TRANSFORMERS="False"
    export MOCK_TOKENIZERS="False"
    export MOCK_SENTENCEPIECE="False"
    
    if [[ "${{ matrix.config }}" == "no-torch" ]]; then
      export MOCK_TORCH="True"
    elif [[ "${{ matrix.config }}" == "no-transformers" ]]; then
      export MOCK_TRANSFORMERS="True"
    elif [[ "${{ matrix.config }}" == "no-tokenizers" ]]; then
      export MOCK_TOKENIZERS="True"
    elif [[ "${{ matrix.config }}" == "all-mock" ]]; then
      export MOCK_TORCH="True"
      export MOCK_TRANSFORMERS="True"
      export MOCK_TOKENIZERS="True"
      export MOCK_SENTENCEPIECE="True"
    fi
    
    # Run the test
    TEST_FILE="fixed_tests/test_hf_${{ matrix.model }}.py"
    ./run_test_with_mock_control.sh --file $TEST_FILE --capture
```

## GitLab CI Integration

For GitLab CI, a similar configuration can be created:

```yaml
stages:
  - validate
  - test

verify-mock-detection:
  stage: validate
  image: python:3.10
  script:
    - cd test/skills
    - python -m pip install pytest pytest-mock unittest-mock
    - python verify_all_mock_detection.py --check-only

test-mock-configurations:
  stage: test
  image: python:3.10
  parallel:
    matrix:
      - MODEL: [bert, gpt2, t5, vit]
        CONFIG: [all-real, no-torch, no-transformers, all-mock]
  script:
    - cd test/skills
    - python -m pip install pytest pytest-mock unittest-mock
    - |
      if [[ "$CONFIG" == "all-real" ]]; then
        pip install torch transformers tokenizers sentencepiece numpy
      else
        pip install numpy unittest-mock
      fi
    - |
      export MOCK_TORCH="False"
      export MOCK_TRANSFORMERS="False"
      export MOCK_TOKENIZERS="False"
      export MOCK_SENTENCEPIECE="False"
      
      if [[ "$CONFIG" == "no-torch" ]]; then
        export MOCK_TORCH="True"
      elif [[ "$CONFIG" == "no-transformers" ]]; then
        export MOCK_TRANSFORMERS="True"
      elif [[ "$CONFIG" == "all-mock" ]]; then
        export MOCK_TORCH="True"
        export MOCK_TRANSFORMERS="True"
        export MOCK_TOKENIZERS="True"
        export MOCK_SENTENCEPIECE="True"
      fi
      
      TEST_FILE="fixed_tests/test_hf_$MODEL.py"
      ./run_test_with_mock_control.sh --file $TEST_FILE --capture
  artifacts:
    paths:
      - test/skills/test_output/
    expire_in: 1 week
```

## Jenkins Integration

For Jenkins, a Jenkinsfile can be created with similar stages:

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.10'
        }
    }
    
    stages {
        stage('Verify Mock Detection') {
            steps {
                dir('test/skills') {
                    sh 'python -m pip install pytest pytest-mock unittest-mock'
                    sh 'python verify_all_mock_detection.py --check-only'
                }
            }
        }
        
        stage('Test Mock Configurations') {
            matrix {
                axes {
                    axis {
                        name 'MODEL'
                        values 'bert', 'gpt2', 't5', 'vit'
                    }
                    axis {
                        name 'CONFIG'
                        values 'all-real', 'no-torch', 'no-transformers', 'all-mock'
                    }
                }
                
                stages {
                    stage('Run Test') {
                        steps {
                            dir('test/skills') {
                                sh 'python -m pip install pytest pytest-mock unittest-mock'
                                script {
                                    if (env.CONFIG == 'all-real') {
                                        sh 'pip install torch transformers tokenizers sentencepiece numpy'
                                    } else {
                                        sh 'pip install numpy unittest-mock'
                                    }
                                    
                                    def mockTorch = 'False'
                                    def mockTransformers = 'False'
                                    def mockTokenizers = 'False'
                                    def mockSentencepiece = 'False'
                                    
                                    if (env.CONFIG == 'no-torch') {
                                        mockTorch = 'True'
                                    } else if (env.CONFIG == 'no-transformers') {
                                        mockTransformers = 'True'
                                    } else if (env.CONFIG == 'all-mock') {
                                        mockTorch = 'True'
                                        mockTransformers = 'True'
                                        mockTokenizers = 'True'
                                        mockSentencepiece = 'True'
                                    }
                                    
                                    sh "MOCK_TORCH=${mockTorch} MOCK_TRANSFORMERS=${mockTransformers} MOCK_TOKENIZERS=${mockTokenizers} MOCK_SENTENCEPIECE=${mockSentencepiece} ./run_test_with_mock_control.sh --file fixed_tests/test_hf_${env.MODEL}.py --capture"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'test/skills/test_output/*.txt', allowEmptyArchive: true
        }
    }
}
```

## Configuring CI/CD for Real Inference Testing

For environments where real inference testing is needed:

1. Create a dedicated runner or agent with GPU support if needed
2. Install all necessary dependencies (torch, transformers, tokenizers, sentencepiece)
3. Configure the test to run without mock environment variables
4. Use the `--capture` flag to save output for later analysis

Example GitHub Actions configuration:

```yaml
test-real-inference:
  runs-on: self-hosted-gpu
  
  steps:
    - name: Checkout code
      uses: actions/checkout@v3
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
        pip install transformers tokenizers sentencepiece numpy
        
    - name: Run tests with real inference
      working-directory: test/skills
      run: |
        for MODEL in bert gpt2 t5 vit; do
          ./run_test_with_mock_control.sh --file fixed_tests/test_hf_$MODEL.py --all-real --capture
        done
```

## Best Practices

1. **Always verify mock detection**: Include a verification step in all CI/CD pipelines
2. **Test multiple configurations**: Test with various mock configurations to ensure tests work in all environments
3. **Save test output**: Use the `--capture` flag to save test output for analysis
4. **Handle timeouts**: Set appropriate timeouts for tests, especially when running with real inference
5. **Provide meaningful artifacts**: Save reports and test output as artifacts for later analysis
6. **Conditional real inference**: Only run real inference tests on branches or PRs that require it
7. **Benchmark periodically**: Run benchmarks with real models periodically (e.g., nightly builds)

## Troubleshooting

If tests fail in CI/CD environments:

1. **Check mock detection**: Verify that mock detection is properly implemented
2. **Check environment variables**: Ensure environment variables are being passed correctly
3. **Check dependencies**: Verify that the necessary dependencies are installed
4. **Run verification script**: Run the verification script locally with the same configuration

## Summary

The mock detection system provides critical visibility and flexibility for running HuggingFace model tests in CI/CD environments. By properly integrating the system into CI/CD workflows, we can ensure that tests run correctly with mock objects when necessary, while still providing validation of real inference in appropriate testing stages.