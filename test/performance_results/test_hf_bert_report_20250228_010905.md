# BERT Performance Test Results

Test run: 2025-02-28 01:09:05

## Implementation Status

| Platform | Status | Notes |
|----------|--------|-------|
| CPU | REAL | Successfully using real implementation |
| CUDA | REAL | Successfully using real implementation with GPU acceleration |
| OpenVINO | REAL | Successfully using real OpenVINO implementation |

**Model used:** /tmp/bert_test_model

## Performance Metrics

| Platform | Processing Speed | Memory Usage | Embedding Size | Batch Size |
|----------|------------------|--------------|----------------|------------|
| CPU | 0.0033s | N/A | 768 | 1 |
| CUDA | 0.0017s | 0.0 MB | 768 | 1 |
| OpenVINO | 0.0016s | N/A | 768 | 1 |

## Test Output Summary

```
connecting to master
connecting to master
Starting BERT test...
Attempting to use primary model: prajjwal1/bert-tiny
Primary model validation failed: prajjwal1/bert-tiny is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`
Trying alternative model: distilbert/distilbert-base-un...
```

