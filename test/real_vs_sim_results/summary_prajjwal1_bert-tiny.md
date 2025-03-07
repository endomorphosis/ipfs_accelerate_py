# Hardware Test Report for prajjwal1/bert-tiny

Generated: 2025-03-06 21:39:24

## Results Summary

| Platform | Status | Implementation Type | Execution Time |
|----------|--------|---------------------|---------------|
| cuda | ❌ Failed | UNKNOWN | 0.000 sec |
| webgpu | ❌ Failed | UNKNOWN | 0.000 sec |

## Implementation Issues

### CUDA

**Error**: TestCase.__init__() got an unexpected keyword argument 'model_id'

**Traceback**:
```
Traceback (most recent call last):
  File "/home/barberb/ipfs_accelerate_py/test/test_single_model_hardware.py", line 180, in test_model_on_platform
    test_instance = TestClass(model_id=model_name)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: TestCase.__init__() got an unexpected keyword argument 'model_id'
```

### WEBGPU

**Error**: TestCase.__init__() got an unexpected keyword argument 'model_id'

**Traceback**:
```
Traceback (most recent call last):
  File "/home/barberb/ipfs_accelerate_py/test/test_single_model_hardware.py", line 180, in test_model_on_platform
    test_instance = TestClass(model_id=model_name)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: TestCase.__init__() got an unexpected keyword argument 'model_id'
```


## Recommendations

### Fix Implementation Issues

- Fix prajjwal1/bert-tiny implementation on cuda
- Fix prajjwal1/bert-tiny implementation on webgpu

