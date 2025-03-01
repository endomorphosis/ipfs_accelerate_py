# Critical Hugging Face Model Test Report

Generated: 2025-03-01T15:09:24.215290

## Summary

- **Total Models**: 3
- **Tested**: 3 (100.0% of total)
- **Passed**: 0 (0.0% of tested)
- **Failed**: 3 (100.0% of tested)
- **Missing Tests**: 0 (0.0% of total)

## Model Results

| Model | Status | CPU | CUDA | OpenVINO | Elapsed Time |
|-------|--------|-----|------|----------|---------------|
| canine | ❌ Failed | N/A | N/A | N/A | 4.5s |
| zoedepth | ❌ Failed | N/A | N/A | N/A | 4.6s |
| visual_bert | ❌ Failed | N/A | N/A | N/A | 4.6s |

## Failure Details

### canine

- **Status**: Failed
- **Error**: `NameError: name 'class_name' is not defined`

```python
Traceback (most recent call last):
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_canine.py", line 89, in <module>
    print(f"Warning: {class_name} module not found, using mock implementation")
                      ^^^^^^^^^^
NameError: name 'class_name' is not defined
```

### zoedepth

- **Status**: Failed
- **Error**: `NameError: name 'class_name' is not defined`

```python
Traceback (most recent call last):
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_zoedepth.py", line 89, in <module>
    print(f"Warning: {class_name} module not found, using mock implementation")
                      ^^^^^^^^^^
NameError: name 'class_name' is not defined
```

### visual_bert

- **Status**: Failed
- **Error**: `NameError: name 'class_name' is not defined`

```python
Traceback (most recent call last):
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_visual_bert.py", line 89, in <module>
    print(f"Warning: {class_name} module not found, using mock implementation")
                      ^^^^^^^^^^
NameError: name 'class_name' is not defined
```

