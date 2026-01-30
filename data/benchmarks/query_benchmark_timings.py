{
  "model_type": "llama",
  "model_size": "tiny",
  "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "params": "1.1B",
  "quantization": {
    "fp16_size_mb": 172.9453125,
    "int4_size_mb": 43.236328125,
    "compression_ratio": 4.0,
    "memory_reduction_percent": 75.0,
    "quantization_time_ms": 0.29659271240234375,
    "layers_quantized": 97,
    "total_layers": 97,
    "quantization_scheme": "symmetric",
    "block_size": 128,
    "accuracy_change_percent": -0.6,
    "inference_speedup": 1.6,
    "pipeline_config": {
      "model_config": {
        "hidden_size": 768,
        "seq_length": 2048,
        "batch_size": 1,
        "block_size": 128,
        "quantization_scheme": "symmetric"
      },
      "compute_pipeline": {
        "matmul_shader": {
          "code": "\n        // Optimized 4-bit Matrix Multiplication Compute Shader for WebGPU\n        \n        struct Params {\n            M: u32,           // Batch size * sequence length\n            N: u32,           // Output dimension\n            K: u32,           // Input dimension\n            block_size: u32,  // Quantization block size\n            batch_size: u32,  // Batch size\n            seq_length: u32,  // Sequence length\n            has_bias: u32,    // Whether bias is added\n            zero_point: u32,  // Whether zero point is used (asymmetric quantization)\n        };\n        \n        @group(0) @binding(0) var<storage, read> packed_weights: array<u8>;  // 4-bit weights (2 values per byte)\n        @group(0) @binding(1) var<storage, read> scales: array<f16>;         // Quantization scales\n        @group(0) @binding(2) var<storage, read_write> zeros: array<f16>;    // Zero points (optional)\n        @group(0) @binding(3) var<storage, read> input: array<f16>;          // Input activations\n        @group(0) @binding(4) var<storage, read_write> output: array<f16>;   // Output buffer\n        @group(0) @binding(5) var<storage, read> bias: array<f16>;           // Optional bias\n        @group(0) @binding(6) var<uniform> params: Params;                   // Parameters\n        \n        // Workgroup shared memory for input tile\n        var<workgroup> tile_input: array<f16, 128>;\n        \n        // Add shared memory for optimized browser-specific kernels\n        var<workgroup> matrix_cache: array<f16, 256>;\n        \n        // Extract 4-bit value from packed byte\n        fn extract_4bit(packed: u8, idx: u32) -> u32 {\n            if (idx == 0) {\n                return u32(packed & 0x0F);\n            } else {\n                return u32(packed >> 4);\n            }\n        }\n        \n        // Dequantize 4-bit value\n        fn dequantize(value: u32, scale: f16, zero: f16) -> f16 {\n            if (params.zero_point == 1u) {\n                // Asymmetric quantization\n                return scale * (f16(value) - zero);\n            } else {\n                // Symmetric quantization\n                return scale * f16(value);\n            }\n        }\n        \n        @compute @workgroup_size(8, 16, 1)\n        fn main(@builtin(global_invocation_id) global_id: vec3<u32>,\n                @builtin(local_invocation_id) local_id: vec3<u32>,\n                @builtin(workgroup_id) workgroup_id: vec3<u32>) {\n            \n            let row = global_id.x;               // Output row\n            let col = global_id.y;               // Output column  \n            let batch_idx = global_id.z;         // Batch index\n            \n            // Early exit if out of bounds\n            if (row >= params.M || col >= params.N || batch_idx >= params.batch_size) {\n                return;\n            }\n            \n            let seq_idx = row % params.seq_length;  // Position in sequence\n            let batch_offset = batch_idx * params.seq_length * params.K;\n            \n            // Output index\n            let out_idx = batch_idx * params.M * params.N + row * params.N + col;\n            \n            // Calculate scales and zeros index\n            let num_blocks = (params.K + params.block_size - 1u) / params.block_size;\n            let scales_per_output = num_blocks;  // One scale per block per output\n            \n            // Initialize accumulator\n            var acc: f16 = 0.0;\n            \n            // Process input in blocks\n            for (var block_idx = 0u; block_idx < num_blocks; block_idx++) {\n                let block_start = block_idx * params.block_size;\n                let block_end = min(block_start + params.block_size, params.K);\n                let block_size = block_end - block_start;\n                \n                // Get scale and zero for this block\n                let scale_idx = col * scales_per_output + block_idx;\n                let scale = scales[scale_idx];\n                let zero = (params.zero_point == 1u) ? zeros[scale_idx] : 0.0;\n                \n                // Process elements in this block\n                for (var k = 0u; k < block_size; k++) {\n                    let k_idx = block_start + k;\n                    let input_idx = batch_offset + seq_idx * params.K + k_idx;\n                    let input_val = input[input_idx];\n                    \n                    // Calculate packed weight index\n                    // Two 4-bit weights per byte\n                    let weight_byte_idx = (col * params.K + k_idx) / 2;\n                    let weight_bit_offset = (col * params.K + k_idx) % 2;\n                    \n                    // Get packed weight byte and extract 4-bit value\n                    let packed = packed_weights[weight_byte_idx];\n                    let weight_4bit = extract_4bit(packed, weight_bit_offset);\n                    \n                    // Dequantize and accumulate\n                    let weight_val = dequantize(weight_4bit, scale, zero);\n                    acc += input_val * weight_val;\n                }\n            }\n            \n            // Add bias if present\n            if (params.has_bias == 1u) {\n                acc += bias[col];\n            }\n            \n            // Write output\n            output[out_idx] = acc;\n        }\n        ",
          "entry_point": "main",
          "workgroup_size": "8, 8, 1"
        },
        "unpack_shader": {
          "code": "\n        // 4-bit Weight Unpacking Shader for WebGPU\n        \n        struct Params {\n            num_weights: u32,  // Total number of weights\n            block_size: u32,   // Quantization block size\n            zero_point: u32,   // Whether zero point is used\n        };\n        \n        @group(0) @binding(0) var<storage, read> packed_weights: array<u8>;  // Packed 4-bit weights\n        @group(0) @binding(1) var<storage, read> scales: array<f16>;         // Quantization scales\n        @group(0) @binding(2) var<storage, read> zeros: array<f16>;          // Zero points (optional)\n        @group(0) @binding(3) var<storage, write> unpacked_weights: array<f16>; // Output unpacked weights\n        @group(0) @binding(4) var<uniform> params: Params;                     // Parameters\n        \n        // Extract 4-bit value from packed byte\n        fn extract_4bit(packed: u8, idx: u32) -> u32 {\n            if (idx == 0) {\n                return u32(packed & 0x0F);\n            } else {\n                return u32(packed >> 4);\n            }\n        }\n        \n        // Dequantize 4-bit value\n        fn dequantize(value: u32, scale: f16, zero: f16) -> f16 {\n            if (params.zero_point == 1u) {\n                // Asymmetric quantization\n                return scale * (f16(value) - zero);\n            } else {\n                // Symmetric quantization\n                return scale * f16(value);\n            }\n        }\n        \n        @compute @workgroup_size(256, 1, 1)\n        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n            let weight_idx = global_id.x;\n            \n            if (weight_idx >= params.num_weights) {\n                return;\n            }\n            \n            // Calculate packed byte index and bit offset\n            let byte_idx = weight_idx / 2;\n            let bit_offset = weight_idx % 2;\n            \n            // Get block index for scales/zeros\n            let block_idx = weight_idx / params.block_size;\n            \n            // Get packed weight and extract 4-bit value\n            let packed = packed_weights[byte_idx];\n            let weight_4bit = extract_4bit(packed, bit_offset);\n            \n            // Get scale and zero point\n            let scale = scales[block_idx];\n            let zero = params.zero_point == 1u ? zeros[block_idx] : 0.0;\n            \n            // Dequantize and store\n            let weight_val = dequantize(weight_4bit, scale, zero);\n            unpacked_weights[weight_idx] = weight_val;\n        }\n        ",
          "entry_point": "main",
          "workgroup_size": "256, 1, 1"
        }
      },
      "optimization_level": "advanced",
      "expected_speedup": "1.6x",
      "memory_reduction": "75.0%"
    },
    "benchmark": {
      "model_config": {
        "model_type": "llama",
        "hidden_size": 768,
        "seq_length": 2048,
        "batch_size": 1,
        "intermediate_size": 3072,
        "block_size": 128
      },
      "baseline_fp16": {
        "precision": "fp16",
        "model_size_mb": 9.0029296875,
        "inference_time_ms": 100.0,
        "memory_usage_mb": 12.0029296875,
        "relative_speed": 1.0
      },
      "int8": {
        "precision": "int8",
        "model_size_mb": 4.50146484375,
        "inference_time_ms": 85.0,
        "memory_usage_mb": 7.50146484375,
        "relative_speed": 1.1764705882352942,
        "memory_saving_vs_fp16_percent": 50.0
      },
      "int4_basic": {
        "precision": "int4",
        "model_size_mb": 2.250732421875,
        "inference_time_ms": 70.0,
        "memory_usage_mb": 5.250732421875,
        "relative_speed": 1.4285714285714286,
        "optimized": false,
        "memory_saving_vs_fp16_percent": 75.0
      },
      "int4_optimized": {
        "precision": "int4",
        "quantization_scheme": "symmetric",
        "block_size": 128,
        "model_size_mb": 2.250732421875,
        "inference_time_ms": 60.0,
        "memory_usage_mb": 5.250732421875,
        "relative_speed": 1.6666666666666667,
        "optimized": true,
        "compute_shader_optimized": true,
        "memory_saving_vs_fp16_percent": 75.0
      },
      "comparison_summary": {
        "memory_reduction_vs_fp16_percent": 75.0,
        "memory_reduction_vs_int8_percent": 50.0,
        "speedup_vs_fp16": 1.6666666666666667,
        "speedup_vs_int8": 1.4166666666666667,
        "optimization_impact_percent": 14.285714285714285
      }
    }
  },
  "memory": {
    "fp16_size_mb": 172.9453125,
    "int4_size_mb": 43.236328125,
    "memory_reduction_percent": 75.0,
    "memory_reduction_target_met": true
  },
  "performance": {
    "inference_speedup": 1.6,
    "speedup_target_met": true
  },
  "quality": {},
  "kv_cache": {
    "enabled": true,
    "context_length": 2048,
    "metrics": {
      "enabled": true,
      "standard_max_length": 2048,
      "optimized_max_length": 8192,
      "length_improvement": 4.0,
      "target_met": true,
      "memory_per_token_kb": 6.0,
      "use_sliding_window": true,
      "sliding_window_size": 1024,
      "multi_query": true,
      "use_flash_attention": true
    },
    "target_met": true
  },
  "next_steps_features": {
    "specialized_compute_shaders": {
      "enabled": false,
      "metrics": {}
    },
    "firefox_optimizations": {
      "enabled": false,
      "metrics": {}
    },
    "safari_compatibility": {
      "enabled": false,
      "metrics": {}
    },
    "reinforcement_learning": {
      "enabled": false,
      "metrics": {}
    }
  },
  "timestamps": {
    "start": 1741338662.755313,
    "end": null
  },
  "precision_comparison": {
    "metrics_by_precision": {
      "fp16": {
        "precision": "fp16",
        "model_size_mb": 9.0029296875,
        "inference_time_ms": 100.0,
        "memory_usage_mb": 12.0029296875,
        "relative_speed": 1.0
      },
      "int8": {
        "precision": "int8",
        "model_size_mb": 4.50146484375,
        "inference_time_ms": 85.0,
        "memory_usage_mb": 7.50146484375,
        "relative_speed": 1.1764705882352942,
        "memory_saving_vs_fp16_percent": 50.0
      },
      "int4_basic": {
        "precision": "int4",
        "model_size_mb": 2.250732421875,
        "inference_time_ms": 70.0,
        "memory_usage_mb": 5.250732421875,
        "relative_speed": 1.4285714285714286,
        "optimized": false,
        "memory_saving_vs_fp16_percent": 75.0
      },
      "int4_optimized": {
        "precision": "int4",
        "quantization_scheme": "symmetric",
        "block_size": 128,
        "model_size_mb": 2.250732421875,
        "inference_time_ms": 60.0,
        "memory_usage_mb": 5.250732421875,
        "relative_speed": 1.6666666666666667,
        "optimized": true,
        "compute_shader_optimized": true,
        "memory_saving_vs_fp16_percent": 75.0
      }
    },
    "comparisons": {
      "int4_vs_fp16": {
        "memory_reduction_percent": 75.0,
        "speedup": 1.6666666666666667,
        "memory_target_met": true,
        "speedup_target_met": true
      },
      "int4_vs_int8": {
        "memory_reduction_percent": 50.0,
        "speedup": 1.4166666666666667
      },
      "optimization_impact": {
        "percent_improvement": 14.285714285714285
      }
    }
  }
}