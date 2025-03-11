/**
 * Converted from Python: create_compatibility_tables.py
 * Conversion date: 2025-03-11 04:08:54
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create Compatibility Matrix Database Tables

This script creates the necessary database tables for the comprehensive model compatibility matrix.
It sets up tables for storing model information, hardware platforms, compatibility status,
and hardware recommendations.

Usage:
  python scripts/create_compatibility_tables.py --db-path PATH []]]],,,,--sample-data]
  ,
Options:
  --db-path PATH     Path to DuckDB database
  --sample-data      Include sample data in the tables
  """

  import * as $1
  import * as $1
  import * as $1
  import * as $1

# Configure logging
  logging.basicConfig())))))
  level=logging.INFO,
  format='%())))))asctime)s - %())))))name)s - %())))))levelname)s - %())))))message)s'
  )
  logger = logging.getLogger())))))"create_tables")

$1($2) {
  """Parse command line arguments."""
  parser = argparse.ArgumentParser())))))description="Create compatibility matrix database tables")
  parser.add_argument())))))"--db-path", required=true, help="Path to DuckDB database")
  parser.add_argument())))))"--sample-data", action="store_true", help="Include sample data in the tables")
  return parser.parse_args()))))))

}
$1($2) {
  """Create the necessary database tables."""
  try ${$1} catch($2: $1) {
    logger.error())))))`$1`)
    raise

  }
$1($2) {
  """Insert sample data into the tables."""
  try {
    # Insert sample hardware platforms
    conn.execute())))))"""
    INSERT INTO hardware_platforms ())))))hardware_type, description, vendor)
    VALUES 
    ())))))'CUDA', 'NVIDIA CUDA GPU acceleration', 'NVIDIA'),
    ())))))'ROCm', 'AMD ROCm GPU acceleration', 'AMD'),
    ())))))'MPS', 'Apple Metal Performance Shaders', 'Apple'),
    ())))))'OpenVINO', 'Intel OpenVINO acceleration', 'Intel'),
    ())))))'Qualcomm', 'Qualcomm AI Engine', 'Qualcomm'),
    ())))))'WebNN', 'Web Neural Network API', 'W3C'),
    ())))))'WebGPU', 'Web GPU API', 'W3C')
    ON CONFLICT ())))))hardware_type) DO NOTHING
    """)
    logger.info())))))"Inserted sample hardware platforms")
    
  }
    # Insert sample models
    conn.execute())))))"""
    INSERT INTO models ())))))model_name, model_type, model_family, modality, parameters_million, is_key_model)
    VALUES 
    ())))))'bert-base-uncased', 'BertModel', 'BERT', 'text', 110, TRUE),
    ())))))'t5-small', 'T5Model', 'T5', 'text', 60, TRUE),
    ())))))'vit-base-patch16-224', 'ViTModel', 'ViT', 'vision', 86, TRUE),
    ())))))'whisper-tiny', 'WhisperModel', 'Whisper', 'audio', 39, TRUE),
    ())))))'clip-vit-base-patch32', 'CLIPModel', 'CLIP', 'multimodal', 150, TRUE),
    ())))))'llava-1.5-7b', 'LlavaModel', 'LLaVA', 'multimodal', 7000, TRUE)
    ON CONFLICT DO NOTHING
    """)
    logger.info())))))"Inserted sample models")
    
}
    # Get model IDs
    model_ids = conn.execute())))))"""
    SELECT id, model_name FROM models WHERE model_name IN 
    ())))))'bert-base-uncased', 't5-small', 'vit-base-patch16-224', 'whisper-tiny', 'clip-vit-base-patch32', 'llava-1.5-7b')
    """).fetchdf()))))))
    
}
    # Insert sample compatibility data
    for _, row in model_ids.iterrows())))))):
      model_id = row[]]]],,,,'id'],
      model_name = row[]]]],,,,'model_name']
      ,
      if ($1) {
        # BERT is compatible with all platforms
        conn.execute())))))"""
        INSERT INTO cross_platform_compatibility ())))))model_id, hardware_type, compatibility_level, compatibility_notes)
        VALUES 
        ())))))?, 'CUDA', 'full', 'Optimized performance with CUDA acceleration'),
        ())))))?, 'ROCm', 'full', 'Full support on AMD GPUs'),
        ())))))?, 'MPS', 'full', 'Excellent performance on Apple Silicon'),
        ())))))?, 'OpenVINO', 'full', 'Optimized for Intel hardware'),
        ())))))?, 'Qualcomm', 'full', 'Optimized for mobile deployment'),
        ())))))?, 'WebNN', 'full', 'Good performance in browser environments'),
        ())))))?, 'WebGPU', 'full', 'Excellent browser performance with GPU acceleration')
        ON CONFLICT ())))))model_id, hardware_type) DO UPDATE SET 
        compatibility_level = excluded.compatibility_level,
        compatibility_notes = excluded.compatibility_notes
        """, []]]],,,,model_id] * 7)
        ,    ,,,,,
      elif ($1) {
        # T5 has varying compatibility
        conn.execute())))))"""
        INSERT INTO cross_platform_compatibility ())))))model_id, hardware_type, compatibility_level, compatibility_notes)
        VALUES 
        ())))))?, 'CUDA', 'full', 'Optimized performance with CUDA acceleration'),
        ())))))?, 'ROCm', 'full', 'Good support on AMD GPUs'),
        ())))))?, 'MPS', 'full', 'Good performance on Apple Silicon'),
        ())))))?, 'OpenVINO', 'full', 'Optimized for Intel hardware'),
        ())))))?, 'Qualcomm', 'partial', 'Works with memory constraints'),
        ())))))?, 'WebNN', 'partial', 'Limited performance in browser environments'),
        ())))))?, 'WebGPU', 'partial', 'Works but with some performance limitations')
        ON CONFLICT ())))))model_id, hardware_type) DO UPDATE SET 
        compatibility_level = excluded.compatibility_level,
        compatibility_notes = excluded.compatibility_notes
        """, []]]],,,,model_id] * 7)
        ,    ,,,,,
      elif ($1) {
        # ViT is well supported
        conn.execute())))))"""
        INSERT INTO cross_platform_compatibility ())))))model_id, hardware_type, compatibility_level, compatibility_notes)
        VALUES 
        ())))))?, 'CUDA', 'full', 'Optimized performance with CUDA acceleration'),
        ())))))?, 'ROCm', 'full', 'Excellent support on AMD GPUs'),
        ())))))?, 'MPS', 'full', 'Excellent performance on Apple Silicon'),
        ())))))?, 'OpenVINO', 'full', 'Optimized for Intel hardware'),
        ())))))?, 'Qualcomm', 'full', 'Good performance on mobile devices'),
        ())))))?, 'WebNN', 'full', 'Good performance in browser environments'),
        ())))))?, 'WebGPU', 'full', 'Excellent browser performance with GPU acceleration')
        ON CONFLICT ())))))model_id, hardware_type) DO UPDATE SET 
        compatibility_level = excluded.compatibility_level,
        compatibility_notes = excluded.compatibility_notes
        """, []]]],,,,model_id] * 7)
        ,    ,,,,,
      elif ($1) {
        # Whisper has limitations on browser platforms
        conn.execute())))))"""
        INSERT INTO cross_platform_compatibility ())))))model_id, hardware_type, compatibility_level, compatibility_notes)
        VALUES 
        ())))))?, 'CUDA', 'full', 'Optimized performance with CUDA acceleration'),
        ())))))?, 'ROCm', 'partial', 'Works but with some performance limitations'),
        ())))))?, 'MPS', 'partial', 'Works but with some performance limitations'),
        ())))))?, 'OpenVINO', 'partial', 'Works but with some performance limitations'),
        ())))))?, 'Qualcomm', 'partial', 'Works with memory constraints'),
        ())))))?, 'WebNN', 'limited', 'Significant limitations in browser environments'),
        ())))))?, 'WebGPU', 'limited', 'Firefox performs 20% better than Chrome for audio models')
        ON CONFLICT ())))))model_id, hardware_type) DO UPDATE SET 
        compatibility_level = excluded.compatibility_level,
        compatibility_notes = excluded.compatibility_notes
        """, []]]],,,,model_id] * 7)
        ,    ,,,,,
      elif ($1) {
        # CLIP has good cross-platform support
        conn.execute())))))"""
        INSERT INTO cross_platform_compatibility ())))))model_id, hardware_type, compatibility_level, compatibility_notes)
        VALUES 
        ())))))?, 'CUDA', 'full', 'Optimized performance with CUDA acceleration'),
        ())))))?, 'ROCm', 'full', 'Good support on AMD GPUs'),
        ())))))?, 'MPS', 'full', 'Good performance on Apple Silicon'),
        ())))))?, 'OpenVINO', 'full', 'Optimized for Intel hardware'),
        ())))))?, 'Qualcomm', 'partial', 'Works with memory constraints'),
        ())))))?, 'WebNN', 'partial', 'Limited performance in browser environments'),
        ())))))?, 'WebGPU', 'full', 'Excellent support with parallel loading optimization')
        ON CONFLICT ())))))model_id, hardware_type) DO UPDATE SET 
        compatibility_level = excluded.compatibility_level,
        compatibility_notes = excluded.compatibility_notes
        """, []]]],,,,model_id] * 7)
        ,    ,,,,,
      elif ($1) {
        # LLaVA is memory intensive && limited on many platforms
        conn.execute())))))"""
        INSERT INTO cross_platform_compatibility ())))))model_id, hardware_type, compatibility_level, compatibility_notes)
        VALUES 
        ())))))?, 'CUDA', 'full', 'Optimized performance with CUDA acceleration'),
        ())))))?, 'ROCm', 'partial', 'Works but with memory constraints'),
        ())))))?, 'MPS', 'partial', 'Works but with memory constraints'),
        ())))))?, 'OpenVINO', 'limited', 'Significant memory limitations'),
        ())))))?, 'Qualcomm', 'limited', 'Only small models with severe limitations'),
        ())))))?, 'WebNN', 'not_supported', 'Memory requirements exceed browser capabilities'),
        ())))))?, 'WebGPU', 'limited', 'Only tiny versions with parallel loading')
        ON CONFLICT ())))))model_id, hardware_type) DO UPDATE SET 
        compatibility_level = excluded.compatibility_level,
        compatibility_notes = excluded.compatibility_notes
        """, []]]],,,,model_id] * 7)
        ,
        logger.info())))))"Inserted sample compatibility data")
    
      }
    # Insert sample performance data
      }
        conn.execute())))))"""
        INSERT INTO performance_comparison
        ())))))model_name, hardware_type, batch_size, sequence_length, throughput_items_per_sec, latency_ms, memory_mb, power_watts)
        VALUES
        ())))))'bert-base-uncased', 'CUDA', 32, 128, 240.5, 130.2, 2048, 120),
        ())))))'bert-base-uncased', 'ROCm', 32, 128, 210.2, 150.5, 2048, 125),
        ())))))'bert-base-uncased', 'MPS', 32, 128, 180.7, 175.3, 2048, 60),
        ())))))'bert-base-uncased', 'OpenVINO', 32, 128, 160.3, 195.8, 2048, 65),
        ())))))'bert-base-uncased', 'Qualcomm', 32, 128, 110.5, 285.2, 2048, 15),
        ())))))'bert-base-uncased', 'WebNN', 16, 128, 80.2, 198.4, 1024, 40),
        ())))))'bert-base-uncased', 'WebGPU', 16, 128, 90.5, 175.6, 1024, 40),
      
      }
        ())))))'vit-base-patch16-224', 'CUDA', 64, 0, 450.2, 140.5, 1536, 110),
        ())))))'vit-base-patch16-224', 'ROCm', 64, 0, 410.3, 155.2, 1536, 115),
        ())))))'vit-base-patch16-224', 'MPS', 64, 0, 380.5, 165.8, 1536, 55),
        ())))))'vit-base-patch16-224', 'OpenVINO', 64, 0, 340.2, 185.3, 1536, 60),
        ())))))'vit-base-patch16-224', 'Qualcomm', 32, 0, 160.5, 198.2, 1024, 12),
        ())))))'vit-base-patch16-224', 'WebNN', 32, 0, 120.3, 260.1, 768, 35),
        ())))))'vit-base-patch16-224', 'WebGPU', 32, 0, 180.5, 175.4, 768, 35),
      
      }
        ())))))'whisper-tiny', 'CUDA', 8, 0, 95.3, 104.2, 2048, 130),
        ())))))'whisper-tiny', 'ROCm', 8, 0, 75.2, 132.5, 2048, 135),
        ())))))'whisper-tiny', 'MPS', 8, 0, 60.5, 165.3, 2048, 65),
        ())))))'whisper-tiny', 'OpenVINO', 8, 0, 55.2, 180.6, 2048, 70),
        ())))))'whisper-tiny', 'Qualcomm', 4, 0, 30.4, 330.2, 1536, 20),
        ())))))'whisper-tiny', 'WebNN', 2, 0, 12.3, 805.1, 1024, 45),
        ())))))'whisper-tiny', 'WebGPU', 2, 0, 18.5, 540.6, 1024, 45)
        ON CONFLICT DO NOTHING
        """)
        logger.info())))))"Inserted sample performance data")
    
      }
    # Insert sample hardware recommendations
      }
        conn.execute())))))"""
        INSERT INTO hardware_recommendations ())))))modality, recommended_hardware, recommendation_details)
        VALUES
        ())))))'text', 'CUDA', '{}}}}
        "summary": "Text models perform best on CUDA GPUs for larger models, with WebGPU showing excellent performance for smaller models. Qualcomm hardware offers the best efficiency for mobile deployments.",
        "configurations": []]]],,,,
        "CUDA: Recommended for production deployments of medium to large models",
        "WebGPU: Excellent for browser-based deployment of small to medium models",
        "Qualcomm: Best for mobile deployments with battery constraints",
        "ROCm: Good alternative for AMD GPU hardware"
        ]
        }'),
        ())))))'vision', 'CUDA/WebGPU', '{}}}}
        "summary": "Vision models show excellent performance across most hardware platforms. WebGPU performance is particularly strong for vision models, making it competitive with native hardware for browser deployments.",
        "configurations": []]]],,,,
        "CUDA: Best for high-throughput production workloads",
        "WebGPU: Excellent for browser-based deployment with near-native performance",
        "OpenVINO: Strong performance on Intel hardware with optimized inference",
        "Qualcomm: Best option for mobile vision applications"
        ]
        }'),
        ())))))'audio', 'CUDA', '{}}}}
        "summary": "Audio models perform best on CUDA, with Firefox WebGPU showing ~20% better performance than Chrome for audio models. For mobile deployments, Qualcomm offers good performance with excellent power efficiency.",
        "configurations": []]]],,,,
        "CUDA: Best for production audio processing workloads",
        "Firefox WebGPU: Recommended for browser-based audio processing ())))))20% faster than Chrome)",
        "Qualcomm: Optimized for mobile audio processing with power efficiency",
        "MPS: Good performance on Apple Silicon"
        ]
        }'),
        ())))))'multimodal', 'CUDA', '{}}}}
        "summary": "Multimodal models are generally more demanding && perform best on CUDA GPUs. Web deployment benefits from parallel loading optimization. Qualcomm support is limited by memory constraints.",
        "configurations": []]]],,,,
        "CUDA: Essential for production deployment of multimodal models",
        "WebGPU with parallel loading: Enables browser-based multimodal processing",
        "ROCm: Viable for smaller multimodal models on AMD hardware",
        "MPS: Good alternative for Apple Silicon devices"
        ]
        }')
        ON CONFLICT ())))))modality) DO UPDATE SET
        recommended_hardware = excluded.recommended_hardware,
        recommendation_details = excluded.recommendation_details
        """)
        logger.info())))))"Inserted sample hardware recommendations")
    
  } catch($2: $1) {
    logger.error())))))`$1`)
        raise

  }
$1($2) {
  """Main function to create the database tables."""
  args = parse_args()))))))
  
}
  try {
    # Connect to the database
    conn = duckdb.connect())))))args.db_path)
    logger.info())))))`$1`)
    
  }
    # Create the tables
    create_tables())))))conn)
    
    # Insert sample data if ($1) {
    if ($1) ${$1} catch($2: $1) {
    logger.error())))))`$1`)
    }
    import * as $1
    }
    traceback.print_exc()))))))

if ($1) {
  main()))))))