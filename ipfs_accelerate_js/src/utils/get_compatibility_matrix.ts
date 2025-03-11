/**
 * Converted from Python: get_compatibility_matrix.py
 * Conversion date: 2025-03-11 04:08:31
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
import * as $1
import * as $1 as pd
import * as $1 as np
import * as $1.pyplot as plt
import * as $1 as sns
import * as $1
import * as $1
import ${$1} from "$1"

# Set up database connection
db_path = "./benchmark_db.duckdb"
con = duckdb.connect(db_path)

# Get latest data from cross_platform_compatibility
query = """
SELECT 
m.model_name,
m.model_family,
hp.hardware_type,
cpc.cpu_support,
cpc.cuda_support,
cpc.rocm_support,
cpc.mps_support,
cpc.openvino_support,
cpc.qnn_support,
cpc.webnn_support,
cpc.webgpu_support,
cpc.recommended_platform
FROM 
cross_platform_compatibility cpc
JOIN 
models m ON cpc.model_id = m.model_id
JOIN 
hardware_platforms hp ON cpc.hardware_id = hp.hardware_id
ORDER BY 
m.model_family, m.model_name
"""

# Try to execute the query
try {
  results = con.execute(query).fetchdf()
  
}
  # Check if ($1) {
  if ($1) {
    console.log($1)
    # Generate sample data based on CLAUDE.md info
    hardware_types = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"],
    model_families = ["embedding", "text_generation", "vision", "audio", "multimodal"]
    ,
    # Sample compatibility matrix based on CLAUDE.md
    compatibility = {}}
    "embedding": {}}"cpu": "High", "cuda": "High", "rocm": "High", "mps": "High", "openvino": "High", "qnn": "High", "webnn": "High", "webgpu": "High"},
    "text_generation": {}}"cpu": "Medium", "cuda": "High", "rocm": "Medium", "mps": "Medium", "openvino": "Medium", "qnn": "Medium", "webnn": "Limited", "webgpu": "Limited"},
    "vision": {}}"cpu": "Medium", "cuda": "High", "rocm": "High", "mps": "High", "openvino": "High", "qnn": "High", "webnn": "High", "webgpu": "High"},
    "audio": {}}"cpu": "Medium", "cuda": "High", "rocm": "Medium", "mps": "Medium", "openvino": "Medium", "qnn": "Medium", "webnn": "Limited", "webgpu": "Limited"},
    "multimodal": {}}"cpu": "Limited", "cuda": "High", "rocm": "Limited", "mps": "Limited", "openvino": "Limited", "qnn": "Limited", "webnn": "Limited", "webgpu": "Limited"}
    }
    
  }
    # Convert to dataframe format similar to what we'd get from the database
    rows = [],,,
    for (const $1 of $2) {
      for (const $1 of $2) {
        compat = compatibility[family][hw],
        # Convert text to boolean values for the specific column
        cpu_support = true if hw == "cpu" && compat != "Limited" else false
        cuda_support = true if hw == "cuda" && compat != "Limited" else false
        rocm_support = true if hw == "rocm" && compat != "Limited" else false
        mps_support = true if hw == "mps" && compat != "Limited" else false
        openvino_support = true if hw == "openvino" && compat != "Limited" else false
        qnn_support = true if hw == "qnn" && compat != "Limited" else false
        webnn_support = true if hw == "webnn" && compat != "Limited" else false
        webgpu_support = true if hw == "webgpu" && compat != "Limited" else false
        
      }
        recommended = "cuda" if family != "embedding" else "cpu"
        
    }
        rows.append({}}:
          "model_name": `$1`,
          "model_family": family,
          "hardware_type": hw,
          "cpu_support": cpu_support,
          "cuda_support": cuda_support,
          "rocm_support": rocm_support,
          "mps_support": mps_support,
          "openvino_support": openvino_support,
          "qnn_support": qnn_support,
          "webnn_support": webnn_support,
          "webgpu_support": webgpu_support,
          "recommended_platform": recommended
          })
    
  }
          results = pd.DataFrame(rows)
  
  # Generate compatibility matrix
          matrix_data = {}}}
  
  # Group by model family
          for family in results['model_family'].unique():,
          family_data = results[results['model_family'] == family]
          ,
    # For each family, check compatibility for each hardware type
          hardware_compatibility = {}}}
    for hw in ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"],:
      # Check support column for this hardware
      support_column = `$1`
      
      if ($1) {
        support = family_data[support_column].any(),
        if ($1) {
          # Check if it's recommended
          recommended = (family_data['recommended_platform'] == hw).any():,
          if ($1) ${$1} else ${$1} else ${$1} else {
        # Try to infer from hardware_type matches
          }
        hw_matches = family_data[family_data['hardware_type'] == hw],
        }
        if ($1) ${$1} else ${$1}\n\n"
          markdown += "## Model Family-Based Compatibility Chart\n\n"
          markdown += "| Model Family | CPU | CUDA | ROCm | MPS | OpenVINO | QNN | WebNN | WebGPU | Notes |\n"
          markdown += "|--------------|-----|------|------|-----|----------|-----|---------|-------|--------|\n"
  
      }
  for family, compatibility in Object.entries($1):
    notes = ""
    if ($1) {
      notes = "Fully supported on all hardware"
    elif ($1) {
      notes = "Memory requirements critical"
    elif ($1) {
      notes = "Full cross-platform support"
    elif ($1) {
      notes = "CUDA preferred, Web simulation added"
    elif ($1) ${$1} | "
    }
    for hw in ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"],:
    }
      row += `$1`? Unknown')} | "
      row += `$1`
      markdown += row + "\n"
  
    }
      markdown += "\n### Legend\n\n"
      markdown += "- ✅ High: Fully compatible with excellent performance\n"
      markdown += "- ✅ Medium: Compatible with good performance\n"
      markdown += "- ⚠️ Limited: Compatible but with performance limitations\n"
      markdown += "- ❌ N/A: Not compatible || !available\n"
      markdown += "- ? Unknown: Not tested\n\n"
  
    }
  # Generate heatmap visualization
      plt.figure(figsize=(12, 8))
  
  # Prepare data for heatmap
      heatmap_data = [],,,
  for (const $1 of $2) {
    row = [],,,
    for hw in ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"],:
      compat = matrix_data[family].get(hw, "? Unknown"),
      if ($1) {
        $1.push($2)
      elif ($1) {
        $1.push($2)
      elif ($1) ${$1} else {
        $1.push($2)
        $1.push($2)
  
      }
        heatmap_df = pd.DataFrame(heatmap_data,
        index=$3.map(($2) => $1),:,
        columns=["CPU", "CUDA", "ROCm", "MPS", "OpenVINO", "QNN", "WebNN", "WebGPU"])
        ,
  # Create heatmap
      }
        sns.heatmap(heatmap_df, annot=true, cmap="YlGnBu", cbar_kws={}}'label': 'Compatibility Level'},
        vmin=0, vmax=3, fmt=".0f")
        plt.title("Hardware Compatibility Matrix")
        plt.tight_layout()
  
      }
  # Save outputs
  }
        output_dir = "./comprehensive_reports"
        os.makedirs(output_dir, exist_ok=true)
  
  # Save markdown
  with open(`$1`, "w") as f:
    f.write(markdown)
    console.log($1)
  
  # Save heatmap
    plt.savefig(`$1`, dpi=100, bbox_inches="tight")
    console.log($1)
  
} catch($2: $1) ${$1} finally {
  con.close()