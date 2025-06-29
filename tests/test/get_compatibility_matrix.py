#!/usr/bin/env python3
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime

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
try:
    results = con.execute(query).fetchdf()
    
    # Check if we have results:
    if len(results) == 0:
        print("No compatibility data found in the database. Generating sample data...")
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
        
        # Convert to dataframe format similar to what we'd get from the database
        rows = [],,,
        for family in model_families:
            for hw in hardware_types:
                compat = compatibility[family][hw],
                # Convert text to boolean values for the specific column
                cpu_support = True if hw == "cpu" and compat != "Limited" else False
                cuda_support = True if hw == "cuda" and compat != "Limited" else False
                rocm_support = True if hw == "rocm" and compat != "Limited" else False
                mps_support = True if hw == "mps" and compat != "Limited" else False
                openvino_support = True if hw == "openvino" and compat != "Limited" else False
                qnn_support = True if hw == "qnn" and compat != "Limited" else False
                webnn_support = True if hw == "webnn" and compat != "Limited" else False
                webgpu_support = True if hw == "webgpu" and compat != "Limited" else False
                
                recommended = "cuda" if family != "embedding" else "cpu"
                
                rows.append({}}:
                    "model_name": f"sample_{}}}}family}",
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
            support_column = f"{}}}}hw}_support"
            
            if support_column in family_data.columns:
                support = family_data[support_column].any(),
                if support:
                    # Check if it's recommended
                    recommended = (family_data['recommended_platform'] == hw).any():,
                    if recommended:
                        hardware_compatibility[hw] = "✅ High",
                    else:
                        hardware_compatibility[hw] = "✅ Medium",,
                else:
                    hardware_compatibility[hw] = "⚠️ Limited",
            else:
                # Try to infer from hardware_type matches
                hw_matches = family_data[family_data['hardware_type'] == hw],
                if len(hw_matches) > 0:
                    hardware_compatibility[hw] = "✅ Medium",,
                else:
                    hardware_compatibility[hw] = "? Unknown"
                    ,
                    matrix_data[family] = hardware_compatibility
                    ,
    # Create a Markdown table
                    markdown = "# Hardware Compatibility Matrix\n\n"
                    markdown += f"Generated: {}}}}datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    markdown += "## Model Family-Based Compatibility Chart\n\n"
                    markdown += "| Model Family | CPU | CUDA | ROCm | MPS | OpenVINO | QNN | WebNN | WebGPU | Notes |\n"
                    markdown += "|--------------|-----|------|------|-----|----------|-----|---------|-------|--------|\n"
    
    for family, compatibility in matrix_data.items():
        notes = ""
        if family == "embedding":
            notes = "Fully supported on all hardware"
        elif family == "text_generation":
            notes = "Memory requirements critical"
        elif family == "vision":
            notes = "Full cross-platform support"
        elif family == "audio":
            notes = "CUDA preferred, Web simulation added"
        elif family == "multimodal":
            notes = "CUDA for production, others are limited"
        
            row = f"| {}}}}family.replace('_', ' ').title()} | "
        for hw in ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"],:
            row += f"{}}}}compatibility.get(hw, '? Unknown')} | "
            row += f"{}}}}notes} |"
            markdown += row + "\n"
    
            markdown += "\n### Legend\n\n"
            markdown += "- ✅ High: Fully compatible with excellent performance\n"
            markdown += "- ✅ Medium: Compatible with good performance\n"
            markdown += "- ⚠️ Limited: Compatible but with performance limitations\n"
            markdown += "- ❌ N/A: Not compatible or not available\n"
            markdown += "- ? Unknown: Not tested\n\n"
    
    # Generate heatmap visualization
            plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
            heatmap_data = [],,,
    for family in matrix_data:
        row = [],,,
        for hw in ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"],:
            compat = matrix_data[family].get(hw, "? Unknown"),
            if "High" in compat:
                row.append(3)
            elif "Medium" in compat:
                row.append(2)
            elif "Limited" in compat:
                row.append(1)
            else:
                row.append(0)
                heatmap_data.append(row)
    
                heatmap_df = pd.DataFrame(heatmap_data,
                index=[f.replace('_', ' ').title() for f in matrix_data.keys()],:,
                columns=["CPU", "CUDA", "ROCm", "MPS", "OpenVINO", "QNN", "WebNN", "WebGPU"])
                ,
    # Create heatmap
                sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", cbar_kws={}}'label': 'Compatibility Level'},
                vmin=0, vmax=3, fmt=".0f")
                plt.title("Hardware Compatibility Matrix")
                plt.tight_layout()
    
    # Save outputs
                output_dir = "./comprehensive_reports"
                os.makedirs(output_dir, exist_ok=True)
    
    # Save markdown
    with open(f"{}}}}output_dir}/hardware_compatibility_matrix.md", "w") as f:
        f.write(markdown)
        print(f"Saved markdown to {}}}}output_dir}/hardware_compatibility_matrix.md")
    
    # Save heatmap
        plt.savefig(f"{}}}}output_dir}/hardware_compatibility_heatmap.png", dpi=100, bbox_inches="tight")
        print(f"Saved heatmap to {}}}}output_dir}/hardware_compatibility_heatmap.png")
    
except Exception as e:
    print(f"Error querying database: {}}}}e}")
    # Create minimal output with error
    output_dir = "./comprehensive_reports"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{}}}}output_dir}/hardware_compatibility_matrix.md", "w") as f:
        f.write(f"# Hardware Compatibility Matrix\n\nError generating matrix: {}}}}e}\n\n")
        f.write("Using information from CLAUDE.md instead.\n\n")
        # Include a simple version of the matrix from CLAUDE.md
        f.write("## Model Family-Based Compatibility Chart\n\n")
        f.write("| Model Family | CPU | CUDA | ROCm | MPS | OpenVINO | QNN | WebNN | WebGPU | Notes |\n")
        f.write("|--------------|-----|------|------|-----|----------|-----|---------|-------|--------|\n")
        f.write("| Embedding | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | Fully supported on all hardware |\n")
        f.write("| Text Generation | ✅ Medium | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Limited | ⚠️ Limited | Memory requirements critical |\n")
        f.write("| Vision | ✅ Medium | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | Full cross-platform support |\n")
        f.write("| Audio | ✅ Medium | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Limited | ⚠️ Limited | CUDA preferred, Web simulation added |\n")
        f.write("| Multimodal | ⚠️ Limited | ✅ High | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | CUDA for production, others are limited |\n")
        
        print(f"Saved fallback markdown to {}}}}output_dir}/hardware_compatibility_matrix.md")

finally:
    con.close()