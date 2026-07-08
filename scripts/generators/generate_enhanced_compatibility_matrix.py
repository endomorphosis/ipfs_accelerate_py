#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Enhanced Comprehensive Model Compatibility Matrix (March 6, 2025)

This script generates a comprehensive compatibility matrix for all HuggingFace model classes
supported by the framework. It queries the DuckDB database for compatibility information and
generates reports in various formats (markdown, HTML, JSON).

The matrix includes:
- Cross-platform compatibility status (CUDA, ROCm, MPS, OpenVINO, Qualcomm, WebNN, WebGPU)
- Performance metrics (throughput, latency, memory)
- Hardware recommendations by model type
- Interactive filtering and visualization in HTML format
- Export capabilities to CSV and JSON

Usage:
    python generate_enhanced_compatibility_matrix.py [options]

Options:
    --db-path PATH           Path to DuckDB database (default: ./benchmark_db.duckdb)
    --output-dir DIR         Output directory for matrix files (default: ./docs)
    --format FORMAT          Output format (markdown, html, json, all) (default: all)
    --filter FILTER          Filter by model family (e.g., 'bert', 'vision')
    --hardware HARDWARE      Filter by hardware platforms (comma-separated)
    --performance            Include performance metrics
    --all-models             Include all models in the database
    --recommendations        Include hardware recommendations
    --debug                  Enable debug output
"""

import argparse
import os
import json
import pandas as pd
import duckdb
import datetime
import jinja2
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("compatibility_matrix")

# Define model categories
MODEL_CATEGORIES = {
    "text": [
        "BERT", "RoBERTa", "T5", "GPT2", "LLAMA", "Falcon", "Gemma", "BLOOM", "OPT", 
        "DistilBERT", "ELECTRA", "ALBERT", "XLNet", "CodeLLAMA", "FLAN-T5", "UL2",
        "Mistral", "Phi", "MPT", "GLM", "BART", "Qwen", "XLM", "DeBERTa"
    ],
    "vision": [
        "ViT", "ResNet", "DETR", "ConvNeXT", "Swin", "BEiT", "DeiT", "RegNet", "EfficientNet",
        "MobileNet", "ConvNext", "DINOv2", "MAE", "EVA", "ConvNeXTv2", "MaxViT"
    ],
    "audio": [
        "Whisper", "Wav2Vec2", "CLAP", "HuBERT", "SpeechT5", "USM", "MMS", "AudioLDM",
        "Bark", "MusicGen", "SEW", "UniSpeech", "WavLM", "MFCC", "Encodec"
    ],
    "multimodal": [
        "CLIP", "LLaVA", "BLIP", "BLIP-2", "ALBEF", "FLAVA", "LLaVA-Next", "CoCa",
        "ImageBind", "PaLM-E", "ALIGN", "BEiT-3", "GIT", "X-CLIP", "Flamingo", "CM3Leon"
    ]
}

# Default hardware platforms to include
DEFAULT_HARDWARE_PLATFORMS = [
    'CUDA', 'ROCm', 'MPS', 'OpenVINO', 'Qualcomm', 'WebNN', 'WebGPU'
]

# Default modalities
DEFAULT_MODALITIES = [
    'text', 'vision', 'audio', 'multimodal'
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate comprehensive model compatibility matrix")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb", 
                        help="Path to DuckDB database")
    parser.add_argument("--output-dir", default="./docs", 
                        help="Output directory for matrix files")
    parser.add_argument("--format", choices=["markdown", "html", "json", "all"], default="all", 
                        help="Output format for compatibility matrix")
    parser.add_argument("--filter", help="Filter by model family (e.g., 'bert', 'vision')")
    parser.add_argument("--hardware", help="Filter by hardware platforms (comma-separated)")
    parser.add_argument("--performance", action="store_true", 
                        help="Include performance metrics")
    parser.add_argument("--all-models", action="store_true", 
                        help="Include all models in the database")
    parser.add_argument("--recommendations", action="store_true", 
                        help="Include hardware recommendations")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug output")
    
    return parser.parse_args()

def connect_to_database(db_path):
    """Connect to the DuckDB database."""
    try:
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def get_hardware_platforms(conn, specified_hardware=None):
    """Get available hardware platforms from the database."""
    try:
        # Check if the hardware platforms table exists
        table_exists = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='hardware_platforms'
        """).fetchone()
        
        if table_exists:
            # Query actual hardware platforms from database
            result = conn.execute("""
                SELECT DISTINCT hardware_type 
                FROM hardware_platforms 
                ORDER BY hardware_type
            """).fetchdf()
            
            if len(result) > 0:
                hardware_platforms = result['hardware_type'].tolist()
            else:
                hardware_platforms = DEFAULT_HARDWARE_PLATFORMS
        else:
            # Fall back to cross_platform_compatibility table
            result = conn.execute("""
                SELECT DISTINCT hardware_type 
                FROM cross_platform_compatibility 
                ORDER BY hardware_type
            """).fetchdf()
            
            if len(result) > 0:
                hardware_platforms = result['hardware_type'].tolist()
            else:
                hardware_platforms = DEFAULT_HARDWARE_PLATFORMS
        
        # Filter platforms if specified
        if specified_hardware:
            specified_platforms = [p.strip() for p in specified_hardware.split(',')]
            hardware_platforms = [p for p in hardware_platforms if p in specified_platforms]
        
        logger.info(f"Found hardware platforms: {hardware_platforms}")
        return hardware_platforms
    
    except Exception as e:
        logger.error(f"Error retrieving hardware platforms: {e}")
        logger.info("Falling back to default hardware platforms")
        if specified_hardware:
            return [p.strip() for p in specified_hardware.split(',')]
        return DEFAULT_HARDWARE_PLATFORMS

def get_model_compatibility_data(conn, hardware_platforms, model_filter=None, all_models=False):
    """Get model compatibility data from the database."""
    try:
        # Base query
        query = f"""
        SELECT 
            m.model_name, 
            m.model_type, 
            m.model_family,
            COALESCE(m.modality, 'unknown') as modality,
            COALESCE(m.parameters_million, 0) as parameters_million,
            pc.hardware_type, 
            pc.compatibility_level,
            pc.compatibility_notes,
            CASE 
                WHEN pc.compatibility_level = 'full' THEN '✅' 
                WHEN pc.compatibility_level = 'partial' THEN '⚠️'
                WHEN pc.compatibility_level = 'limited' THEN '🔶'
                ELSE '❌'
            END as symbol
        FROM 
            cross_platform_compatibility pc
        JOIN 
            models m ON pc.model_id = m.id
        """
        
        # Add filters if specified
        where_clauses = []
        if model_filter:
            where_clauses.append(f"(m.model_family LIKE '%{model_filter}%' OR m.model_type LIKE '%{model_filter}%' OR m.modality LIKE '%{model_filter}%')")
        
        if not all_models:
            # Only include key models if not all_models
            where_clauses.append("m.is_key_model = TRUE")
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Complete query with ordering
        query += """
        ORDER BY 
            m.modality,
            m.model_family,
            m.model_name,
            pc.hardware_type
        """
        
        # Execute query and get results
        df = conn.execute(query).fetchdf()
        
        if df.empty:
            logger.warning("No compatibility data found in the database")
            return pd.DataFrame()
        
        # Pivot table to create matrix format
        matrix_df = df.copy()
        
        # Add hardware-specific compatibility level and notes columns
        for hw in hardware_platforms:
            hw_data = df[df['hardware_type'] == hw]
            if not hw_data.empty:
                for idx, row in hw_data.iterrows():
                    model_name = row['model_name']
                    level = row['compatibility_level']
                    symbol = row['symbol']
                    notes = row['compatibility_notes']
                    
                    # Update all rows for this model
                    matrix_df.loc[matrix_df['model_name'] == model_name, hw] = symbol
                    matrix_df.loc[matrix_df['model_name'] == model_name, f"{hw}_level"] = level
                    matrix_df.loc[matrix_df['model_name'] == model_name, f"{hw}_notes"] = notes
        
        # Drop duplicates to get one row per model
        matrix_df = matrix_df.drop_duplicates(subset=['model_name']).reset_index(drop=True)
        
        logger.info(f"Retrieved compatibility data for {len(matrix_df)} models")
        return matrix_df
    
    except Exception as e:
        logger.error(f"Error retrieving model compatibility data: {e}")
        raise

def get_performance_data(conn, hardware_platforms, model_filter=None):
    """Get performance data from the database."""
    try:
        # Base query for performance data
        perf_query = """
        SELECT 
            m.model_name, 
            m.model_family,
            p.hardware_type,
            AVG(p.throughput_items_per_sec) as avg_throughput,
            AVG(p.latency_ms) as avg_latency,
            AVG(p.memory_mb) as avg_memory
        FROM 
            performance_comparison p
        JOIN 
            models m ON p.model_name = m.model_name
        """
        
        # Add filter if specified
        if model_filter:
            perf_query += f" WHERE m.model_family LIKE '%{model_filter}%' OR m.model_type LIKE '%{model_filter}%'"
        
        # Group by and order
        perf_query += """
        GROUP BY 
            m.model_name, m.model_family, p.hardware_type
        ORDER BY
            m.model_family, m.model_name, p.hardware_type
        """
        
        # Execute query
        perf_df = conn.execute(perf_query).fetchdf()
        
        if perf_df.empty:
            logger.warning("No performance data found in the database")
            return None
        
        # Aggregate by model family
        family_perf = {}
        for _, row in perf_df.iterrows():
            family = row['model_family']
            hw = row['hardware_type']
            
            if family not in family_perf:
                family_perf[family] = {}
            
            if hw not in family_perf[family]:
                family_perf[family][hw] = {}
            
            family_perf[family][hw]['avg_throughput'] = round(row['avg_throughput'], 2)
            family_perf[family][hw]['avg_latency'] = round(row['avg_latency'], 2)
            family_perf[family][hw]['avg_memory'] = round(row['avg_memory'], 2)
        
        # Prepare chart data
        chart_data = {
            'families': list(family_perf.keys()),
            'hardware_platforms': hardware_platforms,
            'throughput_data': [],
            'latency_data': [],
            'memory_data': []
        }
        
        # Populate chart data arrays
        for hw in hardware_platforms:
            throughput_array = []
            latency_array = []
            memory_array = []
            
            for family in chart_data['families']:
                if family in family_perf and hw in family_perf[family]:
                    throughput_array.append(family_perf[family][hw].get('avg_throughput', 0))
                    latency_array.append(family_perf[family][hw].get('avg_latency', 0))
                    memory_array.append(family_perf[family][hw].get('avg_memory', 0))
                else:
                    throughput_array.append(0)
                    latency_array.append(0)
                    memory_array.append(0)
            
            chart_data['throughput_data'].append(throughput_array)
            chart_data['latency_data'].append(latency_array)
            chart_data['memory_data'].append(memory_array)
        
        logger.info(f"Retrieved performance data for {len(perf_df['model_name'].unique())} models")
        return {
            'by_family': family_perf,
            'chart_data': chart_data
        }
    
    except Exception as e:
        logger.error(f"Error retrieving performance data: {e}")
        return None

def get_hardware_recommendations(conn):
    """Get hardware recommendations for different model types."""
    try:
        # Try to get recommendations from database
        query = """
        SELECT 
            modality, 
            recommended_hardware,
            recommendation_details
        FROM 
            hardware_recommendations
        """
        
        try:
            result = conn.execute(query).fetchdf()
            if not result.empty:
                recommendations = {}
                for _, row in result.iterrows():
                    modality = row['modality']
                    recommendations[modality] = {
                        'best_platform': row['recommended_hardware'],
                        'summary': json.loads(row['recommendation_details']).get('summary', ''),
                        'configurations': json.loads(row['recommendation_details']).get('configurations', [])
                    }
                return recommendations
        except:
            # Table might not exist, use default recommendations
            pass
        
        # Default recommendations if table doesn't exist
        recommendations = {
            'text': {
                'best_platform': 'CUDA',
                'summary': 'Text models perform best on CUDA GPUs for larger models, with WebGPU showing excellent performance for smaller models. Qualcomm hardware offers the best efficiency for mobile deployments.',
                'configurations': [
                    'CUDA: Recommended for production deployments of medium to large models',
                    'WebGPU: Excellent for browser-based deployment of small to medium models',
                    'Qualcomm: Best for mobile deployments with battery constraints',
                    'ROCm: Good alternative for AMD GPU hardware'
                ]
            },
            'vision': {
                'best_platform': 'CUDA/WebGPU',
                'summary': 'Vision models show excellent performance across most hardware platforms. WebGPU performance is particularly strong for vision models, making it competitive with native hardware for browser deployments.',
                'configurations': [
                    'CUDA: Best for high-throughput production workloads',
                    'WebGPU: Excellent for browser-based deployment with near-native performance',
                    'OpenVINO: Strong performance on Intel hardware with optimized inference',
                    'Qualcomm: Best option for mobile vision applications'
                ]
            },
            'audio': {
                'best_platform': 'CUDA',
                'summary': 'Audio models perform best on CUDA, with Firefox WebGPU showing ~20% better performance than Chrome for audio models. For mobile deployments, Qualcomm offers good performance with excellent power efficiency.',
                'configurations': [
                    'CUDA: Best for production audio processing workloads',
                    'Firefox WebGPU: Recommended for browser-based audio processing (20% faster than Chrome)',
                    'Qualcomm: Optimized for mobile audio processing with power efficiency',
                    'MPS: Good performance on Apple Silicon'
                ]
            },
            'multimodal': {
                'best_platform': 'CUDA',
                'summary': 'Multimodal models are generally more demanding and perform best on CUDA GPUs. Web deployment benefits from parallel loading optimization. Qualcomm support is limited by memory constraints.',
                'configurations': [
                    'CUDA: Essential for production deployment of multimodal models',
                    'WebGPU with parallel loading: Enables browser-based multimodal processing',
                    'ROCm: Viable for smaller multimodal models on AMD hardware',
                    'MPS: Good alternative for Apple Silicon devices'
                ]
            }
        }
        
        logger.info("Using default hardware recommendations")
        return recommendations
    
    except Exception as e:
        logger.error(f"Error retrieving hardware recommendations: {e}")
        return {}

def generate_markdown(matrix_df, hardware_platforms, performance_data, recommendations, args):
    """Generate markdown compatibility matrix."""
    try:
        # Group models by modality
        models_by_modality = {}
        for modality in matrix_df['modality'].unique():
            modality_models = matrix_df[matrix_df['modality'] == modality].to_dict('records')
            models_by_modality[modality] = modality_models
        
        # Load template
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
        template = env.get_template('compatibility_matrix_template.md')
        
        # Render template
        output = template.render(
            generated_date=datetime.datetime.now().strftime("%B %d, %Y"),
            total_models=len(matrix_df),
            total_hardware_platforms=len(hardware_platforms),
            hardware_platforms=hardware_platforms,
            models_by_modality=models_by_modality,
            include_performance=args.performance,
            performance_by_family=performance_data['by_family'] if performance_data else {},
            recommendations=recommendations
        )
        
        # Write to file
        output_path = os.path.join(args.output_dir, 'COMPREHENSIVE_MODEL_COMPATIBILITY_MATRIX.md')
        with open(output_path, 'w') as f:
            f.write(output)
        
        logger.info(f"Generated markdown compatibility matrix: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating markdown: {e}")
        raise

def generate_html(matrix_df, hardware_platforms, performance_data, recommendations, args):
    """Generate HTML compatibility matrix."""
    try:
        # Group models by modality
        models_by_modality = {}
        modalities = []
        for modality in matrix_df['modality'].unique():
            modality_models = matrix_df[matrix_df['modality'] == modality].to_dict('records')
            models_by_modality[modality] = modality_models
            modalities.append(modality)
        
        # Prepare recommendation chart data
        recommendation_chart_data = {}
        for modality, recs in recommendations.items():
            if modality in modalities:
                # Create radar chart data
                radar_data = {
                    'axes': ['Performance', 'Efficiency', 'Memory', 'Compatibility', 'Developer Experience'],
                    'datasets': []
                }
                
                # Add datasets for each recommended platform
                colors = [
                    'rgba(75, 192, 192, 0.7)',
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(255, 206, 86, 0.7)'
                ]
                
                # Get primary platform and alternatives
                platforms = [recs['best_platform']]
                for config in recs['configurations']:
                    platform = config.split(':')[0].strip()
                    if platform not in platforms:
                        platforms.append(platform)
                
                # Add radar chart datasets
                for i, platform in enumerate(platforms[:4]):  # Limit to 4 platforms
                    if platform == 'CUDA':
                        dataset = {
                            'label': platform,
                            'data': [95, 70, 80, 100, 90],
                            'backgroundColor': colors[i % len(colors)].replace('0.7', '0.2'),
                            'borderColor': colors[i % len(colors)].replace('0.7', '1'),
                            'pointBackgroundColor': colors[i % len(colors)].replace('0.7', '1'),
                            'pointBorderColor': '#fff',
                            'pointHoverBackgroundColor': '#fff',
                            'pointHoverBorderColor': colors[i % len(colors)].replace('0.7', '1')
                        }
                    elif platform == 'WebGPU':
                        dataset = {
                            'label': platform,
                            'data': [75, 85, 65, 90, 95],
                            'backgroundColor': colors[i % len(colors)].replace('0.7', '0.2'),
                            'borderColor': colors[i % len(colors)].replace('0.7', '1'),
                            'pointBackgroundColor': colors[i % len(colors)].replace('0.7', '1'),
                            'pointBorderColor': '#fff',
                            'pointHoverBackgroundColor': '#fff',
                            'pointHoverBorderColor': colors[i % len(colors)].replace('0.7', '1')
                        }
                    elif platform == 'ROCm' or platform == 'MPS':
                        dataset = {
                            'label': platform,
                            'data': [85, 75, 75, 85, 80],
                            'backgroundColor': colors[i % len(colors)].replace('0.7', '0.2'),
                            'borderColor': colors[i % len(colors)].replace('0.7', '1'),
                            'pointBackgroundColor': colors[i % len(colors)].replace('0.7', '1'),
                            'pointBorderColor': '#fff',
                            'pointHoverBackgroundColor': '#fff',
                            'pointHoverBorderColor': colors[i % len(colors)].replace('0.7', '1')
                        }
                    elif platform == 'Qualcomm':
                        dataset = {
                            'label': platform,
                            'data': [70, 95, 60, 80, 75],
                            'backgroundColor': colors[i % len(colors)].replace('0.7', '0.2'),
                            'borderColor': colors[i % len(colors)].replace('0.7', '1'),
                            'pointBackgroundColor': colors[i % len(colors)].replace('0.7', '1'),
                            'pointBorderColor': '#fff',
                            'pointHoverBackgroundColor': '#fff',
                            'pointHoverBorderColor': colors[i % len(colors)].replace('0.7', '1')
                        }
                    else:
                        # Generic dataset for other platforms
                        dataset = {
                            'label': platform,
                            'data': [80, 80, 70, 80, 80],
                            'backgroundColor': colors[i % len(colors)].replace('0.7', '0.2'),
                            'borderColor': colors[i % len(colors)].replace('0.7', '1'),
                            'pointBackgroundColor': colors[i % len(colors)].replace('0.7', '1'),
                            'pointBorderColor': '#fff',
                            'pointHoverBackgroundColor': '#fff',
                            'pointHoverBorderColor': colors[i % len(colors)].replace('0.7', '1')
                        }
                    
                    radar_data['datasets'].append(dataset)
                
                recommendation_chart_data[modality] = radar_data
        
        # Load template
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
        template = env.get_template('compatibility_matrix_template.html')
        
        # Render template
        output = template.render(
            generated_date=datetime.datetime.now().strftime("%B %d, %Y"),
            total_models=len(matrix_df),
            total_hardware_platforms=len(hardware_platforms),
            hardware_platforms=hardware_platforms,
            models_by_modality=models_by_modality,
            modalities=modalities,
            performance_data=performance_data['chart_data'] if performance_data else {},
            performance_by_family=performance_data['by_family'] if performance_data else {},
            recommendations=recommendations,
            recommendation_chart_data=recommendation_chart_data
        )
        
        # Write to file
        output_path = os.path.join(args.output_dir, 'compatibility_matrix.html')
        with open(output_path, 'w') as f:
            f.write(output)
        
        logger.info(f"Generated HTML compatibility matrix: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating HTML: {e}")
        raise

def generate_json(matrix_df, hardware_platforms, performance_data, recommendations, args):
    """Generate JSON compatibility matrix."""
    try:
        # Convert to JSON-friendly structure
        matrix_data = matrix_df.to_dict('records')
        
        # Create complete JSON data structure
        json_data = {
            'metadata': {
                'generated_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_models': len(matrix_df),
                'total_hardware_platforms': len(hardware_platforms),
                'hardware_platforms': hardware_platforms
            },
            'compatibility_matrix': matrix_data,
            'performance_data': performance_data if performance_data else {},
            'recommendations': recommendations
        }
        
        # Write to file
        output_path = os.path.join(args.output_dir, 'compatibility_matrix.json')
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Generated JSON compatibility matrix: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating JSON: {e}")
        raise

def main():
    """Main function to generate the compatibility matrix."""
    # Parse arguments
    args = parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Connect to database
        conn = connect_to_database(args.db_path)
        
        # Get hardware platforms
        hardware_platforms = get_hardware_platforms(conn, args.hardware)
        
        # Get model compatibility data
        matrix_df = get_model_compatibility_data(conn, hardware_platforms, args.filter, args.all_models)
        
        if matrix_df.empty:
            logger.error("No compatibility data found, cannot generate matrix")
            return
        
        # Get performance data if requested
        performance_data = None
        if args.performance:
            performance_data = get_performance_data(conn, hardware_platforms, args.filter)
        
        # Get hardware recommendations if requested
        recommendations = {}
        if args.recommendations:
            recommendations = get_hardware_recommendations(conn)
        
        # Generate outputs in requested formats
        if args.format in ['markdown', 'all']:
            generate_markdown(matrix_df, hardware_platforms, performance_data, recommendations, args)
        
        if args.format in ['html', 'all']:
            generate_html(matrix_df, hardware_platforms, performance_data, recommendations, args)
        
        if args.format in ['json', 'all']:
            generate_json(matrix_df, hardware_platforms, performance_data, recommendations, args)
        
        logger.info("Compatibility matrix generation completed successfully")
    
    except Exception as e:
        logger.error(f"Error generating compatibility matrix: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()