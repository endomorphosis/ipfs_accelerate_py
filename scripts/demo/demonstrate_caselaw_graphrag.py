#!/usr/bin/env python3
"""
Caselaw GraphRAG Demonstration Script

This script demonstrates the Caselaw Access Project (CAP) GraphRAG functionality,
including dataset loading, graph construction, temporal deontic logic analysis,
and a web dashboard for legal case search and analysis.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Any

# Add the parent directory to sys.path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ipfs_accelerate_py.caselaw_graphrag_processor import CaselawGraphRAGProcessor
    from ipfs_accelerate_py.caselaw_dataset_loader import CaselawDatasetLoader
    from ipfs_accelerate_py.temporal_deontic_logic import TemporalDeonticLogicProcessor
    from ipfs_accelerate_py.caselaw_dashboard import CaselawDashboard
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    print("Running in demo mode with mock data...")

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='Demonstrate Caselaw GraphRAG functionality')
    parser.add_argument('--max-samples', type=int, default=100, 
                       help='Maximum number of samples to process')
    parser.add_argument('--quick-demo', action='store_true',
                       help='Run a quick demo with minimal processing')
    parser.add_argument('--temporal-logic', action='store_true',
                       help='Include temporal deontic logic analysis')
    parser.add_argument('--run-dashboard', action='store_true',
                       help='Launch the caselaw dashboard after processing')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port for the dashboard server')
    
    args = parser.parse_args()
    
    if args.quick_demo:
        args.max_samples = min(args.max_samples, 10)
    
    print("🏛️  Caselaw GraphRAG Demonstration")
    print("=" * 50)
    
    # Step 1: Load caselaw dataset
    print("\n📚 Step 1: Loading Caselaw Access Project (CAP) dataset...")
    try:
        dataset_loader = CaselawDatasetLoader()
        dataset_info = dataset_loader.load_sample_dataset(max_samples=args.max_samples)
        print(f"✅ Loaded {dataset_info.get('total_cases', 0)} cases")
        print(f"   - Courts: {len(dataset_info.get('courts', []))}")
        print(f"   - Topics: {len(dataset_info.get('topics', []))}")
        print(f"   - Year range: {dataset_info.get('year_range', 'Unknown')}")
    except Exception as e:
        print(f"⚠️  Using mock dataset due to: {e}")
        dataset_info = create_mock_dataset_info(args.max_samples)
    
    # Step 2: Process with GraphRAG
    print("\n🕸️  Step 2: Processing with GraphRAG...")
    try:
        processor = CaselawGraphRAGProcessor()
        graph_stats = processor.process_cases(dataset_info)
        print(f"✅ GraphRAG processing complete")
        print(f"   - Total nodes: {graph_stats.get('total_nodes', 0)}")
        print(f"   - Total edges: {graph_stats.get('total_edges', 0)}")
        print(f"   - Entity types: {len(graph_stats.get('entity_types', []))}")
        print(f"   - Relationship types: {len(graph_stats.get('relationship_types', []))}")
        
        # Show some example queries
        print("\n🔍 Example queries:")
        queries = ["civil rights", "Supreme Court", "constitutional law"]
        for query in queries:
            results = processor.query_graph(query, limit=3)
            print(f"   '{query}': {len(results)} results")
            
    except Exception as e:
        print(f"⚠️  Using mock GraphRAG results due to: {e}")
        graph_stats = create_mock_graph_stats()
    
    # Step 3: Temporal deontic logic (optional)
    if args.temporal_logic:
        print("\n⏰ Step 3: Temporal Deontic Logic Analysis...")
        try:
            tdl_processor = TemporalDeonticLogicProcessor()
            
            # Process a sample lineage (e.g., qualified immunity doctrine)
            sample_lineage = create_sample_legal_lineage()
            analysis_results = tdl_processor.analyze_lineage(sample_lineage)
            
            print(f"✅ Temporal analysis complete")
            print(f"   - Doctrinal evolution steps: {len(analysis_results.get('evolution_steps', []))}")
            print(f"   - Generated theorems: {len(analysis_results.get('theorems', []))}")
            print(f"   - Consistency checks: {analysis_results.get('consistency_score', 0):.2f}")
            
            # Optional: export to IPLD format
            if analysis_results.get('theorems'):
                print("   - IPLD export structure prepared")
                
        except Exception as e:
            print(f"⚠️  Temporal logic analysis skipped due to: {e}")
    
    # Step 4: Launch dashboard (optional)
    if args.run_dashboard:
        print(f"\n🌐 Step 4: Launching Caselaw Dashboard on port {args.port}...")
        try:
            dashboard = CaselawDashboard(port=args.port)
            print(f"✅ Dashboard available at: http://127.0.0.1:{args.port}")
            print("   - Main UI: /")
            print("   - Search API: /api/search")
            print("   - Legal doctrines: /api/legal-doctrines")
            
            if args.temporal_logic:
                print("   - Temporal analysis: /api/temporal-analysis")
            
            print("\n🎯 Press Ctrl+C to stop the server")
            dashboard.run()
            
        except Exception as e:
            print(f"⚠️  Dashboard launch failed: {e}")
            print("To manually start the dashboard:")
            print(f"python -c \"from ipfs_accelerate_py.caselaw_dashboard import CaselawDashboard; CaselawDashboard({args.port}).run()\"")
    
    print("\n✅ Demonstration complete!")

def create_mock_dataset_info(max_samples: int) -> Dict[str, Any]:
    """Create mock dataset info for demonstration."""
    return {
        'total_cases': max_samples,
        'courts': ['Supreme Court', 'Court of Appeals', 'District Court'],
        'topics': ['Civil Rights', 'Constitutional Law', 'Criminal Law', 'Contract Law'],
        'year_range': '1900-2023',
        'sample_cases': [
            {
                'id': 'us-1',
                'title': 'Brown v. Board of Education',
                'citation': '347 U.S. 483',
                'court': 'Supreme Court',
                'year': 1954,
                'topic': 'Civil Rights',
                'summary': 'Landmark case declaring racial segregation in public schools unconstitutional.'
            },
            {
                'id': 'us-2', 
                'title': 'Miranda v. Arizona',
                'citation': '384 U.S. 436',
                'court': 'Supreme Court', 
                'year': 1966,
                'topic': 'Criminal Law',
                'summary': 'Established Miranda rights for criminal suspects.'
            }
        ]
    }

def create_mock_graph_stats() -> Dict[str, Any]:
    """Create mock graph statistics."""
    return {
        'total_nodes': 1500,
        'total_edges': 3200,
        'entity_types': ['Case', 'Court', 'Judge', 'Legal Doctrine', 'Statute'],
        'relationship_types': ['CITES', 'OVERRULES', 'DISTINGUISHES', 'APPLIES'],
        'court_distribution': {
            'Supreme Court': 45,
            'Court of Appeals': 234,
            'District Court': 321
        },
        'topic_distribution': {
            'Civil Rights': 89,
            'Constitutional Law': 156,
            'Criminal Law': 203,
            'Contract Law': 112
        }
    }

def create_sample_legal_lineage() -> List[Dict[str, Any]]:
    """Create a sample legal doctrine lineage for temporal analysis."""
    return [
        {
            'case_id': 'pierson-v-ray-1967',
            'doctrine': 'qualified immunity',
            'year': 1967,
            'holding': 'Police officers acting under statutory authority have qualified immunity',
            'context': 'Initial qualified immunity doctrine'
        },
        {
            'case_id': 'harlow-v-fitzgerald-1982', 
            'doctrine': 'qualified immunity',
            'year': 1982,
            'holding': 'Qualified immunity protects officials from liability unless rights were clearly established',
            'context': 'Objective reasonableness standard established'
        },
        {
            'case_id': 'saucier-v-katz-2001',
            'doctrine': 'qualified immunity', 
            'year': 2001,
            'holding': 'Two-step analysis: constitutional violation then clearly established law',
            'context': 'Mandatory sequencing introduced'
        }
    ]

if __name__ == '__main__':
    main()