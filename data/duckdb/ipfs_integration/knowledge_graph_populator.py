"""
Knowledge Graph Populator for IPFS Integration (Phase 2C)

This module populates and manages the benchmark knowledge graph with relationships,
semantic search, and intelligent insights.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeGraphPopulator:
    """
    Populates and manages the benchmark knowledge graph.
    
    Features:
    - Automatic relationship discovery from benchmark data
    - Semantic similarity computation
    - Performance pattern detection
    - Hardware compatibility mapping
    """
    
    def __init__(self, knowledge_graph=None):
        """
        Initialize the knowledge graph populator.
        
        Args:
            knowledge_graph: BenchmarkKnowledgeGraph instance
        """
        self.kg = knowledge_graph
        self.relationship_rules: List[Dict[str, Any]] = []
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # Default relationship rules
        self._setup_default_rules()
        
        logger.info("Initialized KnowledgeGraphPopulator")
    
    def _setup_default_rules(self):
        """Setup default relationship discovery rules."""
        self.relationship_rules = [
            {
                'name': 'model_family',
                'type': 'is_variant_of',
                'pattern': lambda node: node.get('model_family'),
                'strength': 0.9
            },
            {
                'name': 'hardware_compatibility',
                'type': 'runs_on',
                'pattern': lambda node: node.get('hardware_type'),
                'strength': 0.8
            },
            {
                'name': 'performance_similarity',
                'type': 'performs_similar_to',
                'pattern': lambda node: (node.get('throughput'), node.get('latency')),
                'strength': 0.7
            }
        ]
    
    def populate_from_benchmarks(
        self,
        benchmarks: List[Dict[str, Any]],
        discover_relationships: bool = True
    ) -> Dict[str, Any]:
        """
        Populate knowledge graph from benchmark results.
        
        Args:
            benchmarks: List of benchmark results
            discover_relationships: Whether to discover relationships automatically
        
        Returns:
            Population statistics
        """
        if not self.kg:
            raise ValueError("Knowledge graph not initialized")
        
        start_time = time.time()
        stats = {
            'nodes_added': 0,
            'edges_added': 0,
            'relationships_discovered': 0
        }
        
        # Add benchmark nodes
        for benchmark in benchmarks:
            node_id = self._create_node_id(benchmark)
            node_type = self._determine_node_type(benchmark)
            properties = self._extract_properties(benchmark)
            
            self.kg.add_node(node_id, node_type, properties)
            stats['nodes_added'] += 1
        
        # Discover and add relationships
        if discover_relationships:
            relationships = self.discover_relationships(benchmarks)
            for rel in relationships:
                self.kg.add_edge(
                    rel['from_node'],
                    rel['to_node'],
                    rel['relationship_type'],
                    rel['properties']
                )
                stats['edges_added'] += 1
            stats['relationships_discovered'] = len(relationships)
        
        execution_time = time.time() - start_time
        stats['execution_time'] = execution_time
        
        logger.info(f"Populated knowledge graph: {stats}")
        return stats
    
    def discover_relationships(
        self,
        benchmarks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Discover relationships between benchmarks using defined rules.
        
        Args:
            benchmarks: List of benchmark results
        
        Returns:
            List of discovered relationships
        """
        relationships = []
        
        # Apply each relationship rule
        for rule in self.relationship_rules:
            rule_relationships = self._apply_relationship_rule(benchmarks, rule)
            relationships.extend(rule_relationships)
        
        # Deduplicate relationships
        unique_relationships = self._deduplicate_relationships(relationships)
        
        return unique_relationships
    
    def _apply_relationship_rule(
        self,
        benchmarks: List[Dict[str, Any]],
        rule: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply a single relationship discovery rule.
        
        Args:
            benchmarks: List of benchmarks
            rule: Relationship rule to apply
        
        Returns:
            Discovered relationships
        """
        relationships = []
        pattern_func = rule['pattern']
        
        # Group benchmarks by pattern
        groups = defaultdict(list)
        for benchmark in benchmarks:
            try:
                pattern_value = pattern_func(benchmark)
                if pattern_value:
                    groups[pattern_value].append(benchmark)
            except Exception as e:
                logger.debug(f"Pattern matching error: {e}")
        
        # Create relationships within groups
        for pattern_value, group_benchmarks in groups.items():
            for i, bench1 in enumerate(group_benchmarks):
                for bench2 in group_benchmarks[i+1:]:
                    relationship = {
                        'from_node': self._create_node_id(bench1),
                        'to_node': self._create_node_id(bench2),
                        'relationship_type': rule['type'],
                        'properties': {
                            'rule_name': rule['name'],
                            'strength': rule['strength'],
                            'pattern_value': str(pattern_value)
                        }
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def compute_semantic_similarity(
        self,
        node1_id: str,
        node2_id: str,
        use_cache: bool = True
    ) -> float:
        """
        Compute semantic similarity between two nodes.
        
        Args:
            node1_id: First node ID
            node2_id: Second node ID
            use_cache: Whether to use cached similarity scores
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Check cache
        cache_key = tuple(sorted([node1_id, node2_id]))
        if use_cache and cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Get nodes
        if not self.kg:
            return 0.0
        
        node1 = self.kg.get_node(node1_id)
        node2 = self.kg.get_node(node2_id)
        
        if not node1 or not node2:
            return 0.0
        
        # Compute similarity based on multiple factors
        similarity = 0.0
        factors = 0
        
        # Type similarity
        if node1.get('type') == node2.get('type'):
            similarity += 0.3
        factors += 1
        
        # Property similarity
        prop1 = node1.get('properties', {})
        prop2 = node2.get('properties', {})
        
        # Model family similarity
        if prop1.get('model_family') == prop2.get('model_family'):
            similarity += 0.3
            factors += 1
        
        # Hardware compatibility
        if prop1.get('hardware_type') == prop2.get('hardware_type'):
            similarity += 0.2
            factors += 1
        
        # Performance similarity (if available)
        if 'throughput' in prop1 and 'throughput' in prop2:
            throughput_diff = abs(prop1['throughput'] - prop2['throughput'])
            max_throughput = max(prop1['throughput'], prop2['throughput'])
            if max_throughput > 0:
                throughput_sim = 1.0 - (throughput_diff / max_throughput)
                similarity += throughput_sim * 0.2
                factors += 1
        
        # Normalize
        if factors > 0:
            similarity = similarity / factors
        
        # Cache result
        self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    def find_similar_benchmarks(
        self,
        benchmark_id: str,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find benchmarks similar to a given benchmark.
        
        Args:
            benchmark_id: Benchmark node ID
            top_k: Number of similar benchmarks to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of similar benchmarks with similarity scores
        """
        if not self.kg:
            return []
        
        # Get all nodes
        all_nodes = self.kg.get_all_nodes()
        
        # Compute similarities
        similarities = []
        for node_id, node_data in all_nodes.items():
            if node_id == benchmark_id:
                continue
            
            similarity = self.compute_semantic_similarity(benchmark_id, node_id)
            
            if similarity >= min_similarity:
                similarities.append({
                    'node_id': node_id,
                    'similarity': similarity,
                    'properties': node_data.get('properties', {})
                })
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def detect_performance_patterns(
        self,
        benchmarks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect performance patterns in benchmark data.
        
        Args:
            benchmarks: List of benchmark results
        
        Returns:
            Detected patterns
        """
        patterns = []
        
        # Group by hardware
        hw_groups = defaultdict(list)
        for benchmark in benchmarks:
            hw_type = benchmark.get('hardware_type', 'unknown')
            hw_groups[hw_type].append(benchmark)
        
        # Analyze each hardware group
        for hw_type, hw_benchmarks in hw_groups.items():
            if len(hw_benchmarks) < 2:
                continue
            
            # Compute statistics
            throughputs = [b.get('throughput', 0) for b in hw_benchmarks if 'throughput' in b]
            latencies = [b.get('latency', 0) for b in hw_benchmarks if 'latency' in b]
            
            if throughputs:
                pattern = {
                    'pattern_type': 'hardware_performance',
                    'hardware_type': hw_type,
                    'benchmark_count': len(hw_benchmarks),
                    'avg_throughput': sum(throughputs) / len(throughputs),
                    'max_throughput': max(throughputs),
                    'min_throughput': min(throughputs)
                }
                
                if latencies:
                    pattern['avg_latency'] = sum(latencies) / len(latencies)
                
                patterns.append(pattern)
        
        return patterns
    
    def create_hardware_compatibility_map(
        self,
        benchmarks: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Create a compatibility map between models and hardware.
        
        Args:
            benchmarks: List of benchmark results
        
        Returns:
            Hardware compatibility map
        """
        compatibility_map = defaultdict(set)
        
        for benchmark in benchmarks:
            model = benchmark.get('model_name')
            hardware = benchmark.get('hardware_type')
            
            if model and hardware:
                # Check if benchmark was successful
                status = benchmark.get('status', 'unknown')
                if status in ['success', 'completed']:
                    compatibility_map[model].add(hardware)
        
        # Convert sets to lists
        return {model: list(hw_set) for model, hw_set in compatibility_map.items()}
    
    def _create_node_id(self, benchmark: Dict[str, Any]) -> str:
        """Create a unique node ID from benchmark data."""
        model = benchmark.get('model_name', 'unknown')
        hardware = benchmark.get('hardware_type', 'unknown')
        timestamp = benchmark.get('timestamp', time.time())
        return f"{model}_{hardware}_{int(timestamp)}"
    
    def _determine_node_type(self, benchmark: Dict[str, Any]) -> str:
        """Determine node type from benchmark data."""
        if 'model_name' in benchmark:
            return 'benchmark_result'
        return 'unknown'
    
    def _extract_properties(self, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant properties from benchmark data."""
        properties = {}
        
        # Copy relevant fields
        for field in ['model_name', 'hardware_type', 'throughput', 'latency',
                      'model_family', 'framework', 'precision', 'batch_size']:
            if field in benchmark:
                properties[field] = benchmark[field]
        
        return properties
    
    def _deduplicate_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate relationships."""
        seen = set()
        unique = []
        
        for rel in relationships:
            # Create a unique key
            key = (
                rel['from_node'],
                rel['to_node'],
                rel['relationship_type']
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        
        return unique
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the populated knowledge graph.
        
        Returns:
            Graph statistics
        """
        if not self.kg:
            return {}
        
        all_nodes = self.kg.get_all_nodes()
        
        # Count node types
        type_counts = defaultdict(int)
        for node_data in all_nodes.values():
            node_type = node_data.get('type', 'unknown')
            type_counts[node_type] += 1
        
        # Count edges by type
        edge_type_counts = defaultdict(int)
        for node_id in all_nodes:
            neighbors = self.kg.get_neighbors(node_id)
            for neighbor_data in neighbors:
                edge_type = neighbor_data.get('relationship')
                edge_type_counts[edge_type] += 1
        
        return {
            'total_nodes': len(all_nodes),
            'node_types': dict(type_counts),
            'total_edges': sum(edge_type_counts.values()),
            'edge_types': dict(edge_type_counts),
            'cache_size': len(self.similarity_cache)
        }
