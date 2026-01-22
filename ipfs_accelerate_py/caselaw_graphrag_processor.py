"""
Caselaw GraphRAG Processor

This module processes legal cases through GraphRAG (Graph-based Retrieval Augmented Generation)
to build knowledge graphs and enable sophisticated legal analysis and querying.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaselawGraphRAGProcessor:
    """Processes legal cases using GraphRAG techniques."""
    
    def __init__(self):
        """Initialize the GraphRAG processor."""
        self.knowledge_graph = {
            'nodes': {},
            'edges': [],
            'node_types': set(),
            'edge_types': set()
        }
        self.entity_extractor = LegalEntityExtractor()
        self.relationship_detector = LegalRelationshipDetector()
        
    def process_cases(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process cases to build knowledge graph.
        
        Args:
            dataset_info: Dataset information from the loader
            
        Returns:
            Graph statistics and processing results
        """
        logger.info("Starting GraphRAG processing of legal cases")
        
        cases = dataset_info.get('all_cases', dataset_info.get('sample_cases', []))
        
        # Process each case
        for case in cases:
            self._process_single_case(case)
        
        # Generate statistics
        stats = self._generate_graph_stats()
        logger.info(f"GraphRAG processing complete: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        
        return stats
    
    def _process_single_case(self, case: Dict[str, Any]) -> None:
        """Process a single legal case.
        
        Args:
            case: Case information dictionary
        """
        case_id = case['id']
        
        # Add case node
        self.knowledge_graph['nodes'][case_id] = {
            'type': 'Case',
            'title': case['title'],
            'citation': case['citation'],
            'court': case['court'],
            'year': case['year'],
            'topic': case['topic'],
            'summary': case['summary']
        }
        self.knowledge_graph['node_types'].add('Case')
        
        # Extract entities from case text
        entities = self.entity_extractor.extract_entities(case)
        
        # Add entity nodes and relationships
        for entity in entities:
            entity_id = f"{entity['type'].lower()}_{entity['text'].lower().replace(' ', '_')}"
            
            # Add entity node if not exists
            if entity_id not in self.knowledge_graph['nodes']:
                self.knowledge_graph['nodes'][entity_id] = {
                    'type': entity['type'],
                    'text': entity['text'],
                    'canonical_form': entity.get('canonical_form', entity['text'])
                }
                self.knowledge_graph['node_types'].add(entity['type'])
            
            # Add relationship between case and entity
            self._add_edge(case_id, entity_id, 'CONTAINS')
        
        # Detect relationships between cases
        self._detect_case_relationships(case)
    
    def _add_edge(self, source: str, target: str, relationship: str, properties: Optional[Dict] = None) -> None:
        """Add an edge to the knowledge graph.
        
        Args:
            source: Source node ID
            target: Target node ID
            relationship: Relationship type
            properties: Optional edge properties
        """
        edge = {
            'source': source,
            'target': target,
            'relationship': relationship,
            'properties': properties or {}
        }
        self.knowledge_graph['edges'].append(edge)
        self.knowledge_graph['edge_types'].add(relationship)
    
    def _detect_case_relationships(self, case: Dict[str, Any]) -> None:
        """Detect relationships between this case and existing cases.
        
        Args:
            case: Current case being processed
        """
        # Simple citation detection in summary/text
        summary = case.get('summary', '') + ' ' + case.get('full_text', '')
        
        # Look for citations to other cases
        for existing_case_id, existing_case in self.knowledge_graph['nodes'].items():
            if existing_case.get('type') != 'Case' or existing_case_id == case['id']:
                continue
            
            existing_title = existing_case.get('title', '')
            
            # Check if this case cites the existing case
            if existing_title.lower() in summary.lower():
                self._add_edge(case['id'], existing_case_id, 'CITES')
            
            # Check for temporal relationships
            current_year = case.get('year', 0)
            existing_year = existing_case.get('year', 0)
            
            if abs(current_year - existing_year) <= 5 and case.get('topic') == existing_case.get('topic'):
                if current_year > existing_year:
                    self._add_edge(case['id'], existing_case_id, 'FOLLOWS')
                elif current_year < existing_year:
                    self._add_edge(existing_case_id, case['id'], 'FOLLOWS')
    
    def _generate_graph_stats(self) -> Dict[str, Any]:
        """Generate statistics about the knowledge graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        nodes_by_type = defaultdict(int)
        edges_by_type = defaultdict(int)
        court_distribution = defaultdict(int)
        topic_distribution = defaultdict(int)
        
        # Count nodes by type
        for node_id, node in self.knowledge_graph['nodes'].items():
            node_type = node.get('type', 'Unknown')
            nodes_by_type[node_type] += 1
            
            if node_type == 'Case':
                court = node.get('court', 'Unknown')
                topic = node.get('topic', 'Unknown')
                court_distribution[court] += 1
                topic_distribution[topic] += 1
        
        # Count edges by type
        for edge in self.knowledge_graph['edges']:
            edge_type = edge.get('relationship', 'Unknown')
            edges_by_type[edge_type] += 1
        
        return {
            'total_nodes': len(self.knowledge_graph['nodes']),
            'total_edges': len(self.knowledge_graph['edges']),
            'entity_types': list(self.knowledge_graph['node_types']),
            'relationship_types': list(self.knowledge_graph['edge_types']),
            'nodes_by_type': dict(nodes_by_type),
            'edges_by_type': dict(edges_by_type),
            'court_distribution': dict(court_distribution),
            'topic_distribution': dict(topic_distribution)
        }
    
    def query_graph(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query the knowledge graph.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching cases and entities
        """
        query_lower = query.lower()
        results = []
        
        # Search through nodes
        for node_id, node in self.knowledge_graph['nodes'].items():
            score = 0
            
            # Check title match
            title = node.get('title', node.get('text', ''))
            if query_lower in title.lower():
                score += 2
            
            # Check summary match
            summary = node.get('summary', '')
            if query_lower in summary.lower():
                score += 1
            
            # Check topic match
            topic = node.get('topic', '')
            if query_lower in topic.lower():
                score += 1.5
            
            # Check court match
            court = node.get('court', '')
            if query_lower in court.lower():
                score += 0.5
            
            if score > 0:
                result = node.copy()
                result['id'] = node_id
                result['relevance'] = min(score / 3.0, 1.0)  # Normalize to 0-1
                results.append(result)
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        return results[:limit]
    
    def get_case_connections(self, case_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all connections for a specific case.
        
        Args:
            case_id: ID of the case
            
        Returns:
            Dictionary of connections grouped by relationship type
        """
        connections = defaultdict(list)
        
        for edge in self.knowledge_graph['edges']:
            if edge['source'] == case_id:
                target_node = self.knowledge_graph['nodes'].get(edge['target'])
                if target_node:
                    connection = {
                        'node_id': edge['target'],
                        'node': target_node,
                        'relationship': edge['relationship'],
                        'direction': 'outgoing'
                    }
                    connections[edge['relationship']].append(connection)
            elif edge['target'] == case_id:
                source_node = self.knowledge_graph['nodes'].get(edge['source'])
                if source_node:
                    connection = {
                        'node_id': edge['source'],
                        'node': source_node,
                        'relationship': edge['relationship'],
                        'direction': 'incoming'
                    }
                    connections[edge['relationship']].append(connection)
        
        return dict(connections)


class LegalEntityExtractor:
    """Extracts legal entities from case text."""
    
    def __init__(self):
        """Initialize the entity extractor."""
        # Legal entity patterns
        self.patterns = {
            'Judge': r'Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            'Statute': r'(\d+\s+U\.S\.C\.?\s*§?\s*\d+|\d+\s+USC\s+\d+)',
            'Court': r'(Supreme Court|Court of Appeals|District Court|Circuit Court)',
            'Legal_Doctrine': r'(qualified immunity|due process|equal protection|judicial review|stare decisis)',
            'Amendment': r'(First Amendment|Second Amendment|Fourth Amendment|Fifth Amendment|Fourteenth Amendment)'
        }
    
    def extract_entities(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from a case.
        
        Args:
            case: Case information
            
        Returns:
            List of extracted entities
        """
        text = f"{case.get('title', '')} {case.get('summary', '')} {case.get('full_text', '')}"
        entities = []
        
        # Add court as entity
        if case.get('court'):
            entities.append({
                'type': 'Court',
                'text': case['court'],
                'canonical_form': case['court']
            })
        
        # Add topic as legal doctrine
        if case.get('topic'):
            entities.append({
                'type': 'Legal_Doctrine',
                'text': case['topic'],
                'canonical_form': case['topic']
            })
        
        # Extract using patterns
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': entity_type,
                    'text': match,
                    'canonical_form': match.strip()
                })
        
        return entities


class LegalRelationshipDetector:
    """Detects relationships between legal entities and cases."""
    
    def __init__(self):
        """Initialize the relationship detector."""
        self.relationship_patterns = {
            'OVERRULES': r'overrul[es]|overruled|reverses?|reversed',
            'DISTINGUISHES': r'distinguish[es]|distinguished|differs?|differed',
            'FOLLOWS': r'follows?|followed|adheres?\s+to|consistent\s+with',
            'APPLIES': r'appl[ys]|applied|applies|invoke[ds]|invoked'
        }
    
    def detect_relationships(self, case1: Dict[str, Any], case2: Dict[str, Any]) -> List[str]:
        """Detect relationships between two cases.
        
        Args:
            case1: First case
            case2: Second case
            
        Returns:
            List of detected relationships
        """
        text1 = f"{case1.get('summary', '')} {case1.get('full_text', '')}"
        text2 = f"{case2.get('summary', '')} {case2.get('full_text', '')}"
        
        relationships = []
        
        for rel_type, pattern in self.relationship_patterns.items():
            # Check if case1 has this relationship with case2
            if (case2.get('title', '').lower() in text1.lower() and 
                re.search(pattern, text1, re.IGNORECASE)):
                relationships.append(rel_type)
        
        return relationships