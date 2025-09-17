"""
Temporal Deontic Logic Processor

This module implements temporal deontic logic analysis for legal case lineages,
tracking the evolution of legal doctrines over time and generating formal
representations of legal reasoning.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalDeonticLogicProcessor:
    """Processes legal doctrine evolution using temporal deontic logic."""
    
    def __init__(self):
        """Initialize the temporal deontic logic processor."""
        self.temporal_operators = {
            'F': 'Eventually (Future)',
            'G': 'Always (Globally)', 
            'X': 'Next',
            'U': 'Until',
            'R': 'Release'
        }
        
        self.deontic_operators = {
            'O': 'Obligatory',
            'P': 'Permitted',
            'F': 'Forbidden',
            'I': 'Indifferent'
        }
        
        self.modal_operators = {
            'K': 'Known',
            'B': 'Believed',
            'D': 'Desired'
        }
    
    def analyze_lineage(self, lineage: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a legal doctrine lineage over time.
        
        Args:
            lineage: List of cases in chronological order
            
        Returns:
            Analysis results including evolution steps, theorems, and consistency
        """
        logger.info(f"Analyzing temporal lineage of {len(lineage)} cases")
        
        # Sort by year to ensure chronological order
        sorted_lineage = sorted(lineage, key=lambda x: x.get('year', 0))
        
        # Analyze evolution steps
        evolution_steps = self._analyze_evolution_steps(sorted_lineage)
        
        # Generate formal theorems
        theorems = self._generate_theorems(sorted_lineage, evolution_steps)
        
        # Check consistency
        consistency_score = self._check_consistency(sorted_lineage, evolution_steps)
        
        # Generate temporal logic formulas
        temporal_formulas = self._generate_temporal_formulas(evolution_steps)
        
        results = {
            'lineage_id': f"doctrine_{sorted_lineage[0].get('doctrine', 'unknown')}",
            'evolution_steps': evolution_steps,
            'theorems': theorems,
            'temporal_formulas': temporal_formulas,
            'consistency_score': consistency_score,
            'analysis_metadata': {
                'total_cases': len(sorted_lineage),
                'time_span': sorted_lineage[-1].get('year', 0) - sorted_lineage[0].get('year', 0),
                'doctrine': sorted_lineage[0].get('doctrine'),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Analysis complete: {len(theorems)} theorems, consistency: {consistency_score:.2f}")
        return results
    
    def _analyze_evolution_steps(self, lineage: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze evolution steps between consecutive cases.
        
        Args:
            lineage: Chronologically sorted lineage
            
        Returns:
            List of evolution step analyses
        """
        evolution_steps = []
        
        for i in range(len(lineage) - 1):
            current_case = lineage[i]
            next_case = lineage[i + 1]
            
            step = {
                'step_id': f"step_{i}",
                'from_case': current_case.get('case_id'),
                'to_case': next_case.get('case_id'),
                'from_year': current_case.get('year'),
                'to_year': next_case.get('year'),
                'time_gap': next_case.get('year', 0) - current_case.get('year', 0),
                'evolution_type': self._classify_evolution_type(current_case, next_case),
                'doctrinal_change': self._analyze_doctrinal_change(current_case, next_case),
                'temporal_relationship': self._determine_temporal_relationship(current_case, next_case)
            }
            
            evolution_steps.append(step)
        
        return evolution_steps
    
    def _classify_evolution_type(self, case1: Dict[str, Any], case2: Dict[str, Any]) -> str:
        """Classify the type of evolution between two cases.
        
        Args:
            case1: Earlier case
            case2: Later case
            
        Returns:
            Evolution type classification
        """
        holding1 = case1.get('holding', '').lower()
        holding2 = case2.get('holding', '').lower()
        
        # Simple heuristics for classification
        if 'overrule' in holding2 or 'reverse' in holding2:
            return 'OVERRULING'
        elif 'expand' in holding2 or 'broaden' in holding2:
            return 'EXPANSION'
        elif 'narrow' in holding2 or 'limit' in holding2:
            return 'LIMITATION'
        elif 'clarif' in holding2 or 'explain' in holding2:
            return 'CLARIFICATION'
        else:
            return 'CONTINUATION'
    
    def _analyze_doctrinal_change(self, case1: Dict[str, Any], case2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the doctrinal change between two cases.
        
        Args:
            case1: Earlier case
            case2: Later case
            
        Returns:
            Doctrinal change analysis
        """
        return {
            'doctrine': case1.get('doctrine'),
            'previous_holding': case1.get('holding'),
            'new_holding': case2.get('holding'),
            'change_magnitude': self._calculate_change_magnitude(case1, case2),
            'change_direction': self._determine_change_direction(case1, case2),
            'affected_principles': self._identify_affected_principles(case1, case2)
        }
    
    def _calculate_change_magnitude(self, case1: Dict[str, Any], case2: Dict[str, Any]) -> float:
        """Calculate the magnitude of change between two cases.
        
        Args:
            case1: Earlier case
            case2: Later case
            
        Returns:
            Change magnitude (0.0 to 1.0)
        """
        # Simple text similarity-based approach
        holding1 = case1.get('holding', '').lower().split()
        holding2 = case2.get('holding', '').lower().split()
        
        if not holding1 or not holding2:
            return 0.5  # Default moderate change
        
        # Calculate Jaccard similarity
        set1 = set(holding1)
        set2 = set(holding2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        similarity = intersection / union if union > 0 else 0
        magnitude = 1.0 - similarity  # Change magnitude is inverse of similarity
        
        return min(max(magnitude, 0.0), 1.0)
    
    def _determine_change_direction(self, case1: Dict[str, Any], case2: Dict[str, Any]) -> str:
        """Determine the direction of doctrinal change.
        
        Args:
            case1: Earlier case
            case2: Later case
            
        Returns:
            Change direction
        """
        holding2 = case2.get('holding', '').lower()
        
        expansive_keywords = ['expand', 'broaden', 'extend', 'increase']
        restrictive_keywords = ['narrow', 'limit', 'restrict', 'reduce']
        
        expansive_count = sum(1 for keyword in expansive_keywords if keyword in holding2)
        restrictive_count = sum(1 for keyword in restrictive_keywords if keyword in holding2)
        
        if expansive_count > restrictive_count:
            return 'EXPANSIVE'
        elif restrictive_count > expansive_count:
            return 'RESTRICTIVE'
        else:
            return 'NEUTRAL'
    
    def _identify_affected_principles(self, case1: Dict[str, Any], case2: Dict[str, Any]) -> List[str]:
        """Identify legal principles affected by the change.
        
        Args:
            case1: Earlier case
            case2: Later case
            
        Returns:
            List of affected principles
        """
        # Extract legal principles from holdings
        principles = []
        
        for case in [case1, case2]:
            holding = case.get('holding', '').lower()
            
            # Common legal principles
            if 'due process' in holding:
                principles.append('Due Process')
            if 'equal protection' in holding:
                principles.append('Equal Protection')
            if 'reasonable' in holding:
                principles.append('Reasonableness Standard')
            if 'immunity' in holding:
                principles.append('Immunity Doctrine')
            if 'constitutional' in holding:
                principles.append('Constitutional Law')
        
        return list(set(principles))
    
    def _determine_temporal_relationship(self, case1: Dict[str, Any], case2: Dict[str, Any]) -> str:
        """Determine the temporal relationship between cases.
        
        Args:
            case1: Earlier case
            case2: Later case
            
        Returns:
            Temporal relationship type
        """
        time_gap = case2.get('year', 0) - case1.get('year', 0)
        
        if time_gap <= 5:
            return 'IMMEDIATE_SUCCESSOR'
        elif time_gap <= 15:
            return 'NEAR_SUCCESSOR'
        else:
            return 'DISTANT_SUCCESSOR'
    
    def _generate_theorems(self, lineage: List[Dict[str, Any]], evolution_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate formal theorems from the analysis.
        
        Args:
            lineage: Case lineage
            evolution_steps: Evolution step analysis
            
        Returns:
            List of generated theorems
        """
        theorems = []
        
        # Theorem 1: Temporal Precedence
        for i, case in enumerate(lineage):
            theorem = {
                'theorem_id': f"temporal_precedence_{i}",
                'type': 'TEMPORAL_PRECEDENCE',
                'formal_statement': f"G(year({case.get('case_id')}) = {case.get('year')})",
                'natural_language': f"Case {case.get('case_id')} was decided in year {case.get('year')}",
                'modalities': ['temporal']
            }
            theorems.append(theorem)
        
        # Theorem 2: Doctrinal Evolution  
        for step in evolution_steps:
            theorem = {
                'theorem_id': f"evolution_{step['step_id']}",
                'type': 'DOCTRINAL_EVOLUTION',
                'formal_statement': f"F(doctrine_evolves({step['from_case']}, {step['to_case']}, {step['evolution_type']}))",
                'natural_language': f"Doctrine evolved from {step['from_case']} to {step['to_case']} via {step['evolution_type']}",
                'modalities': ['temporal', 'deontic']
            }
            theorems.append(theorem)
        
        # Theorem 3: Consistency Principle
        if len(lineage) > 1:
            theorem = {
                'theorem_id': 'consistency_principle',
                'type': 'CONSISTENCY',
                'formal_statement': f"G(consistent_evolution({lineage[0].get('doctrine')}))",
                'natural_language': f"The evolution of {lineage[0].get('doctrine')} maintains internal consistency",
                'modalities': ['temporal', 'deontic', 'epistemic']
            }
            theorems.append(theorem)
        
        return theorems
    
    def _generate_temporal_formulas(self, evolution_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate temporal logic formulas.
        
        Args:
            evolution_steps: Evolution step analysis
            
        Returns:
            List of temporal formulas
        """
        formulas = []
        
        for step in evolution_steps:
            # Temporal succession formula
            formula = {
                'formula_id': f"succession_{step['step_id']}",
                'temporal_formula': f"X(holds({step['to_case']}) → ¬holds({step['from_case']}))",
                'description': f"After {step['to_case']}, {step['from_case']} no longer holds",
                'operators_used': ['X', '→', '¬']
            }
            formulas.append(formula)
            
            # Evolution constraint formula
            if step['evolution_type'] == 'OVERRULING':
                formula = {
                    'formula_id': f"overruling_{step['step_id']}",
                    'temporal_formula': f"F(overrules({step['to_case']}, {step['from_case']}))",
                    'description': f"{step['to_case']} eventually overrules {step['from_case']}",
                    'operators_used': ['F']
                }
                formulas.append(formula)
        
        return formulas
    
    def _check_consistency(self, lineage: List[Dict[str, Any]], evolution_steps: List[Dict[str, Any]]) -> float:
        """Check the consistency of the doctrinal evolution.
        
        Args:
            lineage: Case lineage  
            evolution_steps: Evolution steps
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        consistency_factors = []
        
        # Factor 1: Temporal ordering consistency
        temporal_consistency = 1.0
        for i in range(len(lineage) - 1):
            if lineage[i].get('year', 0) >= lineage[i + 1].get('year', 0):
                temporal_consistency -= 0.1
        consistency_factors.append(max(temporal_consistency, 0.0))
        
        # Factor 2: Doctrinal coherence
        doctrine_consistency = 1.0
        base_doctrine = lineage[0].get('doctrine') if lineage else None
        for case in lineage:
            if case.get('doctrine') != base_doctrine:
                doctrine_consistency -= 0.2
        consistency_factors.append(max(doctrine_consistency, 0.0))
        
        # Factor 3: Evolution type consistency
        evolution_consistency = 1.0
        overruling_count = sum(1 for step in evolution_steps if step['evolution_type'] == 'OVERRULING')
        if overruling_count > 1:  # Multiple overrulings suggest inconsistency
            evolution_consistency -= 0.1 * (overruling_count - 1)
        consistency_factors.append(max(evolution_consistency, 0.0))
        
        # Calculate overall consistency score
        overall_consistency = sum(consistency_factors) / len(consistency_factors) if consistency_factors else 0.0
        return min(max(overall_consistency, 0.0), 1.0)
    
    def export_to_ipld(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Export analysis results to IPLD format.
        
        Args:
            analysis_results: Results from analyze_lineage
            
        Returns:
            IPLD-formatted data structure
        """
        ipld_structure = {
            '@context': 'https://schema.org/LegalCase',
            '@type': 'TemporalDeonticAnalysis',
            'lineage_id': analysis_results['lineage_id'],
            'metadata': analysis_results['analysis_metadata'],
            'evolution_graph': {
                'nodes': [
                    {
                        'id': step['from_case'],
                        'type': 'LegalCase',
                        'year': step['from_year']
                    } for step in analysis_results['evolution_steps']
                ],
                'edges': [
                    {
                        'source': step['from_case'],
                        'target': step['to_case'], 
                        'relationship': step['evolution_type'],
                        'temporal_gap': step['time_gap']
                    } for step in analysis_results['evolution_steps']
                ]
            },
            'formal_theorems': analysis_results['theorems'],
            'temporal_formulas': analysis_results['temporal_formulas'],
            'consistency_metrics': {
                'score': analysis_results['consistency_score'],
                'validation_timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Exported analysis to IPLD format for {analysis_results['lineage_id']}")
        return ipld_structure