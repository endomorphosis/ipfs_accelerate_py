"""
Caselaw Dataset Loader

This module handles loading and processing of caselaw datasets, particularly
from the Caselaw Access Project (CAP) and other legal databases.
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

# Try to import storage wrapper
try:
    from .common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False
        def get_storage_wrapper(*args, **kwargs):
            return None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaselawDatasetLoader:
    """Loads and processes caselaw datasets from various sources."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets. 
                      Uses CASELAW_CACHE_DIR env var or default if not provided.
        """
        self.cache_dir = Path(cache_dir or os.getenv('CASELAW_CACHE_DIR', './caselaw_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage wrapper for distributed storage
        self._storage = get_storage_wrapper() if HAVE_STORAGE_WRAPPER else None
        
        # Sample data for demonstration
        self.sample_cases = [
            {
                'id': 'us-scotus-1954-brown',
                'title': 'Brown v. Board of Education of Topeka',
                'citation': '347 U.S. 483 (1954)',
                'court': 'United States Supreme Court',
                'year': 1954,
                'topic': 'Civil Rights',
                'summary': 'Landmark Supreme Court case declaring state laws establishing racial segregation in public schools to be unconstitutional.',
                'full_text': 'In these days, it is doubtful that any child may reasonably be expected to succeed in life if he is denied the opportunity of an education...',
                'relevance': 0.95
            },
            {
                'id': 'us-scotus-1966-miranda',
                'title': 'Miranda v. Arizona',
                'citation': '384 U.S. 436 (1966)',
                'court': 'United States Supreme Court',
                'year': 1966,
                'topic': 'Criminal Law',
                'summary': 'Supreme Court case establishing Miranda rights - the right to remain silent and right to counsel.',
                'full_text': 'Prior to any questioning, the person must be warned that he has a right to remain silent...',
                'relevance': 0.92
            },
            {
                'id': 'us-scotus-1973-roe',
                'title': 'Roe v. Wade',
                'citation': '410 U.S. 113 (1973)',
                'court': 'United States Supreme Court',
                'year': 1973,
                'topic': 'Constitutional Law',
                'summary': 'Landmark decision establishing a constitutional right to abortion.',
                'full_text': 'This right of privacy, whether it be founded in the Fourteenth Amendment...',
                'relevance': 0.89
            },
            {
                'id': 'us-scotus-1963-gideon',
                'title': 'Gideon v. Wainwright',
                'citation': '372 U.S. 335 (1963)',
                'court': 'United States Supreme Court',
                'year': 1963,
                'topic': 'Criminal Law',
                'summary': 'Supreme Court case establishing the right to counsel in felony cases.',
                'full_text': 'The right of one charged with crime to counsel may not be deemed fundamental...',
                'relevance': 0.87
            },
            {
                'id': 'us-scotus-1896-plessy',
                'title': 'Plessy v. Ferguson',
                'citation': '163 U.S. 537 (1896)',
                'court': 'United States Supreme Court',
                'year': 1896,
                'topic': 'Civil Rights',
                'summary': 'Supreme Court case that established the "separate but equal" doctrine.',
                'full_text': 'Laws permitting, and even requiring, their separation in places where they are liable to be brought into contact...',
                'relevance': 0.84
            },
            {
                'id': 'us-scotus-1803-marbury',
                'title': 'Marbury v. Madison',
                'citation': '5 U.S. 137 (1803)',
                'court': 'United States Supreme Court',
                'year': 1803,
                'topic': 'Constitutional Law',
                'summary': 'Landmark case establishing the principle of judicial review.',
                'full_text': 'It is emphatically the province and duty of the judicial department to say what the law is...',
                'relevance': 0.91
            }
        ]
    
    def load_sample_dataset(self, max_samples: int = 100) -> Dict[str, Any]:
        """Load a sample dataset for demonstration purposes.
        
        Args:
            max_samples: Maximum number of cases to include
            
        Returns:
            Dictionary containing dataset information and sample cases
        """
        logger.info(f"Loading sample caselaw dataset (max {max_samples} cases)")
        
        # For demo purposes, cycle through sample cases to reach max_samples
        cases = []
        for i in range(max_samples):
            case = self.sample_cases[i % len(self.sample_cases)].copy()
            if i >= len(self.sample_cases):
                # Modify ID to make unique
                case['id'] = f"{case['id']}-{i}"
            cases.append(case)
        
        # Extract metadata
        courts = list(set(case['court'] for case in cases))
        topics = list(set(case['topic'] for case in cases))
        years = [case['year'] for case in cases]
        year_range = f"{min(years)}-{max(years)}"
        
        dataset_info = {
            'total_cases': len(cases),
            'courts': courts,
            'topics': topics,
            'year_range': year_range,
            'sample_cases': cases[:10],  # Return first 10 for display
            'all_cases': cases  # Full dataset for processing
        }
        
        logger.info(f"Loaded {len(cases)} cases from {len(courts)} courts")
        logger.info(f"Topics: {', '.join(topics)}")
        logger.info(f"Year range: {year_range}")
        
        return dataset_info
    
    def search_cases(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search cases by query string.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching cases
        """
        query_lower = query.lower()
        results = []
        
        for case in self.sample_cases:
            # Simple text matching
            if (query_lower in case['title'].lower() or 
                query_lower in case['summary'].lower() or
                query_lower in case['topic'].lower()):
                results.append(case)
            
            if len(results) >= limit:
                break
        
        logger.info(f"Found {len(results)} cases matching '{query}'")
        return results
    
    def get_legal_doctrines(self) -> List[Dict[str, Any]]:
        """Get a list of legal doctrines from the dataset.
        
        Returns:
            List of legal doctrine information
        """
        doctrines = [
            {
                'name': 'Qualified Immunity',
                'description': 'Legal doctrine protecting government officials from civil liability',
                'key_cases': ['Harlow v. Fitzgerald', 'Saucier v. Katz'],
                'evolution_timeline': [1967, 1982, 2001, 2020]
            },
            {
                'name': 'Judicial Review',
                'description': 'Power of courts to review and invalidate government actions',
                'key_cases': ['Marbury v. Madison', 'Cooper v. Aaron'],
                'evolution_timeline': [1803, 1958]
            },
            {
                'name': 'Separate but Equal',
                'description': 'Doctrine allowing racial segregation if facilities were equal',
                'key_cases': ['Plessy v. Ferguson', 'Brown v. Board'],
                'evolution_timeline': [1896, 1954]
            },
            {
                'name': 'Due Process',
                'description': 'Constitutional principle requiring fair treatment in legal proceedings',
                'key_cases': ['Gideon v. Wainwright', 'Miranda v. Arizona'],
                'evolution_timeline': [1963, 1966, 1970]
            }
        ]
        
        return doctrines
    
    def load_external_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load dataset from external path (like /storage/teraflopai).
        
        Args:
            dataset_path: Path to external dataset
            
        Returns:
            Dataset information
        """
        external_path = Path(dataset_path)
        
        if not external_path.exists():
            logger.warning(f"External dataset path not found: {dataset_path}")
            return self.load_sample_dataset()
        
        try:
            # Look for JSON files in the external path
            json_files = list(external_path.glob("*.json"))
            
            if json_files:
                logger.info(f"Found {len(json_files)} JSON files in {dataset_path}")
                # Load first JSON file as example
                external_data = None
                
                # Try distributed storage first
                if self._storage and hasattr(self._storage, 'is_distributed') and self._storage.is_distributed:
                    try:
                        cache_key = f"caselaw_external_{json_files[0].name}"
                        cached_content = self._storage.read_file(cache_key)
                        if cached_content:
                            external_data = json.loads(cached_content)
                            logger.debug(f"Loaded external dataset from distributed storage: {cache_key}")
                    except Exception as e:
                        logger.debug(f"Failed to read from distributed storage: {e}")
                
                # Always load from local filesystem
                with open(json_files[0], 'r') as f:
                    local_data = json.load(f)
                    if external_data is None:
                        external_data = local_data
                    
                    # Store in distributed storage for future access
                    if self._storage and hasattr(self._storage, 'is_distributed') and self._storage.is_distributed:
                        try:
                            cache_key = f"caselaw_external_{json_files[0].name}"
                            self._storage.write_file(json.dumps(local_data, indent=2), cache_key, pin=False)
                            logger.debug(f"Cached external dataset to distributed storage: {cache_key}")
                        except Exception as e:
                            logger.debug(f"Failed to write to distributed storage: {e}")
                
                # Convert to our format if needed
                if isinstance(external_data, list):
                    cases = external_data[:100]  # Limit for demo
                else:
                    cases = external_data.get('cases', self.sample_cases)
                
                return {
                    'total_cases': len(cases),
                    'courts': list(set(case.get('court', 'Unknown') for case in cases)),
                    'topics': list(set(case.get('topic', 'General') for case in cases)),
                    'year_range': 'External Dataset',
                    'sample_cases': cases[:10],
                    'all_cases': cases,
                    'source': str(dataset_path)
                }
            else:
                logger.info(f"No JSON files found in {dataset_path}, using sample data")
                
        except Exception as e:
            logger.error(f"Error loading external dataset: {e}")
        
        return self.load_sample_dataset()