"""
Text Processing Utilities

This module contains text processing functions for the RAG system including
entity extraction, key term extraction, and text normalization.
"""

import re
from typing import List

# Optional NLTK support
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    sent_tokenize = None

class TextProcessor:
    """Centralized text processing utilities"""
    
    @staticmethod
    def extract_key_terms(content: str, max_terms: int = 10) -> List[str]:
        """Extract key terms using frequency and capitalization heuristics"""
        # Extract capitalized terms
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', content) + re.findall(r"'([^']+)'", content)
        
        # Extract technical terms
        technical_terms = re.findall(r'\b[A-Z]{2,}\b|\b\w*\d+\w*\b', content)
        
        # Combine and filter
        all_terms = capitalized_terms + quoted_terms + technical_terms
        common_words = {'The', 'This', 'That', 'These', 'Those', 'And', 'Or', 'But', 'For', 'With'}
        key_terms = [term for term in all_terms if len(term) > 2 and term not in common_words]
        
        return sorted(list(set(key_terms)), key=len, reverse=True)[:max_terms]
    
    @staticmethod
    def extract_entities(content: str) -> List[str]:
        """Extract named entities using pattern matching"""
        entities = []
        
        # Date patterns
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}\b'
        ]
        
        # Number patterns
        number_patterns = [
            r'\b\d+(?:\.\d+)?\s*%\b',
            r'\b\d+(?:\.\d+)?\s*(?:kg|g|lb|oz|mm|cm|m|km|ft|in|miles?)\b',
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b'
        ]
        
        for pattern in date_patterns + number_patterns:
            entities.extend(re.findall(pattern, content, re.IGNORECASE))
        
        return list(set(entities))
    
    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores
        
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences using available tools"""
        if HAS_NLTK:
            return sent_tokenize(text)
        else:
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip() and len(s.split()) > 2]

__all__ = ['TextProcessor', 'HAS_NLTK']