"""
Confidence Calculation for RAG Responses

This module calculates confidence scores for RAG responses based on multiple factors
including search quality, source diversity, and answer completeness.
"""

import numpy as np
from typing import List, Dict, Any
from .models import SearchResult

class ConfidenceCalculator:
    """Centralized confidence calculation"""
    
    @staticmethod
    def calculate_confidence(question: str, search_results: List[SearchResult],
                           generation_result: Dict[str, Any]) -> float:
        """Calculate confidence score based on multiple factors"""
        factors = []
        
        # Search quality factors
        if search_results:
            hybrid_scores = [r.hybrid_score for r in search_results[:5]]
            avg_search_quality = np.mean(hybrid_scores) if hybrid_scores else 0.5
            factors.append(('search_quality', avg_search_quality, 0.3))
            
            # Source diversity
            chunk_types = set(r.chunk.chunk_type for r in search_results[:5])
            diversity_score = min(1.0, len(chunk_types) / 3)
            factors.append(('source_diversity', diversity_score, 0.15))
            
            # Semantic density
            avg_density = np.mean([r.chunk.semantic_density for r in search_results[:5]])
            factors.append(('semantic_density', avg_density, 0.1))
        
        # Generation quality factors
        answer_length = len(generation_result.get('answer', '').split())
        length_score = min(1.0, answer_length / 50)
        factors.append(('answer_completeness', length_score, 0.2))
        
        # Query-answer alignment
        question_words = set(question.lower().split())
        answer_words = set(generation_result.get('answer', '').lower().split())
        word_overlap = len(question_words.intersection(answer_words))
        alignment_score = min(1.0, word_overlap / max(len(question_words), 1))
        factors.append(('query_alignment', alignment_score, 0.15))
        
        # Context utilization
        sources_used = generation_result.get('sources_used', 0)
        utilization_score = min(1.0, sources_used / 5)
        factors.append(('context_utilization', utilization_score, 0.1))
        
        # Calculate weighted confidence
        total_weight = sum(weight for _, _, weight in factors)
        confidence = sum(score * weight for _, score, weight in factors) / total_weight
        
        # Apply conservative adjustment
        confidence = confidence * 0.85 + 0.15
        
        return min(1.0, confidence)

__all__ = ['ConfidenceCalculator']