"""
Confidence Calculation for RAG Responses

This module calculates confidence scores for RAG responses based on multiple factors
including search quality, source diversity, and answer completeness.
"""

import numpy as np
from typing import List, Dict, Any
from .models import SearchResult


class ConfidenceCalculator:
    """Centralized confidence calculation with configurable weights"""
    
    # Default weight configuration
    DEFAULT_WEIGHTS = {
        'search_quality': 0.3,
        'source_diversity': 0.15,
        'semantic_density': 0.1,
        'answer_completeness': 0.2,
        'query_alignment': 0.15,
        'context_utilization': 0.1
    }
    
    # Default thresholds
    DEFAULT_ANSWER_LENGTH_THRESHOLD = 50
    DEFAULT_MAX_SOURCES = 5
    DEFAULT_DIVERSITY_DENOMINATOR = 3
    CONSERVATIVE_ADJUSTMENT = 0.85
    CONSERVATIVE_BASE = 0.15
    
    def __init__(self, weights: Dict[str, float] = None):
        """Initialize calculator with optional custom weights"""
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()
    
    def _validate_weights(self):
        """Ensure weights are valid and normalized"""
        total = sum(self.weights.values())
        if total == 0:
            raise ValueError("Sum of weights cannot be zero")
        
        # Normalize weights to sum to 1.0
        for key in self.weights:
            self.weights[key] = self.weights[key] / total
    
    def calculate_confidence(self, question: str, search_results: List[SearchResult],
                           generation_result: Dict[str, Any]) -> float:
        """Calculate confidence score based on multiple factors using configured weights"""
        factors = []
        
        # Search quality factors
        if search_results:
            # Average search quality
            hybrid_scores = [r.hybrid_score for r in search_results[:self.DEFAULT_MAX_SOURCES]]
            avg_search_quality = np.mean(hybrid_scores) if hybrid_scores else 0.5
            factors.append(('search_quality', avg_search_quality, self.weights['search_quality']))
            
            # Source diversity
            chunk_types = set(r.chunk.chunk_type for r in search_results[:self.DEFAULT_MAX_SOURCES])
            diversity_score = min(1.0, len(chunk_types) / self.DEFAULT_DIVERSITY_DENOMINATOR)
            factors.append(('source_diversity', diversity_score, self.weights['source_diversity']))
            
            # Semantic density
            avg_density = np.mean([r.chunk.semantic_density for r in search_results[:self.DEFAULT_MAX_SOURCES]])
            factors.append(('semantic_density', avg_density, self.weights['semantic_density']))
        
        # Generation quality factors
        answer_length = len(generation_result.get('answer', '').split())
        length_score = min(1.0, answer_length / self.DEFAULT_ANSWER_LENGTH_THRESHOLD)
        factors.append(('answer_completeness', length_score, self.weights['answer_completeness']))
        
        # Query-answer alignment
        question_words = set(question.lower().split())
        answer_words = set(generation_result.get('answer', '').lower().split())
        word_overlap = len(question_words.intersection(answer_words))
        alignment_score = min(1.0, word_overlap / max(len(question_words), 1))
        factors.append(('query_alignment', alignment_score, self.weights['query_alignment']))
        
        # Context utilization
        sources_used = generation_result.get('sources_used', generation_result.get('context_used', 0))
        utilization_score = min(1.0, sources_used / self.DEFAULT_MAX_SOURCES)
        factors.append(('context_utilization', utilization_score, self.weights['context_utilization']))
        
        # Calculate weighted confidence
        total_weight = sum(weight for _, _, weight in factors)
        if total_weight == 0:
            return 0.5  # Neutral confidence if no factors
        
        confidence = sum(score * weight for _, score, weight in factors) / total_weight
        
        # Apply conservative adjustment to avoid overconfidence
        confidence = confidence * self.CONSERVATIVE_ADJUSTMENT + self.CONSERVATIVE_BASE
        
        return min(1.0, max(0.0, confidence))
    
    def get_detailed_confidence_breakdown(self, question: str, search_results: List[SearchResult],
                                         generation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed breakdown of confidence calculation for debugging/analysis"""
        factors = []
        
        # Calculate all factors (same as calculate_confidence but return details)
        if search_results:
            hybrid_scores = [r.hybrid_score for r in search_results[:self.DEFAULT_MAX_SOURCES]]
            avg_search_quality = np.mean(hybrid_scores) if hybrid_scores else 0.5
            factors.append({
                'name': 'search_quality',
                'score': avg_search_quality,
                'weight': self.weights['search_quality'],
                'contribution': avg_search_quality * self.weights['search_quality']
            })
            
            chunk_types = set(r.chunk.chunk_type for r in search_results[:self.DEFAULT_MAX_SOURCES])
            diversity_score = min(1.0, len(chunk_types) / self.DEFAULT_DIVERSITY_DENOMINATOR)
            factors.append({
                'name': 'source_diversity',
                'score': diversity_score,
                'weight': self.weights['source_diversity'],
                'contribution': diversity_score * self.weights['source_diversity']
            })
            
            avg_density = np.mean([r.chunk.semantic_density for r in search_results[:self.DEFAULT_MAX_SOURCES]])
            factors.append({
                'name': 'semantic_density',
                'score': avg_density,
                'weight': self.weights['semantic_density'],
                'contribution': avg_density * self.weights['semantic_density']
            })
        
        answer_length = len(generation_result.get('answer', '').split())
        length_score = min(1.0, answer_length / self.DEFAULT_ANSWER_LENGTH_THRESHOLD)
        factors.append({
            'name': 'answer_completeness',
            'score': length_score,
            'weight': self.weights['answer_completeness'],
            'contribution': length_score * self.weights['answer_completeness']
        })
        
        question_words = set(question.lower().split())
        answer_words = set(generation_result.get('answer', '').lower().split())
        word_overlap = len(question_words.intersection(answer_words))
        alignment_score = min(1.0, word_overlap / max(len(question_words), 1))
        factors.append({
            'name': 'query_alignment',
            'score': alignment_score,
            'weight': self.weights['query_alignment'],
            'contribution': alignment_score * self.weights['query_alignment']
        })
        
        sources_used = generation_result.get('sources_used', generation_result.get('context_used', 0))
        utilization_score = min(1.0, sources_used / self.DEFAULT_MAX_SOURCES)
        factors.append({
            'name': 'context_utilization',
            'score': utilization_score,
            'weight': self.weights['context_utilization'],
            'contribution': utilization_score * self.weights['context_utilization']
        })
        
        # Calculate final confidence
        total_weight = sum(f['weight'] for f in factors)
        raw_confidence = sum(f['contribution'] for f in factors) / total_weight if total_weight > 0 else 0.5
        adjusted_confidence = raw_confidence * self.CONSERVATIVE_ADJUSTMENT + self.CONSERVATIVE_BASE
        final_confidence = min(1.0, max(0.0, adjusted_confidence))
        
        return {
            'factors': factors,
            'raw_confidence': raw_confidence,
            'adjusted_confidence': adjusted_confidence,
            'final_confidence': final_confidence,
            'total_weight': total_weight
        }


__all__ = ['ConfidenceCalculator']