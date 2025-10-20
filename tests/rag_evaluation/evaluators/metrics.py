"""
RAG Evaluation Metrics Module

This module implements comprehensive scoring algorithms for evaluating 
Retrieval-Augmented Generation (RAG) system performance across multiple dimensions.

Metrics Categories:
- Accuracy: Factual correctness and completeness
- Relevance: Query-answer alignment and specificity  
- Quality: Coherence, clarity, and structure
- Technical: Performance and reliability metrics
"""

import re
import time
import json
import statistics
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import difflib
import numpy as np


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    query_id: str
    query_text: str
    expected_answer: str
    actual_answer: str
    accuracy_score: float
    relevance_score: float
    quality_score: float
    technical_score: float
    overall_score: float
    response_time: float
    confidence_score: float
    key_facts_found: List[str]
    key_facts_missing: List[str]
    error_analysis: Dict[str, Any]
    timestamp: datetime


class AccuracyEvaluator:
    """Evaluates factual correctness and completeness"""
    
    def __init__(self):
        self.weight = 0.40  # 40% of overall score
        
    def evaluate(self, expected: str, actual: str, key_facts: List[str]) -> Dict[str, Any]:
        """
        Evaluate accuracy based on factual correctness
        
        Args:
            expected: Expected answer text
            actual: Generated answer text
            key_facts: List of key facts that should be present
            
        Returns:
            Dict containing accuracy metrics
        """
        # Normalize text for comparison
        expected_norm = self._normalize_text(expected)
        actual_norm = self._normalize_text(actual)
        
        # Fact-based accuracy
        fact_accuracy = self._evaluate_fact_accuracy(actual_norm, key_facts)
        
        # Semantic similarity (basic word overlap)
        semantic_similarity = self._calculate_semantic_similarity(expected_norm, actual_norm)
        
        # Completeness check
        completeness = self._evaluate_completeness(expected_norm, actual_norm)
        
        # Number accuracy for financial/technical data
        number_accuracy = self._evaluate_number_accuracy(expected, actual)
        
        # Combined accuracy score
        accuracy_score = (
            fact_accuracy * 0.4 +
            semantic_similarity * 0.3 +
            completeness * 0.2 +
            number_accuracy * 0.1
        )
        
        return {
            'accuracy_score': min(accuracy_score, 1.0),
            'fact_accuracy': fact_accuracy,
            'semantic_similarity': semantic_similarity,
            'completeness': completeness,
            'number_accuracy': number_accuracy,
            'facts_found': self._extract_found_facts(actual_norm, key_facts),
            'facts_missing': self._extract_missing_facts(actual_norm, key_facts)
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase, remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove common punctuation for better matching
        text = re.sub(r'[.,;:!?()"]', '', text)
        return text
    
    def _evaluate_fact_accuracy(self, text: str, key_facts: List[str]) -> float:
        """Evaluate how many key facts are present"""
        if not key_facts:
            return 1.0
            
        facts_found = 0
        for fact in key_facts:
            fact_norm = self._normalize_text(fact)
            if fact_norm in text or self._fuzzy_match(fact_norm, text):
                facts_found += 1
                
        return facts_found / len(key_facts)
    
    def _fuzzy_match(self, fact: str, text: str, threshold: float = 0.8) -> bool:
        """Check for fuzzy matches of facts in text"""
        words = text.split()
        fact_words = fact.split()
        
        # Check for partial matches
        for i in range(len(words) - len(fact_words) + 1):
            segment = ' '.join(words[i:i + len(fact_words)])
            similarity = difflib.SequenceMatcher(None, fact, segment).ratio()
            if similarity >= threshold:
                return True
        return False
    
    def _calculate_semantic_similarity(self, expected: str, actual: str) -> float:
        """Calculate basic semantic similarity using word overlap"""
        expected_words = set(expected.split())
        actual_words = set(actual.split())
        
        if not expected_words:
            return 1.0 if not actual_words else 0.0
            
        intersection = expected_words & actual_words
        union = expected_words | actual_words
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Weighted by expected answer coverage
        coverage = len(intersection) / len(expected_words) if expected_words else 0.0
        
        return (jaccard * 0.4 + coverage * 0.6)
    
    def _evaluate_completeness(self, expected: str, actual: str) -> float:
        """Evaluate completeness of the answer"""
        expected_length = len(expected.split())
        actual_length = len(actual.split())
        
        if expected_length == 0:
            return 1.0
            
        # Penalize answers that are too short or too long
        length_ratio = actual_length / expected_length
        
        if 0.7 <= length_ratio <= 1.5:
            completeness = 1.0
        elif length_ratio < 0.7:
            completeness = length_ratio / 0.7
        else:
            completeness = max(0.0, 1.5 / length_ratio)
            
        return completeness
    
    def _evaluate_number_accuracy(self, expected: str, actual: str) -> float:
        """Evaluate accuracy of numbers in the response"""
        expected_numbers = self._extract_numbers(expected)
        actual_numbers = self._extract_numbers(actual)
        
        if not expected_numbers:
            return 1.0
            
        matches = 0
        for exp_num in expected_numbers:
            if any(abs(exp_num - act_num) / max(abs(exp_num), 1) < 0.01 
                   for act_num in actual_numbers):
                matches += 1
                
        return matches / len(expected_numbers)
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text"""
        # Pattern for numbers (including percentages, decimals, currencies)
        pattern = r'[\$]?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?:%|M|K|million|thousand)?'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            try:
                # Remove commas and convert
                num_str = match.replace(',', '')
                num = float(num_str)
                numbers.append(num)
            except ValueError:
                continue
                
        return numbers
    
    def _extract_found_facts(self, text: str, key_facts: List[str]) -> List[str]:
        """Extract facts that were found in the answer"""
        found = []
        for fact in key_facts:
            fact_norm = self._normalize_text(fact)
            if fact_norm in text or self._fuzzy_match(fact_norm, text):
                found.append(fact)
        return found
    
    def _extract_missing_facts(self, text: str, key_facts: List[str]) -> List[str]:
        """Extract facts that were missing from the answer"""
        missing = []
        for fact in key_facts:
            fact_norm = self._normalize_text(fact)
            if not (fact_norm in text or self._fuzzy_match(fact_norm, text)):
                missing.append(fact)
        return missing


class RelevanceEvaluator:
    """Evaluates query-answer alignment and relevance"""
    
    def __init__(self):
        self.weight = 0.25  # 25% of overall score
        
    def evaluate(self, query: str, answer: str, context: str = "") -> Dict[str, Any]:
        """
        Evaluate relevance of answer to query
        
        Args:
            query: User query text
            answer: Generated answer text
            context: Optional context for evaluation
            
        Returns:
            Dict containing relevance metrics
        """
        # Query-answer alignment
        alignment_score = self._evaluate_alignment(query, answer)
        
        # Answer specificity
        specificity_score = self._evaluate_specificity(query, answer)
        
        # Context appropriateness
        context_score = self._evaluate_context_appropriateness(query, answer, context)
        
        # Direct answer detection
        directness_score = self._evaluate_directness(query, answer)
        
        # Combined relevance score
        relevance_score = (
            alignment_score * 0.4 +
            specificity_score * 0.25 +
            context_score * 0.2 +
            directness_score * 0.15
        )
        
        return {
            'relevance_score': min(relevance_score, 1.0),
            'alignment_score': alignment_score,
            'specificity_score': specificity_score,
            'context_score': context_score,
            'directness_score': directness_score
        }
    
    def _evaluate_alignment(self, query: str, answer: str) -> float:
        """Evaluate how well answer aligns with query intent"""
        query_words = set(self._extract_keywords(query))
        answer_words = set(self._extract_keywords(answer))
        
        if not query_words:
            return 1.0
            
        # Calculate keyword overlap
        overlap = len(query_words & answer_words)
        alignment = overlap / len(query_words)
        
        # Bonus for question-type matching
        question_bonus = self._evaluate_question_type_match(query, answer)
        
        return min(alignment + question_bonus, 1.0)
    
    def _evaluate_specificity(self, query: str, answer: str) -> float:
        """Evaluate specificity of the answer"""
        # Check for vague language
        vague_patterns = [
            r'\b(maybe|perhaps|possibly|might|could be)\b',
            r'\b(generally|usually|typically|often)\b',
            r'\b(some|many|several|various)\b'
        ]
        
        vague_count = sum(len(re.findall(pattern, answer.lower())) 
                         for pattern in vague_patterns)
        
        # Check for specific details
        specific_patterns = [
            r'\b\d+\.?\d*%?\b',  # Numbers
            r'\b[A-Z][a-z]+ \d{1,2}, \d{4}\b',  # Dates
            r'\b\$\d+(?:,\d{3})*(?:\.\d{2})?\b',  # Currency
            r'\b[A-Z][A-Za-z]+ [A-Z][A-Za-z]+\b'  # Proper names
        ]
        
        specific_count = sum(len(re.findall(pattern, answer)) 
                           for pattern in specific_patterns)
        
        # Calculate specificity score
        answer_length = len(answer.split())
        if answer_length == 0:
            return 0.0
            
        vague_ratio = vague_count / answer_length
        specific_ratio = specific_count / answer_length
        
        specificity = max(0.0, 1.0 - vague_ratio * 2 + specific_ratio)
        return min(specificity, 1.0)
    
    def _evaluate_context_appropriateness(self, query: str, answer: str, context: str) -> float:
        """Evaluate if answer is appropriate for the context"""
        if not context:
            return 1.0
            
        # Check if answer references context appropriately
        context_words = set(self._extract_keywords(context))
        answer_words = set(self._extract_keywords(answer))
        
        context_usage = len(context_words & answer_words) / len(context_words) if context_words else 1.0
        return min(context_usage, 1.0)
    
    def _evaluate_directness(self, query: str, answer: str) -> float:
        """Evaluate if answer directly addresses the query"""
        # Check for direct answer patterns
        answer_lower = answer.lower()
        
        # Positive indicators
        direct_patterns = [
            r'^(yes|no),?\s',  # Direct yes/no
            r'^\w+\s+is\s+',   # Direct statements
            r'^\d+',           # Direct numbers
            r'^the\s+\w+\s+is\s+'  # Direct definitions
        ]
        
        has_direct_pattern = any(re.search(pattern, answer_lower) 
                               for pattern in direct_patterns)
        
        # Check for hedging language
        hedging_patterns = [
            r'\baccording to\b',
            r'\bbased on\b',
            r'\bit appears that\b',
            r'\bit seems\b'
        ]
        
        has_hedging = any(re.search(pattern, answer_lower) 
                         for pattern in hedging_patterns)
        
        directness = 0.8 if has_direct_pattern else 0.5
        if has_hedging:
            directness *= 0.8
            
        return directness
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _evaluate_question_type_match(self, query: str, answer: str) -> float:
        """Evaluate if answer matches question type"""
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # What questions should have specific items
        if query_lower.startswith('what'):
            if re.search(r'\b(is|are|was|were)\b', answer_lower):
                return 0.1
                
        # How questions should have explanations
        elif query_lower.startswith('how'):
            if re.search(r'\b(by|through|using|via)\b', answer_lower):
                return 0.1
                
        # Why questions should have reasons
        elif query_lower.startswith('why'):
            if re.search(r'\b(because|due to|since|as)\b', answer_lower):
                return 0.1
                
        # When questions should have time references
        elif query_lower.startswith('when'):
            if re.search(r'\b(\d{4}|\d{1,2}/\d{1,2}|january|february|march|april|may|june|july|august|september|october|november|december)\b', answer_lower):
                return 0.1
                
        return 0.0


class QualityEvaluator:
    """Evaluates response coherence, clarity, and structure"""
    
    def __init__(self):
        self.weight = 0.20  # 20% of overall score
        
    def evaluate(self, answer: str) -> Dict[str, Any]:
        """
        Evaluate quality aspects of the answer
        
        Args:
            answer: Generated answer text
            
        Returns:
            Dict containing quality metrics
        """
        # Coherence evaluation
        coherence_score = self._evaluate_coherence(answer)
        
        # Clarity evaluation
        clarity_score = self._evaluate_clarity(answer)
        
        # Structure evaluation
        structure_score = self._evaluate_structure(answer)
        
        # Language quality
        language_score = self._evaluate_language_quality(answer)
        
        # Combined quality score
        quality_score = (
            coherence_score * 0.3 +
            clarity_score * 0.3 +
            structure_score * 0.25 +
            language_score * 0.15
        )
        
        return {
            'quality_score': min(quality_score, 1.0),
            'coherence_score': coherence_score,
            'clarity_score': clarity_score,
            'structure_score': structure_score,
            'language_score': language_score
        }
    
    def _evaluate_coherence(self, answer: str) -> float:
        """Evaluate logical flow and coherence"""
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 1.0 if sentences else 0.0
            
        # Check for transition words and logical connectors
        transition_words = [
            'however', 'therefore', 'furthermore', 'additionally', 'moreover',
            'consequently', 'meanwhile', 'nevertheless', 'thus', 'hence',
            'first', 'second', 'finally', 'also', 'because', 'since'
        ]
        
        transitions_found = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in transition_words):
                transitions_found += 1
                
        # Check for topic consistency (repeated key terms)
        key_terms = self._extract_key_terms(answer)
        consistency_score = self._calculate_topic_consistency(sentences, key_terms)
        
        transition_score = min(transitions_found / max(len(sentences) - 1, 1), 1.0)
        
        return (transition_score * 0.4 + consistency_score * 0.6)
    
    def _evaluate_clarity(self, answer: str) -> float:
        """Evaluate clarity and readability"""
        if not answer.strip():
            return 0.0
            
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Average sentence length (clarity metric)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Optimal sentence length is 15-20 words
        length_score = 1.0
        if avg_sentence_length > 25:
            length_score = max(0.4, 25 / avg_sentence_length)
        elif avg_sentence_length < 8:
            length_score = max(0.6, avg_sentence_length / 8)
            
        # Check for complex punctuation (may indicate confusion)
        complex_punct_count = len(re.findall(r'[();,]{2,}', answer))
        punct_penalty = min(complex_punct_count * 0.1, 0.3)
        
        # Check for clear terminology
        jargon_score = self._evaluate_jargon_usage(answer)
        
        clarity_score = length_score - punct_penalty + jargon_score * 0.2
        return max(0.0, min(clarity_score, 1.0))
    
    def _evaluate_structure(self, answer: str) -> float:
        """Evaluate structural organization"""
        # Check for proper capitalization
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
            
        proper_caps = sum(1 for s in sentences if s and s[0].isupper())
        cap_score = proper_caps / len(sentences)
        
        # Check for lists or enumeration
        has_structure = any(re.search(r'\b(first|second|third|\d+\.|\d+\))', answer.lower())
                           for _ in [1])  # Just checking if pattern exists
        structure_bonus = 0.2 if has_structure else 0.0
        
        # Check for conclusion
        has_conclusion = bool(re.search(r'\b(therefore|thus|in conclusion|overall|finally)\b', 
                                      answer.lower()))
        conclusion_bonus = 0.1 if has_conclusion else 0.0
        
        return min(cap_score + structure_bonus + conclusion_bonus, 1.0)
    
    def _evaluate_language_quality(self, answer: str) -> float:
        """Evaluate language quality and grammar"""
        if not answer.strip():
            return 0.0
            
        # Check for repetitive words
        words = answer.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        repetition_penalty = 0.0
        for word, freq in word_freq.items():
            if len(word) > 4 and freq > len(words) * 0.1:  # Word appears more than 10% of the time
                repetition_penalty += 0.1
                
        # Check for proper punctuation
        punct_score = 1.0
        if not re.search(r'[.!?]$', answer.strip()):
            punct_score = 0.8
            
        language_score = punct_score - min(repetition_penalty, 0.4)
        return max(0.0, language_score)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        # Return words that appear more than once
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        return [word for word, count in word_counts.items() if count > 1]
    
    def _calculate_topic_consistency(self, sentences: List[str], key_terms: List[str]) -> float:
        """Calculate topic consistency across sentences"""
        if not sentences or not key_terms:
            return 1.0
            
        sentence_scores = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            terms_found = sum(1 for term in key_terms if term in sentence_lower)
            score = terms_found / len(key_terms) if key_terms else 0
            sentence_scores.append(score)
            
        # Consistency is measured by how evenly terms are distributed
        return min(np.mean(sentence_scores), 1.0) if sentence_scores else 1.0
    
    def _evaluate_jargon_usage(self, answer: str) -> float:
        """Evaluate appropriate use of technical jargon"""
        # This is a simple heuristic - could be enhanced with domain-specific dictionaries
        technical_indicators = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+[Aa]pi\b',  # API references
            r'\b\d+\.\d+\.\d+\b',  # Version numbers
            r'\b\w+[Ss]erver\b'  # Server references
        ]
        
        jargon_count = sum(len(re.findall(pattern, answer)) 
                          for pattern in technical_indicators)
        
        word_count = len(answer.split())
        if word_count == 0:
            return 1.0
            
        jargon_ratio = jargon_count / word_count
        
        # Appropriate jargon usage (not too much, not too little for technical content)
        if 0.05 <= jargon_ratio <= 0.15:
            return 1.0
        elif jargon_ratio < 0.05:
            return 0.8  # Might be too simplified
        else:
            return max(0.5, 1.0 - (jargon_ratio - 0.15) * 2)  # Too much jargon


class TechnicalEvaluator:
    """Evaluates technical performance metrics"""
    
    def __init__(self):
        self.weight = 0.15  # 15% of overall score
        
    def evaluate(self, response_time: float, confidence_score: float = None, 
                 context_used: int = 0, max_context: int = 3) -> Dict[str, Any]:
        """
        Evaluate technical performance metrics
        
        Args:
            response_time: Time taken to generate response (seconds)
            confidence_score: Model confidence score (0-1)
            context_used: Number of context chunks used
            max_context: Maximum context chunks available
            
        Returns:
            Dict containing technical metrics
        """
        # Response time evaluation
        time_score = self._evaluate_response_time(response_time)
        
        # Confidence evaluation
        confidence_eval = self._evaluate_confidence(confidence_score)
        
        # Context usage evaluation
        context_score = self._evaluate_context_usage(context_used, max_context)
        
        # Combined technical score
        technical_score = (
            time_score * 0.4 +
            confidence_eval * 0.35 +
            context_score * 0.25
        )
        
        return {
            'technical_score': min(technical_score, 1.0),
            'response_time_score': time_score,
            'confidence_score_eval': confidence_eval,
            'context_usage_score': context_score,
            'actual_response_time': response_time,
            'actual_confidence': confidence_score,
            'context_chunks_used': context_used
        }
    
    def _evaluate_response_time(self, response_time: float) -> float:
        """Evaluate response time performance"""
        # Target: <2 seconds, Optimal: <1 second
        if response_time <= 1.0:
            return 1.0
        elif response_time <= 2.0:
            return 1.0 - (response_time - 1.0) * 0.3  # Linear decrease
        elif response_time <= 5.0:
            return 0.7 - (response_time - 2.0) * 0.15  # Steeper decrease
        else:
            return max(0.1, 0.25 - (response_time - 5.0) * 0.05)
    
    def _evaluate_confidence(self, confidence_score: float) -> float:
        """Evaluate confidence score appropriateness"""
        if confidence_score is None:
            return 0.5  # Neutral score if confidence not available
            
        # High confidence is generally good, but overconfidence can be problematic
        if 0.7 <= confidence_score <= 0.9:
            return 1.0  # Optimal range
        elif 0.5 <= confidence_score < 0.7:
            return 0.8  # Reasonable confidence
        elif 0.9 < confidence_score <= 1.0:
            return 0.9  # Slight penalty for overconfidence
        else:
            return confidence_score  # Low confidence scores as-is
    
    def _evaluate_context_usage(self, context_used: int, max_context: int) -> float:
        """Evaluate efficient use of available context"""
        if max_context == 0:
            return 1.0
            
        usage_ratio = context_used / max_context
        
        # Optimal usage is using most but not necessarily all context
        if 0.6 <= usage_ratio <= 1.0:
            return 1.0
        elif 0.3 <= usage_ratio < 0.6:
            return 0.8
        else:
            return 0.5  # Poor context utilization


class ComprehensiveEvaluator:
    """Main evaluator that combines all metrics"""
    
    def __init__(self):
        self.accuracy_evaluator = AccuracyEvaluator()
        self.relevance_evaluator = RelevanceEvaluator()
        self.quality_evaluator = QualityEvaluator()
        self.technical_evaluator = TechnicalEvaluator()
        
    def evaluate_response(self, query_data: Dict[str, Any], 
                         actual_answer: str, response_time: float,
                         confidence_score: float = None) -> EvaluationResult:
        """
        Perform comprehensive evaluation of a RAG response
        
        Args:
            query_data: Dictionary containing query info (question, expected_answer, key_facts, etc.)
            actual_answer: Generated answer text
            response_time: Time taken to generate response
            confidence_score: Model confidence score
            
        Returns:
            EvaluationResult object with all metrics
        """
        start_time = time.time()
        
        # Extract query information
        query_id = query_data.get('query_id', 'unknown')
        query_text = query_data.get('question', '')
        expected_answer = query_data.get('expected_answer', '')
        key_facts = query_data.get('key_facts', [])
        context = query_data.get('context', '')
        
        # Evaluate each dimension
        accuracy_results = self.accuracy_evaluator.evaluate(
            expected_answer, actual_answer, key_facts
        )
        
        relevance_results = self.relevance_evaluator.evaluate(
            query_text, actual_answer, context
        )
        
        quality_results = self.quality_evaluator.evaluate(actual_answer)
        
        technical_results = self.technical_evaluator.evaluate(
            response_time, confidence_score, context_used=3, max_context=3
        )
        
        # Calculate overall score
        overall_score = (
            accuracy_results['accuracy_score'] * self.accuracy_evaluator.weight +
            relevance_results['relevance_score'] * self.relevance_evaluator.weight +
            quality_results['quality_score'] * self.quality_evaluator.weight +
            technical_results['technical_score'] * self.technical_evaluator.weight
        )
        
        # Compile error analysis
        error_analysis = {
            'accuracy_issues': {
                'missing_facts': accuracy_results.get('facts_missing', []),
                'fact_accuracy': accuracy_results.get('fact_accuracy', 0.0)
            },
            'relevance_issues': {
                'alignment_score': relevance_results.get('alignment_score', 0.0),
                'specificity_score': relevance_results.get('specificity_score', 0.0)
            },
            'quality_issues': {
                'coherence_score': quality_results.get('coherence_score', 0.0),
                'clarity_score': quality_results.get('clarity_score', 0.0)
            },
            'technical_issues': {
                'response_time': response_time,
                'time_score': technical_results.get('response_time_score', 0.0)
            }
        }
        
        return EvaluationResult(
            query_id=query_id,
            query_text=query_text,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            accuracy_score=accuracy_results['accuracy_score'],
            relevance_score=relevance_results['relevance_score'],
            quality_score=quality_results['quality_score'],
            technical_score=technical_results['technical_score'],
            overall_score=overall_score,
            response_time=response_time,
            confidence_score=confidence_score or 0.0,
            key_facts_found=accuracy_results.get('facts_found', []),
            key_facts_missing=accuracy_results.get('facts_missing', []),
            error_analysis=error_analysis,
            timestamp=datetime.now()
        )
    
    def batch_evaluate(self, test_cases: List[Dict[str, Any]], 
                      rag_responses: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """
        Evaluate multiple test cases
        
        Args:
            test_cases: List of test case dictionaries
            rag_responses: List of RAG response dictionaries
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        for test_case, response in zip(test_cases, rag_responses):
            result = self.evaluate_response(
                query_data=test_case,
                actual_answer=response.get('answer', ''),
                response_time=response.get('response_time', 0.0),
                confidence_score=response.get('confidence', None)
            )
            results.append(result)
            
        return results