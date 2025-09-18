# Enhanced RAG Evaluation Service with Citation and Verification Metrics
import os
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import Counter

from langchain_mistralai.chat_models import ChatMistralAI

logger = logging.getLogger(__name__)

class EnhancedRAGEvaluationService:
    """Advanced RAG Evaluation with citation quality, verification, and retrieval metrics"""
    
    def __init__(self, mistral_api_key: str, mistral_model: str = "mistral-large-latest"):
        self.mistral_api_key = mistral_api_key
        self.mistral_model = mistral_model
        
        self.llm = ChatMistralAI(
            model=mistral_model,
            temperature=0.1,
            api_key=mistral_api_key
        )
        
        # Evaluation metrics configuration
        self.metrics = {
            'retrieval': ['precision_at_k', 'recall_at_k', 'mrr', 'ndcg'],
            'generation': ['answer_relevancy', 'faithfulness', 'completeness'],
            'verification': ['claim_support_rate', 'entailment_consistency', 'factual_accuracy']
        }
    
    def evaluate_enhanced_rag_response(
        self,
        query: str,
        response: Any,  # Can be Dict[str, Any] or str
        ground_truth: Optional[str] = None,
        relevant_docs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of enhanced RAG response
        
        Args:
            query: The input question
            response: Enhanced RAG response dict or response string
            ground_truth: Optional ground truth answer for comparison
            relevant_docs: Optional list of actually relevant document IDs
        """
        try:
            # Handle both string and dict responses
            if isinstance(response, str):
                # Convert string response to dict format for evaluation
                response_dict = {
                    'answer': response,
                    'sources': [],
                    'confidence_score': 0.5,
                    'verification': {},
                    'processing_time': 0
                }
            elif isinstance(response, dict):
                response_dict = response
            else:
                raise ValueError(f"Response must be string or dict, got {type(response)}")
            
            evaluation_results = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'retrieval_metrics': {},
                'generation_metrics': {},
                'verification_metrics': {},
                'overall_metrics': {}
            }
            
            # 1. Retrieval Quality Metrics
            try:
                evaluation_results['retrieval_metrics'] = self._evaluate_retrieval_quality(
                    query, response_dict, relevant_docs
                )
            except Exception as e:
                logger.warning(f"Retrieval evaluation failed: {str(e)}")
                evaluation_results['retrieval_metrics'] = {'error': str(e)}
            
            # 2. Generation Quality Metrics  
            try:
                evaluation_results['generation_metrics'] = self._evaluate_generation_quality(
                    query, response_dict, ground_truth
                )
            except Exception as e:
                logger.warning(f"Generation evaluation failed: {str(e)}")
                evaluation_results['generation_metrics'] = {'error': str(e)}
            
            # 3. Verification Metrics
            try:
                evaluation_results['verification_metrics'] = self._evaluate_verification_quality(
                    response_dict
                )
            except Exception as e:
                logger.warning(f"Verification evaluation failed: {str(e)}")
                evaluation_results['verification_metrics'] = {'error': str(e)}
            
            # 4. Overall Quality Score
            evaluation_results['overall_metrics'] = self._calculate_overall_metrics(
                evaluation_results
            )
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Enhanced RAG evaluation failed: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _evaluate_retrieval_quality(
        self, 
        query: str, 
        response: Dict[str, Any], 
        relevant_docs: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive retrieval quality evaluation with hit rate, precision, recall, and ranking metrics
        """
        metrics = {}
        
        try:
            sources = response.get('sources', [])
            reranking_info = response.get('reranking', {})
            
            # Basic retrieval metrics
            metrics['documents_retrieved'] = len(sources)
            metrics['candidates_retrieved'] = reranking_info.get('candidates_retrieved', 0)
            metrics['avg_relevance_score'] = np.mean([s.get('relevance_score', 0) for s in sources]) if sources else 0
            metrics['avg_rerank_score'] = reranking_info.get('avg_rerank_score', 0)
            
            # Extract document IDs for evaluation
            retrieved_doc_ids = []
            for source in sources:
                if isinstance(source, dict):
                    # Try different source ID formats
                    doc_id = (source.get('page') or 
                             source.get('document_id') or 
                             source.get('citation_id') or 
                             source.get('source_id', ''))
                    if doc_id:
                        retrieved_doc_ids.append(str(doc_id))
                elif isinstance(source, str):
                    retrieved_doc_ids.append(source)
            
            # Reranking effectiveness
            if sources and len(sources) > 1:
                rerank_scores = [s.get('rerank_score', 0) for s in sources]
                original_scores = [s.get('relevance_score', 0) for s in sources]
                
                # Measure if reranking improved order
                metrics['reranking_improvement'] = self._calculate_ranking_improvement(
                    original_scores, rerank_scores
                )
            else:
                metrics['reranking_improvement'] = 0.0
            
            # Diversity metrics
            metrics['content_type_diversity'] = self._calculate_content_diversity(sources)
            metrics['page_coverage'] = len(set([s.get('page', 'unknown') for s in sources]))
            
            # Comprehensive evaluation if ground truth relevant docs provided
            if relevant_docs and retrieved_doc_ids:
                relevant_set = set(str(doc) for doc in relevant_docs)
                retrieved_set = set(retrieved_doc_ids)
                
                # 1. Hit Rate (did we retrieve at least one relevant document?)
                metrics['hit_rate'] = 1.0 if len(relevant_set.intersection(retrieved_set)) > 0 else 0.0
                
                # 2. Precision@K (what fraction of retrieved docs are relevant?)
                true_positives = len(relevant_set.intersection(retrieved_set))
                metrics['precision_at_k'] = true_positives / len(retrieved_set) if retrieved_set else 0.0
                
                # 3. Recall@K (what fraction of relevant docs were retrieved?)
                metrics['recall_at_k'] = true_positives / len(relevant_set) if relevant_set else 0.0
                
                # 4. F1@K (harmonic mean of precision and recall)
                metrics['f1_at_k'] = (2 * metrics['precision_at_k'] * metrics['recall_at_k'] / 
                                    (metrics['precision_at_k'] + metrics['recall_at_k']) 
                                    if (metrics['precision_at_k'] + metrics['recall_at_k']) > 0 else 0.0)
                
                # 5. Mean Reciprocal Rank (MRR)
                metrics['mrr'] = self._calculate_mrr(retrieved_doc_ids, relevant_set)
                
                # 6. Normalized Discounted Cumulative Gain (NDCG@K)
                metrics['ndcg_at_k'] = self._calculate_ndcg(retrieved_doc_ids, relevant_set)
                
                # 7. Coverage and efficiency metrics
                metrics['total_retrieved'] = len(retrieved_doc_ids)
                metrics['total_relevant'] = len(relevant_set)
                metrics['coverage'] = min(len(retrieved_set) / max(len(relevant_set), 1), 1.0)
                metrics['retrieval_efficiency'] = true_positives / max(metrics['documents_retrieved'], 1)
                
            elif relevant_docs:
                # If relevant docs provided but no retrieved docs found
                metrics.update({
                    'hit_rate': 0.0,
                    'precision_at_k': 0.0,
                    'recall_at_k': 0.0,
                    'f1_at_k': 0.0,
                    'mrr': 0.0,
                    'ndcg_at_k': 0.0,
                    'total_retrieved': len(retrieved_doc_ids),
                    'total_relevant': len(relevant_docs),
                    'coverage': 0.0,
                    'retrieval_efficiency': 0.0
                })
            else:
                # Basic evaluation without ground truth
                metrics.update({
                    'total_retrieved': len(retrieved_doc_ids),
                    'retrieval_depth': len(sources),
                    'source_quality_score': metrics['avg_relevance_score']
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Retrieval evaluation failed: {str(e)}")
            return {'error': str(e)}
    
    def _evaluate_generation_quality(
        self, 
        query: str, 
        response: Dict[str, Any], 
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Comprehensive answer generation quality evaluation with precision, recall, and semantic metrics
        """
        metrics = {}
        
        try:
            answer = response.get('answer', '')
            sources = response.get('sources', [])
            
            # Basic generation metrics
            metrics['answer_length'] = len(answer.split())
            metrics['answer_length_score'] = self._evaluate_answer_length_appropriateness(answer, query)
            metrics['answer_completeness'] = self._evaluate_answer_completeness(query, answer)
            metrics['answer_relevancy'] = self._evaluate_answer_relevancy(query, answer)
            
            # Faithfulness to sources
            context = " ".join([s.get('content_preview', '') for s in sources])
            metrics['faithfulness'] = self._evaluate_faithfulness(answer, context)
            metrics['groundedness'] = self._evaluate_groundedness(answer, sources)
            
            # Language quality metrics
            metrics['fluency'] = self._evaluate_fluency(answer)
            metrics['coherence'] = self._evaluate_coherence(answer)
            metrics['clarity'] = self._evaluate_clarity(answer)
            
            # Consistency with confidence
            confidence = response.get('confidence_score', 0.5)
            metrics['confidence_calibration'] = self._evaluate_confidence_calibration(
                answer, confidence
            )
            
            # Ground truth comparison if available (comprehensive evaluation)
            if ground_truth:
                # Semantic similarity
                metrics['semantic_similarity'] = self._calculate_semantic_similarity(
                    answer, ground_truth
                )
                
                # Content overlap and precision/recall
                content_metrics = self._calculate_content_overlap_metrics(answer, ground_truth)
                metrics.update(content_metrics)
                
                # BLEU and ROUGE scores
                metrics['bleu_score'] = self._calculate_bleu_score(answer, ground_truth)
                rouge_scores = self._calculate_rouge_scores(answer, ground_truth)
                metrics.update(rouge_scores)
                
                # Factual accuracy
                metrics['factual_accuracy'] = self._evaluate_factual_accuracy(answer, ground_truth)
                
                # Answer type correctness (if ground truth indicates expected answer type)
                metrics['answer_type_match'] = self._evaluate_answer_type_match(answer, ground_truth)
                
            else:
                # Basic quality assessment without ground truth
                metrics['information_density'] = self._calculate_information_density(answer)
                metrics['specificity'] = self._evaluate_specificity(answer)
                metrics['helpfulness'] = self._evaluate_helpfulness(query, answer)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Generation evaluation failed: {str(e)}")
            return {'error': str(e)}
    
    def _evaluate_verification_quality(self, response: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate claim verification quality"""
        metrics = {}
        
        try:
            verification = response.get('verification', {})
            
            if not verification:
                return {'verification_available': 0.0}
            
            total_claims = verification.get('total_claims', 0)
            supported_claims = verification.get('supported_claims', 0)
            verification_score = verification.get('verification_score', 0)
            claim_details = verification.get('claim_details', [])
            
            # Basic verification metrics
            metrics['verification_available'] = 1.0
            metrics['claim_support_rate'] = verification_score
            metrics['total_claims_verified'] = total_claims
            metrics['avg_entailment_score'] = self._calculate_avg_entailment_score(claim_details)
            
            # Verification confidence distribution
            if claim_details:
                entailment_scores = [
                    claim.get('best_support', {}).get('entailment_score', 0) 
                    for claim in claim_details
                ]
                metrics['verification_confidence_std'] = np.std(entailment_scores)
                metrics['high_confidence_claims'] = sum(1 for score in entailment_scores if score > 0.8) / len(entailment_scores)
                metrics['low_confidence_claims'] = sum(1 for score in entailment_scores if score < 0.3) / len(entailment_scores)
            
            # Consistency between verification and confidence
            overall_confidence = response.get('confidence_score', 0.5)
            metrics['verification_confidence_consistency'] = 1.0 - abs(verification_score - overall_confidence)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Verification evaluation failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_overall_metrics(self, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall quality scores"""
        try:
            retrieval = evaluation_results.get('retrieval_metrics', {})
            generation = evaluation_results.get('generation_metrics', {})
            verification = evaluation_results.get('verification_metrics', {})
            
            # Component scores (0-1 scale) with comprehensive metrics
            retrieval_metrics = [
                retrieval.get('hit_rate', 0),
                retrieval.get('precision_at_k', 0),
                retrieval.get('recall_at_k', 0),
                retrieval.get('f1_at_k', 0),
                retrieval.get('mrr', 0),
                retrieval.get('ndcg_at_k', 0),
                retrieval.get('avg_rerank_score', 0),
                retrieval.get('content_type_diversity', 0),
                retrieval.get('reranking_improvement', 0)
            ]
            retrieval_score = np.mean([m for m in retrieval_metrics if m > 0]) if any(m > 0 for m in retrieval_metrics) else 0.0
            
            generation_metrics = [
                generation.get('answer_relevancy', 0),
                generation.get('faithfulness', 0),
                generation.get('answer_completeness', 0),
                generation.get('groundedness', 0),
                generation.get('fluency', 0),
                generation.get('coherence', 0),
                generation.get('clarity', 0)
            ]
            generation_score = np.mean([m for m in generation_metrics if m > 0]) if any(m > 0 for m in generation_metrics) else 0.0
            
            # Include semantic and content metrics if available
            if 'semantic_similarity' in generation:
                generation_metrics.extend([
                    generation.get('semantic_similarity', 0),
                    generation.get('content_f1', 0),
                    generation.get('bleu_score', 0),
                    generation.get('rouge_1', 0),
                    generation.get('factual_accuracy', 0)
                ])
                generation_score = np.mean([m for m in generation_metrics if m > 0])
            
            verification_score = verification.get('claim_support_rate', 0)
            
            # Weighted overall score (redistributed weights without citation)
            overall_rag_score = (
                retrieval_score * 0.35 +      # Increased from 0.25
                generation_score * 0.50 +     # Increased from 0.35  
                verification_score * 0.15     # Same
            )
            
            return {
                'retrieval_score': retrieval_score,
                'generation_score': generation_score,
                'verification_score': verification_score,
                'overall_rag_score': overall_rag_score,
                'quality_grade': self._assign_quality_grade(overall_rag_score)
            }
            
        except Exception as e:
            logger.error(f"Overall metrics calculation failed: {str(e)}")
            return {'error': str(e)}
    
    # Helper methods for specific evaluations
    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual factual claims from text"""
        sentences = re.split(r'[.!?]+', text)
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and not sentence.lower().startswith(('however', 'therefore', 'moreover')):
                claims.append(sentence)
        return claims
    
    def _evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """Evaluate how relevant the answer is to the query"""
        prompt = f"""
        Rate the relevancy of this answer to the question on a scale of 0.0 to 1.0.
        
        Question: {query}
        Answer: {answer[:500]}...
        
        Consider:
        - Does the answer directly address the question?
        - Is the information relevant and on-topic?
        - How well does it satisfy the information need?
        
        Respond with only a number between 0.0 and 1.0.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return float(response.content.strip())
        except:
            return 0.5
    
    def _evaluate_answer_completeness(self, query: str, answer: str) -> float:
        """Evaluate how complete the answer is"""
        prompt = f"""
        Rate the completeness of this answer on a scale of 0.0 to 1.0.
        
        Question: {query}
        Answer: {answer[:500]}...
        
        Consider:
        - Are all aspects of the question addressed?
        - Is sufficient detail provided?
        - Are there obvious gaps?
        
        Respond with only a number between 0.0 and 1.0.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return float(response.content.strip())
        except:
            return 0.5
    
    def _evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Evaluate faithfulness to source material"""
        if not context.strip():
            return 0.0
            
        prompt = f"""
        Rate how faithful this answer is to the provided context on a scale of 0.0 to 1.0.
        
        Context: {context[:1000]}...
        Answer: {answer[:500]}...
        
        Consider:
        - Are claims supported by the context?
        - Does the answer contradict the context?
        - Is information fabricated or hallucinated?
        
        Respond with only a number between 0.0 and 1.0.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return float(response.content.strip())
        except:
            return 0.5
    
    def _evaluate_confidence_calibration(self, answer: str, confidence: float) -> float:
        """Evaluate if confidence matches answer quality indicators"""
        # Simple heuristic based on uncertainty words
        uncertainty_words = ['might', 'could', 'possibly', 'perhaps', 'unclear', 'uncertain']
        uncertainty_count = sum(1 for word in uncertainty_words if word in answer.lower())
        
        # High confidence should have low uncertainty indicators
        expected_confidence = max(0.1, 1.0 - (uncertainty_count * 0.2))
        calibration = 1.0 - abs(confidence - expected_confidence)
        
        return max(0.0, calibration)
    
    def _calculate_semantic_similarity(self, answer: str, ground_truth: str) -> float:
        """Calculate semantic similarity between answer and ground truth"""
        # Simple word overlap (could be enhanced with embeddings)
        answer_words = set(answer.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return 0.0
        
        overlap = len(answer_words.intersection(truth_words))
        return overlap / len(truth_words.union(answer_words))
    
    def _calculate_content_overlap(self, answer: str, ground_truth: str) -> float:
        """Calculate content overlap ratio"""
        answer_words = answer.lower().split()
        truth_words = ground_truth.lower().split()
        
        if not truth_words:
            return 0.0
        
        overlap = len(set(answer_words).intersection(set(truth_words)))
        return overlap / max(len(set(answer_words)), len(set(truth_words)))
    
    def _calculate_ranking_improvement(self, original_scores: List[float], rerank_scores: List[float]) -> float:
        """Calculate if reranking improved the order"""
        if len(original_scores) != len(rerank_scores):
            return 0.0
        
        # Check if top reranked item has higher score than top original
        orig_order = sorted(range(len(original_scores)), key=lambda i: original_scores[i], reverse=True)
        rerank_order = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)
        
        # Calculate rank correlation improvement
        improvements = sum(1 for i in range(min(3, len(rerank_scores))) 
                          if rerank_order[i] != orig_order[i])
        
        return improvements / min(3, len(rerank_scores))
    
    def _calculate_content_diversity(self, sources: List[Dict]) -> float:
        """Calculate diversity of content types in sources"""
        if not sources:
            return 0.0
        
        content_types = [s.get('content_type', 'unknown') for s in sources]
        unique_types = len(set(content_types))
        return unique_types / len(content_types)
    
    def _calculate_precision_at_k(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate precision@k"""
        if not retrieved:
            return 0.0
        
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        return relevant_retrieved / len(retrieved)
    
    def _calculate_recall_at_k(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate recall@k"""
        if not relevant:
            return 0.0
        
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        return relevant_retrieved / len(relevant)
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def _calculate_avg_entailment_score(self, claim_details: List[Dict]) -> float:
        """Calculate average entailment score across claims"""
        if not claim_details:
            return 0.0
        
        scores = [
            claim.get('best_support', {}).get('entailment_score', 0) 
            for claim in claim_details
        ]
        
        return np.mean(scores) if scores else 0.0
    
    def _assign_quality_grade(self, score: float) -> str:
        """Assign letter grade based on score"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def batch_evaluate_responses(
        self,
        evaluation_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate multiple responses and provide aggregate metrics"""
        try:
            individual_results = []
            
            for item in evaluation_data:
                result = self.evaluate_enhanced_rag_response(
                    query=item['query'],
                    response=item['response'],
                    ground_truth=item.get('ground_truth'),
                    relevant_docs=item.get('relevant_docs')
                )
                individual_results.append(result)
            
            # Aggregate metrics
            aggregate_metrics = self._calculate_aggregate_metrics(individual_results)
            
            return {
                'individual_results': individual_results,
                'aggregate_metrics': aggregate_metrics,
                'total_evaluations': len(individual_results),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics across multiple evaluations"""
        try:
            # Collect all overall scores
            overall_scores = []
            component_scores = {'retrieval': [], 'generation': [], 'citation': [], 'verification': []}
            
            for result in results:
                if 'overall_metrics' in result:
                    overall = result['overall_metrics']
                    if 'overall_rag_score' in overall:
                        overall_scores.append(overall['overall_rag_score'])
                    
                    for component in component_scores:
                        score_key = f'{component}_score'
                        if score_key in overall:
                            component_scores[component].append(overall[score_key])
            
            # Calculate statistics
            aggregate = {}
            
            if overall_scores:
                aggregate['mean_overall_score'] = np.mean(overall_scores)
                aggregate['std_overall_score'] = np.std(overall_scores)
                aggregate['min_overall_score'] = np.min(overall_scores)
                aggregate['max_overall_score'] = np.max(overall_scores)
            
            # Component statistics
            for component, scores in component_scores.items():
                if scores:
                    aggregate[f'mean_{component}_score'] = np.mean(scores)
                    aggregate[f'std_{component}_score'] = np.std(scores)
            
            # Quality distribution
            if overall_scores:
                grade_counts = Counter()
                for score in overall_scores:
                    grade_counts[self._assign_quality_grade(score)] += 1
                
                aggregate['quality_distribution'] = dict(grade_counts)
            
            return aggregate
            
        except Exception as e:
            logger.error(f"Aggregate metrics calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_mrr(self, retrieved_doc_ids: List[str], relevant_set: set) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ndcg(self, retrieved_doc_ids: List[str], relevant_set: set, k: int = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k"""
        if k is None:
            k = len(retrieved_doc_ids)
        
        # Create relevance scores (1 for relevant, 0 for irrelevant)
        relevance_scores = [1 if doc_id in relevant_set else 0 for doc_id in retrieved_doc_ids[:k]]
        
        # Calculate DCG
        dcg = 0.0
        for i, relevance in enumerate(relevance_scores):
            if relevance > 0:
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1)=0
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevance = sorted([1] * len(relevant_set), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevance):
            if relevance > 0:
                idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _evaluate_answer_length_appropriateness(self, answer: str, query: str) -> float:
        """Evaluate if answer length is appropriate for the question complexity"""
        answer_words = len(answer.split())
        query_words = len(query.split())
        
        # Simple heuristic: more complex questions should have longer answers
        expected_length = min(200, max(50, query_words * 8))
        length_ratio = answer_words / expected_length
        
        # Optimal range is 0.5 to 2.0 times expected length
        if 0.5 <= length_ratio <= 2.0:
            return 1.0
        elif length_ratio < 0.5:
            return length_ratio * 2  # Too short penalty
        else:
            return 2.0 / length_ratio  # Too long penalty
    
    def _evaluate_groundedness(self, answer: str, sources: List[Dict]) -> float:
        """Evaluate how well the answer is grounded in provided sources"""
        if not sources:
            return 0.0
        
        # Count factual claims in answer
        claims = self._extract_claims(answer)
        if not claims:
            return 0.5
        
        # Since citations are removed, evaluate grounding based on source content overlap
        # and presence of specific details from sources
        sources_content = " ".join([s.get('content_preview', '') for s in sources])
        
        if not sources_content:
            return 0.3  # Low grounding if no source context
        
        # Calculate overlap between answer claims and source content
        claim_grounding_scores = []
        for claim in claims:
            # Simple overlap-based grounding check
            claim_words = set(claim.lower().split())
            source_words = set(sources_content.lower().split())
            overlap_ratio = len(claim_words.intersection(source_words)) / max(len(claim_words), 1)
            claim_grounding_scores.append(overlap_ratio)
        
        grounding_score = np.mean(claim_grounding_scores) if claim_grounding_scores else 0.3
        return min(1.0, grounding_score)
    
    def _evaluate_fluency(self, answer: str) -> float:
        """Evaluate language fluency and readability"""
        # Simple heuristics for fluency
        sentences = answer.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Optimal sentence length is 15-25 words
        if 15 <= avg_sentence_length <= 25:
            fluency_score = 1.0
        else:
            fluency_score = max(0.3, 1.0 - abs(avg_sentence_length - 20) / 20)
        
        # Check for basic grammar indicators
        if answer.count('?') + answer.count('!') + answer.count('.') == 0:
            fluency_score *= 0.5  # No punctuation penalty
        
        return fluency_score
    
    def _evaluate_coherence(self, answer: str) -> float:
        """Evaluate logical flow and coherence"""
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.8  # Short answers are inherently coherent
        
        # Look for transition words and logical connectors
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'additionally', 
                          'consequently', 'meanwhile', 'similarly', 'in contrast', 'for example']
        
        transition_count = sum(1 for word in transition_words if word in answer.lower())
        coherence_score = min(1.0, 0.5 + (transition_count / len(sentences)))
        
        return coherence_score
    
    def _evaluate_clarity(self, answer: str) -> float:
        """Evaluate clarity and understandability"""
        # Check for clear structure and organization
        clarity_indicators = {
            'numbered_lists': len(re.findall(r'\d+\.', answer)),
            'bullet_points': answer.count('•') + answer.count('-'),
            'clear_sections': answer.count('\n\n'),
            'question_marks': answer.count('?')
        }
        
        clarity_score = 0.6  # Base score
        
        # Bonus for structure
        if clarity_indicators['numbered_lists'] > 0 or clarity_indicators['bullet_points'] > 0:
            clarity_score += 0.2
        
        # Bonus for organized content
        if clarity_indicators['clear_sections'] > 0:
            clarity_score += 0.1
        
        # Penalty for excessive questions (indicates uncertainty)
        if clarity_indicators['question_marks'] > 2:
            clarity_score -= 0.1
        
        return min(1.0, clarity_score)
    
    def _calculate_content_overlap_metrics(self, answer: str, ground_truth: str) -> Dict[str, float]:
        """Calculate content-based precision, recall, and F1"""
        answer_words = set(answer.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return {'content_precision': 0.0, 'content_recall': 0.0, 'content_f1': 0.0}
        
        overlap = answer_words.intersection(truth_words)
        
        precision = len(overlap) / len(answer_words) if answer_words else 0.0
        recall = len(overlap) / len(truth_words) if truth_words else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return {
            'content_precision': precision,
            'content_recall': recall,
            'content_f1': f1,
            'content_overlap_ratio': len(overlap) / len(answer_words.union(truth_words))
        }
    
    def _calculate_bleu_score(self, answer: str, ground_truth: str) -> float:
        """Calculate BLEU score (simplified n-gram overlap)"""
        try:
            from collections import Counter
            
            answer_tokens = answer.lower().split()
            truth_tokens = ground_truth.lower().split()
            
            if not answer_tokens or not truth_tokens:
                return 0.0
            
            # Calculate 1-gram to 4-gram precision
            bleu_scores = []
            for n in range(1, 5):
                answer_ngrams = [' '.join(answer_tokens[i:i+n]) for i in range(len(answer_tokens)-n+1)]
                truth_ngrams = [' '.join(truth_tokens[i:i+n]) for i in range(len(truth_tokens)-n+1)]
                
                if not answer_ngrams:
                    bleu_scores.append(0.0)
                    continue
                
                answer_counts = Counter(answer_ngrams)
                truth_counts = Counter(truth_ngrams)
                
                overlap = sum(min(answer_counts[ngram], truth_counts[ngram]) for ngram in answer_counts)
                precision = overlap / len(answer_ngrams)
                bleu_scores.append(precision)
            
            # Geometric mean of n-gram precisions
            bleu = np.exp(np.mean([np.log(score + 1e-8) for score in bleu_scores]))
            
            # Brevity penalty
            bp = min(1.0, np.exp(1 - len(truth_tokens) / max(len(answer_tokens), 1)))
            
            return bleu * bp
            
        except Exception:
            return 0.0
    
    def _calculate_rouge_scores(self, answer: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores"""
        try:
            answer_tokens = answer.lower().split()
            truth_tokens = ground_truth.lower().split()
            
            if not answer_tokens or not truth_tokens:
                return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
            
            # ROUGE-1 (unigram overlap)
            answer_unigrams = set(answer_tokens)
            truth_unigrams = set(truth_tokens)
            rouge_1 = len(answer_unigrams.intersection(truth_unigrams)) / len(truth_unigrams)
            
            # ROUGE-2 (bigram overlap)
            answer_bigrams = set(f"{answer_tokens[i]} {answer_tokens[i+1]}" 
                               for i in range(len(answer_tokens)-1))
            truth_bigrams = set(f"{truth_tokens[i]} {truth_tokens[i+1]}" 
                              for i in range(len(truth_tokens)-1))
            rouge_2 = (len(answer_bigrams.intersection(truth_bigrams)) / len(truth_bigrams) 
                      if truth_bigrams else 0.0)
            
            # ROUGE-L (longest common subsequence)
            rouge_l = self._lcs_length(answer_tokens, truth_tokens) / len(truth_tokens)
            
            return {
                'rouge_1': rouge_1,
                'rouge_2': rouge_2,
                'rouge_l': rouge_l
            }
            
        except Exception:
            return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _evaluate_factual_accuracy(self, answer: str, ground_truth: str) -> float:
        """Evaluate factual accuracy compared to ground truth"""
        # Extract numerical facts and entities for comparison
        import re
        
        answer_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))
        truth_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', ground_truth))
        
        # Simple factual accuracy based on number overlap
        if truth_numbers:
            number_accuracy = len(answer_numbers.intersection(truth_numbers)) / len(truth_numbers)
        else:
            number_accuracy = 1.0 if not answer_numbers else 0.8
        
        # Entity overlap (simplified - just proper nouns)
        answer_entities = set(word for word in answer.split() if word[0].isupper())
        truth_entities = set(word for word in ground_truth.split() if word[0].isupper())
        
        if truth_entities:
            entity_accuracy = len(answer_entities.intersection(truth_entities)) / len(truth_entities)
        else:
            entity_accuracy = 1.0
        
        return (number_accuracy + entity_accuracy) / 2
    
    def _evaluate_answer_type_match(self, answer: str, ground_truth: str) -> float:
        """Evaluate if answer type matches expected type from ground truth"""
        # Determine answer types
        def get_answer_type(text):
            text_lower = text.lower().strip()
            if any(word in text_lower for word in ['yes', 'no', 'true', 'false']):
                return 'boolean'
            elif re.search(r'\b\d+(?:\.\d+)?\b', text):
                return 'numerical'
            elif any(word in text_lower for word in ['who', 'what', 'where', 'when', 'why', 'how']):
                return 'descriptive'
            else:
                return 'general'
        
        answer_type = get_answer_type(answer)
        truth_type = get_answer_type(ground_truth)
        
        return 1.0 if answer_type == truth_type else 0.5
    
    def _calculate_information_density(self, answer: str) -> float:
        """Calculate information density (content words per total words)"""
        words = answer.lower().split()
        if not words:
            return 0.0
        
        # Common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
        content_words = [word for word in words if word not in stop_words]
        return len(content_words) / len(words)
    
    def _evaluate_specificity(self, answer: str) -> float:
        """Evaluate how specific vs generic the answer is"""
        # Look for specific indicators
        specific_indicators = [
            len(re.findall(r'\b\d+(?:\.\d+)?\b', answer)),  # Numbers
            len(re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', answer)),  # Proper names
            answer.count('%'),  # Percentages
            len(re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', answer)),  # Dates
        ]
        
        # Generic indicators (penalty)
        generic_words = ['generally', 'usually', 'typically', 'often', 'sometimes', 'maybe', 'perhaps']
        generic_count = sum(1 for word in generic_words if word in answer.lower())
        
        specificity_score = min(1.0, (sum(specific_indicators) * 0.1) - (generic_count * 0.1) + 0.5)
        return max(0.0, specificity_score)
    
    def _evaluate_helpfulness(self, query: str, answer: str) -> float:
        """Evaluate how helpful the answer is for the given query"""
        # Check if answer directly addresses the query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Word overlap
        overlap = len(query_words.intersection(answer_words))
        word_match_score = overlap / len(query_words) if query_words else 0.0
        
        # Check for actionable content
        actionable_words = ['should', 'can', 'will', 'how to', 'step', 'process', 'method']
        actionable_score = sum(1 for word in actionable_words if word in answer.lower()) * 0.1
        
        # Check for complete response indicators
        completeness_indicators = ['in conclusion', 'therefore', 'as a result', 'overall']
        completeness_score = min(0.2, sum(1 for phrase in completeness_indicators if phrase in answer.lower()) * 0.1)
        
        helpfulness = min(1.0, word_match_score + actionable_score + completeness_score)
        return helpfulness