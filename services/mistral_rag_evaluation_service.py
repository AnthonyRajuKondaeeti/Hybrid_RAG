# services/mistral_rag_evaluation_service.py
import os
import csv
import json
import tempfile
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
import numpy as np

from langchain_mistralai.chat_models import ChatMistralAI

logger = logging.getLogger(__name__)

class MistralRAGEvaluationService:
    """RAG Evaluation Service using Mistral API instead of OpenAI"""
    
    def __init__(self, mistral_api_key: str, mistral_model: str = "mistral-large-latest"):
        self.mistral_api_key = mistral_api_key
        self.mistral_model = mistral_model
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize Mistral LLM
        self.llm = ChatMistralAI(
            model=mistral_model,
            temperature=0.1,
            api_key=mistral_api_key
        )
    
    def evaluate_rag_responses(
        self, 
        queries: List[str], 
        rag_responses: List[Dict[str, Any]], 
        file_id: str,
        evaluation_name: str = "rag_evaluation"
    ) -> Dict[str, Any]:
        """
        Evaluate RAG responses using Mistral-based metrics
        
        Args:
            queries: List of questions asked
            rag_responses: List of RAG response objects from your system
            file_id: Document file ID being evaluated
            evaluation_name: Name for this evaluation run
            
        Returns:
            Evaluation results dictionary
        """
        try:
            results = []
            
            for i, (query, response) in enumerate(zip(queries, rag_responses)):
                if not response.get('success'):
                    continue
                
                # Evaluate individual response
                eval_result = self._evaluate_single_response(query, response, f"{file_id}_q{i}")
                results.append(eval_result)
            
            # Calculate summary statistics
            summary = self._calculate_summary_stats(results)
            
            return {
                'success': True,
                'evaluation_name': evaluation_name,
                'timestamp': datetime.now().isoformat(),
                'summary': summary,
                'detailed_results': results,
                'total_queries': len(results),
                'evaluator': 'mistral-custom'
            }
            
        except Exception as e:
            logger.error(f"Mistral RAG evaluation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _evaluate_single_response(self, query: str, response: Dict[str, Any], query_id: str) -> Dict[str, Any]:
        """Evaluate a single RAG response using Mistral-based metrics"""
        
        answer = response.get('answer', '')
        sources = response.get('sources', [])
        context = "\n".join([source.get('content_preview', '') for source in sources])
        
        # Calculate various metrics
        metrics = {
            'query_id': query_id,
            'query': query,
            'answer': answer,
            'answer_relevancy': self._calculate_answer_relevancy(query, answer),
            'faithfulness': self._calculate_faithfulness(answer, context),
            'context_utilization': self._calculate_context_utilization(answer, context),
            'answer_completeness': self._calculate_answer_completeness(query, answer),
            'hallucination_score': self._calculate_hallucination_score(answer, context),
            'semantic_similarity': self._calculate_semantic_similarity(query, answer)
        }
        
        return metrics
    
    def _calculate_answer_relevancy(self, query: str, answer: str) -> float:
        """Calculate how relevant the answer is to the query using Mistral"""
        prompt = f"""
        Rate how relevant this answer is to the given question on a scale of 0.0 to 1.0.
        
        Question: {query}
        Answer: {answer}
        
        Consider:
        - Does the answer directly address the question?
        - Is the answer focused and on-topic?
        - How well does it satisfy the information need?
        
        Respond with only a number between 0.0 and 1.0, where:
        - 1.0 = Perfectly relevant and directly answers the question
        - 0.5 = Somewhat relevant but may be incomplete or tangential
        - 0.0 = Completely irrelevant or doesn't answer the question
        
        Score:"""
        
        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default fallback
    
    def _calculate_faithfulness(self, answer: str, context: str) -> float:
        """Calculate how faithful the answer is to the provided context"""
        if not context.strip():
            return 0.0
        
        prompt = f"""
        Rate how faithful this answer is to the given context on a scale of 0.0 to 1.0.
        
        Context: {context}
        Answer: {answer}
        
        Consider:
        - Are all claims in the answer supported by the context?
        - Does the answer contradict anything in the context?
        - How well does the answer stay within the bounds of the provided information?
        
        Respond with only a number between 0.0 and 1.0, where:
        - 1.0 = Completely faithful, all information comes from context
        - 0.5 = Mostly faithful but some unsupported claims
        - 0.0 = Unfaithful, contradicts or ignores context
        
        Score:"""
        
        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _calculate_context_utilization(self, answer: str, context: str) -> float:
        """Calculate how well the answer utilizes the provided context"""
        if not context.strip():
            return 0.0
        
        prompt = f"""
        Rate how well this answer utilizes the provided context on a scale of 0.0 to 1.0.
        
        Context: {context}
        Answer: {answer}
        
        Consider:
        - Does the answer make good use of the relevant information in the context?
        - Is important context information included in the answer?
        - Is the context information well-integrated into the response?
        
        Respond with only a number between 0.0 and 1.0, where:
        - 1.0 = Excellent utilization, key context well-integrated
        - 0.5 = Moderate utilization, some context used
        - 0.0 = Poor utilization, context mostly ignored
        
        Score:"""
        
        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _calculate_answer_completeness(self, query: str, answer: str) -> float:
        """Calculate how complete the answer is for the given query"""
        prompt = f"""
        Rate how complete this answer is for the given question on a scale of 0.0 to 1.0.
        
        Question: {query}
        Answer: {answer}
        
        Consider:
        - Does the answer fully address all aspects of the question?
        - Are there obvious gaps or missing information?
        - Is the level of detail appropriate for the question?
        
        Respond with only a number between 0.0 and 1.0, where:
        - 1.0 = Very complete, thoroughly answers the question
        - 0.5 = Somewhat complete but missing some aspects
        - 0.0 = Incomplete, major gaps in the answer
        
        Score:"""
        
        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _calculate_hallucination_score(self, answer: str, context: str) -> float:
        """Calculate hallucination score (lower is better)"""
        if not context.strip():
            return 1.0  # High hallucination if no context
        
        prompt = f"""
        Rate the level of hallucination in this answer compared to the context on a scale of 0.0 to 1.0.
        
        Context: {context}
        Answer: {answer}
        
        Consider:
        - Does the answer contain information not present in the context?
        - Are there fabricated facts, numbers, or claims?
        - Does the answer make up details not supported by the context?
        
        Respond with only a number between 0.0 and 1.0, where:
        - 0.0 = No hallucination, all information grounded in context
        - 0.5 = Some unsupported claims but mostly accurate
        - 1.0 = Significant hallucination, many unsupported claims
        
        Score:"""
        
        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _calculate_semantic_similarity(self, query: str, answer: str) -> float:
        """Calculate semantic similarity between query and answer"""
        # Simple word overlap approach (can be enhanced with embeddings)
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(answer_words))
        return overlap / len(query_words)
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate summary statistics from evaluation results"""
        if not results:
            return {}
        
        summary = {}
        numeric_fields = [
            'answer_relevancy', 'faithfulness', 'context_utilization',
            'answer_completeness', 'hallucination_score', 'semantic_similarity'
        ]
        
        for field in numeric_fields:
            values = []
            for result in results:
                try:
                    value = float(result.get(field, 0))
                    values.append(value)
                except (ValueError, TypeError):
                    continue
            
            if values:
                summary[f'{field}_mean'] = sum(values) / len(values)
                summary[f'{field}_min'] = min(values)
                summary[f'{field}_max'] = max(values)
                summary[f'{field}_std'] = np.std(values) if len(values) > 1 else 0.0
        
        # Calculate overall score (lower hallucination is better, so invert it)
        if 'hallucination_score_mean' in summary:
            summary['overall_quality'] = (
                summary.get('answer_relevancy_mean', 0) * 0.25 +
                summary.get('faithfulness_mean', 0) * 0.25 +
                summary.get('context_utilization_mean', 0) * 0.25 +
                (1 - summary.get('hallucination_score_mean', 0)) * 0.25
            )
        
        return summary
    
    def evaluate_single_response(
        self, 
        query: str, 
        response: Dict[str, Any], 
        file_id: str
    ) -> Dict[str, Any]:
        """Evaluate a single RAG response"""
        return self.evaluate_rag_responses([query], [response], file_id, f"single_eval_{file_id}")
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {str(e)}")


# Enhanced RAG Service using Mistral for evaluation
class MistralEnhancedRAGService:
    """Enhanced RAG Service with Mistral-based evaluation capabilities"""
    
    def __init__(self, original_rag_service, mistral_api_key: str, mistral_model: str = "mistral-large-latest"):
        self.rag_service = original_rag_service
        self.evaluation_service = MistralRAGEvaluationService(mistral_api_key, mistral_model)
        self.evaluation_history = []
    
    def answer_question_with_evaluation(
        self, 
        file_id: str, 
        question: str, 
        conversation_history: List[Dict] = None,
        enable_evaluation: bool = True
    ) -> Dict[str, Any]:
        """Answer question and optionally evaluate the response using Mistral"""
        
        # Get the original response
        response = self.rag_service.answer_question(file_id, question, conversation_history)
        
        if enable_evaluation and response.get('success'):
            try:
                # Evaluate the response using Mistral
                eval_result = self.evaluation_service.evaluate_single_response(
                    query=question,
                    response=response,
                    file_id=file_id
                )
                
                # Add evaluation metrics to response
                if eval_result.get('success') and eval_result.get('detailed_results'):
                    metrics = eval_result['detailed_results'][0]
                    response['evaluation'] = {
                        'answer_relevancy': metrics.get('answer_relevancy', 0),
                        'faithfulness': metrics.get('faithfulness', 0),
                        'context_utilization': metrics.get('context_utilization', 0),
                        'answer_completeness': metrics.get('answer_completeness', 0),
                        'hallucination_score': metrics.get('hallucination_score', 0),
                        'semantic_similarity': metrics.get('semantic_similarity', 0),
                        'evaluation_timestamp': eval_result['timestamp'],
                        'evaluator': 'mistral-custom'
                    }
                    
                    # Store evaluation history
                    self.evaluation_history.append({
                        'question': question,
                        'file_id': file_id,
                        'evaluation': response['evaluation'],
                        'timestamp': eval_result['timestamp']
                    })
                
            except Exception as e:
                logger.warning(f"Mistral evaluation failed: {str(e)}")
                response['evaluation_error'] = str(e)
        
        return response
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history"""
        return self.evaluation_history
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}
        
        # Calculate averages across all evaluations
        metrics = ['answer_relevancy', 'faithfulness', 'context_utilization', 
                  'answer_completeness', 'hallucination_score', 'semantic_similarity']
        summary = {}
        
        for metric in metrics:
            values = [eval_data['evaluation'][metric] for eval_data in self.evaluation_history 
                     if metric in eval_data['evaluation']]
            if values:
                summary[f'{metric}_average'] = sum(values) / len(values)
        
        summary['total_evaluations'] = len(self.evaluation_history)
        summary['last_evaluation'] = self.evaluation_history[-1]['timestamp']
        summary['evaluator'] = 'mistral-custom'
        
        return summary