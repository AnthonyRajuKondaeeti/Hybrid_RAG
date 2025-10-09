"""
Enhanced RAG Service V2 - Improved Quality and Reduced Hallucinations

This version implements:
1. Strict source grounding to prevent hallucinations
2. Answer validation against source chunks
3. Confidence scoring based on source alignment
4. Better prompt engineering for precision
5. Fact-checking mechanisms
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import existing services
from services.document_service import DocumentService
from services.conversation_memory import ConversationMemoryManager
from config import Config

class SourceGroundingValidator:
    """Validates answers against source material to prevent hallucinations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def validate_answer_grounding(self, answer: str, source_chunks: List[str], 
                                threshold: float = 0.3) -> Dict[str, Any]:
        """
        Validate if answer content is grounded in source material
        
        Args:
            answer: Generated answer text
            source_chunks: List of source text chunks
            threshold: Minimum similarity threshold for validation
            
        Returns:
            Dict with grounding score, validation result, and details
        """
        if not source_chunks or not answer.strip():
            return {
                'is_grounded': False,
                'grounding_score': 0.0,
                'validated_statements': [],
                'unsupported_content': answer,
                'confidence': 0.0
            }
        
        # Split answer into sentences for granular validation
        sentences = self._split_into_sentences(answer)
        validated_statements = []
        unsupported_content = []
        
        # Combine all source chunks
        combined_sources = ' '.join(source_chunks)
        
        # Validate each sentence
        total_grounding_score = 0.0
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
                
            grounding_score = self._calculate_sentence_grounding(
                sentence, combined_sources
            )
            
            if grounding_score >= threshold:
                validated_statements.append({
                    'text': sentence,
                    'grounding_score': grounding_score,
                    'is_supported': True
                })
            else:
                validated_statements.append({
                    'text': sentence,
                    'grounding_score': grounding_score,
                    'is_supported': False
                })
                unsupported_content.append(sentence)
            
            total_grounding_score += grounding_score
        
        # Calculate overall metrics
        avg_grounding_score = total_grounding_score / len(validated_statements) if validated_statements else 0.0
        supported_ratio = len([s for s in validated_statements if s['is_supported']]) / len(validated_statements) if validated_statements else 0.0
        
        return {
            'is_grounded': supported_ratio >= 0.7,  # 70% of statements must be supported
            'grounding_score': avg_grounding_score,
            'supported_ratio': supported_ratio,
            'validated_statements': validated_statements,
            'unsupported_content': ' '.join(unsupported_content),
            'confidence': min(avg_grounding_score * supported_ratio, 1.0)
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for granular validation"""
        # Simple sentence splitting - can be enhanced with NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_sentence_grounding(self, sentence: str, source_text: str) -> float:
        """Calculate how well a sentence is grounded in source material"""
        try:
            # Encode sentence and source
            sentence_embedding = self.semantic_model.encode([sentence])
            
            # Split source into chunks for better matching
            source_chunks = self._chunk_text(source_text, max_length=200)
            if not source_chunks:
                return 0.0
            
            source_embeddings = self.semantic_model.encode(source_chunks)
            
            # Calculate maximum similarity with any source chunk
            similarities = cosine_similarity(sentence_embedding, source_embeddings)[0]
            max_similarity = float(np.max(similarities))
            
            # Also check for exact phrase matches (higher weight)
            exact_match_score = self._check_exact_phrases(sentence, source_text)
            
            # Combine semantic and exact match scores
            final_score = max(max_similarity, exact_match_score * 0.8)
            
            return min(final_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating grounding score: {e}")
            return 0.0
    
    def _chunk_text(self, text: str, max_length: int = 200) -> List[str]:
        """Split text into smaller chunks for better similarity matching"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _check_exact_phrases(self, sentence: str, source_text: str) -> float:
        """Check for exact phrase matches between sentence and source"""
        sentence_lower = sentence.lower()
        source_lower = source_text.lower()
        
        # Extract key phrases (3+ words)
        words = sentence_lower.split()
        exact_matches = 0
        total_phrases = 0
        
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            total_phrases += 1
            if phrase in source_lower:
                exact_matches += 1
        
        return exact_matches / total_phrases if total_phrases > 0 else 0.0


class PrecisionPromptEngine:
    """Enhanced prompt engineering for better precision and reduced hallucinations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_precision_prompt(self, question: str, context_chunks: List[str], 
                              format_type: str = "general") -> str:
        """
        Create a precision-focused prompt that minimizes hallucinations
        
        Args:
            question: User's question
            context_chunks: Retrieved context chunks
            format_type: Document format for specialized handling
            
        Returns:
            Engineered prompt for precise answers
        """
        
        # Format-specific instructions
        format_instructions = self._get_format_specific_instructions(format_type)
        
        # Create numbered sources for reference
        numbered_sources = ""
        for i, chunk in enumerate(context_chunks, 1):
            numbered_sources += f"\n[Source {i}]: {chunk}\n"
        
        prompt = f"""You are a precise, fact-based assistant. Your task is to answer questions using ONLY the information provided in the sources below.

CRITICAL RULES:
1. ONLY use information that appears in the provided sources
2. If information is not in the sources, clearly state "This information is not available in the provided sources"
3. Always cite which source(s) you're using: [Source 1], [Source 2], etc.
4. Do NOT add your own knowledge or make assumptions
5. Do NOT provide specific numbers, dates, or facts unless they appear exactly in the sources
6. If the sources don't fully answer the question, say so explicitly

{format_instructions}

SOURCES:
{numbered_sources}

QUESTION: {question}

INSTRUCTIONS FOR YOUR ANSWER:
- Start by directly addressing the question
- Use only facts from the sources above
- Cite your sources using [Source X] format
- If the answer requires information not in the sources, explicitly state what's missing
- Be precise with numbers, dates, and technical terms - only use what's explicitly stated
- Keep your answer focused and concise

ANSWER:"""

        return prompt
    
    def _get_format_specific_instructions(self, format_type: str) -> str:
        """Get specialized instructions based on document format"""
        
        format_instructions = {
            "pdf": """
For PDF documents:
- Pay attention to structured data like tables, charts, and technical specifications
- Preserve exact numerical values and measurements
- Note any missing or unclear sections due to formatting""",
            
            "xlsx": """
For spreadsheet data:
- Cite specific cells, rows, or columns when referencing data
- Preserve exact numerical values and calculations
- Be clear about data ranges and time periods""",
            
            "csv": """
For CSV data:
- Reference specific columns and rows
- Preserve data types (numbers vs text)
- Be explicit about data relationships and trends""",
            
            "healthcare": """
For healthcare content:
- Be extremely precise with medical terminology
- Never extrapolate or suggest medical advice
- Cite exact guidelines, ranges, and recommendations
- Flag any incomplete information clearly""",
            
            "technical": """
For technical documents:
- Preserve exact specifications and requirements
- Maintain technical accuracy in terminology
- Reference specific components, versions, or standards""",
            
            "business": """
For business documents:
- Preserve exact financial figures and percentages
- Reference specific time periods and quarters
- Maintain accuracy in business terminology and metrics"""
        }
        
        return format_instructions.get(format_type.lower(), "")


class EnhancedRAGServiceV2:
    """
    Enhanced RAG Service with improved quality controls and reduced hallucinations
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.document_service = DocumentService(config)
        self.conversation_memory = ConversationMemoryManager(config.MONGODB_URI, config.DATABASE_NAME)
        
        # New quality control components
        self.source_validator = SourceGroundingValidator()
        self.prompt_engine = PrecisionPromptEngine()
        
        self.logger.info("Enhanced RAG Service V2 initialized with quality controls")
    
    def answer_question_with_validation(self, question: str, session_id: str,
                                      user_id: str = "default", 
                                      min_confidence: float = 0.6) -> Dict[str, Any]:
        """
        Answer question with comprehensive validation and quality controls
        
        Args:
            question: User's question
            session_id: Session identifier
            user_id: User identifier
            min_confidence: Minimum confidence threshold for answer
            
        Returns:
            Dict with answer, confidence, sources, and validation results
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant documents
            self.logger.info(f"Processing question: {question[:100]}...")
            
            search_results = self.document_service.search_documents(
                query=question,
                limit=10,
                include_metadata=True
            )
            
            if not search_results:
                return self._create_no_results_response(question, session_id)
            
            # Step 2: Extract context and detect format
            context_chunks = []
            source_details = []
            dominant_format = self._detect_dominant_format(search_results)
            
            for result in search_results:
                chunk_text = result.get('content', '')
                if chunk_text and len(chunk_text.strip()) > 20:
                    context_chunks.append(chunk_text)
                    source_details.append({
                        'content': chunk_text,
                        'metadata': result.get('metadata', {}),
                        'score': result.get('score', 0.0)
                    })
            
            if not context_chunks:
                return self._create_no_results_response(question, session_id)
            
            # Step 3: Create precision prompt
            content_type = self._detect_content_type(context_chunks[0])
            precision_prompt = self.prompt_engine.create_precision_prompt(
                question=question,
                context_chunks=context_chunks[:5],  # Limit to top 5 for focus
                format_type=content_type
            )
            
            # Step 4: Generate answer using existing LLM service
            raw_answer = self._generate_llm_response(precision_prompt)
            
            if not raw_answer or len(raw_answer.strip()) < 10:
                return self._create_low_quality_response(question, session_id, "Answer too short")
            
            # Step 5: Validate answer grounding
            validation_result = self.source_validator.validate_answer_grounding(
                answer=raw_answer,
                source_chunks=context_chunks,
                threshold=0.3
            )
            
            # Step 6: Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                validation_result, search_results
            )
            
            # Step 7: Apply quality filters
            if overall_confidence < min_confidence:
                return self._create_low_confidence_response(
                    question, session_id, overall_confidence, validation_result
                )
            
            # Step 8: Create enhanced response
            enhanced_response = self._create_enhanced_response(
                question=question,
                answer=raw_answer,
                source_details=source_details,
                validation_result=validation_result,
                confidence=overall_confidence,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Step 9: Store in conversation memory
            self._store_conversation(session_id, user_id, question, enhanced_response)
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Error in enhanced RAG processing: {e}")
            return self._create_error_response(question, session_id, str(e))
    
    def _detect_dominant_format(self, search_results: List[Dict]) -> str:
        """Detect the dominant document format in results"""
        format_counts = {}
        for result in search_results:
            metadata = result.get('metadata', {})
            file_format = metadata.get('format', 'unknown')
            format_counts[file_format] = format_counts.get(file_format, 0) + 1
        
        if format_counts:
            return max(format_counts, key=format_counts.get)
        return "general"
    
    def _detect_content_type(self, text: str) -> str:
        """Detect content type for specialized handling"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['hba1c', 'diabetes', 'blood pressure', 'medication']):
            return "healthcare"
        elif any(term in text_lower for term in ['revenue', 'profit', 'quarter', 'financial']):
            return "business"
        elif any(term in text_lower for term in ['api', 'system', 'algorithm', 'technical']):
            return "technical"
        else:
            return "general"
    
    def _generate_llm_response(self, prompt: str) -> str:
        """Generate LLM response using existing infrastructure"""
        # This would use your existing LLM service (Mistral, OpenAI, etc.)
        # For now, returning a placeholder - integrate with your actual LLM
        try:
            # Import and use your existing LLM service here
            from services.rag_service import EnhancedRAGService
            temp_service = EnhancedRAGService(self.config)
            
            # Use the existing LLM generation
            response = temp_service._generate_answer_with_llm(prompt)
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {e}")
            return ""
    
    def _calculate_overall_confidence(self, validation_result: Dict, 
                                    search_results: List[Dict]) -> float:
        """Calculate overall confidence based on multiple factors"""
        
        # Base confidence from source grounding
        grounding_confidence = validation_result.get('confidence', 0.0)
        
        # Search relevance confidence
        if search_results:
            avg_search_score = sum(r.get('score', 0.0) for r in search_results) / len(search_results)
            search_confidence = min(avg_search_score, 1.0)
        else:
            search_confidence = 0.0
        
        # Source coverage confidence
        supported_ratio = validation_result.get('supported_ratio', 0.0)
        
        # Weighted combination
        overall_confidence = (
            grounding_confidence * 0.5 +
            search_confidence * 0.3 +
            supported_ratio * 0.2
        )
        
        return min(overall_confidence, 1.0)
    
    def _create_enhanced_response(self, question: str, answer: str, 
                                source_details: List[Dict], validation_result: Dict,
                                confidence: float, processing_time: float) -> Dict[str, Any]:
        """Create comprehensive response with all validation details"""
        
        return {
            'answer': answer,
            'confidence_score': confidence,
            'processing_time': processing_time,
            'validation': {
                'is_grounded': validation_result.get('is_grounded', False),
                'grounding_score': validation_result.get('grounding_score', 0.0),
                'supported_ratio': validation_result.get('supported_ratio', 0.0),
                'unsupported_content': validation_result.get('unsupported_content', ''),
                'statement_analysis': validation_result.get('validated_statements', [])
            },
            'sources': [
                {
                    'content_preview': detail['content'][:200] + '...' if len(detail['content']) > 200 else detail['content'],
                    'relevance_score': detail['score'],
                    'metadata': detail['metadata']
                }
                for detail in source_details[:5]
            ],
            'quality_metrics': {
                'hallucination_risk': 1.0 - validation_result.get('grounding_score', 0.0),
                'answer_precision': validation_result.get('supported_ratio', 0.0),
                'source_coverage': len(source_details),
                'processing_quality': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low'
            },
            'recommendations': self._generate_quality_recommendations(validation_result, confidence)
        }
    
    def _generate_quality_recommendations(self, validation_result: Dict, 
                                        confidence: float) -> List[str]:
        """Generate recommendations based on quality analysis"""
        recommendations = []
        
        if validation_result.get('grounding_score', 0) < 0.7:
            recommendations.append("Answer may contain information not fully supported by sources")
        
        if validation_result.get('supported_ratio', 0) < 0.8:
            recommendations.append("Some statements in the answer lack strong source support")
        
        if confidence < 0.7:
            recommendations.append("Consider rephrasing the question for better results")
        
        if validation_result.get('unsupported_content'):
            recommendations.append("Review highlighted unsupported content carefully")
        
        if not recommendations:
            recommendations.append("High-quality answer with strong source grounding")
        
        return recommendations
    
    def _create_no_results_response(self, question: str, session_id: str) -> Dict[str, Any]:
        """Create response when no relevant documents found"""
        return {
            'answer': "I couldn't find relevant information in the document collection to answer your question. Please try rephrasing your question or check if the relevant documents have been uploaded.",
            'confidence_score': 0.0,
            'processing_time': 0.0,
            'validation': {'is_grounded': False, 'grounding_score': 0.0},
            'sources': [],
            'quality_metrics': {'processing_quality': 'no_results'},
            'recommendations': ["Try rephrasing the question", "Check if relevant documents are uploaded"]
        }
    
    def _create_low_confidence_response(self, question: str, session_id: str,
                                      confidence: float, validation_result: Dict) -> Dict[str, Any]:
        """Create response for low confidence answers"""
        return {
            'answer': f"I found some relevant information, but I'm not confident in the answer quality (confidence: {confidence:.2f}). The information may not be complete or fully accurate based on the available sources.",
            'confidence_score': confidence,
            'validation': validation_result,
            'quality_metrics': {'processing_quality': 'low_confidence'},
            'recommendations': [
                "Try rephrasing the question for better results",
                "Check if more relevant documents are available",
                "Consider breaking complex questions into simpler parts"
            ]
        }
    
    def _create_error_response(self, question: str, session_id: str, error: str) -> Dict[str, Any]:
        """Create response for processing errors"""
        return {
            'answer': "I encountered an error while processing your question. Please try again.",
            'confidence_score': 0.0,
            'error': error,
            'quality_metrics': {'processing_quality': 'error'},
            'recommendations': ["Try again with a simpler question"]
        }
    
    def _store_conversation(self, session_id: str, user_id: str, 
                          question: str, response: Dict[str, Any]) -> None:
        """Store conversation with enhanced metadata"""
        try:
            conversation_data = {
                'question': question,
                'answer': response['answer'],
                'confidence': response['confidence_score'],
                'validation_metrics': response.get('validation', {}),
                'quality_metrics': response.get('quality_metrics', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            self.conversation_memory.store_conversation(
                session_id, user_id, conversation_data
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store conversation: {e}")


# Export the enhanced service
__all__ = ['EnhancedRAGServiceV2', 'SourceGroundingValidator', 'PrecisionPromptEngine']