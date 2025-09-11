import os
import time
import logging
import json
import uuid
import re
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import PromptTemplate

from config import Config

# --- NEW IMPORTS & CLASSES ---
# Helper class for conversation history
class ConversationTurn:
    def __init__(self, question, answer, timestamp, confidence_score, question_type, processing_time, sources_used):
        self.question = question
        self.answer = answer
        self.timestamp = timestamp
        self.confidence_score = confidence_score
        self.question_type = question_type
        self.processing_time = processing_time
        self.sources_used = sources_used

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
            "confidence_score": self.confidence_score,
            "question_type": self.question_type,
            "processing_time": self.processing_time,
            "sources_used": self.sources_used
        }

# Conversation memory manager
class ConversationMemoryManager:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.conversation_history: List[ConversationTurn] = []
        self.query_cache: Dict[str, Any] = {}
        self.session_start = datetime.now()

    def add_turn(self, turn: ConversationTurn):
        self.conversation_history.append(turn)
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history.pop(0)

    def get_context_string(self) -> str:
        context_parts = []
        for turn in self.conversation_history:
            context_parts.append(f"Q: {turn.question}")
            context_parts.append(f"A: {turn.answer}")
        return "\n".join(context_parts)

    def cache_query(self, question: str, result: Any):
        self.query_cache[question.lower()] = result

    def get_cached_result(self, question: str) -> Optional[Any]:
        return self.query_cache.get(question.lower())
    
    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            'total_turns': len(self.conversation_history),
            'cache_size': len(self.query_cache),
        }
    
    def get_related_questions(self, question: str) -> List[str]:
        related = []
        q_words = set(re.findall(r'\b\w+\b', question.lower()))
        for turn in self.conversation_history:
            turn_words = set(re.findall(r'\b\w+\b', turn.question.lower()))
            if len(q_words.intersection(turn_words)) > 2:
                related.append(turn.question)
        return related

# Configure logging
logger = logging.getLogger(__name__)

class RAGService:
    """Service for RAG operations with Qdrant"""
    
    def __init__(self):
        self.config = Config
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=self.config.QDRANT_URL,
            api_key=self.config.QDRANT_API_KEY
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize LLM
        self.llm = ChatMistralAI(
            model=self.config.MISTRAL_MODEL,
            temperature=0.1,
            api_key=self.config.MISTRAL_API_KEY
        )
        
        # Ensure collection exists
        self._ensure_collection()
        
        # --- NEW ATTRIBUTES ---
        self.memory_manager = ConversationMemoryManager()
        self.session_stats = {
            'questions_asked': 0,
            'total_search_time': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'average_confidence': 0.0,
            'cache_hits': 0,
            'session_start': datetime.now()
        }

    def _ensure_collection(self):
        """
        Ensure Qdrant collection exists and has the necessary payload indexes.
        This version is robust and creates indexes even if the collection already exists.
        """
        try:
            collection_name = self.config.COLLECTION_NAME
            
            # Check if the collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                # Create the collection if it doesn't exist
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {collection_name}")

            # --- START: ROBUST INDEX CREATION ---
            # This logic runs every time to ensure indexes are present.
            # Qdrant's create_payload_index is idempotent (safe to run multiple times).

            # Index for 'file_id'
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="file_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True  # Wait for the operation to complete
            )
            logger.info("Ensured payload index exists for 'file_id'.")
            
            # Index for 'is_metadata'
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="is_metadata",
                field_schema=models.PayloadSchemaType.BOOL,
                wait=True # Wait for the operation to complete
            )
            logger.info("Ensured payload index exists for 'is_metadata'.")
            # --- END: ROBUST INDEX CREATION ---

        except Exception as e:
            logger.error(f"Failed to ensure collection and indexes: {str(e)}")
            raise
    
    def store_document_chunks(self, chunks: List[Document], file_id: str, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store document chunks in Qdrant"""
        try:
            logger.info(f"Storing {len(chunks)} chunks for document {file_id}")
            
            points = []
            
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_model.encode(chunk.page_content).tolist()
                
                # Create a valid UUID for the point ID
                point_id = str(uuid.uuid4())
                
                # Prepare payload
                payload = {
                    'content': chunk.page_content,
                    'file_id': file_id,
                    'chunk_id': chunk.metadata.get('chunk_id', f"{file_id}_{i}"),
                    'chunk_index': i,
                    'page': chunk.metadata.get('page', 1),
                    'content_type': chunk.metadata.get('content_type', 'content'),
                    'semantic_score': chunk.metadata.get('semantic_score', 0.5),
                    'key_terms': chunk.metadata.get('key_terms', ''),
                    'word_count': chunk.metadata.get('word_count', len(chunk.page_content.split())),
                    'filename': document_metadata.get('filename', ''),
                    'is_metadata': False,
                    'created_at': datetime.now().isoformat()
                }
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.config.COLLECTION_NAME,
                    points=batch
                )
            
            # Store document metadata
            self._store_document_metadata(file_id, document_metadata)
            
            logger.info(f"Successfully stored {len(chunks)} chunks")
            
            return {
                'success': True,
                'chunks_count': len(chunks),
                'file_id': file_id
            }
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _store_document_metadata(self, file_id: str, metadata: Dict[str, Any]):
        """Store document metadata as a special point"""
        try:
            # Create a dummy embedding for metadata (not used for search)
            dummy_embedding = [0.0] * self.embedding_dim
            
            # Create a valid UUID for metadata point
            metadata_id = str(uuid.uuid4())
            
            payload = {
                'content': f"METADATA for {metadata.get('filename', file_id)}",
                'file_id': file_id,
                'is_metadata': True,
                'document_metadata': metadata,
                'created_at': datetime.now().isoformat()
            }
            
            self.qdrant_client.upsert(
                collection_name=self.config.COLLECTION_NAME,
                points=[models.PointStruct(
                    id=metadata_id,
                    vector=dummy_embedding,
                    payload=payload
                )]
            )
            
        except Exception as e:
            logger.warning(f"Failed to store document metadata: {str(e)}")
    
    def document_exists(self, file_id: str) -> bool:
        """Check if document exists in vector store"""
        try:
            logger.info(f"Checking existence of document with file_id: {file_id}")
            result = self.qdrant_client.scroll(
                collection_name=self.config.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=file_id)
                        ),
                        models.FieldCondition(
                            key="is_metadata",
                            match=models.MatchValue(value=False)
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="is_metadata",
                            match=models.MatchValue(value=True)
                        )
                    ]
                ),
                limit=1
            )
            return len(result[0]) > 0
        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            # Fallback: try without metadata filter
            try:
                result = self.qdrant_client.scroll(
                    collection_name=self.config.COLLECTION_NAME,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_id",
                                match=models.MatchValue(value=file_id)
                            )
                        ]
                    ),
                    limit=1
                )
                return len(result[0]) > 0
            except:
                return False
            
    def _classify_question_type(self, question: str) -> str:
        """Classify question type"""
        question_lower = question.lower()
        
        # Check for general/overview questions first
        if any(word in question_lower for word in ['about', 'overview', 'summarize', 'summary', 'main topic', 'what is this document']):
            return 'general'
        elif any(word in question_lower for word in ['what', 'define', 'meaning', 'explain']):
            return 'definition'
        elif any(word in question_lower for word in ['how many', 'count', 'number']):
            return 'quantitative'
        elif any(word in question_lower for word in ['how', 'method', 'process', 'way']):
            return 'procedural'
        elif any(word in question_lower for word in ['why', 'reason', 'cause']):
            return 'causal'
        elif any(word in question_lower for word in ['when', 'time', 'date']):
            return 'temporal'
        elif any(word in question_lower for word in ['compare', 'difference', 'similar', 'versus']):
            return 'comparative'
        else:
            return 'general'
        
    def answer_question(self, file_id: str, question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Answer question using RAG"""
        try:
            start_time = time.time()
            self.session_stats['questions_asked'] += 1

            # Check cache first
            cached_result = self.memory_manager.get_cached_result(question)
            if cached_result:
                self.session_stats['cache_hits'] += 1
                return cached_result

            # Get conversation context
            conversation_context = self.memory_manager.get_context_string()

            # Classify question type
            question_type = self._classify_question_type(question)
            is_followup_question = self._is_followup_question(question, conversation_context)

            # Adjust search parameters based on question type
            if question_type == 'general' or any(word in question.lower() for word in ['about', 'overview', 'summarize', 'summary']):
                limit = 12
                score_threshold = 0.1
            else:
                limit = 8
                score_threshold = 0.3

            # Search for relevant chunks
            question_embedding = self.embedding_model.encode(question).tolist()
            search_results = self.qdrant_client.search(
                collection_name=self.config.COLLECTION_NAME,
                query_vector=question_embedding,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))]
                ),
                limit=limit,
                score_threshold=score_threshold
            )
            content_results = [result for result in search_results if not result.payload.get('is_metadata', False)]
            
            # Fallback search if no results with threshold
            if not content_results and (question_type == 'general' or any(word in question.lower() for word in ['about', 'overview', 'summarize', 'summary'])):
                search_results = self.qdrant_client.search(
                    collection_name=self.config.COLLECTION_NAME,
                    query_vector=question_embedding,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_id",
                                match=models.MatchValue(value=file_id)
                            )
                        ]
                    ),
                    limit=limit
                )
                content_results = [result for result in search_results if not result.payload.get('is_metadata', False)]
            
            if not content_results:
                return {'success': False, 'error': 'No relevant content found for the question'}

            context_str = "\n\n".join([result.payload['content'] for result in content_results])
            sources = [{
                'page': result.payload.get('page', 'N/A'),
                'content_preview': result.payload['content'][:200] + "..." if len(result.payload['content']) > 200 else result.payload['content'],
                'relevance_score': float(result.score),
                'content_type': result.payload.get('content_type', 'content'),
                'semantic_score': result.payload.get('semantic_score', 0.5)
            } for result in content_results]

            # --- MODIFIED: Prompt with JSON schema ---
            prompt = self._create_structured_prompt(question, context_str, conversation_context)
            
            # --- MODIFIED: LLM call with JSON mode and parsing ---
            try:
                response = self.llm.invoke(prompt, response_format={"type": "json_object"})
                answer_content = json.loads(response.content)
                answer = answer_content.get('answer', 'No answer found.')
                formatted_sources = answer_content.get('sources', [])
                # Reformat sources to match the expected output structure
                parsed_sources = []
                for source_page in formatted_sources:
                    for s in sources:
                        if str(s['page']) == str(source_page):
                            parsed_sources.append(s)
                            break
            except json.JSONDecodeError:
                # Fallback to plain text if JSON parsing fails
                answer = response.content
                parsed_sources = sources
            except Exception as e:
                # Catch other LLM or API errors
                logger.error(f"LLM call or JSON parsing failed: {str(e)}")
                answer = "I'm sorry, I couldn't process the request correctly. Please try again."
                parsed_sources = sources

            # Use the new enhanced confidence score
            avg_relevance = np.mean([s['relevance_score'] for s in parsed_sources]) if parsed_sources else 0.0
            confidence_score = self._calculate_enhanced_confidence_score(answer, avg_relevance, is_followup_question)

            follow_up_suggestions = self._generate_enhanced_followup_suggestions(
                question, answer, conversation_context, self._is_metadata_question(question), self._is_image_question(question), is_followup_question
            )

            processing_time = time.time() - start_time
            self.session_stats['total_search_time'] += processing_time
            self.session_stats['successful_searches'] += 1
            if self.session_stats['successful_searches'] > 0:
                total_confidence = (self.session_stats['average_confidence'] * (self.session_stats['successful_searches'] - 1) + confidence_score)
                self.session_stats['average_confidence'] = total_confidence / self.session_stats['successful_searches']
            
            # Determine final question type for memory
            final_question_type = 'followup' if is_followup_question else question_type
            
            # Add to conversation memory
            conversation_turn = ConversationTurn(
                question=question,
                answer=answer,
                timestamp=datetime.now(),
                confidence_score=confidence_score,
                question_type=final_question_type,
                processing_time=processing_time,
                sources_used=[s['page'] for s in parsed_sources[:3]]
            )
            self.memory_manager.add_turn(conversation_turn)
            self.memory_manager.cache_query(question, {
                'success': True,
                'answer': answer,
                'confidence_score': confidence_score,
                'sources': parsed_sources,
                'follow_up_suggestions': follow_up_suggestions,
                'processing_time': processing_time,
                'question_type': final_question_type,
                'context_chunks_used': len(content_results)
            })

            return {
                'success': True,
                'answer': answer,
                'confidence_score': confidence_score,
                'sources': parsed_sources,
                'follow_up_suggestions': follow_up_suggestions,
                'processing_time': processing_time,
                'question_type': final_question_type,
                'context_chunks_used': len(content_results)
            }
        
        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            self.session_stats['failed_searches'] += 1
            return {'success': False, 'error': str(e)}
        
    def _create_structured_prompt(self, question: str, context: str, conversation_history: str) -> str:
        """Create a prompt that instructs the LLM to return a structured JSON object."""
        
        system_instructions = """You are an expert document analyst. Your task is to provide a complete, well-structured answer to the user's question based ONLY on the provided document context.

    Your response must be a JSON object with the following keys:
    - "answer": A string containing the comprehensive answer to the question.
    - "sources": An array of page numbers (as strings) from which the information was sourced.

    Guidelines for the "answer" field:
    - Provide a comprehensive, well-structured answer based on the context.
    - If the information is not in the context, state this clearly in the answer.
    - Do not include any extra commentary or introductory text outside of the JSON.
    - Maintain consistency with the conversation history when relevant."""

        conversation_section = f"\n\nCONVERSATION HISTORY:\n{conversation_history}\n" if conversation_history else ""
        
        prompt = f"""{system_instructions}
        
    {conversation_section}

    CONTEXT FROM DOCUMENT:
    {context}

    QUESTION: {question}

    PROVIDE A JSON RESPONSE:"""
        return prompt
    
    def _create_rag_prompt(self, question: str, context: str, conversation_history: str, question_type: str) -> str:
        """Create RAG prompt based on question type"""
        
        base_instructions = """You are an expert document analyst. Answer the question based on the provided context from the document.

    Guidelines:
    - Provide comprehensive, well-structured answers based on the context
    - Include specific details, examples, and data points when available
    - For technical questions, explain concepts clearly
    - Use bullet points or numbered lists when appropriate for clarity
    - If information is not in the context, state this clearly
    - Always cite relevant information from the context
    - Maintain consistency with conversation history when relevant"""

        type_specific_instructions = {
            'definition': "Provide a clear, comprehensive definition with examples and context from the document.",
            'quantitative': "List all specific numbers, quantities, metrics, and measurements mentioned in the context.",
            'procedural': "Provide a detailed step-by-step explanation of the process or methodology described.",
            'causal': "Explain the complete cause-and-effect relationships, reasons, and underlying factors mentioned.",
            'temporal': "Provide all relevant dates, times, sequences, and temporal relationships from the context.",
            'comparative': "Provide a thorough comparison highlighting all similarities, differences, and relationships described.",
            'general': "Provide a comprehensive overview covering all major topics, key concepts, important details, and conclusions. Structure your response to cover: 1) Main subject/purpose, 2) Key topics and concepts, 3) Important details and findings, 4) Conclusions or implications."
        }
        
        conversation_section = f"\n\nCONVERSATION HISTORY:\n{conversation_history}\n" if conversation_history else ""
        
        prompt = f"""{base_instructions}

    {type_specific_instructions.get(question_type, type_specific_instructions['general'])}{conversation_section}

    CONTEXT FROM DOCUMENT:
    {context}

    QUESTION: {question}

    PROVIDE A COMPLETE, DETAILED ANSWER:"""
        
        return prompt
    
    def _calculate_confidence_score(self, question: str, answer: str, search_results: List) -> float:
        """Calculate confidence score for the answer"""
        try:
            # Base confidence from search relevance
            if not search_results:
                return 0.0
            
            avg_score = np.mean([result.score for result in search_results])
            base_confidence = min(avg_score * 1.2, 1.0)  # Boost slightly
            
            # Adjust based on answer characteristics
            answer_lower = answer.lower()
            
            # Uncertainty indicators
            uncertainty_phrases = [
                "i don't know", "not sure", "unclear", "might be", "possibly", 
                "perhaps", "maybe", "could be", "appears to", "seems"
            ]
            
            uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in answer_lower)
            
            # Confidence indicators
            confidence_phrases = [
                "according to", "the document states", "specifically mentions",
                "clearly indicates", "explicitly", "precisely"
            ]
            
            confidence_count = sum(1 for phrase in confidence_phrases if phrase in answer_lower)
            
            # Final confidence calculation
            confidence = base_confidence
            confidence *= (1.0 - uncertainty_count * 0.15)  # Reduce for uncertainty
            confidence *= (1.0 + confidence_count * 0.1)    # Boost for confidence
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5
    
    def _generate_follow_up_suggestions(self, question: str, answer: str, context: str) -> List[str]:
        """Generate follow-up question suggestions"""
        try:
            prompt = f"""Based on this Q&A, suggest 3 natural follow-up questions that would help explore the topic further:

Question: {question}
Answer: {answer[:300]}...

Generate follow-up questions that:
- Build on the current answer
- Explore related aspects of the content
- Are specific and actionable
- Would likely have answers in the document

Respond with just the 3 questions, one per line, numbered 1-3."""
            
            response = self.llm.invoke(prompt)
            suggestions = []
            
            for line in response.content.split('\n'):
                line = line.strip()
                # Clean up numbering
                line = re.sub(r'^[\d\.\-\•\*]\s*', '', line)
                if line and len(line) > 10 and '?' in line:
                    suggestions.append(line)
            
            return suggestions[:3]
            
        except Exception as e:
            logger.warning(f"Could not generate follow-up suggestions: {str(e)}")
            return [
                "Can you provide more details about this topic?",
                "What are the key implications mentioned?",
                "Are there any examples or case studies provided?"
            ]
    
    def _is_followup_question(self, question: str, conversation_context: str) -> bool:
        if not conversation_context:
            return False
        followup_indicators = ['can you explain more', 'tell me more', 'elaborate', 'expand on', 'what about', 'how about', 'also', 'in addition', 'furthermore', 'you mentioned', 'earlier you said', 'from the previous', 'continue', 'more details', 'specifically', 'for example']
        question_lower = question.lower()
        has_indicators = any(indicator in question_lower for indicator in followup_indicators)
        has_pronouns = bool(re.search(r'\b(this|that|those|these|it|they)\b', question_lower))
        return has_indicators or has_pronouns
        
    def _is_metadata_question(self, question: str) -> bool:
        metadata_keywords = ['how many pages', 'total pages', 'page count', 'number of pages', 'file size', 'document size', 'size of', 'how big', 'when was', 'creation date', 'modified', 'processed', 'author', 'creator', 'title', 'subject', 'extraction method', 'processing method', 'how was processed', 'success rate', 'processing time', 'how long', 'key terms', 'main topics', 'content types', 'chunks', 'sections', 'parts', 'structure', 'technical terms', 'has technical', 'metadata', 'properties', 'information about document', 'document statistics', 'document analysis']
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in metadata_keywords)
        
    def _is_image_question(self, question: str) -> bool:
        image_keywords = ['how many images', 'number of images', 'images found', 'image count', 'pictures', 'photos', 'photographs', 'figures', 'charts', 'diagrams', 'visual', 'graphics', 'illustrations', 'image', 'images on page', 'what images', 'any images', 'show images', 'contains images', 'visual content', 'visual elements', 'graphical', 'charts and graphs']
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in image_keywords)

    def _calculate_enhanced_confidence_score(self, answer: str, avg_relevance: float, is_followup: bool) -> float:
        try:
            base_confidence = avg_relevance
            uncertainty_phrases = ["i don't know", "cannot answer", "not sure", "unclear", "might be", "possibly", "perhaps", "maybe", "could be", "it seems", "appears to", "likely", "probably"]
            answer_lower = answer.lower()
            uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in answer_lower)
            confidence_indicators = ["specifically states", "clearly indicates", "explicitly mentions", "according to page", "the document shows", "definitively", "precisely", "exactly", "specifically on page"]
            confidence_count = sum(1 for indicator in confidence_indicators if indicator in answer_lower)
            confidence = base_confidence
            confidence *= (1.0 - uncertainty_count * 0.1)
            confidence *= (1.0 + confidence_count * 0.1)
            if is_followup:
                confidence *= 1.1
            if len(answer.split()) > 50:
                confidence *= 1.05
            return max(0.0, min(1.0, confidence))
        except Exception:
            return 0.5
            
    def _generate_enhanced_followup_suggestions(self, question: str, answer: str, conversation_context: str, is_metadata_question: bool = False, is_image_question: bool = False, is_followup_question: bool = False) -> List[str]:
        try:
            if len(answer) < 50:
                if is_image_question:
                    return ["What pages contain the most images?", "Can you describe the types of visual content?"]
                elif is_metadata_question:
                    return ["What are the main content types in this document?", "How was this document structured?"]
                elif is_followup_question:
                    return ["Can you provide more specific examples?", "What other related information is available?"]
                else:
                    return ["Can you elaborate on the key points?", "What supporting evidence is provided?"]
            
            context_hint = "document content and our conversation history"
            if is_image_question:
                context_hint = "visual content and image information"
            elif is_metadata_question:
                context_hint = "document properties and metadata"
            elif is_followup_question:
                context_hint = "the ongoing conversation and related topics"
                
            prompt = f"""Based on this Q&A exchange and conversation history, suggest 3 natural follow-up questions that would help the user explore {context_hint} further:
            
Conversation Context: {conversation_context[-500:] if conversation_context else 'None'}
            
Current Q&A:
Question: {question}
Answer: {answer[:400]}...
            
Generate follow-up questions that:
1. Build naturally on this conversation
2. Explore related aspects of the document
3. Are specific and actionable
4. Avoid repeating previous questions
            
Follow-up questions:"""
            
            response = self.llm.invoke(prompt)
            suggestions = []
            for line in response.content.split('\n'):
                line = line.strip()
                line = re.sub(r'^[\d\.\-\•\*]\s*', '', line)
                if line and len(line) > 15 and '?' in line:
                    suggestions.append(line)
            return suggestions[:3] if suggestions else ["What would you like to know more about?"]
        except Exception as e:
            logger.warning(f"Could not generate follow-up suggestions: {str(e)}")
            return ["What would you like to know more about?"]
    
    def generate_sample_questions(self, chunks: List[Document], document_metadata: Dict[str, Any] = None) -> List[str]:
        """Generate sample questions from document content"""
        try:
            # Select diverse chunks for question generation
            selected_chunks = []
            content_types_seen = set()
            
            for chunk in chunks:
                content_type = chunk.metadata.get('content_type', 'content')
                if content_type not in content_types_seen or len(selected_chunks) < 5:
                    selected_chunks.append(chunk)
                    content_types_seen.add(content_type)
                
                if len(selected_chunks) >= 8:  # Limit for prompt size
                    break
            
            # Create context for question generation
            context_parts = []
            for i, chunk in enumerate(selected_chunks):
                page = chunk.metadata.get('page', 'N/A')
                content_type = chunk.metadata.get('content_type', 'content')
                preview = chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content
                context_parts.append(f"[Page {page} - {content_type.title()}]: {preview}")
            
            context_str = "\n\n".join(context_parts)
            
            # Document info for context
            doc_info = ""
            if document_metadata:
                doc_info = f"Document: {document_metadata.get('filename', 'Unknown')}\n"
                doc_info += f"Pages: {document_metadata.get('total_pages', 'N/A')}\n"
                if document_metadata.get('key_terms'):
                    doc_info += f"Key terms: {', '.join(document_metadata['key_terms'][:10])}\n"
            
            prompt = f"""Based on the following document content, generate exactly 5 sample questions that readers might ask about this document. 
            
{doc_info}
            
DOCUMENT CONTENT SAMPLES:
{context_str}
            
Generate questions that:
1. Cover different aspects of the document (definitions, processes, data, conclusions, etc.)
2. Are specific and answerable from the document content
3. Would be genuinely helpful to someone reading this document
4. Vary in complexity (some simple, some more analytical)
5. Are natural and well-formed
        
Respond with exactly 5 questions, numbered 1-5, one per line."""
            
            response = self.llm.invoke(prompt)
            suggestions = []
            
            for line in response.content.split('\n'):
                line = line.strip()
                # Clean up numbering
                line = re.sub(r'^[0-9]\. ?', '', line)  # This is the new cleanup line
                if line and len(line) > 10 and '?' in line:
                    suggestions.append(line)
            
            # Ensure we have exactly 5 questions
            if len(suggestions) < 5:
                # Add some generic questions if needed
                fallback_questions = [
                    "What are the main topics discussed in this document?",
                    "What are the key findings or conclusions?",
                    "What methodology or approach is described?",
                    "What are the most important concepts defined?",
                    "What practical applications or implications are mentioned?"
                ]
                
                suggestions.extend(fallback_questions)
            
            return suggestions[:5]  # Return exactly 5
            
        except Exception as e:
            logger.warning(f"Could not generate sample questions: {str(e)}")
            return [
                "What are the main topics covered in this document?",
                "What are the key findings or conclusions presented?",
                "What specific data or evidence is provided?",
                "What are the practical implications mentioned?",
                "What recommendations or next steps are suggested?"
            ]
    
    def get_document_chunks(self, file_id: str, limit: int = 10) -> List[Document]:
        """Retrieve document chunks"""
        try:
            result = self.qdrant_client.scroll(
                collection_name=self.config.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=file_id)
                        ),
                        models.FieldCondition(
                            key="is_metadata",
                            match=models.MatchValue(value=False)
                        )
                    ]
                ),
                limit=limit
            )
            
            chunks = []
            for point in result[0]:
                metadata = {k: v for k, v in point.payload.items() if k != 'content'}
                chunks.append(Document(
                    page_content=point.payload['content'],
                    metadata=metadata
                ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents"""
        try:
            result = self.qdrant_client.scroll(
                collection_name=self.config.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="is_metadata",
                            match=models.MatchValue(value=True)
                        )
                    ]
                ),
                limit=100
            )
            
            documents = []
            for point in result[0]:
                doc_metadata = point.payload.get('document_metadata', {})
                documents.append({
                    'file_id': point.payload['file_id'],
                    'filename': doc_metadata.get('filename', 'Unknown'),
                    'total_pages': doc_metadata.get('total_pages', 0),
                    'total_words': doc_metadata.get('total_words', 0),
                    'processed_at': doc_metadata.get('processed_at', ''),
                    'content_types': doc_metadata.get('content_types', {}),
                    'key_terms': doc_metadata.get('key_terms', [])[:10]
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def delete_document(self, file_id: str) -> Dict[str, Any]:
        """Delete document and all its chunks"""
        try:
            # Delete all points with this file_id
            self.qdrant_client.delete(
                collection_name=self.config.COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_id",
                                match=models.MatchValue(value=file_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted document {file_id}")
            
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return {'success': False, 'error': str(e)}
import os
import time
import logging
import json
import uuid
import re
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import PromptTemplate

from config import Config

# --- NEW IMPORTS & CLASSES ---
# Helper class for conversation history
class ConversationTurn:
    def __init__(self, question, answer, timestamp, confidence_score, question_type, processing_time, sources_used):
        self.question = question
        self.answer = answer
        self.timestamp = timestamp
        self.confidence_score = confidence_score
        self.question_type = question_type
        self.processing_time = processing_time
        self.sources_used = sources_used

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
            "confidence_score": self.confidence_score,
            "question_type": self.question_type,
            "processing_time": self.processing_time,
            "sources_used": self.sources_used
        }

# Conversation memory manager
class ConversationMemoryManager:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.conversation_history: List[ConversationTurn] = []
        self.query_cache: Dict[str, Any] = {}
        self.session_start = datetime.now()

    def add_turn(self, turn: ConversationTurn):
        self.conversation_history.append(turn)
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history.pop(0)

    def get_context_string(self) -> str:
        context_parts = []
        for turn in self.conversation_history:
            context_parts.append(f"Q: {turn.question}")
            context_parts.append(f"A: {turn.answer}")
        return "\n".join(context_parts)

    def cache_query(self, question: str, result: Any):
        self.query_cache[question.lower()] = result

    def get_cached_result(self, question: str) -> Optional[Any]:
        return self.query_cache.get(question.lower())
    
    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            'total_turns': len(self.conversation_history),
            'cache_size': len(self.query_cache),
        }
    
    def get_related_questions(self, question: str) -> List[str]:
        related = []
        q_words = set(re.findall(r'\b\w+\b', question.lower()))
        for turn in self.conversation_history:
            turn_words = set(re.findall(r'\b\w+\b', turn.question.lower()))
            if len(q_words.intersection(turn_words)) > 2:
                related.append(turn.question)
        return related

# Configure logging
logger = logging.getLogger(__name__)

class RAGService:
    """Service for RAG operations with Qdrant"""
    
    def __init__(self):
        self.config = Config
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=self.config.QDRANT_URL,
            api_key=self.config.QDRANT_API_KEY
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize LLM
        self.llm = ChatMistralAI(
            model=self.config.MISTRAL_MODEL,
            temperature=0.1,
            api_key=self.config.MISTRAL_API_KEY
        )
        
        # Ensure collection exists
        self._ensure_collection()
        
        # --- NEW ATTRIBUTES ---
        self.memory_manager = ConversationMemoryManager()
        self.session_stats = {
            'questions_asked': 0,
            'total_search_time': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'average_confidence': 0.0,
            'cache_hits': 0,
            'session_start': datetime.now()
        }

    def _ensure_collection(self):
        """
        Ensure Qdrant collection exists and has the necessary payload indexes.
        This version is robust and creates indexes even if the collection already exists.
        """
        try:
            collection_name = self.config.COLLECTION_NAME
            
            # Check if the collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                # Create the collection if it doesn't exist
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {collection_name}")

            # --- START: ROBUST INDEX CREATION ---
            # This logic runs every time to ensure indexes are present.
            # Qdrant's create_payload_index is idempotent (safe to run multiple times).

            # Index for 'file_id'
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="file_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True  # Wait for the operation to complete
            )
            logger.info("Ensured payload index exists for 'file_id'.")
            
            # Index for 'is_metadata'
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="is_metadata",
                field_schema=models.PayloadSchemaType.BOOL,
                wait=True # Wait for the operation to complete
            )
            logger.info("Ensured payload index exists for 'is_metadata'.")
            # --- END: ROBUST INDEX CREATION ---

        except Exception as e:
            logger.error(f"Failed to ensure collection and indexes: {str(e)}")
            raise
    
    def store_document_chunks(self, chunks: List[Document], file_id: str, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store document chunks in Qdrant"""
        try:
            logger.info(f"Storing {len(chunks)} chunks for document {file_id}")
            
            points = []
            
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_model.encode(chunk.page_content).tolist()
                
                # Create a valid UUID for the point ID
                point_id = str(uuid.uuid4())
                
                # Prepare payload
                payload = {
                    'content': chunk.page_content,
                    'file_id': file_id,
                    'chunk_id': chunk.metadata.get('chunk_id', f"{file_id}_{i}"),
                    'chunk_index': i,
                    'page': chunk.metadata.get('page', 1),
                    'content_type': chunk.metadata.get('content_type', 'content'),
                    'semantic_score': chunk.metadata.get('semantic_score', 0.5),
                    'key_terms': chunk.metadata.get('key_terms', ''),
                    'word_count': chunk.metadata.get('word_count', len(chunk.page_content.split())),
                    'filename': document_metadata.get('filename', ''),
                    'is_metadata': False,
                    'created_at': datetime.now().isoformat()
                }
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.config.COLLECTION_NAME,
                    points=batch
                )
            
            # Store document metadata
            self._store_document_metadata(file_id, document_metadata)
            
            logger.info(f"Successfully stored {len(chunks)} chunks")
            
            return {
                'success': True,
                'chunks_count': len(chunks),
                'file_id': file_id
            }
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _store_document_metadata(self, file_id: str, metadata: Dict[str, Any]):
        """Store document metadata as a special point"""
        try:
            # Create a dummy embedding for metadata (not used for search)
            dummy_embedding = [0.0] * self.embedding_dim
            
            # Create a valid UUID for metadata point
            metadata_id = str(uuid.uuid4())
            
            payload = {
                'content': f"METADATA for {metadata.get('filename', file_id)}",
                'file_id': file_id,
                'is_metadata': True,
                'document_metadata': metadata,
                'created_at': datetime.now().isoformat()
            }
            
            self.qdrant_client.upsert(
                collection_name=self.config.COLLECTION_NAME,
                points=[models.PointStruct(
                    id=metadata_id,
                    vector=dummy_embedding,
                    payload=payload
                )]
            )
            
        except Exception as e:
            logger.warning(f"Failed to store document metadata: {str(e)}")
    
    def document_exists(self, file_id: str) -> bool:
        """Check if document exists in vector store"""
        try:
            logger.info(f"Checking existence of document with file_id: {file_id}")
            result = self.qdrant_client.scroll(
                collection_name=self.config.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=file_id)
                        ),
                        models.FieldCondition(
                            key="is_metadata",
                            match=models.MatchValue(value=False)
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="is_metadata",
                            match=models.MatchValue(value=True)
                        )
                    ]
                ),
                limit=1
            )
            return len(result[0]) > 0
        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            # Fallback: try without metadata filter
            try:
                result = self.qdrant_client.scroll(
                    collection_name=self.config.COLLECTION_NAME,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_id",
                                match=models.MatchValue(value=file_id)
                            )
                        ]
                    ),
                    limit=1
                )
                return len(result[0]) > 0
            except:
                return False
            
    def _classify_question_type(self, question: str) -> str:
        """Classify question type"""
        question_lower = question.lower()
        
        # Check for general/overview questions first
        if any(word in question_lower for word in ['about', 'overview', 'summarize', 'summary', 'main topic', 'what is this document']):
            return 'general'
        elif any(word in question_lower for word in ['what', 'define', 'meaning', 'explain']):
            return 'definition'
        elif any(word in question_lower for word in ['how many', 'count', 'number']):
            return 'quantitative'
        elif any(word in question_lower for word in ['how', 'method', 'process', 'way']):
            return 'procedural'
        elif any(word in question_lower for word in ['why', 'reason', 'cause']):
            return 'causal'
        elif any(word in question_lower for word in ['when', 'time', 'date']):
            return 'temporal'
        elif any(word in question_lower for word in ['compare', 'difference', 'similar', 'versus']):
            return 'comparative'
        else:
            return 'general'
        
    def answer_question(self, file_id: str, question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Answer question using RAG"""
        try:
            start_time = time.time()
            self.session_stats['questions_asked'] += 1

            # Check cache first
            cached_result = self.memory_manager.get_cached_result(question)
            if cached_result:
                self.session_stats['cache_hits'] += 1
                return cached_result

            # Get conversation context
            conversation_context = self.memory_manager.get_context_string()

            # Classify question type
            question_type = self._classify_question_type(question)
            is_followup_question = self._is_followup_question(question, conversation_context)

            # Adjust search parameters based on question type
            if question_type == 'general' or any(word in question.lower() for word in ['about', 'overview', 'summarize', 'summary']):
                limit = 12
                score_threshold = 0.1
            else:
                limit = 8
                score_threshold = 0.3

            # Search for relevant chunks
            question_embedding = self.embedding_model.encode(question).tolist()
            search_results = self.qdrant_client.search(
                collection_name=self.config.COLLECTION_NAME,
                query_vector=question_embedding,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))]
                ),
                limit=limit,
                score_threshold=score_threshold
            )
            content_results = [result for result in search_results if not result.payload.get('is_metadata', False)]
            
            # Fallback search if no results with threshold
            if not content_results and (question_type == 'general' or any(word in question.lower() for word in ['about', 'overview', 'summarize', 'summary'])):
                search_results = self.qdrant_client.search(
                    collection_name=self.config.COLLECTION_NAME,
                    query_vector=question_embedding,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_id",
                                match=models.MatchValue(value=file_id)
                            )
                        ]
                    ),
                    limit=limit
                )
                content_results = [result for result in search_results if not result.payload.get('is_metadata', False)]
            
            if not content_results:
                return {'success': False, 'error': 'No relevant content found for the question'}

            context_str = "\n\n".join([result.payload['content'] for result in content_results])
            sources = [{
                'page': result.payload.get('page', 'N/A'),
                'content_preview': result.payload['content'][:200] + "..." if len(result.payload['content']) > 200 else result.payload['content'],
                'relevance_score': float(result.score),
                'content_type': result.payload.get('content_type', 'content'),
                'semantic_score': result.payload.get('semantic_score', 0.5)
            } for result in content_results]

            # --- MODIFIED: Prompt with JSON schema ---
            prompt = self._create_structured_prompt(question, context_str, conversation_context)
            
            # --- MODIFIED: LLM call with JSON mode and parsing ---
            try:
                response = self.llm.invoke(prompt, response_format={"type": "json_object"})
                answer_content = json.loads(response.content)
                answer = answer_content.get('answer', 'No answer found.')
                formatted_sources = answer_content.get('sources', [])
                # Reformat sources to match the expected output structure
                parsed_sources = []
                for source_page in formatted_sources:
                    for s in sources:
                        if str(s['page']) == str(source_page):
                            parsed_sources.append(s)
                            break
            except json.JSONDecodeError:
                # Fallback to plain text if JSON parsing fails
                answer = response.content
                parsed_sources = sources
            except Exception as e:
                # Catch other LLM or API errors
                logger.error(f"LLM call or JSON parsing failed: {str(e)}")
                answer = "I'm sorry, I couldn't process the request correctly. Please try again."
                parsed_sources = sources

            # Use the new enhanced confidence score
            avg_relevance = np.mean([s['relevance_score'] for s in parsed_sources]) if parsed_sources else 0.0
            confidence_score = self._calculate_enhanced_confidence_score(answer, avg_relevance, is_followup_question)

            follow_up_suggestions = self._generate_enhanced_followup_suggestions(
                question, answer, conversation_context, self._is_metadata_question(question), self._is_image_question(question), is_followup_question
            )

            processing_time = time.time() - start_time
            self.session_stats['total_search_time'] += processing_time
            self.session_stats['successful_searches'] += 1
            if self.session_stats['successful_searches'] > 0:
                total_confidence = (self.session_stats['average_confidence'] * (self.session_stats['successful_searches'] - 1) + confidence_score)
                self.session_stats['average_confidence'] = total_confidence / self.session_stats['successful_searches']
            
            # Determine final question type for memory
            final_question_type = 'followup' if is_followup_question else question_type
            
            # Add to conversation memory
            conversation_turn = ConversationTurn(
                question=question,
                answer=answer,
                timestamp=datetime.now(),
                confidence_score=confidence_score,
                question_type=final_question_type,
                processing_time=processing_time,
                sources_used=[s['page'] for s in parsed_sources[:3]]
            )
            self.memory_manager.add_turn(conversation_turn)
            self.memory_manager.cache_query(question, {
                'success': True,
                'answer': answer,
                'confidence_score': confidence_score,
                'sources': parsed_sources,
                'follow_up_suggestions': follow_up_suggestions,
                'processing_time': processing_time,
                'question_type': final_question_type,
                'context_chunks_used': len(content_results)
            })

            return {
                'success': True,
                'answer': answer,
                'confidence_score': confidence_score,
                'sources': parsed_sources,
                'follow_up_suggestions': follow_up_suggestions,
                'processing_time': processing_time,
                'question_type': final_question_type,
                'context_chunks_used': len(content_results)
            }
        
        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            self.session_stats['failed_searches'] += 1
            return {'success': False, 'error': str(e)}
        
    def _create_structured_prompt(self, question: str, context: str, conversation_history: str) -> str:
        """Create a prompt that instructs the LLM to return a structured JSON object."""
        
        system_instructions = """You are an expert document analyst. Your task is to provide a complete, well-structured answer to the user's question based ONLY on the provided document context.

    Your response must be a JSON object with the following keys:
    - "answer": A string containing the comprehensive answer to the question.
    - "sources": An array of page numbers (as strings) from which the information was sourced.

    Guidelines for the "answer" field:
    - Provide a comprehensive, well-structured answer based on the context.
    - If the information is not in the context, state this clearly in the answer.
    - Do not include any extra commentary or introductory text outside of the JSON.
    - Maintain consistency with the conversation history when relevant."""

        conversation_section = f"\n\nCONVERSATION HISTORY:\n{conversation_history}\n" if conversation_history else ""
        
        prompt = f"""{system_instructions}
        
    {conversation_section}

    CONTEXT FROM DOCUMENT:
    {context}

    QUESTION: {question}

    PROVIDE A JSON RESPONSE:"""
        return prompt
    
    def _create_rag_prompt(self, question: str, context: str, conversation_history: str, question_type: str) -> str:
        """Create RAG prompt based on question type"""
        
        base_instructions = """You are an expert document analyst. Answer the question based on the provided context from the document.

    Guidelines:
    - Provide comprehensive, well-structured answers based on the context
    - Include specific details, examples, and data points when available
    - For technical questions, explain concepts clearly
    - Use bullet points or numbered lists when appropriate for clarity
    - If information is not in the context, state this clearly
    - Always cite relevant information from the context
    - Maintain consistency with conversation history when relevant"""

        type_specific_instructions = {
            'definition': "Provide a clear, comprehensive definition with examples and context from the document.",
            'quantitative': "List all specific numbers, quantities, metrics, and measurements mentioned in the context.",
            'procedural': "Provide a detailed step-by-step explanation of the process or methodology described.",
            'causal': "Explain the complete cause-and-effect relationships, reasons, and underlying factors mentioned.",
            'temporal': "Provide all relevant dates, times, sequences, and temporal relationships from the context.",
            'comparative': "Provide a thorough comparison highlighting all similarities, differences, and relationships described.",
            'general': "Provide a comprehensive overview covering all major topics, key concepts, important details, and conclusions. Structure your response to cover: 1) Main subject/purpose, 2) Key topics and concepts, 3) Important details and findings, 4) Conclusions or implications."
        }
        
        conversation_section = f"\n\nCONVERSATION HISTORY:\n{conversation_history}\n" if conversation_history else ""
        
        prompt = f"""{base_instructions}

    {type_specific_instructions.get(question_type, type_specific_instructions['general'])}{conversation_section}

    CONTEXT FROM DOCUMENT:
    {context}

    QUESTION: {question}

    PROVIDE A COMPLETE, DETAILED ANSWER:"""
        
        return prompt
    
    def _calculate_confidence_score(self, question: str, answer: str, search_results: List) -> float:
        """Calculate confidence score for the answer"""
        try:
            # Base confidence from search relevance
            if not search_results:
                return 0.0
            
            avg_score = np.mean([result.score for result in search_results])
            base_confidence = min(avg_score * 1.2, 1.0)  # Boost slightly
            
            # Adjust based on answer characteristics
            answer_lower = answer.lower()
            
            # Uncertainty indicators
            uncertainty_phrases = [
                "i don't know", "not sure", "unclear", "might be", "possibly", 
                "perhaps", "maybe", "could be", "appears to", "seems"
            ]
            
            uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in answer_lower)
            
            # Confidence indicators
            confidence_phrases = [
                "according to", "the document states", "specifically mentions",
                "clearly indicates", "explicitly", "precisely"
            ]
            
            confidence_count = sum(1 for phrase in confidence_phrases if phrase in answer_lower)
            
            # Final confidence calculation
            confidence = base_confidence
            confidence *= (1.0 - uncertainty_count * 0.15)  # Reduce for uncertainty
            confidence *= (1.0 + confidence_count * 0.1)    # Boost for confidence
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5
    
    def _generate_follow_up_suggestions(self, question: str, answer: str, context: str) -> List[str]:
        """Generate follow-up question suggestions"""
        try:
            prompt = f"""Based on this Q&A, suggest 3 natural follow-up questions that would help explore the topic further:

Question: {question}
Answer: {answer[:300]}...

Generate follow-up questions that:
- Build on the current answer
- Explore related aspects of the content
- Are specific and actionable
- Would likely have answers in the document

Respond with just the 3 questions, one per line, numbered 1-3."""
            
            response = self.llm.invoke(prompt)
            suggestions = []
            
            for line in response.content.split('\n'):
                line = line.strip()
                # Clean up numbering
                line = re.sub(r'^[\d\.\-\•\*]\s*', '', line)
                if line and len(line) > 10 and '?' in line:
                    suggestions.append(line)
            
            return suggestions[:3]
            
        except Exception as e:
            logger.warning(f"Could not generate follow-up suggestions: {str(e)}")
            return [
                "Can you provide more details about this topic?",
                "What are the key implications mentioned?",
                "Are there any examples or case studies provided?"
            ]
    
    def _is_followup_question(self, question: str, conversation_context: str) -> bool:
        if not conversation_context:
            return False
        followup_indicators = ['can you explain more', 'tell me more', 'elaborate', 'expand on', 'what about', 'how about', 'also', 'in addition', 'furthermore', 'you mentioned', 'earlier you said', 'from the previous', 'continue', 'more details', 'specifically', 'for example']
        question_lower = question.lower()
        has_indicators = any(indicator in question_lower for indicator in followup_indicators)
        has_pronouns = bool(re.search(r'\b(this|that|those|these|it|they)\b', question_lower))
        return has_indicators or has_pronouns
        
    def _is_metadata_question(self, question: str) -> bool:
        metadata_keywords = ['how many pages', 'total pages', 'page count', 'number of pages', 'file size', 'document size', 'size of', 'how big', 'when was', 'creation date', 'modified', 'processed', 'author', 'creator', 'title', 'subject', 'extraction method', 'processing method', 'how was processed', 'success rate', 'processing time', 'how long', 'key terms', 'main topics', 'content types', 'chunks', 'sections', 'parts', 'structure', 'technical terms', 'has technical', 'metadata', 'properties', 'information about document', 'document statistics', 'document analysis']
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in metadata_keywords)
        
    def _is_image_question(self, question: str) -> bool:
        image_keywords = ['how many images', 'number of images', 'images found', 'image count', 'pictures', 'photos', 'photographs', 'figures', 'charts', 'diagrams', 'visual', 'graphics', 'illustrations', 'image', 'images on page', 'what images', 'any images', 'show images', 'contains images', 'visual content', 'visual elements', 'graphical', 'charts and graphs']
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in image_keywords)

    def _calculate_enhanced_confidence_score(self, answer: str, avg_relevance: float, is_followup: bool) -> float:
        try:
            base_confidence = avg_relevance
            uncertainty_phrases = ["i don't know", "cannot answer", "not sure", "unclear", "might be", "possibly", "perhaps", "maybe", "could be", "it seems", "appears to", "likely", "probably"]
            answer_lower = answer.lower()
            uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in answer_lower)
            confidence_indicators = ["specifically states", "clearly indicates", "explicitly mentions", "according to page", "the document shows", "definitively", "precisely", "exactly", "specifically on page"]
            confidence_count = sum(1 for indicator in confidence_indicators if indicator in answer_lower)
            confidence = base_confidence
            confidence *= (1.0 - uncertainty_count * 0.1)
            confidence *= (1.0 + confidence_count * 0.1)
            if is_followup:
                confidence *= 1.1
            if len(answer.split()) > 50:
                confidence *= 1.05
            return max(0.0, min(1.0, confidence))
        except Exception:
            return 0.5
            
    def _generate_enhanced_followup_suggestions(self, question: str, answer: str, conversation_context: str, is_metadata_question: bool = False, is_image_question: bool = False, is_followup_question: bool = False) -> List[str]:
        try:
            if len(answer) < 50:
                if is_image_question:
                    return ["What pages contain the most images?", "Can you describe the types of visual content?"]
                elif is_metadata_question:
                    return ["What are the main content types in this document?", "How was this document structured?"]
                elif is_followup_question:
                    return ["Can you provide more specific examples?", "What other related information is available?"]
                else:
                    return ["Can you elaborate on the key points?", "What supporting evidence is provided?"]
            
            context_hint = "document content and our conversation history"
            if is_image_question:
                context_hint = "visual content and image information"
            elif is_metadata_question:
                context_hint = "document properties and metadata"
            elif is_followup_question:
                context_hint = "the ongoing conversation and related topics"
                
            prompt = f"""Based on this Q&A exchange and conversation history, suggest 3 natural follow-up questions that would help the user explore {context_hint} further:
            
Conversation Context: {conversation_context[-500:] if conversation_context else 'None'}
            
Current Q&A:
Question: {question}
Answer: {answer[:400]}...
            
Generate follow-up questions that:
1. Build naturally on this conversation
2. Explore related aspects of the document
3. Are specific and actionable
4. Avoid repeating previous questions
            
Follow-up questions:"""
            
            response = self.llm.invoke(prompt)
            suggestions = []
            for line in response.content.split('\n'):
                line = line.strip()
                line = re.sub(r'^[\d\.\-\•\*]\s*', '', line)
                if line and len(line) > 15 and '?' in line:
                    suggestions.append(line)
            return suggestions[:3] if suggestions else ["What would you like to know more about?"]
        except Exception as e:
            logger.warning(f"Could not generate follow-up suggestions: {str(e)}")
            return ["What would you like to know more about?"]
    
    def generate_sample_questions(self, chunks: List[Document], document_metadata: Dict[str, Any] = None) -> List[str]:
        """Generate sample questions from document content"""
        try:
            # Select diverse chunks for question generation
            selected_chunks = []
            content_types_seen = set()
            
            for chunk in chunks:
                content_type = chunk.metadata.get('content_type', 'content')
                if content_type not in content_types_seen or len(selected_chunks) < 5:
                    selected_chunks.append(chunk)
                    content_types_seen.add(content_type)
                
                if len(selected_chunks) >= 8:  # Limit for prompt size
                    break
            
            # Create context for question generation
            context_parts = []
            for i, chunk in enumerate(selected_chunks):
                page = chunk.metadata.get('page', 'N/A')
                content_type = chunk.metadata.get('content_type', 'content')
                preview = chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content
                context_parts.append(f"[Page {page} - {content_type.title()}]: {preview}")
            
            context_str = "\n\n".join(context_parts)
            
            # Document info for context
            doc_info = ""
            if document_metadata:
                doc_info = f"Document: {document_metadata.get('filename', 'Unknown')}\n"
                doc_info += f"Pages: {document_metadata.get('total_pages', 'N/A')}\n"
                if document_metadata.get('key_terms'):
                    doc_info += f"Key terms: {', '.join(document_metadata['key_terms'][:10])}\n"
            
            prompt = f"""Based on the following document content, generate exactly 5 sample questions that readers might ask about this document. 
            
{doc_info}
            
DOCUMENT CONTENT SAMPLES:
{context_str}
            
Generate questions that:
1. Cover different aspects of the document (definitions, processes, data, conclusions, etc.)
2. Are specific and answerable from the document content
3. Would be genuinely helpful to someone reading this document
4. Vary in complexity (some simple, some more analytical)
5. Are natural and well-formed
        
Respond with exactly 5 questions, numbered 1-5, one per line."""
            
            response = self.llm.invoke(prompt)
            suggestions = []
            
            for line in response.content.split('\n'):
                line = line.strip()
                # Clean up numbering
                line = re.sub(r'^[0-9]\. ?', '', line)  # This is the new cleanup line
                if line and len(line) > 10 and '?' in line:
                    suggestions.append(line)
            
            # Ensure we have exactly 5 questions
            if len(suggestions) < 5:
                # Add some generic questions if needed
                fallback_questions = [
                    "What are the main topics discussed in this document?",
                    "What are the key findings or conclusions?",
                    "What methodology or approach is described?",
                    "What are the most important concepts defined?",
                    "What practical applications or implications are mentioned?"
                ]
                
                suggestions.extend(fallback_questions)
            
            return suggestions[:5]  # Return exactly 5
            
        except Exception as e:
            logger.warning(f"Could not generate sample questions: {str(e)}")
            return [
                "What are the main topics covered in this document?",
                "What are the key findings or conclusions presented?",
                "What specific data or evidence is provided?",
                "What are the practical implications mentioned?",
                "What recommendations or next steps are suggested?"
            ]
    
    def get_document_chunks(self, file_id: str, limit: int = 10) -> List[Document]:
        """Retrieve document chunks"""
        try:
            result = self.qdrant_client.scroll(
                collection_name=self.config.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=file_id)
                        ),
                        models.FieldCondition(
                            key="is_metadata",
                            match=models.MatchValue(value=False)
                        )
                    ]
                ),
                limit=limit
            )
            
            chunks = []
            for point in result[0]:
                metadata = {k: v for k, v in point.payload.items() if k != 'content'}
                chunks.append(Document(
                    page_content=point.payload['content'],
                    metadata=metadata
                ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents"""
        try:
            result = self.qdrant_client.scroll(
                collection_name=self.config.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="is_metadata",
                            match=models.MatchValue(value=True)
                        )
                    ]
                ),
                limit=100
            )
            
            documents = []
            for point in result[0]:
                doc_metadata = point.payload.get('document_metadata', {})
                documents.append({
                    'file_id': point.payload['file_id'],
                    'filename': doc_metadata.get('filename', 'Unknown'),
                    'total_pages': doc_metadata.get('total_pages', 0),
                    'total_words': doc_metadata.get('total_words', 0),
                    'processed_at': doc_metadata.get('processed_at', ''),
                    'content_types': doc_metadata.get('content_types', {}),
                    'key_terms': doc_metadata.get('key_terms', [])[:10]
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def delete_document(self, file_id: str) -> Dict[str, Any]:
        """Delete document and all its chunks"""
        try:
            # Delete all points with this file_id
            self.qdrant_client.delete(
                collection_name=self.config.COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_id",
                                match=models.MatchValue(value=file_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted document {file_id}")
            
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return {'success': False, 'error': str(e)}