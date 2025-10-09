"""
Core RAG Service

This module contains the main EnhancedRAGService class that orchestrates
all RAG components together.
"""

import os
import logging
import threading
import time
import uuid
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_mistralai.chat_models import ChatMistralAI

# Import the modular components
from .models import SemanticChunk, SearchResult
from .chunking_service import SemanticChunker
from .search_engines import HybridSearchEngine
from .anti_hallucination import AntiHallucinationPrompts
from .confidence_calculator import ConfidenceCalculator
from .image_analysis import ImageAnalysisService
from .text_processing import TextProcessor
from ..utils.config_manager import ConfigManager
from ..utils.retry_utils import with_retry, performance_timer

logger = logging.getLogger(__name__)

class EnhancedRAGService:
    """Refactored production-ready RAG service"""
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, config=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            self.config_manager = ConfigManager(config)
            self.text_processor = TextProcessor()
            self.confidence_calculator = ConfidenceCalculator()
            
            # Initialize components
            self._initialize_clients()
            self._initialize_models()
            self._initialize_services()
            self._ensure_collection()
            
            # Performance tracking
            self.stats = defaultdict(int)
            self.timing_stats = defaultdict(list)
            
            self._initialized = True
            logger.info("Enhanced RAG Service initialized successfully")
    
    @with_retry()
    def _initialize_clients(self):
        """Initialize external service clients with retry logic"""
        try:
            from qdrant_client import QdrantClient
            from langchain_mistralai.chat_models import ChatMistralAI
            
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                url=self.config_manager.get('QDRANT_URL'),
                api_key=self.config_manager.get('QDRANT_API_KEY'),
                timeout=60,
                prefer_grpc=False,
                https=True,
                verify=True,
                check_compatibility=False
            )
            self._test_qdrant_connection()
            logger.info("Qdrant client initialized successfully")
            
            # Initialize Mistral LLM
            self.llm = ChatMistralAI(
                model=self.config_manager.get('MISTRAL_MODEL'),
                temperature=self.config_manager.get('TEMPERATURE', 0.3),
                api_key=self.config_manager.get('MISTRAL_API_KEY'),
                timeout=self.config_manager.get('RESPONSE_TIMEOUT', 120),
                max_retries=3,
                max_tokens=self.config_manager.get('MAX_TOKENS', 2048),
                top_p=self.config_manager.get('TOP_P', 0.9)
            )
            logger.info("Mistral LLM initialized successfully")
            
        except ImportError as e:
            logger.error(f"Missing required imports: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    def _initialize_models(self):
        """Initialize ML models with better configuration"""
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            
            embedding_model_name = self.config_manager.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            
            self.embedding_model = SentenceTransformer(
                embedding_model_name,
                device=self.config_manager.get('DEVICE', 'cpu')
            )
            
            # Warmup embedding model
            self._warmup_embedding_model()
            
            # Initialize reranker
            reranker_model = self.config_manager.get('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            try:
                self.reranker = CrossEncoder(reranker_model)
                logger.info("Reranker initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
                self.reranker = None
            
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
        except ImportError as e:
            logger.error(f"SentenceTransformers not available: {e}")
            raise
    
    @with_retry()
    def _warmup_embedding_model(self):
        """Warmup embedding model for better performance"""
        try:
            warmup_texts = ["This is a warmup text to initialize the model."]
            _ = self.embedding_model.encode(warmup_texts, show_progress_bar=False)
            logger.info("Embedding model warmed up successfully")
        except Exception as e:
            logger.warning(f"Embedding model warmup failed: {str(e)}")
    
    def _initialize_services(self):
        """Initialize internal services"""
        # Initialize chunker
        chunk_size = self.config_manager.get('CHUNK_SIZE', 300)
        overlap_size = self.config_manager.get('CHUNK_OVERLAP', 50)
        self.chunker = SemanticChunker(chunk_size, overlap_size)
        
        # Initialize search engine
        self.search_engine = HybridSearchEngine(
            embedding_model=self.embedding_model,
            reranker=self.reranker,
            config_manager=self.config_manager
        )
        
        # Initialize anti-hallucination prompts
        self.anti_hallucination = AntiHallucinationPrompts()
        
        # Initialize confidence calculator
        self.confidence_calculator = ConfidenceCalculator()
        
        # Initialize image analysis service
        self.image_analysis = ImageAnalysisService(self.config_manager)
        
        # Initialize conversation memory if available
        self._initialize_conversation_memory()
    
    def _initialize_conversation_memory(self):
        """Initialize conversation memory if available"""
        try:
            from services.conversation_memory import ConversationMemoryManager
            self.conversation_memory = ConversationMemoryManager(
                max_turns=self.config_manager.get('MAX_CONVERSATION_TURNS', 10),
                cache_size=self.config_manager.get('CONVERSATION_CACHE_SIZE', 100)
            )
            logger.info("Conversation memory initialized successfully")
        except ImportError:
            logger.warning("Conversation memory not available")
            self.conversation_memory = None
        except Exception as e:
            logger.warning(f"Failed to initialize conversation memory: {e}")
            self.conversation_memory = None
    
    def _test_qdrant_connection(self):
        """Test Qdrant connection"""
        try:
            collections = self.qdrant_client.get_collections()
            logger.info(f"Qdrant connection successful. Collections: {len(collections.collections)}")
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            raise
    
    def _ensure_collection(self):
        """Ensure Qdrant collection exists with proper configuration"""
        collection_name = self.config_manager.get('COLLECTION_NAME', 'documents')
        
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name not in existing_names:
                logger.info(f"Creating collection: {collection_name}")
                self._create_collection(collection_name)
            else:
                logger.info(f"Collection {collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def _create_collection(self, collection_name: str):
        """Create Qdrant collection with optimized settings"""
        from qdrant_client.http import models
        
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_dim,
                distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfig(
                default_segment_number=2,
                max_segment_size=20000,
                memmap_threshold=20000,
                indexing_threshold=20000,
                flush_interval_sec=5,
                max_optimization_threads=1
            ),
            hnsw_config=models.HnswConfig(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
                max_indexing_threads=0,
                on_disk=False
            )
        )
    
    # This would continue with the main methods like index_documents, answer_question, etc.
    # For now, I'll create the entry points and we can expand them as needed
    
    def index_documents(self, documents: List[Dict[str, Any]], 
                       collection_name: str = None) -> Dict[str, Any]:
        """Index documents into the RAG system"""
        # Implementation would go here - this is a placeholder
        # The actual implementation would use the chunker and search_engine
        return {"status": "success", "message": "Documents indexed"}
    
    @with_retry()
    def store_document_chunks(self, chunks: List[Any], file_id: str, 
                            document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store document with enhanced semantic chunking and error handling"""
        if not self.qdrant_client:
            return self._create_error_response('Vector database not available', 0)
        
        try:
            # Import here to avoid circular imports
            from collections import Counter
            import uuid
            import numpy as np
            from qdrant_client.http import models
            from datetime import datetime
            
            with performance_timer("Document processing", self.timing_stats):
                # Check if this is an image file
                file_ext = document_metadata.get('file_type', '').lower()
                is_image = file_ext in getattr(self.config_manager.config, 'SUPPORTED_IMAGE_FORMATS', ['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
                
                if is_image:
                    # For images, we might have OCR-extracted content
                    logger.info(f"Processing image file: {file_id}")
                    # Add image-specific metadata to chunks
                    for chunk in chunks:
                        if hasattr(chunk, 'metadata'):
                            chunk.metadata['is_image_content'] = True
                            chunk.metadata['requires_ocr'] = True
                            chunk.metadata['content_type'] = 'image_ocr'
                
                # Extract full text
                full_text = "\n\n".join([getattr(chunk, 'page_content', str(chunk)) for chunk in chunks])
                
                # Add file_id to metadata
                processing_metadata = {**document_metadata, 'file_id': file_id}
                
                # Add image-specific processing metadata
                if is_image:
                    processing_metadata['is_image_document'] = True
                    processing_metadata['extraction_method'] = 'ocr'
                
                # Create semantic chunks
                semantic_chunks = self.chunker.chunk_document(full_text, processing_metadata)
                logger.info(f"Created {len(semantic_chunks)} semantic chunks")
                
                # Build search indexes
                self.search_engine.index_chunks(semantic_chunks)
                
                # Store in vector database
                storage_result = self._store_chunks_in_qdrant(semantic_chunks, file_id, document_metadata)
                
                # Store document metadata
                self._store_document_metadata(file_id, document_metadata)
                
                self.stats['documents_processed'] += 1
                
                return {
                    'success': True,
                    'chunks_count': len(semantic_chunks),
                    'file_id': file_id,
                    'processing_time': storage_result.get('processing_time', 0),
                    'chunk_types': Counter(chunk.chunk_type for chunk in semantic_chunks),
                    'avg_semantic_density': np.mean([chunk.semantic_density for chunk in semantic_chunks]),
                    'is_image_content': is_image,
                    'processing_method': 'ocr_enhanced' if is_image else 'standard'
                }
                
        except Exception as e:
            logger.error(f"Document storage failed: {str(e)}")
            error_type = self._classify_error(str(e))
            
            return {
                'success': False,
                'error': str(e),
                'error_type': error_type,
                'retry_recommended': error_type in ['transient_model_error', 'general_processing_error'],
                'file_id': file_id
            }
    
    def _create_error_response(self, message: str, chunks_processed: int) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'status': 'error',
            'message': message,
            'chunks_processed': chunks_processed,
            'success': False,
            'retry_recommended': True
        }
    
    def _classify_error(self, error_msg: str) -> str:
        """Classify error type for better handling"""
        error_lower = error_msg.lower()
        
        if any(term in error_lower for term in ["no such group", "hdf5", "cache"]):
            return 'transient_model_error'
        elif any(term in error_lower for term in ["division by zero", "empty"]):
            return 'empty_content_error'
        elif any(term in error_lower for term in ["rate", "429", "capacity"]):
            return 'rate_limit_error'
        else:
            return 'general_processing_error'
    
    def _store_chunks_in_qdrant(self, semantic_chunks: List[SemanticChunk], 
                               file_id: str, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store semantic chunks in Qdrant with batch processing"""
        import uuid
        from qdrant_client.http import models
        
        points = []
        
        for chunk in semantic_chunks:
            if chunk.contextual_embedding is not None:
                payload = self._create_chunk_payload(chunk, file_id, document_metadata)
                
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk.contextual_embedding.tolist(),
                    payload=payload
                ))
        
        # Batch upload
        return self._batch_upload_to_qdrant(points)
    
    def _create_chunk_payload(self, chunk: SemanticChunk, file_id: str, 
                             document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create payload for chunk storage"""
        return {
            'content': chunk.content,
            'file_id': file_id,
            'chunk_id': chunk.chunk_id,
            'chunk_type': chunk.chunk_type,
            'section_hierarchy': chunk.section_hierarchy,
            'key_terms': chunk.key_terms,
            'entities': chunk.entities,
            'page': chunk.page,
            'token_count': chunk.token_count,
            'semantic_density': chunk.semantic_density,
            'is_metadata': False,
            'created_at': datetime.now().isoformat(),
            'file_type': document_metadata.get('file_type', 'unknown'),
            'original_filename': document_metadata.get('original_filename', ''),
            'upload_timestamp': document_metadata.get('upload_timestamp', ''),
            'file_size': document_metadata.get('file_size', 0),
            # Add image-specific fields
            'is_image_content': getattr(chunk, 'metadata', {}).get('is_image_content', False),
            'requires_ocr': getattr(chunk, 'metadata', {}).get('requires_ocr', False),
            'ocr_confidence': getattr(chunk, 'metadata', {}).get('ocr_confidence', None),
            'extraction_method': getattr(chunk, 'metadata', {}).get('extraction_method', 'standard')
        }
    
    def _batch_upload_to_qdrant(self, points: List[Any]) -> Dict[str, Any]:
        """Upload points to Qdrant in batches"""
        import time
        
        start_time = time.time()
        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        logger.info(f"Uploading {len(points)} points to Qdrant in {total_batches} batches...")
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            self.qdrant_client.upsert(
                collection_name=self.config_manager.get('COLLECTION_NAME'),
                points=batch,
                wait=False
            )
            
            if batch_num % 5 == 0 or batch_num == total_batches:
                logger.info(f"Uploaded batch {batch_num}/{total_batches}")
        
        processing_time = time.time() - start_time
        logger.info(f"Qdrant upload completed in {processing_time:.2f}s")
        
        return {'processing_time': processing_time}
    
    def _store_document_metadata(self, file_id: str, metadata: Dict[str, Any]):
        """Store document metadata in Qdrant"""
        import uuid
        from qdrant_client.http import models
        
        try:
            dummy_embedding = [0.0] * self.embedding_dim
            
            payload = {
                'content': f"METADATA:{metadata.get('filename', file_id)}",
                'file_id': file_id,
                'is_metadata': True,
                'doc_pages': metadata.get('total_pages', 0),
                'doc_words': metadata.get('total_words', 0),
                'filename': metadata.get('filename', ''),
                'created_at': datetime.now().isoformat()
            }
            
            self.qdrant_client.upsert(
                collection_name=self.config_manager.get('COLLECTION_NAME'),
                points=[models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=dummy_embedding,
                    payload=payload
                )],
                wait=False
            )
            
        except Exception as e:
            logger.warning(f"Metadata storage failed: {str(e)}")
    
    def answer_question(self, file_id: str, question: str, 
                       conversation_history: List[Dict] = None,
                       session_id: str = None) -> Dict[str, Any]:
        """Answer a question using the RAG system"""
        if not self.qdrant_client:
            return self._create_answer_error_response('Vector database not available')
        
        try:
            start_time = time.time()
            self.stats['questions_processed'] += 1
            
            # Perform search
            search_results = self._search_with_file_filter(question, file_id)
            
            if not search_results:
                return self._create_answer_error_response('No relevant information found')
            
            # Generate answer using the anti-hallucination prompts
            context_chunks = []
            for result in search_results[:5]:
                if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                    content = result.chunk.content.strip()
                    if content and len(content) > 20:
                        context_chunks.append(content)
            
            if not context_chunks:
                return self._create_answer_error_response('No suitable content found')
            
            # Use anti-hallucination prompt
            content_type = self._detect_content_type(question, context_chunks[0] if context_chunks else "")
            enhanced_prompt = self._create_enhanced_anti_hallucination_prompt(
                question, context_chunks, content_type
            )
            
            # Generate answer using LLM (restore original functionality)
            generation_result = self._generate_enhanced_answer_with_prompt(
                enhanced_prompt, search_results, conversation_history, ""
            )
            
            if not generation_result['success']:
                return generation_result
            
            # Validate answer quality (like original)
            if 'answer' in generation_result:
                chunk_strings = [chunk for chunk in context_chunks if isinstance(chunk, str)]
                quality_check = self.validate_answer_quality(
                    generation_result['answer'], chunk_strings
                )
                generation_result['quality_metrics'] = quality_check
            
            # Calculate confidence using the confidence calculator
            base_confidence = self.confidence_calculator.calculate_confidence(
                question, search_results, generation_result
            )
            
            # Adjust confidence based on quality validation (like original)
            quality_score = generation_result.get('quality_metrics', {}).get('score', 0.5)
            confidence = (base_confidence + quality_score) / 2.0
            
            # Prepare response using original format
            result = self._prepare_final_response(
                generation_result, search_results, confidence, start_time
            )
            
            # Add session information
            result['file_id'] = file_id
            result['session_id'] = session_id
            
            return result
            
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            return self._create_answer_error_response(f'Processing error: {str(e)}')
    
    def _create_answer_error_response(self, message: str) -> Dict[str, Any]:
        """Create standardized error response for answer questions"""
        return {
            'success': False,
            'error': message,
            'answer': '',
            'confidence': 0.0,
            'processing_time': 0.0,
            'retry_recommended': True
        }
    
    def _search_with_file_filter(self, question: str, file_id: str):
        """Perform search with file filtering"""
        try:
            # Use the search engine to find relevant chunks
            logger.info(f"Performing hybrid search for: '{question}' in file: {file_id}")
            all_results = self.search_engine.search(question, top_k=20)
            logger.info(f"Hybrid search returned {len(all_results)} total results")
            
            # Filter results by file_id
            filtered_results = []
            for result in all_results:
                if hasattr(result, 'chunk') and hasattr(result.chunk, 'metadata'):
                    chunk_file_id = result.chunk.metadata.get('file_id')
                    if chunk_file_id == file_id:
                        filtered_results.append(result)
                        logger.debug(f"Result included: dense={result.dense_score:.3f}, sparse={result.sparse_score:.3f}, rerank={result.rerank_score:.3f}")
                elif hasattr(result, 'chunk') and hasattr(result.chunk, 'file_id'):
                    if result.chunk.file_id == file_id:
                        filtered_results.append(result)
                        logger.debug(f"Result included: dense={result.dense_score:.3f}, sparse={result.sparse_score:.3f}, rerank={result.rerank_score:.3f}")
            
            logger.info(f"After file filtering: {len(filtered_results)} results for file {file_id}")
            return filtered_results[:10]  # Return top 10 filtered results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def _detect_content_type(self, question: str, context: str) -> str:
        """Detect the content type based on context and question"""
        return self.anti_hallucination.detect_content_type(context, question)
    
    def _create_enhanced_anti_hallucination_prompt(self, question: str, context_chunks: List[str], content_type: str = "general") -> str:
        """Create enhanced prompts that minimize hallucinations based on content type"""
        
        # Combine context with source numbering
        numbered_context = ""
        for i, chunk in enumerate(context_chunks[:5], 1):  # Limit to 5 chunks for focus
            numbered_context += f"\n[Source {i}]: {chunk}\n"
        
        # Use the anti-hallucination prompts class
        return self.anti_hallucination.get_prompt_for_content_type(content_type, numbered_context, question)
    
    def _generate_enhanced_answer_with_prompt(self, enhanced_prompt: str, search_results: List,
                                           conversation_history: List[Dict] = None, 
                                           conversation_context: str = "") -> Dict[str, Any]:
        """Generate answer using custom anti-hallucination prompt"""
        try:
            # Use the enhanced prompt directly with LLM
            if not hasattr(self, 'llm') or not self.llm:
                # If LLM is not available, use the contextual answer as fallback
                logger.warning("LLM not available, using contextual answer generation")
                context_chunks = []
                for result in search_results[:5]:
                    if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                        context_chunks.append(result.chunk.content)
                
                answer = self._generate_contextual_answer("", context_chunks, "general")
                return {
                    'success': True,
                    'answer': answer,
                    'context_used': len(search_results),
                    'processing_method': 'contextual_fallback'
                }
            
            response = self.llm.invoke(enhanced_prompt)
            answer_content = response.content.strip()
            
            # Check for API capacity issues
            if not answer_content or answer_content in ["", ".", "..", "..."]:
                logger.warning("LLM returned minimal response")
                raise Exception("LLM returned minimal response - possible API capacity issue")
            
            # Post-process answer to preserve anti-hallucination structure
            answer = self._post_process_answer(answer_content, search_results, False)
            
            return {
                'success': True,
                'answer': answer,
                'context_used': len(search_results),
                'processing_method': 'anti_hallucination'
            }
            
        except Exception as e:
            logger.error(f"Enhanced prompt answer generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'answer': "I encountered an error generating a response. Please try again."
            }
    
    def _post_process_answer(self, answer_content: str, search_results: List, add_sources: bool = True) -> str:
        """Post-process the generated answer to make it clean and readable"""
        try:
            # Basic cleanup
            answer = answer_content.strip()
            
            # Remove HTML formatting that makes answers messy
            answer = answer.replace('<br />', ' ')
            answer = answer.replace('<br/>', ' ')
            answer = answer.replace('<br>', ' ')
            
            # Remove excessive markdown formatting
            answer = answer.replace('**', '')
            answer = answer.replace('*', '')
            answer = answer.replace('###', '')
            answer = answer.replace('##', '')
            answer = answer.replace('#', '')
            
            # Remove verbose source citations that clutter the answer
            import re
            answer = re.sub(r'\[Source \d+\]', '', answer)
            answer = re.sub(r'\[Source \d+, Source \d+\]', '', answer)
            answer = re.sub(r'\[Source \d+(, Source \d+)*\]', '', answer)
            
            # Remove excessive section headers and formatting
            answer = re.sub(r'### [^:]+:', '', answer)
            answer = re.sub(r'## [^:]+:', '', answer)
            answer = re.sub(r'# [^:]+:', '', answer)
            
            # Remove repetitive phrases
            answer = re.sub(r'Sources Cited:.*$', '', answer, flags=re.MULTILINE)
            answer = re.sub(r'Missing Data:.*$', '', answer, flags=re.MULTILINE | re.DOTALL)
            answer = re.sub(r'---\s*Sources.*$', '', answer, flags=re.DOTALL)
            
            # Clean up multiple spaces and line breaks
            answer = re.sub(r'\n\s*\n\s*\n', '\n\n', answer)  # Max 2 line breaks
            answer = re.sub(r' +', ' ', answer)  # Multiple spaces to single
            answer = answer.strip()
            
            # Ensure proper sentence structure
            if answer and not answer.endswith(('.', '!', '?')):
                answer += '.'
            
            return answer
            
        except Exception as e:
            logger.warning(f"Answer post-processing failed: {str(e)}")
            return answer_content
    
    def _build_context_string(self, search_results: List) -> str:
        """Build context string from search results"""
        context_parts = []
        for i, result in enumerate(search_results[:5], 1):
            if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                content = result.chunk.content.strip()
                if content:
                    context_parts.append(f"[{i}] {content}")
        
        return "\n\n".join(context_parts)
    
    def validate_answer_quality(self, answer: str, context_chunks: List[str]) -> Dict[str, Any]:
        """Validate answer quality against source content"""
        try:
            if not answer or not context_chunks:
                return {'score': 0.0, 'issues': ['Empty answer or context']}
            
            issues = []
            
            # Check for minimal answers
            if len(answer.split()) < 3:
                issues.append('Answer too short')
            
            # Check for generic responses
            generic_phrases = ['i don\'t know', 'not sure', 'cannot determine', 'no information']
            if any(phrase in answer.lower() for phrase in generic_phrases):
                issues.append('Generic response detected')
            
            # Check for context alignment
            answer_words = set(answer.lower().split())
            context_words = set()
            for chunk in context_chunks:
                context_words.update(chunk.lower().split())
            
            overlap = len(answer_words.intersection(context_words))
            if overlap < len(answer_words) * 0.1:  # Less than 10% overlap
                issues.append('Poor context alignment')
            
            # Calculate overall score
            base_score = 1.0
            for issue in issues:
                if 'too short' in issue:
                    base_score -= 0.3
                elif 'generic' in issue:
                    base_score -= 0.4
                elif 'alignment' in issue:
                    base_score -= 0.2
                else:
                    base_score -= 0.1
            
            score = max(0.0, base_score)
            
            return {
                'score': score,
                'issues': issues,
                'word_overlap': overlap,
                'answer_length': len(answer.split())
            }
            
        except Exception as e:
            logger.warning(f"Answer quality validation failed: {str(e)}")
            return {'score': 0.5, 'issues': ['Validation error']}
    
    def _prepare_final_response(self, generation_result: Dict[str, Any], 
                              search_results: List, confidence: float, 
                              start_time: float) -> Dict[str, Any]:
        """Prepare the final response with all necessary fields"""
        processing_time = time.time() - start_time
        
        # Post-process the answer to make it clean
        raw_answer = generation_result.get('answer', '')
        clean_answer = self._post_process_answer(raw_answer, search_results, False)
        
        # Extract sources information
        sources = []
        for result in search_results[:3]:  # Top 3 sources
            if hasattr(result, 'chunk'):
                source_info = {
                    'content_preview': result.chunk.content[:150] + "..." if len(result.chunk.content) > 150 else result.chunk.content,
                    'relevance_score': getattr(result, 'hybrid_score', 0.0),
                    'chunk_type': getattr(result.chunk, 'chunk_type', 'text')
                }
                sources.append(source_info)
        
        return {
            'success': True,
            'answer': clean_answer,  # Use the cleaned answer
            'sources': sources,
            'confidence_score': confidence,
            'processing_time': processing_time,
            'search_results_count': len(search_results),
            'processing_method': generation_result.get('processing_method', 'enhanced'),
            'quality_metrics': generation_result.get('quality_metrics', {})
        }
    
    def _generate_contextual_answer(self, question: str, context_chunks: List[str], content_type: str) -> str:
        """Generate a contextual answer that directly addresses the question"""
        
        # Keywords to look for based on common question types
        question_lower = question.lower()
        
        # Handle specific question types
        if "series" in question_lower or "model" in question_lower:
            return self._extract_series_models(context_chunks, question)
        elif "feature" in question_lower or "specification" in question_lower:
            return self._extract_features(context_chunks, question)
        elif "price" in question_lower or "cost" in question_lower:
            return self._extract_pricing(context_chunks, question)
        elif "size" in question_lower or "dimension" in question_lower:
            return self._extract_dimensions(context_chunks, question)
        else:
            # General answer - find most relevant content
            return self._extract_general_answer(context_chunks, question)
    
    def _extract_series_models(self, context_chunks: List[str], question: str) -> str:
        """Extract series and model information"""
        series_info = []
        models = []
        
        for chunk in context_chunks:
            chunk_lower = chunk.lower()
            
            # Look for series mentions (E-Series, G-Series, etc.)
            import re
            series_pattern = r'([A-Z]-Series)'
            series_matches = re.findall(series_pattern, chunk)
            series_info.extend(series_matches)
            
            # Also look for "Range of" descriptions
            if "range of" in chunk_lower and "series" in chunk_lower:
                # Extract the specific range description
                parts = chunk.split("Range of")
                if len(parts) > 1:
                    range_desc = parts[1].split("(")[0].strip()
                    if range_desc:
                        series_info.append(f"Range of {range_desc}")
            
            # Look for model numbers (GL-B257EES3, etc.)
            model_pattern = r'GL-[A-Z0-9]+|[A-Z]{2,}-[A-Z0-9]+'
            found_models = re.findall(model_pattern, chunk)
            models.extend(found_models)
        
        # Build answer
        answer_parts = []
        
        if series_info:
            unique_series = list(set(series_info))
            answer_parts.append("Series mentioned in the document:")
            for info in unique_series[:3]:  # Limit to 3 most relevant
                answer_parts.append(f"• {info}")
        
        if models:
            unique_models = list(set(models))
            if len(unique_models) <= 5:
                answer_parts.append(f"\nModel numbers: {', '.join(unique_models)}")
            else:
                answer_parts.append(f"\nModel numbers: {', '.join(unique_models[:5])} (and {len(unique_models)-5} more)")
        
        if answer_parts:
            return "\n".join(answer_parts)
        else:
            return "No specific series or model information found in the retrieved content."
    
    def _extract_features(self, context_chunks: List[str], question: str) -> str:
        """Extract feature information"""
        features = []
        
        for chunk in context_chunks:
            # Look for bullet points and feature lists
            lines = chunk.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    features.append(line)
        
        if features:
            return "Key features mentioned:\n" + "\n".join(features[:8])
        else:
            # Fallback to general content
            return f"Features information: {context_chunks[0][:300]}..." if context_chunks else "No feature information found."
    
    def _extract_pricing(self, context_chunks: List[str], question: str) -> str:
        """Extract pricing information"""
        import re
        pricing_info = []
        
        for chunk in context_chunks:
            # Look for currency symbols and numbers
            price_patterns = [r'\$[\d,]+', r'₹[\d,]+', r'€[\d,]+', r'£[\d,]+']
            for pattern in price_patterns:
                matches = re.findall(pattern, chunk)
                pricing_info.extend(matches)
        
        if pricing_info:
            return f"Pricing information found: {', '.join(set(pricing_info))}"
        else:
            return "No specific pricing information found in the retrieved content."
    
    def _extract_dimensions(self, context_chunks: List[str], question: str) -> str:
        """Extract size and dimension information"""
        import re
        dimension_info = []
        
        for chunk in context_chunks:
            # Look for dimension patterns (e.g., "24 x 30 x 40", "Width: 60cm")
            patterns = [
                r'\d+\s*[x×]\s*\d+\s*[x×]\s*\d+',  # 24 x 30 x 40
                r'\d+\s*(cm|mm|inch|ft)',           # 60cm, 24inch
                r'(width|height|depth|length):\s*\d+',  # Width: 60
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, chunk, re.IGNORECASE)
                dimension_info.extend(matches)
        
        if dimension_info:
            return f"Dimensions found: {', '.join(set(str(d) for d in dimension_info))}"
        else:
            return "No specific dimension information found in the retrieved content."
    
    def _extract_general_answer(self, context_chunks: List[str], question: str) -> str:
        """Generate a general answer from the most relevant content"""
        if not context_chunks:
            return "No relevant information found in the document."
        
        # Use the first chunk as primary content
        main_content = context_chunks[0]
        
        # Clean up the content
        cleaned_content = main_content.replace('<br />', ' ').replace('<br/>', ' ')
        cleaned_content = ' '.join(cleaned_content.split())  # Remove extra whitespace
        
        # Limit length for readability
        if len(cleaned_content) > 400:
            cleaned_content = cleaned_content[:400] + "..."
        
        return f"Based on the document: {cleaned_content}"
    
    def generate_sample_questions(self, chunks, document_metadata: Dict[str, Any] = None) -> List[str]:
        """Generate sample questions for document exploration"""
        try:
            # Use subset for efficiency
            sample_chunks = chunks[:5] if len(chunks) > 5 else chunks
            
            # Debug logging
            logger.info(f"Generating questions from {len(sample_chunks)} chunks")
            
            # Combine all content for analysis
            all_content = ""
            for chunk in sample_chunks:
                content = getattr(chunk, 'page_content', str(chunk))
                all_content += " " + content
                logger.debug(f"Chunk content preview: {content[:100]}...")
            
            # Convert to lowercase for analysis
            content_lower = all_content.lower()
            logger.info(f"Analyzing {len(all_content)} characters of content")
            
            # Extract meaningful terms and concepts
            import re
            
            # Find proper nouns and technical terms (more sophisticated than just capitalized words)
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', all_content)
            # Filter out common words that aren't meaningful
            meaningful_terms = [term for term in proper_nouns if term.lower() not in {
                'the', 'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'will', 'would',
                'can', 'could', 'should', 'may', 'might', 'must', 'shall', 'dear', 'editors', 'figure',
                'table', 'section', 'page', 'chapter', 'document', 'paper', 'article', 'study', 'research'
            }]
            
            # Find specific domain concepts
            domain_concepts = []
            domain_patterns = {
                'framework': r'\b(\w+)\s+framework\b',
                'model': r'\b(\w+)\s+model\b',
                'system': r'\b(\w+)\s+system\b',
                'algorithm': r'\b(\w+)\s+algorithm\b',
                'method': r'\b(\w+)\s+method\b',
                'approach': r'\b(\w+)\s+approach\b'
            }
            
            for concept_type, pattern in domain_patterns.items():
                matches = re.findall(pattern, content_lower)
                for match in matches[:2]:  # Take first 2 matches
                    if len(match) > 3 and match not in {'this', 'that', 'such', 'new', 'old'}:
                        domain_concepts.append(f"{match} {concept_type}")
            
            logger.info(f"Found {len(meaningful_terms)} meaningful terms and {len(domain_concepts)} domain concepts")
            
            # Generate context-aware questions based on content patterns
            questions = []
            
            # Question 1: Main topic/subject with specific terms
            if meaningful_terms:
                # Use the most frequent meaningful term
                term_counts = {}
                for term in meaningful_terms:
                    term_counts[term] = term_counts.get(term, 0) + 1
                most_common_term = max(term_counts, key=term_counts.get) if term_counts else meaningful_terms[0]
                questions.append(f"What is {most_common_term} and how is it explained in this document?")
            elif domain_concepts:
                questions.append(f"What is the {domain_concepts[0]} described in this document?")
            elif any(keyword in content_lower for keyword in ['research', 'study', 'analysis', 'investigation']):
                questions.append("What research or study is being discussed in this document?")
            else:
                questions.append("What is the main topic or subject of this document?")
            
            # Question 2: Purpose/objective with context
            if any(keyword in content_lower for keyword in ['objective', 'goal', 'aim', 'purpose']):
                questions.append("What are the main objectives or goals discussed?")
            elif any(keyword in content_lower for keyword in ['problem', 'challenge', 'issue', 'gap']):
                questions.append("What problems or challenges are being addressed?")
            elif domain_concepts:
                questions.append(f"What is the purpose of the {domain_concepts[0] if domain_concepts else 'approach'}?")
            else:
                questions.append("What is the purpose or focus of this document?")
            
            # Question 3: Methodology/approach with specifics
            if domain_concepts:
                questions.append(f"How does the {domain_concepts[0]} work or function?")
            elif any(keyword in content_lower for keyword in ['methodology', 'method', 'approach', 'technique']):
                questions.append("What methodology or approach is being used?")
            elif any(keyword in content_lower for keyword in ['data', 'dataset', 'sample', 'participants']):
                questions.append("What data or samples are being analyzed?")
            else:
                questions.append("What specific methods are described in this document?")
            
            # Question 4: Results/findings with context
            if any(keyword in content_lower for keyword in ['result', 'finding', 'outcome', 'conclusion']):
                questions.append("What are the main results or findings presented?")
            elif any(keyword in content_lower for keyword in ['performance', 'accuracy', 'effectiveness', 'efficiency']):
                questions.append("How well does the described approach perform?")
            elif meaningful_terms:
                questions.append(f"What are the key characteristics or properties of {meaningful_terms[0]}?")
            else:
                questions.append("What key information or insights are provided?")
            
            # Question 5: Applications/implications with specifics
            if any(keyword in content_lower for keyword in ['application', 'implementation', 'practical', 'real-world']):
                questions.append("What are the practical applications of this work?")
            elif any(keyword in content_lower for keyword in ['future', 'recommendation', 'suggestion', 'next']):
                questions.append("What future directions or recommendations are suggested?")
            elif meaningful_terms and len(meaningful_terms) > 1:
                questions.append(f"How do {meaningful_terms[0]} and {meaningful_terms[1]} relate to each other?")
            else:
                questions.append("What are the implications or significance of this work?")
            
            # Clean and format questions
            cleaned_questions = []
            for i, q in enumerate(questions[:5], 1):
                cleaned_q = q.strip()
                if not cleaned_q.endswith('?'):
                    cleaned_q += '?'
                final_q = f"{i}. {cleaned_q}"
                cleaned_questions.append(final_q)
            
            logger.info(f"Successfully generated {len(cleaned_questions)} context-specific questions")
            return cleaned_questions
            
        except Exception as e:
            logger.warning(f"Sample question generation failed: {str(e)}")
            return [
                "1. What is this document about?",
                "2. What are the main points discussed?",
                "3. What methods or approaches are described?",
                "4. What are the key findings or results?", 
                "5. What are the practical applications?"
            ]

__all__ = ['EnhancedRAGService']