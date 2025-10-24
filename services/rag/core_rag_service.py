"""
Core RAG Service

This module contains the main EnhancedRAGService class that orchestrates
all RAG components together.
"""

import os
import logging
import re
import threading
import time
import uuid
import hashlib
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from types import SimpleNamespace
from dataclasses import dataclass

from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_mistralai.chat_models import ChatMistralAI

# Import the modular components
from .models import SemanticChunk, SearchResult
from .chunking_service import SemanticChunker
from .search_engines import HybridSearchEngine
from .confidence_calculator import ConfidenceCalculator
from .image_analysis import ImageAnalysisService
from .text_processing import TextProcessor
from ..utils.config_manager import ConfigManager
from ..utils.retry_utils import with_retry, performance_timer

logger = logging.getLogger(__name__)

class ErrorResponse:
    """Standardized error response format"""
    
    @staticmethod
    def create(error_message: str, 
               error_type: str = 'general_error',
               retry_recommended: bool = True,
               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create standardized error response"""
        response = {
            'success': False,
            'error': error_message,
            'error_type': error_type,
            'retry_recommended': retry_recommended,
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            response['context'] = context
        
        return response

@dataclass
@dataclass
class SessionSearchResult:
    """Lightweight search result for session queries"""
    chunk: Any
    dense_score: float
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    combined_score: float = 0.0
    relevance_rank: int = 0
    
    @property
    def hybrid_score(self) -> float:
        """Alias for combined_score to maintain compatibility"""
        return self.combined_score


class EnhancedRAGService:
    """Refactored production-ready RAG service"""
    
    _instance = None
    _lock = threading.Lock()
    
    # Configuration constants - Optimized for comprehensive document retrieval
    MAX_CONTEXT_CHUNKS = 8  # Increased to capture more comprehensive content
    MAX_SEARCH_RESULTS = 15  # Increased for better document coverage
    MIN_CHUNK_LENGTH = 15  # Lowered to include concise but important information
    CONFIDENCE_ADJUSTMENT = 0.85
    ANSWER_CACHE_SIZE = 50
    ANSWER_CACHE_TTL = 300  # 5 minutes

    def __init__(self, config):
        # Check if already initialized
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        with self._lock:
            # Double-check inside lock
            if hasattr(self, '_initialized') and self._initialized:
                return
            
            self.config_manager = ConfigManager(config)
            self.text_processor = TextProcessor()
            self.confidence_calculator = ConfidenceCalculator()
            
            # Initialize components
            self._initialize_clients()
            self._initialize_models()
            self._initialize_services()
            
            # Performance tracking
            self.stats = defaultdict(int)
            self.timing_stats = defaultdict(list)
            
            # Adaptive threshold cache
            self.threshold_cache = {}
            self.cache_lock = threading.Lock() 
            self.cache_max_size = 100
            
            # Answer cache for performance
            self._answer_cache = {}
            
            # Rate limiting for API calls
            self._last_api_call = 0
            self._api_call_interval = 1.0  # Minimum 1 second between calls
            self._rate_limit_lock = threading.Lock()
            
            # Mark as initialized LAST
            self._initialized = True
            logger.info("Enhanced RAG Service initialized successfully (session-based collections)")
    
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
            
            # Initialize Mistral LLM with enhanced rate limiting
            self.llm = ChatMistralAI(
                model=self.config_manager.get('MISTRAL_MODEL'),
                temperature=self.config_manager.get('TEMPERATURE', 0.3),
                api_key=self.config_manager.get('MISTRAL_API_KEY'),
                timeout=self.config_manager.get('RESPONSE_TIMEOUT', 120),
                max_retries=5,  # Increased retries for rate limits
                max_tokens=self.config_manager.get('MAX_TOKENS', 2048),
                top_p=self.config_manager.get('TOP_P', 0.9)
            )
            
            # Initialize rate limiting for API calls
            self._init_rate_limiting()
            logger.info("Mistral LLM initialized successfully with rate limiting")
            
        except ImportError as e:
            logger.error(f"Missing required imports: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    def _init_rate_limiting(self):
        """Initialize rate limiting for API calls"""
        self._api_request_times = []
        self._max_requests_per_minute = 10  # Conservative limit
        self._request_interval = 60.0 / self._max_requests_per_minute  # 6 seconds between requests
    
    def _initialize_models(self):
        """Initialize ML models with better configuration and cache handling"""
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            import torch
            
            embedding_model_name = self.config_manager.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            
            # Clear any corrupted cache before loading
            try:
                torch.hub._clear_cache()
            except:
                pass
            
            # Initialize with robust error handling
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.embedding_model = SentenceTransformer(
                        embedding_model_name,
                        device=self.config_manager.get('DEVICE', 'cpu'),
                        cache_folder=None  # Force fresh download if cache is corrupted
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Model loading attempt {attempt + 1} failed: {e}, retrying...")
                        # Clear cache and try again
                        try:
                            import shutil
                            import os
                            cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
                            if os.path.exists(cache_dir):
                                shutil.rmtree(cache_dir, ignore_errors=True)
                        except:
                            pass
                    else:
                        raise e
            
            # Warmup embedding model
            self._warmup_embedding_model()
            
            # Initialize reranker with similar error handling
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
    
    def _rate_limited_llm_call(self, prompt, max_retries=3):
        """Make rate-limited LLM API call with exponential backoff"""
        import time
        import random
        
        for attempt in range(max_retries):
            try:
                # Rate limiting check
                with self._rate_limit_lock:
                    current_time = time.time()
                    
                    # Clean old request times (older than 1 minute)
                    self._api_request_times = [t for t in self._api_request_times if current_time - t < 60]
                    
                    # Check if we need to wait
                    if len(self._api_request_times) >= self._max_requests_per_minute:
                        wait_time = 60 - (current_time - self._api_request_times[0])
                        if wait_time > 0:
                            logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                            time.sleep(wait_time)
                    
                    # Add request time and make the call
                    self._api_request_times.append(current_time)
                
                # Make the actual API call
                response = self.llm.invoke(prompt)
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limit errors
                if any(keyword in error_str for keyword in ['rate', 'limit', '429', 'capacity', 'tier']):
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s...")
                    
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} attempts")
                        raise Exception(f"API rate limit exceeded: {e}")
                else:
                    # Non-rate-limit error, re-raise immediately
                    raise e
        
        raise Exception(f"Failed to make API call after {max_retries} attempts")
    
    def _ensure_collection(self, session_id: str, user_id: str):
        """Ensure Qdrant collection exists with proper configuration"""
        collection_name = self._get_collection_name(session_id, user_id)
        
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name not in existing_names:
                logger.info(f"Creating user-session collection: {collection_name}")
                self._create_collection(collection_name)
            else:
                logger.info(f"User-session collection {collection_name} already exists")
                # Ensure indexes exist for existing collections
                self._ensure_payload_indexes(collection_name)
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def _get_collection_name(self, session_id: str, user_id: str) -> str:
        """Generate collection name based on user and session - always uses {user_id}_{session_id} format"""
        if not session_id or not user_id:
            raise ValueError(f"Both user_id and session_id are required for collection naming. Got user_id='{user_id}', session_id='{session_id}'")
        
        # Check for parameter confusion - detect if either parameter contains an underscore
        if '_' in user_id:
            logger.warning(f"WARNING: user_id contains underscore: '{user_id}' - this might be session_text passed by mistake!")
        if '_' in session_id:
            logger.warning(f"WARNING: session_id contains underscore: '{session_id}' - this might be session_text passed by mistake!")
            
        # Always use user and session-specific collection: user123_session456
        # Keep full format for backward compatibility with existing collections
        safe_user_id = self._sanitize_for_collection_name(user_id)
        safe_session_id = self._sanitize_for_collection_name(session_id)
        collection_name = f"{safe_user_id}_{safe_session_id}"
        
        # Debug logging for collection name generation
        logger.info(f"Collection name generation: user_id='{user_id}' -> '{safe_user_id}', session_id='{session_id}' -> '{safe_session_id}', final='{collection_name}' (length: {len(collection_name)})")
        
        return collection_name
    
    def create_file_id(self, user_id: str, session_id: str, filename: str) -> str:
        """Create a consistent file_id that matches the collection naming scheme"""
        # Use the same sanitization as collection names for consistency
        safe_user_id = self._sanitize_for_collection_name(user_id)
        safe_session_id = self._sanitize_for_collection_name(session_id)
        safe_filename = filename  # Keep filename as-is for readability
        
        file_id = f"{safe_user_id}_{safe_session_id}_{safe_filename}"
        
        logger.info(f"Created file_id: '{file_id}' for user_id='{user_id}', session_id='{session_id}', filename='{filename}'")
        return file_id
    
    def _sanitize_for_collection_name(self, name: str) -> str:
        """Sanitize name to be safe for Qdrant collection names - use underscores for consistency"""
        import re
        # Keep alphanumeric and convert special chars to underscores for consistency with file_id format
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', str(name))
        # Remove consecutive underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        safe_name = safe_name.strip('_')
        # Keep full name - no truncation for backward compatibility
        return safe_name
    
    def delete_session_collection(self, session_id: str, user_id: str) -> bool:
        """Delete user-session specific collection"""
        try:
            collection_name = self._get_collection_name(session_id, user_id)
            collections = self.qdrant_client.get_collections()
            existing_names = [col.name for col in collections.collections]

            if collection_name in existing_names:
                self.qdrant_client.delete_collection(collection_name)
                logger.info(f"Deleted user-session collection: {collection_name}")
                return True
            else:
                logger.info(f"User-session collection {collection_name} does not exist")
                return False
        except Exception as e:
            logger.error(f"Failed to delete user-session collection: {e}")
            return False
    
    def delete_documents_by_file_id(self, file_id: str, session_id: str, user_id: str) -> bool:
        """Delete specific documents by file_id from the user-session collection"""
        try:
            from qdrant_client.http import models
            
            collection_name = self._get_collection_name(session_id, user_id)
            collections = self.qdrant_client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name not in existing_names:
                logger.warning(f"Collection {collection_name} does not exist")
                return False
            
            # Delete all points with the matching file_id
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_id",
                                match=models.MatchValue(value=file_id)
                            )
                        ]
                    )
                ),
                wait=True
            )
            
            logger.info(f"Deleted documents with file_id '{file_id}' from collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents with file_id '{file_id}': {e}")
            return False
    
    def _create_collection(self, collection_name: str):
        """Create Qdrant collection with optimized settings and payload indexes"""
        from qdrant_client.http import models
        
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_dim,
                distance=models.Distance.COSINE
            ),
            optimizers_config={
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 2,
                "max_segment_size": 20000,
                "memmap_threshold": 20000,
                "indexing_threshold": 20000,
                "flush_interval_sec": 5,
                "max_optimization_threads": 1
            },
            hnsw_config={
                "m": 16,
                "ef_construct": 100,
                "full_scan_threshold": 10000,
                "max_indexing_threads": 0,
                "on_disk": False
            }
        )
        
        # Create payload indexes for efficient filtering
        try:
            # Index for file_id field (required for individual file deletion)
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="file_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Index for session_id field (for session-based operations)
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="session_id", 
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Index for user_id field (for user-based operations)
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            logger.info(f"Created payload indexes for collection '{collection_name}'")
            
        except Exception as e:
            logger.warning(f"Failed to create payload indexes for collection '{collection_name}': {e}")
            # Collection is still usable, just without optimized filtering
    
    def _ensure_payload_indexes(self, collection_name: str):
        """Ensure payload indexes exist for a collection (for existing collections)"""
        from qdrant_client.http import models
        
        try:
            # Try to create indexes - Qdrant will ignore if they already exist
            index_fields = [
                ("file_id", models.PayloadSchemaType.KEYWORD),
                ("session_id", models.PayloadSchemaType.KEYWORD), 
                ("user_id", models.PayloadSchemaType.KEYWORD)
            ]
            
            for field_name, field_type in index_fields:
                try:
                    self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type
                    )
                    logger.info(f"Ensured index for field '{field_name}' in collection '{collection_name}'")
                except Exception as field_error:
                    # Index might already exist, which is fine
                    logger.debug(f"Index for '{field_name}' in '{collection_name}': {field_error}")
                    
        except Exception as e:
            logger.warning(f"Failed to ensure indexes for collection '{collection_name}': {e}")
    
    def index_documents(self, documents: List[Dict[str, Any]], 
                       collection_name: str = None) -> Dict[str, Any]:
        """Index documents into the RAG system"""
        # Implementation would go here - this is a placeholder
        # The actual implementation would use the chunker and search_engine
        return {"status": "success", "message": "Documents indexed"}
    
    @with_retry()
    def store_document_chunks(self, chunks: List[Any], file_id: str, 
                            document_metadata: Dict[str, Any], 
                            session_id: str, user_id: str) -> Dict[str, Any]:
        """Store document with enhanced semantic chunking and error handling"""
        if not self.qdrant_client:
            return self._create_storage_error_response(
                'Vector database not available',
                context={'file_id': file_id, 'session_id': session_id, 'user_id': user_id}
            )
        
        try:
            # Ensure user-session collection exists
            self._ensure_collection(session_id, user_id)
            
            with performance_timer("Document processing", self.timing_stats):
                # Check if this is an image file
                file_ext = document_metadata.get('file_type', '').lower()
                is_image = file_ext in getattr(self.config_manager.config, 'SUPPORTED_IMAGE_FORMATS', ['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
                
                if is_image:
                    logger.info(f"Processing image file: {file_id}")
                    for chunk in chunks:
                        if hasattr(chunk, 'metadata'):
                            chunk.metadata['is_image_content'] = True
                            chunk.metadata['requires_ocr'] = True
                            chunk.metadata['content_type'] = 'image_ocr'
                
                # Extract full text
                logger.info(f"Extracting text from {len(chunks)} chunks")
                try:
                    full_text = "\n\n".join([getattr(chunk, 'page_content', str(chunk)) for chunk in chunks])
                    logger.info(f"Extracted {len(full_text)} characters of text")
                except Exception as text_error:
                    logger.error(f"Failed to extract text from chunks: {text_error}")
                    raise
                
                # Add file_id to metadata
                processing_metadata = {**document_metadata, 'file_id': file_id}
                
                if is_image:
                    processing_metadata['is_image_document'] = True
                    processing_metadata['extraction_method'] = 'ocr'
                
                # Create semantic chunks
                logger.info(f"Creating semantic chunks with chunker (chunk_size={self.chunker.target_chunk_size})")
                try:
                    semantic_chunks = self.chunker.chunk_document(full_text, processing_metadata)
                    logger.info(f"Created {len(semantic_chunks)} semantic chunks")
                except Exception as chunk_error:
                    logger.error(f"Semantic chunking failed: {chunk_error}")
                    logger.exception("Full chunking error traceback:")
                    raise
                
                # Light memory cleanup for very large documents
                if len(semantic_chunks) > 2000:
                    logger.info(f"Large document ({len(semantic_chunks)} chunks) - performing light memory cleanup")
                    import gc
                    gc.collect()
                
                # Build search indexes
                logger.info("Building search indexes...")
                try:
                    self.search_engine.index_chunks(semantic_chunks)
                    logger.info("Search indexes built successfully")
                except Exception as index_error:
                    logger.error(f"Search indexing failed: {index_error}")
                    logger.exception("Full indexing error traceback:")
                    raise
                
                # Verify chunks have valid embeddings after indexing
                chunks_without_embeddings = [chunk for chunk in semantic_chunks 
                                           if chunk.contextual_embedding is None or 
                                           (isinstance(chunk.contextual_embedding, np.ndarray) and 
                                            np.allclose(chunk.contextual_embedding, 0))]
                
                if chunks_without_embeddings:
                    logger.error(f"Found {len(chunks_without_embeddings)} chunks without valid embeddings after indexing")
                    return self._create_error_response(
                        f"Failed to generate embeddings for {len(chunks_without_embeddings)} chunks",
                        'embedding_generation_error',
                        {'file_id': file_id, 'failed_chunks': len(chunks_without_embeddings)}
                    )
                
                # Store in vector database
                logger.info("Storing chunks in Qdrant...")
                try:
                    storage_result = self._store_chunks_in_qdrant(semantic_chunks, file_id, document_metadata, session_id, user_id)
                    logger.info("Chunks stored successfully in Qdrant")
                except Exception as storage_error:
                    logger.error(f"Qdrant storage failed: {storage_error}")
                    logger.exception("Full storage error traceback:")
                    raise
                
                if storage_result is None:
                    raise Exception("Failed to store document in vector database")
                
            # Store document metadata
            logger.info("Storing document metadata...")
            self._store_document_metadata(file_id, document_metadata, session_id, user_id)
            
            self.stats['documents_processed'] += 1
            
            return {
                'success': True,
                'chunks_count': len(semantic_chunks),
                'file_id': file_id,
                'processing_time': storage_result.get('processing_time', 0),
                'chunk_types': Counter(chunk.chunk_type for chunk in semantic_chunks),
                'avg_semantic_density': np.mean([chunk.semantic_density for chunk in semantic_chunks]),
                'is_image_content': is_image,
                'processing_method': 'standard'
            }
                
        except Exception as e:
            logger.error(f"Document storage failed at line {e.__traceback__.tb_lineno}: {str(e)}")
            logger.exception("Full error traceback:")
            import traceback
            logger.error(f"Detailed traceback: {traceback.format_exc()}")
            return ErrorResponse.create(
                error_message=str(e),
                error_type=self._classify_error(str(e)),
                context={'file_id': file_id, 'session_id': session_id}
            )
    
    def _process_chunks_in_batches(self, semantic_chunks):
        """Process chunks in smaller batches when full processing fails"""
        batch_size = 50  # Smaller batches for problematic documents
        
        logger.info(f"Processing {len(semantic_chunks)} chunks in batches of {batch_size}")
        
        for i in range(0, len(semantic_chunks), batch_size):
            batch = semantic_chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(semantic_chunks) + batch_size - 1) // batch_size
            
            try:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
                # Generate embeddings for this batch
                texts = [chunk.content for chunk in batch]
                embeddings = self.embedding_model.encode(
                    texts,
                    batch_size=16,  # Even smaller internal batch size
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                
                # Assign embeddings to chunks
                for chunk, embedding in zip(batch, embeddings):
                    chunk.contextual_embedding = embedding
                    
                logger.info(f"Batch {batch_num} completed successfully")
                
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                # Try individual processing for this batch
                for chunk in batch:
                    try:
                        embedding = self.embedding_model.encode([chunk.content], show_progress_bar=False)[0]
                        chunk.contextual_embedding = embedding
                    except Exception as e2:
                        logger.error(f"Individual chunk processing failed: {e2}")
                        # Use zero embedding as absolute fallback
                        chunk.contextual_embedding = np.zeros(self.embedding_dim)
    
    def _light_memory_cleanup(self):
        """
        Perform light memory cleanup without deleting model cache or reinitializing models.
        
        ✅ NEW METHOD: This replaces the aggressive _clear_embedding_cache() method.
        Only does memory cleanup, doesn't touch the HuggingFace cache or reinitialize models.
        """
        try:
            import gc
            import torch
            
            logger.info("Performing light memory cleanup...")
            
            # Clear PyTorch GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            
            # Force garbage collection
            gc.collect()
            logger.info("Python garbage collection completed")
            
            # ✅ THAT'S IT! No cache deletion, no model reinitialization
            logger.info("Light memory cleanup completed successfully")
            
        except Exception as e:
            logger.warning(f"Light memory cleanup failed (non-critical): {e}")
            # Don't raise - memory cleanup failure shouldn't stop processing
    
    # ❌ DEPRECATED: Keep the old method but make it call the light version
    def _clear_embedding_cache(self):
        """
        DEPRECATED: This method is too aggressive and causes problems.
        Redirecting to light memory cleanup instead.
        """
        logger.warning("_clear_embedding_cache called - redirecting to light memory cleanup")
        self._light_memory_cleanup()
    
    def _create_error_response(self, message: str, error_type: str = 'general_error', 
                              context: Dict[str, Any] = None, 
                              additional_fields: Dict[str, Any] = None) -> Dict[str, Any]:
        """Unified error response creation - replaces 3 duplicate functions"""
        error_response = ErrorResponse.create(
            error_message=message,
            error_type=error_type,
            retry_recommended=True,
            context=context or {}
        )
        
        # Add type-specific fields
        if additional_fields:
            error_response.update(additional_fields)
            
        return error_response

    def _create_storage_error_response(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create standardized error response for storage operations - DEPRECATED: Use _create_error_response"""
        return self._create_error_response(message, 'storage_error', context)
    
    def _classify_error(self, error_msg: str) -> str:
        """Classify error type for better handling"""
        error_lower = error_msg.lower()
        
        if any(term in error_lower for term in ["no such group", "hdf5", "cache", "corrupt"]):
            return 'transient_model_error'
        elif any(term in error_lower for term in ["division by zero", "empty", "no chunks"]):
            return 'empty_content_error'
        elif any(term in error_lower for term in ["rate", "429", "capacity", "too many requests"]):
            return 'rate_limit_error'
        elif any(term in error_lower for term in ["memory", "out of memory", "cuda", "gpu"]):
            return 'memory_error'
        elif any(term in error_lower for term in ["embedding", "encode", "model"]):
            return 'model_processing_error'
        else:
            return 'general_processing_error'
    
    def _store_chunks_in_qdrant(self, semantic_chunks: List[SemanticChunk], 
                               file_id: str, document_metadata: Dict[str, Any],
                               session_id: str, user_id: str) -> Dict[str, Any]:
        """Store semantic chunks in Qdrant with batch processing and validation"""
        from qdrant_client.http import models
        
        # ✅ IMPROVED: Better validation with clear error messages
        valid_chunks = []
        invalid_count = 0
        
        for i, chunk in enumerate(semantic_chunks):
            # Check if chunk has embedding
            if chunk.contextual_embedding is None:
                invalid_count += 1
                logger.warning(f"Chunk {i} has no embedding (None)")
                continue
            
            # Check if embedding has correct shape
            if not hasattr(chunk.contextual_embedding, 'shape'):
                invalid_count += 1
                logger.warning(f"Chunk {i} has invalid embedding (no shape attribute)")
                continue
            
            # Check if embedding is not all zeros
            if isinstance(chunk.contextual_embedding, np.ndarray) and np.allclose(chunk.contextual_embedding, 0):
                invalid_count += 1
                logger.warning(f"Chunk {i} has zero embedding")
                continue
            
            valid_chunks.append(chunk)
        
        # ✅ FAIL FAST: If too many chunks are invalid, stop immediately
        if invalid_count > 0:
            invalid_percentage = (invalid_count / len(semantic_chunks)) * 100
            logger.error(f"Found {invalid_count}/{len(semantic_chunks)} chunks with invalid embeddings ({invalid_percentage:.1f}%)")
            
            if invalid_percentage > 10:  # More than 10% invalid is a critical error
                raise Exception(f"Too many chunks ({invalid_count}/{len(semantic_chunks)}) have invalid embeddings. "
                              f"This indicates a problem with embedding generation. Please try again.")
        
        if not valid_chunks:
            raise Exception("No valid chunks with embeddings found for storage")
        
        logger.info(f"Storing {len(valid_chunks)} valid chunks (skipped {invalid_count} invalid)")
        
        # Create points for Qdrant
        points = []
        for chunk in valid_chunks:
            payload = self._create_chunk_payload(chunk, file_id, document_metadata)
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk.contextual_embedding.tolist(),
                payload=payload
            ))
        
        # Batch upload
        return self._batch_upload_to_qdrant(points, session_id, user_id)
    
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
    
    def _batch_upload_to_qdrant(self, points: List[Any], session_id: str, user_id: str) -> Dict[str, Any]:
        """Upload points to Qdrant in batches"""
        collection_name = self._get_collection_name(session_id, user_id)
        start_time = time.time()
        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        logger.info(f"Uploading {len(points)} points to collection '{collection_name}' in {total_batches} batches...")
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True  # Wait for indexing to complete
            )
            
            if batch_num % 5 == 0 or batch_num == total_batches:
                logger.info(f"Uploaded batch {batch_num}/{total_batches}")
        
        processing_time = time.time() - start_time
        logger.info(f"Qdrant upload completed in {processing_time:.2f}s")
        
        return {'processing_time': processing_time}
    
    def _store_document_metadata(self, file_id: str, metadata: Dict[str, Any], session_id: str, user_id: str):
        """Store document metadata in Qdrant with proper user-session isolation"""
        from qdrant_client.http import models
        
        try:
            dummy_embedding = [0.0] * self.embedding_dim
            collection_name = self._get_collection_name(session_id, user_id)
            
            payload = {
                'content': f"METADATA:{metadata.get('filename', file_id)}",
                'file_id': file_id,
                'is_metadata': True,
                'doc_pages': metadata.get('total_pages', 0),
                'doc_words': metadata.get('total_words', 0),
                'filename': metadata.get('filename', ''),
                'created_at': datetime.now().isoformat(),
                'session_id': session_id,
                'user_id': user_id
            }
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=dummy_embedding,
                    payload=payload
                )],
                wait=False
            )
            
            logger.info(f"Metadata stored in collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Metadata storage failed for user {user_id}, session {session_id}: {str(e)}")
            raise
    
    def _cache_key(self, file_id: str, question: str) -> str:
        """Generate cache key for question"""
        return hashlib.md5(f"{file_id}:{question}".encode()).hexdigest()
    
    def answer_question(self, file_id: str, question: str, 
                       conversation_history: List[Dict] = None,
                       session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """Answer a question using the RAG system with caching"""
        # Ensure both session_id and user_id are provided for consistent collection naming
        if not session_id or not user_id:
            return self._create_answer_error_response(
                'Both session_id and user_id are required for user-session isolation',
                context={'file_id': file_id, 'session_id': session_id, 'user_id': user_id}
            )
            
        if not self.qdrant_client:
            return self._create_answer_error_response(
                'Vector database not available',
                context={'file_id': file_id, 'session_id': session_id, 'user_id': user_id}
            )
        
        # Check cache first
        cache_key = self._cache_key(file_id, question)
        if cache_key in self._answer_cache:
            cached = self._answer_cache[cache_key]
            if time.time() - cached['timestamp'] < self.ANSWER_CACHE_TTL:
                logger.info(f"Returning cached answer for: {question[:50]}...")
                cached_result = cached['result'].copy()
                cached_result['cached'] = True
                return cached_result
        
        try:
            start_time = time.time()
            self.stats['questions_processed'] += 1
            
            # Ensure user-session collection exists
            self._ensure_collection(session_id, user_id)
            
            # Generate answer using user-session specific search results
            search_results = self._search_in_session(question, session_id, user_id, file_id)
            
            if not search_results or len(search_results) == 0:
                return self._create_answer_error_response(
                    'No relevant information found in your documents. The document may not contain information about this topic.',
                    context={
                        'file_id': file_id,
                        'session_id': session_id,
                        'user_id': user_id,
                        'question': question,
                        'suggestion': 'Try rephrasing your question or asking about a different topic from the document.'
                    }
                )
            
            # Select best chunks using simplified scoring
            final_chunks = self._select_best_chunks(search_results, self.MAX_CONTEXT_CHUNKS)
            
            if not final_chunks:
                return self._create_answer_error_response('No suitable content found to answer your question')
            
            # Store question context for intelligent post-processing
            self._current_question_context = question
            
            # Create enhanced prompt for accurate, document-specific answers
            enhanced_prompt = self._create_enhanced_anti_hallucination_prompt(
                question, final_chunks, "general"
            )
            
            # Generate answer with improved processing
            generation_result = self._generate_enhanced_answer_with_prompt(
                enhanced_prompt, search_results[:3], conversation_history, ""
            )
            
            if not generation_result['success']:
                return generation_result
            
            # Validate answer quality
            if 'answer' in generation_result:
                quality_check = self.validate_answer_quality(
                    generation_result['answer'], final_chunks
                )
                generation_result['quality_metrics'] = quality_check
            
            # Calculate confidence using the confidence calculator
            base_confidence = self.confidence_calculator.calculate_confidence(
                question, search_results, generation_result
            )
            
            # Adjust confidence based on quality validation
            quality_score = generation_result.get('quality_metrics', {}).get('score', 0.5)
            confidence = (base_confidence + quality_score) / 2.0
            
            # Prepare response
            result = self._prepare_final_response(
                generation_result, search_results, confidence, start_time
            )
            
            # Add user-session information
            result['file_id'] = file_id
            result['session_id'] = session_id
            result['user_id'] = user_id
            result['cached'] = False
            
            # Cache the result
            self._answer_cache[cache_key] = {
                'result': result.copy(),
                'timestamp': time.time()
            }
            
            # Limit cache size
            if len(self._answer_cache) > self.ANSWER_CACHE_SIZE:
                oldest = min(self._answer_cache.items(), key=lambda x: x[1]['timestamp'])
                del self._answer_cache[oldest[0]]
            
            return result
            
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            return self._create_answer_error_response(f'Processing error: {str(e)}')
    
    def _select_best_chunks(self, search_results: List, max_chunks: int = 8) -> List[str]:
        """Select best chunks with comprehensive scoring for all document types"""
        scored_chunks = []
        
        for result in search_results[:self.MAX_SEARCH_RESULTS * 2]:
            if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                content = result.chunk.content.strip()
                
                # Use minimal quality filtering
                if len(content) < self.MIN_CHUNK_LENGTH or self._is_low_quality_chunk(content):
                    continue
                
                # Enhanced quality scoring for comprehensive content
                base_score = result.combined_score * 0.6
                length_score = (min(len(content), 1500) / 1500) * 0.25  # Reward longer, detailed content
                
                # Content richness analysis
                content_lower = content.lower()
                richness_bonus = 0.0
                
                # Count unique meaningful words
                import re
                words = re.findall(r'\b[a-zA-Z]{3,}\b', content_lower)  # Words 3+ chars
                unique_words = len(set(words))
                if unique_words > 20:  # Rich content
                    richness_bonus += 0.15
                elif unique_words > 10:
                    richness_bonus += 0.1
                
                # Boost for structured information (lists, specifications, etc.)
                structure_indicators = ['â€¢', ':', '-', 'features', 'specifications', 'includes', 'capacity']
                structure_count = sum(1 for indicator in structure_indicators if indicator in content_lower)
                if structure_count >= 3:
                    richness_bonus += 0.1
                
                # Technical/detailed content bonus
                detailed_terms = ['technology', 'system', 'model', 'version', 'specifications', 'series']
                detail_count = sum(1 for term in detailed_terms if term in content_lower)
                if detail_count >= 2:
                    richness_bonus += 0.05
                
                quality_score = base_score + length_score + richness_bonus + 0.1
                scored_chunks.append((content, quality_score))
        
        # Sort by quality score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates with improved detection
        seen = set()
        unique_chunks = []
        for chunk, score in scored_chunks[:max_chunks * 3]:  # Check more candidates
            # Use more sophisticated duplicate detection
            chunk_signature = chunk[:200].lower().strip()
            if chunk_signature not in seen:
                seen.add(chunk_signature)
                unique_chunks.append(chunk)
                if len(unique_chunks) >= max_chunks:
                    break
        
        return unique_chunks
    
    def _create_answer_error_response(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create standardized error response for answer generation - DEPRECATED: Use _create_error_response"""
        answer_specific_fields = {
            'answer': '',
            'confidence': 0.0,
            'processing_time': 0.0
        }
        return self._create_error_response(message, 'answer_generation_error', context, answer_specific_fields)
    
    def _search_in_session(self, question: str, session_id: str, user_id: str, file_id: str = None):
        """Perform enhanced search within user-session specific collection for accurate answers"""
        try:
            collection_name = self._get_collection_name(session_id, user_id)
            logger.info(f"Performing enhanced search in user-session collection: {collection_name}")
            
            # Enhanced query processing for better relevance
            processed_question = self._preprocess_question_for_search(question)
            
            # Direct Qdrant search for session-specific collection with adaptive threshold
            query_embedding = self.embedding_model.encode(processed_question).tolist()
            
            # Use adaptive threshold based on collection content
            adaptive_threshold = self._calculate_adaptive_threshold(collection_name, query_embedding)
            
            # Use the new query_points API instead of deprecated search
            from qdrant_client.http import models
            search_results = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=15,  # Get more results for better filtering
                with_payload=True,
                with_vectors=False,
                score_threshold=adaptive_threshold
            ).points
            
            logger.info(f"Session search returned {len(search_results)} results")
            
            # Convert and rank results by relevance
            converted_results = []
            for i, result in enumerate(search_results):
                # Enhanced result filtering for quality
                content = result.payload.get('content', '').strip()
                if len(content) < 20:  # Skip very short chunks
                    continue
                
                # Create SessionSearchResult object
                session_result = SessionSearchResult(
                    chunk=SimpleNamespace(
                        content=content,
                        metadata=result.payload,
                        file_id=result.payload.get('file_id', ''),
                        chunk_type=result.payload.get('chunk_type', 'text'),
                        chunk_id=result.payload.get('chunk_id', ''),
                        page=result.payload.get('page', 0),
                        token_count=result.payload.get('token_count', len(content.split())),
                        semantic_density=result.payload.get('semantic_density', 0.0),
                        section_hierarchy=result.payload.get('section_hierarchy', []),
                        key_terms=result.payload.get('key_terms', []),
                        entities=result.payload.get('entities', [])
                    ),
                    dense_score=result.score,
                    combined_score=result.score,
                    relevance_rank=i
                )
                converted_results.append(session_result)
            
            # Optional file filtering within session (if file_id provided)
            if file_id:
                filtered_results = []
                for result in converted_results:
                    chunk_file_id = result.chunk.metadata.get('file_id')
                    if chunk_file_id == file_id:
                        filtered_results.append(result)
                
                logger.info(f"After file filtering within user-session: {len(filtered_results)} results")
                final_results = filtered_results[:self.MAX_SEARCH_RESULTS] if filtered_results else converted_results[:self.MAX_SEARCH_RESULTS]
            else:
                final_results = converted_results[:self.MAX_SEARCH_RESULTS]
            
            # Sort by relevance score to ensure best results first
            final_results.sort(key=lambda x: x.combined_score, reverse=True)
            
            logger.info(f"Returning {len(final_results)} high-quality results for answer generation")
            # print(final_results)
            return final_results
            
        except Exception as e:
            logger.error(f"Enhanced user-session search failed: {str(e)}")
            logger.warning(f"User-session search failed for user {user_id}, session {session_id}, returning empty results")
            return []
    
    def _preprocess_question_for_search(self, question: str) -> str:
        """Preprocess question to improve search relevance"""
        import re
        
        # Remove question words that don't help with semantic search
        processed = re.sub(r'^(what|how|when|where|why|who|which|can|could|would|should|do|does|did|is|are|was|were)\s+', '', question.lower())
        
        # Remove common filler words but preserve important context
        processed = re.sub(r'\b(please|could you|can you|tell me|explain|describe)\b', '', processed)
        
        # Preserve important interrogative context
        if any(word in question.lower() for word in ['how many', 'how much', 'what type', 'what kind']):
            return question  # Keep original for these specific patterns
        
        return processed.strip() if processed.strip() else question
    
    def _is_low_quality_chunk(self, content: str) -> bool:
        """Filter out low-quality chunks with minimal restrictions for all document types"""
        content_lower = content.lower().strip()
        
        # Very basic quality checks only
        if len(content_lower) < self.MIN_CHUNK_LENGTH:
            return True
            
        # Check for completely empty or whitespace-only content
        if not content.strip():
            return True
            
        # Check for purely repetitive content (very lenient)
        words = content_lower.split()
        if len(words) < 3:  # Too short to be meaningful
            return True
            
        # Only filter extremely repetitive content (90%+ repeated words)
        if len(words) > 5 and len(set(words)) < len(words) * 0.1:
            return True
            
        # Minimal low-quality patterns - only filter obvious junk
        minimal_patterns = ['lorem ipsum', 'placeholder text', 'todo', 'tbd']
        
        # Only filter if content is PRIMARILY these patterns
        pattern_count = sum(1 for pattern in minimal_patterns if pattern in content_lower)
        if pattern_count > 0 and len(content_lower) < 50:  # Short content with patterns
            return True
                
        # Check if content is purely punctuation or numbers (no actual text)
        import re
        text_content = re.sub(r'[\d\s\.\,\-\(\)\[\]Â©â„¢Â®â€¢]+', '', content_lower)
        if len(text_content) < 3:  # Less than 3 actual letters
            return True
                
        return False
    
    def _calculate_adaptive_threshold(self, collection_name: str, query_embedding: list) -> float:
        """Calculate adaptive threshold based on actual score distribution with caching"""
        try:
            query_hash = hashlib.md5(str(query_embedding[:10]).encode()).hexdigest()[:8]
            cache_key = f"{collection_name}_{query_hash}"
            
            # Thread-safe cache check
            with self.cache_lock:
                if cache_key in self.threshold_cache:
                    cached_threshold = self.threshold_cache[cache_key]
                    logger.debug(f"Using cached threshold: {cached_threshold:.3f}")
                    return cached_threshold
            
            # Calculate threshold (outside lock for better performance)
            from qdrant_client.http import models
            sample_results = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=10,
                with_payload=False,
                with_vectors=False
            ).points
            
            if not sample_results:
                return 0.01
            
            scores = [result.score for result in sample_results]
            scores.sort(reverse=True)
            
            max_score = scores[0]
            
            # More permissive adaptive threshold calculation for all document types
            if max_score > 0.5:
                threshold = max(0.05, max_score * 0.2)  # More permissive for high scores
            elif max_score > 0.1:
                median_score = scores[len(scores)//2] if len(scores) > 1 else scores[0]
                threshold = min(max_score * 0.15, median_score * 0.8)  # More permissive
            else:
                threshold = max(0.01, max_score * 0.05)  # Very permissive for low scores
            
            # Less aggressive filtering - allow more results through
            if len(scores) >= 3:
                top_50_percent_score = scores[min(4, len(scores)-1)]  # Top 50% instead of 30%
                threshold = min(threshold, top_50_percent_score)
            
            # Ensure threshold is never too restrictive
            threshold = max(0.01, min(threshold, 0.6))  # Cap at 0.6 instead of 0.8
            
            # Thread-safe cache update
            with self.cache_lock:
                if len(self.threshold_cache) >= self.cache_max_size:
                    oldest_key = next(iter(self.threshold_cache))
                    del self.threshold_cache[oldest_key]
                
                self.threshold_cache[cache_key] = threshold
            
            logger.info(f"Adaptive threshold: {threshold:.3f} (max_score: {max_score:.3f})")
            return threshold
            
        except Exception as e:
            logger.warning(f"Adaptive threshold calculation failed: {e}, using fallback")
            return 0.05

    # === CONSOLIDATED PROMPT TEMPLATES ===
    class PromptTemplates:
        """Centralized prompt management - reduces ~400 lines to ~100"""
        
        @staticmethod
        def create_answer_prompt(question: str, context_chunks: List[str], 
                               enable_citations: bool = False, 
                               min_chunk_length: int = 15) -> str:
            """Unified prompt creation optimized for comprehensive document content"""
            # Process more context for comprehensive answers
            numbered_context = ""
            relevant_chunks = []
            
            if enable_citations:
                # Use numbered sources for citations - include more sources
                for i, chunk in enumerate(context_chunks[:5], 1):  # Increased from 3 to 5
                    if len(chunk.strip()) > min_chunk_length:
                        numbered_context += f"\n[Source {i}]: {chunk.strip()}\n"
                        relevant_chunks.append(chunk)
                if not relevant_chunks:
                    numbered_context = f"\n[Source 1]: {context_chunks[0] if context_chunks else 'No relevant content found'}\n"
            else:
                # Use plain context without numbered sources - include more content
                for chunk in context_chunks[:6]:  # Increased from 3 to 6
                    if len(chunk.strip()) > min_chunk_length:
                        numbered_context += f"\n{chunk.strip()}\n\n"
                        relevant_chunks.append(chunk)
                if not relevant_chunks:
                    numbered_context = f"\n{context_chunks[0] if context_chunks else 'No relevant content found'}\n"
            
            citation_instruction = ("- Include source numbers in brackets [1], [2] when referencing specific information" 
                                  if enable_citations else "- Do not include source citations or reference numbers")
            
            return f"""You are an expert document analyst. Answer using ONLY the provided source material. Be accurate and concise.

STRICT GUIDELINES:
1. Base your answer EXCLUSIVELY on the provided sources
2. If sources don't contain the answer, clearly state this
3. Be specific and factual - avoid generalizations
4. Include ALL essential information and key details
5. Do not add external knowledge or assumptions

CONTEXT: {numbered_context}

QUESTION: {question}

REQUIREMENTS:
- Start directly with the answer
- Use specific facts from sources
- 3-5 sentences with all essential information
{citation_instruction}

Answer:"""

        @staticmethod  
        def create_question_prompt(content: str) -> str:
            """Detailed question generation prompt - maintains original quality"""
            return f"""Based on the following document content, generate 5 specific and meaningful questions that would help someone understand the key aspects of this content. The questions should be:

1. Specific to the content (not generic)
2. Focus on main topics, methods, findings, and applications
3. Be clear and concise
4. Help explore different aspects of the document

Content:
{content[:2000]}

Generate exactly 5 questions, one per line, numbered 1-5:"""

    def _create_enhanced_anti_hallucination_prompt(self, question: str, context_chunks: List[str], content_type: str = "general") -> str:
        """Create enhanced prompts that minimize hallucinations and ensure accurate, concise answers - DEPRECATED: Use PromptTemplates"""
        enable_citations = self.config_manager.get('ENABLE_CITATIONS', False)
        return self.PromptTemplates.create_answer_prompt(question, context_chunks, enable_citations, self.MIN_CHUNK_LENGTH)

    def _generate_enhanced_answer_with_prompt(self, enhanced_prompt: str, search_results: List,
                                           conversation_history: List[Dict] = None, 
                                           conversation_context: str = "") -> Dict[str, Any]:
        """Generate answer using custom anti-hallucination prompt"""
        try:
            # Use the enhanced prompt directly with LLM
            if not hasattr(self, 'llm') or not self.llm:
                logger.warning("LLM not available, using fallback answer generation")
                context_chunks = []
                for result in search_results[:3]:
                    if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                        context_chunks.append(result.chunk.content)
                
                # Simple fallback answer
                answer = "Based on the available content: " + " ".join(context_chunks[:2])[:400]
                if len(answer) > 400:
                    answer = answer[:400] + "..."
                
                return {
                    'success': True,
                    'answer': answer,
                    'context_used': len(search_results),
                    'processing_method': 'fallback'
                }
            
            response = self._rate_limited_llm_call(enhanced_prompt)
            answer_content = response.content.strip()
            
            # Check for API capacity issues
            if not answer_content or answer_content in ["", ".", "..", "..."]:
                logger.warning("LLM returned minimal response")
                raise Exception("LLM returned minimal response - possible API capacity issue")
            
            # Post-process answer
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
        """Post-process the generated answer - simplified"""
        answer = answer_content.strip()
        
        # Basic cleanup only
        answer = self.text_processor.clean_llm_output(answer)
        
        # Remove citations if they are disabled in config
        enable_citations = self.config_manager.get('ENABLE_CITATIONS', False)
        if not enable_citations:
            # Remove citation numbers like [1], [2], [Source 1], etc.
            answer = re.sub(r'\[(?:Source\s*)?\d+\]', '', answer)
            answer = re.sub(r'\[\d+\]', '', answer)
            # Clean up spacing issues from removed citations
            answer = re.sub(r'\s*,\s*,', ',', answer)  # Fix double commas
            answer = re.sub(r'\s*\.\s*\.', '.', answer)  # Fix double periods
            answer = re.sub(r'\s+([.,!?])', r'\1', answer)  # Remove space before punctuation
            answer = re.sub(r'\s+', ' ', answer)  # Clean up multiple spaces
            answer = answer.strip()
        
        # Ensure proper ending
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        return answer
    
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
        
        # Get the answer
        raw_answer = generation_result.get('answer', '')
        clean_answer = self._post_process_answer(raw_answer, search_results, False)
        
        # Extract sources information
        sources = []
        for result in search_results[:3]:  # Top 3 sources
            if hasattr(result, 'chunk'):
                source_info = {
                    'content_preview': result.chunk.content[:150] + "..." if len(result.chunk.content) > 150 else result.chunk.content,
                    'relevance_score': getattr(result, 'combined_score', getattr(result, 'hybrid_score', 0.0)),
                    'chunk_type': getattr(result.chunk, 'chunk_type', 'text') if hasattr(result.chunk, 'chunk_type') else 'text'
                }
                sources.append(source_info)
        # log
        logger.info(f"Final answer prepared with confidence {confidence:.2f} in {processing_time:.2f}s")
        return {
            'success': True,
            'answer': clean_answer,
            'sources': sources,
            'confidence_score': confidence,
            'processing_time': processing_time,
            'search_results_count': len(search_results),
            'processing_method': generation_result.get('processing_method', 'enhanced'),
            'quality_metrics': generation_result.get('quality_metrics', {})
        }
    
    def generate_sample_questions(self, session_id: str, user_id: str, 
                             chunks=None, document_metadata: Dict[str, Any] = None) -> List[str]:
        """Generate sample questions using RAG pipeline from user-session documents"""
        try:
            return self._generate_questions_from_session(session_id, user_id)
        except Exception as e:
            logger.warning(f"Sample question generation failed: {str(e)}")
            # Return simple default questions
            return [
                "What is this document about?",
                "Can you provide a summary?", 
                "What are the main topics covered?",
                "What conclusions can be drawn?"
            ]
    
    def _generate_questions_from_session(self, session_id: str, user_id: str) -> List[str]:
        """Generate questions using RAG pipeline from user-session documents"""
        try:
            collection_name = self._get_collection_name(session_id, user_id)
            logger.info(f"Retrieving content from user-session collection: {collection_name}")
            
            # Get sample content from session collection
            sample_queries = [
                "main topic content overview",
                "key concepts and methods", 
                "important findings results",
                "practical applications uses",
                "conclusions recommendations"
            ]
            
            all_content = []
            for query in sample_queries:
                try:
                    # Use user-session search to get relevant content
                    search_results = self._search_in_session(query, session_id, user_id, file_id=None)
                    for result in search_results[:2]:  # Take top 2 results per query
                        if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                            content = result.chunk.content.strip()
                            if len(content) > 50:  # Only meaningful content
                                all_content.append(content)
                except Exception as e:
                    logger.debug(f"Search failed for query '{query}': {e}")
                    continue
            
            if not all_content:
                logger.warning("No content found in user-session collection")
                # Return simple default questions
                return [
                    "What is this document about?",
                    "Can you provide a summary?", 
                    "What are the main topics covered?",
                    "What conclusions can be drawn?"
                ]
            
            # Combine content for analysis
            combined_content = " ".join(all_content[:10])  # Limit to avoid token limits
            logger.info(f"Analyzing {len(combined_content)} characters from user-session documents")
            
            # Generate questions using LLM with RAG content
            if hasattr(self, 'llm') and self.llm:
                return self._generate_questions_with_llm(combined_content)
            else:
                # Fallback to pattern-based generation
                return self._generate_questions_from_content(combined_content)
                
        except Exception as e:
            logger.error(f"User-session based question generation failed: {e}")
            # Return simple default questions
            return [
                "What is this document about?",
                "Can you provide a summary?", 
                "What are the main topics covered?",
                "What conclusions can be drawn?"
            ]
    
    def _generate_questions_with_llm(self, content: str) -> List[str]:
        """Generate questions using LLM based on document content"""
        try:
            question_prompt = self.PromptTemplates.create_question_prompt(content)

            response = self._rate_limited_llm_call(question_prompt)
            questions_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse questions from response
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, 6)):
                    # Extract the question text after the number
                    question = line.split('.', 1)[1].strip() if '.' in line else line
                    if question:
                        # Ensure question ends with ?
                        if not question.endswith('?'):
                            question += '?'
                        # Keep original numbering and full question
                        questions.append(question)
            
            if len(questions) >= 3:
                logger.info(f"Generated {len(questions)} LLM-based questions")
                return questions[:4]
            else:
                logger.warning("LLM generated insufficient questions, falling back")
                return self._generate_questions_from_content(content)
                
        except Exception as e:
            logger.error(f"LLM question generation failed: {e}")
            return self._generate_questions_from_content(content)
    
    def _generate_questions_from_content(self, content: str) -> List[str]:
        """Generate questions from content using pattern analysis"""
        import re
        
        content_lower = content.lower()
        questions = []
        
        # Extract key terms
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        meaningful_terms = [term for term in proper_nouns[:5] if len(term) > 3]
        
        # Generate context-aware questions
        if meaningful_terms:
            questions.append(f"What is {meaningful_terms[0]} and how is it explained?")
        else:
            questions.append("What is the main topic of this document?")
        
        if any(kw in content_lower for kw in ['objective', 'goal', 'purpose']):
            questions.append("What are the main objectives discussed?")
        else:
            questions.append("What is the purpose of this document?")
        
        if any(kw in content_lower for kw in ['method', 'approach', 'technique']):
            questions.append("What methodology or approach is used?")
        else:
            questions.append("What methods are described?")
        
        if any(kw in content_lower for kw in ['result', 'finding', 'outcome']):
            questions.append("What are the main results or findings?")
        else:
            questions.append("What key information is provided?")
        
        if any(kw in content_lower for kw in ['application', 'practical']):
            questions.append("What are the practical applications?")
        else:
            questions.append("What are the implications of this work?")
        
        # Format questions
        formatted_questions = []
        for i, q in enumerate(questions[:4], 1):
            if not q.startswith(f"{i}."):
                q = f"{i}. {q}"
            formatted_questions.append(q)
        
        return formatted_questions
    



__all__ = ['EnhancedRAGService']