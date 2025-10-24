"""
Search Engines for RAG System

This module contains different search engine implementations including
hybrid search that combines dense and sparse retrieval methods.
"""

import logging
import time
import numpy as np
from functools import lru_cache
from typing import List, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import tempfile
import threading
from .models import SemanticChunk, SearchResult
from .text_processing import TextProcessor
from ..utils.retry_utils import with_retry, performance_timer
from ..utils.config_manager import ConfigManager

# Optional BM25 support
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Okapi = None

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """Optimized hybrid search engine with caching"""
    
    # Configuration constants
    DEFAULT_SPARSE_WEIGHT = 0.3
    DEFAULT_DENSE_WEIGHT = 0.7
    RERANK_HYBRID_WEIGHT = 0.7
    RERANK_SCORE_WEIGHT = 0.3
    QUERY_CACHE_SIZE = 100
    
    def __init__(self, embedding_model: SentenceTransformer, reranker: CrossEncoder = None, 
                 config_manager: ConfigManager = None):
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.config_manager = config_manager
        self.text_processor = TextProcessor()
        
        # Search indexes
        self.dense_index = []
        self.sparse_index = None
        self.chunks_metadata = []
        
        # Search parameters
        self.sparse_weight = self.DEFAULT_SPARSE_WEIGHT
        self.dense_weight = self.DEFAULT_DENSE_WEIGHT
        
        # Query embedding cache
        self._query_embedding_cache = {}
        
        # Lock to protect model reinitialization/usage to avoid concurrent cache writes
        self.model_lock = threading.Lock()

    @with_retry()
    def index_chunks(self, chunks: List[SemanticChunk]):
        """Build indexes with performance optimizations and error handling"""
        if not chunks:
            logger.warning("No chunks to index")
            return
            
        logger.info(f"Indexing {len(chunks)} semantic chunks...")
        
        # ✅ NEW: Progress tracking for large documents
        start_time = time.time()
        
        with performance_timer("Dense embedding generation"):
            self._build_dense_index(chunks)
        
        embedding_time = time.time() - start_time
        
        # ✅ NEW: Report performance metrics
        chunks_per_second = len(chunks) / embedding_time if embedding_time > 0 else 0
        logger.info(f"Dense indexing completed: {len(chunks)} chunks in {embedding_time:.2f}s "
                   f"({chunks_per_second:.1f} chunks/sec)")
        
        if HAS_BM25 and self._should_build_sparse_index(chunks):
            sparse_start = time.time()
            with performance_timer("Sparse index generation"):
                self._build_sparse_index(chunks)
            sparse_time = time.time() - sparse_start
            logger.info(f"Sparse indexing completed in {sparse_time:.2f}s")
        
        self.chunks_metadata = chunks
        total_time = time.time() - start_time
        logger.info(f"Indexing complete: {len(chunks)} chunks indexed in {total_time:.2f}s")

    def _should_build_sparse_index(self, chunks: List[SemanticChunk]) -> bool:
        """Determine if sparse index should be built based on chunk count"""
        max_chunks = self.config_manager.get('MAX_CHUNKS_FOR_DENSE_EMBEDDING', 500) if self.config_manager else 500
        skip_bm25 = self.config_manager.get('SKIP_BM25_FOR_LARGE_FILES', True) if self.config_manager else True
        
        return not (skip_bm25 and len(chunks) > max_chunks)

    def _build_dense_index(self, chunks: List[SemanticChunk]):
        """Build dense embedding index with optimal batching"""
        logger.info(f"Building dense index for {len(chunks)} chunks")
        
        # Prepare texts efficiently
        texts_for_embedding = self._prepare_texts_for_embedding(chunks)
        logger.info(f"Prepared {len(texts_for_embedding)} texts for embedding")
        
        # Generate embeddings in optimal batches
        batch_size = self.config_manager.get('EMBEDDING_BATCH_SIZE', 64) if self.config_manager else 64
        
        # Reduce batch size for large documents to prevent memory issues
        if len(texts_for_embedding) > 2000:
            batch_size = min(batch_size, 10)
            logger.info(f"Large document detected, reducing batch size to {batch_size}")
        elif len(texts_for_embedding) > 1000:
            batch_size = min(batch_size, 20)
            logger.info(f"Medium document detected, reducing batch size to {batch_size}")
            
        logger.info(f"Starting embedding generation with batch size {batch_size}")
        try:
            embeddings = self._generate_embeddings_batch(texts_for_embedding, batch_size)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Embedding generation failed in _build_dense_index: {e}")
            raise
        
        self.dense_index = embeddings
        
        # Store embeddings in chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.contextual_embedding = embedding

    def _prepare_texts_for_embedding(self, chunks: List[SemanticChunk]) -> List[str]:
        """Prepare texts for embedding with performance optimization"""
        max_chunks_threshold = self.config_manager.get('MAX_CHUNKS_FOR_DENSE_EMBEDDING', 500) if self.config_manager else 500
        is_large_file = len(chunks) > max_chunks_threshold
        
        if is_large_file:
            # Fast preparation for large files
            return [chunk.content for chunk in chunks]
        else:
            # Enhanced context for smaller files
            texts = []
            for chunk in chunks:
                enhanced_text = chunk.content
                
                # Add section context (limited)
                if chunk.section_hierarchy and len(chunk.section_hierarchy) <= 3:
                    hierarchy_context = " > ".join(chunk.section_hierarchy[:3])
                    enhanced_text = f"Section: {hierarchy_context}\n{enhanced_text}"
                
                # Add key terms (limited)
                if chunk.key_terms:
                    terms_context = ", ".join(chunk.key_terms[:3])
                    enhanced_text = f"{enhanced_text}\nKey terms: {terms_context}"
                
                texts.append(enhanced_text)
            
            return texts

    def _generate_embeddings_batch(self, texts: List[str], batch_size: int) -> List[np.ndarray]:
        """Generate embeddings in batches with robust error handling for large documents"""
        
        # ✅ IMPROVED: Better adaptive batch sizing
        if len(texts) > 5000:
            batch_size = 8  # Very small batches for very large documents
            logger.info(f"Very large document detected ({len(texts)} chunks), using batch size {batch_size}")
        elif len(texts) > 2000:
            batch_size = 16  # Small batches for large documents
            logger.info(f"Large document detected ({len(texts)} chunks), using batch size {batch_size}")
        elif len(texts) > 1000:
            batch_size = 32  # Medium batches
            logger.info(f"Medium document detected ({len(texts)} chunks), using batch size {batch_size}")
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Starting embedding generation: {len(texts)} texts in {total_batches} batches")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            retry_count = 0
            max_retries = 2  # ✅ REDUCED: Less retries, faster failure
            
            while retry_count < max_retries:
                try:
                    # ✅ SIMPLIFIED: Remove health checks and complex retry logic
                    # Just do the encoding with good settings
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        batch_size=min(16, len(batch_texts)),
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        device='cpu'
                    )
                    embeddings.extend(batch_embeddings)
                    
                    if batch_num % 10 == 0 or batch_num == total_batches:
                        logger.info(f"Processed batch {batch_num}/{total_batches} ({len(embeddings)} total embeddings)")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    error_str = str(e).lower()
                    
                    # ✅ IMPROVED: Better error classification
                    if 'no such group' in error_str or 'hdf5' in error_str or 'cache' in error_str:
                        logger.error(f"Model cache error in batch {batch_num}, attempt {retry_count}/{max_retries}: {e}")
                        
                        if retry_count < max_retries:
                            # ✅ LIGHT CLEANUP: Just clear memory, don't delete cache
                            logger.info("Performing light memory cleanup...")
                            import gc
                            import torch
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                    else:
                        logger.error(f"Unexpected error in batch {batch_num}, attempt {retry_count}/{max_retries}: {e}")
                    
                    # ❌ REMOVE FALLBACK: Don't use zero embeddings - fail fast instead
                    if retry_count >= max_retries:
                        logger.error(f"Failed to process batch {batch_num} after {max_retries} attempts")
                        raise Exception(f"Embedding generation failed at batch {batch_num}/{total_batches}: {str(e)}")
        
        logger.info(f"Embedding generation completed: {len(embeddings)} embeddings generated successfully")
        return embeddings
    
    def _reinitialize_model_after_corruption(self, use_temp_cache: bool = False):
        """Helper method to reinitialize the model after cache corruption

        If use_temp_cache=True, a per-process temporary cache directory will be
        used to avoid shared-cache corruption from concurrent processes.
        """
        import gc
        import torch
        import os
        import shutil
        import tempfile

        # Clear memory and cache aggressively
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # If not using a temp cache, clear shared cache directories
        if not use_temp_cache:
            try:
                cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
                if os.path.exists(cache_dir):
                    logger.info("Clearing corrupted sentence transformers shared cache")
                    shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception as cache_error:
                logger.warning(f"Failed to clear shared cache: {cache_error}")

        # Delete current model instance
        if hasattr(self, 'embedding_model'):
            try:
                del self.embedding_model
            except Exception:
                pass
        gc.collect()

        # Reinitialize model with fresh isolated cache by default when requested
        from sentence_transformers import SentenceTransformer
        embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

        if use_temp_cache:
            temp_cache = tempfile.mkdtemp(prefix="st_cache_")
            logger.info(f"Initializing new embedding model using isolated cache: {temp_cache}")
            self.embedding_model = SentenceTransformer(
                embedding_model_name,
                device='cpu',
                cache_folder=temp_cache
            )
        else:
            logger.info("Initializing new embedding model using shared cache")
            self.embedding_model = SentenceTransformer(
                embedding_model_name,
                device='cpu',
                cache_folder=None  # Force fresh download into shared cache
            )

    def _build_sparse_index(self, chunks: List[SemanticChunk]):
        """Build BM25 sparse index efficiently"""
        corpus = []
        for chunk in chunks:
            tokens = chunk.content.lower().split()
            
            # Add key terms for smaller chunks only
            if len(tokens) < 500:
                for term in chunk.key_terms[:5]:
                    tokens.extend(term.lower().split())
            
            corpus.append(tokens)
        
        self.sparse_index = BM25Okapi(corpus)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding with caching"""
        # Use simple cache key (query text)
        if query in self._query_embedding_cache:
            logger.debug(f"Using cached embedding for query: {query[:50]}...")
            return self._query_embedding_cache[query]
        
        # Generate new embedding
        embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        
        # Cache with size limit
        self._query_embedding_cache[query] = embedding
        if len(self._query_embedding_cache) > self.QUERY_CACHE_SIZE:
            # Remove oldest entry (first key)
            oldest_key = next(iter(self._query_embedding_cache))
            del self._query_embedding_cache[oldest_key]
        
        return embedding

    def search(self, query: str, top_k: int = 20, 
              sparse_weight: float = None, dense_weight: float = None) -> List[SearchResult]:
        """Perform optimized hybrid search with caching"""
        if not self.chunks_metadata:
            return []
        
        # Use provided weights or defaults
        sparse_w = sparse_weight if sparse_weight is not None else self.sparse_weight
        dense_w = dense_weight if dense_weight is not None else self.dense_weight
        
        # Normalize weights
        total_weight = sparse_w + dense_w
        sparse_w /= total_weight
        dense_w /= total_weight
        
        # Perform searches
        dense_scores = self._dense_search(query)
        sparse_scores = self._sparse_search(query)
        
        # Normalize and combine scores
        dense_scores = self.text_processor.normalize_scores(dense_scores)
        sparse_scores = self.text_processor.normalize_scores(sparse_scores)
        
        # Create results
        results = []
        for i, chunk in enumerate(self.chunks_metadata):
            hybrid_score = dense_w * dense_scores[i] + sparse_w * sparse_scores[i]
            
            result = SearchResult(
                chunk=chunk,
                dense_score=dense_scores[i],
                sparse_score=sparse_scores[i],
                rerank_score=0.0,
                hybrid_score=hybrid_score,
                relevance_explanation=self._explain_relevance(query, chunk, dense_scores[i], sparse_scores[i])
            )
            results.append(result)
        
        # Sort and get top candidates
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # Use configuration for candidate selection
        candidate_multiplier = self.config_manager.get('SEARCH_CANDIDATE_MULTIPLIER', 2) if self.config_manager else 2
        top_candidates = results[:min(top_k * candidate_multiplier, len(results))]
        
        # Rerank if available
        if self.reranker and len(top_candidates) > 1:
            top_candidates = self._rerank_results(query, top_candidates)
        
        return top_candidates[:top_k]

    def _dense_search(self, query: str) -> List[float]:
        """Perform dense search using embeddings with caching"""
        query_embedding = self._get_query_embedding(query)
        
        scores = []
        for chunk_embedding in self.dense_index:
            similarity = np.dot(query_embedding, chunk_embedding)
            scores.append(float(similarity))
        
        return scores

    def _sparse_search(self, query: str) -> List[float]:
        """Perform sparse search using BM25"""
        if not self.sparse_index:
            return [0.0] * len(self.chunks_metadata)
        
        query_tokens = query.lower().split()
        scores = self.sparse_index.get_scores(query_tokens)
        return scores.tolist()

    def _explain_relevance(self, query: str, chunk: SemanticChunk, 
                         dense_score: float, sparse_score: float) -> str:
        """Generate relevance explanation"""
        explanations = []
        
        query_terms = set(query.lower().split())
        content_terms = set(chunk.content.lower().split())
        matching_terms = query_terms.intersection(content_terms)
        
        if matching_terms:
            explanations.append(f"Matches: {', '.join(list(matching_terms)[:3])}")
        
        if chunk.key_terms:
            key_matches = [term for term in chunk.key_terms 
                          if any(qt in term.lower() for qt in query_terms)]
            if key_matches:
                explanations.append(f"Key terms: {', '.join(key_matches[:2])}")
        
        if dense_score > 0.8:
            explanations.append("High semantic similarity")
        if sparse_score > 0.8:
            explanations.append("Strong keyword match")
        
        return " | ".join(explanations) if explanations else "General relevance"

    @with_retry()
    def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results using cross-encoder with error handling"""
        try:
            pairs = [[query, result.chunk.content] for result in results]
            rerank_scores = self.reranker.predict(pairs)
            
            for result, score in zip(results, rerank_scores):
                result.rerank_score = float(score)
                # Use class constants for weights
                result.hybrid_score = (result.hybrid_score * self.RERANK_HYBRID_WEIGHT + 
                                     result.rerank_score * self.RERANK_SCORE_WEIGHT)
            
            results.sort(key=lambda x: x.hybrid_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Reranking failed: {str(e)}")
        
        return results


__all__ = ['HybridSearchEngine', 'HAS_BM25']