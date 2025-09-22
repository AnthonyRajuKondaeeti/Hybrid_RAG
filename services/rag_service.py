"""
REFACTORED ENHANCED RAG SERVICE - Production-Ready Document Question Answering System

This refactored module addresses redundancies and architectural issues while maintaining
all functionality of the original system.

IMPROVEMENTS:
============
- Eliminated code duplication and redundant logic
- Centralized error handling and retry mechanisms
- Improved memory efficiency and performance
- Better state management and thread safety
- Cleaner abstractions and separation of concerns
- Consistent configuration handling
"""

import os
import time
import logging
import json
import uuid
import re
import numpy as np
import hashlib
import asyncio
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Protocol
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import wraps, lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.docstore.document import Document
from langchain_mistralai.chat_models import ChatMistralAI

from services.ocr_processor import OCRProcessor

try:
    from config import Config
except ImportError:
    # Fallback config for development
    class Config:
        ENABLE_IMAGE_ANALYSIS = True
        OCR_LANGUAGES = ['en']
        OCR_GPU_ENABLED = False
        OCR_CONFIDENCE_THRESHOLD = 0.5
        OCR_PREPROCESSING = True
        MAX_RETRIES = 3
        RETRY_BASE_DELAY = 1.0
        RETRY_EXPONENTIAL_BASE = 2.0
        RETRY_MAX_DELAY = 60.0
        MAX_CHUNKS_FOR_DENSE_EMBEDDING = 500
        EMBEDDING_BATCH_SIZE = 64
        SKIP_BM25_FOR_LARGE_FILES = True


# Optional dependencies
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logging.warning("rank_bm25 not installed. Sparse search will be limited.")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

logger = logging.getLogger(__name__)

# PROTOCOLS AND INTERFACES

class ChunkProcessor(Protocol):
    """Protocol for document chunking strategies"""
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List['SemanticChunk']:
        ...

class SearchEngine(Protocol):
    """Protocol for search implementations"""
    def index_chunks(self, chunks: List['SemanticChunk']) -> None:
        ...
    
    def search(self, query: str, top_k: int) -> List['SearchResult']:
        ...

# CENTRALIZED ERROR HANDLING

class RetryConfig:
    """Configuration for retry logic"""
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 exponential_base: float = 2.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.exponential_base = exponential_base
        self.max_delay = max_delay

def with_retry(retry_config: RetryConfig = None, 
               transient_errors: Tuple[str, ...] = ("no such group", "rate", "capacity", "429")):
    """Decorator for retry logic with exponential backoff"""
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Check if error is transient
                    is_transient = any(err_type in error_msg for err_type in transient_errors)
                    
                    if is_transient and attempt < retry_config.max_retries - 1:
                        delay = min(
                            retry_config.base_delay * (retry_config.exponential_base ** attempt),
                            retry_config.max_delay
                        )
                        logger.warning(f"Transient error (attempt {attempt + 1}/{retry_config.max_retries}): {str(e)}")
                        logger.info(f"Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        # Non-transient error or final attempt
                        break
            
            # Re-raise the last exception
            raise last_exception
        return wrapper
    return decorator

@contextmanager
def performance_timer(operation_name: str, stats_dict: Dict = None):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"{operation_name} completed in {duration:.2f}s")
        if stats_dict is not None:
            stats_dict.setdefault('timings', []).append(duration)

# CENTRALIZED TEXT PROCESSING

class TextProcessor:
    """Centralized text processing utilities"""
    
    @staticmethod
    def extract_key_terms(content: str, max_terms: int = 10) -> List[str]:
        """Extract key terms using frequency and capitalization heuristics"""
        # Extract capitalized terms
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', content) + re.findall(r"'([^']+)'", content)
        
        # Extract technical terms
        technical_terms = re.findall(r'\b[A-Z]{2,}\b|\b\w*\d+\w*\b', content)
        
        # Combine and filter
        all_terms = capitalized_terms + quoted_terms + technical_terms
        common_words = {'The', 'This', 'That', 'These', 'Those', 'And', 'Or', 'But', 'For', 'With'}
        key_terms = [term for term in all_terms if len(term) > 2 and term not in common_words]
        
        return sorted(list(set(key_terms)), key=len, reverse=True)[:max_terms]
    
    @staticmethod
    def extract_entities(content: str) -> List[str]:
        """Extract named entities using pattern matching"""
        entities = []
        
        # Date patterns
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}\b'
        ]
        
        # Number patterns
        number_patterns = [
            r'\b\d+(?:\.\d+)?\s*%\b',
            r'\b\d+(?:\.\d+)?\s*(?:kg|g|lb|oz|mm|cm|m|km|ft|in|miles?)\b',
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b'
        ]
        
        for pattern in date_patterns + number_patterns:
            entities.extend(re.findall(pattern, content, re.IGNORECASE))
        
        return list(set(entities))
    
    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores
        
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences using available tools"""
        if HAS_NLTK:
            return sent_tokenize(text)
        else:
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip() and len(s.split()) > 2]

# CORE DATA STRUCTURES

@dataclass
class SemanticChunk:
    """Semantic chunk with intelligent boundaries and metadata"""
    content: str
    chunk_type: str
    section_hierarchy: List[str]
    key_terms: List[str]
    entities: List[str]
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    page: int = 1
    token_count: int = 0
    semantic_density: float = 0.0
    contextual_embedding: Optional[np.ndarray] = None
    file_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.token_count:
            self.token_count = len(self.content.split())

@dataclass 
class SearchResult:
    """Enhanced search result with multiple relevance signals"""
    chunk: SemanticChunk
    dense_score: float
    sparse_score: float
    rerank_score: float
    hybrid_score: float
    relevance_explanation: str
    citation_id: str = ""

# CONFIGURATION MANAGER

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config):
        self.config = config
        self._cache = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with caching"""
        if key not in self._cache:
            self._cache[key] = getattr(self.config, key, default)
        return self._cache[key]
    
    def get_retry_config(self) -> RetryConfig:
        """Get retry configuration"""
        return RetryConfig(
            max_retries=self.get('MAX_RETRIES', 3),
            base_delay=self.get('RETRY_BASE_DELAY', 1.0),
            exponential_base=self.get('RETRY_EXPONENTIAL_BASE', 2.0),
            max_delay=self.get('RETRY_MAX_DELAY', 60.0)
        )

# IMPROVED SEMANTIC CHUNKER

class SemanticChunker:
    """Advanced semantic chunking with improved efficiency"""
    
    def __init__(self, target_chunk_size: int = 300, overlap_size: int = 50):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
        self.text_processor = TextProcessor()
        
        # Compile patterns once
        self.header_patterns = [
            re.compile(r'^(#{1,6})\s+(.+)$'),
            re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$'),
            re.compile(r'^([A-Z][A-Z\s]{2,20}):?\s*$'),
            re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$'),
        ]
        
        self.list_patterns = [
            re.compile(r'^\s*(?:[-•*]|\d+\.|\([a-z]\))\s+'),
            re.compile(r'^\s*(?:[A-Z]\.|\d+\))\s+'),
        ]

    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[SemanticChunk]:
        """Create semantically coherent chunks with improved efficiency"""
        if not text.strip():
            return []
            
        metadata = metadata or {}
        
        # Parse document structure
        structured_content = self._parse_document_structure(text)
        
        # Create semantic chunks
        chunks = []
        for section in structured_content:
            section_chunks = self._create_section_chunks(section, metadata)
            chunks.extend(section_chunks)
        
        # Calculate semantic metrics in batch
        self._calculate_semantic_metrics_batch(chunks)
        
        return chunks

    def _parse_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """Parse document into hierarchical structure with improved efficiency"""
        sections = []
        current_hierarchy = []
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if current_content:
                    current_content.append('')
                continue
            
            # Check for headers using compiled patterns
            header_match = self._match_header(line_stripped)
            if header_match:
                # Save previous section
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:  # Only add non-empty sections
                        sections.append({
                            'hierarchy': current_hierarchy.copy(),
                            'content': content,
                            'type': self._classify_content_type(content)
                        })
                
                # Update hierarchy
                level, title = header_match
                self._update_hierarchy(current_hierarchy, level, title)
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append({
                    'hierarchy': current_hierarchy.copy(),
                    'content': content,
                    'type': self._classify_content_type(content)
                })
        
        return sections

    def _match_header(self, line: str) -> Optional[Tuple[int, str]]:
        """Match header patterns using compiled regex"""
        for i, pattern in enumerate(self.header_patterns):
            match = pattern.match(line)
            if match:
                level_indicator = match.group(1)
                title = match.group(2).strip()
                
                if level_indicator.startswith('#'):
                    level = len(level_indicator)
                elif '.' in level_indicator:
                    level = len(level_indicator.split('.'))
                else:
                    level = 1
                
                return level, title
        return None

    def _classify_content_type(self, content: str) -> str:
        """Classify content type with caching"""
        if not content.strip():
            return 'empty'
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Check for lists using compiled patterns
        list_lines = sum(1 for line in lines if any(pattern.match(line) for pattern in self.list_patterns))
        if list_lines > len(lines) * 0.5:
            return 'list'
        
        # Other classifications
        if '|' in content and content.count('|') > 4:
            return 'table'
        
        return 'paragraph' if len(lines) == 1 else 'section'

    def _update_hierarchy(self, hierarchy: List[str], level: int, title: str):
        """Update section hierarchy efficiently"""
        while len(hierarchy) >= level:
            hierarchy.pop()
        hierarchy.append(title)

    def _create_section_chunks(self, section: Dict[str, Any], metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Create chunks from section with improved efficiency"""
        content = section['content']
        if not content.strip():
            return []
        
        # For short sections, keep as single chunk
        word_count = len(content.split())
        if word_count <= self.target_chunk_size:
            return [self._create_chunk(content, section, metadata)]
        
        # Split longer sections
        if section['type'] == 'list':
            return self._chunk_list(content, section, metadata)
        else:
            return self._chunk_by_semantic_boundaries(content, section, metadata)

    def _create_chunk(self, content: str, section: Dict[str, Any], metadata: Dict[str, Any]) -> SemanticChunk:
        """Create a single semantic chunk"""
        return SemanticChunk(
            content=content,
            chunk_type=section['type'],
            section_hierarchy=section['hierarchy'],
            key_terms=self.text_processor.extract_key_terms(content),
            entities=self.text_processor.extract_entities(content),
            page=metadata.get('page', 1),
            file_id=metadata.get('file_id')
        )

    def _chunk_list(self, content: str, section: Dict[str, Any], metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Chunk list content efficiently"""
        chunks = []
        lines = [line for line in content.split('\n') if line.strip()]
        
        current_chunk_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = len(line.split())
            is_new_item = any(pattern.match(line.strip()) for pattern in self.list_patterns)
            
            if (is_new_item and current_tokens > self.target_chunk_size * 0.7 and current_chunk_lines):
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append(self._create_chunk(chunk_content, section, metadata))
                
                current_chunk_lines = [line]
                current_tokens = line_tokens
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
        
        # Add final chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunks.append(self._create_chunk(chunk_content, section, metadata))
        
        return chunks

    def _chunk_by_semantic_boundaries(self, content: str, section: Dict[str, Any], metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Chunk content by semantic boundaries with improved efficiency"""
        chunks = []
        sentences = self.text_processor.split_sentences(content)
        
        current_chunk_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            if current_tokens + sentence_tokens > self.target_chunk_size and current_chunk_sentences:
                # Create chunk with overlap
                chunk_content = ' '.join(current_chunk_sentences)
                chunks.append(self._create_chunk(chunk_content, section, metadata))
                
                # Start new chunk with overlap
                overlap_sentences = max(0, len(current_chunk_sentences) - 3)
                current_chunk_sentences = current_chunk_sentences[overlap_sentences:] + [sentence]
                current_tokens = sum(len(s.split()) for s in current_chunk_sentences)
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_content = ' '.join(current_chunk_sentences)
            chunks.append(self._create_chunk(chunk_content, section, metadata))
        
        return chunks

    def _calculate_semantic_metrics_batch(self, chunks: List[SemanticChunk]):
        """Calculate semantic metrics for all chunks in batch"""
        for chunk in chunks:
            total_words = len(chunk.content.split())
            key_term_words = sum(len(term.split()) for term in chunk.key_terms)
            chunk.semantic_density = key_term_words / total_words if total_words > 0 else 0.0
            chunk.token_count = total_words

# IMPROVED HYBRID SEARCH ENGINE

class HybridSearchEngine:
    """Optimized hybrid search engine"""
    
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
        self.sparse_weight = 0.3
        self.dense_weight = 0.7

    @with_retry()
    def index_chunks(self, chunks: List[SemanticChunk]):
        """Build indexes with performance optimizations and error handling"""
        if not chunks:
            logger.warning("No chunks to index")
            return
            
        logger.info(f"Indexing {len(chunks)} semantic chunks...")
        
        with performance_timer("Dense embedding generation"):
            self._build_dense_index(chunks)
        
        if HAS_BM25 and self._should_build_sparse_index(chunks):
            with performance_timer("Sparse index generation"):
                self._build_sparse_index(chunks)
        
        self.chunks_metadata = chunks
        logger.info(f"Indexing complete. {len(chunks)} chunks indexed.")

    def _should_build_sparse_index(self, chunks: List[SemanticChunk]) -> bool:
        """Determine if sparse index should be built based on chunk count"""
        max_chunks = self.config_manager.get('MAX_CHUNKS_FOR_DENSE_EMBEDDING', 500) if self.config_manager else 500
        skip_bm25 = self.config_manager.get('SKIP_BM25_FOR_LARGE_FILES', True) if self.config_manager else True
        
        return not (skip_bm25 and len(chunks) > max_chunks)

    def _build_dense_index(self, chunks: List[SemanticChunk]):
        """Build dense embedding index with optimal batching"""
        # Prepare texts efficiently
        texts_for_embedding = self._prepare_texts_for_embedding(chunks)
        
        # Generate embeddings in optimal batches
        batch_size = self.config_manager.get('EMBEDDING_BATCH_SIZE', 64) if self.config_manager else 64
        embeddings = self._generate_embeddings_batch(texts_for_embedding, batch_size)
        
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
        """Generate embeddings in batches with optimal settings"""
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    batch_size=32,  # Internal model batch size
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.extend(batch_embeddings)
                
                if batch_num % 5 == 0 or batch_num == total_batches:
                    logger.info(f"Processed batch {batch_num}/{total_batches}")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {str(e)}")
                # Use zero embeddings as fallback
                zero_embedding = np.zeros(self.embedding_model.get_sentence_embedding_dimension())
                embeddings.extend([zero_embedding] * len(batch_texts))
        
        return embeddings

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

    def search(self, query: str, top_k: int = 20, 
              sparse_weight: float = None, dense_weight: float = None) -> List[SearchResult]:
        """Perform optimized hybrid search"""
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
        top_candidates = results[:min(top_k * 2, len(results))]
        
        # Rerank if available
        if self.reranker and len(top_candidates) > 1:
            top_candidates = self._rerank_results(query, top_candidates)
        
        return top_candidates[:top_k]

    def _dense_search(self, query: str) -> List[float]:
        """Perform dense search using embeddings"""
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        
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
                result.hybrid_score = result.hybrid_score * 0.7 + result.rerank_score * 0.3
            
            results.sort(key=lambda x: x.hybrid_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Reranking failed: {str(e)}")
        
        return results

# CONFIDENCE CALCULATOR

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

# ENHANCED RAG SERVICE

class ImageAnalysisService:
    """Service for analyzing standalone images using multimodal capabilities."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.ocr_processor = OCRProcessor()
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM for image analysis."""
        try:
            from langchain_mistralai.chat_models import ChatMistralAI
            self.llm = ChatMistralAI(
                model=self.config_manager.get('MISTRAL_MODEL', 'mistral-small-latest'),
                temperature=0.1,
                api_key=self.config_manager.get('MISTRAL_API_KEY'),
                timeout=60,
                max_retries=3
            )
        except ImportError:
            logger.error("langchain_mistralai not installed")
            self.llm = None
        except Exception as e:
            logger.error(f"Failed to initialize image analysis LLM: {str(e)}")
            self.llm = None
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze an image and extract information."""
        try:
            # First, perform OCR
            ocr_result = self.ocr_processor.process_image_file(image_path)
            
            # Prepare image for LLM analysis (if supported)
            image_analysis = self._analyze_with_llm(image_path, ocr_result)
            
            return {
                'success': True,
                'ocr_result': ocr_result,
                'image_analysis': image_analysis,
                'file_path': image_path,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_path': image_path
            }
    
    def _analyze_with_llm(self, image_path: str, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image using LLM with OCR context."""
        try:
            ocr_text = ocr_result.get('text', '') if ocr_result.get('success') else 'No text extracted'
            
            # Create a comprehensive prompt for image analysis
            analysis_prompt = f"""
Based on the OCR-extracted text from this image, provide a comprehensive analysis:

OCR Text Extracted:
{ocr_text}

Please provide:
1. A summary of what this document/image appears to be
2. Key information extracted
3. Any important details or patterns noticed
4. Potential document type or category

Analysis:"""
            
            if self.llm:
                response = self.llm.invoke(analysis_prompt)
                return {
                    'success': True,
                    'analysis': response.content,
                    'method': 'llm_analysis'
                }
            else:
                return {
                    'success': False,
                    'error': 'LLM not available for image analysis'
                }
                
        except Exception as e:
            logger.warning(f"LLM image analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def answer_question_about_image(self, image_path: str, question: str) -> Dict[str, Any]:
        """Answer questions about a specific image."""
        try:
            # Analyze the image first
            image_analysis = self.analyze_image(image_path)
            
            if not image_analysis['success']:
                return image_analysis
            
            ocr_text = image_analysis['ocr_result'].get('text', '')
            image_desc = image_analysis['image_analysis'].get('analysis', '')
            
            # Create context from OCR and image analysis
            context = f"""
Image Analysis Context:
{image_desc}

OCR-Extracted Text:
{ocr_text}
"""
            
            # Answer the question using the context
            answer_prompt = f"""
Based on the following image analysis and OCR text, please answer the question:

{context}

Question: {question}

Please provide a specific answer based only on the information available from the image. If the information is not available in the image, please state that clearly.

Answer:"""
            
            if self.llm:
                response = self.llm.invoke(answer_prompt)
                
                return {
                    'success': True,
                    'question': question,
                    'answer': response.content,
                    'confidence_score': 0.8 if ocr_text.strip() else 0.4,
                    'sources': [{
                        'type': 'image_ocr',
                        'confidence': image_analysis['ocr_result'].get('confidence', 0),
                        'file_path': image_path
                    }],
                    'processing_method': 'image_qa'
                }
            else:
                return {
                    'success': False,
                    'error': 'LLM not available for question answering'
                }
                
        except Exception as e:
            logger.error(f"Image question answering failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'question': question
            }

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
        # Initialize Qdrant client with improved settings for cloud connectivity
        self.qdrant_client = QdrantClient(
            url=self.config_manager.get('QDRANT_URL'),
            api_key=self.config_manager.get('QDRANT_API_KEY'),
            timeout=60,  # Increased timeout for cloud connections
            prefer_grpc=False,  # Use HTTP for better compatibility with cloud
            https=True,  # Ensure HTTPS is used
            verify=True  # Verify SSL certificates
        )
        self._test_qdrant_connection()
        logger.info("Qdrant client initialized successfully")
        
        # Initialize Mistral LLM
        self.llm = ChatMistralAI(
            model=self.config_manager.get('MISTRAL_MODEL'),
            temperature=0.1,
            api_key=self.config_manager.get('MISTRAL_API_KEY'),
            timeout=60,
            max_retries=3
        )
        logger.info("Mistral LLM initialized successfully")
    
    def _initialize_models(self):
        """Initialize ML models with better configuration"""
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
        
        # Initialize conversation memory if available
        self._initialize_conversation_memory()
        if Config.ENABLE_IMAGE_ANALYSIS:
            try:
                self.image_analysis_service = ImageAnalysisService(self.config_manager)
                logger.info("Image analysis service initialized")
            except Exception as e:
                logger.warning(f"Image analysis service initialization failed: {e}")
                self.image_analysis_service = None
        else:
            self.image_analysis_service = None
    
    def _initialize_conversation_memory(self):
        """Initialize conversation memory manager with error handling"""
        try:
            from services.conversation_memory import ConversationMemoryManager
            self.memory_manager = ConversationMemoryManager()
            logger.info("Conversation memory manager initialized")
        except ImportError:
            logger.warning("Conversation memory not available")
            self.memory_manager = None
        except Exception as e:
            logger.warning(f"Conversation memory initialization failed: {e}")
            self.memory_manager = None
    
    def analyze_image_file(self, image_path: str) -> Dict[str, Any]:
        """Analyze a standalone image file."""
        if not self.image_analysis_service:
            return {
                'success': False,
                'error': 'Image analysis service not available'
            }
        
        return self.image_analysis_service.analyze_image(image_path)
    
    def answer_image_question(self, image_path: str, question: str) -> Dict[str, Any]:
        """Answer questions about an image."""
        if not self.image_analysis_service:
            return {
                'success': False,
                'error': 'Image analysis service not available'
            }
        
        return self.image_analysis_service.answer_question_about_image(image_path, question)

    def _test_qdrant_connection(self):
        """Test Qdrant connection"""
        try:
            collections = self.qdrant_client.get_collections()
            return True
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Qdrant: {e}")
    
    @with_retry()
    def _ensure_collection(self):
        """Setup Qdrant collection with optimizations"""
        collection_name = self.config_manager.get('COLLECTION_NAME')
        
        collections = self.qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True
                )
            )
            logger.info(f"Created collection: {collection_name}")
        
        # Create payload indexes
        self._create_payload_indexes(collection_name)
    
    def _create_payload_indexes(self, collection_name: str):
        """Create payload indexes for better performance"""
        indexes = [
            ("file_id", models.PayloadSchemaType.KEYWORD),
            ("file_type", models.PayloadSchemaType.KEYWORD),
            ("chunk_type", models.PayloadSchemaType.KEYWORD),
            ("section_hierarchy", models.PayloadSchemaType.KEYWORD),
            ("semantic_density", models.PayloadSchemaType.FLOAT),
            ("is_metadata", models.PayloadSchemaType.BOOL)
        ]
        
        for field_name, field_type in indexes:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                    wait=False
                )
            except Exception:
                pass  # Index might already exist
    
    @with_retry()
    def store_document_chunks(self, chunks: List[Document], file_id: str, 
                        document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store document with enhanced semantic chunking and error handling"""
        if not self.qdrant_client:
            return self._create_error_response('Vector database not available', 0)
        
        try:
            with performance_timer("Document processing", self.timing_stats):
                # Check if this is an image file
                file_ext = document_metadata.get('file_type', '').lower()
                is_image = file_ext in getattr(Config, 'SUPPORTED_IMAGE_FORMATS', ['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
                
                if is_image:
                    # For images, we might have OCR-extracted content
                    logger.info(f"Processing image file: {file_id}")
                    # Add image-specific metadata to chunks
                    for chunk in chunks:
                        chunk.metadata['is_image_content'] = True
                        chunk.metadata['requires_ocr'] = True
                        chunk.metadata['content_type'] = 'image_ocr'
                
                # Extract full text
                full_text = "\n\n".join([chunk.page_content for chunk in chunks])
                
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
    
    def _batch_upload_to_qdrant(self, points: List[models.PointStruct]) -> Dict[str, Any]:
        """Upload points to Qdrant in batches"""
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
        """Enhanced question answering with improved error handling"""
        if not self.qdrant_client:
            return self._create_answer_error_response('Vector database not available')
        
        try:
            start_time = time.time()
            self.stats['questions_processed'] += 1
            
            # Check for translation requests
            translation_result = self._handle_translation_request(file_id, question)
            if translation_result:
                return translation_result
            
            # Check cache
            if self.memory_manager and session_id:
                cached_result = self.memory_manager.get_cached_result(question, file_id)
                if cached_result:
                    return cached_result
            
            # Get conversation context
            conversation_context = self._get_conversation_context(session_id, file_id)
            
            # Perform search with direct file filtering
            with performance_timer("Search operation", self.timing_stats):
                search_results = self._search_with_file_filter(question, file_id)
            
            if not search_results:
                return self._create_answer_error_response('No relevant information found')
            
            # Use search results directly (already filtered by file_id)
            file_chunks = search_results
            
            if not file_chunks:
                return self._handle_no_file_chunks(file_id, search_results, start_time)
            
            # Generate answer
            with performance_timer("Answer generation", self.timing_stats):
                generation_result = self._generate_enhanced_answer(
                    question, file_chunks, conversation_history, conversation_context
                )
            
            if not generation_result['success']:
                return generation_result
            
            # Calculate confidence
            confidence = self.confidence_calculator.calculate_confidence(
                question, file_chunks, generation_result
            )
            
            # Prepare response
            result = self._prepare_final_response(
                generation_result, file_chunks, confidence, start_time
            )
            
            # Store in memory if available
            self._store_conversation_turn(session_id, file_id, question, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            return self._create_answer_error_response(str(e), processing_time)
    
    def _create_answer_error_response(self, error_msg: str, processing_time: float = 0.0) -> Dict[str, Any]:
        """Create standardized answer error response"""
        return {
            'success': False,
            'error': error_msg,
            'answer': f'I encountered an error: {error_msg}',
            'sources': [],
            'confidence_score': 0.0,
            'processing_time': processing_time
        }
    
    def _handle_translation_request(self, file_id: str, question: str) -> Optional[Dict[str, Any]]:
        """Handle translation requests - OCR removed, translation not supported"""
        try:
            from services.translation_service import TranslationService
            translation_service = TranslationService()
            
            if not translation_service.is_translation_request(question):
                return None
            
            # OCR content no longer available
            return self._create_translation_error_response(
                'Translation not supported - OCR functionality has been removed'
            )
                
        except ImportError:
            logger.warning("Translation service not available")
            return None
        except Exception as e:
            logger.error(f"Translation handling error: {str(e)}")
            return self._create_translation_error_response(f'Translation service error: {str(e)}')
    
    def _create_translation_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create translation error response"""
        return {
            'success': False,
            'error': error_msg,
            'answer': f'Translation failed: {error_msg}',
            'sources': [],
            'confidence_score': 0.0,
            'processing_time': 0.0,
            'question_type': 'translation'
        }
    
    def _get_conversation_context(self, session_id: str, file_id: str) -> str:
        """Get conversation context string"""
        if self.memory_manager and session_id:
            return self.memory_manager.get_context_string(session_id, file_id)
        return ""
    
    def _search_with_file_filter(self, query: str, file_id: str) -> List[SearchResult]:
        """Perform semantic search directly on Qdrant with file filtering"""
        try:
        # Encode the query
            query_vector = self.embedding_model.encode(query).tolist()
            
            # INCREASE limit for comprehensive retrieval, especially for "list all" type queries
            retrieval_limit = 50 if any(word in query.lower() for word in ['all', 'list', 'series', 'range', 'types']) else 20
            
            # Search Qdrant directly with file filter using new API
            search_results = self.qdrant_client.query_points(
                collection_name=self.config_manager.get('COLLECTION_NAME'),
                query=query_vector,
                query_filter=models.Filter(
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
                limit=retrieval_limit,  # Increased from 15 to 20-50
                with_payload=True
            ).points
            
            # Convert to SearchResult objects
            results = []
            for i, hit in enumerate(search_results):
                payload = hit.payload
                
                # Create SemanticChunk from payload
                # Handle key_terms - it might be a list or string
                key_terms_data = payload.get('key_terms', [])
                if isinstance(key_terms_data, str):
                    key_terms = key_terms_data.split(', ') if key_terms_data else []
                elif isinstance(key_terms_data, list):
                    key_terms = key_terms_data
                else:
                    key_terms = []
                
                chunk = SemanticChunk(
                    content=payload.get('content', ''),
                    chunk_id=payload.get('chunk_id', ''),
                    chunk_type=payload.get('content_type', 'text'),  # Note: using content_type from payload
                    section_hierarchy=[],
                    key_terms=key_terms,
                    entities=[],
                    page=payload.get('page', 1),
                    token_count=payload.get('word_count', 0),  # Using word_count as approximation
                    semantic_density=payload.get('semantic_score', 0.5),
                    file_id=payload.get('file_id', file_id)
                )
                
                # Create SearchResult
                search_result = SearchResult(
                    chunk=chunk,
                    dense_score=hit.score,
                    sparse_score=0.0,  # Not available in direct search
                    rerank_score=0.0,
                    hybrid_score=hit.score,
                    relevance_explanation=f"Direct semantic search score: {hit.score:.4f}",
                    citation_id=str(i + 1)
                )
                
                results.append(search_result)
            
            logger.info(f"Direct search found {len(results)} chunks for file {file_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error in direct search with file filter: {str(e)}")
            return []
    
    def _get_file_chunks(self, file_id: str, search_results: List[SearchResult]) -> List[SearchResult]:
        """Filter search results by file_id efficiently"""
        try:
            # Get chunk IDs from search results
            chunk_ids = [result.chunk.chunk_id for result in search_results]
            
            # Query Qdrant for file associations
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.config_manager.get('COLLECTION_NAME'),
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id)),
                        models.FieldCondition(key="is_metadata", match=models.MatchValue(value=False))
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            # Create set of valid chunk IDs for this file
            file_chunk_ids = {
                point.payload.get('chunk_id') 
                for point in scroll_result[0] 
                if point.payload.get('chunk_id')
            }
            
            # Filter and add citation IDs
            file_chunks = []
            for i, result in enumerate(search_results):
                if result.chunk.chunk_id in file_chunk_ids:
                    result.citation_id = str(len(file_chunks) + 1)
                    file_chunks.append(result)
            
            return file_chunks
            
        except Exception as e:
            logger.error(f"Error filtering chunks by file: {str(e)}")
            # Fallback: assume all chunks are from the file
            for i, result in enumerate(search_results):
                result.citation_id = str(i + 1)
            return search_results
    
    def _handle_no_file_chunks(self, file_id: str, search_results: List[SearchResult], 
                              start_time: float) -> Dict[str, Any]:
        """Handle case when no file chunks are found"""
        logger.warning(f"No file chunks found for file {file_id}")
        
        # Try direct content retrieval as fallback
        try:
            direct_chunks = self._get_direct_file_content(file_id)
            if direct_chunks:
                logger.info("Using direct content retrieval as fallback")
                return self._generate_fallback_answer(direct_chunks, start_time)
        except Exception as e:
            logger.warning(f"Direct content retrieval failed: {str(e)}")
        
        return {
            'success': False,
            'error': 'No relevant information found in this document',
            'answer': 'I could not find relevant information in this document to answer your question.',
            'sources': [],
            'confidence_score': 0.0,
            'processing_time': time.time() - start_time,
            'debug_info': {
                'file_id': file_id,
                'search_results_count': len(search_results),
                'file_chunks_count': 0
            }
        }
    
    def _get_direct_file_content(self, file_id: str) -> List[Dict[str, Any]]:
        """Get direct file content from Qdrant"""
        scroll_result = self.qdrant_client.scroll(
            collection_name=self.config_manager.get('COLLECTION_NAME'),
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id)),
                    models.FieldCondition(key="is_metadata", match=models.MatchValue(value=False))
                ]
            ),
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            {'content': point.payload.get('content', '')}
            for point in scroll_result[0]
            if point.payload.get('content')
        ]
    
    def _generate_fallback_answer(self, direct_chunks: List[Dict[str, Any]], 
                                 start_time: float) -> Dict[str, Any]:
        """Generate fallback answer from direct content"""
        try:
            sample_content = "\n".join([
                chunk['content'][:300] for chunk in direct_chunks[:3]
            ])
            
            prompt = f"""Based on this content, provide a helpful response:

Content: {sample_content}...

Please provide a brief summary of what information is available."""
            
            response = self.llm.invoke(prompt)
            
            return {
                'success': True,
                'answer': response.content.strip(),
                'sources': [],
                'confidence_score': 0.3,
                'processing_time': time.time() - start_time,
                'question_type': 'direct_retrieval'
            }
            
        except Exception as e:
            logger.warning(f"Fallback answer generation failed: {str(e)}")
            raise
    
    @with_retry()
    def _generate_enhanced_answer(self, question: str, search_results: List[SearchResult],
                                conversation_history: List[Dict] = None, 
                                conversation_context: str = "") -> Dict[str, Any]:
        """Generate enhanced answer with retry logic"""
        try:
            # Build context efficiently
            context_str = self._build_context_string(search_results)
            
            # Add conversation context
            context_section = ""
            if conversation_context:
                context_section = f"\nRecent conversation:\n{conversation_context}"
            elif conversation_history:
                context_section = self._build_history_context(conversation_history)
            
            # Create prompt
            prompt = self._create_answer_prompt(question, context_str, context_section)
            
            # Generate response
            response = self.llm.invoke(prompt)
            answer_content = response.content.strip()
            
            # Check for API capacity issues (empty or minimal responses)
            if not answer_content or answer_content in ["", ".", "..", "...", "....", "....."]:
                logger.warning("LLM returned minimal response, likely due to API capacity issues")
                raise Exception("LLM returned minimal response - possible API capacity issue")
            
            answer = self._post_process_answer(answer_content, search_results)
            
            return {
                'success': True,
                'answer': answer,
                'sources_used': len(search_results),
                'context_length': len(context_str)
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Answer generation failed: {error_msg}")
            
            # Check for specific API errors
            if "429" in error_msg or "capacity exceeded" in error_msg.lower():
                logger.error("Mistral API capacity exceeded - using enhanced fallback")
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                logger.error("Mistral API authentication failed")
            
            # Create simple, clean fallback answer
            fallback_answer = self._create_simple_fallback_answer(error_msg)

            logger.info(f"Fallback answer used due to: {error_msg}")
            return {
                'success': True,
                'answer': fallback_answer,
                'sources_used': len(search_results),
                'context_length': 0,
                'fallback_used': True,
                'fallback_reason': error_msg
            }
    
    def _build_context_string(self, search_results: List[SearchResult]) -> str:
        """Build context string from search results efficiently"""
        context_parts = []
    
        # For comprehensive queries, use more sources
        max_sources = 10 if len(search_results) > 5 else 5
        
        for result in search_results[:max_sources]:  # Increased from 5 to 10
            chunk = result.chunk
            context = f"[Source {result.citation_id}]"
            
            if chunk.section_hierarchy:
                context += f" ({chunk.section_hierarchy[-1]})"
            
            # For series/listing queries, use full content
            content = chunk.content[:1200]  # Increased from 800 to 1200
            if len(chunk.content) > 1200:
                content += "..."
            
            context += f"\n{content}"
            context_parts.append(context)
        
        return "\n\n".join(context_parts)
    
    def _build_history_context(self, conversation_history: List[Dict]) -> str:
        """Build conversation history context"""
        if not conversation_history:
            return ""
        
        history_parts = []
        for turn in conversation_history[-3:]:  # Last 3 turns
            if 'question' in turn and 'answer' in turn:
                history_parts.append(f"Q: {turn['question'][:100]}...")
                history_parts.append(f"A: {turn['answer'][:150]}...")
        
        if history_parts:
            return f"\nRecent conversation:\n" + "\n".join(history_parts)
        return ""
    
    def _create_answer_prompt(self, question: str, context_str: str, context_section: str = "") -> str:
        """Create the answer prompt for the LLM"""
        # Detect if this is a comprehensive listing question
        is_listing_query = any(word in question.lower() for word in ['all', 'list', 'series', 'range', 'types', 'found'])
        
        if is_listing_query:
            return f"""You are an expert document analyst. Answer the question using ALL the provided sources comprehensively.

INSTRUCTIONS FOR COMPREHENSIVE LISTING:
1. SCAN ALL SOURCES carefully for complete information
2. LIST ALL series, types, models, or categories mentioned across ALL sources
3. Do NOT miss or omit any series/types found in the sources
4. Group similar information together
5. If multiple sources mention the same series, consolidate the information
6. Be thorough and complete - this is a comprehensive listing request

{context_section}

SOURCES:
{context_str}

QUESTION: {question}

Provide a COMPLETE and COMPREHENSIVE answer that includes ALL series/types found in the sources:"""
        else:
            # Use existing prompt for regular questions
            return f"""You are an expert document analyst. Answer the question using ONLY the provided sources.

INSTRUCTIONS:
1. Keep answers BRIEF and DIRECT - maximum 3-4 sentences
2. Extract specific information like names, dates, numbers, locations, or key facts from the sources
3. Focus on exact details when asked for specific data
4. Use only essential information from sources
5. Be precise and factual - no elaboration
6. If sources lack information, state it concisely
7. Skip introductory phrases - get straight to the point

{context_section}

SOURCES:
{context_str}

QUESTION: {question}

Provide a concise, direct answer (maximum 3-4 sentences):"""
    
    def _create_simple_fallback_answer(self, error_msg: str) -> str:
        """Create simple, clean fallback answer for API issues"""
        # Check for specific API errors and provide appropriate message
        if "429" in error_msg or "capacity exceeded" in error_msg.lower():
            return "The AI service is currently experiencing high demand. Please try again in a few moments."
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            return "The AI service authentication has expired. Please contact support or try again later."
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            return "The AI service is temporarily unavailable. Please try again in a few moments."
        else:
            return "The AI service is temporarily unavailable. Please try again later."
    
    def _detect_missing_series(self, question: str, search_results: List[SearchResult]) -> List[str]:
        """Detect potentially missing series mentioned in question but not found in results"""
        question_lower = question.lower()
        
        # Extract series names from the question
        mentioned_series = []
        series_patterns = [
            r'([a-z]-series)',
            r'(instaview\s+series)',
            r'(dispenser\s+series)',
            r'(french\s+door)',
            r'(\w+\s+series)'
        ]
        
        for pattern in series_patterns:
            matches = re.findall(pattern, question_lower)
            mentioned_series.extend(matches)
        
        # Check which series are actually found in search results
        found_series = []
        for result in search_results:
            content_lower = result.chunk.content.lower()
            for series in mentioned_series:
                if series in content_lower:
                    found_series.append(series)
        
        # Return potentially missing series
        missing = [s for s in mentioned_series if s not in found_series]
        return missing

    def _create_basic_content_fallback(self, question: str, search_results: List[SearchResult]) -> str:
        """Create basic content-based fallback answer"""
        try:
            # Get the most relevant content from top search results
            relevant_content = []
            question_words = set(question.lower().split())
            
            for result in search_results[:3]:  # Use top 3 results
                content = result.chunk.content
                if content and len(content.strip()) > 20:
                    # Find sentences that contain question keywords
                    sentences = content.split('. ')
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 30:  # Meaningful sentence length
                            sentence_words = set(sentence.lower().split())
                            # Check for keyword overlap
                            overlap = len(question_words.intersection(sentence_words))
                            if overlap > 0:
                                relevant_content.append(sentence)
                                if len(relevant_content) >= 3:  # Limit to 3 sentences
                                    break
                
                if len(relevant_content) >= 3:
                    break
            
            if relevant_content:
                # Clean and format the answer
                answer = '. '.join(relevant_content[:3])
                if not answer.endswith('.'):
                    answer += '.'
                
                # Add context note
                answer += " (Note: This information is extracted directly from the document due to a temporary service limitation.)"
                
                logger.info(f"Generated fallback answer: {answer[:100]}...")
                return answer
            else:
                # If no relevant sentences found, use chunk content directly
                if search_results and search_results[0].chunk.content:
                    content = search_results[0].chunk.content[:500]
                    return f"Based on the document content: {content}... (Note: Full analysis unavailable due to temporary service limitations.)"
                else:
                    return "I found relevant content in the document but cannot provide a detailed answer due to temporary service limitations. Please try again in a moment."
                    
        except Exception as e:
            logger.error(f"Error in basic content fallback: {str(e)}")
            return "I found the document but encountered an error while processing your question. Please try again or rephrase your query."

    def _create_summary_fallback(self, search_results: List[SearchResult]) -> str:
        """Create a summary fallback when LLM is unavailable"""
        if not search_results:
            return "No content available to summarize."
        
        logger.info(f"Creating summary fallback from {len(search_results)} results")
        
        # Combine content from all search results
        content_parts = []
        for result in search_results[:5]:  # Use top 5 results
            content = result.chunk.content.strip()
            if content and len(content) > 20:  # Ensure meaningful content
                content_parts.append(content)
        
        if not content_parts:
            return "No readable content found in the document."
        
        # Create a basic summary from the content
        full_content = " ".join(content_parts)
        
        # Extract key information patterns
        import re
        
        summary_parts = []
        
        # Look for names (people)
        names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', full_content)
        if names:
            unique_names = list(set(names[:3]))
            summary_parts.append(f"Key people mentioned: {', '.join(unique_names)}")
        
        # Look for organizations
        organizations = re.findall(r'\b(?:[A-Z][a-z]+\s+)*(?:Company|Corporation|Organization|Department|Agency|Group|Team|Foundation|Association|Institute|University)\b', full_content)
        if organizations:
            unique_orgs = list(set(organizations[:2]))
            summary_parts.append(f"Organizations: {', '.join(unique_orgs)}")
        
        # Look for dates
        dates = re.findall(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{4}|\b[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b', full_content)
        if dates:
            unique_dates = list(set(dates[:2]))
            summary_parts.append(f"Important dates: {', '.join(unique_dates)}")
        
        # Look for numbers/amounts/quantities
        numbers = re.findall(r'\b(?:\d+%|\$\d+(?:,\d{3})*|\d+\.\d+|\d{1,3}(?:,\d{3})*)\b', full_content)
        if numbers:
            unique_numbers = list(set(numbers[:3]))
            summary_parts.append(f"Key figures: {', '.join(unique_numbers)}")
        
        # If we found specific patterns, use them
        if summary_parts:
            summary = ". ".join(summary_parts) + "."
        else:
            # Fall back to first portion of content
            summary = f"This document discusses: {full_content[:300]}..."
            if not summary.endswith('.'):
                summary += "."
        
        # Add service limitation note
        summary += " (Note: Detailed analysis unavailable due to temporary service limitations.)"
        
        logger.info(f"Generated summary fallback: {summary[:100]}...")
        return summary
        
    def _extract_names_info(self, search_results: List[SearchResult]) -> str:
        """Extract names and people information from search results"""
        import re
        names_found = []
        
        for result in search_results:
            content = result.chunk.content
            
            # Look for person names (proper capitalization pattern)
            names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            names_found.extend(names)
            
            # Look for titles with names
            titled_names = re.findall(r'\b(?:Mr|Ms|Mrs|Dr|Prof|Professor)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            names_found.extend(titled_names)
        
        if names_found:
            unique_names = list(set(names_found))[:5]  # Top 5 unique names
            return f"Names mentioned in the document: {', '.join(unique_names)}."
        
        return "I found the document but couldn't extract specific names from the available content."
        
    def _extract_contact_info(self, search_results: List[SearchResult]) -> str:
        """Extract contact information from search results"""
        import re
        contact_info = []
        
        for result in search_results:
            content = result.chunk.content
            
            # Extract emails
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
            if emails:
                contact_info.append(f"Email: {emails[0]}")
            
            # Extract phone numbers (various formats)
            phones = re.findall(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', content)
            if phones:
                contact_info.append(f"Phone: {phones[0]}")
            
            # Extract websites and URLs
            websites = re.findall(r'(?:www\.|https?://)[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}(?:/\S*)?', content)
            if websites:
                contact_info.append(f"Website: {websites[0]}")
                
            # Extract addresses (basic pattern)
            addresses = re.findall(r'\b\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr)\b', content)
            if addresses:
                contact_info.append(f"Address: {addresses[0][:50]}...")
        
        if contact_info:
            return "Contact information found: " + ", ".join(contact_info[:4]) + "."
        return "I found the document but couldn't extract specific contact information from the available content."
    
    def _extract_date_info(self, search_results: List[SearchResult]) -> str:
        """Extract date information from search results"""
        import re
        
        for result in search_results:
            content = result.chunk.content
            
            # Look for date patterns
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{1,2}-\d{1,2}-\d{4}\b',
                r'\b[A-Za-z]+ \d{1,2}, \d{4}\b'
            ]
            
            for pattern in date_patterns:
                dates = re.findall(pattern, content)
                if dates:
                    return f"Date information found: {dates[0]}."
        
        return "I found the document but couldn't extract specific date information from the available content."
    
    def _extract_key_information(self, question: str, search_results: List[SearchResult]) -> str:
        """Extract key information when sentence matching fails"""
        # Get the most relevant content
        if search_results:
            content = search_results[0].chunk.content[:300]  # First 300 chars
            return f"Based on the document content: {content}... (Note: Full answer unavailable due to service limitations)"
        else:
            return "I couldn't find relevant information to answer your question."
    
    def _post_process_answer(self, answer: str, search_results: List[SearchResult]) -> str:
        """Post-process answer for quality and consistency"""
        # Remove markdown formatting
        answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)
        answer = re.sub(r'\*([^*]+)\*', r'\1', answer)
        
        # Clean spacing
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Limit to 4 sentences
        sentences = answer.split('. ')
        if len(sentences) > 4:
            answer = '. '.join(sentences[:4]) + '.'
        
        # Ensure proper ending
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        return answer
    
    def _prepare_final_response(self, generation_result: Dict[str, Any], 
                               file_chunks: List[SearchResult], confidence: float,
                               start_time: float) -> Dict[str, Any]:
        """Prepare final response with all metadata"""
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'answer': generation_result['answer'],
            'confidence_score': confidence,
            'sources': self._prepare_source_information(file_chunks),
            'processing_time': processing_time,
            'retrieval_stats': {
                'total_candidates': len(file_chunks),
                'file_matches': len(file_chunks),
                'avg_hybrid_score': np.mean([r.hybrid_score for r in file_chunks]),
                'chunk_types': Counter(r.chunk.chunk_type for r in file_chunks),
                'search_explanation': self._explain_search_strategy(file_chunks)
            },
            'fallback_used': generation_result.get('fallback_used', False)
        }
    
    def _prepare_source_information(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Prepare source information for response"""
        sources = []
        
        for result in search_results:
            chunk = result.chunk
            
            source_info = {
                'citation_id': result.citation_id,
                'page': chunk.page,
                'chunk_type': chunk.chunk_type,
                'section_path': ' > '.join(chunk.section_hierarchy) if chunk.section_hierarchy else 'Document',
                'content_preview': chunk.content[:400] + '...' if len(chunk.content) > 400 else chunk.content,
                'relevance_scores': {
                    'dense_score': round(result.dense_score, 3),
                    'sparse_score': round(result.sparse_score, 3),
                    'hybrid_score': round(result.hybrid_score, 3),
                    'rerank_score': round(result.rerank_score, 3) if result.rerank_score else None
                },
                'key_terms': chunk.key_terms[:5],
                'entities': chunk.entities[:3],
                'semantic_density': round(chunk.semantic_density, 3),
                'relevance_explanation': result.relevance_explanation
            }
            
            sources.append(source_info)
        
        return sources
    
    def _explain_search_strategy(self, search_results: List[SearchResult]) -> str:
        """Explain search strategy used"""
        if not search_results:
            return "No relevant chunks found"
        
        hybrid_scores = [r.hybrid_score for r in search_results]
        dense_scores = [r.dense_score for r in search_results]
        sparse_scores = [r.sparse_score for r in search_results]
        
        avg_dense = np.mean(dense_scores)
        avg_sparse = np.mean(sparse_scores)
        
        explanations = []
        
        if avg_dense > avg_sparse * 1.5:
            explanations.append("Semantic similarity was primary ranking factor")
        elif avg_sparse > avg_dense * 1.5:
            explanations.append("Keyword matching was primary ranking factor")
        else:
            explanations.append("Balanced semantic and keyword matching")
        
        chunk_types = Counter(r.chunk.chunk_type for r in search_results)
        dominant_type = chunk_types.most_common(1)[0]
        if dominant_type[1] > len(search_results) * 0.6:
            explanations.append(f"Focused on {dominant_type[0]} content")
        
        return " | ".join(explanations)
    
    def _store_conversation_turn(self, session_id: str, file_id: str, 
                                question: str, result: Dict[str, Any]):
        """Store conversation turn in memory manager"""
        if not self.memory_manager or not session_id:
            return
        
        try:
            from services.conversation_memory import ConversationTurn
            
            turn = ConversationTurn(
                question=question,
                answer=result['answer'],
                timestamp=datetime.now(),
                confidence_score=result['confidence_score'],
                question_type=self._classify_question_type(question),
                processing_time=result['processing_time'],
                sources_used=[s.get('section_path', 'Document') for s in result['sources']],
                session_id=session_id,
                file_id=file_id
            )
            
            self.memory_manager.add_turn(turn)
            self.memory_manager.cache_query(question, result, session_id, file_id)
            
        except Exception as e:
            logger.warning(f"Failed to store conversation turn: {str(e)}")
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type for tracking"""
        question_lower = question.lower()
        
        type_patterns = {
            'definition': ['what is', 'what are', 'define', 'definition'],
            'explanation': ['how', 'explain', 'describe'],
            'temporal': ['when', 'date', 'time'],
            'location': ['where', 'location', 'place'],
            'person': ['who', 'person', 'people'],
            'causal': ['why', 'reason', 'because'],
            'quantitative': ['how much', 'how many', 'count', 'number'],
            'listing': ['list', 'enumerate', 'show me']
        }
        
        for q_type, keywords in type_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                return q_type
        
        return 'direct_question' if '?' in question else 'general'
    
    def _ensure_search_index(self, file_id: str = None):
        """Ensure search index is populated for the file"""
        try:
            # Check if we already have content for this file
            if file_id and self.search_engine.chunks_metadata:
                file_chunks_in_index = [
                    chunk for chunk in self.search_engine.chunks_metadata 
                    if hasattr(chunk, 'file_id') and chunk.file_id == file_id
                ]
                
                if file_chunks_in_index:
                    logger.debug(f"Search index ready for file {file_id}")
                    return
            
            logger.info(f"Rebuilding search index for file {file_id}...")
            
            # Get chunks from Qdrant
            filter_conditions = [
                models.FieldCondition(key="is_metadata", match=models.MatchValue(value=False))
            ]
            
            if file_id:
                filter_conditions.append(
                    models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))
                )
            
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.config_manager.get('COLLECTION_NAME'),
                scroll_filter=models.Filter(must=filter_conditions),
                limit=1000,
                with_payload=True,
                with_vectors=True
            )
            
            points = scroll_result[0]
            if not points:
                logger.warning(f"No chunks found for file {file_id}")
                return
            
            # Convert to SemanticChunk objects
            semantic_chunks = []
            for point in points:
                payload = point.payload
                
                chunk = SemanticChunk(
                    content=payload.get('content', ''),
                    chunk_id=payload.get('chunk_id', ''),
                    chunk_type=payload.get('chunk_type', 'text'),
                    section_hierarchy=payload.get('section_hierarchy', []),
                    key_terms=payload.get('key_terms', []),
                    entities=payload.get('entities', []),
                    page=payload.get('page', 1),
                    token_count=payload.get('token_count', 0),
                    semantic_density=payload.get('semantic_density', 0.5),
                    file_id=payload.get('file_id', file_id)
                )
                
                semantic_chunks.append(chunk)
            
            # Rebuild search index
            self.search_engine.index_chunks(semantic_chunks)
            logger.info(f"Search index rebuilt with {len(semantic_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to ensure search index: {str(e)}")
    
    # UTILITY METHODS
    
    def document_exists(self, file_id: str) -> bool:
        """Check if document exists in the system"""
        try:
            result = self.qdrant_client.count(
                collection_name=self.config_manager.get('COLLECTION_NAME'),
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id)),
                        models.FieldCondition(key="is_metadata", match=models.MatchValue(value=False))
                    ]
                )
            )
            return result.count > 0
        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            return False
    
    def delete_document(self, file_id: str) -> Dict[str, Any]:
        """Delete document from the system"""
        try:
            self.qdrant_client.delete(
                collection_name=self.config_manager.get('COLLECTION_NAME'),
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))]
                    )
                ),
                wait=True
            )
            
            logger.info(f"Document {file_id} deleted successfully")
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def generate_sample_questions(self, chunks: List[Document], 
                                document_metadata: Dict[str, Any] = None) -> List[str]:
        """Generate sample questions for document exploration"""
        try:
            # Use subset for efficiency
            sample_chunks = chunks[:5] if len(chunks) > 5 else chunks
            
            # Extract key information
            important_terms = set()
            for chunk in sample_chunks:
                content = chunk.page_content
                caps_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
                important_terms.update(caps_terms[:3])
            
            # Build context
            context_parts = []
            for i, chunk in enumerate(sample_chunks):
                preview = chunk.page_content[:300]
                if len(chunk.page_content) > 300:
                    preview += "..."
                context_parts.append(f"Section {i+1}: {preview}")
            
            context_str = "\n\n".join(context_parts)
            terms_context = ""
            
            if important_terms:
                terms_context = f"\nKey terms: {', '.join(list(important_terms)[:5])}"
            
            prompt = f"""Generate 5 diverse questions for document exploration:

{context_str}{terms_context}

Create questions that:
1. Ask about main concepts
2. Request specific details
3. Compare different aspects
4. Ask about applications
5. Explore relationships

Format as numbered list:"""
            
            response = self.llm.invoke(prompt)
            
            # Parse response
            questions = []
            for line in response.content.split('\n'):
                line = line.strip()
                cleaned = re.sub(r'^\d+\.\s*', '', line)
                if cleaned and len(cleaned) > 10 and '?' in cleaned:
                    questions.append(cleaned)
            
            # Ensure we have 5 questions
            generic_questions = [
                "What are the main topics covered?",
                "What key concepts are explained?", 
                "What specific details are provided?",
                "What are the practical applications?",
                "How do the concepts relate to each other?"
            ]
            
            while len(questions) < 5:
                questions.extend(generic_questions)
            
            return questions[:5]
            
        except Exception as e:
            logger.warning(f"Sample question generation failed: {str(e)}")
            return [
                "What is this document about?",
                "What are the main points?",
                "What specific information is provided?",
                "What are the key takeaways?", 
                "How can this information be applied?"
            ]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'questions_processed': self.stats['questions_processed'],
            'documents_processed': self.stats['documents_processed'],
            'avg_processing_time': np.mean(self.timing_stats.get('timings', [0])),
            'search_engine_stats': {
                'chunks_indexed': len(self.search_engine.chunks_metadata),
                'has_sparse_index': self.search_engine.sparse_index is not None,
                'has_reranker': self.search_engine.reranker is not None
            },
            'config_stats': {
                'embedding_model': self.config_manager.get('EMBEDDING_MODEL', 'unknown'),
                'chunk_size': self.config_manager.get('CHUNK_SIZE', 300),
                'retrieval_k': self.config_manager.get('RETRIEVAL_K', 15)
            }
        }
    
    def get_conversation_stats(self, session_id: str) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self.memory_manager:
            return {'error': 'Conversation memory not available'}
        
        return self.memory_manager.get_session_stats(session_id)
    
    def clear_conversation_history(self, session_id: str) -> bool:
        """Clear conversation history"""
        if not self.memory_manager:
            return False
        
        self.memory_manager.cleanup_session(session_id)
        return True

# FACTORY FUNCTION

def create_enhanced_rag_service(config) -> EnhancedRAGService:
    """Factory function to create enhanced RAG service"""
    return EnhancedRAGService(config)

# BACKWARD COMPATIBILITY

HybridRAGService = EnhancedRAGService