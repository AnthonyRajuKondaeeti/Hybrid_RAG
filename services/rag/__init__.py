"""
RAG Service Module - Modular RAG Components

This module provides a clean, modular implementation of the RAG (Retrieval-Augmented Generation) system.
All components are split into focused modules for better maintainability and testing.
"""

# Re-export main classes for backward compatibility
from .models import SemanticChunk, SearchResult
from .protocols import ChunkProcessor, SearchEngine
from .core_rag_service import EnhancedRAGService
from .chunking_service import SemanticChunker
from .search_engines import HybridSearchEngine, HAS_BM25
from .confidence_calculator import ConfidenceCalculator
from .image_analysis import ImageAnalysisService
from .text_processing import TextProcessor, HAS_NLTK

__all__ = [
    'SemanticChunk',
    'SearchResult', 
    'ChunkProcessor',
    'SearchEngine',
    'EnhancedRAGService',
    'SemanticChunker',
    'HybridSearchEngine',
    'ConfidenceCalculator',
    'ImageAnalysisService',
    'TextProcessor'
]