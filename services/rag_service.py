"""
BACKWARD COMPATIBILITY LAYER - RAG Service

This file maintains backward compatibility with existing applications while
importing from the new modular RAG components.

All imports that your main applications use should work exactly as before.
"""

# Import all components from the modular structure for backward compatibility
from .rag.models import SemanticChunk, SearchResult
from .rag.protocols import ChunkProcessor, SearchEngine
from .rag.core_rag_service import EnhancedRAGService
from .rag.chunking_service import SemanticChunker
from .rag.search_engines import HybridSearchEngine, HAS_BM25
from .rag.confidence_calculator import ConfidenceCalculator
from .rag.image_analysis import ImageAnalysisService
from .rag.text_processing import TextProcessor, HAS_NLTK

# Import utilities
from .utils.retry_utils import RetryConfig, with_retry, performance_timer
from .utils.config_manager import ConfigManager

# Ensure all the classes your main applications expect are available
__all__ = [
    # Main service class
    'EnhancedRAGService',
    
    # Data models
    'SemanticChunk',
    'SearchResult',
    
    # Protocols
    'ChunkProcessor', 
    'SearchEngine',
    
    # Core components
    'SemanticChunker',
    'HybridSearchEngine',
    'ConfidenceCalculator',
    'ImageAnalysisService',
    'TextProcessor',
    'ConfigManager',
    
    # Utilities
    'RetryConfig',
    'with_retry',
    'performance_timer',
    
    # Feature flags
    'HAS_BM25',
    'HAS_NLTK'
]

# Legacy alias for any old code that might use this
RAGService = EnhancedRAGService