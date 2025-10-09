"""
RAG Protocols and Interfaces

This module defines the interfaces and protocols used by the RAG system components.
"""

from typing import Protocol, List, Dict, Any
from .models import SemanticChunk, SearchResult

class ChunkProcessor(Protocol):
    """Protocol for document chunking strategies"""
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[SemanticChunk]:
        ...

class SearchEngine(Protocol):
    """Protocol for search implementations"""
    def index_chunks(self, chunks: List[SemanticChunk]) -> None:
        ...
    
    def search(self, query: str, top_k: int) -> List[SearchResult]:
        ...

__all__ = ['ChunkProcessor', 'SearchEngine']