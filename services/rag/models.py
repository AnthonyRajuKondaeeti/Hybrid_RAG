"""
RAG Data Models and Core Data Structures

This module contains the data classes and models used throughout the RAG system.
"""

import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

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

__all__ = ['SemanticChunk', 'SearchResult']