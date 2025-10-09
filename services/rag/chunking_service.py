"""
Semantic Chunking Service

This module provides intelligent document chunking that preserves semantic boundaries
and creates meaningful chunks for the RAG system.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from .models import SemanticChunk
from .text_processing import TextProcessor

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

__all__ = ['SemanticChunker']