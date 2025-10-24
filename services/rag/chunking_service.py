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
    """Advanced semantic chunking with improved efficiency and comprehensive content capture"""

    # FIXED: More selective header patterns - only match CLEAR headers
    HEADER_PATTERNS = [
        re.compile(r'^(#{1,6})\s+(.+)$'),                              # Markdown headers: # Title
        re.compile(r'^(\d+\.(?:\d+\.)*)\s+(.{3,})$'),                  # Numbered headers: 1.2.3 Title (min 3 chars)
        re.compile(r'^([A-Z][A-Z\s]{5,30})$'),                         # ALL CAPS HEADERS (6-30 chars, standalone)
        re.compile(r'^(Chapter|Section|Part)\s+\d+:?\s*(.*)$', re.IGNORECASE),  # Chapter/Section markers
    ]
    
    LIST_PATTERNS = [
        re.compile(r'^\s*(?:[-•*]|\d+\.|\([a-z]\))\s+'),
        re.compile(r'^\s*(?:[A-Z]\.|\d+\))\s+'),
    ]

    def __init__(self, target_chunk_size: int = 800, overlap_size: int = 100):
        """Initialize with LARGER chunk sizes for comprehensive content"""
        self.target_chunk_size = target_chunk_size  # Increased from 300 to 800
        self.overlap_size = overlap_size  # Increased from 50 to 100
        self.text_processor = TextProcessor()
        self.min_chunk_size = 200  # Minimum chunk size to avoid tiny fragments

    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[SemanticChunk]:
        """Create semantically coherent chunks with comprehensive content"""
        if not text.strip():
            return []
            
        metadata = metadata or {}
        
        # NEW: Use paragraph-based chunking for better content preservation
        if self._should_use_paragraph_chunking(text):
            return self._chunk_by_paragraphs(text, metadata)
        
        # Parse document structure (more conservative)
        structured_content = self._parse_document_structure(text)
        
        # Create semantic chunks
        chunks = []
        for section in structured_content:
            section_chunks = self._create_section_chunks(section, metadata)
            chunks.extend(section_chunks)
        
        # Calculate semantic metrics in batch
        self._calculate_semantic_metrics_batch(chunks)
        
        return chunks

    def _should_use_paragraph_chunking(self, text: str) -> bool:
        """Determine if document should use paragraph-based chunking (for catalogs, brochures, etc.)"""
        # Check for catalog/marketing content indicators
        marketing_indicators = ['features', 'specifications', 'capacity', 'technology', 
                               'design', 'smart', 'connectivity', 'efficiency']
        
        text_lower = text.lower()
        indicator_count = sum(1 for indicator in marketing_indicators if indicator in text_lower)
        
        # If 3+ marketing indicators and not too structured, use paragraph chunking
        lines = text.split('\n')
        header_like_lines = sum(1 for line in lines[:50] if line.strip() and line.strip().isupper())
        
        return indicator_count >= 3 and header_like_lines < 10

    def _chunk_by_paragraphs(self, text: str, metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Chunk by paragraphs for catalog/marketing content - preserves comprehensive context"""
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk_text = []
        current_token_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = len(para.split())
            
            # If adding this paragraph exceeds target, save current chunk
            if current_token_count + para_tokens > self.target_chunk_size and current_chunk_text:
                # Only create chunk if it's substantial
                if current_token_count >= self.min_chunk_size:
                    chunk_content = '\n\n'.join(current_chunk_text)
                    chunks.append(self._create_simple_chunk(chunk_content, metadata))
                
                # Start new chunk with overlap (keep last paragraph)
                if len(current_chunk_text) > 1:
                    current_chunk_text = [current_chunk_text[-1], para]
                    current_token_count = len(current_chunk_text[-1].split()) + para_tokens
                else:
                    current_chunk_text = [para]
                    current_token_count = para_tokens
            else:
                current_chunk_text.append(para)
                current_token_count += para_tokens
        
        # Add final chunk
        if current_chunk_text and current_token_count >= 50:  # At least 50 tokens
            chunk_content = '\n\n'.join(current_chunk_text)
            chunks.append(self._create_simple_chunk(chunk_content, metadata))
        
        return chunks

    def _create_simple_chunk(self, content: str, metadata: Dict[str, Any]) -> SemanticChunk:
        """Create a chunk with minimal processing for speed"""
        return SemanticChunk(
            content=content,
            chunk_type='section',
            section_hierarchy=[],
            key_terms=self.text_processor.extract_key_terms(content),
            entities=self.text_processor.extract_entities(content),
            page=metadata.get('page', 1),
            file_id=metadata.get('file_id')
        )

    def _parse_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """Parse document into hierarchical structure - MORE CONSERVATIVE"""
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
            
            # FIXED: More conservative header detection
            header_match = self._match_header(line_stripped)
            
            # Only treat as header if it's clear and substantial
            if header_match and self._is_valid_header(line_stripped):
                # Save previous section (with minimum size check)
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if len(content.split()) >= 30:  # At least 30 words
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

    def _is_valid_header(self, line: str) -> bool:
        """Check if line is truly a header and not just content"""
        # Reject if too long (headers should be concise)
        if len(line) > 100:
            return False
        
        # Reject if contains certain punctuation patterns (indicates content, not header)
        if line.count('.') > 2 or line.count(',') > 2:
            return False
        
        # Reject if it's all caps but too short (might be acronym)
        if line.isupper() and len(line) < 6:
            return False
        
        return True

    def _match_header(self, line: str) -> Optional[Tuple[int, str]]:
        """Match header patterns - more conservative"""
        for i, pattern in enumerate(self.HEADER_PATTERNS):
            match = pattern.match(line)
            if match:
                try:
                    level_indicator = match.group(1)
                    title = match.group(2).strip() if match.lastindex >= 2 else ""
                    
                    # If title is empty, use the level indicator as title
                    if not title:
                        title = level_indicator.strip()
                    
                    # Calculate level
                    if level_indicator.startswith('#'):
                        level = len(level_indicator)
                    elif '.' in level_indicator:
                        level = len(level_indicator.split('.'))
                    else:
                        level = 1
                    
                    return level, title
                except (IndexError, AttributeError):
                    continue
        
        return None

    def _classify_content_type(self, content: str) -> str:
        """Classify content type"""
        if not content.strip():
            return 'empty'
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Check for lists
        list_lines = sum(1 for line in lines if any(pattern.match(line) for pattern in self.LIST_PATTERNS))
        if list_lines > len(lines) * 0.5:
            return 'list'
        
        # Check for tables
        if '|' in content and content.count('|') > 4:
            return 'table'
        
        return 'paragraph' if len(lines) == 1 else 'section'

    def _update_hierarchy(self, hierarchy: List[str], level: int, title: str):
        """Update section hierarchy"""
        while len(hierarchy) >= level:
            hierarchy.pop()
        hierarchy.append(title)

    def _create_section_chunks(self, section: Dict[str, Any], metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Create chunks from section - PRESERVE MORE CONTENT"""
        content = section['content']
        if not content.strip():
            return []
        
        # ✅ CHANGED: More generous size limits
        word_count = len(content.split())
        
        # Keep larger sections as single chunks
        if word_count <= self.target_chunk_size * 1.5:  # Allow 1.5x target size
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
        """Chunk list content - MORE GENEROUS"""
        chunks = []
        lines = [line for line in content.split('\n') if line.strip()]
        
        current_chunk_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = len(line.split())
            is_new_item = any(pattern.match(line.strip()) for pattern in self.LIST_PATTERNS)
            
            # CHANGED: Higher threshold before splitting (0.9 instead of 0.7)
            if (is_new_item and current_tokens > self.target_chunk_size * 0.9 and current_chunk_lines):
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
        """Chunk content by semantic boundaries - PRESERVE MORE CONTEXT"""
        chunks = []
        sentences = self.text_processor.split_sentences(content)
        
        current_chunk_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            # CHANGED: Allow chunks up to 1.2x target size
            if current_tokens + sentence_tokens > self.target_chunk_size * 1.2 and current_chunk_sentences:
                chunk_content = ' '.join(current_chunk_sentences)
                chunks.append(self._create_chunk(chunk_content, section, metadata))
                
                # Start new chunk with MORE overlap (keep last 5 sentences instead of 3)
                overlap_sentences = max(0, len(current_chunk_sentences) - 5)
                current_chunk_sentences = current_chunk_sentences[overlap_sentences:] + [sentence]
                current_tokens = sum(len(s.split()) for s in current_chunk_sentences)
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk (even if small)
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