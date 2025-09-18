# services/document_service.py
import fitz
import pdfplumber
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
import numpy as np
from collections import Counter
import logging

import docx
import openpyxl
import csv
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from rtfparse.parser import Rtf_Parser
from odf import text, teletype
from odf.opendocument import load
from pptx import Presentation

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentService:
    """Service for processing PDF documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
    
    def process_document(self, filepath: str, file_id: str, original_filename: str) -> Dict[str, Any]:
        """Process PDF document and extract content"""
        try:
            logger.info(f"Processing document: {original_filename}")
            
            # Extract text content
            text_extraction_result = self._extract_text_content(filepath)
            
            if not text_extraction_result['success']:
                return {
                    'success': False,
                    'error': text_extraction_result['error']
                }
            
            documents = text_extraction_result['documents']
            
            # Create chunks
            chunks = self._create_semantic_chunks(documents, file_id, original_filename)
            
            # Extract metadata
            metadata = self._extract_document_metadata(filepath, original_filename, documents)
            
            # Generate processing stats
            processing_stats = {
                'total_pages': len(documents),
                'total_chunks': len(chunks),
                'total_characters': sum(len(doc.page_content) for doc in documents),
                'total_words': sum(len(doc.page_content.split()) for doc in documents),
                'processing_method': text_extraction_result['method'],
                'processed_at': datetime.now().isoformat()
            }
            
            logger.info(f"Document processed successfully: {len(chunks)} chunks created")
            
            return {
                'success': True,
                'chunks': chunks,
                'metadata': metadata,
                'processing_stats': processing_stats
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_text_content(self, filepath: str) -> Dict[str, Any]:
        """Dispatch text extraction based on file extension"""
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # Dispatcher logic
        if file_ext == '.pdf':
            return self._extract_from_pdf(filepath)
        elif file_ext == '.docx':
            return self._extract_from_docx(filepath)
        elif file_ext == '.xlsx' or file_ext == '.xlsm':
            return self._extract_from_excel(filepath)
        elif file_ext == '.csv':
            return self._extract_from_csv(filepath)
        elif file_ext == '.txt':
            return self._extract_from_txt(filepath)
        elif file_ext == '.md':
            return self._extract_from_md(filepath)
        elif file_ext == '.html':
            return self._extract_from_html(filepath)
        elif file_ext == '.epub':
            return self._extract_from_epub(filepath)
        elif file_ext == '.rtf':
            return self._extract_from_rtf(filepath)
        elif file_ext == '.odt':
            return self._extract_from_odt(filepath)
        elif file_ext == '.pptx' or file_ext == '.ppt':
            return self._extract_from_pptx(filepath)
        else:
            return {
                'success': False,
                'error': f"Unsupported file type: {file_ext}"
            }

    # --- PDF Extraction (Existing, moved into a new method) ---
    def _extract_from_pdf(self, filepath: str) -> Dict[str, Any]:
        documents = []
        # PyMuPDF first
        try:
            # Try newer PyMuPDF API first, then fall back to older
            try:
                pdf_doc = fitz.open(filepath)  # Newer API
            except AttributeError:
                import pymupdf as fitz_new  # Alternative import
                pdf_doc = fitz_new.open(filepath)
            
            try:
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc.load_page(page_num)
                    text = page.get_text()
                    if text and text.strip():
                        metadata = {'page': page_num + 1, 'source': filepath, 'extraction_method': 'pymupdf'}
                        documents.append(Document(page_content=text, metadata=metadata))
            finally:
                pdf_doc.close()
                
            if documents:
                    return {'success': True, 'documents': documents, 'method': 'pymupdf'}
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")

        # Fallback to pdfplumber
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        metadata = {'page': page_num + 1, 'source': filepath, 'extraction_method': 'pdfplumber'}
                        documents.append(Document(page_content=text, metadata=metadata))
                if documents:
                    return {'success': True, 'documents': documents, 'method': 'pdfplumber'}
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}")
        
        return {'success': False, 'error': "Failed to extract text from PDF"}
    # --- END PDF Extraction ---
    
    # --- NEW PARSER METHODS ---
    def _extract_from_docx(self, filepath: str) -> Dict[str, Any]:
        try:
            doc = docx.Document(filepath)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            text_content = "\n\n".join(full_text)
            
            if text_content and text_content.strip():
                # For single-page documents, we create one large document object
                metadata = {'page': 1, 'source': filepath, 'extraction_method': 'python-docx'}
                return {'success': True, 'documents': [Document(page_content=text_content, metadata=metadata)], 'method': 'python-docx'}
            
            return {'success': False, 'error': "No text found in DOCX file"}
        except Exception as e:
            logger.error(f"DOCX extraction failed: {str(e)}")
            return {'success': False, 'error': f"Failed to extract text from DOCX: {str(e)}"}
            
    def _extract_from_excel(self, filepath: str) -> Dict[str, Any]:
        try:
            workbook = openpyxl.load_workbook(filepath, data_only=True)
            documents = []
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text_content = ""
                for row in sheet.iter_rows():
                    row_data = [str(cell.value) if cell.value is not None else "" for cell in row]
                    text_content += "\t".join(row_data) + "\n"
                
                if text_content.strip():
                    metadata = {'page': sheet_name, 'source': filepath, 'extraction_method': 'openpyxl'}
                    documents.append(Document(page_content=text_content, metadata=metadata))
            
            if documents:
                return {'success': True, 'documents': documents, 'method': 'openpyxl'}
            
            return {'success': False, 'error': "No text found in Excel file"}
        except Exception as e:
            logger.error(f"Excel extraction failed: {str(e)}")
            return {'success': False, 'error': f"Failed to extract text from Excel: {str(e)}"}

    def _extract_from_csv(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                text_content = "\n".join([",".join(row) for row in reader])
            
            if text_content and text_content.strip():
                metadata = {'page': 1, 'source': filepath, 'extraction_method': 'csv'}
                return {'success': True, 'documents': [Document(page_content=text_content, metadata=metadata)], 'method': 'csv'}
            
            return {'success': False, 'error': "No text found in CSV file"}
        except Exception as e:
            logger.error(f"CSV extraction failed: {str(e)}")
            return {'success': False, 'error': f"Failed to extract text from CSV: {str(e)}"}
            
    def _extract_from_txt(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            if text_content and text_content.strip():
                metadata = {'page': 1, 'source': filepath, 'extraction_method': 'txt'}
                return {'success': True, 'documents': [Document(page_content=text_content, metadata=metadata)], 'method': 'txt'}
            
            return {'success': False, 'error': "No text found in TXT file"}
        except Exception as e:
            logger.error(f"TXT extraction failed: {str(e)}")
            return {'success': False, 'error': f"Failed to extract text from TXT: {str(e)}"}

    def _extract_from_md(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Markdown can be processed as plain text for simplicity
            # A more advanced approach would be to parse the markdown for structure
            
            if text_content and text_content.strip():
                metadata = {'page': 1, 'source': filepath, 'extraction_method': 'md'}
                return {'success': True, 'documents': [Document(page_content=text_content, metadata=metadata)], 'method': 'md'}
            
            return {'success': False, 'error': "No text found in Markdown file"}
        except Exception as e:
            logger.error(f"Markdown extraction failed: {str(e)}")
            return {'success': False, 'error': f"Failed to extract text from Markdown: {str(e)}"}

    def _extract_from_html(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'lxml')
            
            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.extract()
            
            text_content = soup.get_text()
            
            if text_content and text_content.strip():
                metadata = {'page': 1, 'source': filepath, 'extraction_method': 'beautifulsoup'}
                return {'success': True, 'documents': [Document(page_content=text_content, metadata=metadata)], 'method': 'beautifulsoup'}
            
            return {'success': False, 'error': "No text found in HTML file"}
        except Exception as e:
            logger.error(f"HTML extraction failed: {str(e)}")
            return {'success': False, 'error': f"Failed to extract text from HTML: {str(e)}"}
            
    def _extract_from_epub(self, filepath: str) -> Dict[str, Any]:
        try:
            book = epub.read_epub(filepath)
            documents = []
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'lxml')
                    text_content = soup.get_text()
                    if text_content.strip():
                        metadata = {'page': item.get_name(), 'source': filepath, 'extraction_method': 'ebooklib'}
                        documents.append(Document(page_content=text_content, metadata=metadata))
            
            if documents:
                return {'success': True, 'documents': documents, 'method': 'ebooklib'}
                
            return {'success': False, 'error': "No text found in EPUB file"}
        except Exception as e:
            logger.error(f"EPUB extraction failed: {str(e)}")
            return {'success': False, 'error': f"Failed to extract text from EPUB: {str(e)}"}
            
    def _extract_from_rtf(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'rb') as f:
                rtf_parser = RtfParser(f)
                rtf_parser.parse()
                text_content = "".join(rtf_parser.document.dump_text())
                
            if text_content.strip():
                metadata = {'page': 1, 'source': filepath, 'extraction_method': 'rtfparse'}
                return {'success': True, 'documents': [Document(page_content=text_content, metadata=metadata)], 'method': 'rtfparse'}
            
            return {'success': False, 'error': "No text found in RTF file"}
        except Exception as e:
            logger.error(f"RTF extraction failed: {str(e)}")
            return {'success': False, 'error': f"Failed to extract text from RTF: {str(e)}"}
            
    def _extract_from_odt(self, filepath: str) -> Dict[str, Any]:
        try:
            odt_doc = load(filepath)
            text_content = ""
            for elem in odt_doc.getElementsByType(text.P):
                text_content += teletype.extractText(elem) + "\n"
            if text_content.strip():
                metadata = {'page': 1, 'source': filepath, 'extraction_method': 'odfpy'}
                return {'success': True, 'documents': [Document(page_content=text_content, metadata=metadata)], 'method': 'odfpy'}
            return {'success': False, 'error': "No text found in ODT file"}
        except Exception as e:
            logger.error(f"ODT extraction failed: {str(e)}")
            return {'success': False, 'error': f"Failed to extract text from ODT: {str(e)}"}
            
    def _extract_from_pptx(self, filepath: str) -> Dict[str, Any]:
        try:
            prs = Presentation(filepath)
            documents = []
            for slide_num, slide in enumerate(prs.slides):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text_frame") and shape.text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            slide_text.append("".join([run.text for run in paragraph.runs]))
                
                text_content = "\n".join(slide_text)
                if text_content.strip():
                    metadata = {'page': slide_num + 1, 'source': filepath, 'extraction_method': 'python-pptx'}
                    documents.append(Document(page_content=text_content, metadata=metadata))
            
            if documents:
                return {'success': True, 'documents': documents, 'method': 'python-pptx'}
            
            return {'success': False, 'error': "No text found in PPTX file"}
        except Exception as e:
            logger.error(f"PPTX extraction failed: {str(e)}")
            return {'success': False, 'error': f"Failed to extract text from PPTX: {str(e)}"}
    
    def _create_semantic_chunks(self, documents: List[Document], file_id: str, filename: str) -> List[Document]:
        """Improved chunking with better overlap and context preservation"""
        chunks = []
        
        for doc in documents:
            processed_content = self._preprocess_content(doc.page_content)
            
            # IMPROVED: Use smaller chunks with more overlap for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Reduced from 1000
                chunk_overlap=400,  # Increased overlap
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            )
            
            base_chunks = text_splitter.split_text(processed_content)
            
            for i, chunk_content in enumerate(base_chunks):
                if len(chunk_content.strip()) < 50:
                    continue
                
                enhanced_metadata = doc.metadata.copy()
                enhanced_metadata.update({
                    'file_id': file_id,
                    'filename': filename,
                    'chunk_id': f"{file_id}_{doc.metadata.get('page', 0)}_{i}",
                    'chunk_index': i,
                    'semantic_score': self._calculate_semantic_score(chunk_content),
                    'content_density': len(chunk_content.split()) / len(chunk_content),
                    'content_type': self._detect_content_type(chunk_content),
                    'key_terms': ', '.join(self._extract_key_terms(chunk_content)[:5]),
                    'chunk_created_at': datetime.now().isoformat(),
                })
                
                chunks.append(Document(
                    page_content=chunk_content,
                    metadata=enhanced_metadata
                ))
        
        return chunks
    
    def _preprocess_content(self, content: str) -> str:
        """Preprocess content to improve chunking"""
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Fix common PDF extraction issues
        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)  # Fix missing spaces
        content = re.sub(r'([.!?])([A-Z])', r'\1 \2', content)  # Fix sentence boundaries
        
        return content.strip()
    
    def _calculate_semantic_score(self, content: str) -> float:
        """Calculate semantic richness score"""
        words = content.split()
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        word_diversity = unique_words / len(words)
        
        # Presence of meaningful content indicators
        has_verbs = bool(re.search(r'\b(is|are|was|were|have|has|do|does|will|would|can|could)\b', content.lower()))
        has_entities = bool(re.search(r'\b[A-Z][a-z]+\b', content))
        has_numbers = bool(re.search(r'\d+', content))
        
        score = (word_diversity * 0.4 +
                (1 if has_verbs else 0) * 0.3 +
                (1 if has_entities else 0) * 0.2 +
                (1 if has_numbers else 0) * 0.1)
        
        return min(1.0, score)
    
    def _detect_content_type(self, text: str) -> str:
        """Detect content type based on patterns"""
        text_lower = text.lower()
        
        type_patterns = {
            'introduction': ['introduction', 'background', 'overview'],
            'methodology': ['method', 'methodology', 'approach', 'procedure'],
            'results': ['result', 'finding', 'outcome', 'data', 'analysis'],
            'conclusion': ['conclusion', 'discussion', 'summary'],
            'abstract': ['abstract', 'summary'],
            'references': ['reference', 'bibliography', 'citation'],
            'technical': ['algorithm', 'formula', 'equation', 'implementation'],
        }
        
        for content_type, keywords in type_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return content_type
        
        return 'content'
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', text.lower())
        
        stop_words = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'are', 'was', 'were',
            'which', 'have', 'been', 'will', 'can', 'may', 'also', 'such', 'these'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_counts = Counter(filtered_words)
        
        return [word for word, _ in word_counts.most_common(10)]
    
    def _extract_document_metadata(self, filepath: str, filename: str, documents: List[Document]) -> Dict[str, Any]:
        """Extract comprehensive document metadata"""
        total_words = sum(doc.metadata.get('word_count', 0) for doc in documents)
        total_chars = sum(doc.metadata.get('char_count', 0) for doc in documents)
        
        # Content analysis
        content_types = Counter(self._detect_content_type(doc.page_content) for doc in documents)
        all_text = " ".join([doc.page_content for doc in documents[:5]])  # Sample first 5 pages
        key_terms = self._extract_key_terms(all_text)
        
        return {
            'filename': filename,
            'file_size': os.path.getsize(filepath) if os.path.exists(filepath) else 0,
            'total_pages': len(documents),
            'total_words': total_words,
            'total_characters': total_chars,
            'average_words_per_page': total_words / len(documents) if documents else 0,
            'content_types': dict(content_types.most_common()),
            'key_terms': key_terms[:20],
            'processed_at': datetime.now().isoformat(),
            'language': 'english'  # Could be enhanced with language detection
        }
