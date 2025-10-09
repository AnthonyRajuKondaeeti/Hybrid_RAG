"""
Image Analysis Service

This module provides image analysis capabilities for the RAG system,
including OCR and multimodal image understanding.
"""

import logging
from datetime import datetime
from typing import Dict, Any
from ..utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class ImageAnalysisService:
    """Service for analyzing standalone images using multimodal capabilities."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.ocr_processor = None
        self.llm = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize OCR processor and LLM components."""
        try:
            from services.ocr_processor import OCRProcessor
            self.ocr_processor = OCRProcessor()
        except ImportError as e:
            logger.error(f"Failed to import OCRProcessor: {e}")
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM for image analysis."""
        try:
            from langchain_mistralai.chat_models import ChatMistralAI
            self.llm = ChatMistralAI(
                model=self.config_manager.get('MISTRAL_MODEL', 'mistral-small-latest'),
                temperature=self.config_manager.get('TEMPERATURE', 0.3),
                api_key=self.config_manager.get('MISTRAL_API_KEY'),
                timeout=60,
                max_retries=3,
                max_tokens=self.config_manager.get('MAX_TOKENS', 2048),
                top_p=self.config_manager.get('TOP_P', 0.9)
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
            ocr_result = self.ocr_processor.process_image_file(image_path) if self.ocr_processor else {'success': False, 'error': 'OCR not available'}
            
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

__all__ = ['ImageAnalysisService']