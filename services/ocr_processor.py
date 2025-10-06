# services/ocr_processor.py
"""
OCR Processing Service for extracting text from images and scanned documents.
"""

import logging
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import fitz  # PyMuPDF
import re

from config import Config

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    logger.warning("EasyOCR not installed. OCR functionality will be limited.")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class OCRProcessor:
    """Handles OCR for images and scanned documents using EasyOCR."""
    
    def __init__(self):
        self.reader = None
        self.ocr_initialized = False
        self.initialization_attempted = False
        self.supported_languages = Config.OCR_LANGUAGES
        self.confidence_threshold = Config.OCR_CONFIDENCE_THRESHOLD
        self.gpu_enabled = Config.OCR_GPU_ENABLED and HAS_TORCH and torch.cuda.is_available()
        
        # Don't initialize OCR immediately - use lazy initialization
        logger.info("OCR processor created with lazy initialization")
    
    def _initialize_ocr(self):
        """Initialize EasyOCR reader with error handling (lazy initialization)."""
        if self.initialization_attempted:
            return  # Don't try to initialize multiple times
            
        self.initialization_attempted = True
        
        if not HAS_EASYOCR:
            logger.error("EasyOCR not available. Install with: pip install easyocr")
            return
        
        try:
            logger.info(f"Initializing OCR reader with languages: {self.supported_languages}")
            self.reader = easyocr.Reader(
                lang_list=self.supported_languages,
                gpu=self.gpu_enabled
            )
            self.ocr_initialized = True
            logger.info(f"OCR reader initialized successfully (GPU: {self.gpu_enabled})")
        except Exception as e:
            logger.error(f"Failed to initialize OCR reader: {str(e)}")
            self.ocr_initialized = False
    
    def initialize_if_needed(self) -> bool:
        """Manually initialize OCR if not already done. Returns success status."""
        if not self.ocr_initialized and not self.initialization_attempted:
            self._initialize_ocr()
        return self.ocr_initialized
    
    def is_available(self) -> bool:
        """Check if OCR functionality is available (with lazy initialization)."""
        if not self.ocr_initialized and not self.initialization_attempted:
            self._initialize_ocr()
        return self.ocr_initialized and self.reader is not None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for better OCR results with Indian languages."""
        if not Config.OCR_PREPROCESSING:
            return image
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Enhance contrast using CLAHE (good for varied lighting)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply adaptive thresholding with more aggressive settings for Indian scripts
            processed = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
            )
            
            # Morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            
            # Denoise
            processed = cv2.medianBlur(processed, 3)
            
            logger.debug("Applied enhanced preprocessing for Indian language OCR")
            return processed
        except Exception as e:
            logger.warning(f"Enhanced image preprocessing failed: {str(e)}, using original image")
            return image
    
    def _detect_non_english_text(self, text: str) -> bool:
        """Simple detection for non-English text patterns."""
        if not text or not text.strip():
            return False
        
        # Check for common non-English Unicode ranges
        # Devanagari (Hindi): U+0900-U+097F
        # Malayalam: U+0D00-U+0D7F  
        # Tamil: U+0B80-U+0BFF
        # Telugu: U+0C00-U+0C7F
        # Gujarati: U+0A80-U+0AFF
        # Kannada: U+0C80-U+0CFF
        # Bengali: U+0980-U+09FF
        # Punjabi: U+0A00-U+0A7F
        non_english_patterns = [
            r'[\u0900-\u097F]',  # Devanagari (Hindi)
            r'[\u0D00-\u0D7F]',  # Malayalam
            r'[\u0B80-\u0BFF]',  # Tamil
            r'[\u0C00-\u0C7F]',  # Telugu
            r'[\u0A80-\u0AFF]',  # Gujarati
            r'[\u0C80-\u0CFF]',  # Kannada
            r'[\u0980-\u09FF]',  # Bengali
            r'[\u0A00-\u0A7F]',  # Punjabi
            r'[\u0980-\u09FF]',  # Odia
        ]
        
        for pattern in non_english_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def perform_ocr_on_image(self, image_data) -> Dict[str, Any]:
        """Perform OCR on image data (PIL Image, numpy array, or bytes)."""
        if not self.is_available():
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "error": "OCR reader not initialized"
            }
        
        try:
            # Convert input to numpy array
            if isinstance(image_data, bytes):
                image = Image.open(BytesIO(image_data))
                image_array = np.array(image)
            elif isinstance(image_data, Image.Image):
                image_array = np.array(image_data)
            elif isinstance(image_data, np.ndarray):
                image_array = image_data
            else:
                return {
                    "success": False,
                    "text": "",
                    "confidence": 0.0,
                    "error": "Unsupported image format"
                }
            
            # Preprocess image
            if Config.OCR_PREPROCESSING:
                processed_image = self.preprocess_image(image_array)
            else:
                processed_image = image_array
            
            # Perform OCR
            ocr_results = self.reader.readtext(processed_image)
            
            # Extract text and confidence scores with more tolerance for Indian languages
            extracted_texts = []
            confidence_scores = []
            all_texts = []  # Keep track of all detected text for debugging
            
            for (bbox, text, confidence) in ocr_results:
                all_texts.append(f"{text}({confidence:.3f})")
                # More lenient threshold for Indian languages
                if confidence >= self.confidence_threshold:
                    extracted_texts.append(text)
                    confidence_scores.append(confidence)
                elif confidence >= 0.2 and len(text.strip()) > 1:  # Very low threshold for short meaningful text
                    extracted_texts.append(text)
                    confidence_scores.append(confidence)
            
            full_text = " ".join(extracted_texts)
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # Check for non-English text and provide appropriate messaging
            contains_non_english = self._detect_non_english_text(full_text)
            
            # Enhanced logging for debugging
            logger.info(f"OCR detected {len(ocr_results)} text regions, {len(extracted_texts)} above threshold")
            if not extracted_texts and all_texts:
                logger.warning(f"All detected texts below threshold: {all_texts}")
            
            # If non-English text detected, provide appropriate message
            if contains_non_english:
                logger.info("Non-English text detected in OCR result")
                return {
                    "success": True,
                    "text": "This image contains non-English text. Currently, only English text processing is supported. Please provide images with English text for optimal results.",
                    "confidence": float(avg_confidence),
                    "word_count": 0,
                    "processed_at": datetime.now().isoformat(),
                    "non_english_detected": True,
                    "debug_all_detections": all_texts[:5]
                }
            
            return {
                "success": True,
                "text": full_text,
                "confidence": float(avg_confidence),
                "word_count": len(extracted_texts),
                "processed_at": datetime.now().isoformat(),
                "non_english_detected": False,
                "debug_all_detections": all_texts[:5]  # First 5 detections for debugging
            }
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def perform_ocr_on_pdf_page(self, pdf_page) -> Dict[str, Any]:
        """Perform OCR on a specific PDF page."""
        if not self.is_available():
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "error": "OCR reader not initialized"
            }
        
        try:
            # Render PDF page as image
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # Higher resolution
            img_bytes = pix.tobytes("png")
            
            # Perform OCR on the image
            ocr_result = self.perform_ocr_on_image(img_bytes)
            
            if ocr_result["success"]:
                ocr_result["extraction_method"] = "pdf_ocr"
            
            return ocr_result
            
        except Exception as e:
            logger.error(f"PDF page OCR failed: {str(e)}")
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def process_image_file(self, file_path: str) -> Dict[str, Any]:
        """Process a standalone image file."""
        if not self.is_available():
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "error": "OCR reader not initialized"
            }
        
        try:
            # Load image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                ocr_result = self.perform_ocr_on_image(img)
                
                if ocr_result["success"]:
                    ocr_result.update({
                        "file_path": file_path,
                        "extraction_method": "image_ocr",
                        "image_size": img.size
                    })
                
                return ocr_result
                
        except Exception as e:
            logger.error(f"Image file processing failed: {str(e)}")
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_ocr_stats(self) -> Dict[str, Any]:
        """Get OCR processor statistics."""
        return {
            "ocr_available": self.is_available(),
            "gpu_enabled": self.gpu_enabled,
            "languages_supported": self.supported_languages,
            "confidence_threshold": self.confidence_threshold,
            "preprocessing_enabled": Config.OCR_PREPROCESSING
        }