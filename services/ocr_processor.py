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
        self.supported_languages = Config.OCR_LANGUAGES
        self.confidence_threshold = Config.OCR_CONFIDENCE_THRESHOLD
        self.gpu_enabled = Config.OCR_GPU_ENABLED and HAS_TORCH and torch.cuda.is_available()
        
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize EasyOCR reader with error handling."""
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
    
    def is_available(self) -> bool:
        """Check if OCR functionality is available."""
        return self.ocr_initialized and self.reader is not None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        if not Config.OCR_PREPROCESSING:
            return image
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply adaptive thresholding to improve text contrast
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Denoise
            processed = cv2.medianBlur(processed, 3)
            
            return processed
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            return image
    
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
            
            # Extract text and confidence scores
            extracted_texts = []
            confidence_scores = []
            
            for (bbox, text, confidence) in ocr_results:
                if confidence >= self.confidence_threshold:
                    extracted_texts.append(text)
                    confidence_scores.append(confidence)
            
            full_text = " ".join(extracted_texts)
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return {
                "success": True,
                "text": full_text,
                "confidence": float(avg_confidence),
                "word_count": len(extracted_texts),
                "processed_at": datetime.now().isoformat()
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