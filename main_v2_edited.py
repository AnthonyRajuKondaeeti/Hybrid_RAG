from __future__ import print_function
from config import *

import uuid
import sys
import logging
import os
import json
import html
import re
# import jwt  # Commented out - JWT functionality disabled
from bson.json_util import dumps
from functools import wraps
import io
from flask import Flask, jsonify, send_file, Response, session, request
from flask_cors import CORS, cross_origin
from flask_session import Session
from dotenv import load_dotenv
load_dotenv()

# Google Cloud Translate - Commented out
# from google.cloud import translate_v2 as translate
# translate_client = translate.Client()

from qdrant_client.http import models
from qdrant_client import QdrantClient

from datetime import datetime, timezone, timedelta, date
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import time

# Import enhanced services
from services.document_service import DocumentService
from services.ocr_processor import OCRProcessor
from services.rag_service import EnhancedRAGService

# Import utilities
# from file_upload import upload, read_logo, delete_file
# import validations
# from data_logs import user_logs
from utils.response_formatter import ResponseFormatter

# Placeholder functions for missing modules
def upload(file_obj, filename):
    """Placeholder function for file upload - needs implementation"""
    logger.warning("File upload function not implemented - placeholder called")
    return f"placeholder_url_{filename}"

def read_logo(logo_url):
    """Placeholder function for reading logo - needs implementation"""
    logger.warning("Read logo function not implemented - placeholder called")
    return None

def delete_file(file_url):
    """Placeholder function for file deletion - needs implementation"""
    logger.warning("Delete file function not implemented - placeholder called")
    return True

def user_logs(container, email, action, session_id):
    """Placeholder function for user logging - needs implementation"""
    logger.warning(f"User logs function not implemented - would log: {email}, {action}, {session_id}")
    return True

class ValidationPlaceholder:
    """Enhanced validations module replacement"""
    @staticmethod
    def str_to_array(string_data):
        """Convert string to array with multiple format support"""
        # If already a list, return as-is
        if isinstance(string_data, list):
            logger.debug(f"str_to_array: Input already list with {len(string_data)} items")
            return string_data
        
        # If not a string, convert to string first
        if not isinstance(string_data, str):
            result = [str(string_data)] if string_data is not None else []
            logger.debug(f"str_to_array: Converted non-string to array: {result}")
            return result
        
        # Handle empty strings
        if not string_data.strip():
            logger.debug("str_to_array: Empty string converted to empty array")
            return []
        
        try:
            # Try to parse as JSON array first
            import json
            parsed = json.loads(string_data)
            if isinstance(parsed, list):
                result = [str(item).strip() for item in parsed if item is not None]
                logger.debug(f"str_to_array: Parsed JSON array with {len(result)} items")
                return result
            else:
                # Single JSON value
                result = [str(parsed).strip()]
                logger.debug(f"str_to_array: Parsed single JSON value: {result}")
                return result
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, try other formats
            pass
        
        # Handle comma-separated values
        if ',' in string_data:
            result = [item.strip() for item in string_data.split(',') if item.strip()]
            logger.debug(f"str_to_array: Parsed comma-separated values: {len(result)} items")
            return result
        
        # Handle semicolon-separated values
        if ';' in string_data:
            result = [item.strip() for item in string_data.split(';') if item.strip()]
            logger.debug(f"str_to_array: Parsed semicolon-separated values: {len(result)} items")
            return result
        
        # Handle pipe-separated values
        if '|' in string_data:
            result = [item.strip() for item in string_data.split('|') if item.strip()]
            logger.debug(f"str_to_array: Parsed pipe-separated values: {len(result)} items")
            return result
        
        # Handle newline-separated values
        if '\n' in string_data:
            result = [item.strip() for item in string_data.split('\n') if item.strip()]
            logger.debug(f"str_to_array: Parsed newline-separated values: {len(result)} items")
            return result
        
        # Single value - return as single-item array
        result = [string_data.strip()]
        logger.debug(f"str_to_array: Single value converted to array: {result}")
        return result

validations = ValidationPlaceholder()

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, check_compatibility=False)

# MongoDB configuration
server_uri = MONGODB_URL
USER_CONTAINER = "user"
SESSION_CONTAINER = "session"
QA_CONTAINER = "qa"
LOGS_CONTAINER = "user_logs"

# Initialize services with enhanced capabilities
document_service = DocumentService(
    chunk_size=getattr(Config, 'CHUNK_SIZE', 1000),
    chunk_overlap=getattr(Config, 'CHUNK_OVERLAP', 200)
)
ocr_processor = OCRProcessor()

# Session storage (in-memory)
session_rag_services = {}

# Global model cache to prevent redownloading
_global_models_cache = {
    'embedding_model': None,
    'reranker': None,
    'embedding_dim': None,
    'initialized': False
}

def initialize_global_models():
    """Initialize models once globally to prevent redownloading"""
    global _global_models_cache
    
    if _global_models_cache['initialized']:
        logger.info("Global models already initialized, using cached models")
        return _global_models_cache
    
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        
        logger.info("Initializing global models for the first time...")
        
        # Initialize embedding model
        embedding_model_name = getattr(Config, 'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        device = getattr(Config, 'DEVICE', 'cpu')
        
        _global_models_cache['embedding_model'] = SentenceTransformer(
            embedding_model_name,
            device=device
        )
        
        # Warmup embedding model
        warmup_texts = ["This is a warmup text to initialize the model."]
        _ = _global_models_cache['embedding_model'].encode(warmup_texts, show_progress_bar=False)
        logger.info("Global embedding model warmed up successfully")
        
        # Initialize reranker
        reranker_model = getattr(Config, 'RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        try:
            _global_models_cache['reranker'] = CrossEncoder(reranker_model)
            logger.info("Global reranker initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to load global reranker: {e}")
            _global_models_cache['reranker'] = None
        
        _global_models_cache['embedding_dim'] = _global_models_cache['embedding_model'].get_sentence_embedding_dimension()
        _global_models_cache['initialized'] = True
        
        logger.info("Global models initialization completed successfully")
        return _global_models_cache
        
    except Exception as e:
        logger.error(f"Failed to initialize global models: {e}")
        raise

def setup_logging():
    """Configure logging with structured format"""
    log_format = '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log', mode='a')
        ]
    )
    
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('qdrant_client').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# JWT functionality commented out - authentication disabled
# def jwt_required(fn):
#     @wraps(fn)
#     def wrapper(*args, **kwargs):
#         token = request.headers.get('Authorization')
#         if not token:
#             return jsonify({"errors": ["Token is missing"], "status": 401, "success": False}), 401
#         try:
#             token = token.split(' ')[1]
#             payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
#             with MongoClient(server_uri) as mongo_client_server:
#                 db_server = mongo_client_server[DB]
#                 user_container = db_server[USER_CONTAINER]
#                 current_user = user_container.find_one({"email": payload['email']})
#             if current_user:
#                 return fn(current_user, *args, **kwargs)
#         except jwt.ExpiredSignatureError:
#             return jsonify({"errors": ["Token has expired"], "status": 401, "success": False}), 401
#         except jwt.InvalidTokenError:
#             return jsonify({"errors": ["Invalid token"], "status": 401, "success": False}), 401
#     return wrapper

# Placeholder JWT decorator - bypasses authentication
def jwt_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Skip JWT validation - return mock user
        mock_user = {"email": "test@example.com", "_id": "placeholder_user_id"}
        return fn(mock_user, *args, **kwargs)
    return wrapper

def convert_search_results_to_documents(search_results):
    """Helper function to convert search results to Document objects for compatibility"""
    chunks = []
    for result in search_results[:5]:
        # Handle both SearchResult and SemanticChunk objects
        if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
            # SearchResult object containing SemanticChunk
            content = result.chunk.content
            metadata = getattr(result.chunk, 'metadata', {})
            if not metadata:
                # Build metadata from SemanticChunk properties
                metadata = {
                    'chunk_id': getattr(result.chunk, 'chunk_id', ''),
                    'file_id': getattr(result.chunk, 'file_id', ''),
                    'chunk_type': getattr(result.chunk, 'chunk_type', ''),
                    'page': getattr(result.chunk, 'page', 1)
                }
        elif hasattr(result, 'content'):
            # Direct SemanticChunk object
            content = result.content
            metadata = {
                'chunk_id': getattr(result, 'chunk_id', ''),
                'file_id': getattr(result, 'file_id', ''),
                'chunk_type': getattr(result, 'chunk_type', ''),
                'page': getattr(result, 'page', 1)
            }
        else:
            # Fallback for other object types
            content = getattr(result, 'content', '') or getattr(result, 'chunk', '')
            metadata = getattr(result, 'metadata', {})
        
        if content:
            doc = type('Document', (), {
                'page_content': content,
                'metadata': metadata
            })()
            chunks.append(doc)
    
    return chunks

def clean_question_text(question):
    """Clean and format question text to remove markdown and ensure readability"""
    if not question:
        return question
    
    # Remove markdown formatting and asterisks
    question = re.sub(r'\*\*([^*]+)\*\*', r'\1', question)  # Remove bold markdown **text**
    question = re.sub(r'\*([^*]+)\*', r'\1', question)      # Remove italic markdown *text*
    question = re.sub(r'^\*+\s*', '', question)             # Remove leading asterisks
    question = re.sub(r'\s*\*+$', '', question)             # Remove trailing asterisks
    question = re.sub(r'\*{2,}', '', question)              # Remove multiple asterisks
    
    # Remove other markdown elements
    question = re.sub(r'#{1,6}\s*', '', question)           # Remove headers # ## ###
    question = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', question)  # Remove links [text](url)
    question = re.sub(r'`([^`]+)`', r'\1', question)        # Remove code backticks
    question = re.sub(r'_{2,}([^_]+)_{2,}', r'\1', question)  # Remove underline formatting
    
    # Remove markdown list indicators  
    question = re.sub(r'^\s*[-\+\*]\s*', '', question)      # Remove list bullets - + *
    question = re.sub(r'^\s*\d+\.\s*', '', question)        # Remove numbered list items
    
    # Clean quotes and formatting
    question = re.sub(r'^["\'\*\-\+#]+\s*', '', question)   # Remove leading quotes/asterisks/dashes
    question = re.sub(r'\s*["\'\*\-\+#]+$', '', question)   # Remove trailing quotes/asterisks/dashes
    
    # Clean numbering if it's malformed
    question = re.sub(r'^\d+[\.\)]\s*[\*\-"\'\`]*\s*', '', question)  # Remove numbered prefixes
    
    # Remove any remaining markdown artifacts
    question = re.sub(r'^\s*[\>\|]\s*', '', question)       # Remove blockquote > and table |
    question = re.sub(r'~~([^~]+)~~', r'\1', question)      # Remove strikethrough ~~text~~
    question = re.sub(r'==([^=]+)==', r'\1', question)      # Remove highlight ==text==
    
    # Clean spacing and normalize
    question = re.sub(r'\s+', ' ', question).strip()        # Normalize whitespace
    
    # Remove extra punctuation
    question = re.sub(r'[?]{2,}', '?', question)            # Multiple question marks
    question = re.sub(r'[.]{2,}', '', question)             # Multiple periods
    question = re.sub(r'[!]{2,}', '!', question)            # Multiple exclamation marks
    
    # Ensure proper ending
    if question and not question.endswith(('?', '.', '!')):
        question += '?'
    
    # Capitalize first letter
    if question:
        question = question[0].upper() + question[1:] if len(question) > 1 else question.upper()
    
    return question

def get_database_containers():
    """Get standardized database container access"""
    mongo_client_server = MongoClient(server_uri)
    db_server = mongo_client_server[DB]
    return mongo_client_server, db_server

def extract_user_id(user_or_item):
    """Extract MongoDB ObjectId consistently"""
    return user_or_item["_id"]["$oid"]

def find_user_by_session(session_id):
    """Common user lookup pattern - returns user data"""
    with MongoClient(server_uri) as mongo_client_server:
        db_server = mongo_client_server[DB]
        user_container = db_server[USER_CONTAINER]
        user = json.loads(dumps(user_container.find_one({"session_id": session_id})))
        return user

def find_user_by_email_and_session(email, session_id):
    """Find user by email with session validation"""
    with MongoClient(server_uri) as mongo_client_server:
        db_server = mongo_client_server[DB]
        user_container = db_server[USER_CONTAINER]
        user = json.loads(dumps(user_container.find_one({"email": email, "session_id": {'$in': [session_id]}})))
        return user

def generate_questions_with_fallbacks(rag_service, session_id=None, user_id=None, chunks=None, email=None):
    """Unified question generation with all fallback strategies"""
    try:
        # Strategy 1: Session-based generation (highest priority)
        if session_id and user_id:
            logger.info(f"Using session-based question generation for session: {session_id}")
            questions = rag_service.generate_sample_questions(session_id=session_id, user_id=user_id)
            if questions and len(questions) >= 3:
                logger.info(f"Successfully generated {len(questions)} session-based questions")
                return questions[:4]  # Return exactly 4 questions
        
        # Strategy 2: Chunk-based generation
        if chunks:
            logger.info("Using chunk-based question generation")
            questions = rag_service.generate_sample_questions(chunks=chunks)
            if questions and len(questions) >= 3:
                # Clean each question
                cleaned_questions = []
                for q in questions[:4]:
                    cleaned_q = clean_question_text(q)
                    if cleaned_q:
                        cleaned_questions.append(cleaned_q)
                return cleaned_questions
        
        # Strategy 3: MongoDB session data fallback
        if email and session_id:
            logger.warning("Using MongoDB session data fallback for question generation")
            return generate_questions_from_mongodb_session(email, session_id)
        
        # Strategy 4: RAG service without specific session - use fallback session
        logger.warning("Using fallback question generation with available parameters")
        if session_id and user_id:
            questions = rag_service.generate_sample_questions(session_id=session_id, user_id=user_id)
        else:
            questions = get_default_sample_questions()
        if questions:
            cleaned_questions = []
            for q in questions[:4]:
                cleaned_q = clean_question_text(q)
                if cleaned_q:
                    cleaned_questions.append(cleaned_q)
            return cleaned_questions
        
        # Strategy 5: Default questions (last resort)
        logger.warning("All question generation strategies failed, using default questions")
        return get_default_sample_questions()
        
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        return get_default_sample_questions()

def get_enhanced_services_config():
    """Get centralized enhanced services configuration"""
    return {
        'ENHANCED_MODE': True,
        'ENHANCED_SERVICES': True
    }

def format_questions_for_frontend(questions):
    """Format sample questions with HTML breaks for frontend display"""
    formatted_questions = []
    for question in questions:
        # Clean markdown and formatting first
        clean_question = clean_question_text(question)
        # Convert newlines to HTML breaks like in answers
        formatted_question = clean_question.replace('\n', '<br />')
        formatted_questions.append(formatted_question)
    return formatted_questions

def generate_contextual_questions_from_chunks(chunks):
    """Generate contextual questions from document chunks as LLM fallback"""
    questions = []
    for i, chunk in enumerate(chunks[:3]):
        content_preview = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
        questions.extend([
            f"What is discussed in section {i+1}: {content_preview}?",
            f"Can you explain the key points from: {content_preview}?"
        ])
    
    # Add general questions
    questions.extend([
        "What are the main themes across all document sections?",
        "How do the different sections relate to each other?",
        "What conclusions can be drawn from this document?"
    ])
    
    # Clean each question without numbering for better readability
    cleaned_questions = []
    for q in questions[:4]:
        cleaned_q = clean_question_text(q)
        if cleaned_q:
            cleaned_questions.append(cleaned_q)
    
    return cleaned_questions

def generate_sample_questions_from_chunks(rag_service, chunks=None, session_id=None, user_id=None):
    """Generate sample questions from chunks with fallback strategies"""
    try:
        logger.info(f"Attempting to generate questions from {len(chunks) if chunks else 0} chunks, session_id: {session_id}")
        
        # Prioritize session-based generation if session_id is provided
        if session_id:
            logger.info(f"Using session-based question generation for session: {session_id}")
            questions = rag_service.generate_sample_questions(session_id=session_id, user_id=user_id)
        elif chunks:
            # Debug: Log chunk content previews
            for i, chunk in enumerate(chunks[:3]):
                content_preview = chunk.page_content[:150] if hasattr(chunk, 'page_content') else str(chunk)[:150]
                logger.debug(f"Chunk {i+1} preview: {content_preview}...")
            
            # Try chunk-based question generation
            questions = rag_service.generate_sample_questions(session_id or "default", user_id or "default", chunks=chunks)
        else:
            logger.warning("No session_id or chunks provided, using fallback questions")
            questions = get_default_sample_questions()
        
        logger.info(f"RAG service returned {len(questions) if questions else 0} questions")
        
        # Clean each question without numbering for better readability
        cleaned_questions = []
        for q in questions[:4]:
            cleaned_q = clean_question_text(q)
            if cleaned_q:
                cleaned_questions.append(cleaned_q)
        
        logger.info(f"Generated {len(cleaned_questions)} questions using RAG service")
        return cleaned_questions
    except Exception as llm_error:
        logger.warning(f"LLM-based question generation failed: {llm_error}")
        # Fallback to content-based questions
        questions = generate_contextual_questions_from_chunks(chunks)
        logger.info(f"Generated {len(questions)} contextual questions from chunk content")
        return questions

def generate_questions_from_mongodb_session(email, session_id):
    """Generate questions from session data stored in MongoDB as fallback"""
    try:
        with MongoClient(server_uri) as mongo_client_server:
            db_server = mongo_client_server[DB]
            session_container = db_server[SESSION_CONTAINER]
            
            # Get session data
            session_data = json.loads(dumps(session_container.find_one(
                {"user_id": email, "session_id": session_id},
                {'_id': False, 'file_names': True, 'qr_name': True}
            )))
            
            if session_data:
                file_names = session_data.get('file_names', [])
                qr_name = session_data.get('qr_name', 'Document')
                
                # Generate questions based on file names and session name
                questions = [
                    f"What is the main content of {qr_name}?",
                    f"Can you summarize the {len(file_names)} uploaded document(s)?",
                    f"What are the key points in these files: {', '.join(file_names[:3])}?",
                    f"What specific information is contained in {qr_name}?"
                ]
                
                # Clean each question without numbering for better readability
                cleaned_questions = []
                for q in questions[:4]:
                    cleaned_q = clean_question_text(q)
                    if cleaned_q:
                        cleaned_questions.append(cleaned_q)
                
                logger.info(f"Generated questions from MongoDB session data")
                return cleaned_questions
    except Exception as e:
        logger.warning(f"MongoDB-based question generation failed: {e}")
    
    return get_default_sample_questions()

def get_default_sample_questions():
    """Get default sample questions to avoid duplication"""
    default_questions = [
        "What is this document about?",
        "Can you provide a summary?",
        "What are the main topics covered?",
        "What conclusions can be drawn?"
    ]
    
    # Clean the default questions without numbering for better readability
    cleaned_questions = []
    for q in default_questions:
        cleaned_q = clean_question_text(q)
        if cleaned_q:
            cleaned_questions.append(cleaned_q)
    
    return cleaned_questions

def validate_image_file(file):
    """Validate uploaded image file."""
    if not file or not file.filename:
        return False, "No file selected"
    
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    image_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    
    if file_ext not in image_formats:
        return False, f"Unsupported image format. Supported formats: {', '.join(image_formats)}"
    
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    max_size_bytes = 10 * 1024 * 1024
    if file_size > max_size_bytes:
        return False, f"File too large. Maximum size: 10MB"
    
    return True, ""

def get_or_create_rag_service(session_id, user_id):
    """Get existing RAG service or create new one for session using cached models"""
    session_key = f"{user_id}_{session_id}"
    
    if session_key not in session_rag_services:
        try:
            # Initialize global models if not already done
            global_models = initialize_global_models()
            
            # Create RAG service with cached models
            from config import Config as LocalConfig
            from services.utils.config_manager import ConfigManager
            
            # Create RAG service but skip model initialization
            rag_service = EnhancedRAGService.__new__(EnhancedRAGService)
            rag_service.config_manager = ConfigManager(LocalConfig())
            
            # Initialize required attributes
            from services.rag.text_processing import TextProcessor
            from services.rag.confidence_calculator import ConfidenceCalculator
            from collections import defaultdict
            import threading
            
            rag_service.text_processor = TextProcessor()
            rag_service.confidence_calculator = ConfidenceCalculator()
            rag_service.stats = defaultdict(int)
            rag_service.timing_stats = defaultdict(list)
            rag_service.threshold_cache = {}
            rag_service.cache_lock = threading.Lock()
            rag_service.cache_max_size = 100
            rag_service._answer_cache = {}
            rag_service._rate_limit_lock = threading.Lock()  # Add missing rate limit lock
            rag_service._last_api_call = 0  # Rate limiting timestamp
            rag_service._api_call_interval = 1.0  # Minimum 1 second between calls
            rag_service._initialized = True  # Mark as initialized
            
            # Use our cached models instead of initializing new ones
            rag_service.embedding_model = global_models['embedding_model']
            rag_service.reranker = global_models['reranker']
            rag_service.embedding_dim = global_models['embedding_dim']
            
            logger.info(f"Using cached models for session {session_id} - no model download needed")
            
            # Initialize other services that don't load models
            rag_service._initialize_clients()
            rag_service._initialize_services()
            rag_service._initialize_conversation_memory()
            
            session_rag_services[session_key] = rag_service
            logger.info(f"Created new EnhancedRAGService for session {session_id} using cached models")
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedRAGService: {e}")
            raise
    
    return session_rag_services[session_key]

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config["file_text_dict"] = {}
    
    CORS(app, supports_credentials=True, origins=[
        "null", "https://xplorease.sn-ipl.com", "https://data-chat.demetrix.in", 
        "https://demetrix.in", "https://guru-g.demetrix.in", 
        "https://soliboat.solidaridadasia.com", "https://xplorease.com", 
        "http://localhost:3000", "https://localhost:3000", 
        "http://192.168.29.100:3000", "https://bot.sn-ipl.com", 
        "https://bot.xplorease.com", 
        "https://mspo-development.s3.ap-south-1.amazonaws.com"
    ])
    
    return app

app = create_app()

# Initialize global models on startup to prevent first-request delays
try:
    logger.info("Initializing global models on application startup...")
    initialize_global_models()
    logger.info("Application startup completed with models ready")
except Exception as e:
    logger.warning(f"Failed to initialize models on startup: {e}. Models will be loaded on first request.")

def check_image(img):
    return img.mimetype in ['image/tiff', 'image/jpeg', 'image/png', 'image/svg+xml']

def generate_qr_code(chat_url, qr_name, logo_img=None):
    """Single function handling both logo and simple QR generation"""
    import qrcode
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(chat_url)
    qr.make(fit=True)
    
    if logo_img is not None:
        # QR code with logo logic would go here
        # For now, just generate simple QR since logo logic needs implementation
        img = qr.make_image(fill_color="black", back_color="white")
    else:
        img = qr.make_image(fill_color="black", back_color="white")
    
    return img

def generate_qr_code_with_logo(chat_url, logo_img, qr_name):
    """Legacy function - calls unified QR generation"""
    return generate_qr_code(chat_url, qr_name, logo_img)

def generate_simple_qr_code(chat_url, qr_name):
    """Legacy function - calls unified QR generation"""
    return generate_qr_code(chat_url, qr_name)

@app.route("/", methods=["GET"])
def index():
    """API Information - Root endpoint"""
    return jsonify({
        "success": True,
        "message": "XplorEase RAG System API",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "process_file": {
                "method": "POST",
                "description": "Upload and process documents",
                "url": "/process_file"
            },
            "answer_question": {
                "method": "POST", 
                "description": "Ask questions about uploaded documents",
                "url": "/answer_question"
            },
            "generate_sample_questions": {
                "method": "POST",
                "description": "Generate sample questions from documents", 
                "url": "/generate_sample_questions"
            },
            "healthcheck": {
                "method": "GET/POST",
                "description": "Health check endpoint",
                "url": "/healthcheck"
            }
        },
        "documentation": "Enhanced RAG system with semantic search, anti-hallucination, and conversation memory"
    })

@app.route(f"/process_file", methods=["POST"])
# @jwt_required  # JWT authentication disabled
def process_file():  # Removed current_user parameter since JWT is disabled
    """Enhanced file processing with new architecture integration"""
    try:
        logo = request.files.get('logo')
        files = request.files.getlist("file")
        session_id = str(uuid.uuid4().hex)
        email = request.form.get('email')
        qr_name = request.form.get('qr_name', 'Document')
        
        if not email:
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["Email is required"]
            }), 400
        
        if not files or all(not f.filename for f in files):
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["No valid files provided"]
            }), 400
        
    except Exception as e:
        logger.error(f"Request parsing error: {e}")
        return jsonify({
            "success": False,
            "status": 400,
            "errors": ["Please provide the files, email, and session name"]
        }), 400

    # User validation
    with MongoClient(server_uri) as mongo_client_server:
        db_server = mongo_client_server[DB]
        user_container = db_server[USER_CONTAINER]
        user_logs_container = db_server[LOGS_CONTAINER]
        user = json.loads(dumps(user_container.find_one({"email": email})))
        if user is None:
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["User does not exist. Please signup."]
            }), 400
        
        # Extract user's MongoDB _id to use as user_id for collection naming
        user_id = extract_user_id(user)
        
        user_logs(user_logs_container, email, "File Upload", session_id)

    user_name = user_id  # Use the same value for backward compatibility 
    
    # Enhanced logo processing
    logo_url = None
    if logo and check_image(logo):
        logo_name = user_name + "/{}_logo.png".format(user_name)
        logo_name = '{0}/{1}'.format('xplorease', '{}'.format(logo_name))
        try:
            from PIL import Image
            logo_image = Image.open(logo)
            logo_bytes = io.BytesIO()
            logo_image.save(logo_bytes, format="PNG")
            logo_bytes.seek(0)
            logo_url = upload(logo_bytes, logo_name)
        except Exception as e:
            logger.error(f"Logo processing failed: {e}")

    # Enhanced file processing
    processed_files = []
    skipped_files = []
    processing_errors = []
    
    # Define allowed extensions
    config = get_enhanced_services_config()
    ENHANCED_MODE = config['ENHANCED_MODE']
    ENHANCED_SERVICES = config['ENHANCED_SERVICES']
    
    if ENHANCED_MODE:
        allowed_extensions = set(getattr(Config, 'ALLOWED_EXTENSIONS', [
            'pdf', 'html', 'epub', 'rtf', 'odt', 'docx', 'txt',
            'xlsx', 'xlsm', 'csv', 'ppt', 'pptx', 'md',
            'jpg', 'jpeg', 'png', 'bmp', 'tiff'
        ]))
    else:
        allowed_extensions = {
            'pdf', 'html', 'epub', 'rtf', 'odt', 'docx', 'txt',
            'xlsx', 'xlsm', 'csv', 'ppt', 'pptx', 'md',
            'jpg', 'jpeg', 'png', 'bmp', 'tiff'
        }
    
    session_text = user_name + "_" + session_id
    file_urls = []
    file_names = []
    s3_keys = []
    
    # Create uploads directory
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    
    # Initialize RAG service for this session
    try:
        rag_service = get_or_create_rag_service(session_id, user_id)
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        return jsonify({
            "success": False,
            "status": 500,
            "errors": [f"Service initialization error: {str(e)}"]
        }), 500
    
    for file in files:
        if not file.filename:
            continue
            
        file_extension = os.path.splitext(file.filename)[1].lower().strip('.')
        if file_extension not in allowed_extensions:
            skipped_files.append(f'{file.filename} (unsupported file type)')
            continue

        # Enhanced image validation
        is_image = file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        if is_image:
            is_valid, error_msg = validate_image_file(file)
            if not is_valid:
                skipped_files.append(f'{file.filename} ({error_msg})')
                continue

        original_filename = secure_filename(file.filename)
        file_start_time = time.time()
        
        try:
            # File size calculation
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            # Save file temporarily
            save_start = time.time()
            local_path = os.path.join(uploads_dir, original_filename)
            file.save(local_path)
            save_time = time.time() - save_start
            
            # S3 upload for legacy compatibility
            file_names.append(original_filename)
            file_name_s3 = user_name + "/{}_{}".format(session_id, original_filename)
            s3_filename = '{0}/{1}'.format('xplorease', '{}'.format(file_name_s3))
            s3_keys.append(s3_filename)
            
            # Reset file pointer for S3 upload
            file.seek(0)
            file_urls.append(upload(file, s3_filename))
            
            # Enhanced document processing
            doc_process_start = time.time()
            
            if ENHANCED_SERVICES:
                # Use enhanced document service
                result = document_service.process_document(local_path, session_text, original_filename)
                
                if result['success']:
                    # Add file type to metadata
                    result['metadata']['file_type'] = 'image' if is_image else 'document'
                    if is_image:
                        result['metadata']['requires_ocr'] = True
                    
                    # Store in enhanced RAG service
                    # Use consistent file_id creation from RAG service
                    unique_file_id = rag_service.create_file_id(user_id, session_id, original_filename)
                    chunks_result = rag_service.store_document_chunks(
                        result['chunks'], 
                        unique_file_id,  # unique file_id using RAG service method
                        result['metadata'],
                        session_id,  # session_id
                        user_id  # user_id (MongoDB _id)
                    )
                    
                    if chunks_result.get('success', True):
                        doc_process_time = time.time() - doc_process_start
                        file_total_time = time.time() - file_start_time
                        
                        logger.info(f"Successfully processed {original_filename} - Size: {file_size} bytes, Processing time: {doc_process_time:.2f}s")
                        
                        processed_files.append({
                            'filename': original_filename,
                            'file_type': 'image' if is_image else 'document',
                            'processing_stats': result.get('processing_stats', {}),
                            'metadata': result.get('metadata', {}),
                            'timing': {
                                'save_time': save_time,
                                'processing_time': doc_process_time,
                                'total_time': file_total_time,
                                'file_size': file_size
                            }
                        })
                    else:
                        processing_errors.append(f'Failed to store {original_filename}: {chunks_result.get("error", "Unknown error")}')
                else:
                    processing_errors.append(f'Failed to process {original_filename}: {result.get("error", "Unknown error")}')
            else:
                # Legacy processing
                try:
                    doc_result = document_service.process_document(local_path, session_text, original_filename)
                    if doc_result['success']:
                        # Use consistent file_id creation from RAG service
                        unique_file_id = rag_service.create_file_id(user_id, session_id, original_filename)
                        rag_result = rag_service.store_document_chunks(doc_result['chunks'], unique_file_id, doc_result['metadata'], session_id, user_id)
                        if rag_result.get('success', True):
                            processed_files.append({
                                'filename': original_filename,
                                'file_type': 'document',
                                'processing_stats': {},
                                'metadata': doc_result.get('metadata', {})
                            })
                        else:
                            processing_errors.append(f'RAG storage failed for {original_filename}')
                    else:
                        processing_errors.append(f'Document processing failed for {original_filename}')
                except Exception as e:
                    logger.error(f"Legacy processing error for {original_filename}: {e}")
                    processing_errors.append(f'Processing error for {original_filename}: {str(e)}')
            
        except Exception as e:
            logger.error(f'Error processing {original_filename}: {str(e)}')
            processing_errors.append(f'Error with {original_filename}: {str(e)}')
        
        finally:
            # Clean up local file
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {local_path}: {e}")

    # Generate QR code
    chat_url = "https://bot.xplorease.com/?session={}".format(session_id)
    if logo_url is not None:
        logo_img = read_logo(logo_url)
        qr = generate_qr_code_with_logo(chat_url, logo_img, qr_name)
    else:
        qr = generate_simple_qr_code(chat_url, qr_name)

    qr_name_s3 = user_name + "/{}_qr.png".format(session_text)
    qr_name_s3 = '{0}/{1}'.format('xplorease', '{}'.format(qr_name_s3))
    qr_bytes = io.BytesIO()
    qr.save(qr_bytes, format="PNG")
    qr_bytes.seek(0)
    qr_url = upload(qr_bytes, qr_name_s3)

    # Prepare response messages
    messages = []
    if processed_files:
        messages.append(f'Successfully processed {len(processed_files)} file(s).')
    if skipped_files:
        messages.append(f'Skipped {len(skipped_files)} non-supported file(s): {", ".join(skipped_files)}')
    if processing_errors:
        messages.extend(processing_errors)

    if processed_files:
        # Generate automatic sample questions after successful file processing
        sample_questions = []
        try:
            # Use unified question generation with fallbacks
            logger.info(f"Generating automatic sample questions for session: {session_id}")
            sample_questions = generate_questions_with_fallbacks(
                rag_service, 
                session_id=session_id, 
                user_id=user_id, 
                email=email
            )
            logger.info(f"Generated {len(sample_questions)} automatic sample questions for session {session_id}")
                
        except Exception as question_error:
            logger.warning(f"Failed to generate automatic sample questions: {question_error}")
            sample_questions = get_default_sample_questions()
        
        # Format sample questions with HTML like answers
        formatted_sample_questions = format_questions_for_frontend(sample_questions)
        
        # Store session data
        timestamp = datetime.now().isoformat()
        session_item = {
            "session_id": session_id,
            "user_id": email,
            "qr_name": qr_name,
            "chat_url": chat_url,
            "qr_url": qr_url,
            "timestamp": timestamp,
            "logo_url": logo_url,
            "file_urls": file_urls,
            "file_names": file_names,
            "enhanced_mode": ENHANCED_SERVICES,
            "sample_questions": formatted_sample_questions  # Store HTML-formatted for DB
        }
        
        with MongoClient(server_uri) as mongo_client_server:
            db_server = mongo_client_server[DB]
            user_container = db_server[USER_CONTAINER]
            session_container = db_server[SESSION_CONTAINER]
            user_container.update_one({"email": email}, {'$push': {'session_id': session_id}})
            session_container.insert_one(session_item)

        return jsonify({
            "success": True,
            "message": messages,
            "data": [{
                "session_id": session_id,
                "user_id": email,
                "qr_name": qr_name,
                "chat_url": chat_url,
                "qr_url": qr_url,
                "logo_url": logo_url,
                "sample_questions": formatted_sample_questions  # Return formatted for frontend
            }],
            "status": 200
        }), 200
    else:
        return jsonify({
            "success": False,
            "status": 400,
            "errors": messages if messages else ['No files were successfully processed.']
        }), 400

@app.route("/answer_question", methods=["POST"])
def answer_question():
    """Enhanced question answering with conversation memory and improved error handling"""
    try:
        params = request.get_json()
        session_id = params.get("session_id")
        question = params.get("question")
        conversation_history = params.get("conversation_history", [])
        
        if not session_id:
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["Session ID is required"]
            }), 400
            
        if not question:
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["Question is required"]
            }), 400
            
        # Generate unique user_id from MongoDB user _id
        # This will be set after user lookup from session validation
        user_id = None
            
    except Exception as e:
        logger.error(f"Request parsing error: {e}")
        return jsonify({
            "success": False,
            "status": 400,
            "errors": ["Key-Error. Please provide the session id and the question"]
        }), 400

    # Language detection and translation (commented out - Google Translate disabled)
    original_question = question
    detected_language = 'en'
    
    # Google Translate functionality commented out
    # if translate_client:
    #     try:
    #         lang_detection_result = translate_client.detect_language(question)
    #         detected_language = lang_detection_result.get('language', 'en')
    #         
    #         if detected_language != 'en':
    #             lang_eng_result = translate_client.translate(question, target_language='en')
    #             question = lang_eng_result['translatedText']
    #             logger.info(f"Translated question from {detected_language} to English")
    #     except Exception as e:
    #         logger.warning(f'Language translation failed: {e}')

    # Session validation and user lookup
    with MongoClient(server_uri) as mongo_client_server:
        db_server = mongo_client_server[DB]
        user_container = db_server[USER_CONTAINER]
        session_container = db_server[SESSION_CONTAINER]
        item = json.loads(dumps(user_container.find_one({"session_id": session_id})))
        
        if not item:
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["This session does not exist. Please create your session by uploading your documents."]
            }), 400

    # Extract user's MongoDB _id to use as user_id for collection naming
    user_id = extract_user_id(item)
    user_name = user_id  # Keep backward compatibility
    session_text = user_name + "_" + session_id
    
    try:
        # Enhanced question answering
        config = get_enhanced_services_config()
        ENHANCED_SERVICES = config['ENHANCED_SERVICES']
        
        if ENHANCED_SERVICES:
            # Get or create RAG service for this session
            rag_service = get_or_create_rag_service(session_id, user_id)
            
            # Use enhanced RAG service with conversation memory and user_id
            result = rag_service.answer_question(
                file_id=session_text,  # Use session_text as file_id for compatibility
                question=question,
                conversation_history=conversation_history,
                session_id=session_id,
                user_id=user_id  # Include user_id for collection naming
            )
            
            if result.get('success', True):
                answer = result.get('answer', 'No answer generated')
                
                # Enhanced response data
                response_data = {
                    'answer': answer,
                    'confidence_score': result.get('confidence_score', 0.5),
                    'processing_time': result.get('processing_time', 0),
                    'question_type': result.get('question_type', 'general'),
                    'sources_used': len(result.get('sources', [])),
                    'enhanced_features': True
                }
                
                # Add source information if available
                if result.get('sources'):
                    response_data['source_preview'] = result['sources'][0].get('content_preview', '')[:200]
                
            else:
                answer = f"I encountered an error: {result.get('error', 'Unknown error')}"
                response_data = {
                    'answer': answer,
                    'confidence_score': 0.0,
                    'processing_time': 0,
                    'enhanced_features': True,
                    'error': result.get('error', 'Unknown error')
                }
                
        else:
            # Legacy RAG service
            rag_service = get_or_create_rag_service(session_id, user_id)
            answer_result = rag_service.answer_question(
                file_id=session_text, 
                question=question,
                session_id=session_id,
                user_id=user_id
            )
            
            if answer_result.get('success', True):
                answer = answer_result.get('answer', 'No answer generated')
                response_data = {
                    'answer': answer,
                    'confidence_score': 0.7,  # Default confidence
                    'processing_time': 0,
                    'enhanced_features': False
                }
            else:
                answer = f"I encountered an error: {answer_result.get('error', 'Unknown error')}"
                response_data = {
                    'answer': answer,
                    'confidence_score': 0.0,
                    'processing_time': 0,
                    'enhanced_features': False,
                    'error': answer_result.get('error', 'Unknown error')
                }

        # Translate answer back if needed (commented out - Google Translate disabled)
        # if translate_client and detected_language != 'en':
        #     try:
        #         translated_answer = translate_client.translate(answer, target_language=detected_language)
        #         answer = translated_answer['translatedText']
        #         response_data['answer'] = answer
        #         response_data['translated_from'] = 'en'
        #         response_data['translated_to'] = detected_language
        #     except Exception as e:
        #         logger.warning(f'Answer translation failed: {e}')

        # Enhanced HTML entity cleaning and text formatting
        answer = html.unescape(answer)  # Decode HTML entities like &quot; &#x27; etc.
        answer = answer.replace('&quot;', '"').replace('&#x27;', "'").replace('&amp;', '&')  # Additional cleanup
        answer = answer.replace('&lt;', '<').replace('&gt;', '>')  # Clean angle brackets
        # Remove any remaining HTML entities
        answer = re.sub(r'&#?\w+;', '', answer)  # Remove any remaining HTML entities
        response_data['answer'] = answer

        # Store Q&A in database
        timestamp = datetime.now().isoformat()
        qa_item = {
            "user_id": user_id,  # Include user_id for collection naming
            "session_id": session_id,
            "question": original_question,
            "answer": answer,
            "timestamp": timestamp,
            "enhanced_mode": ENHANCED_SERVICES,
            "confidence_score": response_data.get('confidence_score', 0.0)
        }
        
        with MongoClient(server_uri) as mongo_client_server:
            db_server = mongo_client_server[DB]
            qa_container = db_server[QA_CONTAINER]
            user_logs_container = db_server[LOGS_CONTAINER]
            qa_container.insert_one(qa_item)
            user_logs(user_logs_container, item.get('email', 'unknown'), "Question Answered", session_id)

        # Format answer with line breaks for display
        display_answer = answer.replace('\n', '<br />')
        
        # Return response maintaining backward compatibility
        return jsonify({
            "data": [{"answer": display_answer}], 
            "success": True, 
            "status": 200
        }), 200

    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        return jsonify({
            "success": False,
            "status": 500,
            "errors": [f"Failed to process question: {str(e)}"]
        }), 500

@app.route("/generate_sample_questions", methods=["POST"])
# @jwt_required  # JWT authentication disabled
def generate_sample_questions():  # Removed current_user parameter since JWT is disabled
    """Enhanced sample question generation"""
    try:
        params = request.get_json()
        session_id = params.get("session_id")
        email = params.get("email")
        
        if not session_id or not email:
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["Session ID and email are required"]
            }), 400
        
        config = get_enhanced_services_config()
        ENHANCED_SERVICES = config['ENHANCED_SERVICES']
        ENHANCED_MODE = config['ENHANCED_MODE']
        
        if ENHANCED_SERVICES:
            # Use enhanced RAG service for sample questions
            try:
                # Find the user to get the proper session text
                user = find_user_by_email_and_session(email, session_id)
                
                if not user:
                    return jsonify({
                        "success": False,
                        "status": 400,
                        "errors": ["Invalid session"]
                    }), 400
                
                user_name = extract_user_id(user)
                user_id = user_name  # Same value for consistency
                session_text = user_name + "_" + session_id
                
                # Get RAG service
                rag_service = get_or_create_rag_service(session_id, user_name)
                
                # Use unified question generation with fallbacks
                try:
                    logger.info(f"Generating questions using unified pipeline for session: {session_id}")
                    questions = generate_questions_with_fallbacks(
                        rag_service, 
                        session_id=session_id, 
                        user_id=user_id, 
                        email=email
                    )
                        
                except Exception as search_error:
                    logger.warning(f"Unified question generation failed: {search_error}, using default")
                    questions = get_default_sample_questions()
            except Exception as e:
                logger.error(f"Enhanced sample question generation failed: {e}")
                questions = get_default_sample_questions()
        else:
            # Legacy sample question generation
            try:
                rag_service = get_or_create_rag_service(session_id, user_name)
                chunks = rag_service.get_document_chunks(session_id)
                questions = rag_service.generate_sample_questions(session_id, user_name, chunks=chunks)
                
                # Clean each question without numbering for better readability
                cleaned_questions = []
                for q in questions[:4]:
                    cleaned_q = clean_question_text(q)
                    if cleaned_q:
                        cleaned_questions.append(cleaned_q)
                
                questions = cleaned_questions
            except Exception as e:
                logger.error(f"Legacy sample question generation failed: {e}")
                questions = get_default_sample_questions()
        
        # Format sample questions with HTML like answers BEFORE storing
        formatted_questions = format_questions_for_frontend(questions)
        
        # Store HTML-formatted sample questions in session
        with MongoClient(server_uri) as mongo_client_server:
            db_server = mongo_client_server[DB]
            session_container = db_server[SESSION_CONTAINER]
            session_container.update_one(
                {'user_id': email, "session_id": session_id}, 
                {"$set": {'sample_questions': formatted_questions}}
            )
        
        return jsonify({
            "data": [{"answer": formatted_questions}], 
            "success": True, 
            "status": 200
        }), 200
        
    except Exception as e:
        logger.error(f"Sample questions error: {str(e)}")
        return jsonify({
            "success": False,
            "status": 500,
            "errors": [f"Failed to generate sample questions: {str(e)}"]
        }), 500

@app.route("/replace_file", methods=["POST"])
# @jwt_required  # JWT authentication disabled
def replace_file():  # Removed current_user parameter since JWT is disabled
    """Enhanced file replacement with new architecture support"""
    try:
        new_files = request.files.getlist("new_files")
        email = request.form.get("email")
        session_id = request.form.get('session_id')
        old_file_names = request.form.get("file_names")
        old_file_urls = request.form.get("file_urls")
        
        if not all([email, session_id, old_file_names, old_file_urls, new_files]):
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["Missing required parameters"]
            }), 400
        
        old_file_names = validations.str_to_array(old_file_names)
        old_file_urls = validations.str_to_array(old_file_urls)
        
    except Exception as e:
        logger.error(f"Replace file parameter error: {e}")
        return jsonify({
            "success": False,
            "status": 400,
            "errors": ["Invalid request parameters"]
        }), 400

    with MongoClient(server_uri) as mongo_client_server:
        db_server = mongo_client_server[DB]
        user_container = db_server[USER_CONTAINER]
        session_container = db_server[SESSION_CONTAINER]
        user_logs_container = db_server[LOGS_CONTAINER]
        
        user = json.loads(dumps(user_container.find_one({
            "email": email, 
            "session_id": {'$in': [session_id]}
        })))
        
        if not user:
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["User or session not found"]
            }), 400
        
        # Extract user's MongoDB _id to use as user_id for collection naming
        user_id = extract_user_id(user)

        # Validate files exist
        session_item = json.loads(dumps(session_container.find_one({
            "user_id": email, 
            "session_id": session_id, 
            'file_names': {'$in': old_file_names}
        }, {'_id': False, "file_urls": True, "file_names": True})))
        
        if not session_item:
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["Selected file(s) do not exist"]
            }), 400

        user_name = extract_user_id(user)
        session_text = user_name + "_" + session_id
        
        # Get RAG service
        rag_service = get_or_create_rag_service(session_id, user_name)
        
        # Delete old documents from vector store by individual file names
        try:
            for old_file_name in old_file_names:
                # Use individual file name as file_id for deletion
                old_file_id = f"{session_text}_{old_file_name}"
                rag_service.delete_documents_by_file_id(old_file_id, session_id, user_name)
                logger.info(f"Deleted documents for file: {old_file_name}")
        except Exception as e:
            logger.warning(f"Failed to delete old documents from vector store: {e}")
        
        # Delete old files from S3
        for file_url in old_file_urls:
            try:
                delete_file(file_url)
            except Exception as e:
                logger.warning(f"Failed to delete S3 file {file_url}: {e}")
        
        # Process new files
        new_file_urls = []
        new_file_names = []
        s3_keys = []
        processed_files = []
        uploads_dir = "uploads"
        
        # Enhanced mode flag
        config = get_enhanced_services_config()
        ENHANCED_SERVICES = config['ENHANCED_SERVICES']
        
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        for file in new_files:
            if not file.filename:
                continue
                
            file_name = secure_filename(file.filename)
            new_file_names.append(file_name)
            
            file_name_s3 = user_name + "/{}_{}".format(session_id, file_name)
            s3_filename = '{0}/{1}'.format('xplorease', '{}'.format(file_name_s3))
            s3_keys.append(s3_filename)
            new_file_urls.append(upload(file, s3_filename))
            
            # Process with enhanced services
            if ENHANCED_SERVICES:
                try:
                    # Save file temporarily
                    local_path = os.path.join(uploads_dir, file_name)
                    file.save(local_path)
                    
                    # Process with enhanced document service
                    result = document_service.process_document(local_path, session_text, file_name)
                    
                    if result['success']:
                        # Use individual file_id for each file
                        unique_file_id = f"{session_text}_{file_name}"
                        chunks_result = rag_service.store_document_chunks(
                            result['chunks'], 
                            unique_file_id,  # unique file_id instead of session_text
                            result['metadata'],
                            session_id,  # session_id
                            user_id  # user_id (MongoDB _id)
                        )
                        
                        if chunks_result.get('success', True):
                            processed_files.append({
                                'filename': file_name,
                                'processing_stats': result.get('processing_stats', {}),
                                'metadata': result.get('metadata', {})
                            })
                        else:
                            logger.error(f"Failed to store chunks for {file_name}: {chunks_result.get('error', 'Unknown error')}")
                    else:
                        logger.error(f"Failed to process document {file_name}: {result.get('error', 'Unknown error')}")
                    
                    # Clean up
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        
                except Exception as e:
                    logger.error(f"Enhanced file processing failed for {file_name}: {e}")
            else:
                # Legacy processing fallback
                try:
                    local_path = os.path.join(uploads_dir, file_name)
                    file.save(local_path)
                    
                    doc_result = document_service.process_document(local_path, session_text, file_name)
                    if doc_result['success']:
                        # Use individual file_id for each file
                        unique_file_id = f"{session_text}_{file_name}"
                        rag_result = rag_service.store_document_chunks(
                            doc_result['chunks'],
                            unique_file_id,  # unique file_id instead of session_text
                            doc_result['metadata'],
                            session_id,  # session_id
                            user_id  # user_id (MongoDB _id)
                        )
                        
                        if rag_result.get('success', True):
                            processed_files.append({
                                'filename': file_name,
                                'processing_stats': {},
                                'metadata': doc_result.get('metadata', {})
                            })
                        else:
                            logger.error(f"Failed to store chunks for {file_name}: {rag_result.get('error', 'Unknown error')}")
                    else:
                        logger.error(f"Failed to process document {file_name}: {doc_result.get('error', 'Unknown error')}")
                    
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        
                except Exception as e:
                    logger.error(f"Legacy file processing failed for {file_name}: {e}")

        # Remove old files from database
        session_container.update_one(
            {"user_id": email, "session_id": session_id},
            {'$pull': {
                'file_names': {'$in': old_file_names},
                'file_urls': {'$in': old_file_urls}
            }}
        )

        # Update session with new files
        session_container.update_one(
            {"user_id": email, "session_id": session_id},
            {'$push': {
                'file_names': {"$each": new_file_names}, 
                'file_urls': {"$each": new_file_urls}
            }}
        )
        
        user_logs(user_logs_container, email, "replace file(s)", session_id)
        
        # Automatically regenerate sample questions for the new files
        new_sample_questions = []
        try:
            logger.info("Regenerating sample questions for replaced files...")
            
            # Use unified question generation with all fallback strategies
            new_sample_questions = generate_questions_with_fallbacks(
                rag_service, 
                session_id=session_id, 
                user_id=user_id, 
                email=email
            )
                
        except Exception as question_error:
            logger.warning(f"Failed to generate new sample questions: {question_error}")
            new_sample_questions = get_default_sample_questions()
        
        # Format sample questions with HTML like answers BEFORE storing
        formatted_new_questions = format_questions_for_frontend(new_sample_questions)
        
        # Update session with new sample questions (replace old ones) - store HTML formatted
        session_container.update_one(
            {'user_id': email, "session_id": session_id}, 
            {"$set": {'sample_questions': formatted_new_questions}}
        )
        
        logger.info(f"File replacement completed with {len(new_sample_questions)} new sample questions")
        
        return jsonify({
            "success": True,
            "message": ["File(s) replaced successfully", f"Generated {len(new_sample_questions)} new sample questions"],
            "data": {
                "processed_files": len(processed_files),
                "enhanced_mode": ENHANCED_SERVICES,
                "new_sample_questions": formatted_new_questions  # Return formatted for frontend
            },
            "status": 200
        }), 200

@app.route("/delete_selected_files", methods=["POST"])
# @jwt_required  # JWT authentication disabled
def delete_selected_files():  # Removed current_user parameter since JWT is disabled
    try:
        params = request.get_json()
        email = params["email"]
        session_id = params['session_id']
        file_names = validations.str_to_array(params["file_names"])
        file_urls = validations.str_to_array(params["file_urls"])
        
        with MongoClient(server_uri) as mongo_client_server:
            db_server = mongo_client_server[DB]
            user_container = db_server[USER_CONTAINER]
            session_container = db_server[SESSION_CONTAINER]
            user_logs_container = db_server[LOGS_CONTAINER]
            
            user = json.loads(dumps(user_container.find_one({"email": email, "session_id": {'$in': [session_id]}})))
            if user:
                user_name = user['_id']['$oid']
                session_text = user_name + "_" + session_id
                
                # Get RAG service and delete specific documents by file names
                rag_service = get_or_create_rag_service(session_id, user_name)
                
                # Delete documents for each specific file
                for file_name in file_names:
                    try:
                        # Use individual file name as file_id for deletion
                        file_id = f"{session_text}_{file_name}"
                        rag_service.delete_documents_by_file_id(file_id, session_id, user_name)
                        logger.info(f"Deleted documents for file: {file_name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete documents for file {file_name}: {e}")
                
                # Delete from S3
                for url in file_urls:
                    delete_file(url)
                
                # Update MongoDB
                session_container.update_one(
                    {"user_id": email, "session_id": session_id},
                    {'$pull': {'file_names': {'$in': file_names}, 'file_urls': {'$in': file_urls}}}
                )
                
                user_logs(user_logs_container, email, "delete file(s)", session_id)
                
                # Automatically regenerate sample questions after file deletion
                updated_sample_questions = []
                try:
                    logger.info("Regenerating sample questions after file deletion...")
                    
                    # Check if there are still files remaining in the session
                    remaining_session = session_container.find_one(
                        {"user_id": email, "session_id": session_id},
                        {'_id': False, 'file_names': True}
                    )
                    
                    if remaining_session and remaining_session.get('file_names'):
                        # There are still files remaining, generate questions for them
                        logger.info(f"Files remaining in session: {remaining_session.get('file_names')}")
                        search_results = rag_service._search_in_session("overview summary content", session_id, user_id=user_name)
                        
                        if search_results:
                            # Convert search results to Document objects
                            chunks = convert_search_results_to_documents(search_results)
                            
                            if chunks:
                                updated_sample_questions = generate_sample_questions_from_chunks(rag_service, chunks, session_id, user_name)
                                logger.info(f"Generated {len(updated_sample_questions)} questions for remaining files")
                            else:
                                logger.warning("No chunks found for remaining files, using MongoDB fallback")
                                updated_sample_questions = generate_questions_from_mongodb_session(email, session_id)
                        else:
                            logger.warning("No search results for remaining files, using MongoDB fallback")
                            updated_sample_questions = generate_questions_from_mongodb_session(email, session_id)
                    else:
                        # No files remaining, clear all sample questions
                        updated_sample_questions = []
                        logger.info("No files remaining in session, clearing all sample questions")
                    
                    # Format and update session with new sample questions
                    formatted_sample_questions = format_questions_for_frontend(updated_sample_questions)
                    session_container.update_one(
                        {'user_id': email, "session_id": session_id}, 
                        {"$set": {'sample_questions': formatted_sample_questions}}
                    )
                    
                except Exception as question_error:
                    logger.warning(f"Failed to regenerate sample questions after deletion: {question_error}")
                    # Check if there are files remaining to decide what to do
                    try:
                        remaining_session = session_container.find_one(
                            {"user_id": email, "session_id": session_id},
                            {'_id': False, 'file_names': True}
                        )
                        
                        if remaining_session and remaining_session.get('file_names'):
                            # Files remain but question generation failed, try a simpler fallback
                            updated_sample_questions = generate_questions_from_mongodb_session(email, session_id)
                            logger.info("Using MongoDB fallback due to question generation error")
                        else:
                            # No files remain, clear questions
                            updated_sample_questions = []
                            logger.info("No files remaining, clearing sample questions due to error")
                    except:
                        # Ultimate fallback - assume no files and clear questions
                        updated_sample_questions = []
                        logger.warning("Error checking remaining files, clearing sample questions")
                    
                    # Format and store updated questions
                    formatted_sample_questions = format_questions_for_frontend(updated_sample_questions)
                    session_container.update_one(
                        {'user_id': email, "session_id": session_id}, 
                        {"$set": {'sample_questions': formatted_sample_questions}}
                    )
                
                return jsonify({
                    "success": True,
                    "message": [
                        "File(s) deleted successfully", 
                        f"Updated sample questions ({len(updated_sample_questions)} questions)" if updated_sample_questions else "Sample questions cleared (no files remaining)"
                    ],
                    "data": {
                        "updated_sample_questions": formatted_sample_questions,
                        "remaining_files": bool(remaining_session and remaining_session.get('file_names'))
                    },
                    "status": 200
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "status": 400,
                    "errors": ["Selected file(s) do not exist"]
                }), 400
                
    except Exception as e:
        logger.error(f"Delete files error: {str(e)}")
        return jsonify({
            "success": False,
            "status": 500,
            "errors": [f"Delete failed: {str(e)}"]
        }), 500

@app.route("/delete_session", methods=["POST"])
# @jwt_required  # JWT authentication disabled
def delete_session():  # Removed current_user parameter since JWT is disabled
    try:
        params = request.get_json()
        email = params["email"]
        session_id = params['session_id']
        
        with MongoClient(server_uri) as mongo_client_server:
            db_server = mongo_client_server[DB]
            user_container = db_server[USER_CONTAINER]
            session_container = db_server[SESSION_CONTAINER]
            user_logs_container = db_server[LOGS_CONTAINER]
            
            user = json.loads(dumps(user_container.find_one({"email": email, "session_id": {'$in': [session_id]}})))
            if user and user.get('session_id'):
                user_session = json.loads(dumps(session_container.find_one(
                    {"user_id": email, "session_id": session_id},
                    {'_id': False, 'session_id': True, 'qr_name': True, 'qr_url': True, "file_names": True, "file_urls": True}
                )))
                
                if not user_session:
                    return jsonify({
                        "success": False,
                        "status": 400,
                        "errors": ["Session does not exist"]
                    }), 400
                
                user_name = user['_id']['$oid']
                session_text = user_name + "_" + session_id
                
                # Get RAG service and delete from vector store
                try:
                    rag_service = get_or_create_rag_service(session_id, user_name)
                    rag_service.delete_session_collection(session_id, user_name)
                    
                    # Clean up session from memory
                    session_key = f"{user_name}_{session_id}"
                    if session_key in session_rag_services:
                        del session_rag_services[session_key]
                except Exception as e:
                    logger.error(f"Error deleting from vector store: {str(e)}")
                
                # Delete QR code
                delete_file(user_session['qr_url'])
                
                # Delete all files
                for file_url in user_session.get('file_urls', []):
                    delete_file(file_url)
                
                # Update MongoDB
                session_container.delete_one({"user_id": email, "session_id": session_id})
                user_container.update_one({"email": email}, {'$pull': {'session_id': session_id}})
                user_container.update_one({"email": email}, {'$push': {'deleted_session_id': session_id}})
                
                user_logs(user_logs_container, email, "delete session", session_id)
                
                return jsonify({
                    "messages": ["Deleted session successfully"],
                    "success": True,
                    "status": 200
                }), 200
            else:
                return jsonify({
                    "errors": ["Session does not exist"],
                    "success": False,
                    "status": 400
                }), 400
                
    except Exception as e:
        logger.error(f"Delete session error: {str(e)}")
        return jsonify({
            "success": False,
            "status": 500,
            "errors": [f"Delete session failed: {str(e)}"]
        }), 500

@app.route("/append_session_content", methods=["POST"])
# @jwt_required  # JWT authentication disabled
def append_session_content():
    try:
        files = request.files.getlist("file")
        email = request.form["email"]
        session_id = request.form['session_id']
        
        with MongoClient(server_uri) as mongo_client_server:
            db_server = mongo_client_server[DB]
            user_container = db_server[USER_CONTAINER]
            session_container = db_server[SESSION_CONTAINER]
            user_logs_container = db_server[LOGS_CONTAINER]
            
            user = json.loads(dumps(user_container.find_one({"email": email, "session_id": {'$in': [session_id]}})))
            if not user or not user.get('session_id'):
                return jsonify({
                    "success": False,
                    "status": 400,
                    "errors": ["User or session not found"]
                }), 400
            
            # Extract user's MongoDB _id to use as user_id for collection naming
            user_id = user["_id"]["$oid"]  # This is the actual MongoDB ObjectId as string
            
            user_name = user['_id']['$oid']
            session_text = user_name + "_" + session_id
            
            # Process new files (similar to process_file but append to existing session)
            processed_files = []
            new_file_urls = []
            new_file_names = []
            
            # Get RAG service
            rag_service = get_or_create_rag_service(session_id, user_name)
            
            for file in files:
                if not file.filename:
                    continue
                    
                file_name = secure_filename(file.filename)
                new_file_names.append(file_name)
                
                # Process file and add to existing session
                try:
                    local_path = os.path.join("uploads", file_name)
                    file.save(local_path)
                    
                    result = document_service.process_document(local_path, session_text, file_name)
                    if result['success']:
                        # Use individual file_id for each file
                        unique_file_id = f"{session_text}_{file_name}"
                        chunks_result = rag_service.store_document_chunks(
                            result['chunks'], 
                            unique_file_id,  # unique file_id instead of session_text
                            result['metadata'],
                            session_id,  # session_id
                            user_id  # user_id (MongoDB _id)
                        )
                        
                        if chunks_result.get('success', True):
                            processed_files.append(file_name)
                            new_file_urls.append(f"placeholder_url_{file_name}")
                    
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")
            
            # Update session with new files
            if processed_files:
                session_container.update_one(
                    {"user_id": email, "session_id": session_id},
                    {'$push': {
                        'file_names': {"$each": new_file_names}, 
                        'file_urls': {"$each": new_file_urls}
                    }}
                )
                
                user_logs(user_logs_container, email, "append file(s)", session_id)
                
                return jsonify({
                    "success": True,
                    "message": [f"Successfully appended {len(processed_files)} file(s)"],
                    "status": 200
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "status": 400,
                    "errors": ["No files were successfully processed"]
                }), 400
                
    except Exception as e:
        logger.error(f"Append session content error: {str(e)}")
        return jsonify({
            "success": False,
            "status": 500,
            "errors": [f"Append failed: {str(e)}"]
        }), 500

@app.route("/get_session_details", methods=["POST"])
def get_session_details():
    """Get session details including uploaded files"""
    try:
        params = request.get_json()
        session_id = params["session_id"]
        email = params["email"]
    except Exception as e:
        logger.error(f"Missing parameters: {e}")
        return jsonify({
            "success": False,
            "status": 400,
            "errors": ["Please provide session_id and email"]
        }), 400

    with MongoClient(server_uri) as mongo_client_server:
        db_server = mongo_client_server[DB]
        user_container = db_server[USER_CONTAINER]
        session_container = db_server[SESSION_CONTAINER]
        
        # Check if user has access to this session
        user = user_container.find_one({"email": email, "session_id": {'$in': [session_id]}})
        if not user:
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["Invalid session or unauthorized access"]
            }), 400
        
        # Get session details
        session_details = json.loads(dumps(session_container.find_one(
            {"user_id": email, "session_id": session_id},
            {'_id': False}
        )))
        
        if not session_details:
            return jsonify({
                "success": False,
                "status": 400,
                "errors": ["Session not found"]
            }), 400
        
        # Format file information
        files_list = []
        file_names = session_details.get('file_names', [])
        file_urls = session_details.get('file_urls', [])
        
        for i in range(len(file_names)):
            files_list.append({
                'file_name': file_names[i],
                'file_url': file_urls[i] if i < len(file_urls) else None
            })
        
        session_details['uploaded_files'] = files_list
        if 'file_names' in session_details:
            session_details.pop('file_names')
        if 'file_urls' in session_details:
            session_details.pop('file_urls')
        
        return jsonify({
            "success": True,
            "data": [session_details],
            "status": 200
        }), 200

@app.route("/healthcheck", methods=["GET", "POST"])
def healthcheck():
    """Health check endpoint with enhanced info"""
    try:
        return jsonify({
            "status": "healthy",
            "message": "I'm alive :)",
            "version": "2.0.0-enhanced",
            "timestamp": datetime.now().isoformat(),
            "features": ["enhanced_rag", "ocr_processing", "multi_format_support", "conversation_memory"]
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return "I'm alive :)", 200

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'status': 413,
        'errors': ['File too large. Maximum size is 50MB.']
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'status': 404,
        'errors': ['Endpoint not found']
    }), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {str(e)}")
    return jsonify({
        'success': False,
        'status': 500,
        'errors': ['Internal server error']
    }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=SERVER_PORT, debug=True, threaded=True)
