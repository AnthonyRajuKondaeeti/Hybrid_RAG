from flask import Flask, request, jsonify, current_app, sessions
from flask_cors import CORS
import os
import uuid
import hashlib
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import logging
import jwt
from functools import wraps
from typing import Dict, Any, List

# Ensure environment is loaded first
from dotenv import load_dotenv
load_dotenv()

from qdrant_client.http import models 
from langchain_core.documents import Document

# Import Config after loading environment
try:
    from config import Config
    # Make Config available as a module-level variable
    APP_CONFIG = Config
except ImportError as e:
    print(f"app Failed to import Config: {e}")
    raise

from services.document_service import DocumentService
from services.ocr_processor import OCRProcessor
from services.rag_service import EnhancedRAGService  # Using the enhanced version
from utils.response_formatter import ResponseFormatter

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
    
    # Reduce noise from third-party libraries
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('qdrant_client').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Simple JWT secret (you should use a proper secret in production)
JWT_SECRET = "your-jwt-secret-key"

def log_file_processing_summary(filename, file_size, file_ext, save_time, processing_time, vector_time, total_time):
    """Consolidated file processing log"""
    rate_mb_per_sec = (file_size / 1024 / 1024) / total_time if total_time > 0 else 0
    
    logger.info(f"Document processing: {filename} | "
               f"Size: {file_size:,}B | Total: {total_time:.2f}s | "
               f"Rate: {rate_mb_per_sec:.2f}MB/s")
    
    # Only warn for genuinely slow processing
    if (file_ext in ['xlsx', 'xlsm'] and total_time > 15) or total_time > 45:
        logger.warning(f"Slow processing: {filename} took {total_time:.2f}s")

def validate_session(session_id, sessions, require_rag_service=True):
    """Centralized session validation"""
    if not session_id:
        return {'valid': False, 'error': 'session_id is required', 'status': 400}
    
    if session_id not in sessions:
        logger.warning(f"Invalid session: {session_id[:8]}...")
        return {'valid': False, 'error': 'Invalid session', 'status': 400}
    
    session_data = sessions[session_id]
    
    if require_rag_service and 'rag_service' not in session_data:
        logger.error(f"Session {session_id[:8]}... missing rag_service")
        return {'valid': False, 'error': 'Session corrupted - missing RAG service', 'status': 500}
    
    return {'valid': True, 'session_data': session_data}

def create_response(success, data=None, message=None, errors=None, status=200):
    """Simplified response creator"""
    response = {
        'success': success,
        'status': status
    }
    
    if success:
        if data is not None:
            response['data'] = data
        if message:
            response['message'] = message if isinstance(message, list) else [message]
    else:
        response['errors'] = errors if isinstance(errors, list) else [errors] if errors else ['Unknown error']
    
    return jsonify(response), status

def log_session_action(session_id, action, details=None):
    """Standardized session action logging"""
    session_prefix = session_id[:8] + "..." if session_id else "unknown"
    log_msg = f"Session {session_prefix}: {action}"
    if details:
        log_msg += f" | {details}"
    logger.info(log_msg)

def validate_image_file(file):
    """Validate uploaded image file."""
    if not file or not file.filename:
        return False, "No file selected"
    
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    image_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    
    if file_ext not in image_formats:
        return False, f"Unsupported image format. Supported formats: {', '.join(image_formats)}"
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    max_size_bytes = 10 * 1024 * 1024  # 10MB limit for images
    if file_size > max_size_bytes:
        return False, f"File too large. Maximum size: 10MB"
    
    return True, ""

def create_app():
    app = Flask(__name__)
    
    # Validate configuration
    APP_CONFIG.validate()
    
    # Configuration
    app.config.from_object(APP_CONFIG)
    
    # Enable CORS for V2 compatibility
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })
    
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize services
    document_service = DocumentService(
        chunk_size=APP_CONFIG.CHUNK_SIZE,
        chunk_overlap=APP_CONFIG.CHUNK_OVERLAP
    )
    ocr_processor = OCRProcessor()
    # In-memory session storage (use Redis in production)
    sessions: Dict[str, Dict[str, Any]] = {}
    
    def verify_jwt_token(f):
        """JWT token verification decorator"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = None
            
            # Get token from Authorization header
            if 'Authorization' in request.headers:
                auth_header = request.headers['Authorization']
                try:
                    token = auth_header.split(" ")[1]  # Bearer <token>
                except IndexError:
                    return jsonify({
                        'success': False,
                        'status': 401,
                        'errors': ['Invalid authorization header format']
                    }), 401
            
            if not token:
                return jsonify({
                    'success': False,
                    'status': 401,
                    'errors': ['Token is missing']
                }), 401
            
            try:
                # Decode JWT token
                payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
                current_user_email = payload['email']
                request.current_user = current_user_email
            except jwt.ExpiredSignatureError:
                return jsonify({
                    'success': False,
                    'status': 401,
                    'errors': ['Token has expired']
                }), 401
            except jwt.InvalidTokenError:
                return jsonify({
                    'success': False,
                    'status': 401,
                    'errors': ['Token is invalid']
                }), 401
            
            return f(*args, **kwargs)
        return decorated_function
    
    def generate_file_id_from_email_filename(email: str, filename: str) -> str:
        """Generate consistent file_id based on email and filename"""
        combined = f"{email}_{filename}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def generate_session_id() -> str:
        """Generate session ID"""
        return uuid.uuid4().hex
    
    # V2 Compatible Endpoints
    
    @app.route('/process_file', methods=['POST'])
    @verify_jwt_token
    def process_file():
        """V2 Compatible: Upload and process document"""
        try:
            # Get form data
            email = request.form.get('email')
            qr_name = request.form.get('qr_name', 'Document')
            
            if not email:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['Email is required']
                }), 400
            
            # Collect all uploaded files from any file parameter
            files = []
            for field_name in request.files:
                if field_name.startswith('file'):
                    files.extend(request.files.getlist(field_name))
            
            # Remove empty files
            files = [f for f in files if f.filename and f.filename.strip() != '']
            
            if not files:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['No valid files provided']
                }), 400

            processed_files = []
            skipped_files = []
            processing_errors = []
            
            # Define a list of allowed extensions (documents only)
            ALLOWED_EXTENSIONS = {
                'pdf', 'html', 'epub', 'rtf', 'odt', 'docx', 'txt',
                'xlsx', 'xlsm', 'csv', 'ppt', 'pptx', 'md',
                'jpg', 'jpeg', 'png', 'bmp', 'tiff'  # Add image formats
            }
            
            # Initialize a NEW HybridRAGService instance for this session
            try:
                from config import Config as LocalConfig
                rag_service = EnhancedRAGService(LocalConfig)
                logger.info("EnhancedRAGService initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import Config: {e}")
                return jsonify({
                    'success': False,
                    'status': 500,
                    'errors': [f'Configuration import error: {str(e)}']
                }), 500
            except Exception as e:
                logger.error(f"Failed to initialize EnhancedRAGService: {e}")
                return jsonify({
                    'success': False,
                    'status': 500,
                    'errors': [f'Service initialization error: {str(e)}']
                }), 500
                
            session_id = generate_session_id()
            
            for file in files:
                file_extension = os.path.splitext(file.filename)[1].lower().strip('.')
                if file_extension not in ALLOWED_EXTENSIONS:
                    skipped_files.append(f'{file.filename} (unsupported file type)')
                    continue

                # Additional validation for image files
                is_image = file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
                if is_image:
                    is_valid, error_msg = validate_image_file(file)
                    if not is_valid:
                        skipped_files.append(f'{file.filename} ({error_msg})')
                        continue

                original_filename = secure_filename(file.filename)
                file_id = generate_file_id_from_email_filename(email, original_filename)
                
                # Save file temporarily
                filename = f"{file_id}_{original_filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                try:
                    file_start_time = time.time()
                    file_size = len(file.read())
                    file.seek(0)  # Reset file pointer after reading size
                
                    # Save file temporarily  
                    save_start = time.time()
                    file.save(filepath)
                    save_time = time.time() - save_start
                    
                    # Determine file type for processing
                    file_type = 'image' if is_image else 'document'
                    
                    # Process document with DocumentService (now handles images with OCR)
                    doc_process_start = time.time()
                    result = document_service.process_document(filepath, file_id, original_filename)
                    doc_process_time = time.time() - doc_process_start
                    
                    if result['success']:
                        # Add file type to metadata
                        result['metadata']['file_type'] = file_type
                        if is_image:
                            result['metadata']['requires_ocr'] = True
                        
                        # Store in vector database
                        vector_start = time.time()
                        chunks_result = rag_service.store_document_chunks(
                            result['chunks'], 
                            file_id,
                            result['metadata']
                        )
                        vector_time = time.time() - vector_start
                                                
                        if chunks_result['success']:
                            file_total_time = time.time() - file_start_time
                            processing_time = result.get('processing_time', file_total_time - save_time - vector_time)
                            
                            log_file_processing_summary(
                                original_filename, file_size, file_extension,
                                save_time, processing_time, vector_time, file_total_time
                            )
                                   
                            processed_files.append({
                                'file_id': file_id,
                                'filename': original_filename,
                                'qr_name': qr_name,
                                'file_type': file_type,  # 'image' or 'document'
                                'processing_stats': result.get('processing_stats', {}),
                                'metadata': result.get('metadata', {}),
                                'timing_breakdown': {
                                    'file_save_time': save_time,
                                    'processing_time': processing_time,
                                    'vector_storage_time': vector_time,
                                    'total_time': file_total_time,
                                    'file_size': file_size,
                                    'processing_rate_mb_per_sec': (file_size / 1024 / 1024) / file_total_time if file_total_time > 0 else 0
                                }
                            })
                        else:
                            # Handle storage errors (existing error handling remains the same)
                            error_info = chunks_result.get('error', 'Unknown storage error')
                            error_type = chunks_result.get('error_type', 'general_processing_error')
                            retry_recommended = chunks_result.get('retry_recommended', True)
                            
                            if error_type == 'transient_model_error':
                                friendly_error = f'Temporary processing issue with {original_filename}. Please try uploading again.'
                                logger.warning(f"Transient error for {original_filename}: {error_info}")
                            elif error_type == 'empty_content_error':
                                friendly_error = f'No readable content found in {original_filename}. File may be empty or corrupted.'
                                logger.warning(f"Empty content error for {original_filename}: {error_info}")
                            else:
                                friendly_error = f'Failed to store {original_filename}: {error_info}'
                                logger.error(f"Storage error for {original_filename}: {error_info}")
                            
                            processing_errors.append(friendly_error)
                            
                            if retry_recommended:
                                processing_errors.append(f'Temporary issue with {original_filename}. Please retry.')
                    else:
                        processing_errors.append(f'Failed to process {original_filename}: {result["error"]}')
                
                except Exception as e:
                    file_total_time = time.time() - file_start_time if 'file_start_time' in locals() else 0
                    logger.error(f'Error processing {original_filename} after {file_total_time:.2f}s: {str(e)}')
                    processing_errors.append(f'Error with {original_filename}: {str(e)}')
                
                finally:
                    # Clean up uploaded file
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except:
                            pass
            
            # Prepare response messages
            messages = []
            if processed_files:
                messages.append(f'Successfully processed {len(processed_files)} file(s).')
            if skipped_files:
                messages.append(f'Skipped {len(skipped_files)} non-supported file(s): {", ".join(skipped_files)}')
            if processing_errors:
                messages.extend(processing_errors)
            
            if processed_files:
                # Store the HybridRAGService instance and file info in the sessions dictionary
                sessions[session_id] = {
                    'rag_service': rag_service,
                    'files': processed_files,
                    'created_at': datetime.now().isoformat(),
                    'email': email
                }
                
                chat_url = f"https://bot.xplorease.com/?session={session_id}"
                qr_url = f"https://mspo-development.s3.ap-south-1.amazonaws.com/xplorease/{email}/{session_id}_qr.png"
                logo_url = None
                
                if 'logo' in request.files and request.files['logo'].filename:
                    logo_url = f"https://mspo-development.s3.ap-south-1.amazonaws.com/xplorease/{email}/{email}_logo.png"
                
                # Create data array with all processed files
                data = []
                for processed_file in processed_files:
                    file_data = {
                        'session_id': session_id,
                        'user_id': email,
                        'qr_name': qr_name,
                        'file_id': processed_file['file_id'],
                        'filename': processed_file['filename'],
                        'file_type': processed_file.get('file_type', 'document'),
                        'chat_url': chat_url,
                        'qr_url': qr_url,
                        'logo_url': logo_url,
                        'processing_stats': processed_file.get('processing_stats', {}),
                        'document_metadata': processed_file.get('metadata', {})
                    }
                    
                    data.append(file_data)
                
                return jsonify({
                    'success': True,
                    'status': 200,
                    'message': messages,
                    'data': data
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': messages if messages else ['No files were successfully processed.']
                }), 400

        except Exception as e:
            logger.error(f"Process file error: {str(e)}")
            return jsonify({
                'success': False,
                'status': 500,
                'errors': [f'Processing failed: {str(e)}']
            }), 500
    
    @app.route('/answer_question', methods=['POST'])
    @verify_jwt_token
    def answer_question():
        """V2 Compatible: Answer question about document with enhanced RAG"""
        import traceback
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['No JSON data provided']
                }), 400

            session_id = data.get('session_id')
            question = data.get('question')
            file_id = data.get('file_id')
            conversation_history = data.get('conversation_history', [])
            include_evaluation = data.get('include_evaluation', False)  # Optional evaluation

            if not session_id:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['session_id is required']
                }), 400

            if not question:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['question is required']
                }), 400

            if not file_id:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['file_id is required to specify which document to query.']
                }), 400

            validation = validate_session(session_id, sessions, require_rag_service=True)
            if not validation['valid']:
                return create_response(False, errors=[validation['error']], status=validation['status'])

            rag_service = validation['session_data']['rag_service']
            
            # Validate that the RAG service is properly initialized
            if not hasattr(rag_service, 'answer_question'):
                logger.error(f"RAG service in session {session_id} is corrupted")
                return jsonify({
                    'success': False,
                    'status': 500,
                    'errors': ['RAG service corrupted']
                }), 500

            try:
                # Process question with the enhanced RAG service
                log_session_action(session_id, "Processing question", f"file: {file_id[:8]}...")

                
                result = rag_service.answer_question(
                    file_id=file_id,
                    question=question,
                    conversation_history=conversation_history,
                    session_id=session_id  # Add session_id for conversation memory
                )
                
                logger.info(f"RAG service returned: success={result.get('success', False)}")
                
            except Exception as inner_e:
                tb = traceback.format_exc()
                logger.error(f"Critical error in rag_service.answer_question: {str(inner_e)}")
                logger.error(f"Inner traceback:\n{tb}")
                
                # Return a graceful error response
                return jsonify({
                    'success': False,
                    'status': 500,
                    'errors': [f'RAG processing failed: {str(inner_e)}'],
                    'debug_info': {
                        'inner_error': str(inner_e),
                        'error_location': 'rag_service.answer_question'
                    }
                }), 500

            if result['success']:
                # Clean, essential response data only
                response_data = {
                    'answer': result['answer'],
                    'confidence_score': result.get('confidence_score', 0.5),
                    'processing_time': result.get('processing_time', 0),
                    'question_type': result.get('question_type', 'general'),
                    'source_type': 'document'  # Document processing only
                }
                
                # Add translation-specific information if available
                translation_info = result.get('translation_info')
                if translation_info:
                    response_data['translation_info'] = {
                        'original_text': translation_info.get('original_text'),
                        'source_language': translation_info.get('source_language'),
                        'target_language': translation_info.get('target_language'),
                        'processing_type': translation_info.get('processing_type'),
                        'context': translation_info.get('context'),
                        'method': translation_info.get('method', 'AI')
                    }
                    if 'note' in translation_info:
                        response_data['translation_info']['note'] = translation_info['note']
                
                # Add simplified sources without citation dependencies
                sources = result.get('sources', [])
                if sources:
                    simplified_sources = []
                    for source in sources:
                        simplified_source = {
                            'content_preview': source.get('content_preview', ''),
                            'page': source.get('page', 1),
                            'relevance_score': source.get('relevance_scores', {}).get('hybrid_score', 0),
                            'key_terms': source.get('key_terms', [])[:3],  # Limit to top 3 key terms
                            'source_type': source.get('source_type', 'document')
                        }
                        simplified_sources.append(simplified_source)
                    response_data['sources'] = simplified_sources

                # Optional: Include cleaned evaluation metrics (disabled for now)
                # if include_evaluation:
                #     try:
                #         evaluation = evaluation_service.evaluate_enhanced_rag_response(
                #             query=question,
                #             response=result
                #         )
                #         
                #         # Extract only essential metrics, excluding citation-dependent ones
                #         cleaned_evaluation = {
                #             'overall_score': evaluation.get('overall_metrics', {}).get('overall_rag_score', 0),
                #             'retrieval_score': evaluation.get('overall_metrics', {}).get('retrieval_score', 0),
                #             'generation_score': evaluation.get('overall_metrics', {}).get('generation_score', 0),
                #             'quality_grade': evaluation.get('overall_metrics', {}).get('quality_grade', 'Unknown'),
                #             'key_metrics': {
                #                 'answer_relevancy': evaluation.get('generation_metrics', {}).get('answer_relevancy', 0),
                #                 'clarity': evaluation.get('generation_metrics', {}).get('clarity', 0),
                #                 'coherence': evaluation.get('generation_metrics', {}).get('coherence', 0),
                #                 'documents_retrieved': evaluation.get('retrieval_metrics', {}).get('documents_retrieved', 0),
                #                 'retrieval_depth': evaluation.get('retrieval_metrics', {}).get('retrieval_depth', 0)
                #             }
                #         }
                #         response_data['evaluation'] = cleaned_evaluation
                #         
                #     except Exception as eval_error:
                #         logger.warning(f"Evaluation failed: {str(eval_error)}")
                #         response_data['evaluation_error'] = str(eval_error)

                # V2 compatible response format (but with enhanced data)
                return jsonify({
                    'success': True,
                    'status': 200,
                    'data': [response_data]  # V2 expects data as array
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'status': 500,
                    'errors': [result.get('error', 'Unknown error occurred')]
                }), 500

        except Exception as e:
            tb_outer = traceback.format_exc()
            
            # Enhanced error logging for debugging
            logger.error(f"Question processing failed for session {session_id[:8]}...: {str(e)}")
            
            return jsonify({
                'success': False,
                'status': 500,
                'errors': [f'Failed to process question: {str(e)}'],
                'debug_info': {
                    'error_type': type(e).__name__,
                    'session_exists': session_id in sessions if session_id else False,
                    'question_length': len(question) if question else 0
                }
            }), 500
    
    @app.route('/sample_questions', methods=['POST'])
    @verify_jwt_token
    def sample_questions():
        """V2 Compatible: Get sample questions"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['No JSON data provided']
                }), 400
            
            session_id = data.get('session_id')
            file_id = data.get('file_id')
            
            if not session_id:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['session_id is required']
                }), 400
            
            if not file_id:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['file_id is required to specify which document to query.']
                }), 400

            # Check if session exists and retrieve the specific HybridRAGService instance
            validation = validate_session(session_id, sessions, require_rag_service=True)
            if not validation['valid']:
                return create_response(False, errors=[validation['error']], status=validation['status'])

            rag_service = validation['session_data']['rag_service']
            
            # Generate sample questions using enhanced RAG capabilities
            try:
                # Get document chunks for sample generation
                try:
                    # Get document chunks for sample generation using scroll method
                    from qdrant_client.http import models
                    
                    scroll_result = rag_service.qdrant_client.scroll(
                        collection_name=rag_service.config_manager.get('COLLECTION_NAME'),
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id)),
                                models.FieldCondition(key="is_metadata", match=models.MatchValue(value=False))
                            ]
                        ),
                        limit=5,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    # Convert to Document objects
                    chunks = []
                    for point in scroll_result[0]:
                        chunks.append(type('Document', (), {
                            'page_content': point.payload.get('content', ''),
                            'metadata': point.payload
                        })())
                        
                except Exception as e:
                    logger.error(f"Error retrieving chunks: {str(e)}")
                    chunks = []
                
                if not chunks:
                    return jsonify({
                        'success': False,
                        'status': 404,
                        'errors': ['No content found for document']
                    }), 404
                
                # Use RAG service to generate intelligent sample questions
                sample_questions = rag_service.generate_sample_questions(chunks)
                
                # Format as numbered list (V2 compatibility)
                formatted_questions = [f"{i+1}. {q}" for i, q in enumerate(sample_questions[:5])]
                
                return create_response(True, data=[{'answer': formatted_questions}])
                
            except Exception as e:
                logger.error(f"Sample question generation error: {str(e)}")
                # Fallback questions
                fallback_questions = [
                    "1. What is this document about?",
                    "2. Can you provide a summary?",
                    "3. What are the main topics covered?",
                    "4. Are there any key findings mentioned?",
                    "5. What conclusions can be drawn?"
                ]
                
                return jsonify({
                    'success': True,
                    'status': 200,
                    'data': [{
                        'answer': fallback_questions
                    }]
                }), 200
            
        except Exception as e:
            logger.error(f"Sample questions error: {str(e)}")
            return jsonify({
                'success': False,
                'status': 500,
                'errors': [f'Failed to generate sample questions: {str(e)}']
            }), 500
    
    @app.route('/generate_sample_questions', methods=['POST'])
    @verify_jwt_token
    def generate_sample_questions():
        """V2 Compatible: Alternative endpoint for sample questions"""
        return sample_questions()
    
    @app.route('/replace_file', methods=['POST'])
    @verify_jwt_token
    def replace_file():
        """V2 Compatible: Replace file (implemented as delete + upload)"""
        try:
            # Get form data
            email = request.form.get('email')
            session_id = request.form.get('session_id')
            file_id_to_delete = request.form.get('file_id')
            
            if not email or not session_id or not file_id_to_delete:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['Email, session_id, and file_id are required']
                }), 400
            
            # Check if session exists
            validation = validate_session(session_id, sessions, require_rag_service=True)
            if not validation['valid']:
                return create_response(False, errors=[validation['error']], status=validation['status'])
            
            # Check if new file is provided
            if 'new_files' not in request.files:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['No new file provided']
                }), 400
            
            file = request.files['new_files']
            
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['No file selected']
                }), 400
            
            # Retrieve the HybridRAGService instance from the session
            session_data = sessions[session_id]
            rag_service = session_data['rag_service']
            
            # Delete old document
            rag_service.delete_document(file_id_to_delete)
            
            # Process new file
            original_filename = secure_filename(file.filename)
            new_file_id = generate_file_id_from_email_filename(email, original_filename)
            
            # Save file temporarily
            filename = f"{new_file_id}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Process new document
                result = document_service.process_document(filepath, new_file_id, original_filename)
                
                if result['success']:
                    chunks_result = rag_service.store_document_chunks(
                        result['chunks'], 
                        new_file_id,
                        result['metadata']
                    )
                    
                    if chunks_result['success']:
                        # Update the session with the new file info
                        sessions[session_id]['files'] = [
                            f for f in sessions[session_id]['files'] if f['file_id'] != file_id_to_delete
                        ]
                        sessions[session_id]['files'].append({
                            'file_id': new_file_id,
                            'filename': original_filename,
                            'qr_name': 'Document',
                            'processing_stats': result.get('processing_stats', {}),
                            'metadata': result.get('metadata', {})
                        })
                        
                        return jsonify({
                            'success': True,
                            'status': 200,
                            'message': ['File(s) Replaced successfully'],
                            'data': {
                                'new_file_id': new_file_id,
                                'processing_stats': result.get('processing_stats', {}),
                                'document_metadata': result.get('metadata', {})
                            }
                        }), 200
                    else:
                        return jsonify({
                            'success': False,
                            'status': 500,
                            'errors': [f'Failed to store new document: {chunks_result["error"]}']
                        }), 500
                else:
                    return jsonify({
                        'success': False,
                        'status': 500,
                        'errors': [f'Failed to process new document: {result["error"]}']
                    }), 500
                    
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                
        except Exception as e:
            logger.error(f"Replace file error: {str(e)}")
            return jsonify({
                'success': False,
                'status': 500,
                'errors': [f'Replace failed: {str(e)}']
            }), 500
    
    @app.route('/delete_selected_files', methods=['POST'])
    @verify_jwt_token
    def delete_selected_files():
        """V2 Compatible: Delete selected files"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['No JSON data provided']
                }), 400
            
            session_id = data.get('session_id')
            file_ids_to_delete = data.get('file_ids', [])
            
            if not session_id:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['session_id is required']
                }), 400
            
            # Check if session exists
            validation = validate_session(session_id, sessions, require_rag_service=True)
            if not validation['valid']:
                return create_response(False, errors=[validation['error']], status=validation['status'])

            rag_service = sessions[session_id]['rag_service']
            deleted_count = 0
            
            for file_id in file_ids_to_delete:
                result = rag_service.delete_document(file_id)
                if result['success']:
                    deleted_count += 1
                    # Remove the file from the session's file list
                    sessions[session_id]['files'] = [
                        f for f in sessions[session_id]['files'] if f['file_id'] != file_id
                    ]

            # If all files for the session are deleted, remove the session from memory
            if not sessions[session_id]['files']:
                del sessions[session_id]
                
            return jsonify({
                'success': True,
                'status': 200,
                'message': [f'{deleted_count} file(s) deleted successfully']
            }), 200

        except Exception as e:
            logger.error(f"Delete files error: {str(e)}")
            return jsonify({
                'success': False,
                'status': 500,
                'errors': [f'Delete failed: {str(e)}']
            }), 500
    
    # New Enhanced Endpoints
    
    @app.route('/get_session_info', methods=['GET'])
    @verify_jwt_token
    def get_session_info():
        """Get information about a specific session"""
        try:
            session_id = request.args.get('session_id')
            
            if not session_id:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['session_id parameter is required']
                }), 400
            
            if session_id not in sessions:
                return jsonify({
                    'success': False,
                    'status': 404,
                    'errors': ['Session not found']
                }), 404
            
            session_data = sessions[session_id]
            
            return jsonify({
                'success': True,
                'status': 200,
                'data': {
                    'session_id': session_id,
                    'created_at': session_data.get('created_at'),
                    'email': session_data.get('email'),
                    'files': session_data.get('files', []),
                    'total_files': len(session_data.get('files', []))
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Get session info error: {str(e)}")
            return jsonify({
                'success': False,
                'status': 500,
                'errors': [f'Failed to get session info: {str(e)}']
            }), 500
    
    # @app.route('/evaluate_response', methods=['POST'])
    # @verify_jwt_token
    # def evaluate_response():
    #     """
    #     Comprehensive RAG response evaluation with hit rate, precision, recall, 
    #     and advanced retrieval/generation metrics
    #     """
    #     try:
    #         data = request.get_json()
    #         
    #         if not data:
    #             return create_response(False, errors=['No JSON data provided'], status=400)
    #         
    #         query = data.get('query')
    #         response = data.get('response')
    #         ground_truth = data.get('ground_truth')
    #         relevant_docs = data.get('relevant_docs')
    #         
    #         if not query or not response:
    #             return create_response(False, errors=['Query and response are required'], status=400)
    #         
    #         # Perform comprehensive evaluation
    #         evaluation = evaluation_service.evaluate_enhanced_rag_response(
    #             query=query,
    #             response=response,
    #             ground_truth=ground_truth,
    #             relevant_docs=relevant_docs
    #         )
    #         
    #         # Enhance response with metric explanations
    #         enhanced_evaluation = {
    #             **evaluation,
    #             'metrics_explanation': {
    #                 'retrieval_metrics': {
    #                     'hit_rate': 'Percentage of queries where at least one relevant document was retrieved',
    #                     'precision_at_k': 'Fraction of retrieved documents that are relevant',
    #                     'recall_at_k': 'Fraction of relevant documents that were retrieved',
    #                     'f1_at_k': 'Harmonic mean of precision and recall',
    #                     'mrr': 'Mean Reciprocal Rank - position of first relevant document',
    #                     'ndcg_at_k': 'Normalized Discounted Cumulative Gain - ranking quality metric'
    #                 },
    #                 'generation_metrics': {
    #                     'semantic_similarity': 'Semantic similarity to ground truth answer',
    #                     'bleu_score': 'BLEU score comparing n-gram overlap with ground truth',
    #                     'rouge_scores': 'ROUGE-1, ROUGE-2, ROUGE-L for different text overlap measures',
    #                     'content_precision': 'Precision of content words compared to ground truth',
    #                     'content_recall': 'Recall of content words compared to ground truth',
    #                     'factual_accuracy': 'Accuracy of factual claims compared to ground truth'
    #                 }
    #             },
    #             'evaluation_summary': {
    #                 'overall_score': evaluation.get('overall_metrics', {}).get('overall_rag_score', 0),
    #                 'retrieval_score': evaluation.get('overall_metrics', {}).get('retrieval_score', 0),
    #                 'generation_score': evaluation.get('overall_metrics', {}).get('generation_score', 0),
    #                 'quality_grade': evaluation.get('overall_metrics', {}).get('quality_grade', 'Unknown')
    #             }
    #         }
    #         
    #         return create_response(True, data=enhanced_evaluation, message="Evaluation completed successfully")

    #         
    #     except Exception as e:
    #         logger.error(f"Evaluation error: {str(e)}")
    #         return create_response(False, errors=['No JSON data provided'], status=400)
    
    @app.route('/health_check', methods=['GET'])
    def health_check():
        """Enhanced health check"""
        try:
            # Basic health info
            active_sessions = len(sessions)
            total_files = sum(len(s.get('files', [])) for s in sessions.values())

            return create_response(True, data={
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0-enhanced',
                'active_sessions': active_sessions,
                'total_files_processed': total_files,
                'features_enabled': ['enhanced_rag', 'ocr_processing', 'multi_format_support']
            }, message="Service is running")
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return jsonify({
                'success': False,
                'status': 500,
                'message': 'Health check failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    # Error handlers
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
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)