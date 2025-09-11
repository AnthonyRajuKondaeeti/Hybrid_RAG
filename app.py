from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
import os
import uuid
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
import logging
import jwt
from functools import wraps
from typing import Dict, Any, List


from qdrant_client.http import models 
from config import Config
from services.document_service import DocumentService
from services.rag_service import RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple JWT secret (you should use a proper secret in production)
JWT_SECRET = "your-jwt-secret-key"

def create_app():
    app = Flask(__name__)
    
    # Validate configuration
    Config.validate()
    
    # Configuration
    app.config.from_object(Config)
    
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
    
    # Initialize services that don't need to be per-session
    document_service = DocumentService()
    
    # In-memory session storage (use Redis in production)
    # This dictionary now stores a dictionary for each session, containing the RAGService instance and file_id.
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
            
            # Define a list of allowed extensions
            ALLOWED_EXTENSIONS = {
                'pdf', 'html', 'epub', 'rtf', 'odt', 'docx', 'txt',
                'xlsx', 'xlsm', 'csv', 'ppt', 'pptx', 'md'
            }
            
            # Initialize a NEW RAGService instance for this session
            rag_service = RAGService()
            session_id = generate_session_id()
            
            for file in files:
                file_extension = os.path.splitext(file.filename)[1].lower().strip('.')
                if file_extension not in ALLOWED_EXTENSIONS:
                    skipped_files.append(f'{file.filename} (unsupported file type)')
                    continue

                original_filename = secure_filename(file.filename)
                file_id = generate_file_id_from_email_filename(email, original_filename)
                
                # Save file temporarily
                filename = f"{file_id}_{original_filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                try:
                    file.save(filepath)
                    
                    # Process document with DocumentService
                    result = document_service.process_document(filepath, file_id, original_filename)
                    
                    if result['success']:
                        # Store in vector database using the new instance
                        chunks_result = rag_service.store_document_chunks(
                            result['chunks'], 
                            file_id,
                            result['metadata']
                        )
                        
                        if chunks_result['success']:
                            processed_files.append({
                                'file_id': file_id,
                                'filename': original_filename,
                                'qr_name': qr_name
                            })
                        else:
                            processing_errors.append(f'Failed to store {original_filename}: {chunks_result["error"]}')
                    else:
                        processing_errors.append(f'Failed to process {original_filename}: {result["error"]}')
                
                except Exception as e:
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
                messages.append(f'Skipped {len(skipped_files)} non-PDF file(s): {", ".join(skipped_files)}')
            if processing_errors:
                messages.extend(processing_errors)
            
            if processed_files:
                # Store the RAGService instance and file info in the sessions dictionary
                sessions[session_id] = {
                    'rag_service': rag_service,
                    'files': processed_files
                }
                
                chat_url = f"https://bot.xplorease.com/?session={session_id}"
                qr_url = f"https://mspo-development.s3.ap-south-1.amazonaws.com/xplorease/{email}/{session_id}_qr.png"
                logo_url = None
                
                if 'logo' in request.files and request.files['logo'].filename:
                    logo_url = f"https://mspo-development.s3.ap-south-1.amazonaws.com/xplorease/{email}/{email}_logo.png"
                
                # Create data array with all processed files
                data = []
                for processed_file in processed_files:
                    data.append({
                        'session_id': session_id,
                        'user_id': email,
                        'qr_name': qr_name,
                        'file_id': processed_file['file_id'],
                        'filename': processed_file['filename'],
                        'chat_url': chat_url,
                        'qr_url': qr_url,
                        'logo_url': logo_url
                    })
                
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
        """V2 Compatible: Answer question about document"""
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
            file_id = data.get('file_id')  # Now accepts a file_id to specify which file to query
            
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

            # Check if session exists and retrieve the specific RAGService instance
            if session_id not in sessions:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['Invalid session']
                }), 400
            
            session_data = sessions[session_id]
            rag_service = session_data['rag_service']
            
            # Process question with the specific RAGService instance
            result = rag_service.answer_question(
                file_id=file_id,
                question=question
            )
            
            if result['success']:
                # V2 compatible response format
                return jsonify({
                    'success': True,
                    'status': 200,
                    'data': [{
                        'answer': result['answer']
                    }]
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'status': 500,
                    'errors': [result['error']]
                }), 500
                
        except Exception as e:
            logger.error(f"Answer question error: {str(e)}")
            return jsonify({
                'success': False,
                'status': 500,
                'errors': [f'Failed to process question: {str(e)}']
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
            file_id = data.get('file_id') # Now requires a file_id
            
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

            # Check if session exists and retrieve the specific RAGService instance
            if session_id not in sessions:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['Invalid session']
                }), 400
            
            session_data = sessions[session_id]
            rag_service = session_data['rag_service']
            
            # Get chunks for sample generation
            chunks = rag_service.get_document_chunks(file_id, limit=10)
            
            if not chunks:
                return jsonify({
                    'success': False,
                    'status': 404,
                    'errors': ['No chunks found for document']
                }), 404
            
            # Generate sample questions with the specific RAGService instance
            sample_questions = rag_service.generate_sample_questions(chunks)
            
            # Format as numbered list (V2 compatibility)
            formatted_questions = [f"{i+1}. {q}" for i, q in enumerate(sample_questions)]
            
            return jsonify({
                'success': True,
                'status': 200,
                'data': [{
                    'answer': formatted_questions
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
            if session_id not in sessions:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['Invalid session']
                }), 400
            
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
            
            # Retrieve the RAGService instance from the session
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
            
            # Process new document
            result = document_service.process_document(filepath, new_file_id, original_filename)
            
            if result['success']:
                chunks_result = rag_service.store_document_chunks(
                    result['chunks'], 
                    new_file_id,
                    result['metadata']
                )
                
                if chunks_result['success']:
                    # Update the session with the new file info and clear memory
                    sessions[session_id]['files'] = [
                        f for f in sessions[session_id]['files'] if f['file_id'] != file_id_to_delete
                    ]
                    sessions[session_id]['files'].append({
                        'file_id': new_file_id,
                        'filename': original_filename,
                        'qr_name': 'Document'
                    })
                    rag_service.memory_manager.conversation_history.clear()
                    rag_service.memory_manager.query_cache.clear()
                    
                    return jsonify({
                        'success': True,
                        'status': 200,
                        'message': ['File(s) Replaced successfully'],
                        'data': {
                            'new_file_id': new_file_id # <-- New key for the file ID
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
                
        except Exception as e:
            logger.error(f"Replace file error: {str(e)}")
            return jsonify({
                'success': False,
                'status': 500,
                'errors': [f'Replace failed: {str(e)}']
            }), 500
        
        finally:
            # Clean up uploaded file
            if 'filepath' in locals() and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
    
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
            file_ids_to_delete = data.get('file_ids', []) # Accepts a list of file_ids
            
            if not session_id:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['session_id is required']
                }), 400
            
            # Check if session exists
            if session_id not in sessions:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['Selected file(s) do not exist']
                }), 400

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
    
    @app.route('/health_check', methods=['GET'])
    def health_check():
        """Health check"""
        return jsonify({
            'success': True,
            'status': 200,
            'message': 'XplorEase V2 Compatible RAG API Server is running',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0-compatible'
        }), 200
    
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