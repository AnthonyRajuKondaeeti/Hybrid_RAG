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
from services.rag_service import HybridRAGService  # Using the enhanced version
from services.mistral_rag_evaluation_service import EnhancedRAGEvaluationService
from utils.response_formatter import ResponseFormatter

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
    
    # Initialize services
    document_service = DocumentService()
    evaluation_service = EnhancedRAGEvaluationService(
        mistral_api_key=Config.MISTRAL_API_KEY,
        mistral_model=Config.MISTRAL_MODEL
    )
    
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
            
            # Define a list of allowed extensions
            ALLOWED_EXTENSIONS = {
                'pdf', 'html', 'epub', 'rtf', 'odt', 'docx', 'txt',
                'xlsx', 'xlsm', 'csv', 'ppt', 'pptx', 'md'
            }
            
            # Initialize a NEW HybridRAGService instance for this session
            rag_service = HybridRAGService(Config)
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
                        # Store in vector database using the enhanced instance
                        chunks_result = rag_service.store_document_chunks(
                            result['chunks'], 
                            file_id,
                            result['metadata']
                        )
                        
                        if chunks_result['success']:
                            processed_files.append({
                                'file_id': file_id,
                                'filename': original_filename,
                                'qr_name': qr_name,
                                'processing_stats': result.get('processing_stats', {}),
                                'metadata': result.get('metadata', {})
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
                    data.append({
                        'session_id': session_id,
                        'user_id': email,
                        'qr_name': qr_name,
                        'file_id': processed_file['file_id'],
                        'filename': processed_file['filename'],
                        'chat_url': chat_url,
                        'qr_url': qr_url,
                        'logo_url': logo_url,
                        'processing_stats': processed_file.get('processing_stats', {}),
                        'document_metadata': processed_file.get('metadata', {})
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

            # Check if session exists and retrieve the specific HybridRAGService instance
            if session_id not in sessions:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['Invalid session']
                }), 400

            session_data = sessions[session_id]
            rag_service = session_data['rag_service']

            try:
                # Process question with the enhanced RAG service
                result = rag_service.answer_question(
                    file_id=file_id,
                    question=question,
                    conversation_history=conversation_history
                )
            except Exception as inner_e:
                tb = traceback.format_exc()
                logger.error(f"Error in rag_service.answer_question: {str(inner_e)}\n{tb}")
                return jsonify({
                    'success': False,
                    'status': 500,
                    'errors': [f'Error in answer_question: {str(inner_e)}', tb]
                }), 500

            if result['success']:
                # Clean, essential response data only
                response_data = {
                    'answer': result['answer'],
                    'confidence_score': result.get('confidence_score', 0.5),
                    'processing_time': result.get('processing_time', 0),
                    'question_type': result.get('question_type', 'general')
                }
                
                # Add simplified sources without citation dependencies
                sources = result.get('sources', [])
                if sources:
                    simplified_sources = []
                    for source in sources:
                        simplified_source = {
                            'content_preview': source.get('content_preview', ''),
                            'page': source.get('page', 1),
                            'relevance_score': source.get('relevance_scores', {}).get('hybrid_score', 0),
                            'key_terms': source.get('key_terms', [])[:3]  # Limit to top 3 key terms
                        }
                        simplified_sources.append(simplified_source)
                    response_data['sources'] = simplified_sources

                # Optional: Include cleaned evaluation metrics (without citation dependencies)
                if include_evaluation:
                    try:
                        evaluation = evaluation_service.evaluate_enhanced_rag_response(
                            query=question,
                            response=result
                        )
                        
                        # Extract only essential metrics, excluding citation-dependent ones
                        cleaned_evaluation = {
                            'overall_score': evaluation.get('overall_metrics', {}).get('overall_rag_score', 0),
                            'retrieval_score': evaluation.get('overall_metrics', {}).get('retrieval_score', 0),
                            'generation_score': evaluation.get('overall_metrics', {}).get('generation_score', 0),
                            'quality_grade': evaluation.get('overall_metrics', {}).get('quality_grade', 'Unknown'),
                            'key_metrics': {
                                'answer_relevancy': evaluation.get('generation_metrics', {}).get('answer_relevancy', 0),
                                'clarity': evaluation.get('generation_metrics', {}).get('clarity', 0),
                                'coherence': evaluation.get('generation_metrics', {}).get('coherence', 0),
                                'documents_retrieved': evaluation.get('retrieval_metrics', {}).get('documents_retrieved', 0),
                                'retrieval_depth': evaluation.get('retrieval_metrics', {}).get('retrieval_depth', 0)
                            }
                        }
                        response_data['evaluation'] = cleaned_evaluation
                        
                    except Exception as eval_error:
                        logger.warning(f"Evaluation failed: {str(eval_error)}")
                        response_data['evaluation_error'] = str(eval_error)

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
            logger.error(f"Answer question error: {str(e)}\n{tb_outer}")
            return jsonify({
                'success': False,
                'status': 500,
                'errors': [f'Failed to process question: {str(e)}', tb_outer]
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
            if session_id not in sessions:
                return jsonify({
                    'success': False,
                    'status': 400,
                    'errors': ['Invalid session']
                }), 400
            
            session_data = sessions[session_id]
            rag_service = session_data['rag_service']
            
            # Generate sample questions using enhanced RAG capabilities
            try:
                # Get document chunks for sample generation
                chunks = rag_service.get_document_chunks(file_id, limit=5)
                
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
                
                return jsonify({
                    'success': True,
                    'status': 200,
                    'data': [{
                        'answer': formatted_questions
                    }]
                }), 200
                
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
    
    @app.route('/evaluate_response', methods=['POST'])
    @verify_jwt_token
    def evaluate_response():
        """
        Comprehensive RAG response evaluation with hit rate, precision, recall, 
        and advanced retrieval/generation metrics
        """
        try:
            data = request.get_json()
            
            if not data:
                return ResponseFormatter.error_response(
                    "No JSON data provided",
                    status=400
                )
            
            query = data.get('query')
            response = data.get('response')
            ground_truth = data.get('ground_truth')
            relevant_docs = data.get('relevant_docs')
            
            if not query or not response:
                return ResponseFormatter.validation_error(
                    errors=['Query and response are required'],
                    message="Missing required fields for evaluation"
                )
            
            # Perform comprehensive evaluation
            evaluation = evaluation_service.evaluate_enhanced_rag_response(
                query=query,
                response=response,
                ground_truth=ground_truth,
                relevant_docs=relevant_docs
            )
            
            # Enhance response with metric explanations
            enhanced_evaluation = {
                **evaluation,
                'metrics_explanation': {
                    'retrieval_metrics': {
                        'hit_rate': 'Percentage of queries where at least one relevant document was retrieved',
                        'precision_at_k': 'Fraction of retrieved documents that are relevant',
                        'recall_at_k': 'Fraction of relevant documents that were retrieved',
                        'f1_at_k': 'Harmonic mean of precision and recall',
                        'mrr': 'Mean Reciprocal Rank - position of first relevant document',
                        'ndcg_at_k': 'Normalized Discounted Cumulative Gain - ranking quality metric'
                    },
                    'generation_metrics': {
                        'semantic_similarity': 'Semantic similarity to ground truth answer',
                        'bleu_score': 'BLEU score comparing n-gram overlap with ground truth',
                        'rouge_scores': 'ROUGE-1, ROUGE-2, ROUGE-L for different text overlap measures',
                        'content_precision': 'Precision of content words compared to ground truth',
                        'content_recall': 'Recall of content words compared to ground truth',
                        'factual_accuracy': 'Accuracy of factual claims compared to ground truth'
                    }
                },
                'evaluation_summary': {
                    'overall_score': evaluation.get('overall_metrics', {}).get('overall_rag_score', 0),
                    'retrieval_score': evaluation.get('overall_metrics', {}).get('retrieval_score', 0),
                    'generation_score': evaluation.get('overall_metrics', {}).get('generation_score', 0),
                    'quality_grade': evaluation.get('overall_metrics', {}).get('quality_grade', 'Unknown')
                }
            }
            
            return ResponseFormatter.success_response(
                data=enhanced_evaluation,
                message="RAG response evaluation completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return ResponseFormatter.error_response(
                f"Evaluation failed: {str(e)}",
                status=500
            )
    
    @app.route('/health_check', methods=['GET'])
    def health_check():
        """Enhanced health check"""
        try:
            # Basic health info
            health_info = {
                'success': True,
                'status': 200,
                'message': 'XplorEase V2 Compatible Enhanced RAG API Server is running',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0-enhanced',
                'features': {
                    'document_processing': True,
                    'enhanced_rag': True,
                    'citations': True,
                    'verification': True,
                    'reranking': True,
                    'evaluation': True
                },
                'statistics': {
                    'active_sessions': len(sessions),
                    'total_files_processed': sum(len(session.get('files', [])) for session in sessions.values())
                }
            }
            
            return jsonify(health_info), 200
            
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