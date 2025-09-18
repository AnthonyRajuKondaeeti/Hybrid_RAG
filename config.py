# File: config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Mistral API Configuration
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
    MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', 'mistral-large-latest')
    
    # Qdrant Configuration
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'document_chunks')
    
    # Flask Configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 50 * 1024 * 1024))  # 50MB
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    
    # JWT Configuration
    JWT_SECRET = os.getenv('JWT_SECRET')
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', 24))
    
    # Session Management
    SESSION_CLEANUP_HOURS = int(os.getenv('SESSION_CLEANUP_HOURS', 24))
    MAX_SESSIONS = int(os.getenv('MAX_SESSIONS', 1000))
    
    # Enhanced System Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    RERANKER_MODEL = os.getenv('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    VERIFIER_MODEL = os.getenv('VERIFIER_MODEL', 'cross-encoder/nli-deberta-v3-base')
    
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 300))  # Optimized for semantic chunking
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))
    RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', 20))
    RERANK_TOP_K = int(os.getenv('RERANK_TOP_K', 8))
    MAX_CONVERSATION_HISTORY = int(os.getenv('MAX_CONVERSATION_HISTORY', 10))
    DEVICE = os.getenv('DEVICE', 'cpu')
    
    # Enhanced Features
    ENABLE_RERANKING = os.getenv('ENABLE_RERANKING', 'true').lower() == 'true'
    ENABLE_VERIFICATION = os.getenv('ENABLE_VERIFICATION', 'true').lower() == 'true'
    ENABLE_CITATIONS = os.getenv('ENABLE_CITATIONS', 'false').lower() == 'true'
    ENABLE_EVALUATION = os.getenv('ENABLE_EVALUATION', 'true').lower() == 'true'
    
    # Performance Configuration
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 100))
    PROCESSING_TIMEOUT = int(os.getenv('PROCESSING_TIMEOUT', 300))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/rag_service.log')
    ENABLE_METRICS_LOGGING = os.getenv('ENABLE_METRICS_LOGGING', 'true').lower() == 'true'
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    CORS_METHODS = os.getenv('CORS_METHODS', 'GET,POST,PUT,DELETE,OPTIONS')
    CORS_HEADERS = os.getenv('CORS_HEADERS', 'Content-Type,Authorization')
    
    # File Processing Configuration
    ALLOWED_EXTENSIONS = os.getenv('ALLOWED_EXTENSIONS', 'pdf,html,epub,rtf,odt,docx,txt,xlsx,xlsm,csv,ppt,pptx,md').split(',')
    MAX_PAGES_PER_DOCUMENT = int(os.getenv('MAX_PAGES_PER_DOCUMENT', 1000))
    EXTRACTION_TIMEOUT = int(os.getenv('EXTRACTION_TIMEOUT', 180))
    
    @staticmethod
    def validate():
        required_vars = ['MISTRAL_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY', 'JWT_SECRET']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        # Validate JWT secret length
        jwt_secret = os.getenv('JWT_SECRET')
        if jwt_secret and len(jwt_secret) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters long for security")
        
        # Validate chunk size
        chunk_size = int(os.getenv('CHUNK_SIZE', 300))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
        if chunk_overlap >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")