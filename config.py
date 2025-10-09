# File: config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Mistral API Configuration
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
    MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', 'mistral-small-latest')
    
    # LLM Generation Parameters
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2048))  # Prevent answer truncation
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.3))  # Control response creativity
    TOP_P = float(os.getenv('TOP_P', 0.9))  # Nucleus sampling parameter
    
    # Qdrant Configuration
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'document_chunks')
    
    # MongoDB Configuration
    MONGODB_URL = os.getenv('MONGODB_URL')
    DB_NAME = os.getenv('DB_NAME', 'Xplorease_db')
    
    # Flask Configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 50 * 1024 * 1024))  # 50MB
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    
    # JWT Configuration
    JWT_SECRET = os.getenv('JWT_SECRET', 'default-jwt-secret-for-development-only-change-in-production')
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', 24))
    
    # Session Management
    SESSION_CLEANUP_HOURS = int(os.getenv('SESSION_CLEANUP_HOURS', 24))
    MAX_SESSIONS = int(os.getenv('MAX_SESSIONS', 1000))
    
    # Enhanced System Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    RERANKER_MODEL = os.getenv('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    VERIFIER_MODEL = os.getenv('VERIFIER_MODEL', 'cross-encoder/nli-deberta-v3-base')
    
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 800))  # Increased from 300 for better content analysis
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 100))  # Increased proportionally
    RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', 20))
    RERANK_TOP_K = int(os.getenv('RERANK_TOP_K', 8))
    MAX_CONVERSATION_HISTORY = int(os.getenv('MAX_CONVERSATION_HISTORY', 10))
    DEVICE = os.getenv('DEVICE', 'cpu')
    
    # Enhanced Features
    ENABLE_RERANKING = os.getenv('ENABLE_RERANKING', 'true').lower() == 'true'
    ENABLE_VERIFICATION = os.getenv('ENABLE_VERIFICATION', 'true').lower() == 'true'
    ENABLE_CITATIONS = os.getenv('ENABLE_CITATIONS', 'false').lower() == 'true'
    ENABLE_EVALUATION = os.getenv('ENABLE_EVALUATION', 'true').lower() == 'true'
    
    # Performance Optimization Settings
    MAX_CHUNKS_FOR_DENSE_EMBEDDING = int(os.getenv('MAX_CHUNKS_FOR_DENSE_EMBEDDING', 500))
    EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
    FAST_EMBEDDING_MODEL = os.getenv('FAST_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    SLOW_EMBEDDING_MODEL = os.getenv('SLOW_EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
    SKIP_BM25_FOR_LARGE_FILES = os.getenv('SKIP_BM25_FOR_LARGE_FILES', 'true').lower() == 'true'
    QDRANT_BATCH_SIZE = int(os.getenv('QDRANT_BATCH_SIZE', 100))
    
    # Multimodal and Enhanced Features
    ENABLE_OCR = os.getenv('ENABLE_OCR', 'true').lower() == 'true'
    ENABLE_IMAGE_ANALYSIS = os.getenv('ENABLE_IMAGE_ANALYSIS', 'true').lower() == 'true'
    ENABLE_CONVERSATION_MEMORY = os.getenv('ENABLE_CONVERSATION_MEMORY', 'true').lower() == 'true'
    # Enhanced OCR configuration - English only for simplicity
    # Non-English text will be handled with appropriate messaging
    OCR_LANGUAGES = os.getenv('OCR_LANGUAGES', 'en').split(',')  # English only
    OCR_GPU_ENABLED = os.getenv('OCR_GPU_ENABLED', 'false').lower() == 'true'
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv('OCR_CONFIDENCE_THRESHOLD', 0.3))  # Lowered for better quote detection
    OCR_PREPROCESSING = os.getenv('OCR_PREPROCESSING', 'true').lower() == 'true'
    
    # Conversation Memory Settings
    MAX_CONVERSATION_TURNS = int(os.getenv('MAX_CONVERSATION_TURNS', 5))
    CONVERSATION_CACHE_SIZE = int(os.getenv('CONVERSATION_CACHE_SIZE', 100))
    CONVERSATION_CACHE_TTL_HOURS = int(os.getenv('CONVERSATION_CACHE_TTL_HOURS', 1))
    
    # Image Processing Settings
    MAX_IMAGE_SIZE_MB = int(os.getenv('MAX_IMAGE_SIZE_MB', 10))
    SUPPORTED_IMAGE_FORMATS = os.getenv('SUPPORTED_IMAGE_FORMATS', 'jpg,jpeg,png,bmp,tiff').split(',')
    IMAGE_PROCESSING_TIMEOUT = int(os.getenv('IMAGE_PROCESSING_TIMEOUT', 30))
    
    # Translation Settings
    ENABLE_TRANSLATION = os.getenv('ENABLE_TRANSLATION', 'true').lower() == 'true'
    DEFAULT_TARGET_LANGUAGE = os.getenv('DEFAULT_TARGET_LANGUAGE', 'English')
    TRANSLATION_TIMEOUT = int(os.getenv('TRANSLATION_TIMEOUT', 60))
    AUTO_DETECT_LANGUAGE = os.getenv('AUTO_DETECT_LANGUAGE', 'true').lower() == 'true'
    TRANSLATION_RATE_LIMIT = float(os.getenv('TRANSLATION_RATE_LIMIT', 1.0))
    TRANSLATION_MAX_RETRIES = int(os.getenv('TRANSLATION_MAX_RETRIES', 3))
    TRANSLATION_RETRY_DELAY = float(os.getenv('TRANSLATION_RETRY_DELAY', 2.0))
    
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
    ALLOWED_EXTENSIONS = os.getenv('ALLOWED_EXTENSIONS', 'pdf,html,epub,rtf,odt,docx,txt,xlsx,xlsm,csv,ppt,pptx,md,jpg,jpeg,png,bmp,tiff').split(',')
    MAX_PAGES_PER_DOCUMENT = int(os.getenv('MAX_PAGES_PER_DOCUMENT', 1000))
    EXTRACTION_TIMEOUT = int(os.getenv('EXTRACTION_TIMEOUT', 180))
    
    # Retry Configuration
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    RETRY_BASE_DELAY = float(os.getenv('RETRY_BASE_DELAY', 1.0))
    RETRY_EXPONENTIAL_BASE = float(os.getenv('RETRY_EXPONENTIAL_BASE', 2.0))
    RETRY_MAX_DELAY = float(os.getenv('RETRY_MAX_DELAY', 60.0))
    
    # Legacy variable names for backward compatibility with xplorease_main_v2
    SECRET_KEY = JWT_SECRET  # Flask secret key
    DB = DB_NAME  # Database name
    SERVER_PORT = PORT  # Server port
    EMBEDDING_DIMENSIONS = 384  # For all-MiniLM-L6-v2 model
    
    @staticmethod
    def validate():
        required_vars = ['MISTRAL_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY', 'MONGODB_URL']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        # Validate MongoDB URL format
        mongodb_url = os.getenv('MONGODB_URL')
        if mongodb_url and not mongodb_url.startswith(('mongodb://', 'mongodb+srv://')):
            raise ValueError("MONGODB_URL must start with 'mongodb://' or 'mongodb+srv://'")
        
        # Validate chunk size
        chunk_size = int(os.getenv('CHUNK_SIZE', 800))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 100))
        if chunk_overlap >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")

# Create module-level variables for backward compatibility
SECRET_KEY = Config.JWT_SECRET
DB = Config.DB_NAME
SERVER_PORT = Config.PORT
QDRANT_URL = Config.QDRANT_URL
QDRANT_API_KEY = Config.QDRANT_API_KEY
MONGODB_URL = Config.MONGODB_URL
MISTRAL_API_KEY = Config.MISTRAL_API_KEY
MISTRAL_MODEL = Config.MISTRAL_MODEL
COLLECTION_NAME = Config.COLLECTION_NAME
EMBEDDING_DIMENSIONS = 384

# LLM Generation Parameters (new)
MAX_TOKENS = Config.MAX_TOKENS
TEMPERATURE = Config.TEMPERATURE
TOP_P = Config.TOP_P