# File: config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Mistral API Configuration
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
    MISTRAL_MODEL = "mistral-large-latest"
    
    # Qdrant Configuration
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    COLLECTION_NAME = "document_chunks"
    
    # Flask Configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 50 * 1024 * 1024))  # 50MB
    
    # System Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CONVERSATION_HISTORY = 10
    DEVICE = "cpu"
    
    @staticmethod
    def validate():
        required_vars = ['MISTRAL_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
# File: config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Mistral API Configuration
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
    MISTRAL_MODEL = "mistral-large-latest"
    
    # Qdrant Configuration
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    COLLECTION_NAME = "document_chunks"
    
    # Flask Configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 50 * 1024 * 1024))  # 50MB
    
    # System Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CONVERSATION_HISTORY = 10
    DEVICE = "cpu"
    
    @staticmethod
    def validate():
        required_vars = ['MISTRAL_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")