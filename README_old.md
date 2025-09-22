# 🚀 Xplorease V2 - AI-Powered Document Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1+-green.svg)](https://flask.palletsprojects.com/)
[![Mistral AI](https://img.shields.io/badge/Mistral-AI-orange.svg)](https://mistral.ai/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red.svg)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Advanced Document Intelligence Platform** powered by Retrieval-Augmented Generation (RAG), enabling intelligent document processing, multi-format support, and conversational AI interactions with your documents.

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🔧 Technology Stack](#-technology-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Configuration](#️-configuration)
- [📡 API Documentation](#-api-documentation)
- [🔒 Authentication](#-authentication)
- [💾 Supported File Formats](#-supported-file-formats)
- [🧪 Testing](#-testing)
- [🐛 Troubleshooting](#-troubleshooting)
- [📊 Performance](#-performance)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## ✨ Features

### 🎯 Core Capabilities

- **📄 Multi-Format Document Processing** - PDF, DOCX, TXT, Excel, PowerPoint, and image files (OCR)
- **🤖 Advanced RAG Engine** - Powered by Mistral AI with semantic search and contextual understanding
- **💬 Conversational AI** - Natural language queries with conversation memory and context retention
- **🔍 Hybrid Search** - Combines semantic similarity and keyword matching for optimal retrieval
- **🖼️ Image Processing** - OCR support for scanned documents and image files
- **📊 Document Analytics** - Processing statistics, confidence scores, and performance metrics

### 🛡️ Enterprise Features

- **🔐 JWT Authentication** - Secure token-based API access
- **⚡ High Performance** - Optimized chunking, caching, and parallel processing
- **🌐 CORS Support** - Cross-origin resource sharing for web applications
- **📱 QR Code Integration** - Generate shareable QR codes for document sessions
- **🔄 Session Management** - Persistent conversation context and multi-document support
- **📈 Monitoring & Logging** - Comprehensive logging with performance tracking

### 🧠 AI & ML Features

- **🎯 Intelligent Chunking** - Semantic-aware document segmentation
- **🔄 Conversation Memory** - Context-aware responses across multiple queries
- **⚖️ Confidence Scoring** - Response reliability indicators
- **🚀 Fallback Handling** - Graceful degradation when AI services are unavailable
- **🔧 Configurable Models** - Support for different embedding and language models

## 🏗️ Architecture

```mermaid
graph TB
    A[Client/Frontend] --> B[Flask API Gateway]
    B --> C[Authentication Layer]
    C --> D[Document Service]
    C --> E[RAG Service]
    C --> F[OCR Processor]

    D --> G[File Processing]
    E --> H[Enhanced RAG Engine]
    F --> I[Image Analysis]

    H --> J[Mistral AI LLM]
    H --> K[Sentence Transformers]
    H --> L[Qdrant Vector DB]

    G --> M[Document Chunks]
    M --> L

    subgraph "AI Services"
        J
        K
        direction TB
    end

    subgraph "Storage Layer"
        L
        N[Session Storage]
        O[File Storage]
        direction TB
    end
```

## 🔧 Technology Stack

### **Backend Core**

- **🐍 Python 3.12+** - Modern Python with latest features
- **🌶️ Flask 3.1+** - Lightweight and flexible web framework
- **🔑 PyJWT** - JSON Web Token implementation
- **🌐 Flask-CORS** - Cross-Origin Resource Sharing support

### **AI & Machine Learning**

- **🧠 Mistral AI** - State-of-the-art large language model
- **🔗 LangChain** - RAG framework and AI orchestration
- **📊 Sentence Transformers** - High-quality embedding models
- **🔍 Rank BM25** - Sparse retrieval for hybrid search
- **🖼️ EasyOCR** - Optical Character Recognition

### **Data & Storage**

- **🗄️ Qdrant** - Vector database for semantic search
- **📄 Document Processing Stack**:
  - PyMuPDF (PDF processing)
  - PDFPlumber (advanced PDF parsing)
  - python-docx (Word documents)
  - openpyxl (Excel files)
  - python-pptx (PowerPoint)
  - BeautifulSoup4 (HTML/web content)

### **Development & Deployment**

- **🔧 python-dotenv** - Environment variable management
- **📝 Comprehensive Logging** - Structured application logging
- **⚡ Performance Optimization** - Caching and async processing

## 📁 Project Structure

```
Xplorease_V2-main/
│
├── 📱 app.py                      # Main Flask application & API routes
├── ⚙️ config.py                   # Configuration management & validation
├── 🚀 run.py                      # Application entry point
├── 🔒 .env                        # Environment variables (create from template)
├── 📦 requirements.txt            # Python dependencies
├── 📝 README.md                   # Project documentation
│
├── 🔧 services/                   # Core business logic
│   ├── __init__.py
│   ├── 📄 document_service.py     # Document processing & text extraction
│   ├── 🤖 rag_service.py          # Enhanced RAG implementation
│   ├── 👁️ ocr_processor.py        # Image & OCR processing
│   └── 💭 conversation_memory.py  # Session & conversation management
│
├── 🛠️ utils/                     # Utility functions
│   ├── __init__.py
│   ├── 🔐 jwt_utils.py            # JWT token management
│   └── 📋 response_formatter.py   # API response standardization
│
├── 🧪 Test Files/                # Testing & validation
│   ├── test_qdrant_manual.py      # Qdrant connection testing
│   ├── test_fast_ocr.py           # OCR functionality testing
│   ├── test_generalized_rag.py    # RAG system testing
│   └── test_performance_improvements.py
│
├── 📁 static/                     # Static files (CSS, JS, images)
├── 📤 uploads/                    # Temporary file upload storage
├── 🐍 env/                        # Python virtual environment
└── 📊 app.log                     # Application logs
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** with pip
- **Qdrant Cloud** account ([Get free tier](https://cloud.qdrant.io/))
- **Mistral AI** API key ([Get API key](https://console.mistral.ai/))

### 1. 📥 Clone & Setup

```bash
# Clone the repository
git clone <repository-url>
cd Xplorease_V2-main

# Create and activate virtual environment
python -m venv env

# Windows
env\Scripts\activate

# Linux/Mac
source env/bin/activate
```

### 2. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. ⚙️ Environment Configuration

Create a `.env` file in the root directory:

```env
# 🧠 Mistral AI Configuration
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_MODEL=mistral-small-latest
MISTRAL_API_URL=https://api.mistral.ai/v1/chat/completions

# 🗄️ Qdrant Vector Database
QDRANT_URL=https://your-cluster-url.qdrant.cloud:6333
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=document_chunks

# 📊 Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
DEVICE=cpu

# 🌐 Application Settings
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=52428800  # 50MB max file size
HOST=0.0.0.0
PORT=5000
FLASK_ENV=development

# 🔧 Processing Configuration
CHUNK_SIZE=300
CHUNK_OVERLAP=50
ENABLE_IMAGE_ANALYSIS=true

# 📱 Integration URLs (Optional)
S3_BUCKET_URL=https://your-s3-bucket.amazonaws.com
```

### 4. 🧪 Test Configuration

```bash
# Test Qdrant connection
python test_qdrant_manual.py

# Test OCR functionality
python test_fast_ocr.py

# Generate JWT token for API testing
python utils/jwt_utils.py
```

### 5. 🚀 Launch Application

```bash
python run.py
```

🎉 **Server running at:** `http://localhost:5000`

## ⚙️ Configuration

### Environment Variables Reference

| Variable                | Description                | Default            | Required |
| ----------------------- | -------------------------- | ------------------ | -------- |
| `MISTRAL_API_KEY`       | Mistral AI API key         | -                  | ✅       |
| `QDRANT_URL`            | Qdrant database URL        | -                  | ✅       |
| `QDRANT_API_KEY`        | Qdrant API key             | -                  | ✅       |
| `COLLECTION_NAME`       | Vector collection name     | `document_chunks`  | ❌       |
| `EMBEDDING_MODEL`       | Sentence transformer model | `all-MiniLM-L6-v2` | ❌       |
| `CHUNK_SIZE`            | Document chunk size        | `300`              | ❌       |
| `CHUNK_OVERLAP`         | Chunk overlap size         | `50`               | ❌       |
| `MAX_CONTENT_LENGTH`    | Max file size (bytes)      | `52428800`         | ❌       |
| `ENABLE_IMAGE_ANALYSIS` | Enable OCR processing      | `true`             | ❌       |

### Model Configuration

```python
# Available embedding models
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",     # Fast, good quality
    "sentence-transformers/all-mpnet-base-v2",    # Best quality
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  # QA optimized
]

# Available language models
MISTRAL_MODELS = [
    "mistral-small-latest",   # Fast, cost-effective
    "mistral-medium-latest",  # Balanced performance
    "mistral-large-latest"    # Best quality
]
```

## 📡 API Documentation

### 🔒 Authentication

All endpoints require JWT authentication. Include token in Authorization header:

```http
Authorization: Bearer <your_jwt_token>
```

**Generate Token:**

```bash
python utils/jwt_utils.py
```

### 🌐 Core Endpoints

#### 1. 📄 Upload & Process Documents

Process multiple documents for AI-powered querying.

```http
POST /process_file
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

**Request:**

```javascript
const formData = new FormData();
formData.append("email", "user@example.com");
formData.append("qr_name", "My Document Set");
formData.append("file1", pdfFile);
formData.append("file2", docxFile);
formData.append("file3", imageFile); // OCR support
```

**Response:**

```json
{
  "success": true,
  "status": 200,
  "message": ["Successfully processed 3 file(s)."],
  "data": [
    {
      "session_id": "a1b2c3d4e5f6...",
      "file_id": "abc123def456...",
      "filename": "document.pdf",
      "file_type": "document",
      "user_id": "user@example.com",
      "qr_name": "My Document Set",
      "chat_url": "https://bot.xplorease.com/?session=a1b2c3d4e5f6",
      "qr_url": "https://example.com/qr/a1b2c3d4e5f6_qr.png",
      "processing_stats": {
        "total_chunks": 45,
        "total_words": 2156,
        "total_pages": 12,
        "processing_time": 3.45,
        "avg_semantic_density": 0.78
      },
      "timing_breakdown": {
        "file_save_time": 0.12,
        "processing_time": 2.89,
        "vector_storage_time": 0.44,
        "total_time": 3.45,
        "processing_rate_mb_per_sec": 1.23
      }
    }
  ]
}
```

#### 2. ❓ Ask Questions (Enhanced RAG)

Query documents using natural language with conversation context.

```http
POST /answer_question
Content-Type: application/json
Authorization: Bearer <token>
```

**Request:**

```json
{
  "session_id": "a1b2c3d4e5f6...",
  "file_id": "abc123def456...",
  "question": "What are the main findings in this research paper?",
  "conversation_history": [
    {
      "question": "What is this document about?",
      "answer": "This is a research paper on AI applications..."
    }
  ],
  "include_evaluation": false
}
```

**Response:**

```json
{
  "success": true,
  "status": 200,
  "data": [
    {
      "answer": "The main findings include three key discoveries: 1) AI models show 85% improvement in accuracy when using RAG, 2) Processing time reduced by 60% with optimized chunking, and 3) User satisfaction increased by 40% with conversational interfaces.",
      "confidence_score": 0.89,
      "processing_time": 2.34,
      "question_type": "analytical",
      "source_type": "document",
      "sources": [
        {
          "content_preview": "Our experiments demonstrate that RAG-enhanced models achieve...",
          "page": 5,
          "relevance_score": 0.92,
          "key_terms": ["RAG", "accuracy", "improvement"],
          "source_type": "document"
        }
      ]
    }
  ]
}
```

#### 3. 💡 Generate Sample Questions

Get AI-generated questions based on document content.

```http
POST /sample_questions
Content-Type: application/json
Authorization: Bearer <token>
```

**Request:**

```json
{
  "session_id": "a1b2c3d4e5f6...",
  "file_id": "abc123def456..."
}
```

**Response:**

```json
{
  "success": true,
  "status": 200,
  "data": [
    {
      "answer": [
        "1. What is the main purpose of this document?",
        "2. What are the key findings or conclusions?",
        "3. What methodology was used in this research?",
        "4. What are the practical applications mentioned?",
        "5. What future research directions are suggested?"
      ]
    }
  ]
}
```

#### 4. 🔄 File Management

**Replace File:**

```http
POST /replace_file
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

**Delete Files:**

```http
POST /delete_selected_files
Content-Type: application/json
Authorization: Bearer <token>
```

**Session Info:**

```http
GET /get_session_info?session_id=a1b2c3d4e5f6...
Authorization: Bearer <token>
```

#### 5. 🏥 Health Check

```http
GET /health_check
```

**Response:**

```json
{
  "success": true,
  "status": 200,
  "data": {
    "timestamp": "2025-09-22T10:30:45.123456",
    "version": "2.0.0-enhanced",
    "active_sessions": 5,
    "total_files_processed": 23,
    "features_enabled": [
      "enhanced_rag",
      "ocr_processing",
      "multi_format_support",
      "conversation_memory"
    ]
  },
  "message": "Service is running"
}
```

## 🔒 Authentication

### JWT Token Generation

```bash
# Generate a new JWT token
python utils/jwt_utils.py

# Output:
# Generated JWT Token: eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
# Email: user@example.com
# Expires: 2025-09-23 10:30:45
```

### Using Tokens in Requests

```javascript
// JavaScript/Fetch
fetch('http://localhost:5000/process_file', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer ' + token
  },
  body: formData
});

// cURL
curl -X POST "http://localhost:5000/answer_question" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "question": "..."}'
```

## 💾 Supported File Formats

### 📄 Document Formats

- **PDF** (.pdf) - Advanced parsing with PDFPlumber & PyMuPDF
- **Microsoft Word** (.docx) - Full text and metadata extraction
- **Plain Text** (.txt) - Direct text processing
- **Markdown** (.md) - Structured document processing
- **Rich Text Format** (.rtf) - Formatted text extraction
- **OpenDocument Text** (.odt) - Open standard documents

### 📊 Spreadsheet Formats

- **Microsoft Excel** (.xlsx, .xlsm) - Multi-sheet processing
- **CSV** (.csv) - Structured data extraction

### 📽️ Presentation Formats

- **Microsoft PowerPoint** (.ppt, .pptx) - Slide content extraction

### 🌐 Web Formats

- **HTML** (.html) - Web content processing
- **EPUB** (.epub) - E-book format support

### 🖼️ Image Formats (OCR)

- **JPEG** (.jpg, .jpeg) - Photo and scan processing
- **PNG** (.png) - High-quality image processing
- **BMP** (.bmp) - Bitmap image support
- **TIFF** (.tiff) - High-resolution document scans

### File Size Limits

- **Maximum file size**: 50MB per file
- **Multiple files**: Up to 10 files per request
- **Total session limit**: 500MB per session

## 🧪 Testing

### Unit Tests

```bash
# Test Qdrant connection
python test_qdrant_manual.py

# Test OCR functionality
python test_fast_ocr.py

# Test RAG performance
python test_generalized_rag.py

# Performance benchmarking
python test_performance_improvements.py
```

### Manual API Testing

```bash
# 1. Generate JWT token
python utils/jwt_utils.py

# 2. Test file upload
curl -X POST "http://localhost:5000/process_file" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "email=test@example.com" \
  -F "file=@sample.pdf"

# 3. Test question answering
curl -X POST "http://localhost:5000/answer_question" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "SESSION_ID",
    "file_id": "FILE_ID",
    "question": "What is this document about?"
  }'
```

### Performance Benchmarks

| Operation            | Small Doc (<1MB) | Medium Doc (1-10MB) | Large Doc (10-50MB) |
| -------------------- | ---------------- | ------------------- | ------------------- |
| **Upload & Process** | <2s              | <10s                | <30s                |
| **Question Answer**  | <3s              | <5s                 | <8s                 |
| **Vector Search**    | <500ms           | <1s                 | <2s                 |
| **OCR Processing**   | <5s              | <15s                | <45s                |

## 🐛 Troubleshooting

### Common Issues

#### 🔴 Qdrant Connection Failed

```bash
# Verify Qdrant configuration
python test_qdrant_manual.py

# Check environment variables
echo $QDRANT_URL
echo $QDRANT_API_KEY
```

#### 🔴 Mistral API Errors

```
Error: 429 - Rate limit exceeded
Solution: Implement request queuing or upgrade Mistral plan

Error: 401 - Unauthorized
Solution: Verify MISTRAL_API_KEY in .env file
```

#### 🔴 Large File Processing

```
Error: File too large (>50MB)
Solution: Split document or compress before upload

Error: OCR timeout
Solution: Increase processing timeout or reduce image resolution
```

#### 🔴 Memory Issues

```bash
# Monitor memory usage
python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available // 1024**3}GB')
"

# Reduce chunk size in config.py
CHUNK_SIZE = 200  # Smaller chunks use less memory
```

### Debug Mode

```bash
# Enable debug logging
export FLASK_ENV=development

# Run with verbose logging
python run.py --debug
```

### Log Analysis

```bash
# View recent errors
tail -f app.log | grep ERROR

# Filter by session
grep "session_abc123" app.log

# Performance analysis
grep "processing_time" app.log | awk '{print $NF}' | sort -n
```

## 📊 Performance

### System Requirements

| Component   | Minimum | Recommended | Enterprise |
| ----------- | ------- | ----------- | ---------- |
| **CPU**     | 2 cores | 4 cores     | 8+ cores   |
| **RAM**     | 4GB     | 8GB         | 16GB+      |
| **Storage** | 10GB    | 50GB        | 500GB+     |
| **Network** | 10Mbps  | 100Mbps     | 1Gbps+     |

### Optimization Tips

#### 🚀 Performance Tuning

```python
# config.py optimizations
CHUNK_SIZE = 250          # Smaller = faster search, larger = better context
CHUNK_OVERLAP = 25        # Reduce for speed, increase for accuracy
DEVICE = "cuda"           # Use GPU if available
ENABLE_CACHING = True     # Cache embeddings and responses
```

#### 🗄️ Database Optimization

```python
# Qdrant optimizations
VECTOR_SIZE = 384         # Use smaller embeddings for speed
DISTANCE_METRIC = "Cosine" # Faster than Euclidean
ON_DISK_STORAGE = True    # Reduce memory usage
```

#### 📊 Monitoring

- **Response Times**: Track via `processing_time` in responses
- **Error Rates**: Monitor failed requests in logs
- **Resource Usage**: CPU, memory, and disk utilization
- **API Limits**: Mistral AI and Qdrant usage tracking

## 🤝 Contributing

### Development Setup

```bash
# Clone for development
git clone <repository-url>
cd Xplorease_V2-main

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # Linux/Mac
# dev-env\Scripts\activate   # Windows

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest  # Code formatting and testing

# Set up pre-commit hooks
pip install pre-commit
pre-commit install
```

### Code Standards

```bash
# Format code
black . --line-length 100

# Check style
flake8 . --max-line-length 100

# Run tests
pytest tests/
```

### Contribution Guidelines

1. **🔀 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **💻 Code** with proper documentation and tests
4. **✅ Test** thoroughly across different file formats
5. **🔧 Format** code using Black and check with Flake8
6. **📝 Commit** changes (`git commit -m 'Add amazing feature'`)
7. **🚀 Push** to branch (`git push origin feature/amazing-feature`)
8. **📬 Create** a Pull Request

### Priority Areas for Contribution

- **🌍 Multi-language Support** - Add support for non-English documents
- **📊 Enhanced Analytics** - Advanced document analysis and insights
- **🔗 Integration APIs** - Connect with popular document management systems
- **⚡ Performance Optimizations** - Faster processing and search algorithms
- **🧪 Testing Coverage** - Comprehensive test suite expansion

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Mistral AI](https://mistral.ai/)** - Advanced language model capabilities
- **[Qdrant](https://qdrant.tech/)** - High-performance vector database
- **[LangChain](https://python.langchain.com/)** - RAG framework and AI orchestration
- **[Sentence Transformers](https://www.sbert.net/)** - State-of-the-art embeddings
- **[Flask](https://flask.palletsprojects.com/)** - Lightweight and flexible web framework

---

## 📞 Support & Contact

- **📧 Email**: support@xplorease.com
- **📖 Documentation**: [Full API Documentation](https://docs.xplorease.com)
- **🐛 Issues**: [GitHub Issues](https://github.com/AnthonyRajuKondaeeti/Xplorease_V2/issues)
- **💬 Discord**: [Join our community](https://discord.gg/xplorease)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by the Xplorease Team

</div>
Xplorease_V2-main/
│
├── app.py                 # Main Flask application
├── config.py              # Configuration management
├── run.py                 # Application entry point
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
│
├── services/              # Core business logic
│   ├── __init__.py
│   ├── document_service.py    # PDF processing & text extraction
│   └── rag_service.py         # RAG implementation & AI logic
│
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── jwt_utils.py           # JWT token management
│   └── response_formatter.py  # API response formatting
│
├── static/                # Static files
├── uploads/              # Uploaded documents storage
└── env/                  # Python virtual environment
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Pip package manager
- Qdrant Cloud account
- Mistral AI API key

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Xplorease_V2-main
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Mistral AI Configuration
MISTRAL_API_KEY=your_mistral_api_key_here

# Qdrant Configuration
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key

# Application Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=52428800  # 50MB
HOST=0.0.0.0
PORT=5000
FLASK_ENV=development
```

### 5. Run the Application

```bash
python run.py
```

The API will be available at `http://localhost:5000`

## 📡 API Endpoints

### Authentication

All endpoints (except health check) require JWT authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

### 🔑 Generate JWT Token

```bash
python utils/jwt_utils.py
```

### Core Endpoints

#### 1. 📄 Process Documents

Upload and process PDF documents for RAG.

```http
POST /process_file
Content-Type: multipart/form-data
Authorization: Bearer <token>

Form Data:
- email: user@example.com
- qr_name: Document Name (optional)
- file: <pdf_file_1>
- file2: <pdf_file_2> (optional)
```

**Response:**

```json
{
  "success": true,
  "status": 200,
  "data": [
    {
      "session_id": "abc123...",
      "file_id": "def456...",
      "filename": "document.pdf",
      "user_id": "user@example.com",
      "qr_name": "Document Name",
      "chat_url": "https://bot.xplorease.com/?session=abc123...",
      "qr_url": "https://example.com/qr/abc123_qr.png",
      "logo_url": null
    }
  ],
  "message": ["Successfully processed 1 file(s)."]
}
```

#### 2. ❓ Ask Questions

Query processed documents using natural language.

```http
POST /answer_question
Content-Type: application/json
Authorization: Bearer <token>

{
    "session_id": "abc123...",
    "question": "What is the main topic of this document?"
}
```

**Response:**

```json
{
  "success": true,
  "status": 200,
  "data": {
    "answer": "The main topic discusses...",
    "confidence_score": 0.95,
    "sources": ["page 1", "page 3"],
    "processing_time": 1.23,
    "question_type": "factual"
  }
}
```

#### 3. 💡 Get Sample Questions

Generate relevant questions based on document content.

```http
POST /sample_questions
Content-Type: application/json
Authorization: Bearer <token>

{
    "session_id": "abc123..."
}
```

#### 4. 🔄 Replace Document

Replace an existing document with a new one.

```http
POST /replace_file
Content-Type: multipart/form-data
Authorization: Bearer <token>

Form Data:
- session_id: abc123...
- file: <new_pdf_file>
- qr_name: New Document Name (optional)
```

#### 5. 🗑️ Delete Documents

Remove selected documents from a session.

```http
POST /delete_selected_files
Content-Type: application/json
Authorization: Bearer <token>

{
    "session_id": "abc123...",
    "file_ids": ["file_id_1", "file_id_2"]
}
```

#### 6. ❤️ Health Check

Check API status (no authentication required).

```http
GET /health_check
```

## 🧪 Testing with Postman

### 1. Import Collection

Create a new Postman collection and add the following environment variables:

- `base_url`: `http://localhost:5000`
- `jwt_token`: `<your_generated_token>`

### 2. Set Authorization

For each request (except health check):

- Go to Authorization tab
- Select "Bearer Token"
- Use `{{jwt_token}}` as the token value

### 3. Test Document Upload

- Method: POST
- URL: `{{base_url}}/process_file`
- Body: form-data
  - email: your-email@example.com
  - file: (select PDF file)
  - qr_name: Test Document

### 4. Test Q&A

- Method: POST
- URL: `{{base_url}}/answer_question`
- Body: raw JSON

```json
{
  "session_id": "your_session_id_from_upload",
  "question": "What is this document about?"
}
```

## 🔧 Configuration

### Environment Variables

| Variable             | Description                  | Required | Default       |
| -------------------- | ---------------------------- | -------- | ------------- |
| `MISTRAL_API_KEY`    | Mistral AI API key           | Yes      | -             |
| `QDRANT_URL`         | Qdrant cloud instance URL    | Yes      | -             |
| `QDRANT_API_KEY`     | Qdrant API key               | Yes      | -             |
| `UPLOAD_FOLDER`      | Directory for uploaded files | No       | `uploads`     |
| `MAX_CONTENT_LENGTH` | Max file upload size (bytes) | No       | `52428800`    |
| `HOST`               | Server host                  | No       | `0.0.0.0`     |
| `PORT`               | Server port                  | No       | `5000`        |
| `FLASK_ENV`          | Flask environment            | No       | `development` |

### Model Configuration

The application uses several pre-configured models:

- **LLM**: Mistral Large Latest
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters

## 🔍 Features Deep Dive

### 1. Document Processing Pipeline

1. **File Validation**: Checks file type (PDF only)
2. **Text Extraction**: Uses PyMuPDF and PDFPlumber for robust text extraction
3. **Chunking**: Intelligent text splitting with overlap for context preservation
4. **Embedding Generation**: Creates vector representations using sentence transformers
5. **Vector Storage**: Stores embeddings in Qdrant for fast similarity search

### 2. RAG Implementation

- **Retrieval**: Semantic search using cosine similarity
- **Context Assembly**: Combines relevant chunks with conversation history
- **Generation**: Uses Mistral AI for response generation
- **Quality Control**: Confidence scoring and source attribution

### 3. Session Management

- **Per-Session RAG**: Each upload session has its own RAG instance
- **Conversation Memory**: Maintains context across multiple questions
- **File Management**: Track and manage multiple documents per session

### 4. Security Features

- **JWT Authentication**: Secure token-based authentication
- **File Validation**: Strict file type and size validation
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Input Sanitization**: Secure filename handling

## 📊 Performance Considerations

- **Vector Search**: O(log n) search time with Qdrant indexing
- **Memory Management**: Efficient chunk processing and storage
- **Caching**: Query result caching for repeated questions
- **Concurrent Processing**: Supports multiple simultaneous sessions

## 🛠️ Development

### Code Structure

- **Services Layer**: Separated business logic for documents and RAG
- **Configuration Management**: Centralized config with validation
- **Error Handling**: Comprehensive error handling and logging
- **Type Hints**: Full type annotation for better code quality

### Adding New Features

1. Create new service in `services/` directory
2. Add configuration in `config.py`
3. Register endpoints in `app.py`
4. Update documentation

### Testing

```bash
# Generate test JWT token
python utils/jwt_utils.py

# Run the application
python run.py

# Test endpoints using Postman or curl
```

## 🚨 Error Handling

The API returns standardized error responses:

```json
{
  "success": false,
  "status": 400,
  "errors": ["Detailed error message"]
}
```

Common error codes:

- `400`: Bad Request (invalid input)
- `401`: Unauthorized (invalid/missing token)
- `404`: Not Found (session/file not found)
- `500`: Internal Server Error

## 📝 Logging

The application includes comprehensive logging:

- **Level**: INFO and above
- **Format**: Timestamp - Logger - Level - Message
- **Output**: Console (can be configured for file output)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

## This project is licensed under the MIT License - see the LICENSE file for details.

**Made with ❤️ by the Xplorease Team**
