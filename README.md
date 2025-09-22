# 🚀 Xplorease V2 - AI Document Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1+-green.svg)](https://flask.palletsprojects.com/)
[![Mistral AI](https://img.shields.io/badge/Mistral-AI-orange.svg)](https://mistral.ai/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red.svg)](https://qdrant.tech/)

> **AI-powered document processing platform** with RAG capabilities, multi-format support, and conversational AI interactions.

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📡 API Documentation](#-api-documentation)
- [💾 Supported Formats](#-supported-formats)
- [🧪 Testing](#-testing)
- [🔧 Configuration](#-configuration)
- [📊 Performance](#-performance)

## ✨ Features

### Core Capabilities
- **📄 Multi-Format Processing** - PDF, DOCX, Excel, PowerPoint, images (OCR)
- **🤖 Advanced RAG** - Mistral AI + semantic search + conversation memory
- **🔍 Hybrid Search** - Semantic similarity + keyword matching
- **🔐 JWT Authentication** - Secure API access
- **📊 Real-time Analytics** - Processing stats and performance metrics

### Technical Features
- **🧠 Intelligent Chunking** - Semantic-aware document segmentation
- **💬 Conversation Memory** - Context retention across queries
- **⚡ High Performance** - Optimized processing and caching
- **🚀 Graceful Fallbacks** - Clean error handling when AI services are down

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [Qdrant Cloud](https://cloud.qdrant.io/) account
- [Mistral AI](https://console.mistral.ai/) API key

### Setup

```bash
# 1. Clone and setup
git clone <repository-url>
cd Xplorease_V2-main
python -m venv env

# Windows: env\Scripts\activate
# Linux/Mac: source env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (.env file)
MISTRAL_API_KEY=your_mistral_api_key
QDRANT_URL=https://your-cluster.qdrant.cloud:6333
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=document_chunks

# 4. Test and run
python test_qdrant_manual.py  # Test connections
python utils/jwt_utils.py     # Generate JWT token
python run.py                 # Start server
```

🎉 **Server running at:** `http://localhost:5000`

## 📡 API Documentation

### Authentication
All endpoints require JWT token:
```http
Authorization: Bearer <token>
```

### Core Endpoints

#### 📄 Upload Documents
```http
POST /process_file
Content-Type: multipart/form-data

Form Data:
- email: user@example.com
- file1: <document1>
- file2: <document2>
```

**Response:**
```json
{
  "success": true,
  "data": [{
    "session_id": "abc123...",
    "file_id": "def456...",
    "filename": "document.pdf",
    "processing_stats": {
      "total_chunks": 45,
      "processing_time": 3.45
    }
  }]
}
```

#### ❓ Ask Questions
```http
POST /answer_question
Content-Type: application/json

{
  "session_id": "abc123...",
  "file_id": "def456...",
  "question": "What are the main findings?"
}
```

**Response:**
```json
{
  "success": true,
  "data": [{
    "answer": "The main findings include...",
    "confidence_score": 0.89,
    "processing_time": 2.34,
    "sources": [...]
  }]
}
```

#### 💡 Sample Questions
```http
POST /sample_questions
{
  "session_id": "abc123...",
  "file_id": "def456..."
}
```

#### 🔄 File Management
```http
POST /replace_file        # Replace document
POST /delete_selected_files   # Delete documents
GET /get_session_info     # Session information
GET /health_check         # API status
```

## 💾 Supported Formats

| Category | Formats | Processing |
|----------|---------|------------|
| **Documents** | PDF, DOCX, TXT, MD, RTF, ODT | Text extraction |
| **Spreadsheets** | XLSX, XLSM, CSV | Data parsing |
| **Presentations** | PPT, PPTX | Content extraction |
| **Images** | JPG, PNG, BMP, TIFF | OCR processing |
| **Web** | HTML, EPUB | Content parsing |

**Limits:** 50MB per file, 10 files per request

## 🧪 Testing

### Quick Tests
```bash
# Connection tests
python test_qdrant_manual.py
python test_fast_ocr.py

# Generate JWT token
python utils/jwt_utils.py

# API testing with cURL
curl -X POST "http://localhost:5000/process_file" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "email=test@example.com" \
  -F "file=@sample.pdf"
```

### Performance Benchmarks
| Operation | Small (<1MB) | Medium (1-10MB) | Large (10-50MB) |
|-----------|--------------|-----------------|-----------------|
| Upload & Process | <2s | <10s | <30s |
| Question Answer | <3s | <5s | <8s |
| OCR Processing | <5s | <15s | <45s |

## 🔧 Configuration

### Environment Variables
```env
# Required
MISTRAL_API_KEY=your_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key

# Optional
CHUNK_SIZE=300
CHUNK_OVERLAP=50
MAX_CONTENT_LENGTH=52428800
ENABLE_IMAGE_ANALYSIS=true
```

### Model Options
```python
# Embedding models
"all-MiniLM-L6-v2"      # Fast, good quality
"all-mpnet-base-v2"     # Best quality
"multi-qa-MiniLM-L6-cos-v1"  # QA optimized

# Mistral models
"mistral-small-latest"   # Fast, cost-effective
"mistral-medium-latest"  # Balanced
"mistral-large-latest"   # Best quality
```

## 📊 Performance

### System Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4GB | 8GB+ |
| Storage | 10GB | 50GB+ |

### Optimization Tips
```python
# Performance tuning
CHUNK_SIZE = 250         # Smaller = faster search
DEVICE = "cuda"          # Use GPU if available
ENABLE_CACHING = True    # Cache responses
```

### Common Issues & Solutions
```bash
# Qdrant connection failed
python test_qdrant_manual.py

# Mistral API errors
# 429: Rate limit → upgrade plan
# 401: Check MISTRAL_API_KEY

# Memory issues
CHUNK_SIZE = 200  # Reduce chunk size
```

## 🏗️ Architecture

```
Client → Flask API → [Document Service, RAG Service, OCR Processor]
                  ↓
        [Mistral AI, Sentence Transformers, Qdrant Vector DB]
```

### Project Structure
```
Xplorease_V2-main/
├── app.py                    # Main Flask application
├── config.py                 # Configuration management
├── run.py                    # Application entry point
├── services/
│   ├── document_service.py   # Document processing
│   ├── rag_service.py        # RAG implementation
│   ├── ocr_processor.py      # Image processing
│   └── conversation_memory.py # Session management
├── utils/
│   ├── jwt_utils.py          # Authentication
│   └── response_formatter.py # API responses
└── tests/                    # Testing scripts
```

### Technology Stack
- **Backend:** Python 3.12+, Flask 3.1+
- **AI/ML:** Mistral AI, LangChain, Sentence Transformers
- **Database:** Qdrant Vector DB
- **Processing:** PyMuPDF, PDFPlumber, EasyOCR
- **Auth:** JWT, Flask-CORS

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/name`
3. Code with tests and documentation
4. Format: `black . --line-length 100`
5. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by the Xplorease Team

**📧 Support:** support@xplorease.com | **🐛 Issues:** [GitHub Issues](https://github.com/AnthonyRajuKondaeeti/Xplorease_V2/issues)

</div>