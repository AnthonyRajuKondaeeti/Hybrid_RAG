# Xplorease V2 - Document Intelligence & RAG API

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Overview

Xplorease V2 is an advanced **Document Intelligence Platform** that combines **Retrieval-Augmented Generation (RAG)** with modern AI technologies to provide intelligent document processing and conversational Q&A capabilities. The platform allows users to upload PDF documents, process them using advanced NLP techniques, and interact with the content through natural language queries.

### ✨ Key Features

- 📄 **Multi Document Processing** - Upload and process multiple files simultaneously
- 🤖 **RAG-Powered Q&A** - Ask questions about document content with context-aware responses
- 🔍 **Semantic Search** - Advanced vector-based document search using embeddings
- 🧠 **Conversation Memory** - Maintains context across multiple queries in a session
- 🔐 **JWT Authentication** - Secure API access with token-based authentication
- 📊 **QR Code Generation** - Generate QR codes for easy document sharing
- 🌐 **RESTful API** - Clean, well-documented API endpoints
- ⚡ **High Performance** - Optimized for speed and scalability

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend/     │    │   Flask API     │    │   Qdrant DB     │
│   Postman       │────│   (RAG Engine)  │────│   (Vector Store)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌─────────────────┐
                       │   Mistral AI    │
                       │   (LLM Model)   │
                       └─────────────────┘
```

### 🔧 Technology Stack

- **Backend Framework**: Flask 2.0+
- **AI/ML Stack**:
  - Mistral AI (Large Language Model)
  - Sentence Transformers (Embeddings)
  - LangChain (RAG Framework)
- **Vector Database**: Qdrant Cloud
- **Document Processing**: PyMuPDF, PDFPlumber
- **Authentication**: JWT (JSON Web Tokens)
- **Other**: CORS, Werkzeug, Python-dotenv

## 📁 Project Structure

```
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
