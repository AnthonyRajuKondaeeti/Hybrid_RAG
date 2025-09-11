# File: run.py
"""
Production-ready Flask application runner
"""
import os
from app import create_app
from config import Config

# Create application
app = create_app()

if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Starting XplorEase V2 Compatible RAG API Server...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print(f"Qdrant URL: {Config.QDRANT_URL}")
    print(f"V2 Compatible Mode: Enabled")
    
    app.run(
        host=host,
        port=port,
        debug=debug
    )
# File: run.py
"""
Production-ready Flask application runner
"""
import os
from app import create_app
from config import Config

# Create application
app = create_app()

if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Starting XplorEase V2 Compatible RAG API Server...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print(f"Qdrant URL: {Config.QDRANT_URL}")
    print(f"V2 Compatible Mode: Enabled")
    
    app.run(
        host=host,
        port=port,
        debug=debug
    )