import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import sys

load_dotenv()

def test_connection():
    print("🔍 Testing Qdrant Cloud Connection...")
    print("=" * 50)
    
    url = os.getenv('QDRANT_URL')
    api_key = os.getenv('QDRANT_API_KEY')
    
    print(f"URL: {url}")
    print(f"API Key: {'✓ Present' if api_key else '✗ Missing'}")
    print()
    
    # Test different connection methods
    connection_configs = [
        {
            "name": "Standard Connection",
            "config": {
                "url": url,
                "api_key": api_key,
                "timeout": 60
            }
        },
        {
            "name": "HTTP-only Connection", 
            "config": {
                "url": url,
                "api_key": api_key,
                "timeout": 60,
                "prefer_grpc": False
            }
        },
        {
            "name": "HTTP with SSL verification",
            "config": {
                "url": url,
                "api_key": api_key,
                "timeout": 60,
                "prefer_grpc": False,
                "https": True,
                "verify": True
            }
        }
    ]
    
    for test_config in connection_configs:
        print(f"\n🔌 Testing: {test_config['name']}")
        try:
            client = QdrantClient(**test_config['config'])
            collections = client.get_collections()
            print(f"✅ SUCCESS! Found {len(collections.collections)} collections")
            print(f"Collections: {[c.name for c in collections.collections]}")
            return True
        except Exception as e:
            print(f"❌ FAILED: {e}")
    
    return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\n🎉 Connection successful! Your Qdrant Cloud setup is working.")
    else:
        print("\n❌ All connection attempts failed.")
        print("\n🔧 Next steps:")
        print("1. Check your Qdrant Cloud dashboard - ensure cluster is running")
        print("2. Verify your API key has proper permissions")
        print("3. Check Windows Firewall/antivirus settings")
        print("4. Try accessing your cluster URL in a web browser")