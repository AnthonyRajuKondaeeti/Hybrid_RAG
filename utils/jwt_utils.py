import jwt
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use the same secret as in the main app
JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-jwt-key-change-this-in-production-make-it-32-chars-minimum')

def generate_jwt_token(email: str, expires_in_hours: int = 24) -> str:
    """
    Generate JWT token for testing
    
    Args:
        email: User email
        expires_in_hours: Token expiration time in hours
    
    Returns:
        JWT token string
    """
    payload = {
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
        'iat': datetime.utcnow()
    }
    
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_jwt_token(token: str) -> dict:
    """
    Verify and decode JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded payload or None if invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        print("Token has expired")
        return None
    except jwt.InvalidTokenError:
        print("Token is invalid")
        return None

if __name__ == "__main__":
    # Generate test tokens
    test_emails = [
        "tonykondaveetijmj98@gmail.com",
        "test@example.com"  # Additional test email
    ]
    
    print("Generated JWT Tokens for Testing:")
    print("=" * 60)
    print(f"Using JWT Secret: {JWT_SECRET}")
    print("=" * 60)
    
    for email in test_emails:
        token = generate_jwt_token(email)
        print(f"\nEmail: {email}")
        print(f"Token: {token}")
        print(f"Authorization Header: Bearer {token}")
        
        # Verify the token
        payload = verify_jwt_token(token)
        if payload:
            print(f"Verified: {payload['email']} (expires: {datetime.fromtimestamp(payload['exp'])})")
        else:
            print("Token verification failed")
    
    print("\n" + "=" * 60)
    print("Copy the 'Authorization Header' value to use in Postman:")
    print("1. In Postman, go to Authorization tab")
    print("2. Select 'Bearer Token' type")
    print("3. Paste the token (without 'Bearer ' prefix)")
    print("OR")
    print("4. Add Header: Authorization = Bearer <token>")