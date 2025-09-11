import jwt
from datetime import datetime, timedelta

# Use the same secret as in the app
JWT_SECRET = "your-jwt-secret-key"

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
        # "rosmin.varghese@demetrix.in",
        # "rosmin.varghese@sn-ipl.com",
        # "info@demetrix.in"
        "tonykondaveetijmj98@gmail.com"
    ]
    
    print("Generated JWT Tokens for Testing:")
    print("=" * 50)
    
    for email in test_emails:
        token = generate_jwt_token(email)
        print(f"\nEmail: {email}")
        print(f"Token: {token}")
        
        # Verify the token
        payload = verify_jwt_token(token)
        if payload:
            print(f"Verified: {payload['email']} (expires: {datetime.fromtimestamp(payload['exp'])})")
        else:
            print("Token verification failed")
    
    print("\n" + "=" * 50)
    print("Copy these tokens to use in your Postman collection Authorization headers")
    print("Format: Bearer <token>")
import jwt
from datetime import datetime, timedelta

# Use the same secret as in the app
JWT_SECRET = "your-jwt-secret-key"

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
        # "rosmin.varghese@demetrix.in",
        # "rosmin.varghese@sn-ipl.com",
        # "info@demetrix.in"
        "tonykondaveetijmj98@gmail.com"
    ]
    
    print("Generated JWT Tokens for Testing:")
    print("=" * 50)
    
    for email in test_emails:
        token = generate_jwt_token(email)
        print(f"\nEmail: {email}")
        print(f"Token: {token}")
        
        # Verify the token
        payload = verify_jwt_token(token)
        if payload:
            print(f"Verified: {payload['email']} (expires: {datetime.fromtimestamp(payload['exp'])})")
        else:
            print("Token verification failed")
    
    print("\n" + "=" * 50)
    print("Copy these tokens to use in your Postman collection Authorization headers")
    print("Format: Bearer <token>")