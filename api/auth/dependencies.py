from typing import Dict, Optional
from fastapi import Header, HTTPException

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict:
    """
    Mock authentication dependency.
    In a real app, this would decode a JWT token.
    For now, it just returns a dummy user or passes if no auth is strictly enforced.
    """
    # For testing purposes, we allow unauthenticated access or check for a simple header
    # if authorization is None:
    #     raise HTTPException(status_code=401, detail="Missing authentication header")
    
    return {
        "id": "user_123",
        "username": "test_user",
        "email": "test@example.com"
    }
