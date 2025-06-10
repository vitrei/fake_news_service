import requests
from typing import Optional, Dict, Any

def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user profile from user_profile_builder service (port 8010)"""
    try:
        url = f"http://localhost:8010/users/{user_id}"
        print(f"DEBUG: Making request to: {url}")
        
        response = requests.get(url)
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response content: {response.text[:200]}...")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"ERROR getting user profile: {e}")
        print(f"DEBUG: Failed URL was: {url}")
        return None
    


