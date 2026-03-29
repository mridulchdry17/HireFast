"""
LinkedIn API service for posting job descriptions.
"""
import requests
from typing import Dict, List, Optional
from app.config import Config
from app.utils.helpers import split_text_for_post, get_form_link_for_role

class LinkedInService:
    """Service for LinkedIn API operations."""
    
    def __init__(self):
        self.user_tokens = {}
    
    def store_user_token(self, user_id: str, access_token: str, id_token: Optional[str] = None):
        """Store user tokens for API access."""
        self.user_tokens[user_id] = {
            'access_token': access_token,
            'id_token': id_token
        }
    
    def remove_user_token(self, user_id: str):
        """Remove user tokens."""
        if user_id in self.user_tokens:
            del self.user_tokens[user_id]

    def get_profile_with_token(self, access_token: str) -> Dict:
        """Fetch OpenID userinfo with a fresh access token (used before session user id exists)."""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        try:
            response = requests.get(
                "https://api.linkedin.com/v2/userinfo", headers=headers
            )
            if response.status_code != 200:
                response = requests.get(
                    "https://api.linkedin.com/v2/me", headers=headers
                )
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to get profile: {response.text}"}
        except Exception as e:
            return {"error": f"Profile request failed: {str(e)}"}
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Get user profile information."""
        if user_id not in self.user_tokens:
            return {'error': 'User not authenticated'}
        return self.get_profile_with_token(self.user_tokens[user_id]['access_token'])
    
    def post_job_description(self, user_id: str, role: str, job_description: str, job_id: str = None) -> Dict:
        """
        Post job description to LinkedIn.
        
        Args:
            user_id: User ID
            role: Job role
            job_description: Job description text
            job_id: ID of the job posting in our database
            
        Returns:
            Dictionary with post status and IDs
        """
        if user_id not in self.user_tokens:
            return {'post_status': 'error', 'error': 'User not authenticated'}
        
        # Determine the application link: Internal Portal first, then Google Form fallback
        if job_id:
            base = (Config.APP_BASE_URL or "http://127.0.0.1:5000").rstrip("/")
            form_link = f"{base}/apply/{job_id}"
        else:
            form_link = get_form_link_for_role(role)
            
        jd_with_link = f"{job_description.strip()}\n\nApply here: {form_link}"
        
        # Split into chunks
        chunks = split_text_for_post(jd_with_link)
        if not chunks:
            return {'post_status': 'error', 'error': 'No content to post'}
        
        access_token = self.user_tokens[user_id]['access_token']
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0'
        }
        
        # Get user profile to get person URN
        profile_data = self.get_user_profile(user_id)
        if 'error' in profile_data:
            return {'post_status': 'error', 'error': profile_data['error']}
        
        person_urn = profile_data.get('sub') or profile_data.get('id')
        if not person_urn:
            return {'post_status': 'error', 'error': 'Could not retrieve user URN'}
        
        post_ids = []
        for idx, chunk in enumerate(chunks):
            payload = {
                'author': f'urn:li:person:{person_urn}',
                'lifecycleState': 'PUBLISHED',
                'specificContent': {
                    'com.linkedin.ugc.ShareContent': {
                        'shareCommentary': {'text': chunk},
                        'shareMediaCategory': 'NONE'
                    }
                },
                'visibility': {'com.linkedin.ugc.MemberNetworkVisibility': 'PUBLIC'}
            }
            
            print(f"Posting chunk {idx+1}/{len(chunks)} with person_urn: {person_urn}")
            
            try:
                response = requests.post(
                    'https://api.linkedin.com/v2/ugcPosts',
                    json=payload,
                    headers=headers
                )
                
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
                
                if response.status_code in [200, 201]:
                    post_id = response.headers.get('X-RestLi-Id')
                    post_url = f'https://www.linkedin.com/feed/update/{post_id}' if post_id else None
                    post_ids.append({'id': post_id, 'url': post_url})
                    print(f"Successfully posted with ID: {post_id}")
                else:
                    return {'post_status': 'error', 'error': f'Post failed: {response.text}', 'post_ids': post_ids}
            except Exception as e:
                print(f"Exception during posting: {str(e)}")
                return {'post_status': 'error', 'error': f'Post request failed: {str(e)}', 'post_ids': post_ids}
        
        return {'post_status': 'success', 'post_ids': post_ids}
