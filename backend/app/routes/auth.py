"""
Authentication routes for LinkedIn OAuth.
"""
import os
import secrets
import requests
from urllib.parse import urlencode

from flask import Blueprint, request, jsonify, redirect, url_for, session
from app.config import Config
from app.services.linkedin_service import LinkedInService

auth_bp = Blueprint('auth', __name__)

# Initialize LinkedIn service
linkedin_service = LinkedInService()


def _linkedin_redirect_uri() -> str:
    """
    Must match an Authorized redirect URL in the LinkedIn Developer Portal (exact string).

    Uses Config.APP_BASE_URL (from backend/.env) → {APP_BASE_URL}/callback.
    Optional LINKEDIN_REDIRECT_URI env overrides. If APP_BASE_URL is empty, uses this request host.
    """
    explicit = (os.environ.get("LINKEDIN_REDIRECT_URI") or "").strip()
    if explicit:
        return explicit
    base = (Config.APP_BASE_URL or "").strip().rstrip("/")
    if base:
        return f"{base}/callback"
    return request.url_root.rstrip("/") + "/callback"


@auth_bp.route('/login')
def login():
    """Initiate LinkedIn OAuth flow."""
    if not Config.LINKEDIN_CLIENT_ID or not Config.LINKEDIN_CLIENT_SECRET:
        return jsonify({'error': 'LinkedIn Client ID or Secret not set'}), 500
    
    state = secrets.token_urlsafe(16)
    session['state'] = state
    session['user_id'] = secrets.token_urlsafe(16)

    redirect_uri = _linkedin_redirect_uri()
    auth_params = {
        'response_type': 'code',
        'client_id': Config.LINKEDIN_CLIENT_ID,
        'redirect_uri': redirect_uri,
        'state': state,
        'scope': 'openid profile w_member_social',
    }

    auth_url = f"{Config.LINKEDIN_AUTH_URL}?{urlencode(auth_params)}"
    return redirect(auth_url)

@auth_bp.route('/callback')
def callback():
    """Handle LinkedIn OAuth callback."""
    code = request.args.get('code')
    state = request.args.get('state')
    error = request.args.get('error')
    error_description = request.args.get('error_description')
    saved_state = session.get('state')
    
    if error:
        return jsonify({'error': f'OAuth error: {error}', 'description': error_description}), 400
    
    if not code or state != saved_state:
        return jsonify({'error': 'Invalid login attempt'}), 400
    
    token_data = {
        'grant_type': 'authorization_code',
        'code': code,
        'client_id': Config.LINKEDIN_CLIENT_ID,
        'client_secret': Config.LINKEDIN_CLIENT_SECRET,
        'redirect_uri': _linkedin_redirect_uri(),
    }
    
    try:
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(Config.LINKEDIN_TOKEN_URL, data=token_data, headers=headers)
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to get token', 'details': response.text}), 500
        
        token_response = response.json()
        
        # Store tokens in LinkedIn service
        linkedin_service.store_user_token(
            session['user_id'],
            token_response['access_token'],
            token_response.get('id_token')
        )
        
        return redirect(url_for('landing_page'))
        
    except Exception as e:
        return jsonify({'error': f'Authentication failed: {str(e)}'}), 500

@auth_bp.route('/logout')
def logout():
    """Logout user and clear session."""
    user_id = session.get('user_id')
    if user_id:
        linkedin_service.remove_user_token(user_id)
    session.clear()
    return redirect(url_for('landing_page'))

@auth_bp.route('/check-auth')
def check_auth():
    """Check if user is authenticated and token is valid."""
    user_id = session.get('user_id')
    is_authenticated = user_id is not None and user_id in linkedin_service.user_tokens
    
    if is_authenticated:
        # Test token validity by getting user profile
        profile_data = linkedin_service.get_user_profile(user_id)
        
        if 'error' in profile_data:
            return jsonify({
                'authenticated': is_authenticated,
                'token_valid': False,
                'error': profile_data['error']
            })
        else:
            return jsonify({
                'authenticated': is_authenticated,
                'token_valid': True,
                'profile_status': 'Profile accessible'
            })
    
    return jsonify({'authenticated': is_authenticated})
