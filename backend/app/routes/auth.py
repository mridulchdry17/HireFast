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
from app.models.db_models import db, RecruiterUser

auth_bp = Blueprint('auth', __name__)

# Initialize LinkedIn service
linkedin_service = LinkedInService()


def _linkedin_redirect_uri() -> str:
    """
    Must match an Authorized redirect URL in the LinkedIn Developer Portal (exact string).

    Do not use Config.APP_BASE_URL here: it defaults to 127.0.0.1 when APP_BASE_URL is unset,
    which would always send users to localhost in prod. Only use APP_BASE_URL if set in the
    environment; otherwise use this request's public URL (same host the user used for /login).

    Optional: LINKEDIN_REDIRECT_URI env overrides everything.
    """
    explicit = (os.environ.get("LINKEDIN_REDIRECT_URI") or "").strip()
    if explicit:
        return explicit
    env_base = (os.environ.get("APP_BASE_URL") or "").strip().rstrip("/")
    if env_base:
        return f"{env_base}/callback"
    return request.url_root.rstrip("/") + "/callback"


def _upsert_recruiter_from_linkedin_profile(prof: dict) -> RecruiterUser:
    """Create or update recruiter row keyed by LinkedIn OIDC ``sub`` (stable id)."""
    sub = prof.get("sub") or prof.get("id")
    if not sub:
        raise ValueError("LinkedIn profile missing subject (sub)")
    sub = str(sub).strip()
    email = (prof.get("email") or "").strip() or None
    given = (prof.get("given_name") or "").strip() or None
    family = (prof.get("family_name") or "").strip() or None
    name = (prof.get("name") or "").strip()
    if not name and (given or family):
        name = " ".join(x for x in (given, family) if x) or None
    picture = prof.get("picture") or prof.get("picture_url")

    user = RecruiterUser.query.filter_by(linkedin_sub=sub).first()
    if user:
        if email:
            user.email = email
        if name:
            user.full_name = name
        if given:
            user.given_name = given
        if family:
            user.family_name = family
        if picture:
            user.picture_url = picture
    else:
        user = RecruiterUser(
            linkedin_sub=sub,
            email=email,
            full_name=name,
            given_name=given,
            family_name=family,
            picture_url=picture,
        )
        db.session.add(user)
    db.session.commit()
    return user


@auth_bp.route('/login')
def login():
    """Initiate LinkedIn OAuth flow."""
    if not Config.LINKEDIN_CLIENT_ID or not Config.LINKEDIN_CLIENT_SECRET:
        return jsonify({'error': 'LinkedIn Client ID or Secret not set'}), 500
    
    state = secrets.token_urlsafe(16)
    session['state'] = state

    redirect_uri = _linkedin_redirect_uri()
    # Token exchange must use the exact same redirect_uri string as the authorize request.
    session["linkedin_oauth_redirect_uri"] = redirect_uri
    auth_params = {
        'response_type': 'code',
        'client_id': Config.LINKEDIN_CLIENT_ID,
        'redirect_uri': redirect_uri,
        'state': state,
        # email: primary email via OIDC userinfo (enable in LinkedIn app + product)
        'scope': 'openid profile email w_member_social',
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
    
    redirect_uri = session.pop("linkedin_oauth_redirect_uri", None) or _linkedin_redirect_uri()
    token_data = {
        'grant_type': 'authorization_code',
        'code': code,
        'client_id': Config.LINKEDIN_CLIENT_ID,
        'client_secret': Config.LINKEDIN_CLIENT_SECRET,
        'redirect_uri': redirect_uri,
    }
    
    try:
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(Config.LINKEDIN_TOKEN_URL, data=token_data, headers=headers)
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to get token', 'details': response.text}), 500
        
        token_response = response.json()
        access_token = token_response['access_token']

        prof = linkedin_service.get_profile_with_token(access_token)
        if 'error' in prof:
            return jsonify({'error': 'Could not load LinkedIn profile', 'details': prof['error']}), 500

        try:
            user = _upsert_recruiter_from_linkedin_profile(prof)
        except ValueError as e:
            return jsonify({'error': str(e)}), 500

        # Stable internal id for session, Composio, and LinkedIn token map
        session['user_id'] = user.id
        linkedin_service.store_user_token(
            user.id,
            access_token,
            token_response.get('id_token'),
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
    
    user_payload = None
    if user_id:
        row = RecruiterUser.query.get(user_id)
        if row:
            user_payload = {
                'id': row.id,
                'full_name': row.full_name,
                'email': row.email,
                'job_title': row.job_title,
                'company': row.company,
                'picture_url': row.picture_url,
            }

    if is_authenticated:
        profile_data = linkedin_service.get_user_profile(user_id)
        
        if 'error' in profile_data:
            return jsonify({
                'authenticated': is_authenticated,
                'token_valid': False,
                'error': profile_data['error'],
                'user': user_payload,
            })
        else:
            return jsonify({
                'authenticated': is_authenticated,
                'token_valid': True,
                'profile_status': 'Profile accessible',
                'user': user_payload,
            })
    
    return jsonify({'authenticated': is_authenticated, 'user': user_payload})


@auth_bp.route('/api/me', methods=['GET'])
def api_me():
    """Current recruiter profile (from DB)."""
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error': 'Unauthorized'}), 401
    user = RecruiterUser.query.get(uid)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({
        'id': user.id,
        'email': user.email,
        'full_name': user.full_name,
        'given_name': user.given_name,
        'family_name': user.family_name,
        'picture_url': user.picture_url,
        'job_title': user.job_title,
        'company': user.company,
        'linkedin_connected': uid in linkedin_service.user_tokens,
    })


@auth_bp.route('/api/me/profile', methods=['PUT'])
def api_me_profile():
    """Update editable profile fields (name/email can override LinkedIn until next login refresh)."""
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error': 'Unauthorized'}), 401
    user = RecruiterUser.query.get(uid)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    data = request.get_json() or {}
    if 'full_name' in data:
        v = data.get('full_name')
        user.full_name = (v or '').strip() or None
    if 'email' in data:
        v = data.get('email')
        user.email = (v or '').strip() or None
    if 'job_title' in data:
        v = data.get('job_title')
        user.job_title = (v or '').strip() or None
    if 'company' in data:
        v = data.get('company')
        user.company = (v or '').strip() or None

    db.session.commit()
    return jsonify({'success': True})
