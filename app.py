import os
from flask import Flask, request, jsonify, send_file, redirect, url_for, session
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional
from langchain_groq import ChatGroq
import secrets
import requests
from langchain.prompts import PromptTemplate
import re

load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# LinkedIn OAuth 2.0 settings
CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
REDIRECT_URI = "http://127.0.0.1:5000/callback"
AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"

user_tokens = {}

class HRHiringState(TypedDict):
    role: Optional[str]
    job_description: Optional[str]
    approval: Optional[bool]

# --- LangGraph Workflow Setup ---
def generate_jd_node(state: HRHiringState) -> dict:
    role = state.get('role')
    if not role:
        return {'post_status': 'error', 'error': 'Role is required'}
    template = (
        "You are an HR and copywriting expert. Write a concise, professional job description for the role of {role}, optimized for posting on LinkedIn.\n\n"
        "Guidelines:\n"
        "- Keep the total length under 1200 characters to allow space for the application link.\n"
        "- Use clear, professional language with a compelling hook.\n"
        "- Include key responsibilities, qualifications, and a call-to-action.\n"
        "- Add relevant hashtags (e.g., #Hiring, #JobOpportunity) and emojis for engagement.\n"
        "- Avoid bullet points or excessive formatting; use plain text with newlines for readability.\n\n"
        "Return the response as a plain text paragraph.\n\n"
        "Role: {role}"
    )
    prompt = PromptTemplate.from_template(template).format(role=role)
    llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model='llama3-8b-8192')
    try:
        jd = llm.invoke(prompt)
        return {'role': role, 'job_description': jd.content}
    except Exception as e:
        return {'post_status': 'error', 'error': f'JD generation failed: {str(e)}'}

def human_approval_node(state: HRHiringState) -> dict:
    return {'role': state.get('role'), 'job_description': state.get('job_description'), 'approval': state.get('approval')}

# Restored chunking function for LinkedIn's 1300-character limit
def split_text_for_post(text, max_length=1300):
    text = re.sub(r'\*\*|\*', '', text)  # Remove Markdown
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
    words = text.split()
    chunks = []
    current = ''
    for word in words:
        if len(current) + len(word) + 1 <= max_length:
            current += (' ' if current else '') + word
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1} length: {len(chunk)}")
    return chunks

# Role to application link mapping
ROLE_FORM_LINKS = {
    "ai": "https://forms.gle/8R5Eg8TaQHgQBFE19",
    "machine learning": "https://forms.gle/8R5Eg8TaQHgQBFE19",
    "data": "https://forms.gle/8R5Eg8TaQHgQBFE19",
    "marketing": "https://forms.gle/8R5Eg8TaQHgQBFE19",
}
GENERIC_FORM_LINK = "https://forms.gle/8R5Eg8TaQHgQBFE19"

def get_form_link_for_role(role: str) -> str:
    role_lower = role.lower()
    for keyword, link in ROLE_FORM_LINKS.items():
        if keyword in role_lower:
            return link
    return GENERIC_FORM_LINK

# LinkedIn posting node
def post_linkedin_node(state: HRHiringState) -> dict:
    jd = state.get('job_description')
    role = state.get('role', '')
    user_id = session.get('user_id')
    if not jd or not user_id or user_id not in user_tokens:
        return {'post_status': 'error', 'error': 'Missing JD or not authenticated'}

    # Append the application link
    form_link = get_form_link_for_role(role)
    jd_with_link = f"{jd.strip()}\n\nApply here: {form_link}"

    # Split into chunks (though typically one post for LinkedIn)
    chunks = split_text_for_post(jd_with_link)
    if not chunks:
        return {'post_status': 'error', 'error': 'No content to post'}

    access_token = user_tokens[user_id]['access_token']
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'X-Restli-Protocol-Version': '2.0.0'
    }

    # Get user ID from OpenID Connect ID token or fallback to profile API
    user_id = session.get('user_id')
    person_urn = None
    
    if user_id in user_tokens and user_tokens[user_id].get('id_token'):
        # Try to extract user ID from ID token (simplified approach)
        # In production, you should properly decode and validate the JWT
        try:
            # For now, we'll use the profile API as fallback
            profile_response = requests.get('https://api.linkedin.com/v2/userinfo', headers=headers)
            if profile_response.status_code == 200:
                profile_data = profile_response.json()
                person_urn = profile_data.get('sub')  # OpenID Connect subject identifier
        except:
            pass
    
    # Fallback to profile API if OpenID Connect approach fails
    if not person_urn:
        profile_response = requests.get('https://api.linkedin.com/v2/me', headers=headers)
        if profile_response.status_code != 200:
            return {'post_status': 'error', 'error': f'Failed to get user profile: {profile_response.text}'}
        profile_data = profile_response.json()
        person_urn = profile_data.get('id')
    
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
        print(f"Payload: {payload}")
        try:
            response = requests.post(
                'https://api.linkedin.com/v2/ugcPosts',
                json=payload,
                headers=headers
            )
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
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

def approval_conditional(state: HRHiringState) -> str:
    return 'post_linkedin' if state.get('approval') else 'generate_jd'

# Update StateGraph
builder = StateGraph(HRHiringState)
builder.add_node('generate_jd', generate_jd_node)
builder.add_node('human_approval', human_approval_node)
builder.add_node('post_linkedin', post_linkedin_node)
builder.add_edge(START, 'generate_jd')
builder.add_edge('generate_jd', 'human_approval')
builder.add_conditional_edges('human_approval', approval_conditional)
builder.add_edge('post_linkedin', END)
graph = builder.compile()

# --- Flask Endpoints ---
@app.route('/')
def landing_page():
    return send_file('index.html')

@app.route('/login')
def login():
    if not CLIENT_ID or not CLIENT_SECRET:
        return jsonify({'error': 'LinkedIn Client ID or Secret not set'}), 500
    state = secrets.token_urlsafe(16)
    session['state'] = state
    session['user_id'] = secrets.token_urlsafe(16)
    auth_params = {
        'response_type': 'code',
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'state': state,
        'scope': 'openid profile w_member_social'  # Use OpenID Connect scopes
    }
    auth_url = f"{AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in auth_params.items())}"
    return redirect(auth_url)

@app.route('/callback')
def callback():
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
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'redirect_uri': REDIRECT_URI
    }
    try:
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(TOKEN_URL, data=token_data, headers=headers)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to get token', 'details': response.text}), 500
        token_response = response.json()
        
        # Store both access token and ID token
        user_tokens[session['user_id']] = {
            'access_token': token_response['access_token'],
            'id_token': token_response.get('id_token')  # OpenID Connect ID token
        }
        return redirect(url_for('landing_page'))
    except Exception as e:
        return jsonify({'error': f'Authentication failed: {str(e)}'}), 500

@app.route('/logout')
def logout():
    user_id = session.get('user_id')
    if user_id and user_id in user_tokens:
        del user_tokens[user_id]
    session.clear()
    return redirect(url_for('landing_page'))

@app.route('/check-auth')
def check_auth():
    user_id = session.get('user_id')
    is_authenticated = user_id is not None and user_id in user_tokens
    if is_authenticated:
        access_token = user_tokens[user_id]['access_token']
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0'
        }
        try:
            # Try OpenID Connect userinfo endpoint first
            test_response = requests.get('https://api.linkedin.com/v2/userinfo', headers=headers)
            if test_response.status_code != 200:
                # Fallback to profile API
                test_response = requests.get('https://api.linkedin.com/v2/me', headers=headers)
            
            token_valid = test_response.status_code == 200
            return jsonify({
                'authenticated': is_authenticated,
                'token_valid': token_valid,
                'profile_status': test_response.status_code,
                'profile_response': test_response.text[:200] if not token_valid else 'Profile accessible'
            })
        except Exception as e:
            return jsonify({
                'authenticated': is_authenticated,
                'token_valid': False,
                'error': str(e)
            })
    return jsonify({'authenticated': is_authenticated})

@app.route('/generate-jd', methods=['POST'])
def generate_jd_api():
    data = request.get_json()
    if not data or 'role' not in data:
        return jsonify({'error': 'Role is required'}), 400
    state = {'role': data.get('role')}
    result = generate_jd_node(state)
    return jsonify(result)

@app.route('/approve-jd', methods=['POST'])
def approve_jd_api():
    data = request.get_json()
    if not data or 'role' not in data or 'job_description' not in data:
        return jsonify({'error': 'Role and job_description are required'}), 400
    state = {
        'role': data.get('role'),
        'job_description': data.get('job_description'),
        'approval': data.get('approval', False)
    }
    result = human_approval_node(state)
    if not data.get('approval'):
        result = generate_jd_node({'role': data.get('role')})
    return jsonify(result)

@app.route('/post-jd', methods=['POST'])
def post_jd_api():
    data = request.get_json()
    if not data or 'role' not in data or 'job_description' not in data:
        return jsonify({'error': 'Role and job_description are required'}), 400
    state = {
        'role': data.get('role'),
        'job_description': data.get('job_description'),
        'approval': True
    }
    result = post_linkedin_node(state)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)