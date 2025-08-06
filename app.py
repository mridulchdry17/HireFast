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

# OAuth 2.0 settings
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = "http://127.0.0.1:5000/callback"
AUTH_URL = "https://twitter.com/i/oauth2/authorize"
TOKEN_URL = "https://api.twitter.com/2/oauth2/token"

user_tokens = {}

class HRHiringState(TypedDict):
    role: Optional[str]
    job_description: Optional[str]
    approval: Optional[bool]

# --- LangGraph Workflow Setup ---
def generate_jd_node(state: HRHiringState) -> dict:
    role = state.get('role')
    # Use a concise, Twitter-friendly prompt
    template = (
    "You are an HR and copywriting expert. Write a minimal, impactful job description for the role of {role}, optimized for posting as a thread on Twitter/X.\n\n"
    "Guidelines:\n"
    "- Use short, simple, standalone sentences (max 280 characters each).\n"
    "- Avoid bullet points, formatting, or special characters.\n"
    "- Focus only on the most essential responsibilities, skills, and benefits.\n"
    "- Be concise and clear. Each sentence should make sense on its own.\n"
    "- Do not repeat information or include greetings, hashtags, or emojis.\n\n"
    "Return the response as a plain text paragraph with tweet-sized lines separated by newlines. Do not number the lines or label them.\n\n"
    "Role: {role}"
    )
    prompt = PromptTemplate.from_template(template).format(role=role)
    llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model='llama3-8b-8192')
    jd = llm.invoke(prompt)
    return {'role': role, 'job_description': jd.content}

def human_approval_node(state: HRHiringState) -> dict:
    # Approval is provided by the user via the frontend, so just pass through
    return {'role': state.get('role'), 'job_description': state.get('job_description'), 'approval': state.get('approval')}

# Improved chunking function
def split_text_for_thread(text, max_length=280):
    # Remove Markdown formatting
    text = re.sub(r'\*\*|\*', '', text)
    # Replace multiple newlines with a single space
    text = re.sub(r'\n+', ' ', text)
    # Split into words and build chunks
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
    # Print chunk lengths for debugging
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1} length: {len(chunk)}")
    return chunks

# Role to Google Form mapping
ROLE_FORM_LINKS = {
    "ai": "https://forms.gle/hWuPeNMyZCq8cePj7",
    "machine learning": "https://forms.gle/ai-form-link",
    "data": "https://forms.gle/ai-form-link",
    "marketing": "https://forms.gle/marketing-form-link",
    # Add more as needed
}
GENERIC_FORM_LINK = "https://forms.gle/generic-form-link"

def get_form_link_for_role(role: str) -> str:
    role_lower = role.lower()
    for keyword, link in ROLE_FORM_LINKS.items():
        if keyword in role_lower:
            return link
    return GENERIC_FORM_LINK

# Update post_twitter_node to append the form link

def post_twitter_node(state: HRHiringState) -> dict:
    import time
    jd = state.get('job_description')
    role = state.get('role', '')
    user_id = session.get('user_id')
    if not jd or not user_id or user_id not in user_tokens:
        return {'post_status': 'error', 'error': 'Missing JD or not authenticated'}
    # Append the correct Google Form link
    form_link = get_form_link_for_role(role)
    jd_with_link = f"{jd.strip()}\n\nApply here: {form_link}"
    access_token = user_tokens[user_id]['access_token']
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    chunks = split_text_for_thread(jd_with_link)
    tweet_ids = []
    in_reply_to = None
    for idx, chunk in enumerate(chunks):
        payload = {'text': chunk}
        if in_reply_to:
            payload['reply'] = {'in_reply_to_tweet_id': in_reply_to}
        print(f"Posting chunk {idx+1}/{len(chunks)}: {payload}")
        response = requests.post('https://api.twitter.com/2/tweets', json=payload, headers=headers)
        print(f"Response status: {response.status_code}, Response: {response.text}")
        if response.status_code == 201:
            tweet_id = response.json().get('data', {}).get('id')
            tweet_ids.append(tweet_id)
            in_reply_to = tweet_id
            time.sleep(1)  # Delay to ensure tweet is available for reply
        else:
            return {'post_status': 'error', 'error': response.text, 'tweet_ids': tweet_ids}
    return {'post_status': 'success', 'tweet_ids': tweet_ids}

def approval_conditional(state: HRHiringState) -> str:
    return 'post_twitter' if state.get('approval') else 'generate_jd'

builder = StateGraph(HRHiringState)
builder.add_node('generate_jd', generate_jd_node)
builder.add_node('human_approval', human_approval_node)
builder.add_node('post_twitter', post_twitter_node)
builder.add_edge(START, 'generate_jd')
builder.add_edge('generate_jd', 'human_approval')
builder.add_conditional_edges('human_approval', approval_conditional)
builder.add_edge('post_twitter', END)
graph = builder.compile()

# --- Flask Endpoints using LangGraph nodes ---
@app.route('/')
def landing_page():
    return send_file('index.html')

@app.route('/login')
def login():
    if not CLIENT_ID:
        return jsonify({'error': 'Client ID not set'}), 500
    state = secrets.token_urlsafe(16)
    session['state'] = state
    session['user_id'] = secrets.token_urlsafe(16)
    code_verifier = secrets.token_urlsafe(32)
    import hashlib, base64
    code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest()).decode().rstrip('=')
    session['code_verifier'] = code_verifier
    auth_params = {
        'response_type': 'code',
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': 'tweet.read tweet.write users.read offline.access',
        'state': state,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256'
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
    code_verifier = session.get('code_verifier')
    if not code_verifier:
        return jsonify({'error': 'Code verifier not found in session'}), 400
    token_data = {
        'grant_type': 'authorization_code',
        'code': code,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'redirect_uri': REDIRECT_URI,
        'code_verifier': code_verifier
    }
    try:
        import base64
        credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': f'Basic {encoded_credentials}'
        }
        response = requests.post(TOKEN_URL, data=token_data, headers=headers)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to get token', 'details': response.text}), 500
        token_response = response.json()
        user_tokens[session['user_id']] = {
            'access_token': token_response['access_token'],
            'refresh_token': token_response.get('refresh_token')
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
    return jsonify({'authenticated': is_authenticated})

@app.route('/generate-jd', methods=['POST'])
def generate_jd_api():
    data = request.get_json()
    state = {'role': data.get('role')}
    result = generate_jd_node(state)
    return jsonify(result)

@app.route('/approve-jd', methods=['POST'])
def approve_jd_api():
    data = request.get_json()
    state = {
        'role': data.get('role'),
        'job_description': data.get('job_description'),
        'approval': data.get('approval')
    }
    result = human_approval_node(state)
    # If approval is False, regenerate JD
    if not data.get('approval'):
        result = generate_jd_node({'role': data.get('role')})
    return jsonify(result)

@app.route('/post-jd', methods=['POST'])
def post_jd_api():
    data = request.get_json()
    state = {
        'role': data.get('role'),
        'job_description': data.get('job_description'),
        'approval': True
    }
    result = post_twitter_node(state)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 
    