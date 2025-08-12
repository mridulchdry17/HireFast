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
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from langchain_community.document_loaders import PyPDFLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
# New imports for personal Google Account OAuth
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import pickle
import json

load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# LinkedIn OAuth 2.0 settings
CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
REDIRECT_URI = "http://127.0.0.1:5000/callback"
AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"

# Google Calendar OAuth settings
GOOGLE_CALENDAR_SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events'
]
GOOGLE_CALENDAR_CREDENTIALS_FILE = 'google_calendar_credentials.json'
GOOGLE_CALENDAR_TOKEN_FILE = 'google_calendar_token.pickle'

user_tokens = {}

def get_google_calendar_credentials():
    """
    Get Google Calendar credentials using OAuth 2.0 for personal account.
    Returns credentials that can send email invites.
    """
    creds = None
    
    # Check if we have a valid token file
    if os.path.exists(GOOGLE_CALENDAR_TOKEN_FILE):
        with open(GOOGLE_CALENDAR_TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(GOOGLE_CALENDAR_CREDENTIALS_FILE):
                return {
                    'error': 'Google Calendar credentials file not found. Please download OAuth 2.0 credentials from Google Cloud Console.'
                }
            
            flow = InstalledAppFlow.from_client_secrets_file(
                GOOGLE_CALENDAR_CREDENTIALS_FILE, 
                GOOGLE_CALENDAR_SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(GOOGLE_CALENDAR_TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

@app.route('/google-calendar-auth')
def google_calendar_auth():
    """Initiate Google Calendar OAuth flow."""
    try:
        if not os.path.exists(GOOGLE_CALENDAR_CREDENTIALS_FILE):
            return jsonify({
                'error': 'Google Calendar credentials file not found',
                'instructions': [
                    '1. Go to Google Cloud Console',
                    '2. Create a new project or select existing one',
                    '3. Enable Google Calendar API',
                    '4. Go to Credentials',
                    '5. Create OAuth 2.0 Client ID (Desktop application)',
                    '6. Download the JSON file',
                    '7. Rename it to google_calendar_credentials.json',
                    '8. Place it in the project root directory'
                ]
            }), 400
        
        # This will trigger the OAuth flow
        creds = get_google_calendar_credentials()
        
        if isinstance(creds, dict) and 'error' in creds:
            return jsonify(creds), 400
        
        return jsonify({
            'status': 'success',
            'message': 'Google Calendar authentication successful! You can now send calendar invites.'
        })
        
    except Exception as e:
        return jsonify({'error': f'Authentication failed: {str(e)}'}), 500

@app.route('/check-google-calendar-auth')
def check_google_calendar_auth():
    """Check if Google Calendar is authenticated."""
    try:
        creds = get_google_calendar_credentials()
        
        if isinstance(creds, dict) and 'error' in creds:
            return jsonify({
                'authenticated': False,
                'error': creds['error']
            })
        
        # Test the credentials by trying to access calendar
        service = build('calendar', 'v3', credentials=creds)
        calendar_list = service.calendarList().list().execute()
        
        return jsonify({
            'authenticated': True,
            'calendars': [cal['summary'] for cal in calendar_list.get('items', [])]
        })
        
    except Exception as e:
        return jsonify({
            'authenticated': False,
            'error': str(e)
        })

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
    "- Avoid bullet points or excessive formatting; use plain text with newlines for readability.\n"
    "- Do NOT start with phrases like 'Here is the job description' or any introductory statements; start directly with the content.\n\n"
    "Return the response as a plain text paragraph only.\n\n"
    "Role: {role}"
    )

    prompt = PromptTemplate.from_template(template).format(role=role)
    llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model='openai/gpt-oss-120b')
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

def extract_drive_file_id(drive_url):
    """Extracts the file ID from a Google Drive share link."""
    if not drive_url:
        print("Empty drive URL provided")
        return None
        
    print(f"Extracting file ID from: {drive_url}")
    
    # Try different patterns
    patterns = [
        r'/d/([\w-]+)',  # Standard share link
        r'id=([\w-]+)',  # ID parameter
        r'/file/d/([\w-]+)',  # Direct file link
        r'/drive/folders/([\w-]+)',  # Folder link (for debugging)
        r'([\w-]{25,})'  # Generic long ID pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_url)
        if match:
            file_id = match.group(1)
            print(f"Found file ID: {file_id} using pattern: {pattern}")
            return file_id
    
    print(f"No file ID found in URL: {drive_url}")
    return None

GOOGLE_SHEET_ID = "1PrMvW7un7b1zdF_262QwtaP7_xNhf5zxylYIqnOSq_U"
DRIVE_FOLDER_ID = "1lFh4bkL2PYtyjFcVMiw0SuVEEeP3hXMNuCH4zQgGb8UOyLO1fxs528Gq5jxV8z4lsH3h-9KF"
CREDENTIALS_PATH = "credentials.json"

def fetch_applications(sheet_id=GOOGLE_SHEET_ID, credentials_path=CREDENTIALS_PATH, range_name='Form Responses 1'):
    """
    Reads applicant data from a Google Sheet and extracts Drive file IDs for resumes.
    Skips the first column (timestamp).
    Returns a list of dicts: {name, college, email, intro, resume_file_id}
    """
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
    values = result.get('values', [])
    
    print(f"Raw sheet data: {values}")
    
    if not values or len(values) < 2:
        print("No data found in sheet")
        return []
        
    headers = values[0][1:]  # Skip timestamp
    data_rows = values[1:]
    
    print(f"Headers: {headers}")
    print(f"Data rows: {data_rows}")
    
    applicants = []
    for i, row in enumerate(data_rows):
        print(f"Processing row {i+1}: {row}")
        
        if len(row) < len(headers) + 1:
            print(f"Row {i+1} is incomplete, skipping")
            continue  # skip incomplete rows
            
        row_dict = dict(zip(headers, row[1:]))  # Skip timestamp in each row
        print(f"Row {i+1} dict: {row_dict}")
        
        resume_url = row_dict.get('RESUME ', '')  # Note the trailing space to match the actual header
        print(f"Resume URL for row {i+1}: '{resume_url}'")
        
        file_id = extract_drive_file_id(resume_url)
        print(f"Extracted file ID for row {i+1}: '{file_id}'")
        
        applicant = {
            'name': row_dict.get('Name', ''),
            'college': row_dict.get('College Name and Course', ''),
            'email': row_dict.get('Email', ''),
            'intro': row_dict.get('Short Intro about you', ''),
            'resume_file_id': file_id
        }
        
        print(f"Created applicant: {applicant}")
        applicants.append(applicant)
    
    print(f"Final applicants list: {applicants}")
    return applicants

def fetch_applicants_with_resumes(sheet_id=GOOGLE_SHEET_ID, credentials_path=CREDENTIALS_PATH, range_name='Form Responses 1'):
    """
    Reads applicant data from Google Sheet and extracts Drive file IDs for resumes.
    Returns a list of dicts with complete applicant info: {name, college, email, intro, resume_file_id}
    """
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
    values = result.get('values', [])
    
    print(f"Raw sheet data: {values}")
    
    if not values or len(values) < 2:
        print("No data found in sheet")
        return []
        
    headers = values[0]
    data_rows = values[1:]
    
    print(f"Headers: {headers}")
    print(f"Data rows: {data_rows}")
    
    # Find the RESUME column index (handle trailing spaces)
    resume_col_index = None
    for i, header in enumerate(headers):
        if header.strip() == 'RESUME':
            resume_col_index = i
            break
    
    if resume_col_index is None:
        print("RESUME column not found!")
        print(f"Available headers: {[h.strip() for h in headers]}")
        return []
    
    print(f"RESUME column found at index: {resume_col_index}")
    
    applicants = []
    for i, row in enumerate(data_rows):
        print(f"Processing row {i+1}: {row}")
        
        if len(row) > resume_col_index:
            resume_url = row[resume_col_index]
            print(f"Resume URL for row {i+1}: '{resume_url}'")
            
            file_id = extract_drive_file_id(resume_url)
            print(f"Extracted file ID for row {i+1}: '{file_id}'")
            
            if file_id:
                # Create applicant dict with all available data
                applicant = {
                    'name': row[1] if len(row) > 1 else '',
                    'college': row[2] if len(row) > 2 else '',
                    'email': row[3] if len(row) > 3 else '',
                    'intro': row[4] if len(row) > 4 else '',
                    'resume_file_id': file_id
                }
                applicants.append(applicant)
        else:
            print(f"Row {i+1} doesn't have RESUME column data")
    
    print(f"Final applicants list: {applicants}")
    return applicants

def fetch_resume_file_ids(sheet_id=GOOGLE_SHEET_ID, credentials_path=CREDENTIALS_PATH, range_name='Form Responses 1'):
    """
    Reads only the RESUME column from Google Sheet and extracts Drive file IDs.
    Returns a list of file IDs for resumes.
    """
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
    values = result.get('values', [])
    
    print(f"Raw sheet data: {values}")
    
    if not values or len(values) < 2:
        print("No data found in sheet")
        return []
        
    headers = values[0]
    data_rows = values[1:]
    
    print(f"Headers: {headers}")
    print(f"Data rows: {data_rows}")
    
    # Find the RESUME column index (handle trailing spaces)
    resume_col_index = None
    for i, header in enumerate(headers):
        if header.strip() == 'RESUME':
            resume_col_index = i
            break
    
    if resume_col_index is None:
        print("RESUME column not found!")
        print(f"Available headers: {[h.strip() for h in headers]}")
        return []
    
    print(f"RESUME column found at index: {resume_col_index}")
    
    file_ids = []
    for i, row in enumerate(data_rows):
        print(f"Processing row {i+1}: {row}")
        
        if len(row) > resume_col_index:
            resume_url = row[resume_col_index]
            print(f"Resume URL for row {i+1}: '{resume_url}'")
            
            file_id = extract_drive_file_id(resume_url)
            print(f"Extracted file ID for row {i+1}: '{file_id}'")
            
            if file_id:
                file_ids.append(file_id)
        else:
            print(f"Row {i+1} doesn't have RESUME column data")
    
    print(f"Final file IDs list: {file_ids}")
    return file_ids

def download_pdf_from_drive(file_id, credentials_path):
    """Download a PDF file from Google Drive and return its bytes."""
    from googleapiclient.discovery import build
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = None
    try:
        from googleapiclient.http import MediaIoBaseDownload
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return fh.read()
    except Exception as e:
        print(f"Error downloading file {file_id}: {e}")
        return None

def extract_text_from_pdf_bytes(pdf_bytes):
    """Extract text from PDF bytes using LangChain's PyPDFLoader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        text = " ".join([doc.page_content for doc in docs])
    finally:
        os.remove(tmp_file_path)
    return text

def select_best_candidates(applicants, job_description, credentials_path):
    """
    For each applicant, download and parse the PDF resume, compare to job description,
    and return the top 2 candidates (by cosine similarity).
    """
    print(f"Starting candidate selection for {len(applicants)} applicants")
    resume_texts = []
    valid_applicants = []
    
    for i, applicant in enumerate(applicants):
        file_id = applicant['resume_file_id']
        print(f"Processing applicant {i+1}: {applicant['name']} (file ID: {file_id})")
        
        pdf_bytes = download_pdf_from_drive(file_id, credentials_path)
        if not pdf_bytes:
            print(f"Failed to download PDF for applicant: {applicant['name']}")
            continue
            
        try:
            print(f"Extracting text from PDF for applicant: {applicant['name']}")
            text = extract_text_from_pdf_bytes(pdf_bytes)
            print(f"Extracted text length: {len(text)} characters")
            
            if text.strip():
                resume_texts.append(text)
                valid_applicants.append(applicant)
                print(f"Successfully processed applicant: {applicant['name']}")
            else:
                print(f"Empty text extracted for applicant: {applicant['name']}")
        except Exception as e:
            print(f"Error parsing PDF for applicant {applicant['name']}: {e}")
            continue
    
    print(f"Successfully processed {len(valid_applicants)} applicants")
    
    if not resume_texts:
        print("No valid resumes found")
        return []
        
    if len(resume_texts) == 1:
        print("Only one applicant found, returning it")
        return [{'applicant': valid_applicants[0], 'similarity_score': 1.0}]
    
    try:
        print("Vectorizing resumes and job description")
        # Vectorize resumes and job description
        vectorizer = TfidfVectorizer().fit(resume_texts + [job_description])
        resume_vecs = vectorizer.transform(resume_texts)
        jd_vec = vectorizer.transform([job_description])
        scores = cosine_similarity(resume_vecs, jd_vec).flatten()
        
        print(f"Similarity scores: {scores}")
        
        # Get top 2 indices (or all if less than 2)
        num_to_return = min(2, len(scores))
        top_indices = scores.argsort()[-num_to_return:][::-1]
        
        top_candidates = []
        for idx in top_indices:
            top_candidates.append({
                'applicant': valid_applicants[idx],
                'similarity_score': float(scores[idx])
            })
            print(f"Top candidate: {valid_applicants[idx]['name']} with score: {scores[idx]}")
            
        return top_candidates
        
    except Exception as e:
        print(f"Error in vectorization/similarity calculation: {e}")
        # Fallback: return all valid applicants with default scores
        fallback_candidates = []
        for applicant in valid_applicants:
            fallback_candidates.append({
                'applicant': applicant,
                'similarity_score': 0.5
            })
        return fallback_candidates[:2] if len(fallback_candidates) >= 2 else fallback_candidates

def pick_best_resume(applicants):
    """Return the applicant with the highest similarity_score from a list."""
    if not applicants:
        return None
    best = max(applicants, key=lambda x: x.get('similarity_score', 0))
    return best

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
def schedule_interviews_node(state: HRHiringState) -> dict:
    """
    Schedule interviews for the top candidates.
    This node will be called after selecting the best candidates.
    """
    # Get the top candidates from the state
    top_candidates = state.get('top_candidates', [])
    
    if not top_candidates:
        return {'status': 'no_candidates_to_schedule'}
    
    # For now, return the candidates for manual scheduling
    # In a full implementation, you would integrate with Google Calendar API
    return {
        'status': 'candidates_ready_for_scheduling',
        'top_candidates': top_candidates
    }

builder = StateGraph(HRHiringState)
builder.add_node('generate_jd', generate_jd_node)
builder.add_node('human_approval', human_approval_node)
builder.add_node('post_linkedin', post_linkedin_node)
builder.add_node('fetch_applications', fetch_applications)
builder.add_node('select_best_resumes', select_best_candidates)
builder.add_node('schedule_interviews', schedule_interviews_node)

builder.add_edge(START, 'generate_jd')
builder.add_edge('generate_jd', 'human_approval')
builder.add_conditional_edges('human_approval', approval_conditional)
builder.add_edge('post_linkedin', 'fetch_applications')
builder.add_edge('fetch_applications', 'select_best_resumes')
builder.add_edge('select_best_resumes', 'schedule_interviews')
builder.add_edge('schedule_interviews', END)

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

@app.route('/fetch-applications')
def fetch_applications_api():
    """Fetch applications from Google Sheet using hardcoded IDs."""
    try:
        applicants = fetch_applications()
        return jsonify({'applicants': applicants})
    except Exception as e:
        return jsonify({'error': f'Failed to fetch applications: {str(e)}'}), 500

@app.route('/select-best-resumes', methods=['POST'])
def select_best_resumes_api():
    """Select best candidates using the provided job description."""
    data = request.get_json()
    if not data or 'job_description' not in data:
        return jsonify({'error': 'Job description is required'}), 400
    
    print(f"Job description length: {len(data['job_description'])} characters")
    
    try:
        # First get the applicants with resume data
        applicants = fetch_applicants_with_resumes()
        
        if not applicants:
            return jsonify({'error': 'No applicants found'}), 400
        
        print(f"Found {len(applicants)} applicants")
        
        # Then select the best candidates
        top_candidates = select_best_candidates(
            applicants, 
            data['job_description'], 
            CREDENTIALS_PATH
        )
        
        if not top_candidates:
            return jsonify({'error': 'No valid candidates found'}), 400
            
        # Pick the single best candidate
        best_candidate = max(top_candidates, key=lambda x: x.get('similarity_score', 0))
        
        print(f"Selected best candidate: {best_candidate['applicant']['name']} with score: {best_candidate['similarity_score']}")
        return jsonify({
            'best_candidate': best_candidate,
            'all_top_candidates': top_candidates  # Keep for debugging
        })
            
    except Exception as e:
        print(f"Error in select_best_resumes_api: {str(e)}")
        return jsonify({'error': f'Failed to select best candidates: {str(e)}'}), 500

@app.route('/test-sheet')
def test_sheet():
    """Test endpoint to see raw data from Google Sheet."""
    try:
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=GOOGLE_SHEET_ID, range='Form Responses 1').execute()
        values = result.get('values', [])
        
        return jsonify({
            'raw_data': values,
            'row_count': len(values),
            'headers': values[0] if values else [],
            'data_rows': values[1:] if len(values) > 1 else []
        })
    except Exception as e:
        return jsonify({'error': f'Failed to read sheet: {str(e)}'}), 500

@app.route('/fetch-resumes')
def fetch_resumes_api():
    """Fetch only resume file IDs from Google Sheet."""
    try:
        file_ids = fetch_resume_file_ids()
        return jsonify({'resume_file_ids': file_ids})
    except Exception as e:
        return jsonify({'error': f'Failed to fetch resumes: {str(e)}'}), 500

@app.route('/view-resume/<file_id>')
def view_resume_api(file_id):
    """Serve a PDF resume file for viewing."""
    try:
        pdf_bytes = download_pdf_from_drive(file_id, CREDENTIALS_PATH)
        if not pdf_bytes:
            return jsonify({'error': 'Resume not found'}), 404
        
        # Return the PDF as a file response
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=False,
            download_name=f'resume_{file_id}.pdf'
        )
    except Exception as e:
        return jsonify({'error': f'Failed to load resume: {str(e)}'}), 500

def create_calendar_event(candidate_email, candidate_name, interview_date, interview_time, interviewer_email=None):
    """
    Create a Google Calendar event for the interview using personal Google Account.
    This can send actual email invites to candidates.
    """
    try:
        # Get personal Google Account credentials
        creds = get_google_calendar_credentials()
        
        if isinstance(creds, dict) and 'error' in creds:
            return creds
        
        # Build the Calendar service
        service = build('calendar', 'v3', credentials=creds)
        
        # Parse date and time
        from datetime import datetime, timedelta
        try:
            # Combine date and time
            datetime_str = f"{interview_date} {interview_time}"
            start_time = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            end_time = start_time + timedelta(hours=1)  # 1-hour interview
            
            # Format for Google Calendar API
            start_time_str = start_time.isoformat() + 'Z'
            end_time_str = end_time.isoformat() + 'Z'
        except ValueError as e:
            return {'error': f'Invalid date/time format: {str(e)}'}
        
        # Create event with attendees to send email invites
        event = {
            'summary': f'Interview with {candidate_name}',
            'description': f'Interview for the position.\n\nCandidate: {candidate_name}\nEmail: {candidate_email}',
            'start': {
                'dateTime': start_time_str,
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_time_str,
                'timeZone': 'UTC',
            },
            'attendees': [
                {'email': candidate_email},
            ],
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},  # 1 day before
                    {'method': 'popup', 'minutes': 30},  # 30 minutes before
                ],
            },
        }
        
        # Add interviewer if provided
        if interviewer_email:
            event['attendees'].append({'email': interviewer_email})
        
        # Insert the event and send email invites
        event = service.events().insert(
            calendarId='primary',  # Use primary calendar
            body=event,
            sendUpdates='all'  # Send email notifications to all attendees
        ).execute()
        
        return {
            'status': 'success',
            'event_id': event.get('id'),
            'event_link': event.get('htmlLink'),
            'message': f'Calendar event created successfully for {candidate_name}. Email invite sent to {candidate_email}.'
        }
        
    except Exception as e:
        return {'error': f'Failed to create calendar event: {str(e)}'}

@app.route('/schedule-interview', methods=['POST'])
def schedule_interview_api():
    """Schedule a calendar invite for a candidate."""
    data = request.get_json()
    if not data or 'candidate_email' not in data or 'candidate_name' not in data:
        return jsonify({'error': 'Candidate email and name are required'}), 400
    
    candidate_email = data['candidate_email']
    candidate_name = data['candidate_name']
    interview_date = data.get('interview_date', '')
    interview_time = data.get('interview_time', '')
    interviewer_email = data.get('interviewer_email', '')  # Optional
    
    try:
        # Create the calendar event
        result = create_calendar_event(
            candidate_email, 
            candidate_name, 
            interview_date, 
            interview_time,
            interviewer_email
        )
        
        if result.get('status') == 'success':
            return jsonify({
                'status': 'success',
                'message': result['message'],
                'event_id': result.get('event_id'),
                'event_link': result.get('event_link'),
                'candidate_name': candidate_name,
                'candidate_email': candidate_email,
                'interview_date': interview_date,
                'interview_time': interview_time
            })
        else:
            return jsonify({'error': result.get('error', 'Unknown error')}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to schedule interview: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)




"""WITH MCP SERVER"""
    


# import os
# from flask import Flask, request, jsonify, send_file, redirect, url_for, session
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START, END
# from typing import TypedDict, Optional
# from langchain_groq import ChatGroq
# import secrets
# import requests
# from langchain.prompts import PromptTemplate
# import re
# from google.oauth2 import service_account
# from googleapiclient.discovery import build
# import io
# from langchain_community.document_loaders import PyPDFLoader
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import tempfile
# import asyncio
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
# from observee_agents import call_mcpauth_login, get_available_servers, chat_with_tools
# import json

# load_dotenv(override=True)

# app = Flask(__name__)
# app.secret_key = os.urandom(24)

# # LinkedIn OAuth settings
# CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
# CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
# REDIRECT_URI = "http://127.0.0.1:5000/callback"
# AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
# TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"

# # Google Sheets and Drive settings
# GOOGLE_SHEET_ID = "1PrMvW7un7b1zdF_262QwtaP7_xNhf5zxylYIqnOSq_U"
# DRIVE_FOLDER_ID = "1lFh4bkL2PYtyjFcVMiw0SuVEEeP3hXMNuCH4zQgGb8UOyLO1fxs528Gq5jxV8z4lsH3h-9KF"
# CREDENTIALS_PATH = "credentials.json"

# # Observee MCP settings
# OBSERVEE_CLIENT_ID = "0a2bb94e-b62c-4158-881d-57de87bfadcf"
# OBSERVEE_API_KEY = "obs_hyFy_2HNIAnDsRfK-EKhFP8sUzNZ3dO8uk_r6qRnLuQ"

# user_tokens = {}
# user_observee_client_ids = {}  

# async def get_observee_client():
#     """Get Observee MCP client for Google Calendar operations."""
#     return MultiServerMCPClient(
#         {
#             "observee": {
#                 "transport": "streamable_http",
#                 "url": f"https://mcp.observee.ai/mcp?client_id={OBSERVEE_CLIENT_ID}",
#                 "headers": {
#                     "Authorization": f"Bearer {OBSERVEE_API_KEY}"
#                 },
#             }
#         }
#     )

# def schedule_interview_with_observee(candidate_email, candidate_name, interview_date, interview_time, interviewer_email=None, user_id=None):
#     """
#     Schedule interview using Observee MCP Auth SDK for user-specific calendar access.
#     This path requires the user to have completed the Connect Calendar flow.
#     """
#     try:
#         # Verify calendar auth is available for this user
#         if not (user_id and user_id in user_observee_client_ids):
#             return {
#                 'status': 'error',
#                 'error': 'Please connect your calendar first. Click "Connect Calendar" to authenticate.'
#             }

#         client_id = user_observee_client_ids[user_id]

#         # Provide strict instructions to the MCP agent to actually create the event
#         # and respond with a parseable JSON payload.
#         attendees_line = candidate_email if not interviewer_email else f"{candidate_email}, {interviewer_email}"
#         scheduling_message = (
#             "You are connected to Observee MCP with Gmail/Google Calendar tools. "
#             "Create a REAL Google Calendar event in the authenticated user's primary calendar. "
#             "ONLY use the calendar tool; do not simulate. "
#             "Details:"\
#             f" Event Title: Interview with {candidate_name}."\
#             f" Date: {interview_date}."\
#             f" Start Time: {interview_time}."\
#             " Duration: 60 minutes."\
#             f" Attendees: {attendees_line}."\
#             f" Description: Interview for the position. Candidate: {candidate_name}, Email: {candidate_email}. "
#             "Return ONLY compact JSON with keys: status (success|error), event_id, event_link, message."
#         )

#         result = chat_with_tools(
#             message=scheduling_message,
#             provider="groq",
#             model="openai/gpt-oss-120b",
#             client_id=client_id
#         )

#         # result = chat_with_tools(
#         #     provider="groq",
#         #     model="openai/gpt-oss-120b",
#         #     enable_filtering=True,
#         #     expand_by_server=True,
#         #     client_id=client_id,
#         #     observee_api_key=os.getenv("OBSERVEE_API_KEY")
#         # )

#         # Normalize result to string for parsing
#         result_text = result if isinstance(result, str) else str(result)

#         # Try to parse a JSON object from the result
#         parsed = None
#         try:
#             parsed = json.loads(result_text)
#         except Exception:
#             try:
#                 import re as _re
#                 match = _re.search(r"\{[\s\S]*\}", result_text)
#                 if match:
#                     parsed = json.loads(match.group(0))
#             except Exception:
#                 parsed = None

#         # Determine success only if we have concrete event identifiers
#         if isinstance(parsed, dict):
#             event_id = parsed.get('event_id')
#             event_link = parsed.get('event_link') or parsed.get('htmlLink')
#             status_flag = parsed.get('status')

#             if status_flag == 'success' and (event_id or event_link):
#                 return {
#                     'status': 'success',
#                     'message': parsed.get('message') or f'Interview scheduled successfully for {candidate_name}.',
#                     'details': parsed,
#                 }

#         # If we reach here, we did not get a definitive success response
#         return {
#             'status': 'error',
#             'error': 'Calendar event could not be confirmed. Please ensure calendar is connected and try again.',
#             'details': result_text
#         }

#     except Exception as e:
#         return {
#             'status': 'error',
#             'error': f'Failed to schedule interview: {str(e)}'
#         }

# class HRHiringState(TypedDict):
#     role: Optional[str]
#     job_description: Optional[str]
#     approval: Optional[bool]

# # --- LangGraph Workflow Setup ---
# def generate_jd_node(state: HRHiringState) -> dict:
#     """Generate job description using Groq LLM."""
#     role = state.get('role', 'Software Engineer')
#     llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")
    
#     prompt = PromptTemplate(
#         input_variables=["role"],
#         template="Generate a comprehensive job description for a {role} position. Include requirements, responsibilities, and qualifications under 1200 characters"
#     )
    
#     chain = prompt | llm
#     result = chain.invoke({"role": role})
    
#     return {"job_description": result.content}

# def human_approval_node(state: HRHiringState) -> dict:
#     """Human approval node - returns current state."""
#     return state

# # Restored chunking function for LinkedIn's 1300-character limit
# def split_text_for_post(text, max_length=1300):
#     """Split text into chunks suitable for social media posts."""
#     words = text.split()
#     chunks = []
#     current_chunk = []
#     current_length = 0
    
#     for word in words:
#         if current_length + len(word) + 1 <= max_length:
#             current_chunk.append(word)
#             current_length += len(word) + 1
#         else:
#             if current_chunk:
#                 chunks.append(' '.join(current_chunk))
#             current_chunk = [word]
#             current_length = len(word)
    
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))
    
#     return chunks

# # Role to application link mapping
# ROLE_FORM_LINKS = {
#     "data": "https://forms.gle/8R5Eg8TaQHgQBFE19",
#     "software": "https://forms.gle/8R5Eg8TaQHgQBFE19",
#     "developer": "https://forms.gle/8R5Eg8TaQHgQBFE19",
#     "engineer": "https://forms.gle/8R5Eg8TaQHgQBFE19"
# }
# GENERIC_FORM_LINK = "https://forms.gle/8R5Eg8TaQHgQBFE19"

# def get_form_link_for_role(role: str) -> str:
#     role_lower = role.lower()
#     for keyword, link in ROLE_FORM_LINKS.items():
#         if keyword in role_lower:
#             return link
#     return GENERIC_FORM_LINK

# def extract_drive_file_id(drive_url):
#     """Extract file ID from Google Drive share link."""
#     if not drive_url:
#         return None
    
#     patterns = [
#         r'/d/([\w-]+)',
#         r'id=([\w-]+)',
#         r'/file/d/([\w-]+)',
#         r'([\w-]{25,})'
#     ]
    
#     for pattern in patterns:
#         match = re.search(pattern, drive_url)
#         if match:
#             return match.group(1)
    
#     return None

# def fetch_applications(sheet_id=GOOGLE_SHEET_ID, credentials_path=CREDENTIALS_PATH, range_name='Form Responses 1'):
#     """
#     Reads applicant data from a Google Sheet and extracts Drive file IDs for resumes.
#     Skips the first column (timestamp).
#     Returns a list of dicts: {name, college, email, intro, resume_file_id}
#     """
#     SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
#     creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
#     service = build('sheets', 'v4', credentials=creds)
#     sheet = service.spreadsheets()
#     result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
#     values = result.get('values', [])
    
#     print(f"Raw sheet data: {values}")
    
#     if not values or len(values) < 2:
#         print("No data found in sheet")
#         return []
        
#     headers = values[0][1:]  # Skip timestamp
#     data_rows = values[1:]
    
#     print(f"Headers: {headers}")
#     print(f"Data rows: {data_rows}")
    
#     applicants = []
#     for i, row in enumerate(data_rows):
#         print(f"Processing row {i+1}: {row}")
        
#         if len(row) < len(headers) + 1:
#             print(f"Row {i+1} is incomplete, skipping")
#             continue  # skip incomplete rows
            
#         row_dict = dict(zip(headers, row[1:]))  # Skip timestamp in each row
#         print(f"Row {i+1} dict: {row_dict}")
        
#         resume_url = row_dict.get('RESUME ', '')  # Note the trailing space to match the actual header
#         print(f"Resume URL for row {i+1}: '{resume_url}'")
        
#         file_id = extract_drive_file_id(resume_url)
#         print(f"Extracted file ID for row {i+1}: '{file_id}'")
        
#         applicant = {
#             'name': row_dict.get('Name', ''),
#             'college': row_dict.get('College Name and Course', ''),
#             'email': row_dict.get('Email', ''),
#             'intro': row_dict.get('Short Intro about you', ''),
#             'resume_file_id': file_id
#         }
        
#         print(f"Created applicant: {applicant}")
#         applicants.append(applicant)
    
#     print(f"Final applicants list: {applicants}")
#     return applicants

# def fetch_applicants_with_resumes(sheet_id=GOOGLE_SHEET_ID, credentials_path=CREDENTIALS_PATH, range_name='Form Responses 1'):
#     """
#     Reads applicant data from Google Sheet and extracts Drive file IDs for resumes.
#     Returns a list of dicts with complete applicant info: {name, college, email, intro, resume_file_id}
#     """
#     SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
#     creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
#     service = build('sheets', 'v4', credentials=creds)
#     sheet = service.spreadsheets()
#     result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
#     values = result.get('values', [])
    
#     print(f"Raw sheet data: {values}")
    
#     if not values or len(values) < 2:
#         print("No data found in sheet")
#         return []
        
#     headers = values[0]
#     data_rows = values[1:]
    
#     print(f"Headers: {headers}")
#     print(f"Data rows: {data_rows}")
    
#     # Find the RESUME column index (handle trailing spaces)
#     resume_col_index = None
#     for i, header in enumerate(headers):
#         if header.strip() == 'RESUME':
#             resume_col_index = i
#             break
    
#     if resume_col_index is None:
#         print("RESUME column not found!")
#         print(f"Available headers: {[h.strip() for h in headers]}")
#         return []
    
#     print(f"RESUME column found at index: {resume_col_index}")
    
#     applicants = []
#     for i, row in enumerate(data_rows):
#         print(f"Processing row {i+1}: {row}")
        
#         if len(row) > resume_col_index:
#             resume_url = row[resume_col_index]
#             print(f"Resume URL for row {i+1}: '{resume_url}'")
            
#             file_id = extract_drive_file_id(resume_url)
#             print(f"Extracted file ID for row {i+1}: '{file_id}'")
            
#             if file_id:
#                 # Create applicant dict with all available data
#                 applicant = {
#                     'name': row[1] if len(row) > 1 else '',
#                     'college': row[2] if len(row) > 2 else '',
#                     'email': row[3] if len(row) > 3 else '',
#                     'intro': row[4] if len(row) > 4 else '',
#                     'resume_file_id': file_id
#                 }
#                 applicants.append(applicant)
#         else:
#             print(f"Row {i+1} doesn't have RESUME column data")
    
#     print(f"Final applicants list: {applicants}")
#     return applicants

# def fetch_resume_file_ids(sheet_id=GOOGLE_SHEET_ID, credentials_path=CREDENTIALS_PATH, range_name='Form Responses 1'):
#     """
#     Reads only the RESUME column from Google Sheet and extracts Drive file IDs.
#     Returns a list of file IDs for resumes.
#     """
#     SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
#     creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
#     service = build('sheets', 'v4', credentials=creds)
#     sheet = service.spreadsheets()
#     result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
#     values = result.get('values', [])
    
#     print(f"Raw sheet data: {values}")
    
#     if not values or len(values) < 2:
#         print("No data found in sheet")
#         return []
        
#     headers = values[0]
#     data_rows = values[1:]
    
#     print(f"Headers: {headers}")
#     print(f"Data rows: {data_rows}")
    
#     # Find the RESUME column index (handle trailing spaces)
#     resume_col_index = None
#     for i, header in enumerate(headers):
#         if header.strip() == 'RESUME':
#             resume_col_index = i
#             break
    
#     if resume_col_index is None:
#         print("RESUME column not found!")
#         print(f"Available headers: {[h.strip() for h in headers]}")
#         return []
    
#     print(f"RESUME column found at index: {resume_col_index}")
    
#     file_ids = []
#     for i, row in enumerate(data_rows):
#         print(f"Processing row {i+1}: {row}")
        
#         if len(row) > resume_col_index:
#             resume_url = row[resume_col_index]
#             print(f"Resume URL for row {i+1}: '{resume_url}'")
            
#             file_id = extract_drive_file_id(resume_url)
#             print(f"Extracted file ID for row {i+1}: '{file_id}'")
            
#             if file_id:
#                 file_ids.append(file_id)
#         else:
#             print(f"Row {i+1} doesn't have RESUME column data")
    
#     print(f"Final file IDs list: {file_ids}")
#     return file_ids

# def download_pdf_from_drive(file_id, credentials_path):
#     """Download PDF from Google Drive and return bytes."""
#     creds = service_account.Credentials.from_service_account_file(
#         credentials_path, 
#         scopes=["https://www.googleapis.com/auth/drive.readonly"]
#     )
#     service = build('drive', 'v3', credentials=creds)
#     request = service.files().get_media(fileId=file_id)
#     fh = io.BytesIO()
    
#     try:
#         from googleapiclient.http import MediaIoBaseDownload
#         downloader = MediaIoBaseDownload(fh, request)
#         done = False
#         while not done:
#             status, done = downloader.next_chunk()
#         fh.seek(0)
#         return fh.read()
#     except Exception as e:
#         return None

# def extract_text_from_pdf_bytes(pdf_bytes):
#     """Extract text from PDF bytes using LangChain."""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(pdf_bytes)
#         tmp_file_path = tmp_file.name

#     try:
#         loader = PyPDFLoader(tmp_file_path)
#         docs = loader.load()
#         text = " ".join([doc.page_content for doc in docs])
#     finally:
#         os.remove(tmp_file_path)
#     return text

# def select_best_candidates(applicants, job_description, credentials_path):
#     """
#     For each applicant, download and parse the PDF resume, compare to job description,
#     and return the top 2 candidates (by cosine similarity).
#     """
#     print(f"Starting candidate selection for {len(applicants)} applicants")
#     resume_texts = []
#     valid_applicants = []
    
#     for i, applicant in enumerate(applicants):
#         file_id = applicant['resume_file_id']
#         print(f"Processing applicant {i+1}: {applicant['name']} (file ID: {file_id})")
        
#         pdf_bytes = download_pdf_from_drive(file_id, credentials_path)
#         if not pdf_bytes:
#             print(f"Failed to download PDF for applicant: {applicant['name']}")
#             continue
            
#         try:
#             print(f"Extracting text from PDF for applicant: {applicant['name']}")
#             text = extract_text_from_pdf_bytes(pdf_bytes)
#             print(f"Extracted text length: {len(text)} characters")
            
#             if text.strip():
#                 resume_texts.append(text)
#                 valid_applicants.append(applicant)
#                 print(f"Successfully processed applicant: {applicant['name']}")
#             else:
#                 print(f"Empty text extracted for applicant: {applicant['name']}")
#         except Exception as e:
#             print(f"Error parsing PDF for applicant {applicant['name']}: {e}")
#             continue
    
#     print(f"Successfully processed {len(valid_applicants)} applicants")
    
#     if not resume_texts:
#         print("No valid resumes found")
#         return []
        
#     if len(resume_texts) == 1:
#         print("Only one applicant found, returning it")
#         return [{'applicant': valid_applicants[0], 'similarity_score': 1.0}]
    
#     try:
#         print("Vectorizing resumes and job description")
#         # Vectorize resumes and job description
#         vectorizer = TfidfVectorizer().fit(resume_texts + [job_description])
#         resume_vecs = vectorizer.transform(resume_texts)
#         jd_vec = vectorizer.transform([job_description])
#         scores = cosine_similarity(resume_vecs, jd_vec).flatten()
        
#         print(f"Similarity scores: {scores}")
        
#         # Get top 2 indices (or all if less than 2)
#         num_to_return = min(2, len(scores))
#         top_indices = scores.argsort()[-num_to_return:][::-1]
        
#         top_candidates = []
#         for idx in top_indices:
#             top_candidates.append({
#                 'applicant': valid_applicants[idx],
#                 'similarity_score': float(scores[idx])
#             })
#             print(f"Top candidate: {valid_applicants[idx]['name']} with score: {scores[idx]}")
            
#         return top_candidates
        
#     except Exception as e:
#         print(f"Error in vectorization/similarity calculation: {e}")
#         # Fallback: return all valid applicants with default scores
#         fallback_candidates = []
#         for applicant in valid_applicants:
#             fallback_candidates.append({
#                 'applicant': applicant,
#                 'similarity_score': 0.5
#             })
#         return fallback_candidates[:2] if len(fallback_candidates) >= 2 else fallback_candidates

# def pick_best_resume(applicants):
#     """Return the applicant with the highest similarity_score from a list."""
#     if not applicants:
#         return None
#     best = max(applicants, key=lambda x: x.get('similarity_score', 0))
#     return best

# # LinkedIn posting node
# def post_linkedin_node(state: HRHiringState) -> dict:
#     jd = state.get('job_description')
#     role = state.get('role', '')
#     user_id = session.get('user_id')
#     if not jd or not user_id or user_id not in user_tokens:
#         return {'post_status': 'error', 'error': 'Missing JD or not authenticated'}

#     # Append the application link
#     form_link = get_form_link_for_role(role)
#     jd_with_link = f"{jd.strip()}\n\nApply here: {form_link}"

#     # Split into chunks (though typically one post for LinkedIn)
#     chunks = split_text_for_post(jd_with_link)
#     if not chunks:
#         return {'post_status': 'error', 'error': 'No content to post'}

#     access_token = user_tokens[user_id]['access_token']
#     headers = {
#         'Authorization': f'Bearer {access_token}',
#         'Content-Type': 'application/json',
#         'X-Restli-Protocol-Version': '2.0.0'
#     }

#     # Get user ID from OpenID Connect ID token or fallback to profile API
#     user_id = session.get('user_id')
#     person_urn = None
    
#     if user_id in user_tokens and user_tokens[user_id].get('id_token'):
#         # Try to extract user ID from ID token (simplified approach)
#         # In production, you should properly decode and validate the JWT
#         try:
#             # For now, we'll use the profile API as fallback
#             profile_response = requests.get('https://api.linkedin.com/v2/userinfo', headers=headers)
#             if profile_response.status_code == 200:
#                 profile_data = profile_response.json()
#                 person_urn = profile_data.get('sub')  # OpenID Connect subject identifier
#         except:
#             pass
    
#     # Fallback to profile API if OpenID Connect approach fails
#     if not person_urn:
#         profile_response = requests.get('https://api.linkedin.com/v2/me', headers=headers)
#         if profile_response.status_code != 200:
#             return {'post_status': 'error', 'error': f'Failed to get user profile: {profile_response.text}'}
#         profile_data = profile_response.json()
#         person_urn = profile_data.get('id')
    
#     if not person_urn:
#         return {'post_status': 'error', 'error': 'Could not retrieve user URN'}

#     post_ids = []
#     for idx, chunk in enumerate(chunks):
#         payload = {
#             'author': f'urn:li:person:{person_urn}',
#             'lifecycleState': 'PUBLISHED',
#             'specificContent': {
#                 'com.linkedin.ugc.ShareContent': {
#                     'shareCommentary': {'text': chunk},
#                     'shareMediaCategory': 'NONE'
#                 }
#             },
#             'visibility': {'com.linkedin.ugc.MemberNetworkVisibility': 'PUBLIC'}
#         }
#         print(f"Posting chunk {idx+1}/{len(chunks)} with person_urn: {person_urn}")
#         print(f"Payload: {payload}")
#         try:
#             response = requests.post(
#                 'https://api.linkedin.com/v2/ugcPosts',
#                 json=payload,
#                 headers=headers
#             )
#             print(f"Response status: {response.status_code}")
#             print(f"Response headers: {dict(response.headers)}")
#             print(f"Response text: {response.text}")
#             if response.status_code in [200, 201]:
#                 post_id = response.headers.get('X-RestLi-Id')
#                 post_url = f'https://www.linkedin.com/feed/update/{post_id}' if post_id else None
#                 post_ids.append({'id': post_id, 'url': post_url})
#                 print(f"Successfully posted with ID: {post_id}")
#             else:
#                 return {'post_status': 'error', 'error': f'Post failed: {response.text}', 'post_ids': post_ids}
#         except Exception as e:
#             print(f"Exception during posting: {str(e)}")
#             return {'post_status': 'error', 'error': f'Post request failed: {str(e)}', 'post_ids': post_ids}
    
#     return {'post_status': 'success', 'post_ids': post_ids}

# def approval_conditional(state: HRHiringState) -> str:
#     """Conditional routing based on approval."""
#     return "approved" if state.get('approval') else "rejected"

# # Update StateGraph
# def schedule_interviews_node(state: HRHiringState) -> dict:
#     """
#     Schedule interviews for the top candidates.
#     This node will be called after selecting the best candidates.
#     """
#     # Get the top candidates from the state
#     top_candidates = state.get('top_candidates', [])
    
#     if not top_candidates:
#         return {'status': 'no_candidates_to_schedule'}
    
#     # For now, return the candidates for manual scheduling
#     # In a full implementation, you would integrate with Google Calendar API
#     return {
#         'status': 'candidates_ready_for_scheduling',
#         'top_candidates': top_candidates
#     }

# builder = StateGraph(HRHiringState)
# builder.add_node('generate_jd', generate_jd_node)
# builder.add_node('human_approval', human_approval_node)
# builder.add_node('post_linkedin', post_linkedin_node)
# builder.add_node('fetch_applications', fetch_applications)
# builder.add_node('select_best_resumes', select_best_candidates)
# builder.add_node('schedule_interviews', schedule_interviews_node)

# builder.add_edge(START, 'generate_jd')
# builder.add_edge('generate_jd', 'human_approval')
# builder.add_conditional_edges('human_approval', approval_conditional)
# builder.add_edge('post_linkedin', 'fetch_applications')
# builder.add_edge('fetch_applications', 'select_best_resumes')
# builder.add_edge('select_best_resumes', 'schedule_interviews')
# builder.add_edge('schedule_interviews', END)

# graph = builder.compile()

# # --- Flask Endpoints ---
# @app.route('/')
# def landing_page():
#     return send_file('index.html')

# @app.route('/login')
# def login():
#     if not CLIENT_ID or not CLIENT_SECRET:
#         return jsonify({'error': 'LinkedIn Client ID or Secret not set'}), 500
#     state = secrets.token_urlsafe(16)
#     session['state'] = state
#     session['user_id'] = secrets.token_urlsafe(16)
#     auth_params = {
#         'response_type': 'code',
#         'client_id': CLIENT_ID,
#         'redirect_uri': REDIRECT_URI,
#         'state': state,
#         'scope': 'openid profile w_member_social'  # Use OpenID Connect scopes
#     }
#     auth_url = f"{AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in auth_params.items())}"
#     return redirect(auth_url)

# @app.route('/callback')
# def callback():
#     code = request.args.get('code')
#     state = request.args.get('state')
#     error = request.args.get('error')
#     error_description = request.args.get('error_description')
#     saved_state = session.get('state')
#     if error:
#         return jsonify({'error': f'OAuth error: {error}', 'description': error_description}), 400
#     if not code or state != saved_state:
#         return jsonify({'error': 'Invalid login attempt'}), 400
#     token_data = {
#         'grant_type': 'authorization_code',
#         'code': code,
#         'client_id': CLIENT_ID,
#         'client_secret': CLIENT_SECRET,
#         'redirect_uri': REDIRECT_URI
#     }
#     try:
#         headers = {'Content-Type': 'application/x-www-form-urlencoded'}
#         response = requests.post(TOKEN_URL, data=token_data, headers=headers)
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to get token', 'details': response.text}), 500
#         token_response = response.json()
        
#         # Store both access token and ID token
#         user_tokens[session['user_id']] = {
#             'access_token': token_response['access_token'],
#             'id_token': token_response.get('id_token')  # OpenID Connect ID token
#         }
#         return redirect(url_for('landing_page'))
#     except Exception as e:
#         return jsonify({'error': f'Authentication failed: {str(e)}'}), 500

# @app.route('/logout')
# def logout():
#     user_id = session.get('user_id')
#     if user_id and user_id in user_tokens:
#         del user_tokens[user_id]
#     session.clear()
#     return redirect(url_for('landing_page'))

# @app.route('/check-auth')
# def check_auth():
#     user_id = session.get('user_id')
#     is_authenticated = user_id is not None and user_id in user_tokens
#     if is_authenticated:
#         access_token = user_tokens[user_id]['access_token']
#         headers = {
#             'Authorization': f'Bearer {access_token}',
#             'Content-Type': 'application/json',
#             'X-Restli-Protocol-Version': '2.0.0'
#         }
#         try:
#             # Try OpenID Connect userinfo endpoint first
#             test_response = requests.get('https://api.linkedin.com/v2/userinfo', headers=headers)
#             if test_response.status_code != 200:
#                 # Fallback to profile API
#                 test_response = requests.get('https://api.linkedin.com/v2/me', headers=headers)
            
#             token_valid = test_response.status_code == 200
#             return jsonify({
#                 'authenticated': is_authenticated,
#                 'token_valid': token_valid,
#                 'profile_status': test_response.status_code,
#                 'profile_response': test_response.text[:200] if not token_valid else 'Profile accessible'
#             })
#         except Exception as e:
#             return jsonify({
#                 'authenticated': is_authenticated,
#                 'token_valid': False,
#                 'error': str(e)
#             })
#     return jsonify({'authenticated': is_authenticated})

# @app.route('/generate-jd', methods=['POST'])
# def generate_jd_api():
#     data = request.get_json()
#     if not data or 'role' not in data:
#         return jsonify({'error': 'Role is required'}), 400
#     state = {'role': data.get('role')}
#     result = generate_jd_node(state)
#     return jsonify(result)

# @app.route('/approve-jd', methods=['POST'])
# def approve_jd_api():
#     data = request.get_json()
#     if not data or 'role' not in data or 'job_description' not in data:
#         return jsonify({'error': 'Role and job_description are required'}), 400
#     state = {
#         'role': data.get('role'),
#         'job_description': data.get('job_description'),
#         'approval': data.get('approval', False)
#     }
#     result = human_approval_node(state)
#     if not data.get('approval'):
#         result = generate_jd_node({'role': data.get('role')})
#     return jsonify(result)

# @app.route('/post-jd', methods=['POST'])
# def post_jd_api():
#     data = request.get_json()
#     if not data or 'role' not in data or 'job_description' not in data:
#         return jsonify({'error': 'Role and job_description are required'}), 400
#     state = {
#         'role': data.get('role'),
#         'job_description': data.get('job_description'),
#         'approval': True
#     }
#     result = post_linkedin_node(state)
#     return jsonify(result)

# @app.route('/fetch-applications')
# def fetch_applications_api():
#     """Fetch applications from Google Sheet using hardcoded IDs."""
#     try:
#         applicants = fetch_applications()
#         return jsonify({'applicants': applicants})
#     except Exception as e:
#         return jsonify({'error': f'Failed to fetch applications: {str(e)}'}), 500

# @app.route('/select-best-resumes', methods=['POST'])
# def select_best_resumes_api():
#     """Select best candidates using the provided job description."""
#     data = request.get_json()
#     if not data or 'job_description' not in data:
#         return jsonify({'error': 'Job description is required'}), 400
    
#     print(f"Job description length: {len(data['job_description'])} characters")
    
#     try:
#         # First get the applicants with resume data
#         applicants = fetch_applicants_with_resumes()
        
#         if not applicants:
#             return jsonify({'error': 'No applicants found'}), 400
        
#         print(f"Found {len(applicants)} applicants")
        
#         # Then select the best candidates
#         top_candidates = select_best_candidates(
#             applicants, 
#             data['job_description'], 
#             CREDENTIALS_PATH
#         )
        
#         if not top_candidates:
#             return jsonify({'error': 'No valid candidates found'}), 400
            
#         # Pick the single best candidate
#         best_candidate = max(top_candidates, key=lambda x: x.get('similarity_score', 0))
        
#         print(f"Selected best candidate: {best_candidate['applicant']['name']} with score: {best_candidate['similarity_score']}")
#         return jsonify({
#             'best_candidate': best_candidate,
#             'all_top_candidates': top_candidates  # Keep for debugging
#         })
            
#     except Exception as e:
#         print(f"Error in select_best_resumes_api: {str(e)}")
#         return jsonify({'error': f'Failed to select best candidates: {str(e)}'}), 500

# @app.route('/test-sheet')
# def test_sheet():
#     """Test endpoint to see raw data from Google Sheet."""
#     try:
#         SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
#         creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
#         service = build('sheets', 'v4', credentials=creds)
#         sheet = service.spreadsheets()
#         result = sheet.values().get(spreadsheetId=GOOGLE_SHEET_ID, range='Form Responses 1').execute()
#         values = result.get('values', [])
        
#         return jsonify({
#             'raw_data': values,
#             'row_count': len(values),
#             'headers': values[0] if values else [],
#             'data_rows': values[1:] if len(values) > 1 else []
#         })
#     except Exception as e:
#         return jsonify({'error': f'Failed to read sheet: {str(e)}'}), 500

# @app.route('/fetch-resumes')
# def fetch_resumes_api():
#     """Fetch only resume file IDs from Google Sheet."""
#     try:
#         file_ids = fetch_resume_file_ids()
#         return jsonify({'resume_file_ids': file_ids})
#     except Exception as e:
#         return jsonify({'error': f'Failed to fetch resumes: {str(e)}'}), 500

# @app.route('/view-resume/<file_id>')
# def view_resume_api(file_id):
#     """Serve a PDF resume file for viewing."""
#     try:
#         pdf_bytes = download_pdf_from_drive(file_id, CREDENTIALS_PATH)
#         if not pdf_bytes:
#             return jsonify({'error': 'Resume not found'}), 404
        
#         # Return the PDF as a file response
#         return send_file(
#             io.BytesIO(pdf_bytes),
#             mimetype='application/pdf',
#             as_attachment=False,
#             download_name=f'resume_{file_id}.pdf'
#         )
#     except Exception as e:
#         return jsonify({'error': f'Failed to load resume: {str(e)}'}), 500

# def create_calendar_event(candidate_email, candidate_name, interview_date, interview_time, interviewer_email=None):
#     """
#     Create a Google Calendar event for the interview using personal Google Account.
#     This can send actual email invites to candidates.
#     """
#     try:
#         # Get personal Google Account credentials
#         creds = get_google_calendar_credentials()
        
#         if isinstance(creds, dict) and 'error' in creds:
#             return creds
        
#         # Build the Calendar service
#         service = build('calendar', 'v3', credentials=creds)
        
#         # Parse date and time
#         from datetime import datetime, timedelta
#         try:
#             # Combine date and time
#             datetime_str = f"{interview_date} {interview_time}"
#             start_time = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
#             end_time = start_time + timedelta(hours=1)  # 1-hour interview
            
#             # Format for Google Calendar API
#             start_time_str = start_time.isoformat() + 'Z'
#             end_time_str = end_time.isoformat() + 'Z'
#         except ValueError as e:
#             return {'error': f'Invalid date/time format: {str(e)}'}
        
#         # Create event with attendees to send email invites
#         event = {
#             'summary': f'Interview with {candidate_name}',
#             'description': f'Interview for the position.\n\nCandidate: {candidate_name}\nEmail: {candidate_email}',
#             'start': {
#                 'dateTime': start_time_str,
#                 'timeZone': 'UTC',
#             },
#             'end': {
#                 'dateTime': end_time_str,
#                 'timeZone': 'UTC',
#             },
#             'attendees': [
#                 {'email': candidate_email},
#             ],
#             'reminders': {
#                 'useDefault': False,
#                 'overrides': [
#                     {'method': 'email', 'minutes': 24 * 60},  # 1 day before
#                     {'method': 'popup', 'minutes': 30},  # 30 minutes before
#                 ],
#             },
#         }
        
#         # Add interviewer if provided
#         if interviewer_email:
#             event['attendees'].append({'email': interviewer_email})
        
#         # Insert the event and send email invites
#         event = service.events().insert(
#             calendarId='primary',  # Use primary calendar
#             body=event,
#             sendUpdates='all'  # Send email notifications to all attendees
#         ).execute()
        
#         return {
#             'status': 'success',
#             'event_id': event.get('id'),
#             'event_link': event.get('htmlLink'),
#             'message': f'Calendar event created successfully for {candidate_name}. Email invite sent to {candidate_email}.'
#         }
        
#     except Exception as e:
#         return {'error': f'Failed to create calendar event: {str(e)}'}

# @app.route('/schedule-interview', methods=['POST'])
# def schedule_interview_api():
#     """Schedule a calendar invite for a candidate."""
#     data = request.get_json()
#     if not data or 'candidate_email' not in data or 'candidate_name' not in data:
#         return jsonify({'error': 'Candidate email and name are required'}), 400
    
#     candidate_email = data['candidate_email']
#     candidate_name = data['candidate_name']
#     interview_date = data.get('interview_date', '')
#     interview_time = data.get('interview_time', '')
#     interviewer_email = data.get('interviewer_email', '')  # Optional
    
#     try:
#         # Create the calendar event
#         result = create_calendar_event(
#             candidate_email, 
#             candidate_name, 
#             interview_date, 
#             interview_time,
#             interviewer_email
#         )
        
#         if result.get('status') == 'success':
#             return jsonify({
#                 'status': 'success',
#                 'message': result['message'],
#                 'event_id': result.get('event_id'),
#                 'event_link': result.get('event_link'),
#                 'candidate_name': candidate_name,
#                 'candidate_email': candidate_email,
#                 'interview_date': interview_date,
#                 'interview_time': interview_time
#             })
#         else:
#             return jsonify({'error': result.get('error', 'Unknown error')}), 500
            
#     except Exception as e:
#         return jsonify({'error': f'Failed to schedule interview: {str(e)}'}), 500

# # Frontend Routes
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/auth')
# def auth_page():
#     return render_template('auth.html')

# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html')

# @app.route('/job-description')
# def job_description():
#     return render_template('job_description.html')

# @app.route('/candidates')
# def candidates():
#     return render_template('candidates.html')

# @app.route('/interviews')
# def interviews():
#     return render_template('interviews.html')

# if __name__ == '__main__':
#     app.run(debug=True)



"""MCP WITH OLD"""



# import os
# from flask import Flask, request, jsonify, send_file, redirect, url_for, session
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START, END
# from typing import TypedDict, Optional
# from langchain_groq import ChatGroq
# import secrets
# import requests
# from langchain.prompts import PromptTemplate
# import re
# from google.oauth2 import service_account
# from googleapiclient.discovery import build
# import io
# from langchain_community.document_loaders import PyPDFLoader
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import tempfile
# import asyncio
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
# from observee_agents import call_mcpauth_login, get_available_servers, chat_with_tools,chat_with_tools_stream

# load_dotenv(override=True)

# app = Flask(__name__)
# app.secret_key = os.urandom(24)

# # LinkedIn OAuth settings
# CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
# CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
# REDIRECT_URI = "http://127.0.0.1:5000/callback"
# AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
# TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"

# # Google Sheets and Drive settings
# GOOGLE_SHEET_ID = "1PrMvW7un7b1zdF_262QwtaP7_xNhf5zxylYIqnOSq_U"
# DRIVE_FOLDER_ID = "1lFh4bkL2PYtyjFcVMiw0SuVEEeP3hXMNuCH4zQgGb8UOyLO1fxs528Gq5jxV8z4lsH3h-9KF"
# CREDENTIALS_PATH = "credentials.json"

# # Observee MCP settings
# OBSERVEE_CLIENT_ID = "0a2bb94e-b62c-4158-881d-57de87bfadcf"
# OBSERVEE_API_KEY = "obs_hyFy_2HNIAnDsRfK-EKhFP8sUzNZ3dO8uk_r6qRnLuQ"

# user_tokens = {}
# user_observee_client_ids = {}  

# async def get_observee_client():
#     """Get Observee MCP client for Google Calendar operations."""
#     return MultiServerMCPClient(
#         {
#             "observee": {
#                 "transport": "streamable_http",
#                 "url": f"https://mcp.observee.ai/mcp?client_id={OBSERVEE_CLIENT_ID}",
#                 "headers": {
#                     "Authorization": f"Bearer {OBSERVEE_API_KEY}"
#                 },
#             }
#         }
#     )

# def schedule_interview_with_observee(candidate_email, candidate_name, interview_date, interview_time, interviewer_email=None, user_id=None):
#     """
#     Schedule interview using Observee Auth SDK - user-specific calendar access!
#     """
#     try:
#         # Check if user has authenticated calendar access
#         if user_id and user_id in user_observee_client_ids:
#             client_id = user_observee_client_ids[user_id]
            
#             # Use the new Auth SDK approach
#             scheduling_message = f"""
#             Schedule an interview meeting for {candidate_name} on {interview_date} at {interview_time} for 1 hour.
#             Attendees: {candidate_email}
#             Summary: Interview with {candidate_name}
#             Description: Interview for the position. Candidate: {candidate_name}, Email: {candidate_email}
#             """
            
#             result = chat_with_tools_stream(
#                 message=scheduling_message,
#                 provider="groq",  
#                 model="llama-3.1-8b-instant",
#                 client_id=client_id,
#                 observee_api_key=f"{OBSERVEE_API_KEY}"
#             )
            
#             if result and "scheduled" in str(result).lower() or "created" in str(result).lower():
#                 return {
#                     'status': 'success',
#                     'message': f'Interview scheduled successfully for {candidate_name}. Email invite sent to {candidate_email}.',
#                     'details': str(result)
#                 }
#             else:
#                 return {
#                     'status': 'success',
#                     'message': f'Interview scheduling completed for {candidate_name}.',
#                     'details': str(result)
#                 }
        
#         else:
#             # Fallback to legacy approach if user hasn't authenticated
#             return {
#                 'status': 'error',
#                 'error': 'Please connect your calendar first. Click "Connect Calendar" to authenticate.'
#             }
        
#     except Exception as e:
#         return {
#             'status': 'error',
#             'error': f'Failed to schedule interview: {str(e)}'
#         }

# class HRHiringState(TypedDict):
#     role: Optional[str]
#     job_description: Optional[str]
#     approval: Optional[bool]

# # --- LangGraph Workflow Setup ---
# def generate_jd_node(state: HRHiringState) -> dict:
#     """Generate job description using Groq LLM."""
#     role = state.get('role', 'Software Engineer')
#     llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")
    
#     prompt = PromptTemplate(
#         input_variables=["role"],
#         template="Generate a comprehensive job description for a {role} position. Include requirements, responsibilities, and qualifications under 1200 characters"
#     )
    
#     chain = prompt | llm
#     result = chain.invoke({"role": role})
    
#     return {"job_description": result.content}

# def human_approval_node(state: HRHiringState) -> dict:
#     """Human approval node - returns current state."""
#     return state

# # Restored chunking function for LinkedIn's 1300-character limit
# def split_text_for_post(text, max_length=1300):
#     """Split text into chunks suitable for social media posts."""
#     words = text.split()
#     chunks = []
#     current_chunk = []
#     current_length = 0
    
#     for word in words:
#         if current_length + len(word) + 1 <= max_length:
#             current_chunk.append(word)
#             current_length += len(word) + 1
#         else:
#             if current_chunk:
#                 chunks.append(' '.join(current_chunk))
#             current_chunk = [word]
#             current_length = len(word)
    
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))
    
#     return chunks

# # Role to application link mapping
# ROLE_FORM_LINKS = {
#     "data": "https://forms.gle/8R5Eg8TaQHgQBFE19",
#     "software": "https://forms.gle/8R5Eg8TaQHgQBFE19",
#     "developer": "https://forms.gle/8R5Eg8TaQHgQBFE19",
#     "engineer": "https://forms.gle/8R5Eg8TaQHgQBFE19"
# }
# GENERIC_FORM_LINK = "https://forms.gle/8R5Eg8TaQHgQBFE19"

# def get_form_link_for_role(role: str) -> str:
#     role_lower = role.lower()
#     for keyword, link in ROLE_FORM_LINKS.items():
#         if keyword in role_lower:
#             return link
#     return GENERIC_FORM_LINK

# def extract_drive_file_id(drive_url):
#     """Extract file ID from Google Drive share link."""
#     if not drive_url:
#         return None
    
#     patterns = [
#         r'/d/([\w-]+)',
#         r'id=([\w-]+)',
#         r'/file/d/([\w-]+)',
#         r'([\w-]{25,})'
#     ]
    
#     for pattern in patterns:
#         match = re.search(pattern, drive_url)
#         if match:
#             return match.group(1)
    
#     return None

# def fetch_applications(sheet_id=GOOGLE_SHEET_ID, credentials_path=CREDENTIALS_PATH, range_name='Form Responses 1'):
#     """
#     Reads applicant data from a Google Sheet and extracts Drive file IDs for resumes.
#     Skips the first column (timestamp).
#     Returns a list of dicts: {name, college, email, intro, resume_file_id}
#     """
#     SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
#     creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
#     service = build('sheets', 'v4', credentials=creds)
#     sheet = service.spreadsheets()
#     result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
#     values = result.get('values', [])
    
#     print(f"Raw sheet data: {values}")
    
#     if not values or len(values) < 2:
#         print("No data found in sheet")
#         return []
        
#     headers = values[0][1:]  # Skip timestamp
#     data_rows = values[1:]
    
#     print(f"Headers: {headers}")
#     print(f"Data rows: {data_rows}")
    
#     applicants = []
#     for i, row in enumerate(data_rows):
#         print(f"Processing row {i+1}: {row}")
        
#         if len(row) < len(headers) + 1:
#             print(f"Row {i+1} is incomplete, skipping")
#             continue  # skip incomplete rows
            
#         row_dict = dict(zip(headers, row[1:]))  # Skip timestamp in each row
#         print(f"Row {i+1} dict: {row_dict}")
        
#         resume_url = row_dict.get('RESUME ', '')  # Note the trailing space to match the actual header
#         print(f"Resume URL for row {i+1}: '{resume_url}'")
        
#         file_id = extract_drive_file_id(resume_url)
#         print(f"Extracted file ID for row {i+1}: '{file_id}'")
        
#         applicant = {
#             'name': row_dict.get('Name', ''),
#             'college': row_dict.get('College Name and Course', ''),
#             'email': row_dict.get('Email', ''),
#             'intro': row_dict.get('Short Intro about you', ''),
#             'resume_file_id': file_id
#         }
        
#         print(f"Created applicant: {applicant}")
#         applicants.append(applicant)
    
#     print(f"Final applicants list: {applicants}")
#     return applicants

# def fetch_applicants_with_resumes(sheet_id=GOOGLE_SHEET_ID, credentials_path=CREDENTIALS_PATH, range_name='Form Responses 1'):
#     """
#     Reads applicant data from Google Sheet and extracts Drive file IDs for resumes.
#     Returns a list of dicts with complete applicant info: {name, college, email, intro, resume_file_id}
#     """
#     SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
#     creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
#     service = build('sheets', 'v4', credentials=creds)
#     sheet = service.spreadsheets()
#     result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
#     values = result.get('values', [])
    
#     print(f"Raw sheet data: {values}")
    
#     if not values or len(values) < 2:
#         print("No data found in sheet")
#         return []
        
#     headers = values[0]
#     data_rows = values[1:]
    
#     print(f"Headers: {headers}")
#     print(f"Data rows: {data_rows}")
    
#     # Find the RESUME column index (handle trailing spaces)
#     resume_col_index = None
#     for i, header in enumerate(headers):
#         if header.strip() == 'RESUME':
#             resume_col_index = i
#             break
    
#     if resume_col_index is None:
#         print("RESUME column not found!")
#         print(f"Available headers: {[h.strip() for h in headers]}")
#         return []
    
#     print(f"RESUME column found at index: {resume_col_index}")
    
#     applicants = []
#     for i, row in enumerate(data_rows):
#         print(f"Processing row {i+1}: {row}")
        
#         if len(row) > resume_col_index:
#             resume_url = row[resume_col_index]
#             print(f"Resume URL for row {i+1}: '{resume_url}'")
            
#             file_id = extract_drive_file_id(resume_url)
#             print(f"Extracted file ID for row {i+1}: '{file_id}'")
            
#             if file_id:
#                 # Create applicant dict with all available data
#                 applicant = {
#                     'name': row[1] if len(row) > 1 else '',
#                     'college': row[2] if len(row) > 2 else '',
#                     'email': row[3] if len(row) > 3 else '',
#                     'intro': row[4] if len(row) > 4 else '',
#                     'resume_file_id': file_id
#                 }
#                 applicants.append(applicant)
#         else:
#             print(f"Row {i+1} doesn't have RESUME column data")
    
#     print(f"Final applicants list: {applicants}")
#     return applicants

# def fetch_resume_file_ids(sheet_id=GOOGLE_SHEET_ID, credentials_path=CREDENTIALS_PATH, range_name='Form Responses 1'):
#     """
#     Reads only the RESUME column from Google Sheet and extracts Drive file IDs.
#     Returns a list of file IDs for resumes.
#     """
#     SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
#     creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
#     service = build('sheets', 'v4', credentials=creds)
#     sheet = service.spreadsheets()
#     result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
#     values = result.get('values', [])
    
#     print(f"Raw sheet data: {values}")
    
#     if not values or len(values) < 2:
#         print("No data found in sheet")
#         return []
        
#     headers = values[0]
#     data_rows = values[1:]
    
#     print(f"Headers: {headers}")
#     print(f"Data rows: {data_rows}")
    
#     # Find the RESUME column index (handle trailing spaces)
#     resume_col_index = None
#     for i, header in enumerate(headers):
#         if header.strip() == 'RESUME':
#             resume_col_index = i
#             break
    
#     if resume_col_index is None:
#         print("RESUME column not found!")
#         print(f"Available headers: {[h.strip() for h in headers]}")
#         return []
    
#     print(f"RESUME column found at index: {resume_col_index}")
    
#     file_ids = []
#     for i, row in enumerate(data_rows):
#         print(f"Processing row {i+1}: {row}")
        
#         if len(row) > resume_col_index:
#             resume_url = row[resume_col_index]
#             print(f"Resume URL for row {i+1}: '{resume_url}'")
            
#             file_id = extract_drive_file_id(resume_url)
#             print(f"Extracted file ID for row {i+1}: '{file_id}'")
            
#             if file_id:
#                 file_ids.append(file_id)
#         else:
#             print(f"Row {i+1} doesn't have RESUME column data")
    
#     print(f"Final file IDs list: {file_ids}")
#     return file_ids

# def download_pdf_from_drive(file_id, credentials_path):
#     """Download PDF from Google Drive and return bytes."""
#     creds = service_account.Credentials.from_service_account_file(
#         credentials_path, 
#         scopes=["https://www.googleapis.com/auth/drive.readonly"]
#     )
#     service = build('drive', 'v3', credentials=creds)
#     request = service.files().get_media(fileId=file_id)
#     fh = io.BytesIO()
    
#     try:
#         from googleapiclient.http import MediaIoBaseDownload
#         downloader = MediaIoBaseDownload(fh, request)
#         done = False
#         while not done:
#             status, done = downloader.next_chunk()
#         fh.seek(0)
#         return fh.read()
#     except Exception as e:
#         return None

# def extract_text_from_pdf_bytes(pdf_bytes):
#     """Extract text from PDF bytes using LangChain."""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(pdf_bytes)
#         tmp_file_path = tmp_file.name

#     try:
#         loader = PyPDFLoader(tmp_file_path)
#         docs = loader.load()
#         text = " ".join([doc.page_content for doc in docs])
#     finally:
#         os.remove(tmp_file_path)
#     return text

# def select_best_candidates(applicants, job_description, credentials_path):
#     """
#     For each applicant, download and parse the PDF resume, compare to job description,
#     and return the top 2 candidates (by cosine similarity).
#     """
#     print(f"Starting candidate selection for {len(applicants)} applicants")
#     resume_texts = []
#     valid_applicants = []
    
#     for i, applicant in enumerate(applicants):
#         file_id = applicant['resume_file_id']
#         print(f"Processing applicant {i+1}: {applicant['name']} (file ID: {file_id})")
        
#         pdf_bytes = download_pdf_from_drive(file_id, credentials_path)
#         if not pdf_bytes:
#             print(f"Failed to download PDF for applicant: {applicant['name']}")
#             continue
            
#         try:
#             print(f"Extracting text from PDF for applicant: {applicant['name']}")
#             text = extract_text_from_pdf_bytes(pdf_bytes)
#             print(f"Extracted text length: {len(text)} characters")
            
#             if text.strip():
#                 resume_texts.append(text)
#                 valid_applicants.append(applicant)
#                 print(f"Successfully processed applicant: {applicant['name']}")
#             else:
#                 print(f"Empty text extracted for applicant: {applicant['name']}")
#         except Exception as e:
#             print(f"Error parsing PDF for applicant {applicant['name']}: {e}")
#             continue
    
#     print(f"Successfully processed {len(valid_applicants)} applicants")
    
#     if not resume_texts:
#         print("No valid resumes found")
#         return []
        
#     if len(resume_texts) == 1:
#         print("Only one applicant found, returning it")
#         return [{'applicant': valid_applicants[0], 'similarity_score': 1.0}]
    
#     try:
#         print("Vectorizing resumes and job description")
#         # Vectorize resumes and job description
#         vectorizer = TfidfVectorizer().fit(resume_texts + [job_description])
#         resume_vecs = vectorizer.transform(resume_texts)
#         jd_vec = vectorizer.transform([job_description])
#         scores = cosine_similarity(resume_vecs, jd_vec).flatten()
        
#         print(f"Similarity scores: {scores}")
        
#         # Get top 2 indices (or all if less than 2)
#         num_to_return = min(2, len(scores))
#         top_indices = scores.argsort()[-num_to_return:][::-1]
        
#         top_candidates = []
#         for idx in top_indices:
#             top_candidates.append({
#                 'applicant': valid_applicants[idx],
#                 'similarity_score': float(scores[idx])
#             })
#             print(f"Top candidate: {valid_applicants[idx]['name']} with score: {scores[idx]}")
            
#         return top_candidates
        
#     except Exception as e:
#         print(f"Error in vectorization/similarity calculation: {e}")
#         # Fallback: return all valid applicants with default scores
#         fallback_candidates = []
#         for applicant in valid_applicants:
#             fallback_candidates.append({
#                 'applicant': applicant,
#                 'similarity_score': 0.5
#             })
#         return fallback_candidates[:2] if len(fallback_candidates) >= 2 else fallback_candidates

# def pick_best_resume(applicants):
#     """Return the applicant with the highest similarity_score from a list."""
#     if not applicants:
#         return None
#     best = max(applicants, key=lambda x: x.get('similarity_score', 0))
#     return best

# # LinkedIn posting node
# def post_linkedin_node(state: HRHiringState) -> dict:
#     jd = state.get('job_description')
#     role = state.get('role', '')
#     user_id = session.get('user_id')
#     if not jd or not user_id or user_id not in user_tokens:
#         return {'post_status': 'error', 'error': 'Missing JD or not authenticated'}

#     # Append the application link
#     form_link = get_form_link_for_role(role)
#     jd_with_link = f"{jd.strip()}\n\nApply here: {form_link}"

#     # Split into chunks (though typically one post for LinkedIn)
#     chunks = split_text_for_post(jd_with_link)
#     if not chunks:
#         return {'post_status': 'error', 'error': 'No content to post'}

#     access_token = user_tokens[user_id]['access_token']
#     headers = {
#         'Authorization': f'Bearer {access_token}',
#         'Content-Type': 'application/json',
#         'X-Restli-Protocol-Version': '2.0.0'
#     }

#     # Get user ID from OpenID Connect ID token or fallback to profile API
#     user_id = session.get('user_id')
#     person_urn = None
    
#     if user_id in user_tokens and user_tokens[user_id].get('id_token'):
#         # Try to extract user ID from ID token (simplified approach)
#         # In production, you should properly decode and validate the JWT
#         try:
#             # For now, we'll use the profile API as fallback
#             profile_response = requests.get('https://api.linkedin.com/v2/userinfo', headers=headers)
#             if profile_response.status_code == 200:
#                 profile_data = profile_response.json()
#                 person_urn = profile_data.get('sub')  # OpenID Connect subject identifier
#         except:
#             pass
    
#     # Fallback to profile API if OpenID Connect approach fails
#     if not person_urn:
#         profile_response = requests.get('https://api.linkedin.com/v2/me', headers=headers)
#         if profile_response.status_code != 200:
#             return {'post_status': 'error', 'error': f'Failed to get user profile: {profile_response.text}'}
#         profile_data = profile_response.json()
#         person_urn = profile_data.get('id')
    
#     if not person_urn:
#         return {'post_status': 'error', 'error': 'Could not retrieve user URN'}

#     post_ids = []
#     for idx, chunk in enumerate(chunks):
#         payload = {
#             'author': f'urn:li:person:{person_urn}',
#             'lifecycleState': 'PUBLISHED',
#             'specificContent': {
#                 'com.linkedin.ugc.ShareContent': {
#                     'shareCommentary': {'text': chunk},
#                     'shareMediaCategory': 'NONE'
#                 }
#             },
#             'visibility': {'com.linkedin.ugc.MemberNetworkVisibility': 'PUBLIC'}
#         }
#         print(f"Posting chunk {idx+1}/{len(chunks)} with person_urn: {person_urn}")
#         print(f"Payload: {payload}")
#         try:
#             response = requests.post(
#                 'https://api.linkedin.com/v2/ugcPosts',
#                 json=payload,
#                 headers=headers
#             )
#             print(f"Response status: {response.status_code}")
#             print(f"Response headers: {dict(response.headers)}")
#             print(f"Response text: {response.text}")
#             if response.status_code in [200, 201]:
#                 post_id = response.headers.get('X-RestLi-Id')
#                 post_url = f'https://www.linkedin.com/feed/update/{post_id}' if post_id else None
#                 post_ids.append({'id': post_id, 'url': post_url})
#                 print(f"Successfully posted with ID: {post_id}")
#             else:
#                 return {'post_status': 'error', 'error': f'Post failed: {response.text}', 'post_ids': post_ids}
#         except Exception as e:
#             print(f"Exception during posting: {str(e)}")
#             return {'post_status': 'error', 'error': f'Post request failed: {str(e)}', 'post_ids': post_ids}
    
#     return {'post_status': 'success', 'post_ids': post_ids}

# def approval_conditional(state: HRHiringState) -> str:
#     """Conditional routing based on approval."""
#     return "approved" if state.get('approval') else "rejected"

# # Update StateGraph
# def schedule_interviews_node(state: HRHiringState) -> dict:
#     """
#     Schedule interviews for the top candidates.
#     This node will be called after selecting the best candidates.
#     """
#     # Get the top candidates from the state
#     top_candidates = state.get('top_candidates', [])
    
#     if not top_candidates:
#         return {'status': 'no_candidates_to_schedule'}
    
#     # For now, return the candidates for manual scheduling
#     # In a full implementation, you would integrate with Google Calendar API
#     return {
#         'status': 'candidates_ready_for_scheduling',
#         'top_candidates': top_candidates
#     }

# builder = StateGraph(HRHiringState)
# builder.add_node('generate_jd', generate_jd_node)
# builder.add_node('human_approval', human_approval_node)
# builder.add_node('post_linkedin', post_linkedin_node)
# builder.add_node('fetch_applications', fetch_applications)
# builder.add_node('select_best_resumes', select_best_candidates)
# builder.add_node('schedule_interviews', schedule_interviews_node)

# builder.add_edge(START, 'generate_jd')
# builder.add_edge('generate_jd', 'human_approval')
# builder.add_conditional_edges('human_approval', approval_conditional)
# builder.add_edge('post_linkedin', 'fetch_applications')
# builder.add_edge('fetch_applications', 'select_best_resumes')
# builder.add_edge('select_best_resumes', 'schedule_interviews')
# builder.add_edge('schedule_interviews', END)

# graph = builder.compile()

# # --- Flask Endpoints ---
# @app.route('/')
# def landing_page():
#     return send_file('index.html')

# @app.route('/login')
# def login():
#     if not CLIENT_ID or not CLIENT_SECRET:
#         return jsonify({'error': 'LinkedIn Client ID or Secret not set'}), 500
#     state = secrets.token_urlsafe(16)
#     session['state'] = state
#     session['user_id'] = secrets.token_urlsafe(16)
#     auth_params = {
#         'response_type': 'code',
#         'client_id': CLIENT_ID,
#         'redirect_uri': REDIRECT_URI,
#         'state': state,
#         'scope': 'openid profile w_member_social'  # Use OpenID Connect scopes
#     }
#     auth_url = f"{AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in auth_params.items())}"
#     return redirect(auth_url)

# @app.route('/callback')
# def callback():
#     code = request.args.get('code')
#     state = request.args.get('state')
#     error = request.args.get('error')
#     error_description = request.args.get('error_description')
#     saved_state = session.get('state')
#     if error:
#         return jsonify({'error': f'OAuth error: {error}', 'description': error_description}), 400
#     if not code or state != saved_state:
#         return jsonify({'error': 'Invalid login attempt'}), 400
#     token_data = {
#         'grant_type': 'authorization_code',
#         'code': code,
#         'client_id': CLIENT_ID,
#         'client_secret': CLIENT_SECRET,
#         'redirect_uri': REDIRECT_URI
#     }
#     try:
#         headers = {'Content-Type': 'application/x-www-form-urlencoded'}
#         response = requests.post(TOKEN_URL, data=token_data, headers=headers)
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to get token', 'details': response.text}), 500
#         token_response = response.json()
        
#         # Store both access token and ID token
#         user_tokens[session['user_id']] = {
#             'access_token': token_response['access_token'],
#             'id_token': token_response.get('id_token')  # OpenID Connect ID token
#         }
#         return redirect(url_for('landing_page'))
#     except Exception as e:
#         return jsonify({'error': f'Authentication failed: {str(e)}'}), 500

# @app.route('/logout')
# def logout():
#     user_id = session.get('user_id')
#     if user_id and user_id in user_tokens:
#         del user_tokens[user_id]
#     session.clear()
#     return redirect(url_for('landing_page'))

# @app.route('/check-auth')
# def check_auth():
#     user_id = session.get('user_id')
#     is_authenticated = user_id is not None and user_id in user_tokens
#     if is_authenticated:
#         access_token = user_tokens[user_id]['access_token']
#         headers = {
#             'Authorization': f'Bearer {access_token}',
#             'Content-Type': 'application/json',
#             'X-Restli-Protocol-Version': '2.0.0'
#         }
#         try:
#             # Try OpenID Connect userinfo endpoint first
#             test_response = requests.get('https://api.linkedin.com/v2/userinfo', headers=headers)
#             if test_response.status_code != 200:
#                 # Fallback to profile API
#                 test_response = requests.get('https://api.linkedin.com/v2/me', headers=headers)
            
#             token_valid = test_response.status_code == 200
#             return jsonify({
#                 'authenticated': is_authenticated,
#                 'token_valid': token_valid,
#                 'profile_status': test_response.status_code,
#                 'profile_response': test_response.text[:200] if not token_valid else 'Profile accessible'
#             })
#         except Exception as e:
#             return jsonify({
#                 'authenticated': is_authenticated,
#                 'token_valid': False,
#                 'error': str(e)
#             })
#     return jsonify({'authenticated': is_authenticated})

# @app.route('/generate-jd', methods=['POST'])
# def generate_jd_api():
#     data = request.get_json()
#     if not data or 'role' not in data:
#         return jsonify({'error': 'Role is required'}), 400
#     state = {'role': data.get('role')}
#     result = generate_jd_node(state)
#     return jsonify(result)

# @app.route('/approve-jd', methods=['POST'])
# def approve_jd_api():
#     data = request.get_json()
#     if not data or 'role' not in data or 'job_description' not in data:
#         return jsonify({'error': 'Role and job_description are required'}), 400
#     state = {
#         'role': data.get('role'),
#         'job_description': data.get('job_description'),
#         'approval': data.get('approval', False)
#     }
#     result = human_approval_node(state)
#     if not data.get('approval'):
#         result = generate_jd_node({'role': data.get('role')})
#     return jsonify(result)

# @app.route('/post-jd', methods=['POST'])
# def post_jd_api():
#     data = request.get_json()
#     if not data or 'role' not in data or 'job_description' not in data:
#         return jsonify({'error': 'Role and job_description are required'}), 400
#     state = {
#         'role': data.get('role'),
#         'job_description': data.get('job_description'),
#         'approval': True
#     }
#     result = post_linkedin_node(state)
#     return jsonify(result)

# @app.route('/fetch-applications')
# def fetch_applications_api():
#     """Fetch applications from Google Sheet using hardcoded IDs."""
#     try:
#         applicants = fetch_applications()
#         return jsonify({'applicants': applicants})
#     except Exception as e:
#         return jsonify({'error': f'Failed to fetch applications: {str(e)}'}), 500

# @app.route('/select-best-resumes', methods=['POST'])
# def select_best_resumes_api():
#     """Select best candidates using the provided job description."""
#     data = request.get_json()
#     if not data or 'job_description' not in data:
#         return jsonify({'error': 'Job description is required'}), 400
    
#     print(f"Job description length: {len(data['job_description'])} characters")
    
#     try:
#         # First get the applicants with resume data
#         applicants = fetch_applicants_with_resumes()
        
#         if not applicants:
#             return jsonify({'error': 'No applicants found'}), 400
        
#         print(f"Found {len(applicants)} applicants")
        
#         # Then select the best candidates
#         top_candidates = select_best_candidates(
#             applicants, 
#             data['job_description'], 
#             CREDENTIALS_PATH
#         )
        
#         if not top_candidates:
#             return jsonify({'error': 'No valid candidates found'}), 400
            
#         # Pick the single best candidate
#         best_candidate = max(top_candidates, key=lambda x: x.get('similarity_score', 0))
        
#         print(f"Selected best candidate: {best_candidate['applicant']['name']} with score: {best_candidate['similarity_score']}")
#         return jsonify({
#             'best_candidate': best_candidate,
#             'all_top_candidates': top_candidates  # Keep for debugging
#         })
            
#     except Exception as e:
#         print(f"Error in select_best_resumes_api: {str(e)}")
#         return jsonify({'error': f'Failed to select best candidates: {str(e)}'}), 500

# @app.route('/test-sheet')
# def test_sheet():
#     """Test endpoint to see raw data from Google Sheet."""
#     try:
#         SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
#         creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
#         service = build('sheets', 'v4', credentials=creds)
#         sheet = service.spreadsheets()
#         result = sheet.values().get(spreadsheetId=GOOGLE_SHEET_ID, range='Form Responses 1').execute()
#         values = result.get('values', [])
        
#         return jsonify({
#             'raw_data': values,
#             'row_count': len(values),
#             'headers': values[0] if values else [],
#             'data_rows': values[1:] if len(values) > 1 else []
#         })
#     except Exception as e:
#         return jsonify({'error': f'Failed to read sheet: {str(e)}'}), 500

# @app.route('/fetch-resumes')
# def fetch_resumes_api():
#     """Fetch only resume file IDs from Google Sheet."""
#     try:
#         file_ids = fetch_resume_file_ids()
#         return jsonify({'resume_file_ids': file_ids})
#     except Exception as e:
#         return jsonify({'error': f'Failed to fetch resumes: {str(e)}'}), 500

# @app.route('/view-resume/<file_id>')
# def view_resume_api(file_id):
#     """Serve a PDF resume file for viewing."""
#     try:
#         pdf_bytes = download_pdf_from_drive(file_id, CREDENTIALS_PATH)
#         if not pdf_bytes:
#             return jsonify({'error': 'Resume not found'}), 404
        
#         # Return the PDF as a file response
#         return send_file(
#             io.BytesIO(pdf_bytes),
#             mimetype='application/pdf',
#             as_attachment=False,
#             download_name=f'resume_{file_id}.pdf'
#         )
#     except Exception as e:
#         return jsonify({'error': f'Failed to load resume: {str(e)}'}), 500

# @app.route('/schedule-interview', methods=['POST'])
# def schedule_interview_api():
#     """Schedule a calendar invite for a candidate."""
#     data = request.get_json()
#     if not data or 'candidate_email' not in data or 'candidate_name' not in data:
#         return jsonify({'error': 'Candidate email and name are required'}), 400
    
#     candidate_email = data['candidate_email']
#     candidate_name = data['candidate_name']
#     interview_date = data.get('interview_date', '')
#     interview_time = data.get('interview_time', '')
#     interviewer_email = data.get('interviewer_email', '')  # Optional
    
#     try:
#         # Get user_id for personalized calendar access
#         user_id = session.get('user_id')
        
#         result = schedule_interview_with_observee(
#             candidate_email, 
#             candidate_name, 
#             interview_date, 
#             interview_time,
#             interviewer_email,
#             user_id
#         )
        
#         if result.get('status') == 'success':
#             return jsonify({
#                 'status': 'success',
#                 'message': result['message'],
#                 'details': result.get('details', ''),
#                 'candidate_name': candidate_name,
#                 'candidate_email': candidate_email,
#                 'interview_date': interview_date,
#                 'interview_time': interview_time
#             })
#         else:
#             return jsonify({'error': result.get('error', 'Unknown error')}), 500
            
#     except Exception as e:
#         return jsonify({'error': f'Failed to schedule interview: {str(e)}'}), 500

# @app.route('/connect-calendar', methods=['POST'])
# def connect_calendar():
#     """Start calendar authentication flow for user."""
#     user_id = session.get('user_id')
#     if not user_id:
#         return jsonify({'error': 'Please login first'}), 401
    
#     try:
#         # Generate unique client_id for this user
#         import uuid
#         client_id = str(uuid.uuid4())
        
#         # Start authentication flow
#         response = call_mcpauth_login(
#             auth_server="gmail",  # This includes Google Calendar
#             client_id=client_id
#         )
        
#         if response and 'url' in response:
#             # Store the client_id for this user
#             user_observee_client_ids[user_id] = client_id
            
#             return jsonify({
#                 'status': 'success',
#                 'auth_url': response['url'],
#                 'client_id': client_id,
#                 'message': 'Please visit the URL to authenticate your Google Calendar'
#             })
#         else:
#             return jsonify({'error': 'Failed to start authentication'}), 500
            
#     except Exception as e:
#         return jsonify({'error': f'Failed to connect calendar: {str(e)}'}), 500

# @app.route('/check-calendar-status')
# def check_calendar_status():
#     """Check if user has connected their calendar."""
#     user_id = session.get('user_id')
#     if not user_id:
#         return jsonify({'connected': False, 'message': 'Please login first'})
    
#     if user_id in user_observee_client_ids:
#         client_id = user_observee_client_ids[user_id]
        
#         try:
#             # For now, assume if user has a client_id, they're connected
#             # In a real implementation, you'd verify the specific client_id
#             return jsonify({
#                 'connected': True,
#                 'message': 'Calendar connected successfully!',
#                 'services': ['gmail']  # Assume Gmail/Calendar is available
#             })
                
#         except Exception as e:
#             return jsonify({
#                 'connected': False,
#                 'message': f'Error checking calendar status: {str(e)}'
#             })
#     else:
#         return jsonify({
#             'connected': False,
#             'message': 'Calendar not connected. Click "Connect Calendar" to get started.'
#         })

# @app.route('/check-observee-status')
# def check_observee_status():
#     """Check if Observee MCP is working."""
#     try:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
        
#         client = loop.run_until_complete(get_observee_client())
#         tools = loop.run_until_complete(client.get_tools())
        
#         loop.close()
        
#         calendar_tools = [tool for tool in tools if 'calendar' in tool.name.lower()]
        
#         return jsonify({
#             'status': 'success',
#             'message': 'Observee MCP is working!',
#             'total_tools': len(tools),
#             'calendar_tools': len(calendar_tools),
#             'available_calendar_tools': [tool.name for tool in calendar_tools]
#         })
        
#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'error': f'Observee MCP connection failed: {str(e)}'
#         }), 500

# if __name__ == '__main__':
#     app.run(debug=True)