"""
Hiring workflow routes for job description generation and posting.
"""
from flask import Blueprint, request, jsonify, session, send_file
import io
from app.services.ai_service import AIService
from app.services.linkedin_service import LinkedInService
from app.services.google_service import GoogleService
from app.services.resume_service import ResumeService
from app.models.hiring import HRHiringState

hiring_bp = Blueprint('hiring', __name__)

# Initialize services
ai_service = AIService()
google_service = GoogleService()
resume_service = ResumeService()

# Import LinkedIn service from auth module to share instance
from app.routes.auth import linkedin_service

@hiring_bp.route('/generate-jd', methods=['POST'])
def generate_jd_api():
    """Generate job description for a role."""
    data = request.get_json()
    if not data or 'role' not in data:
        return jsonify({'error': 'Role is required'}), 400
    
    result = ai_service.generate_job_description(
        role=data.get('role'),
        company_name=data.get('company_name'),
        location=data.get('location'),
        employment_type=data.get('employment_type'),
        additional_requirements=data.get('additional_requirements')
    )
    if result.get('post_status') == 'error':
        return jsonify(result), 500
    return jsonify(result)

@hiring_bp.route('/approve-jd', methods=['POST'])
def approve_jd_api():
    """Approve or reject job description."""
    data = request.get_json()
    if not data or 'role' not in data or 'job_description' not in data:
        return jsonify({'error': 'Role and job_description are required'}), 400
    
    role = data.get('role')
    job_description = data.get('job_description')
    approval = data.get('approval', False)
    
    if not approval:
        # Generate new JD if not approved
        result = ai_service.generate_job_description(role)
    else:
        result = {
            'role': role,
            'job_description': job_description,
            'approval': approval
        }
    
    if result.get('post_status') == 'error':
        return jsonify(result), 500
    return jsonify(result)

@hiring_bp.route('/post-jd', methods=['POST'])
def post_jd_api():
    """Post job description to LinkedIn and store it for candidate matching."""
    data = request.get_json()
    if not data or 'role' not in data or 'job_description' not in data:
        return jsonify({'error': 'Role and job_description are required'}), 400
    
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User not authenticated'}), 401
    
    result = linkedin_service.post_job_description(
        user_id,
        data.get('role'),
        data.get('job_description')
    )
    
    # Store the JD in session for later use in candidate matching
    if result.get('post_status') == 'success':
        session['last_posted_jd'] = data.get('job_description')
        session['last_posted_role'] = data.get('role')
        session.permanent = True
    
    if result.get('post_status') == 'error':
        return jsonify(result), 500
    return jsonify(result)

@hiring_bp.route('/get-latest-jd')
def get_latest_jd_api():
    """Get the latest posted job description from session."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User not authenticated'}), 401
    
    latest_jd = session.get('last_posted_jd')
    latest_role = session.get('last_posted_role')
    
    if not latest_jd:
        return jsonify({
            'error': 'No job description found. Please post a job first.',
            'has_jd': False
        }), 404
    
    return jsonify({
        'job_description': latest_jd,
        'role': latest_role,
        'has_jd': True
    })

@hiring_bp.route('/fetch-applications')
def fetch_applications_api():
    """Fetch applications from Google Sheet."""
    try:
        applicants = google_service.fetch_applications()
        return jsonify({'applicants': applicants})
    except Exception as e:
        return jsonify({'error': f'Failed to fetch applications: {str(e)}'}), 500

@hiring_bp.route('/select-best-resumes', methods=['POST'])
def select_best_resumes_api():
    """Select best candidates using the provided job description."""
    data = request.get_json()
    if not data or 'job_description' not in data:
        return jsonify({'error': 'Job description is required'}), 400
    
    print(f"Job description length: {len(data['job_description'])} characters")
    
    try:
        # First get the applicants with resume data
        applicants = google_service.fetch_applicants_with_resumes()
        
        if not applicants:
            return jsonify({'error': 'No applicants found'}), 400
        
        print(f"Found {len(applicants)} applicants")
        
        # Then select the best candidates
        top_candidates = resume_service.select_best_candidates(
            applicants, 
            data['job_description']
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

@hiring_bp.route('/test-sheet')
def test_sheet():
    """Test endpoint to see raw data from Google Sheet."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from app.config import Config
        
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        creds = service_account.Credentials.from_service_account_file(
            Config.CREDENTIALS_PATH, scopes=SCOPES
        )
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=Config.GOOGLE_SHEET_ID, 
            range='Form Responses 1'
        ).execute()
        values = result.get('values', [])
        
        return jsonify({
            'raw_data': values,
            'row_count': len(values),
            'headers': values[0] if values else [],
            'data_rows': values[1:] if len(values) > 1 else []
        })
    except Exception as e:
        return jsonify({'error': f'Failed to read sheet: {str(e)}'}), 500

@hiring_bp.route('/fetch-resumes')
def fetch_resumes_api():
    """Fetch only resume file IDs from Google Sheet."""
    try:
        # This would need to be implemented in GoogleService
        # For now, return a placeholder
        return jsonify({'message': 'Endpoint not yet implemented'})
    except Exception as e:
        return jsonify({'error': f'Failed to fetch resumes: {str(e)}'}), 500

@hiring_bp.route('/view-resume/<file_id>')
def view_resume_api(file_id):
    """Serve a PDF resume file for viewing."""
    try:
        pdf_bytes = google_service.download_pdf_from_drive(file_id)
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
