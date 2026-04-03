"""
Hiring workflow routes for job description generation and posting.
"""
from flask import Blueprint, request, jsonify, session, send_file
from datetime import datetime, timedelta
from sqlalchemy import or_
import io
from app.services.ai_service import AIService
from app.services.linkedin_service import LinkedInService
from app.services.job_service import JobService
from app.services.google_service import GoogleService
from app.services.resume_service import ResumeService
from app.models.hiring import HRHiringState
from app.models.db_models import db, JobPosting, Application, AIInterviewSession

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
        
    # Save the generated JD using JobService
    try:
        new_job = JobService.create_job(
            role=data.get('role'),
            company_name=data.get('company_name'),
            location=data.get('location'),
            employment_type=data.get('employment_type'),
            job_description=result.get('job_description')
        )
        result['job_id'] = new_job.id
    except Exception as e:
        print(f"Error saving JD to database: {e}")
        
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
        
    # If regenerated, use JobService
    if not approval:
        try:
            new_job = JobService.create_job(
                role=role,
                company_name=data.get('company_name'),
                job_description=result.get('job_description')
            )
            result['job_id'] = new_job.id
        except Exception as e:
            print(f"Error saving JD to database: {e}")
            
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
        data.get('job_description'),
        job_id=data.get('job_id')
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
    """Get the latest posted job description — uses session first, falls back to DB."""
    latest_jd = session.get('last_posted_jd')
    latest_role = session.get('last_posted_role')

    # Fall back to DB if session was cleared (e.g. server restart)
    if not latest_jd:
        from app.repositories.job_repository import JobRepository
        latest_job = JobRepository.get_latest_job()
        if latest_job and latest_job.job_description:
            latest_jd = latest_job.job_description
            latest_role = latest_job.role
            session['last_posted_jd'] = latest_jd
            session['last_posted_role'] = latest_role

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



def _build_dashboard_activity():
    """Recent events from JobPosting, Application, AIInterviewSession (merged, newest first)."""
    rows = []

    for job in JobPosting.query.order_by(JobPosting.created_at.desc()).limit(4).all():
        rows.append(
            {
                "kind": "job",
                "title": f"Job description saved — {job.role}",
                "detail": job.company_name or None,
                "at": job.created_at.isoformat() + "Z" if job.created_at else None,
                "accent": "violet",
            }
        )

    for app in Application.query.order_by(Application.created_at.desc()).limit(12).all():
        job = db.session.get(JobPosting, app.job_id)
        role = job.role if job else "Unknown role"
        rows.append(
            {
                "kind": "application",
                "title": f"New application — {role}",
                "detail": app.candidate_name,
                "at": app.created_at.isoformat() + "Z" if app.created_at else None,
                "accent": "emerald",
            }
        )

    for sess in AIInterviewSession.query.order_by(AIInterviewSession.created_at.desc()).limit(12).all():
        if sess.status == "completed":
            title = f"AI interview completed — {sess.job_role}"
        elif sess.started_at:
            title = f"AI interview in progress — {sess.job_role}"
        else:
            title = f"AI interview session — {sess.job_role}"
        rows.append(
            {
                "kind": "interview",
                "title": title,
                "detail": sess.candidate_name,
                "at": sess.created_at.isoformat() + "Z" if sess.created_at else None,
                "accent": "sky",
            }
        )

    rows = [r for r in rows if r.get("at")]
    rows.sort(key=lambda r: r["at"], reverse=True)
    return rows[:10]


@hiring_bp.route('/dashboard-summary')
def dashboard_summary_api():
    """Aggregates for the Next.js dashboard — real DB stats (no session required)."""
    try:
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        start_of_month = datetime(now.year, now.month, 1)
        start_of_today = datetime(now.year, now.month, now.day)

        job_count = JobPosting.query.count()
        jobs_this_week = JobPosting.query.filter(JobPosting.created_at >= week_ago).count()

        application_count = Application.query.count()
        applications_this_week = Application.query.filter(Application.created_at >= week_ago).count()
        applications_today = Application.query.filter(Application.created_at >= start_of_today).count()

        ai_interview_sessions = AIInterviewSession.query.count()
        interviews_engaged = AIInterviewSession.query.filter(
            or_(
                AIInterviewSession.started_at.isnot(None),
                AIInterviewSession.status.in_(["in_progress", "completed"]),
            )
        ).count()
        interview_sessions_this_week = AIInterviewSession.query.filter(
            AIInterviewSession.created_at >= week_ago
        ).count()

        completed_interviews_this_month = AIInterviewSession.query.filter(
            AIInterviewSession.completed_at.isnot(None),
            AIInterviewSession.completed_at >= start_of_month,
        ).count()

        latest_job = JobPosting.query.order_by(JobPosting.created_at.desc()).first()
        workflow = None
        if latest_job:
            workflow = {
                "role": latest_job.role,
                "company_name": latest_job.company_name,
                "created_at": latest_job.created_at.isoformat() + "Z" if latest_job.created_at else None,
                "step_label": "Job description created",
            }

        return jsonify(
            {
                "job_count": job_count,
                "jobs_this_week": jobs_this_week,
                "application_count": application_count,
                "applications_this_week": applications_this_week,
                "applications_today": applications_today,
                "ai_interview_sessions": ai_interview_sessions,
                "interviews_engaged": interviews_engaged,
                "interview_sessions_this_week": interview_sessions_this_week,
                "completed_interviews_this_month": completed_interviews_this_month,
                "recent_activity": _build_dashboard_activity(),
                "workflow": workflow,
            }
        )
    except Exception as e:
        print(f"Error dashboard summary: {e}")
        return jsonify({"error": f"Failed to load summary: {str(e)}"}), 500


@hiring_bp.route('/fetch-applications')
def fetch_applications_api():
    """Fetch applications from Internal DB."""
    try:
        internal_applicants = JobService.get_all_applications()
        return jsonify({'applicants': internal_applicants})
    except Exception as e:
        print(f"Error fetching applications: {e}")
        return jsonify({'error': f'Failed to fetch applications: {str(e)}'}), 500

@hiring_bp.route('/select-best-resumes', methods=['POST'])
def select_best_resumes_api():
    """Select best candidates from DB applications using the provided job description."""
    data = request.get_json()
    if not data or 'job_description' not in data:
        return jsonify({'error': 'Job description is required'}), 400

    print(f"Job description length: {len(data['job_description'])} characters")

    try:
        # Fetch applicants from Internal DB (new architecture)
        db_applicants = JobService.get_all_applications()

        if not db_applicants:
            return jsonify({'error': 'No applicants found. Candidates must apply via the job link first.'}), 400

        print(f"Found {len(db_applicants)} applicants in DB")

        # Map DB format → format expected by resume_service
        applicants = [
            {
                'name': app['name'],
                'email': app['email'],
                'resume_path': app.get('resume_path', ''),
                'similarity_score': app.get('similarity_score', 0.0),
            }
            for app in db_applicants
            if app.get('resume_path')  # only include applicants who uploaded a resume
        ]

        if not applicants:
            return jsonify({'error': 'No applicants with resumes found.'}), 400

        # Select best candidates using local resume files
        top_candidates = resume_service.select_best_candidates(
            applicants,
            data['job_description']
        )

        if not top_candidates:
            return jsonify({'error': 'No valid candidates found after resume parsing.'}), 400

        best_candidate = max(top_candidates, key=lambda x: x.get('similarity_score', 0))

        print(f"Selected best candidate: {best_candidate['applicant']['name']} "
              f"with score: {best_candidate['similarity_score']}")

        return jsonify({
            'best_candidate': best_candidate,
            'all_top_candidates': top_candidates
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
