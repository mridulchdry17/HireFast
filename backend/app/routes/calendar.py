from flask import Blueprint, request, jsonify, redirect, session, current_app
from app.services.composio_service import ComposioService
from app.services.ai_interview_service import AIInterviewService

calendar_bp = Blueprint('calendar', __name__)

# Initialize services
composio_service = ComposioService()
ai_interview_service = AIInterviewService()

def _get_user_id():
    """Get user ID from session, falling back to query param for testing."""
    return session.get('user_id') or request.args.get('user_id', 'default_user')


def _calendar_callback_url() -> str:
    """Composio return URL after Google OAuth: {APP_BASE_URL}/scheduling."""
    base = (current_app.config.get('APP_BASE_URL') or '').strip().rstrip('/')
    return f"{base}/scheduling"


def _public_base_url() -> str:
    """Public origin for links in emails/calendar (same as APP_BASE_URL)."""
    return (current_app.config.get('APP_BASE_URL') or '').strip().rstrip('/')


@calendar_bp.route('/connect-calendar')
def connect_calendar():
    """Initiate Composio Google Calendar connection flow."""
    user_id = _get_user_id()
    redirect_url = _calendar_callback_url()
    auth_url = composio_service.get_auth_url(user_id, redirect_url=redirect_url)
    if auth_url:
        return redirect(auth_url)
    return jsonify({'error': 'Failed to generate connection URL'}), 500

@calendar_bp.route('/check-calendar-status')
def check_calendar_status():
    """Check if the user has connected their calendar via Composio."""
    user_id = _get_user_id()
    connected = composio_service.check_connection_status(user_id)
    
    if connected:
        return jsonify({
            'status': 'success',
            'connected': True,
            'message': 'Google Calendar connected successfully via Composio!'
        })
    else:
        return jsonify({
            'status': 'pending',
            'connected': False,
            'message': 'Calendar not connected. Please click "Connect Calendar".'
        })

@calendar_bp.route('/schedule-interview', methods=['POST'])
def schedule_interview_api():
    """Schedule a calendar invite for a candidate via Composio."""
    data = request.get_json()
    if not data or 'candidate_email' not in data or 'candidate_name' not in data:
        return jsonify({'error': 'Candidate email and name are required'}), 400
    
    # Use session user_id first, fall back to body param
    user_id = session.get('user_id') or data.get('user_id', 'default_user')
    candidate_email = data['candidate_email']
    candidate_name = data['candidate_name']
    interview_date = data.get('interview_date', '')
    interview_time = data.get('interview_time', '')
    duration = data.get('duration', 60)  # Default to 60 mins if not provided
    
    interview_mode = data.get('interview_mode', 'human')
    notes = data.get('notes', '')
    
    # If AI Screening, create the session first
    description = notes
    create_meeting_room = True

    if interview_mode == 'ai':
        # Create persistent session
        ai_session = ai_interview_service.create_interview_session(
            application_id=data.get('application_id'), # Optional if scheduled from direct form
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            job_role=data.get('position', 'Candidate'),
            resume_path=data.get('resume_path', '')
        )
        
        # Get absolute URL for the interview link
        relative_link = ai_session['interview_link']
        base_url = _public_base_url()
        interview_link = f"{base_url}{relative_link}"
        
        description = f"AI Screening Interview.\n\nPlease complete this interview within 48 hours at this link: {interview_link}\n\n{notes}"
        create_meeting_room = False
        # For AI interviews, time/duration is just a placeholder in calendar
        # but we'll use the user's selected date/time as the "Invitation start"
    
    # Create the calendar event using Composio
    # We pass the custom description (with link) to Composio
    result = composio_service.create_interview_event(
        user_id,
        candidate_email, 
        candidate_name, 
        interview_date, 
        interview_time,
        duration_mins=int(duration),
        description=description,
        create_meeting_room=create_meeting_room
    )
    
    if result.get('status') == 'success':
        return jsonify({
            'status': 'success',
            'message': f'{"AI Screening invite" if interview_mode == "ai" else "Interview"} scheduled successfully!',
            'candidate_name': candidate_name,
            'candidate_email': candidate_email,
            'interview_mode': interview_mode
        })
    else:
        return jsonify({'error': result.get('message', 'Failed to schedule event')}), 500
