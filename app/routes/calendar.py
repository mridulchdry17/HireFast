from flask import Blueprint, request, jsonify, redirect, session
from app.services.composio_service import ComposioService

calendar_bp = Blueprint('calendar', __name__)

# Initialize Composio service
composio_service = ComposioService()

def _get_user_id():
    """Get user ID from session, falling back to query param for testing."""
    return session.get('user_id') or request.args.get('user_id', 'default_user')

@calendar_bp.route('/connect-calendar')
def connect_calendar():
    """Initiate Composio Google Calendar connection flow."""
    user_id = _get_user_id()
    auth_url = composio_service.get_auth_url(user_id)
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
    
    # Create the calendar event using Composio
    result = composio_service.create_interview_event(
        user_id,
        candidate_email, 
        candidate_name, 
        interview_date, 
        interview_time,
        duration_mins=int(duration)
    )
    
    if result.get('status') == 'success':
        return jsonify({
            'status': 'success',
            'message': 'Interview scheduled successfully via Composio!',
            'candidate_name': candidate_name,
            'candidate_email': candidate_email
        })
    else:
        return jsonify({'error': result.get('message', 'Failed to schedule event')}), 500
