"""
Calendar integration routes for scheduling interviews.
"""
from flask import Blueprint, request, jsonify
from app.services.google_service import GoogleService

calendar_bp = Blueprint('calendar', __name__)

# Initialize Google service
google_service = GoogleService()

@calendar_bp.route('/google-calendar-auth')
def google_calendar_auth():
    """Initiate Google Calendar OAuth flow."""
    try:
        import os
        from app.config import Config
        
        if not os.path.exists(Config.GOOGLE_CALENDAR_CREDENTIALS_FILE):
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
                    '8. Place it in the credentials directory'
                ]
            }), 400
        
        # This will trigger the OAuth flow
        creds = google_service.get_google_calendar_credentials()
        
        if isinstance(creds, dict) and 'error' in creds:
            return jsonify(creds), 400
        
        return jsonify({
            'status': 'success',
            'message': 'Google Calendar authentication successful! You can now send calendar invites.'
        })
        
    except Exception as e:
        return jsonify({'error': f'Authentication failed: {str(e)}'}), 500

@calendar_bp.route('/check-google-calendar-auth')
def check_google_calendar_auth():
    """Check if Google Calendar is authenticated."""
    result = google_service.check_calendar_auth()
    return jsonify(result)

@calendar_bp.route('/schedule-interview', methods=['POST'])
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
        result = google_service.create_calendar_event(
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
