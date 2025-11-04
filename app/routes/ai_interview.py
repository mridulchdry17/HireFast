"""
AI Interview routes for conducting automated interviews.
"""
import os
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from app.services.ai_interview_service import AIInterviewService
from app.services.audio_service import AudioService
from app.services.resume_parser_service import ResumeParserService
from app.services.google_service import GoogleService
from app.models.ai_interview import AIInterview

ai_interview_bp = Blueprint('ai_interview', __name__)

# Initialize services
print(f"🔍 DEBUG: Initializing AI interview services...")
ai_interview_service = AIInterviewService()
audio_service = AudioService()
resume_parser_service = ResumeParserService()
google_service = GoogleService()
print(f"🔍 DEBUG: All services initialized successfully")

@ai_interview_bp.route('/ai-interviewer/<interview_id>')
def ai_interviewer_page(interview_id):
    """Main AI interviewer page for candidates."""
    print(f"🔍 DEBUG: Accessing interview page for ID: {interview_id}")
    
    try:
        # Get interview status
        print(f"🔍 DEBUG: Getting interview status from service...")
        interview_status = ai_interview_service.get_interview_status(interview_id)
        print(f"🔍 DEBUG: Interview status result: {interview_status}")
        
        if 'error' in interview_status:
            print(f"🔍 DEBUG: Interview not found, redirecting to direct AI interview page")
            # If interview not found, redirect to direct AI interview page
            from flask import redirect, url_for
            return redirect('/direct-ai-interview')
        
        interview = interview_status['interview']
        print(f"🔍 DEBUG: Found interview: {interview.get('candidate_name', 'Unknown')} - {interview.get('job_role', 'Unknown role')}")
        
        return render_template('ai_interview.html', 
                             interview=interview,
                             conversation_history=interview_status.get('conversation_history', []))
        
    except Exception as e:
        print(f"🔍 DEBUG: Exception in ai_interviewer_page: {str(e)}")
        # Fallback: redirect to direct AI interview page
        from flask import redirect
        return redirect('/direct-ai-interview')

@ai_interview_bp.route('/api/ai-interview/create', methods=['POST'])
def create_ai_interview():
    """Create a new AI interview session for a candidate."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['candidate_id', 'candidate_name', 'candidate_email', 'job_role']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Get resume text if resume_file_id is provided
        resume_text = ""
        if 'resume_file_id' in data and data['resume_file_id']:
            try:
                # Download and parse resume from Google Drive
                pdf_bytes = google_service.download_pdf_from_drive(data['resume_file_id'])
                if pdf_bytes:
                    resume_text = resume_parser_service.extract_text_from_pdf_bytes(pdf_bytes)
                else:
                    resume_text = "Resume not available"
            except Exception as e:
                resume_text = f"Resume parsing failed: {str(e)}"
        else:
            resume_text = "No resume provided"
        
        # Create interview session
        interview = ai_interview_service.create_interview_session(
            candidate_id=data['candidate_id'],
            candidate_name=data['candidate_name'],
            candidate_email=data['candidate_email'],
            job_role=data['job_role'],
            resume_text=resume_text
        )
        
        return jsonify({
            'success': True,
            'interview': interview,
            'interview_link': f"/ai-interviewer/{interview['id']}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to create interview: {str(e)}'}), 500

@ai_interview_bp.route('/api/ai-interview/<interview_id>/start', methods=['POST'])
def start_interview(interview_id):
    """Start an AI interview session."""
    try:
        result = ai_interview_service.start_interview(interview_id)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            'interview': result['interview'],
            'question': result['question'],
            'status': result['status']
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start interview: {str(e)}'}), 500

@ai_interview_bp.route('/api/ai-interview/<interview_id>/answer', methods=['POST'])
def submit_answer(interview_id):
    """Submit an answer to an interview question."""
    try:
        data = request.get_json()
        
        if 'answer_text' not in data:
            return jsonify({'error': 'Answer text is required'}), 400
        
        answer_text = data['answer_text']
        question_id = data.get('question_id', '')
        answer_audio_path = None
        
        # Handle audio answer if provided
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            if audio_file and audio_file.filename:
                # Save audio file
                filename = secure_filename(f"{interview_id}_{uuid.uuid4().hex[:8]}.wav")
                audio_path = os.path.join(audio_service.audio_storage_path, filename)
                audio_file.save(audio_path)
                answer_audio_path = audio_path
                
                # Convert speech to text if no text provided
                if not answer_text:
                    try:
                        answer_text = audio_service.speech_to_text(audio_path)
                    except Exception as e:
                        return jsonify({'error': f'Speech-to-text failed: {str(e)}'}), 500
        
        # Submit answer and get evaluation
        result = ai_interview_service.submit_answer(
            interview_id=interview_id,
            question_id=question_id,
            answer_text=answer_text,
            answer_audio_path=answer_audio_path
        )
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'evaluation': result['evaluation'],
            'interview_complete': result['interview_complete'],
            'next_question': result.get('next_question'),
            'progress': result.get('progress'),
            'overall_score': result.get('overall_score')
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to submit answer: {str(e)}'}), 500

@ai_interview_bp.route('/api/ai-interview/<interview_id>/status')
def get_interview_status(interview_id):
    """Get the current status of an interview."""
    try:
        result = ai_interview_service.get_interview_status(interview_id)
        
        if 'error' in result:
            return jsonify(result), 404
        
        return jsonify({
            'success': True,
            'interview': result['interview'],
            'conversation_history': result['conversation_history']
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get interview status: {str(e)}'}), 500

@ai_interview_bp.route('/api/ai-interview/<interview_id>/record-audio', methods=['POST'])
def record_audio(interview_id):
    """Record audio for an interview answer."""
    try:
        data = request.get_json()
        duration = data.get('duration', 10)  # Default 10 seconds
        
        # Record audio
        audio_path = audio_service.record_audio(duration=duration)
        
        # Convert to text
        try:
            transcribed_text = audio_service.speech_to_text(audio_path)
        except Exception as e:
            return jsonify({'error': f'Speech-to-text failed: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'audio_path': audio_path,
            'transcribed_text': transcribed_text
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to record audio: {str(e)}'}), 500

@ai_interview_bp.route('/api/ai-interview/<interview_id>/speak', methods=['POST'])
def speak_text(interview_id):
    """Convert text to speech for interview questions."""
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        play_immediately = data.get('play_immediately', True)
        
        # Generate speech
        audio_path = audio_service.speak_text(text, play_immediately=play_immediately)
        
        return jsonify({
            'success': True,
            'audio_path': audio_path
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate speech: {str(e)}'}), 500

@ai_interview_bp.route('/api/ai-interview/all')
def get_all_interviews():
    """Get all interview sessions (for admin dashboard)."""
    try:
        interviews = ai_interview_service.get_all_interviews()
        
        return jsonify({
            'success': True,
            'interviews': interviews,
            'total_count': len(interviews)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get interviews: {str(e)}'}), 500

@ai_interview_bp.route('/api/ai-interview/<interview_id>/complete', methods=['POST'])
def complete_interview(interview_id):
    """Manually complete an interview session."""
    try:
        result = ai_interview_service.get_interview_status(interview_id)
        
        if 'error' in result:
            return jsonify(result), 404
        
        interview = result['interview']
        interview['status'] = 'completed'
        interview['completed_at'] = datetime.now()
        
        return jsonify({
            'success': True,
            'interview': interview,
            'message': 'Interview completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to complete interview: {str(e)}'}), 500

@ai_interview_bp.route('/api/ai-interview/quick-create', methods=['POST'])
def create_quick_interview():
    """Create a quick AI interview session for direct platform use."""
    try:
        data = request.get_json()
        
        # Validate required fields
        candidate_name = data.get('candidate_name', 'Anonymous Candidate')
        job_role = data.get('job_role', 'General Position')
        resume_text = data.get('resume_text', '')
        
        # Create quick interview session
        interview = ai_interview_service.create_quick_interview(
            candidate_name=candidate_name,
            job_role=job_role,
            resume_text=resume_text
        )
        
        return jsonify({
            'success': True,
            'interview': interview,
            'interview_link': f"/ai-interviewer/{interview['id']}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to create quick interview: {str(e)}'}), 500

# Email invite functionality removed - AI interviews are now direct platform features

@ai_interview_bp.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio files for the interview interface."""
    try:
        audio_path = os.path.join(audio_service.audio_storage_path, filename)
        
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        return send_file(audio_path, mimetype='audio/wav')
        
    except Exception as e:
        return jsonify({'error': f'Failed to serve audio: {str(e)}'}), 500
