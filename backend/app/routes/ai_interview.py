"""
AI Interview routes — updated to use the new DB-backed AIInterviewService.
"""
import base64
import os
import uuid
from flask import Blueprint, request, jsonify, render_template, redirect
from werkzeug.utils import secure_filename
from app.services.ai_interview_service import AIInterviewService
from app.services.audio_service import AudioService
from app.services.job_service import JobService

ai_interview_bp = Blueprint('ai_interview', __name__)

ai_interview_service = AIInterviewService()
audio_service = AudioService()


# ─── Pages ──────────────────────────────────────────────────────────────────────

@ai_interview_bp.route('/ai-interviewer/<session_id>')
def ai_interviewer_page(session_id):
    """Candidate-facing AI interview page."""
    status_data = ai_interview_service.get_interview_status(session_id)
    if 'error' in status_data:
        return redirect('/direct-ai-interview')
    
    session = status_data['session']
    if status_data.get('is_expired') and session['status'] != 'completed':
        return render_template('ai_interview.html', 
                             interview=session, 
                             is_expired=True,
                             error="This interview link has expired. Please contact the recruiter to extend the deadline.")

    return render_template('ai_interview.html', 
                           interview=session, 
                           conversation_history=status_data.get('conversation_history', []))


# ─── REST API ────────────────────────────────────────────────────────────────────

@ai_interview_bp.route('/api/ai-interview/create', methods=['POST'])
def create_ai_interview():
    """Create an AI interview session for a candidate from the DB."""
    try:
        data = request.get_json()

        # Fetch the application from DB to get resume_path
        application_id = data.get('application_id', '')
        all_apps = JobService.get_all_applications()
        candidate_app = next((a for a in all_apps if str(a['id']) == str(application_id)), None)

        if not candidate_app and not data.get('candidate_name'):
            return jsonify({'error': 'application_id or candidate_name required'}), 400

        candidate_name = data.get('candidate_name') or candidate_app['name']
        candidate_email = data.get('candidate_email') or candidate_app.get('email', '')
        job_role = data.get('job_role') or candidate_app.get('role', 'Position')
        resume_path = data.get('resume_path') or (candidate_app.get('resume_path') if candidate_app else '')

        session = ai_interview_service.create_interview_session(
            application_id=application_id,
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            job_role=job_role,
            resume_path=resume_path,
        )

        return jsonify({
            'success': True,
            'session': session,
            'interview_link': session['interview_link'],
        })

    except Exception as e:
        return jsonify({'error': f'Failed to create interview: {str(e)}'}), 500


@ai_interview_bp.route('/api/ai-interview/<session_id>/start', methods=['POST'])
def start_interview(session_id):
    """Start an AI interview session and get the first question."""
    try:
        result = ai_interview_service.start_interview(session_id)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_interview_bp.route('/api/ai-interview/<session_id>/answer', methods=['POST'])
def submit_answer(session_id):
    """Submit a text or voice answer and get evaluation + next question."""
    try:
        answer_text = ''
        question_id = ''

        if request.content_type and 'multipart' in request.content_type:
            answer_text = request.form.get('answer_text', '')
            question_id = request.form.get('question_id', '')

            if 'audio_file' in request.files:
                audio_file = request.files['audio_file']
                if audio_file and audio_file.filename and not answer_text:
                    filename = secure_filename(f"{session_id}_{uuid.uuid4().hex[:8]}.wav")
                    audio_path = os.path.join(audio_service.audio_storage_path, filename)
                    audio_file.save(audio_path)
                    try:
                        answer_text = audio_service.speech_to_text(audio_path)
                    except Exception as e:
                        return jsonify({'error': f'Speech-to-text failed: {str(e)}'}), 500
                    finally:
                        # Upload temp is only needed for Whisper; do not keep voice blobs on disk
                        if audio_path and os.path.exists(audio_path):
                            audio_service.cleanup_audio_file(audio_path)
        else:
            data = request.get_json() or {}
            answer_text = data.get('answer_text', '')
            question_id = data.get('question_id', '')

        if not answer_text:
            return jsonify({'error': 'answer_text is required'}), 400

        result = ai_interview_service.submit_answer(session_id, question_id, answer_text)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify({'success': True, **result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_interview_bp.route('/api/ai-interview/<session_id>/status')
def get_interview_status(session_id):
    """Get current interview status and conversation history."""
    try:
        result = ai_interview_service.get_interview_status(session_id)
        if 'error' in result:
            return jsonify(result), 404
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_interview_bp.route('/api/ai-interview/by-application/<application_id>')
def get_session_by_application(application_id):
    """Get the interview session for a specific application (for UI status badge)."""
    try:
        session = ai_interview_service.get_session_by_application(application_id)
        if session:
            return jsonify({'success': True, 'session': session})
        return jsonify({'success': True, 'session': None})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_interview_bp.route('/api/ai-interview/all')
def get_all_interviews():
    """Get all interview sessions (admin view)."""
    try:
        interviews = ai_interview_service.get_all_interviews()
        return jsonify({'success': True, 'interviews': interviews, 'total': len(interviews)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_interview_bp.route('/api/ai-interview/<session_id>/speak', methods=['POST'])
def speak_text(session_id):
    """Generate TTS in memory and return base64 for the browser (no static/audio file)."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        ai_interview_service.touch_session_activity(session_id)
        mp3_bytes = audio_service.text_to_speech_mp3_bytes(text)
        audio_b64 = base64.b64encode(mp3_bytes).decode("ascii")
        return jsonify({"success": True, "audio_base64": audio_b64, "mime": "audio/mpeg"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ai_interview_bp.route('/audio/<filename>')
def serve_audio(filename):
    """Serve recorded audio files."""
    from flask import send_file
    audio_path = os.path.join(audio_service.audio_storage_path, filename)
    if not os.path.exists(audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(audio_path, mimetype='audio/wav')
@ai_interview_bp.route('/api/ai-interview/<session_id>/extend', methods=['POST'])
def extend_interview(session_id):
    """Extend the interview deadline (Admin only)."""
    try:
        data = request.get_json() or {}
        hours = data.get('hours', 48)
        result = ai_interview_service.extend_expiration(session_id, hours)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
