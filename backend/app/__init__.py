"""
Flask application factory for HireFast.
"""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from app.config import config
from app.routes import auth_bp, hiring_bp, calendar_bp, ai_interview_bp, candidate_portal_bp
from app.models.db_models import db
import os
import threading


def _ensure_ai_interview_last_activity_column():
    """SQLite: add last_activity_at if DB was created before this column existed."""
    try:
        from sqlalchemy import inspect, text

        insp = inspect(db.engine)
        cols = [c["name"] for c in insp.get_columns("ai_interview_session")]
        if "last_activity_at" in cols:
            return
        with db.engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE ai_interview_session ADD COLUMN last_activity_at DATETIME"
                )
            )
    except Exception as e:
        print(f"Note: ai_interview_session.last_activity_at migration: {e}")


def _start_idle_interview_audio_cleanup(app):
    """Every ~10 minutes, delete static/audio/{session_id}_* for idle pending/in_progress sessions."""

    def worker():
        import time
        from app.services.ai_interview_service import AIInterviewService

        while True:
            time.sleep(600)
            with app.app_context():
                try:
                    n = AIInterviewService().cleanup_idle_session_audio_files(idle_minutes=30)
                    if n:
                        print(f"[audio cleanup] removed {n} idle session file(s)")
                except Exception as exc:
                    print(f"[audio cleanup] error: {exc}")

    threading.Thread(target=worker, daemon=True).start()


def create_app(config_name='default'):
    """
    Create and configure the Flask application.
    
    Args:
        config_name: Configuration name ('development', 'production', 'default')
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
                static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'))
    app.config.from_object(config[config_name])
    
    # Split deployment: allow Vercel / local Next.js to call the API (set CORS_ORIGINS in production)
    _origins = app.config.get('CORS_ORIGINS') or ''
    if _origins.strip() == '*':
        CORS(app, resources={r'/*': {'origins': '*'}})
    else:
        origin_list = [o.strip() for o in _origins.split(',') if o.strip()]
        if origin_list:
            CORS(app, resources={r'/*': {'origins': origin_list}}, supports_credentials=True)
    
    # Initialize DB
    db.init_app(app)
    
    # Create upload and db folders
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    with app.app_context():
        db.create_all()
        _ensure_ai_interview_last_activity_column()

    _start_idle_interview_audio_cleanup(app)
    
    @app.route('/health')
    def health():
        """Load balancer / deployment check for split FE/BE."""
        return jsonify({'status': 'ok', 'service': 'hirefast-api'})
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(hiring_bp)
    app.register_blueprint(calendar_bp)
    app.register_blueprint(ai_interview_bp)
    app.register_blueprint(candidate_portal_bp)
    
    # Main routes
    @app.route('/')
    def landing_page():
        return render_template('index.html')
    
    @app.route('/dashboard')
    def dashboard():
        return render_template('dashboard.html')
    
    @app.route('/job-posting')
    def job_posting():
        return render_template('job-posting.html')
    
    @app.route('/candidates')
    def candidates():
        return render_template('candidates.html')
    
    @app.route('/scheduling')
    def scheduling():
        return render_template('scheduling.html')
    
    @app.route('/analytics')
    def analytics():
        return render_template('analytics.html')
    
    @app.route('/settings')
    def settings():
        return render_template('settings.html')
    
    @app.route('/contact')
    def contact():
        return render_template('contact.html')
    
    @app.route('/ai-interview-demo')
    def ai_interview_demo():
        return render_template('ai_interview_demo.html')
    
    @app.route('/workflow')
    def workflow():
        return render_template('workflow.html')
    
    @app.route('/direct-ai-interview')
    def direct_ai_interview():
        return render_template('direct_ai_interview.html')
    
    # API Routes
    @app.route('/create-ai-interview', methods=['POST'])
    def create_ai_interview_api():
        """Create an AI interview for a candidate (legacy endpoint)."""
        from app.services.ai_interview_service import AIInterviewService
        data = request.get_json()
        if not data or 'candidate_email' not in data or 'candidate_name' not in data:
            return jsonify({'error': 'candidate_email and candidate_name are required'}), 400

        try:
            ai_interview_service = AIInterviewService()
            interview = ai_interview_service.create_interview_session(
                application_id=data.get('application_id', ''),
                candidate_name=data['candidate_name'],
                candidate_email=data['candidate_email'],
                job_role=data.get('job_role', 'Software Engineer'),
                resume_path=data.get('resume_path', ''),
            )
            return jsonify({
                'status': 'success',
                'message': f'AI interview created for {data["candidate_name"]}',
                'interview': interview,
                'interview_link': interview['interview_link'],
            })
        except Exception as e:
            return jsonify({'error': f'Failed to create AI interview: {str(e)}'}), 500

    @app.route('/create-direct-ai-interview', methods=['POST'])
    def create_direct_ai_interview_api():
        """Create a direct AI interview with optional resume text (for demo page)."""
        from werkzeug.utils import secure_filename
        import uuid

        try:
            candidate_name = request.form.get('candidate_name') or (request.get_json() or {}).get('candidate_name')
            candidate_email = request.form.get('candidate_email') or (request.get_json() or {}).get('candidate_email')
            job_role = request.form.get('job_role') or (request.get_json() or {}).get('job_role')

            if not candidate_name or not candidate_email or not job_role:
                return jsonify({'error': 'Missing required fields'}), 400

            # Save uploaded resume to uploads/ if provided
            resume_path = ''
            if 'resume_file' in request.files:
                file = request.files['resume_file']
                if file and file.filename:
                    filename = secure_filename(f"{uuid.uuid4().hex[:8]}_{file.filename}")
                    upload_dir = app.config.get('UPLOAD_FOLDER', 'uploads')
                    os.makedirs(upload_dir, exist_ok=True)
                    file.save(os.path.join(upload_dir, filename))
                    resume_path = filename

            from app.services.ai_interview_service import AIInterviewService
            ai_interview_service = AIInterviewService()
            interview = ai_interview_service.create_interview_session(
                application_id='',
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                job_role=job_role,
                resume_path=resume_path,
            )
            return jsonify({
                'status': 'success',
                'message': f'AI interview created for {candidate_name}',
                'interview': interview,
                'interview_link': interview['interview_link'],
            })
        except Exception as e:
            return jsonify({'error': f'Failed to create direct AI interview: {str(e)}'}), 500
    
    return app
