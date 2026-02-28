"""
Flask application factory for HireFast.
"""
from flask import Flask, render_template, jsonify, request
from app.config import config
from app.routes import auth_bp, hiring_bp, calendar_bp, ai_interview_bp, candidate_portal_bp
from app.models.db_models import db
import os

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
    
    # Initialize DB
    db.init_app(app)
    
    # Create upload and db folders
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    with app.app_context():
        db.create_all()
    
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
        """Create an AI interview for a candidate."""
        from app.services.ai_interview_service import AIInterviewService
        from app.services.resume_parser_service import ResumeParserService
        from app.services.google_service import GoogleService
        
        data = request.get_json()
        if not data or 'candidate_email' not in data or 'candidate_name' not in data:
            return jsonify({'error': 'Candidate email and name are required'}), 400
        
        candidate_email = data['candidate_email']
        candidate_name = data['candidate_name']
        job_role = data.get('job_role', 'Software Engineer')
        resume_file_id = data.get('resume_file_id', '')
        
        try:
            # Create AI interview session
            ai_interview_service = AIInterviewService()
            resume_parser_service = ResumeParserService()
            google_service = GoogleService()
            
            # Get resume text if available
            resume_text = ""
            if resume_file_id:
                try:
                    pdf_bytes = google_service.download_pdf_from_drive(resume_file_id)
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
                candidate_id=data.get('candidate_id', candidate_email),
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                job_role=job_role,
                resume_text=resume_text
            )
            
            return jsonify({
                'status': 'success',
                'message': f'AI interview created for {candidate_name}',
                'interview': interview,
                'interview_link': f"/ai-interviewer/{interview['id']}"
            })
            
        except Exception as e:
            return jsonify({'error': f'Failed to create AI interview: {str(e)}'}), 500

    @app.route('/create-direct-ai-interview', methods=['POST'])
    def create_direct_ai_interview_api():
        """Create a direct AI interview with resume upload."""
        from werkzeug.utils import secure_filename
        import os
        import uuid
        
        print(f"🔍 DEBUG: create_direct_ai_interview_api called")
        
        try:
            # Get form data
            candidate_name = request.form.get('candidate_name')
            candidate_email = request.form.get('candidate_email')
            job_role = request.form.get('job_role')
            experience_level = request.form.get('experience_level', 'Mid Level')
            resume_text = request.form.get('resume_text', '')
            question_count = int(request.form.get('question_count', 5))
            interview_type = request.form.get('interview_type', 'mixed')
            candidate_id = request.form.get('candidate_id', candidate_email)
            
            print(f"🔍 DEBUG: Form data - Name: {candidate_name}, Email: {candidate_email}, Role: {job_role}")
            
            if not candidate_name or not candidate_email or not job_role:
                print(f"🔍 DEBUG: Missing required fields")
                return jsonify({'error': 'Missing required fields'}), 400
            
            # Handle file upload
            resume_file_path = None
            if 'resume_file' in request.files:
                file = request.files['resume_file']
                if file and file.filename:
                    # Save uploaded file
                    filename = secure_filename(f"{candidate_id}_{uuid.uuid4().hex[:8]}_{file.filename}")
                    upload_dir = "static/uploads"
                    os.makedirs(upload_dir, exist_ok=True)
                    file_path = os.path.join(upload_dir, filename)
                    file.save(file_path)
                    resume_file_path = file_path
                    
                    # Extract text from uploaded file
                    try:
                        from app.services.resume_parser_service import ResumeParserService
                        resume_parser = ResumeParserService()
                        
                        if file.filename.lower().endswith('.pdf'):
                            with open(file_path, 'rb') as f:
                                pdf_bytes = f.read()
                            extracted_text = resume_parser.extract_text_from_pdf_bytes(pdf_bytes)
                        else:
                            # For DOC/DOCX files, you would need additional libraries
                            extracted_text = "Resume file uploaded (text extraction for DOC/DOCX not implemented yet)"
                        
                        if extracted_text and extracted_text.strip():
                            resume_text = extracted_text
                    except Exception as e:
                        print(f"Error extracting text from file: {e}")
                        resume_text = "Resume file uploaded but text extraction failed"
            
            # Create AI interview session
            from app.services.ai_interview_service import AIInterviewService
            
            ai_interview_service = AIInterviewService()
            
            print(f"🔍 DEBUG: Creating interview session...")
            
            # Create interview session
            interview = ai_interview_service.create_interview_session(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                job_role=job_role,
                resume_text=resume_text
            )
            
            # Update interview settings
            interview['total_questions'] = question_count
            interview['interview_type'] = interview_type
            interview['experience_level'] = experience_level
            
            print(f"🔍 DEBUG: Interview created successfully with ID: {interview['id']}")
            print(f"🔍 DEBUG: Returning interview link: /ai-interviewer/{interview['id']}")
            
            return jsonify({
                'status': 'success',
                'message': f'AI interview created for {candidate_name}',
                'interview': interview,
                'interview_link': f"/ai-interviewer/{interview['id']}"
            })
            
        except Exception as e:
            return jsonify({'error': f'Failed to create direct AI interview: {str(e)}'}), 500
    
    return app
