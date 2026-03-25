from flask import Blueprint, render_template, request, jsonify, current_app
from app.services.job_service import JobService
from app.services.resume_parser_service import ResumeParserService
import os
import uuid
from werkzeug.utils import secure_filename

candidate_portal_bp = Blueprint('candidate_portal', __name__)

@candidate_portal_bp.route('/apply/<job_id>', methods=['GET'])
def application_page(job_id):
    """Display the application form for a specific job."""
    job = JobService.get_job_by_id(job_id)
    if not job:
        return "Job not found", 404
    return render_template('apply.html', job=job)

@candidate_portal_bp.route('/apply/<job_id>', methods=['POST'])
def submit_application(job_id):
    """Handle job application submission."""
    job = JobService.get_job_by_id(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    try:
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        
        if not name or not email or 'resume' not in request.files:
            return jsonify({'error': 'Name, email, and resume are required'}), 400
            
        resume_file = request.files['resume']
        if not resume_file or resume_file.filename == '':
            return jsonify({'error': 'No resume file selected'}), 400
            
        # Save resume locally
        filename = secure_filename(f"{uuid.uuid4().hex[:8]}_{resume_file.filename}")
        resume_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(resume_path)
        
        # Parse resume and score
        resume_parser = ResumeParserService()
        try:
            with open(resume_path, 'rb') as f:
                pdf_bytes = f.read()
            resume_text = resume_parser.extract_text_from_pdf_bytes(pdf_bytes)
            
            # Use AI Service for similarity scoring (basic implementation)
            from app.services.ai_service import AIService
            ai_service = AIService()
            
            # Simple scoring - we can refine this later
            # For now, let's just store the application
            score = 0.0 # Default
        except Exception as e:
            print(f"Scoring error: {e}")
            score = 0.0
            
        # Save application using JobService (Clean Architecture)
        new_app = JobService.create_application(
            job_id=job.id,
            name=name,
            email=email,
            resume_path=filename,
            resume_text=resume_text if 'resume_text' in locals() else ""
        )
        
        return jsonify({
            'success': True, 
            'message': 'Application submitted successfully!',
            'application_id': new_app.id
        })
        
    except Exception as e:
        print(f"Submission error: {e}")
        return jsonify({'error': f'Failed to submit application: {str(e)}'}), 500
