from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()

class JobPosting(db.Model):
    """Model for Job Postings."""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    role = db.Column(db.String(100), nullable=False)
    company_name = db.Column(db.String(100))
    location = db.Column(db.String(100))
    employment_type = db.Column(db.String(50))
    job_description = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    applications = db.relationship('Application', backref='job', lazy=True)

class Application(db.Model):
    """Model for Candidate Applications."""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = db.Column(db.String(36), db.ForeignKey('job_posting.id'), nullable=False)
    candidate_name = db.Column(db.String(100), nullable=False)
    candidate_email = db.Column(db.String(100), nullable=False)
    resume_path = db.Column(db.String(255))  # Local path or cloud link
    resume_file_id = db.Column(db.String(100)) # Drive file ID if used
    similarity_score = db.Column(db.Float)
    ai_interview_status = db.Column(db.String(20), default='not_sent')
    ai_interview_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
