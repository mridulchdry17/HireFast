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
    resume_path = db.Column(db.String(255))
    resume_file_id = db.Column(db.String(100))
    similarity_score = db.Column(db.Float)
    ai_interview_status = db.Column(db.String(20), default='not_sent')
    ai_interview_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AIInterviewSession(db.Model):
    """Persistent AI interview session — survives server restarts."""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    application_id = db.Column(db.String(36), db.ForeignKey('application.id'), nullable=True)
    candidate_name = db.Column(db.String(100), nullable=False)
    candidate_email = db.Column(db.String(100), nullable=False)
    job_role = db.Column(db.String(100), nullable=False)
    resume_text = db.Column(db.Text)
    status = db.Column(db.String(20), default='pending')
    total_questions = db.Column(db.Integer, default=5)
    current_question = db.Column(db.Integer, default=0)
    overall_score = db.Column(db.Float)
    conversation_history = db.Column(db.Text)  # JSON string
    expires_at = db.Column(db.DateTime)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    questions = db.relationship('AIInterviewQuestion', backref='session', lazy=True,
                                cascade='all, delete-orphan')

class AIInterviewQuestion(db.Model):
    """Stores each question so evaluator gets the actual question text (not a placeholder)."""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(36), db.ForeignKey('ai_interview_session.id'), nullable=False)
    question_number = db.Column(db.Integer, nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
