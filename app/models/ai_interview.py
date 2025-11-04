"""
AI Interview models for the hiring workflow.
"""
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime

class AIInterview(TypedDict):
    """Model for AI interview session."""
    id: Optional[str]
    candidate_id: str
    candidate_name: str
    candidate_email: str
    job_role: str
    resume_text: str
    status: str  # 'pending', 'in_progress', 'completed', 'cancelled'
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_questions: int
    current_question: int
    overall_score: Optional[float]
    interview_link: str
    created_at: datetime

class InterviewQuestion(TypedDict):
    """Model for interview questions."""
    id: Optional[str]
    interview_id: str
    question_number: int
    question_text: str
    question_type: str  # 'technical', 'behavioral', 'situational'
    context: Optional[str]  # Additional context for the question
    created_at: datetime

class InterviewAnswer(TypedDict):
    """Model for candidate answers."""
    id: Optional[str]
    interview_id: str
    question_id: str
    question_number: int
    answer_text: str
    answer_audio_path: Optional[str]  # Path to audio file if voice answer
    answer_type: str  # 'text', 'voice', 'mixed'
    duration_seconds: Optional[int]  # For voice answers
    created_at: datetime

class InterviewEvaluation(TypedDict):
    """Model for AI evaluation of answers."""
    id: Optional[str]
    interview_id: str
    question_id: str
    answer_id: str
    score: float  # 0-10 scale
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    evaluation_criteria: Dict[str, float]  # Detailed scoring breakdown
    created_at: datetime

class InterviewSession(TypedDict):
    """Complete interview session with all components."""
    interview: AIInterview
    questions: List[InterviewQuestion]
    answers: List[InterviewAnswer]
    evaluations: List[InterviewEvaluation]
    conversation_history: List[Dict[str, str]]  # For LLM context
