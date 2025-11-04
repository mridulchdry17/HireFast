"""
Data models for the hiring workflow.
"""
from typing import TypedDict, Optional, List, Dict, Any

class HRHiringState(TypedDict):
    """State model for the HR hiring workflow."""
    role: Optional[str]
    job_description: Optional[str]
    approval: Optional[bool]
    post_status: Optional[str]
    error: Optional[str]
    top_candidates: Optional[List[Dict[str, Any]]]
    post_ids: Optional[List[Dict[str, str]]]

class Applicant(TypedDict):
    """Model for applicant data."""
    name: str
    college: str
    email: str
    intro: str
    resume_file_id: str

class Candidate(TypedDict):
    """Model for candidate with similarity score."""
    applicant: Applicant
    similarity_score: float
    ai_interview_status: Optional[str]  # 'not_sent', 'sent', 'in_progress', 'completed'
    ai_interview_score: Optional[float]
    ai_interview_link: Optional[str]
