"""
Routes package for API endpoints.
"""
from .auth import auth_bp
from .hiring import hiring_bp
from .calendar import calendar_bp
from .ai_interview import ai_interview_bp
from .candidate_portal import candidate_portal_bp

__all__ = ['auth_bp', 'hiring_bp', 'calendar_bp', 'ai_interview_bp', 'candidate_portal_bp']
