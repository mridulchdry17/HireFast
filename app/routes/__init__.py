"""
Routes package for API endpoints.
"""
from .auth import auth_bp
from .hiring import hiring_bp
from .calendar import calendar_bp

__all__ = ['auth_bp', 'hiring_bp', 'calendar_bp']
