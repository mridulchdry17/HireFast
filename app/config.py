"""
Configuration management for the HireFast application.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database settings
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        f"sqlite:///{os.path.join(BASE_DIR, 'hirefast.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Storage settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit
    
    # LinkedIn OAuth 2.0 settings
    LINKEDIN_CLIENT_ID = os.environ.get("LINKEDIN_CLIENT_ID")
    LINKEDIN_CLIENT_SECRET = os.environ.get("LINKEDIN_CLIENT_SECRET")
    LINKEDIN_REDIRECT_URI = "http://127.0.0.1:5000/callback"
    LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
    LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
    
    # Google Calendar OAuth settings
    GOOGLE_CALENDAR_SCOPES = [
        'https://www.googleapis.com/auth/calendar',
        'https://www.googleapis.com/auth/calendar.events'
    ]
    GOOGLE_CALENDAR_CREDENTIALS_FILE = 'credentials/google_calendar_credentials.json'
    GOOGLE_CALENDAR_TOKEN_FILE = 'google_calendar_token.pickle'
    
    # Google Sheets settings
    GOOGLE_SHEET_ID = "1PrMvW7un7b1zdF_262QwtaP7_xNhf5zxylYIqnOSq_U"
    DRIVE_FOLDER_ID = "1lFh4bkL2PYtyjFcVMiw0SuVEEeP3hXMNuCH4zQgGb8UOyLO1fxs528Gq5jxV8z4lsH3h-9KF"
    CREDENTIALS_PATH = "credentials/credentials.json"
    
    # AI/LLM settings
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY")
    
    # Application form links
    ROLE_FORM_LINKS = {
        "ai": "https://forms.gle/8R5Eg8TaQHgQBFE19",
        "machine learning": "https://forms.gle/8R5Eg8TaQHgQBFE19",
        "data": "https://forms.gle/8R5Eg8TaQHgQBFE19",
        "marketing": "https://forms.gle/8R5Eg8TaQHgQBFE19",
    }
    GENERIC_FORM_LINK = "https://forms.gle/8R5Eg8TaQHgQBFE19"

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
