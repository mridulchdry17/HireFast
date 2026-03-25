"""
Configuration management for the HireFast application.
"""
import os
from dotenv import load_dotenv

# backend/ directory (parent of app/)
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_env_file = os.path.join(_BASE_DIR, '.env')
if os.path.isfile(_env_file):
    load_dotenv(_env_file, override=True)

class Config:
    """Base configuration class."""
    
    # CORS: comma-separated origins for split FE (Vercel) + BE (VM). Use * only for local hacks.
    # Example: https://hirefast.vercel.app,http://localhost:3000
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000')
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    # Public base URL for OAuth/callback redirects (VM IP/domain in prod, localhost in dev)
    APP_BASE_URL = os.environ.get('APP_BASE_URL', 'http://127.0.0.1:5000').rstrip('/')
    
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
    LINKEDIN_REDIRECT_URI = os.environ.get(
        "LINKEDIN_REDIRECT_URI", f"{APP_BASE_URL}/callback"
    ).strip()
    LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
    LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
    
    # Google Calendar OAuth settings
    GOOGLE_CALENDAR_SCOPES = [
        'https://www.googleapis.com/auth/calendar',
        'https://www.googleapis.com/auth/calendar.events'
    ]
    GOOGLE_CALENDAR_CREDENTIALS_FILE = os.path.join(BASE_DIR, 'credentials', 'google_calendar_credentials.json')
    GOOGLE_CALENDAR_TOKEN_FILE = os.path.join(BASE_DIR, 'google_calendar_token.pickle')
    
    # Google Sheets settings
    GOOGLE_SHEET_ID = "1PrMvW7un7b1zdF_262QwtaP7_xNhf5zxylYIqnOSq_U"
    DRIVE_FOLDER_ID = "1lFh4bkL2PYtyjFcVMiw0SuVEEeP3hXMNuCH4zQgGb8UOyLO1fxs528Gq5jxV8z4lsH3h-9KF"
    CREDENTIALS_PATH = os.path.join(BASE_DIR, 'credentials', 'credentials.json')
    
    # AI/LLM settings
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY")
    # Composio dashboard: Auth Config for Google Calendar (required for connect-calendar / link flow)
    COMPOSIO_GOOGLE_CALENDAR_AUTH_CONFIG_ID = os.environ.get(
        "COMPOSIO_GOOGLE_CALENDAR_AUTH_CONFIG_ID", ""
    ).strip()
    # Optional full override; defaults to APP_BASE_URL + /scheduling.
    COMPOSIO_CALENDAR_REDIRECT_URL = os.environ.get(
        "COMPOSIO_CALENDAR_REDIRECT_URL", f"{APP_BASE_URL}/scheduling"
    ).strip()
    
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
