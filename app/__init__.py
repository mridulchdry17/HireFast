"""
Flask application factory for HireFast.
"""
from flask import Flask, render_template
from app.config import config
from app.routes import auth_bp, hiring_bp, calendar_bp
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
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(hiring_bp)
    app.register_blueprint(calendar_bp)
    
    # Main route
    @app.route('/')
    def landing_page():
        return render_template('index.html')
    
    return app
