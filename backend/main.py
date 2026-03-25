"""
Main application entry point for HireFast.

Production (VM), from repo root: gunicorn --chdir backend -w 2 -b 0.0.0.0:5000 "main:app"
Or: cd backend && gunicorn -w 2 -b 0.0.0.0:5000 "main:app"
Set FLASK_CONFIG=production and CORS_ORIGINS for split FE/BE deploy.
"""
import os
from app import create_app

_config = os.environ.get('FLASK_CONFIG', 'default')
app = create_app(_config)

if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(host=host, port=port, debug=debug)