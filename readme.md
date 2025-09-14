# HireFast - AI-Powered HR Automation Platform

A modern, structured Flask application that automates the hiring process using AI for job description generation, candidate selection, and interview scheduling.

## 🏗️ Project Structure

```
HireFast/
├── app/                          # Main application package
│   ├── __init__.py              # Flask app factory
│   ├── config.py                # Configuration management
│   ├── models/                  # Data models
│   │   ├── __init__.py
│   │   └── hiring.py           # HR hiring state models
│   ├── services/                # Business logic
│   │   ├── __init__.py
│   │   ├── linkedin_service.py  # LinkedIn API operations
│   │   ├── google_service.py    # Google Sheets/Calendar operations
│   │   ├── resume_service.py    # Resume processing
│   │   └── ai_service.py        # AI/LLM operations
│   ├── routes/                  # API routes
│   │   ├── __init__.py
│   │   ├── auth.py             # Authentication routes
│   │   ├── hiring.py           # Hiring workflow routes
│   │   └── calendar.py         # Calendar integration routes
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── helpers.py          # Helper functions
├── templates/                   # HTML templates
│   ├── base.html
│   └── index_new.html
├── static/                      # Static assets
│   ├── css/
│   │   └── main.css
│   ├── js/
│   │   └── main.js
│   └── images/
├── credentials/                 # Credential files
│   ├── google_calendar_credentials.json
│   └── credentials.json
├── main.py                      # Application entry point
├── requirements.txt
├── env.example
└── README_NEW.md
```

## 🚀 Features

- **AI-Powered Job Description Generation**: Generate professional job descriptions using Groq's LLM
- **LinkedIn Integration**: Post job descriptions directly to LinkedIn
- **Resume Processing**: Automatically analyze and rank candidates based on resume content
- **Google Sheets Integration**: Fetch applications from Google Forms responses
- **Calendar Scheduling**: Schedule interviews with Google Calendar integration
- **Modern UI**: Clean, responsive interface with glass morphism design

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd HireFast
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your actual credentials
   ```

5. **Set up Google credentials**
   - Place your Google service account credentials in `credentials/credentials.json`
   - Place your Google Calendar OAuth credentials in `credentials/google_calendar_credentials.json`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Flask Configuration
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# LinkedIn OAuth Configuration
LINKEDIN_CLIENT_ID=your-linkedin-client-id
LINKEDIN_CLIENT_SECRET=your-linkedin-client-secret

# AI/LLM Configuration
GROQ_API_KEY=your-groq-api-key
```

### Google Services Setup

1. **Google Sheets API**:
   - Create a service account in Google Cloud Console
   - Download the JSON credentials file
   - Place it in `credentials/credentials.json`
   - Share your Google Sheet with the service account email

2. **Google Calendar API**:
   - Enable Google Calendar API in Google Cloud Console
   - Create OAuth 2.0 credentials (Desktop application)
   - Download the JSON file and place it in `credentials/google_calendar_credentials.json`

### LinkedIn App Setup

1. Create a LinkedIn app at https://www.linkedin.com/developers/
2. Add the redirect URI: `http://127.0.0.1:5000/callback`
3. Request the following scopes: `openid`, `profile`, `w_member_social`

## 🏃‍♂️ Running the Application

```bash
python main.py
```

The application will be available at `http://127.0.0.1:5000`

## 📚 API Endpoints

### Authentication
- `GET /login` - Initiate LinkedIn OAuth flow
- `GET /callback` - Handle OAuth callback
- `GET /logout` - Logout user
- `GET /check-auth` - Check authentication status

### Hiring Workflow
- `POST /generate-jd` - Generate job description
- `POST /approve-jd` - Approve/reject job description
- `POST /post-jd` - Post job description to LinkedIn
- `GET /fetch-applications` - Fetch applications from Google Sheet
- `POST /select-best-resumes` - Select best candidates

### Calendar Integration
- `GET /google-calendar-auth` - Initiate Google Calendar OAuth
- `GET /check-google-calendar-auth` - Check Calendar auth status
- `POST /schedule-interview` - Schedule interview

## 🏗️ Architecture

The application follows a modular, service-oriented architecture:

- **Models**: Define data structures and state management
- **Services**: Contain business logic and external API integrations
- **Routes**: Handle HTTP requests and responses
- **Utils**: Provide helper functions and utilities
- **Config**: Centralized configuration management

## 🔧 Development

### Adding New Features

1. **New Service**: Add to `app/services/`
2. **New Route**: Add to `app/routes/` and register in `app/__init__.py`
3. **New Model**: Add to `app/models/`
4. **New Utility**: Add to `app/utils/`

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Document functions and classes
- Keep functions small and focused

## 🚀 Deployment

### Production Configuration

1. Set `FLASK_DEBUG=False` in your environment
2. Use a production WSGI server like Gunicorn
3. Set up proper logging
4. Use environment variables for sensitive data
5. Set up SSL/TLS certificates

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the code comments

## 🔄 Migration from Old Structure

If you're migrating from the old monolithic structure:

1. Your existing `.env` file will work
2. Move credential files to the `credentials/` directory
3. Update any hardcoded paths in your configuration
4. Test all functionality to ensure everything works

The new structure provides better maintainability, testability, and scalability while preserving all existing functionality.
