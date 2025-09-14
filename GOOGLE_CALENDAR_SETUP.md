# Google Calendar OAuth Setup Guide

This guide will help you set up personal Google Account authentication for the HR automation system, enabling it to send actual email invites to candidates.

## Prerequisites

- A Google Account (personal or Google Workspace)
- Access to Google Cloud Console

## Step-by-Step Setup

### 1. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top
3. Click "New Project"
4. Enter a project name (e.g., "HR-Automation")
5. Click "Create"

### 2. Enable Google Calendar API

1. In your new project, go to "APIs & Services" > "Library"
2. Search for "Google Calendar API"
3. Click on "Google Calendar API"
4. Click "Enable"

### 3. Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. If prompted, configure the OAuth consent screen:
   - User Type: External
   - App name: "HR Automation"
   - User support email: Your email
   - Developer contact information: Your email
   - Save and continue through the steps

4. Create OAuth 2.0 Client ID:
   - Application type: Desktop application
   - Name: "HR Automation Desktop Client"
   - Click "Create"

5. Download the JSON file:
   - Click the download button (⬇️) next to your new client ID
   - Rename the downloaded file to `google_calendar_credentials.json`
   - Place it in your project root directory (same folder as `app.py`)

### 4. Configure OAuth Consent Screen (if needed)

If you haven't configured the consent screen yet:

1. Go to "APIs & Services" > "OAuth consent screen"
2. Choose "External" user type
3. Fill in the required information:
   - App name: "HR Automation"
   - User support email: Your email
   - Developer contact information: Your email
4. Add scopes:
   - Click "Add or Remove Scopes"
   - Search for and add:
     - `https://www.googleapis.com/auth/calendar`
     - `https://www.googleapis.com/auth/calendar.events`
5. Add test users (your email address)
6. Save and continue

### 5. Test the Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run your Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and go to `http://127.0.0.1:5000`

4. Click "🔐 Authenticate Google Calendar" in the HR Automation section

5. Follow the OAuth flow:
   - A browser window will open
   - Sign in with your Google Account
   - Grant the requested permissions
   - You'll be redirected back to the application

## How It Works

### Before (Service Account)
- ❌ Could not send email invites to external users
- ❌ Required manual invitation to calendar events
- ❌ Limited by Google's service account restrictions

### After (Personal OAuth)
- ✅ Can send actual email invites to candidates
- ✅ Automatic calendar notifications
- ✅ Full calendar integration capabilities

## File Structure

After setup, you should have these files in your project:

```
Agentic-HR/
├── app.py
├── index.html
├── credentials.json              # Service account (for Sheets/Drive)
├── google_calendar_credentials.json  # OAuth credentials (for Calendar)
├── google_calendar_token.pickle     # Generated after first auth
└── requirements.txt
```

## Troubleshooting

### "Google Calendar credentials file not found"
- Make sure you downloaded the OAuth credentials JSON file
- Rename it to `google_calendar_credentials.json`
- Place it in the project root directory

### "Access blocked" during OAuth
- Make sure you added your email as a test user in the OAuth consent screen
- If using a Google Workspace account, contact your admin

### "Invalid client" error
- Check that the `google_calendar_credentials.json` file is in the correct location
- Verify the file contains valid OAuth 2.0 credentials

### Calendar events not sending emails
- Ensure you're using the personal OAuth flow (not service account)
- Check that the Google Calendar API is enabled
- Verify the OAuth consent screen has the correct scopes

## Security Notes

- Keep your `google_calendar_credentials.json` file secure
- The `google_calendar_token.pickle` file contains your access tokens
- Never commit these files to version control
- Add them to your `.gitignore` file

## Benefits

With this setup, your HR automation system can:

1. **Send Real Email Invites**: Candidates receive actual calendar invitations
2. **Automatic Notifications**: Google Calendar handles all email notifications
3. **Professional Experience**: Candidates get proper calendar events with all details
4. **No Manual Work**: No need to manually invite candidates to events

## Next Steps

Once authenticated, you can:

1. Select the best candidates from resumes
2. Schedule interviews with one click
3. Send professional calendar invites automatically
4. Track interview responses and attendance

The system will now provide a complete end-to-end HR automation experience! 