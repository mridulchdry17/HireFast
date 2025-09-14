"""
Google services for Sheets and Calendar integration.
"""
import os
import pickle
import io
import tempfile
from typing import List, Dict, Optional, Any
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from app.config import Config

class GoogleService:
    """Service for Google API operations."""
    
    def __init__(self):
        self.calendar_credentials = None
    
    def get_google_calendar_credentials(self) -> Any:
        """
        Get Google Calendar credentials using OAuth 2.0 for personal account.
        Returns credentials that can send email invites.
        """
        creds = None
        
        # Check if we have a valid token file
        if os.path.exists(Config.GOOGLE_CALENDAR_TOKEN_FILE):
            with open(Config.GOOGLE_CALENDAR_TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(Config.GOOGLE_CALENDAR_CREDENTIALS_FILE):
                    return {
                        'error': 'Google Calendar credentials file not found. Please download OAuth 2.0 credentials from Google Cloud Console.'
                    }
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    Config.GOOGLE_CALENDAR_CREDENTIALS_FILE, 
                    Config.GOOGLE_CALENDAR_SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(Config.GOOGLE_CALENDAR_TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
        
        return creds
    
    def check_calendar_auth(self) -> Dict:
        """Check if Google Calendar is authenticated."""
        try:
            creds = self.get_google_calendar_credentials()
            
            if isinstance(creds, dict) and 'error' in creds:
                return {
                    'authenticated': False,
                    'error': creds['error']
                }
            
            # Test the credentials by trying to access calendar
            service = build('calendar', 'v3', credentials=creds)
            calendar_list = service.calendarList().list().execute()
            
            return {
                'authenticated': True,
                'calendars': [cal['summary'] for cal in calendar_list.get('items', [])]
            }
            
        except Exception as e:
            return {
                'authenticated': False,
                'error': str(e)
            }
    
    def fetch_applications(self, range_name: str = 'Form Responses 1') -> List[Dict]:
        """
        Read applicant data from Google Sheet and extract Drive file IDs for resumes.
        Returns a list of dicts: {name, college, email, intro, resume_file_id}
        """
        from app.utils.helpers import extract_drive_file_id
        
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        creds = service_account.Credentials.from_service_account_file(
            Config.CREDENTIALS_PATH, scopes=SCOPES
        )
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=Config.GOOGLE_SHEET_ID, 
            range=range_name
        ).execute()
        values = result.get('values', [])
        
        print(f"Raw sheet data: {values}")
        
        if not values or len(values) < 2:
            print("No data found in sheet")
            return []
        
        headers = values[0][1:]  # Skip timestamp
        data_rows = values[1:]
        
        print(f"Headers: {headers}")
        print(f"Data rows: {data_rows}")
        
        applicants = []
        for i, row in enumerate(data_rows):
            print(f"Processing row {i+1}: {row}")
            
            if len(row) < len(headers) + 1:
                print(f"Row {i+1} is incomplete, skipping")
                continue
            
            row_dict = dict(zip(headers, row[1:]))  # Skip timestamp in each row
            print(f"Row {i+1} dict: {row_dict}")
            
            resume_url = row_dict.get('RESUME ', '')  # Note the trailing space
            print(f"Resume URL for row {i+1}: '{resume_url}'")
            
            file_id = extract_drive_file_id(resume_url)
            print(f"Extracted file ID for row {i+1}: '{file_id}'")
            
            applicant = {
                'name': row_dict.get('Name', ''),
                'college': row_dict.get('College Name and Course', ''),
                'email': row_dict.get('Email', ''),
                'intro': row_dict.get('Short Intro about you', ''),
                'resume_file_id': file_id
            }
            
            print(f"Created applicant: {applicant}")
            applicants.append(applicant)
        
        print(f"Final applicants list: {applicants}")
        return applicants
    
    def fetch_applicants_with_resumes(self, range_name: str = 'Form Responses 1') -> List[Dict]:
        """
        Read applicant data from Google Sheet and extract Drive file IDs for resumes.
        Returns a list of dicts with complete applicant info.
        """
        from app.utils.helpers import extract_drive_file_id
        
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        creds = service_account.Credentials.from_service_account_file(
            Config.CREDENTIALS_PATH, scopes=SCOPES
        )
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=Config.GOOGLE_SHEET_ID, 
            range=range_name
        ).execute()
        values = result.get('values', [])
        
        print(f"Raw sheet data: {values}")
        
        if not values or len(values) < 2:
            print("No data found in sheet")
            return []
        
        headers = values[0]
        data_rows = values[1:]
        
        print(f"Headers: {headers}")
        print(f"Data rows: {data_rows}")
        
        # Find the RESUME column index (handle trailing spaces)
        resume_col_index = None
        for i, header in enumerate(headers):
            if header.strip() == 'RESUME':
                resume_col_index = i
                break
        
        if resume_col_index is None:
            print("RESUME column not found!")
            print(f"Available headers: {[h.strip() for h in headers]}")
            return []
        
        print(f"RESUME column found at index: {resume_col_index}")
        
        applicants = []
        for i, row in enumerate(data_rows):
            print(f"Processing row {i+1}: {row}")
            
            if len(row) > resume_col_index:
                resume_url = row[resume_col_index]
                print(f"Resume URL for row {i+1}: '{resume_url}'")
                
                file_id = extract_drive_file_id(resume_url)
                print(f"Extracted file ID for row {i+1}: '{file_id}'")
                
                if file_id:
                    # Create applicant dict with all available data
                    applicant = {
                        'name': row[1] if len(row) > 1 else '',
                        'college': row[2] if len(row) > 2 else '',
                        'email': row[3] if len(row) > 3 else '',
                        'intro': row[4] if len(row) > 4 else '',
                        'resume_file_id': file_id
                    }
                    applicants.append(applicant)
            else:
                print(f"Row {i+1} doesn't have RESUME column data")
        
        print(f"Final applicants list: {applicants}")
        return applicants
    
    def download_pdf_from_drive(self, file_id: str) -> Optional[bytes]:
        """Download a PDF file from Google Drive and return its bytes."""
        creds = service_account.Credentials.from_service_account_file(
            Config.CREDENTIALS_PATH, 
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        service = build('drive', 'v3', credentials=creds)
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        
        try:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            return fh.read()
        except Exception as e:
            print(f"Error downloading file {file_id}: {e}")
            return None
    
    def create_calendar_event(self, candidate_email: str, candidate_name: str, 
                            interview_date: str, interview_time: str, 
                            interviewer_email: Optional[str] = None) -> Dict:
        """
        Create a Google Calendar event for the interview using personal Google Account.
        This can send actual email invites to candidates.
        """
        try:
            # Get personal Google Account credentials
            creds = self.get_google_calendar_credentials()
            
            if isinstance(creds, dict) and 'error' in creds:
                return creds
            
            # Build the Calendar service
            service = build('calendar', 'v3', credentials=creds)
            
            # Parse date and time
            from datetime import datetime, timedelta
            try:
                # Combine date and time
                datetime_str = f"{interview_date} {interview_time}"
                start_time = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
                end_time = start_time + timedelta(hours=1)  # 1-hour interview
                
                # Format for Google Calendar API
                start_time_str = start_time.isoformat() + 'Z'
                end_time_str = end_time.isoformat() + 'Z'
            except ValueError as e:
                return {'error': f'Invalid date/time format: {str(e)}'}
            
            # Create event with attendees to send email invites
            event = {
                'summary': f'Interview with {candidate_name}',
                'description': f'Interview for the position.\n\nCandidate: {candidate_name}\nEmail: {candidate_email}',
                'start': {
                    'dateTime': start_time_str,
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time_str,
                    'timeZone': 'UTC',
                },
                'attendees': [
                    {'email': candidate_email},
                ],
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},  # 1 day before
                        {'method': 'popup', 'minutes': 30},  # 30 minutes before
                    ],
                },
            }
            
            # Add interviewer if provided
            if interviewer_email:
                event['attendees'].append({'email': interviewer_email})
            
            # Insert the event and send email invites
            event = service.events().insert(
                calendarId='primary',  # Use primary calendar
                body=event,
                sendUpdates='all'  # Send email notifications to all attendees
            ).execute()
            
            return {
                'status': 'success',
                'event_id': event.get('id'),
                'event_link': event.get('htmlLink'),
                'message': f'Calendar event created successfully for {candidate_name}. Email invite sent to {candidate_email}.'
            }
            
        except Exception as e:
            return {'error': f'Failed to create calendar event: {str(e)}'}
