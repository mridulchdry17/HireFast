# Calendar Integration with Composio

## Overview

The HR automation system now includes **personalized calendar integration** using Composio. Each user can connect their own Google Calendar, and interview scheduling events will be sent to their personal calendar.

## How It Works

### 1. User Authentication Flow
- User clicks "Connect Calendar" button
- System generates a unique `client_id` for the user
- User is redirected to Google OAuth flow
- User authenticates with their Google account
- Calendar access is granted to their personal account

### 2. Interview Scheduling
- When scheduling interviews, the system uses the user's `client_id`
- Calendar events are created in the user's personal Google Calendar
- Email invites are sent from the user's account
- Each user has their own separate calendar

## Features

✅ **Personal Calendar Access** - Each user gets their own calendar  
✅ **Secure Authentication** - Uses Google OAuth 2.0  
✅ **No Manual Setup** - Users just click and authenticate  
✅ **Automatic Email Invites** - Sends invites from user's account  
✅ **Multi-User Support** - Unlimited users can connect their calendars  

## User Experience

### Step 1: Connect Calendar
1. User clicks "Connect Calendar" button
2. Browser opens Google OAuth page
3. User logs into their Google account
4. User grants calendar access permissions
5. User returns to the app

### Step 2: Schedule Interviews
1. User selects best candidates
2. User clicks "Schedule Interview" on a candidate
3. User enters date/time details
4. System creates calendar event in user's personal calendar
5. Email invite is sent to the candidate

## Technical Implementation

### Backend Changes
- Added `/connect-calendar` endpoint for authentication
- Added `/check-calendar-status` endpoint for status checking
- Updated `create_interview_event()` to use user-specific `user_id`
- Uses Composio for secure OAuth management

### Frontend Changes
- Added Calendar Integration section in UI
- Added "Connect Calendar" and "Check Calendar Status" buttons
- Added real-time status updates
- Added authentication flow guidance

## API Endpoints

### POST `/connect-calendar`
Starts the calendar authentication flow for the current user.

**Response:**
```json
{
  "status": "success",
  "auth_url": "https://composio.ai/...",
  "user_id": "user-uuid",
  "message": "Please visit the URL to authenticate your Google Calendar"
}
```

### GET `/check-calendar-status`
Checks if the current user has connected their calendar.

**Response:**
```json
{
  "connected": true,
  "message": "Calendar connected successfully!",
  "services": ["gmail"]
}
```

## Installation

1. Install the Composio LangChain package:
```bash
pip install composio-langchain composio-core
```

2. Set your Composio API key as environment variable:
```bash
export COMPOSIO_API_KEY="ak_your_api_key_here"
```

3. Run the application:
```bash
python app.py
```

## Benefits

- **No Shared Credentials** - Each user has their own calendar access
- **Professional Experience** - Events appear in user's personal calendar
- **Scalable** - Works for unlimited users
- **Secure** - Uses industry-standard OAuth 2.0
- **User-Friendly** - Simple click-to-connect experience

## Troubleshooting

### Calendar Not Connecting
- Ensure user completes the full OAuth flow
- Check that user grants calendar permissions
- Verify Composio API key is set correctly

### Interview Scheduling Fails
- Ensure user has connected their calendar first
- Check that user's `client_id` is stored correctly
- Verify Google Calendar API is enabled

### Multiple Users
- Each user gets their own `client_id`
- Events go to each user's personal calendar
- No conflicts between different users
