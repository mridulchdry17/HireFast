import os
from typing import Optional, Dict, Any
from composio import Composio
from composio.client.enums import Action, App
from app.config import Config


class ComposioService:
    """Service for handling Google Calendar via Composio (Multi-user rollout)."""

    def __init__(self):
        self.api_key = Config.COMPOSIO_API_KEY
        if not self.api_key:
            print("Warning: COMPOSIO_API_KEY not found in environment.")
        # Use the top-level Composio client (composio-core >= 0.7.x)
        self.client = Composio(api_key=self.api_key)

    def get_auth_url(self, user_id: str, redirect_url: str = "http://127.0.0.1:5000/scheduling") -> Optional[str]:
        """Generate a Google Calendar authentication URL for a specific user entity."""
        try:
            entity = self.client.get_entity(id=user_id)
            connection = entity.initiate_connection(
                app_name=App.GOOGLECALENDAR,
                redirect_url=redirect_url,
            )
            return connection.redirectUrl
        except Exception as e:
            print(f"Composio connection error: {e}")
            return None

    def check_connection_status(self, user_id: str) -> bool:
        """Check if the user has an active Google Calendar connection."""
        try:
            entity = self.client.get_entity(id=user_id)
            connections = entity.get_connections()
            # Filter for active GOOGLECALENDAR connections
            google_cal_connections = [
                c for c in connections
                if c.appName.lower() == "googlecalendar" and c.status == "ACTIVE"
            ]
            return len(google_cal_connections) > 0
        except Exception as e:
            print(f"Composio status check error: {e}")
            return False

    def create_interview_event(
        self,
        user_id: str,
        candidate_email: str,
        candidate_name: str,
        interview_date: str,
        interview_time: str,
        duration_mins: int = 60,
        description: str = None,
        create_meeting_room: bool = True
    ) -> Dict[str, Any]:
        """Create a Google Calendar event using Composio for a specific user."""
        try:
            entity = self.client.get_entity(id=user_id)

            from datetime import datetime, timedelta
            try:
                time_formats = ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"]
                start_dt = None
                for fmt in time_formats:
                    try:
                        start_dt = datetime.strptime(f"{interview_date} {interview_time}", f"%Y-%m-%d {fmt}")
                        break
                    except ValueError:
                        continue
                if start_dt is None:
                    raise ValueError(f"Could not parse time '{interview_time}'. Use HH:MM format (e.g. 14:30)")
                end_dt = start_dt + timedelta(minutes=duration_mins)
                start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError as ve:
                print(f"[Composio] Date/time parse error: {ve}")
                return {"status": "error", "message": f"Invalid date or time format: {ve}"}

            duration_hours = duration_mins // 60
            duration_minutes = duration_mins % 60

            # Use provided description or fall back to default
            event_description = description if description else f"Interview scheduled via HireFast.\nCandidate: {candidate_name} ({candidate_email})"

            params = {
                "start_datetime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "timezone": "UTC",
                "summary": f"Interview with {candidate_name}",
                "description": event_description,
                "attendees": [candidate_email],
                "event_duration_hour": duration_hours,
                "event_duration_minutes": duration_minutes,
                "create_meeting_room": create_meeting_room,
            }

            print(f"[Composio] Scheduling: user={user_id} candidate={candidate_email} start={start_iso}")

            try:
                result = entity.execute_action(
                    action=Action.GOOGLECALENDAR_CREATE_EVENT,
                    params=params,
                )
            except AttributeError:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = entity.execute(
                        action=Action.GOOGLECALENDAR_CREATE_EVENT,
                        params=params,
                    )

            print(f"[Composio] Result: {result}")

            if isinstance(result, dict) and (result.get("successfull") is False or result.get("error")):
                err = result.get("error", str(result))
                print(f"[Composio] Event creation failed: {err}")
                return {"status": "error", "message": err}

            return {"status": "success", "data": result}

        except Exception as e:
            print(f"[Composio] Exception: {e}")
            return {"status": "error", "message": str(e)}


