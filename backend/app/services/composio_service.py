import os
from typing import Optional, Dict, Any

from composio import Composio
from composio.client.enums import Action
from app.config import Config


class ComposioService:
    """Google Calendar via Composio (SDK v3: tools + connected_accounts, no get_entity)."""

    def __init__(self):
        self.api_key = Config.COMPOSIO_API_KEY
        self._auth_config_id = Config.COMPOSIO_GOOGLE_CALENDAR_AUTH_CONFIG_ID
        self._composio: Optional[Composio] = None
        if not self.api_key:
            print("Warning: COMPOSIO_API_KEY not found in environment.")
            return
        try:
            kwargs = {"api_key": self.api_key}
            tv = Config.COMPOSIO_TOOLKIT_VERSION_GOOGLECALENDAR
            if tv:
                kwargs["toolkit_versions"] = {"googlecalendar": tv}
            try:
                self._composio = Composio(**kwargs)
            except TypeError:
                # Older Composio() without toolkit_versions
                self._composio = Composio(api_key=self.api_key)
        except Exception as e:
            print(f"Composio init error: {e}")
            self._composio = None

    def get_auth_url(self, user_id: str, redirect_url: Optional[str] = None) -> Optional[str]:
        """OAuth link for the user to connect Google Calendar (Composio Link)."""
        if not self._composio:
            print("Composio: client not initialized (missing COMPOSIO_API_KEY?).")
            return None
        if not self._auth_config_id:
            print(
                "Composio: set COMPOSIO_GOOGLE_CALENDAR_AUTH_CONFIG_ID in backend/.env "
                "(Auth Config ID from Composio dashboard for Google Calendar)."
            )
            return None
        try:
            if not redirect_url:
                # Only use env APP_BASE_URL; never Config default (localhost) when env unset.
                base = (os.environ.get("APP_BASE_URL") or "").strip().rstrip("/") or "http://127.0.0.1:5000"
                redirect_url = f"{base}/scheduling"
            req = self._composio.connected_accounts.link(
                user_id=user_id,
                auth_config_id=self._auth_config_id,
                callback_url=redirect_url,
            )
            return req.redirect_url
        except Exception as e:
            print(f"Composio connection error: {e}")
            return None

    def check_connection_status(self, user_id: str) -> bool:
        """True if user has an ACTIVE Google Calendar connected account."""
        if not self._composio:
            return False
        try:
            resp = self._composio.connected_accounts.list(
                user_ids=[user_id],
                toolkit_slugs=["googlecalendar"],
                statuses=["ACTIVE"],
                limit=10,
            )
            items = getattr(resp, "items", None) or []
            return len(items) > 0
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
        create_meeting_room: bool = True,
    ) -> Dict[str, Any]:
        """Create a Google Calendar event via Composio tools.execute."""
        if not self._composio:
            return {"status": "error", "message": "Composio not configured (COMPOSIO_API_KEY)."}

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
            duration_hours = duration_mins // 60
            duration_minutes = duration_mins % 60

            event_description = description if description else (
                f"Interview scheduled via HireFast.\nCandidate: {candidate_name} ({candidate_email})"
            )

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

            print(f"[Composio] Scheduling: user={user_id} candidate={candidate_email}")

            slug = Action.GOOGLECALENDAR_CREATE_EVENT.slug
            exec_kw: Dict[str, Any] = {"arguments": params, "user_id": user_id}
            tv = Config.COMPOSIO_TOOLKIT_VERSION_GOOGLECALENDAR
            if tv:
                exec_kw["version"] = tv
            elif Config.COMPOSIO_DANGEROUSLY_SKIP_TOOLKIT_VERSION_CHECK:
                exec_kw["dangerously_skip_version_check"] = True

            tools = getattr(self._composio, "tools", None)
            if not tools or not hasattr(tools, "execute"):
                return {
                    "status": "error",
                    "message": "Composio SDK has no tools.execute; upgrade composio (see requirements.txt).",
                }

            try:
                result = tools.execute(slug, **exec_kw)
            except TypeError:
                # SDK without versioning kwargs
                result = tools.execute(slug, arguments=params, user_id=user_id)

            print(f"[Composio] Result: {result}")

            if isinstance(result, dict):
                if result.get("successful") is False or result.get("error"):
                    err = result.get("error", str(result))
                    print(f"[Composio] Event creation failed: {err}")
                    return {"status": "error", "message": err}
                return {"status": "success", "data": result.get("data", result)}

            return {"status": "success", "data": result}

        except Exception as e:
            print(f"[Composio] Exception: {e}")
            return {"status": "error", "message": str(e)}
