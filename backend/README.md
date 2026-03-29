# HireFast API (Flask)

Run all commands from this directory (or use `gunicorn --chdir backend` from the repo root).

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env
python main.py
```

Production:

```bash
export FLASK_CONFIG=production
gunicorn -w 2 -b 0.0.0.0:5000 "main:app"
```

Place Google credential files under `credentials/` (see `GOOGLE_CALENDAR_SETUP.md` in the repo root). The app uses **Postgres** only: set `DATABASE_URL` in `.env` (e.g. [Neon](https://neon.tech) free tier). Uploads still live under `uploads/`.
