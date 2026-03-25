# HireFast split deployment (Vercel + Azure VM)

This repo contains:

- **Backend:** Flask app in **`backend/`** (`main.py`, `app/`, `templates/`, `static/`).
- **Frontend:** Next.js app in **`frontend/`** (deploy to Vercel).

The browser talks to **your Vercel domain** only. Next.js **`/api/proxy/*`** forwards requests server-side to your Flask **`BACKEND_URL`**, so you avoid HTTPS (Vercel) → HTTP (VM) mixed-content issues in the browser.

---

## 1. Backend on the VM

1. Install Python 3.10+ and dependencies:

   ```bash
   cd /path/to/HireFast/backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set `SECRET_KEY`, `CORS_ORIGINS` (include `https://<your-project>.vercel.app` if you ever call the API directly from the browser), and optional `DATABASE_URL`.

3. Run with Gunicorn (production):

   ```bash
   export FLASK_CONFIG=production
   gunicorn -w 2 -b 0.0.0.0:5000 "main:app"
   ```

   From the **repository root** (without `cd backend`):

   ```bash
   export FLASK_CONFIG=production
   gunicorn --chdir backend -w 2 -b 0.0.0.0:5000 "main:app"
   ```

4. Open **Azure NSG** inbound port **5000** (or terminate TLS on 443 with Nginx and proxy to `127.0.0.1:5000`).

5. Health check: `GET http://<VM_IP>:5000/health` → `{"status":"ok","service":"hirefast-api"}`.

---

## 2. Frontend on Vercel

1. In the [Vercel](https://vercel.com) project, set **Root Directory** to **`frontend`**.

2. Add environment variable **server-side only**:

   - **`BACKEND_URL`** = `http://<YOUR_VM_PUBLIC_IP>:5000` (or `https://api.yourdomain.com` once you add HTTPS).

   Do **not** expose the VM URL as `NEXT_PUBLIC_*` unless you intend to bypass the proxy.

3. Deploy. Your app URL will look like `https://<project>.vercel.app`.

---

## 3. Local development

**Terminal A — Flask**

```bash
cd HireFast/backend
source .venv/bin/activate
pip install -r requirements.txt
export FLASK_HOST=127.0.0.1
export FLASK_PORT=5000
python main.py
```

**Terminal B — Next.js**

```bash
cd HireFast/frontend
cp .env.example .env.local
# Ensure BACKEND_URL=http://127.0.0.1:5000
npm install
npm run dev
```

Open `http://localhost:3000` → Dashboard loads data via `/api/proxy/...` to Flask.

---

## 4. Optional: direct API URL (no proxy)

Set in `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:5000
```

Then the browser calls Flask directly (must allow origin in `CORS_ORIGINS`). Useful for local debugging; **Vercel HTTPS + VM HTTP** usually blocks this in the browser.

---

## 5. Legacy Flask HTML UI

Routes still serve Jinja templates (`/dashboard`, `/candidates`, …) from **`backend/`** for backward compatibility. The new product UI lives under **`frontend/`**.

**If you previously ran Flask from the repo root:** copy your `.env` to `backend/.env` and use `hirefast.db` / `uploads/` under `backend/` (or set `DATABASE_URL`).

---

## 6. Repository layout

```
HireFast/
├── backend/     # Flask — run Gunicorn here (or --chdir backend)
├── frontend/    # Next.js — Vercel root directory
└── DEPLOYMENT.md
```
