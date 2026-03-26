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

6. **Candidate AI interview (microphone):** Browsers only allow `getUserMedia` on **HTTPS** or **localhost**. If candidates open `http://<VM_IP>:5000/ai-interviewer/...`, voice recording will not work; they can still type answers, or you should put **TLS in front** (Nginx + Let’s Encrypt on a domain, or Cloudflare) and set `APP_BASE_URL` to the `https://` URL.

---

## 2. Cloudflare Quick Tunnel (HTTPS to Flask, no paid domain)

Use this when you want a **public `https://` URL** to your VM backend **without** buying a domain. Cloudflare gives a random hostname like `https://something.trycloudflare.com` that forwards to your Flask process.

### 2.1 Install `cloudflared` on the VM

**Linux amd64** (typical Azure VM):

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
cloudflared --version
```

Use **`cloudflared-linux-arm64.deb`** on ARM64 VMs.

### 2.2 Run Flask on the VM (same as §1)

From `backend/` with your venv:

```bash
export FLASK_CONFIG=production
gunicorn -w 2 -b 127.0.0.1:5000 "main:app"
```

Binding to **`127.0.0.1:5000`** is enough for the tunnel (only local access). You can keep **`0.0.0.0:5000`** if you still need direct `http://<VM_IP>:5000` for debugging.

### 2.3 Start the quick tunnel

In **another** shell on the same VM:

```bash
cloudflared tunnel --url http://127.0.0.1:5000
```

After a few seconds, note the URL, e.g.:

`https://bright-tree-458.trycloudflare.com`

That is your **HTTPS** base URL for the API.

**Background (optional):**

```bash
nohup cloudflared tunnel --url http://127.0.0.1:5000 > /tmp/cloudflared.log 2>&1 &
```

### 2.4 Connect HireFast to the tunnel

| Where | What to set |
|--------|-------------|
| **Vercel** → Environment variables | `BACKEND_URL` = `https://bright-tree-458.trycloudflare.com` (use **your** URL, **no** trailing slash). Redeploy the project after saving. |
| **`backend/.env` on the VM** | `APP_BASE_URL=https://bright-tree-458.trycloudflare.com` — used for LinkedIn/Composio redirects and absolute interview links. |
| **`backend/.env`** | `CORS_ORIGINS=https://<your-project>.vercel.app` (and any other origins that call Flask from the browser). |
| **LinkedIn Developer app** | Authorized redirect: `https://<tunnel-host>/callback` |
| **Composio** (Google Calendar Auth Config) | Redirect allowlist: `https://<tunnel-host>/scheduling` |

Test: `curl https://<tunnel-host>/health` → `{"status":"ok","service":"hirefast-api"}`.

Your Next.js app still calls **`/api/proxy/...`** on Vercel; the proxy server reads **`BACKEND_URL`** and forwards to the tunnel, which hits Gunicorn on the VM.

### 2.5 Caveats

- **Quick Tunnel hostnames change** each time you restart `cloudflared` (unless you move to a [named tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) with a stable route). After a restart, update **Vercel**, **`backend/.env`**, **LinkedIn**, and **Composio** with the new URL.
- If the tunnel process stops, the public HTTPS URL stops working until you start it again.

---

## 3. Frontend on Vercel

1. In the [Vercel](https://vercel.com) project, set **Root Directory** to **`frontend`**.

2. Add environment variable **server-side only**:

   - **`BACKEND_URL`** = `http://<YOUR_VM_PUBLIC_IP>:5000`, **or** your Cloudflare tunnel `https://<name>.trycloudflare.com` from §2 (recommended for HTTPS and AI interview microphone).

   Do **not** expose the VM URL as `NEXT_PUBLIC_*` unless you intend to bypass the proxy.

3. Deploy. Your app URL will look like `https://<project>.vercel.app`.

---

## 4. Local development

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

## 5. Optional: direct API URL (no proxy)

Set in `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:5000
```

Then the browser calls Flask directly (must allow origin in `CORS_ORIGINS`). Useful for local debugging; **Vercel HTTPS + VM HTTP** usually blocks this in the browser.

---

## 6. Legacy Flask HTML UI

Routes still serve Jinja templates (`/dashboard`, `/candidates`, …) from **`backend/`** for backward compatibility. The new product UI lives under **`frontend/`**.

**Environment:** Flask reads **`backend/.env` only** — not a `.env` at the repo root.

---

## 7. Repository layout

```
HireFast/
├── backend/     # Flask — run Gunicorn here (or --chdir backend)
├── frontend/    # Next.js — Vercel root directory
└── DEPLOYMENT.md
```
