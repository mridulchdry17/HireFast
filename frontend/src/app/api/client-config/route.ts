import { NextResponse } from "next/server";

/**
 * Exposes a safe "open the Flask UI" URL to the browser.
 * Prefer NEXT_PUBLIC_MAIN_APP_URL when the UI is on a different origin than BACKEND_URL;
 * otherwise BACKEND_URL alone is enough (typical: same VM:port for API + Jinja pages).
 */
export async function GET() {
  const fromPublic = process.env.NEXT_PUBLIC_MAIN_APP_URL?.trim().replace(/\/$/, "");
  const fromBackend = process.env.BACKEND_URL?.trim().replace(/\/$/, "");
  const mainAppUrl = fromPublic || fromBackend || null;
  return NextResponse.json({ mainAppUrl });
}
