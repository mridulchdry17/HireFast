/**
 * API base URL for the Flask backend.
 *
 * - Default (no NEXT_PUBLIC_API_URL): browser calls go to Next.js `/api/proxy/*`,
 *   which forwards server-side to BACKEND_URL (avoids HTTPS→HTTP mixed content on Vercel).
 * - Set NEXT_PUBLIC_API_URL=http://localhost:5000 to talk to Flask directly (needs CORS).
 */
export function apiUrl(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  const direct = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
  if (direct) {
    return `${direct}${p}`;
  }
  return `/api/proxy${p}`;
}
