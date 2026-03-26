/**
 * Server-only: resolves the public Flask / main-app origin.
 * Used by server components (home page). Client code should GET /api/client-config instead.
 */
export function resolveMainAppUrl(): string {
  const fromPublic = process.env.NEXT_PUBLIC_MAIN_APP_URL?.trim().replace(/\/$/, "");
  if (fromPublic) return fromPublic;
  return process.env.BACKEND_URL?.trim().replace(/\/$/, "") ?? "";
}
