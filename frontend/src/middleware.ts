import { NextRequest, NextResponse } from "next/server";

/**
 * Transparent reverse proxy — makes your Vercel domain serve Flask's full UI.
 *
 * How it works:
 *   When BACKEND_URL is set: almost every path (including /dashboard) → rewritten to Flask (Jinja + static).
 *   Only the marketing home page / stays on Next.js (React).
 *
 * What is NOT proxied (handled by Next.js itself):
 *   /                    — landing page (React)
 *   /api/proxy/*, /api/client-config — Next.js route handlers
 *   /_next/static/*      — Next.js compiled assets
 *   /_next/image/*       — Next.js image optimisation
 *   /favicon.ico         — favicon
 *
 * When BACKEND_URL is not set, requests fall through to Next.js (including /dashboard React fallback
 * and the home "configure BACKEND_URL" hint).
 */

const BACKEND = (process.env.BACKEND_URL ?? "").trim().replace(/\/$/, "");

// Next.js-owned API routes — everything else under /api/* belongs to Flask.
const NEXTJS_API_ROUTES = ["/api/proxy/", "/api/client-config"];

/** Routes that must render the Next.js app even when BACKEND_URL is set (otherwise middleware rewrites to Flask). */
function isNextjsAppRoute(pathname: string): boolean {
  return pathname === "/" || pathname === "/index.html";
}

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  // Only protect Next.js's own API routes; Flask also uses /api/* (e.g. /api/ai-interview/*)
  if (NEXTJS_API_ROUTES.some((r) => pathname.startsWith(r))) {
    return NextResponse.next();
  }

  // Landing page only — never rewrite to Flask
  if (isNextjsAppRoute(pathname)) {
    return NextResponse.next();
  }

  // No backend configured → fall through to Next.js pages (config hint)
  if (!BACKEND) {
    return NextResponse.next();
  }

  // Proxy everything else to Flask transparently
  const target = new URL(pathname + req.nextUrl.search, BACKEND);
  return NextResponse.rewrite(target);
}

export const config = {
  matcher: [
    // Run on all paths except Next.js internals and static assets
    "/((?!_next/static|_next/image|favicon\\.ico).*)",
  ],
};
