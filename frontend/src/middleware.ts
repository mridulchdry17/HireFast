import { NextRequest, NextResponse } from "next/server";

/**
 * Transparent reverse proxy — makes your Vercel domain serve Flask's full UI.
 *
 * How it works:
 *   Browser → hire-fast-lime.vercel.app/dashboard
 *   Middleware → fetches BACKEND_URL/dashboard (Flask Jinja template)
 *   Response returned to browser (users never see the backend URL / VM IP)
 *
 * What is NOT proxied (handled by Next.js itself):
 *   /api/*               — Next.js route handlers (proxy, client-config)
 *   /_next/static/*      — Next.js compiled assets
 *   /_next/image/*       — Next.js image optimisation
 *   /favicon.ico         — favicon
 *
 * When BACKEND_URL is not set, everything falls through to Next.js pages
 * (page.tsx shows a "configure BACKEND_URL" hint — useful during local dev).
 */

const BACKEND = (process.env.BACKEND_URL ?? "").trim().replace(/\/$/, "");

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  // Keep Next.js API routes — /api/proxy, /api/client-config, etc.
  if (pathname.startsWith("/api/")) {
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
