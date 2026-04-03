import { NextRequest, NextResponse } from "next/server";

/**
 * Transparent reverse proxy — Vercel serves Flask Jinja + static for (almost) all browser paths.
 *
 * When BACKEND_URL is set: requests are rewritten to Flask — including / and /dashboard.
 * Exceptions: Next.js API routes listed below (proxy helper + client-config).
 *
 * When BACKEND_URL is not set: fall through to Next.js (local frontend dev, / shows a hint page).
 */

const BACKEND = (process.env.BACKEND_URL ?? "").trim().replace(/\/$/, "");

const NEXTJS_API_ROUTES = ["/api/proxy/", "/api/client-config"];

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  if (NEXTJS_API_ROUTES.some((r) => pathname.startsWith(r))) {
    return NextResponse.next();
  }

  if (!BACKEND) {
    return NextResponse.next();
  }

  const target = new URL(pathname + req.nextUrl.search, BACKEND);
  return NextResponse.rewrite(target);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon\\.ico).*)"],
};
