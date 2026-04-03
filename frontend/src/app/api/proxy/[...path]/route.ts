import { NextRequest, NextResponse } from "next/server";

function backendOrigin(): string | null {
  const u = process.env.BACKEND_URL?.trim().replace(/\/$/, "");
  return u || null;
}

async function proxy(req: NextRequest, pathSegments: string[]) {
  const base = backendOrigin();
  if (!base) {
    return NextResponse.json(
      {
        error: "BACKEND_URL is not set",
        detail:
          "Add BACKEND_URL to frontend/.env.local (e.g. http://127.0.0.1:5001) or Vercel project env. It must be your Flask server origin with no trailing slash.",
      },
      { status: 503 },
    );
  }

  const path = pathSegments.length ? `/${pathSegments.join("/")}` : "/";
  const target = new URL(path + req.nextUrl.search, base);

  const init: RequestInit = {
    method: req.method,
    cache: "no-store",
    headers: {
      Accept: req.headers.get("Accept") || "*/*",
    },
  };

  if (!["GET", "HEAD"].includes(req.method)) {
    const body = await req.arrayBuffer();
    if (body.byteLength > 0) {
      init.body = body;
    }
    const ct = req.headers.get("Content-Type");
    if (ct) {
      (init.headers as Record<string, string>)["Content-Type"] = ct;
    }
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, init);
  } catch (e) {
    const msg = e instanceof Error ? e.message : "fetch failed";
    return NextResponse.json(
      {
        error: "Upstream unreachable",
        detail: `${msg}. Check BACKEND_URL (${base}) and that Flask is running.`,
      },
      { status: 502 },
    );
  }

  const body = await upstream.arrayBuffer();
  const res = new NextResponse(body, { status: upstream.status });

  const ct = upstream.headers.get("Content-Type");
  if (ct) res.headers.set("Content-Type", ct);

  return res;
}

type Ctx = { params: Promise<{ path?: string[] }> };

export async function GET(req: NextRequest, ctx: Ctx) {
  const { path: segments } = await ctx.params;
  return proxy(req, segments ?? []);
}

export async function POST(req: NextRequest, ctx: Ctx) {
  const { path: segments } = await ctx.params;
  return proxy(req, segments ?? []);
}

export async function PUT(req: NextRequest, ctx: Ctx) {
  const { path: segments } = await ctx.params;
  return proxy(req, segments ?? []);
}

export async function PATCH(req: NextRequest, ctx: Ctx) {
  const { path: segments } = await ctx.params;
  return proxy(req, segments ?? []);
}

export async function DELETE(req: NextRequest, ctx: Ctx) {
  const { path: segments } = await ctx.params;
  return proxy(req, segments ?? []);
}
