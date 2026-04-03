import Link from "next/link";

/**
 * Shown only when BACKEND_URL is unset (e.g. local Next dev without Flask).
 * In production, middleware rewrites / to Flask — this file is not used.
 */
export const dynamic = "force-dynamic";

export default function Home() {
  return (
    <div className="flex min-h-dvh flex-col items-center justify-center bg-[#070b14] px-4 py-12 text-center text-slate-200">
      <div className="mx-auto max-w-md rounded-2xl border border-white/[0.08] bg-white/[0.04] px-6 py-10 shadow-xl">
        <h1 className="bg-gradient-to-r from-sky-300 to-violet-300 bg-clip-text text-2xl font-bold text-transparent">
          HireFast (Next.js dev)
        </h1>
        <p className="mt-4 text-sm leading-relaxed text-slate-400">
          Set <code className="rounded bg-white/10 px-1.5 py-0.5 text-slate-300">BACKEND_URL</code> in{" "}
          <code className="rounded bg-white/10 px-1.5 py-0.5 text-slate-300">frontend/.env.local</code> to proxy to
          Flask — the real home page is <span className="text-slate-300">backend/templates/index.html</span>.
        </p>
        <p className="mt-4 text-xs text-slate-500">
          Or run the Flask app and open its URL directly for the full UI.
        </p>
        <Link
          href="/dashboard"
          className="mt-6 inline-block rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-5 py-2.5 text-sm font-semibold text-white"
        >
          Try /dashboard (proxied when configured)
        </Link>
      </div>
    </div>
  );
}
