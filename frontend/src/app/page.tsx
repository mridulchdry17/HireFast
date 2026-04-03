import Link from "next/link";
import { resolveMainAppUrl } from "@/lib/mainAppUrl";

/** Read BACKEND_URL / NEXT_PUBLIC_* on each request so Vercel env is always current. */
export const dynamic = "force-dynamic";

export default function Home() {
  const mainApp = resolveMainAppUrl();

  return (
    <div className="flex min-h-dvh flex-col items-center justify-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 px-4 py-12 text-center text-slate-100 sm:px-6">
      <div className="mx-auto max-w-lg rounded-2xl border border-white/10 bg-white/[0.03] px-6 py-10 shadow-xl shadow-black/20 sm:px-10">
        <h1 className="bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-3xl font-bold text-transparent sm:text-4xl">
          HireFast
        </h1>
        <p className="mt-4 text-sm leading-relaxed text-slate-400 sm:text-base">
          With <code className="rounded bg-white/5 px-1 text-slate-400">BACKEND_URL</code> set, the same domain proxies
          to your Flask app — dashboard, jobs, candidates, and settings are all served from the server (Jinja), not
          duplicated here.
        </p>
        <div className="mt-8 flex flex-col items-stretch gap-3 sm:flex-row sm:justify-center">
          <Link
            href="/dashboard"
            className="rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-6 py-3 text-center text-sm font-semibold text-white shadow-lg shadow-indigo-500/20"
          >
            View live stats
          </Link>
          {mainApp ? (
            <a
              href={mainApp}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl border border-white/15 bg-slate-900/50 px-6 py-3 text-center text-sm font-medium text-slate-200 hover:bg-white/5"
            >
              Open full app (Flask)
            </a>
          ) : null}
        </div>
        {!mainApp ? (
          <p className="mt-6 text-xs text-slate-500">
            Set <code className="rounded bg-white/5 px-1.5 py-0.5 text-slate-400">BACKEND_URL</code> on Vercel (your
            Flask VM URL, same one the proxy uses). Optional:{" "}
            <code className="rounded bg-white/5 px-1.5 py-0.5 text-slate-400">NEXT_PUBLIC_MAIN_APP_URL</code> if the UI
            lives on a different URL than the API.
          </p>
        ) : null}
      </div>
    </div>
  );
}
