import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-dvh flex-col items-center justify-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 px-4 py-10 text-center text-slate-100 sm:px-6">
      <h1 className="bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-3xl font-bold text-transparent sm:text-4xl md:text-5xl">
        HireFast
      </h1>
      <p className="mt-3 max-w-md text-sm text-slate-400 sm:text-base">
        AI-powered HR automation. Frontend on Vercel, API on your VM — connected via the
        Next.js proxy.
      </p>
      <div className="mt-8 flex w-full max-w-md flex-col items-stretch gap-3 sm:mt-10 sm:max-w-none sm:flex-row sm:flex-wrap sm:items-center sm:justify-center sm:gap-4">
        <Link
          href="/dashboard"
          className="rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-6 py-3 text-center text-sm font-semibold text-white shadow-lg shadow-indigo-500/25 sm:inline-block"
        >
          Open dashboard
        </Link>
        <p className="text-center text-xs text-slate-500 sm:text-sm">
          Deploy: see <code className="text-slate-400">DEPLOYMENT.md</code> in the repo root.
        </p>
      </div>
    </div>
  );
}
