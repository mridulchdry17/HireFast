import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 px-6 text-center text-slate-100">
      <h1 className="bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-4xl font-bold text-transparent sm:text-5xl">
        HireFast
      </h1>
      <p className="mt-3 max-w-md text-slate-400">
        AI-powered HR automation. Frontend on Vercel, API on your VM — connected via the
        Next.js proxy.
      </p>
      <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
        <Link
          href="/dashboard"
          className="rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-500/25"
        >
          Open dashboard
        </Link>
        <p className="text-sm text-slate-500">
          Deploy: see <code className="text-slate-400">DEPLOYMENT.md</code> in the repo root.
        </p>
      </div>
    </div>
  );
}
