"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { apiUrl } from "@/lib/api";

type Health = { status?: string; service?: string };
type SummaryPayload = {
  job_count?: number;
  application_count?: number;
  ai_interview_sessions?: number;
  error?: string;
};
type ApiErr = { error?: string; detail?: string };

function readApiError(body: unknown): string | null {
  if (!body || typeof body !== "object") return null;
  const o = body as ApiErr;
  if (o.error && o.detail) return `${o.error}: ${o.detail}`;
  if (o.detail) return o.detail;
  if (o.error) return o.error;
  return null;
}

export default function DashboardPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [summary, setSummary] = useState<SummaryPayload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const h = await fetch(apiUrl("/health"));
        const raw = await h.json().catch(() => null);
        if (!h.ok) {
          if (!cancelled) setErr(readApiError(raw) ?? `API error (${h.status})`);
        } else if (!cancelled) {
          setHealth(raw as Health);
        }
      } catch {
        if (!cancelled) setErr("Could not reach API (check BACKEND_URL / proxy).");
      }
      try {
        const a = await fetch(apiUrl("/dashboard-summary"));
        const text = await a.text();
        let raw: unknown = null;
        try {
          raw = text ? JSON.parse(text) : null;
        } catch {
          if (!cancelled) {
            setErr((prev) => prev ?? "Dashboard summary response was not valid JSON (often fixed by redeploying after a proxy update).");
          }
          return;
        }
        if (!cancelled) {
          if (!a.ok) {
            setErr((prev) => prev ?? readApiError(raw) ?? `Summary error (${a.status})`);
          } else {
            const s = raw as SummaryPayload;
            if (s?.error) {
              const msg = s.error;
              setErr((prev) => prev ?? msg);
            } else if (raw && typeof raw === "object") {
              setSummary(s);
            }
          }
        }
      } catch {
        if (!cancelled) setErr((prev) => prev ?? "Failed to load dashboard summary.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!sidebarOpen) return;
    const onResize = () => {
      if (window.matchMedia("(min-width: 768px)").matches) setSidebarOpen(false);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [sidebarOpen]);

  const nav = [
    { href: "/dashboard", label: "Dashboard", active: true },
    { href: "#", label: "Create Job Post" },
    { href: "#", label: "Candidates" },
    { href: "#", label: "Schedule Interviews" },
    { href: "#", label: "Connect Calendar" },
    { href: "#", label: "Analytics" },
    { href: "#", label: "Settings" },
  ];

  const sidebarInner = (
    <>
      <div className="border-b border-white/10 p-4 sm:p-6">
        <h1 className="bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-lg font-bold text-transparent sm:text-xl">
          HireFast
        </h1>
        <p className="mt-1 text-xs text-slate-400">AI-Powered HR Automation</p>
      </div>
      <nav className="flex flex-1 flex-col gap-1 overflow-y-auto p-2 sm:p-3">
        {nav.map((item) => (
          <Link
            key={item.label}
            href={item.href}
            onClick={() => setSidebarOpen(false)}
            className={`rounded-xl px-3 py-2.5 text-sm transition ${
              item.active
                ? "bg-gradient-to-r from-indigo-600 to-violet-600 text-white"
                : "text-slate-400 hover:bg-white/5 hover:text-white"
            }`}
          >
            {item.label}
          </Link>
        ))}
      </nav>
      <div className="border-t border-white/10 p-3 sm:p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-violet-600/90 text-xs font-semibold sm:h-10 sm:w-10 sm:text-sm">
            HF
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium">HireFast</p>
            <p className="text-xs text-slate-400">Admin</p>
          </div>
        </div>
      </div>
    </>
  );

  return (
    <div className="flex min-h-dvh bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-slate-100">
      {sidebarOpen && (
        <button
          type="button"
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm md:hidden"
          aria-label="Close menu"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[min(18rem,92vw)] shrink-0 flex-col overflow-x-hidden border-r border-white/10 bg-slate-950/95 backdrop-blur transition-transform duration-200 md:static md:z-0 md:w-72 md:min-w-[18rem] md:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        {sidebarInner}
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-white/10 bg-slate-950/90 px-3 py-3 backdrop-blur md:hidden">
          <button
            type="button"
            className="inline-flex items-center justify-center rounded-lg border border-white/15 p-2 text-slate-200 hover:bg-white/10"
            aria-expanded={sidebarOpen}
            aria-label={sidebarOpen ? "Close navigation" : "Open navigation"}
            onClick={() => setSidebarOpen((o) => !o)}
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
              {sidebarOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
          <span className="truncate text-sm font-semibold text-white">Dashboard</span>
        </header>

        <main className="relative flex-1 overflow-auto">
          <div
            className="pointer-events-none absolute inset-0 opacity-[0.12]"
            style={{
              backgroundImage: `linear-gradient(rgba(148,163,184,0.08) 1px, transparent 1px),
              linear-gradient(90deg, rgba(148,163,184,0.08) 1px, transparent 1px)`,
              backgroundSize: "24px 24px",
            }}
          />
          <div className="relative z-10 p-4 sm:p-6 lg:p-8">
            <div className="mb-6 flex flex-col gap-4 sm:mb-8 sm:flex-row sm:flex-wrap sm:items-start sm:justify-between">
              <div className="min-w-0">
                <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">Dashboard</h2>
                <p className="mt-1 text-sm text-slate-400 sm:text-base">
                  Welcome back! Here&apos;s what&apos;s happening with your hiring process.
                </p>
                <p className="mt-2 max-w-2xl text-xs leading-relaxed text-slate-500">
                  Stats below are read from your Flask database via the API. Posting jobs, LinkedIn, and calendar flows
                  stay on your main server until those screens are built here.
                </p>
              </div>
            </div>

            {err && (
              <div className="mb-6 rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                {err}
              </div>
            )}

            <div className="mb-4 text-xs text-slate-500">
              API:{" "}
              {health ? (
                <span className="text-emerald-400">
                  {health.service} — {health.status}
                </span>
              ) : (
                <span>checking…</span>
              )}
            </div>

            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {[
                {
                  label: "Job postings (DB)",
                  value: summary ? String(summary.job_count ?? 0) : "—",
                  trend: "JobPosting rows",
                  accent: "from-blue-500 to-indigo-600",
                },
                {
                  label: "Applications (DB)",
                  value: summary ? String(summary.application_count ?? 0) : "—",
                  trend: "Application rows",
                  accent: "from-emerald-500 to-teal-600",
                },
                {
                  label: "AI interview sessions",
                  value: summary ? String(summary.ai_interview_sessions ?? 0) : "—",
                  trend: "AIInterviewSession rows",
                  accent: "from-pink-500 to-violet-600",
                },
                {
                  label: "Hires this month",
                  value: "—",
                  trend: "Not stored in DB yet",
                  accent: "from-amber-500 to-orange-600",
                },
              ].map((c) => (
                <div
                  key={c.label}
                  className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur sm:p-5"
                >
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-xs text-slate-400 sm:text-sm">{c.label}</p>
                    <div className={`h-9 w-9 shrink-0 rounded-xl bg-gradient-to-br ${c.accent} opacity-90 sm:h-10 sm:w-10`} />
                  </div>
                  <p className="mt-3 text-2xl font-bold sm:mt-4 sm:text-3xl">{c.value}</p>
                  <p className="mt-1 text-xs text-slate-500 sm:text-sm">{c.trend}</p>
                </div>
              ))}
            </div>

            <div className="mt-8 grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4 sm:p-6">
                <h3 className="mb-2 text-base font-semibold sm:text-lg">Where to work</h3>
                <p className="mb-4 text-sm text-slate-400">
                  Job posts, candidates, LinkedIn, and calendar are not wired in this Next.js shell yet. Use your Flask
                  app on the VM for the full workflow.
                </p>
                <div className="flex flex-col gap-2">
                  {process.env.NEXT_PUBLIC_MAIN_APP_URL ? (
                    <a
                      href={process.env.NEXT_PUBLIC_MAIN_APP_URL.replace(/\/$/, "")}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-4 py-3 text-center text-sm font-semibold text-white"
                    >
                      Open full HireFast (Flask)
                    </a>
                  ) : (
                    <p className="rounded-xl border border-dashed border-white/15 bg-slate-900/40 px-4 py-3 text-left text-sm text-slate-400">
                      Set <code className="text-slate-300">NEXT_PUBLIC_MAIN_APP_URL</code> on Vercel (your VM URL,
                      e.g. <code className="text-slate-300">http://40.x.x.x:5000</code>) to show a button here.
                    </p>
                  )}
                </div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4 sm:p-6">
                <h3 className="mb-2 text-base font-semibold sm:text-lg">Activity</h3>
                <p className="text-sm leading-relaxed text-slate-400">
                  There is no live activity feed in this preview. Your Flask templates and database hold the real
                  history; we only surface aggregate counts above.
                </p>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
