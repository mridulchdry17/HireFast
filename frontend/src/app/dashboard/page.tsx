"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { apiUrl } from "@/lib/api";
import { formatRelativeTime } from "@/lib/formatRelativeTime";

type Health = { status?: string; service?: string };
type ActivityItem = {
  kind: string;
  title: string;
  detail?: string | null;
  at: string | null;
  accent?: string;
};
type Workflow = {
  role: string;
  company_name?: string | null;
  created_at?: string | null;
  step_label?: string;
};
type SummaryPayload = {
  job_count?: number;
  jobs_this_week?: number;
  application_count?: number;
  applications_this_week?: number;
  applications_today?: number;
  ai_interview_sessions?: number;
  interviews_engaged?: number;
  interview_sessions_this_week?: number;
  completed_interviews_this_month?: number;
  recent_activity?: ActivityItem[];
  workflow?: Workflow | null;
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

function accentRing(accent: string | undefined): string {
  switch (accent) {
    case "emerald":
      return "bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.45)]";
    case "sky":
      return "bg-sky-400 shadow-[0_0_12px_rgba(56,189,248,0.45)]";
    case "violet":
      return "bg-violet-400 shadow-[0_0_12px_rgba(167,139,250,0.45)]";
    default:
      return "bg-slate-400";
  }
}

function IconBriefcase({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M21 13.255V19a2 2 0 01-2 2H5a2 2 0 01-2-2v-5.745M21 13.255V8a2 2 0 00-2-2h-1.5M21 13.255l-3.182-1.591a2 2 0 00-1.636 0L12 13.745M5 6V4a2 2 0 012-2h10a2 2 0 012 2v2M5 6h14M5 6v7.255a2 2 0 001.118 1.79l5 2.824a2 2 0 001.764 0l5-2.824A2 2 0 0021 13.255"
      />
    </svg>
  );
}

function IconUsers({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.75}
        d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
      />
    </svg>
  );
}

function IconCalendar({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.75}
        d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
      />
    </svg>
  );
}

function IconCheckCircle({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
  );
}

export default function DashboardPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [summary, setSummary] = useState<SummaryPayload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mainAppUrl, setMainAppUrl] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cfg = await fetch("/api/client-config").then((r) => r.json());
        if (!cancelled && cfg?.mainAppUrl && typeof cfg.mainAppUrl === "string") {
          setMainAppUrl(cfg.mainAppUrl);
        }
      } catch {
        /* ignore */
      }
      try {
        const h = await fetch(apiUrl("/health"));
        const raw = await h.json().catch(() => null);
        if (!h.ok) {
          if (!cancelled) setErr(readApiError(raw) ?? `API error (${h.status})`);
        } else if (!cancelled) {
          setHealth(raw as Health);
        }
      } catch {
        if (!cancelled) setErr("Could not reach API (check BACKEND_URL / NEXT_PUBLIC_API_URL).");
      }
      try {
        const a = await fetch(apiUrl("/dashboard-summary"));
        const text = await a.text();
        let raw: unknown = null;
        try {
          raw = text ? JSON.parse(text) : null;
        } catch {
          if (!cancelled) {
            setErr(
              (prev) =>
                prev ?? "Dashboard summary was not valid JSON (check API proxy / backend).",
            );
          }
          return;
        }
        if (!cancelled) {
          if (!a.ok) {
            setErr((prev) => prev ?? readApiError(raw) ?? `Summary error (${a.status})`);
          } else {
            const s = raw as SummaryPayload;
            if (s?.error) {
              setErr((prev) => prev ?? s.error ?? null);
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

  const base = mainAppUrl?.replace(/\/$/, "") ?? "";

  const nav = useMemo(
    () => [
      { href: "/dashboard", label: "Dashboard", active: true, next: true as const },
      { href: base ? `${base}/job-posting` : "#", label: "Create Job Post", next: false as const },
      { href: base ? `${base}/candidates` : "#", label: "Candidates", next: false as const },
      { href: base ? `${base}/scheduling` : "#", label: "Schedule Interviews", next: false as const },
      { href: base ? `${base}/connect-calendar` : "#", label: "Connect Calendar", next: false as const },
      { href: base ? `${base}/analytics` : "#", label: "Analytics", next: false as const },
      { href: base ? `${base}/settings` : "#", label: "Settings", next: false as const },
    ],
    [base],
  );

  const stats = useMemo(() => {
    if (!summary) return null;
    const jw = summary.jobs_this_week ?? 0;
    const aw = summary.applications_this_week ?? 0;
    const at = summary.applications_today ?? 0;
    const iw = summary.interview_sessions_this_week ?? 0;
    return [
      {
        label: "Active job posts",
        value: summary.job_count ?? 0,
        hint: jw > 0 ? `+${jw} this week` : "No new roles this week",
        gradient: "from-indigo-500 via-blue-500 to-cyan-500",
        icon: IconBriefcase,
      },
      {
        label: "Total applications",
        value: summary.application_count ?? 0,
        hint: at > 0 ? `+${at} today` : aw > 0 ? `+${aw} this week` : "Waiting for applicants",
        gradient: "from-emerald-500 to-teal-500",
        icon: IconUsers,
      },
      {
        label: "Interview pipeline",
        value: summary.interviews_engaged ?? summary.ai_interview_sessions ?? 0,
        hint:
          iw > 0
            ? `${iw} session touches this week`
            : "Sessions started or completed",
        gradient: "from-fuchsia-500 to-violet-600",
        icon: IconCalendar,
      },
      {
        label: "Completed interviews",
        value: summary.completed_interviews_this_month ?? 0,
        hint: "This month (AI)",
        gradient: "from-amber-500 to-orange-500",
        icon: IconCheckCircle,
      },
    ];
  }, [summary]);

  const sidebarInner = (
    <>
      <div className="border-b border-white/10 p-4 sm:p-6">
        <h1 className="bg-gradient-to-r from-blue-300 via-indigo-300 to-violet-300 bg-clip-text text-lg font-bold tracking-tight text-transparent sm:text-xl">
          HireFast
        </h1>
        <p className="mt-1 text-xs text-slate-400">AI-Powered HR Automation</p>
      </div>
      <nav className="flex flex-1 flex-col gap-1 overflow-y-auto p-2 sm:p-3">
        {nav.map((item) => {
          const cls = `group flex items-center gap-2 rounded-xl px-3 py-2.5 text-sm transition-all duration-200 ${
            item.active
              ? "bg-gradient-to-r from-indigo-600/90 to-violet-600/90 text-white shadow-lg shadow-indigo-900/40"
              : "text-slate-400 hover:bg-white/[0.06] hover:text-white"
          }`;
          if (item.next) {
            return (
              <Link key={item.label} href={item.href} onClick={() => setSidebarOpen(false)} className={cls}>
                <span className="h-1.5 w-1.5 rounded-full bg-white/40 opacity-0 transition group-hover:opacity-100" />
                {item.label}
              </Link>
            );
          }
          return (
            <a
              key={item.label}
              href={item.href}
              onClick={() => setSidebarOpen(false)}
              className={cls}
              {...(item.href.startsWith("http") ? { target: "_blank", rel: "noopener noreferrer" } : {})}
            >
              {item.label}
            </a>
          );
        })}
      </nav>
      <div className="border-t border-white/10 p-3 sm:p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 text-sm font-semibold text-white shadow-inner shadow-white/10">
            HF
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-white">Recruiter</p>
            <p className="truncate text-xs text-slate-500">Profile in main app</p>
          </div>
        </div>
      </div>
    </>
  );

  return (
    <div className="relative flex min-h-dvh overflow-hidden bg-[#070b14] text-slate-100">
      <div
        className="pointer-events-none absolute -left-40 top-0 h-[420px] w-[420px] rounded-full bg-indigo-600/25 blur-[100px]"
        aria-hidden
      />
      <div
        className="pointer-events-none absolute -right-32 bottom-0 h-[380px] w-[380px] rounded-full bg-violet-600/20 blur-[90px]"
        aria-hidden
      />
      <div
        className="pointer-events-none absolute left-1/2 top-24 h-64 w-[80%] max-w-3xl -translate-x-1/2 rounded-full bg-cyan-500/10 blur-[80px]"
        aria-hidden
      />

      {sidebarOpen && (
        <button
          type="button"
          className="fixed inset-0 z-30 bg-black/70 backdrop-blur-sm md:hidden"
          aria-label="Close menu"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[min(18rem,92vw)] shrink-0 flex-col overflow-x-hidden border-r border-white/[0.07] bg-[#0a0f1c]/95 shadow-2xl shadow-black/40 backdrop-blur-xl transition-transform duration-300 ease-out md:static md:z-0 md:w-72 md:min-w-[18rem] md:translate-x-0 md:shadow-none ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        {sidebarInner}
      </aside>

      <div className="relative flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-white/[0.06] bg-[#070b14]/85 px-3 py-3 backdrop-blur-xl md:hidden">
          <button
            type="button"
            className="inline-flex items-center justify-center rounded-xl border border-white/10 bg-white/5 p-2 text-slate-200 transition hover:bg-white/10"
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
            className="pointer-events-none absolute inset-0 opacity-[0.14]"
            style={{
              backgroundImage: `linear-gradient(rgba(148,163,184,0.06) 1px, transparent 1px),
              linear-gradient(90deg, rgba(148,163,184,0.06) 1px, transparent 1px)`,
              backgroundSize: "32px 32px",
            }}
          />
          <div className="relative z-10 mx-auto max-w-7xl p-4 sm:p-6 lg:p-8">
            <div className="mb-8 flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
              <div className="min-w-0 animate-[fade-up_0.5s_ease-out]">
                <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] font-medium uppercase tracking-wider text-slate-400">
                  Live data
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                </div>
                <h2 className="mt-3 text-3xl font-bold tracking-tight text-white sm:text-4xl">Dashboard</h2>
                <p className="mt-2 max-w-xl text-sm leading-relaxed text-slate-400 sm:text-base">
                  Welcome back. Metrics below are pulled from your HireFast database (jobs, applications, AI
                  interviews).
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  API:{" "}
                  {health ? (
                    <span className="text-emerald-400/90">
                      {health.service} — {health.status}
                    </span>
                  ) : (
                    <span className="text-slate-500">connecting…</span>
                  )}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                {base ? (
                  <a
                    href={base}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.06] px-4 py-2 text-xs font-medium text-slate-200 transition hover:border-white/20 hover:bg-white/10"
                  >
                    Open full HireFast
                    <span aria-hidden className="text-slate-500">
                      ↗
                    </span>
                  </a>
                ) : (
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-500">
                    Set <code className="text-slate-400">BACKEND_URL</code> for quick links
                  </span>
                )}
              </div>
            </div>

            {err && (
              <div className="mb-6 rounded-2xl border border-amber-500/35 bg-gradient-to-r from-amber-500/10 to-transparent px-4 py-3 text-sm text-amber-100/95 backdrop-blur">
                {err}
              </div>
            )}

            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {stats
                ? stats.map((c, i) => (
                    <div
                      key={c.label}
                      className="group relative overflow-hidden rounded-2xl border border-white/[0.08] bg-gradient-to-b from-white/[0.07] to-white/[0.02] p-5 shadow-xl shadow-black/20 transition duration-300 hover:border-white/[0.14] hover:shadow-2xl"
                      style={{ animationDelay: `${i * 60}ms` }}
                    >
                      <div className="pointer-events-none absolute inset-0 opacity-0 transition duration-500 group-hover:opacity-100">
                        <div
                          className={`absolute -right-8 -top-8 h-32 w-32 rounded-full bg-gradient-to-br ${c.gradient} opacity-20 blur-2xl`}
                        />
                      </div>
                      <div className="relative flex items-start justify-between gap-3">
                        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{c.label}</p>
                        <div
                          className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br ${c.gradient} text-white shadow-lg ring-1 ring-white/10`}
                        >
                          <c.icon className="h-5 w-5 opacity-95" />
                        </div>
                      </div>
                      <p className="relative mt-4 font-mono text-3xl font-bold tabular-nums tracking-tight text-white sm:text-4xl">
                        {c.value}
                      </p>
                      <p className="relative mt-2 text-xs text-slate-500 sm:text-sm">{c.hint}</p>
                    </div>
                  ))
                : [0, 1, 2, 3].map((k) => (
                    <div
                      key={k}
                      className="h-[140px] animate-pulse rounded-2xl border border-white/5 bg-white/[0.04]"
                    />
                  ))}
            </div>

            <div className="mt-10 grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl border border-white/[0.08] bg-gradient-to-b from-white/[0.06] to-transparent p-6 shadow-xl shadow-black/20">
                <h3 className="text-lg font-semibold text-white">Quick actions</h3>
                <p className="mt-1 text-sm text-slate-500">Jump into the full workflow on your main HireFast URL.</p>
                <div className="mt-5 flex flex-col gap-2">
                  {base ? (
                    <>
                      <a
                        href={`${base}/job-posting`}
                        className="group flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-4 py-3.5 text-center text-sm font-semibold text-white shadow-lg shadow-indigo-900/40 transition hover:brightness-110"
                      >
                        <span className="text-lg leading-none">+</span> Create job post
                      </a>
                      <a
                        href={`${base}/candidates`}
                        className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-3 text-center text-sm font-medium text-slate-200 transition hover:bg-white/[0.08]"
                      >
                        View candidates
                      </a>
                      <a
                        href={`${base}/scheduling`}
                        className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-3 text-center text-sm font-medium text-slate-200 transition hover:bg-white/[0.08]"
                      >
                        Schedule interviews
                      </a>
                    </>
                  ) : (
                    <p className="rounded-xl border border-dashed border-white/15 bg-slate-950/40 px-4 py-4 text-sm text-slate-400">
                      Configure <code className="text-slate-300">NEXT_PUBLIC_MAIN_APP_URL</code> or{" "}
                      <code className="text-slate-300">BACKEND_URL</code> in <code className="text-slate-300">client-config</code>{" "}
                      to enable one-click links to job posting and candidates.
                    </p>
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-white/[0.08] bg-gradient-to-b from-white/[0.06] to-transparent p-6 shadow-xl shadow-black/20">
                <h3 className="text-lg font-semibold text-white">Recent activity</h3>
                <p className="mt-1 text-sm text-slate-500">Latest applications, jobs, and AI interviews.</p>
                <ul className="mt-5 space-y-0">
                  {summary?.recent_activity && summary.recent_activity.length > 0 ? (
                    summary.recent_activity.map((item, idx) => (
                      <li
                        key={`${item.at}-${idx}`}
                        className="flex gap-3 border-b border-white/[0.05] py-3 last:border-0"
                      >
                        <div className="relative flex flex-col items-center pt-1.5">
                          <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${accentRing(item.accent)}`} />
                          {idx < (summary.recent_activity?.length ?? 0) - 1 && (
                            <span className="absolute top-4 h-full w-px bg-gradient-to-b from-white/15 to-transparent" />
                          )}
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium leading-snug text-slate-100">{item.title}</p>
                          {item.detail ? (
                            <p className="mt-0.5 truncate text-xs text-slate-500">{item.detail}</p>
                          ) : null}
                          <p className="mt-1 text-[11px] text-slate-600">{formatRelativeTime(item.at)}</p>
                        </div>
                      </li>
                    ))
                  ) : (
                    <li className="py-8 text-center text-sm text-slate-500">
                      {summary ? "No activity yet — create a job or receive applications." : "Loading…"}
                    </li>
                  )}
                </ul>
              </div>
            </div>

            <div className="mt-6 rounded-2xl border border-white/[0.08] bg-gradient-to-r from-indigo-950/40 via-slate-900/40 to-violet-950/40 p-6 shadow-xl shadow-black/30">
              <h3 className="text-lg font-semibold text-white">Current hiring workflow</h3>
              {summary?.workflow ? (
                <div className="mt-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0">
                    <p className="text-xs font-medium uppercase tracking-wide text-indigo-300/80">
                      {summary.workflow.step_label ?? "Latest role"}
                    </p>
                    <p className="mt-1 truncate text-xl font-semibold text-white">{summary.workflow.role}</p>
                    {summary.workflow.company_name ? (
                      <p className="mt-0.5 text-sm text-slate-400">{summary.workflow.company_name}</p>
                    ) : null}
                  </div>
                  <div className="flex shrink-0 items-center gap-3">
                    <span className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/15 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-emerald-200">
                      <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      In database
                    </span>
                  </div>
                </div>
              ) : (
                <p className="mt-3 text-sm text-slate-500">
                  No job postings yet. Create one in the main app to see it here.
                </p>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
