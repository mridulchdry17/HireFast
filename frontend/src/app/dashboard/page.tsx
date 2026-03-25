"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { apiUrl } from "@/lib/api";

type Health = { status?: string; service?: string };
type ApplicantsPayload = { applicants?: unknown[]; error?: string };

export default function DashboardPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [applicants, setApplicants] = useState<unknown[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const h = await fetch(apiUrl("/health"));
        const hj = (await h.json()) as Health;
        if (!cancelled) setHealth(hj);
      } catch {
        if (!cancelled) setErr("Could not reach API (check BACKEND_URL / proxy).");
      }
      try {
        const a = await fetch(apiUrl("/fetch-applications"));
        const aj = (await a.json()) as ApplicantsPayload;
        if (!cancelled) {
          if (aj.error) setErr(aj.error);
          else setApplicants(aj.applicants ?? []);
        }
      } catch {
        if (!cancelled) setErr((prev) => prev ?? "Failed to load applicants.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const nav = [
    { href: "/dashboard", label: "Dashboard", active: true },
    { href: "#", label: "Create Job Post" },
    { href: "#", label: "Candidates" },
    { href: "#", label: "Schedule Interviews" },
    { href: "#", label: "Connect Calendar" },
    { href: "#", label: "Analytics" },
    { href: "#", label: "Settings" },
  ];

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-slate-100">
      <aside className="flex w-64 flex-col border-r border-white/10 bg-slate-950/90 backdrop-blur">
        <div className="border-b border-white/10 p-6">
          <h1 className="bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-xl font-bold text-transparent">
            HireFast
          </h1>
          <p className="mt-1 text-xs text-slate-400">AI-Powered HR Automation</p>
        </div>
        <nav className="flex flex-1 flex-col gap-1 p-3">
          {nav.map((item) => (
            <Link
              key={item.label}
              href={item.href}
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
        <div className="border-t border-white/10 p-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-violet-600 text-sm font-semibold">
              JD
            </div>
            <div>
              <p className="text-sm font-medium">John Doe</p>
              <p className="text-xs text-slate-400">HR Manager</p>
            </div>
          </div>
        </div>
      </aside>

      <main className="relative flex-1 overflow-auto">
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.12]"
          style={{
            backgroundImage: `linear-gradient(rgba(148,163,184,0.15) 1px, transparent 1px),
              linear-gradient(90deg, rgba(148,163,184,0.15) 1px, transparent 1px)`,
            backgroundSize: "24px 24px",
          }}
        />
        <div className="relative z-10 p-8">
          <div className="mb-8 flex flex-wrap items-start justify-between gap-4">
            <div>
              <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
              <p className="mt-1 text-slate-400">
                Welcome back! Here&apos;s what&apos;s happening with your hiring process.
              </p>
            </div>
            <div className="flex flex-wrap gap-3 text-sm">
              <span className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                LinkedIn Connected
              </span>
              <span className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                Google Calendar Connected
              </span>
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
              { label: "Active Job Posts", value: "—", trend: "+3 this week", accent: "from-blue-500 to-indigo-600" },
              { label: "Total Applications", value: applicants ? String(applicants.length) : "—", trend: "+18 today", accent: "from-emerald-500 to-teal-600" },
              { label: "Interviews Scheduled", value: "8", trend: "3 this week", accent: "from-pink-500 to-violet-600" },
              { label: "Hires This Month", value: "5", trend: "+2 from last month", accent: "from-amber-500 to-orange-600" },
            ].map((c) => (
              <div
                key={c.label}
                className="rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur"
              >
                <div className="flex items-start justify-between">
                  <p className="text-sm text-slate-400">{c.label}</p>
                  <div className={`h-10 w-10 rounded-xl bg-gradient-to-br ${c.accent} opacity-90`} />
                </div>
                <p className="mt-4 text-3xl font-bold">{c.value}</p>
                <p className="mt-1 text-sm text-emerald-400/90">{c.trend}</p>
              </div>
            ))}
          </div>

          <div className="mt-10 grid gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
              <h3 className="mb-4 text-lg font-semibold">Quick Actions</h3>
              <div className="flex flex-col gap-3">
                <button
                  type="button"
                  className="rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-4 py-3 text-left text-sm font-semibold text-white"
                >
                  + Create New Job Post
                </button>
                <button
                  type="button"
                  className="rounded-xl border border-white/10 bg-slate-900/80 px-4 py-3 text-left text-sm text-slate-200"
                >
                  View All Candidates
                </button>
                <button
                  type="button"
                  className="rounded-xl border border-white/10 bg-slate-900/80 px-4 py-3 text-left text-sm text-slate-200"
                >
                  Schedule Interview
                </button>
              </div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
              <h3 className="mb-4 text-lg font-semibold">Recent Activity</h3>
              <ul className="space-y-4 text-sm">
                <li className="flex gap-3">
                  <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-emerald-400" />
                  <div>
                    <p>New application for Software Engineer</p>
                    <p className="text-xs text-slate-500">2 minutes ago</p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-blue-400" />
                  <div>
                    <p>Interview scheduled with Sarah Johnson</p>
                    <p className="text-xs text-slate-500">1 hour ago</p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-violet-400" />
                  <div>
                    <p>Job post published to LinkedIn</p>
                    <p className="text-xs text-slate-500">3 hours ago</p>
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
