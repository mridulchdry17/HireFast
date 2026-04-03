import { NextResponse } from "next/server";
import { resolveMainAppUrl } from "@/lib/mainAppUrl";

/** Exposes safe public config to the browser (main Flask URL for sidebar links). */
export async function GET() {
  const main = resolveMainAppUrl();
  return NextResponse.json({ mainAppUrl: main || null });
}
