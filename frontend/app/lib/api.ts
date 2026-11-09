export const DEMO = process.env.NEXT_PUBLIC_DEMO === "1";
export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** The public demo reads saved results instead of calling a server. */
export function apiUrl(path: string) {
  return DEMO ? `/demo${path.replace(/^\/api/, "")}.json` : `${API}${path}`;
}
