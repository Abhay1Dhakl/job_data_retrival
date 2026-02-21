export async function POST(request) {
  const apiBaseUrl = process.env.API_BASE_URL || "http://localhost:8000";
  const payload = await request.json();

  const response = await fetch(`${apiBaseUrl}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store"
  });

  const body = await response.text();
  return new Response(body, {
    status: response.status,
    headers: { "Content-Type": response.headers.get("content-type") || "application/json" }
  });
}
