# Web network monitor

Authenticated users can run a test at `/network` or in the dashboard. Existing JWT authentication and `VITE_API_URL` are used. Production supports `https://ai-ctdrs-api.onrender.com/api`. There is no Android dependency.

The browser makes five GET requests to `/network/latency`, times successful responses with `performance.now()`, and averages them. It then consumes a 1,000,000-byte random binary response from GET `/network/probe?size_bytes=1000000`, and POSTs 1,000,000 random bytes to `/network/probe`. Speeds are actual payload bits divided by elapsed seconds, including HTTP and backend overhead. Responses are not cached; incompressible bytes prevent compression from inflating measured speed. A backend cold start can raise latency. These small tests measure application transfer throughput to this backend, not maximum ISP capacity.

Browser online state and optional Network Information API type, effective type, RTT and downlink are separate from measured metrics. Unsupported values display “Not available in this browser”. HTTP reachability does not prove general internet access, and successful DNS resolution can be cached. Failed HTTP requests cannot isolate a DNS fault. No packet-loss measurement is claimed.

Health classification uses actual HTTP latency/download/upload, in this order:

- UNAVAILABLE: backend unreachable or any required measurement missing.
- POOR: latency >=600 ms, download <1 Mbps, or upload <0.5 Mbps.
- DEGRADED: latency >=250 ms, download <3 Mbps, or upload <1 Mbps.
- EXCELLENT: latency <100 ms, download >=25 Mbps, upload >=5 Mbps.
- GOOD: all other complete results.

These are application diagnostic thresholds, not AI cyberattack classifications. The backend independently computes the classification on save. Client measurements are user-submitted diagnostics, not server-attested evidence.

Save explicitly POSTs `/network/reports`. GET `/network/reports`, `/network/reports/latest`, and `/network/reports/{id}` use existing ownership checks. Administrators can list/filter/inspect all reports; latest always means the current account's latest report. `/admin/network` is protected by the existing administrator route gate. Reports have a copyable provider summary without a public sharing endpoint.

Uploads are streamed and discarded with a 2,000,000-byte maximum, including when Content-Length is absent. Downloads have the same size maximum. Each worker limits probe requests to 20 per user per rolling minute. There is no arbitrary target URL or proxy. Typed metadata accepts only browser metrics and HTTP probe observations. Private IPs, DNS server lists, carrier/signal data and packet-loss submissions are disabled. No stable device ID is collected (`device_id` is the literal `browser`).

Migration `0006_network_reports` and the existing table are retained; legacy nullable columns remain for compatibility but are not populated by browser tests. Apply normal Alembic migrations through the existing deployment procedure when deployment is separately authorized. No production database changes were made during this work.

Manual browser verification after authorized deployment: login, open `/network`, run and save a test, reload history, inspect/copy a provider report, test an offline connection, and repeat in a browser without Network Information API support. Verify another user cannot see the saved report and an administrator can inspect it. Check production CORS and `VITE_API_URL`, especially cross-origin binary uploads. This implementation has not been deployed.

## Production route verification (2026-10-06)

Read-only checks against `https://ai-ctdrs-api.onrender.com` returned 200 for `/health` and `/api/auth/status`, but 404 `{"detail":"Not Found"}` for every network route below (without authentication). The local backend mounts all of these under `/api` and returns 401 for unauthenticated requests. Thus the deployed backend does not expose the network implementation present in this checkout. Changing frontend paths or adding duplicate endpoints cannot fix that deployment mismatch. The page initially loads GET `/api/network/reports`, explaining the displayed error even before running a test.

- GET `/api/network/latency`
- GET `/api/network/probe`
- POST `/api/network/probe`
- GET `/api/network/reports`
- GET `/api/network/reports/latest`
- GET `/api/network/reports/{report_id}`
- POST `/api/network/reports`

After deployment is explicitly authorized, release the backend containing the existing network router and migration `0006` using the existing Docker startup. Set Vercel `VITE_API_URL=https://ai-ctdrs-api.onrender.com/api` at build time. Set Render `CORS_ORIGINS` to the exact frontend origin (no path or trailing slash), using the existing comma-separated configuration for additional origins. Check OPTIONS with that origin, `Access-Control-Request-Method: POST`, and `Access-Control-Request-Headers: authorization,content-type`; expect `Access-Control-Allow-Origin` to match. The actual Vercel origin must be supplied before this can be verified. No deployment or production database changes are authorized or performed here.
