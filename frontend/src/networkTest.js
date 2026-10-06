import { api } from './api';

export function assessCondition(r) {
  if (r.internet_status !== 'reachable' || [r.latency_ms, r.download_speed_mbps, r.upload_speed_mbps].some(v => v == null)) return 'unavailable';
  if (r.latency_ms >= 600 || r.download_speed_mbps < 1 || r.upload_speed_mbps < 0.5) return 'poor';
  if (r.latency_ms >= 250 || r.download_speed_mbps < 3 || r.upload_speed_mbps < 1) return 'degraded';
  if (r.latency_ms < 100 && r.download_speed_mbps >= 25 && r.upload_speed_mbps >= 5) return 'excellent';
  return 'good';
}

export async function runNetworkTest() {
  const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  const result = {
    device_id: 'browser', connection_type: connection?.type || null,
    connectivity_status: navigator.onLine ? 'online' : 'offline',
    internet_status: 'unreachable', dns_status: 'HTTP request failed; DNS cause cannot be isolated',
    latency_ms: null, packet_loss_pct: null, download_speed_mbps: null, upload_speed_mbps: null,
    metadata: { effective_type: connection?.effectiveType || null, rtt_ms: connection?.rtt ?? null,
      downlink_mbps: connection?.downlink ?? null, latency_samples_ms: [], failed_http_requests: 0 },
    created_at: new Date().toISOString(),
  };
  const options = { timeout: 20000, params: { nonce: Date.now() } };
  for (let i = 0; i < 5; i++) {
    const start = performance.now();
    try {
      const response = await api.get('/network/latency', options);
      if (response.data.status !== 'reachable') throw new Error('Invalid probe response');
      result.metadata.latency_samples_ms.push(performance.now() - start);
    } catch (error) {
      if ([401, 403, 429].includes(error.response?.status)) throw error;
      result.metadata.failed_http_requests++;
    }
  }
  const samples = result.metadata.latency_samples_ms;
  if (samples.length) {
    result.internet_status = 'reachable';
    result.dns_status = 'Backend HTTP reachable; DNS may be cached';
    result.latency_ms = samples.reduce((a, b) => a + b, 0) / samples.length;
    const size = 1_000_000;
    try {
      const start = performance.now();
      const response = await api.get('/network/probe', { ...options, params: { ...options.params, size_bytes: size }, responseType: 'arraybuffer' });
      const elapsed = performance.now() - start;
      if (response.data.byteLength === size && elapsed > 0) result.download_speed_mbps = size * 8 / (elapsed * 1000);
    } catch (error) { if ([401, 403, 429].includes(error.response?.status)) throw error; }
    try {
      const payload = new Uint8Array(size);
      for (let i = 0; i < size; i += 65536) crypto.getRandomValues(payload.subarray(i, Math.min(i + 65536, size)));
      const start = performance.now();
      const response = await api.post('/network/probe', payload, { timeout: 20000, headers: { 'Content-Type': 'application/octet-stream' } });
      const elapsed = performance.now() - start;
      if (response.data.bytes_received === size && elapsed > 0) result.upload_speed_mbps = size * 8 / (elapsed * 1000);
    } catch (error) { if ([401, 403, 429].includes(error.response?.status)) throw error; }
  }
  result.overall_status = assessCondition(result);
  return result;
}
