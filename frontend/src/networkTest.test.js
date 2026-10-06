import { describe, it, expect, vi, beforeEach } from 'vitest';
import { assessCondition, runNetworkTest } from './networkTest';
import { api } from './api';
vi.mock('./api', () => ({ api: { get: vi.fn(), post: vi.fn() } }));
beforeEach(() => vi.clearAllMocks());
describe('browser network diagnostics', () => {
  it('uses received bytes and acknowledged upload with unavailable browser fields', async () => {
    api.get.mockImplementation(path => Promise.resolve({ data: path.endsWith('latency') ? { status: 'reachable' } : new ArrayBuffer(1000000) }));
    api.post.mockResolvedValue({ data: { bytes_received: 1000000 } });
    const r = await runNetworkTest();
    expect(r.metadata.latency_samples_ms).toHaveLength(5);
    expect(r.download_speed_mbps).toBeGreaterThan(0);
    expect(r.upload_speed_mbps).toBeGreaterThan(0);
    expect(r.packet_loss_pct).toBeNull();
    expect(r.connection_type).toBeNull();
    expect(api.post.mock.calls[0][1].byteLength).toBe(1000000);
  });
  it('records HTTP failures without inventing packet loss or throughput', async () => {
    api.get.mockRejectedValue(new Error('Network error'));
    const r = await runNetworkTest();
    expect(r.overall_status).toBe('unavailable');
    expect(r.latency_ms).toBeNull();
    expect(r.download_speed_mbps).toBeNull();
    expect(r.metadata.failed_http_requests).toBe(5);
    expect(api.post).not.toHaveBeenCalled();
  });
  it('derives documented health conditions', () => {
    const r = { internet_status: 'reachable', latency_ms: 20, download_speed_mbps: 50, upload_speed_mbps: 10 };
    expect(assessCondition(r)).toBe('excellent');
    expect(assessCondition({ ...r, latency_ms: 100 })).toBe('good');
    expect(assessCondition({ ...r, latency_ms: 250 })).toBe('degraded');
    expect(assessCondition({ ...r, upload_speed_mbps: 0.2 })).toBe('poor');
    expect(assessCondition({ ...r, upload_speed_mbps: null })).toBe('unavailable');
  });
});
