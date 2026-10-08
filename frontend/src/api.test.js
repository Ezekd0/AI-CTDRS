import { describe, expect, it, afterEach } from 'vitest';
import { api, predictionErrorMessage } from './api';

describe('detection request diagnostics', () => {
  afterEach(() => localStorage.clear());
  it('sends the stored JWT and unchanged detection features to the existing endpoint', async () => {
    localStorage.setItem('access_token', 'admin-test-token');
    const features = { duration: 10, src_bytes: 1024, dst_bytes: 2048, src_packets: 20, dst_packets: 15 };
    await api.post('/predict', { dataset: 'nsl-kdd', task: 'binary', model: 'random_forest', features }, {
      adapter: async config => {
        expect(config.method).toBe('post');
        expect(api.getUri(config)).toBe(`${api.defaults.baseURL}/predict`);
        expect(config.headers.Authorization).toBe('Bearer admin-test-token');
        expect(JSON.parse(config.data).features).toEqual(features);
        return { status: 201, data: {}, headers: {}, config };
      },
    });
  });
  it.each([401, 403, 404, 422, 500, 503])('shows HTTP %s rather than a generic network error', status => {
    expect(predictionErrorMessage({ message: 'Network Error', response: { status, data: { detail: 'Backend detail' } } })).toBe(`HTTP ${status}: Backend detail`);
  });
  it('handles a non-JSON response and distinguishes blocked responses and timeouts', () => {
    expect(predictionErrorMessage({ response: { status: 502, data: '<html>Bad Gateway</html>' } })).toBe('HTTP 502: Detection request failed');
    expect(predictionErrorMessage({ code: 'ERR_NETWORK' })).toContain('CORS');
    expect(predictionErrorMessage({ code: 'ECONNABORTED' })).toContain('may still be processing');
  });
});
