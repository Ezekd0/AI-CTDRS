import { it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { NetworkMonitorPage, AdminNetworkReportsPanel } from './NetworkPages';
import { api } from './api';
import { runNetworkTest } from './networkTest';
vi.mock('./api', () => ({ api: { get: vi.fn(), post: vi.fn() } }));
vi.mock('./networkTest', () => ({ runNetworkTest: vi.fn() }));
beforeEach(() => vi.resetAllMocks());
it('runs a test, displays measurements, and saves only on request', async () => {
  const report = { device_id: 'browser', connectivity_status: 'online', internet_status: 'reachable', latency_ms: 40, download_speed_mbps: 30, upload_speed_mbps: 6, overall_status: 'excellent', metadata: {}, created_at: '2026-10-06T08:00:00Z' };
  api.get.mockResolvedValueOnce({ data: { items: [] } }).mockResolvedValue({ data: { items: [{ ...report, id: 'r1', user_id: 'user-1' }] } });
  api.post.mockResolvedValue({ data: { ...report, id: 'r1' } });
  runNetworkTest.mockResolvedValue(report);
  render(<NetworkMonitorPage/>);
  fireEvent.click(screen.getByRole('button', { name: 'RUN NETWORK TEST' }));
  await screen.findByText('AI-CTDRS NETWORK DIAGNOSTIC');
  expect(screen.getByText('EXCELLENT')).toBeInTheDocument();
  expect(screen.getAllByText('Not available in this browser').length).toBeGreaterThan(0);
  expect(api.post).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Save report' }));
  await waitFor(() => expect(api.post).toHaveBeenCalledWith('/network/reports', report));
  await screen.findByRole('button', { name: 'Report saved' });
  fireEvent.click(await screen.findByRole('button', { name: 'Inspect' }));
  expect(screen.getAllByText('40.00 ms')).toHaveLength(2);
});


function deferred() {
  let resolve, reject;
  const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
}

it('keeps the latest admin filter results when earlier requests finish later', async () => {
  const all = deferred(), alice = deferred(), bob = deferred();
  api.get.mockImplementation((path, options) => path === '/admin/users'
    ? Promise.resolve({ data: { items: [{ id: 'alice', email: 'alice@example.com' }, { id: 'bob', email: 'bob@example.com' }] } })
    : (options.params.user_id === 'alice' ? alice : options.params.user_id === 'bob' ? bob : all).promise);
  render(<AdminNetworkReportsPanel/>);
  await screen.findByRole('option', { name: 'alice@example.com' });
  const filter = screen.getByRole('combobox');
  fireEvent.change(filter, { target: { value: 'alice' } });
  fireEvent.change(filter, { target: { value: 'bob' } });
  const report = { id: 'bob-report', user_id: 'bob', created_at: '2026-10-06T08:00:00Z', overall_status: 'good', latency_ms: 42 };
  await act(async () => { bob.resolve({ data: { items: [report] } }); });
  expect(screen.getByText('42.00 ms')).toBeInTheDocument();
  await act(async () => { alice.resolve({ data: { items: [{ ...report, id: 'alice-report', user_id: 'alice', latency_ms: 999 }] } }); });
  expect(screen.getByText('42.00 ms')).toBeInTheDocument();
  expect(screen.queryByText('999.00 ms')).not.toBeInTheDocument();
  await act(async () => { all.reject(new Error('Superseded request failed')); });
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  expect(filter).toHaveValue('bob');
});

it('clears old admin results while the new filter is loading', async () => {
  const next = deferred();
  api.get.mockImplementation((path, options) => path === '/admin/users'
    ? Promise.resolve({ data: { items: [{ id: 'alice', email: 'alice@example.com' }] } })
    : options.params.user_id ? next.promise : Promise.resolve({ data: { items: [{ id: 'old', user_id: 'old-user', created_at: '2026-10-06T08:00:00Z', latency_ms: 123 }] } }));
  render(<AdminNetworkReportsPanel/>);
  await screen.findByText('123.00 ms');
  fireEvent.change(screen.getByRole('combobox'), { target: { value: 'alice' } });
  expect(screen.queryByText('123.00 ms')).not.toBeInTheDocument();
  await act(async () => { next.resolve({ data: { items: [] } }); });
});

it('ignores pending admin requests after unmounting', async () => {
  const users = deferred(), reports = deferred();
  api.get.mockImplementation(path => path === '/admin/users' ? users.promise : reports.promise);
  const { unmount } = render(<AdminNetworkReportsPanel/>);
  unmount();
  // Accessing response data proves whether a stale fulfillment handler ran.
  const readData = vi.fn(() => ({ items: [] }));
  await act(async () => {
    users.resolve({ get data() { return readData(); } });
    reports.resolve({ get data() { return readData(); } });
  });
  expect(readData).not.toHaveBeenCalled();
});
