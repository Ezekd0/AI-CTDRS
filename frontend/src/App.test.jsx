import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';
import { api } from './api';

vi.mock('./api', async () => {
  const actual = await vi.importActual('./api');
  return { ...actual, login: vi.fn(), logout: vi.fn(), api: { ...actual.api, get: vi.fn(), post: vi.fn(), patch: vi.fn() } };
});

describe('authentication and dashboard integration', () => {
  beforeEach(() => { localStorage.clear(); vi.clearAllMocks(); window.history.replaceState({}, "", "/login"); api.get.mockImplementation(path => Promise.resolve({ data: path === "/auth/me" ? { id: "u1", email: "analyst@example.com", role: "analyst" } : { statistics: {}, recent_incidents: [] } })); });
  it('renders login and authenticates', async () => {
    const { login } = await import('./api');
    login.mockResolvedValue({ access_token: 'token', role: 'analyst' });
    render(<App />);
    expect(screen.getByRole('heading', { name: 'Sign in' })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Email'), { target: { value: 'analyst@example.com' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'password' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));
    await waitFor(() => expect(screen.getByText('Submit network-feature input')).toBeInTheDocument());
  });

  it('shows unauthorized state after a 401 event', async () => {
    localStorage.setItem('access_token', 'expired');
    render(<App />);
    window.dispatchEvent(new Event('auth:logout'));
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Sign in' })).toBeInTheDocument());
  });

  it.each([[false, false], [true, false], [false, true], [true, true]])('submits SHAP=%s and LIME=%s with a bounded prediction timeout', async (shap, lime) => {
    localStorage.setItem('access_token', 'analyst-token');
    window.history.replaceState({}, '', '/dashboard');
    api.post.mockImplementation(() => new Promise(() => {}));
    render(<App />);
    const shapBox = await screen.findByLabelText('Generate SHAP');
    const limeBox = screen.getByLabelText('Generate LIME');
    fireEvent.change(screen.getByRole('textbox'), { target: { value: JSON.stringify({ duration: 0, protocol_type: 'tcp' }) } });
    if (shapBox.checked !== shap) fireEvent.click(shapBox);
    if (limeBox.checked !== lime) fireEvent.click(limeBox);
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    await waitFor(() => expect(api.post).toHaveBeenCalledWith('/predict', expect.objectContaining({ generate_shap: shap, generate_lime: lime }), { timeout: 30000 }));
    expect(screen.getByRole('button', { name: 'Running pipeline…' })).toBeDisabled();
  });

  it('shows explanation timeout and unavailability on the successful detection details', async () => {
    localStorage.setItem('access_token', 'analyst-token');
    window.history.replaceState({}, '', '/dashboard');
    const detection = { id: 'd1', confidence: 0.95, predicted_attack: 'normal', severity: 'low', explanation_status: { shap: 'timed_out: server-side limit', lime: 'unavailable: warming up' } };
    api.get.mockImplementation(path => Promise.resolve({ data: path === '/auth/me' ? { id: 'u1', role: 'analyst' } : path === '/detections/d1' ? detection : { statistics: {}, recent_incidents: [] } }));
    api.post.mockResolvedValue({ data: { detection_id: 'd1', prediction: 'normal', probability: 0.95, explanation: detection.explanation_status } });
    render(<App />);
    await screen.findByLabelText('Generate SHAP');
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '{"duration": 0}' } });
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    expect(await screen.findByText('timed_out: server-side limit')).toBeInTheDocument();
    expect(screen.getByText('unavailable: warming up')).toBeInTheDocument();
    expect(screen.getByText('SHAP status')).toBeInTheDocument();
    expect(screen.getByText('LIME status')).toBeInTheDocument();
  });
});

function fillSignup(confirm = 'RegistrationPassword123!') {
  fireEvent.change(screen.getByLabelText('Full name'), { target: { value: 'Test User' } });
  fireEvent.change(screen.getByLabelText('Email'), { target: { value: 'test@example.com' } });
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'RegistrationPassword123!' } });
  fireEvent.change(screen.getByLabelText('Confirm password'), { target: { value: confirm } });
}

describe('registration and protected routes', () => {
  beforeEach(() => {
    localStorage.clear(); vi.clearAllMocks();
    window.history.replaceState({}, '', '/signup');
    api.get.mockImplementation(path => Promise.resolve({ data: path === '/auth/me' ? { id: 'u1', role: 'viewer', email: 'test@example.com' } : { statistics: {}, recent_incidents: [] } }));
  });
  it('validates confirmation, registers, returns to login and enters dashboard', async () => {
    api.post.mockResolvedValue({ data: { id: 'u1' } });
    const { login } = await import('./api'); login.mockResolvedValue({ access_token: 'token', role: 'viewer' });
    render(<App />);
    fillSignup('different');
    fireEvent.click(screen.getByRole('button', { name: 'Sign up' }));
    expect(screen.getByRole('alert')).toHaveTextContent('Passwords do not match');
    expect(api.post).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText('Confirm password'), { target: { value: 'RegistrationPassword123!' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign up' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Registration successful'));
    expect(api.post).toHaveBeenCalledWith('/auth/register', { full_name: 'Test User', email: 'test@example.com', password: 'RegistrationPassword123!' });
    fireEvent.click(screen.getByRole('link', { name: 'Return to Login' }));
    fireEvent.change(screen.getByLabelText('Email'), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'RegistrationPassword123!' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));
    await waitFor(() => expect(window.location.pathname).toBe('/dashboard'));
    expect(api.get).toHaveBeenCalledWith('/auth/me');
  });
  it('shows loading and duplicate-registration errors', async () => {
    let reject; api.post.mockImplementation(() => new Promise((_, fail) => { reject = fail; }));
    render(<App />); fillSignup();
    fireEvent.click(screen.getByRole('button', { name: 'Sign up' }));
    expect(screen.getByRole('button', { name: 'Creating account…' })).toBeDisabled();
    reject({ response: { status: 409, data: { detail: 'An account with this email already exists' } } });
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('An account with this email already exists'));
    expect(screen.getByRole('button', { name: 'Sign up' })).toBeEnabled();
  });
  it('rejects short passwords and blank names before calling registration', () => {
    render(<App />); fillSignup();
    fireEvent.change(screen.getByLabelText('Full name'), { target: { value: '  ' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign up' }));
    expect(screen.getByRole('alert')).toHaveTextContent('Enter your full name');
    fireEvent.change(screen.getByLabelText('Full name'), { target: { value: 'User' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'short' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign up' }));
    expect(screen.getByRole('alert')).toHaveTextContent('12–128 characters');
    expect(api.post).not.toHaveBeenCalled();
  });
  it('redirects unauthenticated users from protected URLs', async () => {
    window.history.replaceState({}, '', '/admin/users'); render(<App />);
    await waitFor(() => expect(window.location.pathname).toBe('/login'));
    expect(screen.getByRole('heading', { name: 'Sign in' })).toBeInTheDocument();
  });
  it('redirects ordinary users from admin URLs without requesting admin data', async () => {
    localStorage.setItem('access_token', 'viewer-token');
    window.history.replaceState({}, '', '/admin/users'); render(<App />);
    await waitFor(() => expect(window.location.pathname).toBe('/dashboard'));
    expect(screen.queryByRole('heading', { name: 'User management' })).not.toBeInTheDocument();
    expect(api.get.mock.calls.some(([path]) => path.startsWith('/admin'))).toBe(false);
  });
  it('loads admin overview and manages account status with server filters', async () => {
    localStorage.setItem('access_token', 'admin-token'); window.history.replaceState({}, '', '/admin');
    const user = { id: 'u2', full_name: 'Example User', email: 'example@example.com', role: 'viewer', is_active: true, created_at: '2026-10-05T09:00:00Z' };
    api.get.mockImplementation(path => Promise.resolve({ data: path === '/auth/me' ? { id: 'admin', role: 'administrator' } : path === '/admin/overview' ? { total_users: 2, active_users: 2, registrations_last_7_days: 1, total_detections: 0, recent_registrations: [user], recent_detections: [], system_status: { api: 'ok', database: 'connected', response_mode: 'SIMULATION' } } : { items: [user], total: 1 } }));
    api.patch.mockResolvedValue({ data: { ...user, is_active: false } });
    render(<App />);
    await waitFor(() => expect(screen.getByText('Total users')).toBeInTheDocument());
    fireEvent.click(screen.getByRole('link', { name: 'Manage users →' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Disable' })).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: 'Disable' }));
    await waitFor(() => expect(api.patch).toHaveBeenCalledWith('/admin/users/u2/status', { is_active: false }));
    fireEvent.change(screen.getByLabelText('Search users'), { target: { value: 'Example' } });
    fireEvent.change(screen.getByLabelText('Filter status'), { target: { value: 'true' } });
    await waitFor(() => expect(api.get).toHaveBeenCalledWith('/admin/users', { params: { offset: 0, limit: 25, search: 'Example', is_active: 'true' } }));
    expect(window.location.pathname).toBe('/admin/users');
  });
});
