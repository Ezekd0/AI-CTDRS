import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';
import { api } from './api';

vi.mock('./api', async () => {
  const actual = await vi.importActual('./api');
  return { ...actual, login: vi.fn(), logout: vi.fn(), api: { ...actual.api, get: vi.fn(), post: vi.fn() } };
});

describe('authentication and dashboard integration', () => {
  beforeEach(() => { localStorage.clear(); vi.clearAllMocks(); });
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
    api.get.mockResolvedValue({ data: { statistics: {}, recent_incidents: [] } });
    render(<App />);
    window.dispatchEvent(new Event('auth:logout'));
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Sign in' })).toBeInTheDocument());
  });
});
